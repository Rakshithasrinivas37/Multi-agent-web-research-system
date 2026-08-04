"""LangGraph workflow for the multi-agent research system."""

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.browser_agent import BrowserAgent
from src.agents.change_detection_agent import ChangeDetectionAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.memory.shared_memory import SharedMemory
from src.rag import index_research_results
from src.tools.text_utils import clean_text

class ResearchState(TypedDict, total=False):
    objective: str
    research_plan: dict[str, Any]
    browser_results: list[dict[str, Any]]
    change_detection: dict[str, Any]
    rag_index: dict[str, Any]
    memory_path: str
    history_db_path: str
    chroma_path: str
    model: str
    synthesis: dict[str, Any]
    max_concurrency: int
    errors: list[str]

def planner_node(state: ResearchState) -> ResearchState:
    """Create a research plan and write it to shared memory."""

    objective = (state.get("objective") or "").strip()
    if not objective:
        return {"errors": ["objective is required"]}

    memory_path = state.get("memory_path") or "data/shared_memory.json"
    model = clean_text(state.get("model"))
    planner = PlannerAgent(use_llm=True, model=model or None)
    plan = planner.plan(objective)
    planner.write_to_memory(plan, memory_path)

    return {
        "objective": plan.objective,
        "research_plan": plan.to_dict(),
        "memory_path": memory_path,
        "model": planner.model,
        "errors": [],
    }

def read_research_plan_from_memory(memory_path: str) -> dict[str, Any]:
    """Read planner output from shared memory."""

    memory = SharedMemory(memory_path)
    planner_output = memory.read_agent_output("planner")
    plan = planner_output.get("research_plan", {})
    return plan if isinstance(plan, dict) else {}

async def browser_node(state: ResearchState) -> ResearchState:
    """Read research plan from state or shared memory, then run browser tasks."""

    memory_path = state.get("memory_path") or "data/shared_memory.json"
    plan = read_research_plan_from_memory(memory_path)
    tasks = plan.get("tasks") or plan.get("subtasks") or []
    if not tasks:
        return {
            **state,
            "memory_path": memory_path,
            "errors": [*state.get("errors", []), "browser_node requires research_plan.tasks"],
        }

    browser = BrowserAgent(
        max_concurrency=state.get("max_concurrency", 3),
    )
    results = await browser.run_tasks(tasks)

    memory = SharedMemory(memory_path)
    memory.write_agent_output("browser", {"results": results})

    return {
        **state,
        "research_plan": plan,
        "browser_results": results,
        "memory_path": memory_path,
        "errors": state.get("errors", []),
    }

def change_detection_node(state: ResearchState) -> ResearchState:
    """Compare browser results, persist the diff, and index current content for RAG."""

    memory_path = state.get("memory_path") or "data/shared_memory.json"
    history_db_path = state.get("history_db_path") or "data/browser_history.db"
    chroma_path = state.get("chroma_path") or "data/chroma"
    objective = clean_text(state.get("objective"))
    current_results = state.get("browser_results", [])
    plan = state.get("research_plan", {})

    if not current_results:
        return {
            **state,
            "memory_path": memory_path,
            "history_db_path": history_db_path,
            "errors": [*state.get("errors", []), "change_detection_node requires browser_results"],
        }

    change_detector = ChangeDetectionAgent(history_db_path)
    diff = change_detector.detect_with_history(objective, current_results, plan)
    change_detector.write_to_memory(diff, memory_path)
    rag_index = index_rag_after_change_detection(
        browser_results=current_results,
        research_plan=plan,
        change_detection=diff,
        memory_path=memory_path,
        chroma_path=chroma_path,
    )
    errors = state.get("errors", [])
    if rag_index.get("status") == "error":
        errors = [*errors, clean_text(rag_index.get("error"))]

    return {
        **state,
        "change_detection": diff,
        "rag_index": rag_index,
        "memory_path": memory_path,
        "history_db_path": history_db_path,
        "chroma_path": chroma_path,
        "errors": errors,
    }

def synthesis_node(state: ResearchState) -> ResearchState:
    """Retrieve indexed evidence and write report-ready synthesis to memory."""

    memory_path = state.get("memory_path") or "data/shared_memory.json"
    chroma_path = state.get("chroma_path") or "data/chroma"
    plan = state.get("research_plan") or read_research_plan_from_memory(memory_path)
    if not plan:
        return {
            **state,
            "memory_path": memory_path,
            "errors": [*state.get("errors", []), "synthesis_node requires research_plan"],
        }

    synthesis_agent = SynthesisAgent(
        model=state.get("model"),
        chroma_path=chroma_path,
    )
    try:
        synthesis = synthesis_agent.synthesize(plan)
    except Exception as error:
        return {
            **state,
            "memory_path": memory_path,
            "errors": [*state.get("errors", []), clean_text(error)],
        }

    synthesis_agent.write_to_memory(synthesis, memory_path)
    return {
        **state,
        "synthesis": synthesis,
        "memory_path": memory_path,
        "chroma_path": chroma_path,
        "errors": state.get("errors", []),
    }

def index_rag_after_change_detection(
    browser_results: list[dict[str, Any]],
    research_plan: dict[str, Any],
    change_detection: dict[str, Any],
    memory_path: str,
    chroma_path: str,
) -> dict[str, Any]:
    """Index browser content after change detection and write the summary to memory."""

    try:
        index_summary = index_research_results(
            browser_results=browser_results,
            research_plan=research_plan,
            change_detection=change_detection,
            chroma_path=chroma_path,
        )
    except RuntimeError as error:
        index_summary = {"status": "error", "error": str(error), "chroma_path": chroma_path}
        memory = SharedMemory(memory_path)
        memory.write_agent_output("rag_index", {"index": index_summary})
        return index_summary

    memory = SharedMemory(memory_path)
    memory.write_agent_output("rag_index", {"index": index_summary})
    return index_summary

def build_planner_graph():
    """Build a graph with only the planner node."""

    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", END)
    return graph.compile()

def build_research_graph():
    """Build the planner-to-browser-to-change-detection research graph."""

    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("browser", browser_node)
    graph.add_node("change_detection", change_detection_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "browser")
    graph.add_edge("browser", "change_detection")
    graph.add_node("synthesis", synthesis_node)
    graph.add_edge("change_detection", "synthesis")
    graph.add_edge("synthesis", END)
    return graph.compile()

def run_planner_graph(objective: str, memory_path: str = "data/shared_memory.json") -> ResearchState:
    """Run the LangGraph workflow through the planner node."""

    graph = build_planner_graph()
    return graph.invoke(
        {
            "objective": objective,
            "memory_path": memory_path,
        }
    )

async def run_research_graph(
    objective: str,
    memory_path: str = "data/shared_memory.json",
    history_db_path: str = "data/browser_history.db",
    chroma_path: str = "data/chroma",
    max_concurrency: int = 3,
    model: str | None = None,
) -> ResearchState:
    """Run the LangGraph workflow through planner, browser, change detection, and RAG indexing."""

    graph = build_research_graph()
    return await graph.ainvoke(
        {
            "objective": objective,
            "memory_path": memory_path,
            "history_db_path": history_db_path,
            "chroma_path": chroma_path,
            "max_concurrency": max_concurrency,
            "model": model or "",
        }
    )
