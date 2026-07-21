"""LangGraph workflow for the multi-agent research system."""

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.browser_agent import BrowserAgent
from src.agents.change_detection_agent import ChangeDetectionAgent
from src.agents.planner_agent import PlannerAgent
from src.memory.shared_memory import SharedMemory
from src.tools.text_utils import clean_text

class ResearchState(TypedDict, total=False):
    objective: str
    research_plan: dict[str, Any]
    browser_results: list[dict[str, Any]]
    change_detection: dict[str, Any]
    memory_path: str
    history_db_path: str
    max_concurrency: int
    errors: list[str]

def planner_node(state: ResearchState) -> ResearchState:
    """Create a research plan and write it to shared memory."""

    objective = (state.get("objective") or "").strip()
    if not objective:
        return {"errors": ["objective is required"]}

    memory_path = state.get("memory_path") or "data/shared_memory.json"
    planner = PlannerAgent(use_llm=True)
    plan = planner.plan(objective)
    planner.write_to_memory(plan, memory_path)

    return {
        "objective": plan.objective,
        "research_plan": plan.to_dict(),
        "memory_path": memory_path,
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
    """Compare previous and current browser results and write a compact diff."""

    memory_path = state.get("memory_path") or "data/shared_memory.json"
    history_db_path = state.get("history_db_path") or "data/browser_history.db"
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

    return {
        **state,
        "change_detection": diff,
        "memory_path": memory_path,
        "history_db_path": history_db_path,
        "errors": state.get("errors", []),
    }

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
    graph.add_edge("change_detection", END)
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
    max_concurrency: int = 3,
) -> ResearchState:
    """Run the LangGraph workflow through planner, browser, and change detection nodes."""

    graph = build_research_graph()
    return await graph.ainvoke(
        {
            "objective": objective,
            "memory_path": memory_path,
            "history_db_path": history_db_path,
            "max_concurrency": max_concurrency,
        }
    )
