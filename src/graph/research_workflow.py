"""LangGraph workflow for the multi-agent research system."""

from functools import wraps
from inspect import iscoroutinefunction
from time import perf_counter
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.browser_agent import BrowserAgent
from src.agents.change_detection_agent import ChangeDetectionAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.report_agent import ReportAgent
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
    report: dict[str, Any]
    max_concurrency: int
    errors: list[str]
    agent_timings: list[dict[str, Any]]


def log_agent_step(agent_name: str):
    """Print node completion time and attach timing metadata to state."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(state: ResearchState) -> ResearchState:
            started_at = perf_counter()
            try:
                result = await func(state)
            except Exception:
                elapsed = perf_counter() - started_at
                print(f"[{agent_name}] failed after {elapsed:.2f}s")
                raise
            return add_agent_timing(agent_name, result, elapsed=perf_counter() - started_at)

        @wraps(func)
        def sync_wrapper(state: ResearchState) -> ResearchState:
            started_at = perf_counter()
            try:
                result = func(state)
            except Exception:
                elapsed = perf_counter() - started_at
                print(f"[{agent_name}] failed after {elapsed:.2f}s")
                raise
            return add_agent_timing(agent_name, result, elapsed=perf_counter() - started_at)

        return async_wrapper if iscoroutinefunction(func) else sync_wrapper

    return decorator


def add_agent_timing(agent_name: str, state: ResearchState, elapsed: float) -> ResearchState:
    errors = state.get("errors", []) if isinstance(state, dict) else []
    status = "completed with errors" if errors else "completed"
    print(f"[{agent_name}] {status} in {elapsed:.2f}s")
    timing = {"agent": agent_name, "status": status, "elapsed_seconds": round(elapsed, 2)}
    return {**state, "agent_timings": [*state.get("agent_timings", []), timing]}


@log_agent_step("planner")
def planner_node(state: ResearchState) -> ResearchState:
    """Create a research plan and write it to shared memory."""

    objective = (state.get("objective") or "").strip()
    if not objective:
        return {"errors": ["objective is required"]}

    memory_path = state.get("memory_path") or "data/shared_memory.json"
    model = clean_text(state.get("model"))
    planner = PlannerAgent(use_llm=True, model=model or None)
    plan = planner.plan(objective)
    plan_dict = plan.to_dict()
    errors = validate_research_plan(plan_dict, objective)
    if errors:
        return {
            **state,
            "objective": plan.objective,
            "research_plan": plan_dict,
            "memory_path": memory_path,
            "model": planner.model,
            "errors": errors,
        }

    planner.write_to_memory(plan, memory_path)

    return {
        "objective": plan.objective,
        "research_plan": plan_dict,
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


@log_agent_step("browser")
async def browser_node(state: ResearchState) -> ResearchState:
    """Read research plan from state or shared memory, then run browser tasks."""

    memory_path = state.get("memory_path") or "data/shared_memory.json"
    if state.get("errors"):
        return {**state, "memory_path": memory_path}

    plan = read_research_plan_from_memory(memory_path)
    plan_errors = validate_research_plan(plan, clean_text(state.get("objective")))
    if plan_errors:
        return {
            **state,
            "memory_path": memory_path,
            "errors": [*state.get("errors", []), *plan_errors],
        }

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
    result_errors = validate_browser_results(results)

    memory = SharedMemory(memory_path)
    memory.write_agent_output("browser", {"results": results})

    return {
        **state,
        "research_plan": plan,
        "browser_results": results,
        "memory_path": memory_path,
        "errors": [*state.get("errors", []), *result_errors],
    }


@log_agent_step("change_detection")
def change_detection_node(state: ResearchState) -> ResearchState:
    """Compare browser results, persist the diff, and index current content for RAG."""

    memory_path = state.get("memory_path") or "data/shared_memory.json"
    history_db_path = state.get("history_db_path") or "data/browser_history.db"
    chroma_path = state.get("chroma_path") or "data/chroma"
    if state.get("errors"):
        return {**state, "memory_path": memory_path, "history_db_path": history_db_path, "chroma_path": chroma_path}

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
    diff_errors = validate_change_detection(diff)
    change_detector.write_to_memory(diff, memory_path)
    rag_index = index_rag_after_change_detection(
        browser_results=current_results,
        research_plan=plan,
        change_detection=diff,
        memory_path=memory_path,
        chroma_path=chroma_path,
    )
    errors = [*state.get("errors", []), *diff_errors, *validate_rag_index(rag_index)]
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


@log_agent_step("synthesis")
def synthesis_node(state: ResearchState) -> ResearchState:
    """Retrieve indexed evidence and write report-ready synthesis to memory."""

    memory_path = state.get("memory_path") or "data/shared_memory.json"
    chroma_path = state.get("chroma_path") or "data/chroma"
    if state.get("errors"):
        return {**state, "memory_path": memory_path, "chroma_path": chroma_path}

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

    synthesis_errors = validate_synthesis_payload(synthesis, plan)
    if synthesis_errors:
        return {
            **state,
            "synthesis": synthesis,
            "memory_path": memory_path,
            "chroma_path": chroma_path,
            "errors": [*state.get("errors", []), *synthesis_errors],
        }

    synthesis_agent.write_to_memory(synthesis, memory_path)
    return {
        **state,
        "synthesis": synthesis,
        "memory_path": memory_path,
        "chroma_path": chroma_path,
        "errors": state.get("errors", []),
    }


@log_agent_step("report")
def report_node(state: ResearchState) -> ResearchState:
    """Generate the final report from synthesis output."""

    memory_path = state.get("memory_path") or "data/shared_memory.json"
    if state.get("errors"):
        return {**state, "memory_path": memory_path}

    report_context = state.get("synthesis")
    if not report_context:
        memory = SharedMemory(memory_path)
        report_context = memory.read_agent_output("synthesis").get("report_context", {})
    if not report_context:
        return {
            **state,
            "memory_path": memory_path,
            "errors": [*state.get("errors", []), "report_node requires synthesis.report_context"],
        }

    research_plan = state.get("research_plan") or read_research_plan_from_memory(memory_path)
    context_errors = validate_synthesis_payload(report_context, research_plan)
    if context_errors:
        return {
            **state,
            "memory_path": memory_path,
            "errors": [*state.get("errors", []), *context_errors],
        }

    report_agent = ReportAgent(model=state.get("model"))
    try:
        report = report_agent.generate(
            report_context,
            output_format=clean_text(research_plan.get("output_format")) or "report",
        )
    except Exception as error:
        return {
            **state,
            "memory_path": memory_path,
            "errors": [*state.get("errors", []), clean_text(error)],
        }

    report_errors = validate_report_payload(report, research_plan)
    if report_errors:
        return {
            **state,
            "report": report,
            "memory_path": memory_path,
            "errors": [*state.get("errors", []), *report_errors],
        }

    report_agent.write_to_memory(report, memory_path)
    return {
        **state,
        "report": report,
        "memory_path": memory_path,
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


def validate_research_plan(plan: dict[str, Any], expected_objective: str = "") -> list[str]:
    errors = []
    if not isinstance(plan, dict) or not plan:
        return ["planner_node produced empty research_plan"]
    if not clean_text(plan.get("objective")):
        errors.append("planner_node research_plan.objective is required")
    if expected_objective and clean_text(plan.get("objective")).lower() != clean_text(expected_objective).lower():
        errors.append("planner_node research_plan.objective does not match requested objective")
    if not plan.get("sub_questions"):
        errors.append("planner_node research_plan.sub_questions is required")
    tasks = plan.get("tasks") or []
    if not isinstance(tasks, list) or not tasks:
        errors.append("planner_node research_plan.tasks is required")
    for index, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            errors.append(f"planner_node task {index} is not an object")
            continue
        for key in ("query_context", "url", "source_type", "extraction_goal"):
            if not clean_text(task.get(key)):
                errors.append(f"planner_node task {index}.{key} is required")
    return errors


def validate_browser_results(results: list[dict[str, Any]]) -> list[str]:
    if not isinstance(results, list) or not results:
        return ["browser_node produced no browser_results"]
    usable_sources = 0
    for result in results:
        if not isinstance(result, dict):
            continue
        for source in result.get("sources", []) or []:
            if clean_text(source.get("url")) and clean_text(source.get("full_content")):
                usable_sources += 1
    return [] if usable_sources else ["browser_node produced no useful sources"]


def validate_change_detection(diff: dict[str, Any]) -> list[str]:
    if not isinstance(diff, dict) or not diff:
        return ["change_detection_node produced empty diff"]
    return [] if clean_text(diff.get("summary")) else ["change_detection_node diff.summary is required"]


def validate_rag_index(index: dict[str, Any]) -> list[str]:
    if not isinstance(index, dict) or not index:
        return ["rag_index produced empty index summary"]
    if clean_text(index.get("status")) != "success":
        return [f"rag_index status is {clean_text(index.get('status')) or 'missing'}"]
    try:
        indexed_chunks = int(index.get("indexed_chunks") or 0)
    except (TypeError, ValueError):
        indexed_chunks = 0
    if indexed_chunks <= 0:
        return ["rag_index indexed zero chunks"]
    return []


def validate_synthesis_payload(payload: dict[str, Any], research_plan: dict[str, Any]) -> list[str]:
    errors = []
    if not isinstance(payload, dict) or not payload:
        return ["synthesis_node produced empty synthesis payload"]
    if not clean_text(payload.get("objective")):
        errors.append("synthesis_node payload.objective is required")
    plan_objective = clean_text(research_plan.get("objective")) if isinstance(research_plan, dict) else ""
    if plan_objective and clean_text(payload.get("objective")).lower() != plan_objective.lower():
        errors.append("synthesis_node payload.objective does not match research_plan.objective")
    if not clean_text(payload.get("synthesis")):
        errors.append("synthesis_node payload.synthesis is required")
    if not isinstance(payload.get("sources"), list) or not payload.get("sources"):
        errors.append("synthesis_node payload.sources is required")
    return errors


def validate_report_payload(payload: dict[str, Any], research_plan: dict[str, Any]) -> list[str]:
    errors = []
    if not isinstance(payload, dict) or not payload:
        return ["report_node produced empty report payload"]
    plan_objective = clean_text(research_plan.get("objective")) if isinstance(research_plan, dict) else ""
    if plan_objective and clean_text(payload.get("objective")).lower() != plan_objective.lower():
        errors.append("report_node payload.objective does not match research_plan.objective")
    report = clean_text(payload.get("report"))
    if not report:
        errors.append("report_node payload.report is required")
    if "references" not in report.lower():
        errors.append("report_node report must include a References section")
    return errors

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
    graph.add_node("report", report_node)
    graph.add_edge("change_detection", "synthesis")
    graph.add_edge("synthesis", "report")
    graph.add_edge("report", END)
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
