"""LangGraph workflow for the multi-agent research system."""

from functools import wraps
from inspect import iscoroutinefunction
import os
import re
from time import perf_counter
import traceback
from typing import Any, Sequence, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.browser_agent import BrowserAgent
from src.agents.change_detection_agent import ChangeDetectionAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.report_agent import (
    ReportAgent,
    report_context_gap_items,
    report_context_gap_queries,
    rewrite_missing_sub_question_queries,
)
from src.agents.synthesis_agent import SynthesisAgent
from src.memory.shared_memory import SharedMemory
from src.rag import index_research_results
from src.tools.progress import emit_progress
from src.tools.text_utils import clean_text

DEFAULT_AGENT_RESPONSE_ATTEMPTS = 3
DEFAULT_REPORT_RESPONSE_ATTEMPTS = 1
DEFAULT_REPORT_GAP_SYNTHESIS_MODEL = "qwen/qwen3.6-27b"


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
            emit_progress("agent_started", f"{agent_name} started", agent=agent_name)
            try:
                result = await func(state)
            except Exception:
                elapsed = perf_counter() - started_at
                print(f"[{agent_name}] failed after {elapsed:.2f}s")
                emit_progress(
                    "agent_failed",
                    f"{agent_name} failed after {elapsed:.2f}s",
                    agent=agent_name,
                    metadata={"elapsed_seconds": round(elapsed, 2)},
                )
                raise
            return add_agent_timing(agent_name, result, elapsed=perf_counter() - started_at)

        @wraps(func)
        def sync_wrapper(state: ResearchState) -> ResearchState:
            started_at = perf_counter()
            emit_progress("agent_started", f"{agent_name} started", agent=agent_name)
            try:
                result = func(state)
            except Exception:
                elapsed = perf_counter() - started_at
                print(f"[{agent_name}] failed after {elapsed:.2f}s")
                emit_progress(
                    "agent_failed",
                    f"{agent_name} failed after {elapsed:.2f}s",
                    agent=agent_name,
                    metadata={"elapsed_seconds": round(elapsed, 2)},
                )
                raise
            return add_agent_timing(agent_name, result, elapsed=perf_counter() - started_at)

        return async_wrapper if iscoroutinefunction(func) else sync_wrapper

    return decorator


def add_agent_timing(agent_name: str, state: ResearchState, elapsed: float) -> ResearchState:
    errors = state.get("errors", []) if isinstance(state, dict) else []
    status = "failed" if errors else "completed"
    print(f"[{agent_name}] {status} in {elapsed:.2f}s")
    if errors:
        for error in errors:
            print(f"[{agent_name}] error:\n{format_error_text(error)}")
    timing = {"agent": agent_name, "status": status, "elapsed_seconds": round(elapsed, 2)}
    emit_progress(
        "agent_completed",
        f"{agent_name} {status} in {elapsed:.2f}s",
        agent=agent_name,
        metadata=timing,
    )
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
    plan = None
    plan_dict = {}
    errors = []
    for attempt in range(1, DEFAULT_AGENT_RESPONSE_ATTEMPTS + 1):
        try:
            plan = planner.plan(objective)
            plan_dict = plan.to_dict()
            errors = validate_research_plan(plan_dict, objective)
        except Exception as error:
            errors = [format_exception_details(error)]
        if not errors:
            break
        if attempt < DEFAULT_AGENT_RESPONSE_ATTEMPTS:
            print_retry_response_error("planner", attempt, errors)

    if errors:
        return {
            **state,
            "objective": plan.objective if plan else objective,
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
    synthesis = {}
    synthesis_errors = []
    for attempt in range(1, DEFAULT_AGENT_RESPONSE_ATTEMPTS + 1):
        try:
            synthesis = synthesis_agent.synthesize(plan, browser_results=state.get("browser_results", []))
            synthesis_errors = validate_synthesis_payload(synthesis, plan)
        except Exception as error:
            synthesis_errors = [format_exception_details(error)]
        if not synthesis_errors:
            break
        if attempt < DEFAULT_AGENT_RESPONSE_ATTEMPTS:
            print_retry_response_error("synthesis", attempt, synthesis_errors)

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
    chroma_path = state.get("chroma_path") or "data/chroma"
    if state.get("errors"):
        return {**state, "memory_path": memory_path, "chroma_path": chroma_path}

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
    report_context = report_context_with_browser_memory(report_context, state, memory_path)

    research_plan = state.get("research_plan") or read_research_plan_from_memory(memory_path)
    context_errors = validate_synthesis_payload(report_context, research_plan)
    if context_errors:
        return {
            **state,
            "memory_path": memory_path,
            "errors": [*state.get("errors", []), *context_errors],
        }

    report_agent = ReportAgent(model=state.get("model"))
    report = {}
    report_errors = []
    output_format = clean_text(research_plan.get("output_format")) or "report"
    for attempt in range(1, DEFAULT_REPORT_RESPONSE_ATTEMPTS + 1):
        try:
            preflight_gap_items = report_context_gap_items(report_context, research_plan)
            if preflight_gap_items and not report_context_gap_retry_used(report_context):
                report_context = refresh_synthesis_for_report_gaps(
                    state=state,
                    research_plan=research_plan,
                    report_context=report_context,
                    missing_questions=preflight_gap_items,
                    retry_queries=report_context_gap_queries(report_context, research_plan),
                    memory_path=memory_path,
                    chroma_path=chroma_path,
                )
            report = report_agent.generate(report_context, output_format=output_format)
            missing_questions = report_missing_sub_questions(report)
            if missing_questions and not report_context_gap_retry_used(report_context):
                report_diagnostics = report.get("diagnostics", {}) if isinstance(report, dict) else {}
                report_context = refresh_synthesis_for_report_gaps(
                    state=state,
                    research_plan=research_plan,
                    report_context=report_context,
                    missing_questions=missing_questions,
                    retry_queries=report_diagnostics.get("report_retry_queries", []),
                    memory_path=memory_path,
                    chroma_path=chroma_path,
                )
                report = report_agent.generate(report_context, output_format=output_format)
            report_errors = report_missing_question_errors(report)
        except Exception as error:
            report_errors = [format_exception_details(error)]
        else:
            report_errors = report_errors or validate_report_payload(report, research_plan)
            if not report_errors:
                break

        if attempt < DEFAULT_REPORT_RESPONSE_ATTEMPTS:
            print_retry_response_error("report", attempt, report_errors)

    if report_errors:
        return {
            **state,
            "report": report,
            "memory_path": memory_path,
            "chroma_path": chroma_path,
            "errors": [*state.get("errors", []), *report_errors],
        }

    report_agent.write_to_memory(report, memory_path)
    return {
        **state,
        "report": report,
        "memory_path": memory_path,
        "chroma_path": chroma_path,
        "synthesis": report_context,
        "errors": state.get("errors", []),
    }


def report_missing_sub_questions(report: dict[str, Any]) -> list[str]:
    diagnostics = report.get("diagnostics", {}) if isinstance(report, dict) else {}
    questions = diagnostics.get("report_missing_sub_questions", []) if isinstance(diagnostics, dict) else []
    return [clean_text(question) for question in questions if clean_text(question)]


def report_missing_question_errors(report: dict[str, Any]) -> list[str]:
    missing_questions = report_missing_sub_questions(report)
    if not missing_questions:
        return []
    missing_text = "; ".join(missing_questions[:3])
    return [f"report_node report does not answer planner sub-questions: {missing_text}"]


def report_context_gap_retry_used(report_context: dict[str, Any]) -> bool:
    diagnostics = report_context.get("diagnostics", {}) if isinstance(report_context, dict) else {}
    return bool(isinstance(diagnostics, dict) and diagnostics.get("report_gap_retry"))


def report_context_with_browser_memory(
    report_context: dict[str, Any],
    state: ResearchState,
    memory_path: str,
) -> dict[str, Any]:
    if not isinstance(report_context, dict) or report_context.get("browser_results"):
        return report_context
    browser_results = state.get("browser_results")
    if not browser_results:
        browser_results = SharedMemory(memory_path).read_agent_output("browser").get("results", [])
    if not browser_results:
        return report_context
    return {**report_context, "browser_results": browser_results}


def refresh_synthesis_for_report_gaps(
    state: ResearchState,
    research_plan: dict[str, Any],
    report_context: dict[str, Any],
    missing_questions: Sequence[str],
    retry_queries: Sequence[str],
    memory_path: str,
    chroma_path: str,
) -> dict[str, Any]:
    """Rerun synthesis with rewritten queries for report coverage gaps."""

    if not retry_queries:
        retry_queries = rewrite_missing_sub_question_queries(
            clean_text(research_plan.get("objective")),
            missing_questions,
        )
    print(f"[report] refreshing synthesis for {len(missing_questions)} coverage gap(s)")
    gap_plan = research_plan_for_report_gaps(research_plan, missing_questions, retry_queries)
    synthesis_agent = SynthesisAgent(model=report_gap_synthesis_model(), chroma_path=chroma_path)
    refreshed = synthesis_agent.synthesize(gap_plan)
    merged = merge_report_context(report_context, refreshed)
    synthesis_agent.write_to_memory(merged, memory_path)
    return merged


def report_gap_synthesis_model() -> str:
    return clean_text(os.environ.get("REPORT_GAP_SYNTHESIS_MODEL")) or DEFAULT_REPORT_GAP_SYNTHESIS_MODEL


def research_plan_for_report_gaps(
    research_plan: dict[str, Any],
    missing_questions: Sequence[str],
    retry_queries: Sequence[str],
) -> dict[str, Any]:
    objective = clean_text(research_plan.get("objective"))
    questions = [clean_text(question) for question in missing_questions if clean_text(question)]
    queries = [clean_text(query) for query in retry_queries if clean_text(query)] or questions
    tasks = [
        {
            "query_context": question,
            "url": clean_text(queries[index]) if index < len(queries) else question,
            "source_type": "rag",
            "extraction_goal": f"Retrieve evidence needed to answer: {question}",
        }
        for index, question in enumerate(questions)
    ]
    return {
        **research_plan,
        "objective": objective,
        "sub_questions": questions,
        "tasks": tasks,
        "synthesis_instruction": (
            "Retrieve and synthesize only the evidence needed to answer the report's missing planner "
            "sub-questions. Preserve citations and mark gaps instead of guessing."
        ),
    }


def merge_report_context(original: dict[str, Any], refreshed: dict[str, Any]) -> dict[str, Any]:
    """Append targeted synthesis evidence to the original report context."""

    original_synthesis = clean_text(original.get("synthesis"))
    sources, citation_map = append_reindexed_sources(
        original.get("sources", []),
        refreshed.get("sources", []),
    )
    refreshed_synthesis = remap_report_gap_citations(clean_text(refreshed.get("synthesis")), citation_map)
    synthesis = clean_text(
        f"{original_synthesis}\n\n## Targeted Evidence Refresh\n{refreshed_synthesis}"
    )
    chunks = list(original.get("supporting_chunks", []) or [])
    chunks.extend(reindex_supporting_chunks(refreshed.get("supporting_chunks", []) or [], citation_map))
    diagnostics = {
        **(original.get("diagnostics", {}) if isinstance(original.get("diagnostics"), dict) else {}),
        "report_gap_retry": True,
        "report_gap_retry_queries": refreshed.get("retrieval_queries", []),
    }
    return {
        **original,
        "synthesis": synthesis,
        "sources": sources,
        "supporting_chunks": chunks,
        "diagnostics": diagnostics,
    }


def append_reindexed_sources(
    original_sources: Sequence[dict[str, Any]],
    refreshed_sources: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    merged = [dict(source) for source in original_sources or [] if isinstance(source, dict)]
    existing_by_url = {clean_text(source.get("url")).lower(): source_index(source, index) for index, source in enumerate(merged, start=1)}
    next_index = max([source_index(source, index) for index, source in enumerate(merged, start=1)] or [0]) + 1
    citation_map = {}
    for fallback_index, source in enumerate(refreshed_sources or [], start=1):
        if not isinstance(source, dict):
            continue
        old_index = source_index(source, fallback_index)
        url_key = clean_text(source.get("url")).lower()
        if url_key and url_key in existing_by_url:
            citation_map[old_index] = existing_by_url[url_key]
            continue
        copied = dict(source)
        copied["index"] = next_index
        citation_map[old_index] = next_index
        merged.append(copied)
        if url_key:
            existing_by_url[url_key] = next_index
        next_index += 1
    return merged, citation_map


def source_index(source: dict[str, Any], fallback: int) -> int:
    try:
        return int(source.get("index") or fallback)
    except (TypeError, ValueError):
        return fallback


def remap_report_gap_citations(text: str, citation_map: dict[int, int]) -> str:
    if not citation_map:
        return text

    def replace(match: re.Match[str]) -> str:
        old_index = int(match.group(1))
        return f"[{citation_map.get(old_index, old_index)}]"

    return re.sub(r"\[(\d+)\]", replace, text)


def reindex_supporting_chunks(chunks: Sequence[dict[str, Any]], citation_map: dict[int, int]) -> list[dict[str, Any]]:
    reindexed = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        copied = dict(chunk)
        old_index = copied.get("source_index")
        try:
            copied["source_index"] = citation_map.get(int(old_index), old_index)
        except (TypeError, ValueError):
            pass
        reindexed.append(copied)
    return reindexed


def print_retry_response_error(agent_name: str, attempt: int, errors: Sequence[str]) -> None:
    error_text = "\n---\n".join(format_error_text(error) for error in errors if format_error_text(error))
    print(f"[{agent_name}] retrying after response error ({attempt}/{DEFAULT_AGENT_RESPONSE_ATTEMPTS}):\n{error_text}")


def format_exception_details(error: Exception) -> str:
    details = "".join(traceback.format_exception(type(error), error, error.__traceback__)).strip()
    return details or f"{type(error).__name__}: {error}"


def format_error_text(error: Any) -> str:
    return str(error or "").strip()


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
    try:
        stored_chunks = int(index.get("stored_chunks") or 0)
    except (TypeError, ValueError):
        stored_chunks = 0
    if indexed_chunks <= 0 and stored_chunks <= 0:
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


def next_node_or_end(next_node: str):
    """Route to the next node only when the previous node completed without errors."""

    def route(state: ResearchState) -> str:
        return END if state.get("errors") else next_node

    return route


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
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("report", report_node)
    graph.add_conditional_edges("planner", next_node_or_end("browser"))
    graph.add_conditional_edges("browser", next_node_or_end("change_detection"))
    graph.add_conditional_edges("change_detection", next_node_or_end("synthesis"))
    graph.add_conditional_edges("synthesis", next_node_or_end("report"))
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
