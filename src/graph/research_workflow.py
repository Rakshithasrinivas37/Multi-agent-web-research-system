"""LangGraph workflow for the multi-agent research system."""

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.planner_agent import PlannerAgent


class ResearchState(TypedDict, total=False):
    objective: str
    research_plan: dict[str, Any]
    memory_path: str
    use_llm: bool
    errors: list[str]


def planner_node(state: ResearchState) -> ResearchState:
    """Create a research plan and write it to shared memory."""

    objective = (state.get("objective") or "").strip()
    if not objective:
        return {"errors": ["objective is required"]}

    memory_path = state.get("memory_path") or "data/shared_memory.json"
    planner = PlannerAgent(use_llm=state.get("use_llm", True))
    plan = planner.plan(objective)
    planner.write_to_memory(plan, memory_path)

    return {
        "objective": plan.objective,
        "research_plan": plan.to_dict(),
        "memory_path": memory_path,
        "errors": [],
    }


def build_research_graph():
    """Build the current graph with planner as the first node."""

    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", END)
    return graph.compile()


def run_planner_graph(objective: str, memory_path: str = "data/shared_memory.json", use_llm: bool = True) -> ResearchState:
    """Run the LangGraph workflow through the planner node."""

    graph = build_research_graph()
    return graph.invoke(
        {
            "objective": objective,
            "memory_path": memory_path,
            "use_llm": use_llm,
        }
    )
