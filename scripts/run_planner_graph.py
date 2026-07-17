import argparse
import json
from pathlib import Path
import sys

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.research_workflow import run_planner_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the planner node with LangGraph.")
    parser.add_argument(
        "--objective",
        default=None,
        help="Research objective. If omitted, the script asks for it.",
    )
    parser.add_argument(
        "--memory-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "shared_memory.json",
        help="Path to save shared memory JSON.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use deterministic fallback planning instead of Groq.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    args = parse_args()
    objective = args.objective or input("Research objective: ").strip()
    if not objective:
        print("Research objective is required.")
        return 1

    state = run_planner_graph(
        objective=objective,
        memory_path=str(args.memory_output),
        use_llm=not args.no_llm,
    )

    if state.get("errors"):
        print(json.dumps({"errors": state["errors"]}, indent=2))
        return 1

    print(f"Updated shared memory: {args.memory_output}")
    print(json.dumps(state.get("research_plan", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
