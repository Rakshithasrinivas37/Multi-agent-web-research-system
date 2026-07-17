import argparse
import json
from pathlib import Path
import sys

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs) -> bool:
        """No-op when python-dotenv is not installed."""

        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.planner_agent import PlannerAgent


def parse_args() -> argparse.Namespace:
    """Parse planner CLI arguments."""

    parser = argparse.ArgumentParser(description="Generate a research plan.")
    parser.add_argument(
        "--objective",
        default=None,
        help="Research objective. If omitted, the script asks for it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "research_plan.json",
        help="Path to save the JSON plan when --save-plan is used.",
    )
    parser.add_argument(
        "--save-plan",
        action="store_true",
        help="Also save the planner output as a standalone JSON plan file.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use deterministic fallback planning instead of Groq.",
    )
    parser.add_argument(
        "--memory-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "shared_memory.json",
        help="Path to save shared memory JSON.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Do not write planner output to shared memory.",
    )
    return parser.parse_args()


def main() -> int:
    """Generate and save a structured research plan."""

    load_dotenv(PROJECT_ROOT / ".env")

    args = parse_args()
    objective = args.objective or input("Research objective: ").strip()
    if not objective:
        print("Research objective is required.")
        return 1

    planner = PlannerAgent(use_llm=not args.no_llm)
    plan = planner.plan(objective)

    if args.save_plan:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    if not args.no_memory:
        planner.write_to_memory(plan, str(args.memory_output))

    if args.save_plan:
        print(f"Saved research plan: {args.output}")
    if not args.no_memory:
        print(f"Updated shared memory: {args.memory_output}")
    print(json.dumps(plan.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
