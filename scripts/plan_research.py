import argparse
import json
from pathlib import Path
import sys


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
        help="Path to save the JSON plan.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use deterministic fallback planning instead of Groq.",
    )
    return parser.parse_args()


def main() -> int:
    """Generate and save a structured research plan."""

    args = parse_args()
    objective = args.objective or input("Research objective: ").strip()
    if not objective:
        print("Research objective is required.")
        return 1

    planner = PlannerAgent(use_llm=not args.no_llm)
    plan = planner.plan(objective)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

    print(f"Saved research plan: {args.output}")
    print(json.dumps(plan.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
