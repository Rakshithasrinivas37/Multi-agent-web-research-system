import argparse
import asyncio
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

from src.agents.browser_agent import BrowserAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run browser agent on a research plan.")
    parser.add_argument(
        "--plan",
        type=Path,
        default=PROJECT_ROOT / "data" / "research_plan.json",
        help="Path to research_plan.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "browser_results.json",
        help="Path to save browser results.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=3,
        help="Number of tasks to process in parallel.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip Groq extraction and save raw fallback snippets.",
    )
    return parser.parse_args()


async def async_main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))

    agent = BrowserAgent(
        max_concurrency=args.max_concurrency,
        use_llm=not args.no_llm,
    )
    tasks = plan.get("tasks") or plan.get("subtasks") or []
    if not tasks:
        print("No tasks found. Expected 'tasks' or 'subtasks' in the research plan.")
        return 1

    results = await agent.run_tasks(tasks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved browser results: {args.output}")
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
