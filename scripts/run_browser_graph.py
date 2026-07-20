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

from src.graph.research_workflow import browser_node


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the browser node from shared memory.")
    parser.add_argument(
        "--memory",
        type=Path,
        default=PROJECT_ROOT / "data" / "shared_memory.json",
        help="Path to shared memory JSON.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=3,
        help="Number of tasks to process in parallel.",
    )
    return parser.parse_args()


async def async_main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    args = parse_args()
    state = await browser_node(
        {
            "memory_path": str(args.memory),
            "max_concurrency": args.max_concurrency,
        }
    )

    if state.get("errors"):
        print(json.dumps({"errors": state["errors"]}, indent=2))
        return 1

    print(f"Updated shared memory: {args.memory}")
    print(json.dumps(state.get("browser_results", []), indent=2))
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
