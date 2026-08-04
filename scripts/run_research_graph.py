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

from src.graph.research_workflow import run_research_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run planner and browser nodes with LangGraph.")
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
        "--history-db",
        type=Path,
        default=PROJECT_ROOT / "data" / "browser_history.db",
        help="SQLite database used for previous browser results.",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "chroma",
        help="Persistent ChromaDB directory used for RAG indexing.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=3,
        help="Number of browser tasks to process in parallel.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Groq model for planner and synthesis. Defaults to RESEARCH_PLANNER_MODEL.",
    )
    return parser.parse_args()


async def async_main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    args = parse_args()
    objective = args.objective or input("Research objective: ").strip()
    if not objective:
        print("Research objective is required.")
        return 1

    state = await run_research_graph(
        objective=objective,
        memory_path=str(args.memory_output),
        history_db_path=str(args.history_db),
        chroma_path=str(args.chroma_path),
        max_concurrency=args.max_concurrency,
        model=args.model,
    )

    if state.get("errors"):
        print(json.dumps({"errors": state["errors"]}, indent=2))
        return 1

    print(f"Updated shared memory: {args.memory_output}")
    print(f"Updated browser history DB: {args.history_db}")
    print(f"Updated ChromaDB store: {args.chroma_path}")
    print(
        json.dumps(
            {
                "research_plan": state.get("research_plan"),
                "browser_results": state.get("browser_results"),
                "change_detection": state.get("change_detection"),
                "rag_index": state.get("rag_index"),
                "synthesis": state.get("synthesis"),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
