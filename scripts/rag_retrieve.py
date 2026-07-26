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

from src.rag.retrieval import hybrid_retrieve, retrieval_results_to_dicts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid RAG retrieval against the Chroma store.")
    parser.add_argument("--query", required=True, help="Question or search query.")
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "chroma",
        help="Persistent ChromaDB directory.",
    )
    parser.add_argument("--collection-name", default="research_rag", help="Chroma collection name.")
    parser.add_argument("--history-key", default="", help="Optional history key filter.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of final hybrid results.")
    parser.add_argument("--semantic-k", type=int, default=20, help="Number of semantic search candidates.")
    parser.add_argument("--bm25-k", type=int, default=20, help="Number of BM25 keyword search candidates.")
    parser.add_argument("--semantic-weight", type=float, default=0.7, help="Hybrid weight for semantic scores.")
    parser.add_argument("--bm25-weight", type=float, default=0.3, help="Hybrid weight for BM25 scores.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON output.")
    return parser.parse_args()


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    results = hybrid_retrieve(
        query=args.query,
        chroma_path=args.chroma_path,
        collection_name=args.collection_name,
        top_k=args.top_k,
        semantic_k=args.semantic_k,
        bm25_k=args.bm25_k,
        history_key=args.history_key,
        semantic_weight=args.semantic_weight,
        bm25_weight=args.bm25_weight,
    )

    if args.json:
        print(json.dumps({"query": args.query, "results": retrieval_results_to_dicts(results)}, indent=2))
        return 0

    for rank, result in enumerate(results, start=1):
        metadata = result.metadata
        print(f"\n[{rank}] score={result.score:.4f} semantic={result.semantic_score:.4f} bm25={result.bm25_score:.4f}")
        print(f"id: {result.id}")
        print(f"title: {metadata.get('title', '')}")
        print(f"url: {metadata.get('url', '')}")
        print(result.document[:700].strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
