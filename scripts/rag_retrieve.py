import argparse
import json
import os
from pathlib import Path
import sys

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.indexing import select_embedding_device
from src.rag.retrieval import (
    display_document_preview,
    hybrid_retrieve,
    normalize_source_url,
    result_source_urls_from_metadata,
    retrieval_results_to_dicts,
)


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
    parser.add_argument("--semantic-weight", type=float, default=0.55, help="Hybrid weight for semantic scores.")
    parser.add_argument("--bm25-weight", type=float, default=0.35, help="Hybrid weight for BM25 scores.")
    parser.add_argument("--authority-weight", type=float, default=0.10, help="Hybrid weight for source authority scores.")
    parser.add_argument("--no-diversify", action="store_true", help="Do not diversify final results by URL.")
    parser.add_argument("--rerank", action="store_true", help="Rerank hybrid candidates with a CrossEncoder model.")
    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Sentence Transformers CrossEncoder model used when --rerank is enabled.",
    )
    parser.add_argument("--rerank-k", type=int, default=20, help="Number of hybrid candidates to rerank.")
    parser.add_argument("--rerank-weight", type=float, default=0.70, help="Final score weight assigned to reranker scores.")
    parser.add_argument(
        "--device",
        default="",
        help="Embedding device for semantic search: cuda, cpu, mps, or auto. Defaults to RAG_EMBEDDING_DEVICE/auto.",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON output.")
    return parser.parse_args()


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    requested_device = args.device or os.environ.get("RAG_EMBEDDING_DEVICE", "")
    selected_device = select_embedding_device(requested_device or "auto")
    if not args.json:
        print(f"embedding device: {selected_device}")
        if args.rerank:
            print(f"reranker model: {args.reranker_model}")

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
        authority_weight=args.authority_weight,
        embedding_device=selected_device,
        diversify_urls=not args.no_diversify,
        rerank=args.rerank,
        reranker_model=args.reranker_model,
        rerank_k=args.rerank_k,
        rerank_weight=args.rerank_weight,
    )

    if args.json:
        print(json.dumps({"query": args.query, "results": retrieval_results_to_dicts(results)}, indent=2))
        return 0

    for rank, result in enumerate(results, start=1):
        metadata = result.metadata
        print(
            f"\n[{rank}] score={result.score:.4f} "
            f"semantic={result.semantic_score:.4f} "
            f"bm25={result.bm25_score:.4f} "
            f"authority={result.authority_score:.4f} "
            f"rerank={result.rerank_score:.4f}"
        )
        print(f"id: {result.id}")
        print(f"title: {metadata.get('title', '')}")
        print(f"url: {metadata.get('url', '')}")
        alternate_urls = [
            item for item in result_source_urls_from_metadata(metadata)
            if item != normalize_source_url(metadata.get("url", ""))
        ]
        if alternate_urls:
            print(f"alternate_urls: {', '.join(alternate_urls)}")
        print(display_document_preview(result.document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
