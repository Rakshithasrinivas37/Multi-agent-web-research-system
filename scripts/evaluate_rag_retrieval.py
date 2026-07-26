import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retrieval import (
    display_document_preview,
    evaluate_retrieval,
    hybrid_retrieve,
    metrics_to_dicts,
    retrieval_results_to_dicts,
)
from src.rag.indexing import select_embedding_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hybrid RAG retrieval with RAGAS ID metrics plus MRR.")
    parser.add_argument("--query", default="", help="Single query to evaluate.")
    parser.add_argument("--eval-file", type=Path, default=None, help="JSON file containing evaluation cases.")
    parser.add_argument("--relevant-id", action="append", default=[], help="Relevant Chroma chunk ID. Can be repeated.")
    parser.add_argument("--relevant-url", action="append", default=[], help="Relevant source URL. Can be repeated.")
    parser.add_argument("--k", default="1,3,5,10", help="Comma-separated k values.")
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "chroma",
        help="Persistent ChromaDB directory.",
    )
    parser.add_argument("--collection-name", default="research_rag", help="Chroma collection name.")
    parser.add_argument("--history-key", default="", help="Optional history key filter.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to retrieve for evaluation.")
    parser.add_argument("--semantic-k", type=int, default=30, help="Number of semantic search candidates.")
    parser.add_argument("--bm25-k", type=int, default=30, help="Number of BM25 keyword search candidates.")
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
        help="Embedding/reranker device: cuda, cpu, mps, or auto. Defaults to RAG_EMBEDDING_DEVICE/auto.",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON output.")
    return parser.parse_args()


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    selected_device = select_embedding_device(args.device or os.environ.get("RAG_EMBEDDING_DEVICE", "") or "auto")
    k_values = parse_k_values(args.k)
    cases = load_cases(args)
    if not cases:
        print("Provide --query with relevant labels, or --eval-file.")
        return 1

    outputs = []
    for case in cases:
        query = case["query"]
        results = hybrid_retrieve(
            query=query,
            chroma_path=args.chroma_path,
            collection_name=args.collection_name,
            top_k=max(args.top_k, max(k_values)),
            semantic_k=args.semantic_k,
            bm25_k=args.bm25_k,
            history_key=case.get("history_key") or args.history_key,
            semantic_weight=args.semantic_weight,
            bm25_weight=args.bm25_weight,
            authority_weight=args.authority_weight,
            diversify_urls=not args.no_diversify,
            embedding_device=selected_device,
            rerank=args.rerank,
            reranker_model=args.reranker_model,
            rerank_k=args.rerank_k,
            rerank_weight=args.rerank_weight,
        )
        metrics = evaluate_retrieval(
            retrieved=results,
            relevant_ids=set(case.get("relevant_ids", [])),
            relevant_urls=set(case.get("relevant_urls", [])),
            k_values=k_values,
        )
        outputs.append(
            {
                "query": query,
                "relevant_ids": case.get("relevant_ids", []),
                "relevant_urls": case.get("relevant_urls", []),
                "metrics": metrics_to_dicts(metrics),
                "results": retrieval_results_to_dicts(results),
            }
        )

    if args.json:
        print(json.dumps({"cases": outputs}, indent=2))
        return 0

    for output in outputs:
        print(f"\nQuery: {output['query']}")
        print_evaluation_metric_scores(output["metrics"])
        print_retrieved_context_scores(output)
    return 0


def print_evaluation_metric_scores(metrics: list[dict[str, Any]]) -> None:
    print("RAGAS evaluation metric scores:")
    for metric in metrics:
        print(f"  k={metric['k']}")
        print(f"    RAGAS ID Context Recall@k: {metric['recall_at_k']:.4f}")
        print(f"    RAGAS ID Context Precision@k: {metric['precision_at_k']:.4f}")
        print(f"    MRR@k: {metric['mrr']:.4f}")


def print_retrieved_context_scores(output: dict[str, Any]) -> None:
    relevant_ids = set(output.get("relevant_ids", []))
    relevant_urls = set(output.get("relevant_urls", []))
    print("Retrieved context scores:")
    for rank, result in enumerate(output["results"], start=1):
        metadata = result.get("metadata", {})
        url = str(metadata.get("url", ""))
        is_relevant = result.get("id") in relevant_ids or url in relevant_urls
        print(
            f"  [{rank}] "
            f"relevant={str(is_relevant).lower()} "
            f"hybrid={result['score']:.4f} "
            f"semantic={result['semantic_score']:.4f} "
            f"bm25={result['bm25_score']:.4f} "
            f"authority={result.get('authority_score', 0.0):.4f} "
            f"rerank={result.get('rerank_score', 0.0):.4f} "
            f"semantic_rank={result.get('semantic_rank')} "
            f"bm25_rank={result.get('bm25_rank')}"
        )
        print(f"      id: {result.get('id', '')}")
        print(f"      url: {url}")
        preview = display_document_preview(str(result.get("document", "")), max_chars=220).replace("\n", " ")
        if preview:
            print(f"      preview: {preview}")


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.eval_file:
        with args.eval_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            payload = payload.get("cases", [])
        return [normalize_case(item) for item in payload if isinstance(item, dict)]

    if args.query:
        return [
            {
                "query": args.query,
                "relevant_ids": args.relevant_id,
                "relevant_urls": args.relevant_url,
                "history_key": args.history_key,
            }
        ]
    return []


def normalize_case(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": str(item.get("query", "")),
        "relevant_ids": list_values(item.get("relevant_ids") or item.get("relevant_id")),
        "relevant_urls": list_values(item.get("relevant_urls") or item.get("relevant_url")),
        "history_key": str(item.get("history_key", "")),
    }


def list_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value:
        return [value]
    return []


def parse_k_values(value: str) -> list[int]:
    parsed = []
    for item in value.split(","):
        try:
            parsed.append(max(1, int(item.strip())))
        except ValueError:
            continue
    return parsed or [1, 3, 5, 10]


if __name__ == "__main__":
    raise SystemExit(main())
