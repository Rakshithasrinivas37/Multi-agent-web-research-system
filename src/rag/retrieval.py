"""Hybrid RAG retrieval using semantic search and BM25 keyword search."""

from __future__ import annotations

import asyncio
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from src.rag.indexing import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    SentenceTransformerEmbeddingFunction,
    chunk_id,
    get_collection,
    select_embedding_device,
    to_int,
)
from src.tools.text_utils import clean_text


DEFAULT_TOP_K = 5
DEFAULT_SEMANTIC_K = 20
DEFAULT_BM25_K = 20
DEFAULT_SEMANTIC_WEIGHT = 0.7
DEFAULT_BM25_WEIGHT = 0.3
DEFAULT_BM25_SCAN_LIMIT = 1000
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


@dataclass(frozen=True)
class RetrievalResult:
    id: str
    document: str
    metadata: dict[str, Any]
    score: float
    semantic_score: float
    bm25_score: float
    semantic_rank: Optional[int] = None
    bm25_rank: Optional[int] = None


@dataclass(frozen=True)
class RetrievalMetrics:
    k: int
    retrieved: int
    relevant: int
    relevant_retrieved: int
    recall_at_k: float
    precision_at_k: float
    mrr: float


class LangChainSentenceTransformerEmbeddings:
    """LangChain embeddings adapter using the same model/device as indexing."""

    def __init__(self, device: str = "") -> None:
        self.embedding_function = SentenceTransformerEmbeddingFunction(device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedding_function(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embedding_function([text])[0]


def hybrid_retrieve(
    query: str,
    chroma_path: Union[str, Path] = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = DEFAULT_TOP_K,
    semantic_k: int = DEFAULT_SEMANTIC_K,
    bm25_k: int = DEFAULT_BM25_K,
    history_key: str = "",
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    bm25_scan_limit: int = DEFAULT_BM25_SCAN_LIMIT,
    embedding_device: str = "",
) -> list[RetrievalResult]:
    """Retrieve chunks using semantic vector similarity plus BM25 keyword matching."""

    query = clean_text(query)
    if not query:
        return []

    top_k = max(1, top_k)
    semantic_k = max(top_k, semantic_k)
    bm25_k = max(top_k, bm25_k)
    where = metadata_filter(history_key)
    collection = get_collection(chroma_path, collection_name)

    semantic_results = semantic_search(
        query=query,
        chroma_path=chroma_path,
        collection_name=collection_name,
        semantic_k=semantic_k,
        where=where,
        embedding_device=embedding_device,
    )
    bm25_results = bm25_search(
        collection,
        query,
        bm25_k=bm25_k,
        where=where,
        scan_limit=max(bm25_scan_limit, bm25_k),
    )

    merged = merge_ranked_results(
        semantic_results=semantic_results,
        bm25_results=bm25_results,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight,
    )
    return merged[:top_k]


def semantic_search(
    query: str,
    chroma_path: Union[str, Path],
    collection_name: str,
    semantic_k: int,
    where: Optional[dict[str, Any]] = None,
    embedding_device: str = "",
) -> list[RetrievalResult]:
    vector_store = get_langchain_chroma(
        chroma_path=chroma_path,
        collection_name=collection_name,
        embedding_device=embedding_device,
    )
    try:
        results = vector_store.similarity_search_with_relevance_scores(
            query,
            k=max(1, semantic_k),
            filter=where,
        )
    except Exception:
        return []

    rows = []
    for index, (document, score) in enumerate(results):
        rows.append(
            RetrievalResult(
                id=document_id(document),
                document=clean_text(document.page_content),
                metadata=document.metadata if isinstance(document.metadata, dict) else {},
                score=0.0,
                semantic_score=to_float(score, 0.0),
                bm25_score=0.0,
                semantic_rank=index + 1,
            )
        )
    return rows


def bm25_search(
    collection: Any,
    query: str,
    bm25_k: int,
    where: Optional[dict[str, Any]] = None,
    scan_limit: int = DEFAULT_BM25_SCAN_LIMIT,
) -> list[RetrievalResult]:
    get_args = {"include": ["documents", "metadatas"], "limit": max(1, scan_limit)}
    if where:
        get_args["where"] = where

    try:
        result = collection.get(**get_args)
    except Exception:
        return []

    ids = result.get("ids", []) if isinstance(result, dict) else []
    documents = result.get("documents", []) if isinstance(result, dict) else []
    metadatas = result.get("metadatas", []) if isinstance(result, dict) else []
    langchain_documents = build_langchain_documents(ids=ids, documents=documents, metadatas=metadatas)
    if not langchain_documents:
        return []

    try:
        BM25Retriever = langchain_bm25_retriever()
        retriever = BM25Retriever.from_documents(
            langchain_documents,
            k=max(1, bm25_k),
            preprocess_func=tokenize,
        )
        ranked_documents = retriever.invoke(query)
    except Exception:
        return []

    rows = []
    for bm25_rank, document in enumerate(ranked_documents[: max(1, bm25_k)], start=1):
        rows.append(
            RetrievalResult(
                id=document_id(document),
                document=clean_text(document.page_content),
                metadata=document.metadata if isinstance(document.metadata, dict) else {},
                score=0.0,
                semantic_score=0.0,
                bm25_score=1.0 / bm25_rank,
                bm25_rank=bm25_rank,
            )
        )
    return [row for row in rows if row.id]


def merge_ranked_results(
    semantic_results: Sequence[RetrievalResult],
    bm25_results: Sequence[RetrievalResult],
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
) -> list[RetrievalResult]:
    semantic_scores = normalize_scores({item.id: item.semantic_score for item in semantic_results})
    bm25_scores = normalize_scores({item.id: item.bm25_score for item in bm25_results})
    by_id: dict[str, RetrievalResult] = {}

    for item in list(semantic_results) + list(bm25_results):
        existing = by_id.get(item.id)
        by_id[item.id] = merge_result(existing, item) if existing else item

    merged = []
    for item_id, item in by_id.items():
        semantic_score = semantic_scores.get(item_id, 0.0)
        bm25_score = bm25_scores.get(item_id, 0.0)
        score = (semantic_weight * semantic_score) + (bm25_weight * bm25_score)
        merged.append(
            RetrievalResult(
                id=item.id,
                document=item.document,
                metadata=item.metadata,
                score=score,
                semantic_score=semantic_score,
                bm25_score=bm25_score,
                semantic_rank=item.semantic_rank,
                bm25_rank=item.bm25_rank,
            )
        )

    return sorted(merged, key=lambda item: item.score, reverse=True)


def evaluate_retrieval(
    retrieved: Sequence[RetrievalResult],
    relevant_ids: Optional[set[str]] = None,
    relevant_urls: Optional[set[str]] = None,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> list[RetrievalMetrics]:
    """Compute RAGAS ID-based Recall@k/Precision@k plus local MRR."""

    relevant_ids = {clean_text(item) for item in (relevant_ids or set()) if clean_text(item)}
    relevant_urls = {clean_text(item) for item in (relevant_urls or set()) if clean_text(item)}
    reference_context_ids = sorted(relevant_ids.union(relevant_urls))
    relevant_count = len(reference_context_ids)
    if relevant_count == 0:
        return [
            RetrievalMetrics(
                k=max(1, k),
                retrieved=min(len(retrieved), max(1, k)),
                relevant=0,
                relevant_retrieved=0,
                recall_at_k=0.0,
                precision_at_k=0.0,
                mrr=0.0,
            )
            for k in k_values
        ]

    metrics = []
    for k in k_values:
        k = max(1, int(k))
        top_results = list(retrieved[:k])
        retrieved_context_ids = [evaluation_context_id(result, relevant_urls) for result in top_results]
        relevant_flags = [is_relevant(item, relevant_ids, relevant_urls) for item in top_results]
        first_relevant_rank = next((index + 1 for index, flag in enumerate(relevant_flags) if flag), None)
        ragas_scores = score_ragas_id_metrics(
            retrieved_context_ids=retrieved_context_ids,
            reference_context_ids=reference_context_ids,
        )
        metrics.append(
            RetrievalMetrics(
                k=k,
                retrieved=len(top_results),
                relevant=relevant_count,
                relevant_retrieved=len(set(retrieved_context_ids).intersection(reference_context_ids)),
                recall_at_k=ragas_scores["recall"],
                precision_at_k=ragas_scores["precision"],
                mrr=(1.0 / first_relevant_rank) if first_relevant_rank else 0.0,
            )
        )
    return metrics


def retrieval_results_to_dicts(results: Sequence[RetrievalResult]) -> list[dict[str, Any]]:
    return [asdict(item) for item in results]


def metrics_to_dicts(metrics: Sequence[RetrievalMetrics]) -> list[dict[str, Any]]:
    return [asdict(item) for item in metrics]


def score_ragas_id_metrics(retrieved_context_ids: list[str], reference_context_ids: list[str]) -> dict[str, float]:
    SingleTurnSample, IDBasedContextPrecision, IDBasedContextRecall = ragas_id_metric_classes()
    sample = SingleTurnSample(
        retrieved_context_ids=retrieved_context_ids,
        reference_context_ids=reference_context_ids,
    )
    return {
        "precision": score_ragas_metric(IDBasedContextPrecision(), sample),
        "recall": score_ragas_metric(IDBasedContextRecall(), sample),
    }


def ragas_id_metric_classes() -> tuple[Any, Any, Any]:
    try:
        from ragas.dataset_schema import SingleTurnSample
    except ImportError:
        try:
            from ragas import SingleTurnSample
        except ImportError as error:
            raise RuntimeError(
                "ragas is not installed. Install it with `pip install ragas` or `pip install -r requirements.txt`."
            ) from error

    try:
        from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall
    except ImportError as error:
        raise RuntimeError(
            "RAGAS ID-based retrieval metrics are unavailable. Upgrade ragas with `pip install -U ragas`."
        ) from error
    return SingleTurnSample, IDBasedContextPrecision, IDBasedContextRecall


def score_ragas_metric(metric: Any, sample: Any) -> float:
    if hasattr(metric, "single_turn_score"):
        return ragas_metric_value(metric.single_turn_score(sample))
    if hasattr(metric, "score"):
        return ragas_metric_value(metric.score(sample))
    if hasattr(metric, "single_turn_ascore"):
        return ragas_metric_value(asyncio.run(metric.single_turn_ascore(sample)))
    raise RuntimeError(f"Unsupported RAGAS metric API for {type(metric).__name__}")


def ragas_metric_value(result: Any) -> float:
    if hasattr(result, "value"):
        return to_float(result.value, 0.0)
    return to_float(result, 0.0)


def metadata_filter(history_key: str) -> Optional[dict[str, Any]]:
    history_key = clean_text(history_key)
    return {"history_key": history_key} if history_key else None


def merge_result(existing: RetrievalResult, incoming: RetrievalResult) -> RetrievalResult:
    return RetrievalResult(
        id=existing.id,
        document=existing.document or incoming.document,
        metadata=existing.metadata or incoming.metadata,
        score=0.0,
        semantic_score=max(existing.semantic_score, incoming.semantic_score),
        bm25_score=max(existing.bm25_score, incoming.bm25_score),
        semantic_rank=existing.semantic_rank or incoming.semantic_rank,
        bm25_rank=existing.bm25_rank or incoming.bm25_rank,
    )


def get_langchain_chroma(chroma_path: Union[str, Path], collection_name: str, embedding_device: str = "") -> Any:
    try:
        from langchain_chroma import Chroma
    except ImportError as error:
        raise RuntimeError(
            "langchain-chroma is not installed. Install it with "
            "`pip install langchain-chroma` or `pip install -r requirements.txt`."
        ) from error

    return Chroma(
        collection_name=collection_name,
        persist_directory=str(chroma_path),
        embedding_function=LangChainSentenceTransformerEmbeddings(device=embedding_device),
    )


def langchain_bm25_retriever() -> Any:
    try:
        from langchain_community.retrievers import BM25Retriever
    except ImportError as error:
        raise RuntimeError(
            "LangChain BM25 retrieval dependencies are not installed. Install "
            "`langchain-community rank-bm25` or `pip install -r requirements.txt`."
        ) from error
    return BM25Retriever


def build_langchain_documents(ids: Sequence[Any], documents: Sequence[Any], metadatas: Sequence[Any]) -> list[Any]:
    try:
        from langchain_core.documents import Document
    except ImportError as error:
        raise RuntimeError(
            "langchain-core is not installed. Install it with `pip install langchain-core`."
        ) from error

    langchain_documents = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        metadata = {**metadata, "chunk_id": clean_text(ids[index] if index < len(ids) else "")}
        langchain_documents.append(Document(page_content=clean_text(document), metadata=metadata))
    return langchain_documents


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(clean_text(text).lower())


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    min_score = min(scores.values())
    if max_score <= 0:
        return {key: 0.0 for key in scores}
    if max_score == min_score:
        return {key: 1.0 if value > 0 else 0.0 for key, value in scores.items()}
    return {key: (value - min_score) / (max_score - min_score) for key, value in scores.items()}


def document_id(document: Any) -> str:
    metadata = document.metadata if isinstance(document.metadata, dict) else {}
    explicit_id = clean_text(metadata.get("chunk_id")) or clean_text(getattr(document, "id", ""))
    if explicit_id:
        return explicit_id

    history_key = clean_text(metadata.get("history_key"))
    url = clean_text(metadata.get("url"))
    chunk_index = to_int(metadata.get("chunk_index"), 0)
    return chunk_id(history_key, url, chunk_index)


def is_relevant(result: RetrievalResult, relevant_ids: set[str], relevant_urls: set[str]) -> bool:
    return result.id in relevant_ids or clean_text(result.metadata.get("url")) in relevant_urls


def evaluation_context_id(result: RetrievalResult, relevant_urls: set[str]) -> str:
    url = clean_text(result.metadata.get("url"))
    return url if url in relevant_urls else result.id


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
