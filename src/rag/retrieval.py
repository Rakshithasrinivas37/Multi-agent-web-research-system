"""Hybrid RAG retrieval using semantic search and BM25 keyword search."""

from __future__ import annotations

import asyncio
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union
from urllib.parse import urlparse

from langchain_core.cross_encoders import BaseCrossEncoder

from src.agents.change_detection_agent import normalize_url
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
DEFAULT_SEMANTIC_WEIGHT = 0.55
DEFAULT_BM25_WEIGHT = 0.35
DEFAULT_AUTHORITY_WEIGHT = 0.10
DEFAULT_BM25_SCAN_LIMIT = 1000
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_K = 20
DEFAULT_RERANK_WEIGHT = 0.70
DEFAULT_SOURCE_URL_K = 1
DEFAULT_FEATURE_WEIGHT = 0.15
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
MIN_SOURCE_COVERAGE_OVERLAP = 0.08
SOURCE_COVERAGE_STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "how",
    "the",
    "what",
    "when",
    "where",
    "which",
    "with",
}


@dataclass(frozen=True)
class RetrievalResult:
    id: str
    document: str
    metadata: dict[str, Any]
    score: float
    semantic_score: float
    bm25_score: float
    authority_score: float = 0.0
    rerank_score: float = 0.0
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
    history_keys: Sequence[str] | None = None,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    authority_weight: float = DEFAULT_AUTHORITY_WEIGHT,
    bm25_scan_limit: int = DEFAULT_BM25_SCAN_LIMIT,
    embedding_device: str = "",
    diversify_urls: bool = True,
    rerank: bool = False,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    rerank_k: int = DEFAULT_RERANK_K,
    rerank_weight: float = DEFAULT_RERANK_WEIGHT,
) -> list[RetrievalResult]:
    """Retrieve chunks using semantic vector similarity plus BM25 keyword matching."""

    query = clean_text(query)
    if not query:
        return []

    top_k = max(1, top_k)
    semantic_k = max(top_k, semantic_k)
    bm25_k = max(top_k, bm25_k)

    clean_history_keys = clean_history_key_list(history_keys)
    if clean_history_keys:
        scoped_results = []
        seen_ids = set()
        for scoped_history_key in clean_history_keys:
            for result in hybrid_retrieve(
                query=query,
                chroma_path=chroma_path,
                collection_name=collection_name,
                top_k=top_k,
                semantic_k=semantic_k,
                bm25_k=bm25_k,
                history_key=scoped_history_key,
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight,
                authority_weight=authority_weight,
                bm25_scan_limit=bm25_scan_limit,
                embedding_device=embedding_device,
                diversify_urls=False,
                rerank=rerank,
                reranker_model=reranker_model,
                rerank_k=rerank_k,
                rerank_weight=rerank_weight,
            ):
                if result.id in seen_ids:
                    continue
                seen_ids.add(result.id)
                scoped_results.append(result)
        scoped_results = sorted(scoped_results, key=lambda item: item.score, reverse=True)
        if diversify_urls:
            scoped_results = diversify_by_url(scoped_results, top_k=top_k)
        return scoped_results[:top_k]

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
        query=query,
        semantic_results=semantic_results,
        bm25_results=bm25_results,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight,
        authority_weight=authority_weight,
    )
    if rerank:
        merged = rerank_results(
            query=query,
            results=merged,
            top_n=max(top_k, rerank_k),
            model_name=reranker_model,
            device=embedding_device,
            rerank_weight=rerank_weight,
        )
    if diversify_urls:
        merged = diversify_by_url(merged, top_k=top_k)
    return merged[:top_k]


def multi_query_hybrid_retrieve(
    queries: Sequence[str],
    chroma_path: Union[str, Path] = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = DEFAULT_TOP_K,
    per_query_k: int = DEFAULT_TOP_K,
    semantic_k: int = DEFAULT_SEMANTIC_K,
    bm25_k: int = DEFAULT_BM25_K,
    history_key: str = "",
    history_keys: Sequence[str] | None = None,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    authority_weight: float = DEFAULT_AUTHORITY_WEIGHT,
    bm25_scan_limit: int = DEFAULT_BM25_SCAN_LIMIT,
    embedding_device: str = "",
    diversify_urls: bool = True,
    rerank: bool = False,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    rerank_k: int = DEFAULT_RERANK_K,
    rerank_weight: float = DEFAULT_RERANK_WEIGHT,
) -> list[RetrievalResult]:
    """Run hybrid retrieval for multiple queries and merge the best chunks."""
    clean_queries = [clean_text(query) for query in queries if clean_text(query)]
    if not clean_queries:
        return []

    by_id: dict[str, RetrievalResult] = {}
    for query in clean_queries:
        results = hybrid_retrieve(
            query=query,
            chroma_path=chroma_path,
            collection_name=collection_name,
            top_k=max(1, per_query_k),
            semantic_k=semantic_k,
            bm25_k=bm25_k,
            history_key=history_key,
            history_keys=history_keys,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
            authority_weight=authority_weight,
            bm25_scan_limit=bm25_scan_limit,
            embedding_device=embedding_device,
            diversify_urls=False,
            rerank=rerank,
            reranker_model=reranker_model,
            rerank_k=rerank_k,
            rerank_weight=rerank_weight,
        )
        for result in results:
            existing = by_id.get(result.id)
            if existing is None or result.score > existing.score:
                by_id[result.id] = result

    merged = sorted(by_id.values(), key=lambda item: item.score, reverse=True)
    if diversify_urls:
        merged = diversify_by_url(merged, top_k=top_k)
    return merged[: max(1, top_k)]


def source_url_coverage_retrieve(
    source_urls: Sequence[str],
    query: str | Sequence[str],
    chroma_path: Union[str, Path] = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    history_key: str = "",
    history_keys: Sequence[str] | None = None,
    top_k_per_url: int = DEFAULT_SOURCE_URL_K,
    scan_limit: int = DEFAULT_BM25_SCAN_LIMIT,
) -> list[RetrievalResult]:
    """Retrieve query-matched chunks from specific planned URLs when indexed."""
    target_urls = dedupe_urls(source_urls)
    if not target_urls:
        return []

    clean_history_keys = clean_history_key_list(history_keys)
    if clean_history_keys:
        scoped_results = []
        seen_ids = set()
        for scoped_history_key in clean_history_keys:
            for result in source_url_coverage_retrieve(
                source_urls=target_urls,
                query=query,
                chroma_path=chroma_path,
                collection_name=collection_name,
                history_key=scoped_history_key,
                top_k_per_url=top_k_per_url,
                scan_limit=scan_limit,
            ):
                if result.id in seen_ids:
                    continue
                seen_ids.add(result.id)
                scoped_results.append(result)
        return scoped_results

    query_text = joined_query_text(query)
    collection = get_collection(chroma_path, collection_name)
    get_args = {
        "include": ["documents", "metadatas"],
        "limit": max(scan_limit, len(target_urls) * max(1, top_k_per_url) * 10),
    }
    where = metadata_filter(history_key)
    if where:
        get_args["where"] = where

    try:
        result = collection.get(**get_args)
    except Exception:
        return []

    ids = result.get("ids", []) if isinstance(result, dict) else []
    documents = result.get("documents", []) if isinstance(result, dict) else []
    metadatas = result.get("metadatas", []) if isinstance(result, dict) else []
    rows_by_url: dict[str, list[tuple[Any, Any, Any]]] = {url: [] for url in target_urls}

    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        matched_url = first_matching_url(target_urls, result_source_urls_from_metadata(metadata))
        if not matched_url:
            continue
        rows_by_url[matched_url].append(
            (
                ids[index] if index < len(ids) else "",
                document,
                metadata,
            )
        )

    selected = []
    selected_ids = set()
    for target_url in target_urls:
        ranked_documents = rank_source_rows(rows_by_url.get(target_url, []), query_text, top_k_per_url)
        for rank, document in enumerate(ranked_documents, start=1):
            item_id = document_id(document)
            if not item_id or item_id in selected_ids:
                continue
            selected_ids.add(item_id)
            selected.append(
                RetrievalResult(
                    id=item_id,
                    document=clean_text(document.page_content),
                    metadata=document.metadata if isinstance(document.metadata, dict) else {},
                    score=source_authority_score(document.metadata),
                    semantic_score=0.0,
                    bm25_score=1.0 / rank,
                    authority_score=source_authority_score(document.metadata),
                    rerank_score=0.0,
                    bm25_rank=rank,
                )
            )
    return selected


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
                authority_score=source_authority_score(document.metadata),
                rerank_score=0.0,
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
                authority_score=source_authority_score(document.metadata),
                rerank_score=0.0,
                bm25_rank=bm25_rank,
            )
        )
    return [row for row in rows if row.id]


def rank_source_rows(
    rows: Sequence[tuple[Any, Any, Any]],
    query: str,
    top_k: int,
) -> list[Any]:
    """Rank chunks from one source URL with BM25, falling back to chunk order."""
    if not rows:
        return []

    ids = [row[0] for row in rows]
    documents = [row[1] for row in rows]
    metadatas = [row[2] for row in rows]
    langchain_documents = build_langchain_documents(ids=ids, documents=documents, metadatas=metadatas)
    if not langchain_documents:
        return []

    if query:
        try:
            BM25Retriever = langchain_bm25_retriever()
            retriever = BM25Retriever.from_documents(
                langchain_documents,
                k=max(1, top_k),
                preprocess_func=tokenize,
            )
            ranked = query_feature_rerank(retriever.invoke(query), query)
            return relevant_source_documents(ranked, query, top_k)
        except Exception:
            pass

    return sorted(
        langchain_documents,
        key=lambda document: to_int(
            document.metadata.get("chunk_index") if isinstance(document.metadata, dict) else None,
            0,
        ),
    )[: max(1, top_k)]


def relevant_source_documents(documents: Sequence[Any], query: str, top_k: int) -> list[Any]:
    """Keep URL-coverage chunks that still match the retrieval query."""

    query_terms = relevance_terms(query)
    if not query_terms:
        return list(documents[: max(1, top_k)])

    selected = []
    for document in documents:
        doc_terms = relevance_terms(getattr(document, "page_content", ""))
        overlap = len(query_terms & doc_terms) / max(1, min(len(query_terms), len(doc_terms)))
        if overlap >= MIN_SOURCE_COVERAGE_OVERLAP:
            selected.append(document)
        if len(selected) >= max(1, top_k):
            break
    return selected


def relevance_terms(text: str) -> set[str]:
    return {
        token
        for token in tokenize(text)
        if len(token) > 2 and token not in SOURCE_COVERAGE_STOPWORDS
    }


def merge_ranked_results(
    query: str,
    semantic_results: Sequence[RetrievalResult],
    bm25_results: Sequence[RetrievalResult],
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    authority_weight: float = DEFAULT_AUTHORITY_WEIGHT,
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
        authority_score = source_authority_score(item.metadata)
        score = (
            (semantic_weight * semantic_score)
            + (bm25_weight * bm25_score)
            + (authority_weight * authority_score)
            + (DEFAULT_FEATURE_WEIGHT * query_feature_score(query, item.document))
        )
        merged.append(
            RetrievalResult(
                id=item.id,
                document=item.document,
                metadata=item.metadata,
                score=score,
                semantic_score=semantic_score,
                bm25_score=bm25_score,
                authority_score=authority_score,
                rerank_score=item.rerank_score,
                semantic_rank=item.semantic_rank,
                bm25_rank=item.bm25_rank,
            )
        )

    return sorted(merged, key=lambda item: item.score, reverse=True)


def query_feature_rerank(documents: Sequence[Any], query: str) -> list[Any]:
    """Lightly prefer chunks that preserve the query's terms and phrases."""

    return [
        document
        for _, document in sorted(
            enumerate(documents),
            key=lambda pair: query_feature_score(query, getattr(pair[1], "page_content", "")) + (0.05 / (pair[0] + 1)),
            reverse=True,
        )
    ]


def query_feature_score(query: str, document: str) -> float:
    query_terms = relevance_terms(query)
    document_terms = relevance_terms(document)
    if not query_terms or not document_terms:
        return 0.0

    term_overlap = len(query_terms & document_terms) / max(1, min(len(query_terms), len(document_terms)))
    phrase_overlap = query_phrase_overlap(query, document)
    return min(1.0, (0.75 * term_overlap) + (0.25 * phrase_overlap))


def query_phrase_overlap(query: str, document: str) -> float:
    query_tokens = [token for token in tokenize(query) if token not in SOURCE_COVERAGE_STOPWORDS]
    document_text = clean_text(document).lower()
    phrases = []
    for size in (2, 3, 4):
        phrases.extend(" ".join(query_tokens[index : index + size]) for index in range(len(query_tokens) - size + 1))
    phrases = [phrase for phrase in phrases if len(phrase) > 5]
    if not phrases:
        return 0.0
    return sum(1 for phrase in phrases if phrase in document_text) / len(phrases)


def evaluate_retrieval(
    retrieved: Sequence[RetrievalResult],
    relevant_ids: Optional[set[str]] = None,
    relevant_urls: Optional[set[str]] = None,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> list[RetrievalMetrics]:
    """Compute RAGAS ID-based Recall@k/Precision@k plus local MRR."""

    relevant_ids = {clean_text(item) for item in (relevant_ids or set()) if clean_text(item)}
    relevant_urls = {normalize_source_url(item) for item in (relevant_urls or set()) if normalize_source_url(item)}
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
    except (ImportError, ModuleNotFoundError) as first_error:
        try:
            from ragas import SingleTurnSample
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "RAGAS could not be imported. This can happen when ragas and LangChain package versions "
                f"are incompatible. Original error: {error or first_error}"
            ) from error

    try:
        from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "RAGAS ID-based retrieval metrics are unavailable. Upgrade ragas or adjust LangChain versions. "
            f"Original error: {error}"
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


def clean_history_key_list(history_keys: Sequence[str] | None) -> list[str]:
    seen = set()
    clean_keys = []
    for history_key in history_keys or []:
        key = clean_text(history_key)
        if not key or key in seen:
            continue
        seen.add(key)
        clean_keys.append(key)
    return clean_keys


def merge_result(existing: RetrievalResult, incoming: RetrievalResult) -> RetrievalResult:
    return RetrievalResult(
        id=existing.id,
        document=existing.document or incoming.document,
        metadata=existing.metadata or incoming.metadata,
        score=0.0,
        semantic_score=max(existing.semantic_score, incoming.semantic_score),
        bm25_score=max(existing.bm25_score, incoming.bm25_score),
        authority_score=max(existing.authority_score, incoming.authority_score),
        rerank_score=max(existing.rerank_score, incoming.rerank_score),
        semantic_rank=existing.semantic_rank or incoming.semantic_rank,
        bm25_rank=existing.bm25_rank or incoming.bm25_rank,
    )


def rerank_results(
    query: str,
    results: Sequence[RetrievalResult],
    top_n: int = DEFAULT_RERANK_K,
    model_name: str = DEFAULT_RERANKER_MODEL,
    device: str = "",
    rerank_weight: float = DEFAULT_RERANK_WEIGHT,
) -> list[RetrievalResult]:
    """Rerank hybrid candidates with a CrossEncoder query/chunk relevance model."""
    candidates = list(results[: max(1, top_n)])
    remainder = list(results[max(1, top_n):])
    if not candidates:
        return []

    reranker = langchain_cross_encoder_reranker(
        model_name=model_name,
        device=device,
        top_n=len(candidates),
    )
    reranked_documents = reranker.compress_documents(
        documents=retrieval_results_to_langchain_documents(candidates),
        query=query,
    )
    normalized_scores = rank_scores([document_id(document) for document in reranked_documents])
    rerank_weight = min(1.0, max(0.0, rerank_weight))

    reranked = []
    for item in candidates:
        rerank_score = normalized_scores.get(item.id, 0.0)
        combined_score = (rerank_weight * rerank_score) + ((1.0 - rerank_weight) * item.score)
        reranked.append(
            RetrievalResult(
                id=item.id,
                document=item.document,
                metadata=item.metadata,
                score=combined_score,
                semantic_score=item.semantic_score,
                bm25_score=item.bm25_score,
                authority_score=item.authority_score,
                rerank_score=rerank_score,
                semantic_rank=item.semantic_rank,
                bm25_rank=item.bm25_rank,
            )
        )

    reranked = sorted(reranked, key=lambda item: item.score, reverse=True)
    return reranked + remainder


def langchain_cross_encoder_reranker(model_name: str, device: str, top_n: int) -> Any:
    """Build LangChain's CrossEncoderReranker with a Hugging Face cross-encoder."""
    try:
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    except ImportError as error:
        raise RuntimeError(
            "LangChain reranking dependencies are unavailable. Install "
            "`langchain-classic sentence-transformers` or `pip install -r requirements.txt`."
        ) from error

    selected_device = select_embedding_device(device or "auto")
    cross_encoder = SentenceTransformerCrossEncoderAdapter(
        model_name=clean_text(model_name) or DEFAULT_RERANKER_MODEL,
        device=selected_device,
    )
    return CrossEncoderReranker(model=cross_encoder, top_n=max(1, top_n))


class SentenceTransformerCrossEncoderAdapter(BaseCrossEncoder):
    """LangChain BaseCrossEncoder adapter for sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str, device: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers and langchain-core are required for reranking. "
                "Install them with `pip install -r requirements.txt`."
            ) from error
        self.client = CrossEncoder(model_name, device=device)

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        scores = self.client.predict(text_pairs, show_progress_bar=False)
        return [to_float(score, 0.0) for score in list(scores)]


def retrieval_results_to_langchain_documents(results: Sequence[RetrievalResult]) -> list[Any]:
    try:
        from langchain_core.documents import Document
    except ImportError as error:
        raise RuntimeError(
            "langchain-core is not installed. Install it with `pip install langchain-core`."
        ) from error

    documents = []
    for result in results:
        documents.append(
            Document(
                page_content=display_document_preview(result.document, max_chars=1400),
                metadata={**result.metadata, "chunk_id": result.id},
            )
        )
    return documents


def diversify_by_url(results: Sequence[RetrievalResult], top_k: int) -> list[RetrievalResult]:
    """Prefer source diversity, then fill remaining slots by score."""
    top_k = max(1, top_k)
    selected = []
    selected_ids = set()
    seen_urls = set()

    for result in results:
        url = clean_text(result.metadata.get("url"))
        if url and url in seen_urls:
            continue
        selected.append(result)
        selected_ids.add(result.id)
        if url:
            seen_urls.add(url)
        if len(selected) >= top_k:
            return selected

    for result in results:
        if result.id in selected_ids:
            continue
        selected.append(result)
        if len(selected) >= top_k:
            break
    return selected


def source_authority_score(metadata: Any) -> float:
    """Small ranking boost for primary/authoritative sources."""
    metadata = metadata if isinstance(metadata, dict) else {}
    urls = " ".join(result_source_urls_from_metadata(metadata)).lower()
    title = clean_text(metadata.get("title")).lower()
    source_type = clean_text(metadata.get("source_type")).lower()
    source_quality = clean_text(metadata.get("source_quality")).lower()

    score = 0.0
    if source_type in {"arxiv", "pdf", "paper"} or "arxiv.org" in urls:
        score = max(score, 1.0)
    if any(domain in urls for domain in ("nature.com", "ibm.com", ".edu", "docs.", "documentation")):
        score = max(score, 0.75)
    if source_type in {"webpage", "news"}:
        score = max(score, 0.45)
    if any(domain in urls for domain in ("geeksforgeeks.org", "medium.com", "blog", "erdem.pl")):
        score = min(max(score, 0.35), 0.45)
    if "useful" in source_quality:
        score = max(score, 0.55 if score == 0 else score)
    if any(word in title for word in ("paper", "arxiv", "documentation", "docs")):
        score = max(score, 0.7)
    return min(1.0, max(0.0, score))


def display_document_preview(document: str, max_chars: int = 700) -> str:
    """Remove stored source headers from CLI previews."""
    lines = clean_text(document).splitlines()
    content_lines = [
        line for line in lines
        if not line.startswith("Source: ") and not line.startswith("URL: ") and not line.startswith("Task: ")
    ]
    preview = clean_text("\n".join(content_lines))
    return preview[:max(80, max_chars)].strip()


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


def rank_scores(ids: Sequence[str]) -> dict[str, float]:
    """Convert a reranked ID order into scores between 1.0 and 0.0."""
    ids = [clean_text(item) for item in ids if clean_text(item)]
    if not ids:
        return {}
    if len(ids) == 1:
        return {ids[0]: 1.0}
    denominator = len(ids) - 1
    return {item_id: 1.0 - (index / denominator) for index, item_id in enumerate(ids)}


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
    return result.id in relevant_ids or any(url in relevant_urls for url in result_source_urls(result))


def evaluation_context_id(result: RetrievalResult, relevant_urls: set[str]) -> str:
    for url in result_source_urls(result):
        if url in relevant_urls:
            return url
    return result.id


def result_source_urls(result: RetrievalResult) -> list[str]:
    return result_source_urls_from_metadata(result.metadata)


def result_source_urls_from_metadata(metadata: Any) -> list[str]:
    metadata = metadata if isinstance(metadata, dict) else {}
    urls = []
    for key in ("url", "source_url"):
        url = normalize_source_url(metadata.get(key))
        if url and url not in urls:
            urls.append(url)
    for url in split_metadata_urls(metadata.get("task_urls")):
        if url and url not in urls:
            urls.append(url)
    return urls


def split_metadata_urls(value: Any) -> list[str]:
    if isinstance(value, list):
        candidates = value
    else:
        candidates = clean_text(value).split("|")
    return [url for url in (normalize_source_url(item) for item in candidates) if url]


def normalize_source_url(value: Any) -> str:
    url = normalize_url(clean_text(value))
    if not url.startswith(("http://", "https://")):
        return ""
    return canonical_source_url(url)


def canonical_source_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    match = re.match(r"^/(abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?(?:\.pdf)?$", path)
    if parsed.netloc.lower() == "arxiv.org" and match:
        return f"{parsed.scheme}://arxiv.org/pdf/{match.group(2)}"
    return url


def dedupe_urls(urls: Sequence[str]) -> list[str]:
    seen = set()
    deduped = []
    for url in urls:
        normalized = normalize_source_url(url)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def joined_query_text(query: str | Sequence[str]) -> str:
    if isinstance(query, str):
        return clean_text(query)
    return clean_text(" ".join(clean_text(item) for item in query if clean_text(item)))


def first_matching_url(target_urls: Sequence[str], candidate_urls: Sequence[str]) -> str:
    candidates = set(dedupe_urls(candidate_urls))
    for url in target_urls:
        if url in candidates:
            return url
    return ""


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
