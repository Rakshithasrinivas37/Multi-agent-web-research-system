"""Hybrid RAG retrieval using semantic search and BM25 keyword search."""

from __future__ import annotations

import asyncio
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from langchain_core.cross_encoders import BaseCrossEncoder

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
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


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


def merge_ranked_results(
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
    url = clean_text(metadata.get("url")).lower()
    title = clean_text(metadata.get("title")).lower()
    source_type = clean_text(metadata.get("source_type")).lower()
    source_quality = clean_text(metadata.get("source_quality")).lower()

    score = 0.0
    if source_type in {"arxiv", "pdf", "paper"} or "arxiv.org" in url:
        score = max(score, 1.0)
    if any(domain in url for domain in ("nature.com", "ibm.com", ".edu", "docs.", "documentation")):
        score = max(score, 0.75)
    if source_type in {"webpage", "news"}:
        score = max(score, 0.45)
    if any(domain in url for domain in ("geeksforgeeks.org", "medium.com", "blog", "erdem.pl")):
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
    return result.id in relevant_ids or clean_text(result.metadata.get("url")) in relevant_urls


def evaluation_context_id(result: RetrievalResult, relevant_urls: set[str]) -> str:
    url = clean_text(result.metadata.get("url"))
    return url if url in relevant_urls else result.id


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
