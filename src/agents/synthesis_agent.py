"""Synthesis agent for report-ready research context."""

from __future__ import annotations

import os
from typing import Any

from src.memory.shared_memory import SharedMemory
from src.rag.generation import (
    DEFAULT_CONTEXT_BLOCK_CHARS,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_REPORT_AUTHORITY_WEIGHT,
    DEFAULT_REPORT_BM25_WEIGHT,
    DEFAULT_REPORT_MAX_TOKENS,
    DEFAULT_REPORT_PER_QUERY_K,
    DEFAULT_REPORT_SEMANTIC_WEIGHT,
    DEFAULT_REPORT_SUPPORTING_CHUNKS,
    DEFAULT_REPORT_SOURCE_URL_K,
    DEFAULT_REPORT_TOP_K,
    rag_generation_model,
    synthesize_report_from_research_plan,
)
from src.rag.retrieval import (
    DEFAULT_BM25_K,
    DEFAULT_BM25_SCAN_LIMIT,
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_RERANK_K,
    DEFAULT_RERANK_WEIGHT,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_SEMANTIC_K,
)
from src.tools.text_utils import clean_text


class SynthesisAgent:
    """Retrieve planner evidence and synthesize it for a downstream report agent."""

    def __init__(
        self,
        model: str | None = None,
        chroma_path: str = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        top_k: int = DEFAULT_REPORT_TOP_K,
        per_query_k: int = DEFAULT_REPORT_PER_QUERY_K,
        semantic_k: int = DEFAULT_SEMANTIC_K,
        bm25_k: int = DEFAULT_BM25_K,
        semantic_weight: float = DEFAULT_REPORT_SEMANTIC_WEIGHT,
        bm25_weight: float = DEFAULT_REPORT_BM25_WEIGHT,
        authority_weight: float = DEFAULT_REPORT_AUTHORITY_WEIGHT,
        bm25_scan_limit: int = DEFAULT_BM25_SCAN_LIMIT,
        embedding_device: str = "",
        diversify_urls: bool = True,
        rerank: bool = False,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        rerank_k: int = DEFAULT_RERANK_K,
        rerank_weight: float = DEFAULT_RERANK_WEIGHT,
        include_planned_source_urls: bool = True,
        source_url_k: int = DEFAULT_REPORT_SOURCE_URL_K,
        rewrite_query: bool = False,
        include_retrieved_chunks: bool = True,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        max_tokens: int = DEFAULT_REPORT_MAX_TOKENS,
        retrieved_chunk_chars: int = DEFAULT_CONTEXT_BLOCK_CHARS,
        supporting_chunk_count: int = DEFAULT_REPORT_SUPPORTING_CHUNKS,
    ) -> None:
        self.model = (
            clean_text(model)
            or rag_generation_model()
        )
        self.chroma_path = env_text("RAG_CHROMA_PATH", chroma_path, DEFAULT_CHROMA_PATH)
        self.collection_name = env_text("RAG_COLLECTION_NAME", collection_name, DEFAULT_COLLECTION_NAME)
        self.top_k = env_int("RAG_REPORT_TOP_K", top_k, DEFAULT_REPORT_TOP_K)
        self.per_query_k = env_int("RAG_REPORT_PER_QUERY_K", per_query_k, DEFAULT_REPORT_PER_QUERY_K)
        self.semantic_k = env_int("RAG_SEMANTIC_K", semantic_k, DEFAULT_SEMANTIC_K)
        self.bm25_k = env_int("RAG_BM25_K", bm25_k, DEFAULT_BM25_K)
        self.semantic_weight = env_float("RAG_SEMANTIC_WEIGHT", semantic_weight, DEFAULT_REPORT_SEMANTIC_WEIGHT)
        self.bm25_weight = env_float("RAG_BM25_WEIGHT", bm25_weight, DEFAULT_REPORT_BM25_WEIGHT)
        self.authority_weight = env_float("RAG_AUTHORITY_WEIGHT", authority_weight, DEFAULT_REPORT_AUTHORITY_WEIGHT)
        self.bm25_scan_limit = env_int("RAG_BM25_SCAN_LIMIT", bm25_scan_limit, DEFAULT_BM25_SCAN_LIMIT)
        self.embedding_device = env_text("RAG_EMBEDDING_DEVICE", embedding_device, "")
        self.diversify_urls = env_bool("RAG_DIVERSIFY_URLS", diversify_urls, True)
        self.rerank = env_bool("RAG_RERANK", rerank, False)
        self.reranker_model = env_text("RAG_RERANKER_MODEL", reranker_model, DEFAULT_RERANKER_MODEL)
        self.rerank_k = env_int("RAG_RERANK_K", rerank_k, DEFAULT_RERANK_K)
        self.rerank_weight = env_float("RAG_RERANK_WEIGHT", rerank_weight, DEFAULT_RERANK_WEIGHT)
        self.include_planned_source_urls = env_bool(
            "RAG_INCLUDE_PLANNED_SOURCE_URLS",
            include_planned_source_urls,
            True,
        )
        self.source_url_k = env_int("RAG_SOURCE_URL_K", source_url_k, DEFAULT_REPORT_SOURCE_URL_K)
        self.rewrite_query = env_bool("RAG_REWRITE_QUERY", rewrite_query, False)
        self.include_retrieved_chunks = env_bool("RAG_INCLUDE_RETRIEVED_CHUNKS", include_retrieved_chunks, True)
        self.max_context_chars = env_int("RAG_MAX_CONTEXT_CHARS", max_context_chars, DEFAULT_MAX_CONTEXT_CHARS)
        self.max_tokens = env_int("RAG_REPORT_MAX_TOKENS", max_tokens, DEFAULT_REPORT_MAX_TOKENS)
        self.retrieved_chunk_chars = env_int(
            "RAG_RETRIEVED_CHUNK_CHARS",
            retrieved_chunk_chars,
            DEFAULT_CONTEXT_BLOCK_CHARS,
        )
        self.supporting_chunk_count = env_int(
            "RAG_SUPPORTING_CHUNKS",
            supporting_chunk_count,
            DEFAULT_REPORT_SUPPORTING_CHUNKS,
        )

    def synthesize(
        self,
        research_plan: dict[str, Any],
        browser_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create report-agent-ready notes from indexed research evidence."""

        if not isinstance(research_plan, dict) or not research_plan:
            raise ValueError("research_plan is required")

        payload = synthesize_report_from_research_plan(
            research_plan=research_plan,
            chroma_path=self.chroma_path,
            collection_name=self.collection_name,
            top_k=self.top_k,
            per_query_k=self.per_query_k,
            semantic_k=self.semantic_k,
            bm25_k=self.bm25_k,
            semantic_weight=self.semantic_weight,
            bm25_weight=self.bm25_weight,
            authority_weight=self.authority_weight,
            bm25_scan_limit=self.bm25_scan_limit,
            embedding_device=self.embedding_device,
            diversify_urls=self.diversify_urls,
            rerank=self.rerank,
            reranker_model=self.reranker_model,
            rerank_k=self.rerank_k,
            rerank_weight=self.rerank_weight,
            include_planned_source_urls=self.include_planned_source_urls,
            source_url_k=self.source_url_k,
            rewrite_query=self.rewrite_query,
            model=self.model,
            max_context_chars=self.max_context_chars,
            max_tokens=self.max_tokens,
            include_retrieved_chunks=self.include_retrieved_chunks,
            retrieved_chunk_chars=self.retrieved_chunk_chars,
            supporting_chunk_count=self.supporting_chunk_count,
            browser_results=browser_results or [],
        )
        payload["synthesis_config"] = self.config_summary()
        return payload

    def config_summary(self) -> dict[str, Any]:
        """Return the runtime knobs used by synthesis for debugging."""

        return {
            "model": self.model,
            "top_k": self.top_k,
            "per_query_k": self.per_query_k,
            "semantic_k": self.semantic_k,
            "bm25_k": self.bm25_k,
            "diversify_urls": self.diversify_urls,
            "source_url_k": self.source_url_k,
            "rewrite_query": self.rewrite_query,
            "max_context_chars": self.max_context_chars,
            "max_tokens": self.max_tokens,
            "supporting_chunk_count": self.supporting_chunk_count,
            "include_retrieved_chunks": self.include_retrieved_chunks,
        }

    def write_to_memory(
        self,
        synthesis_payload: dict[str, Any],
        memory_path: str = "data/shared_memory.json",
    ) -> None:
        """Persist synthesis output for the report agent."""

        memory = SharedMemory(memory_path)
        memory.write_agent_output("synthesis", {"report_context": synthesis_payload})


def env_text(name: str, current: str, default: str) -> str:
    value = clean_text(os.environ.get(name))
    if value and clean_text(current) == clean_text(default):
        return value
    return current


def env_int(name: str, current: int, default: int) -> int:
    value = os.environ.get(name)
    if current != default or not value:
        return current
    try:
        return int(value)
    except ValueError:
        return current


def env_float(name: str, current: float, default: float) -> float:
    value = os.environ.get(name)
    if current != default or not value:
        return current
    try:
        return float(value)
    except ValueError:
        return current


def env_bool(name: str, current: bool, default: bool) -> bool:
    value = clean_text(os.environ.get(name)).lower()
    if current != default or not value:
        return current
    return value not in {"0", "false", "no", "off"}
