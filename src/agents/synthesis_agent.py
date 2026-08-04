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
    DEFAULT_RAG_GENERATION_MODEL,
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
            or clean_text(os.environ.get("RESEARCH_PLANNER_MODEL"))
            or clean_text(os.environ.get("RAG_GENERATION_MODEL"))
            or DEFAULT_RAG_GENERATION_MODEL
        )
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.top_k = top_k
        self.per_query_k = per_query_k
        self.semantic_k = semantic_k
        self.bm25_k = bm25_k
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.authority_weight = authority_weight
        self.bm25_scan_limit = bm25_scan_limit
        self.embedding_device = embedding_device
        self.diversify_urls = diversify_urls
        self.rerank = rerank
        self.reranker_model = reranker_model
        self.rerank_k = rerank_k
        self.rerank_weight = rerank_weight
        self.include_planned_source_urls = include_planned_source_urls
        self.source_url_k = source_url_k
        self.rewrite_query = rewrite_query
        self.include_retrieved_chunks = include_retrieved_chunks
        self.max_context_chars = max_context_chars
        self.max_tokens = max_tokens
        self.retrieved_chunk_chars = retrieved_chunk_chars
        self.supporting_chunk_count = supporting_chunk_count

    def synthesize(self, research_plan: dict[str, Any]) -> dict[str, Any]:
        """Create report-agent-ready notes from indexed research evidence."""

        if not isinstance(research_plan, dict) or not research_plan:
            raise ValueError("research_plan is required")

        return synthesize_report_from_research_plan(
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
        )

    def write_to_memory(
        self,
        synthesis_payload: dict[str, Any],
        memory_path: str = "data/shared_memory.json",
    ) -> None:
        """Persist synthesis output for the report agent."""

        memory = SharedMemory(memory_path)
        memory.write_agent_output("synthesis", {"report_context": synthesis_payload})
