"""Generate a RAG answer from already-retrieved context chunks."""

from __future__ import annotations

import os
from typing import Any, Sequence

from src.rag.retrieval import (
    DEFAULT_BM25_K,
    DEFAULT_BM25_SCAN_LIMIT,
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_RERANK_K,
    DEFAULT_RERANK_WEIGHT,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_SEMANTIC_K,
    RetrievalResult,
    display_document_preview,
    multi_query_hybrid_retrieve,
    result_source_urls_from_metadata,
)
from src.tools.text_utils import clean_text


DEFAULT_RAG_GENERATION_MODEL = "llama-3.1-8b-instant"
DEFAULT_MAX_CONTEXT_CHARS = 12000
DEFAULT_MAX_TOKENS = 900
DEFAULT_REPORT_MAX_TOKENS = 1400
DEFAULT_REPORT_TOP_K = 12
DEFAULT_REPORT_PER_QUERY_K = 20
DEFAULT_REPORT_SEMANTIC_WEIGHT = 0.30
DEFAULT_REPORT_BM25_WEIGHT = 0.30
DEFAULT_REPORT_AUTHORITY_WEIGHT = 0.40
DEFAULT_CONTEXT_BLOCK_CHARS = 1400


def synthesize_report_from_research_plan(
    research_plan: dict[str, Any],
    chroma_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = DEFAULT_REPORT_TOP_K,
    per_query_k: int = DEFAULT_REPORT_PER_QUERY_K,
    semantic_k: int = DEFAULT_SEMANTIC_K,
    bm25_k: int = DEFAULT_BM25_K,
    history_key: str = "",
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
    model: str | None = None,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_tokens: int = DEFAULT_REPORT_MAX_TOKENS,
) -> dict[str, Any]:
    """Retrieve with planner-derived queries, then synthesize report-ready notes."""
    objective = clean_text(research_plan.get("objective"))
    if not objective:
        raise ValueError("research_plan.objective is required")

    queries = planner_tasks_to_rag_queries(research_plan)
    retrieved_context = multi_query_hybrid_retrieve(
        queries=queries,
        chroma_path=chroma_path,
        collection_name=collection_name,
        top_k=top_k,
        per_query_k=per_query_k,
        semantic_k=semantic_k,
        bm25_k=bm25_k,
        history_key=history_key,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight,
        authority_weight=authority_weight,
        bm25_scan_limit=bm25_scan_limit,
        embedding_device=embedding_device,
        diversify_urls=diversify_urls,
        rerank=rerank,
        reranker_model=reranker_model,
        rerank_k=rerank_k,
        rerank_weight=rerank_weight,
    )
    payload = synthesize_context_for_report(
        objective=objective,
        retrieved_context=retrieved_context,
        synthesis_instruction=clean_text(research_plan.get("synthesis_instruction")),
        model=model,
        max_context_chars=max_context_chars,
        max_tokens=max_tokens,
    )
    payload["queries"] = queries
    payload["retrieved_count"] = len(retrieved_context)
    return payload


def planner_tasks_to_rag_queries(research_plan: dict[str, Any]) -> list[str]:
    """Build retrieval queries from PlannerAgent task output."""
    objective = clean_text(research_plan.get("objective"))
    queries = []

    for task in research_plan.get("tasks", []):
        if not isinstance(task, dict):
            continue
        expected_signals = task.get("expected_signals", [])
        expected_signals_text = " ".join(clean_text(item) for item in expected_signals if clean_text(item))
        parts = [
            task.get("query_context"),
            task.get("extraction_goal"),
            task.get("target_name"),
            task.get("url"),
            expected_signals_text,
        ]
        query = clean_text(" ".join(clean_text(part) for part in parts if clean_text(part)))
        if query:
            queries.append(query)

    for sub_question in research_plan.get("sub_questions", []):
        query = clean_text(sub_question)
        if query:
            queries.append(query)

    if objective:
        queries.append(objective)
    return dedupe_preserve_order(queries)


def generate_answer_from_context(
    question: str,
    retrieved_context: Sequence[RetrievalResult],
    model: str | None = None,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Use Groq to answer a question from retrieved RAG chunks only."""
    question = clean_text(question)
    if not question:
        raise ValueError("question is required")
    if not retrieved_context:
        return {"answer": "I could not find retrieved context to answer the question.", "sources": []}
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set")

    try:
        from groq import Groq
    except ImportError as error:
        raise RuntimeError("groq package is not installed. Install it with `pip install -r requirements.txt`.") from error

    context_text, sources = build_generation_context(retrieved_context, max_context_chars=max_context_chars)
    prompt = f"""Question:
{question}

Retrieved context:
{context_text}

Answer using only the retrieved context. If the context does not contain the answer, say that clearly.
Use only the numbered source markers that appear in the retrieved context, like [1] or [2].
Do not cite source names, authors, dates, or papers unless they are present in the retrieved context."""

    response = Groq().chat.completions.create(
        model=clean_text(model or os.environ.get("RAG_GENERATION_MODEL")) or DEFAULT_RAG_GENERATION_MODEL,
        temperature=0,
        max_tokens=max(100, max_tokens),
        messages=[
            {
                "role": "system",
                "content": "You are a careful RAG answer generator. Do not use outside knowledge.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    answer = clean_text(response.choices[0].message.content)
    return {"answer": answer, "sources": sources, "model": response.model}


def synthesize_context_for_report(
    objective: str,
    retrieved_context: Sequence[RetrievalResult],
    synthesis_instruction: str = "",
    model: str | None = None,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_tokens: int = DEFAULT_REPORT_MAX_TOKENS,
) -> dict[str, Any]:
    """Synthesize retrieved chunks into a compact payload for a report agent."""
    objective = clean_text(objective)
    if not objective:
        raise ValueError("objective is required")
    if not retrieved_context:
        return {
            "objective": objective,
            "synthesis": "No retrieved context was available for report synthesis.",
            "sources": [],
        }
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set")

    try:
        from groq import Groq
    except ImportError as error:
        raise RuntimeError("groq package is not installed. Install it with `pip install -r requirements.txt`.") from error

    context_text, sources = build_generation_context(retrieved_context, max_context_chars=max_context_chars)
    instruction = clean_text(synthesis_instruction) or "Synthesize the retrieved evidence into report-ready research notes."
    prompt = f"""Research objective:
{objective}

Synthesis instruction:
{instruction}

Retrieved context from multiple sources:
{context_text}

Create report-agent-ready research notes using only the retrieved context.
Return concise Markdown with these sections:
- Key Findings
- Technical Details
- Source Evidence
- Conflicts Or Gaps
- Recommended Report Angle

Use only the numbered source markers that appear in the retrieved context, like [1], [2], [3].
Every evidence-backed claim must include at least one source marker.
Do not invent source names, authors, dates, titles, papers, or citations that are not present in the retrieved context."""

    response = Groq().chat.completions.create(
        model=clean_text(model or os.environ.get("RAG_GENERATION_MODEL")) or DEFAULT_RAG_GENERATION_MODEL,
        temperature=0,
        max_tokens=max(300, max_tokens),
        messages=[
            {
                "role": "system",
                "content": "You synthesize retrieved RAG evidence for a downstream report agent. Do not use outside knowledge.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    synthesis = clean_text(response.choices[0].message.content)
    return {
        "objective": objective,
        "synthesis_instruction": instruction,
        "synthesis": synthesis,
        "sources": sources,
        "model": response.model,
    }


def build_generation_context(
    retrieved_context: Sequence[RetrievalResult],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> tuple[str, list[dict[str, Any]]]:
    """Convert retrieved chunks into compact numbered context blocks.

    The prompt budget is spread across unique source URLs first so synthesis sees
    evidence from more than just the highest-ranked long chunks.
    """
    blocks = []
    sources = []
    used_chars = 0
    max_context_chars = max(1000, max_context_chars)
    ordered_results = source_balanced_results(retrieved_context)
    block_char_limit = context_block_char_limit(ordered_results, max_context_chars)

    for index, result in enumerate(ordered_results, start=1):
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        url = primary_source_url(metadata)
        title = clean_text(metadata.get("title")) or url or f"Source {index}"
        chunk = display_document_preview(result.document, max_chars=block_char_limit)
        if not chunk:
            continue

        block = f"[{index}] {title}\nURL: {url}\n{chunk}"
        if used_chars + len(block) > max_context_chars:
            remaining = max_context_chars - used_chars
            if remaining < 500:
                break
            block = block[:remaining].rstrip()

        blocks.append(block)
        sources.append(
            {
                "index": index,
                "id": result.id,
                "url": url,
                "title": title,
                "score": result.score,
            }
        )
        used_chars += len(block)

    return "\n\n".join(blocks), sources


def source_balanced_results(retrieved_context: Sequence[RetrievalResult]) -> list[RetrievalResult]:
    """Interleave results so each source URL gets represented before repeats."""
    buckets: dict[str, list[RetrievalResult]] = {}
    source_order = []

    for result in retrieved_context:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        url = primary_source_url(metadata) or result.id
        if url not in buckets:
            buckets[url] = []
            source_order.append(url)
        buckets[url].append(result)

    ordered = []
    while True:
        added = False
        for url in source_order:
            bucket = buckets[url]
            if not bucket:
                continue
            ordered.append(bucket.pop(0))
            added = True
        if not added:
            break
    return ordered


def context_block_char_limit(
    retrieved_context: Sequence[RetrievalResult],
    max_context_chars: int,
) -> int:
    """Choose a per-block preview size that leaves room for many sources."""
    nonempty_count = sum(1 for result in retrieved_context if clean_text(result.document))
    if nonempty_count <= 0:
        return DEFAULT_CONTEXT_BLOCK_CHARS
    target_blocks = min(nonempty_count, 10)
    per_block_budget = max_context_chars // max(1, target_blocks)
    return max(700, min(DEFAULT_CONTEXT_BLOCK_CHARS, per_block_budget - 120))


def primary_source_url(metadata: dict[str, Any]) -> str:
    urls = result_source_urls_from_metadata(metadata)
    return urls[0] if urls else clean_text(metadata.get("url"))


def dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    seen = set()
    deduped = []
    for item in items:
        text = clean_text(item)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped
