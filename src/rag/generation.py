"""Generate a RAG answer from already-retrieved context chunks."""

from __future__ import annotations

import os
import re
import hashlib
import json
from typing import Any, Sequence

from src.agents.change_detection_agent import objective_key
from src.rag.indexing import get_collection
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
    normalize_source_url,
    result_source_urls_from_metadata,
    source_url_coverage_retrieve,
)
from src.tools.groq_retry import create_chat_completion_with_retries, is_groq_request_too_large_error
from src.tools.progress import emit_progress
from src.tools.text_utils import clean_text


DEFAULT_RAG_GENERATION_MODEL = "llama-3.1-8b-instant"
DEFAULT_GAP_QUERY_MODEL = "qwen/qwen3.6-27b"
DEFAULT_SUBQUESTION_QUERY_REWRITE_MODEL = "qwen/qwen3.6-27b"
DEFAULT_MAX_CONTEXT_CHARS = 18000
DEFAULT_MAX_TOKENS = 900
DEFAULT_REPORT_MAX_TOKENS = 1400
DEFAULT_REPORT_TOP_K = 20
DEFAULT_REPORT_PER_QUERY_K = 25
DEFAULT_REPORT_SEMANTIC_WEIGHT = 0.30
DEFAULT_REPORT_BM25_WEIGHT = 0.30
DEFAULT_REPORT_AUTHORITY_WEIGHT = 0.40
DEFAULT_CONTEXT_BLOCK_CHARS = 750
DEFAULT_REPORT_SOURCE_URL_K = 2
DEFAULT_REPORT_SUPPORTING_CHUNKS = 6
DEFAULT_BROWSER_SIGNAL_SOURCES = 6
DEFAULT_BROWSER_SIGNAL_SNIPPETS = 2
DEFAULT_BROWSER_SIGNAL_CANDIDATES = 40
DEFAULT_GAP_RETRIEVAL_TOP_K = 12
DEFAULT_GAP_RETRIEVAL_PER_QUERY_K = 4
DEFAULT_GAP_RETRIEVAL_MAX_QUERIES = 6
DEFAULT_PRECISION_QUERY_LIMIT = 8
DEFAULT_SUBQUESTION_QUERY_VARIANTS = 3
DEFAULT_SYNTHESIS_CHUNK_PRINT_LIMIT = 30
DEFAULT_SYNTHESIS_CHUNKS_PER_QUESTION = 3
DEFAULT_SYNTHESIS_MAX_CHUNKS = 18
MIN_EVIDENCE_CHARS = 120
MIN_EVIDENCE_TOKENS = 12
DEFAULT_OBJECTIVE_SCOPE_SIMILARITY = 0.40
DEFAULT_OBJECTIVE_SCOPE_MAX_KEYS = 6
URL_PATTERN = re.compile(r"https?://[^\s\])}>\"']+")
OBJECTIVE_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
BROWSER_SIGNAL_PATTERN = re.compile(
    r"(?i)\b("
    r"definition|defined\s+as|equation|formula|theorem|algorithm|"
    r"benchmark|score|accuracy|precision|recall|f1|auc|bleu|rouge|"
    r"latency|throughput|cost|price|rate|table|"
    r"api|signature|parameter|argument|class|function|method|"
    r"complexity|memory|runtime|quadratic|linear|o\("
    r")\b"
)
BROWSER_FORMULA_SIGNAL_PATTERN = re.compile(
    r"(?i)(?:"
    r"\\(?:frac|sum|sqrt|operatorname)|"
    r"[=∑Σ√αβγδθλµπ]|"
    r"\b(?:equation|formula|softmax|sqrt|tanh|exp|log|argmax|argmin)\b|"
    r"\b[A-Za-z_][A-Za-z0-9_]*\s*\([^)]{0,80}\)\s*="
    r")"
)
BROWSER_METRIC_SIGNAL_PATTERN = re.compile(
    r"(?i)\b\d+(?:\.\d+)?\s*(?:%|bleu|rouge|f1|auc|accuracy|precision|recall|ms|s|tokens/s|score)\b"
)
BROWSER_API_SIGNAL_PATTERN = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_.]*\([^)]{1,120}\)|\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_.]*\b"
)
OBJECTIVE_STOPWORDS = {
    "a",
    "an",
    "and",
    "architecture",
    "architectures",
    "based",
    "compare",
    "different",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "research",
    "the",
    "to",
    "what",
    "with",
}
COVERAGE_GENERIC_TERMS = OBJECTIVE_STOPWORDS | {
    "based",
    "core",
    "deep",
    "evidence",
    "known",
    "learning",
    "main",
    "major",
    "mechanism",
    "mechanisms",
    "recent",
    "research",
    "standard",
}
COVERAGE_EVIDENCE_TERMS = {
    "api",
    "application",
    "applications",
    "benchmark",
    "benchmarks",
    "challenge",
    "challenges",
    "complexity",
    "definition",
    "equation",
    "formula",
    "implementation",
    "limitation",
    "limitations",
    "metric",
    "metrics",
    "performance",
    "result",
    "results",
    "score",
    "scores",
}
QUERY_FILLER_TERMS = OBJECTIVE_STOPWORDS | {
    "about",
    "also",
    "are",
    "be",
    "been",
    "being",
    "can",
    "could",
    "did",
    "does",
    "doing",
    "eg",
    "e.g",
    "its",
    "main",
    "should",
    "such",
    "their",
    "them",
    "they",
    "using",
    "versus",
    "was",
    "were",
    "would",
}
EVIDENCE_QUERY_HINTS = {
    "api": ["official documentation", "api signature", "parameters", "usage example"],
    "applications": ["applications", "use cases", "examples"],
    "benchmark": ["benchmark", "results table", "scores", "metrics"],
    "comparison": ["comparison", "differences", "tradeoffs"],
    "complexity": ["time complexity", "memory complexity", "big O", "scaling"],
    "definition": ["definition", "overview", "purpose", "concept"],
    "equation": ["equation", "formula", "mathematical derivation", "score function", "variables"],
    "limitations": ["limitations", "challenges", "bottlenecks", "open questions"],
}


def rag_generation_model(model: str | None = None) -> str:
    """Use the same default model selection as the planner agent."""

    return (
        clean_model_name(model)
        or clean_model_name(os.environ.get("RESEARCH_PLANNER_MODEL"))
        or DEFAULT_RAG_GENERATION_MODEL
    )


def sub_question_query_rewrite_model(model: str | None = None) -> str:
    """Model used only for rewriting planner sub-questions into retrieval queries."""

    return (
        clean_model_name(os.environ.get("RAG_SUBQUESTION_QUERY_MODEL"))
        or DEFAULT_SUBQUESTION_QUERY_REWRITE_MODEL
        or rag_generation_model(model)
    )


def gap_query_model(model: str | None = None) -> str:
    """Model used for rewriting synthesis evidence gaps into retrieval queries."""

    return (
        clean_model_name(os.environ.get("RAG_GAP_QUERY_MODEL"))
        or DEFAULT_GAP_QUERY_MODEL
        or rag_generation_model(model)
    )


def clean_model_name(value: Any) -> str:
    """Normalize env-provided model names without changing valid ids."""

    return clean_text(value).strip("\"'“”‘’")


def retrieve_full_collection_enabled() -> bool:
    """Default RAG retrieval to the full Chroma collection, not one run scope."""

    value = clean_text(os.environ.get("RAG_RETRIEVE_FULL_COLLECTION", "true")).lower()
    return value not in {"0", "false", "no", "off"}


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
    include_planned_source_urls: bool = True,
    source_url_k: int = DEFAULT_REPORT_SOURCE_URL_K,
    rewrite_query: bool = True,
    print_rewritten_query: bool = True,
    model: str | None = None,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_tokens: int = DEFAULT_REPORT_MAX_TOKENS,
    include_retrieved_chunks: bool = False,
    retrieved_chunk_chars: int = DEFAULT_CONTEXT_BLOCK_CHARS,
    supporting_chunk_count: int = DEFAULT_REPORT_SUPPORTING_CHUNKS,
    browser_results: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Retrieve with planner-derived queries, then synthesize report-ready notes."""
    objective = clean_text(research_plan.get("objective"))
    if not objective:
        raise ValueError("research_plan.objective is required")

    queries = planner_tasks_to_rag_queries(research_plan)
    emit_progress(
        "tool_called",
        "Synthesis retrieving evidence from RAG index",
        agent="synthesis",
        tool="rag-retrieval",
        metadata={"query_count": len(queries)},
    )
    current_history_key = clean_text(history_key) or objective_key(objective, research_plan)
    objective_scope = resolve_objective_history_scope(
        objective=objective,
        current_history_key=current_history_key,
        chroma_path=chroma_path,
        collection_name=collection_name,
    )
    allowed_history_keys = objective_scope["history_keys"]
    retrieval_history_keys = [] if retrieve_full_collection_enabled() else allowed_history_keys
    retrieval_scope = "full_collection" if not retrieval_history_keys else "objective_history_keys"
    llm_query_result = llm_sub_question_retrieval_query_result(
        research_plan=research_plan,
        model=model,
    ) if rewrite_query else empty_llm_query_result()
    llm_sub_question_queries = llm_query_result["queries"]
    if print_rewritten_query and llm_sub_question_queries:
        print("LLM rewritten sub-question retrieval queries:")
        for query in llm_sub_question_queries:
            print(f"- {query}")
    rewritten_query = "\n".join(llm_sub_question_queries)
    precision_queries = precision_retrieval_queries(research_plan, objective=objective)
    retrieval_queries = dedupe_preserve_order(
        llm_sub_question_queries + queries + precision_queries
    )
    retrieved_context = multi_query_hybrid_retrieve(
        queries=retrieval_queries,
        chroma_path=chroma_path,
        collection_name=collection_name,
        top_k=top_k,
        per_query_k=per_query_k,
        semantic_k=semantic_k,
        bm25_k=bm25_k,
        history_keys=retrieval_history_keys,
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
    planned_source_urls = planner_source_urls(research_plan)
    question_source_urls = planner_question_source_urls(research_plan)
    source_coverage_results = []
    if include_planned_source_urls and planned_source_urls:
        source_coverage_results = source_url_coverage_retrieve(
            source_urls=missing_or_weak_source_urls(planned_source_urls, retrieved_context),
            query=planner_context_query(research_plan, retrieval_queries),
            chroma_path=chroma_path,
            collection_name=collection_name,
            history_keys=retrieval_history_keys,
            top_k_per_url=max(source_url_k, 3),
            scan_limit=bm25_scan_limit,
        )
        retrieved_context = merge_retrieved_context(retrieved_context, source_coverage_results)

    question_source_coverage_results = []
    if include_planned_source_urls and question_source_urls:
        question_source_coverage_results = question_source_coverage_retrieve(
            question_source_urls=question_source_urls,
            retrieved_context=retrieved_context,
            chroma_path=chroma_path,
            collection_name=collection_name,
            history_keys=retrieval_history_keys,
            top_k_per_url=source_url_k,
            scan_limit=bm25_scan_limit,
        )
        retrieved_context = merge_retrieved_context(retrieved_context, question_source_coverage_results)

    browser_signal_context = browser_signal_results(browser_results or [], objective=objective)
    if browser_signal_context:
        retrieved_context = merge_retrieved_context(browser_signal_context, retrieved_context)

    synthesis_instruction = clean_text(research_plan.get("synthesis_instruction"))
    planner_questions = planner_sub_questions(research_plan)
    synthesis_context = select_synthesis_context(
        retrieved_context,
        planner_questions,
        question_source_urls=question_source_urls,
    )
    print_synthesis_chunks(synthesis_context, label="initial")
    payload = synthesize_context_for_report(
        objective=objective,
        retrieved_context=synthesis_context,
        synthesis_instruction=synthesis_instruction,
        planner_questions=planner_questions,
        question_source_urls=question_source_urls,
        model=model,
        max_context_chars=max_context_chars,
        max_tokens=max_tokens,
    )
    gap_query_plan = synthesis_gap_retrieval_plan(
        payload.get("synthesis"),
        objective=objective,
        synthesis_instruction=synthesis_instruction,
        sources=payload.get("sources", []),
        model=model,
        max_queries=DEFAULT_GAP_RETRIEVAL_MAX_QUERIES,
    )
    gap_queries = gap_query_plan["queries"]
    gap_retry_results = []
    gap_retry_count = 0
    gap_new_chunk_count = 0
    if gap_queries:
        gap_retry_results = multi_query_hybrid_retrieve(
            queries=gap_queries,
            chroma_path=chroma_path,
            collection_name=collection_name,
            top_k=DEFAULT_GAP_RETRIEVAL_TOP_K,
            per_query_k=DEFAULT_GAP_RETRIEVAL_PER_QUERY_K,
            semantic_k=semantic_k,
            bm25_k=max(bm25_k, DEFAULT_GAP_RETRIEVAL_TOP_K),
            history_keys=retrieval_history_keys,
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
        retry_context = merge_retrieved_context(gap_retry_results, retrieved_context)
        gap_new_chunk_count = max(0, len(retry_context) - len(retrieved_context))
        if gap_new_chunk_count:
            retrieved_context = retry_context
            gap_retry_count = 1
            synthesis_context = select_synthesis_context(
                retrieved_context,
                planner_questions,
                question_source_urls=question_source_urls,
            )
            print_synthesis_chunks(synthesis_context, label="gap-refresh")
            payload = synthesize_context_for_report(
                objective=objective,
                retrieved_context=synthesis_context,
                synthesis_instruction=synthesis_instruction,
                planner_questions=planner_questions,
                question_source_urls=question_source_urls,
                model=model,
                max_context_chars=max_context_chars,
                max_tokens=max_tokens,
            )
    payload["queries"] = queries
    payload["rewritten_query"] = rewritten_query
    payload["retrieval_queries"] = retrieval_queries
    payload["llm_sub_question_retrieval_queries"] = llm_sub_question_queries
    payload["llm_sub_question_query_model"] = llm_query_result["model"]
    payload["llm_sub_question_query_error"] = llm_query_result["error"]
    payload["llm_sub_question_query_raw_response"] = llm_query_result["raw_response"]
    payload["precision_retrieval_queries"] = precision_queries
    payload["gap_retrieval_queries"] = gap_queries
    payload["gap_query_model"] = gap_query_plan["model"]
    payload["gap_query_source"] = gap_query_plan["source"]
    payload["llm_gap_query_count"] = len(gap_query_plan["llm_queries"])
    payload["fallback_gap_query_count"] = len(gap_query_plan["fallback_queries"])
    payload["gap_query_error"] = gap_query_plan["llm_error"]
    payload["llm_gap_query_raw_response"] = gap_query_plan["llm_raw_response"]
    payload["gap_retrieved_count"] = len(gap_retry_results)
    payload["gap_new_chunk_count"] = gap_new_chunk_count
    payload["gap_retry_count"] = gap_retry_count
    payload["history_key"] = current_history_key
    payload["allowed_history_keys"] = allowed_history_keys
    payload["retrieval_history_keys"] = retrieval_history_keys
    payload["retrieval_scope"] = retrieval_scope
    payload["objective_scope"] = objective_scope
    payload["planned_source_urls"] = planned_source_urls
    payload["question_source_urls"] = question_source_urls
    payload["source_coverage_count"] = len(source_coverage_results)
    payload["question_source_coverage_count"] = len(question_source_coverage_results)
    payload["browser_signal_count"] = len(browser_signal_context)
    payload["retrieved_count"] = len(retrieved_context)
    payload["synthesis_context_count"] = len(synthesis_context)
    payload["citation_policy"] = (
        "Use only the numbered source indexes in sources. Prefer primary papers, "
        "official documentation, academic sources, and authoritative surveys for "
        "technical claims. Use secondary explainers only for intuition. If the "
        "synthesis marks a formula, number, API detail, or definition as missing "
        "evidence, do not add it to the report unless it appears in a supporting chunk."
    )
    citation_audit = audit_synthesis_citations(payload.get("synthesis"), payload.get("sources", []))
    payload["citation_audit"] = citation_audit
    payload["sub_question_specs"] = planner_sub_question_specs(research_plan)
    payload["coverage_by_question"] = build_coverage_by_question(
        payload.get("synthesis"),
        payload["sub_question_specs"],
        payload.get("sources", []),
        evidence_packs=payload.get("evidence_packs", []),
        retrieved_context=synthesis_context,
    )
    payload["supporting_chunks"] = report_supporting_chunks(
        retrieved_context,
        payload.get("sources", []),
        max_chunks=supporting_chunk_count,
        max_chars=retrieved_chunk_chars,
        cited_source_indexes=dedupe_ints(
            [
                *citation_audit.get("valid_referenced_source_indexes", []),
                *coverage_source_indexes(payload["coverage_by_question"]),
            ]
        ),
    )
    payload["diagnostics"] = synthesis_diagnostics(payload, retrieved_context)
    if include_retrieved_chunks:
        payload["retrieved_chunks"] = compact_retrieved_chunks(
            retrieved_context,
            payload.get("sources", []),
            max_chars=retrieved_chunk_chars,
        )
    return payload


def print_synthesis_chunks(retrieved_context: Sequence[RetrievalResult], label: str = "initial") -> None:
    """Print final chunks passed to the synthesis LLM after retrieval/reranking."""

    if os.environ.get("RAG_PRINT_SYNTHESIS_CHUNKS", "1").lower() in {"0", "false", "no"}:
        return
    try:
        limit = int(os.environ.get("RAG_PRINT_SYNTHESIS_CHUNKS_LIMIT", DEFAULT_SYNTHESIS_CHUNK_PRINT_LIMIT))
    except ValueError:
        limit = DEFAULT_SYNTHESIS_CHUNK_PRINT_LIMIT
    limit = max(1, limit)
    print(f"[synthesis] chunks passed to synthesizer after reranking ({label}): {len(retrieved_context)}")
    for rank, result in enumerate(retrieved_context[:limit], start=1):
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        url = primary_source_url(metadata)
        title = clean_text(metadata.get("title")) or url or "unknown source"
        source_kind = "primary/paper" if primary_source_url_like(url) else "secondary"
        preview = clean_text(display_document_preview(result.document, max_chars=180))
        rerank_note = f", rerank={result.rerank_score:.3f}" if result.rerank_score else ""
        print(
            f"[synthesis] chunk {rank}: score={result.score:.3f}{rerank_note}, "
            f"{source_kind}, url={url}, title={title}, preview={preview}"
        )
    if len(retrieved_context) > limit:
        print(f"[synthesis] ... {len(retrieved_context) - limit} more chunk(s) not printed")


def planner_tasks_to_rag_queries(research_plan: dict[str, Any]) -> list[str]:
    """Build retrieval queries from PlannerAgent output.

    Sub-questions are the primary retrieval queries because they are already
    concise search intents. Task details are used only as a fallback.
    """
    objective = clean_text(research_plan.get("objective"))
    sub_question_queries = []
    tasks = [task for task in research_plan.get("tasks", []) if isinstance(task, dict)]

    for sub_question in research_plan.get("sub_questions", []):
        query = clean_text(sub_question)
        if query:
            sub_question_queries.extend(
                sub_question_retrieval_queries(
                    query,
                    objective=objective,
                    task_details=matching_task_details(query, tasks),
                )
            )

    if sub_question_queries:
        if objective:
            sub_question_queries.append(objective)
        return dedupe_preserve_order(sub_question_queries)

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

    if objective:
        queries.append(objective)
    return dedupe_preserve_order(queries)


def sub_question_retrieval_queries(
    question: str,
    objective: str = "",
    task_details: str = "",
    max_variants: int = DEFAULT_SUBQUESTION_QUERY_VARIANTS,
) -> list[str]:
    """Rewrite one planner sub-question into broad retrieval intents."""

    question = clean_text(question)
    objective = clean_text(objective)
    task_details = clean_text(task_details)
    if not question:
        return []

    source_terms = source_query_terms(task_details)
    topic = retrieval_topic_phrase(f"{question} {task_details}", limit=12)
    key_terms = " ".join(query_keywords(question, limit=10)) or topic or objective
    hints = " ".join(broad_query_hints(f"{question} {task_details}"))
    variants = [
        clean_text(f"What source-backed context explains {topic} {hints}?"),
        clean_text(f"Which evidence gives {key_terms} details, examples, equations, benchmarks, or limitations?"),
        clean_text(f"Where do authoritative sources discuss {objective} {topic} {source_terms} source sections evidence?"),
    ]
    return dedupe_preserve_order(variant[:700] for variant in variants)[: max(1, max_variants)]


def llm_sub_question_retrieval_query_result(
    research_plan: dict[str, Any],
    model: str | None = None,
    max_variants: int = DEFAULT_SUBQUESTION_QUERY_VARIANTS,
) -> dict[str, Any]:
    """Use the generation model to rewrite planner sub-questions into RAG queries."""

    objective = clean_text(research_plan.get("objective"))
    questions = planner_sub_questions(research_plan)
    selected_model = sub_question_query_rewrite_model(model)
    if not questions:
        return empty_llm_query_result(model=selected_model)
    if not os.environ.get("GROQ_API_KEY"):
        return empty_llm_query_result(model=selected_model, error="GROQ_API_KEY is not set")

    try:
        from groq import Groq
    except ImportError as error:
        return empty_llm_query_result(model=selected_model, error=str(error))

    tasks = [task for task in research_plan.get("tasks", []) if isinstance(task, dict)]
    prompt = f"""Research objective:
{objective}

Planner sub-questions and optional task details:
{format_sub_question_rewrite_items(questions, tasks)}

Rewrite each planner sub-question into 2-3 broad, high-recall RAG retrieval queries.
Requirements:
- Return exactly 2 or 3 queries for every sub-question, in the same order as the planner list.
- Write each query as a broad natural-language retrieval question, not a short keyword label.
- For each sub-question, create complementary queries: one broad concept/context query, one evidence/detail query, and one source/section query when source details exist.
- Do not output only narrow labels like "entity + equation"; include surrounding vocabulary that may appear near the answer in papers, docs, reports, or extracted web text.
- Keep each query tied to one planner sub-question. Do not let benchmark, limitation, or formula queries drift into unrelated sub-question groups.
- Preserve named entities, URLs, titles, years, model names, datasets, metrics, APIs, equations, aliases, and important technical terms from the planner/task details.
- For equation/formula questions, include nearby evidence terms such as equation, formula, derivation, score function, alignment, matrix, variables, components, or operation names when relevant.
- For benchmark questions, include dataset names, metrics, result table, scores, comparison, performance, and evaluation terms when relevant.
- For API/implementation questions, include official documentation, class/function names, signature, parameters, usage, and example terms when relevant.
- For limitation/complexity questions, include complexity, memory, runtime, scaling, tradeoffs, bottleneck, sparse, efficient, or alternatives when relevant.
- Prefer 8-22 words per query.
- Do not answer the question and do not add citations.
- Do not include reasoning, <think> text, or explanations.
- Never output placeholder labels or generic query names.
- Return JSON only in this shape:
{{"items":[{{"sub_question":"...","queries":["topic overview evidence","topic method comparison source"]}}]}}"""

    try:
        response = create_chat_completion_with_retries(
            Groq(),
            model=selected_model,
            temperature=0,
            max_tokens=900,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite planner sub-questions into broad, complementary RAG retrieval "
                        "queries that maximize recall across papers, docs, reports, and web extracts. "
                        "Return JSON only."
                    ),
                },
                {"role": "user", "content": prompt[:6000]},
            ],
        )
    except Exception as error:  # pragma: no cover - exercised through integration runs.
        return empty_llm_query_result(model=selected_model, error=str(error))

    raw_response = str(response.choices[0].message.content or "")
    queries = valid_retrieval_queries(parse_llm_retrieval_queries(raw_response), research_plan)
    raw_response_text = clean_text(raw_response)
    max_queries = max(1, len(questions) * max(1, max_variants))
    if not queries:
        return {
            "queries": [],
            "model": clean_text(getattr(response, "model", "")) or selected_model,
            "error": "LLM query rewrite returned no usable queries",
            "raw_response": raw_response_text[:1000],
        }
    queries = complete_sub_question_query_coverage(queries, research_plan, max_variants=max_variants)
    return {
        "queries": dedupe_preserve_order(query[:700] for query in queries)[:max_queries],
        "model": clean_text(getattr(response, "model", "")) or selected_model,
        "error": "",
        "raw_response": raw_response_text[:1000],
    }


def empty_llm_query_result(model: str = "", error: str = "") -> dict[str, Any]:
    return {"queries": [], "model": clean_text(model), "error": clean_text(error), "raw_response": ""}


def complete_sub_question_query_coverage(
    queries: Sequence[str],
    research_plan: dict[str, Any],
    max_variants: int = DEFAULT_SUBQUESTION_QUERY_VARIANTS,
) -> list[str]:
    questions = planner_sub_questions(research_plan)
    if not questions:
        return dedupe_preserve_order(queries)

    objective = clean_text(research_plan.get("objective"))
    tasks = [task for task in research_plan.get("tasks", []) if isinstance(task, dict)]
    clean_queries = dedupe_preserve_order(queries)
    grouped = []
    used = set()
    min_variants = min(max(1, max_variants), 2)
    for question in questions:
        question_queries = [
            query for query in clean_queries
            if query not in used and query_matches_sub_question(question, query)
        ][: max(1, max_variants)]
        used.update(question_queries)
        fallback_queries = sub_question_retrieval_queries(
            question,
            objective=objective,
            task_details=matching_task_details(question, tasks),
            max_variants=max_variants,
        )
        target_count = max(1, max_variants) if not question_queries else min_variants
        for query in fallback_queries:
            if len(question_queries) >= target_count:
                break
            if query not in question_queries:
                question_queries.append(query)
        grouped.extend(question_queries[: max(1, max_variants)])

    max_queries = max(1, len(questions) * max(1, max_variants))
    return dedupe_preserve_order(grouped)[:max_queries]


def query_list_covers_sub_question(question: str, queries: Sequence[str]) -> bool:
    return any(query_matches_sub_question(question, query) for query in queries)


def query_matches_sub_question(question: str, query: str) -> bool:
    query_terms = query_tokens(query)
    named_terms = question_named_terms(question)
    if named_terms and not any(term_matches_query(term, query_terms, query) for term in named_terms):
        return False

    ignored_terms = COVERAGE_GENERIC_TERMS | COVERAGE_EVIDENCE_TERMS | {"attention", "method", "methods", "topic", "topics"}
    topic_terms = [term for term in query_keywords(question, limit=8) if term.lower() not in ignored_terms]
    normalized = {term.lower().replace("‑", "-").replace("–", "-") for term in topic_terms}
    evidence_terms = {
        term.lower()
        for term in query_keywords(question, limit=12)
        if term.lower() in COVERAGE_EVIDENCE_TERMS
    }
    if not normalized:
        if evidence_terms:
            return bool(evidence_terms & query_terms)
        return bool(clean_text(query))
    topic_overlap = len(normalized & query_terms)
    if len(normalized) == 1 and evidence_terms:
        return topic_overlap > 0 and bool(evidence_terms & query_terms)
    return topic_overlap >= min(2, len(normalized))


def term_matches_query(term: str, query_terms: set[str], query: str) -> bool:
    normalized = clean_text(term).lower().replace("‑", "-").replace("–", "-")
    compact = normalized.replace("-", "")
    query_text = clean_text(query).lower().replace("‑", "-").replace("–", "-")
    return normalized in query_terms or compact in {token.replace("-", "") for token in query_terms} or normalized in query_text


def question_named_terms(question: str) -> list[str]:
    normalized = clean_text(question).replace("‑", "-").replace("–", "-").replace("—", "-")
    terms = re.findall(r"\b[A-Z][A-Za-z0-9_.-]{2,}\b|\b[A-Z]{2,}\b", normalized)
    terms.extend(re.findall(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b", normalized))
    return dedupe_preserve_order(term.lower() for term in terms if term.lower() not in OBJECTIVE_STOPWORDS)


def format_sub_question_rewrite_items(questions: Sequence[str], tasks: Sequence[dict[str, Any]]) -> str:
    lines = []
    for index, question in enumerate(questions, start=1):
        details = matching_task_details(question, tasks)
        detail_text = f"\n   task_details: {details}" if details else ""
        lines.append(f"{index}. {question}{detail_text}")
    return "\n".join(lines)


def parse_llm_retrieval_queries(raw_response: str) -> list[str]:
    text = clean_query_generation_response(raw_response)
    if generated_query_output_is_noise(text):
        return []
    try:
        parsed = json.loads(json_payload(text))
    except (TypeError, ValueError):
        return fallback_json_like_queries(text) or fallback_line_queries(text)
    return extract_query_strings(parsed)


def clean_query_generation_response(text: Any) -> str:
    return strip_thinking_blocks(clean_markdown_fence(str(text or "")))


def strip_thinking_blocks(text: Any) -> str:
    value = str(text or "")
    value = re.sub(r"(?is)<think>.*?</think>", " ", value)
    open_think = re.search(r"(?is)<think>", value)
    if open_think:
        prefix = value[: open_think.start()]
        tail = value[open_think.end() :]
        payload = re.search(r"(\{.*\}|\[.*\])", tail, flags=re.DOTALL)
        value = f"{prefix}\n{payload.group(1)}" if payload else prefix
    return value.strip()


def json_payload(text: str) -> str:
    text = clean_text(text)
    if text.startswith("{") or text.startswith("["):
        return text
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    return match.group(1) if match else text


def clean_markdown_fence(text: str) -> str:
    return re.sub(r"^```(?:json)?|```$", "", str(text or "").strip(), flags=re.IGNORECASE).strip()


def fallback_json_like_queries(text: str) -> list[str]:
    """Recover query arrays when a model returns almost-JSON that json.loads rejects."""

    queries = []
    for match in re.finditer(r'"queries"\s*:\s*\[(.*?)\]', str(text or ""), flags=re.DOTALL | re.IGNORECASE):
        queries.extend(re.findall(r'"([^"]+)"', match.group(1)))
    return dedupe_preserve_order(queries)


def extract_query_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        queries = []
        for key, item in value.items():
            key_name = key.lower()
            if key_name in {"sub_question", "question"}:
                continue
            if key_name in {"query", "text"} and isinstance(item, str):
                queries.append(item)
            else:
                queries.extend(extract_query_strings(item))
        return dedupe_preserve_order(queries)
    if isinstance(value, list):
        queries = []
        for item in value:
            queries.extend(extract_query_strings(item))
        return dedupe_preserve_order(queries)
    if isinstance(value, str):
        return fallback_line_queries(value)
    return []


def fallback_line_queries(text: str) -> list[str]:
    lines = []
    for line in str(text or "").splitlines():
        line = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip().strip('"')
        if clean_text(line):
            lines.append(line)
    return dedupe_preserve_order(lines)


def valid_retrieval_queries(queries: Sequence[str], research_plan: dict[str, Any]) -> list[str]:
    objective_terms = query_tokens(clean_text(research_plan.get("objective")))
    question_terms = set()
    for question in planner_sub_questions(research_plan):
        question_terms.update(query_tokens(question))
    allowed_terms = objective_terms | question_terms
    return [
        query
        for query in dedupe_preserve_order(queries)
        if is_valid_retrieval_query(query, allowed_terms)
    ]


def is_valid_retrieval_query(query: str, allowed_terms: set[str]) -> bool:
    text = clean_text(query)
    if not is_valid_generated_query(text, max_chars=420):
        return False
    if len(re.findall(r"\S+", text)) < 3:
        return False
    return True


def is_valid_generated_query(query: str, max_chars: int = 600) -> bool:
    text = clean_text(query)
    lowered = text.lower()
    if not text or len(text) > max_chars:
        return False
    if generated_query_output_is_noise(text):
        return False
    if not re.search(r"[A-Za-z]", text):
        return False
    if re.fullmatch(r"(?:\[\d+\]|\(?\d+\)?|and|or|,|\s)+", lowered):
        return False
    if re.fullmatch(r"(?:query|retrieval query|search query|example query)\s*\d*", lowered):
        return False
    if re.fullmatch(r"(?:query|retrieval|search|example)(?:\s+\d+)?", lowered):
        return False
    if re.search(r"\b(?:query|retrieval query|search query|example query)\s*\d+\b", lowered):
        return False
    return True


def generated_query_output_is_noise(text: Any) -> bool:
    lowered = clean_text(text).lower()
    if not lowered:
        return False
    noise_signals = (
        "<think",
        "</think",
        "thinking process",
        "analyze user input",
        "research objective:",
        "planner sub-questions",
        "requirements:",
        "output json",
        "return json only",
        "do not answer",
    )
    return any(signal in lowered for signal in noise_signals)


def precision_retrieval_queries(
    research_plan: dict[str, Any],
    objective: str = "",
    max_queries: int = DEFAULT_PRECISION_QUERY_LIMIT,
) -> list[str]:
    """Build deterministic high-signal queries for exact evidence retrieval."""

    objective_text = clean_text(objective or research_plan.get("objective"))
    synthesis_instruction = clean_text(research_plan.get("synthesis_instruction"))
    tasks = [task for task in research_plan.get("tasks", []) if isinstance(task, dict)]
    candidates = [
        *planner_sub_questions(research_plan),
        *instruction_requirement_items(synthesis_instruction),
    ]
    queries = []
    for item in dedupe_preserve_order(candidates):
        base = clean_text(f"{objective_text} {item} {matching_task_details(item, tasks)}")
        suffixes = precision_query_suffixes(item)
        if not base or not suffixes:
            continue
        for suffix in suffixes:
            queries.append(clean_text(f"{base} {suffix}")[:700])
            if len(queries) >= max(1, max_queries):
                return dedupe_preserve_order(queries)
    return dedupe_preserve_order(queries)[: max(1, max_queries)]


def precision_query_suffixes(text: str) -> list[str]:
    lowered = clean_text(text).lower()
    suffixes = []
    if re.search(r"\b(equation|formula|mathematical|formulation|derive|scaled|additive|multiplicative)\b", lowered):
        suffixes.append("exact equation formula derivation symbols softmax sqrt sum matrix alignment")
    if re.search(r"\b(benchmark|result|score|performance|accuracy|bleu|glue|imagenet|top-1|metric)\b", lowered):
        suffixes.append("benchmark table results scores metrics accuracy BLEU GLUE ImageNet top-1")
    if re.search(r"\b(api|implementation|framework|library|pytorch|tensorflow|keras|hugging face|transformers)\b", lowered):
        suffixes.append("official documentation API signature parameters usage example class function")
    if re.search(r"\b(complexity|efficient|variant|limitation|memory|time|quadratic|linear|sparse|low-rank)\b", lowered):
        suffixes.append("computational complexity time memory O(n^2) O(n) algorithm approximation limitation")
    if re.search(r"\b(paper|contribution|introduced|architecture|method|model|et al)\b", lowered):
        suffixes.append("original paper method contribution architecture equations results")
    return dedupe_preserve_order(suffixes)


def matching_task_details(question: str, tasks: Sequence[dict[str, Any]]) -> str:
    """Return task details that make a planner sub-question more retrievable."""

    question_tokens = query_tokens(question)
    parts = []
    for task in tasks:
        context = clean_text(task.get("query_context"))
        context_tokens = query_tokens(context)
        if not context_tokens:
            continue
        overlap = len(question_tokens & context_tokens) / max(1, min(len(question_tokens), len(context_tokens)))
        if context.lower() != clean_text(question).lower() and overlap < 0.45:
            continue
        expected_signals = " ".join(clean_text(item) for item in task.get("expected_signals", []) if clean_text(item))
        parts.extend([task.get("extraction_goal"), expected_signals, task.get("url")])
    return clean_text(" ".join(clean_text(part) for part in parts if clean_text(part)))


def query_tokens(text: str) -> set[str]:
    text = clean_text(text).replace("‑", "-").replace("–", "-").replace("—", "-")
    stopwords = {"and", "are", "for", "from", "how", "the", "what", "with"}
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_+#.-]+", text)
        if len(token) > 2 and token.lower() not in stopwords
    }


def query_keywords(text: str, limit: int = 10) -> list[str]:
    keywords = []
    for token in re.findall(r"[A-Za-z0-9_+#.-]+", clean_text(text)):
        lowered = token.lower().strip(".")
        if len(lowered) <= 2 or lowered in QUERY_FILLER_TERMS:
            continue
        keywords.append(token.strip(".,;:()[]{}"))
    return dedupe_preserve_order(keywords)[: max(1, limit)]


def retrieval_topic_phrase(text: str, limit: int = 14) -> str:
    normalized = clean_text(text).replace("‑", "-").replace("–", "-").replace("—", "-")
    normalized = re.sub(r"(?i)\b(?:what|how|why|when|where|which)\s+(?:is|are|does|do|did|can|should)?\b", " ", normalized)
    normalized = re.sub(r"(?i)\b(?:as|e\.g\.|eg)\b", " ", normalized)
    return " ".join(query_keywords(normalized, limit=limit))


def source_query_terms(text: str, limit: int = 8) -> str:
    urls = [match.group(0).rstrip(".,;:") for match in URL_PATTERN.finditer(clean_text(text))]
    terms = [term for term in query_keywords(text, limit=limit) if not URL_PATTERN.search(term)]
    return clean_text(" ".join([*urls[:2], *terms[:limit]]))[:300]


def broad_query_hints(text: str) -> list[str]:
    lowered = clean_text(text).lower()
    hints = ["overview", "evidence"]
    hint_rules = [
        (r"\b(defin|concept|what is)\b", ["definition", "concept"]),
        (r"\b(equation|formula|mathematical|formulation)\b", ["formula", "equation"]),
        (r"\b(benchmark|metric|score|accuracy|performance|result)\b", ["benchmark", "metrics"]),
        (r"\b(api|implementation|framework|library|function|class)\b", ["implementation", "api"]),
        (r"\b(complexity|limitation|memory|runtime|efficient|trade[- ]?off)\b", ["complexity", "limitations"]),
        (r"\b(compare|comparison|versus|difference|variant|type)\b", ["comparison", "variants"]),
        (r"\b(application|use case|example)\b", ["applications", "examples"]),
    ]
    for pattern, words in hint_rules:
        if re.search(pattern, lowered):
            hints.extend(words)
    return dedupe_preserve_order(hints)


def planner_sub_questions(research_plan: dict[str, Any]) -> list[str]:
    """Return planner sub-questions without mixing in fallback task text."""

    return dedupe_preserve_order(
        clean_text(question)
        for question in research_plan.get("sub_questions", [])
        if clean_text(question)
    )


def planner_sub_question_specs(research_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return stable per-question coverage specs from planner output."""

    raw_specs = research_plan.get("sub_question_specs")
    specs = []
    if isinstance(raw_specs, list):
        for index, item in enumerate(raw_specs, start=1):
            if not isinstance(item, dict):
                continue
            question = clean_text(item.get("question"))
            if not question:
                continue
            specs.append(
                {
                    "question_id": clean_text(item.get("question_id")) or f"q{index:03d}",
                    "question": question,
                    "required_evidence": clean_string_list(item.get("required_evidence")) or infer_question_evidence_types(question),
                }
            )

    if specs:
        return specs
    return [
        {
            "question_id": f"q{index:03d}",
            "question": question,
            "required_evidence": infer_question_evidence_types(question),
        }
        for index, question in enumerate(planner_sub_questions(research_plan), start=1)
    ]


def build_coverage_by_question(
    synthesis: Any,
    sub_question_specs: Sequence[dict[str, Any]],
    sources: Sequence[dict[str, Any]],
    evidence_packs: Sequence[dict[str, Any]] | None = None,
    retrieved_context: Sequence[RetrievalResult] | None = None,
) -> list[dict[str, Any]]:
    """Summarize synthesis coverage for each planner sub-question."""

    synthesis_text = normalize_citation_markers(synthesis)
    source_index_set = source_indexes(sources)
    packs_by_question = evidence_packs_by_question(evidence_packs or [])
    coverage = []
    for spec in sub_question_specs:
        question = clean_text(spec.get("question")) if isinstance(spec, dict) else ""
        required_evidence = clean_string_list(spec.get("required_evidence")) if isinstance(spec, dict) else []
        section = synthesis_section_for_question(synthesis_text, question)
        cited = [index for index in citation_markers(section) if index in source_index_set]
        status = infer_synthesis_coverage_status(section)
        pack = packs_by_question.get(question_key(question), {})
        pack_indexes = pack_source_indexes(pack)
        context_matches = coverage_matches_for_question(
            question,
            retrieved_context or [],
            sources,
            required_evidence=required_evidence,
        )
        context_indexes = [match["source_index"] for match in context_matches]
        source_indexes_for_question = dedupe_ints([*cited, *pack_indexes, *context_indexes])
        if status == "covered" and cited and not pack_indexes and not context_matches:
            status = "partial"
        if pack_indexes:
            status = merge_coverage_status(status, clean_text(pack.get("coverage")) or "partial")
        status = merge_coverage_status(status, infer_context_coverage_status(context_matches, section))
        coverage.append(
            {
                "question_id": clean_text(spec.get("question_id")) if isinstance(spec, dict) else "",
                "question": question,
                "required_evidence": required_evidence,
                "status": status,
                "source_indexes": source_indexes_for_question,
                "has_citations": bool(source_indexes_for_question),
                "evidence_count": (len(pack.get("chunks", [])) if isinstance(pack, dict) else 0) + len(context_matches),
                "missing_reason": coverage_missing_reason(status, context_matches, section),
            }
        )
    return coverage


def coverage_matches_for_question(
    question: str,
    retrieved_context: Sequence[RetrievalResult],
    sources: Sequence[dict[str, Any]],
    required_evidence: Sequence[str] | None = None,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Return chunks that support a planner question with topic and evidence signals."""

    question_terms = coverage_question_tokens(question)
    topic_terms = {term for term in question_terms if term not in COVERAGE_EVIDENCE_TERMS}
    evidence_types = clean_string_list(list(required_evidence or [])) or infer_question_evidence_types(question)
    source_index_by_id = citation_index_by_chunk_id(sources or [])
    source_index_by_url = citation_index_by_source_url(sources or [])
    matches = []
    for result in retrieved_context:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        url = primary_source_url(metadata)
        source_index = source_index_by_id.get(result.id) or source_index_by_url.get(source_url_key(url))
        if source_index is None:
            continue
        title = clean_text(metadata.get("title")) or url
        content = retrieved_chunk_preview(result.document, metadata, max_chars=500) or clean_text(result.document)[:500]
        if len(query_tokens(content)) < 3:
            continue
        text = clean_text(" ".join([title, url, content]))
        chunk_terms = coverage_question_tokens(text)
        topic_overlap = topic_terms & chunk_terms
        evidence_score = evidence_type_score(text, evidence_types)
        if topic_terms and not topic_overlap:
            continue
        if evidence_types and not evidence_score:
            continue
        score = len(question_terms & chunk_terms) + evidence_score + (1 if is_primary_source(metadata) else 0)
        matches.append(
            (
                score,
                {
                    "source_index": source_index,
                    "id": result.id,
                    "title": title,
                    "is_primary_source": is_primary_source(metadata),
                },
            )
        )
    return [
        {
            "source_index": chunk.get("source_index"),
            "id": chunk.get("id"),
            "title": chunk.get("title"),
            "score": score,
        }
        for score, chunk in sorted(matches, key=lambda item: -item[0])[: max(1, limit)]
        if chunk.get("source_index") is not None
    ]


def coverage_question_tokens(text: str) -> set[str]:
    return {
        token
        for token in query_tokens(text)
        if token not in COVERAGE_GENERIC_TERMS
    }


def evidence_type_score(text: str, evidence_types: Sequence[str]) -> int:
    lowered = clean_text(text).lower()
    if not evidence_types:
        return 1 if lowered else 0
    patterns = {
        "api": r"\b(api|class|function|method|parameter|argument|signature|constructor|usage example|official docs?)\b",
        "applications": r"\b(application|use case|deployed|used for|vision|nlp|classification|translation)\b",
        "benchmark": r"\b(benchmark|score|metric|accuracy|bleu|glue|imagenet|result|performance|\d+(?:\.\d+)?\s*%)\b",
        "comparison": r"\b(compare|comparison|versus| vs |different|difference|whereas|while)\b",
        "complexity": r"\b(complexity|runtime|memory|quadratic|linear|o\(|o\(n|efficient|scalability)\b",
        "definition": r"\b(definition|defined as|refers to|means|is a|are a|purpose)\b",
        "equation": r"(?:\\(?:frac|sum|sqrt)|[=∑Σ√]|softmax|equation|formula|where\s+[A-Za-z])",
        "limitations": r"\b(limitation|challenge|drawback|constraint|bottleneck|weakness|open question)\b",
    }
    score = 0
    for evidence_type in evidence_types:
        key = clean_text(evidence_type).lower()
        pattern = patterns.get(key)
        if pattern and re.search(pattern, lowered):
            score += 2
        elif key and key in lowered:
            score += 1
    return score


def infer_context_coverage_status(matches: Sequence[dict[str, Any]], section: str = "") -> str:
    if matches:
        return "covered"
    return "partial" if citation_markers(section) else "missing"


def merge_coverage_status(*statuses: str) -> str:
    ranks = {"missing": 0, "partial": 1, "covered": 2}
    best = "missing"
    for status in statuses:
        normalized = clean_text(status).lower()
        if ranks.get(normalized, 0) > ranks[best]:
            best = normalized
    return best


def coverage_missing_reason(status: str, matches: Sequence[dict[str, Any]], section: str) -> str:
    if status == "covered":
        return ""
    if not matches:
        return "No retrieved chunk matched both the planner topic and required evidence signals."
    if not citation_markers(section):
        return "Supporting chunks exist, but the synthesis section did not cite them directly."
    return "Coverage is partial."


def coverage_gap_items(coverage: Sequence[dict[str, Any]]) -> list[str]:
    gaps = []
    for item in coverage or []:
        if not isinstance(item, dict) or clean_text(item.get("status")).lower() == "covered":
            continue
        question = clean_text(item.get("question"))
        reason = clean_text(item.get("missing_reason"))
        gaps.append(clean_text(f"{question}: {reason}") if reason else question)
    return dedupe_preserve_order(gaps)


def synthesis_section_for_question(synthesis: str, question: str) -> str:
    if not question:
        return ""
    question_text = clean_text(question)
    location = synthesis.lower().find(question_text.lower())
    if location < 0:
        tokens = query_tokens(question_text)
        return "\n".join(
            line for line in synthesis.splitlines()
            if len(tokens & query_tokens(line)) >= max(2, min(4, len(tokens)))
        )
    next_heading = re.search(r"\n\s*(?:#{1,6}\s+|\d+\.\s+|[-*]\s+)?(?:What|How|Why|Which|When|Where)\b", synthesis[location + len(question_text):], flags=re.IGNORECASE)
    end = location + len(question_text) + next_heading.start() if next_heading else min(len(synthesis), location + 1800)
    return synthesis[location:end]


def infer_synthesis_coverage_status(text: str) -> str:
    lowered = clean_text(text).lower()
    if not lowered:
        return "missing"
    if re.search(r"\bmissing evidence\b|\bnot present\b|\bno explicit\b|\bunavailable\b", lowered):
        return "missing"
    if re.search(r"\bpartial(?:ly)?\b|\blimited\b|\bweak\b", lowered):
        return "partial"
    if re.search(r"\bcovered\b|\bstrong\b|\bsupported\b", lowered) or citation_markers(text):
        return "covered"
    return "partial"


def evidence_packs_by_question(evidence_packs: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    packs = {}
    for pack in evidence_packs or []:
        if not isinstance(pack, dict):
            continue
        key = question_key(pack.get("question"))
        if key:
            packs[key] = pack
    return packs


def pack_source_indexes(pack: dict[str, Any]) -> list[int]:
    indexes = []
    for chunk in pack.get("chunks", []) if isinstance(pack, dict) else []:
        if not isinstance(chunk, dict):
            continue
        index = chunk.get("source_index")
        if isinstance(index, int) and index not in indexes:
            indexes.append(index)
    return indexes


def coverage_source_indexes(coverage: Sequence[dict[str, Any]]) -> list[int]:
    values = []
    for item in coverage or []:
        if isinstance(item, dict):
            values.extend(item.get("source_indexes", []))
    return dedupe_ints(values)


def dedupe_ints(values: Sequence[Any]) -> list[int]:
    deduped = []
    seen = set()
    for value in values or []:
        if isinstance(value, bool):
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number in seen:
            continue
        seen.add(number)
        deduped.append(number)
    return deduped


def question_key(question: Any) -> str:
    return clean_text(question).lower()


def infer_question_evidence_types(question: str) -> list[str]:
    lowered = clean_text(question).lower()
    checks = [
        ("definition", r"\b(what is|definition|define|purpose|overview)\b"),
        ("equation", r"\b(equations?|formulas?|formulations?|mathematical|components?)\b"),
        ("comparison", r"\b(compare|comparison|versus| vs |differ|differences?)\b"),
        ("benchmark", r"\b(benchmark|score|performance|metric|accuracy|bleu|glue|imagenet|result)\b"),
        ("api", r"\b(api|pytorch|tensorflow|keras|implementation|code|signature|usage)\b"),
        ("complexity", r"\b(complexity|memory|time|efficient|linear|quadratic|scalability)\b"),
        ("applications", r"\b(application|use case|vision|nlp|computer vision)\b"),
        ("limitations", r"\b(limitation|challenge|drawback|open question)\b"),
    ]
    evidence = [name for name, pattern in checks if re.search(pattern, lowered)]
    return evidence or ["evidence"]


def clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [clean_text(item) for item in value if clean_text(item)]


def planner_context_query(research_plan: dict[str, Any], fallback_queries: Sequence[str]) -> str:
    """Compact planner text used to retrieve better chunks from known sources."""

    parts = [
        research_plan.get("objective"),
        research_plan.get("synthesis_instruction"),
        *planner_sub_questions(research_plan),
    ]
    for task in research_plan.get("tasks", []):
        if not isinstance(task, dict):
            continue
        signals = " ".join(clean_text(item) for item in task.get("expected_signals", []) if clean_text(item))
        parts.extend([task.get("query_context"), task.get("extraction_goal"), signals])

    query = clean_text(" ".join(clean_text(part) for part in parts if clean_text(part)))
    if query:
        return query[:4000]
    return clean_text(" ".join(fallback_queries))[:4000]


def resolve_objective_history_scope(
    objective: str,
    current_history_key: str,
    chroma_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    similarity_threshold: float = DEFAULT_OBJECTIVE_SCOPE_SIMILARITY,
    max_history_keys: int = DEFAULT_OBJECTIVE_SCOPE_MAX_KEYS,
) -> dict[str, Any]:
    """Choose current plus similar previous objective scopes for retrieval."""

    objective = clean_text(objective)
    current_history_key = clean_text(current_history_key)
    selected = []
    if current_history_key:
        selected.append(
            {
                "history_key": current_history_key,
                "objective": objective,
                "similarity": 1.0,
                "reason": "current_objective",
            }
        )

    for record in list_indexed_objective_scopes(chroma_path, collection_name):
        history_key_value = clean_text(record.get("history_key"))
        previous_objective = clean_text(record.get("objective"))
        if not history_key_value or not previous_objective or history_key_value == current_history_key:
            continue
        similarity = objective_similarity(objective, previous_objective)
        if similarity < similarity_threshold:
            continue
        selected.append(
            {
                "history_key": history_key_value,
                "objective": previous_objective,
                "similarity": similarity,
                "reason": "similar_previous_objective",
            }
        )

    selected = sorted(
        selected,
        key=lambda item: (
            0 if item.get("reason") == "current_objective" else 1,
            -float(item.get("similarity") or 0.0),
            clean_text(item.get("objective")).lower(),
        ),
    )[: max(1, max_history_keys)]
    return {
        "history_keys": [item["history_key"] for item in selected],
        "selected_objectives": selected,
        "similarity_threshold": similarity_threshold,
    }


def list_indexed_objective_scopes(
    chroma_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    limit: int = 5000,
) -> list[dict[str, str]]:
    """Read distinct objective/history_key pairs already stored in Chroma metadata."""

    try:
        collection = get_collection(chroma_path, collection_name)
        result = collection.get(include=["metadatas"], limit=max(1, limit))
    except Exception:
        return []

    metadatas = result.get("metadatas", []) if isinstance(result, dict) else []
    records = []
    seen = set()
    for metadata in metadatas:
        if not isinstance(metadata, dict):
            continue
        history_key_value = clean_text(metadata.get("history_key"))
        objective_value = clean_text(metadata.get("objective"))
        if not history_key_value or not objective_value:
            continue
        key = (history_key_value, objective_value.lower())
        if key in seen:
            continue
        seen.add(key)
        records.append({"history_key": history_key_value, "objective": objective_value})
    return records


def objective_similarity(left: str, right: str) -> float:
    left_tokens = objective_tokens(left)
    right_tokens = objective_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens.intersection(right_tokens)
    if not overlap:
        return 0.0
    jaccard = len(overlap) / len(left_tokens.union(right_tokens))
    overlap_coefficient = len(overlap) / min(len(left_tokens), len(right_tokens))
    return max(jaccard, overlap_coefficient)


def objective_tokens(text: str) -> set[str]:
    tokens = set()
    for token in OBJECTIVE_TOKEN_PATTERN.findall(clean_text(text).lower()):
        normalized = normalize_objective_token(token)
        if not normalized or normalized in OBJECTIVE_STOPWORDS or len(normalized) < 3:
            continue
        tokens.add(normalized)
    return tokens


def normalize_objective_token(token: str) -> str:
    token = clean_text(token).lower()
    if token == "indian":
        return "india"
    if token in {"cultures", "cultural"}:
        return "culture"
    if token in {"transformers"}:
        return "transformer"
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def rewrite_query_from_planner_queries(
    objective: str,
    queries: Sequence[str],
    model: str | None = None,
    max_input_chars: int = 8000,
) -> str:
    """Rewrite all planner query text into one dense retrieval query."""
    objective = clean_text(objective)
    query_text = "\n".join(f"- {query}" for query in dedupe_preserve_order(queries))
    query_text = query_text[: max(1000, max_input_chars)].strip()
    if not query_text:
        return objective
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set")

    try:
        from groq import Groq
    except ImportError as error:
        raise RuntimeError("groq package is not installed. Install it with `pip install -r requirements.txt`.") from error

    prompt = f"""Research objective:
{objective}

Planner retrieval query contents:
{query_text}

Rewrite the planner query contents into one optimized RAG retrieval query.
Requirements:
- Preserve the core research intent, entities, URLs, technical terms, expected evidence, and subquestions.
- Remove duplicates and filler words.
- Keep it topic-agnostic and useful for semantic search plus BM25 keyword retrieval.
- Return only the rewritten query text, no bullets, no markdown, no explanation."""

    response = create_chat_completion_with_retries(
        Groq(),
        model=rag_generation_model(model),
        temperature=0,
        max_tokens=350,
        messages=[
            {
                "role": "system",
                "content": "You rewrite planner output into a single high-recall retrieval query. Return only the query.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    rewritten_query = clean_text(response.choices[0].message.content)
    return rewritten_query or objective


def planner_source_urls(research_plan: dict[str, Any]) -> list[str]:
    """Extract source URLs from any planner fields without topic-specific rules."""
    urls = []
    for value in nested_values(research_plan):
        if not isinstance(value, str):
            continue
        urls.extend(match.group(0).rstrip(".,;:") for match in URL_PATTERN.finditer(value))
    return dedupe_source_urls(urls)


def planner_question_source_urls(research_plan: dict[str, Any]) -> dict[str, list[str]]:
    """Map planner sub-questions to the task URLs assigned to them."""

    specs = planner_sub_question_specs(research_plan)
    question_by_id = {
        clean_text(spec.get("question_id")).lower(): clean_text(spec.get("question"))
        for spec in specs
        if clean_text(spec.get("question_id")) and clean_text(spec.get("question"))
    }
    question_by_text = {question_key(spec.get("question")): clean_text(spec.get("question")) for spec in specs}
    mapped: dict[str, list[str]] = {clean_text(spec.get("question")): [] for spec in specs if clean_text(spec.get("question"))}

    for task in research_plan.get("tasks", []) or []:
        if not isinstance(task, dict):
            continue
        url = first_url(task.get("url"))
        context = clean_text(task.get("query_context"))
        question = question_by_id.get(context.lower()) or question_by_text.get(question_key(context))
        if not url or not question:
            continue
        mapped.setdefault(question, []).append(url)

    return {question: dedupe_source_urls(urls) for question, urls in mapped.items() if urls}


def first_url(value: Any) -> str:
    text = clean_text(value)
    match = URL_PATTERN.search(text)
    return normalize_source_url(match.group(0).rstrip(".,;:")) if match else ""


def nested_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        values = []
        for item in value.values():
            values.extend(nested_values(item))
        return values
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(nested_values(item))
        return values
    return [value]


def missing_source_urls(
    planned_urls: Sequence[str],
    retrieved_context: Sequence[RetrievalResult],
) -> list[str]:
    covered_urls = set()
    for result in retrieved_context:
        covered_urls.update(result_source_urls_from_metadata(result.metadata))
    return [url for url in planned_urls if url not in covered_urls]


def missing_or_weak_source_urls(
    planned_urls: Sequence[str],
    retrieved_context: Sequence[RetrievalResult],
) -> list[str]:
    """Return planned URLs that do not have a meaningful retrieved evidence chunk."""

    covered_urls = set()
    for result in retrieved_context:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        chunk = retrieved_chunk_preview(result.document, metadata, max_chars=MIN_EVIDENCE_CHARS * 3)
        if not is_meaningful_evidence(chunk):
            continue
        covered_urls.update(result_source_urls_from_metadata(metadata))
    return [url for url in planned_urls if url not in covered_urls]


def question_source_coverage_retrieve(
    question_source_urls: dict[str, list[str]],
    retrieved_context: Sequence[RetrievalResult],
    chroma_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    history_keys: Sequence[str] | None = None,
    top_k_per_url: int = DEFAULT_REPORT_SOURCE_URL_K,
    scan_limit: int = DEFAULT_BM25_SCAN_LIMIT,
) -> list[RetrievalResult]:
    """Retrieve missing planned-source chunks with each planner question as query."""

    selected = []
    running_context = list(retrieved_context)
    seen_ids = {result.id for result in retrieved_context}
    for question, urls in question_source_urls.items():
        missing_urls = uncovered_source_urls_for_question(question, urls, running_context)
        if not missing_urls:
            continue
        results = source_url_coverage_retrieve(
            source_urls=missing_urls,
            query=evidence_focused_question_query(question),
            chroma_path=chroma_path,
            collection_name=collection_name,
            history_keys=history_keys,
            top_k_per_url=max(1, top_k_per_url),
            scan_limit=scan_limit,
        )
        for result in results:
            if result.id in seen_ids:
                continue
            seen_ids.add(result.id)
            selected.append(result)
        running_context = merge_retrieved_context(running_context, results)
    return selected


def uncovered_source_urls_for_question(
    question: str,
    urls: Sequence[str],
    retrieved_context: Sequence[RetrievalResult],
) -> list[str]:
    """Return planned URLs without a question-relevant chunk in current context."""

    missing = []
    for url in dedupe_source_urls(urls):
        if any(result_matches_source_urls(result, [url]) and result_supports_question(question, result) for result in retrieved_context):
            continue
        missing.append(url)
    return missing


def evidence_focused_question_query(question: str) -> str:
    """Add evidence-type hints so planned-source recovery finds the right section."""

    question = clean_text(question)
    hints = []
    for evidence_type in infer_question_evidence_types(question):
        hints.extend(EVIDENCE_QUERY_HINTS.get(evidence_type, []))
    lowered = question.lower()
    if "multiplicative" in lowered:
        hints.extend(["dot product", "bilinear score", "general score"])
    if "additive" in lowered:
        hints.extend(["feedforward score", "tanh"])
    return clean_text(f"{question} {' '.join(dedupe_preserve_order(hints))}")[:1000]


def merge_retrieved_context(*groups: Sequence[RetrievalResult]) -> list[RetrievalResult]:
    merged = []
    seen_ids = set()
    for group in groups:
        for result in group:
            if result.id in seen_ids:
                continue
            seen_ids.add(result.id)
            merged.append(result)
    return merged


def browser_signal_results(browser_results: Sequence[Any], objective: str = "") -> list[RetrievalResult]:
    """Promote high-signal browser excerpts into synthesis context."""

    results = []
    seen = set()
    source_count = 0
    for source in browser_result_sources(browser_results):
        url = normalize_source_url(source.get("url"))
        content = clean_text(source.get("full_content") or source.get("content") or source.get("text"))
        if not url or not content:
            continue
        snippets = high_signal_browser_snippets(content, max_snippets=DEFAULT_BROWSER_SIGNAL_SNIPPETS)
        if not snippets:
            continue
        source_count += 1
        title = clean_text(source.get("title")) or url
        for snippet_index, snippet in enumerate(snippets):
            key = (url, snippet.lower()[:240])
            if key in seen:
                continue
            seen.add(key)
            digest = hashlib.sha256(f"{url}:{snippet_index}:{snippet}".encode("utf-8")).hexdigest()[:24]
            results.append(
                RetrievalResult(
                    id=f"browser-signal-{digest}",
                    document=snippet,
                    metadata={
                        "title": title,
                        "url": url,
                        "source_url": url,
                        "source_type": clean_text(source.get("source_type")) or "browser",
                        "source_quality": "high_signal_browser",
                        "query_contexts": clean_text(source.get("query_context") or objective),
                    },
                    score=1.0,
                    semantic_score=0.0,
                    bm25_score=0.0,
                    authority_score=0.4,
                )
            )
        if source_count >= DEFAULT_BROWSER_SIGNAL_SOURCES:
            break
    return results


def browser_result_sources(browser_results: Sequence[Any]) -> list[dict[str, Any]]:
    sources = []
    for result in browser_results or []:
        if not isinstance(result, dict):
            continue
        for source in result.get("sources", []) or []:
            if isinstance(source, dict):
                sources.append(source)
    return sources


def high_signal_browser_snippets(content: str, max_snippets: int = DEFAULT_BROWSER_SIGNAL_SNIPPETS) -> list[str]:
    candidates = []
    for match in BROWSER_SIGNAL_PATTERN.finditer(content):
        snippet = browser_signal_snippet(content, match.start())
        if is_meaningful_evidence(snippet):
            candidates.append((browser_signal_score(snippet), match.start(), snippet))
        if len(candidates) >= DEFAULT_BROWSER_SIGNAL_CANDIDATES:
            break
    ranked = sorted(candidates, key=lambda item: (-item[0], item[1]))
    return dedupe_preserve_order([snippet for _, _, snippet in ranked])[:max_snippets]


def browser_signal_score(snippet: str) -> int:
    value = clean_text(snippet)
    score = 0
    score += min(5, len(BROWSER_FORMULA_SIGNAL_PATTERN.findall(value))) * 4
    score += min(4, len(BROWSER_METRIC_SIGNAL_PATTERN.findall(value))) * 3
    score += min(3, len(BROWSER_API_SIGNAL_PATTERN.findall(value))) * 2
    if re.search(r"(?i)\b(?:equation|formula|definition|defined as|theorem|algorithm)\b", value):
        score += 3
    if re.search(r"(?i)\b(?:benchmark|score|accuracy|latency|throughput|cost|complexity)\b", value):
        score += 2
    return score


def browser_signal_snippet(content: str, position: int, max_chars: int = 900) -> str:
    start = max(0, position - max_chars // 2)
    end = min(len(content), position + max_chars // 2)
    snippet = clean_text(content[start:end])
    if start > 0:
        sentence_start = re.search(r"(?<=[.!?])\s+[A-Z0-9`(]", snippet)
        if sentence_start:
            snippet = snippet[sentence_start.end() - 1 :]
    if end < len(content):
        sentence_end = list(re.finditer(r"[.!?](?=\s+[A-Z0-9`(]|$)", snippet))
        if sentence_end:
            snippet = snippet[: sentence_end[-1].end()]
    return clean_text(snippet[:max_chars])


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

    emit_progress(
        "tool_called",
        "RAG generation calling Groq",
        agent="synthesis",
        tool="groq",
        metadata={"model": rag_generation_model(model)},
    )
    response = create_chat_completion_with_retries(
        Groq(),
        model=rag_generation_model(model),
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
    planner_questions: Sequence[str] | None = None,
    question_source_urls: dict[str, list[str]] | None = None,
    model: str | None = None,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_tokens: int = DEFAULT_REPORT_MAX_TOKENS,
) -> dict[str, Any]:
    """Synthesize retrieved chunks into a detailed evidence package for a report agent."""
    objective = clean_text(objective)
    if not objective:
        raise ValueError("objective is required")
    if not retrieved_context:
        return {
            "objective": objective,
            "synthesis": "No retrieved context was available for report synthesis.",
            "sources": [],
            "evidence_packs": [],
        }
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set")

    try:
        from groq import Groq
    except ImportError as error:
        raise RuntimeError("groq package is not installed. Install it with `pip install -r requirements.txt`.") from error

    planner_question_text = format_planner_questions(planner_questions or [])
    instruction = clean_text(synthesis_instruction) or "Synthesize the retrieved evidence into report-ready research notes."
    instruction_requirement_text = format_instruction_requirements(instruction)
    client = Groq()
    last_error: Exception | None = None

    for attempt in range(3):
        context_budget = max(8000, int(max_context_chars * (0.70 ** attempt)))
        token_budget = max(700, int(max_tokens * (0.80 ** attempt)))
        context_text, sources = build_generation_context(retrieved_context, max_context_chars=context_budget)
        evidence_packs = build_sub_question_evidence_packs(
            planner_questions or [],
            retrieved_context,
            sources,
            question_source_urls=question_source_urls,
        )
        source_priority_guidance = build_source_priority_guidance(sources)
        prompt = f"""Research objective:
{objective}

Synthesis instruction:
{instruction}

Instruction requirements to satisfy:
{instruction_requirement_text}

Planner sub-questions to cover:
{planner_question_text}

Source priority guidance:
{source_priority_guidance}

Per-question evidence packs:
{format_evidence_packs_for_prompt(evidence_packs)}

Retrieved context from multiple sources:
{context_text}

Create a detailed report-agent-ready evidence package using only the retrieved context.
Do not write the final report. Prepare rich notes that another agent can turn into a technical report.

Return Markdown with these sections:
1. Instruction Coverage Checklist
   - For each instruction requirement, mark Covered, Partial, or Missing Evidence.
   - Cite source markers for Covered/Partial items and name the exact missing facts for Missing Evidence items.
2. Coverage Map
   - For each planner sub-question, state whether the retrieved context has strong, partial, or missing evidence.
3. Section Notes By Planner Question
   - Repeat each planner sub-question as a subsection.
   - Include the direct answer, important evidence, equations/formulas/API details when available, and gaps.
   - Keep enough detail for a report agent to write a full section without needing to infer missing facts.
4. Cross-Source Synthesis
   - Connect repeated ideas across sources and identify how the sources complement each other.
5. Technical Details To Preserve
   - Preserve exact equations, definitions, model components, implementation details, and benchmark values only when present in the retrieved context.
6. Conflicts Or Gaps
   - List missing evidence, weak citations, source conflicts, or claims that need caution.
7. Recommended Report Structure
   - Suggest report sections and which source markers support each section.
   - Include every Covered/Partial instruction requirement and explicit gap notes for Missing Evidence requirements.

Use only plain ASCII numbered source markers that appear in the retrieved context, exactly like [1], [2], [3].
Every evidence-backed claim must include at least one source marker.
For equations, formulas, API signatures, benchmark numbers, and historical attribution, cite original papers, official documentation, academic sources, or authoritative surveys first.
If a primary/official source and a secondary explainer both support the same technical claim, cite the primary/official source and omit the secondary citation.
Do not mark a requirement or planner question as Missing Evidence when a primary/official source in the retrieved context contains evidence for it; cite that source and mark it Covered or Partial instead.
Use secondary explainers only for intuition, examples, or background wording.
Do not compress important technical details into vague summaries.
Do not use Markdown tables.
Never use citation formats like 【1】, 【1†L1-L4】, footnotes, or URLs inline.
If a requested equation, number, API detail, or definition is not explicitly present in the retrieved context, mark it as missing evidence and tell the report agent not to add it.
Before finishing, check the Instruction Coverage Checklist against the Recommended Report Structure so requested items are not silently dropped.
Do not invent source names, authors, dates, titles, papers, benchmark numbers, equations, or citations that are not present in the retrieved context."""

        try:
            emit_progress(
                "tool_called",
                "Synthesis calling Groq to create report context",
                agent="synthesis",
                tool="groq",
                metadata={"model": rag_generation_model(model), "attempt": attempt + 1},
            )
            response = create_chat_completion_with_retries(
                client,
                model=rag_generation_model(model),
                temperature=0,
                max_tokens=max(300, token_budget),
                messages=[
                    {
                        "role": "system",
                        "content": "You synthesize retrieved RAG evidence for a downstream report agent. Do not use outside knowledge.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as error:
            last_error = error
            if is_request_too_large_error(error) and attempt < 2:
                continue
            raise

        synthesis = normalize_citation_markers(response.choices[0].message.content)
        return {
            "objective": objective,
            "synthesis_instruction": instruction,
            "planner_questions": list(planner_questions or []),
            "synthesis": synthesis,
            "sources": sources,
            "evidence_packs": evidence_packs,
            "model": response.model,
            "synthesis_attempts": attempt + 1,
            "synthesis_context_chars": context_budget,
            "synthesis_max_tokens": max(300, token_budget),
        }

    if last_error:
        raise last_error
    raise RuntimeError("synthesis failed before calling the generation model")


def format_planner_questions(questions: Sequence[str]) -> str:
    clean_questions = dedupe_preserve_order(questions)
    if not clean_questions:
        return "No planner sub-questions were provided. Cover the research objective directly."
    return "\n".join(f"- {question}" for question in clean_questions)


def format_instruction_requirements(instruction: str) -> str:
    """Return a compact checklist extracted from a free-form synthesis instruction."""

    requirements = instruction_requirement_items(instruction)
    if not requirements:
        return "- No explicit synthesis requirements were provided."
    return "\n".join(f"- {requirement}" for requirement in requirements)


def instruction_requirement_items(instruction: str) -> list[str]:
    text = clean_text(instruction)
    if not text:
        return []

    markers = list(re.finditer(r"(?:^|\s)(?:\(\d+\)|\d+[.)])\s+", text))
    if len(markers) >= 2:
        items = []
        for index, marker in enumerate(markers):
            start = marker.end()
            end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            item = clean_text(text[start:end].strip(" ;,."))
            item = re.sub(r"(?:,\s*)?\band$", "", item).strip(" ;,.")
            if item:
                items.append(item)
        return dedupe_preserve_order(items)

    bullet_items = [
        clean_text(line.lstrip("-* ").strip())
        for line in str(instruction or "").splitlines()
        if line.strip().startswith(("-", "*"))
    ]
    return dedupe_preserve_order(item for item in bullet_items if item) or [text]


def synthesis_gap_retrieval_queries(
    synthesis: Any,
    objective: str = "",
    synthesis_instruction: str = "",
    sources: Sequence[dict[str, Any]] | None = None,
    model: str | None = None,
    max_queries: int = DEFAULT_GAP_RETRIEVAL_MAX_QUERIES,
) -> list[str]:
    """Build LLM-generated retrieval queries from synthesis coverage gaps."""

    return synthesis_gap_retrieval_plan(
        synthesis,
        objective=objective,
        synthesis_instruction=synthesis_instruction,
        sources=sources,
        model=model,
        max_queries=max_queries,
    )["queries"]


def synthesis_gap_retrieval_plan(
    synthesis: Any,
    objective: str = "",
    synthesis_instruction: str = "",
    sources: Sequence[dict[str, Any]] | None = None,
    model: str | None = None,
    max_queries: int = DEFAULT_GAP_RETRIEVAL_MAX_QUERIES,
) -> dict[str, Any]:
    """Build gap-retrieval queries plus diagnostics about their origin."""

    gaps = synthesis_gap_items(synthesis)
    if not gaps:
        return gap_retrieval_plan(
            queries=[],
            llm_queries=[],
            fallback_queries=[],
        )
    fallback_queries = fallback_gap_retrieval_queries(
        gaps,
        objective=objective,
        max_queries=max_queries,
    )
    llm_result = llm_gap_retrieval_query_result(
        gaps,
        objective=objective,
        synthesis_instruction=synthesis_instruction,
        sources=sources or [],
        model=model,
        max_queries=max_queries,
    )
    llm_queries = llm_result["queries"]
    queries = dedupe_preserve_order([*llm_queries, *fallback_queries])[: max(1, max_queries)]
    return gap_retrieval_plan(
        queries=queries,
        llm_queries=llm_queries,
        fallback_queries=fallback_queries,
        llm_error=llm_result["error"],
        llm_raw_response=llm_result["raw_response"],
        model=llm_result.get("model", ""),
    )


def gap_retrieval_plan(
    queries: Sequence[str],
    llm_queries: Sequence[str],
    fallback_queries: Sequence[str],
    llm_error: str = "",
    llm_raw_response: str = "",
    model: str = "",
) -> dict[str, Any]:
    llm_count = len(llm_queries)
    fallback_count = len(fallback_queries)
    if llm_count and fallback_count:
        source = "mixed"
    elif llm_count:
        source = "llm"
    elif fallback_count:
        source = "fallback"
    else:
        source = "none"
    return {
        "queries": list(queries),
        "llm_queries": list(llm_queries),
        "fallback_queries": list(fallback_queries),
        "model": clean_text(model) or DEFAULT_GAP_QUERY_MODEL,
        "source": source,
        "llm_error": clean_text(llm_error),
        "llm_raw_response": clean_text(llm_raw_response)[:1000],
    }


def fallback_gap_retrieval_queries(
    gaps: Sequence[str],
    objective: str = "",
    max_queries: int = DEFAULT_GAP_RETRIEVAL_MAX_QUERIES,
) -> list[str]:
    objective = clean_text(objective)
    queries = []
    for gap in gaps:
        gap_text = re.sub(r"(?i)\b(?:missing|partial|weak)\s+evidence\s*:?", " ", clean_text(gap))
        key_terms = " ".join(query_keywords(gap_text, limit=12))
        hints = " ".join(broad_query_hints(gap_text))
        query = clean_text(f"{objective} {key_terms} {hints} source evidence")
        if query:
            queries.append(query[:600])
    return dedupe_preserve_order(queries)[: max(1, max_queries)]


def llm_gap_retrieval_queries(
    gaps: Sequence[str],
    objective: str = "",
    synthesis_instruction: str = "",
    sources: Sequence[dict[str, Any]] | None = None,
    model: str | None = None,
    max_queries: int = DEFAULT_GAP_RETRIEVAL_MAX_QUERIES,
) -> list[str]:
    """Ask the generation LLM for precise RAG queries for missing evidence."""

    return llm_gap_retrieval_query_result(
        gaps,
        objective=objective,
        synthesis_instruction=synthesis_instruction,
        sources=sources,
        model=model,
        max_queries=max_queries,
    )["queries"]


def llm_gap_retrieval_query_result(
    gaps: Sequence[str],
    objective: str = "",
    synthesis_instruction: str = "",
    sources: Sequence[dict[str, Any]] | None = None,
    model: str | None = None,
    max_queries: int = DEFAULT_GAP_RETRIEVAL_MAX_QUERIES,
) -> dict[str, Any]:
    """Ask the gap-query model for queries and preserve failure diagnostics."""

    clean_gaps = dedupe_preserve_order(gaps)
    if not clean_gaps:
        return llm_gap_query_result([], model=gap_query_model(model), error="no_gap_items")
    if not os.environ.get("GROQ_API_KEY"):
        return llm_gap_query_result([], model=gap_query_model(model), error="missing_groq_api_key")

    try:
        from groq import Groq
    except ImportError as error:
        return llm_gap_query_result([], model=gap_query_model(model), error=f"groq_import_error: {clean_text(error)}")

    selected_model = gap_query_model(model)
    prompt_gaps = clean_gaps[: max(1, max_queries)]
    gap_text = format_gap_items_for_query_prompt(prompt_gaps)
    source_hints = format_gap_query_source_hints(sources or [])
    prompt = f"""Research objective:
{clean_text(objective)}

Synthesis instruction:
{clean_text(synthesis_instruction)}

Missing or partial evidence items:
{gap_text}

Available source hints:
{source_hints}

Generate broad RAG retrieval queries to find relevant evidence in indexed chunks.
Requirements:
- Return one query per missing item, using the same item id format: G1: query text.
- Generate at most {len(prompt_gaps)} queries.
- Keep each query centered on the missing item, but broad enough to retrieve surrounding source context.
- Prefer compact noun phrases over full missing-evidence sentences.
- Include literal technical terms, aliases, API names, benchmark names, model names, author names, and source titles that match the item.
- Preserve the target entity from the missing item; do not move benchmarks, formulas, metrics, or limitations from one model/source to another.
- For missing equations, include the named equation/architecture and broad terms like formula, derivation, components, or notation.
- For missing API details, include official class/function names and parameters.
- Do not answer the research question.
- Do not add bullets, markdown tables, quotes, or explanation."""

    raw_response = ""
    try:
        response = create_chat_completion_with_retries(
            Groq(),
            model=selected_model,
            temperature=0,
            max_tokens=500,
            messages=[
                {
                    "role": "system",
                    "content": "You generate concise high-recall RAG retrieval queries for missing evidence. Return only queries.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        raw_response = clean_text(response.choices[0].message.content)
    except Exception as error:
        return llm_gap_query_result([], model=selected_model, error=f"groq_api_error: {type(error).__name__}: {clean_text(error)}")

    if not raw_response:
        return llm_gap_query_result([], model=selected_model, error="empty_response")
    queries = parse_gap_query_lines(raw_response, max_queries=max_queries)
    if not queries:
        return llm_gap_query_result([], model=selected_model, error="parsed_empty_response", raw_response=raw_response)
    return llm_gap_query_result(queries, model=selected_model, raw_response=raw_response)


def llm_gap_query_result(
    queries: Sequence[str],
    model: str = "",
    error: str = "",
    raw_response: str = "",
) -> dict[str, Any]:
    return {
        "queries": list(queries),
        "model": clean_text(model),
        "error": clean_text(error),
        "raw_response": clean_text(raw_response)[:1000],
    }


def parse_gap_query_lines(text: Any, max_queries: int = DEFAULT_GAP_RETRIEVAL_MAX_QUERIES) -> list[str]:
    queries = []
    for line in split_gap_query_response(strip_thinking_blocks(text)):
        query = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        query = re.sub(r"^G\d+\s*:\s*", "", query, flags=re.IGNORECASE).strip()
        query = clean_text(query.strip("\"'`"))
        if is_valid_generated_query(query, max_chars=600):
            queries.append(query[:600])
    return dedupe_preserve_order(queries)[: max(1, max_queries)]


def split_gap_query_response(text: Any) -> list[str]:
    response = str(text or "").strip()
    if not response:
        return []
    inline_items = [
        match.group(0).strip()
        for match in re.finditer(r"G\d+\s*:\s*.*?(?=\s+G\d+\s*:|$)", response, flags=re.IGNORECASE | re.DOTALL)
    ]
    if len(inline_items) > 1:
        return inline_items
    return response.splitlines()


def format_gap_items_for_query_prompt(gaps: Sequence[str]) -> str:
    lines = []
    for index, gap in enumerate(gaps, start=1):
        lines.append(f"G{index}: {clean_text(gap)[:500]}")
    return "\n".join(lines)


def format_gap_query_source_hints(sources: Sequence[dict[str, Any]]) -> str:
    lines = []
    for source in sources[:12]:
        if not isinstance(source, dict):
            continue
        title = clean_text(source.get("title"))
        url = clean_text(source.get("url"))
        index = source.get("index")
        if title or url:
            marker = f"[{index}] " if isinstance(index, int) else ""
            lines.append(f"{marker}{title} {url}".strip())
    return "\n".join(lines) or "No source hints available."


def synthesis_gap_items(synthesis: Any) -> list[str]:
    text = normalize_citation_markers(synthesis)
    gaps = []
    for line in text.splitlines():
        line_text = clean_text(line)
        if not line_text:
            continue
        table_gap = synthesis_gap_from_table_row(line_text)
        if table_gap:
            gaps.append(table_gap)
            continue
        lowered = line_text.lower()
        if any(
            phrase in lowered
            for phrase in (
                "missing evidence",
                "not present in the retrieved",
                "not present in the cited",
                "not quoted in the retrieved",
                "no explicit",
                "no concrete",
            )
        ):
            gaps.append(strip_markdown_markup(line_text)[:300])
    return dedupe_preserve_order(gaps)


def synthesis_gap_from_table_row(line: str) -> str:
    if not line.startswith("|") or "---" in line:
        return ""
    cells = [strip_markdown_markup(cell) for cell in line.strip("|").split("|")]
    if len(cells) < 2:
        return ""
    first_cell = clean_text(cells[0]).lower()
    if first_cell in {"requirement", "planner sub-question", "source", "status"}:
        return ""
    status = clean_text(cells[1]).lower()
    notes = clean_text(" ".join(cells[2:]))
    if "missing" not in status and "partial" not in status and "missing evidence" not in notes.lower():
        return ""
    return clean_text(f"{cells[0]} {notes}")


def strip_markdown_markup(text: str) -> str:
    value = re.sub(r"`([^`]*)`", r"\1", str(text or ""))
    value = re.sub(r"[*_#]+", "", value)
    value = re.sub(r"\[(\d+)\]", "", value)
    return clean_text(value)


def is_request_too_large_error(error: Exception) -> bool:
    return is_groq_request_too_large_error(error)


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
    source_by_key: dict[str, dict[str, Any]] = {}
    used_chars = 0
    max_context_chars = max(1000, max_context_chars)
    ordered_results = source_balanced_results(retrieved_context)
    block_char_limit = context_block_char_limit(ordered_results, max_context_chars)

    for retrieval_rank, result in enumerate(ordered_results, start=1):
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        url = primary_source_url(metadata)
        title = clean_text(metadata.get("title")) or url or f"Source {len(sources) + 1}"
        chunk = retrieved_chunk_preview(result.document, metadata, max_chars=block_char_limit)
        if not is_meaningful_evidence(chunk):
            continue

        source_key = source_url_key(url) or clean_text(result.id)
        source = source_by_key.get(source_key)
        if source is None:
            citation_index = len(sources) + 1
            source = {
                "index": citation_index,
                "retrieval_rank": retrieval_rank,
                "id": result.id,
                "ids": [result.id],
                "url": url,
                "title": title,
                "score": result.score,
            }
            source_by_key[source_key] = source
            sources.append(source)
        else:
            citation_index = source["index"]
            if result.id not in source["ids"]:
                source["ids"].append(result.id)
            source["score"] = max(float(source.get("score") or 0.0), float(result.score or 0.0))

        block = f"[{citation_index}] {title}\nURL: {url}\n{chunk}"
        if used_chars + len(block) > max_context_chars:
            remaining = max_context_chars - used_chars
            if remaining < 500:
                break
            block = block[:remaining].rstrip()

        blocks.append(block)
        used_chars += len(block)

    return "\n\n".join(blocks), sources


def source_url_key(url: Any) -> str:
    return normalize_source_url(url) or clean_text(url).lower()


def select_synthesis_context(
    retrieved_context: Sequence[RetrievalResult],
    planner_questions: Sequence[str] | None = None,
    question_source_urls: dict[str, list[str]] | None = None,
    max_chunks: int = DEFAULT_SYNTHESIS_MAX_CHUNKS,
    per_question: int = DEFAULT_SYNTHESIS_CHUNKS_PER_QUESTION,
) -> list[RetrievalResult]:
    """Return a compact, topic-balanced context for the synthesis LLM."""

    candidates = meaningful_retrieval_results(retrieved_context)
    questions = [clean_text(question) for question in planner_questions or [] if clean_text(question)]
    if not questions:
        return source_balanced_results(candidates)[:max(1, max_chunks)]

    selected = []
    seen = set()
    selected_by_question: dict[str, int] = {}

    def add_for_question(question: str, result: RetrievalResult) -> bool:
        key = retrieval_result_key(result)
        if key in seen or len(selected) >= max_chunks:
            return False
        selected.append(result)
        seen.add(key)
        selected_by_question[question_key(question)] = selected_by_question.get(question_key(question), 0) + 1
        return True

    for question in questions:
        for result in question_ranked_results(question, candidates, question_source_urls)[:1]:
            add_for_question(question, result)
            break

    for question in questions:
        for result in question_ranked_results(question, candidates, question_source_urls):
            if selected_by_question.get(question_key(question), 0) >= max(1, per_question):
                break
            add_for_question(question, result)
            if len(selected) >= max_chunks:
                return selected

    for result in source_balanced_results(candidates):
        if len(selected) >= max_chunks:
            break
        key = retrieval_result_key(result)
        if key in seen:
            continue
        selected.append(result)
        seen.add(key)
        if len(selected) >= max_chunks:
            break
    return selected


def question_ranked_results(
    question: str,
    candidates: Sequence[RetrievalResult],
    question_source_urls: dict[str, list[str]] | None = None,
) -> list[RetrievalResult]:
    preferred = [
        result
        for result in source_matched_results(candidates, question_source_urls_for(question, question_source_urls))
        if result_supports_question(question, result)
    ]
    ranked = rank_results_for_question(question, preferred) + rank_results_for_question(question, candidates)
    return unique_retrieval_results(ranked)


def source_matched_results(results: Sequence[RetrievalResult], urls: Sequence[str]) -> list[RetrievalResult]:
    if not urls:
        return []
    return [result for result in results if result_matches_source_urls(result, urls)]


def result_matches_source_urls(result: RetrievalResult, urls: Sequence[str]) -> bool:
    target_urls = set(dedupe_source_urls(urls))
    if not target_urls:
        return False
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    return bool(target_urls & set(dedupe_source_urls(result_source_urls_from_metadata(metadata))))


def question_source_urls_for(question: str, question_source_urls: dict[str, list[str]] | None) -> list[str]:
    if not question_source_urls:
        return []
    return question_source_urls.get(question) or question_source_urls.get(question_key(question), [])


def unique_retrieval_results(results: Sequence[RetrievalResult]) -> list[RetrievalResult]:
    unique = []
    seen = set()
    for result in results:
        key = retrieval_result_key(result)
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


def meaningful_retrieval_results(retrieved_context: Sequence[RetrievalResult]) -> list[RetrievalResult]:
    results = []
    seen = set()
    for result in retrieved_context:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        chunk = retrieved_chunk_preview(result.document, metadata, max_chars=MIN_EVIDENCE_CHARS * 4)
        if not is_meaningful_evidence(chunk):
            continue
        key = retrieval_result_key(result, preview=chunk)
        if key in seen:
            continue
        seen.add(key)
        results.append(result)
    return results


def rank_results_for_question(question: str, candidates: Sequence[RetrievalResult]) -> list[RetrievalResult]:
    question_terms = query_tokens(question)
    evidence_types = infer_question_evidence_types(question)
    ranked = []
    for position, result in enumerate(candidates):
        text = retrieval_result_text(result)
        terms = query_tokens(text)
        overlap = len(question_terms & terms)
        evidence_score = evidence_type_score(text, evidence_types)
        if overlap or evidence_score:
            score = overlap * 3 + evidence_score + retrieval_result_priority(result)
            ranked.append((score, -position, result))
    return [result for _, _, result in sorted(ranked, reverse=True)]


def result_supports_question(question: str, result: RetrievalResult) -> bool:
    return text_supports_question(question, retrieval_result_text(result))


def text_supports_question(question: str, text: str, required_evidence: Sequence[str] | None = None) -> bool:
    text = clean_text(text)
    question_terms = coverage_question_tokens(question)
    chunk_terms = coverage_question_tokens(text)
    topic_terms = {term for term in question_terms if term not in COVERAGE_EVIDENCE_TERMS}
    if topic_terms and not (topic_terms & chunk_terms):
        return False
    evidence_types = clean_string_list(list(required_evidence or [])) or infer_question_evidence_types(question)
    strict_evidence = [item for item in evidence_types if item != "evidence"]
    evidence_score = evidence_type_score(text, strict_evidence)
    if strict_evidence and not evidence_score:
        return False
    return bool((question_terms & chunk_terms) or evidence_score)


def retrieval_result_text(result: RetrievalResult) -> str:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    return clean_text(
        " ".join(
            [
                clean_text(metadata.get("title")),
                clean_text(primary_source_url(metadata)),
                clean_text(metadata.get("query_contexts")),
                retrieved_chunk_preview(result.document, metadata, max_chars=DEFAULT_CONTEXT_BLOCK_CHARS)
                or clean_text(result.document)[:DEFAULT_CONTEXT_BLOCK_CHARS],
            ]
        )
    )


def retrieval_result_priority(result: RetrievalResult) -> int:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    priority = 2 if is_primary_source(metadata) else 0
    if result.score is not None and result.score > 0:
        priority += 1
    return priority


def retrieval_result_key(result: RetrievalResult, preview: str = "") -> str:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    stable_id = clean_text(metadata.get("parent_id")) or clean_text(result.id)
    url = primary_source_url(metadata)
    content = clean_text(preview or result.document)[:240].lower()
    return clean_text(f"{stable_id}:{url}:{content}") or clean_text(result.id)


def build_source_priority_guidance(sources: Sequence[dict[str, Any]]) -> str:
    """Tell the synthesis LLM which retrieved sources are best for technical claims."""

    primary = []
    secondary = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        index = source.get("index")
        url = clean_text(source.get("url"))
        title = clean_text(source.get("title")) or url
        if not isinstance(index, int) or not url:
            continue
        item = f"[{index}] {title} ({url})"
        if primary_source_url_like(url):
            primary.append(item)
        else:
            secondary.append(item)

    lines = [
        "Prefer primary/official sources for technical claims, formulas, API details, and benchmark evidence.",
    ]
    if primary:
        lines.append("Primary/official candidates:")
        lines.extend(f"- {item}" for item in primary[:12])
    if secondary:
        lines.append("Secondary/background candidates:")
        lines.extend(f"- {item}" for item in secondary[:8])
    return "\n".join(lines)


def primary_source_url_like(url: str) -> bool:
    normalized_url = clean_text(url).lower()
    return (
        "arxiv.org/pdf/" in normalized_url
        or "arxiv.org/abs/" in normalized_url
        or "docs." in normalized_url
        or "pytorch.org" in normalized_url
        or "tensorflow.org" in normalized_url
        or "openreview.net/pdf" in normalized_url
        or "doi.org" in normalized_url
        or ".edu" in normalized_url
    )


def normalize_citation_markers(text: str) -> str:
    """Convert model-specific citation glyphs to plain [n] markers."""

    normalized = str(text or "").strip()
    normalized = re.sub(r"【\s*(\d+)(?:[^】]*)?】", r"[\1]", normalized)
    normalized = re.sub(r"\[\s*(\d+)\s*\]", r"[\1]", normalized)
    normalized = re.sub(r"[ \t]+$", "", normalized, flags=re.MULTILINE)
    normalized = re.sub(r"\n{4,}", "\n\n\n", normalized)
    return normalized


def compact_retrieved_chunks(
    retrieved_context: Sequence[RetrievalResult],
    sources: Sequence[dict[str, Any]] | None = None,
    max_chars: int = DEFAULT_CONTEXT_BLOCK_CHARS,
) -> list[dict[str, Any]]:
    """Serialize selected retrieved chunks for a downstream report agent."""

    source_index_by_id = citation_index_by_chunk_id(sources or [])
    source_index_by_url = citation_index_by_source_url(sources or [])
    chunks = []
    for rank, result in enumerate(retrieved_context, start=1):
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        url = primary_source_url(metadata)
        title = clean_text(metadata.get("title")) or url or f"Source {rank}"
        content = retrieved_chunk_preview(result.document, metadata, max_chars=max_chars)
        if not is_meaningful_evidence(content):
            continue
        chunks.append(
            {
                "source_index": source_index_by_id.get(result.id) or source_index_by_url.get(source_url_key(url)),
                "retrieval_rank": rank,
                "id": result.id,
                "url": url,
                "title": title,
                "score": result.score,
                "source_type": clean_text(metadata.get("source_type")),
                "source_quality": clean_text(metadata.get("source_quality")),
                "is_primary_source": is_primary_source(metadata),
                "content": content,
            }
        )
    return chunks


def build_sub_question_evidence_packs(
    planner_questions: Sequence[str],
    retrieved_context: Sequence[RetrievalResult],
    sources: Sequence[dict[str, Any]],
    question_source_urls: dict[str, list[str]] | None = None,
    max_chunks_per_question: int = 4,
) -> list[dict[str, Any]]:
    """Group source-numbered chunks under each planner sub-question."""

    chunks = [
        chunk
        for chunk in compact_retrieved_chunks(retrieved_context, sources=sources, max_chars=500)
        if chunk.get("source_index") is not None
    ]
    packs = []
    for question in dedupe_preserve_order(planner_questions):
        question_terms = query_tokens(question)
        scored = []
        for chunk in chunks:
            text = " ".join([clean_text(chunk.get("title")), clean_text(chunk.get("content"))])
            overlap = len(question_terms & query_tokens(text))
            preferred = chunk_matches_source_urls(chunk, question_source_urls_for(question, question_source_urls))
            supported = text_supports_question(question, text)
            if overlap or preferred or supported:
                scored.append(
                    (
                        1 if supported else 0,
                        1 if preferred else 0,
                        1 if chunk.get("is_primary_source") else 0,
                        overlap,
                        float(chunk.get("score") or 0.0),
                        chunk,
                    )
                )
        selected = [
            chunk
            for *_score, chunk in sorted(scored, key=lambda item: (-item[0], -item[1], -item[2], -item[3], -item[4]))
            [:max_chunks_per_question]
        ]
        supported_count = sum(
            1
            for chunk in selected
            if text_supports_question(question, " ".join([clean_text(chunk.get("title")), clean_text(chunk.get("content"))]))
        )
        packs.append(
            {
                "question": question,
                "coverage": "covered" if supported_count else ("partial" if selected else "missing"),
                "planned_source_urls": question_source_urls_for(question, question_source_urls),
                "chunks": selected,
            }
        )
    return packs


def chunk_matches_source_urls(chunk: dict[str, Any], urls: Sequence[str]) -> bool:
    target_urls = set(dedupe_source_urls(urls))
    if not target_urls:
        return False
    chunk_urls = dedupe_source_urls([clean_text(chunk.get("url"))])
    return bool(target_urls & set(chunk_urls))


def format_evidence_packs_for_prompt(evidence_packs: Sequence[dict[str, Any]]) -> str:
    if not evidence_packs:
        return "- No per-question evidence packs were built."
    lines = []
    for pack in evidence_packs:
        question = clean_text(pack.get("question")) if isinstance(pack, dict) else ""
        coverage = clean_text(pack.get("coverage")) if isinstance(pack, dict) else "missing"
        lines.append(f"- {question} ({coverage})")
        for chunk in (pack.get("chunks", []) if isinstance(pack, dict) else [])[:4]:
            marker = f"[{chunk.get('source_index')}]"
            title = clean_text(chunk.get("title")) or clean_text(chunk.get("url")) or "Retrieved chunk"
            content = clean_text(chunk.get("content"))[:260]
            lines.append(f"  - {marker} {title}: {content}")
    return "\n".join(lines)


def report_supporting_chunks(
    retrieved_context: Sequence[RetrievalResult],
    sources: Sequence[dict[str, Any]],
    max_chunks: int = DEFAULT_REPORT_SUPPORTING_CHUNKS,
    max_chars: int = DEFAULT_CONTEXT_BLOCK_CHARS,
    cited_source_indexes: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Return citation-linked chunks that are safest for the report agent."""

    compact_chunks = compact_retrieved_chunks(
        retrieved_context,
        sources=sources,
        max_chars=max_chars,
    )
    citation_backed = [chunk for chunk in compact_chunks if chunk.get("source_index") is not None]
    cited_indexes = {index for index in cited_source_indexes or [] if isinstance(index, int)}
    ordered = sorted(
        citation_backed,
        key=lambda chunk: (
            0 if chunk.get("source_index") in cited_indexes else 1,
            0 if chunk.get("is_primary_source") else 1,
            chunk.get("source_index") or 10**6,
            -(float(chunk.get("score") or 0.0)),
        ),
    )
    target_count = max(1, max_chunks, len(cited_indexes))
    return ordered[:target_count]


def audit_synthesis_citations(
    synthesis: Any,
    sources: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize whether synthesis notes cite only available source markers."""

    referenced_indexes = citation_markers(synthesis)
    valid_indexes = source_indexes(sources)
    valid_referenced = [index for index in referenced_indexes if index in valid_indexes]
    invalid_referenced = [index for index in referenced_indexes if index not in valid_indexes]
    uncited_sources = [index for index in sorted(valid_indexes) if index not in referenced_indexes]
    return {
        "referenced_source_indexes": referenced_indexes,
        "valid_referenced_source_indexes": valid_referenced,
        "invalid_source_indexes": invalid_referenced,
        "uncited_source_indexes": uncited_sources,
        "source_count": len(valid_indexes),
        "has_invalid_citations": bool(invalid_referenced),
    }


def citation_markers(text: Any) -> list[int]:
    """Return unique plain numeric Markdown citation markers in first-seen order."""

    markers = []
    seen = set()
    for match in re.finditer(r"\[(\d+)\]", normalize_citation_markers(text)):
        index = int(match.group(1))
        if index in seen:
            continue
        seen.add(index)
        markers.append(index)
    return markers


def source_indexes(sources: Sequence[dict[str, Any]]) -> set[int]:
    indexes = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        index = source.get("index")
        if isinstance(index, int):
            indexes.add(index)
    return indexes


def citation_index_by_chunk_id(sources: Sequence[dict[str, Any]]) -> dict[str, int]:
    indexes = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        index = source.get("index")
        if not isinstance(index, int):
            continue
        raw_ids = source.get("ids") if isinstance(source.get("ids"), list) else []
        ids = [source.get("id"), *raw_ids]
        for chunk_id in ids:
            clean_id = clean_text(chunk_id)
            if clean_id:
                indexes[clean_id] = index
    return indexes


def citation_index_by_source_url(sources: Sequence[dict[str, Any]]) -> dict[str, int]:
    indexes = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        index = source.get("index")
        key = source_url_key(source.get("url"))
        if key and isinstance(index, int):
            indexes[key] = index
    return indexes


def synthesis_diagnostics(payload: dict[str, Any], retrieved_context: Sequence[RetrievalResult]) -> dict[str, Any]:
    sources = payload.get("sources", [])
    supporting_chunks = payload.get("supporting_chunks", [])
    allowed_history_keys = payload.get("allowed_history_keys", [])
    retrieval_history_keys = payload.get("retrieval_history_keys", [])
    objective_scope = payload.get("objective_scope", {})
    primary_source_count = sum(
        1 for result in retrieved_context
        if is_primary_source(result.metadata if isinstance(result.metadata, dict) else {})
    )
    evidence_chunk_count = count_meaningful_evidence_chunks(retrieved_context)
    return {
        "retrieved_count": len(retrieved_context),
        "evidence_chunk_count": evidence_chunk_count,
        "filtered_weak_chunk_count": max(0, len(retrieved_context) - evidence_chunk_count),
        "source_count": len(sources) if isinstance(sources, list) else 0,
        "supporting_chunk_count": len(supporting_chunks) if isinstance(supporting_chunks, list) else 0,
        "primary_source_count": primary_source_count,
        "invalid_citation_count": len(payload.get("citation_audit", {}).get("invalid_source_indexes", [])),
        "cited_source_count": len(payload.get("citation_audit", {}).get("valid_referenced_source_indexes", [])),
        "source_coverage_count": payload.get("source_coverage_count", 0),
        "question_source_coverage_count": payload.get("question_source_coverage_count", 0),
        "browser_signal_count": payload.get("browser_signal_count", 0),
        "gap_retrieval_query_count": len(payload.get("gap_retrieval_queries", [])),
        "gap_query_model": payload.get("gap_query_model", ""),
        "gap_query_source": payload.get("gap_query_source", ""),
        "llm_gap_query_count": payload.get("llm_gap_query_count", 0),
        "fallback_gap_query_count": payload.get("fallback_gap_query_count", 0),
        "gap_query_error": payload.get("gap_query_error", ""),
        "llm_gap_query_raw_response": payload.get("llm_gap_query_raw_response", ""),
        "gap_retrieved_count": payload.get("gap_retrieved_count", 0),
        "gap_new_chunk_count": payload.get("gap_new_chunk_count", 0),
        "gap_retry_count": payload.get("gap_retry_count", 0),
        "retrieval_scope": payload.get("retrieval_scope", ""),
        "retrieval_history_key_count": len(retrieval_history_keys) if isinstance(retrieval_history_keys, list) else 0,
        "allowed_history_key_count": len(allowed_history_keys) if isinstance(allowed_history_keys, list) else 0,
        "similar_previous_objective_count": max(
            0,
            len(objective_scope.get("selected_objectives", [])) - 1
            if isinstance(objective_scope, dict)
            else 0,
        ),
    }


def is_primary_source(metadata: dict[str, Any]) -> bool:
    source_type = clean_text(metadata.get("source_type")).lower()
    url = primary_source_url(metadata).lower()
    return (
        source_type in {"arxiv", "academic", "docs", "benchmarks", "pricing"}
        or "arxiv.org/pdf/" in url
        or "arxiv.org/abs/" in url
        or "openreview.net/pdf" in url
        or "tensorflow.org" in url
        or "pytorch.org" in url
        or ".edu" in url
        or "doi.org" in url
    )


def retrieved_chunk_preview(document: str, metadata: dict[str, Any], max_chars: int) -> str:
    """Return chunk body text without dropping content after stored headers."""

    body = strip_stored_chunk_headers(document, metadata)
    if not body and stored_chunk_header_present(document):
        return ""
    if not body:
        body = display_document_preview(document, max_chars=max_chars)
    if not body:
        body = clean_text(document)
    preview = body[: max(80, max_chars)].strip()
    return preview if is_meaningful_evidence(preview) else ""


def strip_stored_chunk_headers(document: str, metadata: dict[str, Any]) -> str:
    """Remove Source/URL/Task headers added during indexing while preserving body."""

    raw_text = str(document or "")
    if "\n" in raw_text:
        content_lines = [
            line for line in raw_text.splitlines()
            if not line.startswith("Source: ") and not line.startswith("URL: ") and not line.startswith("Task: ")
        ]
        body = clean_text("\n".join(content_lines))
        if body:
            return body

    text = clean_text(raw_text)
    if not text:
        return ""

    url = clean_text(metadata.get("url"))
    source_url = clean_text(metadata.get("source_url"))
    title = clean_text(metadata.get("title")) or url or source_url or "Untitled Source"
    task = clean_text(metadata.get("query_contexts"))

    candidate_headers = [
        clean_text(f"Source: {title} URL: {url} Task: {task}"),
        clean_text(f"Source: {title} URL: {source_url} Task: {task}"),
        clean_text(f"Source: {url} URL: {url} Task: {task}"),
        clean_text(f"Source: {source_url} URL: {source_url} Task: {task}"),
    ]
    for header in candidate_headers:
        if header and text.startswith(header):
            return clean_text(text[len(header):])

    task_marker = clean_text(f"Task: {task}")
    if task_marker and task_marker in text:
        _, _, tail = text.partition(task_marker)
        return clean_text(tail)

    return text


def stored_chunk_header_present(document: str) -> bool:
    """Detect chunks that begin with indexing metadata headers."""

    text = clean_text(document).lower()
    return text.startswith("source:") and " url:" in text and " task:" in text


def is_meaningful_evidence(text: str) -> bool:
    """Reject empty and metadata-only chunks before synthesis/report packaging."""

    value = clean_text(text)
    if len(value) < MIN_EVIDENCE_CHARS:
        return False
    if len(re.findall(r"[A-Za-z0-9_]+", value)) < MIN_EVIDENCE_TOKENS:
        return False
    if stored_chunk_header_present(value):
        return False
    return True


def count_meaningful_evidence_chunks(retrieved_context: Sequence[RetrievalResult]) -> int:
    count = 0
    for result in retrieved_context:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        chunk = retrieved_chunk_preview(result.document, metadata, max_chars=MIN_EVIDENCE_CHARS * 3)
        if is_meaningful_evidence(chunk):
            count += 1
    return count


def source_balanced_results(retrieved_context: Sequence[RetrievalResult]) -> list[RetrievalResult]:
    """Interleave results while surfacing primary/official sources first."""
    buckets: dict[str, list[RetrievalResult]] = {}
    source_order = []
    source_order_index = {}
    source_priority = {}

    for result in retrieved_context:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        url = primary_source_url(metadata) or result.id
        if url not in buckets:
            buckets[url] = []
            source_order_index[url] = len(source_order)
            source_order.append(url)
            source_priority[url] = primary_source_rank(metadata)
        else:
            source_priority[url] = min(source_priority[url], primary_source_rank(metadata))
        buckets[url].append(result)

    ordered_sources = sorted(source_order, key=lambda url: (source_priority.get(url, 1), source_order_index[url]))
    ordered = []
    while True:
        added = False
        for url in ordered_sources:
            bucket = buckets[url]
            if not bucket:
                continue
            ordered.append(bucket.pop(0))
            added = True
        if not added:
            break
    return ordered


def primary_source_rank(metadata: dict[str, Any]) -> int:
    return 0 if is_primary_source(metadata) else 1


def context_block_char_limit(
    retrieved_context: Sequence[RetrievalResult],
    max_context_chars: int,
) -> int:
    """Choose a per-block preview size that leaves room for many sources."""
    nonempty_count = sum(1 for result in retrieved_context if clean_text(result.document))
    if nonempty_count <= 0:
        return DEFAULT_CONTEXT_BLOCK_CHARS
    target_blocks = min(nonempty_count, 14)
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


def dedupe_source_urls(urls: Sequence[str]) -> list[str]:
    seen = set()
    deduped = []
    for url in urls:
        normalized = normalize_source_url(url)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped
