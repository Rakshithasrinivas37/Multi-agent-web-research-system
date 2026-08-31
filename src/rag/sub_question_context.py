"""Query rewriting and per-sub-question RAG retrieval helpers."""

from __future__ import annotations

import json
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from src.rag.indexing import get_collection
from src.rag.query_helpers import (
    COVERAGE_EVIDENCE_TERMS,
    COVERAGE_GENERIC_TERMS,
    OBJECTIVE_STOPWORDS,
    clean_model_name,
    clean_string_list,
    dedupe_preserve_order,
    facet_present,
    infer_question_evidence_types,
    query_keywords,
    query_tokens,
    question_key,
    question_required_facets,
    retrieval_topic_phrase,
    source_query_terms,
)
from src.rag.retrieval import RetrievalResult, expand_parent_context_results, multi_query_hybrid_retrieve, source_url_coverage_retrieve
from src.tools.groq_retry import create_chat_completion_with_retries
from src.tools.text_utils import clean_text


DEFAULT_SUBQUESTION_QUERY_REWRITE_MODEL = "qwen/qwen3.6-27b"
DEFAULT_SUBQUESTION_HF_QUERY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_SUBQUESTION_QUERY_VARIANTS = 3
DEFAULT_SUBQUESTION_RETRIEVAL_MAX_WORKERS = 4
DEFAULT_SUBQUESTION_RERANK_MAX_WORKERS = 2
DEFAULT_SYNTHESIS_CHUNKS_PER_QUESTION = 6
DEFAULT_SYNTHESIS_MAX_CHUNKS = 48
DEFAULT_CONTEXT_BLOCK_CHARS = 750
DEFAULT_PRECISION_QUERY_LIMIT = 8
DEFAULT_HF_QUERY_MAX_INPUT_TOKENS = 4096
DEFAULT_HF_QUERY_MAX_NEW_TOKENS = 900
GENERIC_QUERY_PHRASES = (
    r"\boverview evidence definition concept\b",
    r"\bevidence details examples equations benchmarks limitations\b",
    r"\bdetails examples equations benchmarks limitations\b",
    r"\bprimary source official docs paper\b",
    r"\bextract source-backed evidence\b",
    r"\bsource-backed evidence\b",
    r"\bsource-backed context explains\b",
    r"\bwhich evidence gives\b",
    r"\bevidence gives\b",
    r"\bwhere do authoritative sources discuss\b",
    r"\bauthoritative sources discuss\b",
    r"\banswering authoritative source\b",
    r"\bauthoritative source https\b",
    r"\bsource context\b",
)
QUERY_INSTRUCTION_TERMS = {
    "answer",
    "answering",
    "authoritative",
    "backed",
    "discuss",
    "evidence",
    "extract",
    "https",
    "retrieval",
    "source",
    "source-backed",
    "sources",
}
EVIDENCE_TERMS_BY_TYPE = {
    "api": {"api", "class", "function", "method", "parameter", "argument", "signature", "usage", "example"},
    "applications": {"application", "applications", "task", "tasks", "used for", "nlp", "vision", "computer vision", "image", "classification", "translation", "speech", "recognition"},
    "benchmark": {"benchmark", "benchmarks", "score", "scores", "result", "results", "performance", "accuracy", "bleu", "glue", "wmt", "imagenet", "cifar", "vtab", "top-1"},
    "complexity": {"complexity", "runtime", "memory", "quadratic", "linear", "scalability", "sequence length"},
    "definition": {"definition", "defined as", "refers to", "means", "purpose", "overview"},
    "equation": {"equation", "formula", "formulation", "softmax", "sqrt", "tanh", "exp", "alpha", "sum"},
    "limitations": {"limitation", "limitations", "challenge", "challenges", "drawback", "bottleneck", "open question"},
}


def generation_helpers() -> Any:
    """Import generation lazily to avoid a heavy circular module split."""

    from src.rag import generation

    return generation


def sub_question_query_rewrite_model(model: str | None = None) -> str:
    """Model used only for rewriting planner sub-questions into retrieval queries."""

    return clean_model_name(os.environ.get("RAG_SUBQUESTION_QUERY_MODEL")) or DEFAULT_SUBQUESTION_QUERY_REWRITE_MODEL


def query_rewrite_provider() -> str:
    provider = clean_text(os.environ.get("RAG_QUERY_REWRITE_PROVIDER", "auto")).lower()
    if provider in {"hf", "huggingface", "local", "transformers"}:
        return "hf"
    if provider in {"groq", "api"}:
        return "groq"
    if provider in {"off", "false", "none", "disabled"}:
        return "off"
    return "auto"


def hf_sub_question_query_model() -> str:
    return clean_model_name(os.environ.get("RAG_SUBQUESTION_HF_MODEL")) or DEFAULT_SUBQUESTION_HF_QUERY_MODEL


def sub_question_retrieval_queries(
    question: str,
    objective: str = "",
    task_details: str = "",
    max_variants: int = DEFAULT_SUBQUESTION_QUERY_VARIANTS,
) -> list[str]:
    """Rewrite one planner sub-question into compact retrieval queries."""

    question = clean_text(question)
    objective = clean_text(objective)
    task_details = clean_text(task_details)
    if not question:
        return []

    evidence_types = infer_question_evidence_types(f"{question} {task_details}")
    evidence_terms = query_evidence_terms(evidence_types)
    facets = question_required_facets(f"{question} {task_details}")
    source_terms = source_query_terms(task_details)
    topic = retrieval_topic_phrase(f"{question} {task_details}", limit=12)
    key_terms = " ".join(query_keywords(f"{question} {task_details}", limit=10)) or topic or objective
    variants = []
    for facet in facets[: max(1, max_variants)]:
        variants.append(clean_retrieval_query(f"{facet} {objective} {topic_without_other_facets(topic, facet, facets)} {evidence_terms}"))
    variants.extend(
        [
            clean_retrieval_query(f"{topic} {evidence_terms}"),
            clean_retrieval_query(f"{key_terms} {evidence_terms}"),
            clean_retrieval_query(f"{objective} {topic} {evidence_terms}"),
            clean_retrieval_query(f"{source_terms} {topic} {evidence_terms}"),
        ]
    )
    return dedupe_preserve_order(variant for variant in variants if variant)[: max(1, max_variants)]


def topic_without_other_facets(topic: str, facet: str, facets: Sequence[str]) -> str:
    text = clean_text(topic)
    for other in facets:
        if clean_text(other).lower() == clean_text(facet).lower():
            continue
        text = re.sub(rf"\b{re.escape(clean_text(other))}\b", " ", text, flags=re.IGNORECASE)
    return clean_text(text)


def query_evidence_terms(evidence_types: Sequence[str]) -> str:
    term_map = {
        "api": ["official documentation", "api signature", "parameters", "usage"],
        "applications": ["applications", "use cases", "tasks"],
        "benchmark": ["benchmark", "results", "scores", "metrics"],
        "comparison": ["comparison", "differences", "tradeoffs"],
        "complexity": ["complexity", "runtime", "memory", "scaling"],
        "definition": ["definition", "purpose"],
        "equation": ["equation", "formula", "variables", "components"],
        "examples": ["examples", "cases"],
        "implementation": ["implementation", "api signature", "parameters"],
        "limitations": ["limitations", "challenges", "bottlenecks"],
    }
    terms = []
    for evidence_type in evidence_types or ["evidence"]:
        terms.extend(term_map.get(clean_text(evidence_type).lower(), []))
    return clean_text(" ".join(dedupe_preserve_order(terms)))


def clean_retrieval_query(query: str, max_words: int = 18) -> str:
    text = clean_text(query)
    text = re.sub(r"^(?:what|which|where|how)\s+(?:is|are|does|do|did|can|should|has|have)?\s*", "", text, flags=re.IGNORECASE)
    for phrase in GENERIC_QUERY_PHRASES:
        text = re.sub(phrase, " ", text, flags=re.IGNORECASE)
    tokens = []
    seen = set()
    for token in re.findall(r"https?://[^\s]+|[A-Za-z0-9_+#./-]+", text):
        cleaned = token.strip(".,;:()[]{}\"'")
        if not cleaned:
            continue
        key = cleaned.lower().replace("‑", "-").replace("–", "-")
        if key in seen:
            continue
        if key in QUERY_INSTRUCTION_TERMS:
            continue
        seen.add(key)
        tokens.append(cleaned)
    return clean_text(" ".join(tokens[:max_words]))[:300]


def planner_tasks_to_rag_queries(research_plan: dict[str, Any]) -> list[str]:
    """Build retrieval queries from PlannerAgent output."""

    objective = clean_text(research_plan.get("objective"))
    tasks = [task for task in research_plan.get("tasks", []) if isinstance(task, dict)]
    sub_question_queries = []
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
        return dedupe_preserve_order([*sub_question_queries, objective])

    queries = []
    for task in tasks:
        expected_signals_text = " ".join(clean_text(item) for item in task.get("expected_signals", []) if clean_text(item))
        query = clean_text(
            " ".join(
                clean_text(part)
                for part in [
                    task.get("query_context"),
                    task.get("extraction_goal"),
                    task.get("target_name"),
                    task.get("url"),
                    expected_signals_text,
                ]
                if clean_text(part)
            )
        )
        if query:
            queries.append(query)
    return dedupe_preserve_order([*queries, objective])


def precision_retrieval_queries(
    research_plan: dict[str, Any],
    objective: str = "",
    max_queries: int = DEFAULT_PRECISION_QUERY_LIMIT,
) -> list[str]:
    """Build deterministic high-signal queries for exact evidence retrieval."""

    helpers = generation_helpers()
    objective_text = clean_text(objective or research_plan.get("objective"))
    synthesis_instruction = clean_text(research_plan.get("synthesis_instruction"))
    tasks = [task for task in research_plan.get("tasks", []) if isinstance(task, dict)]
    candidates = [
        *helpers.planner_sub_questions(research_plan),
        *helpers.instruction_requirement_items(synthesis_instruction),
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

    question_terms = query_tokens(question)
    parts = []
    for task in tasks:
        context = clean_text(task.get("query_context"))
        context_terms = query_tokens(context)
        if not context_terms:
            continue
        overlap = len(question_terms & context_terms) / max(1, min(len(question_terms), len(context_terms)))
        if context.lower() != clean_text(question).lower() and overlap < 0.45:
            continue
        expected_signals = " ".join(clean_text(item) for item in task.get("expected_signals", []) if clean_text(item))
        parts.extend([task.get("extraction_goal"), expected_signals, task.get("url")])
    return clean_text(" ".join(clean_text(part) for part in parts if clean_text(part)))


def retrieve_sub_question_context(
    research_plan: dict[str, Any],
    questions: Sequence[str],
    objective: str,
    chroma_path: str,
    collection_name: str,
    history_keys: Sequence[str] | None,
    candidate_chunks: int,
    final_chunks: int,
    per_query_k: int,
    semantic_k: int,
    bm25_k: int,
    semantic_weight: float,
    bm25_weight: float,
    authority_weight: float,
    bm25_scan_limit: int,
    embedding_device: str,
    rerank: bool,
    reranker_model: str,
    rerank_k: int,
    rerank_weight: float,
    browser_results: Sequence[Any] | None = None,
) -> list[RetrievalResult]:
    """Retrieve and rerank an independent evidence set for each planner sub-question."""

    return flatten_sub_question_context_groups(
        retrieve_sub_question_context_groups(
            research_plan=research_plan,
            questions=questions,
            objective=objective,
            chroma_path=chroma_path,
            collection_name=collection_name,
            history_keys=history_keys,
            candidate_chunks=candidate_chunks,
            final_chunks=final_chunks,
            per_query_k=per_query_k,
            semantic_k=semantic_k,
            bm25_k=bm25_k,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
            authority_weight=authority_weight,
            bm25_scan_limit=bm25_scan_limit,
            embedding_device=embedding_device,
            rerank=rerank,
            reranker_model=reranker_model,
            rerank_k=rerank_k,
            rerank_weight=rerank_weight,
            browser_results=browser_results,
        )
    )


def retrieve_sub_question_context_groups(
    research_plan: dict[str, Any],
    questions: Sequence[str],
    objective: str,
    chroma_path: str,
    collection_name: str,
    history_keys: Sequence[str] | None,
    candidate_chunks: int,
    final_chunks: int,
    per_query_k: int,
    semantic_k: int,
    bm25_k: int,
    semantic_weight: float,
    bm25_weight: float,
    authority_weight: float,
    bm25_scan_limit: int,
    embedding_device: str,
    rerank: bool,
    reranker_model: str,
    rerank_k: int,
    rerank_weight: float,
    browser_results: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve an inspectable chunk group for each planner sub-question."""

    helpers = generation_helpers()
    tasks = [task for task in research_plan.get("tasks", []) if isinstance(task, dict)]
    candidate_chunks = max(1, candidate_chunks)
    final_chunks = max(1, min(final_chunks, candidate_chunks))
    clean_questions = dedupe_preserve_order(questions)
    question_source_urls = helpers.planner_question_source_urls(research_plan)
    evidence_by_question = {
        question_key(spec.get("question")): clean_string_list(spec.get("required_evidence"))
        for spec in helpers.planner_sub_question_specs(research_plan)
        if clean_text(spec.get("question"))
    }

    def retrieve_question(question: str) -> dict[str, Any]:
        required_evidence = evidence_by_question.get(question_key(question)) or infer_question_evidence_types(question)
        query_set = per_question_context_queries(question, objective=objective, tasks=tasks)
        if not query_set:
            return sub_question_context_group(question, [], [], [])
        fallback_sources = []
        candidates = multi_query_hybrid_retrieve(
            queries=query_set,
            chroma_path=chroma_path,
            collection_name=collection_name,
            top_k=candidate_chunks,
            per_query_k=max(1, min(per_query_k, candidate_chunks)),
            semantic_k=semantic_k,
            bm25_k=bm25_k,
            history_keys=history_keys,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
            authority_weight=authority_weight,
            bm25_scan_limit=bm25_scan_limit,
            embedding_device=embedding_device,
            diversify_urls=False,
            rerank=rerank,
            reranker_model=reranker_model,
            rerank_k=max(rerank_k, candidate_chunks),
            rerank_weight=rerank_weight,
        )
        if len(candidates) < final_chunks:
            try:
                source_candidates = source_url_coverage_retrieve(
                    source_urls=helpers.question_source_urls_for(question, question_source_urls),
                    query=query_set,
                    chroma_path=chroma_path,
                    collection_name=collection_name,
                    history_keys=history_keys,
                    top_k_per_url=candidate_chunks,
                    scan_limit=bm25_scan_limit,
                )
            except Exception as error:
                print(f"[synthesis] per-question source URL retrieval failed: {error}")
                source_candidates = []
            if source_candidates:
                fallback_sources.append("source_url")
                candidates = helpers.merge_retrieved_context(candidates, source_candidates)
        if len(candidates) < final_chunks:
            fallback_candidates = collection_scan_question_retrieve(
                question=question,
                queries=query_set,
                chroma_path=chroma_path,
                collection_name=collection_name,
                history_keys=history_keys,
                top_k=candidate_chunks,
                scan_limit=bm25_scan_limit,
                question_source_urls=question_source_urls,
                required_evidence=required_evidence,
            )
            if fallback_candidates:
                fallback_sources.append("collection_scan")
                candidates = helpers.merge_retrieved_context(candidates, fallback_candidates)
        browser_candidates = browser_question_context_retrieve(
            question=question,
            queries=query_set,
            browser_results=browser_results or [],
            top_k=candidate_chunks,
            question_source_urls=question_source_urls,
            required_evidence=required_evidence,
        )
        if browser_candidates:
            fallback_sources.append("browser_results")
            candidates = helpers.merge_retrieved_context(candidates, browser_candidates)
        missing_facets = missing_facets_for_results(question, candidates)
        if missing_facets:
            facet_candidates = facet_rescue_context_retrieve(
                question=question,
                missing_facets=missing_facets,
                queries=query_set,
                chroma_path=chroma_path,
                collection_name=collection_name,
                history_keys=history_keys,
                top_k=candidate_chunks,
                scan_limit=bm25_scan_limit,
                question_source_urls=question_source_urls,
                required_evidence=required_evidence,
            )
            if facet_candidates:
                fallback_sources.append("facet_scan")
                candidates = helpers.merge_retrieved_context(candidates, facet_candidates)
        if not candidates:
            print(f"[synthesis] no per-question chunks found for: {question[:120]}")
        ranked_candidates = rank_collection_scan_results(
            question=question,
            queries=query_set,
            candidates=candidates,
            question_source_urls=question_source_urls,
            required_evidence=required_evidence,
        )
        selected = helpers.meaningful_retrieval_results(ranked_candidates) or ranked_candidates or list(candidates)
        selected = select_facet_covered_results(question, selected, final_chunks)
        tagged = [tag_result_for_question(result, question) for result in selected[:final_chunks]]
        return sub_question_context_group(
            question,
            query_set,
            candidates,
            tagged,
            fallback_sources=fallback_sources,
        )

    max_workers = sub_question_retrieval_max_workers(len(clean_questions), rerank=rerank)
    if max_workers <= 1:
        return [retrieve_question(question) for question in clean_questions]
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rag-subquestion") as executor:
        return list(executor.map(retrieve_question, clean_questions))


def sub_question_context_group(
    question: str,
    queries: Sequence[str],
    candidates: Sequence[RetrievalResult],
    chunks: Sequence[RetrievalResult],
    fallback_used: bool = False,
    fallback_sources: Sequence[str] | None = None,
) -> dict[str, Any]:
    clean_fallback_sources = [clean_text(source) for source in fallback_sources or [] if clean_text(source)]
    return {
        "question": clean_text(question),
        "queries": list(queries),
        "candidate_count": len(candidates),
        "chunk_count": len(chunks),
        "fallback_used": bool(fallback_used or clean_fallback_sources),
        "fallback_sources": clean_fallback_sources,
        "chunks": list(chunks),
    }


def browser_question_context_retrieve(
    question: str,
    queries: Sequence[str],
    browser_results: Sequence[Any],
    top_k: int,
    question_source_urls: dict[str, list[str]] | None = None,
    required_evidence: Sequence[str] | None = None,
) -> list[RetrievalResult]:
    """Build per-question evidence chunks from already extracted browser sources."""

    helpers = generation_helpers()
    if not browser_results:
        return []

    query_text = clean_text(" ".join([question, *queries]))
    evidence_terms = browser_evidence_query_terms(question, required_evidence=required_evidence)
    facet_terms = {facet.lower() for facet in question_required_facets(question)}
    query_terms = query_tokens(query_text) | evidence_terms | facet_terms
    priority_terms = evidence_terms | facet_terms
    source_urls = helpers.question_source_urls_for(question, question_source_urls)
    candidates = []
    for result_index, browser_result in enumerate(browser_results):
        if not isinstance(browser_result, dict):
            continue
        task_context = clean_text(browser_result.get("query_context"))
        for source_index, source in enumerate(browser_result.get("sources") or []):
            if not isinstance(source, dict):
                continue
            url = clean_text(source.get("url"))
            title = clean_text(source.get("title")) or url or f"Browser source {source_index + 1}"
            metadata = {
                "url": url,
                "source_url": url,
                "title": title,
                "source_type": clean_text(source.get("source_type")),
                "source_quality": clean_text(source.get("source_quality")),
                "source_authority": clean_text(source.get("source_authority")),
                "query_contexts": task_context,
            }
            for snippet_index, snippet in enumerate(browser_source_snippets(source, query_terms=query_terms, priority_terms=priority_terms)):
                content = clean_text(snippet)
                if not helpers.is_meaningful_evidence(content):
                    continue
                candidates.append(
                    RetrievalResult(
                        id=browser_question_chunk_id(url, content, result_index, source_index, snippet_index),
                        document=content,
                        metadata=metadata,
                        score=0.0,
                        semantic_score=0.0,
                        bm25_score=0.0,
                        authority_score=1.0 if helpers.is_primary_source(metadata) else 0.0,
                    )
                )

    ranked = rank_collection_scan_results(
        question=question,
        queries=queries,
        candidates=candidates,
        question_source_urls=question_source_urls,
        required_evidence=required_evidence,
    )
    return ranked[: max(1, top_k)]


def browser_source_snippets(
    source: dict[str, Any],
    query_terms: set[str],
    priority_terms: set[str] | None = None,
    max_snippets: int = 8,
) -> list[str]:
    """Return compact source snippets likely to answer a planner sub-question."""

    snippets = []
    for key in ("full_content", "content_preview"):
        text = clean_text(source.get(key))
        if text:
            if priority_terms:
                snippets.extend(text_windows_for_terms(text, query_terms=priority_terms, max_windows=5))
            snippets.extend(text_windows_for_terms(text, query_terms=query_terms))

    for key in ("important_sections", "extracted_facts", "evidence"):
        for item in source.get(key) or []:
            text = browser_source_item_text(item)
            if text:
                snippets.append(text)

    seen = set()
    unique = []
    for snippet in snippets:
        value = clean_text(snippet)
        if not value:
            continue
        key = value[:300].lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
        if len(unique) >= max_snippets:
            break
    return unique


def browser_evidence_query_terms(question: str, required_evidence: Sequence[str] | None = None) -> set[str]:
    """Terms used to recover exact evidence from browser text before vector chunks lose it."""

    terms = set()
    evidence_types = clean_string_list(list(required_evidence or [])) or infer_question_evidence_types(question)
    for evidence_type in evidence_types:
        terms.update(EVIDENCE_TERMS_BY_TYPE.get(clean_text(evidence_type).lower(), set()))
    return {term.lower() for term in terms if len(term) >= 3}


def browser_source_item_text(item: Any) -> str:
    if isinstance(item, str):
        return clean_text(item)
    if not isinstance(item, dict):
        return clean_text(item)
    parts = []
    for key in ("heading", "title", "section", "fact", "evidence", "content", "text", "summary"):
        value = clean_text(item.get(key))
        if value:
            parts.append(value)
    return clean_text(" ".join(parts))


def text_windows_for_terms(text: str, query_terms: set[str], max_windows: int = 4, window_chars: int = 900) -> list[str]:
    value = clean_text(text)
    if not value:
        return []
    lowered = value.lower()
    positions = []
    for term in sorted(query_terms, key=len, reverse=True):
        if len(term) < 3:
            continue
        match = re.search(rf"\b{re.escape(term.lower())}\b", lowered)
        if match:
            positions.append(match.start())
        if len(positions) >= max_windows:
            break
    if not positions:
        positions = [0]

    windows = []
    for position in positions:
        start = max(0, position - (window_chars // 3))
        end = min(len(value), start + window_chars)
        windows.append(value[start:end])
    return windows


def browser_question_chunk_id(url: str, content: str, result_index: int, source_index: int, snippet_index: int) -> str:
    digest = hashlib.sha1(f"{url}|{content}".encode("utf-8", errors="ignore")).hexdigest()[:24]
    return f"browser-question-{result_index}-{source_index}-{snippet_index}-{digest}"


def collection_scan_question_retrieve(
    question: str,
    queries: Sequence[str],
    chroma_path: str,
    collection_name: str,
    history_keys: Sequence[str] | None,
    top_k: int,
    scan_limit: int,
    question_source_urls: dict[str, list[str]] | None = None,
    required_evidence: Sequence[str] | None = None,
) -> list[RetrievalResult]:
    """Fallback per-question retrieval by directly scoring Chroma rows."""

    helpers = generation_helpers()
    try:
        collection = get_collection(chroma_path, collection_name)
        raw = collection.get(
            include=["documents", "metadatas"],
            limit=max(1, scan_limit, top_k * 25),
        )
    except Exception as error:
        print(f"[synthesis] per-question collection scan failed: {error}")
        return []

    ids = raw.get("ids", []) if isinstance(raw, dict) else []
    documents = raw.get("documents", []) if isinstance(raw, dict) else []
    metadatas = raw.get("metadatas", []) if isinstance(raw, dict) else []
    allowed_history_keys = set(clean_string_list(history_keys or []))
    candidates = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        if allowed_history_keys and clean_text(metadata.get("history_key")) not in allowed_history_keys:
            continue
        item_id = clean_text(ids[index] if index < len(ids) else "")
        content = clean_text(document)
        if not item_id or not content:
            continue
        candidates.append(
            RetrievalResult(
                id=item_id,
                document=content,
                metadata=metadata,
                score=0.0,
                semantic_score=0.0,
                bm25_score=0.0,
                authority_score=1.0 if helpers.is_primary_source(metadata) else 0.0,
            )
        )

    ranked = rank_collection_scan_results(
        question=question,
        queries=queries,
        candidates=candidates,
        question_source_urls=question_source_urls,
        required_evidence=required_evidence,
    )
    selected = ranked[: max(1, top_k)]
    return expand_parent_context_results(selected, chroma_path=chroma_path)


def missing_facets_for_results(question: str, results: Sequence[RetrievalResult]) -> list[str]:
    helpers = generation_helpers()
    facets = question_required_facets(question)
    if not facets:
        return []
    text = clean_text(" ".join(helpers.retrieval_result_text(result) for result in results))
    return [facet for facet in facets if not facet_present(facet, text)]


def facet_rescue_context_retrieve(
    question: str,
    missing_facets: Sequence[str],
    queries: Sequence[str],
    chroma_path: str,
    collection_name: str,
    history_keys: Sequence[str] | None,
    top_k: int,
    scan_limit: int,
    question_source_urls: dict[str, list[str]] | None = None,
    required_evidence: Sequence[str] | None = None,
) -> list[RetrievalResult]:
    helpers = generation_helpers()
    evidence_terms = " ".join(sorted(browser_evidence_query_terms(question, required_evidence=required_evidence))[:8])
    facet_queries = dedupe_preserve_order(
        clean_text(f"{facet} {evidence_terms}") for facet in missing_facets if clean_text(facet)
    )
    if not facet_queries:
        return []
    return collection_scan_question_retrieve(
        question=question,
        queries=dedupe_preserve_order([*queries, *facet_queries]),
        chroma_path=chroma_path,
        collection_name=collection_name,
        history_keys=history_keys,
        top_k=max(1, top_k),
        scan_limit=scan_limit,
        question_source_urls=question_source_urls,
        required_evidence=required_evidence,
    )


def select_facet_covered_results(
    question: str,
    ranked_results: Sequence[RetrievalResult],
    limit: int,
) -> list[RetrievalResult]:
    helpers = generation_helpers()
    results = list(ranked_results)
    facets = question_required_facets(question)
    if not facets or not results:
        return results[: max(1, limit)]

    selected = []
    seen = set()
    for facet in facets:
        for result in results:
            key = helpers.retrieval_result_key(result)
            if key in seen:
                continue
            if facet_present(facet, helpers.retrieval_result_text(result)):
                selected.append(result)
                seen.add(key)
                break

    for result in results:
        if len(selected) >= max(1, limit):
            break
        key = helpers.retrieval_result_key(result)
        if key in seen:
            continue
        selected.append(result)
        seen.add(key)
    return selected[: max(1, limit)]


def rank_collection_scan_results(
    question: str,
    queries: Sequence[str],
    candidates: Sequence[RetrievalResult],
    question_source_urls: dict[str, list[str]] | None = None,
    required_evidence: Sequence[str] | None = None,
) -> list[RetrievalResult]:
    """Rank direct collection rows for one planner question."""

    helpers = generation_helpers()
    query_text = clean_text(" ".join([question, *queries]))
    query_terms = query_tokens(query_text)
    topic_terms = {term for term in query_terms if term not in COVERAGE_GENERIC_TERMS}
    evidence_types = clean_string_list(list(required_evidence or [])) or infer_question_evidence_types(question)
    source_urls = helpers.question_source_urls_for(question, question_source_urls)
    facets = question_required_facets(question)
    scored = []
    fallback = []

    for position, result in enumerate(candidates):
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        chunk = helpers.retrieved_chunk_preview(result.document, metadata, max_chars=DEFAULT_CONTEXT_BLOCK_CHARS * 2)
        if not helpers.is_meaningful_evidence(chunk):
            continue
        fallback.append(result)
        text = helpers.retrieval_result_text(result)
        text_terms = query_tokens(text)
        overlap = len(query_terms & text_terms)
        topic_overlap = len(topic_terms & text_terms)
        evidence_score = helpers.evidence_type_score(text, evidence_types)
        signal_score = helpers.evidence_signal_score(text, evidence_types)
        table_score = metadata_table_signal_score(metadata, evidence_types)
        facet_score = sum(1 for facet in facets if facet_present(facet, text))
        preferred_source = helpers.result_matches_source_urls(result, source_urls)
        primary_source = helpers.is_primary_source(metadata)
        primary_signal = primary_source and signal_score
        exact_signal = signal_score >= 10
        if not (overlap or topic_overlap or evidence_score or signal_score or facet_score or preferred_source):
            continue
        score = (
            (topic_overlap * 4.0)
            + (overlap * 1.5)
            + (facet_score * 5.0)
            + (evidence_score * 3.0)
            + (signal_score * 5.0)
            + table_score
            + (20.0 if exact_signal else 0.0)
            + (10.0 if primary_signal else 0.0)
            + (5.0 if preferred_source else 0.0)
            + helpers.retrieval_result_priority(result)
            + (1.0 / (position + 1))
        )
        scored.append((score, result))

    ranked = [
        collection_scan_scored_result(result, score)
        for score, result in sorted(scored, key=lambda item: item[0], reverse=True)
    ]
    if ranked:
        return helpers.unique_retrieval_results(ranked)
    return helpers.source_balanced_results(fallback)


def metadata_table_signal_score(metadata: dict[str, Any], evidence_types: Sequence[str]) -> float:
    if not (metadata.get("chunk_kind") == "table" or metadata.get("has_table_signal")):
        return 0.0
    evidence = {clean_text(item).lower() for item in evidence_types}
    if evidence & {"benchmark", "comparison", "applications"}:
        return 8.0
    return 3.0


def collection_scan_scored_result(result: RetrievalResult, score: float) -> RetrievalResult:
    return RetrievalResult(
        id=result.id,
        document=result.document,
        metadata=result.metadata,
        score=float(score),
        semantic_score=result.semantic_score,
        bm25_score=result.bm25_score,
        authority_score=result.authority_score,
        rerank_score=result.rerank_score,
        semantic_rank=result.semantic_rank,
        bm25_rank=result.bm25_rank,
    )


def flatten_sub_question_context_groups(groups: Sequence[dict[str, Any]]) -> list[RetrievalResult]:
    helpers = generation_helpers()
    results = []
    for group in groups:
        chunks = group.get("chunks", []) if isinstance(group, dict) else []
        results = helpers.merge_retrieved_context(results, chunks)
    return results


def sub_question_context_counts(groups: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "question": clean_text(group.get("question")),
            "query_count": len(group.get("queries", [])),
            "candidate_count": int(group.get("candidate_count") or 0),
            "chunk_count": int(group.get("chunk_count") or 0),
            "fallback_used": bool(group.get("fallback_used")),
            "fallback_sources": list(group.get("fallback_sources", [])),
        }
        for group in groups
        if isinstance(group, dict)
    ]


def tag_result_for_question(result: RetrievalResult, question: str) -> RetrievalResult:
    metadata = dict(result.metadata or {})
    metadata["synthesis_question"] = clean_text(question)
    return RetrievalResult(
        id=result.id,
        document=result.document,
        metadata=metadata,
        score=result.score,
        semantic_score=result.semantic_score,
        bm25_score=result.bm25_score,
        authority_score=result.authority_score,
        rerank_score=result.rerank_score,
        semantic_rank=result.semantic_rank,
        bm25_rank=result.bm25_rank,
    )


def select_question_first_synthesis_context(
    question_context_results: Sequence[RetrievalResult],
    fallback_context: Sequence[RetrievalResult],
    planner_questions: Sequence[str],
    question_source_urls: dict[str, list[str]] | None = None,
) -> list[RetrievalResult]:
    """Prefer per-question RAG chunks; use global/browser context only as fallback."""

    helpers = generation_helpers()
    max_chunks = max(
        DEFAULT_SYNTHESIS_MAX_CHUNKS,
        len(planner_questions) * DEFAULT_SYNTHESIS_CHUNKS_PER_QUESTION,
    )
    if question_context_results:
        selected = list(question_context_results)[:max_chunks]
        seen = {helpers.retrieval_result_key(result) for result in selected}
        for result in helpers.select_synthesis_context(
            fallback_context,
            planner_questions,
            question_source_urls=question_source_urls,
            max_chunks=max_chunks,
        ):
            if len(selected) >= max_chunks:
                break
            key = helpers.retrieval_result_key(result)
            if key in seen:
                continue
            selected.append(result)
            seen.add(key)
        return selected
    return helpers.select_synthesis_context(
        fallback_context,
        planner_questions,
        question_source_urls=question_source_urls,
        max_chunks=max_chunks,
    )


def sub_question_retrieval_max_workers(question_count: int, rerank: bool = False) -> int:
    """Bound parallel sub-question retrieval, especially when CrossEncoder rerank is enabled."""

    default_workers = DEFAULT_SUBQUESTION_RERANK_MAX_WORKERS if rerank else DEFAULT_SUBQUESTION_RETRIEVAL_MAX_WORKERS
    try:
        workers = int(os.environ.get("RAG_SUBQUESTION_RETRIEVAL_MAX_WORKERS", default_workers))
    except (TypeError, ValueError):
        workers = default_workers
    return min(max(1, workers), max(1, question_count))


def per_question_context_queries(question: str, objective: str = "", tasks: Sequence[dict[str, Any]] = ()) -> list[str]:
    """Build broad plus exact-evidence queries for one planner sub-question."""

    helpers = generation_helpers()
    task_details = matching_task_details(question, tasks)
    queries = sub_question_retrieval_queries(question, objective=objective, task_details=task_details)
    exact_queries = [
        clean_text(f"{objective} {question} {suffix} {task_details}")[:700]
        for suffix in precision_query_suffixes(question)
    ]
    return dedupe_preserve_order([*queries, *exact_queries])


def llm_sub_question_retrieval_query_result(
    research_plan: dict[str, Any],
    model: str | None = None,
    max_variants: int = DEFAULT_SUBQUESTION_QUERY_VARIANTS,
) -> dict[str, Any]:
    """Rewrite planner sub-questions into RAG queries with Groq plus optional HF fallback."""

    provider = query_rewrite_provider()
    if provider == "off":
        return empty_llm_query_result(model=sub_question_query_rewrite_model(model), error="query rewrite provider is disabled", provider="off")
    if provider == "hf":
        return hf_sub_question_retrieval_query_result(research_plan, model=model, max_variants=max_variants)

    groq_result = groq_sub_question_retrieval_query_result(research_plan, model=model, max_variants=max_variants)
    if groq_result.get("queries") or provider == "groq":
        return groq_result

    hf_result = hf_sub_question_retrieval_query_result(
        research_plan,
        model=model,
        max_variants=max_variants,
        fallback_reason=groq_result.get("error") or "Groq query rewrite returned no usable queries",
    )
    if hf_result.get("queries"):
        return hf_result

    error = "; ".join(
        item for item in [groq_result.get("error"), f"HF fallback failed: {hf_result.get('error')}"] if clean_text(item)
    )
    return {
        **groq_result,
        "error": error or "query rewrite returned no usable queries",
        "provider": "auto",
        "fallback_provider": "hf",
        "fallback_error": hf_result.get("error", ""),
        "fallback_raw_response": hf_result.get("raw_response", ""),
    }


def groq_sub_question_retrieval_query_result(
    research_plan: dict[str, Any],
    model: str | None = None,
    max_variants: int = DEFAULT_SUBQUESTION_QUERY_VARIANTS,
) -> dict[str, Any]:
    """Use Groq to rewrite planner sub-questions into RAG queries."""

    helpers = generation_helpers()
    objective = clean_text(research_plan.get("objective"))
    questions = helpers.planner_sub_questions(research_plan)
    selected_model = sub_question_query_rewrite_model(model)
    if not questions:
        return empty_llm_query_result(model=selected_model, provider="groq")
    if not os.environ.get("GROQ_API_KEY"):
        return empty_llm_query_result(model=selected_model, error="GROQ_API_KEY is not set", provider="groq")

    try:
        from groq import Groq
    except ImportError as error:
        return empty_llm_query_result(model=selected_model, error=str(error), provider="groq")

    tasks = [task for task in research_plan.get("tasks", []) if isinstance(task, dict)]
    prompt = sub_question_rewrite_prompt(objective, questions, tasks)

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
                        "You rewrite planner sub-questions into compact keyword-style RAG search "
                        "queries that maximize recall across papers, docs, reports, and web extracts. "
                        "Return JSON only."
                    ),
                },
                {"role": "user", "content": prompt[:6000]},
            ],
        )
    except Exception as error:  # pragma: no cover - exercised through integration runs.
        return empty_llm_query_result(model=selected_model, error=str(error), provider="groq")

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
            "provider": "groq",
        }
    queries = complete_sub_question_query_coverage(queries, research_plan, max_variants=max_variants)
    return {
        "queries": dedupe_preserve_order(query[:700] for query in queries)[:max_queries],
        "model": clean_text(getattr(response, "model", "")) or selected_model,
        "error": "",
        "raw_response": raw_response_text[:1000],
        "provider": "groq",
    }


def sub_question_rewrite_prompt(objective: str, questions: Sequence[str], tasks: Sequence[dict[str, Any]]) -> str:
    return f"""Research objective:
{objective}

Planner sub-questions and optional task details:
{format_sub_question_rewrite_items(questions, tasks)}

Rewrite each planner sub-question into 2-3 compact, high-recall RAG search queries.
Requirements:
- Return exactly 2 or 3 queries for every sub-question, in the same order as the planner list.
- Base each query on the actual planner sub-question topic and named entities. Task details may add source names, URLs, APIs, datasets, metrics, or paper titles only when they match that sub-question.
- Write compact keyword-style queries as noun phrases, not full sentences or questions.
- Do not copy instruction words into queries: no "extract", "source-backed", "authoritative", "evidence gives", "source context", or "answering".
- Do not start queries with "what", "which", "where", "how", or phrases like "source-backed context", "which evidence gives", or "authoritative sources discuss".
- Do not append catch-all tails like "evidence details examples equations benchmarks limitations" or "overview evidence definition concept".
- For each sub-question, create complementary queries: one broad concept query, one exact evidence query, and one source-targeted query for a named facet, source, API, dataset, or metric when present.
- For compound/list questions, each query should focus on a named facet, dataset, method, framework, metric, or source from the question instead of repeating the whole question.
- If a sub-question names multiple facets, split them across queries rather than mixing all facets into every query.
- Prefer noun phrases that can match both semantic search and BM25 keyword search.
- Keep each query tied to one planner sub-question. Do not let benchmark, limitation, or formula queries drift into unrelated sub-question groups.
- Preserve named entities, URLs, titles, years, model names, datasets, metrics, APIs, equations, aliases, and important technical terms from the planner/task details.
- For equation/formula questions, include nearby evidence terms such as equation, formula, derivation, score function, alignment, matrix, variables, components, or operation names when relevant.
- For benchmark questions, include dataset names, metrics, result table, scores, comparison, performance, and evaluation terms when relevant.
- For API/implementation questions, include official documentation, class/function names, signature, parameters, usage, and example terms when relevant.
- For limitation/complexity questions, include complexity, memory, runtime, scaling, tradeoffs, bottleneck, sparse, efficient, or alternatives when relevant.
- Prefer 6-14 words per query. Do not exceed 18 words unless a URL or API name requires it.
- Do not answer the question and do not add citations.
- Do not include reasoning, <think> text, or explanations.
- Never output placeholder labels or generic query names.
- Good shape: "named method equation formula variables"; "named dataset benchmark scores metrics"; "framework API signature parameters".
- Bad shape: "topic evidence details examples equations benchmarks limitations"; "Which evidence gives topic details".
- Return JSON only in this shape:
{{"items":[{{"sub_question":"...","queries":["named topic exact evidence terms","named facet comparison terms"]}}]}}"""


def hf_sub_question_retrieval_query_result(
    research_plan: dict[str, Any],
    model: str | None = None,
    max_variants: int = DEFAULT_SUBQUESTION_QUERY_VARIANTS,
    fallback_reason: str = "",
) -> dict[str, Any]:
    """Use a local Hugging Face transformers model to rewrite retrieval queries."""

    helpers = generation_helpers()
    objective = clean_text(research_plan.get("objective"))
    questions = helpers.planner_sub_questions(research_plan)
    selected_model = hf_sub_question_query_model()
    if not questions:
        return empty_llm_query_result(model=selected_model, provider="hf")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    except Exception as error:
        return empty_llm_query_result(model=selected_model, error=str(error), provider="hf")

    tasks = [task for task in research_plan.get("tasks", []) if isinstance(task, dict)]
    prompt = sub_question_rewrite_prompt(objective, questions, tasks)
    device = hf_query_device(torch)
    tokenizer = None
    model_obj = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(selected_model, trust_remote_code=True)
        model_cls = AutoModelForSeq2SeqLM if hf_query_model_is_seq2seq(selected_model) else AutoModelForCausalLM
        model_kwargs = {"trust_remote_code": True}
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        model_obj = model_cls.from_pretrained(selected_model, **model_kwargs)
        model_obj.to(device)
        model_obj.eval()
        raw_response = hf_generate_query_response(
            tokenizer=tokenizer,
            model_obj=model_obj,
            prompt=prompt,
            device=device,
            is_seq2seq=model_cls is AutoModelForSeq2SeqLM,
            torch=torch,
        )
    except Exception as error:
        return empty_llm_query_result(model=selected_model, error=str(error), provider="hf")
    finally:
        try:
            del model_obj
            del tokenizer
        except Exception:
            pass
        clear_hf_query_memory()

    queries = valid_retrieval_queries(parse_llm_retrieval_queries(raw_response), research_plan)
    raw_response_text = clean_text(raw_response)
    max_queries = max(1, len(questions) * max(1, max_variants))
    if not queries:
        return {
            "queries": [],
            "model": selected_model,
            "error": "HF query rewrite returned no usable queries",
            "raw_response": raw_response_text[:1000],
            "provider": "hf",
            "fallback_reason": clean_text(fallback_reason),
        }
    queries = complete_sub_question_query_coverage(queries, research_plan, max_variants=max_variants)
    return {
        "queries": dedupe_preserve_order(query[:700] for query in queries)[:max_queries],
        "model": selected_model,
        "error": "",
        "raw_response": raw_response_text[:1000],
        "provider": "hf",
        "fallback_reason": clean_text(fallback_reason),
    }


def hf_query_device(torch: Any) -> str:
    requested = clean_text(os.environ.get("RAG_SUBQUESTION_HF_DEVICE", "auto")).lower()
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hf_query_model_is_seq2seq(model_name: str) -> bool:
    lowered = clean_text(model_name).lower()
    return any(name in lowered for name in ("t5", "flan", "bart"))


def hf_generate_query_response(
    tokenizer: Any,
    model_obj: Any,
    prompt: str,
    device: str,
    is_seq2seq: bool,
    torch: Any,
) -> str:
    system_prompt = (
        "You rewrite planner sub-questions into broad, complementary RAG retrieval "
        "queries. Return JSON only."
    )
    if not is_seq2seq and hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt[:6000]}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = f"{system_prompt}\n\n{prompt[:6000]}\n\nJSON:"

    max_input_tokens = env_int("RAG_HF_QUERY_MAX_INPUT_TOKENS", DEFAULT_HF_QUERY_MAX_INPUT_TOKENS)
    max_new_tokens = env_int("RAG_HF_QUERY_MAX_NEW_TOKENS", DEFAULT_HF_QUERY_MAX_NEW_TOKENS)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generate_kwargs = {
        "max_new_tokens": max(64, max_new_tokens),
        "do_sample": False,
    }
    pad_token_id = hf_pad_token_id(tokenizer)
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = pad_token_id
    with torch.no_grad():
        outputs = model_obj.generate(**inputs, **generate_kwargs)
    if is_seq2seq:
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    input_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)


def hf_pad_token_id(tokenizer: Any) -> int | None:
    return getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def clear_hf_query_memory() -> None:
    try:
        import gc
        import torch
    except Exception:
        return
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except Exception:
        pass


def empty_llm_query_result(model: str = "", error: str = "", provider: str = "") -> dict[str, Any]:
    return {
        "queries": [],
        "model": clean_text(model),
        "error": clean_text(error),
        "raw_response": "",
        "provider": clean_text(provider),
    }


def complete_sub_question_query_coverage(
    queries: Sequence[str],
    research_plan: dict[str, Any],
    max_variants: int = DEFAULT_SUBQUESTION_QUERY_VARIANTS,
) -> list[str]:
    helpers = generation_helpers()
    questions = helpers.planner_sub_questions(research_plan)
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
    helpers = generation_helpers()
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
    helpers = generation_helpers()
    objective_terms = query_tokens(clean_text(research_plan.get("objective")))
    question_terms = set()
    for question in helpers.planner_sub_questions(research_plan):
        question_terms.update(query_tokens(question))
    allowed_terms = objective_terms | question_terms
    return [
        cleaned
        for query in dedupe_preserve_order(queries)
        if (cleaned := clean_retrieval_query(query))
        if is_valid_retrieval_query(cleaned, allowed_terms)
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
