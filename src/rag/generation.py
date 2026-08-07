"""Generate a RAG answer from already-retrieved context chunks."""

from __future__ import annotations

import os
import re
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
from src.tools.text_utils import clean_text


DEFAULT_RAG_GENERATION_MODEL = "llama-3.1-8b-instant"
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
MIN_EVIDENCE_CHARS = 120
MIN_EVIDENCE_TOKENS = 12
DEFAULT_OBJECTIVE_SCOPE_SIMILARITY = 0.40
DEFAULT_OBJECTIVE_SCOPE_MAX_KEYS = 6
URL_PATTERN = re.compile(r"https?://[^\s\])}>\"']+")
OBJECTIVE_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
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
) -> dict[str, Any]:
    """Retrieve with planner-derived queries, then synthesize report-ready notes."""
    objective = clean_text(research_plan.get("objective"))
    if not objective:
        raise ValueError("research_plan.objective is required")

    queries = planner_tasks_to_rag_queries(research_plan)
    current_history_key = clean_text(history_key) or objective_key(objective, research_plan)
    objective_scope = resolve_objective_history_scope(
        objective=objective,
        current_history_key=current_history_key,
        chroma_path=chroma_path,
        collection_name=collection_name,
    )
    allowed_history_keys = objective_scope["history_keys"]
    rewritten_query = rewrite_query_from_planner_queries(
        objective=objective,
        queries=queries,
        model=model,
    ) if rewrite_query else ""
    if print_rewritten_query and rewritten_query:
        print("Rewritten retrieval query:")
        print(rewritten_query)
    retrieval_queries = [rewritten_query] if rewritten_query else queries
    retrieved_context = multi_query_hybrid_retrieve(
        queries=retrieval_queries,
        chroma_path=chroma_path,
        collection_name=collection_name,
        top_k=top_k,
        per_query_k=per_query_k,
        semantic_k=semantic_k,
        bm25_k=bm25_k,
        history_keys=allowed_history_keys,
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
    source_coverage_results = []
    if include_planned_source_urls and planned_source_urls:
        source_coverage_results = source_url_coverage_retrieve(
            source_urls=missing_or_weak_source_urls(planned_source_urls, retrieved_context),
            query=retrieval_queries,
            chroma_path=chroma_path,
            collection_name=collection_name,
            history_keys=allowed_history_keys,
            top_k_per_url=source_url_k,
            scan_limit=bm25_scan_limit,
        )
        retrieved_context = merge_retrieved_context(retrieved_context, source_coverage_results)

    payload = synthesize_context_for_report(
        objective=objective,
        retrieved_context=retrieved_context,
        synthesis_instruction=clean_text(research_plan.get("synthesis_instruction")),
        planner_questions=planner_sub_questions(research_plan),
        model=model,
        max_context_chars=max_context_chars,
        max_tokens=max_tokens,
    )
    payload["queries"] = queries
    payload["rewritten_query"] = rewritten_query
    payload["retrieval_queries"] = retrieval_queries
    payload["history_key"] = current_history_key
    payload["allowed_history_keys"] = allowed_history_keys
    payload["objective_scope"] = objective_scope
    payload["planned_source_urls"] = planned_source_urls
    payload["source_coverage_count"] = len(source_coverage_results)
    payload["retrieved_count"] = len(retrieved_context)
    payload["citation_policy"] = (
        "Use only the numbered source indexes in sources. Prefer primary papers, "
        "official documentation, academic sources, and authoritative surveys for "
        "technical claims. Use secondary explainers only for intuition. If the "
        "synthesis marks a formula, number, API detail, or definition as missing "
        "evidence, do not add it to the report unless it appears in a supporting chunk."
    )
    citation_audit = audit_synthesis_citations(payload.get("synthesis"), payload.get("sources", []))
    payload["citation_audit"] = citation_audit
    payload["supporting_chunks"] = report_supporting_chunks(
        retrieved_context,
        payload.get("sources", []),
        max_chunks=supporting_chunk_count,
        max_chars=retrieved_chunk_chars,
        cited_source_indexes=citation_audit.get("valid_referenced_source_indexes", []),
    )
    payload["diagnostics"] = synthesis_diagnostics(payload, retrieved_context)
    if include_retrieved_chunks:
        payload["retrieved_chunks"] = compact_retrieved_chunks(
            retrieved_context,
            payload.get("sources", []),
            max_chars=retrieved_chunk_chars,
        )
    return payload


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
            details = matching_task_details(query, tasks)
            sub_question_queries.append(clean_text(f"{query} {details}"))

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
    stopwords = {"and", "are", "for", "from", "how", "the", "what", "with"}
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_+#.-]+", clean_text(text))
        if len(token) > 2 and token.lower() not in stopwords
    }


def planner_sub_questions(research_plan: dict[str, Any]) -> list[str]:
    """Return planner sub-questions without mixing in fallback task text."""

    return dedupe_preserve_order(
        clean_text(question)
        for question in research_plan.get("sub_questions", [])
        if clean_text(question)
    )


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

    response = Groq().chat.completions.create(
        model=clean_text(model or os.environ.get("RAG_GENERATION_MODEL")) or DEFAULT_RAG_GENERATION_MODEL,
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
    planner_questions: Sequence[str] | None = None,
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
        }
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set")

    try:
        from groq import Groq
    except ImportError as error:
        raise RuntimeError("groq package is not installed. Install it with `pip install -r requirements.txt`.") from error

    planner_question_text = format_planner_questions(planner_questions or [])
    instruction = clean_text(synthesis_instruction) or "Synthesize the retrieved evidence into report-ready research notes."
    client = Groq()
    last_error: Exception | None = None

    for attempt in range(3):
        context_budget = max(8000, int(max_context_chars * (0.70 ** attempt)))
        token_budget = max(700, int(max_tokens * (0.80 ** attempt)))
        context_text, sources = build_generation_context(retrieved_context, max_context_chars=context_budget)
        source_priority_guidance = build_source_priority_guidance(sources)
        prompt = f"""Research objective:
{objective}

Synthesis instruction:
{instruction}

Planner sub-questions to cover:
{planner_question_text}

Source priority guidance:
{source_priority_guidance}

Retrieved context from multiple sources:
{context_text}

Create a detailed report-agent-ready evidence package using only the retrieved context.
Do not write the final report. Prepare rich notes that another agent can turn into a technical report.

Return Markdown with these sections:
1. Coverage Map
   - For each planner sub-question, state whether the retrieved context has strong, partial, or missing evidence.
2. Section Notes By Planner Question
   - Repeat each planner sub-question as a subsection.
   - Include the direct answer, important evidence, equations/formulas/API details when available, and gaps.
   - Keep enough detail for a report agent to write a full section without needing to infer missing facts.
3. Cross-Source Synthesis
   - Connect repeated ideas across sources and identify how the sources complement each other.
4. Technical Details To Preserve
   - Preserve exact equations, definitions, model components, implementation details, and benchmark values only when present in the retrieved context.
5. Conflicts Or Gaps
   - List missing evidence, weak citations, source conflicts, or claims that need caution.
6. Recommended Report Structure
   - Suggest report sections and which source markers support each section.

Use only plain ASCII numbered source markers that appear in the retrieved context, exactly like [1], [2], [3].
Every evidence-backed claim must include at least one source marker.
For equations, formulas, API signatures, benchmark numbers, and historical attribution, cite original papers, official documentation, academic sources, or authoritative surveys first.
If a primary/official source and a secondary explainer both support the same technical claim, cite the primary/official source and omit the secondary citation.
Use secondary explainers only for intuition, examples, or background wording.
Do not compress important technical details into vague summaries.
Do not use Markdown tables.
Never use citation formats like 【1】, 【1†L1-L4】, footnotes, or URLs inline.
If a requested equation, number, API detail, or definition is not explicitly present in the retrieved context, mark it as missing evidence and tell the report agent not to add it.
Do not invent source names, authors, dates, titles, papers, benchmark numbers, equations, or citations that are not present in the retrieved context."""

        try:
            response = client.chat.completions.create(
                model=clean_text(model or os.environ.get("RAG_GENERATION_MODEL")) or DEFAULT_RAG_GENERATION_MODEL,
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


def is_request_too_large_error(error: Exception) -> bool:
    message = clean_text(error).lower()
    return (
        "request too large" in message
        or "tokens per minute" in message
        or "rate_limit_exceeded" in message
        or "tpm" in message
    )


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

    for retrieval_rank, result in enumerate(ordered_results, start=1):
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        url = primary_source_url(metadata)
        citation_index = len(blocks) + 1
        title = clean_text(metadata.get("title")) or url or f"Source {citation_index}"
        chunk = retrieved_chunk_preview(result.document, metadata, max_chars=block_char_limit)
        if not is_meaningful_evidence(chunk):
            continue

        block = f"[{citation_index}] {title}\nURL: {url}\n{chunk}"
        if used_chars + len(block) > max_context_chars:
            remaining = max_context_chars - used_chars
            if remaining < 500:
                break
            block = block[:remaining].rstrip()

        blocks.append(block)
        sources.append(
            {
                "index": citation_index,
                "retrieval_rank": retrieval_rank,
                "id": result.id,
                "url": url,
                "title": title,
                "score": result.score,
            }
        )
        used_chars += len(block)

    return "\n\n".join(blocks), sources


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
                "source_index": source_index_by_id.get(result.id),
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
        chunk_id = clean_text(source.get("id"))
        index = source.get("index")
        if not chunk_id or not isinstance(index, int):
            continue
        indexes[chunk_id] = index
    return indexes


def synthesis_diagnostics(payload: dict[str, Any], retrieved_context: Sequence[RetrievalResult]) -> dict[str, Any]:
    sources = payload.get("sources", [])
    supporting_chunks = payload.get("supporting_chunks", [])
    allowed_history_keys = payload.get("allowed_history_keys", [])
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
