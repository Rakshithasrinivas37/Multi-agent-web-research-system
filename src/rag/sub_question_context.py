"""Query rewriting and per-sub-question RAG retrieval helpers."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from src.rag.indexing import get_collection
from src.rag.retrieval import RetrievalResult, expand_parent_context_results, multi_query_hybrid_retrieve
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
DEFAULT_HF_QUERY_MAX_INPUT_TOKENS = 4096
DEFAULT_HF_QUERY_MAX_NEW_TOKENS = 900


def generation_helpers() -> Any:
    """Import generation lazily to avoid a heavy circular module split."""

    from src.rag import generation

    return generation


def clean_model_name(value: Any) -> str:
    """Normalize env-provided model names without changing valid ids."""

    return clean_text(value).strip("\"'“”‘’")


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
    """Rewrite one planner sub-question into broad retrieval intents."""

    helpers = generation_helpers()
    question = clean_text(question)
    objective = clean_text(objective)
    task_details = clean_text(task_details)
    if not question:
        return []

    source_terms = helpers.source_query_terms(task_details)
    topic = helpers.retrieval_topic_phrase(f"{question} {task_details}", limit=12)
    key_terms = " ".join(helpers.query_keywords(question, limit=10)) or topic or objective
    hints = " ".join(helpers.broad_query_hints(f"{question} {task_details}"))
    variants = [
        clean_text(f"What source-backed context explains {topic} {hints}?"),
        clean_text(f"Which evidence gives {key_terms} details, examples, equations, benchmarks, or limitations?"),
        clean_text(f"Where do authoritative sources discuss {objective} {topic} {source_terms} source sections evidence?"),
    ]
    return helpers.dedupe_preserve_order(variant[:700] for variant in variants)[: max(1, max_variants)]


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
) -> list[dict[str, Any]]:
    """Retrieve an inspectable chunk group for each planner sub-question."""

    helpers = generation_helpers()
    tasks = [task for task in research_plan.get("tasks", []) if isinstance(task, dict)]
    candidate_chunks = max(1, candidate_chunks)
    final_chunks = max(1, min(final_chunks, candidate_chunks))
    clean_questions = helpers.dedupe_preserve_order(questions)
    question_source_urls = helpers.planner_question_source_urls(research_plan)

    def retrieve_question(question: str) -> dict[str, Any]:
        query_set = per_question_context_queries(question, objective=objective, tasks=tasks)
        if not query_set:
            return sub_question_context_group(question, [], [], [])
        fallback_used = False
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
            fallback_candidates = collection_scan_question_retrieve(
                question=question,
                queries=query_set,
                chroma_path=chroma_path,
                collection_name=collection_name,
                history_keys=history_keys,
                top_k=candidate_chunks,
                scan_limit=bm25_scan_limit,
                question_source_urls=question_source_urls,
            )
            fallback_used = bool(fallback_candidates)
            candidates = helpers.merge_retrieved_context(candidates, fallback_candidates)
        selected = helpers.meaningful_retrieval_results(candidates) or list(candidates)
        tagged = [tag_result_for_question(result, question) for result in selected[:final_chunks]]
        return sub_question_context_group(question, query_set, candidates, tagged, fallback_used=fallback_used)

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
) -> dict[str, Any]:
    return {
        "question": clean_text(question),
        "queries": list(queries),
        "candidate_count": len(candidates),
        "chunk_count": len(chunks),
        "fallback_used": bool(fallback_used),
        "chunks": list(chunks),
    }


def collection_scan_question_retrieve(
    question: str,
    queries: Sequence[str],
    chroma_path: str,
    collection_name: str,
    history_keys: Sequence[str] | None,
    top_k: int,
    scan_limit: int,
    question_source_urls: dict[str, list[str]] | None = None,
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
    allowed_history_keys = set(helpers.clean_string_list(history_keys or []))
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
    )
    selected = ranked[: max(1, top_k)]
    return expand_parent_context_results(selected, chroma_path=chroma_path)


def rank_collection_scan_results(
    question: str,
    queries: Sequence[str],
    candidates: Sequence[RetrievalResult],
    question_source_urls: dict[str, list[str]] | None = None,
) -> list[RetrievalResult]:
    """Rank direct collection rows for one planner question."""

    helpers = generation_helpers()
    query_text = clean_text(" ".join([question, *queries]))
    query_terms = helpers.query_tokens(query_text)
    topic_terms = {term for term in query_terms if term not in helpers.COVERAGE_GENERIC_TERMS}
    evidence_types = helpers.infer_question_evidence_types(question)
    source_urls = helpers.question_source_urls_for(question, question_source_urls)
    scored = []
    fallback = []

    for position, result in enumerate(candidates):
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        chunk = helpers.retrieved_chunk_preview(result.document, metadata, max_chars=DEFAULT_CONTEXT_BLOCK_CHARS * 2)
        if not helpers.is_meaningful_evidence(chunk):
            continue
        fallback.append(result)
        text = helpers.retrieval_result_text(result)
        text_terms = helpers.query_tokens(text)
        overlap = len(query_terms & text_terms)
        topic_overlap = len(topic_terms & text_terms)
        evidence_score = helpers.evidence_type_score(text, evidence_types)
        preferred_source = helpers.result_matches_source_urls(result, source_urls)
        if not (overlap or topic_overlap or evidence_score or preferred_source):
            continue
        score = (
            (topic_overlap * 4.0)
            + (overlap * 1.5)
            + (evidence_score * 3.0)
            + (8.0 if preferred_source else 0.0)
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
    task_details = helpers.matching_task_details(question, tasks)
    queries = sub_question_retrieval_queries(question, objective=objective, task_details=task_details)
    exact_queries = [
        clean_text(f"{objective} {question} {suffix} {task_details}")[:700]
        for suffix in helpers.precision_query_suffixes(question)
    ]
    return helpers.dedupe_preserve_order([*queries, *exact_queries])


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
                        "You rewrite planner sub-questions into broad, complementary RAG retrieval "
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
        "queries": helpers.dedupe_preserve_order(query[:700] for query in queries)[:max_queries],
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
        "queries": helpers.dedupe_preserve_order(query[:700] for query in queries)[:max_queries],
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
        return helpers.dedupe_preserve_order(queries)

    objective = clean_text(research_plan.get("objective"))
    tasks = [task for task in research_plan.get("tasks", []) if isinstance(task, dict)]
    clean_queries = helpers.dedupe_preserve_order(queries)
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
            task_details=helpers.matching_task_details(question, tasks),
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
    return helpers.dedupe_preserve_order(grouped)[:max_queries]


def query_list_covers_sub_question(question: str, queries: Sequence[str]) -> bool:
    return any(query_matches_sub_question(question, query) for query in queries)


def query_matches_sub_question(question: str, query: str) -> bool:
    helpers = generation_helpers()
    query_terms = helpers.query_tokens(query)
    named_terms = question_named_terms(question)
    if named_terms and not any(term_matches_query(term, query_terms, query) for term in named_terms):
        return False

    ignored_terms = helpers.COVERAGE_GENERIC_TERMS | helpers.COVERAGE_EVIDENCE_TERMS | {"attention", "method", "methods", "topic", "topics"}
    topic_terms = [term for term in helpers.query_keywords(question, limit=8) if term.lower() not in ignored_terms]
    normalized = {term.lower().replace("‑", "-").replace("–", "-") for term in topic_terms}
    evidence_terms = {
        term.lower()
        for term in helpers.query_keywords(question, limit=12)
        if term.lower() in helpers.COVERAGE_EVIDENCE_TERMS
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
    helpers = generation_helpers()
    normalized = clean_text(question).replace("‑", "-").replace("–", "-").replace("—", "-")
    terms = re.findall(r"\b[A-Z][A-Za-z0-9_.-]{2,}\b|\b[A-Z]{2,}\b", normalized)
    terms.extend(re.findall(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b", normalized))
    return helpers.dedupe_preserve_order(term.lower() for term in terms if term.lower() not in helpers.OBJECTIVE_STOPWORDS)


def format_sub_question_rewrite_items(questions: Sequence[str], tasks: Sequence[dict[str, Any]]) -> str:
    helpers = generation_helpers()
    lines = []
    for index, question in enumerate(questions, start=1):
        details = helpers.matching_task_details(question, tasks)
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

    helpers = generation_helpers()
    queries = []
    for match in re.finditer(r'"queries"\s*:\s*\[(.*?)\]', str(text or ""), flags=re.DOTALL | re.IGNORECASE):
        queries.extend(re.findall(r'"([^"]+)"', match.group(1)))
    return helpers.dedupe_preserve_order(queries)


def extract_query_strings(value: Any) -> list[str]:
    helpers = generation_helpers()
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
        return helpers.dedupe_preserve_order(queries)
    if isinstance(value, list):
        queries = []
        for item in value:
            queries.extend(extract_query_strings(item))
        return helpers.dedupe_preserve_order(queries)
    if isinstance(value, str):
        return fallback_line_queries(value)
    return []


def fallback_line_queries(text: str) -> list[str]:
    helpers = generation_helpers()
    lines = []
    for line in str(text or "").splitlines():
        line = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip().strip('"')
        if clean_text(line):
            lines.append(line)
    return helpers.dedupe_preserve_order(lines)


def valid_retrieval_queries(queries: Sequence[str], research_plan: dict[str, Any]) -> list[str]:
    helpers = generation_helpers()
    objective_terms = helpers.query_tokens(clean_text(research_plan.get("objective")))
    question_terms = set()
    for question in helpers.planner_sub_questions(research_plan):
        question_terms.update(helpers.query_tokens(question))
    allowed_terms = objective_terms | question_terms
    return [
        query
        for query in helpers.dedupe_preserve_order(queries)
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
