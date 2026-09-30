"""Lightweight GraphRAG expansion over retrieved vector/BM25 candidates.

This module intentionally builds an in-memory graph from the candidate chunks
already returned by vector/BM25 retrieval.  It does not introduce a separate
database or indexing job; instead it combines vector search with graph-style
entity neighborhood promotion during each synthesis run.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any, Sequence

from src.rag.retrieval import RetrievalResult
from src.tools.text_utils import clean_text


DEFAULT_GRAPHRAG_ENABLED = "1"
DEFAULT_GRAPHRAG_ENTITY_LIMIT = 12
DEFAULT_GRAPHRAG_TOP_K = 4

ENTITY_STOPWORDS = {
    "abstract", "attention", "figure", "introduction", "learning", "model", "models",
    "neural", "paper", "section", "source", "table", "the", "this", "using",
}

TECHNICAL_PHRASES = (
    "self-attention",
    "multi-head attention",
    "scaled dot-product attention",
    "dot-product attention",
    "additive attention",
    "encoder-decoder attention",
    "vision transformer",
    "transformer",
    "bahdanau",
    "luong",
    "bleu",
    "wmt 2014",
    "query",
    "queries",
    "keys",
    "values",
)


def graphrag_enabled() -> bool:
    return clean_text(os.environ.get("RAG_GRAPHRAG_ENABLED", DEFAULT_GRAPHRAG_ENABLED)).lower() not in {"0", "false", "no", "off"}


def graph_expand_question_results(
    question: str,
    candidates: Sequence[RetrievalResult],
    seed_results: Sequence[RetrievalResult],
    final_chunks: int,
) -> dict[str, Any]:
    """Blend vector seeds with entity-neighborhood results from the candidate graph."""

    if not graphrag_enabled() or not candidates or final_chunks <= 0:
        return empty_graph_result(seed_results, enabled=graphrag_enabled())

    graph = build_candidate_entity_graph(candidates)
    seed_ids = {result.id for result in seed_results}
    question_entities = extract_graph_entities(question)
    scored: list[tuple[float, int, RetrievalResult]] = []

    for position, result in enumerate(candidates):
        entities = graph["entities_by_id"].get(result.id, set())
        if not entities:
            continue
        question_overlap = len(entities & question_entities)
        seed_overlap = sum(len(entities & graph["entities_by_id"].get(seed.id, set())) for seed in seed_results)
        entity_frequency = sum(graph["entity_counts"].get(entity, 0) for entity in entities)
        if result.id in seed_ids:
            graph_score = 4.0 + question_overlap * 3.0 + seed_overlap * 0.25
        else:
            graph_score = question_overlap * 3.0 + seed_overlap * 1.25 + min(entity_frequency, 8) * 0.15
        if graph_score <= 0:
            continue
        scored.append((graph_score, -position, tag_graph_result(result, entities, graph_score)))

    graph_ranked = [result for _, _, result in sorted(scored, reverse=True)]
    blended = blend_graph_and_vector_results(seed_results, graph_ranked, final_chunks)
    graph_added = [result for result in blended if result.id not in seed_ids]
    used_entities = sorted({entity for result in blended for entity in graph["entities_by_id"].get(result.id, set())})[:DEFAULT_GRAPHRAG_ENTITY_LIMIT]
    return {
        "enabled": True,
        "mode": "graphrag_vector",
        "results": blended,
        "candidate_count": len(candidates),
        "seed_count": len(seed_results),
        "graph_added_count": len(graph_added),
        "entity_count": len(graph["entity_counts"]),
        "entities": used_entities,
    }


def empty_graph_result(seed_results: Sequence[RetrievalResult], enabled: bool = False) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "mode": "vector",
        "results": list(seed_results),
        "candidate_count": 0,
        "seed_count": len(seed_results),
        "graph_added_count": 0,
        "entity_count": 0,
        "entities": [],
    }


def build_candidate_entity_graph(candidates: Sequence[RetrievalResult]) -> dict[str, Any]:
    entities_by_id: dict[str, set[str]] = {}
    entity_to_ids: dict[str, set[str]] = defaultdict(set)
    for result in candidates:
        text = " ".join([
            clean_text(result.metadata.get("title")) if isinstance(result.metadata, dict) else "",
            clean_text(result.metadata.get("url")) if isinstance(result.metadata, dict) else "",
            clean_text(result.document),
        ])
        entities = set(extract_graph_entities(text))
        entities_by_id[result.id] = entities
        for entity in entities:
            entity_to_ids[entity].add(result.id)
    return {
        "entities_by_id": entities_by_id,
        "entity_to_ids": entity_to_ids,
        "entity_counts": Counter({entity: len(ids) for entity, ids in entity_to_ids.items()}),
    }


def extract_graph_entities(text: Any, limit: int = DEFAULT_GRAPHRAG_ENTITY_LIMIT) -> set[str]:
    value = clean_text(text)
    lowered = value.lower()
    entities: list[str] = []
    for phrase in TECHNICAL_PHRASES:
        if phrase in lowered:
            entities.append(phrase)
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:[- ][A-Z]?[A-Za-z0-9]+){0,4}\b|\b[A-Z]{2,}\b", value):
        entity = clean_text(match.group(0)).lower()
        if len(entity) < 3 or entity in ENTITY_STOPWORDS:
            continue
        if any(word in ENTITY_STOPWORDS for word in entity.split()):
            continue
        entities.append(entity)
    return set(entities[:limit])


def tag_graph_result(result: RetrievalResult, entities: set[str], graph_score: float) -> RetrievalResult:
    metadata = dict(result.metadata or {})
    metadata["retrieval_mode"] = "graphrag_vector"
    metadata["graphrag_entities"] = sorted(entities)[:DEFAULT_GRAPHRAG_ENTITY_LIMIT]
    metadata["graphrag_score"] = round(graph_score, 4)
    return replace(result, metadata=metadata, score=result.score + graph_score)


def blend_graph_and_vector_results(
    seed_results: Sequence[RetrievalResult],
    graph_ranked: Sequence[RetrievalResult],
    final_chunks: int,
) -> list[RetrievalResult]:
    selected: list[RetrievalResult] = []
    seen = set()
    seed_keep = max(1, final_chunks - min(DEFAULT_GRAPHRAG_TOP_K, max(1, final_chunks // 2)))
    for result in seed_results[:seed_keep]:
        if result.id not in seen:
            selected.append(result)
            seen.add(result.id)
    for result in graph_ranked:
        if len(selected) >= final_chunks:
            break
        if result.id not in seen:
            selected.append(result)
            seen.add(result.id)
    for result in seed_results:
        if len(selected) >= final_chunks:
            break
        if result.id not in seen:
            selected.append(result)
            seen.add(result.id)
    return selected[:final_chunks]
