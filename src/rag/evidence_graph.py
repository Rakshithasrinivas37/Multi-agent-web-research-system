"""Lightweight evidence graph helpers for GraphRAG-style chunk selection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from src.tools.text_utils import clean_text


ENTITY_STOPWORDS = {
    "attention",
    "abstract",
    "introduction",
    "section",
    "figure",
    "table",
    "method",
    "model",
    "models",
    "results",
    "paper",
    "source",
    "using",
    "used",
}

DOMAIN_ENTITY_PATTERNS = {
    "additive_attention": r"\b(?:additive|bahdanau)\b",
    "multiplicative_attention": r"\b(?:multiplicative|luong|dot[- ]?product|bilinear)\b",
    "self_attention": r"\bself[- ]attention\b",
    "multi_head_attention": r"\bmulti[- ]head\b|\bmultihead\b",
    "scaled_dot_product": r"\bscaled\s+dot[- ]product\b|\bsoftmax\b.+\bsqrt\b",
    "transformer": r"\btransformer|attention\s+is\s+all\s+you\s+need\b",
    "complexity": r"\b(?:complexity|quadratic|linear|o\(|memory|runtime)\b",
    "benchmark": r"\b(?:benchmark|bleu|accuracy|imagenet|cifar|score|performance|\d+(?:\.\d+)?\s*%)\b",
    "api": r"\b(?:api|class|function|method|signature|parameter|constructor)\b",
    "tensorflow": r"\b(?:tensorflow|keras|tf\.keras)\b",
    "pytorch": r"\b(?:pytorch|torch\.nn|torch\.)\b",
}


@dataclass(frozen=True)
class EvidenceGraph:
    """Small in-memory graph keyed by chunk id and extracted evidence entities."""

    chunk_entities: dict[str, set[str]]
    entity_chunks: dict[str, set[str]]
    source_chunks: dict[str, set[str]]
    chunk_by_id: dict[str, dict[str, Any]]


def build_evidence_graph(chunks: Sequence[dict[str, Any]]) -> EvidenceGraph:
    """Build a chunk/entity graph from retrieved chunks."""

    chunk_entities: dict[str, set[str]] = {}
    entity_chunks: dict[str, set[str]] = {}
    source_chunks: dict[str, set[str]] = {}
    chunk_by_id: dict[str, dict[str, Any]] = {}
    for fallback_index, chunk in enumerate(chunks or []):
        if not isinstance(chunk, dict):
            continue
        chunk_id = chunk_graph_id(chunk, fallback_index)
        chunk_by_id[chunk_id] = chunk
        entities = evidence_entities(chunk_text_for_graph(chunk))
        chunk_entities[chunk_id] = entities
        for entity in entities:
            entity_chunks.setdefault(entity, set()).add(chunk_id)
        source_key = chunk_source_key(chunk)
        if source_key:
            source_chunks.setdefault(source_key, set()).add(chunk_id)
    return EvidenceGraph(
        chunk_entities=chunk_entities,
        entity_chunks=entity_chunks,
        source_chunks=source_chunks,
        chunk_by_id=chunk_by_id,
    )


def expand_chunks_for_question(
    question: str,
    selected_chunks: Sequence[dict[str, Any]],
    candidate_chunks: Sequence[dict[str, Any]],
    max_chunks: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expand selected evidence with graph-neighbor chunks connected by entities."""

    selected = [chunk for chunk in selected_chunks or [] if isinstance(chunk, dict)]
    if len(selected) >= max(1, max_chunks):
        return selected[:max(1, max_chunks)], {"added": 0, "matched_entities": []}

    graph = build_evidence_graph(candidate_chunks)
    selected_ids = {chunk_graph_id(chunk, index) for index, chunk in enumerate(selected)}
    question_entities = evidence_entities(question)
    seed_entities = set(question_entities)
    for chunk_id in selected_ids:
        seed_entities.update(graph.chunk_entities.get(chunk_id, set()))

    scored: list[tuple[int, float, dict[str, Any], set[str]]] = []
    for fallback_index, chunk in enumerate(candidate_chunks or []):
        if not isinstance(chunk, dict):
            continue
        chunk_id = chunk_graph_id(chunk, fallback_index)
        if chunk_id in selected_ids:
            continue
        entities = graph.chunk_entities.get(chunk_id, set())
        overlap = seed_entities & entities
        if not overlap:
            continue
        score = len(overlap) * 4
        score += len(question_entities & entities) * 3
        if chunk.get("is_primary_source"):
            score += 2
        if chunk.get("has_formula_signal") or chunk.get("has_api_signal") or chunk.get("has_benchmark_signal"):
            score += 1
        scored.append((score, float(chunk.get("score") or 0.0), chunk, overlap))

    expanded = list(selected)
    matched_entities: list[str] = []
    for _, _, chunk, overlap in sorted(scored, key=lambda item: (-item[0], -item[1])):
        if len(expanded) >= max(1, max_chunks):
            break
        if chunk_duplicate(chunk, expanded):
            continue
        expanded.append(chunk)
        matched_entities.extend(sorted(overlap))
    return expanded, {
        "added": max(0, len(expanded) - len(selected)),
        "matched_entities": sorted(set(matched_entities))[:12],
        "candidate_count": len(scored),
    }


def evidence_graph_summary(chunks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    graph = build_evidence_graph(chunks)
    return {
        "chunk_count": len(graph.chunk_by_id),
        "entity_count": len(graph.entity_chunks),
        "edge_count": sum(max(0, len(chunk_ids) - 1) for chunk_ids in graph.entity_chunks.values()),
    }


def evidence_entities(text: Any) -> set[str]:
    """Extract stable entities useful for technical evidence linking."""

    value = clean_text(text)
    lowered = value.lower()
    entities: set[str] = set()
    for label, pattern in DOMAIN_ENTITY_PATTERNS.items():
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            entities.add(label)
    for match in re.finditer(r"\b(?:torch|tf|keras|nn)\.[A-Za-z_][A-Za-z0-9_.]*\b", value):
        entities.add(match.group(0).lower())
    for match in re.finditer(r"\b\d{4}\.\d{4,5}\b", value):
        entities.add(f"arxiv:{match.group(0)}")
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9]+(?:[- ][A-Z][A-Za-z0-9]+){0,3}\b", value):
        entity = clean_text(match.group(0)).lower().replace(" ", "_")
        if len(entity) >= 4 and entity not in ENTITY_STOPWORDS:
            entities.add(entity)
    return {entity for entity in entities if entity and entity not in ENTITY_STOPWORDS}


def chunk_text_for_graph(chunk: dict[str, Any]) -> str:
    return clean_text(
        " ".join(
            [
                clean_text(chunk.get("title")),
                clean_text(chunk.get("url")),
                clean_text(chunk.get("content")),
            ]
        )
    )


def chunk_graph_id(chunk: dict[str, Any], fallback_index: int = 0) -> str:
    return clean_text(chunk.get("id")) or clean_text(f"{chunk.get('source_index')}:{chunk.get('url')}:{fallback_index}")


def chunk_source_key(chunk: dict[str, Any]) -> str:
    return clean_text(chunk.get("url")).lower() or clean_text(chunk.get("source_index"))


def chunk_duplicate(chunk: dict[str, Any], selected: Sequence[dict[str, Any]]) -> bool:
    key = clean_text(f"{chunk.get('source_index')}:{chunk.get('id')}:{chunk.get('content')[:120]}").lower()
    for item in selected or []:
        item_key = clean_text(f"{item.get('source_index')}:{item.get('id')}:{clean_text(item.get('content'))[:120]}").lower()
        if key and key == item_key:
            return True
    return False
