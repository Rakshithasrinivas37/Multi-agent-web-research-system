"""Extract compact source-backed evidence spans from retrieved chunks."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Sequence

from src.tools.text_utils import clean_text


DEFAULT_EVIDENCE_PACK_CHUNK_CHARS = 900
DEFAULT_EVIDENCE_SPANS_PER_QUESTION = 8
DEFAULT_EVIDENCE_SPANS_PER_CHUNK = 2
DEFAULT_EVIDENCE_SPAN_CHARS = 700
DEFAULT_REPORT_EVIDENCE_SPANS_PER_QUESTION = 2

EVIDENCE_SIGNAL_PATTERNS = {
    "api": (
        (r"\b(?:api|class|function|method|parameter|argument|signature|constructor|official docs?|usage example)\b", 3),
        (r"\b[A-Za-z_][A-Za-z0-9_.]*\([^)]{1,120}\)", 4),
    ),
    "applications": ((r"\b(?:application|applications|use case|used for|task|translation|classification|vision|speech|recognition|nlp)\b", 3),),
    "benchmark": (
        (r"\b\d+(?:\.\d+)?\s*(?:%|bleu|rouge|f1|auc|accuracy|precision|recall|top[- ]?1|top[- ]?5|score|points?)\b", 6),
        (r"\b(?:wmt|glue|superglue|imagenet|cifar|squad|vtab|coco|benchmark|leaderboard|test set|validation set)\b", 3),
        (r"\b(?:achieves?|reports?|outperforms?|improv(?:e|es|ed|ing)|state-of-the-art|results?)\b", 2),
    ),
    "comparison": ((r"\b(?:compare|comparison|versus| vs |different|difference|whereas|while|trade[- ]?off)\b", 3),),
    "complexity": ((r"\b(?:o\([^)]+\)|quadratic|linear|sub-quadratic|runtime|memory|complexity|scalability)\b", 4),),
    "definition": ((r"\b(?:defined as|definition|refers to|means|is a|are a|purpose)\b", 3),),
    "equation": (
        (r"(?:\\(?:frac|sum|sqrt|operatorname)|[=∑Σ√αβγδθλµπ])", 4),
        (r"\b(?:softmax|sqrt|tanh|exp|log|equation|formula|formulation)\b", 3),
        (r"\b[A-Za-z_][A-Za-z0-9_]*\s*\([^)]{0,80}\)\s*=", 4),
    ),
    "examples": ((r"\b(?:example|case study|task|application|introduced|proposed|experiment)\b", 2),),
    "implementation": (
        (r"\b(?:implementation|api|class|function|method|parameter|argument|signature|constructor|usage example|official docs?)\b", 3),
        (r"\b[A-Za-z_][A-Za-z0-9_.]*\([^)]{1,120}\)", 4),
    ),
    "limitations": ((r"\b(?:limitation|limitations|challenge|drawback|constraint|bottleneck|weakness|open question)\b", 3),),
}


def evidence_spans_for_question(
    question: str,
    chunks: Sequence[dict[str, Any]],
    evidence_types: Sequence[str],
    max_spans: int = DEFAULT_EVIDENCE_SPANS_PER_QUESTION,
) -> list[dict[str, Any]]:
    spans = []
    question_terms = _query_tokens(question)
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for span in _evidence_spans_from_chunk(question, chunk, evidence_types):
            text = clean_text(span.get("text"))
            if not text:
                continue
            evidence_type = clean_text(span.get("evidence_type")) or "evidence"
            score = (
                len(question_terms & _query_tokens(text))
                + _evidence_type_score(text, [evidence_type]) * 3
                + _evidence_signal_score(text, [evidence_type]) * 4
                + (4 if chunk.get("is_primary_source") else 0)
                + (4 if evidence_type == "table" else 0)
            )
            spans.append(
                {
                    "source_index": chunk.get("source_index"),
                    "chunk_id": clean_text(chunk.get("id")),
                    "title": clean_text(chunk.get("title")),
                    "url": clean_text(chunk.get("url")),
                    "evidence_type": evidence_type,
                    "text": _compact_text(text, DEFAULT_EVIDENCE_SPAN_CHARS),
                    "score": score,
                }
            )
    ordered = sorted(spans, key=lambda item: (-float(item.get("score") or 0.0), item.get("source_index") or 10**6))
    return _unique_evidence_spans(span for span in ordered if span.get("source_index") is not None)[: max(1, max_spans)]


def supporting_chunks_from_evidence_spans(
    evidence_packs: Sequence[dict[str, Any]],
    max_spans_per_question: int = DEFAULT_REPORT_EVIDENCE_SPANS_PER_QUESTION,
) -> list[dict[str, Any]]:
    chunks = []
    for pack in evidence_packs or []:
        if not isinstance(pack, dict):
            continue
        question = clean_text(pack.get("question"))
        spans = [span for span in pack.get("evidence_spans", []) if isinstance(span, dict)]
        for span in spans[: max(1, max_spans_per_question)]:
            source_index = span.get("source_index")
            text = clean_text(span.get("text"))
            if not isinstance(source_index, int) or not text:
                continue
            evidence_type = clean_text(span.get("evidence_type")) or "evidence"
            span_id = hashlib.sha1(f"{question}|{source_index}|{evidence_type}|{text}".encode("utf-8")).hexdigest()[:16]
            chunks.append(
                {
                    "source_index": source_index,
                    "retrieval_rank": 0,
                    "id": f"evidence-span-{span_id}",
                    "url": clean_text(span.get("url")),
                    "title": clean_text(span.get("title")) or "Evidence span",
                    "score": span.get("score", 0.0),
                    "source_type": "",
                    "source_quality": "",
                    "is_primary_source": False,
                    "synthesis_question": question,
                    "chunk_kind": "evidence_span",
                    "content": clean_text(f"{question}\nEvidence type: {evidence_type}\n{text}"),
                }
            )
    return chunks


def merge_supporting_chunks(
    priority_chunks: Sequence[dict[str, Any]],
    fallback_chunks: Sequence[dict[str, Any]],
    max_chunks: int,
) -> list[dict[str, Any]]:
    merged = []
    seen = set()
    for chunk in [*list(priority_chunks or []), *list(fallback_chunks or [])]:
        if not isinstance(chunk, dict):
            continue
        key = clean_text(f"{chunk.get('source_index')}:{chunk.get('id')}:{chunk.get('content')}").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(chunk)
        if len(merged) >= max(1, max_chunks):
            break
    return merged


def _evidence_spans_from_chunk(
    question: str,
    chunk: dict[str, Any],
    evidence_types: Sequence[str],
) -> list[dict[str, str]]:
    content = clean_text(chunk.get("content"))
    if not content:
        return []
    if chunk.get("chunk_kind") == "table" or chunk.get("has_table_signal"):
        return [{"evidence_type": "table", "text": _compact_table_span(content)}]

    candidates = []
    query_terms = _query_tokens(question)
    wanted = _clean_string_list(evidence_types) or _infer_evidence_types(question)
    for segment in _evidence_segments(content):
        segment_terms = _query_tokens(segment)
        has_overlap = bool(query_terms & segment_terms)
        has_signal = bool(_evidence_type_score(segment, wanted) or _evidence_signal_score(segment, wanted))
        if query_terms and not has_overlap and not has_signal:
            continue
        evidence_type = _best_evidence_type(segment, wanted)
        if evidence_type:
            score = len(query_terms & segment_terms) + _evidence_type_score(segment, [evidence_type]) + _evidence_signal_score(segment, [evidence_type])
            candidates.append((score, {"evidence_type": evidence_type, "text": segment}))
    return [span for _, span in sorted(candidates, key=lambda item: item[0], reverse=True)][:DEFAULT_EVIDENCE_SPANS_PER_CHUNK]


def _evidence_segments(text: str) -> list[str]:
    lines = [clean_text(line) for line in str(text or "").splitlines() if clean_text(line)]
    sentences = [clean_text(sentence) for sentence in re.split(r"(?<=[.!?])\s+|\n+", clean_text(text)) if clean_text(sentence)]
    return _dedupe_preserve_order([*lines, *sentences])


def _best_evidence_type(text: str, evidence_types: Sequence[str]) -> str:
    scored = []
    for evidence_type in _clean_string_list(evidence_types):
        score = _evidence_type_score(text, [evidence_type]) + _evidence_signal_score(text, [evidence_type])
        if score:
            scored.append((score, evidence_type))
    return sorted(scored, key=lambda item: item[0], reverse=True)[0][1] if scored else ("evidence" if clean_text(text) else "")


def _compact_table_span(content: str) -> str:
    text = clean_text(content)
    marker = "Table data JSON:"
    if marker not in text:
        return text
    try:
        payload = json.loads(text.split(marker, 1)[1].strip())
    except json.JSONDecodeError:
        return text
    headers = ", ".join(_clean_string_list(payload.get("headers"))[:12])
    records = payload.get("records") if isinstance(payload.get("records"), list) else []
    rows = []
    for record in records[:6]:
        if isinstance(record, dict):
            rows.append("; ".join(f"{clean_text(key)}={clean_text(value)}" for key, value in record.items() if clean_text(value)))
    return clean_text(f"Table columns: {headers}. Rows: {' | '.join(rows)}") or text


def _unique_evidence_spans(spans: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = []
    seen = set()
    for span in spans:
        key = clean_text(f"{span.get('source_index')}:{span.get('evidence_type')}:{span.get('text')}").lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(span)
    return unique


def _infer_evidence_types(question: str) -> list[str]:
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
    return [name for name, pattern in checks if re.search(pattern, lowered)] or ["evidence"]


def _evidence_type_score(text: str, evidence_types: Sequence[str]) -> int:
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
        "examples": r"\b(example|case|task|paper|introduced|proposed|contribution|experiment|application)\b",
        "implementation": r"\b(implementation|api|class|function|method|parameter|argument|signature|constructor|usage example|official docs?)\b",
        "limitations": r"\b(limitation|challenge|drawback|constraint|bottleneck|weakness|open question)\b",
        "table": r"\b(table|row|column|dataset|result|metric)\b",
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


def _evidence_signal_score(text: str, evidence_types: Sequence[str]) -> int:
    lowered = clean_text(text).lower()
    score = 0
    for evidence_type in _clean_string_list(evidence_types):
        for pattern, weight in EVIDENCE_SIGNAL_PATTERNS.get(evidence_type.lower(), ()):
            if re.search(pattern, lowered):
                score += weight
    return score


def _query_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_]+", clean_text(text).lower()) if len(token) > 2}


def _clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [clean_text(item) for item in value if clean_text(item)]


def _dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    deduped = []
    seen = set()
    for item in items:
        key = clean_text(item).lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(clean_text(item))
    return deduped


def _compact_text(value: Any, max_chars: int) -> str:
    text = clean_text(value)
    return text if len(text) <= max_chars else text[:max_chars].rstrip()
