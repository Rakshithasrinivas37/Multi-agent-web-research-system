"""Shared query and coverage helpers for RAG generation/retrieval."""

from __future__ import annotations

import re
from typing import Any, Sequence

from src.tools.text_utils import clean_text


URL_PATTERN = re.compile(r"https?://[^\s\])}>\"']+")
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
    "cost",
    "costs",
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
    "examples": ["examples", "cases", "tasks"],
    "implementation": ["implementation", "api signature", "parameters", "usage example"],
    "limitations": ["limitations", "challenges", "bottlenecks", "open questions"],
}
COVERAGE_FACET_STOPWORDS = QUERY_FILLER_TERMS | COVERAGE_GENERIC_TERMS | COVERAGE_EVIDENCE_TERMS | {
    "academic",
    "authoritative",
    "background",
    "common",
    "concept",
    "concepts",
    "detail",
    "details",
    "example",
    "examples",
    "official",
    "paper",
    "papers",
    "primary",
    "source",
    "sources",
    "topic",
    "topics",
    "trade",
    "off",
    "have",
    "has",
    "key",
}


def clean_model_name(value: Any) -> str:
    """Normalize env-provided model names without changing valid ids."""

    return clean_text(value).strip("\"'“”‘’")


def clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [clean_text(item) for item in value if clean_text(item)]


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
    return [name for name, pattern in checks if re.search(pattern, lowered)] or ["evidence"]


def coverage_question_tokens(text: str) -> set[str]:
    return {
        token
        for token in query_tokens(text)
        if token not in COVERAGE_GENERIC_TERMS
    }


def question_required_facets(question: str) -> list[str]:
    """Extract named/listed facets that a compound planner question expects."""

    text = clean_text(question).replace("‑", "-").replace("–", "-").replace("—", "-")
    candidates: list[str] = []

    def add(raw: str) -> None:
        candidates.extend(split_facet_candidates(raw))

    for match in re.findall(r"\(([^)]{2,160})\)", text):
        add(match)
    for pattern in (
        r"\b(?:e\.g\.|eg|including|includes?|such as|like)\s+([^?.;:()]+)",
        r"\b(?:between|among|across|beyond)\s+([^?.;:()]+)",
        r"\b(?:compared\s+(?:with|to)|versus|vs\.?)\s+([^?.;:()]+)",
        r"\b(?:in|for)\s+([A-Z][A-Za-z0-9_.-]*(?:\s+(?:and|or|/)\s+[A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+){0,2})+)",
    ):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            add(match.group(1))
    for match in re.finditer(r"\b(?:[A-Z]{2,}[A-Za-z0-9_.-]*|[A-Z][a-z]+(?:[A-Z][A-Za-z0-9_.-]+)+)\b", text):
        candidates.append(match.group(0))
    for match in re.finditer(
        r"\b([A-Za-z0-9][A-Za-z0-9_.+-]*(?:\s+[A-Za-z0-9][A-Za-z0-9_.+-]*){0,2})\s+"
        r"(attention|api|dataset|benchmark|model|architecture|framework|library|method|task|application)\b",
        text,
        flags=re.IGNORECASE,
    ):
        modifier = match.group(1)
        head = match.group(2)
        if re.search(r"\b(?:and|or)\b", modifier, flags=re.IGNORECASE):
            candidates.extend(f"{part} {head}" for part in re.split(r"\b(?:and|or)\b", modifier, flags=re.IGNORECASE))
        else:
            candidates.append(f"{modifier} {head}")

    return dedupe_preserve_order(
        facet for facet in (normalize_facet(candidate) for candidate in candidates)
        if facet
    )


def split_facet_candidates(text: str) -> list[str]:
    cleaned = clean_text(text)
    cleaned = re.sub(r"\b(?:and how|and what|where|why|when)\b.*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:e\.g\.|eg|i\.e\.|ie)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:and|or|versus|vs\.?)\b|/", ",", cleaned, flags=re.IGNORECASE)
    return [
        normalized
        for part in cleaned.split(",")
        if (normalized := normalize_facet(part))
    ]


def normalize_facet(value: Any) -> str:
    text = clean_text(value).strip(" .,:;()[]{}")
    text = text.replace("‑", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"^(?:the|a|an|of|to|in|for|with|without)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:introduced|trade[- ]?offs?|trade\s+off|proposes?|proposed|compares?|compared|addresses?|addressed|"
        r"works?|reduces?|improves?|uses?|reports?|achieves?|provides?|describes?)\b.*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    tokens = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.+#/-]*", text):
        cleaned = token.strip("._/#-")
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in COVERAGE_FACET_STOPWORDS:
            continue
        tokens.append(cleaned)
    if tokens == ["attention"]:
        return ""
    if len(tokens) == 1 and (len(tokens[0]) < 3 or tokens[0].lower() in COVERAGE_FACET_STOPWORDS):
        return ""
    return " ".join(tokens)[:90]


def facet_present(facet: str, text: str) -> bool:
    normalized = normalize_facet(facet)
    if not normalized:
        return False
    lowered_text = clean_text(text).replace("‑", "-").replace("–", "-").replace("—", "-").lower()
    lowered_facet = normalized.lower()
    if lowered_facet in lowered_text:
        return True
    facet_tokens = coverage_question_tokens(lowered_facet)
    text_tokens = coverage_question_tokens(lowered_text)
    return bool(facet_tokens and facet_tokens <= text_tokens)


def facet_evidence_window(facet: str, text: str) -> str:
    normalized = normalize_facet(facet).lower()
    if not normalized:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", clean_text(text))
    windows = [
        sentence
        for sentence in sentences
        if facet_present(normalized, sentence)
    ]
    return " ".join(windows) or text
