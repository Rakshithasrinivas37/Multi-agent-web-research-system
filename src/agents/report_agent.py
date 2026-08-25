"""Small report agent for turning synthesis evidence into a cited report."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Sequence

from src.memory.shared_memory import SharedMemory
from src.tools.groq_retry import create_chat_completion_with_retries
from src.tools.progress import emit_progress
from src.tools.text_utils import clean_text


DEFAULT_REPORT_AGENT_MODEL = "llama-3.1-8b-instant"
DEFAULT_REPORT_MAX_TOKENS = 4000
DEFAULT_REPORT_PROMPT_CHARS = 12000
DEFAULT_REPORT_TOTAL_TOKEN_BUDGET = 10000
DEFAULT_REPORT_OUTPUT_DIR = "data/reports"
DEFAULT_EVIDENCE_CHARS = 3000
DEFAULT_SYNTHESIS_CHARS = 1600
DEFAULT_EVIDENCE_PACK_CHARS = 1000
DEFAULT_COVERAGE_CHARS = 1200
DEFAULT_SOURCE_CHARS = 1400
DEFAULT_CHUNK_CHARS = 420

REPORT_SYSTEM_PROMPT = (
    "You write concise, well-structured, cited reports from supplied evidence only. "
    "Do not use outside knowledge. If evidence is missing, state the gap instead of answering from memory."
)

REPORT_PROMPT_RULES = """Grounding requirement (strict - read this first):
- Use ONLY the supplied sources, evidence packs, supporting evidence, and synthesis notes.
- Do not use any fact, figure, date, name, definition, or background knowledge from your own training.
- If retrieved context is silent on a topic, state the gap instead of filling it from general knowledge.
- If a sentence has no source support, delete it or make it an explicit limitation.

Required report schema:
1. Executive Summary
2. Introduction and Context
3. One main section per planner sub-question topic, in order
4. Cross-cutting Analysis and Synthesis
5. Limitations and Open Questions
6. Conclusion
7. References

Coverage requirement (mandatory):
- Every planner sub-question must map to exactly one topic section under heading 3.
- Treat synthesis coverage as the coverage contract. If a question is marked missing, write a short evidence-gap subsection instead of inventing an answer.
- Treat per-question evidence packs as the strongest topic-by-topic evidence map. Covered packs must be explained in the matching section; partial packs must include caveats.
- For missing coverage items, do not include formulas, API names, benchmark values, examples, or detailed explanations.
- Each topic section must directly answer its sub-question using only retrieved context.
- For questions asking for a definition, equation, components, complexity, API, or benchmark metric, include a clearly labeled "Core equation", "Core formula", "API", or "Metric evidence" line only when that detail appears in evidence.
- When multiple equivalent equations appear in evidence, show the most general/source-backed equation first.

Evidence and citation rules:
- Cite every factual claim inline with a real available source marker like [1].
- Never state a number, date, name, or quote that does not appear in the supplied context.
- If sources conflict, present both with citations.
- End with ## References, listing only sources actually cited."""

EVIDENCE_SNIPPET_SIGNALS = ["definition", "equation", "formula", "benchmark", "score", "result", "complexity", "api", "limitation", "challenge"]

STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "by", "can", "do", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "their", "to", "what", "when",
    "where", "which", "with",
}


class ReportAgent:
    """Generate a final report from synthesis-agent context."""

    def __init__(self, model: str | None = None) -> None:
        self.model = (
            clean_text(model)
            or clean_text(os.environ.get("RESEARCH_PLANNER_MODEL"))
            or clean_text(os.environ.get("RAG_GENERATION_MODEL"))
            or DEFAULT_REPORT_AGENT_MODEL
        )

    def generate(self, report_context: dict[str, Any], output_format: str = "report") -> dict[str, Any]:
        if not isinstance(report_context, dict) or not report_context:
            raise ValueError("report_context is required")
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set")

        try:
            from groq import Groq
        except ImportError as error:
            raise RuntimeError("groq package is not installed. Install it with `pip install -r requirements.txt`.") from error

        objective = clean_text(report_context.get("objective"))
        synthesis = clean_markdown(normalize_citation_markers(report_context.get("synthesis")))
        if not objective:
            raise ValueError("report_context.objective is required")
        if not synthesis:
            raise ValueError("report_context.synthesis is required")

        planner_questions = [clean_text(q) for q in report_context.get("planner_questions", []) if clean_text(q)]
        evidence_packs = report_context.get("evidence_packs", [])
        coverage_questions = dedupe_text([*planner_questions, *evidence_pack_questions(evidence_packs)])
        sources = sources_with_browser_results(report_context.get("sources", []), report_context.get("browser_results", []))
        evidence = format_supporting_evidence(report_context, sources=sources)
        pack_text = format_evidence_packs(evidence_packs)
        sources = evidence_backed_sources(sources, evidence, synthesis, pack_text)
        prompt = build_report_prompt(
            objective=objective,
            output_format=output_format,
            planner_questions=coverage_questions,
            synthesis=synthesis,
            evidence=evidence,
            sources=sources,
            citation_policy=clean_text(report_context.get("citation_policy")),
            coverage_by_question=report_context.get("coverage_by_question", []),
            evidence_packs=evidence_packs,
        )

        emit_progress(
            "tool_called",
            "Report agent calling Groq to generate final report",
            agent="report",
            tool="groq",
            metadata={"model": self.model},
        )
        report, model = generate_single_report(Groq(), self.model, prompt)
        report = normalize_final_report(report, sources)
        synthesis_gaps = synthesis_coverage_gap_questions(report_context, coverage_questions)
        coverage = report_sub_question_coverage_check(report, coverage_questions)
        schema_issues = report_schema_issues(report, coverage_questions)
        report_issues = report_quality_issues(report, sources, evidence_text=f"{evidence}\n{synthesis}\n{pack_text}")
        review = report_self_critique(report_issues, coverage, schema_issues)

        return {
            "objective": objective,
            "output_format": clean_text(output_format) or "report",
            "report": report,
            "sources": sources,
            "model": model,
            "diagnostics": {
                "source_count": len(sources),
                "evidence_pack_count": len(evidence_packs) if isinstance(evidence_packs, list) else 0,
                "supporting_chunk_count": len(report_context.get("supporting_chunks", []) or []),
                "retrieved_chunk_count": len(report_context.get("retrieved_chunks", []) or []),
                "report_length": len(report),
                "report_generation_mode": "single",
                "report_issues": report_issues,
                "report_schema_issues": schema_issues,
                "report_missing_sub_questions": coverage["missing"],
                "report_evidence_gap_questions": synthesis_gaps,
                "report_coverage_check": coverage,
                "report_retry_queries": rewrite_missing_sub_question_queries(objective, dedupe_text([*coverage["missing"], *synthesis_gaps])),
                "report_review_trace": [review],
                "report_token_budget": DEFAULT_REPORT_TOTAL_TOKEN_BUDGET,
                "report_estimated_token_cap": report_generation_token_cap(),
            },
        }

    def write_to_memory(
        self,
        report_payload: dict[str, Any],
        memory_path: str = "data/shared_memory.json",
        report_path: str | None = None,
    ) -> None:
        saved_path = write_report_file(report_payload, memory_path, report_path)
        SharedMemory(memory_path).write_agent_output("report", {"final_report": {**report_payload, "report_path": saved_path}})


def build_report_prompt(
    objective: str,
    output_format: str,
    planner_questions: Sequence[str],
    synthesis: str,
    evidence: str,
    sources: Sequence[dict[str, Any]],
    citation_policy: str = "",
    coverage_by_question: Sequence[dict[str, Any]] | None = None,
    evidence_packs: Sequence[dict[str, Any]] | None = None,
) -> str:
    prompt = f"""Research objective:
{objective}

Requested output format:
{clean_text(output_format) or "report"}

Citation policy:
{citation_policy or "Use only numbered source markers from the available sources."}

{REPORT_PROMPT_RULES}

Write the final Markdown report from the evidence below. Explain each supported topic in clear prose before equations, tables, APIs, or technical details.

Planner sub-questions to cover:
{format_planner_questions(planner_questions)}

Suggested topic headings:
{format_report_section_outline(planner_questions)}

Available sources:
{compact_text(format_sources(sources), DEFAULT_SOURCE_CHARS)}

Supporting evidence:
{compact_text(evidence, DEFAULT_EVIDENCE_CHARS)}

Synthesis notes:
{compact_text(synthesis, DEFAULT_SYNTHESIS_CHARS)}

Synthesis coverage by planner question:
{compact_text(format_question_coverage(coverage_by_question or []), DEFAULT_COVERAGE_CHARS)}

Evidence gaps from synthesis:
{format_missing_evidence_constraints(synthesis)}

Per-question evidence packs:
{compact_text(format_evidence_packs(evidence_packs or []), DEFAULT_EVIDENCE_PACK_CHARS)}

Write the final Markdown report. Explain each supported topic in clear prose before equations, tables, APIs, or technical details."""
    return trim_report_prompt(prompt)


def generate_single_report(client: Any, model: str, prompt: str) -> tuple[str, str]:
    print(f"Generating single report with model {model}...")
    prompt = trim_report_prompt(prompt)
    response = create_chat_completion_with_retries(
        client,
        model=model,
        temperature=0,
        max_tokens=DEFAULT_REPORT_MAX_TOKENS,
        messages=[
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return normalize_citation_markers(response.choices[0].message.content), clean_text(getattr(response, "model", "")) or model


def report_generation_token_cap() -> int:
    return (DEFAULT_REPORT_PROMPT_CHARS + 3) // 4 + DEFAULT_REPORT_MAX_TOKENS


def format_planner_questions(questions: Sequence[str]) -> str:
    items = [clean_text(q) for q in questions if clean_text(q)]
    return "\n".join(f"- {q}" for q in items) or "- Cover the research objective directly."


def format_report_section_outline(questions: Sequence[str]) -> str:
    items = [clean_text(q) for q in questions if clean_text(q)]
    if not items:
        return "- Use clear sections that answer the objective."
    return "\n".join(f"## {i}. {planner_question_heading(q)}" for i, q in enumerate(items, 1))


def format_question_coverage(coverage_by_question: Sequence[dict[str, Any]]) -> str:
    lines = []
    for item in coverage_by_question or []:
        if not isinstance(item, dict):
            continue
        question = clean_text(item.get("question"))
        if not question:
            continue
        question_id = clean_text(item.get("question_id")) or "question"
        status = clean_text(item.get("status")) or "unknown"
        evidence = ", ".join(clean_text(value) for value in item.get("required_evidence", []) if clean_text(value))
        source_indexes = [str(value) for value in item.get("source_indexes", []) if isinstance(value, int)]
        sources = ", ".join(f"[{index}]" for index in source_indexes) or "no cited sources"
        lines.append(f"- {question_id}: {status}; evidence={evidence or 'evidence'}; sources={sources}; question={question}")
    return "\n".join(lines) or "- No structured coverage map was provided; use planner questions and synthesis notes."


def evidence_pack_questions(evidence_packs: Sequence[Any]) -> list[str]:
    return dedupe_text(
        clean_text(pack.get("question"))
        for pack in evidence_packs or []
        if isinstance(pack, dict) and clean_text(pack.get("question"))
    )


def format_evidence_packs(evidence_packs: Sequence[dict[str, Any]]) -> str:
    lines = []
    for pack in evidence_packs or []:
        if not isinstance(pack, dict):
            continue
        question = clean_text(pack.get("question"))
        if not question:
            continue
        coverage = clean_text(pack.get("coverage")) or "unknown"
        lines.append(f"- {coverage}: {question}")
        chunks = pack.get("chunks", [])
        chunks = chunks if isinstance(chunks, list) else []
        for chunk in chunks[:1]:
            if not isinstance(chunk, dict):
                continue
            source_index = chunk.get("source_index")
            marker = f"[{source_index}]" if isinstance(source_index, int) else "[uncited]"
            title = compact_text(clean_text(chunk.get("title")) or clean_text(chunk.get("url")) or "Evidence chunk", 90)
            content = sanitize_evidence_content(chunk.get("content"))[:220]
            if content:
                lines.append(f"  - {marker} {title}: {content}")
    return "\n".join(lines) or "- No per-question evidence packs were provided."


def planner_question_heading(question: str) -> str:
    text = clean_text(question).rstrip("?")
    heading = re.sub(
        r"^(what|how|why|when|where|which)\s+(is|are|does|do|did|can|should)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    heading = re.sub(r"^(what|how|why|when|where|which)\s+", "", heading, flags=re.IGNORECASE)
    heading = re.sub(r"\b(e\.g\.|eg|examples?|evidence|results?)\b", "", heading, flags=re.IGNORECASE)
    heading = re.sub(r"\s+", " ", heading).strip(" .,:;")
    words = []
    for word in heading.split():
        clean_word = word.strip(".,:;()[]{}")
        words.append(clean_word if any(char.isupper() for char in clean_word[1:]) else clean_word.capitalize())
    return " ".join(words)[:90].strip() or "Research Finding"


def format_supporting_evidence(
    report_context: dict[str, Any],
    max_chars: int = DEFAULT_EVIDENCE_CHARS,
    sources: Sequence[dict[str, Any]] | None = None,
) -> str:
    chunks = list(report_context.get("supporting_chunks") or []) + list(report_context.get("retrieved_chunks") or [])
    blocks: list[dict[str, Any]] = []
    seen = set()
    source_index_by_url = {normalize_url(source.get("url")): source.get("index") for source in sources or [] if isinstance(source, dict)}
    query_text = " ".join(clean_text(q) for q in report_context.get("planner_questions", []) if clean_text(q))
    terms = list(detail_terms(query_text))[:20]

    def add_block(source_index: Any, title: Any, url: Any, content: Any) -> None:
        index = source_index if isinstance(source_index, int) else source_index_by_url.get(normalize_url(url))
        content = sanitize_evidence_content(content)[:DEFAULT_CHUNK_CHARS]
        if not content:
            return
        key = clean_text(f"{index}:{url}:{content[:120]}").lower()
        if key in seen:
            return
        seen.add(key)
        marker = f"[{index}]" if isinstance(index, int) else "[uncited]"
        block = f"{marker} {clean_text(title) or clean_text(url)}\n{content}"
        score = source_priority(url) * 20 + evidence_snippet_score(content, terms, EVIDENCE_SNIPPET_SIGNALS)
        blocks.append({"source_index": index, "url": clean_text(url), "block": block, "score": score})

    for chunk in chunks:
        if isinstance(chunk, dict):
            add_block(chunk.get("source_index") if isinstance(chunk.get("source_index"), int) else chunk.get("index"), chunk.get("title"), chunk.get("url"), chunk.get("content"))

    for source in browser_result_sources(report_context.get("browser_results", [])):
        url = source.get("url")
        add_block(source_index_by_url.get(normalize_url(url)), source.get("title"), url, best_evidence_snippet(source, query_text))

    return compact_evidence_blocks(blocks, max_chars)


def browser_result_sources(browser_results: Sequence[Any]) -> list[dict[str, Any]]:
    sources = []
    for result in browser_results or []:
        if not isinstance(result, dict):
            continue
        sources.extend(source for source in result.get("sources", []) or [] if isinstance(source, dict))
    return sources


def best_evidence_snippet(source: dict[str, Any], query_text: str) -> str:
    content = clean_text(source.get("full_content") or source.get("content") or source.get("content_preview"))
    if not content:
        return ""
    terms = list(detail_terms(query_text))[:20]
    candidates = [0]
    lowered = content.lower()
    for term in [*terms, *EVIDENCE_SNIPPET_SIGNALS]:
        location = lowered.find(term.lower())
        if location >= 0:
            candidates.append(location)
    best = max(candidates, key=lambda pos: evidence_snippet_score(content[pos: pos + DEFAULT_CHUNK_CHARS], terms, EVIDENCE_SNIPPET_SIGNALS))
    start = max(0, best - DEFAULT_CHUNK_CHARS // 4)
    return clean_text(content[start: start + DEFAULT_CHUNK_CHARS])


def evidence_snippet_score(snippet: str, terms: Sequence[str], signals: Sequence[str]) -> int:
    lowered = snippet.lower()
    return sum(1 for term in terms if term in lowered) + 2 * sum(1 for signal in signals if signal in lowered)


def sanitize_evidence_content(text: Any) -> str:
    """Remove paper-internal numeric citations so they cannot be mistaken for source markers."""

    return clean_text(re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", clean_text(text)))


def compact_evidence_blocks(blocks: Sequence[dict[str, Any]], max_chars: int) -> str:
    ordered = sorted(blocks, key=lambda item: (-int(item.get("score") or 0), len(clean_text(item.get("block")))))
    selected, seen_sources, used = [], set(), 0
    for pass_number in (1, 2):
        for item in ordered:
            source_key = item.get("source_index") or normalize_url(item.get("url"))
            if pass_number == 1 and source_key in seen_sources:
                continue
            block = clean_text(item.get("block"))
            if not block or block in selected:
                continue
            if used + len(block) > max_chars:
                continue
            selected.append(block)
            seen_sources.add(source_key)
            used += len(block)
    return "\n\n".join(selected)


def format_sources(sources: Sequence[dict[str, Any]]) -> str:
    lines = []
    for fallback, source in enumerate(sources, 1):
        if not isinstance(source, dict):
            continue
        index = source.get("index") if isinstance(source.get("index"), int) else fallback
        title = clean_text(source.get("title")) or clean_text(source.get("url")) or f"Source {index}"
        url = clean_text(source.get("url"))
        lines.append(f"[{index}] {title} - {url}")
    return "\n".join(lines) or "No sources provided."


def evidence_backed_sources(sources: Sequence[dict[str, Any]], *evidence_texts: Any) -> list[dict[str, Any]]:
    cited = set()
    for text in evidence_texts:
        cited.update(citation_markers(text))
    if not cited:
        return list(sources or [])
    backed = [source for source in sources or [] if isinstance(source, dict) and source.get("index") in cited]
    return backed or list(sources or [])


def source_priority(url: Any) -> int:
    value = clean_text(url).lower()
    if any(signal in value for signal in ("arxiv.org", "openreview.net", "doi.org", "pytorch.org", "tensorflow.org", "docs.")) or ".edu" in value:
        return 2
    return 1 if value else 0


def sources_with_browser_results(sources: Sequence[Any], browser_results: Sequence[Any]) -> list[dict[str, Any]]:
    merged = []
    existing = set()
    used_indexes = set()
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        item = dict(source)
        index = item.get("index")
        if not isinstance(index, int):
            continue
        merged.append(item)
        used_indexes.add(index)
        url = normalize_url(item.get("url"))
        if url:
            existing.add(url)

    next_index = max(used_indexes, default=0) + 1
    for result in browser_results or []:
        if not isinstance(result, dict):
            continue
        for source in result.get("sources", []) or []:
            if not isinstance(source, dict):
                continue
            url = normalize_url(source.get("url"))
            if url and url not in existing:
                existing.add(url)
                while next_index in used_indexes:
                    next_index += 1
                used_indexes.add(next_index)
                merged.append({"index": next_index, "title": source.get("title"), "url": source.get("url")})
                next_index += 1
    return merged


def dedupe_sources(sources: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = []
    seen = set()
    used_indexes = set()
    next_index = 1
    for source in sources or []:
        url = normalize_url(source.get("url"))
        if url and url in seen:
            continue
        if url:
            seen.add(url)
        item = dict(source)
        index = item.get("index")
        if not isinstance(index, int) or index in used_indexes:
            while next_index in used_indexes:
                next_index += 1
            index = next_index
        item["index"] = index
        used_indexes.add(index)
        deduped.append(item)
    return deduped


def normalize_final_report(report: str, sources: Sequence[dict[str, Any]]) -> str:
    text = normalize_markdown_headings(remove_unavailable_citation_markers(clean_markdown(report), source_index_set(sources)))
    body = strip_references(text)
    return clean_markdown(f"{body}\n\n{references_section(body, sources)}")


def references_section(report: str, sources: Sequence[dict[str, Any]]) -> str:
    by_index = {source.get("index"): source for source in sources if isinstance(source, dict)}
    lines = ["## References"]
    used = citation_markers(report)
    if not used:
        lines.append("No cited source markers were used.")
        return "\n".join(lines)
    for index in used:
        source = by_index.get(index)
        if source:
            lines.append(f"[{index}] {clean_text(source.get('url'))}")
    return "\n".join(lines)


def strip_references(report: str) -> str:
    lines = []
    skipping = False
    for line in clean_markdown(report).splitlines():
        if is_references_heading(line):
            skipping = True
            continue
        if skipping and line.startswith("## ") and not is_references_heading(line):
            skipping = False
        if not skipping:
            lines.append(line)
    return clean_markdown("\n".join(lines))


def report_sub_question_coverage_check(
    report: str,
    planner_questions: Sequence[str],
) -> dict[str, Any]:
    questions = [clean_text(q) for q in planner_questions if clean_text(q)]
    missing = missing_sub_question_coverage(report, questions)
    missing_keys = {normalize_heading(q) for q in missing}
    return {
        "total": len(questions),
        "covered_count": sum(1 for q in questions if normalize_heading(q) not in missing_keys),
        "missing_count": len(missing),
        "missing": missing,
        "items": [{"question": q, "heading": planner_question_heading(q), "status": "missing" if normalize_heading(q) in missing_keys else "covered"} for q in questions],
    }


def synthesis_coverage_gap_questions(
    report_context: dict[str, Any],
    planner_questions: Sequence[str] | None = None,
) -> list[str]:
    """Return planner questions synthesis marked as missing, partial, or weak."""

    if not isinstance(report_context, dict):
        return []
    canonical = {normalize_heading(q): q for q in planner_questions or [] if clean_text(q)}
    gaps = []
    for item in report_context.get("coverage_by_question", []) or []:
        if not isinstance(item, dict) or not synthesis_coverage_status_is_gap(item.get("status")):
            continue
        question = clean_text(item.get("question"))
        if question:
            gaps.append(canonical.get(normalize_heading(question), question))
    return dedupe_text(gaps)


def synthesis_coverage_status_is_gap(status: Any) -> bool:
    lowered = clean_text(status).lower()
    return bool(lowered and any(term in lowered for term in ("missing", "partial", "weak", "insufficient", "unsupported", "failed", "error")))


def missing_sub_question_coverage(report: str, planner_questions: Sequence[str]) -> list[str]:
    report_terms = set(technical_question_terms(report))
    missing = []
    for question in planner_questions:
        terms = [term for term in technical_question_terms(question) if term not in STOPWORDS]
        named = named_terms(question)
        important = named or terms[:4]
        required = 1 if len(important) <= 2 else 2
        named_matches = sum(1 for term in named if term in report_terms)
        term_matches = sum(1 for term in terms[:6] if term in report_terms)
        if named_matches < required and term_matches < required:
            missing.append(question)
    return missing


def report_schema_issues(report: str, planner_questions: Sequence[str]) -> list[str]:
    headings = {normalize_heading(h) for h in h2_headings(report)}
    required = {
        "executive summary": ("executive summary",),
        "introduction/context": ("introduction and context", "introduction", "context"),
        "cross-cutting analysis/synthesis": ("cross-cutting analysis and synthesis", "cross-source synthesis", "cross-cutting analysis"),
        "limitations/open questions": ("limitations and open questions", "limitations", "open questions"),
        "conclusion": ("conclusion",),
        "references": ("references", "sources"),
    }
    issues = []
    for label, aliases in required.items():
        if not any(normalize_heading(alias) in headings for alias in aliases):
            issues.append(f"missing schema section: {label}")
    for question in planner_questions:
        heading = normalize_heading(planner_question_heading(question))
        if heading and not any(headings_match(heading, actual) for actual in headings):
            issues.append(f"missing planner topic section: {planner_question_heading(question)}")
    return issues


def report_self_critique(report_issues: Sequence[str], coverage_check: dict[str, Any], schema_issues: Sequence[str]) -> dict[str, Any]:
    unresolved = [clean_text(issue) for issue in [*report_issues, *schema_issues] if clean_text(issue)]
    unresolved.extend(f"missing planner topic: {q}" for q in coverage_check.get("missing", []) if clean_text(q))
    print(f"[report] self-critique: {len(unresolved)} issue(s)")
    return {"source": "deterministic", "unresolved_issues": unresolved, "coverage_missing": coverage_check.get("missing", []), "schema_issues": list(schema_issues)}


def report_quality_issues(
    report: str,
    sources: Sequence[dict[str, Any]] | None = None,
    evidence_text: str = "",
) -> list[str]:
    issues = []
    text = clean_markdown(report)
    if not text:
        return ["report is empty"]
    if not any(is_references_heading(line) for line in text.splitlines()):
        issues.append("report must include a References section")
    if has_placeholder_source_marker(text):
        issues.append("report contains placeholder or non-source citation markers")
    source_indexes = source_index_set(sources or [])
    invalid = unavailable_citation_markers(report, source_indexes)
    if invalid:
        issues.append(f"report uses unavailable citations: {format_citation_indexes(invalid)}")
    unsupported_metrics = unsupported_benchmark_metrics(text, evidence_text)
    if unsupported_metrics:
        issues.append(f"report includes benchmark metrics not present in evidence: {', '.join(unsupported_metrics[:5])}")
    return issues


def normalize_markdown_headings(markdown: str) -> str:
    lines = []
    for line in clean_markdown(markdown).splitlines():
        lines.append(re.sub(r"^(\s{0,3}#{1,6}\s+)#{1,6}\s+", r"\1", line))
    return "\n".join(lines)


def has_placeholder_source_marker(text: str) -> bool:
    return bool(re.search(r"\[\s*(?:[—-]|uncited|citation needed|source needed)[^\]]*\]", text, flags=re.IGNORECASE))


def unsupported_benchmark_metrics(report: str, evidence_text: str) -> list[str]:
    evidence = strip_markdown(evidence_text).lower()
    if not evidence:
        return []
    unsupported = []
    for line in clean_markdown(report).splitlines():
        if line.lstrip().startswith("#") or is_references_heading(line):
            continue
        text = strip_markdown(line)
        lowered = text.lower()
        if not re.search(r"\b(benchmark|bleu|glue|imagenet|accuracy|top[- ]?[15]|f1|auc|rouge)\b|%", lowered):
            continue
        for match in re.finditer(r"(?<![-\[])\b\d+(?:\.\d+)?\s*%?", text):
            value = clean_text(match.group(0)).lower()
            if is_non_metric_number(text, match):
                continue
            if value and value not in evidence and value not in unsupported:
                unsupported.append(value)
    return unsupported


def is_non_metric_number(text: str, match: re.Match[str]) -> bool:
    value = clean_text(match.group(0)).lower()
    if re.fullmatch(r"(?:19|20)\d{2}", value):
        return True
    if value.endswith("%"):
        return False
    before = text[max(0, match.start() - 2):match.start()]
    after = text[match.end():match.end() + 2]
    return bool(re.search(r"[-._/]$", before) or re.search(r"^[-._/]", after))


def rewrite_missing_sub_question_queries(objective: str, questions: Sequence[str]) -> list[str]:
    return [
        clean_text(f"{objective} {question} source-backed evidence details examples equations benchmarks limitations")[:700]
        for question in questions
        if clean_text(question)
    ]


def report_context_gap_items(report_context: dict[str, Any], research_plan: dict[str, Any]) -> list[str]:
    synthesis = clean_text(report_context.get("synthesis")) if isinstance(report_context, dict) else ""
    questions = [clean_text(q) for q in research_plan.get("sub_questions", []) if clean_text(q)] if isinstance(research_plan, dict) else []
    missing = synthesis_coverage_gap_questions(report_context, questions)
    missing.extend(q for q in missing_sub_question_coverage(synthesis, questions) if q not in missing)
    gap_text = "\n".join(missing_evidence_constraints(synthesis)).lower()
    for question in questions:
        if question not in missing and any(term in gap_text for term in detail_terms(question)):
            missing.append(question)
    return missing


def report_context_gap_queries(report_context: dict[str, Any], research_plan: dict[str, Any]) -> list[str]:
    objective = clean_text(research_plan.get("objective")) if isinstance(research_plan, dict) else clean_text(report_context.get("objective"))
    return rewrite_missing_sub_question_queries(objective, report_context_gap_items(report_context, research_plan))


def missing_evidence_constraints(synthesis: Any) -> list[str]:
    constraints = []
    for line in clean_markdown(synthesis).splitlines():
        value = clean_text(line)
        lowered = value.lower()
        if any(signal in lowered for signal in ("missing", "not present", "not in the retrieved", "gap")):
            constraints.append(strip_markdown(value)[:300])
    return dedupe_text(constraints)


def format_missing_evidence_constraints(synthesis: Any) -> str:
    items = missing_evidence_constraints(synthesis)
    return "\n".join(f"- {item}" for item in items) if items else "- No explicit missing-evidence constraints."


def write_report_file(report_payload: dict[str, Any], memory_path: str = "data/shared_memory.json", report_path: str | None = None) -> str:
    report = clean_markdown(report_payload.get("report"))
    if not report:
        raise ValueError("report_payload.report is required")
    output_path = Path(report_path) if report_path else default_report_path(report_payload, memory_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    return str(output_path)


def default_report_path(report_payload: dict[str, Any], memory_path: str) -> Path:
    base = Path(memory_path).parent
    output_dir = Path(DEFAULT_REPORT_OUTPUT_DIR) if str(base) in {"", "."} else base / "reports"
    return output_dir / f"{slugify_filename(report_payload.get('objective'))}.md"


def slugify_filename(text: Any, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", clean_text(text).lower()).strip("-")
    return (slug[:max_length].strip("-") or "research-report")


def clean_markdown(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = normalize_citation_markers(text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def normalize_citation_markers(text: Any) -> str:
    value = str(text or "")
    value = re.sub(r"【\s*(\d+)(?:[^】]*)?】", r"[\1]", value)
    value = re.sub(r"\[\s*(\d+(?:\s*,\s*\d+)+)\s*\]", lambda m: " ".join(f"[{p.strip()}]" for p in m.group(1).split(",")), value)
    value = re.sub(r"\[\s*(\d+)\s*\]", r"[\1]", value)
    return value


def remove_unavailable_citation_markers(text: str, available_indexes: set[int]) -> str:
    if not available_indexes:
        return re.sub(r"\[\d+\]", "", text)
    return re.sub(r"\[(\d+)\]", lambda m: m.group(0) if int(m.group(1)) in available_indexes else "", text)


def unavailable_citation_markers(text: str, available_indexes: set[int]) -> list[int]:
    return sorted({index for index in citation_markers(text) if available_indexes and index not in available_indexes})


def citation_markers(text: Any) -> list[int]:
    seen, markers = set(), []
    for match in re.finditer(r"\[(\d+)\]", normalize_citation_markers(text)):
        index = int(match.group(1))
        if index not in seen:
            seen.add(index)
            markers.append(index)
    return markers


def source_index_set(sources: Sequence[dict[str, Any]]) -> set[int]:
    return {source.get("index") for source in sources if isinstance(source, dict) and isinstance(source.get("index"), int)}


def h2_headings(markdown: str) -> list[str]:
    headings = []
    in_fence = False
    for line in clean_markdown(markdown).splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            headings.append(match.group(1).strip())
    return headings


def is_references_heading(line: str) -> bool:
    return normalize_heading(line.lstrip("#").strip()) in {"references", "reference", "sources"}


def normalize_heading(text: Any) -> str:
    value = strip_markdown(text).replace("‑", "-").replace("–", "-").replace("—", "-")
    value = re.sub(r"^\d+[.)]\s*", "", value)
    value = re.sub(r"[^a-zA-Z0-9]+", " ", value)
    return clean_text(value).lower()


def headings_match(expected: str, actual: str) -> bool:
    if expected == actual or expected in actual or actual in expected:
        return True
    expected_terms = detail_terms(expected)
    actual_terms = detail_terms(actual)
    if not expected_terms:
        return False
    if len(expected_terms & actual_terms) >= min(3, len(expected_terms)):
        return True
    required = min(3, max(2, len(expected_terms) // 3))
    return len(expected_terms & actual_terms) >= required


def detail_terms(text: Any) -> set[str]:
    return {token.lower().replace("‑", "-").replace("–", "-") for token in re.findall(r"[A-Za-z][A-Za-z0-9_+.-]{2,}", strip_markdown(text)) if token.lower() not in STOPWORDS}


def technical_question_terms(text: Any) -> list[str]:
    value = strip_markdown(text).replace("‑", "-").replace("–", "-")
    terms = re.findall(r"[A-Za-z][A-Za-z0-9_+.-]{2,}", value)
    phrases = []
    for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9_+.-]*(?:-[A-Za-z0-9_+.-]+)+\b", value):
        phrase = match.group(0).lower()
        phrases.extend([phrase, phrase.replace("-", "")])
    return dedupe_text([*(term.lower() for term in terms), *phrases])


def named_terms(text: Any) -> list[str]:
    terms = []
    for token in re.findall(r"\b[A-Z][A-Za-z0-9_+.-]{2,}\b|\b[A-Z]{2,}\b", clean_text(text)):
        if token.lower() not in STOPWORDS:
            terms.append(token.lower())
    lowered = clean_text(text).lower()
    for phrase in ("multiheadattention", "multi-head", "self-attention", "vision transformer", "scaled dot", "cross-attention"):
        if phrase in lowered:
            terms.append(phrase)
    return dedupe_text(terms)


def strip_markdown(text: Any) -> str:
    value = re.sub(r"`([^`]*)`", r"\1", str(text or ""))
    value = re.sub(r"[*_#|]+", " ", value)
    value = re.sub(r"\[(\d+)\]", "", value)
    return clean_text(value)


def compact_text(value: Any, max_chars: int) -> str:
    text = clean_markdown(value)
    return text if len(text) <= max_chars else text[:max_chars].rstrip()


def trim_report_prompt(prompt: Any, max_chars: int = DEFAULT_REPORT_PROMPT_CHARS) -> str:
    text = clean_markdown(prompt)
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars].rstrip()
    return trimmed.rsplit("\n\n", 1)[0].rstrip() if "\n\n" in trimmed else trimmed


def dedupe_text(items: Sequence[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        value = clean_text(item)
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def normalize_url(value: Any) -> str:
    return clean_text(value).rstrip("/").lower()


def format_citation_indexes(indexes: Sequence[int]) -> str:
    return ", ".join(f"[{index}]" for index in sorted(set(indexes)))


# Compatibility aliases used by older tests/scripts.
normalize_report_for_validation = normalize_final_report
hard_report_issues = lambda issues: [issue for issue in issues if clean_text(issue)]
markdown_completion_issues = lambda markdown: [] if clean_text(markdown) else ["section is empty"]
format_evidence_coverage_brief = lambda **kwargs: ""
format_memory_signal_evidence = lambda *args, **kwargs: ""
format_planner_evidence_packet = lambda *args, **kwargs: ""
format_source_priority_guidance = lambda sources: ""
remove_conflicting_missing_evidence_statements = lambda report, evidence_text="": report
remove_placeholder_citations = lambda text: re.sub(r"\[(?:uncited|citation needed|source needed)\]", "", str(text), flags=re.IGNORECASE)
ensure_planner_question_sections = lambda report, *args, **kwargs: report
