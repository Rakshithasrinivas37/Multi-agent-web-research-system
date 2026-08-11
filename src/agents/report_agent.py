"""Report agent for final research report generation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Sequence

from src.memory.shared_memory import SharedMemory
from src.tools.groq_retry import create_chat_completion_with_retries as groq_chat_completion_with_retries
from src.tools.progress import emit_progress
from src.tools.text_utils import clean_text


DEFAULT_REPORT_AGENT_MODEL = "llama-3.1-8b-instant"
DEFAULT_REPORT_AGENT_MAX_TOKENS = 4200
DEFAULT_REPORT_AGENT_CONTEXT_CHARS = 30000
DEFAULT_REPORT_AGENT_CHUNK_CHARS = 1000
DEFAULT_REPORT_TOTAL_TOKEN_BUDGET = 10000
DEFAULT_REPORT_SINGLE_PROMPT_CHARS = 12000
DEFAULT_REPORT_SINGLE_MAX_TOKENS = 4000
DEFAULT_REPORT_OUTPUT_DIR = "data/reports"
DEFAULT_REPORT_DIAGNOSTICS_CHARS = 1200
DEFAULT_REPORT_MEMORY_EVIDENCE_CHARS = 3600
DEFAULT_REPORT_COVERAGE_BRIEF_CHARS = 2400
REPORT_REPAIR_MAX_ATTEMPTS = 0


class ReportAgent:
    """Generate a final report from synthesis output and cited evidence."""

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = DEFAULT_REPORT_AGENT_MAX_TOKENS,
        max_context_chars: int = DEFAULT_REPORT_AGENT_CONTEXT_CHARS,
    ) -> None:
        self.model = (
            clean_text(model)
            or clean_text(os.environ.get("RESEARCH_PLANNER_MODEL"))
            or clean_text(os.environ.get("RAG_GENERATION_MODEL"))
            or DEFAULT_REPORT_AGENT_MODEL
        )
        self.max_tokens = max_tokens
        self.max_context_chars = max_context_chars

    def generate(
        self,
        report_context: dict[str, Any],
        output_format: str = "report",
    ) -> dict[str, Any]:
        """Write the final report using only synthesis-agent context."""

        if not isinstance(report_context, dict) or not report_context:
            raise ValueError("report_context is required")
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set")

        try:
            from groq import Groq
        except ImportError as error:
            raise RuntimeError("groq package is not installed. Install it with `pip install -r requirements.txt`.") from error

        objective = clean_text(report_context.get("objective"))
        synthesis = normalize_citation_markers(report_context.get("synthesis"))
        if not objective:
            raise ValueError("report_context.objective is required")
        if not synthesis:
            raise ValueError("report_context.synthesis is required")

        original_sources = sources_with_browser_results(
            report_context.get("sources", []),
            report_context.get("browser_results", []),
        )
        sources, citation_aliases = dedupe_sources_by_url(original_sources)
        synthesis = remap_citation_markers(synthesis, citation_aliases)
        available_source_indexes = source_index_set(sources)
        synthesis = remove_unavailable_citation_markers(synthesis, available_source_indexes)
        source_text = format_sources(sources)
        evidence_text = format_supporting_evidence(report_context, citation_aliases=citation_aliases)
        memory_evidence_text = format_memory_signal_evidence(report_context, sources, citation_aliases, evidence_text)
        if memory_evidence_text:
            evidence_text = clean_markdown(f"Memory evidence signals:\n{memory_evidence_text}\n\n{evidence_text}")
        evidence_text = remove_unavailable_citation_markers(evidence_text, available_source_indexes)
        missing_evidence_text = format_missing_evidence_constraints(synthesis)
        coverage_brief = format_evidence_coverage_brief(
            planner_questions=[clean_text(question) for question in report_context.get("planner_questions", []) or []],
            synthesis=synthesis,
            evidence_text=evidence_text,
            missing_evidence_text=missing_evidence_text,
        )
        citation_policy = clean_text(report_context.get("citation_policy")) or (
            "Use only numbered source markers from the provided sources."
        )
        diagnostics = compact_markdown(report_context.get("diagnostics", {}), max_chars=DEFAULT_REPORT_DIAGNOSTICS_CHARS)
        client = Groq()
        output_format_text = clean_text(output_format) or "report"
        planner_questions = [clean_text(question) for question in report_context.get("planner_questions", []) or []]
        single_prompt = build_single_report_prompt(
            objective=objective,
            output_format=output_format_text,
            citation_policy=citation_policy,
            planner_questions=planner_questions,
            coverage_brief=coverage_brief,
            synthesis=synthesis,
            missing_evidence_text=missing_evidence_text,
            evidence_text=evidence_text,
            source_text=source_text,
            diagnostics=diagnostics,
        )
        emit_progress(
            "tool_called",
            "Report agent calling Groq to generate final report",
            agent="report",
            tool="groq",
            metadata={"model": self.model},
        )
        report, report_model = generate_single_report(
            client=client,
            model=self.model,
            prompt=single_prompt,
        )
        estimated_token_cap = report_generation_token_cap()
        report, repair_count, report_issues = finalize_report(
            client=client,
            model=self.model,
            report=report,
            objective=objective,
            output_format=output_format_text,
            synthesis=synthesis,
            evidence_text=evidence_text,
            source_text=source_text,
            missing_evidence_text=missing_evidence_text,
            sources=sources,
            report_max_tokens=min(max(1200, self.max_tokens), DEFAULT_REPORT_AGENT_MAX_TOKENS),
            report_context_chars=min(self.max_context_chars, DEFAULT_REPORT_SINGLE_PROMPT_CHARS),
        )
        missing_sub_questions = missing_sub_question_coverage(report, planner_questions)
        return {
            "objective": objective,
            "output_format": output_format_text,
            "report": report,
            "sources": sources,
            "model": report_model,
            "diagnostics": {
                "source_count": len(sources),
                "deduped_source_count": len(original_sources or []) - len(sources),
                "supporting_chunk_count": len(report_context.get("supporting_chunks", []) or []),
                "missing_evidence_constraint_count": missing_evidence_constraint_count(missing_evidence_text),
                "report_length": len(report),
                "report_generation_mode": "single",
                "report_repair_count": repair_count,
                "report_issues": report_issues,
                "report_missing_sub_questions": missing_sub_questions,
                "report_retry_queries": rewrite_missing_sub_question_queries(objective, missing_sub_questions),
                "report_token_budget": DEFAULT_REPORT_TOTAL_TOKEN_BUDGET,
                "report_estimated_token_cap": estimated_token_cap,
            },
        }

    def write_to_memory(
        self,
        report_payload: dict[str, Any],
        memory_path: str = "data/shared_memory.json",
        report_path: str | None = None,
    ) -> None:
        """Persist the final report to a Markdown file and shared memory."""

        saved_path = write_report_file(report_payload, memory_path, report_path)
        report_payload = {**report_payload, "report_path": saved_path}
        memory = SharedMemory(memory_path)
        memory.write_agent_output("report", {"final_report": report_payload})


def build_single_report_prompt(
    objective: str,
    output_format: str,
    citation_policy: str,
    planner_questions: Sequence[str],
    coverage_brief: str,
    synthesis: str,
    missing_evidence_text: str,
    evidence_text: str,
    source_text: str,
    diagnostics: Any,
) -> str:
    return f"""Research objective:
{objective}

Requested output format:
{output_format}

Citation policy:
{citation_policy}

Generation requirements:
- Return polished Markdown.
- Include a clear title.
- Include an executive summary under the exact heading "## Executive Summary".
- Write a detailed technical report, not a short summary.
- Include technical sections that match the objective, synthesis, and requested output format.
- Answer every planner sub-question explicitly using the available evidence.
- Prefer concrete definitions, structured comparisons, measurements, examples, and implementation details when supported by synthesis or supporting chunks.
- Use synthesis-agent notes and supporting chunks as the only factual basis; do not fill gaps from model prior knowledge.
- Explain important technical terms or notation when applicable, and use tables when they make comparisons clearer.
- Include exact technical details only when they are supported by synthesis or supporting chunks.
- Include supported equations, API signatures, and code snippets when they appear in synthesis or supporting chunks.
- Treat partial evidence as usable: write the supported part, then place only the unresolved part in "Evidence Gaps".
- Reconcile Missing-evidence constraints against Supporting evidence chunks before writing.
- If supporting chunks contain a detail that synthesis previously marked missing, include the supported detail and do not say it is missing.
- Do not generalize a missing detail to a broader topic; include supported parts and name only the exact unsupported detail.
- Do not reproduce details listed in Missing-evidence constraints unless they are present in Supporting evidence chunks.
- If a missing item remains important, state only that the provided evidence describes it but does not include the exact detail.
- Include a brief "Evidence Gaps" section when Missing-evidence constraints identify partial or missing required items.
- Cite claims using only plain source markers from Available sources, exactly like [1], [2], [3].
- For precise claims, prefer original papers, official docs, academic sources, or authoritative surveys.
- Do not cite sources that are not listed.
- Do not use citation formats like 【1】, footnotes, line citations, or URLs inline.
- End with a References section mapping only used source markers to source URLs.
- If evidence is incomplete, mention the limitation instead of inventing details.
- Before finalizing, remove contradictions such as saying a detail is missing and then including that detail.

Available sources:
{source_text}

Planner sub-questions that must be answered:
{format_planner_questions(planner_questions)}

Evidence coverage brief:
{compact_markdown(coverage_brief, DEFAULT_REPORT_COVERAGE_BRIEF_CHARS)}

Supporting evidence chunks:
{compact_markdown(evidence_text, 5000)}

Missing-evidence constraints:
{compact_markdown(missing_evidence_text, 1400)}

Synthesis-agent notes:
{compact_markdown(synthesis, 3200)}


Retrieval diagnostics:
{diagnostics}

Generate the final report using only the synthesis-agent notes, supporting evidence chunks, and available sources above."""


def generate_single_report(client: Any, model: str, prompt: str) -> tuple[str, str]:
    print(f"Generating single report with model {model}...")
    return groq_chat_text(
        client=client,
        model=model,
        system_prompt="You are a careful report-writing agent. Use only provided synthesis context and evidence.",
        user_prompt=prompt,
        max_tokens=DEFAULT_REPORT_SINGLE_MAX_TOKENS,
        max_context_chars=DEFAULT_REPORT_SINGLE_PROMPT_CHARS,
    )


def report_generation_token_cap() -> int:
    """Estimate worst-case report-agent model tokens from configured prompt/output caps."""

    return estimated_tokens(DEFAULT_REPORT_SINGLE_PROMPT_CHARS) + DEFAULT_REPORT_SINGLE_MAX_TOKENS


def format_planner_questions(questions: Sequence[str]) -> str:
    clean_questions = [clean_text(question) for question in questions if clean_text(question)]
    if not clean_questions:
        return "- No planner sub-questions were provided. Cover the research objective directly."
    return "\n".join(f"- {question}" for question in clean_questions)


def missing_sub_question_coverage(report: str, planner_questions: Sequence[str]) -> list[str]:
    """Return planner questions whose specific terms are not reflected in the report."""

    questions = [clean_text(question) for question in planner_questions if clean_text(question)]
    if not questions:
        return []
    report_terms = detail_terms(report)
    common_terms = common_question_terms(questions)
    question_terms = {"what", "when", "where", "which", "whose", "why", "does", "used"}
    missing = []
    for question in questions:
        terms = [term for term in detail_terms(question) if term not in common_terms and term not in question_terms]
        if not terms:
            continue
        overlap_count = sum(1 for term in set(terms) if report_has_question_term(report_terms, term))
        required_overlap = 1
        if overlap_count < required_overlap:
            missing.append(question)
    return missing


def report_has_question_term(report_terms: set[str], term: str) -> bool:
    if term in report_terms:
        return True
    prefix = term[:5]
    return len(prefix) >= 5 and any(value.startswith(prefix) or term.startswith(value[:5]) for value in report_terms)


def common_question_terms(questions: Sequence[str]) -> set[str]:
    threshold = max(2, (len(questions) + 1) // 2)
    counts: dict[str, int] = {}
    for question in questions:
        for term in detail_terms(question):
            counts[term] = counts.get(term, 0) + 1
    return {term for term, count in counts.items() if count >= threshold}


def rewrite_missing_sub_question_queries(objective: str, questions: Sequence[str]) -> list[str]:
    objective_text = clean_text(objective)
    queries = []
    for question in questions:
        query = clean_text(
            f"{objective_text} {question} source-backed evidence details definitions examples equations limitations"
        )
        if query:
            queries.append(query[:600])
    return dedupe_preserve_order(queries)


def report_context_gap_items(report_context: dict[str, Any], research_plan: dict[str, Any]) -> list[str]:
    """Find missing or partial evidence items before the final report call."""

    if not isinstance(report_context, dict):
        return []
    synthesis = clean_markdown(report_context.get("synthesis"))
    if not synthesis:
        return []
    planner_questions = report_context.get("planner_questions") or research_plan.get("sub_questions") or []
    planner_questions = [clean_text(question) for question in planner_questions if clean_text(question)]
    explicit_constraints = missing_evidence_constraints(synthesis)
    question_gaps = planner_question_gap_items(synthesis, planner_questions)
    uncovered_questions = missing_sub_question_coverage(synthesis, planner_questions)
    return dedupe_preserve_order([*question_gaps, *uncovered_questions, *explicit_constraints])


def report_context_gap_queries(report_context: dict[str, Any], research_plan: dict[str, Any]) -> list[str]:
    objective = clean_text(research_plan.get("objective")) or clean_text(report_context.get("objective"))
    return rewrite_missing_sub_question_queries(
        objective,
        report_context_gap_items(report_context, research_plan),
    )


def planner_question_gap_items(synthesis: str, planner_questions: Sequence[str]) -> list[str]:
    questions = [clean_text(question) for question in planner_questions if clean_text(question)]
    if not questions:
        return []
    current_heading = ""
    gap_items = []
    for line in clean_markdown(synthesis).splitlines():
        line_text = clean_text(line)
        if not line_text:
            continue
        if line_text.startswith("#"):
            current_heading = strip_markdown_markup(line_text)
            continue
        if not line_mentions_gap(line_text):
            continue
        for question in questions:
            if question_matches_gap_context(question, f"{current_heading} {line_text}"):
                gap_items.append(f"{question}: {strip_markdown_markup(line_text)}")
                break
    return dedupe_preserve_order(gap_items)


def line_mentions_gap(line: str) -> bool:
    lowered = clean_text(strip_markdown_markup(line)).lower()
    return any(
        marker in lowered
        for marker in (
            "partial",
            "missing",
            "gap",
            "none in the retrieved",
            "not present",
            "not extracted",
            "does not contain",
        )
    )


def question_matches_gap_context(question: str, context: str) -> bool:
    question_terms = detail_terms(question)
    context_terms = detail_terms(context)
    if not question_terms or not context_terms:
        return False
    common_question_words = {"what", "when", "where", "which", "whose", "does", "used"}
    specific_terms = {term for term in question_terms if term not in common_question_words}
    required = min(2, max(1, len(specific_terms) // 3))
    return len(specific_terms & context_terms) >= required


def estimated_tokens(char_count: int) -> int:
    return max(1, (max(0, char_count) + 3) // 4)


def groq_chat_text(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    max_context_chars: int,
) -> tuple[str, str]:
    response = groq_chat_completion_with_retries(
        client,
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt[:max_context_chars]},
        ],
    )
    return normalize_citation_markers(response.choices[0].message.content), clean_text(getattr(response, "model", "")) or model


def markdown_completion_issues(markdown: str) -> list[str]:
    text = clean_markdown(markdown)
    lines = [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and not re.fullmatch(r"[-*_]{3,}", line.strip())
    ]
    if not lines:
        return ["section is empty"]

    issues = []
    if text.count("```") % 2:
        issues.append("section has an unclosed fenced code block")
    if text.count(r"\[") != text.count(r"\]"):
        issues.append("section has an unclosed display math block")
    if text.count(r"\(") != text.count(r"\)"):
        issues.append("section has an unclosed inline math expression")
    if text.count("**") % 2:
        issues.append("section has unclosed bold markdown")

    last_line = lines[-1].strip()
    if last_line.startswith("|") and not last_line.endswith("|"):
        issues.append("section ends with an unfinished table row")
    if re.search(r"<(?:sub|sup)>[^<]*$", last_line, flags=re.IGNORECASE):
        issues.append("section ends with an unfinished HTML tag")
    if unfinished_final_line(last_line):
        issues.append("section appears to stop mid-sentence")
    return dedupe_preserve_order(issues)


def unfinished_final_line(line: str) -> bool:
    text = clean_text(strip_markdown_markup(line))
    if not text:
        return False
    lowered = text.lower().rstrip()
    if lowered.endswith((",", ";", ":", "-", " and", " or", " of", " with", " by", " to", " from", " the")):
        return True
    if line.startswith("|"):
        return False
    return text[-1] not in ".!?)]}`'\""


def short_issue_label(text: str, max_length: int = 80) -> str:
    value = clean_text(strip_markdown_markup(text))
    if len(value) <= max_length:
        return value
    return value[:max_length].rstrip(" ,.;:") + "..."


def normalize_final_report(report: str, sources: Sequence[dict[str, Any]]) -> str:
    body = strip_all_references_blocks(report)
    body = remove_unavailable_citation_markers(body, source_index_set(sources))
    body = remove_empty_math_blocks(body)
    body = remove_malformed_table_rows(body)
    body = trim_incomplete_section_tails(body)
    body = remove_empty_sections(body)
    body = remove_incomplete_sections(body)
    body = remove_empty_sections(body)
    body = ensure_executive_summary_section(body)
    return clean_markdown(f"{body}\n\n{references_section(body, sources)}")


def ensure_executive_summary_section(markdown: str) -> str:
    text = clean_markdown(markdown)
    if has_h2_executive_summary(text):
        return text

    lines = text.splitlines()
    summary_bounds = find_summary_section_bounds(lines)
    if summary_bounds:
        start, end = summary_bounds
        summary_lines = ["## Executive Summary", *nonempty_lines(lines[start + 1 : end])]
        remaining = lines[:start] + lines[end:]
    else:
        summary_lines = ["## Executive Summary", fallback_executive_summary(lines)]
        remaining = lines

    insert_at = executive_summary_insert_index(remaining)
    updated = [*remaining[:insert_at], "", *summary_lines, "", *remaining[insert_at:]]
    return clean_markdown("\n".join(updated))


def has_h2_executive_summary(markdown: str) -> bool:
    return any(
        line.startswith("## ") and normalized_heading(line.lstrip("#").strip()) == "executive summary"
        for line in clean_markdown(markdown).splitlines()
    )


def find_summary_section_bounds(lines: Sequence[str]) -> tuple[int, int] | None:
    for index, line in enumerate(lines):
        match = re.match(r"^(#{2,3})\s+(.+)$", line)
        if not match or normalized_heading(match.group(2)) not in {"summary", "executive summary"}:
            continue
        level = len(match.group(1))
        end = len(lines)
        for next_index, next_line in enumerate(lines[index + 1 :], start=index + 1):
            next_match = re.match(r"^(#{2,3})\s+", next_line)
            if next_match and len(next_match.group(1)) <= level:
                end = next_index
                break
        return index, end
    return None


def nonempty_lines(lines: Sequence[str]) -> list[str]:
    return [line for line in lines if clean_text(line)]


def executive_summary_insert_index(lines: Sequence[str]) -> int:
    for index, line in enumerate(lines):
        if not clean_text(line):
            continue
        if line.startswith("# ") or (line.startswith("**") and line.endswith("**")):
            return index + 1
        return index
    return 0


def fallback_executive_summary(lines: Sequence[str]) -> str:
    for line in lines:
        text = clean_text(strip_markdown_markup(line))
        if not text or line.startswith("#") or line.startswith("|") or re.fullmatch(r"[-*_]{3,}", line.strip()):
            continue
        return text[:500].rstrip()
    return "This report summarizes the available evidence for the research objective."


def normalized_heading(text: str) -> str:
    return clean_text(re.sub(r"^\d+[.)]\s*", "", text)).lower()


def strip_all_references_blocks(markdown: str) -> str:
    lines = clean_markdown(markdown).splitlines()
    kept = []
    skip = False
    for line in lines:
        if is_references_heading(line):
            remove_trailing_separators(kept)
            skip = True
            continue
        if skip and line.startswith("## ") and not is_references_heading(line):
            skip = False
        if not skip:
            kept.append(line)
    return trim_trailing_separators(clean_markdown("\n".join(kept)))


def trim_trailing_separators(markdown: str) -> str:
    lines = clean_markdown(markdown).splitlines()
    remove_trailing_separators(lines)
    return clean_markdown("\n".join(lines))


def remove_trailing_separators(lines: list[str]) -> None:
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and re.fullmatch(r"[-*_]{3,}", lines[-1].strip()):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()


def is_references_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    text = normalized_heading(strip_markdown_markup(stripped).strip()).rstrip(":")
    return text in {"references", "reference"}


def references_section(markdown: str, sources: Sequence[dict[str, Any]]) -> str:
    source_by_index = {
        source.get("index"): source
        for source in sources
        if isinstance(source, dict) and isinstance(source.get("index"), int)
    }
    lines = ["## References"]
    for index in citation_markers(markdown):
        source = source_by_index.get(index)
        if not source:
            continue
        url = clean_text(source.get("url"))
        if url:
            lines.append(f"[{index}] {url}")
    if len(lines) == 1:
        lines.append("No cited source markers were used.")
    return "\n".join(lines)


def source_index_set(sources: Sequence[dict[str, Any]]) -> set[int]:
    return {
        source["index"]
        for source in sources
        if isinstance(source, dict) and isinstance(source.get("index"), int)
    }


def unavailable_citation_markers(text: Any, available_indexes: set[int]) -> list[int]:
    if not available_indexes:
        return []
    return [index for index in citation_markers(text) if index not in available_indexes]


def remove_unavailable_citation_markers(text: Any, available_indexes: set[int]) -> str:
    normalized = normalize_citation_markers(text)
    if not available_indexes:
        return normalized

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return match.group(0) if index in available_indexes else ""

    return clean_markdown(re.sub(r"\[(\d+)\]", replace, normalized))


def missing_reference_entries_for_used_citations(report: str, available_indexes: set[int]) -> list[int]:
    body = strip_all_references_blocks(report)
    used = {index for index in citation_markers(body) if index in available_indexes}
    references = reference_entry_indexes(report)
    return sorted(used - references)


def reference_entry_indexes(report: str) -> set[int]:
    indexes = set()
    in_references = False
    for line in clean_markdown(report).splitlines():
        if is_references_heading(line):
            in_references = True
            continue
        if in_references and line.startswith("## ") and not is_references_heading(line):
            break
        if in_references:
            match = re.match(r"^\[(\d+)\]\s+", line.strip())
            if match:
                indexes.add(int(match.group(1)))
    return indexes


def format_citation_indexes(indexes: Sequence[int]) -> str:
    return ", ".join(f"[{index}]" for index in sorted(set(indexes)))


def citation_markers(text: Any) -> list[int]:
    markers = []
    seen = set()
    for match in re.finditer(r"\[(\d+)\]", normalize_citation_markers(text)):
        index = int(match.group(1))
        if index in seen:
            continue
        seen.add(index)
        markers.append(index)
    return markers


def compact_markdown(value: Any, max_chars: int) -> str:
    text = clean_markdown(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def finalize_report(
    client: Any,
    model: str,
    report: str,
    objective: str,
    output_format: str,
    synthesis: str,
    evidence_text: str,
    source_text: str,
    missing_evidence_text: str,
    sources: Sequence[dict[str, Any]],
    report_max_tokens: int,
    report_context_chars: int,
) -> tuple[str, int, list[str]]:
    if not clean_text(report):
        raise ValueError("report_agent produced empty report")

    repaired = normalize_report_for_validation(report, sources, evidence_text)
    issues = report_quality_issues(repaired, evidence_text, sources=sources)
    total_repair_count = 0
    for _ in range(REPORT_REPAIR_MAX_ATTEMPTS):
        if not issues:
            break
        repaired, repair_count, issues = repair_report_if_needed(
            client=client,
            model=model,
            report=repaired,
            objective=objective,
            output_format=output_format,
            synthesis=synthesis,
            evidence_text=evidence_text,
            source_text=source_text,
            missing_evidence_text=missing_evidence_text,
            sources=sources,
            max_tokens=report_max_tokens,
            max_context_chars=report_context_chars,
        )
        total_repair_count += repair_count
        repaired = normalize_report_for_validation(repaired, sources, evidence_text)
        issues = report_quality_issues(repaired, evidence_text, sources=sources)

    blocking_issues = hard_report_issues(issues)
    if blocking_issues:
        raise ValueError(f"report_agent produced invalid report: {'; '.join(blocking_issues)}")
    return repaired, total_repair_count, issues


def hard_report_issues(issues: Sequence[str]) -> list[str]:
    """Return report issues that should block saving the final report."""

    return [issue for issue in issues if clean_text(issue)]


def normalize_report_for_validation(
    report: str,
    sources: Sequence[dict[str, Any]],
    evidence_text: str,
) -> str:
    normalized = normalize_final_report(report, sources)
    normalized = remove_weak_implementation_api_sections(normalized)
    normalized = ensure_supported_api_details(normalized, evidence_text)
    normalized = remove_resolved_evidence_gap_rows(normalized, evidence_text)
    cleaned = remove_conflicting_missing_evidence_statements(normalized, evidence_text)
    normalized = normalize_final_report(cleaned, sources)
    normalized = remove_weak_implementation_api_sections(normalized)
    normalized = ensure_supported_api_details(normalized, evidence_text)
    normalized = remove_resolved_evidence_gap_rows(normalized, evidence_text)
    return remove_conflicting_missing_evidence_statements(normalized, evidence_text)


def ensure_supported_api_details(report: str, evidence_text: str) -> str:
    """Add a compact API section when supported API identifiers were omitted."""

    if not evidence_has_api_signal(evidence_text) or evidence_has_api_signal(report):
        return report
    items = supported_api_items(evidence_text)
    if not items:
        return report
    lines = ["## Implementation APIs"]
    for api_name, marker in items[:6]:
        citation = f" [{marker}]" if marker else ""
        lines.append(f"- `{api_name}` is present in the supporting evidence{citation}.")
    return append_section_before_references(report, "\n".join(lines))


def supported_api_items(text: str) -> list[tuple[str, str]]:
    items = []
    seen = set()
    for line in clean_markdown(text).splitlines():
        marker_match = re.search(r"\[(\d+)\]", line)
        marker = marker_match.group(1) if marker_match else ""
        for match in attention_api_matches(line):
            api_name = match.group(0).rstrip(".")
            key = api_name.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append((api_name, marker))
    return items


def remove_weak_implementation_api_sections(report: str) -> str:
    """Drop API sections that only contain incidental helper APIs."""

    lines = clean_markdown(report).splitlines()
    for level, heading, _ in reversed(markdown_sections(report)):
        if normalized_heading(heading) != "implementation apis":
            continue
        start = section_start_index(lines, level, heading)
        if start < 0:
            continue
        end = section_end_index(lines, start, level)
        section_text = "\n".join(lines[start:end])
        if attention_api_signal(section_text):
            continue
        del lines[start:end]
    return clean_markdown("\n".join(lines))


def append_section_before_references(report: str, section: str) -> str:
    lines = clean_markdown(report).splitlines()
    for index, line in enumerate(lines):
        if is_references_heading(line):
            return clean_markdown("\n".join([*lines[:index], "", section, "", *lines[index:]]))
    return clean_markdown(f"{report}\n\n{section}")


def remove_conflicting_missing_evidence_statements(report: str, evidence_text: str) -> str:
    """Drop missing-evidence sentences contradicted by supplied evidence."""

    if not clean_text(evidence_text) or not has_missing_claim(report):
        return report
    resolved_terms = detail_terms(evidence_text) | report_body_terms_without_gaps(report)
    cleaned_lines = []
    for line in clean_markdown(report).splitlines():
        if conflicting_missing_line(line, resolved_terms):
            continue
        if is_references_heading(line) or line.lstrip().startswith("#"):
            if not conflicting_missing_sentence(strip_markdown_markup(line), resolved_terms):
                cleaned_lines.append(line)
            continue
        sentences = split_sentences_preserving_markdown(line)
        kept = [
            sentence
            for sentence in sentences
            if not conflicting_missing_sentence(sentence, resolved_terms)
        ]
        cleaned_line = clean_text(" ".join(kept))
        if conflicting_missing_sentence(cleaned_line, resolved_terms):
            cleaned_line = ""
        if cleaned_line or not clean_text(line):
            cleaned_lines.append(cleaned_line)
    return clean_markdown("\n".join(cleaned_lines))


def remove_resolved_evidence_gap_rows(report: str, evidence_text: str) -> str:
    """Drop Evidence Gaps rows whose topic is already covered by report/evidence."""

    resolved_terms = detail_terms(evidence_text) | report_body_terms_without_gaps(report)
    if not resolved_terms:
        return report
    cleaned_lines = []
    in_gap_section = False
    for line in clean_markdown(report).splitlines():
        if line.lstrip().startswith("#"):
            in_gap_section = "evidence gap" in normalized_heading(line)
            cleaned_lines.append(line)
            continue
        if in_gap_section and resolved_gap_line(line, resolved_terms):
            continue
        cleaned_lines.append(line)
    return remove_empty_sections(remove_empty_table_blocks(clean_markdown("\n".join(cleaned_lines))))


def report_body_terms_without_gaps(report: str) -> set[str]:
    lines = []
    in_gap_section = False
    for line in clean_markdown(report).splitlines():
        if line.lstrip().startswith("#"):
            in_gap_section = "evidence gap" in normalized_heading(line)
        if not in_gap_section and not line_missing_claim(line):
            lines.append(line)
    return detail_terms("\n".join(lines))


def resolved_gap_line(line: str, resolved_terms: set[str]) -> bool:
    text = strip_markdown_markup(line)
    if not clean_text(text) or re.fullmatch(r"\|?\s*-{3,}(?:\s*\|\s*-{3,})*\s*\|?", line.strip()):
        return False
    if line.strip().startswith("|"):
        cells = [clean_text(strip_markdown_markup(cell)) for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or clean_text(cells[0]).lower() in {"planner sub-question", "topic", "requirement"}:
            return False
    terms = detail_terms(text)
    strong_terms = terms & {
        "bahdanau",
        "bleu",
        "formula",
        "glue",
        "keras",
        "linformer",
        "multiheadattention",
        "performer",
        "pytorch",
        "scaled",
        "softmax",
        "tensorflow",
        "transformer",
    }
    overlap = terms & resolved_terms
    return bool(strong_terms & resolved_terms) or len(overlap) >= 2


def remove_empty_table_blocks(markdown: str) -> str:
    lines = clean_markdown(markdown).splitlines()
    cleaned = []
    index = 0
    while index < len(lines):
        if not lines[index].strip().startswith("|"):
            cleaned.append(lines[index])
            index += 1
            continue
        start = index
        while index < len(lines) and lines[index].strip().startswith("|"):
            index += 1
        block = [line for line in lines[start:index] if not malformed_table_row(line)]
        data_rows = [
            line for line in block
            if not re.fullmatch(r"\|?\s*-{3,}(?:\s*\|\s*-{3,})*\s*\|?", line.strip())
        ]
        if len(data_rows) > 1:
            cleaned.extend(block)
    return clean_markdown("\n".join(cleaned))


def remove_malformed_table_rows(markdown: str) -> str:
    lines = [line for line in clean_markdown(markdown).splitlines() if not malformed_table_row(line)]
    return clean_markdown("\n".join(lines))


def malformed_table_row(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    cells = [clean_text(cell) for cell in stripped.strip("|").split("|")]
    return not any(cells)


def split_sentences_preserving_markdown(text: str) -> list[str]:
    if not clean_text(text):
        return [""]
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if clean_text(part)]


def conflicting_missing_sentence(sentence: str, evidence_terms: set[str]) -> bool:
    if not report_missing_sentences(sentence):
        return False
    return has_exact_detail_signal(sentence) or bool(detail_terms(sentence) & evidence_terms)


def conflicting_missing_line(line: str, evidence_terms: set[str]) -> bool:
    text = strip_markdown_markup(line)
    if line.strip().startswith("|") and table_row_missing_claim(line):
        return resolved_gap_line(line, evidence_terms)
    if bullet_missing_claim(line):
        return resolved_gap_line(line, evidence_terms)
    return False


def line_missing_claim(line: str) -> bool:
    text = strip_markdown_markup(line)
    return bool(report_missing_sentences(text)) or bullet_missing_claim(line) or table_row_missing_claim(line)


def bullet_missing_claim(line: str) -> bool:
    text = strip_markdown_markup(line).lower()
    return bool(
        re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", line)
        and any(phrase in text for phrase in ("missing", "not present", "not provided", "not available", "no explicit"))
    )


def table_row_missing_claim(line: str) -> bool:
    cells = [clean_text(strip_markdown_markup(cell)).lower() for cell in line.strip().strip("|").split("|")]
    if len(cells) < 2 or "---" in line:
        return False
    return any(cell in {"missing", "partial"} for cell in cells[1:]) or any(
        phrase in " ".join(cells)
        for phrase in ("not present", "not provided", "not available", "not included", "no explicit")
    )


def repair_report_if_needed(
    client: Any,
    model: str,
    report: str,
    objective: str,
    output_format: str,
    synthesis: str,
    evidence_text: str,
    source_text: str,
    missing_evidence_text: str,
    sources: Sequence[dict[str, Any]],
    max_tokens: int,
    max_context_chars: int,
) -> tuple[str, int, list[str]]:
    """Run one model repair pass for report-level validation issues."""

    repaired = normalize_citation_markers(report)
    repair_count = 0
    issues = report_quality_issues(repaired, evidence_text, sources=sources)
    if not issues:
        return repaired, repair_count, issues

    repair_prompt = f"""Research objective:
{objective}

Requested output format:
{output_format}

Current report:
{repaired}

Detected report issues:
{format_issue_list(issues)}

Synthesis-agent notes:
{synthesis}

Missing-evidence constraints:
{missing_evidence_text}

Supporting evidence chunks:
{evidence_text}

Available sources:
{source_text}

Repair the report.
Requirements:
- Return only the corrected Markdown report.
- Keep the report detailed and technical.
- Remove stale missing-evidence statements when supporting evidence now contains the detail.
- Remove exact formulas, examples, numbers, or complexity notation from sentences that say those details are missing or unsupported.
- If a concrete detail appears in supporting evidence, it may be included with a citation.
- Do not invent unsupported details.
- Preserve unresolved Missing-evidence constraints in a brief "Evidence Gaps" section instead of writing unsupported facts.
- Keep only source markers from Available sources.
- Remove or replace unavailable source markers using only the Available sources and supporting evidence.
- Do not include section-level References blocks.
- End with a References section mapping used source markers to URLs."""

    candidate, _ = groq_chat_text(
        client=client,
        model=model,
        system_prompt="You repair technical reports for evidence consistency, citation validity, and completeness.",
        user_prompt=repair_prompt,
        max_tokens=max_tokens,
        max_context_chars=max_context_chars,
    )
    if clean_text(candidate):
        repaired = candidate
        repair_count += 1
    issues = report_quality_issues(repaired, evidence_text, sources=sources)
    return repaired, repair_count, issues


def report_quality_issues(
    report: str,
    evidence_text: str,
    sources: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    issues = []
    text = clean_text(report)
    if not text:
        return ["report is empty"]
    reference_count = references_heading_count(report)
    if reference_count == 0:
        issues.append("report must include a References section")
    if reference_count > 1:
        issues.append("report must not include section-level References blocks")
    source_indexes = source_index_set(sources or [])
    if source_indexes:
        invalid_citations = unavailable_citation_markers(report, source_indexes)
        if invalid_citations:
            issues.append(f"report uses unavailable citations: {format_citation_indexes(invalid_citations)}")
        missing_reference_entries = missing_reference_entries_for_used_citations(report, source_indexes)
        if missing_reference_entries:
            issues.append(f"report References section is missing cited sources: {format_citation_indexes(missing_reference_entries)}")
    incomplete_sections = incomplete_report_sections(report)
    if incomplete_sections:
        issues.append(f"report contains incomplete sections: {', '.join(incomplete_sections[:3])}")
    issues.extend(report_contract_issues(report, evidence_text))
    if missing_statement_contains_unsupported_detail(report):
        issues.append("report includes exact details inside missing-evidence statements")
    if stale_missing_detail_statement(report, evidence_text):
        issues.append("report may contain stale missing-evidence statements contradicted by supporting evidence")
    return dedupe_preserve_order(issues)


def report_contract_issues(report: str, evidence_text: str) -> list[str]:
    """Generic final-report contract derived from available evidence."""

    if not clean_text(evidence_text):
        return []
    issues = []
    if "executive summary" not in {normalized_heading(heading) for heading, _ in h2_sections(report)}:
        issues.append("report must include an Executive Summary section")
    if evidence_has_formula_signal(evidence_text) and not evidence_has_formula_signal(report):
        issues.append("report omits supported equations or formulas")
    if evidence_has_code_signal(evidence_text) and not evidence_has_code_signal(report):
        issues.append("report omits supported code snippets")
    if evidence_has_api_signal(evidence_text) and not evidence_has_api_signal(report):
        issues.append("report omits supported API details")
    return issues


def evidence_has_formula_signal(text: str) -> bool:
    value = str(text or "")
    return bool(
        re.search(r"\\(?:frac|sum|sqrt|top|operatorname)|\bsoftmax\s*\(|\bAttention\s*\(", value)
        or re.search(r"[A-Za-z0-9_{}()\\]+\s*=\s*[^=\n]{4,}", value)
    )


def evidence_has_code_signal(text: str) -> bool:
    value = str(text or "")
    return "```" in value or bool(re.search(r"\b(import|from|def|class)\s+[A-Za-z_]", value))


def evidence_has_api_signal(text: str) -> bool:
    return attention_api_signal(text)


def attention_api_signal(text: Any) -> bool:
    return bool(attention_api_matches(str(text or "")))


def attention_api_matches(text: str) -> list[re.Match[str]]:
    pattern = (
        r"\b(?:"
        r"torch\.nn\.MultiheadAttention|"
        r"torch\.nn\.functional\.scaled_dot_product_attention|"
        r"tf\.keras\.layers\.MultiHeadAttention|"
        r"keras\.layers\.MultiHeadAttention|"
        r"transformers\.[A-Za-z0-9_.]*(?:Attention|Model)"
        r")\b"
    )
    return list(re.finditer(pattern, str(text or "")))


def references_heading_count(report: str) -> int:
    return sum(1 for line in clean_markdown(report).splitlines() if is_references_heading(line))


def incomplete_report_sections(report: str) -> list[str]:
    incomplete = []
    for heading, section_text in h2_sections(report):
        if normalized_heading(heading) in {"references"}:
            continue
        if markdown_completion_issues(section_text):
            incomplete.append(short_issue_label(heading, max_length=60))
    return incomplete


def h2_sections(markdown: str) -> list[tuple[str, str]]:
    return [(heading, text) for _, heading, text in markdown_sections(markdown)]


def markdown_sections(markdown: str) -> list[tuple[int, str, str]]:
    sections = []
    lines = clean_markdown(markdown).splitlines()
    headings = [
        (index, len(match.group(1)), line.lstrip("#").strip())
        for index, line in enumerate(lines)
        if (match := re.match(r"^(#{2,3})\s+", line))
    ]
    for heading_index, (start, level, heading) in enumerate(headings):
        end = len(lines)
        for next_start, next_level, _ in headings[heading_index + 1 :]:
            if next_level <= level:
                end = next_start
                break
        sections.append((level, heading, "\n".join(lines[start:end]).strip()))
    return sections


def section_start_index(lines: Sequence[str], level: int, heading: str) -> int:
    pattern = re.compile(rf"^#{{{level}}}\s+{re.escape(heading)}\s*$")
    for index, line in enumerate(lines):
        if pattern.match(line):
            return index
    return -1


def section_end_index(lines: Sequence[str], start: int, level: int) -> int:
    for index in range(start + 1, len(lines)):
        match = re.match(r"^(#{2,3})\s+", lines[index])
        if match and len(match.group(1)) <= level:
            return index
    return len(lines)


def trim_incomplete_section_tails(markdown: str) -> str:
    lines = clean_markdown(markdown).splitlines()
    for start, level, _ in reversed([
        (index, len(match.group(1)), line.lstrip("#").strip())
        for index, line in enumerate(lines)
        if (match := re.match(r"^(#{2,3})\s+", line))
    ]):
        end = len(lines)
        for index in range(start + 1, len(lines)):
            match = re.match(r"^(#{2,3})\s+", lines[index])
            if match and len(match.group(1)) <= level:
                end = index
                break
        trim_incomplete_tail_lines(lines, start, end)
    return clean_markdown("\n".join(lines))


def trim_incomplete_tail_lines(lines: list[str], start: int, end: int, max_removed_lines: int = 8) -> None:
    current_end = end
    for _ in range(max_removed_lines):
        section_text = "\n".join(lines[start:current_end]).strip()
        issues = markdown_completion_issues(section_text)
        if not issues or issues == ["section is empty"]:
            return
        index = last_content_line_index(lines, start, current_end)
        if index <= start:
            return
        lines[index] = ""
        current_end = index


def last_content_line_index(lines: Sequence[str], start: int, end: int) -> int:
    index = end - 1
    while index > start and (not lines[index].strip() or re.fullmatch(r"[-*_]{3,}", lines[index].strip())):
        index -= 1
    return index


def remove_empty_sections(markdown: str) -> str:
    lines = clean_markdown(markdown).splitlines()
    for start, level, _ in reversed([
        (index, len(match.group(1)), line.lstrip("#").strip())
        for index, line in enumerate(lines)
        if (match := re.match(r"^(#{2,3})\s+", line))
    ]):
        end = len(lines)
        for index in range(start + 1, len(lines)):
            match = re.match(r"^(#{2,3})\s+", lines[index])
            if match and len(match.group(1)) <= level:
                end = index
                break
        if not section_has_content(lines[start + 1 : end]):
            del lines[start:end]
    return clean_markdown("\n".join(lines))


def remove_empty_math_blocks(markdown: str) -> str:
    """Remove display math blocks that contain no formula content."""

    text = clean_markdown(markdown)
    text = re.sub(r"\\\[\s*\\\]", "", text, flags=re.DOTALL)
    text = re.sub(r"\$\$\s*\$\$", "", text, flags=re.DOTALL)
    return clean_markdown(text)


def remove_incomplete_sections(markdown: str) -> str:
    lines = clean_markdown(markdown).splitlines()
    headings = [
        (index, len(match.group(1)), line.lstrip("#").strip())
        for index, line in enumerate(lines)
        if (match := re.match(r"^(#{2,3})\s+", line))
    ]
    for heading_pos, (start, level, heading) in reversed(list(enumerate(headings))):
        if normalized_heading(heading) == "references":
            continue
        end = len(lines)
        for next_start, next_level, _ in headings[heading_pos + 1 :]:
            if next_level <= level:
                end = next_start
                break
        section_text = "\n".join(lines[start:end]).strip()
        issues = markdown_completion_issues(section_text)
        if issues and issues != ["section is empty"]:
            del lines[start:end]
    return clean_markdown("\n".join(lines))


def section_has_content(lines: Sequence[str]) -> bool:
    for line in lines:
        stripped = line.strip()
        if stripped and not re.fullmatch(r"[-*_]{3,}", stripped):
            return True
    return False


def stale_missing_detail_statement(report: str, evidence_text: str) -> bool:
    if not has_missing_claim(report):
        return False
    return bool(overlapping_detail_terms(missing_claim_texts(report), evidence_text))


def has_missing_claim(text: str) -> bool:
    return bool(missing_claim_texts(text))


def missing_claim_texts(text: str) -> list[str]:
    claims = list(report_missing_sentences(text))
    for line in clean_markdown(text).splitlines():
        if line.strip().startswith("|") and table_row_missing_claim(line):
            claims.append(strip_markdown_markup(line))
        elif line_missing_claim(line):
            claims.append(strip_markdown_markup(line))
    return dedupe_preserve_order(claims)


def report_missing_sentences(text: str) -> list[str]:
    missing_phrases = (
        "not explicitly",
        "not present",
        "not provided",
        "not available",
        "not included",
        "not mentioned",
        "not found",
        "not retrieved",
        "not covered",
        "not supported",
        "not shown",
        "not specified",
        "not contain",
        "not reproduced",
        "cannot be cited",
        "cannot be supplied",
        "cannot be reproduced",
        "no explicit statement",
        "no explicit evidence",
        "no precise citation",
        "no details",
        "no direct citation",
        "no direct source",
        "no evidence",
        "no source",
        "is missing",
        "are missing",
        "is absent",
        "are absent",
        "unavailable",
        "missing detail",
        "evidence is incomplete",
    )
    sentences = re.split(r"(?<=[.!?])\s+|\n+", clean_text(text))
    return [
        sentence
        for sentence in sentences
        if any(phrase in sentence.lower() for phrase in missing_phrases)
    ]


def overlapping_detail_terms(missing_sentences: Sequence[str], evidence_text: str) -> set[str]:
    evidence_terms = detail_terms(evidence_text)
    overlaps = set()
    for sentence in missing_sentences:
        overlaps.update(detail_terms(sentence) & evidence_terms)
    return overlaps


def missing_statement_contains_unsupported_detail(report: str) -> bool:
    for sentence in report_missing_sentences(report):
        if has_exact_detail_signal(sentence):
            return True
    return False


def has_exact_detail_signal(text: str) -> bool:
    value = str(text or "")
    return bool(
        re.search(r"\\\(|\\\[|=[^=]|\\sum|\\frac|O\(", value)
        or re.search(r"\be\.g\.\s*,", value, flags=re.IGNORECASE)
    )


def detail_terms(text: str) -> set[str]:
    stopwords = {
        "about",
        "above",
        "after",
        "also",
        "available",
        "because",
        "before",
        "being",
        "cannot",
        "contain",
        "detail",
        "details",
        "does",
        "evidence",
        "exact",
        "from",
        "include",
        "includes",
        "including",
        "included",
        "missing",
        "present",
        "provided",
        "shown",
        "that",
        "their",
        "there",
        "these",
        "this",
        "with",
        "would",
    }
    terms = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{3,}", clean_text(text)):
        term = normalize_detail_term(token)
        if term not in stopwords:
            terms.add(term)
        if term.startswith("torch."):
            terms.add("pytorch")
        if term.startswith(("tf.", "keras.")):
            terms.add("tensorflow")
    return terms


def normalize_detail_term(token: str) -> str:
    value = token.lower().strip("._+-")
    if value.endswith("ically") and len(value) > 9:
        return value[:-6] + "ic"
    if value.endswith("ally") and len(value) > 7:
        return value[:-4]
    if value.endswith("ies") and len(value) > 5:
        return value[:-3] + "y"
    if value.endswith("s") and len(value) > 5:
        return value[:-1]
    return value


def format_issue_list(issues: Sequence[str]) -> str:
    return "\n".join(f"- {clean_text(issue)}" for issue in issues if clean_text(issue)) or "- No issues."


def format_sources(sources: Sequence[Any]) -> str:
    lines = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        index = source.get("index")
        url = clean_text(source.get("url"))
        title = clean_text(source.get("title")) or url
        if isinstance(index, int) and url:
            lines.append(f"[{index}] {title}\nURL: {url}")
    return "\n\n".join(lines) or "No sources provided."


def sources_with_browser_results(sources: Sequence[Any], browser_results: Sequence[Any]) -> list[Any]:
    merged = [source for source in sources or [] if isinstance(source, dict)]
    seen = {normalize_source_key(clean_text(source.get("url"))) for source in merged}
    next_index = max([source.get("index") for source in merged if isinstance(source.get("index"), int)] or [0]) + 1
    for source in browser_sources(browser_results):
        url = clean_text(source.get("url"))
        key = normalize_source_key(url)
        if not url or key in seen:
            continue
        copied = {
            "index": next_index,
            "url": url,
            "title": clean_text(source.get("title")) or url,
        }
        merged.append(copied)
        seen.add(key)
        next_index += 1
    return merged


def browser_sources(browser_results: Sequence[Any]) -> list[dict[str, Any]]:
    sources = []
    if not isinstance(browser_results, list):
        return sources
    for result in browser_results:
        if not isinstance(result, dict):
            continue
        for source in result.get("sources", []) or []:
            if isinstance(source, dict):
                sources.append(source)
    return sources


def format_memory_signal_evidence(
    report_context: dict[str, Any],
    sources: Sequence[dict[str, Any]],
    citation_aliases: dict[int, int] | None,
    existing_evidence: str,
) -> str:
    citation_aliases = citation_aliases or {}
    source_index_by_url = {
        normalize_source_key(clean_text(source.get("url"))): citation_aliases.get(source.get("index"), source.get("index"))
        for source in sources
        if isinstance(source, dict) and isinstance(source.get("index"), int) and clean_text(source.get("url"))
    }
    existing = clean_text(existing_evidence).lower()
    topic_terms = detail_terms(
        " ".join(
            [
                clean_text(report_context.get("objective")),
                *[clean_text(question) for question in report_context.get("planner_questions", []) or []],
            ]
        )
    )
    candidates = []
    sequence = 0
    seen = set()
    for source in browser_sources(report_context.get("browser_results", [])):
        marker_index = source_index_by_url.get(normalize_source_key(clean_text(source.get("url"))))
        if not isinstance(marker_index, int):
            continue
        snippets = high_signal_snippets(source.get("full_content") or source.get("content_preview"), existing)
        for snippet in snippets[:3]:
            key = clean_text(snippet).lower()
            if not key or key in seen or key in existing:
                continue
            if topic_terms and not (detail_terms(snippet) & topic_terms):
                continue
            seen.add(key)
            sequence += 1
            candidates.append((signal_priority(snippet), -sequence, f"[{marker_index}] {snippet}"))

    lines = []
    used_chars = 0
    for _, _, line in sorted(candidates, reverse=True):
        if used_chars + len(line) > DEFAULT_REPORT_MEMORY_EVIDENCE_CHARS:
            continue
        lines.append(line)
        used_chars += len(line)
    return "\n".join(dedupe_preserve_order(lines))


def high_signal_snippets(text: Any, existing_evidence: str = "", limit: int = 6) -> list[str]:
    snippets = []
    for sentence in evidence_sentences(text):
        if not has_report_signal(sentence):
            continue
        snippet = focused_signal_snippet(sentence)
        if snippet and snippet.lower() not in existing_evidence:
            snippets.append(snippet)
        if len(snippets) >= limit:
            break
    return dedupe_preserve_order(snippets)


def evidence_sentences(text: Any) -> list[str]:
    value = clean_text(text)
    if not value:
        return []
    return [
        clean_text(part)
        for part in re.split(r"(?<=[.!?])\s+|\n+", value)
        if clean_text(part)
    ]


def has_report_signal(text: str) -> bool:
    value = clean_text(text)
    return bool(
        evidence_has_formula_signal(value)
        or evidence_has_api_signal(value)
        or re.search(r"\b\d+(?:\.\d+)?\s*(?:BLEU|GLUE|accuracy|perplexity|F1|AUC|%|tokens?/sec)\b", value, re.I)
        or re.search(r"\b(?:benchmark|score|improves?|achieves?|outperform|complexity|low-rank|linear|quadratic)\b", value, re.I)
        or re.search(r"\bO\([^)]+\)", value)
    )


def signal_priority(text: str) -> int:
    value = clean_text(text)
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:BLEU|GLUE|accuracy|perplexity|F1|AUC|%|tokens?/sec)\b", value, re.I):
        return 5
    if evidence_has_formula_signal(value) or re.search(r"\bO\([^)]+\)", value):
        return 4
    if evidence_has_api_signal(value):
        return 3
    if re.search(r"\b(?:complexity|low-rank|linear|quadratic|benchmark|score)\b", value, re.I):
        return 2
    return 1


def focused_signal_snippet(sentence: str, max_chars: int = 420) -> str:
    text = clean_text(sentence)
    if len(text) <= max_chars:
        return text
    signal = re.search(
        r"\b\d+(?:\.\d+)?\s*(?:BLEU|GLUE|accuracy|perplexity|F1|AUC|%|tokens?/sec)\b|"
        r"\bO\([^)]+\)|\b(?:softmax|torch\.|tf\.|keras\.|low-rank|complexity)\b",
        text,
        re.I,
    )
    center = signal.start() if signal else 0
    start = max(0, center - max_chars // 2)
    end = min(len(text), start + max_chars)
    return text[start:end].strip(" ,.;")


def format_evidence_coverage_brief(
    planner_questions: Sequence[str],
    synthesis: str,
    evidence_text: str,
    missing_evidence_text: str,
) -> str:
    supported_signals = high_signal_snippets(evidence_text, limit=10)
    lines = [
        "Use this brief to decide covered, partial, and missing sections before writing.",
        "If a topic has supported details and unresolved gaps, write the supported details first and put only the unresolved details in Evidence Gaps.",
    ]
    if supported_signals:
        lines.append("Supported evidence signals:")
        lines.extend(f"- {signal}" for signal in supported_signals[:8])
    remaining_gaps = unresolved_gap_brief(missing_evidence_text, evidence_text)
    if remaining_gaps:
        lines.append("Potential unresolved gaps from synthesis:")
        lines.extend(f"- {gap}" for gap in remaining_gaps[:6])
    if planner_questions:
        lines.append("Planner-question coverage rule: every planner question needs either supported content or a precise unresolved gap.")
    return clean_markdown("\n".join(lines))


def unresolved_gap_brief(missing_evidence_text: str, evidence_text: str) -> list[str]:
    evidence_terms = detail_terms(evidence_text)
    gaps = []
    for line in clean_markdown(missing_evidence_text).splitlines():
        gap = clean_text(line.lstrip("- "))
        if not gap:
            continue
        overlap = detail_terms(gap) & evidence_terms
        if overlap and has_report_signal(gap):
            continue
        gaps.append(gap)
    return dedupe_preserve_order(gaps)


def format_missing_evidence_constraints(synthesis: str) -> str:
    """Extract gap notes that the report must not turn into unsupported details."""

    constraints = missing_evidence_constraints(synthesis)
    if not constraints:
        return "No missing-evidence constraints were identified."
    return "\n".join(f"- {constraint}" for constraint in constraints)


def missing_evidence_constraints(synthesis: str) -> list[str]:
    constraints = []
    for line in clean_markdown(synthesis).splitlines():
        line_text = clean_text(line)
        if not line_text:
            continue
        table_constraint = missing_evidence_from_table_row(line_text)
        if table_constraint:
            constraints.append(table_constraint)
            continue
        if line_text.startswith("|"):
            continue
        lowered = line_text.lower()
        if any(
            phrase in lowered
            for phrase in (
                "missing detail",
                "missing evidence",
                "not present in the retrieved",
                "not present in the cited",
                "not quoted in the retrieved",
                "do not add",
            )
        ):
            constraints.append(strip_markdown_markup(line_text))
    return dedupe_preserve_order(constraints)


def missing_evidence_from_table_row(line: str) -> str:
    if not line.startswith("|") or "---" in line:
        return ""
    cells = [strip_markdown_markup(cell) for cell in line.strip("|").split("|")]
    if len(cells) < 2:
        return ""
    first_cell = clean_text(cells[0]).lower()
    if first_cell in {"requirement", "planner sub-question", "source", "status"}:
        return ""
    status = clean_text(cells[1]).lower()
    if any(word in status for word in ("covered", "resolved", "available")):
        return ""
    notes = clean_text(" ".join(cells[2:]))
    if (
        "missing" not in status
        and "partial" not in status
        and "missing" not in notes.lower()
        and "not present" not in notes.lower()
    ):
        return ""
    return clean_text(f"{cells[0]}: {notes}")


def missing_evidence_constraint_count(text: str) -> int:
    return sum(1 for line in clean_markdown(text).splitlines() if line.startswith("- "))


def strip_markdown_markup(text: str) -> str:
    value = re.sub(r"`([^`]*)`", r"\1", str(text or ""))
    value = re.sub(r"[*_#]+", "", value)
    value = re.sub(r"\[(\d+)\]", "", value)
    return clean_text(value)


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


def dedupe_sources_by_url(sources: Sequence[Any]) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Keep one citation marker per unique URL and alias duplicate markers."""

    deduped = []
    first_index_by_url: dict[str, int] = {}
    citation_aliases: dict[int, int] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        index = source.get("index")
        url = clean_text(source.get("url"))
        if not isinstance(index, int) or not url:
            continue
        key = normalize_source_key(url)
        if key in first_index_by_url:
            citation_aliases[index] = first_index_by_url[key]
            continue
        first_index_by_url[key] = index
        deduped.append(source)
    return deduped, citation_aliases


def normalize_source_key(url: str) -> str:
    key = clean_text(url).lower().rstrip("/")
    arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/([^?#/]+)", key)
    if arxiv_match:
        return f"arxiv.org/abs/{arxiv_match.group(1).removesuffix('.pdf')}"
    return key


def write_report_file(
    report_payload: dict[str, Any],
    memory_path: str = "data/shared_memory.json",
    report_path: str | None = None,
) -> str:
    """Save report Markdown to disk and return its path."""

    report = clean_markdown(report_payload.get("report"))
    if not report:
        raise ValueError("report_payload.report is required")

    output_path = Path(report_path) if report_path else default_report_path(report_payload, memory_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    return str(output_path)


def default_report_path(report_payload: dict[str, Any], memory_path: str) -> Path:
    memory_parent = Path(memory_path).parent
    if str(memory_parent) in {"", "."}:
        output_dir = Path(DEFAULT_REPORT_OUTPUT_DIR)
    else:
        output_dir = memory_parent / "reports"
    objective = clean_text(report_payload.get("objective")) or "research-report"
    return output_dir / f"{slugify_filename(objective)}.md"


def slugify_filename(text: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", clean_text(text).lower()).strip("-")
    return (slug[:max_length].strip("-") or "research-report")


def format_supporting_evidence(
    report_context: dict[str, Any],
    citation_aliases: dict[int, int] | None = None,
) -> str:
    chunks = report_context.get("supporting_chunks") or report_context.get("retrieved_chunks") or []
    if not isinstance(chunks, list):
        return "No supporting chunks provided."

    citation_aliases = citation_aliases or {}
    blocks = []
    used_chars = 0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        source_index = chunk.get("source_index")
        if not isinstance(source_index, int):
            source_index = chunk.get("index")
        if isinstance(source_index, int):
            source_index = citation_aliases.get(source_index, source_index)
        url = clean_text(chunk.get("url"))
        title = clean_text(chunk.get("title")) or url or "Source"
        content = clean_text(chunk.get("content"))
        if not content:
            continue
        content = content[:DEFAULT_REPORT_AGENT_CHUNK_CHARS].strip()
        marker = f"[{source_index}]" if isinstance(source_index, int) else "[uncited]"
        block = f"{marker} {title}\nURL: {url}\nEvidence: {content}"
        if used_chars + len(block) > DEFAULT_REPORT_AGENT_CONTEXT_CHARS // 2:
            break
        blocks.append(block)
        used_chars += len(block)
    return "\n\n".join(blocks) or "No supporting chunks provided."


def normalize_citation_markers(text: str) -> str:
    normalized = clean_markdown(text)
    normalized = re.sub(r"【\s*(\d+)(?:[^】]*)?】", r"[\1]", normalized)
    normalized = re.sub(r"\[\s*(\d+)\s*\]", r"[\1]", normalized)
    return normalized


def remap_citation_markers(text: str, citation_aliases: dict[int, int]) -> str:
    if not citation_aliases:
        return text

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return f"[{citation_aliases.get(index, index)}]"

    return re.sub(r"\[(\d+)\]", replace, text)


def clean_markdown(value: Any) -> str:
    """Normalize Markdown spacing without collapsing line breaks."""

    text = strip_thinking_blocks(str(value or "")).replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    normalized = "\n".join(lines).strip()
    normalized = re.sub(r"\n{4,}", "\n\n\n", normalized)
    return normalized


def strip_thinking_blocks(text: str) -> str:
    """Remove leaked model reasoning blocks before report planning or validation."""

    return re.sub(r"<think\b[^>]*>.*?</think>", "", str(text or ""), flags=re.IGNORECASE | re.DOTALL)
