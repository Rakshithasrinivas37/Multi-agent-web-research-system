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
DEFAULT_REPORT_EVIDENCE_PACKET_CHARS = 3600
DEFAULT_REPORT_MIN_QUESTION_SECTION_WORDS = 30


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
        source_priority_text = format_source_priority_guidance(sources)
        evidence_text = format_supporting_evidence(report_context, citation_aliases=citation_aliases)
        memory_evidence_text = format_memory_signal_evidence(report_context, sources, citation_aliases, evidence_text)
        if memory_evidence_text:
            evidence_text = clean_markdown(f"Memory evidence signals:\n{memory_evidence_text}\n\n{evidence_text}")
        evidence_text = remove_unavailable_citation_markers(evidence_text, available_source_indexes)
        planner_questions = [clean_text(question) for question in report_context.get("planner_questions", []) or []]
        evidence_packet = format_planner_evidence_packet(
            report_context=report_context,
            planner_questions=planner_questions,
            sources=sources,
            citation_aliases=citation_aliases,
        )
        if evidence_packet:
            evidence_text = clean_markdown(f"Planner evidence packet:\n{evidence_packet}\n\n{evidence_text}")
        missing_evidence_text = format_missing_evidence_constraints(synthesis)
        coverage_brief = format_evidence_coverage_brief(
            planner_questions=planner_questions,
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
        single_prompt = build_single_report_prompt(
            objective=objective,
            output_format=output_format_text,
            citation_policy=citation_policy,
            planner_questions=planner_questions,
            evidence_packet=evidence_packet,
            coverage_brief=coverage_brief,
            synthesis=synthesis,
            missing_evidence_text=missing_evidence_text,
            evidence_text=evidence_text,
            source_text=source_text,
            source_priority_text=source_priority_text,
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
        report, report_issues = finalize_report(
            report=report,
            synthesis=synthesis,
            evidence_text=evidence_text,
            sources=sources,
        )
        missing_sub_questions = missing_sub_question_coverage(report, planner_questions)
        if missing_sub_questions:
            labels = ", ".join(short_issue_label(question, 80) for question in missing_sub_questions[:3])
            raise ValueError(f"report_agent produced invalid report: report does not cover planner sub-questions: {labels}")
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
                "report_repair_count": 0,
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
    evidence_packet: str,
    coverage_brief: str,
    synthesis: str,
    missing_evidence_text: str,
    evidence_text: str,
    source_text: str,
    source_priority_text: str,
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
- Build the main body as one "##" section per planner sub-question, in the same order as the Report section outline.
- For every planner sub-question, explain the concept in prose first: what it is, how it works, why it matters, and the evidence-backed details.
- Do not make the report equation-heavy. Include equations only when the planner sub-question asks for a formula, equation, math, mathematical formulation, score function, or core formulation.
- If a planner sub-question does not ask for equations, summarize any math briefly in prose and focus on applications, architecture, benchmarks, limitations, comparisons, or implementation details as requested.
- Answer every planner sub-question explicitly using the available evidence; do not let the executive summary be the only coverage for a topic.
- Prefer concrete definitions, structured comparisons, measurements, examples, and implementation details when supported by synthesis or supporting chunks.
- Use synthesis-agent notes and supporting chunks as the only factual basis; do not fill gaps from model prior knowledge.
- Explain important technical terms or notation when applicable, and use tables when they make comparisons clearer.
- Include exact technical details only when they are supported by synthesis or supporting chunks.
- Include supported API signatures and code snippets when they appear in synthesis or supporting chunks.
- When multiple sources support the same claim, cite primary/official sources first.
- Treat partial evidence as usable: write the supported part, then place only the unresolved part in "Evidence Gaps".
- Treat the Planner evidence packet below as the source of truth for covered, partial, and missing topics.
- If the Planner evidence packet marks a topic as Covered, write the supported answer directly and do not repeat missing/partial statements from synthesis for that topic.
- If the Planner evidence packet marks a topic as Partial, write the supported details first and put only the exact missing subtopic in Evidence Gaps.
- If the Planner evidence packet marks a topic as Missing, mention it only in Evidence Gaps.
- Reconcile Missing-evidence constraints against Supporting evidence chunks before writing.
- If supporting chunks contain a detail that synthesis previously marked missing, include the supported detail and do not say it is missing.
- Do not generalize a missing detail to a broader topic; include supported parts and name only the exact unsupported detail.
- Do not reproduce details listed in Missing-evidence constraints unless they are present in Supporting evidence chunks.
- If a missing item remains important, state only that the provided evidence describes it but does not include the exact detail.
- Include a brief "Evidence Gaps" section when Missing-evidence constraints identify partial or missing required items.
- Cite claims using only plain source markers from Available sources, exactly like [1], [2], [3].
- For precise claims, prefer original papers, official docs, academic sources, or authoritative surveys.
- Do not cite sources that are not listed.
- Before using a citation marker, verify that the cited source title or URL matches the named concept in the sentence.
- Do not use citation formats like 【1】, footnotes, line citations, or URLs inline.
- Do not write placeholder citations such as [uncited], [citation needed], or [source needed].
- Do not add named methods, models, papers, or variants unless they appear in the synthesis notes, supporting evidence, or source list.
- Do not write placeholder citation markers like [Evidence Gap]; cite a listed source or state the exact unresolved gap in prose.
- End with a References section mapping only used source markers to source URLs.
- If evidence is incomplete, mention the limitation instead of inventing details.
- Before finalizing, remove contradictions such as saying a detail is missing and then including that detail.

Available sources:
{source_text}

Source priority:
{source_priority_text}

Planner sub-questions that must be answered:
{format_planner_questions(planner_questions)}

Report section outline:
{format_report_section_outline(planner_questions)}

Planner evidence packet:
{compact_markdown(evidence_packet, DEFAULT_REPORT_EVIDENCE_PACKET_CHARS)}

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

Generate the final report using the Planner evidence packet and Supporting evidence chunks as primary truth. Use synthesis-agent notes only as secondary organization context."""


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


def format_report_section_outline(questions: Sequence[str]) -> str:
    clean_questions = [clean_text(question) for question in questions if clean_text(question)]
    if not clean_questions:
        return "- Use clear sections that directly answer the research objective."
    lines = []
    for index, question in enumerate(clean_questions, start=1):
        equation_policy = (
            "include supported equations"
            if question_requests_equations(question)
            else "use prose; avoid equations unless essential to the answer"
        )
        lines.append(f"## {index}. {short_issue_label(question, 100)}")
        lines.append(f"   - Explain the topic in detail from evidence; {equation_policy}.")
    return "\n".join(lines)


def question_requests_equations(question: str) -> bool:
    lowered = clean_text(question).lower()
    return any(
        phrase in lowered
        for phrase in (
            "equation",
            "formula",
            "mathematical",
            "math",
            "formulation",
            "score function",
            "core formulation",
        )
    )


def missing_sub_question_coverage(report: str, planner_questions: Sequence[str]) -> list[str]:
    """Return planner questions whose specific terms are not reflected in the report."""

    questions = [clean_text(question) for question in planner_questions if clean_text(question)]
    if not questions:
        return []
    common_terms = common_question_terms(questions)
    question_terms = {"what", "when", "where", "which", "whose", "why", "does", "used", "have", "been", "e.g", "eg"}
    missing = []
    has_sections = bool(h2_sections(report))
    for question in questions:
        terms = [term for term in detail_terms(question) if term not in common_terms and term not in question_terms]
        if not terms:
            continue
        section_text = best_question_section_text(report, terms)
        if not section_text:
            missing.append(question)
            continue
        section_terms = detail_terms(section_text)
        named_terms = named_topic_candidates_from_text(question)
        overlap_count = sum(1 for term in set(terms) if report_has_question_term(section_terms, term))
        required_overlap = min(2, len(set(terms))) if has_sections else 1
        too_short = has_sections and meaningful_word_count(section_text) < DEFAULT_REPORT_MIN_QUESTION_SECTION_WORDS
        missing_named_terms = has_sections and any(not topic_in_text(term, section_text) for term in named_terms)
        if overlap_count < required_overlap or too_short or missing_named_terms:
            missing.append(question)
    return missing


def best_question_section_text(report: str, terms: Sequence[str]) -> str:
    sections = h2_sections(report)
    if not sections:
        report_terms = detail_terms(report)
        return report if any(report_has_question_term(report_terms, term) for term in terms) else ""
    best_text = ""
    best_score = 0
    for heading, section_text in sections:
        if normalized_heading(heading) in {"executive summary", "references", "evidence gaps"}:
            continue
        section_terms = detail_terms(f"{heading} {section_text}")
        score = sum(1 for term in set(terms) if report_has_question_term(section_terms, term))
        if score > best_score:
            best_score = score
            best_text = section_text
    return best_text if best_score else ""


def meaningful_word_count(text: str) -> int:
    return len(re.findall(r"\b[A-Za-z][A-Za-z0-9-]*\b", strip_markdown_markup(text)))


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
    body = remove_evidence_gap_placeholders(body)
    body = remove_placeholder_citations(body)
    body = remove_unavailable_citation_markers(body, source_index_set(sources))
    body = correct_mismatched_topic_citations(body, sources)
    body = remove_empty_math_blocks(body)
    body = remove_malformed_table_rows(body)
    body = repair_executive_summary_section(body)
    body = trim_incomplete_section_tails(body)
    body = remove_empty_sections(body)
    body = remove_incomplete_sections(body)
    body = remove_empty_evidence_gap_sections(body)
    body = remove_empty_sections(body)
    body = ensure_executive_summary_section(body)
    body = repair_executive_summary_section(body)
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


def repair_executive_summary_section(markdown: str) -> str:
    """Make the summary validation-safe without changing report facts."""

    lines = clean_markdown(markdown).splitlines()
    bounds = find_summary_section_bounds(lines)
    if not bounds:
        return markdown
    start, end = bounds
    body_lines = lines[start + 1 : end]
    if not section_has_content(body_lines):
        lines[start + 1 : end] = [fallback_executive_summary(lines[:start] + lines[end:])]
        return clean_markdown("\n".join(lines))

    last_index = last_content_line_index(lines, start, end)
    if last_index > start and markdown_completion_issues("\n".join(lines[start:last_index + 1])) == ["section appears to stop mid-sentence"]:
        lines[last_index] = complete_sentence(lines[last_index])
    return clean_markdown("\n".join(lines))


def complete_sentence(line: str) -> str:
    text = line.rstrip()
    if not text or text[-1] in ".!?)]}`'\"":
        return text
    return f"{text}."


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
        return complete_sentence(text[:500].rstrip())
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


def correct_mismatched_topic_citations(markdown: str, sources: Sequence[dict[str, Any]]) -> str:
    """Repair cited topic markers when another available source clearly matches."""

    lines = []
    for line in clean_markdown(markdown).splitlines():
        updated = line
        remove_line = False
        for topic, signals in topic_source_signals().items():
            updated, unsupported = correct_topic_citation_line(updated, topic, signals, sources)
            if unsupported:
                remove_line = True
                break
        if not remove_line:
            lines.append(updated)
    return clean_markdown("\n".join(lines))


def correct_topic_citation_line(
    line: str,
    topic: str,
    signals: Sequence[str],
    sources: Sequence[dict[str, Any]],
) -> tuple[str, bool]:
    pattern = topic_citation_pattern(topic)
    if not pattern.search(line):
        return line, False
    matching_index = matching_topic_source_index(sources, signals)
    source_by_index = source_by_index_map(sources)

    def replace(match: re.Match[str]) -> str:
        index = int(match.group("index"))
        source = source_by_index.get(index)
        if source_matches_topic(source, signals):
            return match.group(0)
        if matching_index:
            return f"{match.group('prefix')}[{matching_index}]"
        return ""

    updated = pattern.sub(replace, line)
    if matching_index:
        return updated, False
    return updated, clean_text(updated) != clean_text(line)


def topic_citation_mismatches(report: str, sources: Sequence[dict[str, Any]]) -> list[str]:
    mismatches = []
    source_by_index = source_by_index_map(sources)
    for line in clean_markdown(report).splitlines():
        for topic, signals in topic_source_signals().items():
            for match in topic_citation_pattern(topic).finditer(line):
                index = int(match.group("index"))
                if not source_matches_topic(source_by_index.get(index), signals):
                    mismatches.append(f"{topic} [{index}]")
    return dedupe_preserve_order(mismatches)


def topic_citation_pattern(topic: str) -> re.Pattern[str]:
    topic_pattern = re.escape(topic).replace(r"\ ", r"\s+")
    boundary = r"\b(?:linformer|performer|conformer|bahdanau|luong|deberta|vision\s+transformer|vit)\b"
    return re.compile(
        rf"(?P<prefix>\b{topic_pattern}\b(?:(?!{boundary}).){{0,160}}?)\[(?P<index>\d+)\]",
        flags=re.IGNORECASE,
    )


def topic_source_signals() -> dict[str, tuple[str, ...]]:
    return {
        "bahdanau": ("bahdanau", "1409.0473"),
        "conformer": ("conformer", "2005.08100"),
        "deberta": ("deberta", "superglue", "syncedreview"),
        "linformer": ("linformer", "2006.04768"),
        "luong": ("luong", "1508.04025"),
        "performer": ("performer", "2009.14794"),
        "vision transformer": ("vision transformer", "2010.11929", "vit"),
        "vit": ("vision transformer", "2010.11929", "vit"),
    }


def source_by_index_map(sources: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {
        source["index"]: source
        for source in sources
        if isinstance(source, dict) and isinstance(source.get("index"), int)
    }


def matching_topic_source_index(sources: Sequence[dict[str, Any]], signals: Sequence[str]) -> int | None:
    for source in sources:
        if source_matches_topic(source, signals) and isinstance(source.get("index"), int):
            return source["index"]
    return None


def source_matches_topic(source: dict[str, Any] | None, signals: Sequence[str]) -> bool:
    if not isinstance(source, dict):
        return False
    haystack = clean_text(f"{source.get('title', '')} {source.get('url', '')}").lower()
    return any(signal.lower() in haystack for signal in signals)


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
    report: str,
    synthesis: str,
    evidence_text: str,
    sources: Sequence[dict[str, Any]],
) -> tuple[str, list[str]]:
    if not clean_text(report):
        raise ValueError("report_agent produced empty report")

    repaired = normalize_report_for_validation(report, sources, evidence_text, synthesis=synthesis)
    issues = report_quality_issues(repaired, evidence_text, sources=sources)
    blocking_issues = hard_report_issues(issues)
    if blocking_issues:
        raise ValueError(f"report_agent produced invalid report: {'; '.join(blocking_issues)}")
    return repaired, issues


def hard_report_issues(issues: Sequence[str]) -> list[str]:
    """Return report issues that should block saving the final report."""

    return [issue for issue in issues if clean_text(issue)]


def normalize_report_for_validation(
    report: str,
    sources: Sequence[dict[str, Any]],
    evidence_text: str,
    synthesis: str = "",
) -> str:
    resolved_evidence_text = covered_synthesis_signal_text(synthesis, evidence_text)
    normalized = normalize_final_report(report, sources)
    normalized = remove_weak_implementation_api_sections(normalized)
    normalized = remove_unsupported_named_topic_lines(normalized, resolved_evidence_text, sources)
    normalized = remove_resolved_evidence_gap_rows(normalized, resolved_evidence_text)
    normalized = remove_conflicting_missing_evidence_statements(normalized, resolved_evidence_text)
    return normalize_final_report(normalized, sources)


def covered_synthesis_signal_text(synthesis: str, evidence_text: str) -> str:
    """Add synthesis-covered topics to evidence signals used for cleanup."""

    covered = []
    for line in clean_markdown(synthesis).splitlines():
        signal = covered_synthesis_table_signal(line)
        if signal:
            covered.append(signal)
    if not covered:
        return evidence_text
    return clean_markdown(f"{evidence_text}\n\nSynthesis covered topics:\n" + "\n".join(covered))


def covered_synthesis_table_signal(line: str) -> str:
    if not line.startswith("|") or "---" in line:
        return ""
    cells = [clean_text(strip_markdown_markup(cell)) for cell in line.strip("|").split("|")]
    if len(cells) < 2:
        return ""
    topic = cells[0]
    status = cells[1].lower()
    if topic.lower() in {"requirement", "planner question", "planner sub-question", "topic"}:
        return ""
    if "partial" in status or "missing" in status:
        return ""
    if not any(word in status for word in ("covered", "resolved", "available", "strong")):
        return ""
    signal = clean_text(" ".join([topic, *cells[2:]]))
    signal = clean_text(re.sub(r"[-‑–—/]", " ", signal))
    if "equation" in signal.lower():
        signal = f"{signal} formula formulation"
    return signal


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
    if not any(cells):
        return True
    if re.fullmatch(r"\|?\s*:?-{3,}:?(?:\s*\|\s*:?-{3,}:?)*\s*\|?", stripped):
        return False
    return all(cell in {"-", "--", "---", "—", "–", "n/a", "na"} for cell in cells)


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
        topic_mismatches = topic_citation_mismatches(report, sources or [])
        if topic_mismatches:
            issues.append(f"report cites mismatched sources for named topics: {', '.join(topic_mismatches[:5])}")
    incomplete_sections = incomplete_report_sections(report)
    if incomplete_sections:
        issues.append(f"report contains incomplete sections: {', '.join(incomplete_sections[:3])}")
    issues.extend(report_contract_issues(report, evidence_text))
    if placeholder_citation_markers(report):
        issues.append("report contains placeholder citation markers")
    unsupported_topics = unsupported_named_topic_mentions(report, evidence_text, sources or [])
    if unsupported_topics:
        issues.append(f"report mentions unsupported named topics: {', '.join(unsupported_topics[:5])}")
    if missing_statement_contains_unsupported_detail(report):
        issues.append("report includes exact details inside missing-evidence statements")
    if stale_missing_detail_statement(report, evidence_text):
        issues.append("report may contain stale missing-evidence statements contradicted by supporting evidence")
    return dedupe_preserve_order(issues)


def placeholder_citation_markers(text: str) -> list[str]:
    matches = re.findall(
        r"\[\s*(uncited|citation needed|source needed|needs citation|no citation)\s*\]",
        clean_markdown(text),
        flags=re.IGNORECASE,
    )
    return dedupe_preserve_order([clean_text(match).lower() for match in matches])


def unsupported_named_topic_mentions(
    report: str,
    evidence_text: str,
    sources: Sequence[dict[str, Any]],
) -> list[str]:
    evidence_haystack = evidence_topic_text(evidence_text, sources)
    return [
        topic
        for topic in report_named_topic_candidates(report)
        if not topic_in_text(topic, evidence_haystack)
    ]


def remove_unsupported_named_topic_lines(
    report: str,
    evidence_text: str,
    sources: Sequence[dict[str, Any]],
) -> str:
    unsupported = set(unsupported_named_topic_mentions(report, evidence_text, sources))
    lines = []
    in_evidence_gaps = False
    for line in clean_markdown(report).splitlines():
        if line.startswith("## "):
            in_evidence_gaps = normalized_heading(line.lstrip("#").strip()) == "evidence gaps"
        text = strip_markdown_markup(line)
        if unsupported_inference_line(text):
            continue
        if re.match(r"^#{2,3}\s+", line):
            lines.append(line)
            continue
        if not in_evidence_gaps and any(topic_in_text(topic, text) for topic in unsupported):
            continue
        lines.append(line)
    return clean_markdown("\n".join(lines))


def unsupported_inference_line(text: str) -> bool:
    lowered = clean_text(text).lower()
    return any(
        phrase in lowered
        for phrase in (
            "not listed but inferred",
            "inferred from the",
            "inferred from provided",
            "not directly provided but inferred",
        )
    )


def report_named_topic_candidates(report: str) -> list[str]:
    """Extract likely named methods/models from repeated mentions and table rows."""

    counts: dict[str, int] = {}
    table_candidates: set[str] = set()
    in_evidence_gaps = False
    for line in strip_all_references_blocks(clean_markdown(report)).splitlines():
        if line.startswith("## "):
            in_evidence_gaps = normalized_heading(line.lstrip("#").strip()) == "evidence gaps"
        text = clean_text(strip_markdown_markup(line))
        if not text or line.lstrip().startswith("#") or in_evidence_gaps:
            continue
        if line.strip().startswith("|") and "---" not in line:
            table_candidates.update(named_topic_candidates_from_text(line))
        if named_topic_context(text):
            for candidate in named_topic_candidates_from_text(text):
                counts[candidate] = counts.get(candidate, 0) + 1
    return [
        topic
        for topic in dedupe_preserve_order([*table_candidates, *counts])
        if topic in table_candidates or counts.get(topic, 0) > 1 or distinctive_named_topic(topic)
    ]


def evidence_topic_text(evidence_text: str, sources: Sequence[dict[str, Any]]) -> str:
    source_text = " ".join(
        f"{source.get('title', '')} {source.get('url', '')}"
        for source in sources
        if isinstance(source, dict)
    )
    return clean_text(f"{evidence_text} {source_text}").lower()


def named_topic_context(text: str) -> bool:
    lowered = clean_text(text).lower()
    return any(
        word in lowered
        for word in (
            "model",
            "architecture",
            "variant",
            "method",
            "paper",
            "mechanism",
            "attention",
            "uses",
            "reduces",
            "combines",
            "introduced",
            "proposed",
            "implements",
            "outperforms",
            "achieves",
        )
    )


def named_topic_candidates_from_text(text: str) -> list[str]:
    candidates = []
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)?(?:\s+[A-Z][A-Za-z0-9]+)?\b", strip_markdown_markup(text)):
        candidate = clean_text(match.group(0))
        if len(candidate) >= 4 and candidate.lower() not in named_topic_stopwords():
            candidates.append(candidate.lower())
    return dedupe_preserve_order(candidates)


def distinctive_named_topic(text: str) -> bool:
    value = clean_text(text)
    if value.lower() in named_topic_stopwords():
        return False
    return bool(re.search(r"[A-Z]", value[1:]) or re.search(r"\d", value) or re.fullmatch(r"[A-Z]{2,}s?", value))


def named_topic_stopwords() -> set[str]:
    return {
        "attention",
        "evidence",
        "references",
        "summary",
        "executive summary",
        "section",
        "table",
        "figure",
        "field",
        "value",
        "complete",
        "broken",
        "row",
        "detail",
        "details",
        "reference",
        "core idea",
        "typical use",
        "cases",
        "number",
        "dimensionality",
        "employs",
        "introduces",
        "optimizes",
        "sequence",
        "source",
        "sources",
        "api",
        "apis",
        "model",
        "models",
        "decoder",
        "encoder",
        "method",
        "methods",
        "architecture",
        "variant",
        "variants",
        "transformer",
        "neural",
        "language",
        "image",
        "translation",
        "classification",
        "benchmark",
        "benchmarks",
        "starting",
        "modern",
        "while",
        "what",
        "this",
        "proposed",
        "each",
        "task",
        "self",
        "transformers",
        "english",
        "german",
        "bleu",
        "comparable",
        "formally",
        "intuitively",
        "specifically",
        "typically",
        "generally",
        "overall",
        "therefore",
        "however",
    }


def topic_in_text(topic: str, text: str) -> bool:
    pattern = re.escape(topic).replace(r"\ ", r"\s+")
    return bool(re.search(rf"\b{pattern}\b", clean_markdown(text), flags=re.IGNORECASE))


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
        heading_key = normalized_heading(heading)
        if heading_key in {"references"}:
            continue
        if heading_key == "executive summary" and section_has_content(section_text.splitlines()[1:]):
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


def remove_evidence_gap_placeholders(markdown: str) -> str:
    """Remove inline placeholder markers that are not real source citations."""

    text = clean_markdown(markdown)
    text = re.sub(r"\s*\[\s*[-–—]*\s*Evidence\s+Gap\s*[-–—]*\s*\]", "", text, flags=re.IGNORECASE)
    return clean_markdown(text)


def remove_placeholder_citations(markdown: str) -> str:
    """Remove non-source citation placeholders from generated Markdown."""

    text = clean_markdown(markdown)
    text = re.sub(
        r"\s*\[\s*(?:uncited|citation needed|source needed|needs citation|no citation)\s*\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s+(?:and reproduced )?in (?:the )?uncited excerpt\.?",
        ".",
        text,
        flags=re.IGNORECASE,
    )
    return clean_markdown(text)


def remove_empty_evidence_gap_sections(markdown: str) -> str:
    lines = clean_markdown(markdown).splitlines()
    for start, level, heading in reversed([
        (index, len(match.group(1)), line.lstrip("#").strip())
        for index, line in enumerate(lines)
        if (match := re.match(r"^(#{2,3})\s+", line))
    ]):
        if normalized_heading(heading) != "evidence gaps":
            continue
        end = section_end_index(lines, start, level)
        if not evidence_gap_section_has_content(lines[start + 1 : end]):
            del lines[start:end]
    return clean_markdown("\n".join(lines))


def evidence_gap_section_has_content(lines: Sequence[str]) -> bool:
    boilerplate_phrases = (
        "these gaps are noted for completeness",
        "these gaps are acknowledged",
        "avoid overstating unsupported details",
        "report includes all verifiable information",
        "no evidence gaps",
        "no missing evidence",
        "no missing-evidence",
    )
    for line in lines:
        text = clean_text(strip_markdown_markup(line)).lower()
        if not text or re.fullmatch(r"[-*_]{3,}", text):
            continue
        if any(phrase in text for phrase in boilerplate_phrases):
            continue
        return True
    return False


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
        "do not include",
        "does not include",
        "not reproduced",
        "cannot be cited",
        "cannot be supplied",
        "cannot be reproduced",
        "no explicit statement",
        "no explicit evidence",
        "no precise citation",
        "no details",
        "no direct citation",
        "no direct equation",
        "no direct equations",
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


def format_source_priority_guidance(sources: Sequence[Any]) -> str:
    primary = []
    background = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        index = source.get("index")
        url = clean_text(source.get("url"))
        title = clean_text(source.get("title")) or url
        if not isinstance(index, int) or not url:
            continue
        item = f"[{index}] {title}"
        if is_primary_or_official_source(url):
            primary.append(item)
        else:
            background.append(item)

    lines = ["Prefer these sources for equations, benchmarks, APIs, and paper-specific claims when supported."]
    if primary:
        lines.append("Primary/official sources: " + ", ".join(primary[:12]))
    if background:
        lines.append(
            "Use background sources only when no primary/official source supports the claim: "
            + ", ".join(background[:8])
        )
    return "\n".join(lines)


def is_primary_or_official_source(url: str) -> bool:
    value = clean_text(url).lower()
    primary_signals = (
        "arxiv.org/abs/",
        "arxiv.org/pdf/",
        "openreview.net/",
        "aclanthology.org/",
        "doi.org/",
        "pytorch.org/docs",
        "tensorflow.org/api_docs",
        ".edu/",
        ".gov/",
    )
    return any(signal in value for signal in primary_signals)


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


def format_planner_evidence_packet(
    report_context: dict[str, Any],
    planner_questions: Sequence[str],
    sources: Sequence[dict[str, Any]],
    citation_aliases: dict[int, int] | None = None,
) -> str:
    """Build a compact source-backed coverage map for final report generation."""

    questions = [clean_text(question) for question in planner_questions if clean_text(question)]
    if not questions:
        return ""
    items = planner_evidence_items(report_context, sources, citation_aliases or {})
    all_evidence_text = clean_markdown("\n".join(item["text"] for item in items))
    lines = [
        "Use this packet as the source of truth. Covered topics must be answered directly; do not repeat stale missing-evidence notes for them.",
    ]
    used_chars = len(lines[0])
    for question in questions:
        expected_topics = expected_question_topics(question)
        covered_topics = [
            topic
            for topic in expected_topics
            if topic_has_evidence(topic, all_evidence_text, sources)
        ]
        missing_topics = [topic for topic in expected_topics if topic not in covered_topics]
        matches = ranked_question_evidence(question, items, expected_topics)[:2]
        if matches:
            status = "Partial" if missing_topics else "Covered"
        elif covered_topics:
            status = "Partial" if missing_topics else "Covered"
        else:
            status = "Missing"
        block_lines = [
            f"Question: {question}",
            f"Status: {status}",
        ]
        if covered_topics:
            block_lines.append(f"Covered topics: {', '.join(covered_topics)}")
        if matches:
            block_lines.append("Best evidence:")
            block_lines.extend(f"- {item['marker']} {item['text']}" for item in matches)
        if missing_topics:
            block_lines.append(f"Allowed Evidence Gaps: {', '.join(missing_topics)}")
        elif status == "Covered":
            block_lines.append("Allowed Evidence Gaps: none for this question")
        block = "\n".join(block_lines)
        if used_chars + len(block) > DEFAULT_REPORT_EVIDENCE_PACKET_CHARS:
            break
        lines.extend(["", block])
        used_chars += len(block)
    return clean_markdown("\n".join(lines))


def planner_evidence_items(
    report_context: dict[str, Any],
    sources: Sequence[dict[str, Any]],
    citation_aliases: dict[int, int],
) -> list[dict[str, Any]]:
    items = []
    seen = set()
    for chunk in list(report_context.get("supporting_chunks") or []) + list(report_context.get("retrieved_chunks") or []):
        if not isinstance(chunk, dict):
            continue
        source_index = chunk.get("source_index")
        if not isinstance(source_index, int):
            source_index = chunk.get("index")
        if isinstance(source_index, int):
            source_index = citation_aliases.get(source_index, source_index)
        text = focused_signal_snippet(chunk.get("content"), max_chars=360)
        if not text:
            text = clean_text(chunk.get("content"))[:360]
        add_planner_evidence_item(items, seen, source_index, text, chunk.get("title"), chunk.get("url"))

    source_index_by_url = {
        normalize_source_key(clean_text(source.get("url"))): citation_aliases.get(source.get("index"), source.get("index"))
        for source in sources
        if isinstance(source, dict) and isinstance(source.get("index"), int) and clean_text(source.get("url"))
    }
    for source in browser_sources(report_context.get("browser_results", [])):
        source_index = source_index_by_url.get(normalize_source_key(clean_text(source.get("url"))))
        for snippet in high_signal_snippets(source.get("full_content") or source.get("content_preview"), limit=4):
            add_planner_evidence_item(items, seen, source_index, snippet, source.get("title"), source.get("url"))
    return items


def add_planner_evidence_item(
    items: list[dict[str, Any]],
    seen: set[str],
    source_index: Any,
    text: Any,
    title: Any = "",
    url: Any = "",
) -> None:
    snippet = clean_text(text)
    if not snippet:
        return
    marker = f"[{source_index}]" if isinstance(source_index, int) else "[uncited]"
    key = clean_text(f"{marker} {snippet}").lower()
    if key in seen:
        return
    seen.add(key)
    items.append(
        {
            "marker": marker,
            "source_index": source_index if isinstance(source_index, int) else None,
            "title": clean_text(title),
            "url": clean_text(url),
            "text": snippet[:420],
            "terms": detail_terms(f"{title} {url} {snippet}"),
            "priority": signal_priority(snippet),
        }
    )


def ranked_question_evidence(
    question: str,
    items: Sequence[dict[str, Any]],
    expected_topics: Sequence[str],
) -> list[dict[str, Any]]:
    question_terms = detail_terms(question)
    ranked = []
    for item in items:
        overlap = len(question_terms & set(item.get("terms", set())))
        topic_bonus = sum(2 for topic in expected_topics if evidence_text_has_topic(clean_text(item.get("text")), topic))
        score = overlap + topic_bonus + int(item.get("priority", 0))
        if score <= 1:
            continue
        ranked.append((score, item))
    return [item for _, item in sorted(ranked, key=lambda value: value[0], reverse=True)]


def expected_question_topics(question: str) -> list[str]:
    text = clean_text(question).lower()
    topic_aliases = {
        "bahdanau": ("bahdanau", "additive"),
        "luong": ("luong", "multiplicative"),
        "transformer": ("transformer", "self-attention", "multi-head", "vaswani"),
        "pytorch": ("pytorch", "torch"),
        "tensorflow": ("tensorflow", "tf.keras", "keras"),
        "glue": ("glue", "superglue"),
        "wmt": ("wmt", "bleu"),
        "vit/imagenet": ("vit", "vision transformer", "imagenet"),
        "linformer": ("linformer",),
        "performer": ("performer",),
        "sparse attention": ("sparse",),
        "relative positional attention": ("relative positional", "positional"),
    }
    topics = [
        topic
        for topic, aliases in topic_aliases.items()
        if any(alias in text for alias in aliases)
    ]
    return dedupe_preserve_order(topics)


def topic_has_evidence(topic: str, evidence_text: str, sources: Sequence[dict[str, Any]]) -> bool:
    if evidence_text_has_topic(evidence_text, topic):
        return True
    signals = packet_topic_signals(topic)
    return any(source_matches_topic(source, signals) for source in sources)


def evidence_text_has_topic(text: str, topic: str) -> bool:
    value = clean_text(text).lower()
    return any(signal in value for signal in packet_topic_signals(topic))


def packet_topic_signals(topic: str) -> tuple[str, ...]:
    signals = {
        "bahdanau": ("bahdanau", "1409.0473", "additive attention"),
        "luong": ("luong", "1508.04025", "multiplicative attention"),
        "transformer": ("transformer", "1706.03762", "multi-head", "scaled dot-product"),
        "pytorch": ("pytorch", "torch.nn.multiheadattention"),
        "tensorflow": ("tensorflow", "tf.keras", "keras.layers.multiheadattention"),
        "glue": ("glue", "superglue"),
        "wmt": ("wmt", "bleu"),
        "vit/imagenet": ("vision transformer", "2010.11929", "imagenet", "vit"),
        "linformer": ("linformer",),
        "performer": ("performer",),
        "sparse attention": ("sparse attention", "sparse"),
        "relative positional attention": ("relative positional", "positional bias"),
    }
    return signals.get(topic, (topic,))


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
    normalized = re.sub(
        r"\[\s*(\d+(?:\s*,\s*\d+)+)\s*\]",
        lambda match: " ".join(f"[{index.strip()}]" for index in match.group(1).split(",")),
        normalized,
    )
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

    value = re.sub(r"<think\b[^>]*>.*?</think>", "", str(text or ""), flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"<think\b[^>]*>.*$", "", value, flags=re.IGNORECASE | re.DOTALL)
