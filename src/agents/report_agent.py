"""Report agent for final research report generation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
from pathlib import Path
from typing import Any, Sequence

from src.memory.shared_memory import SharedMemory
from src.tools.groq_retry import create_chat_completion_with_retries as groq_chat_completion_with_retries
from src.tools.text_utils import clean_text


DEFAULT_REPORT_AGENT_MODEL = "llama-3.1-8b-instant"
DEFAULT_REPORT_AGENT_MAX_TOKENS = 4200
DEFAULT_REPORT_AGENT_CONTEXT_CHARS = 30000
DEFAULT_REPORT_AGENT_CHUNK_CHARS = 1000
DEFAULT_REPORT_SECTION_CONTEXT_CHARS = 9000
DEFAULT_REPORT_SECTION_MAX_TOKENS = 2200
DEFAULT_REPORT_SUMMARY_CONTEXT_CHARS = 9000
DEFAULT_REPORT_MAX_SECTIONS = 6
DEFAULT_REPORT_SECTION_CONCURRENCY = 3
DEFAULT_REPORT_SINGLE_PROMPT_CHARS = 12000
DEFAULT_REPORT_SINGLE_MAX_TOKENS = 2200
DEFAULT_REPORT_OUTPUT_DIR = "data/reports"
REPORT_REPAIR_MAX_ATTEMPTS = 2
SECTION_REPAIR_MAX_ATTEMPTS = 2


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

        original_sources = report_context.get("sources", [])
        sources, citation_aliases = dedupe_sources_by_url(original_sources)
        synthesis = remap_citation_markers(synthesis, citation_aliases)
        source_text = format_sources(sources)
        evidence_text = format_supporting_evidence(report_context, citation_aliases=citation_aliases)
        missing_evidence_text = format_missing_evidence_constraints(synthesis)
        citation_policy = clean_text(report_context.get("citation_policy")) or (
            "Use only numbered source markers from the provided sources."
        )
        diagnostics = report_context.get("diagnostics", {})
        client = Groq()
        output_format_text = clean_text(output_format) or "report"
        single_prompt = build_single_report_prompt(
            objective=objective,
            output_format=output_format_text,
            citation_policy=citation_policy,
            synthesis=synthesis,
            missing_evidence_text=missing_evidence_text,
            evidence_text=evidence_text,
            source_text=source_text,
            diagnostics=diagnostics,
        )
        generation_mode = "single"
        section_count = 0
        section_concurrency = 0
        report_model = self.model
        if len(single_prompt) <= min(self.max_context_chars, DEFAULT_REPORT_SINGLE_PROMPT_CHARS):
            report, report_model = generate_single_report(
                client=client,
                model=self.model,
                prompt=single_prompt,
            )
        else:
            section_specs = report_section_specs(report_context, synthesis, output_format)
            sections, section_models = generate_report_sections(
                client_factory=Groq,
                model=self.model,
                objective=objective,
                output_format=output_format_text,
                section_specs=section_specs,
                report_context=report_context,
                citation_aliases=citation_aliases,
                sources=sources,
                synthesis=synthesis,
                missing_evidence_text=missing_evidence_text,
                citation_policy=citation_policy,
                diagnostics=diagnostics,
            )
            executive_summary, summary_model = generate_executive_summary(
                client=client,
                model=self.model,
                objective=objective,
                output_format=output_format_text,
                sections=sections,
                sources=sources,
                citation_policy=citation_policy,
            )
            report = assemble_report(
                objective=objective,
                executive_summary=executive_summary,
                sections=sections,
                sources=sources,
            )
            generation_mode = "sectioned_parallel"
            section_count = len(sections)
            section_concurrency = min(DEFAULT_REPORT_SECTION_CONCURRENCY, max(1, section_count))
            report_model = summary_model or (section_models[-1] if section_models else self.model)
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
            report_context_chars=min(self.max_context_chars, DEFAULT_REPORT_SECTION_CONTEXT_CHARS),
        )
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
                "report_generation_mode": generation_mode,
                "report_section_count": section_count,
                "report_section_concurrency": section_concurrency,
                "report_repair_count": repair_count,
                "report_issues": report_issues,
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

Synthesis-agent notes:
{synthesis}

Missing-evidence constraints:
{missing_evidence_text}

Supporting evidence chunks:
{evidence_text}

Available sources:
{source_text}

Retrieval diagnostics:
{diagnostics}

Generate the final report using only the synthesis-agent notes, supporting evidence chunks, and available sources above.
Requirements:
- Return polished Markdown.
- Include a clear title.
- Include an executive summary.
- Write a detailed technical report, not a short summary.
- Include technical sections that match the objective, synthesis, and requested output format.
- Prefer concrete definitions, structured comparisons, measurements, examples, and implementation details when supported by synthesis or supporting chunks.
- Explain important technical terms or notation when applicable, and use tables when they make comparisons clearer.
- Include exact technical details only when they are supported by synthesis or supporting chunks.
- Reconcile Missing-evidence constraints against Supporting evidence chunks before writing.
- If supporting chunks contain a detail that synthesis previously marked missing, include the supported detail and do not say it is missing.
- Do not reproduce details listed in Missing-evidence constraints unless they are present in Supporting evidence chunks.
- If a missing item remains important, state only that the provided evidence describes it but does not include the exact detail.
- Cite claims using only plain source markers from Available sources, exactly like [1], [2], [3].
- For precise claims, prefer original papers, official docs, academic sources, or authoritative surveys.
- Do not cite sources that are not listed.
- Do not use citation formats like 【1】, footnotes, line citations, or URLs inline.
- End with a References section mapping only used source markers to source URLs.
- If evidence is incomplete, mention the limitation instead of inventing details.
- Before finalizing, remove contradictions such as saying a detail is missing and then including that detail."""


def generate_single_report(client: Any, model: str, prompt: str) -> tuple[str, str]:
    return groq_chat_text(
        client=client,
        model=model,
        system_prompt="You are a careful report-writing agent. Use only provided synthesis context and evidence.",
        user_prompt=prompt,
        max_tokens=DEFAULT_REPORT_SINGLE_MAX_TOKENS,
        max_context_chars=DEFAULT_REPORT_SINGLE_PROMPT_CHARS,
    )


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


def generate_report_sections(
    client_factory: Any,
    model: str,
    objective: str,
    output_format: str,
    section_specs: Sequence[dict[str, str]],
    report_context: dict[str, Any],
    citation_aliases: dict[int, int],
    sources: Sequence[dict[str, Any]],
    synthesis: str,
    missing_evidence_text: str,
    citation_policy: str,
    diagnostics: Any,
) -> tuple[list[str], list[str]]:
    """Generate final report sections with small prompts to avoid TPM failures."""

    max_workers = min(DEFAULT_REPORT_SECTION_CONCURRENCY, max(1, len(section_specs)))
    generated: dict[int, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                generate_one_report_section,
                client=client_factory(),
                model=model,
                objective=objective,
                output_format=output_format,
                spec=spec,
                report_context=report_context,
                citation_aliases=citation_aliases,
                sources=sources,
                synthesis=synthesis,
                missing_evidence_text=missing_evidence_text,
                citation_policy=citation_policy,
                diagnostics=diagnostics,
            ): index
            for index, spec in enumerate(section_specs)
        }
        for future in as_completed(futures):
            generated[futures[future]] = future.result()

    sections = []
    models = []
    for index in sorted(generated):
        section, section_model = generated[index]
        if clean_text(section):
            sections.append(section)
            models.append(section_model)
    if not sections:
        raise ValueError("report_agent produced no report sections")
    return sections, models


def generate_one_report_section(
    client: Any,
    model: str,
    objective: str,
    output_format: str,
    spec: dict[str, str],
    report_context: dict[str, Any],
    citation_aliases: dict[int, int],
    sources: Sequence[dict[str, Any]],
    synthesis: str,
    missing_evidence_text: str,
    citation_policy: str,
    diagnostics: Any,
) -> tuple[str, str]:
    title = clean_text(spec.get("title")) or "Detailed Findings"
    instruction = clean_text(spec.get("instruction")) or title
    section_query = clean_text(f"{title} {instruction}")
    section_synthesis = relevant_text_for_section(synthesis, section_query, max_chars=3200)
    section_chunks = relevant_supporting_chunks_for_instruction(
        report_context=report_context,
        citation_aliases=citation_aliases,
        instruction=instruction,
        fallback_query=section_query,
        max_chunks=6,
        max_chars=600,
    )
    source_indexes = citation_markers(section_synthesis)
    source_indexes.extend(chunk_source_indexes(section_chunks))
    section_sources = sources_for_indexes(sources, source_indexes, max_sources=10) or list(sources[:8])
    available_citations = source_index_set(section_sources)
    section_evidence_text = format_chunk_blocks(section_chunks) or "No section-specific supporting chunks were found."
    prompt = build_section_prompt(
        objective=objective,
        output_format=output_format,
        title=title,
        instruction=instruction,
        citation_policy=citation_policy,
        section_synthesis=section_synthesis,
        section_evidence_text=section_evidence_text,
        missing_evidence_text=compact_markdown(missing_evidence_text, max_chars=1200),
        section_source_text=format_sources(section_sources),
        diagnostics=compact_markdown(diagnostics, max_chars=800),
    )
    section, response_model = groq_chat_text(
        client=client,
        model=model,
        system_prompt="You write one section of a technical report using only provided synthesis context and evidence.",
        user_prompt=prompt,
        max_tokens=DEFAULT_REPORT_SECTION_MAX_TOKENS,
        max_context_chars=DEFAULT_REPORT_SECTION_CONTEXT_CHARS,
    )
    section = strip_references_section(section)
    section = ensure_section_heading(section, title)
    section, section_repair_model = repair_section_if_needed(
        client=client,
        model=model,
        section=section,
        title=title,
        instruction=instruction,
        prompt=prompt,
        available_citations=available_citations,
    )
    if section_repair_model:
        response_model = section_repair_model
    return section, response_model


def repair_section_if_needed(
    client: Any,
    model: str,
    section: str,
    title: str,
    instruction: str,
    prompt: str,
    available_citations: set[int],
) -> tuple[str, str]:
    repaired = ensure_section_heading(section, title)
    repair_model = ""
    for _ in range(SECTION_REPAIR_MAX_ATTEMPTS):
        issues = section_quality_issues(repaired, instruction, available_citations=available_citations)
        if not issues:
            break
        repair_prompt = f"""Original section prompt:
{compact_markdown(prompt, max_chars=5200)}

Current section draft:
{compact_markdown(repaired, max_chars=2200)}

Detected section issues:
{format_issue_list(issues)}

Repair only this section.
Requirements:
- Return the complete corrected section only.
- Start with exactly this heading: ## {title}
- Finish all sentences, lists, tables, code blocks, and math blocks.
- Cover every item in the section instruction using the provided evidence, or state that the provided evidence is incomplete.
- Use only citation markers listed in the section sources.
- Remove unsupported or unavailable citation markers instead of inventing references.
- Do not include the report title, executive summary, conclusion, or References section."""
        candidate, response_model = groq_chat_text(
            client=client,
            model=model,
            system_prompt="You repair one technical report section for completeness and evidence consistency.",
            user_prompt=repair_prompt,
            max_tokens=DEFAULT_REPORT_SECTION_MAX_TOKENS,
            max_context_chars=DEFAULT_REPORT_SECTION_CONTEXT_CHARS,
        )
        candidate = ensure_section_heading(
            strip_references_section(candidate),
            title,
        )
        if clean_text(candidate):
            repaired = candidate
            repair_model = response_model
    return repaired, repair_model


def build_section_prompt(
    objective: str,
    output_format: str,
    title: str,
    instruction: str,
    citation_policy: str,
    section_synthesis: str,
    section_evidence_text: str,
    missing_evidence_text: str,
    section_source_text: str,
    diagnostics: str,
) -> str:
    return f"""Research objective:
{objective}

Requested output format:
{output_format}

Section to write:
{title}

Section instruction:
{instruction}

Citation policy:
{citation_policy}

Relevant synthesis notes:
{section_synthesis}

Relevant supporting evidence chunks:
{section_evidence_text}

Missing-evidence constraints:
{missing_evidence_text}

Available sources for this section:
{section_source_text}

Retrieval diagnostics:
{diagnostics}

Write only this report section.
Requirements:
- Start with exactly this heading: ## {title}
- Write detailed, evidence-backed technical content for this section.
- Prefer concrete definitions, comparisons, measurements, examples, and implementation details when supported.
- Include exact technical details only when they are present in the synthesis notes or supporting chunks.
- If evidence is incomplete, state the limitation instead of inventing details.
- Cite claims using only plain source markers listed above, like [1] or [2].
- Do not use citation markers that are absent from the section source list.
- Do not include the report title, executive summary, conclusion, or References section."""


def section_quality_issues(
    section: str,
    instruction: str,
    available_citations: set[int] | None = None,
) -> list[str]:
    issues = []
    text = clean_markdown(section)
    if not text:
        return ["section is empty"]
    issues.extend(markdown_completion_issues(text))
    invalid_citations = unavailable_citation_markers(text, available_citations or set())
    if invalid_citations:
        issues.append(f"section uses unavailable citations: {format_citation_indexes(invalid_citations)}")
    missing_items = missing_instruction_items(text, instruction)
    if missing_items:
        issues.append(f"section is missing requested coverage: {', '.join(missing_items[:3])}")
    if section_too_brief_for_instruction(text, instruction):
        issues.append("section appears too brief for the grouped instruction")
    return dedupe_preserve_order(issues)


def markdown_completion_issues(markdown: str) -> list[str]:
    text = clean_markdown(markdown)
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
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


def missing_instruction_items(section: str, instruction: str) -> list[str]:
    section_terms = detail_terms(section)
    missing = []
    for item in section_instruction_items(instruction):
        item_terms = {
            term
            for term in detail_terms(item)
            if len(term) >= 4
        }
        if not item_terms:
            continue
        required = min(3, max(1, len(item_terms) // 3))
        if len(item_terms & section_terms) < required:
            missing.append(short_issue_label(item))
    return missing


def section_instruction_items(instruction: str) -> list[str]:
    items = [
        clean_text(line.lstrip("-* ").strip())
        for line in str(instruction or "").splitlines()
        if line.strip().startswith(("-", "*"))
    ]
    return dedupe_preserve_order(items) or dedupe_preserve_order([instruction])


def short_issue_label(text: str, max_length: int = 80) -> str:
    value = clean_text(strip_markdown_markup(text))
    if len(value) <= max_length:
        return value
    return value[:max_length].rstrip(" ,.;:") + "..."


def section_too_brief_for_instruction(section: str, instruction: str) -> bool:
    item_count = len(section_instruction_items(instruction))
    min_chars = 450 if item_count <= 1 else min(1800, 450 * item_count)
    return len(clean_text(section)) < min_chars


def generate_executive_summary(
    client: Any,
    model: str,
    objective: str,
    output_format: str,
    sections: Sequence[str],
    sources: Sequence[dict[str, Any]],
    citation_policy: str,
) -> tuple[str, str]:
    """Generate a compact executive summary from completed sections."""

    section_preview = compact_markdown("\n\n".join(sections), max_chars=6500)
    summary_sources = sources_for_indexes(sources, citation_markers(section_preview), max_sources=12)
    prompt = f"""Research objective:
{objective}

Requested output format:
{output_format}

Citation policy:
{citation_policy}

Generated report sections:
{section_preview}

Available sources:
{format_sources(summary_sources)}

Write only the executive summary for the final report.
Requirements:
- Start with exactly this heading: ## Executive Summary
- Summarize the most important findings from the generated sections.
- Keep it concise but specific.
- If a generated section says a detail is missing, not reproduced, or not supported, do not include that detail as a fact in the summary.
- Cite source-backed claims with only the available source markers.
- Do not include a References section."""

    summary, response_model = groq_chat_text(
        client=client,
        model=model,
        system_prompt="You summarize completed report sections without adding unsupported claims.",
        user_prompt=prompt,
        max_tokens=700,
        max_context_chars=DEFAULT_REPORT_SUMMARY_CONTEXT_CHARS,
    )
    summary = strip_references_section(summary)
    return ensure_section_heading(summary, "Executive Summary"), response_model


def assemble_report(
    objective: str,
    executive_summary: str,
    sections: Sequence[str],
    sources: Sequence[dict[str, Any]],
) -> str:
    body_parts = [
        f"# {clean_text(objective).rstrip('?')}",
        ensure_section_heading(executive_summary, "Executive Summary"),
        *[strip_section_local_references(section) for section in sections if clean_text(section)],
    ]
    body = clean_markdown("\n\n".join(body_parts))
    return clean_markdown(f"{body}\n\n{references_section(body, sources)}")


def normalize_final_report(report: str, sources: Sequence[dict[str, Any]]) -> str:
    body = strip_all_references_blocks(report)
    return clean_markdown(f"{body}\n\n{references_section(body, sources)}")


def report_section_specs(
    report_context: dict[str, Any],
    synthesis: str,
    output_format: str,
) -> list[dict[str, str]]:
    instruction_items = instruction_requirement_items(report_context.get("synthesis_instruction"))
    structure_items = recommended_report_structure_items(synthesis)
    planner_items = [clean_text(question) for question in report_context.get("planner_questions", []) or []]
    items = [*instruction_items, *structure_items] if instruction_items else [*structure_items, *planner_items]
    items = dedupe_preserve_order(items)
    if not items:
        items = [clean_text(output_format) or "Detailed Findings"]
    return [
        {
            "title": section_title_from_instruction(" ".join(group), index),
            "instruction": "\n".join(f"- {item}" for item in group),
        }
        for index, group in enumerate(group_report_items(items, DEFAULT_REPORT_MAX_SECTIONS), start=1)
    ]


def group_report_items(items: Sequence[str], max_groups: int) -> list[list[str]]:
    clean_items = dedupe_preserve_order(items)
    if not clean_items:
        return []
    group_count = min(max(1, max_groups), len(clean_items))
    groups = [[] for _ in range(group_count)]
    for index, item in enumerate(clean_items):
        groups[min(group_count - 1, index * group_count // len(clean_items))].append(item)
    return [group for group in groups if group]


def instruction_requirement_items(instruction: Any) -> list[str]:
    text = clean_text(instruction)
    if not text:
        return []

    markers = list(re.finditer(r"(?:^|\s)(?:\(\d+\)|\d+[.)])\s+", text))
    if len(markers) >= 2:
        items = []
        for index, marker in enumerate(markers):
            start = marker.end()
            end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            item = clean_text(text[start:end].strip(" ;,."))
            item = re.sub(r"(?:,\s*)?\band$", "", item).strip(" ;,.")
            if item:
                items.append(item)
        return dedupe_preserve_order(items)

    bullet_items = [
        clean_text(line.lstrip("-* ").strip())
        for line in str(instruction or "").splitlines()
        if line.strip().startswith(("-", "*"))
    ]
    if bullet_items:
        return dedupe_preserve_order(item for item in bullet_items if item)

    listed_text = re.sub(r"^.*?\b(?:include|cover|covering)\s*:\s*", "", text, flags=re.IGNORECASE)
    listed_text = re.sub(r"\.\s*(?:cite|ensure|use)\b.*$", "", listed_text, flags=re.IGNORECASE)
    listed_items = split_instruction_list(listed_text)
    if len(listed_items) >= 3:
        return dedupe_preserve_order(listed_items)
    return []


def split_instruction_list(text: str) -> list[str]:
    items = []
    current = []
    depth = 0
    for char in str(text or ""):
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        if char in {",", ";"} and depth == 0:
            item = clean_text("".join(current).strip(" .;:,"))
            if item:
                items.append(item)
            current = []
            continue
        current.append(char)
    item = clean_text("".join(current).strip(" .;:,"))
    if item:
        items.append(item)
    if len(items) > 1:
        last = items[-1]
        if last.lower().startswith("and "):
            items[-1] = clean_text(last[4:])
    return [item for item in items if item]


def recommended_report_structure_items(synthesis: str) -> list[str]:
    section = markdown_section(synthesis, "Recommended Report Structure")
    if not section:
        return []
    items = []
    for line in section.splitlines():
        item = clean_text(re.sub(r"^[-*\d.)\s]+", "", line))
        if item and len(item) >= 8:
            items.append(strip_markdown_markup(item))
    return dedupe_preserve_order(items)


def markdown_section(markdown: str, heading: str) -> str:
    lines = clean_markdown(markdown).splitlines()
    target = normalized_heading(heading)
    capture = False
    section_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            heading_text = normalized_heading(stripped.lstrip("#").strip())
            if capture:
                break
            capture = heading_text == target
            continue
        if capture:
            section_lines.append(line)
    return "\n".join(section_lines).strip()


def normalized_heading(text: str) -> str:
    return clean_text(re.sub(r"^\d+[.)]\s*", "", text)).lower()


def section_title_from_instruction(instruction: str, index: int) -> str:
    text = clean_text(strip_markdown_markup(instruction))
    text = re.sub(r"\b(what|which|how|why|does|do|did|are|is|was|were|the|a|an)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(in|of|on|for|to|from|with|and|or|as|by|used|using|include|includes|have|has|been)\b", " ", text, flags=re.IGNORECASE)
    words = re.findall(r"[A-Za-z0-9+/#_.-]+", clean_text(text))[:8]
    if not words:
        return f"Section {index}"
    return " ".join(word[:1].upper() + word[1:] for word in words)


def relevant_text_for_section(text: str, query: str, max_chars: int) -> str:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", clean_markdown(text)) if clean_text(block)]
    if not blocks:
        return "No synthesis notes were provided for this section."
    query_terms = detail_terms(query)
    ranked = []
    for index, block in enumerate(blocks):
        score = len(detail_terms(block) & query_terms)
        ranked.append((score, index, block))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = [block for score, _, block in ranked if score > 0]
    if not selected:
        selected = blocks[:3]
    return truncate_blocks(selected, max_chars=max_chars)


def relevant_supporting_chunks(
    report_context: dict[str, Any],
    citation_aliases: dict[int, int],
    query: str,
    max_chunks: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    chunks = report_context.get("supporting_chunks") or report_context.get("retrieved_chunks") or []
    if not isinstance(chunks, list):
        return []
    query_terms = detail_terms(query)
    ranked = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        content = clean_text(chunk.get("content"))
        if not content:
            continue
        source_index = chunk_source_index(chunk, citation_aliases)
        content_terms = detail_terms(content)
        score = len(content_terms & query_terms)
        if chunk.get("is_primary_source"):
            score += 1
        ranked.append((score, index, {**chunk, "source_index": source_index, "content": content[:max_chars].strip()}))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = [chunk for score, _, chunk in ranked if score > 0][:max_chunks]
    if not selected:
        selected = [chunk for _, _, chunk in ranked[:max_chunks]]
    return selected


def relevant_supporting_chunks_for_instruction(
    report_context: dict[str, Any],
    citation_aliases: dict[int, int],
    instruction: str,
    fallback_query: str,
    max_chunks: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    selected = []
    seen_ids = set()
    queries = section_instruction_items(instruction) or [fallback_query]
    per_query_limit = max(1, min(2, max_chunks))
    for query in queries:
        for chunk in relevant_supporting_chunks(
            report_context=report_context,
            citation_aliases=citation_aliases,
            query=query,
            max_chunks=per_query_limit,
            max_chars=max_chars,
        ):
            key = clean_text(chunk.get("id")) or clean_text(chunk.get("url")) or clean_text(chunk.get("content"))[:80]
            if key in seen_ids:
                continue
            seen_ids.add(key)
            selected.append(chunk)
            if len(selected) >= max_chunks:
                return selected
    if selected:
        return selected
    return relevant_supporting_chunks(
        report_context=report_context,
        citation_aliases=citation_aliases,
        query=fallback_query,
        max_chunks=max_chunks,
        max_chars=max_chars,
    )


def chunk_source_index(chunk: dict[str, Any], citation_aliases: dict[int, int]) -> int | None:
    source_index = chunk.get("source_index")
    if not isinstance(source_index, int):
        source_index = chunk.get("index")
    if isinstance(source_index, int):
        return citation_aliases.get(source_index, source_index)
    return None


def chunk_source_indexes(chunks: Sequence[dict[str, Any]]) -> list[int]:
    indexes = []
    for chunk in chunks:
        source_index = chunk.get("source_index") if isinstance(chunk, dict) else None
        if isinstance(source_index, int):
            indexes.append(source_index)
    return indexes


def format_chunk_blocks(chunks: Sequence[dict[str, Any]]) -> str:
    blocks = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        source_index = chunk.get("source_index")
        marker = f"[{source_index}]" if isinstance(source_index, int) else "[uncited]"
        title = clean_text(chunk.get("title")) or clean_text(chunk.get("url")) or "Source"
        url = clean_text(chunk.get("url"))
        content = clean_text(chunk.get("content"))
        if content:
            blocks.append(f"{marker} {title}\nURL: {url}\nEvidence: {content}")
    return "\n\n".join(blocks)


def sources_for_indexes(
    sources: Sequence[dict[str, Any]],
    indexes: Sequence[int],
    max_sources: int,
) -> list[dict[str, Any]]:
    wanted = {index for index in indexes if isinstance(index, int)}
    selected = [
        source
        for source in sources
        if isinstance(source, dict)
        and isinstance(source.get("index"), int)
        and source["index"] in wanted
    ]
    if len(selected) < min(len(sources), max_sources):
        selected_indexes = {source["index"] for source in selected}
        for source in sources:
            if not isinstance(source, dict):
                continue
            index = source.get("index")
            if not isinstance(index, int) or index in selected_indexes:
                continue
            selected.append(source)
            selected_indexes.add(index)
            if len(selected) >= max_sources:
                break
    return selected[:max_sources]


def ensure_section_heading(markdown: str, title: str) -> str:
    content = clean_markdown(markdown)
    expected = f"## {clean_text(title)}"
    if not content:
        return expected
    first_line = content.splitlines()[0].strip()
    if first_line == expected:
        return content
    if first_line.startswith("## "):
        return "\n".join([expected, *content.splitlines()[1:]]).strip()
    return f"{expected}\n{content}"


def strip_references_section(markdown: str) -> str:
    lines = clean_markdown(markdown).splitlines()
    kept = []
    for line in lines:
        if is_references_heading(line):
            break
        kept.append(line)
    return clean_markdown("\n".join(kept))


def strip_section_local_references(markdown: str) -> str:
    lines = clean_markdown(markdown).splitlines()
    kept = []
    skip = False
    for line in lines:
        if is_references_heading(line):
            remove_trailing_separators(kept)
            skip = True
            continue
        if skip and line.startswith("## "):
            skip = False
        if not skip:
            kept.append(line)
    return trim_trailing_separators(clean_markdown("\n".join(kept)))


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


def truncate_blocks(blocks: Sequence[str], max_chars: int) -> str:
    selected = []
    used_chars = 0
    for block in blocks:
        text = clean_text(block)
        if not text:
            continue
        if used_chars + len(text) > max_chars:
            remaining = max_chars - used_chars
            if remaining > 400:
                selected.append(text[:remaining].rstrip())
            break
        selected.append(text)
        used_chars += len(text)
    return "\n\n".join(selected) or "No relevant notes were found."


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

    repaired = normalize_final_report(report, sources)
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
        repaired = normalize_final_report(repaired, sources)
        issues = report_quality_issues(repaired, evidence_text, sources=sources)

    if issues:
        raise ValueError(f"report_agent produced invalid report: {'; '.join(issues)}")
    return repaired, total_repair_count, issues


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
    if missing_statement_contains_unsupported_detail(report):
        issues.append("report includes exact details inside missing-evidence statements")
    if stale_missing_detail_statement(report, evidence_text):
        issues.append("report may contain stale missing-evidence statements contradicted by supporting evidence")
    return dedupe_preserve_order(issues)


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
    sections = []
    heading = ""
    lines = []
    for line in clean_markdown(markdown).splitlines():
        if line.startswith("## "):
            if heading:
                sections.append((heading, "\n".join(lines).strip()))
            heading = line.lstrip("#").strip()
            lines = [line]
            continue
        if heading:
            lines.append(line)
    if heading:
        sections.append((heading, "\n".join(lines).strip()))
    return sections


def stale_missing_detail_statement(report: str, evidence_text: str) -> bool:
    if not has_missing_claim(report):
        return False
    return bool(overlapping_detail_terms(report_missing_sentences(report), evidence_text))


def has_missing_claim(text: str) -> bool:
    return bool(report_missing_sentences(text))


def report_missing_sentences(text: str) -> list[str]:
    missing_phrases = (
        "not explicitly",
        "not present",
        "not provided",
        "not available",
        "not included",
        "not mentioned",
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
        "is missing",
        "are missing",
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
    return clean_text(url).lower().rstrip("/")


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

    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    normalized = "\n".join(lines).strip()
    normalized = re.sub(r"\n{4,}", "\n\n\n", normalized)
    return normalized
