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
DEFAULT_RETRY_EVIDENCE_CHARS = 8000
DEFAULT_RETRY_SYNTHESIS_CHARS = 1600
DEFAULT_RETRY_EVIDENCE_PACK_CHARS = 1000
DEFAULT_RETRY_COVERAGE_CHARS = 1200
DEFAULT_RETRY_SOURCE_CHARS = 1400
DEFAULT_RETRY_PACK_CHARS = 220
DEFAULT_FOCUSED_EVIDENCE_CHARS = 9000
DEFAULT_FOCUSED_CHUNKS_PER_QUESTION = 4
DEFAULT_FOCUSED_CHUNK_CHARS = 1500

DEFAULT_REPORT_GENERATION_MODE = "sections"  # "single" or "sections"  
DEFAULT_SECTION_MAX_TOKENS = 1500
DEFAULT_SECTION_RETRY_ATTEMPTS = 2
DEFAULT_SECTION_EVIDENCE_CHUNKS = 4
DEFAULT_SECTION_EVIDENCE_CHUNK_CHARS = 1500
DEFAULT_FRAME_MAX_TOKENS = 700
DEFAULT_FRAME_SECTION_CHARS = 900

SECTION_SYSTEM_PROMPT = (
    "You write ONE section of a larger cited research report from supplied evidence only. "
    "Do not use outside knowledge. Cite every claim inline with a real source marker like [1]. "
    "Output only that one section - no other headings, no preamble, no closing remarks."
)

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
- Treat synthesis coverage as a signal, but cited per-question evidence overrides stale gap notes.
- If a per-question evidence pack is covered and includes cited chunks, the matching section must use those chunks and cite at least one of their source markers.
- If any per-question evidence pack lists source markers, the matching section must cite at least one listed marker whenever it makes supported claims for that sub-question.
- Never label a covered evidence pack as an evidence gap. If exact details are incomplete, write the supported answer first with citations, then name only the missing detail as a caveat.
- If a question is marked missing and no cited evidence is supplied for it, write a short evidence-gap subsection instead of inventing an answer.
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

    def __init__(self, model: str | None = None, generation_mode: str | None = None) -> None:
        self.model = (
            clean_text(model)
            or clean_text(os.environ.get("RESEARCH_PLANNER_MODEL"))
            or clean_text(os.environ.get("RAG_GENERATION_MODEL"))
            or DEFAULT_REPORT_AGENT_MODEL
        )
        # "single" = one completion call for the whole report (legacy).
        # "sections" = one focused call per topic section plus small framing
        # calls (exec summary, cross-cutting, limitations, conclusion),
        # stitched together. Costs more calls but each call is small,
        # grounded in only that question's evidence, and individually
        # retried, so a weak/missing evidence pack for one sub-question
        # can no longer degrade the whole report.
        self.generation_mode = (
            clean_text(generation_mode)
            or clean_text(os.environ.get("REPORT_GENERATION_MODE"))
            or DEFAULT_REPORT_GENERATION_MODE
        ).lower()

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
        coverage_by_question = resolve_report_coverage(
            report_context.get("coverage_by_question", []),
            evidence_packs,
            coverage_questions,
        )
        coverage_conflicts = report_coverage_conflicts(report_context.get("coverage_by_question", []), evidence_packs)
        if coverage_conflicts:
            print(f"[report] resolved {len(coverage_conflicts)} stale coverage gap(s) from evidence packs")
        per_question_synthesis = report_context.get("per_question_synthesis", [])
        evidence = format_question_focused_evidence(report_context, coverage_questions, sources=sources, evidence_packs=evidence_packs)
        if not evidence:
            evidence = format_supporting_evidence(report_context, sources=sources)
        pack_text = format_evidence_packs(evidence_packs)
        per_question_synthesis_text = format_per_question_synthesis(per_question_synthesis)
        sources = evidence_backed_sources(sources, evidence, synthesis, pack_text, per_question_synthesis_text)

        emit_progress(
            "tool_called",
            "Report agent calling Groq to generate final report",
            agent="report",
            tool="groq",
            metadata={"model": self.model, "generation_mode": self.generation_mode},
        )
        client = Groq()
        # Built unconditionally: the self-critique repair-retry below always
        # falls back to the single-shot compact prompt, even when the
        # initial draft was produced in "sections" mode, so prompt_inputs
        # must exist regardless of which branch generated the first draft.
        prompt_inputs = {
            "objective": objective,
            "output_format": output_format,
            "planner_questions": coverage_questions,
            "synthesis": synthesis,
            "evidence": evidence,
            "sources": sources,
            "citation_policy": clean_text(report_context.get("citation_policy")),
            "coverage_by_question": coverage_by_question,
            "evidence_packs": evidence_packs,
            "per_question_synthesis": per_question_synthesis,
        }
        section_diagnostics: dict[str, Any] = {}
        if self.generation_mode == "sections":
            report, model, section_diagnostics = generate_report_by_sections(
                client,
                self.model,
                objective=objective,
                output_format=output_format,
                coverage_questions=coverage_questions,
                evidence_packs=evidence_packs,
                sources=sources,
                synthesis=synthesis,
                per_question_synthesis=per_question_synthesis,
            )
        else:
            prompt = build_report_prompt(**prompt_inputs)
            fallback_prompt = build_report_prompt(**prompt_inputs, compact=True)
            report, model = generate_single_report(client, self.model, prompt, fallback_prompt=fallback_prompt)
        report = normalize_final_report(report, sources)
        validation = validate_report_output(report, sources, coverage_questions, evidence, synthesis, pack_text, evidence_packs, report_context)
        review_trace = [validation["review"]]
        if report_needs_revision(validation):
            feedback = format_report_revision_feedback(validation)
            print(f"[report] retrying after self-critique: {clean_text(feedback)[:240]}")
            repair_prompt = build_report_prompt(**prompt_inputs, compact=True, repair_feedback=feedback)
            report, model = generate_single_report(client, self.model, repair_prompt, fallback_prompt=repair_prompt)
            report = normalize_final_report(report, sources)
            validation = validate_report_output(report, sources, coverage_questions, evidence, synthesis, pack_text, evidence_packs, report_context)
            review_trace.append(validation["review"])
        report, deterministic_repairs = apply_report_evidence_pack_repairs(
            report,
            evidence_packs,
            validation,
            coverage_questions,
            sources,
            per_question_synthesis=per_question_synthesis,
        )
        if deterministic_repairs:
            print(f"[report] applied {len(deterministic_repairs)} deterministic evidence-pack repair(s)")
            validation = validate_report_output(report, sources, coverage_questions, evidence, synthesis, pack_text, evidence_packs, report_context)
            review_trace.append({**validation["review"], "source": "deterministic_repair", "repairs": deterministic_repairs})

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
                "report_issues": validation["report_issues"],
                "report_schema_issues": validation["schema_issues"],
                "report_missing_sub_questions": validation["coverage"]["missing"],
                "report_evidence_gap_questions": validation["synthesis_gaps"],
                "report_false_gap_questions": validation["false_gap_questions"],
                "report_pack_citation_gap_questions": validation["pack_citation_gap_questions"],
                "report_coverage_conflicts": coverage_conflicts,
                "report_coverage_check": validation["coverage"],
                "report_retry_queries": rewrite_missing_sub_question_queries(
                    objective,
                    dedupe_text(
                        [
                            *validation["coverage"]["missing"],
                            *validation["synthesis_gaps"],
                            *validation["false_gap_questions"],
                            *validation["pack_citation_gap_questions"],
                        ]
                    ),
                ),
                "report_review_trace": review_trace,
                "report_revision_attempts": len(review_trace) - 1,
                "report_deterministic_repairs": deterministic_repairs,
                "report_token_budget": DEFAULT_REPORT_TOTAL_TOKEN_BUDGET,
                "report_generation_mode": self.generation_mode,
                "report_section_diagnostics": section_diagnostics,
                "report_prompt_chars": len(prompt) if self.generation_mode != "sections" else None,
                "report_fallback_prompt_chars": len(fallback_prompt) if self.generation_mode != "sections" else None,
                "report_estimated_token_cap": (
                    report_generation_token_cap(len(prompt)) if self.generation_mode != "sections" else None
                ),
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
    per_question_synthesis: Sequence[dict[str, Any]] | None = None,
    compact: bool = False,
    repair_feedback: str = "",
) -> str:
    source_text = format_sources(sources)
    evidence_text = clean_markdown(evidence)
    synthesis_text = clean_markdown(synthesis)
    coverage_text = format_question_coverage(coverage_by_question or [])
    pack_text = format_evidence_packs(evidence_packs or [])
    per_question_synthesis_text = format_per_question_synthesis(per_question_synthesis or [])
    if compact:
        source_text = compact_text(source_text, DEFAULT_RETRY_SOURCE_CHARS)
        evidence_text = compact_text(evidence_text, DEFAULT_RETRY_EVIDENCE_CHARS)
        synthesis_text = compact_text(synthesis_text, DEFAULT_RETRY_SYNTHESIS_CHARS)
        coverage_text = compact_text(coverage_text, DEFAULT_RETRY_COVERAGE_CHARS)
        pack_text = compact_text(
            format_evidence_packs(
                evidence_packs or [],
                max_chunks_per_pack=1,
                chunk_chars=DEFAULT_RETRY_PACK_CHARS,
            ),
            DEFAULT_RETRY_EVIDENCE_PACK_CHARS,
        )
        per_question_synthesis_text = compact_text(per_question_synthesis_text, DEFAULT_RETRY_EVIDENCE_PACK_CHARS)
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
{source_text}

Supporting evidence:
{evidence_text}

Synthesis notes:
{synthesis_text}

Synthesis coverage by planner question:
{coverage_text}

{format_repair_feedback(repair_feedback)}

Evidence gaps from synthesis:
{format_missing_evidence_constraints(synthesis)}

Per-question evidence packs:
{pack_text}

Per-question synthesis notes:
{per_question_synthesis_text}

Write the final Markdown report. Explain each supported topic in clear prose before equations, tables, APIs, or technical details."""
    return trim_report_prompt(prompt) if compact else prompt


def generate_report_by_sections(
    client: Any,
    model: str,
    objective: str,
    output_format: str,
    coverage_questions: Sequence[str],
    evidence_packs: Sequence[dict[str, Any]],
    sources: Sequence[dict[str, Any]],
    synthesis: str,
    per_question_synthesis: Sequence[dict[str, Any]] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Generate the report as separate, individually-grounded section calls.

    Each topic section only ever sees its own question's evidence pack, so a
    thin or missing evidence pack for one sub-question can no longer starve
    or corrupt the rest of the report. Framing sections (exec summary,
    introduction, cross-cutting analysis, limitations, conclusion) are
    generated afterwards from the already-written topic sections, so they
    describe what was actually produced instead of guessing ahead of time.
    """

    source_text = format_sources(sources)
    packs_by_question = {
        normalize_heading(pack.get("question")): pack
        for pack in evidence_packs or []
        if isinstance(pack, dict)
    }
    synthesis_by_question = per_question_synthesis_by_question(per_question_synthesis or [])
    last_model = model

    topic_sections: list[str] = []
    section_diagnostics: list[dict[str, Any]] = []
    for question in coverage_questions:
        pack = packs_by_question.get(normalize_heading(question), {})
        synthesis_note = synthesis_by_question.get(normalize_heading(question), {})
        section, used_model, retried = generate_topic_section(
            client,
            model,
            objective,
            question,
            pack,
            source_text,
            synthesis_note=synthesis_note,
        )
        last_model = used_model or last_model
        topic_sections.append(section)
        section_diagnostics.append({
            "question": question,
            "retried": retried,
            "had_usable_evidence": evidence_pack_has_usable_cited_evidence(pack),
            "had_per_question_synthesis": per_question_synthesis_has_cited_evidence(synthesis_note),
            "chars": len(section),
        })

    topics_digest = "\n\n".join(compact_text(section, DEFAULT_FRAME_SECTION_CHARS) for section in topic_sections)

    intro, last_model = generate_frame_section(
        client, model, "Introduction and Context",
        instructions=(
            "Write a short introduction (3-5 sentences) that frames the research objective and previews the "
            "topics covered below, using only claims already present in the topic sections. Cite using the "
            "same source markers used in those sections; do not introduce a new fact without one."
        ),
        objective=objective, source_text=source_text, body_digest=topics_digest, fallback_model=last_model,
    )
    cross_cutting, last_model = generate_frame_section(
        client, model, "Cross-cutting Analysis and Synthesis",
        instructions=(
            "Write a cross-cutting analysis that connects the topic sections below into a coherent narrative: "
            "note relationships, tensions, or a progression across topics. Use only claims and citations already "
            "present in the topic sections - do not add new facts."
        ),
        objective=objective, source_text=source_text, body_digest=topics_digest, fallback_model=last_model,
    )
    limitations, last_model = generate_frame_section(
        client, model, "Limitations and Open Questions",
        instructions=(
            "List, as short bullet points, which sub-questions below have thin or missing evidence and what "
            "specific detail is unresolved. Base this only on gaps already stated in the topic sections; do not "
            "invent new limitations and do not repeat claims that were already supported with a citation."
        ),
        objective=objective, source_text=source_text, body_digest=topics_digest, fallback_model=last_model,
    )
    conclusion, last_model = generate_frame_section(
        client, model, "Conclusion",
        instructions=(
            "Write a short conclusion (3-5 sentences) summarising what is well-supported across the topic "
            "sections and what remains open, consistent with the limitations already identified. Do not add new "
            "facts or citations that are not already present above."
        ),
        objective=objective, source_text=source_text, body_digest=topics_digest, fallback_model=last_model,
    )
    exec_summary, last_model = generate_frame_section(
        client, model, "Executive Summary",
        instructions=(
            "Write a 3-4 sentence executive summary of the whole report below, naming the main topics covered "
            "and, in one clause, the main limitation. Do not add new facts or citations that are not already "
            "present above."
        ),
        objective=objective, source_text=source_text,
        body_digest=f"{intro}\n\n{topics_digest}\n\n{cross_cutting}\n\n{limitations}\n\n{conclusion}",
        fallback_model=last_model,
    )

    report = "\n\n".join([
        f"## 1. Executive Summary\n{strip_leading_heading(exec_summary)}",
        f"## 2. Introduction and Context\n{strip_leading_heading(intro)}",
        "## 3. Topic Sections",
        *(f"### 3.{i}. {planner_question_heading(q)}\n{strip_leading_heading(section)}" for i, (q, section) in enumerate(zip(coverage_questions, topic_sections), 1)),
        f"## 4. Cross-cutting Analysis and Synthesis\n{strip_leading_heading(cross_cutting)}",
        f"## 5. Limitations and Open Questions\n{strip_leading_heading(limitations)}",
        f"## 6. Conclusion\n{strip_leading_heading(conclusion)}",
    ])
    return report, last_model, {"topic_sections": section_diagnostics}


def generate_topic_section(
    client: Any,
    model: str,
    objective: str,
    question: str,
    pack: dict[str, Any],
    source_text: str,
    synthesis_note: dict[str, Any] | None = None,
) -> tuple[str, str, bool]:
    heading = planner_question_heading(question)
    prompt = build_topic_section_prompt(objective, question, heading, pack, source_text, synthesis_note=synthesis_note)
    response = create_chat_completion_with_retries(
        client, model=model, temperature=0, max_tokens=DEFAULT_SECTION_MAX_TOKENS,
        retry_attempts=DEFAULT_SECTION_RETRY_ATTEMPTS,
        messages=[
            {"role": "system", "content": SECTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    section = normalize_citation_markers(clean_markdown(response.choices[0].message.content))
    used_model = clean_text(getattr(response, "model", "")) or model

    retried = False
    if section_needs_retry(section, pack, synthesis_note=synthesis_note):
        retried = True
        retry_prompt = f"""{prompt}

Your previous draft failed a check: it was empty, did not cite any of the available evidence markers above, or
wrongly called covered evidence a gap. Fix this using only the evidence given above.

Previous draft:
{section}"""
        response = create_chat_completion_with_retries(
            client, model=model, temperature=0, max_tokens=DEFAULT_SECTION_MAX_TOKENS,
            retry_attempts=DEFAULT_SECTION_RETRY_ATTEMPTS,
            messages=[
                {"role": "system", "content": SECTION_SYSTEM_PROMPT},
                {"role": "user", "content": retry_prompt},
            ],
        )
        section = normalize_citation_markers(clean_markdown(response.choices[0].message.content))
        used_model = clean_text(getattr(response, "model", "")) or used_model

    if not section.lstrip().startswith("#"):
        section = f"## {heading}\n\n{section}"
    return section, used_model, retried


def build_topic_section_prompt(
    objective: str,
    question: str,
    heading: str,
    pack: dict[str, Any],
    source_text: str,
    synthesis_note: dict[str, Any] | None = None,
) -> str:
    evidence_text = format_single_question_evidence(question, pack)
    synthesis_text = format_single_question_synthesis(question, synthesis_note or {})
    return f"""Research objective:
{objective}

You are writing ONLY this section of the report:
## {heading}

Sub-question this section must answer:
{question}

Available sources:
{source_text}

Evidence retrieved for this question only:
{evidence_text}

Per-question synthesis notes for this question:
{synthesis_text}

Rules:
- Use only the retrieved evidence and per-question synthesis notes above; never use outside knowledge.
- Treat cited per-question synthesis notes as report-ready support for this exact question.
- If per-question synthesis notes cite sources for an answer, use those cited claims before naming any gap.
- Cite every factual claim inline with a real marker shown above, like [1].
- If the evidence answers the question, write clear prose first, then any equation/formula/API/metric line only
  if that exact detail appears in the evidence.
- If the evidence above says no chunks were retrieved for this question, write one short sentence naming the
  gap - do not invent an answer, and do not call it a gap if cited evidence or cited synthesis is shown above.
- Output only the section content in Markdown (you may include the "## {heading}" heading). Do not write any
  other section, heading, or closing remarks."""


def format_single_question_evidence(question: str, pack: dict[str, Any]) -> str:
    chunks = pack.get("chunks", []) if isinstance(pack, dict) else []
    ranked = rank_question_chunks(question, chunks, planned_urls=pack.get("planned_source_urls", []) if isinstance(pack, dict) else [])
    lines = []
    for chunk in ranked[:DEFAULT_SECTION_EVIDENCE_CHUNKS]:
        content = sanitize_evidence_content(chunk.get("content"))[:DEFAULT_SECTION_EVIDENCE_CHUNK_CHARS].rstrip()
        if not content:
            continue
        source_index = chunk.get("source_index") if isinstance(chunk.get("source_index"), int) else chunk.get("index")
        marker = f"[{source_index}]" if isinstance(source_index, int) else "[uncited]"
        title = compact_text(clean_text(chunk.get("title")) or clean_text(chunk.get("url")) or "Evidence chunk", 90)
        lines.append(f"{marker} {title}:\n{content}")
    return "\n\n".join(lines) or "No cited retrieved evidence chunks were found for this question."


def section_needs_retry(section_text: str, pack: dict[str, Any], synthesis_note: dict[str, Any] | None = None) -> bool:
    if len(clean_text(strip_markdown(section_text))) < 40:
        return True
    has_usable_evidence = evidence_pack_has_usable_cited_evidence(pack) if isinstance(pack, dict) else False
    has_usable_synthesis = per_question_synthesis_has_cited_evidence(synthesis_note or {})
    if not has_usable_evidence and not has_usable_synthesis:
        return False
    available = set(pack_source_indexes(pack)) if isinstance(pack, dict) else set()
    available.update(per_question_synthesis_source_indexes(synthesis_note or {}))
    has_cited = bool(available & set(citation_markers(section_text)))
    has_gap_claim = bool(re.search(evidence_gap_pattern(), section_text.lower()))
    return (not has_cited) or has_gap_claim


def generate_frame_section(
    client: Any,
    model: str,
    heading: str,
    instructions: str,
    objective: str,
    source_text: str,
    body_digest: str,
    fallback_model: str,
) -> tuple[str, str]:
    prompt = f"""Research objective:
{objective}

Available sources:
{source_text}

Already-written report content (topic sections and/or earlier framing sections):
{body_digest}

Write ONLY the "{heading}" section of the report.
{instructions}
Never introduce a new fact, number, or citation that is not already present in the content above.
Output only the section content in Markdown. Do not write any other heading or closing remarks."""
    try:
        response = create_chat_completion_with_retries(
            client, model=model, temperature=0, max_tokens=DEFAULT_FRAME_MAX_TOKENS,
            retry_attempts=DEFAULT_SECTION_RETRY_ATTEMPTS,
            messages=[
                {"role": "system", "content": SECTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as error:
        print(f"[report] frame section '{heading}' failed ({clean_text(error)[:160]}); using digest fallback")
        return compact_text(body_digest, DEFAULT_FRAME_SECTION_CHARS) or f"{heading} could not be generated.", fallback_model
    section = normalize_citation_markers(clean_markdown(response.choices[0].message.content))
    used_model = clean_text(getattr(response, "model", "")) or fallback_model
    return section, used_model


def strip_leading_heading(section_text: str) -> str:
    lines = clean_markdown(section_text).splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        lines = lines[1:]
    return clean_markdown("\n".join(lines))


def generate_single_report(client: Any, model: str, prompt: str, fallback_prompt: str | None = None) -> tuple[str, str]:
    print(f"Generating single report with model {model}...")
    try:
        response = create_report_completion(client, model, prompt, retry_attempts=1)
    except Exception as error:
        if not fallback_prompt or not report_prompt_too_large_error(error):
            raise
        print("[report] prompt too large; retrying with compact evidence context")
        response = create_report_completion(client, model, fallback_prompt, retry_attempts=3)
    return normalize_citation_markers(response.choices[0].message.content), clean_text(getattr(response, "model", "")) or model


def create_report_completion(client: Any, model: str, prompt: str, retry_attempts: int) -> Any:
    return create_chat_completion_with_retries(
        client,
        model=model,
        temperature=0,
        max_tokens=DEFAULT_REPORT_MAX_TOKENS,
        retry_attempts=retry_attempts,
        messages=[
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )


def report_prompt_too_large_error(error: Exception) -> bool:
    message = clean_text(error).lower()
    return (
        "context_length_exceeded" in message
        or "please reduce the length of the messages or completion" in message
        or "please reduce your message size" in message
        or "request too large" in message
    )


def report_generation_token_cap(prompt_chars: int | None = None) -> int:
    if prompt_chars is None:
        return DEFAULT_REPORT_TOTAL_TOKEN_BUDGET
    return (max(0, prompt_chars) + 3) // 4 + DEFAULT_REPORT_MAX_TOKENS


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


def format_evidence_packs(
    evidence_packs: Sequence[dict[str, Any]],
    max_chunks_per_pack: int | None = None,
    chunk_chars: int | None = None,
) -> str:
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
        source_markers = ", ".join(f"[{index}]" for index in pack_source_indexes(pack))
        if source_markers:
            lines.append(f"  - Use cited evidence from {source_markers} for supported claims; do not call cited evidence absent.")
        if evidence_pack_has_formula_evidence(pack):
            lines.append(f"  - Formula/equation evidence is present in {source_markers or 'cited chunks'}; include it with citation before naming any remaining gap.")
        selected_chunks = chunks
        if max_chunks_per_pack is not None:
            selected_chunks = rank_question_chunks(question, chunks)[: max(0, max_chunks_per_pack)]
        for chunk in selected_chunks:
            if not isinstance(chunk, dict):
                continue
            source_index = chunk.get("source_index")
            marker = f"[{source_index}]" if isinstance(source_index, int) else "[uncited]"
            title = compact_text(clean_text(chunk.get("title")) or clean_text(chunk.get("url")) or "Evidence chunk", 90)
            content = sanitize_evidence_content(chunk.get("content"))
            if chunk_chars is not None:
                content = content[:chunk_chars].rstrip()
            if content:
                lines.append(f"  - {marker} {title}: {content}")
    return "\n".join(lines) or "- No per-question evidence packs were provided."


def format_per_question_synthesis(per_question_synthesis: Sequence[dict[str, Any]]) -> str:
    lines = []
    for item in per_question_synthesis or []:
        if not isinstance(item, dict):
            continue
        question = clean_text(item.get("question"))
        synthesis = clean_markdown(item.get("synthesis"))
        if not question or not synthesis:
            continue
        source_markers = ", ".join(f"[{index}]" for index in per_question_synthesis_source_indexes(item))
        lines.append(f"- {question}")
        if source_markers:
            lines.append(f"  - Use cited synthesis from {source_markers} before naming any evidence gap.")
        lines.append(f"  - {compact_text(synthesis, DEFAULT_RETRY_COVERAGE_CHARS)}")
    return "\n".join(lines) or "- No per-question synthesis notes were provided."


def format_single_question_synthesis(question: str, synthesis_note: dict[str, Any]) -> str:
    if not isinstance(synthesis_note, dict):
        return "No cited per-question synthesis notes were found for this question."
    synthesis = clean_markdown(synthesis_note.get("synthesis"))
    if not synthesis:
        return "No cited per-question synthesis notes were found for this question."
    source_markers = ", ".join(f"[{index}]" for index in per_question_synthesis_source_indexes(synthesis_note))
    header = f"Synthesis source markers: {source_markers or 'none'}"
    return f"{header}\n{synthesis}"


def per_question_synthesis_by_question(per_question_synthesis: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        normalize_heading(item.get("question")): item
        for item in per_question_synthesis or []
        if isinstance(item, dict) and clean_text(item.get("question"))
    }


def per_question_synthesis_source_indexes(synthesis_note: dict[str, Any]) -> list[int]:
    if not isinstance(synthesis_note, dict):
        return []
    return dedupe_ints([*synthesis_note.get("source_indexes", []), *citation_markers(synthesis_note.get("synthesis"))])


def per_question_synthesis_has_cited_evidence(synthesis_note: dict[str, Any]) -> bool:
    if not isinstance(synthesis_note, dict):
        return False
    synthesis = clean_text(synthesis_note.get("synthesis"))
    if not synthesis or re.search(evidence_gap_pattern(), synthesis.lower()) and not citation_markers(synthesis):
        return False
    return bool(per_question_synthesis_source_indexes(synthesis_note))


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
    max_chars: int | None = None,
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
        content = sanitize_evidence_content(content)
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


def format_question_focused_evidence(
    report_context: dict[str, Any],
    questions: Sequence[str],
    sources: Sequence[dict[str, Any]] | None = None,
    evidence_packs: Sequence[dict[str, Any]] | None = None,
    max_chars: int = DEFAULT_FOCUSED_EVIDENCE_CHARS,
) -> str:
    """Build compact evidence blocks that keep support for every planner question."""

    clean_questions = [clean_text(q) for q in questions if clean_text(q)]
    if not clean_questions:
        return ""
    all_chunks = all_report_chunks(report_context, evidence_packs)
    if not all_chunks:
        return ""
    source_index_by_url = {normalize_url(source.get("url")): source.get("index") for source in sources or [] if isinstance(source, dict)}
    packs_by_question = {normalize_heading(pack.get("question")): pack for pack in evidence_packs or [] if isinstance(pack, dict)}
    per_question_budget = max(650, max_chars // max(1, len(clean_questions)))
    blocks = []
    used = 0
    for question in clean_questions:
        pack = packs_by_question.get(normalize_heading(question), {})
        planned_urls = pack.get("planned_source_urls", []) if isinstance(pack, dict) else []
        ranked = rank_question_chunks(question, all_chunks, planned_urls=planned_urls)
        lines = [f"Question: {question}"]
        remaining = per_question_budget
        for chunk in ranked[:DEFAULT_FOCUSED_CHUNKS_PER_QUESTION]:
            content = sanitize_evidence_content(chunk.get("content"))[:DEFAULT_FOCUSED_CHUNK_CHARS].rstrip()
            if not content:
                continue
            source_index = chunk.get("source_index") if isinstance(chunk.get("source_index"), int) else chunk.get("index")
            if not isinstance(source_index, int):
                source_index = source_index_by_url.get(normalize_url(chunk.get("url")))
            marker = f"[{source_index}]" if isinstance(source_index, int) else "[uncited]"
            title = compact_text(clean_text(chunk.get("title")) or clean_text(chunk.get("url")) or "Evidence chunk", 90)
            line = f"- {marker} {title}: {content}"
            if len(line) > remaining and len(lines) > 1:
                continue
            lines.append(line)
            remaining -= len(line)
        block = "\n".join(lines)
        if len(lines) == 1:
            block += "\n- No cited retrieved evidence was selected for this question."
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def all_report_chunks(report_context: dict[str, Any], evidence_packs: Sequence[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for pack in evidence_packs or []:
        if isinstance(pack, dict):
            chunks.extend(chunk for chunk in pack.get("chunks", []) or [] if isinstance(chunk, dict))
    chunks.extend(chunk for chunk in report_context.get("supporting_chunks", []) or [] if isinstance(chunk, dict))
    chunks.extend(chunk for chunk in report_context.get("retrieved_chunks", []) or [] if isinstance(chunk, dict))
    deduped = []
    seen = set()
    for chunk in chunks:
        content = sanitize_evidence_content(chunk.get("content"))
        key = clean_text(f"{chunk.get('source_index')}:{chunk.get('url')}:{content[:160]}").lower()
        if content and key not in seen:
            seen.add(key)
            item = dict(chunk)
            item["content"] = content
            deduped.append(item)
    return deduped


def rank_question_chunks(
    question: str,
    chunks: Sequence[dict[str, Any]],
    planned_urls: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    target_urls = {normalize_url(url) for url in planned_urls or [] if normalize_url(url)}
    return sorted(
        [chunk for chunk in chunks if isinstance(chunk, dict)],
        key=lambda chunk: -question_chunk_score(question, chunk, target_urls),
    )


def question_chunk_score(question: str, chunk: dict[str, Any], planned_urls: set[str] | None = None) -> int:
    text = clean_text(" ".join([clean_text(chunk.get("title")), clean_text(chunk.get("url")), clean_text(chunk.get("content"))]))
    terms = detail_terms(question)
    overlap = len(terms & detail_terms(text))
    source_url = normalize_url(chunk.get("url"))
    assigned = normalize_heading(chunk.get("synthesis_question") or chunk.get("question")) == normalize_heading(question)
    planned = bool(source_url and source_url in (planned_urls or set()))
    score = overlap
    score += 8 if assigned else 0
    score += 6 if planned else 0
    score += 4 if chunk.get("is_primary_source") else 0
    score += source_priority(source_url) * 2
    score += evidence_snippet_score(text, list(terms), evidence_signals_for_question(question))
    return score


def evidence_signals_for_question(question: str) -> list[str]:
    lowered = clean_text(question).lower()
    signals = list(EVIDENCE_SNIPPET_SIGNALS)
    if any(term in lowered for term in ("equation", "formula", "mathematical", "component")):
        signals.extend(["=", "softmax", "sqrt", "tanh", "exp", "sum", "∑", "alpha", "attention("])
    if any(term in lowered for term in ("benchmark", "performance", "score", "metric", "improve")):
        signals.extend(["benchmark", "score", "result", "improve", "accuracy", "bleu", "glue", "wmt", "%"])
    if any(term in lowered for term in ("complexity", "scale", "cost", "sequence length")):
        signals.extend(["complexity", "quadratic", "linear", "memory", "o(", "sequence length"])
    if any(term in lowered for term in ("limitation", "challenge", "risk", "open question")):
        signals.extend(["limitation", "challenge", "bottleneck", "cost", "interpretability", "locality"])
    return dedupe_text(signals)


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
    return content


def evidence_snippet_score(snippet: str, terms: Sequence[str], signals: Sequence[str]) -> int:
    lowered = snippet.lower()
    return sum(1 for term in terms if term in lowered) + 2 * sum(1 for signal in signals if signal in lowered)


def sanitize_evidence_content(text: Any) -> str:
    """Remove paper-internal numeric citations so they cannot be mistaken for source markers."""

    return clean_text(re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", clean_text(text)))


def compact_evidence_blocks(blocks: Sequence[dict[str, Any]], max_chars: int | None) -> str:
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
            if max_chars is not None and used + len(block) > max_chars:
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


def validate_report_output(
    report: str,
    sources: Sequence[dict[str, Any]],
    planner_questions: Sequence[str],
    evidence: str,
    synthesis: str,
    pack_text: str,
    evidence_packs: Sequence[dict[str, Any]],
    report_context: dict[str, Any],
) -> dict[str, Any]:
    """Run separate, inspectable checks for final report quality."""

    synthesis_gaps = synthesis_coverage_gap_questions(report_context, planner_questions)
    coverage = report_sub_question_coverage_check(report, planner_questions)
    schema_issues = report_schema_issues(report, planner_questions)
    false_gaps = dedupe_text(
        [
            *report_evidence_gap_contradictions(report, evidence_packs, planner_questions),
            *report_synthesis_gap_contradictions(
                report,
                report_context.get("per_question_synthesis", []),
                planner_questions,
            ),
        ]
    )
    pack_citation_gaps = dedupe_text(
        [
            *report_pack_citation_gaps(report, evidence_packs, planner_questions),
            *report_per_question_synthesis_citation_gaps(
                report,
                report_context.get("per_question_synthesis", []),
                planner_questions,
            ),
        ]
    )
    report_issues = report_quality_issues(report, sources, evidence_text=f"{evidence}\n{synthesis}\n{pack_text}")
    report_issues.extend(f"report marks covered evidence as a gap: {question}" for question in false_gaps)
    report_issues.extend(f"report section does not cite its evidence pack: {question}" for question in pack_citation_gaps)
    review = report_self_critique(report_issues, coverage, schema_issues)
    return {
        "coverage": coverage,
        "schema_issues": schema_issues,
        "synthesis_gaps": synthesis_gaps,
        "false_gap_questions": false_gaps,
        "pack_citation_gap_questions": pack_citation_gaps,
        "report_issues": dedupe_text(report_issues),
        "review": review,
    }


def report_needs_revision(validation: dict[str, Any]) -> bool:
    return bool(
        validation.get("report_issues")
        or validation.get("schema_issues")
        or validation.get("synthesis_gaps")
        or validation.get("false_gap_questions")
        or validation.get("pack_citation_gap_questions")
        or validation.get("coverage", {}).get("missing")
    )


def format_report_revision_feedback(validation: dict[str, Any]) -> str:
    issues = [
        *validation.get("report_issues", []),
        *validation.get("schema_issues", []),
        *(f"missing planner topic: {q}" for q in validation.get("coverage", {}).get("missing", [])),
        *(f"synthesis gap to respect: {q}" for q in validation.get("synthesis_gaps", [])),
        *(f"false evidence gap to remove and replace with cited evidence: {q}" for q in validation.get("false_gap_questions", [])),
        *(f"missing evidence-pack citation to add in matching section: {q}" for q in validation.get("pack_citation_gap_questions", [])),
    ]
    return "\n".join(f"- {issue}" for issue in dedupe_text(issues)) or "- No unresolved issue."


def format_repair_feedback(repair_feedback: str) -> str:
    feedback = clean_text(repair_feedback)
    if not feedback:
        return ""
    return f"""Repair feedback from previous draft:
{repair_feedback}

Revise the report to fix every item above. If a topic has covered cited evidence, do not write it as an evidence gap."""


def apply_report_evidence_pack_repairs(
    report: str,
    evidence_packs: Sequence[dict[str, Any]],
    validation: dict[str, Any],
    planner_questions: Sequence[str],
    sources: Sequence[dict[str, Any]],
    per_question_synthesis: Sequence[dict[str, Any]] | None = None,
) -> tuple[str, list[str]]:
    """Patch unresolved per-question evidence failures with concise cited notes."""

    target_questions = dedupe_text(
        [
            *validation.get("false_gap_questions", []),
            *validation.get("pack_citation_gap_questions", []),
        ]
    )
    if not target_questions:
        return report, []
    packs_by_question = {
        normalize_heading(pack.get("question")): pack
        for pack in evidence_packs or []
        if isinstance(pack, dict) and evidence_pack_has_usable_cited_evidence(pack)
    }
    synthesis_by_question = per_question_synthesis_by_question(per_question_synthesis or [])
    report_text = strip_references(clean_markdown(report))
    repairs = []
    for question in target_questions:
        pack = packs_by_question.get(normalize_heading(question))
        synthesis_note = synthesis_by_question.get(normalize_heading(question), {})
        note = per_question_synthesis_repair_note(question, synthesis_note)
        if not note and pack:
            note = evidence_pack_repair_note(question, pack)
        if not note:
            continue
        report_text = upsert_section_repair_note(
            report_text,
            question,
            note,
            remove_gap_lines=question in validation.get("false_gap_questions", []),
        )
        repairs.append(question)
    if not repairs:
        return report, []
    return normalize_final_report(report_text, sources), repairs


def per_question_synthesis_repair_note(question: str, synthesis_note: dict[str, Any]) -> str:
    if not per_question_synthesis_has_cited_evidence(synthesis_note):
        return ""
    synthesis = clean_markdown(synthesis_note.get("synthesis"))
    if not synthesis:
        return ""
    snippet = compact_text(synthesis, 620)
    source_markers = set(per_question_synthesis_source_indexes(synthesis_note))
    if not (source_markers & set(citation_markers(snippet))):
        snippet = f"{snippet} {format_citation_indexes(source_markers)}"
    return f"**Per-question synthesis support:** {snippet}"


def evidence_pack_repair_note(question: str, pack: dict[str, Any]) -> str:
    ranked_chunks = rank_question_chunks(question, pack.get("chunks", []) or [], planned_urls=pack.get("planned_source_urls", []))
    for chunk in ranked_chunks:
        if not isinstance(chunk, dict) or not isinstance(chunk.get("source_index"), int):
            continue
        content = sanitize_evidence_content(chunk.get("content"))
        if not content:
            continue
        marker = f"[{chunk['source_index']}]"
        label = "Core evidence" if evidence_pack_has_formula_evidence(pack) else "Evidence-pack support"
        snippet = compact_text(content, 420)
        if marker not in snippet:
            snippet = f"{snippet} {marker}"
        return f"**{label}:** {snippet}"
    return ""


def upsert_section_repair_note(report: str, question: str, note: str, remove_gap_lines: bool = False) -> str:
    lines = clean_markdown(report).splitlines()
    bounds = section_bounds_for_question(lines, question)
    if not bounds:
        return clean_markdown(f"{report}\n\n## Evidence Pack Corrections\n\n### {planner_question_heading(question)}\n{note}")
    start, end = bounds
    section_lines = lines[start:end]
    if remove_gap_lines:
        section_lines = [line for line in section_lines if not line_has_gap_claim(line)]
    if note not in "\n".join(section_lines):
        section_lines.extend(["", note])
    return clean_markdown("\n".join([*lines[:start], *section_lines, *lines[end:]]))


def section_bounds_for_question(lines: Sequence[str], question: str) -> tuple[int, int] | None:
    expected = normalize_heading(planner_question_heading(question))
    question_terms = detail_terms(question)
    best: tuple[int, int] | None = None
    best_score = 0
    heading_positions = [
        (index, match.group(1).strip())
        for index, line in enumerate(lines)
        if (match := re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line))
    ]
    for position, (start, heading) in enumerate(heading_positions):
        end = heading_positions[position + 1][0] if position + 1 < len(heading_positions) else len(lines)
        actual = normalize_heading(heading)
        score = 5 if expected and headings_match(expected, actual) else 0
        score += len(question_terms & detail_terms(heading))
        if score > best_score:
            best_score = score
            best = (start, end)
    return best if best_score else None


def line_has_gap_claim(line: str) -> bool:
    return bool(re.search(evidence_gap_pattern(), clean_text(line).lower()))


def evidence_gap_pattern() -> str:
    return (
        r"(evidence\s+gap|evidence\s+not\s+provided|not\s+provided|missing\s+evidence|"
        r"(?:is|are)\s+missing|"
        r"no\s+source-backed|cannot\s+be\s+(?:given|reproduced|answered|provided)|"
        r"do\s+not\s+(?:contain|provide|include)|does\s+not\s+(?:contain|provide|include)|"
        r"none\s+.*\s+provide|not\s+available|absent\s+from\s+.*\s+evidence|"
        r"not\s+present\s+in\s+.*\s+(?:evidence|sources|material))"
    )


def resolve_report_coverage(
    coverage_by_question: Sequence[dict[str, Any]],
    evidence_packs: Sequence[dict[str, Any]],
    planner_questions: Sequence[str],
) -> list[dict[str, Any]]:
    """Prefer cited per-question evidence packs over stale missing coverage rows."""

    raw_by_question = {
        normalize_heading(item.get("question")): dict(item)
        for item in coverage_by_question or []
        if isinstance(item, dict) and clean_text(item.get("question"))
    }
    packs_by_question = {
        normalize_heading(pack.get("question")): pack
        for pack in evidence_packs or []
        if isinstance(pack, dict) and clean_text(pack.get("question"))
    }
    questions = dedupe_text([*planner_questions, *(pack.get("question") for pack in evidence_packs or [] if isinstance(pack, dict))])
    resolved = []
    for index, question in enumerate(questions, 1):
        key = normalize_heading(question)
        item = raw_by_question.get(key, {"question_id": f"q{index:03d}", "question": question})
        pack = packs_by_question.get(key)
        pack_indexes = pack_source_indexes(pack) if isinstance(pack, dict) else []
        if pack_indexes and evidence_pack_has_cited_evidence(pack):
            item["status"] = clean_text(pack.get("coverage")) or "covered"
            item["source_indexes"] = dedupe_ints([*item.get("source_indexes", []), *pack_indexes])
            item["has_citations"] = True
            item["evidence_count"] = max(int(item.get("evidence_count") or 0), len(pack.get("chunks", []) or []))
            item["missing_reason"] = ""
        resolved.append(item)
    return resolved


def report_coverage_conflicts(
    coverage_by_question: Sequence[dict[str, Any]],
    evidence_packs: Sequence[dict[str, Any]],
) -> list[str]:
    packs_by_question = {
        normalize_heading(pack.get("question")): pack
        for pack in evidence_packs or []
        if isinstance(pack, dict) and evidence_pack_has_cited_evidence(pack)
    }
    conflicts = []
    for item in coverage_by_question or []:
        if not isinstance(item, dict) or not synthesis_coverage_status_is_gap(item.get("status")):
            continue
        question = clean_text(item.get("question"))
        if normalize_heading(question) in packs_by_question:
            conflicts.append(question)
    return dedupe_text(conflicts)


def report_evidence_gap_contradictions(
    report: str,
    evidence_packs: Sequence[dict[str, Any]],
    planner_questions: Sequence[str] | None = None,
) -> list[str]:
    """Find sections that call a covered evidence pack missing."""

    canonical = {normalize_heading(q): q for q in planner_questions or [] if clean_text(q)}
    contradictions = []
    for pack in evidence_packs or []:
        if not isinstance(pack, dict) or not evidence_pack_has_usable_cited_evidence(pack):
            continue
        question = clean_text(pack.get("question"))
        section = report_section_for_question(report, question)
        if section and section_claims_missing_supported_evidence(section, question, pack):
            contradictions.append(canonical.get(normalize_heading(question), question))
    return dedupe_text(contradictions)


def report_pack_citation_gaps(
    report: str,
    evidence_packs: Sequence[dict[str, Any]],
    planner_questions: Sequence[str] | None = None,
) -> list[str]:
    """Find planner sections that use a topic but omit its evidence-pack source markers."""

    canonical = {normalize_heading(q): q for q in planner_questions or [] if clean_text(q)}
    gaps = []
    for pack in evidence_packs or []:
        if not isinstance(pack, dict) or not evidence_pack_has_usable_cited_evidence(pack):
            continue
        question = clean_text(pack.get("question"))
        section = report_section_for_question(report, question)
        if not section or section_cites_pack_source(section, pack):
            continue
        if section_mentions_pack_topic(section, question):
            gaps.append(canonical.get(normalize_heading(question), question))
    return dedupe_text(gaps)


def report_synthesis_gap_contradictions(
    report: str,
    per_question_synthesis: Sequence[dict[str, Any]],
    planner_questions: Sequence[str] | None = None,
) -> list[str]:
    """Find sections that call cited per-question synthesis evidence missing."""

    canonical = {normalize_heading(q): q for q in planner_questions or [] if clean_text(q)}
    contradictions = []
    for item in per_question_synthesis or []:
        if not isinstance(item, dict) or not per_question_synthesis_has_cited_evidence(item):
            continue
        question = clean_text(item.get("question"))
        section = report_section_for_question(report, question) or clean_markdown(report)
        if section and section_claims_missing_per_question_synthesis(section, question, item):
            contradictions.append(canonical.get(normalize_heading(question), question))
    return dedupe_text(contradictions)


def report_per_question_synthesis_citation_gaps(
    report: str,
    per_question_synthesis: Sequence[dict[str, Any]],
    planner_questions: Sequence[str] | None = None,
) -> list[str]:
    """Find topic sections that drop the citations used by their per-question synthesis."""

    canonical = {normalize_heading(q): q for q in planner_questions or [] if clean_text(q)}
    gaps = []
    for item in per_question_synthesis or []:
        if not isinstance(item, dict) or not per_question_synthesis_has_cited_evidence(item):
            continue
        question = clean_text(item.get("question"))
        section = report_section_for_question(report, question) or clean_markdown(report)
        if not section or section_cites_per_question_synthesis(section, item):
            continue
        if section_mentions_pack_topic(section, question):
            gaps.append(canonical.get(normalize_heading(question), question))
    return dedupe_text(gaps)


def section_claims_missing_per_question_synthesis(section: str, question: str, synthesis_note: dict[str, Any]) -> bool:
    lowered = clean_text(section).lower()
    gap_terms = evidence_gap_pattern()
    if not re.search(gap_terms, lowered):
        return False
    if not section_cites_per_question_synthesis(section, synthesis_note):
        return True
    for term in named_terms(question):
        if term in lowered and re.search(rf"\b{re.escape(term)}\b.{{0,140}}{gap_terms}|{gap_terms}.{{0,140}}\b{re.escape(term)}\b", lowered):
            return True
    return False


def section_mentions_pack_topic(section: str, question: str) -> bool:
    text_terms = detail_terms(section)
    question_terms = [term for term in detail_terms(question) if term not in STOPWORDS]
    named = named_terms(question)
    if named and any(term in text_terms for term in named):
        return True
    return len(set(question_terms[:8]) & text_terms) >= 2


def report_section_for_question(report: str, question: str) -> str:
    expected = normalize_heading(planner_question_heading(question))
    question_terms = detail_terms(question)
    best = ""
    best_score = 0
    for heading, section in markdown_sections(report):
        actual = normalize_heading(heading)
        score = 0
        if expected and headings_match(expected, actual):
            score += 5
        score += len(question_terms & detail_terms(heading))
        if score > best_score:
            best_score = score
            best = section
    return best if best_score else ""


def markdown_sections(markdown: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    heading = ""
    lines: list[str] = []
    in_fence = False
    for line in clean_markdown(markdown).splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            if lines:
                sections.append((heading, lines))
            heading, lines = match.group(1).strip(), [line]
        else:
            lines.append(line)
    if lines:
        sections.append((heading, lines))
    return [(heading, "\n".join(lines)) for heading, lines in sections]


def section_claims_missing_supported_evidence(section: str, question: str, pack: dict[str, Any]) -> bool:
    lowered = clean_text(section).lower()
    gap_terms = evidence_gap_pattern()
    if re.search(gap_terms, lowered) and (evidence_pack_has_formula_evidence(pack) or not section_cites_pack_source(section, pack)):
        return True
    for term in named_terms(question):
        if term in lowered and re.search(rf"\b{re.escape(term)}\b.{{0,140}}{gap_terms}|{gap_terms}.{{0,140}}\b{re.escape(term)}\b", lowered):
            return True
    return False


def section_cites_pack_source(section: str, pack: dict[str, Any]) -> bool:
    return bool(set(citation_markers(section)) & set(pack_source_indexes(pack)))


def section_cites_per_question_synthesis(section: str, synthesis_note: dict[str, Any]) -> bool:
    return bool(set(citation_markers(section)) & set(per_question_synthesis_source_indexes(synthesis_note)))


def pack_source_indexes(pack: dict[str, Any]) -> list[int]:
    return dedupe_ints(
        chunk.get("source_index")
        for chunk in pack.get("chunks", []) or []
        if isinstance(chunk, dict)
    )


def dedupe_ints(values: Sequence[Any]) -> list[int]:
    deduped = []
    seen = set()
    for value in values or []:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if isinstance(value, bool) or number in seen:
            continue
        seen.add(number)
        deduped.append(number)
    return deduped


def synthesis_coverage_gap_questions(
    report_context: dict[str, Any],
    planner_questions: Sequence[str] | None = None,
) -> list[str]:
    """Return planner questions synthesis marked as missing, partial, or weak."""

    if not isinstance(report_context, dict):
        return []
    canonical = {normalize_heading(q): q for q in planner_questions or [] if clean_text(q)}
    covered_packs = {
        normalize_heading(pack.get("question"))
        for pack in report_context.get("evidence_packs", []) or []
        if isinstance(pack, dict) and evidence_pack_has_cited_evidence(pack)
    }
    covered_synthesis = {
        normalize_heading(item.get("question"))
        for item in report_context.get("per_question_synthesis", []) or []
        if isinstance(item, dict) and per_question_synthesis_has_cited_evidence(item)
    }
    gaps = []
    for item in report_context.get("coverage_by_question", []) or []:
        if not isinstance(item, dict) or not synthesis_coverage_status_is_gap(item.get("status")):
            continue
        question = clean_text(item.get("question"))
        if normalize_heading(question) in covered_packs or normalize_heading(question) in covered_synthesis:
            continue
        if question:
            gaps.append(canonical.get(normalize_heading(question), question))
    return dedupe_text(gaps)


def evidence_pack_has_cited_evidence(pack: dict[str, Any]) -> bool:
    coverage = clean_text(pack.get("coverage")).lower()
    if synthesis_coverage_status_is_gap(coverage):
        return False
    return any(isinstance(chunk, dict) and isinstance(chunk.get("source_index"), int) and clean_text(chunk.get("content")) for chunk in pack.get("chunks", []) or [])


def evidence_pack_has_usable_cited_evidence(pack: dict[str, Any]) -> bool:
    return any(isinstance(chunk, dict) and isinstance(chunk.get("source_index"), int) and clean_text(chunk.get("content")) for chunk in pack.get("chunks", []) or [])


def evidence_pack_has_formula_evidence(pack: dict[str, Any]) -> bool:
    question = clean_text(pack.get("question"))
    if not re.search(r"\b(equations?|formulas?|mathematical|formulation|compatibility|alignment|score)\b", question, flags=re.IGNORECASE):
        return False
    for chunk in pack.get("chunks", []) or []:
        if not isinstance(chunk, dict) or not isinstance(chunk.get("source_index"), int):
            continue
        text = clean_text(chunk.get("content"))
        if text and re.search(r"(=|softmax|tanh|sqrt|\\bsum\\b|∑|⊤|\\^T|\\bwhere\\b.+\\bmatrix|\\bscore function\\b)", text, flags=re.IGNORECASE):
            return True
    return False


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
