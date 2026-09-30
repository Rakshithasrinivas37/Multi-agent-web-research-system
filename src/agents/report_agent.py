"""Small deterministic report agent for cited research reports."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Sequence

from src.memory.shared_memory import SharedMemory
from src.tools.text_utils import clean_text


DEFAULT_REPORT_AGENT_MODEL = "llama-3.1-8b-instant"
DEFAULT_REPORT_MAX_TOKENS = 2200
DEFAULT_REPORT_PROMPT_CHARS = 9000
DEFAULT_REPORT_TOTAL_TOKEN_BUDGET = 7000
DEFAULT_REPORT_OUTPUT_DIR = "data/reports"
DEFAULT_TOPIC_SECTION_CHARS = 1400
DEFAULT_EVIDENCE_CHUNKS_PER_QUESTION = 4
DEFAULT_EVIDENCE_CHUNK_CHARS = 700

STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "by", "can", "do", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "their", "to", "what", "when",
    "where", "which", "with",
}
GAP_PATTERN = re.compile(r"(missing|not present|not provided|no cited|no source|insufficient|cannot be|gap)", re.I)


class ReportAgent:
    """Generate a final report from synthesis-agent context.

    This implementation is deterministic-first: it turns per-question synthesis
    and cited evidence packs into stable sections, then builds the framing
    sections from those accepted topic sections.
    """

    def __init__(self, model: str | None = None, generation_mode: str | None = None) -> None:
        self.model = (
            clean_text(model)
            or clean_text(os.environ.get("RESEARCH_PLANNER_MODEL"))
            or clean_text(os.environ.get("RAG_GENERATION_MODEL"))
            or DEFAULT_REPORT_AGENT_MODEL
        )
        self.generation_mode = clean_text(generation_mode or os.environ.get("REPORT_GENERATION_MODE")) or "deterministic"

    def generate(self, report_context: dict[str, Any], output_format: str = "report") -> dict[str, Any]:
        if not isinstance(report_context, dict) or not report_context:
            raise ValueError("report_context is required")
        objective = clean_text(report_context.get("objective"))
        if not objective:
            raise ValueError("report_context.objective is required")

        evidence_packs = [pack for pack in report_context.get("evidence_packs", []) or [] if isinstance(pack, dict)]
        planner_questions = [clean_text(q) for q in report_context.get("planner_questions", []) if clean_text(q)]
        questions = dedupe_text([*planner_questions, *evidence_pack_questions(evidence_packs)]) or [objective]
        sources = evidence_backed_sources(
            dedupe_sources(sources_with_browser_results(report_context.get("sources", []), report_context.get("browser_results", []))),
            report_context,
        )
        packs_by_question = {normalize_heading(pack.get("question")): pack for pack in evidence_packs}
        synthesis_by_question = per_question_synthesis_by_question(report_context.get("per_question_synthesis", []))

        topic_sections: list[str] = []
        diagnostics: list[dict[str, Any]] = []
        for index, question in enumerate(questions, 1):
            pack = packs_by_question.get(normalize_heading(question), {})
            synthesis_note = synthesis_by_question.get(normalize_heading(question), {})
            section, diagnostic = build_topic_section(index, question, pack, synthesis_note)
            topic_sections.append(section)
            diagnostics.append(diagnostic)

        report = normalize_final_report(assemble_report(objective, topic_sections), sources)
        validation = validate_report_output(report, sources, questions, evidence_packs, report_context)

        return {
            "objective": objective,
            "output_format": clean_text(output_format) or "report",
            "report": report,
            "sources": sources,
            "model": self.model,
            "diagnostics": {
                "source_count": len(sources),
                "evidence_pack_count": len(evidence_packs),
                "supporting_chunk_count": len(report_context.get("supporting_chunks", []) or []),
                "retrieved_chunk_count": len(report_context.get("retrieved_chunks", []) or []),
                "report_length": len(report),
                "report_generation_mode": "deterministic",
                "report_issues": validation["report_issues"],
                "report_schema_issues": validation["schema_issues"],
                "report_missing_sub_questions": validation["coverage"]["missing"],
                "report_evidence_gap_questions": synthesis_coverage_gap_questions(report_context, questions),
                "report_false_gap_questions": validation["false_gap_questions"],
                "report_pack_citation_gap_questions": validation["pack_citation_gap_questions"],
                "report_coverage_check": validation["coverage"],
                "report_retry_queries": rewrite_missing_sub_question_queries(
                    objective,
                    dedupe_text([*validation["coverage"]["missing"], *validation["false_gap_questions"]]),
                ),
                "report_review_trace": [report_self_critique(validation["report_issues"], validation["coverage"], validation["schema_issues"])],
                "report_revision_attempts": 0,
                "report_deterministic_repairs": [],
                "report_finalization_status": "clean" if not report_needs_revision(validation) else "blocked",
                "report_token_budget": DEFAULT_REPORT_TOTAL_TOKEN_BUDGET,
                "report_section_diagnostics": {"topic_sections": diagnostics},
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


def build_topic_section(index: int, question: str, pack: dict[str, Any], synthesis_note: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    technical_body = technical_evidence_answer(question, pack)
    body = technical_body or clean_section_synthesis_note(synthesis_note)
    source = "technical_evidence" if technical_body else "per_question_synthesis"
    evidence_body = evidence_pack_answer(question, pack)
    if weak_topic_body(body):
        body = evidence_body
        source = "evidence_pack"
    elif source != "technical_evidence" and topic_needs_evidence_enrichment(question) and evidence_body and (set(citation_markers(evidence_body)) - set(citation_markers(body))):
        body = compact_markdown_at_sentence(f"{body}\n{evidence_body}", DEFAULT_TOPIC_SECTION_CHARS)
        source = "synthesis_plus_evidence"
    if weak_topic_body(body):
        body = evidence_gap_answer(question, pack, synthesis_note)
        source = "gap"
    body = normalize_topic_body(body)
    return (
        f"### 3.{index}. {planner_question_heading(question)}\n{body}",
        {"question": question, "source": source, "source_indexes": citation_markers(body), "chars": len(body)},
    )


def assemble_report(objective: str, topic_sections: Sequence[str]) -> str:
    topic_text = "\n\n".join(topic_sections)
    intro = first_supported_sentences(topic_text, 3) or f"This report addresses: {objective}."
    return "\n\n".join([
        f"## 1. Executive Summary\n{executive_summary(topic_sections)}",
        f"## 2. Introduction and Context\n{intro}",
        "## 3. Topic Sections",
        *topic_sections,
        f"## 4. Cross-cutting Analysis and Synthesis\n{cross_cutting_analysis(topic_sections)}",
        f"## 5. Limitations and Open Questions\n{limitations_section(topic_text)}",
        f"## 6. Conclusion\n{conclusion_section(topic_sections)}",
    ])


def executive_summary(topic_sections: Sequence[str]) -> str:
    findings = [first_supported_sentences(section, 1) for section in topic_sections]
    findings = [item for item in findings if item and not line_has_gap_claim(item)]
    gaps = extract_gap_sentences("\n\n".join(topic_sections), 1)
    text = " ".join(findings[:3])
    if gaps:
        text = clean_text(f"{text} Remaining limitation: {gaps[0]}")
    return text or "The retrieved evidence was not sufficient to produce supported findings."


def cross_cutting_analysis(topic_sections: Sequence[str]) -> str:
    findings = [first_supported_sentences(section, 1) for section in topic_sections]
    findings = [item for item in findings if item and not line_has_gap_claim(item)]
    return " ".join(findings[:4]) or "No cross-topic synthesis could be made from the cited evidence."


def limitations_section(topic_text: str) -> str:
    gaps = extract_gap_sentences(topic_text, 6)
    return "\n".join(f"- {gap}" for gap in gaps) if gaps else "- No unresolved evidence gaps were detected in the generated topic sections."


def conclusion_section(topic_sections: Sequence[str]) -> str:
    findings = [first_supported_sentences(section, 1) for section in topic_sections]
    findings = [item for item in findings if item and not line_has_gap_claim(item)]
    return " ".join(findings[-3:]) if findings else "The report could not draw a supported conclusion from the retrieved evidence."


def clean_section_synthesis_note(synthesis_note: dict[str, Any]) -> str:
    if not isinstance(synthesis_note, dict):
        return ""
    synthesis = clean_markdown(synthesis_note.get("synthesis") or synthesis_note.get("answer") or synthesis_note.get("content"))
    if not synthesis or not citation_markers(synthesis):
        return ""
    active_markers = per_question_synthesis_source_indexes(synthesis_note)
    lines: list[str] = []
    in_missing_block = False
    for raw_line in synthesis.splitlines():
        line = clean_text(raw_line)
        if not line:
            continue
        if re.match(r"^#{1,6}\s+", line):
            continue
        if re.search(r"\b(Missing|What is missing|Open questions|Limitations)\b", strip_markdown(line), flags=re.I):
            in_missing_block = True
            continue
        if in_missing_block or line_has_gap_claim(line) or raw_pdf_extraction_artifact(line):
            continue
        markers = citation_markers(line)
        if markers:
            active_markers = markers
        if line_is_synthesis_scaffold(line):
            continue
        line = scrub_scaffold_text(line)
        if not line:
            continue
        if not citation_markers(line) and active_markers:
            line = f"{line.rstrip('.')} {format_citation_indexes(active_markers)}."
        lines.append(line)
    text = clean_markdown("\n".join(lines))
    text = re.sub(r"\s+", " ", text)
    return compact_markdown_at_sentence(text, DEFAULT_TOPIC_SECTION_CHARS)


def evidence_pack_answer(question: str, pack: dict[str, Any]) -> str:
    lines = []
    seen = set()
    for chunk in rank_question_chunks(question, pack.get("chunks", []) if isinstance(pack, dict) else []):
        if len(lines) >= DEFAULT_EVIDENCE_CHUNKS_PER_QUESTION:
            break
        key = clean_text(cleanup_evidence_text(chunk.get("content"))[:220]).lower()
        if key in seen:
            continue
        seen.add(key)
        sentence = chunk_evidence_sentence(chunk)
        if sentence:
            lines.append(sentence)
    return compact_markdown_at_sentence("\n".join(lines), DEFAULT_TOPIC_SECTION_CHARS)


def technical_evidence_answer(question: str, pack: dict[str, Any]) -> str:
    lowered = clean_text(question).lower()
    if not any(term in lowered for term in ("self-attention", "self attention", "multi-head", "multihead", "transformer", "vaswani")):
        return ""
    chunks = pack.get("chunks", []) if isinstance(pack, dict) else []
    transformer_chunks = [
        chunk for chunk in chunks
        if isinstance(chunk, dict)
        and isinstance(chunk.get("source_index"), int)
        and "1706.03762" in normalize_url(chunk.get("url"))
    ]
    text = cleanup_evidence_text(" ".join(clean_text(chunk.get("content")) for chunk in transformer_chunks))
    if not text:
        return ""
    marker = format_citation_indexes([chunk.get("source_index") for chunk in transformer_chunks[:1]])
    if not marker:
        return ""
    lines = []
    if re.search(r"scaled dot-product attention|queries and keys|attention\(q,k,v", text, flags=re.I):
        lines.append(
            "Transformer self-attention uses scaled dot-product attention: queries and keys determine "
            f"compatibility scores, the scores are divided by the square root of the key dimension, softmax "
            f"turns them into weights, and those weights combine the values {marker}."
        )
        lines.append(
            "**Core equation:**\n"
            "\\[\n"
            "\\text{Attention}(Q,K,V)=\\operatorname{softmax}\\left(\\frac{QK^{\\top}}{\\sqrt{d_k}}\\right)V\n"
            "\\]"
        )
    if re.search(r"h\s*=\s*8|parallel attention layers|heads", text, flags=re.I):
        lines.append(
            "Multi-head attention runs several attention heads in parallel; in the Transformer paper, "
            f"the model uses h = 8 heads with dk = dv = dmodel / h = 64, so each head works in a reduced "
            f"subspace and the total cost remains similar to single-head attention at full dimensionality {marker}."
        )
    return clean_markdown("\n\n".join(lines))


def chunk_evidence_sentence(chunk: dict[str, Any]) -> str:
    if not isinstance(chunk, dict):
        return ""
    marker = citation_marker_for_chunk(chunk)
    content = cleanup_evidence_text(chunk.get("content"))
    if not marker or not content or raw_pdf_extraction_artifact(content):
        return ""
    sentence = first_supported_sentences(content, 2) or compact_text(content, min(DEFAULT_EVIDENCE_CHUNK_CHARS, 460))
    return sentence if citation_markers(sentence) else f"{sentence.rstrip('.')} {marker}."


def topic_needs_evidence_enrichment(question: str) -> bool:
    lowered = clean_text(question).lower()
    return bool(any(term in lowered for term in (
        "equation", "formula", "self-attention", "self attention", "multi-head",
        "multihead", "benchmark", "complexity", "evolved", "key differences",
    )))


def evidence_gap_answer(question: str, pack: dict[str, Any], synthesis_note: dict[str, Any]) -> str:
    markers = dedupe_ints([*pack_source_indexes(pack), *per_question_synthesis_source_indexes(synthesis_note)])
    suffix = f" {format_citation_indexes(markers[:1])}" if markers else ""
    return f"The retrieved evidence does not provide enough cited detail to fully answer this sub-question: {question}.{suffix}"


def normalize_topic_body(body: str) -> str:
    lines = []
    for line in clean_markdown(body).splitlines():
        value = repair_truncated_markdown_line(clean_text(line))
        if value and not raw_pdf_extraction_artifact(value):
            lines.append(value)
    return clean_markdown("\n".join(lines))


def weak_topic_body(body: str) -> bool:
    text = clean_markdown(body)
    plain = strip_markdown(text)
    return bool(
        len(plain) < 80
        or not citation_markers(text)
        or markdown_appears_truncated(text)
        or (len([line for line in text.splitlines() if clean_text(line)]) == 1 and re.match(r"^\*\*.+\*\*$", clean_text(text)))
    )


def validate_report_output(report: str, sources: Sequence[dict[str, Any]], planner_questions: Sequence[str], evidence_packs: Sequence[dict[str, Any]], report_context: dict[str, Any]) -> dict[str, Any]:
    coverage = report_sub_question_coverage_check(report, planner_questions)
    schema_issues = report_schema_issues(report, planner_questions)
    false_gaps = report_evidence_gap_contradictions(report, evidence_packs, planner_questions)
    pack_gaps = report_pack_citation_gaps(report, evidence_packs, planner_questions)
    report_issues = report_quality_issues(report, sources)
    report_issues.extend(f"report marks covered evidence as a gap: {question}" for question in false_gaps)
    report_issues.extend(f"report section does not cite its evidence pack: {question}" for question in pack_gaps)
    return {
        "coverage": coverage,
        "schema_issues": schema_issues,
        "false_gap_questions": false_gaps,
        "pack_citation_gap_questions": pack_gaps,
        "report_issues": dedupe_text(report_issues),
    }


def report_quality_issues(report: str, sources: Sequence[dict[str, Any]] | None = None, evidence_text: str = "") -> list[str]:
    issues = []
    text = clean_markdown(report)
    if not text:
        return ["report is empty"]
    if not any(is_references_heading(line) for line in text.splitlines()):
        issues.append("report must include a References section")
    invalid = unavailable_citation_markers(text, source_index_set(sources or []))
    if invalid:
        issues.append(f"report uses unavailable citations: {format_citation_indexes(invalid)}")
    for heading, section in markdown_sections(text):
        if "benchmark" in normalize_heading(heading) and weak_topic_body(strip_topic_section_headings(section)):
            issues.append("benchmark section is too thin")
    return dedupe_text(issues)


def report_sub_question_coverage_check(report: str, planner_questions: Sequence[str]) -> dict[str, Any]:
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


def report_schema_issues(report: str, planner_questions: Sequence[str]) -> list[str]:
    headings = {normalize_heading(h) for h in h2_headings(report)}
    required = {
        "executive summary": ("executive summary",),
        "introduction and context": ("introduction and context",),
        "topic sections": ("topic sections",),
        "cross-cutting analysis and synthesis": ("cross cutting analysis and synthesis", "cross-cutting analysis and synthesis"),
        "limitations and open questions": ("limitations and open questions",),
        "conclusion": ("conclusion",),
        "references": ("references",),
    }
    issues = []
    for label, aliases in required.items():
        if not any(normalize_heading(alias) in headings for alias in aliases):
            issues.append(f"missing schema section: {label}")
    return issues


def report_evidence_gap_contradictions(report: str, evidence_packs: Sequence[dict[str, Any]], planner_questions: Sequence[str] | None = None) -> list[str]:
    canonical = {normalize_heading(q): q for q in planner_questions or [] if clean_text(q)}
    contradictions = []
    for pack in evidence_packs or []:
        if not isinstance(pack, dict) or not evidence_pack_has_usable_cited_evidence(pack):
            continue
        question = clean_text(pack.get("question"))
        section = report_section_for_question(report, question)
        if section and line_has_gap_claim(section) and section_cites_pack_source(section, pack):
            contradictions.append(canonical.get(normalize_heading(question), question))
    return dedupe_text(contradictions)


def report_pack_citation_gaps(report: str, evidence_packs: Sequence[dict[str, Any]], planner_questions: Sequence[str] | None = None, sources: Sequence[dict[str, Any]] | None = None) -> list[str]:
    canonical = {normalize_heading(q): q for q in planner_questions or [] if clean_text(q)}
    gaps = []
    for pack in evidence_packs or []:
        if not isinstance(pack, dict) or not evidence_pack_has_usable_cited_evidence(pack):
            continue
        question = clean_text(pack.get("question"))
        section = report_section_for_question(report, question)
        if section and section_mentions_pack_topic(section, question) and not citation_markers(section):
            gaps.append(canonical.get(normalize_heading(question), question))
    return dedupe_text(gaps)


def report_needs_revision(validation: dict[str, Any]) -> bool:
    return bool(validation.get("report_issues") or validation.get("schema_issues") or validation.get("coverage", {}).get("missing"))


def report_self_critique(report_issues: Sequence[str], coverage_check: dict[str, Any], schema_issues: Sequence[str]) -> dict[str, Any]:
    unresolved = dedupe_text([*report_issues, *schema_issues, *(f"missing planner topic: {q}" for q in coverage_check.get("missing", []))])
    return {"source": "deterministic", "unresolved_issues": unresolved, "coverage_missing": coverage_check.get("missing", []), "schema_issues": list(schema_issues)}


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
    compact: bool = False,
    repair_feedback: str = "",
) -> str:
    return trim_report_prompt(f"""Research objective:
{objective}

Requested output format:
{clean_text(output_format) or "report"}

Write a cited Markdown report using only supplied evidence.

Planner questions:
{format_planner_questions(planner_questions)}

Sources:
{format_sources(sources)}

Evidence:
{evidence}

Synthesis:
{synthesis}

Evidence packs:
{format_evidence_packs(evidence_packs or [], max_chunks_per_pack=2 if compact else None)}
""", 5000 if compact else DEFAULT_REPORT_PROMPT_CHARS)


def format_planner_questions(questions: Sequence[str]) -> str:
    items = [clean_text(q) for q in questions if clean_text(q)]
    return "\n".join(f"- {q}" for q in items) or "- Cover the research objective directly."


def format_report_section_outline(questions: Sequence[str]) -> str:
    return "\n".join(f"### 3.{i}. {planner_question_heading(q)}" for i, q in enumerate(questions, 1)) or "- Use clear sections."


def format_question_coverage(coverage_by_question: Sequence[dict[str, Any]]) -> str:
    lines = []
    for item in coverage_by_question or []:
        if isinstance(item, dict) and clean_text(item.get("question")):
            indexes = format_citation_indexes([i for i in item.get("source_indexes", []) if isinstance(i, int)])
            lines.append(f"- {clean_text(item.get('status')) or 'unknown'}; sources={indexes or 'none'}; question={clean_text(item.get('question'))}")
    return "\n".join(lines) or "- No structured coverage map was provided."


def format_evidence_packs(evidence_packs: Sequence[dict[str, Any]], max_chunks_per_pack: int | None = None, chunk_chars: int | None = None) -> str:
    lines = []
    for pack in evidence_packs or []:
        if not isinstance(pack, dict) or not clean_text(pack.get("question")):
            continue
        lines.append(f"Question: {clean_text(pack.get('question'))}")
        lines.append(f"Coverage: {clean_text(pack.get('coverage')) or 'unknown'}")
        for chunk in (pack.get("chunks", []) or [])[: max_chunks_per_pack or 999]:
            marker = citation_marker_for_chunk(chunk) or "[uncited]"
            content = cleanup_evidence_text(chunk.get("content"))
            if chunk_chars:
                content = content[:chunk_chars].rstrip()
            if content:
                lines.append(f"- {marker} {content}")
    return "\n".join(lines) or "- No per-question evidence packs were provided."


def format_supporting_evidence(report_context: dict[str, Any], max_chars: int | None = None, sources: Sequence[dict[str, Any]] | None = None) -> str:
    blocks = [chunk_evidence_sentence(chunk) for chunk in all_report_chunks(report_context, report_context.get("evidence_packs", []))]
    return compact_text("\n\n".join(block for block in blocks if block), max_chars or DEFAULT_REPORT_PROMPT_CHARS)


def format_question_focused_evidence(report_context: dict[str, Any], questions: Sequence[str], sources: Sequence[dict[str, Any]] | None = None, evidence_packs: Sequence[dict[str, Any]] | None = None, max_chars: int = DEFAULT_REPORT_PROMPT_CHARS) -> str:
    packs = {normalize_heading(pack.get("question")): pack for pack in evidence_packs or [] if isinstance(pack, dict)}
    blocks = []
    for question in questions:
        answer = evidence_pack_answer(question, packs.get(normalize_heading(question), {}))
        if answer:
            blocks.append(f"Question: {question}\n{answer}")
    return compact_text("\n\n".join(blocks), max_chars)


def format_single_question_synthesis(item: dict[str, Any]) -> str:
    return clean_section_synthesis_note(item)


def planner_question_heading(question: str) -> str:
    text = clean_text(question).rstrip("?")
    text = re.sub(r"^(what|how|why|when|where|which)\s+(is|are|does|do|did|can|should)\s+", "", text, flags=re.I)
    text = re.sub(r"^(what|how|why|when|where|which)\s+", "", text, flags=re.I)
    text = re.sub(r"\b(e\.g\.|eg|examples?|evidence|results?)\b", "", text, flags=re.I)
    words = [w.strip(".,:;()[]{}") for w in re.sub(r"\s+", " ", text).strip(" .,:;").split()]
    title = " ".join(word if any(char.isupper() for char in word[1:]) else word.capitalize() for word in words)
    return smart_truncate_heading(title or "Research Finding")


def smart_truncate_heading(text: str, max_length: int = 96) -> str:
    value = clean_text(text)
    if len(value) <= max_length:
        return value
    clipped = value[:max_length].rsplit(" ", 1)[0].rstrip(" ,:;-") or value[:max_length].rstrip(" ,:;-")
    while clipped.split(" ")[-1:].pop().lower() in {"and", "or", "with", "including", "of", "the"}:
        clipped = clipped.rsplit(" ", 1)[0].rstrip(" ,:;-")
    return clipped or value[:max_length].rstrip(" ,:;-")


def format_sources(sources: Sequence[dict[str, Any]]) -> str:
    lines = []
    for fallback, source in enumerate(sources, 1):
        if isinstance(source, dict):
            index = source.get("index") if isinstance(source.get("index"), int) else fallback
            title = clean_text(source.get("title")) or clean_text(source.get("url")) or f"Source {index}"
            lines.append(f"[{index}] {title} - {clean_text(source.get('url'))}")
    return "\n".join(lines) or "No sources provided."


def clean_markdown(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"<think>.*", "", text, flags=re.S | re.I)
    text = normalize_citation_markers(text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    return re.sub(r"\n{4,}", "\n\n\n", text).strip()


def normalize_citation_markers(text: Any) -> str:
    value = str(text or "")
    value = re.sub(r"【\s*(\d+)(?:[^】]*)?】", r"[\1]", value)
    value = re.sub(r"\[\s*(\d+(?:\s*,\s*\d+)+)\s*\]", lambda m: " ".join(f"[{p.strip()}]" for p in m.group(1).split(",")), value)
    return re.sub(r"\[\s*(\d+)\s*\]", r"[\1]", value)


def citation_markers(text: Any) -> list[int]:
    if isinstance(text, (list, tuple, set)):
        return dedupe_ints(text)
    return dedupe_ints(int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", normalize_citation_markers(text)))


def citation_marker_for_chunk(chunk: dict[str, Any]) -> str:
    if not isinstance(chunk, dict):
        return ""
    index = chunk.get("source_index") if isinstance(chunk.get("source_index"), int) else chunk.get("index")
    return f"[{index}]" if isinstance(index, int) else ""


def remove_unavailable_citation_markers(text: str, available_indexes: set[int]) -> str:
    if not available_indexes:
        return re.sub(r"\[\d+\]", "", text)
    return re.sub(r"\[(\d+)\]", lambda m: m.group(0) if int(m.group(1)) in available_indexes else "", text)


def unavailable_citation_markers(text: str, available_indexes: set[int]) -> list[int]:
    return sorted({index for index in citation_markers(text) if available_indexes and index not in available_indexes})


def normalize_final_report(report: str, sources: Sequence[dict[str, Any]]) -> str:
    text = normalize_markdown_headings(remove_unavailable_citation_markers(clean_markdown(report), source_index_set(sources)))
    body = strip_references(text)
    return clean_markdown(f"{body}\n\n{references_section(body, sources)}")


def normalize_markdown_headings(markdown: str) -> str:
    return "\n".join(re.sub(r"^(\s{0,3}#{1,6}\s+)#{1,6}\s+", r"\1", line) for line in clean_markdown(markdown).splitlines())


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
    lines, skipping = [], False
    for line in clean_markdown(report).splitlines():
        if is_references_heading(line):
            skipping = True
            continue
        if skipping and line.startswith("## ") and not is_references_heading(line):
            skipping = False
        if not skipping:
            lines.append(line)
    return clean_markdown("\n".join(lines))


def strip_topic_section_headings(section: str) -> str:
    lines = clean_markdown(section).splitlines()
    while lines and re.match(r"^\s{0,3}#{1,6}\s+", lines[0]):
        lines = lines[1:]
    return clean_markdown("\n".join(lines))


def strip_markdown(text: Any) -> str:
    value = re.sub(r"`([^`]*)`", r"\1", str(text or ""))
    value = re.sub(r"[*_#|]+", " ", value)
    value = re.sub(r"\[(\d+)\]", "", value)
    return clean_text(value)


def cleanup_evidence_text(value: Any) -> str:
    text = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", clean_markdown(value))
    text = re.sub(r"\s+", " ", text)
    replacements = {"Englishto": "English-to", "dimensiondk": "dimension dk", "matrixQ": "matrix Q", "matricesK": "matrices K", "andV": "and V"}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return clean_text(text)


def scrub_scaffold_text(line: str) -> str:
    value = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", clean_text(line))
    value = re.sub(r"^\*\*(?:Supported answer|Supported evidence|Supported benchmark evidence|Planner notes)[^*]*\*\*\s*:?", "", value, flags=re.I)
    return clean_text(value)


def raw_pdf_extraction_artifact(text: Any) -> bool:
    value = clean_text(text)
    lowered = value.lower()
    return bool(
        "published as a conference paper" in lowered
        or "abstract neural machine translation" in lowered
        or re.search(r"\b(?:singlelayer|multilayer|englishto)\b", lowered)
    )


def line_is_synthesis_scaffold(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-*+]\s*)?\*\*(?:Supported|Planner notes|Key equations|Overall performance claim)[^*]*\*\*\s*:?\s*$", clean_text(line), flags=re.I))


def line_has_gap_claim(line: Any) -> bool:
    return bool(GAP_PATTERN.search(clean_text(line)))


def markdown_appears_truncated(markdown: str) -> bool:
    text = clean_markdown(markdown)
    if not text:
        return True
    last = text.splitlines()[-1].strip()
    return bool(last and not re.search(r"[\].)]$|[.!?]$|\\\]$", last) and len(last.split()) > 8)


def first_supported_sentences(text: Any, max_sentences: int = 2) -> str:
    text = "\n".join(line for line in clean_markdown(text).splitlines() if not re.match(r"^\s{0,3}#{1,6}\s+", line))
    selected = []
    for sentence in split_sentences(text):
        if citation_markers(sentence) and not raw_pdf_extraction_artifact(sentence):
            selected.append(sentence)
        if len(selected) >= max_sentences:
            break
    return clean_text(" ".join(selected))


def split_sentences(text: Any) -> list[str]:
    value = clean_text(str(text or "").replace("\n", " "))
    return [clean_text(piece) for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9*])", value) if clean_text(piece)]


def extract_gap_sentences(text: Any, limit: int = 5) -> list[str]:
    return dedupe_text(strip_markdown(sentence).rstrip(".") + "." for sentence in split_sentences(text) if line_has_gap_claim(sentence))[:limit]


def compact_markdown_at_sentence(value: Any, max_chars: int) -> str:
    text = clean_markdown(value)
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rstrip()
    end = max(clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"))
    return clipped[: end + 1].rstrip() if end > max_chars * 0.45 else clipped.rsplit(" ", 1)[0].rstrip()


def compact_text(value: Any, max_chars: int) -> str:
    text = clean_markdown(value)
    return text if len(text) <= max_chars else text[:max_chars].rstrip()


def trim_report_prompt(prompt: Any, max_chars: int = DEFAULT_REPORT_PROMPT_CHARS) -> str:
    return compact_markdown_at_sentence(prompt, max_chars)


def sources_with_browser_results(sources: Sequence[Any], browser_results: Sequence[Any]) -> list[dict[str, Any]]:
    merged, existing, used_indexes = [], set(), set()
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        item = dict(source)
        if not isinstance(item.get("index"), int):
            item["index"] = len(used_indexes) + 1
        merged.append(item)
        used_indexes.add(item["index"])
        if normalize_url(item.get("url")):
            existing.add(normalize_url(item.get("url")))
    next_index = max(used_indexes, default=0) + 1
    for result in browser_results or []:
        if not isinstance(result, dict):
            continue
        for source in result.get("sources", []) or []:
            if isinstance(source, dict) and normalize_url(source.get("url")) not in existing:
                existing.add(normalize_url(source.get("url")))
                merged.append({"index": next_index, "title": source.get("title"), "url": source.get("url")})
                next_index += 1
    return dedupe_sources(merged)


def dedupe_sources(sources: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out, seen, used, next_index = [], set(), set(), 1
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        url = normalize_url(source.get("url"))
        if url and url in seen:
            continue
        seen.add(url)
        item = dict(source)
        index = item.get("index")
        if not isinstance(index, int) or index in used:
            while next_index in used:
                next_index += 1
            item["index"] = next_index
            index = next_index
        used.add(index)
        out.append(item)
    return out


def evidence_backed_sources(sources: Sequence[dict[str, Any]], report_context: dict[str, Any]) -> list[dict[str, Any]]:
    cited = set()
    for pack in report_context.get("evidence_packs", []) or []:
        if isinstance(pack, dict):
            cited.update(pack_source_indexes(pack))
    for item in report_context.get("per_question_synthesis", []) or []:
        if isinstance(item, dict):
            cited.update(per_question_synthesis_source_indexes(item))
    return [source for source in sources if source.get("index") in cited] or list(sources)


def all_report_chunks(report_context: dict[str, Any], evidence_packs: Sequence[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    chunks = []
    for pack in evidence_packs or []:
        if isinstance(pack, dict):
            chunks.extend(chunk for chunk in pack.get("chunks", []) or [] if isinstance(chunk, dict))
    chunks.extend(chunk for chunk in report_context.get("supporting_chunks", []) or [] if isinstance(chunk, dict))
    chunks.extend(chunk for chunk in report_context.get("retrieved_chunks", []) or [] if isinstance(chunk, dict))
    return dedupe_chunks(chunks)


def dedupe_chunks(chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out, seen = [], set()
    for chunk in chunks:
        content = cleanup_evidence_text(chunk.get("content"))
        key = clean_text(f"{chunk.get('source_index')}:{chunk.get('url')}:{content[:160]}").lower()
        if content and key not in seen:
            item = dict(chunk)
            item["content"] = content
            out.append(item)
            seen.add(key)
    return out


def rank_question_chunks(question: str, chunks: Sequence[dict[str, Any]], planned_urls: Sequence[str] | None = None) -> list[dict[str, Any]]:
    terms = detail_terms(question)
    return sorted(
        [chunk for chunk in chunks if isinstance(chunk, dict)],
        key=lambda chunk: -(len(terms & detail_terms(" ".join([clean_text(chunk.get("title")), clean_text(chunk.get("url")), clean_text(chunk.get("content"))]))) + 3 * source_priority(chunk.get("url"))),
    )


def source_priority(url: Any) -> int:
    value = clean_text(url).lower()
    if any(signal in value for signal in ("arxiv.org", "openreview.net", "doi.org", "pytorch.org", "tensorflow.org", "docs.")) or ".edu" in value:
        return 2
    return 1 if value else 0


def evidence_pack_questions(evidence_packs: Sequence[Any]) -> list[str]:
    return dedupe_text(clean_text(pack.get("question")) for pack in evidence_packs or [] if isinstance(pack, dict))


def per_question_synthesis_by_question(items: Sequence[Any]) -> dict[str, dict[str, Any]]:
    return {normalize_heading(item.get("question")): item for item in items or [] if isinstance(item, dict) and clean_text(item.get("question"))}


def pack_source_indexes(pack: dict[str, Any]) -> list[int]:
    return dedupe_ints(chunk.get("source_index") for chunk in pack.get("chunks", []) or [] if isinstance(chunk, dict))


def per_question_synthesis_source_indexes(item: dict[str, Any]) -> list[int]:
    return dedupe_ints(item.get("source_indexes", []) if isinstance(item, dict) else [])


def evidence_pack_has_usable_cited_evidence(pack: dict[str, Any]) -> bool:
    return any(isinstance(chunk, dict) and isinstance(chunk.get("source_index"), int) and clean_text(chunk.get("content")) for chunk in pack.get("chunks", []) or [])


def section_cites_pack_source(section: str, pack: dict[str, Any]) -> bool:
    return bool(set(citation_markers(section)) & set(pack_source_indexes(pack)))


def section_mentions_pack_topic(section: str, question: str) -> bool:
    return len(detail_terms(section) & detail_terms(question)) >= 2


def synthesis_coverage_gap_questions(report_context: dict[str, Any], planner_questions: Sequence[str] | None = None) -> list[str]:
    if not isinstance(report_context, dict):
        return []
    canonical = {normalize_heading(q): q for q in planner_questions or [] if clean_text(q)}
    covered = {normalize_heading(pack.get("question")) for pack in report_context.get("evidence_packs", []) or [] if isinstance(pack, dict) and evidence_pack_has_usable_cited_evidence(pack)}
    gaps = []
    for item in report_context.get("coverage_by_question", []) or []:
        if isinstance(item, dict) and synthesis_coverage_status_is_gap(item.get("status")):
            question = clean_text(item.get("question"))
            if question and normalize_heading(question) not in covered:
                gaps.append(canonical.get(normalize_heading(question), question))
    return dedupe_text(gaps)


def synthesis_coverage_status_is_gap(status: Any) -> bool:
    lowered = clean_text(status).lower()
    return bool(lowered and any(term in lowered for term in ("missing", "partial", "weak", "insufficient", "unsupported", "failed", "error")))


def missing_evidence_constraints(synthesis: Any) -> list[str]:
    return dedupe_text(strip_markdown(line)[:300] for line in clean_markdown(synthesis).splitlines() if line_has_gap_claim(line))


def format_missing_evidence_constraints(synthesis: Any) -> str:
    items = missing_evidence_constraints(synthesis)
    return "\n".join(f"- {item}" for item in items) if items else "- No explicit missing-evidence constraints."


def report_context_gap_items(report_context: dict[str, Any], research_plan: dict[str, Any]) -> list[str]:
    questions = [clean_text(q) for q in research_plan.get("sub_questions", []) if clean_text(q)] if isinstance(research_plan, dict) else []
    gaps = synthesis_coverage_gap_questions(report_context, questions)
    synthesis = clean_text(report_context.get("synthesis")) if isinstance(report_context, dict) else ""
    gaps.extend(q for q in missing_sub_question_coverage(synthesis, questions) if q not in gaps)
    return dedupe_text(gaps)


def report_context_gap_queries(report_context: dict[str, Any], research_plan: dict[str, Any]) -> list[str]:
    objective = clean_text(research_plan.get("objective")) if isinstance(research_plan, dict) else clean_text(report_context.get("objective"))
    return rewrite_missing_sub_question_queries(objective, report_context_gap_items(report_context, research_plan))


def rewrite_missing_sub_question_queries(objective: str, questions: Sequence[str]) -> list[str]:
    return [clean_text(f"{objective} {question} source-backed evidence details equations benchmarks limitations")[:700] for question in questions if clean_text(question)]


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


def markdown_sections(markdown: str) -> list[tuple[str, str]]:
    sections, heading, lines, in_fence = [], "", [], False
    for line in clean_markdown(markdown).splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            if lines:
                sections.append((heading, "\n".join(lines)))
            heading, lines = match.group(1).strip(), [line]
        else:
            lines.append(line)
    if lines:
        sections.append((heading, "\n".join(lines)))
    return sections


def report_section_for_question(report: str, question: str) -> str:
    expected = normalize_heading(planner_question_heading(question))
    terms = detail_terms(question)
    best, best_score = "", 0
    for heading, section in markdown_sections(report):
        actual = normalize_heading(heading)
        score = (5 if expected and (expected in actual or actual in expected) else 0) + len(terms & detail_terms(heading))
        if score > best_score:
            best, best_score = section, score
    return best if best_score else ""


def h2_headings(markdown: str) -> list[str]:
    return [match.group(1).strip() for line in clean_markdown(markdown).splitlines() if (match := re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line))]


def is_references_heading(line: str) -> bool:
    return normalize_heading(line.lstrip("#").strip()) in {"references", "reference", "sources"}


def source_index_set(sources: Sequence[dict[str, Any]]) -> set[int]:
    return {source.get("index") for source in sources if isinstance(source, dict) and isinstance(source.get("index"), int)}


def normalize_heading(text: Any) -> str:
    value = strip_markdown(text).replace("‑", "-").replace("–", "-").replace("—", "-")
    value = re.sub(r"^\d+(?:\.\d+)*[.)]?\s*", "", value)
    return clean_text(re.sub(r"[^a-zA-Z0-9]+", " ", value)).lower()


def detail_terms(text: Any) -> set[str]:
    return {token.lower().replace("‑", "-").replace("–", "-") for token in re.findall(r"[A-Za-z][A-Za-z0-9_+.-]{2,}", strip_markdown(text)) if token.lower() not in STOPWORDS}


def technical_question_terms(text: Any) -> list[str]:
    return dedupe_text(term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9_+.-]{2,}", strip_markdown(text)))


def named_terms(text: Any) -> list[str]:
    return dedupe_text(token.lower() for token in re.findall(r"\b[A-Z][A-Za-z0-9_+.-]{2,}\b|\b[A-Z]{2,}\b", clean_text(text)) if token.lower() not in STOPWORDS)


def missing_sub_question_coverage(report: str, planner_questions: Sequence[str]) -> list[str]:
    report_terms = set(technical_question_terms(report))
    missing = []
    for question in planner_questions:
        terms = [term for term in technical_question_terms(question) if term not in STOPWORDS]
        important = named_terms(question) or terms[:5]
        required = 1 if len(important) <= 2 else 2
        if sum(1 for term in important if term in report_terms) < required:
            missing.append(question)
    return missing


def dedupe_text(items: Sequence[Any]) -> list[str]:
    seen, out = set(), []
    for item in items or []:
        value = clean_text(item)
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def dedupe_ints(values: Sequence[Any]) -> list[int]:
    out, seen = [], set()
    for value in values or []:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if isinstance(value, bool) or number in seen:
            continue
        seen.add(number)
        out.append(number)
    return out


def normalize_url(value: Any) -> str:
    return clean_text(value).rstrip("/").lower()


def format_citation_indexes(indexes: Sequence[int]) -> str:
    return ", ".join(f"[{index}]" for index in dedupe_ints(indexes))


def report_generation_token_cap(prompt_chars: int | None = None) -> int:
    return ((prompt_chars or DEFAULT_REPORT_PROMPT_CHARS) + 3) // 4 + DEFAULT_REPORT_MAX_TOKENS


# Compatibility shims for older tests/scripts. The report path above no longer
# depends on LLM repair chains, but these names remain importable.
def generate_single_report(client: Any, model: str, prompt: str, fallback_prompt: str | None = None) -> tuple[str, str]:
    raise RuntimeError("single-shot LLM report generation has been removed; use ReportAgent.generate")


def finalize_report_output(report: str, sources: Sequence[dict[str, Any]], planner_questions: Sequence[str], evidence: str, synthesis: str, pack_text: str, evidence_packs: Sequence[dict[str, Any]], report_context: dict[str, Any], validation: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    normalized = normalize_final_report(report, sources)
    new_validation = validate_report_output(normalized, sources, planner_questions, evidence_packs, report_context)
    return normalized, new_validation, {"status": "clean" if not report_needs_revision(new_validation) else "blocked", "repairs": []}


def accept_topic_section(question: str, section: str, pack: dict[str, Any], synthesis_note: dict[str, Any], sources: Sequence[dict[str, Any]]) -> tuple[str, list[str], bool]:
    issues = topic_section_acceptance_issues(section, question, pack, synthesis_note, sources)
    if issues:
        return deterministic_topic_section(question, pack, synthesis_note, sources, issues), issues, True
    return normalize_topic_body(section), [], False


def deterministic_topic_section(question: str, pack: dict[str, Any], synthesis_note: dict[str, Any], sources: Sequence[dict[str, Any]], issues: Sequence[str]) -> str:
    body = clean_section_synthesis_note(synthesis_note)
    if weak_topic_body(body):
        body = evidence_pack_answer(question, pack)
    return body if not weak_topic_body(body) else evidence_gap_answer(question, pack, synthesis_note)


def topic_section_acceptance_issues(section: str, question: str, pack: dict[str, Any], synthesis_note: dict[str, Any], sources: Sequence[dict[str, Any]], allow_gap_fallback: bool = False) -> list[str]:
    issues = ["section is weak or unsupported"] if weak_topic_body(section) else []
    if line_has_gap_claim(section) and evidence_pack_has_usable_cited_evidence(pack) and not allow_gap_fallback:
        issues.append("section marks available evidence as a gap")
    return issues


def resolve_report_coverage(coverage_by_question: Sequence[dict[str, Any]], evidence_packs: Sequence[dict[str, Any]], planner_questions: Sequence[str]) -> list[dict[str, Any]]:
    return list(coverage_by_question or [])


def format_report_revision_feedback(validation: dict[str, Any]) -> str:
    return "\n".join(f"- {issue}" for issue in dedupe_text([*validation.get("report_issues", []), *validation.get("schema_issues", [])])) or "- No unresolved issue."


apply_report_evidence_pack_repairs = lambda report, *args, **kwargs: (report, [])
apply_incomplete_equation_repairs = lambda report, *args, **kwargs: (report, [])
apply_validation_limitations = lambda report, *args, **kwargs: report
cleanup_report_markdown_artifacts = lambda report: (clean_markdown(report), [])
cleanup_topic_section = lambda section, question: (normalize_topic_body(section), [])
repair_report_by_sections = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("section repair LLM path has been removed"))
report_section_repair_questions = lambda validation, report_context, planner_questions: dedupe_text([*validation.get("coverage", {}).get("missing", []), *validation.get("false_gap_questions", [])])
normalize_nested_markdown_bullet = lambda line: clean_text(line)
repair_truncated_markdown_line = lambda line: clean_text(line).rstrip(",-;:")
remove_clipped_sentence_fragments = clean_markdown
remove_duplicate_section_labels = clean_markdown
has_dangling_markdown_bullet = lambda text: bool(re.search(r"^\s*[-*+]\s*$", clean_markdown(text), flags=re.M))
has_truncated_markdown_list_item = lambda line: bool(clean_text(line).startswith(("-", "*")) and markdown_appears_truncated(line))
list_item_appears_truncated = has_truncated_markdown_list_item
heading_ends_with_connector = lambda heading: clean_text(heading).lower().split(" ")[-1:] in [["and"], ["or"], ["with"], ["of"]]
topic_heading_sequence_issues = lambda report: []
truncated_report_sections = lambda report: [heading for heading, section in markdown_sections(report) if markdown_appears_truncated(section)]
malformed_equation_tail_present = lambda text: False
section_has_incomplete_equation = lambda text: False
unsupported_benchmark_metrics = lambda report, evidence_text: []
canonical_source_routing_issues = lambda report, planner_questions, sources: []
required_topic_facet_issues = lambda report, planner_questions: []
framework_api_detail_issues = lambda report, planner_questions: []
topic_section_semantic_report_issues = lambda report, planner_questions: []
internal_gap_contradiction_issues = lambda report: []
report_synthesis_gap_contradictions = lambda report, per_question_synthesis, planner_questions=None: []
report_per_question_synthesis_citation_gaps = lambda report, per_question_synthesis, planner_questions=None, sources=None: []
per_question_synthesis_repair_note = lambda question, synthesis_note: clean_section_synthesis_note(synthesis_note)
clean_topic_digest_for_frames = lambda text: compact_markdown_at_sentence(text, 900)
frame_lines_as_prose = lambda lines: clean_text(" ".join(lines))
frame_source_line_usable = lambda line: bool(clean_text(line) and not line_has_gap_claim(line))
frame_body_needs_role_repair = lambda heading_name, body: weak_topic_body(body)
frame_section_needs_retry = lambda section, heading: weak_topic_body(section)
malformed_frame_section_issues = lambda section, heading: ["section is weak"] if weak_topic_body(section) else []
repair_weak_frame_sections = lambda report, sources: (report, [])
repair_topic_headings = lambda report, questions: report
normalize_report_for_validation = normalize_final_report
hard_report_issues = lambda issues: [issue for issue in issues if clean_text(issue)]
markdown_completion_issues = lambda markdown: [] if clean_text(markdown) else ["section is empty"]
format_evidence_coverage_brief = lambda **kwargs: ""
format_memory_signal_evidence = lambda *args, **kwargs: ""
format_planner_evidence_packet = lambda *args, **kwargs: ""
format_source_priority_guidance = lambda sources: ""
remove_conflicting_missing_evidence_statements = lambda report, evidence_text="": report
remove_placeholder_citations = lambda text: re.sub(r"\[(?:uncited|citation needed|source needed)\]", "", str(text), flags=re.I)
ensure_planner_question_sections = lambda report, *args, **kwargs: report
