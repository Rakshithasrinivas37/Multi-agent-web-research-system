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
DEFAULT_EVIDENCE_CHARS = 5200
DEFAULT_SYNTHESIS_CHARS = 3400
DEFAULT_CHUNK_CHARS = 900

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
        synthesis = normalize_citation_markers(report_context.get("synthesis"))
        if not objective:
            raise ValueError("report_context.objective is required")
        if not synthesis:
            raise ValueError("report_context.synthesis is required")

        planner_questions = [clean_text(q) for q in report_context.get("planner_questions", []) if clean_text(q)]
        sources = dedupe_sources(sources_with_browser_results(report_context.get("sources", []), report_context.get("browser_results", [])))
        evidence = format_supporting_evidence(report_context)
        prompt = build_report_prompt(
            objective=objective,
            output_format=output_format,
            planner_questions=planner_questions,
            synthesis=synthesis,
            evidence=evidence,
            sources=sources,
            citation_policy=clean_text(report_context.get("citation_policy")),
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
        coverage = report_sub_question_coverage_check(report, planner_questions)
        schema_issues = report_schema_issues(report, planner_questions)
        report_issues = report_quality_issues(report, sources)
        review = report_self_critique(report_issues, coverage, schema_issues)

        return {
            "objective": objective,
            "output_format": clean_text(output_format) or "report",
            "report": report,
            "sources": sources,
            "model": model,
            "diagnostics": {
                "source_count": len(sources),
                "supporting_chunk_count": len(report_context.get("supporting_chunks", []) or []),
                "retrieved_chunk_count": len(report_context.get("retrieved_chunks", []) or []),
                "report_length": len(report),
                "report_generation_mode": "single",
                "report_issues": report_issues,
                "report_schema_issues": schema_issues,
                "report_missing_sub_questions": coverage["missing"],
                "report_coverage_check": coverage,
                "report_retry_queries": rewrite_missing_sub_question_queries(objective, coverage["missing"]),
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
) -> str:
    return f"""Research objective:
{objective}

Requested output format:
{clean_text(output_format) or "report"}

Citation policy:
{citation_policy or "Use only numbered source markers from the available sources."}

Required report schema (use these exact headings, in this order):
1. Executive Summary
2. Introduction and Context
3. One main section per planner sub-question topic, in order
4. Cross-cutting Analysis and Synthesis
5. Limitations and Open Questions
6. Conclusion
7. References

Planner sub-questions to cover:
{format_planner_questions(planner_questions)}

Suggested topic headings:
{format_report_section_outline(planner_questions)}

Available sources:
{format_sources(sources)}

Supporting evidence:
{compact_text(evidence, DEFAULT_EVIDENCE_CHARS)}

Synthesis notes:
{compact_text(synthesis, DEFAULT_SYNTHESIS_CHARS)}

Write the final Markdown report.

Grounding requirement (strict — read this first):
- Use ONLY the information in "Available sources," "Supporting evidence," and "Synthesis notes" above. Treat this as the complete and only knowledge you have access to.
- Do not use any fact, figure, date, name, definition, or background knowledge from your own training. Even facts you are confident are true must not be included unless they appear in the retrieved context above.
- If the retrieved context is silent on something a sub-question asks about, do not fill the gap from general knowledge. State the gap explicitly in that section and in Limitations/Open Questions instead.
- Do not include "outside evidence scope" explanations. A gap statement is enough when evidence is missing.
- If you find yourself writing a sentence with no source to cite for it, delete the sentence or move it to Limitations/Open Questions as a stated gap — do not soften it into an uncited claim.

Coverage requirement (mandatory):
- Every planner sub-question above must map to exactly one section under heading 3, using the suggested topic heading or a clearer equivalent.
- Each of those sections must explicitly answer its sub-question using only the retrieved context — not just mention the topic. If the evidence only partially answers a sub-question, answer what is supported and name the missing piece in that section AND in Limitations/Open Questions.
- Do not merge two sub-questions into one section unless they are genuinely the same question asked two ways — if you do this, say so explicitly.
- Do not add sections that don't map to a sub-question, except the fixed schema sections above.

Evidence and citation rules:
- Every factual claim must trace to a source in "Available sources." Cite inline using [n] immediately after the claim, not bundled at the end of a paragraph.
- Never state a number, date, name, or quote that does not appear in the evidence or synthesis notes.
- If two sources conflict, present both and cite both — do not silently pick one.
- Do not cite a source for a claim it doesn't actually support.

Writing rules:
- Explain each topic in plain prose before any equations, tables, or technical detail.
- Write for a reader who has not seen the sub-questions — sections should read as a coherent report, not as Q&A pairs.
- Keep the Cross-cutting Analysis section genuinely cross-cutting: identify tensions, agreements, or patterns across sections rather than repeating section content.

Before writing References, run this self-check silently and correct any failures before output — do not show this checklist in the final report:
- [ ] Every claim in the report can be traced to a specific line in the retrieved context — none came from outside knowledge
- [ ] Every planner sub-question has a matching section that directly answers it using only retrieved context
- [ ] Every claim has an inline citation to a real, relevant source
- [ ] No invented facts, figures, or attributions
- [ ] Every evidence gap is named specifically (not "some information was missing" — state exactly what is missing and for which sub-question)
- [ ] Executive Summary accurately reflects the sections below it, including any major limitations

End with ## References, listing only sources actually cited in the report, numbered to match inline markers."""

def generate_single_report(client: Any, model: str, prompt: str) -> tuple[str, str]:
    print(f"Generating single report with model {model}...")
    response = create_chat_completion_with_retries(
        client,
        model=model,
        temperature=0,
        max_tokens=DEFAULT_REPORT_MAX_TOKENS,
        messages=[
            {"role": "system", "content": "You write concise, well-structured, cited technical reports from provided evidence only."},
            {"role": "user", "content": prompt[:DEFAULT_REPORT_PROMPT_CHARS]},
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
    return "\n".join(f"## {i}. {planner_question_heading(q)}\nCoverage target: {q}" for i, q in enumerate(items, 1))


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


def format_supporting_evidence(report_context: dict[str, Any], max_chars: int = DEFAULT_EVIDENCE_CHARS) -> str:
    chunks = list(report_context.get("supporting_chunks") or []) + list(report_context.get("retrieved_chunks") or [])
    blocks: list[str] = []
    seen = set()
    used = 0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        source_index = chunk.get("source_index") if isinstance(chunk.get("source_index"), int) else chunk.get("index")
        content = clean_text(chunk.get("content"))[:DEFAULT_CHUNK_CHARS]
        if not content:
            continue
        key = clean_text(f"{source_index}:{chunk.get('url')}:{content[:120]}").lower()
        if key in seen:
            continue
        seen.add(key)
        marker = f"[{source_index}]" if isinstance(source_index, int) else "[uncited]"
        block = f"{marker} {clean_text(chunk.get('title')) or clean_text(chunk.get('url'))}\n{content}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


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


def sources_with_browser_results(sources: Sequence[Any], browser_results: Sequence[Any]) -> list[dict[str, Any]]:
    merged = [dict(source) for source in sources or [] if isinstance(source, dict)]
    existing = {normalize_url(source.get("url")) for source in merged}
    for result in browser_results or []:
        if not isinstance(result, dict):
            continue
        for source in result.get("sources", []) or []:
            if not isinstance(source, dict):
                continue
            url = normalize_url(source.get("url"))
            if url and url not in existing:
                existing.add(url)
                merged.append({"title": source.get("title"), "url": source.get("url")})
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
    text = remove_unavailable_citation_markers(clean_markdown(report), source_index_set(sources))
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


def report_sub_question_coverage_check(report: str, planner_questions: Sequence[str]) -> dict[str, Any]:
    questions = [clean_text(q) for q in planner_questions if clean_text(q)]
    missing = missing_sub_question_coverage(report, questions)
    missing_set = set(missing)
    return {
        "total": len(questions),
        "covered_count": len(questions) - len(missing),
        "missing_count": len(missing),
        "missing": missing,
        "items": [{"question": q, "heading": planner_question_heading(q), "status": "missing" if q in missing_set else "covered"} for q in questions],
    }


def missing_sub_question_coverage(report: str, planner_questions: Sequence[str]) -> list[str]:
    report_terms = detail_terms(report)
    missing = []
    for question in planner_questions:
        terms = [term for term in detail_terms(question) if term not in STOPWORDS]
        important = named_terms(question) or terms[:4]
        required = 1 if len(important) <= 2 else 2
        if sum(1 for term in important if term in report_terms) < required:
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


def report_quality_issues(report: str, sources: Sequence[dict[str, Any]] | None = None) -> list[str]:
    issues = []
    text = clean_markdown(report)
    if not text:
        return ["report is empty"]
    if not any(is_references_heading(line) for line in text.splitlines()):
        issues.append("report must include a References section")
    source_indexes = source_index_set(sources or [])
    invalid = unavailable_citation_markers(report, source_indexes)
    if invalid:
        issues.append(f"report uses unavailable citations: {format_citation_indexes(invalid)}")
    return issues


def rewrite_missing_sub_question_queries(objective: str, questions: Sequence[str]) -> list[str]:
    return [
        clean_text(f"{objective} {question} source-backed evidence details examples equations benchmarks limitations")[:700]
        for question in questions
        if clean_text(question)
    ]


def report_context_gap_items(report_context: dict[str, Any], research_plan: dict[str, Any]) -> list[str]:
    synthesis = clean_text(report_context.get("synthesis")) if isinstance(report_context, dict) else ""
    questions = [clean_text(q) for q in research_plan.get("sub_questions", []) if clean_text(q)] if isinstance(research_plan, dict) else []
    missing = missing_sub_question_coverage(synthesis, questions)
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
    required = min(4, max(2, len(expected_terms) // 2))
    return len(expected_terms & actual_terms) >= required


def detail_terms(text: Any) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9_+.-]{2,}", strip_markdown(text)) if token.lower() not in STOPWORDS}


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
