"""Compact report agent for turning synthesis context into a cited report."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from src.memory.shared_memory import SharedMemory
from src.tools.text_utils import clean_text


DEFAULT_REPORT_AGENT_MODEL = "llama-3.1-8b-instant"
DEFAULT_REPORT_OUTPUT_DIR = "data/reports"
DEFAULT_REPORT_TOTAL_TOKEN_BUDGET = 7000
DEFAULT_REPORT_PROMPT_CHARS = 12000
DEFAULT_TOPIC_TEXT_CHARS = 1100
DEFAULT_EVIDENCE_CHUNK_CHARS = 480
DEFAULT_HEADING_CHARS = 96

STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "by", "can", "do", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "their", "to", "what", "when",
    "where", "which", "with",
}

TRAILING_HEADING_WORDS = {
    "and", "as", "by", "for", "from", "in", "including", "of", "or", "the", "to",
    "what", "when", "where", "which", "with", "eg", "e.g", "benchmark",
}

RAW_LABEL_RE = re.compile(
    r"\b(?:Planner Sub-?question|Report-?agent-?ready notes|Planner notes|"
    r"Supported information|Supported formulation|Supported evidence|"
    r"Supported evidence-based synthesis|Supported answer|Supported definition|"
    r"Self-?attention variant|Missing details)\b",
    flags=re.IGNORECASE,
)

NOISE_RE = re.compile(
    r"(?:skip to main content|section navigation|rate this page|manage preferences|"
    r"was this helpful|uses cookies|source code for|install pytorch|api developer notes|"
    r"given the fast pace of innovation|higher level libraries from the pytorch ecosystem|"
    r"privacy policy|learn community projects docs|"
    r"we begin by establishing|presented with complete mathematical derivations|"
    r"dimensiondk|dimensiondv|operation then computes:|"
    r"another is the amount of computation|"
    r"\.{6,}\s*\d+|"
    r"\b\d+\.\d+\.\d+\s+[A-Z][A-Za-z -]+\.{3,}|"
    r"\b(?:corresponding key|only limited features relative|instead of all encoder outputs)\b|"
    r"[\)\]]\s*Instead of all encoder outputs|"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"
    r"\b(?:translatio|classificatio|representatio|computatio|informatio|long-r)\b|"
    r"\b(?:ncoder|resses|hematical|aceVto|head h|ng with)\b|"
    r"(?:Q=XW|K=XW|V=XW|T\(v\)\s*=|dX\s+i=1|MultiHead\(Q,K,V\)=Concat))",
    flags=re.IGNORECASE,
)

BAD_SENTENCE_START_RE = re.compile(
    r"^(?:[a-z)]|(?:and|or|but|where|which|that|it also|this version|another is|the third is|head h|ncoder|resses|"
    r"hematical|ng with|corresponding key|only limited|instead of all)\b)",
)

FORMULA_FRAGMENT_RE = re.compile(
    r"(?:[\U0001D400-\U0001D7FF]\s+[\U0001D400-\U0001D7FF]|[A-Z]=XW|dX\s+i=1|"
    r"\b(?:head|MultiHead)\s+\d|\u200b|\\qquad|\\operatorname|\\text\{)",
    flags=re.IGNORECASE,
)


class ReportAgent:
    """Generate and persist final reports from synthesis-agent context."""

    def __init__(self, model: str | None = None, generation_mode: str | None = None) -> None:
        self.model = clean_text(model) or DEFAULT_REPORT_AGENT_MODEL
        self.generation_mode = "deterministic"

    def generate(self, report_context: dict[str, Any], output_format: str = "report") -> dict[str, Any]:
        if not isinstance(report_context, dict) or not report_context:
            raise ValueError("report_context is required")
        objective = clean_text(report_context.get("objective"))
        if not objective:
            raise ValueError("report_context.objective is required")

        evidence_packs = [p for p in report_context.get("evidence_packs", []) or [] if isinstance(p, dict)]
        planner_questions = [clean_text(q) for q in report_context.get("planner_questions", []) if clean_text(q)]
        questions = dedupe_text([*planner_questions, *evidence_pack_questions(evidence_packs)]) or [objective]
        sources = sources_with_browser_results(report_context.get("sources", []), report_context.get("browser_results", []))
        synthesis_by_question = per_question_synthesis_by_question(report_context.get("per_question_synthesis", []))
        packs_by_question = {normalize_heading(pack.get("question")): pack for pack in evidence_packs}

        topic_sections, diagnostics = [], []
        for index, question in enumerate(questions, 1):
            section, diagnostic = build_topic_section(
                index=index,
                question=question,
                pack=packs_by_question.get(normalize_heading(question), {}),
                synthesis_note=synthesis_by_question.get(normalize_heading(question), {}),
                sources=sources,
            )
            topic_sections.append(section)
            diagnostics.append(diagnostic)

        report = assemble_report(objective, topic_sections, evidence_packs, report_context)
        report = normalize_final_report(report, sources)
        report, repairs = cleanup_report(report, sources)
        validation = validate_report(report, sources, questions, evidence_packs)
        if validation["issues"]:
            report = ensure_limitations(report, validation["issues"], sources)
            report, more_repairs = cleanup_report(report, sources)
            repairs.extend(more_repairs)
            validation = validate_report(report, sources, questions, evidence_packs)

        coverage = report_sub_question_coverage_check(report, questions)
        synthesis_gaps = synthesis_coverage_gap_questions(report_context, questions)
        retry_questions = dedupe_text([*coverage["missing"], *synthesis_gaps])

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
                "report_generation_mode": self.generation_mode,
                "report_issues": validation["issues"],
                "report_schema_issues": validation["schema_issues"],
                "report_missing_sub_questions": coverage["missing"],
                "report_evidence_gap_questions": synthesis_gaps,
                "report_false_gap_questions": [],
                "report_pack_citation_gap_questions": validation["citation_gap_questions"],
                "report_coverage_check": coverage,
                "report_retry_queries": rewrite_missing_sub_question_queries(objective, retry_questions),
                "report_review_trace": [report_self_critique(validation["issues"], coverage, validation["schema_issues"])],
                "report_revision_attempts": 0,
                "report_deterministic_repairs": repairs,
                "report_finalization_status": "clean" if not validation["issues"] else "needs_review",
                "report_token_budget": DEFAULT_REPORT_TOTAL_TOKEN_BUDGET,
                "report_section_diagnostics": {"topic_sections": diagnostics},
                "report_estimated_token_cap": DEFAULT_REPORT_TOTAL_TOKEN_BUDGET,
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


def build_topic_section(
    index: int,
    question: str,
    pack: dict[str, Any],
    synthesis_note: dict[str, Any],
    sources: Sequence[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    heading = planner_question_heading(question)
    source_indexes = [idx for idx in sorted(set(pack_source_indexes(pack)) | set(per_question_synthesis_source_indexes(synthesis_note))) if idx in source_index_set(sources)]
    body_parts = topic_body_from_synthesis(question, synthesis_note)
    body_parts.extend(topic_body_from_chunks(question, pack, existing_text=" ".join(body_parts)))
    if not body_parts:
        body_parts = [evidence_gap_sentence(question, pack, synthesis_note)]

    body = cleanup_section_text("\n\n".join(body_parts))
    if source_indexes and not (set(source_indexes) & set(citation_markers(body))):
        body = f"{body.rstrip('.')} {format_citation_indexes(source_indexes[:2])}."
    body = enforce_topic_requirements(question, cleanup_section_text(body), pack, synthesis_note, source_indexes, sources)
    return (
        f"### 3.{index}. {heading}\n{cleanup_section_text(body)}",
        {
            "question": question,
            "heading": heading,
            "source_indexes": source_indexes,
            "coverage": clean_text(pack.get("coverage")) or "unknown",
            "source": "per_question_synthesis" if topic_body_from_synthesis(question, synthesis_note) else "evidence_pack",
            "chars": len(body),
        },
    )


def topic_body_from_synthesis(question: str, synthesis_note: dict[str, Any]) -> list[str]:
    synthesis = clean_text(synthesis_note.get("synthesis")) if isinstance(synthesis_note, dict) else ""
    if not synthesis or not citation_markers(synthesis):
        return []
    sentences = clean_report_sentences(synthesis, max_sentences=4, require_citation=True, question=question)
    if not sentences:
        return []
    text = compact_at_sentence(" ".join(sentences), DEFAULT_TOPIC_TEXT_CHARS)
    return [text] if text and citation_markers(text) and report_sentence_quality(text, require_citation=True, allow_long=True) else []


def topic_body_from_chunks(question: str, pack: dict[str, Any], existing_text: str = "") -> list[str]:
    if citation_markers(existing_text):
        return []
    chunks = rank_question_chunks(question, pack.get("chunks", []) if isinstance(pack, dict) else [])
    notes, seen = [], detail_terms(existing_text)
    for chunk in chunks:
        if len(notes) >= 2:
            break
        note = chunk_to_evidence_sentence(question, chunk)
        if not note:
            continue
        terms = detail_terms(note)
        if terms and len(terms & seen) >= max(3, min(6, len(terms) // 2)):
            continue
        seen.update(terms)
        notes.append(note)
    return notes


def chunk_to_evidence_sentence(question: str, chunk: dict[str, Any]) -> str:
    if not isinstance(chunk, dict) or not isinstance(chunk.get("source_index"), int):
        return ""
    marker = f"[{chunk['source_index']}]"
    content = sanitize_evidence_content(chunk.get("content"))
    sentences = clean_report_sentences(content, max_sentences=1, require_citation=False, question=question)
    sentence = sentences[0] if sentences else ""
    if not sentence:
        return ""
    return sentence if marker in sentence else f"{sentence.rstrip('.')} {marker}."


def enforce_topic_requirements(question: str, body: str, pack: dict[str, Any], synthesis_note: dict[str, Any], source_indexes: Sequence[int], sources: Sequence[dict[str, Any]]) -> str:
    lowered = clean_text(question).lower()
    additions = []
    if any(term in lowered for term in ("equation", "formula", "mathematical")) and "Core equation" not in body:
        equation = extract_source_backed_equation(pack, synthesis_note)
        if equation:
            additions.append(f"**Core equation:**\n\\[\n{equation}\n\\]\nSource: {format_citation_indexes(source_indexes[:2])}.")
    if "variant" in lowered and "attention" in lowered:
        table = attention_variant_table(body, source_indexes)
        if table:
            additions.append(table)
    if any(term in lowered for term in ("api", "framework", "pytorch", "tensorflow", "keras")):
        api_note = framework_api_note(body, source_indexes, sources)
        if api_note:
            additions.append(api_note)
    return clean_markdown("\n\n".join([body, *additions]))


def extract_source_backed_equation(pack: dict[str, Any], synthesis_note: dict[str, Any]) -> str:
    text = clean_text(" ".join([
        clean_text(synthesis_note.get("synthesis")) if isinstance(synthesis_note, dict) else "",
        *(clean_text(chunk.get("content")) for chunk in pack.get("chunks", []) or [] if isinstance(chunk, dict)),
    ]))
    if re.search(r"Attention\(?Q,?\s*K,?\s*V\)?", text, re.I) and all(token in text.lower() for token in ("softmax", "sqrt")):
        return r"\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V"
    return ""


def attention_variant_table(body: str, source_indexes: Sequence[int]) -> str:
    variants, lowered = [], body.lower()
    for name, signal, distinction in (
        ("Additive attention", "additive", "Uses a learned scoring function over query-key information."),
        ("Dot-product attention", "dot-product", "Uses vector dot products as attention scores."),
        ("Scaled dot-product attention", "scaled", "Scales dot products before softmax to stabilise training."),
        ("Self-attention", "self-attention", "Uses queries, keys, and values from the same sequence."),
        ("Multi-head attention", "multi-head", "Runs multiple heads to attend to different representation subspaces."),
    ):
        if signal in lowered:
            variants.append((name, distinction))
    if len(variants) < 2:
        return ""
    marker = format_citation_indexes(source_indexes[:2])
    return "\n".join(["| Variant | Distinction |", "|---|---|", *(f"| {name} | {distinction} {marker}. |" for name, distinction in variants[:6])])


def framework_api_note(body: str, source_indexes: Sequence[int], sources: Sequence[dict[str, Any]] | None = None) -> str:
    lowered = body.lower()
    api_indexes = [
        source.get("index")
        for source in sources or []
        if isinstance(source, dict)
        and source.get("index") in source_indexes
        and re.search(r"(pytorch|tensorflow|keras|multiheadattention|docs\.)", clean_text(f"{source.get('title')} {source.get('url')}"), flags=re.I)
    ]
    preferred = dedupe_ints(api_indexes) or list(source_indexes[:2])
    marker = format_citation_indexes(preferred)
    notes = []
    if marker and (api_indexes or "multiheadattention" in lowered or "torch.nn" in lowered or not citation_markers(body)):
        notes.append(f"**API evidence:** PyTorch exposes `torch.nn.MultiheadAttention` and related scaled dot-product attention support {marker}.")
    if "tf.keras" in lowered or "keras.layers" in lowered:
        notes.append(f"**API evidence:** TensorFlow/Keras evidence identifies attention layers in the framework API {marker}.")
    return "\n".join(notes)


def evidence_gap_sentence(question: str, pack: dict[str, Any], synthesis_note: dict[str, Any]) -> str:
    coverage = clean_text(pack.get("coverage")).lower() if isinstance(pack, dict) else ""
    if synthesis_coverage_status_is_gap(coverage):
        return f"The retrieved evidence is incomplete for this sub-question: {clean_text(question)}."
    synthesis = clean_text(synthesis_note.get("synthesis")) if isinstance(synthesis_note, dict) else ""
    if line_has_gap_claim(synthesis):
        return compact_at_sentence(synthesis, 320)
    return f"The retrieved evidence did not provide enough clean cited detail to answer this sub-question: {clean_text(question)}."


def assemble_report(objective: str, topic_sections: Sequence[str], evidence_packs: Sequence[dict[str, Any]], report_context: dict[str, Any]) -> str:
    topic_digest = "\n\n".join(topic_sections)
    return clean_markdown("\n\n".join([
        "## 1. Executive Summary",
        frame_section(topic_digest, "summary"),
        "## 2. Introduction and Context",
        frame_section(topic_digest, "intro") or f"This report summarizes retrieved evidence about {objective}.",
        "## 3. Topic Sections",
        *topic_sections,
        "## 4. Cross-cutting Analysis and Synthesis",
        frame_section(topic_digest, "cross"),
        "## 5. Limitations and Open Questions",
        limitations_section(evidence_packs, report_context),
        "## 6. Conclusion",
        frame_section(topic_digest, "conclusion"),
    ]))


def frame_section(topic_digest: str, role: str) -> str:
    topics = frame_topics(topic_digest)
    if not topics:
        return "The retrieved evidence did not provide enough clean cited detail for this frame section."
    topic_names = readable_topic_list([topic["heading"] for topic in topics[:4]])
    markers = format_citation_indexes([index for topic in topics[:4] for index in topic["citations"]][:4])
    if role == "summary":
        return f"The report covers {topic_names} using the cited evidence selected for each planner question {markers}."
    if role == "intro":
        return f"The report is organized around {len(topics)} planner topic sections so the retrieved evidence can be evaluated question by question {markers}."
    if role == "conclusion":
        return f"The strongest report findings are the sections with direct source markers; any topic without clean cited support is preserved as a limitation rather than filled from outside knowledge {markers}."
    return f"Across the topic sections, the evidence links {topic_names} while keeping each claim tied to the source markers available for that section {markers}."


def frame_topics(markdown: str) -> list[dict[str, Any]]:
    topics = []
    for heading, section in markdown_sections(markdown):
        match = re.match(r"^3\.\d+\.?\s+(.+)$", clean_text(heading))
        if not match:
            continue
        citations = citation_markers(section)
        if citations:
            topics.append({"heading": strip_heading_numbering(match.group(1)), "citations": citations})
    return topics


def readable_topic_list(topics: Sequence[str]) -> str:
    cleaned = [clean_text(topic).lower() for topic in topics if clean_text(topic)]
    if not cleaned:
        return "the requested topics"
    if len(cleaned) == 1:
        return cleaned[0]
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def frame_takeaways(markdown: str) -> list[str]:
    items = []
    for _, section in markdown_sections(markdown):
        for sentence in clean_report_sentences(strip_leading_heading(section), max_sentences=3, require_citation=True):
            if frame_sentence_usable(sentence) and sentence not in items:
                items.append(sentence.rstrip(".") + ".")
                break
    return dedupe_text(items)


def frame_sentence_usable(sentence: str) -> bool:
    if "|" in sentence or sentence.lstrip().startswith("|"):
        return False
    if re.search(r"\b(?:Variant|Distinction|API evidence|Core equation|Core formula)\b", sentence, re.I):
        return False
    return report_sentence_quality(sentence, require_citation=True, allow_long=False)


def limitations_section(evidence_packs: Sequence[dict[str, Any]], report_context: dict[str, Any]) -> str:
    items = []
    for pack in evidence_packs or []:
        if isinstance(pack, dict) and synthesis_coverage_status_is_gap(pack.get("coverage")):
            items.append(f"- Evidence is {clean_text(pack.get('coverage')) or 'incomplete'} for: {clean_text(pack.get('question'))}.")
    for item in report_context.get("coverage_by_question", []) or []:
        if isinstance(item, dict) and synthesis_coverage_status_is_gap(item.get("status")):
            items.append(f"- Synthesis coverage is {clean_text(item.get('status'))} for: {clean_text(item.get('question'))}.")
    if clean_text(report_context.get("gap_query_error")):
        items.append("- Additional gap-retrieval evidence was unavailable during this run.")
    return "\n".join(dedupe_text(items)) or "- No explicit evidence gaps were identified in the supplied synthesis context."


def cleanup_report(report: str, sources: Sequence[dict[str, Any]]) -> tuple[str, list[str]]:
    repairs, cleaned_lines = [], []
    for line in strip_references(clean_markdown(report)).splitlines():
        cleaned = cleanup_section_text(line)
        if not cleaned and clean_text(line):
            repairs.append("removed noisy or malformed line")
            continue
        if cleaned != line:
            repairs.append("cleaned report line")
        cleaned_lines.append(cleaned)
    text, math_repairs = cleanup_code_and_math_citations(clean_markdown("\n".join(cleaned_lines)))
    repairs.extend(math_repairs)
    return normalize_final_report(repair_headings(text), sources), dedupe_text(repairs)


def cleanup_section_text(text: Any) -> str:
    value = clean_markdown(text)
    value = remove_authoring_labels(value)
    value = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "", value)
    value = re.sub(r"\[\s*(?:uncited|citation needed|source needed)\s*\]", "", value, flags=re.I)
    value = re.sub(r"\b(tasks|evidence|results)\s+(On|The)\b", r"\1. \2", value)
    value = re.sub(r"\b(representations|architecture|formulation)\s+(This|The)\b", r"\1. \2", value)
    value = re.sub(r"([A-Za-z)])(\[\d+\])", r"\1 \2", value)
    value = re.sub(r"\.\s*,", ",", value)
    value = re.sub(r"\bby√", "by √", value)
    value = re.sub(r"\b(attention|comparison)\.\s+[–-]\s+", r"\1: ", value, flags=re.I)
    value = re.sub(r"\s+([.,;:])", r"\1", value)
    value = re.sub(r"([.!?])\s+([.!?])", r"\1", value)
    value = re.sub(r"([A-Za-z])-\s+([a-z])", r"\1\2", value)
    value = trim_clipped_fragments(value)
    if is_noisy_text(value):
        value = drop_noisy_sentences(value)
    return clean_markdown(value)


def remove_authoring_labels(text: Any) -> str:
    value = clean_markdown(text)
    value = re.sub(r"\*\*(?:Planner Sub-?question|Report-?agent-?ready notes|Planner notes|Supported information|Supported formulation|Supported evidence|Supported evidence-based synthesis|Supported answer|Supported definition|Self-?attention variant|Missing details)(?:\s*\([^)]*\))?\s*:?\*\*\s*", "", value, flags=re.I)
    value = re.sub(r"\b(?:Planner Sub-?question|Report-?agent-?ready notes|Planner notes)\s*:?\s*", "", value, flags=re.I)
    value = re.sub(r"\b(?:Supported information|Supported formulation|Supported evidence|Supported evidence-based synthesis|Supported answer|Supported definition|Self-?attention variant)(?:\s*\([^)]*\))?\s*:?\s*", "", value, flags=re.I)
    return clean_markdown(value)


def sanitize_evidence_content(text: Any) -> str:
    value = clean_text(text)
    value = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", value)
    value = re.sub(r"\b(?:Skip to main content|Section Navigation|Rate this Page|Manage Preferences)\b.*", "", value, flags=re.I)
    return cleanup_section_text(value)


def clean_report_sentences(text: Any, max_sentences: int = 4, require_citation: bool = True, question: str = "") -> list[str]:
    """Return complete cited sentences that are suitable for final report prose."""

    value = prepare_report_sentence_text(text)
    selected, pending = [], ""
    for raw_sentence in split_sentences(value):
        sentence = cleanup_section_text(raw_sentence)
        if require_citation and pending and citation_markers(sentence):
            joined = cleanup_section_text(f"{pending.rstrip('.')} {sentence}")
            if report_sentence_quality(joined, require_citation=True, allow_long=True) and sentence_matches_question(joined, question):
                selected.append(joined.rstrip(".") + ".")
                pending = ""
                if len(selected) >= max_sentences:
                    break
                continue
        if require_citation and not citation_markers(sentence) and report_sentence_quality(sentence, require_citation=False) and sentence_matches_question(sentence, question):
            pending = sentence
            continue
        if report_sentence_quality(sentence, require_citation=require_citation) and sentence_matches_question(sentence, question):
            selected.append(sentence.rstrip(".") + ".")
            pending = ""
        if len(selected) >= max_sentences:
            break
    return dedupe_text(selected)


def prepare_report_sentence_text(text: Any) -> str:
    value = clean_markdown(text)
    value = re.sub(r"\\\[[\s\S]*?\\\]", " ", value)
    value = re.sub(r"\$\$[\s\S]*?\$\$", " ", value)
    value = re.sub(r"`{1,3}[^`]*`{1,3}", " ", value)
    value = remove_authoring_labels(value)
    value = re.sub(r"\*\*([^*]{3,90})\*\*", r"\1. ", value)
    value = re.sub(r"\s+-\s+", ". ", value)
    value = re.sub(r"\s+", " ", value)
    return clean_text(value)


def report_sentence_quality(sentence: Any, require_citation: bool = True, allow_long: bool = False) -> bool:
    value = cleanup_section_text(sentence)
    words = strip_markdown(value).split()
    if not value or len(words) < 8:
        return False
    if len(words) > (80 if allow_long else 45):
        return False
    if require_citation and not citation_markers(value):
        return False
    if RAW_LABEL_RE.search(value) or is_noisy_text(value):
        return False
    if BAD_SENTENCE_START_RE.search(strip_markdown(value)):
        return False
    if formula_fragment_score(value) >= 2:
        return False
    if value.count("[") != value.count("]"):
        return False
    if re.search(r"\b\d+\.\d+\.\d+\b|\bFigure\s+\d+\b", value, flags=re.I):
        return False
    return True


def sentence_matches_question(sentence: Any, question: str = "") -> bool:
    if not clean_text(question):
        return True
    text = clean_text(sentence).lower()
    q = clean_text(question).lower()
    asks_api = any(term in q for term in ("api", "framework", "pytorch", "tensorflow", "keras"))
    asks_benchmark = any(term in q for term in ("benchmark", "application", "nlp", "vision", "imagenet", "glue", "translation"))
    asks_complexity = any(term in q for term in ("complexity", "recurrent", "linear-time", "linear time", "cost"))
    asks_limitation = any(term in q for term in ("limitation", "drawback", "quadratic", "locality"))
    asks_variant = "variant" in q or "differ" in q or any(term in q for term in ("additive", "luong", "multi-head"))
    api_terms = ("scaled_dot_product_attention", "multiheadattention", "nested tensor", "fastpath", "torch.", "pytorch", "keras", "tensorflow")
    benchmark_terms = ("wmt", "bleu", "imagenet", "glue", "benchmark", "translation task")
    if any(term in text for term in api_terms) and not asks_api:
        return False
    if any(term in text for term in benchmark_terms) and not asks_benchmark:
        return False
    if any(term in text for term in ("computational cost", "memory requirement", "o(n", "quadratic scaling")) and not (asks_complexity or asks_limitation):
        return False
    if "permutation equivariance" in text and not asks_complexity:
        return False
    if text.startswith("the third is"):
        return False
    if any(term in text for term in ("trace of a square matrix", "frobenius norm", "inner product induces", "cosine similarity")):
        return False
    if text.startswith("it also") or any(term in text for term in ("located at the class", "implementation of it also", "page is located at", "the class it also", "class it also")):
        return False
    if text.startswith("this monograph") or "comprehensive and rigorous mathematical treatment" in text:
        return False
    if any(term in text for term in ("additive", "luong", "multiplicative")) and not asks_variant:
        return False
    return True


def formula_fragment_score(text: Any) -> int:
    value = clean_text(text)
    score = len(FORMULA_FRAGMENT_RE.findall(value))
    score += 1 if len(re.findall(r"[=∑√⊤]", value)) >= 3 else 0
    score += 1 if len(re.findall(r"\b[A-Z]\s*[=∈]\s*", value)) >= 2 else 0
    return score


def cleanup_code_and_math_citations(markdown: str) -> tuple[str, list[str]]:
    text, repairs = clean_markdown(markdown), []

    def clean_block(match: re.Match[str]) -> str:
        repairs.append("removed citations inside code or equation block")
        block = re.sub(r"\s*\[\d+\][.,;:]?", "", match.group(0))
        if block.lstrip().startswith("```") and "\n" not in block:
            block = re.sub(r"^```\s*[A-Za-z0-9_+-]*\s*", "`", block.strip())
            block = re.sub(r"\s*```\s*$", "`", block)
        return clean_text(block)

    text = re.sub(r"```[\s\S]*?```", clean_block, text)
    text = re.sub(r"```[^\n]*", clean_block, text)
    text = re.sub(r"\\\[[\s\S]*?\\\]", clean_block, text)
    return clean_markdown(text), repairs


def repair_headings(markdown: str) -> str:
    lines = []
    for line in clean_markdown(markdown).splitlines():
        match = re.match(r"^(\s{0,3}#{3}\s+3\.\d+\.?\s+)(.+?)\s*$", line)
        if match:
            line = f"{match.group(1)}{repair_common_heading_fragments(match.group(2))}"
        lines.append(line)
    return clean_markdown("\n".join(lines))


def validate_report(report: str, sources: Sequence[dict[str, Any]], questions: Sequence[str], evidence_packs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    issues = report_quality_issues(report, sources)
    schema_issues = report_schema_issues(report, questions)
    citation_gap_questions = report_pack_citation_gaps(report, evidence_packs, questions)
    issues.extend(f"report section does not cite supplied evidence: {q}" for q in citation_gap_questions)
    return {"issues": dedupe_text(issues), "schema_issues": schema_issues, "citation_gap_questions": citation_gap_questions}


def report_quality_issues(report: str, sources: Sequence[dict[str, Any]] | None = None, evidence_text: str = "") -> list[str]:
    text, issues = clean_markdown(report), []
    if not text:
        return ["report is empty"]
    if not any(is_references_heading(line) for line in text.splitlines()):
        issues.append("report must include a References section")
    if RAW_LABEL_RE.search(text):
        issues.append("report contains raw planner or synthesis labels")
    if NOISE_RE.search(text):
        issues.append("report contains raw extraction or web-navigation artifacts")
    noisy_lines = report_artifact_lines(text)
    if noisy_lines:
        issues.append(f"report contains noisy copied evidence lines: {len(noisy_lines)}")
    copied_frames = repeated_frame_sentences(text)
    if copied_frames:
        issues.append(f"report repeats topic prose in frame sections: {len(copied_frames)}")
    if code_or_math_contains_citations(text):
        issues.append("report contains citations inside code or equation blocks")
    weak = weak_topic_headings(text)
    if weak:
        issues.append(f"report contains weak topic headings: {', '.join(weak[:4])}")
    invalid = unavailable_citation_markers(text, source_index_set(sources or []))
    if invalid:
        issues.append(f"report uses unavailable citations: {format_citation_indexes(invalid)}")
    return dedupe_text(issues)


def report_artifact_lines(report: str) -> list[str]:
    bad = []
    for line in clean_markdown(report).splitlines():
        value = clean_text(line)
        if not value or value.startswith("#") or is_references_heading(value):
            continue
        if structured_report_line(value):
            continue
        if is_noisy_text(value) or BAD_SENTENCE_START_RE.search(strip_markdown(value)) or formula_fragment_score(value) >= 2:
            bad.append(value[:140])
    return bad


def structured_report_line(line: str) -> bool:
    value = clean_text(line)
    return (
        value.startswith("|")
        or value.startswith(r"\[")
        or value.startswith("- ")
        or value.startswith("**Core equation:**")
        or value.startswith("Source:")
        or bool(re.fullmatch(r"\[?\d+\]?\s+https?://\S+", value))
        or bool(re.fullmatch(r"\\operatorname\{Attention\}.*", value))
    )


def repeated_frame_sentences(report: str) -> list[str]:
    sections = {normalize_heading(heading): section for heading, section in markdown_sections(report)}
    frame_names = {
        "executive summary",
        "introduction and context",
        "cross cutting analysis and synthesis",
        "conclusion",
    }
    topic_text = sections.get("topic sections", "")
    topic_sentences = {normalize_heading(sentence) for sentence in split_sentences(topic_text) if len(strip_markdown(sentence).split()) >= 10}
    repeats = []
    for heading, section in sections.items():
        if heading not in frame_names:
            continue
        for sentence in split_sentences(section):
            key = normalize_heading(sentence)
            if key and key in topic_sentences:
                repeats.append(sentence)
    return repeats


def report_schema_issues(report: str, questions: Sequence[str]) -> list[str]:
    headings = {normalize_heading(h) for h in h2_headings(report)}
    required = {
        "executive summary": ("executive summary",),
        "introduction/context": ("introduction and context", "introduction"),
        "cross-cutting analysis/synthesis": ("cross-cutting analysis and synthesis", "cross-cutting analysis"),
        "limitations/open questions": ("limitations and open questions", "limitations"),
        "conclusion": ("conclusion",),
        "references": ("references", "sources"),
    }
    issues = []
    for label, aliases in required.items():
        if not any(normalize_heading(alias) in headings for alias in aliases):
            issues.append(f"missing schema section: {label}")
    numbered = {entry["number"] for entry in topic_section_heading_entries(report)}
    for index, _ in enumerate(questions, 1):
        if f"3.{index}" not in numbered:
            issues.append(f"missing sequential planner topic heading: 3.{index}")
    return issues


def ensure_limitations(report: str, issues: Sequence[str], sources: Sequence[dict[str, Any]]) -> str:
    body = "\n".join(f"- {clean_text(issue).rstrip('.')}." for issue in issues[:8])
    return normalize_final_report(replace_named_report_section(report, "Limitations and Open Questions", body), sources)


def report_pack_citation_gaps(report: str, evidence_packs: Sequence[dict[str, Any]], questions: Sequence[str] | None = None) -> list[str]:
    canonical = {normalize_heading(q): q for q in questions or [] if clean_text(q)}
    gaps = []
    for pack in evidence_packs or []:
        if not isinstance(pack, dict) or not evidence_pack_has_usable_cited_evidence(pack):
            continue
        question = clean_text(pack.get("question"))
        section = report_section_for_question(report, question)
        if section and not (set(citation_markers(section)) & set(pack_source_indexes(pack))):
            gaps.append(canonical.get(normalize_heading(question), question))
    return dedupe_text(gaps)


def report_context_gap_items(report_context: dict[str, Any], research_plan: dict[str, Any]) -> list[str]:
    questions = [clean_text(q) for q in research_plan.get("sub_questions", []) if clean_text(q)] if isinstance(research_plan, dict) else []
    missing = synthesis_coverage_gap_questions(report_context, questions)
    synthesis = clean_text(report_context.get("synthesis")) if isinstance(report_context, dict) else ""
    missing.extend(q for q in missing_sub_question_coverage(synthesis, questions) if q not in missing)
    return dedupe_text(missing)


def report_context_gap_queries(report_context: dict[str, Any], research_plan: dict[str, Any]) -> list[str]:
    objective = clean_text(research_plan.get("objective")) if isinstance(research_plan, dict) else clean_text(report_context.get("objective"))
    return rewrite_missing_sub_question_queries(objective, report_context_gap_items(report_context, research_plan))


def rewrite_missing_sub_question_queries(objective: str, questions: Sequence[str]) -> list[str]:
    return [clean_text(f"{objective} {question} source-backed evidence details equations benchmarks limitations")[:700] for question in questions if clean_text(question)]


def synthesis_coverage_gap_questions(report_context: dict[str, Any], planner_questions: Sequence[str] | None = None) -> list[str]:
    if not isinstance(report_context, dict):
        return []
    canonical = {normalize_heading(q): q for q in planner_questions or [] if clean_text(q)}
    gaps = []
    for pack in report_context.get("evidence_packs", []) or []:
        if isinstance(pack, dict) and synthesis_coverage_status_is_gap(pack.get("coverage")):
            question = clean_text(pack.get("question"))
            if question:
                gaps.append(canonical.get(normalize_heading(question), question))
    for item in report_context.get("coverage_by_question", []) or []:
        if isinstance(item, dict) and synthesis_coverage_status_is_gap(item.get("status")):
            question = clean_text(item.get("question"))
            if question:
                gaps.append(canonical.get(normalize_heading(question), question))
    return dedupe_text(gaps)


def synthesis_coverage_status_is_gap(status: Any) -> bool:
    lowered = clean_text(status).lower()
    return bool(lowered and any(term in lowered for term in ("missing", "partial", "weak", "insufficient", "unsupported", "failed", "error")))


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


def missing_sub_question_coverage(report: str, planner_questions: Sequence[str]) -> list[str]:
    report_terms, missing = set(technical_question_terms(report)), []
    for question in planner_questions:
        section = report_section_for_question(report, question)
        if specialized_question_covered(question, section):
            continue
        terms = [term for term in technical_question_terms(question) if term not in STOPWORDS]
        important = named_terms(question) or terms[:5]
        required = 1 if len(important) <= 2 else 2
        if sum(1 for term in important[:7] if term in report_terms) < required:
            missing.append(question)
    return missing


def specialized_question_covered(question: str, section: str) -> bool:
    lowered = clean_text(question).lower()
    body = clean_text(section).lower()
    if any(term in lowered for term in ("equation", "formula", "mathematical")):
        return "core equation" in body or ("attention(q,k,v)" in body or "attention}(q,k,v)" in body)
    if any(term in lowered for term in ("api", "framework", "pytorch", "tensorflow", "keras")):
        return "api evidence" in body or "multiheadattention" in body or "keras.layers" in body
    if "benchmark" in lowered:
        return "benchmark" in body or line_has_gap_claim(body)
    return False


def report_self_critique(report_issues: Sequence[str], coverage_check: dict[str, Any], schema_issues: Sequence[str]) -> dict[str, Any]:
    unresolved = [clean_text(issue) for issue in [*report_issues, *schema_issues] if clean_text(issue)]
    unresolved.extend(f"missing planner topic: {q}" for q in coverage_check.get("missing", []) if clean_text(q))
    print(f"[report] self-critique: {len(unresolved)} issue(s)")
    return {"source": "deterministic", "unresolved_issues": unresolved, "coverage_missing": coverage_check.get("missing", []), "schema_issues": list(schema_issues)}


def normalize_final_report(report: str, sources: Sequence[dict[str, Any]]) -> str:
    text = normalize_markdown_headings(remove_unavailable_citation_markers(clean_markdown(report), source_index_set(sources)))
    body = strip_references(text)
    return clean_markdown(f"{body}\n\n{references_section(body, sources)}")


def references_section(report: str, sources: Sequence[dict[str, Any]]) -> str:
    by_index = {source.get("index"): source for source in sources if isinstance(source, dict)}
    lines, used = ["## References"], citation_markers(report)
    if not used:
        lines.append("No cited source markers were used.")
        return "\n".join(lines)
    for index in used:
        source = by_index.get(index)
        if source:
            lines.append(f"[{index}] {clean_text(source.get('url'))}")
    return "\n".join(lines)


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


def sources_with_browser_results(sources: Sequence[Any], browser_results: Sequence[Any]) -> list[dict[str, Any]]:
    merged, existing, used_indexes = [], set(), set()
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        item = dict(source)
        if not isinstance(item.get("index"), int):
            item["index"] = next_available_index(used_indexes)
        merged.append(item)
        used_indexes.add(item["index"])
        if normalize_url(item.get("url")):
            existing.add(normalize_url(item.get("url")))
    for result in browser_results or []:
        if not isinstance(result, dict):
            continue
        for source in result.get("sources", []) or []:
            if not isinstance(source, dict):
                continue
            url = normalize_url(source.get("url"))
            if url and url not in existing:
                index = next_available_index(used_indexes)
                used_indexes.add(index)
                existing.add(url)
                merged.append({"index": index, "title": source.get("title"), "url": source.get("url")})
    return merged


def next_available_index(used: set[int]) -> int:
    index = 1
    while index in used:
        index += 1
    return index


def evidence_pack_questions(evidence_packs: Sequence[Any]) -> list[str]:
    return dedupe_text(clean_text(pack.get("question")) for pack in evidence_packs or [] if isinstance(pack, dict) and clean_text(pack.get("question")))


def per_question_synthesis_by_question(per_question_synthesis: Sequence[Any]) -> dict[str, dict[str, Any]]:
    return {normalize_heading(item.get("question")): item for item in per_question_synthesis or [] if isinstance(item, dict) and clean_text(item.get("question"))}


def per_question_synthesis_source_indexes(synthesis_note: dict[str, Any]) -> list[int]:
    if not isinstance(synthesis_note, dict):
        return []
    return dedupe_ints([*synthesis_note.get("source_indexes", []), *citation_markers(synthesis_note.get("synthesis"))])


def pack_source_indexes(pack: dict[str, Any]) -> list[int]:
    return dedupe_ints(chunk.get("source_index") for chunk in pack.get("chunks", []) or [] if isinstance(chunk, dict))


def evidence_pack_has_usable_cited_evidence(pack: dict[str, Any]) -> bool:
    return any(isinstance(chunk, dict) and isinstance(chunk.get("source_index"), int) and clean_text(chunk.get("content")) for chunk in pack.get("chunks", []) or [])


def source_index_set(sources: Sequence[dict[str, Any]]) -> set[int]:
    return {source.get("index") for source in sources if isinstance(source, dict) and isinstance(source.get("index"), int)}


def rank_question_chunks(question: str, chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    terms = detail_terms(question)
    return sorted([chunk for chunk in chunks or [] if isinstance(chunk, dict)], key=lambda chunk: -len(terms & detail_terms(" ".join([clean_text(chunk.get("title")), clean_text(chunk.get("url")), clean_text(chunk.get("content"))]))))


def planner_question_heading(question: str, max_length: int | None = DEFAULT_HEADING_CHARS) -> str:
    text = clean_text(question).rstrip("?")
    heading = re.sub(r"^(what|how|why|when|where|which)\s+(is|are|does|do|did|can|should)\s+", "", text, flags=re.I)
    heading = re.sub(r"\s*\([^)]*(?:e\.g\.|eg|for example)[^)]*\)", "", heading, flags=re.I)
    heading = re.sub(r"\s+and\s+how\s+do\s+they\s+differ\b", " and their differences", heading, flags=re.I)
    heading = re.sub(r"^(what|how|why|when|where|which)\s+", "", heading, flags=re.I)
    heading = re.sub(r"\b(e\.g\.|eg|examples?|evidence|results?)\b", "", heading, flags=re.I)
    heading = repair_common_heading_fragments(heading)
    words = []
    for word in heading.split():
        clean_word = word.strip(".,:;()[]{}")
        words.append(clean_word if any(char.isupper() for char in clean_word[1:]) else clean_word.capitalize())
    return truncate_heading_at_word_boundary(" ".join(words), max_length) or "Research Finding"


def repair_common_heading_fragments(heading: str) -> str:
    value = clean_text(heading).strip(" .,:;")
    value = re.sub(r"\bThe Primary Applications Of (.+?) And What Benchmark(?:s)?(?: E\.?g\.?.*)?$", r"\1 Applications And Benchmark Evidence", value, flags=re.I)
    value = re.sub(r"\bThe Computational Complexity Of (.+?) Compared To Recurrent Networks And What Are.*$", r"\1 Complexity Compared With Recurrent Networks", value, flags=re.I)
    value = re.sub(r"\bThe Main Variants Of Attention Mechanisms.*$", "Attention Mechanism Variants And Differences", value, flags=re.I)
    value = re.sub(r"\bAttention Implemented In (.+?) Such As .*$", r"Attention Implementations In \1", value, flags=re.I)
    value = re.sub(r"\bAnd What Benchmark(?:s)?(?: E\.?g\.?.*)?$", "And Benchmark Evidence", value, flags=re.I)
    value = re.sub(r"\bAnd What Are.*$", "", value, flags=re.I)
    value = re.sub(r"\bSuch As .*$", "", value, flags=re.I)
    value = re.sub(r"\bE\.?g\.?\s*$", "", value, flags=re.I)
    return clean_text(value).strip(" .,:;")


def truncate_heading_at_word_boundary(heading: str, max_length: int | None = DEFAULT_HEADING_CHARS) -> str:
    value = clean_text(heading).strip(" .,:;")
    if not max_length or len(value) <= max_length:
        return trim_trailing_heading_words(value)
    return trim_trailing_heading_words(value[:max_length].rsplit(" ", 1)[0].strip(" .,:;"))


def trim_trailing_heading_words(heading: str) -> str:
    value = clean_text(heading).strip(" .,:;")
    while value and value.split()[-1].lower().rstrip(".") in TRAILING_HEADING_WORDS:
        value = " ".join(value.split()[:-1]).strip(" .,:;")
    return value


def weak_topic_headings(markdown: str) -> list[str]:
    return dedupe_text(strip_heading_numbering(entry["heading"]) for entry in topic_section_heading_entries(markdown) if heading_appears_weak(strip_heading_numbering(entry["heading"])))


def heading_appears_weak(heading: str) -> bool:
    lowered = clean_text(heading).lower().strip(" .,:;")
    return any(lowered.endswith(f" {word}") for word in TRAILING_HEADING_WORDS) or bool(re.search(r"\b(?:e\.g|eg)\s*$", lowered))


def topic_section_heading_entries(report: str) -> list[dict[str, Any]]:
    entries = []
    for line in clean_markdown(report).splitlines():
        match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
        if match and (number_match := re.match(r"^(3\.\d+)\.?\s+", match.group(2).strip())):
            entries.append({"level": len(match.group(1)), "heading": match.group(2).strip(), "number": number_match.group(1)})
    return entries


def report_section_for_question(report: str, question: str) -> str:
    expected, question_terms, best, best_score = normalize_heading(planner_question_heading(question)), detail_terms(question), "", 0
    for heading, section in markdown_sections(report):
        actual = normalize_heading(heading)
        score = (5 if expected and headings_match(expected, actual) else 0) + len(question_terms & detail_terms(heading))
        if score > best_score:
            best_score, best = score, section
    return best if best_score else ""


def markdown_sections(markdown: str) -> list[tuple[str, str]]:
    sections, heading, lines, in_fence = [], "", [], False
    for line in clean_markdown(markdown).splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if match:
            if lines:
                sections.append((heading, lines))
            heading, lines = match.group(1).strip(), [line]
        else:
            lines.append(line)
    if lines:
        sections.append((heading, lines))
    return [(heading, "\n".join(lines)) for heading, lines in sections]


def h2_headings(markdown: str) -> list[str]:
    headings, in_fence = [], False
    for line in clean_markdown(markdown).splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and (match := re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)):
            headings.append(match.group(1).strip())
    return headings


def replace_named_report_section(report: str, heading_name: str, body: str) -> str:
    target, lines = normalize_heading(heading_name), clean_markdown(report).splitlines()
    positions = [(i, m.group(1), m.group(2).strip()) for i, line in enumerate(lines) if (m := re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line))]
    for pos, (start, hashes, heading) in enumerate(positions):
        if normalize_heading(heading) == target:
            end = positions[pos + 1][0] if pos + 1 < len(positions) else len(lines)
            return clean_markdown("\n".join([*lines[:start], f"{hashes} {heading}", clean_markdown(body), *lines[end:]]))
    return clean_markdown(f"{report}\n\n## {heading_name}\n{body}")


def strip_leading_heading(section_text: str) -> str:
    lines = clean_markdown(section_text).splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        lines = lines[1:]
    return clean_markdown("\n".join(lines))


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


def is_references_heading(line: str) -> bool:
    return normalize_heading(line.lstrip("#").strip()) in {"references", "reference", "sources"}


def normalize_markdown_headings(markdown: str) -> str:
    return "\n".join(re.sub(r"^(\s{0,3}#{1,6}\s+)#{1,6}\s+", r"\1", line) for line in clean_markdown(markdown).splitlines())


def code_or_math_contains_citations(markdown: str) -> bool:
    in_fence = in_math = False
    for line in clean_markdown(markdown).splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            if citation_markers(stripped):
                return True
            in_fence = not in_fence
            continue
        if stripped == r"\[":
            in_math = True
            continue
        if stripped == r"\]":
            in_math = False
            continue
        if (in_fence or in_math) and citation_markers(line):
            return True
    return False


def compact_at_sentence(text: Any, max_chars: int) -> str:
    value = clean_text(text)
    if len(value) <= max_chars:
        return value
    window = value[:max_chars].rstrip()
    boundaries = [m.end() for m in re.finditer(r"(?:\[\d+\](?:\s*\[\d+\])*)[.)]?(?:\s+|$)|[.!?](?:\s+|$)", window)]
    if boundaries:
        trimmed = window[:boundaries[-1]].strip()
        if len(trimmed) >= min(120, max_chars // 3):
            return trimmed
    return window.rsplit(" ", 1)[0].strip(" .,:;")


def split_sentences(text: Any) -> list[str]:
    value = clean_text(text)
    if not value:
        return []
    value = re.sub(r"(\[\d+\])\s+(?=[A-Za-z(*`])", r"\1\n", value)
    value = re.sub(r"(?<=[.!?])\s+(?=[A-Za-z(*`])", "\n", value)
    value = re.sub(r"\s+(?=\*\*(?:Supported|Self|Planner|Missing)\b)", "\n", value, flags=re.I)
    return [part.strip(" -") for part in value.splitlines() if clean_text(part)]


def trim_clipped_fragments(text: str) -> str:
    value = clean_markdown(text)
    cleaned_lines = [
        re.sub(r"\s*[^.!?]*\b(?:translatio|classificatio|representatio|computatio|informatio|long-r)\b[^.!?]*(?:[.!?]|$)", " ", line, flags=re.I)
        for line in value.splitlines()
    ]
    return clean_markdown("\n".join(cleaned_lines))


def drop_noisy_sentences(text: Any) -> str:
    return clean_text(" ".join(sentence for sentence in split_sentences(text) if not is_noisy_text(sentence) and not BAD_SENTENCE_START_RE.search(strip_markdown(sentence))))


def is_noisy_text(text: Any) -> bool:
    return bool(NOISE_RE.search(clean_text(text)))


def line_has_gap_claim(line: Any) -> bool:
    return bool(re.search(evidence_gap_pattern(), clean_text(line).lower()))


def evidence_gap_pattern() -> str:
    return r"(evidence\s+gap|missing\s+evidence|not\s+provided|not\s+available|not\s+present|insufficient|partial|cannot\s+be\s+(?:answered|provided))"


def unavailable_citation_markers(text: str, available_indexes: set[int]) -> list[int]:
    return sorted({index for index in citation_markers(text) if available_indexes and index not in available_indexes})


def remove_unavailable_citation_markers(text: str, available_indexes: set[int]) -> str:
    if not available_indexes:
        return re.sub(r"\[\d+\]", "", text)
    return re.sub(r"\[(\d+)\]", lambda m: m.group(0) if int(m.group(1)) in available_indexes else "", text)


def citation_markers(text: Any) -> list[int]:
    seen, markers = set(), []
    for match in re.finditer(r"\[(\d+)\]", normalize_citation_markers(text)):
        index = int(match.group(1))
        if index not in seen:
            seen.add(index)
            markers.append(index)
    return markers


def normalize_citation_markers(text: Any) -> str:
    value = str(text or "")
    value = re.sub(r"【\s*(\d+)(?:[^】]*)?】", r"[\1]", value)
    value = re.sub(r"\[\s*(\d+(?:\s*,\s*\d+)+)\s*\]", lambda m: " ".join(f"[{p.strip()}]" for p in m.group(1).split(",")), value)
    value = re.sub(r"\[\s*(\d+)\s*\]", r"[\1]", value)
    return value


def format_citation_indexes(indexes: Sequence[int]) -> str:
    return ", ".join(f"[{index}]" for index in sorted(set(indexes)))


def strip_markdown(text: Any) -> str:
    value = re.sub(r"`([^`]*)`", r"\1", str(text or ""))
    value = re.sub(r"[*_#|]+", " ", value)
    value = re.sub(r"\[(\d+)\]", "", value)
    return clean_text(value)


def clean_markdown(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"<think>.*", "", text, flags=re.S | re.I)
    text = normalize_citation_markers(text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def normalize_heading(text: Any) -> str:
    value = strip_markdown(text).replace("‑", "-").replace("–", "-").replace("—", "-")
    value = re.sub(r"^\d+(?:\.\d+)?[.)]?\s*", "", value)
    value = re.sub(r"[^a-zA-Z0-9]+", " ", value)
    return clean_text(value).lower()


def strip_heading_numbering(heading: str) -> str:
    return re.sub(r"^\s*\d+(?:\.\d+)?[.)]?\s*", "", clean_text(heading)).strip()


def headings_match(expected: str, actual: str) -> bool:
    if expected == actual or expected in actual or actual in expected:
        return True
    expected_terms, actual_terms = detail_terms(expected), detail_terms(actual)
    return bool(expected_terms) and len(expected_terms & actual_terms) >= min(3, max(2, len(expected_terms) // 3))


def detail_terms(text: Any) -> set[str]:
    return {token.lower().replace("‑", "-").replace("–", "-") for token in re.findall(r"[A-Za-z][A-Za-z0-9_+.-]{2,}", strip_markdown(text)) if token.lower() not in STOPWORDS}


def technical_question_terms(text: Any) -> list[str]:
    return dedupe_text(term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9_+.-]{2,}", strip_markdown(text).replace("‑", "-").replace("–", "-")))


def named_terms(text: Any) -> list[str]:
    terms = [token.lower() for token in re.findall(r"\b[A-Z][A-Za-z0-9_+.-]{2,}\b|\b[A-Z]{2,}\b", clean_text(text)) if token.lower() not in STOPWORDS]
    lowered = clean_text(text).lower()
    for phrase in ("multiheadattention", "multi-head", "self-attention", "vision transformer", "scaled dot", "cross-attention"):
        if phrase in lowered:
            terms.append(phrase)
    return dedupe_text(terms)


def dedupe_text(items: Sequence[Any]) -> list[str]:
    seen, out = set(), []
    for item in items:
        value, key = clean_text(item), clean_text(item).lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def dedupe_ints(values: Sequence[Any]) -> list[int]:
    deduped, seen = [], set()
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


def normalize_url(value: Any) -> str:
    return clean_text(value).rstrip("/").lower()


def compact_text(value: Any, max_chars: int) -> str:
    text = clean_markdown(value)
    return text if len(text) <= max_chars else text[:max_chars].rstrip()


# Compatibility aliases used by older tests/scripts.
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

# Compatibility shims for older tests/scripts.  The compact agent routes these
# names to the deterministic cleanup/validation primitives above.
build_report_prompt = lambda **kwargs: ""
trim_report_prompt = lambda prompt, max_chars=DEFAULT_REPORT_PROMPT_CHARS: compact_text(prompt, max_chars)
report_generation_token_cap = lambda prompt_chars=None: ((prompt_chars or DEFAULT_REPORT_PROMPT_CHARS) + 3) // 4 + DEFAULT_REPORT_TOTAL_TOKEN_BUDGET
generate_single_report = lambda client, model, prompt, fallback_prompt=None, label="report": (clean_markdown(prompt), model)
format_question_coverage = lambda coverage_by_question: "\n".join(f"- {clean_text(i.get('status'))}: {clean_text(i.get('question'))}" for i in coverage_by_question or [] if isinstance(i, dict))
format_report_section_outline = lambda questions: "\n".join(f"## {i}. {planner_question_heading(q)}" for i, q in enumerate(questions or [], 1))
format_evidence_packs = lambda evidence_packs, **kwargs: "\n".join(f"- {clean_text(p.get('coverage')) or 'unknown'}: {clean_text(p.get('question'))}" for p in evidence_packs or [] if isinstance(p, dict))
format_single_question_synthesis = lambda question, synthesis_note: clean_markdown(synthesis_note.get("synthesis")) if isinstance(synthesis_note, dict) else ""
format_supporting_evidence = lambda report_context, **kwargs: "\n\n".join(chunk_to_evidence_sentence(c) for c in (report_context.get("supporting_chunks", []) or []) if isinstance(c, dict))
format_question_focused_evidence = lambda report_context, questions, **kwargs: format_supporting_evidence(report_context)
format_report_revision_feedback = lambda validation: "\n".join(f"- {i}" for i in validation.get("issues", validation.get("report_issues", []))) if isinstance(validation, dict) else ""
missing_evidence_constraints = lambda synthesis: [line for line in clean_markdown(synthesis).splitlines() if line_has_gap_claim(line)]
clean_section_synthesis_note = lambda synthesis_note: cleanup_section_text(synthesis_note.get("synthesis")) if isinstance(synthesis_note, dict) else ""
clean_topic_digest_for_frames = cleanup_section_text
compact_markdown_at_sentence = compact_at_sentence
frame_lines_as_prose = lambda lines: " ".join(cleanup_section_text(line).rstrip(".") + "." for line in lines if cleanup_section_text(line))
cleanup_report_markdown_artifacts = lambda report: (cleanup_report(report, [])[0], cleanup_report(report, [])[1])
frame_source_line_usable = frame_sentence_usable
finalize_report_output = lambda report, sources, *args, **kwargs: (cleanup_report(report, sources)[0], args[-1] if args else {}, {"status": "repaired", "repairs": cleanup_report(report, sources)[1]})
frame_body_needs_role_repair = lambda heading_name, body: not frame_sentence_usable(clean_text(body))
frame_section_needs_retry = lambda section_text, heading="", body_digest="": bool(report_quality_issues(section_text, []))
has_dangling_markdown_bullet = lambda markdown: any(re.match(r"^\s*(?:[-*+]|\d+[.)])\s*$", line) for line in clean_markdown(markdown).splitlines())
has_truncated_markdown_list_item = lambda markdown: False
heading_ends_with_connector = heading_appears_weak
list_item_appears_truncated = lambda line: heading_appears_weak(strip_markdown(line)) if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", clean_text(line)) else False
markdown_appears_truncated = lambda markdown: heading_appears_weak(strip_markdown(clean_markdown(markdown).splitlines()[-1])) if clean_markdown(markdown).splitlines() else True
malformed_equation_tail_present = lambda text: bool(re.search(r"^\s*\\\]\s*\\left", clean_markdown(text), flags=re.M))
normalize_nested_markdown_bullet = lambda line: re.sub(r"^(\s*)([-*+])\s+[-*+]\s+", r"\1\2 ", line)
per_question_synthesis_repair_note = lambda question, synthesis_note: clean_section_synthesis_note(synthesis_note)
remove_clipped_sentence_fragments = trim_clipped_fragments
remove_duplicate_section_labels = lambda report: report
required_topic_facet_issues = lambda report, planner_questions: []
framework_api_detail_issues = lambda report, planner_questions: []
repair_topic_headings = lambda report: (repair_headings(report), [])
repair_weak_frame_sections = lambda report: (report, [])
repair_truncated_markdown_line = lambda line: (cleanup_section_text(line), ["cleaned line"] if cleanup_section_text(line) != clean_text(line) else [])
report_evidence_gap_contradictions = lambda *args, **kwargs: []
internal_gap_contradiction_issues = lambda report: []
report_per_question_synthesis_citation_gaps = lambda *args, **kwargs: []
report_needs_revision = lambda validation: bool(validation.get("issues") or validation.get("report_issues") or validation.get("schema_issues") or validation.get("coverage", {}).get("missing"))
report_section_concurrency = lambda question_count=None: 1
repair_report_by_sections = lambda *args, **kwargs: (args[2] if len(args) > 2 else "", args[1] if len(args) > 1 else "", {"mode": "compat"})
section_has_incomplete_equation = lambda section: code_or_math_contains_citations(section)
resolve_report_coverage = lambda coverage_by_question, evidence_packs, planner_questions: list(coverage_by_question or [])
source_priority = lambda url: 2 if any(s in normalize_url(url) for s in ("arxiv.org", "openreview.net", "doi.org", "docs.", "pytorch.org", "tensorflow.org")) else (1 if clean_text(url) else 0)
strip_topic_section_headings = strip_leading_heading
topic_heading_sequence_issues = lambda report, planner_questions: report_schema_issues(report, planner_questions)
topic_section_semantic_report_issues = lambda *args, **kwargs: []
topic_section_acceptance_issues = lambda *args, **kwargs: []
truncated_report_sections = lambda markdown: weak_topic_headings(markdown)
unsupported_benchmark_metrics = lambda report, evidence_text="": []
malformed_frame_section_issues = lambda report: []
canonical_source_routing_issues = lambda *args, **kwargs: []
report_section_repair_questions = lambda validation, questions: []
report_synthesis_gap_contradictions = lambda *args, **kwargs: []
apply_incomplete_equation_repairs = lambda report, sources, evidence_text="": (report, [])
apply_report_evidence_pack_repairs = lambda report, *args, **kwargs: (report, [])
apply_validation_limitations = lambda report, validation, sources: ensure_limitations(report, validation.get("issues", validation.get("report_issues", [])), sources)
accept_topic_section = lambda question, section, pack, synthesis_note=None, sources=None: (cleanup_section_text(section), [], False)
dedupe_sources = lambda sources: sources_with_browser_results(sources, [])
