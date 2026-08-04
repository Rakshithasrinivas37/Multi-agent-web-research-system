"""Report agent for final research report generation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Sequence

from src.memory.shared_memory import SharedMemory
from src.tools.text_utils import clean_text


DEFAULT_REPORT_AGENT_MODEL = "llama-3.1-8b-instant"
DEFAULT_REPORT_AGENT_MAX_TOKENS = 3200
DEFAULT_REPORT_AGENT_CONTEXT_CHARS = 30000
DEFAULT_REPORT_AGENT_CHUNK_CHARS = 1000
DEFAULT_REPORT_OUTPUT_DIR = "data/reports"


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
        synthesis = clean_text(report_context.get("synthesis"))
        if not objective:
            raise ValueError("report_context.objective is required")
        if not synthesis:
            raise ValueError("report_context.synthesis is required")

        source_text = format_sources(report_context.get("sources", []))
        evidence_text = format_supporting_evidence(report_context)
        citation_policy = clean_text(report_context.get("citation_policy")) or (
            "Use only numbered source markers from the provided sources."
        )
        diagnostics = report_context.get("diagnostics", {})

        prompt = f"""Research objective:
{objective}

Requested output format:
{clean_text(output_format) or "report"}

Citation policy:
{citation_policy}

Synthesis-agent notes:
{synthesis}

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
- Include technical sections that match the objective and synthesis.
- Include equations only when they are supported by synthesis or supporting chunks.
- Cite claims using only plain source markers from Available sources, exactly like [1], [2], [3].
- For formulas, APIs, benchmark claims, and historical attribution, prefer original papers, official docs, academic sources, or authoritative surveys.
- Do not cite sources that are not listed.
- Do not use citation formats like 【1】, footnotes, line citations, or URLs inline.
- End with a References section mapping source markers to source URLs.
- If evidence is incomplete, mention the limitation instead of inventing details."""

        response = Groq().chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=max(500, self.max_tokens),
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful report-writing agent. Use only provided synthesis context and evidence.",
                },
                {"role": "user", "content": prompt[: self.max_context_chars]},
            ],
        )
        report = normalize_citation_markers(response.choices[0].message.content)
        return {
            "objective": objective,
            "output_format": clean_text(output_format) or "report",
            "report": report,
            "sources": report_context.get("sources", []),
            "model": response.model,
            "diagnostics": {
                "source_count": len(report_context.get("sources", []) or []),
                "supporting_chunk_count": len(report_context.get("supporting_chunks", []) or []),
                "report_length": len(report),
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


def write_report_file(
    report_payload: dict[str, Any],
    memory_path: str = "data/shared_memory.json",
    report_path: str | None = None,
) -> str:
    """Save report Markdown to disk and return its path."""

    report = clean_text(report_payload.get("report"))
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


def format_supporting_evidence(report_context: dict[str, Any]) -> str:
    chunks = report_context.get("supporting_chunks") or report_context.get("retrieved_chunks") or []
    if not isinstance(chunks, list):
        return "No supporting chunks provided."

    blocks = []
    used_chars = 0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        source_index = chunk.get("source_index")
        if not isinstance(source_index, int):
            source_index = chunk.get("index")
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
    normalized = clean_text(text)
    normalized = re.sub(r"【\s*(\d+)(?:[^】]*)?】", r"[\1]", normalized)
    normalized = re.sub(r"\[\s*(\d+)\s*\]", r"[\1]", normalized)
    return normalized
