"""Generate a RAG answer from already-retrieved context chunks."""

from __future__ import annotations

import os
from typing import Any, Sequence

from src.rag.retrieval import RetrievalResult, display_document_preview, result_source_urls_from_metadata
from src.tools.text_utils import clean_text


DEFAULT_RAG_GENERATION_MODEL = "llama-3.1-8b-instant"
DEFAULT_MAX_CONTEXT_CHARS = 12000
DEFAULT_MAX_TOKENS = 900
DEFAULT_REPORT_MAX_TOKENS = 1400


def generate_answer_from_context(
    question: str,
    retrieved_context: Sequence[RetrievalResult],
    model: str | None = None,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Use Groq to answer a question from retrieved RAG chunks only."""
    question = clean_text(question)
    if not question:
        raise ValueError("question is required")
    if not retrieved_context:
        return {"answer": "I could not find retrieved context to answer the question.", "sources": []}
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set")

    try:
        from groq import Groq
    except ImportError as error:
        raise RuntimeError("groq package is not installed. Install it with `pip install -r requirements.txt`.") from error

    context_text, sources = build_generation_context(retrieved_context, max_context_chars=max_context_chars)
    prompt = f"""Question:
{question}

Retrieved context:
{context_text}

Answer using only the retrieved context. If the context does not contain the answer, say that clearly.
Use source markers like [1] or [2] when you use information from a source."""

    response = Groq().chat.completions.create(
        model=clean_text(model or os.environ.get("RAG_GENERATION_MODEL")) or DEFAULT_RAG_GENERATION_MODEL,
        temperature=0,
        max_tokens=max(100, max_tokens),
        messages=[
            {
                "role": "system",
                "content": "You are a careful RAG answer generator. Do not use outside knowledge.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    answer = clean_text(response.choices[0].message.content)
    return {"answer": answer, "sources": sources, "model": response.model}


def synthesize_context_for_report(
    objective: str,
    retrieved_context: Sequence[RetrievalResult],
    synthesis_instruction: str = "",
    model: str | None = None,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_tokens: int = DEFAULT_REPORT_MAX_TOKENS,
) -> dict[str, Any]:
    """Synthesize retrieved chunks into a compact payload for a report agent."""
    objective = clean_text(objective)
    if not objective:
        raise ValueError("objective is required")
    if not retrieved_context:
        return {
            "objective": objective,
            "synthesis": "No retrieved context was available for report synthesis.",
            "sources": [],
        }
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set")

    try:
        from groq import Groq
    except ImportError as error:
        raise RuntimeError("groq package is not installed. Install it with `pip install -r requirements.txt`.") from error

    context_text, sources = build_generation_context(retrieved_context, max_context_chars=max_context_chars)
    instruction = clean_text(synthesis_instruction) or "Synthesize the retrieved evidence into report-ready research notes."
    prompt = f"""Research objective:
{objective}

Synthesis instruction:
{instruction}

Retrieved context from multiple sources:
{context_text}

Create report-agent-ready research notes using only the retrieved context.
Return concise Markdown with these sections:
- Key Findings
- Technical Details
- Source Evidence
- Conflicts Or Gaps
- Recommended Report Angle

Use source markers like [1], [2], [3] for every evidence-backed claim."""

    response = Groq().chat.completions.create(
        model=clean_text(model or os.environ.get("RAG_GENERATION_MODEL")) or DEFAULT_RAG_GENERATION_MODEL,
        temperature=0,
        max_tokens=max(300, max_tokens),
        messages=[
            {
                "role": "system",
                "content": "You synthesize retrieved RAG evidence for a downstream report agent. Do not use outside knowledge.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    synthesis = clean_text(response.choices[0].message.content)
    return {
        "objective": objective,
        "synthesis_instruction": instruction,
        "synthesis": synthesis,
        "sources": sources,
        "model": response.model,
    }


def build_generation_context(
    retrieved_context: Sequence[RetrievalResult],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> tuple[str, list[dict[str, Any]]]:
    """Convert retrieved chunks into compact numbered context blocks."""
    blocks = []
    sources = []
    used_chars = 0
    max_context_chars = max(1000, max_context_chars)

    for index, result in enumerate(retrieved_context, start=1):
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        url = primary_source_url(metadata)
        title = clean_text(metadata.get("title")) or url or f"Source {index}"
        chunk = display_document_preview(result.document, max_chars=2200)
        if not chunk:
            continue

        block = f"[{index}] {title}\nURL: {url}\n{chunk}"
        if used_chars + len(block) > max_context_chars:
            remaining = max_context_chars - used_chars
            if remaining < 500:
                break
            block = block[:remaining].rstrip()

        blocks.append(block)
        sources.append(
            {
                "index": index,
                "id": result.id,
                "url": url,
                "title": title,
                "score": result.score,
            }
        )
        used_chars += len(block)

    return "\n\n".join(blocks), sources


def primary_source_url(metadata: dict[str, Any]) -> str:
    urls = result_source_urls_from_metadata(metadata)
    return urls[0] if urls else clean_text(metadata.get("url"))
