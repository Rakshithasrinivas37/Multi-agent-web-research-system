"""Browser agent for running planner tasks in parallel."""

import asyncio
import re
from typing import Any, Optional
from urllib.parse import urlparse

from src.tools.firecrawl_tool import extract_with_firecrawl_or_httpx
from src.tools.pdf_tool import extract_pdf_text, is_pdf_url
from src.tools.playwright_tool import render_with_playwright
from src.tools.tavily_search import search_with_tavily
from src.tools.text_utils import clean_content, clean_list, clean_text

MIN_TASK_OVERLAP = 0.06
MIN_TRUSTED_DIRECT_CONTENT_LENGTH = 1000
MAX_NOISE_RATIO = 0.65
SOURCE_STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "how",
    "into",
    "compare",
    "comparison",
    "different",
    "research",
    "source",
    "sources",
    "the",
    "what",
    "which",
    "with",
}
LOW_VALUE_HOST_PARTS = (
    "academia.edu",
    "coursehero",
    "facebook.com",
    "medium.com",
    "pinterest.",
    "quora.com",
    "reddit.com",
    "researchgate.net",
    "scribd.com",
    "slideshare.net",
    "studocu",
    "twitter.com",
    "x.com",
)
NOISE_LINE_TERMS = (
    "accept cookies",
    "advertisement",
    "all rights reserved",
    "cookie policy",
    "follow us",
    "log in",
    "newsletter",
    "privacy policy",
    "related articles",
    "share this",
    "sign in",
    "sign up",
    "skip to",
    "subscribe",
    "terms of use",
)


class BrowserAgent:
    """Runs research-plan tasks and extracts task-focused evidence."""

    def __init__(self, max_concurrency: int = 3) -> None:
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def run_tasks(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        jobs = [self.run_task(task) for task in tasks]
        return await asyncio.gather(*jobs)

    async def run_task(self, task: dict[str, Any]) -> dict[str, Any]:
        async with self.semaphore:
            try:
                url = clean_text(task.get("url", ""))
                if url.startswith("SEARCH:"):
                    return await self.search_task(task)
                url = arxiv_pdf_url(url)
                if is_extractable_pdf_url(url):
                    return await self.pdf_task(task, url)
                return await self.url_task(task, url)
            except Exception as error:
                return self.failed_result(task, str(error))

    async def search_task(self, task: dict[str, Any]) -> dict[str, Any]:
        task_url = clean_text(task.get("url"))
        query = search_query_for_task(task)
        errors = []
        try:
            results = await asyncio.to_thread(search_with_tavily, query)
        except Exception as error:
            errors.append(f"Tavily search failed: {error}")
            results = []

        results = search_candidates_for_task(task, rank_search_results(task, results))

        sources = []
        for item in results:
            url = clean_text(item.get("url"))
            if not is_http_url(url):
                errors.append(f"Skipped invalid search result URL: {url}")
                continue

            try:
                page = await self.scrape_search_result(task, url)
            except Exception as error:
                errors.append(f"Failed to scrape search result {url}: {error}")
                snippet = item.get("content") or item.get("snippet") or ""
                page = {
                    "url": url,
                    "title": clean_text(item.get("title")),
                    "content": snippet,
                    "method": "tavily_snippet",
                    "errors": [str(error)],
                }

            extraction = await self.extract(task, page["url"], page["title"] or clean_text(item.get("title")), page["content"])
            payload = self.source_payload(page, item.get("source_type", "search_result"), page["method"], extraction, task)
            if not source_is_useful(payload, task):
                errors.append(f"Skipped low-quality source: {url} ({source_quality_note(payload, task)})")
                continue
            sources.append(payload)
            errors.extend(page.get("errors", []))

        return {
            "task_id": task.get("task_id"),
            "query_context": task.get("query_context"),
            "task_url": task_url,
            "status": task_status(sources, errors),
            "query": query,
            "sources": sources,
            "errors": errors if sources else errors + ["No valid Tavily results found."],
        }

    async def scrape_search_result(self, task: dict[str, Any], url: str) -> dict[str, Any]:
        url = arxiv_pdf_url(url)
        if is_extractable_pdf_url(url):
            text = await asyncio.to_thread(extract_pdf_text, url)
            return {"url": url, "title": "", "content": text, "method": "pdf", "errors": [] if text else ["No PDF text extracted."]}
        if use_playwright_for_search_result(task, url):
            try:
                return await render_with_playwright(url)
            except Exception:
                return await extract_with_firecrawl_or_httpx(url)
        return await extract_with_firecrawl_or_httpx(url)

    async def url_task(self, task: dict[str, Any], url: str) -> dict[str, Any]:
        if task.get("use_playwright"):
            page = await render_with_playwright(url)
        else:
            page = await extract_with_firecrawl_or_httpx(url)

        extraction = await self.extract(task, page["url"], page["title"], page["content"])
        result = self.source_result(task, page, task.get("source_type", "webpage"), page["method"], extraction)
        if result["status"] in {"blocked", "failed"}:
            return await self.retry_direct_url_with_search(task, result)
        return result

    async def pdf_task(self, task: dict[str, Any], url: str) -> dict[str, Any]:
        try:
            pdf_text = await asyncio.to_thread(extract_pdf_text, url)
        except Exception as error:
            result = self.failed_result(task, str(error))
            return await self.retry_direct_url_with_search(task, result)

        extraction = await self.extract(task, url, "", pdf_text)
        page = {"url": url, "title": "", "content": pdf_text, "errors": [] if pdf_text else ["No PDF text extracted."]}
        status = "success" if pdf_text else "failed"
        result = self.source_result(task, page, "pdf", "pdf", extraction, status)
        if result["status"] in {"blocked", "failed"}:
            return await self.retry_direct_url_with_search(task, result)
        return result

    async def retry_direct_url_with_search(self, task: dict[str, Any], direct_result: dict[str, Any]) -> dict[str, Any]:
        fallback_task = fallback_search_task(task, direct_result)
        fallback_result = await self.search_task(fallback_task)
        fallback_sources = fallback_result.get("sources", [])
        fallback_errors = fallback_result.get("errors", [])
        direct_errors = direct_result.get("errors", [])

        if fallback_sources:
            status = "partial"
            sources = fallback_sources
            errors = [
                *direct_errors,
                f"Direct URL extraction was {direct_result.get('status')}; used Tavily fallback search.",
                *fallback_errors,
            ]
        else:
            status = direct_result.get("status", "failed")
            sources = direct_result.get("sources", [])
            errors = [
                *direct_errors,
                f"Direct URL extraction was {direct_result.get('status')}; Tavily fallback search found no usable sources.",
                *fallback_errors,
            ]

        return {
            "task_id": task.get("task_id"),
            "query_context": task.get("query_context"),
            "task_url": task.get("url"),
            "status": status,
            "sources": sources,
            "errors": errors,
            "fallback_used": bool(fallback_sources),
            "fallback_query": fallback_result.get("query"),
            "fallback_status": fallback_result.get("status"),
            "direct_url_status": direct_result.get("status"),
            "direct_url_sources": direct_result.get("sources", []),
            "direct_url_errors": direct_errors,
        }

    async def extract(self, task: dict[str, Any], url: str, title: str, content: str) -> dict[str, Any]:
        return fallback_extraction(clean_source_content(content), task)

    def source_result(
        self,
        task: dict[str, Any],
        page: dict[str, Any],
        source_type: str,
        method: str,
        extraction: dict[str, Any],
        status: str = "success",
    ) -> dict[str, Any]:
        payload = self.source_payload(page, source_type, method, extraction, task)
        errors = list(page.get("errors", []))
        if is_bot_blocked_page(page):
            errors.append(bot_block_note())
            payload["source_quality"] = "blocked"
            payload["source_quality_note"] = bot_block_note()
            payload["fallback_needed"] = True
            payload["fallback_recommended"] = fallback_recommendation(task)
            status = "blocked"
        if status == "success" and not source_is_useful(payload, task):
            errors.append(source_quality_note(payload, task))
            status = "failed"

        return {
            "task_id": task.get("task_id"),
            "query_context": task.get("query_context"),
            "task_url": task.get("url"),
            "status": status,
            "sources": [payload],
            "errors": errors,
        }

    def source_payload(
        self,
        page: dict[str, Any],
        source_type: str,
        method: str,
        extraction: dict[str, Any],
        task: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        raw_text = clean_content(page.get("content", ""))
        full_text = clean_source_content(raw_text)
        payload = {
            "url": page["url"],
            "title": page["title"],
            "source_type": source_type,
            "extraction_method": method,
            "source_quality": "unchecked",
            "source_authority": source_authority_level(page["url"], source_type, task or {}),
            "content_length": len(full_text),
            "content_noise_score": content_noise_score(raw_text),
            "content_preview": full_text[:2000],
            "full_content": full_text,
            "errors": list(page.get("errors", [])),
            **extraction,
        }
        if is_bot_blocked_page(page):
            payload["blocked"] = True
            payload["fallback_needed"] = True
            payload["fallback_recommended"] = fallback_recommendation(task or {})
        if is_pricing_task(task or {}):
            payload["pricing_rows"] = extract_pricing_rows(full_text)
        if payload.get("blocked"):
            payload["source_quality"] = "blocked"
            payload["source_quality_note"] = bot_block_note()
        else:
            payload["source_quality"] = source_quality_label(payload, task or {})
            payload["source_quality_note"] = source_quality_note(payload, task or {})
        return payload

    def failed_result(self, task: dict[str, Any], error: str) -> dict[str, Any]:
        return {
            "task_id": task.get("task_id"),
            "query_context": task.get("query_context"),
            "task_url": task.get("url"),
            "status": "failed",
            "sources": [],
            "errors": [error],
        }


def fallback_extraction(content: str, task: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    text = clean_content(content)
    task = task or {}
    if is_pricing_task(task):
        rows = extract_pricing_rows(text)
        facts = rows[:30] or extract_pricing_facts(text)
        sections = extract_pricing_sections(text)
        return {
            "extraction_status": "fallback",
            "relevance": "high" if facts else ("medium" if text else "low"),
            "extracted_facts": facts or extract_important_sentences(text, task=task),
            "evidence": facts[:8],
            "important_sections": sections or extract_important_sections(text),
            "notes": "Fallback pricing extraction used.",
        }

    return {
        "extraction_status": "fallback",
        "relevance": "medium" if text else "low",
        "extracted_facts": extract_important_sentences(text, task=task),
        "evidence": [],
        "important_sections": extract_important_sections(text),
        "notes": "Deterministic browser extraction used.",
    }


def is_pricing_task(task: dict[str, Any]) -> bool:
    text = " ".join(
        [
            clean_text(task.get("source_type")),
            clean_text(task.get("query_context")),
            clean_text(task.get("extraction_goal")),
            " ".join(clean_list(task.get("expected_signals"))),
        ]
    ).lower()
    return any(word in text for word in ("price", "pricing", "tokens", "cached input", "batch", "free tier", "rate limits"))


def extract_pricing_facts(text: str, limit: int = 30) -> list[str]:
    if not text:
        return []

    snippets: list[str] = []
    for match in re.finditer(r"\$[\d,]+(?:\.\d+)?", text):
        snippet = clean_text(text[max(0, match.start() - 180) : min(len(text), match.end() + 260)])
        if useful_pricing_snippet(snippet) and snippet not in snippets:
            snippets.append(snippet[:700])
        if len(snippets) >= limit:
            return snippets
    if snippets:
        return snippets

    patterns = (
        r"(?i)(?:model|input|output|cached|cache|batch|price|pricing|token|free tier|rate limit|context)[^.]{0,220}(?:\$[\d.,]+|percent|free|tokens?|requests?)[^.]{0,220}",
        r"(?i)[A-Za-z0-9_.-]{2,40}[^.]{0,160}\$[\d.,]+[^.]{0,180}(?:input|output|tokens?|cached|batch|context)?",
        r"(?i)(?:\$[\d.,]+)[^.]{0,220}(?:per|/)\s*(?:1m|million|token|tokens|request|requests|minute|hour|day)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            snippet = clean_text(match.group(0))
            if useful_pricing_snippet(snippet) and snippet not in snippets:
                snippets.append(snippet[:700])
            if len(snippets) >= limit:
                return snippets

    return snippets


def extract_pricing_rows(text: str, limit: int = 80) -> list[str]:
    rows = []
    for line in text.splitlines():
        row = normalize_pricing_row(line)
        if not row or row in rows:
            continue
        if useful_pricing_row(row):
            rows.append(row[:1000])
        if len(rows) >= limit:
            return rows
    return rows


def normalize_pricing_row(row: str) -> str:
    row = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", str(row or ""))
    row = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", row)
    row = re.sub(r"https?://\S+", " ", row)
    row = re.sub(r"\s*\|\s*", " | ", row)
    return clean_text(row)


def useful_pricing_row(row: str) -> bool:
    lower = row.lower()
    if not useful_pricing_snippet(row):
        return False
    if "$" not in row and "free of charge" not in lower:
        return False

    model_or_provider = any(
        word in lower
        for word in (
            "gpt",
            "claude",
            "gemini",
            "llama",
            "mistral",
            "qwen",
            "deepseek",
            "groq",
            "openai",
            "anthropic",
            "google",
            "model",
        )
    )
    pricing_context = any(
        word in lower
        for word in (
            "input",
            "output",
            "cached",
            "cache",
            "token",
            "mtok",
            "batch",
            "context",
            "rate limit",
            "requests",
            "tpm",
            "rpm",
        )
    )
    table_like = "|" in row and row.count("$") >= 2
    compact_price_row = len(row) < 260 and row.count("$") >= 1 and model_or_provider
    return (model_or_provider and pricing_context) or table_like or compact_price_row


def extract_pricing_sections(text: str, limit: int = 12) -> list[str]:
    keywords = (
        "pricing",
        "model input",
        "cached input",
        "cache writes",
        "output",
        "batch",
        "free tier",
        "rate limit",
        "context",
        "tokens",
        "discount",
    )
    sections = []
    lower_text = text.lower()
    for keyword in keywords:
        start = lower_text.find(keyword)
        while start != -1:
            section = clean_text(text[max(0, start - 250) : start + 900])
            if section and section not in sections:
                sections.append(section)
            if len(sections) >= limit:
                return sections
            start = lower_text.find(keyword, start + len(keyword))
    return sections


def useful_snippet(snippet: str) -> bool:
    lower = snippet.lower()
    noisy = ("linkedin", "twitter", "facebook", "gmail", "copy_link", "cookie")
    return len(snippet) > 30 and not any(word in lower for word in noisy)


def useful_pricing_snippet(snippet: str) -> bool:
    lower = snippet.lower()
    noisy = (
        "linkedin",
        "twitter",
        "facebook",
        "gmail",
        "copy_link",
        "cookie",
        "privacy policy",
        "terms of service",
        "careers",
        "get started",
        "our services",
        "our work",
        "partnership",
        "log in",
        "sign up",
        "navbar",
        "footer",
    )
    if len(snippet) <= 20 or any(word in lower for word in noisy):
        return False
    if snippet.count("[") + snippet.count("]") > 4:
        return False
    if snippet.count("http") > 0:
        return False
    return True


def extract_important_sentences(text: str, limit: int = 12, task: Optional[dict[str, Any]] = None) -> list[str]:
    if not text:
        return []

    keywords = important_sentence_keywords(task or {})
    sentences = [sentence.strip() for sentence in text.replace("\n", " ").split(".") if sentence.strip()]
    important = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)]
    chosen = important[:limit] if important else sentences[:limit]
    return [sentence[:500] for sentence in chosen]


def important_sentence_keywords(task: dict[str, Any]) -> tuple[str, ...]:
    generic = {
        "api",
        "application",
        "architecture",
        "benchmark",
        "comparison",
        "component",
        "definition",
        "equation",
        "evidence",
        "example",
        "implementation",
        "limitation",
        "metric",
        "performance",
        "result",
        "source",
    }
    task_terms = {
        term
        for term in relevance_terms(task_relevance_text(task))
        if len(term) > 3 and term not in SOURCE_STOPWORDS
    }
    return tuple(sorted(generic | set(list(task_terms)[:20])))


def extract_important_sections(text: str, limit: int = 10) -> list[str]:
    if not text:
        return []
    words = text.split()
    chunks = [" ".join(words[index : index + 80]) for index in range(0, min(len(words), 800), 80)]
    return [chunk for chunk in chunks if chunk][:limit]


def is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def arxiv_pdf_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.lower() != "arxiv.org":
        return url
    if parsed.path.startswith(("/abs/", "/html/")):
        paper_id = re.sub(r"^/(abs|html)/", "", parsed.path).strip("/")
        return f"https://arxiv.org/pdf/{paper_id}"
    return url


def is_arxiv_pdf_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc.lower() == "arxiv.org" and parsed.path.startswith("/pdf/")


def is_extractable_pdf_url(url: str) -> bool:
    return is_pdf_url(url) or is_arxiv_pdf_url(url)


def search_query_for_task(task: dict[str, Any]) -> str:
    return clean_text(task.get("url")).removeprefix("SEARCH:").strip()


def fallback_search_task(task: dict[str, Any], direct_result: dict[str, Any]) -> dict[str, Any]:
    fallback_task = dict(task)
    fallback_task["url"] = f"SEARCH:{direct_url_fallback_query(task, direct_result)}"
    fallback_task["source_type"] = "search"
    fallback_task["use_playwright"] = False
    return fallback_task


def direct_url_fallback_query(task: dict[str, Any], direct_result: dict[str, Any]) -> str:
    original_url = clean_text(task.get("url"))
    host = urlparse(original_url).netloc.replace("www.", "")
    parts = [
        clean_text(task.get("target_name")) if clean_text(task.get("target_name")) != "General Research" else "",
        clean_text(task.get("query_context")),
        clean_text(task.get("extraction_goal")),
        " ".join(clean_list(task.get("expected_signals"))),
        clean_text(task.get("source_type")),
        host,
        "alternative source",
    ]
    query = clean_text(" ".join(part for part in parts if part))
    if not query:
        query = clean_text(original_url) or "research source"
    return dedupe_query_words(query)


def dedupe_query_words(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9+.#:/_-]+", clean_text(text))
    seen = set()
    result = []
    for word in words:
        key = word.lower()
        if key in seen or key in {"a", "an", "and", "for", "of", "the", "to", "what", "which"}:
            continue
        seen.add(key)
        result.append(word)
    return " ".join(result)[:300].strip()


def task_status(sources: list[dict[str, Any]], errors: list[str]) -> str:
    if not sources:
        return "failed"
    return "partial" if errors else "success"


def source_is_useful(payload: dict[str, Any], task: dict[str, Any]) -> bool:
    if payload.get("blocked"):
        return False
    if blocked_or_empty_source(payload):
        return False
    if payload.get("content_noise_score", 0) > MAX_NOISE_RATIO and payload.get("content_length", 0) < 1200:
        return False
    if is_bad_dictionary_source(payload, task) or is_not_found_source(payload):
        return False
    if is_pricing_task(task):
        return bool(payload.get("pricing_rows") or payload.get("extracted_facts"))
    if is_trusted_direct_source(payload, task):
        return True
    if not source_matches_target(payload, task):
        return False
    if not source_matches_task(payload, task):
        return False
    return bool(payload.get("extracted_facts") or payload.get("important_sections") or payload.get("content_length", 0) >= 800)


def source_quality_label(payload: dict[str, Any], task: dict[str, Any]) -> str:
    if not source_is_useful(payload, task):
        return "weak"
    authority = clean_text(payload.get("source_authority"))
    if authority in {"primary", "official", "authoritative"}:
        return f"useful_{authority}"
    return "useful_secondary"


def source_quality_note(payload: dict[str, Any], task: dict[str, Any]) -> str:
    if payload.get("blocked") or is_bot_blocked_text(
        " ".join(
            [
                clean_text(payload.get("title")),
                clean_text(payload.get("content_preview")),
                clean_text(payload.get("full_content"))[:1000],
                " ".join(clean_list(payload.get("errors"))),
            ]
        )
    ):
        return bot_block_note()
    if blocked_or_empty_source(payload):
        return "source content is empty, blocked, or too short"
    if payload.get("content_noise_score", 0) > MAX_NOISE_RATIO and payload.get("content_length", 0) < 1200:
        return "source content is mostly boilerplate or navigation noise"
    if is_bad_dictionary_source(payload, task):
        return "dictionary result is not relevant to this research task"
    if is_not_found_source(payload):
        return "source appears to be a not-found or missing page"
    if is_pricing_task(task) and not (payload.get("pricing_rows") or payload.get("extracted_facts")):
        return "no pricing rows or pricing facts found"
    if is_trusted_direct_source(payload, task):
        return "trusted direct authoritative source"
    if not source_matches_target(payload, task):
        return "source does not match the task target topic"
    if not source_matches_task(payload, task):
        return "source content has low overlap with the task"
    return "source passed basic quality checks"


def clean_source_content(content: str) -> str:
    lines = []
    seen: dict[str, int] = {}
    for line in clean_content(content).splitlines():
        line = clean_text(line)
        if not line or is_noise_line(line):
            continue
        key = line.lower()
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1 and len(line) < 160:
            continue
        lines.append(line)
    return "\n".join(lines)


def is_noise_line(line: str) -> bool:
    lower = line.lower()
    if len(line) <= 2:
        return True
    if len(line) < 120 and any(term in lower for term in NOISE_LINE_TERMS):
        return True
    if len(line.split()) <= 4 and any(term in lower for term in ("menu", "home", "login", "share", "cookie")):
        return True
    return False


def content_noise_score(content: str) -> float:
    lines = [clean_text(line) for line in clean_content(content).splitlines() if clean_text(line)]
    if not lines:
        return 1.0
    noisy = sum(1 for line in lines if is_noise_line(line))
    return noisy / len(lines)


def source_authority_level(url: str, source_type: str = "", task: Optional[dict[str, Any]] = None) -> str:
    parsed = urlparse(clean_text(url))
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.lower()
    source_type = clean_text(source_type).lower()

    if host == "arxiv.org" or host.endswith("doi.org") or source_type in {"arxiv", "academic", "paper"}:
        return "primary"
    if source_type in {"docs", "pricing", "careers"} or host.startswith("docs.") or "/docs" in path or "/api_docs" in path:
        return "official"
    if host.endswith(".gov") or host.endswith(".edu") or source_type in {"pdf", "benchmarks"}:
        return "authoritative"
    if task and source_matches_target({"url": url, "title": "", "content_preview": host, "full_content": ""}, task):
        return "topic_match"
    return "secondary"


def is_trusted_direct_source(payload: dict[str, Any], task: dict[str, Any]) -> bool:
    task_url = clean_text(task.get("url"))
    source_url = clean_text(payload.get("url"))
    if not task_url or task_url.startswith("SEARCH:") or not source_url:
        return False
    if payload.get("content_length", 0) < MIN_TRUSTED_DIRECT_CONTENT_LENGTH:
        return False
    return same_planned_url(task_url, source_url) and is_authoritative_source_url(source_url)


def same_planned_url(task_url: str, source_url: str) -> bool:
    return normalize_url_key(arxiv_pdf_url(task_url)) == normalize_url_key(arxiv_pdf_url(source_url))


def is_authoritative_source_url(url: str) -> bool:
    parsed = urlparse(clean_text(url))
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.lower()
    return (
        host == "arxiv.org"
        or host == "openreview.net"
        or host.endswith(".edu")
        or host.endswith(".gov")
        or host.endswith("doi.org")
        or host.startswith("docs.")
        or "docs." in host
        or "/api_docs" in path
        or path.endswith(".pdf")
    )


def blocked_or_empty_source(payload: dict[str, Any]) -> bool:
    text = " ".join(
        [
            clean_text(payload.get("title")),
            clean_text(payload.get("content_preview")),
            clean_text(payload.get("full_content"))[:1000],
        ]
    ).lower()
    return payload.get("content_length", 0) < 300 or is_bot_blocked_text(text)


def is_bad_dictionary_source(payload: dict[str, Any], task: dict[str, Any]) -> bool:
    host = urlparse(clean_text(payload.get("url"))).netloc.lower()
    if not any(domain in host for domain in ("dictionary.com", "merriam-webster.com", "vocabulary.com")):
        return False
    task_text = task_relevance_text(task).lower()
    return not any(word in task_text for word in ("definition", "meaning", "etymology", "word"))


def is_not_found_source(payload: dict[str, Any]) -> bool:
    text = " ".join([clean_text(payload.get("title")), clean_text(payload.get("content_preview"))]).lower()
    return any(term in text for term in ("404", "page not found", "does not exist", "no article"))


def source_matches_target(payload: dict[str, Any], task: dict[str, Any]) -> bool:
    target_name = clean_text(task.get("target_name"))
    if not target_name or target_name.lower() == "general research":
        return True
    target_terms = relevance_terms(target_name)
    source_terms = relevance_terms(source_relevance_text(payload))
    return not target_terms or bool(target_terms & source_terms)


def source_matches_task(payload: dict[str, Any], task: dict[str, Any]) -> bool:
    task_terms = relevance_terms(task_relevance_text(task))
    if not task_terms:
        return True
    source_terms = relevance_terms(source_relevance_text(payload))
    overlap = len(task_terms & source_terms) / max(1, min(len(task_terms), len(source_terms)))
    return overlap >= MIN_TASK_OVERLAP


def source_relevance_text(payload: dict[str, Any]) -> str:
    return " ".join(
        [
            clean_text(payload.get("title")),
            clean_text(payload.get("content_preview")),
            clean_text(payload.get("full_content"))[:3000],
        ]
    )


def task_relevance_text(task: dict[str, Any]) -> str:
    return " ".join(
        [
            clean_text(task.get("query_context")),
            clean_text(task.get("extraction_goal")),
            clean_text(task.get("target_name")),
            " ".join(clean_list(task.get("expected_signals"))),
        ]
    )


def relevance_terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+", clean_text(text))
        if len(token) > 2 and token.lower() not in SOURCE_STOPWORDS
    }


def is_bot_blocked_page(page: dict[str, Any]) -> bool:
    text = " ".join(
        [
            clean_text(page.get("title")),
            clean_text(page.get("content"))[:3000],
            " ".join(clean_list(page.get("errors"))),
        ]
    )
    return is_bot_blocked_text(text)


def is_bot_blocked_text(text: str) -> bool:
    lower = clean_text(text).lower()
    blocked_terms = (
        "access denied",
        "attention required",
        "bot detection",
        "captcha",
        "cf-chl",
        "checking if the site connection is secure",
        "checking your browser",
        "cloudflare",
        "enable javascript and cookies",
        "error 1005",
        "error 1015",
        "forbidden",
        "http 403",
        "http 429",
        "http 503",
        "just a moment",
        "performance and security by cloudflare",
        "please verify you are a human",
        "rate limited",
        "request blocked",
        "unusual traffic",
        "verify you are human",
    )
    return any(term in lower for term in blocked_terms)


def bot_block_note() -> str:
    return "page appears blocked by Cloudflare, CAPTCHA, bot protection, rate limits, or access controls"


def fallback_recommendation(task: dict[str, Any]) -> str:
    if clean_text(task.get("url")).startswith("SEARCH:"):
        return "Use another Tavily result or retry later."
    return "Use Tavily search, another authoritative source URL, cached content, or manual review."


def search_candidates_for_task(task: dict[str, Any], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for item in results:
        url = clean_text(item.get("url"))
        key = normalize_url_key(url)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def normalize_url_key(url: str) -> str:
    if not is_http_url(url):
        return ""
    parsed = urlparse(url)
    return f"{parsed.netloc.lower()}{parsed.path.rstrip('/').lower()}"


def use_playwright_for_search_result(task: dict[str, Any], url: str) -> bool:
    return is_pdf_url(url) is False and bool(task.get("use_playwright"))


def rank_search_results(task: dict[str, Any], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(results, key=lambda item: search_result_score(task, clean_text(item.get("url"))), reverse=True)


def search_result_score(task: dict[str, Any], url: str) -> int:
    if not is_http_url(url):
        return -100

    score = 0
    target_name = clean_text(task.get("target_name")).lower()
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    text = f"{host} {path}"
    if target_name and target_name != "general research" and target_name in text:
        score += 40
    if is_authoritative_source_url(url):
        score += 35
    if source_authority_level(url, clean_text(task.get("source_type")), task) in {"primary", "official"}:
        score += 20
    if any(part in host for part in LOW_VALUE_HOST_PARTS):
        score -= 60
    if any(word in text for word in ("forum", "login", "signin", "signup", "tag/", "category/")):
        score -= 15
    task_text = task_relevance_text(task).lower()
    if any(word in task_text for word in ("api", "documentation", "implementation", "code", "usage")):
        score += sum(15 for word in ("docs", "api_docs", "reference", "developer") if word in text)
    if any(word in task_text for word in ("equation", "formula", "paper", "benchmark", "metric", "dataset")):
        score += sum(15 for word in ("arxiv", "doi", "paper", "proceedings", "benchmark", "dataset") if word in text)
    if is_pricing_task(task):
        score += sum(20 for word in ("pricing", "price", "docs", "console", "platform") if word in text)
        score -= sum(80 for word in ("community", "forum", "wikipedia", "aipricing", "reddit", "medium.com") if word in text)
    return score
