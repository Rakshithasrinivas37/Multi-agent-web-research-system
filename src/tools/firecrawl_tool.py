"""Firecrawl extraction tool."""

import asyncio
import os
from typing import Any

from src.tools.playwright_tool import fetch_with_httpx
from src.tools.text_utils import clean_text


async def extract_with_firecrawl_or_httpx(url: str) -> dict[str, Any]:
    content = await asyncio.to_thread(extract_with_firecrawl, url)
    if content:
        return {
            "url": url,
            "title": "",
            "content": content,
            "method": "firecrawl",
            "errors": [],
        }
    return await fetch_with_httpx(url, "Firecrawl unavailable or returned no content; used httpx fallback.")


def extract_with_firecrawl(url: str) -> str:
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        return ""

    try:
        from firecrawl import FirecrawlApp
    except ImportError:
        return ""

    try:
        app = FirecrawlApp(api_key=api_key)
        try:
            result = app.scrape_url(url, formats=["markdown"])
        except TypeError:
            result = app.scrape_url(url, params={"formats": ["markdown"]})
    except Exception:
        return ""

    if isinstance(result, dict):
        return clean_text(result.get("markdown") or result.get("content") or "")
    return clean_text(getattr(result, "markdown", "") or getattr(result, "content", ""))
