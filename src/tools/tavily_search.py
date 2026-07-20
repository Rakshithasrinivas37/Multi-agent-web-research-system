"""Tavily search tool."""

import os
from typing import Any


def search_with_tavily(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")

    from tavily import TavilyClient

    response = TavilyClient(api_key=api_key).search(
        query=query,
        max_results=max_results,
        search_depth="basic",
    )
    return response.get("results", [])
