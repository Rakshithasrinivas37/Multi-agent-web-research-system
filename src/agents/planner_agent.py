"""LLM-first planner agent for research workflows."""

import json
import os
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import httpx


@dataclass(frozen=True)
class ResearchTask:
    task_id: str
    query_context: str
    url: str
    source_type: str
    priority: int
    extraction_goal: str
    target_type: str = "discovery"
    target_name: str = "General Research"
    use_playwright: bool = False
    expected_signals: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResearchPlan:
    objective: str
    research_mode: str
    sub_questions: list[str]
    tasks: list[ResearchTask]
    synthesis_instruction: str
    output_format: str
    companies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "research_mode": self.research_mode,
            "companies": self.companies,
            "sub_questions": self.sub_questions,
            "tasks": [asdict(task) for task in self.tasks],
            "synthesis_instruction": self.synthesis_instruction,
            "output_format": self.output_format,
        }


SOURCE_TYPES = {
    "webpage",
    "search",
    "wikipedia",
    "arxiv",
    "academic",
    "technical_overview",
    "benchmarks",
    "implementation",
    "news",
    "pricing",
    "docs",
    "careers",
    "reviews",
}

SOURCE_ALIASES = {
    "blog": "news",
    "company_blog": "news",
    "documentation": "docs",
    "official_docs": "docs",
    "paper": "arxiv",
    "papers": "arxiv",
    "price": "pricing",
    "pricing_page": "pricing",
    "research_paper": "academic",
    "search_query": "search",
}

RESEARCH_MODES = {"competitor_intel", "knowledge_research", "technical_deep_dive", "market_research"}
OUTPUT_FORMATS = {"comparison_table", "deep_dive", "summary", "report"}


class PlannerAgent:
    PROMPT = f"""You are the planner in a multi-agent web research system.

Given one research objective, decide the research mode, companies/topics,
sub-questions, URLs or search queries, source types, and synthesis guidance.

Return only valid JSON:
{{
  "research_mode": "competitor_intel|knowledge_research|technical_deep_dive|market_research",
  "companies": ["company names if the objective compares organizations/products"],
  "sub_questions": ["specific question the research should answer"],
  "tasks": [{{
    "query_context": "which sub-question this task answers",
    "url": "https://real-source-url.com OR SEARCH:precise search query",
    "source_type": "one of {', '.join(sorted(SOURCE_TYPES))}",
    "priority": 1,
    "extraction_goal": "what the next agent should extract",
    "target_type": "company|discovery",
    "target_name": "company/topic name or General Research",
    "use_playwright": false,
    "expected_signals": ["facts or fields to look for"]
  }}],
  "synthesis_instruction": "specific instructions for the final answer",
  "output_format": "comparison_table|deep_dive|summary|report"
}}

Rules:
- You decide all companies, sources, URLs, queries, and sub-questions.
- Prefer real source URLs when you are confident they exist.
- Use SEARCH: when you are not confident about the exact URL.
- In competitor_intel mode, cover every company across the important sub-questions.
- Keep the plan compact enough to execute: usually 6 to 10 tasks.
- Return JSON only. No markdown."""

    def __init__(self, use_llm: bool = True, model: Optional[str] = None, validate_urls: Optional[bool] = None) -> None:
        self.use_llm = use_llm
        self.model = model or os.environ.get("RESEARCH_PLANNER_MODEL", "llama-3.1-8b-instant")
        env_value = os.environ.get("RESEARCH_PLANNER_VALIDATE_URLS", "1").lower()
        self.validate_urls = validate_urls if validate_urls is not None else env_value not in {"0", "false", "no"}

    def plan(self, objective: str) -> ResearchPlan:
        if self.use_llm:
            try:
                return self._plan_with_groq(objective)
            except Exception as error:
                print(f"[planner_agent] Groq planner unavailable; using fallback planner: {error}")
        return self._fallback_plan(objective)

    def _plan_with_groq(self, objective: str) -> ResearchPlan:
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set")
        try:
            from groq import Groq
        except ImportError as error:
            raise RuntimeError("groq package is not installed") from error

        client = Groq()
        request = f"Create a compact research plan for: {objective}"
        messages = [{"role": "user", "content": request}]
        last_error: Optional[Exception] = None

        for attempt in range(1, 3):
            response = client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=1600,
                messages=[{"role": "system", "content": self.PROMPT}, *messages],
            )
            raw = (response.choices[0].message.content or "").strip()
            try:
                return self._parse_plan(raw, objective)
            except (json.JSONDecodeError, KeyError, ValueError) as error:
                last_error = error
                messages = [
                    {"role": "user", "content": request},
                    {"role": "user", "content": f"Previous output failed validation: {error}. Return shorter valid JSON only."},
                ]
                print(f"[planner_agent] Plan parse failed on attempt {attempt}: {error}")

        raise RuntimeError(f"Groq planner failed after 2 attempts: {last_error}")

    def _parse_plan(self, raw: str, objective: str) -> ResearchPlan:
        data = json.loads(strip_fence(raw))
        companies = clean_list(data.get("companies"))
        sub_questions = clean_list(data.get("sub_questions")) or [objective]
        mode = clean_mode(data.get("research_mode"))
        output_format = clean_output_format(data.get("output_format"))
        tasks = [self._task_from_dict(item, index) for index, item in enumerate(data.get("tasks", []), 1) if isinstance(item, dict)]

        if not tasks:
            raise ValueError("planner returned no tasks")

        tasks = [self._safe_task(task) for task in tasks]
        tasks = dedupe_and_renumber(tasks)
        validate_plan(tasks, mode, companies, sub_questions)

        return ResearchPlan(
            objective=objective,
            research_mode=mode,
            companies=companies,
            sub_questions=sub_questions,
            tasks=tasks,
            synthesis_instruction=clean_text(data.get("synthesis_instruction")) or f"Synthesize findings for: {objective}. Cite source URLs.",
            output_format=output_format,
        )

    def _task_from_dict(self, item: dict[str, Any], index: int) -> ResearchTask:
        source_type = clean_source_type(item.get("source_type"))
        url = normalize_url(clean_text(item.get("url")), source_type)
        if url.startswith("SEARCH:"):
            source_type = "search"

        return ResearchTask(
            task_id=f"task_{index:03d}",
            query_context=clean_text(item.get("query_context")),
            url=url,
            source_type=source_type,
            priority=to_int(item.get("priority"), index),
            extraction_goal=clean_text(item.get("extraction_goal")),
            target_type=clean_target_type(item.get("target_type")),
            target_name=clean_text(item.get("target_name")) or "General Research",
            use_playwright=bool(item.get("use_playwright")) and not url.startswith("SEARCH:"),
            expected_signals=clean_list(item.get("expected_signals")),
        )

    def _safe_task(self, task: ResearchTask) -> ResearchTask:
        url = dedupe_search(task.url)
        source_type = "search" if url.startswith("SEARCH:") else task.source_type

        if self.validate_urls and valid_http_url(url) and not url_alive(url):
            url = search_from_task(task)
            source_type = "search"

        return replace(
            task,
            url=url,
            source_type=source_type,
            use_playwright=task.use_playwright and not url.startswith("SEARCH:"),
        )

    def _fallback_plan(self, objective: str) -> ResearchPlan:
        tasks = [
            ResearchTask(
                task_id="task_001",
                query_context=f"Find authoritative sources for {objective}",
                url=f"SEARCH:{objective} authoritative sources",
                source_type="search",
                priority=1,
                extraction_goal="Find source URLs, titles, snippets, and credibility signals.",
                expected_signals=["source URLs", "titles", "snippets"],
            ),
            ResearchTask(
                task_id="task_002",
                query_context=f"Find evidence and examples for {objective}",
                url=f"SEARCH:{objective} evidence examples",
                source_type="search",
                priority=2,
                extraction_goal="Find evidence, examples, metrics, and practical context.",
                expected_signals=["evidence", "examples", "metrics"],
            ),
        ]
        return ResearchPlan(
            objective=objective,
            research_mode="knowledge_research",
            companies=[],
            sub_questions=[objective],
            tasks=tasks,
            synthesis_instruction=f"Synthesize findings for: {objective}. Cite source URLs.",
            output_format="summary",
        )


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [clean_text(item) for item in value if clean_text(item)]


def clean_source_type(value: Any) -> str:
    source_type = clean_text(value).lower().replace("-", "_").replace(" ", "_")
    source_type = SOURCE_ALIASES.get(source_type, source_type)
    return source_type if source_type in SOURCE_TYPES else "webpage"


def clean_target_type(value: Any) -> str:
    target_type = clean_text(value).lower()
    return target_type if target_type in {"company", "discovery"} else "discovery"


def clean_mode(value: Any) -> str:
    mode = clean_text(value)
    return mode if mode in RESEARCH_MODES else "knowledge_research"


def clean_output_format(value: Any) -> str:
    output_format = clean_text(value)
    return output_format if output_format in OUTPUT_FORMATS else "summary"


def to_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def normalize_url(url: str, source_type: str) -> str:
    if not url:
        return "SEARCH:research sources"
    if url.startswith("SEARCH:"):
        return f"SEARCH:{dedupe_words(url.removeprefix('SEARCH:')) or 'research sources'}"

    query = query_from_search_url(url)
    if query:
        return f"SEARCH:{dedupe_words(query)}"
    if valid_http_url(url):
        return url
    if url.startswith("www.") and valid_http_url(f"https://{url}"):
        return f"https://{url}"
    if source_type == "search" or " " in url or "." not in url:
        return f"SEARCH:{dedupe_words(url)}"
    return f"SEARCH:{url}"


def query_from_search_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if not any(domain in host for domain in ("google.", "bing.com", "duckduckgo.com")):
        return ""
    return parse_qs(parsed.query).get("q", [""])[0].strip()


def dedupe_search(url: str) -> str:
    if not url.startswith("SEARCH:"):
        return url
    return f"SEARCH:{dedupe_words(url.removeprefix('SEARCH:')) or 'research sources'}"


def dedupe_words(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9+.#-]+", text)
    seen = set()
    result = []
    for word in words:
        key = word.lower()
        if key in seen:
            continue
        result.append(word)
        seen.add(key)
    return " ".join(result).strip()


def search_from_task(task: ResearchTask) -> str:
    parts = [task.target_name if task.target_name != "General Research" else "", task.query_context, task.extraction_goal]
    return f"SEARCH:{dedupe_words(' '.join(parts)) or 'research sources'}"


def valid_task_url(url: str) -> bool:
    return (url.startswith("SEARCH:") and bool(url.removeprefix("SEARCH:").strip())) or valid_http_url(url)


def valid_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def url_alive(url: str) -> bool:
    try:
        response = httpx.head(url, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        return response.status_code not in {404, 410}
    except httpx.HTTPError:
        return True


def dedupe_and_renumber(tasks: list[ResearchTask]) -> list[ResearchTask]:
    chosen: dict[tuple[str, str, str], ResearchTask] = {}
    for task in tasks:
        key = (task.target_name.lower(), task.source_type, task.url.lower())
        current = chosen.get(key)
        if current is None or task.priority < current.priority:
            chosen[key] = task

    ordered = sorted(chosen.values(), key=lambda task: (task.priority, task.target_name.lower(), task.query_context.lower()))
    return [replace(task, task_id=f"task_{index:03d}") for index, task in enumerate(ordered, 1)]


def validate_plan(tasks: list[ResearchTask], mode: str, companies: list[str], sub_questions: list[str]) -> None:
    if mode == "competitor_intel" and not companies:
        raise ValueError("competitor_intel mode requires companies")
    if not sub_questions:
        raise ValueError("planner returned no sub_questions")
    if len(tasks) < 2:
        raise ValueError("planner returned fewer than two tasks")

    for task in tasks:
        if not task.query_context or not task.extraction_goal:
            raise ValueError(f"{task.task_id} is missing context or extraction goal")
        if task.source_type not in SOURCE_TYPES:
            raise ValueError(f"{task.task_id} has invalid source_type {task.source_type!r}")
        if not valid_task_url(task.url):
            raise ValueError(f"{task.task_id} has invalid url {task.url!r}")
