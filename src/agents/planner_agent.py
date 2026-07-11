"""
Planner Agent
=============

This is a junior-friendly planner for the multi-agent research system.

What it does:
1. Takes one research objective from the user.
2. Asks Groq to break it into sub-questions and source tasks.
3. Fixes common LLM mistakes before the next agent runs.
4. Falls back to simple rule-based planning when Groq is unavailable.

Task URLs can be one of two forms:
- "https://..." for a page that can be scraped.
- "SEARCH:..." for a query that the search agent should resolve.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import httpx


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchTask:
    """One small job for a search or scraper agent."""

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
    """Final plan returned by the planner."""

    objective: str
    research_mode: str
    sub_questions: list[str]
    tasks: list[ResearchTask]
    synthesis_instruction: str
    output_format: str
    companies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the plan into JSON-serializable data."""

        return {
            "objective": self.objective,
            "research_mode": self.research_mode,
            "companies": self.companies,
            "sub_questions": self.sub_questions,
            "tasks": [asdict(task) for task in self.tasks],
            "synthesis_instruction": self.synthesis_instruction,
            "output_format": self.output_format,
        }


# ---------------------------------------------------------------------------
# Simple configuration
# ---------------------------------------------------------------------------


SOURCE_TYPES = {
    "webpage": {
        "goal": "Extract key facts, claims, links, and evidence from the page.",
        "signals": ["page title", "key claims", "source links"],
    },
    "search": {
        "goal": "Find useful source URLs, result titles, and snippets.",
        "signals": ["result titles", "candidate URLs", "snippets"],
    },
    "wikipedia": {
        "goal": "Extract definitions, background, and references.",
        "signals": ["definitions", "background", "references"],
    },
    "arxiv": {
        "goal": "Extract paper title, authors, abstract, and method claims.",
        "signals": ["paper title", "authors", "abstract", "method claims"],
    },
    "academic": {
        "goal": "Extract paper names, claims, methods, and publication context.",
        "signals": ["papers", "authors", "research claims"],
    },
    "technical_overview": {
        "goal": "Extract concepts, architecture details, tradeoffs, and use cases.",
        "signals": ["concepts", "architecture", "strengths", "limitations"],
    },
    "benchmarks": {
        "goal": "Extract datasets, metrics, results, and performance tradeoffs.",
        "signals": ["metrics", "datasets", "results", "latency"],
    },
    "implementation": {
        "goal": "Extract code examples, framework details, and practical pitfalls.",
        "signals": ["code examples", "frameworks", "pitfalls"],
    },
    "news": {
        "goal": "Extract announcements, dates, article titles, and summaries.",
        "signals": ["announcements", "dates", "article URLs"],
    },
    "pricing": {
        "goal": "Extract model names, token costs, tiers, limits, and enterprise notes.",
        "signals": ["model prices", "token units", "tiers", "limits"],
    },
    "docs": {
        "goal": "Extract official API names, examples, limits, and setup guidance.",
        "signals": ["API names", "examples", "limits"],
    },
    "careers": {
        "goal": "Extract roles, teams, locations, salary hints, benefits, and training.",
        "signals": ["roles", "teams", "locations", "skills"],
    },
    "reviews": {
        "goal": "Extract ratings, praise, complaints, and feature requests.",
        "signals": ["ratings", "complaints", "praise"],
    },
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

VALID_RESEARCH_MODES = {
    "competitor_intel",
    "knowledge_research",
    "technical_deep_dive",
    "market_research",
}

VALID_OUTPUT_FORMATS = {"comparison_table", "deep_dive", "summary", "report"}


KNOWN_COMPANIES = {
    "openai": {
        "name": "OpenAI",
        "aliases": ["openai", "chatgpt", "gpt", "sora"],
        "domains": ["openai.com", "platform.openai.com", "developers.openai.com"],
        "sources": {
            "news": "https://openai.com/news/",
            "pricing": "https://platform.openai.com/docs/pricing",
            "docs": "https://platform.openai.com/docs",
        },
    },
    "anthropic": {
        "name": "Anthropic",
        "aliases": ["anthropic", "claude"],
        "domains": ["anthropic.com", "claude.com", "docs.anthropic.com", "platform.claude.com"],
        "sources": {
            "news": "https://www.anthropic.com/news",
            "pricing": "https://platform.claude.com/docs/en/about-claude/pricing",
            "docs": "https://docs.anthropic.com",
        },
    },
    "groq": {
        "name": "Groq",
        "aliases": ["groq", "groqcloud"],
        "domains": ["groq.com", "console.groq.com"],
        "sources": {
            "news": "https://groq.com/news/",
            "pricing": "https://groq.com/pricing",
            "docs": "https://console.groq.com/docs",
        },
    },
    "google": {
        "name": "Google",
        "aliases": ["google", "gemini", "vertex ai", "google ai"],
        "domains": ["google.com", "google.dev", "ai.google.dev", "cloud.google.com"],
        "sources": {
            "news": "https://blog.google/technology/ai/",
            "pricing": "https://ai.google.dev/gemini-api/docs/pricing",
            "docs": "https://ai.google.dev/gemini-api/docs",
        },
    },
}


KNOWN_TOPIC_SOURCES = {
    "attention": [
        {
            "query_context": "What did the Transformer paper establish about self-attention?",
            "url": "https://arxiv.org/abs/1706.03762",
            "source_type": "arxiv",
            "target_name": "Attention Mechanism",
        },
        {
            "query_context": "How did Bahdanau attention introduce neural machine translation alignment?",
            "url": "https://arxiv.org/abs/1409.0473",
            "source_type": "arxiv",
            "target_name": "Bahdanau Attention",
        },
        {
            "query_context": "How does Lilian Weng explain attention and transformer mechanisms?",
            "url": "https://lilianweng.github.io/posts/2018-06-24-attention/",
            "source_type": "webpage",
            "target_name": "Attention Mechanism",
        },
    ],
    "transformer": [
        {
            "query_context": "What did the original Transformer paper introduce?",
            "url": "https://arxiv.org/abs/1706.03762",
            "source_type": "arxiv",
            "target_name": "Transformer",
        }
    ],
    "lstm": [
        {
            "query_context": "How do LSTMs work and why were they introduced?",
            "url": "https://colah.github.io/posts/2015-08-Understanding-LSTMs",
            "source_type": "webpage",
            "target_name": "LSTM",
        }
    ],
    "rnn": [
        {
            "query_context": "What is a recurrent neural network?",
            "url": "https://en.wikipedia.org/wiki/Recurrent_neural_network",
            "source_type": "wikipedia",
            "target_name": "RNN",
        }
    ],
}

TRUSTED_HOSTS = {
    "arxiv.org",
    "colah.github.io",
    "docs.python.org",
    "github.com",
    "huggingface.co",
    "lilianweng.github.io",
    "pytorch.org",
    "tensorflow.org",
    "wikipedia.org",
}

UNTRUSTED_HOSTS = {"medium.com", "towardsdatascience.com"}


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class PlannerAgent:
    """Create a research plan from one objective."""

    SYSTEM_PROMPT = f"""You create plans for a multi-agent web research system.

Return ONLY valid JSON with this shape:
{{
  "research_mode": "competitor_intel|knowledge_research|technical_deep_dive|market_research",
  "companies": ["company names if any"],
  "sub_questions": ["answerable research question"],
  "tasks": [{{
    "query_context": "which sub-question this task answers",
    "url": "https://real-url.com OR SEARCH:search query",
    "source_type": "one of {', '.join(SOURCE_TYPES)}",
    "priority": 1,
    "extraction_goal": "what to extract",
    "target_type": "company|discovery",
    "target_name": "company name or General Research",
    "use_playwright": false,
    "expected_signals": ["signals to look for"]
  }}],
  "synthesis_instruction": "specific comparison or summary guidance",
  "output_format": "comparison_table|deep_dive|summary|report"
}}

Rules:
- Use competitor_intel when named companies or products are compared.
- In competitor_intel, create one task for every company and every sub-question.
- Use SEARCH: when you are not sure the URL is real.
- SEARCH tasks must have use_playwright=false.
- Prefer official company URLs for pricing, docs, and news.
- For attention mechanism, include arXiv 1706.03762, arXiv 1409.0473, and Lilian Weng's attention post.
- Make synthesis_instruction specific to the objective."""

    def __init__(
        self,
        use_llm: bool = True,
        model: Optional[str] = None,
        validate_urls: Optional[bool] = None,
    ) -> None:
        """Set Groq model and URL validation behavior."""

        self.use_llm = use_llm
        self.model = model or os.environ.get("RESEARCH_PLANNER_MODEL", "llama-3.1-8b-instant")
        env_value = os.environ.get("RESEARCH_PLANNER_VALIDATE_URLS", "1").lower()
        self.validate_urls = validate_urls if validate_urls is not None else env_value not in {"0", "false", "no"}

    def plan(self, objective: str) -> ResearchPlan:
        """Create a plan with Groq, or use fallback rules if Groq fails."""

        if self.use_llm:
            try:
                return self._plan_with_groq(objective)
            except Exception as error:
                print(f"[planner_agent] Groq planner unavailable; using fallback planner: {error}")
        return self._fallback_plan(objective)

    def _plan_with_groq(self, objective: str) -> ResearchPlan:
        """Ask Groq for JSON and repair the result."""

        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set")

        try:
            from groq import Groq
        except ImportError as error:
            raise RuntimeError("groq package is not installed") from error

        client = Groq()
        messages = [{"role": "user", "content": f"Create a research plan for: {objective}"}]
        last_error: Optional[Exception] = None

        for attempt in range(1, 4):
            response = client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=2048,
                messages=[{"role": "system", "content": self.SYSTEM_PROMPT}, *messages],
            )
            raw = (response.choices[0].message.content or "").strip()
            messages.append({"role": "assistant", "content": raw})

            try:
                return self._parse_plan(raw, objective)
            except (json.JSONDecodeError, KeyError, ValueError) as error:
                last_error = error
                messages.append({"role": "user", "content": f"Fix this error and return only JSON: {error}"})
                print(f"[planner_agent] Plan parse failed on attempt {attempt}: {error}")

        raise RuntimeError(f"Groq planner failed after 3 attempts: {last_error}")

    def _parse_plan(self, raw: str, objective: str) -> ResearchPlan:
        """Convert LLM JSON into a clean ResearchPlan."""

        data = json.loads(strip_json_fence(raw))
        companies = clean_strings(data.get("companies", [])) or detect_companies(objective)
        sub_questions = clean_strings(data.get("sub_questions", [])) or default_questions(objective)
        mode = fix_mode(data.get("research_mode", ""), objective, companies)

        tasks = [
            self._task_from_dict(item, index)
            for index, item in enumerate(data.get("tasks", []), 1)
            if isinstance(item, dict)
        ]
        if not tasks:
            tasks = fallback_tasks(objective, companies)

        tasks = self._add_known_topic_sources(objective, tasks)
        tasks = self._repair_tasks(objective, tasks)
        tasks = self._add_missing_company_tasks(mode, companies, sub_questions, tasks)
        tasks = self._repair_tasks(objective, tasks)
        self._validate_tasks(tasks, mode, companies, sub_questions)

        return ResearchPlan(
            objective=objective,
            research_mode=mode,
            companies=companies,
            sub_questions=sub_questions,
            tasks=sorted(tasks, key=lambda task: task.priority),
            synthesis_instruction=fix_synthesis(data.get("synthesis_instruction", ""), objective),
            output_format=fix_output_format(data.get("output_format", ""), objective, mode),
        )

    def _task_from_dict(self, data: dict[str, Any], index: int) -> ResearchTask:
        """Build one task from LLM JSON."""

        source_type = clean_source_type(data.get("source_type", "webpage"))
        return ResearchTask(
            task_id=f"task_{index:03d}",
            query_context=str(data.get("query_context", "")).strip(),
            url=normalize_url(str(data.get("url", "")).strip(), source_type),
            source_type=source_type,
            priority=to_int(data.get("priority", index), index),
            extraction_goal=str(data.get("extraction_goal", "")).strip(),
            target_type=clean_target_type(data.get("target_type", "discovery")),
            target_name=str(data.get("target_name", "General Research")).strip() or "General Research",
            use_playwright=bool(data.get("use_playwright", False)),
            expected_signals=clean_strings(data.get("expected_signals", [])),
        )

    def _repair_tasks(self, objective: str, tasks: list[ResearchTask]) -> list[ResearchTask]:
        """Fix task IDs, source types, URLs, and Playwright flags."""

        fixed_tasks = []
        for index, task in enumerate(tasks, 1):
            source_type = infer_source_type(objective, task)
            url = normalize_url(task.url, source_type)
            url = dedupe_search(url)
            url = self._replace_with_known_company_url(url, source_type, task.target_name)
            url = self._make_url_safe(url, source_type, task)

            if url.startswith("SEARCH:"):
                source_type = "search"

            fixed_tasks.append(
                replace(
                    task,
                    task_id=f"task_{index:03d}",
                    source_type=source_type,
                    url=url,
                    target_type=clean_target_type(task.target_type),
                    use_playwright=source_type in {"pricing", "careers", "reviews"} and not url.startswith("SEARCH:"),
                    expected_signals=task.expected_signals or source_info(source_type)["signals"],
                    extraction_goal=task.extraction_goal or source_info(source_type)["goal"],
                )
            )
        return fixed_tasks

    def _add_known_topic_sources(self, objective: str, tasks: list[ResearchTask]) -> list[ResearchTask]:
        """Add trusted sources for common technical topics."""

        existing_urls = {task.url for task in tasks}
        new_tasks = list(tasks)

        for keyword, sources in KNOWN_TOPIC_SOURCES.items():
            if keyword not in objective.lower():
                continue

            for source in sources:
                if source["url"] in existing_urls:
                    continue
                source_type = clean_source_type(source["source_type"])
                new_tasks.append(
                    ResearchTask(
                        task_id="",
                        query_context=source["query_context"],
                        url=source["url"],
                        source_type=source_type,
                        priority=1,
                        extraction_goal=source_info(source_type)["goal"],
                        target_type="discovery",
                        target_name=source["target_name"],
                        expected_signals=source_info(source_type)["signals"],
                    )
                )
                existing_urls.add(source["url"])

        return new_tasks

    def _add_missing_company_tasks(
        self,
        mode: str,
        companies: list[str],
        sub_questions: list[str],
        tasks: list[ResearchTask],
    ) -> list[ResearchTask]:
        """Make competitor plans symmetric across companies and questions."""

        if mode != "competitor_intel" or not companies or not sub_questions:
            return tasks

        fixed_tasks = list(tasks)
        for company in companies:
            for question in sub_questions:
                if any(task_matches_company_question(task, company, question) for task in fixed_tasks):
                    continue

                source_type = source_type_for_question(question)
                fixed_tasks.append(
                    ResearchTask(
                        task_id="",
                        query_context=f"{company}: {question}",
                        url=self._known_company_url(company, source_type) or f"SEARCH:{company} {question}",
                        source_type=source_type,
                        priority=2,
                        extraction_goal=goal_for(source_type, question),
                        target_type="company",
                        target_name=company,
                        use_playwright=source_type in {"pricing", "careers", "reviews"},
                        expected_signals=source_info(source_type)["signals"],
                    )
                )

        return fixed_tasks

    def _replace_with_known_company_url(self, url: str, source_type: str, target_name: str) -> str:
        """Use maintained official URLs for known companies."""

        if url.startswith("SEARCH:") or source_type not in {"pricing", "docs", "news"}:
            return url

        if is_known_company_url(url, target_name):
            return self._known_company_url(target_name, source_type) or url

        return url

    def _make_url_safe(self, url: str, source_type: str, task: ResearchTask) -> str:
        """Convert risky or dead direct URLs into SEARCH tasks."""

        if url.startswith("SEARCH:"):
            return url

        if source_type in {"pricing", "careers", "reviews"} and not is_known_company_url(url, task.target_name):
            return search_query_for(task, source_type)

        if is_untrusted_url(url, source_type):
            return search_query_for(task, source_type)

        if self.validate_urls and not url_is_alive(url):
            return self._known_company_url(task.target_name, source_type) or search_query_for(task, source_type)

        return url

    def _known_company_url(self, company_name: str, source_type: str) -> str:
        """Return a maintained company source URL if we have one."""

        company = find_known_company(company_name)
        if not company:
            return ""
        return company["sources"].get(source_type, "")

    def _validate_tasks(
        self,
        tasks: list[ResearchTask],
        mode: str,
        companies: list[str],
        sub_questions: list[str],
    ) -> None:
        """Raise clear errors if the plan is still broken after repair."""

        if len(tasks) < 2:
            raise ValueError("planner returned fewer than two tasks")

        for task in tasks:
            if not task.query_context or not task.url or not task.extraction_goal:
                raise ValueError(f"{task.task_id} is missing required fields")
            if task.source_type not in SOURCE_TYPES:
                raise ValueError(f"{task.task_id} has unsupported source_type {task.source_type!r}")
            if not valid_task_url(task.url):
                raise ValueError(f"{task.task_id} has invalid url {task.url!r}")

        if mode == "competitor_intel" and companies and sub_questions:
            for company in companies:
                count = sum(1 for task in tasks if task.target_name.lower() == company.lower())
                if count < len(sub_questions):
                    raise ValueError(f"{company} has only {count}/{len(sub_questions)} required tasks")

    def _fallback_plan(self, objective: str) -> ResearchPlan:
        """Simple rule-based plan for local development."""

        companies = detect_companies(objective)
        mode = "competitor_intel" if companies else infer_mode(objective)
        tasks = fallback_tasks(objective, companies)
        tasks = self._add_known_topic_sources(objective, tasks)
        tasks = self._repair_tasks(objective, tasks)
        sub_questions = default_questions(objective)
        tasks = self._add_missing_company_tasks(mode, companies, sub_questions, tasks)
        tasks = self._repair_tasks(objective, tasks)

        return ResearchPlan(
            objective=objective,
            research_mode=mode,
            companies=companies,
            sub_questions=sub_questions,
            tasks=tasks,
            synthesis_instruction=fix_synthesis("", objective),
            output_format=fix_output_format("", objective, mode),
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def source_info(source_type: str) -> dict[str, Any]:
    """Return metadata for a source type."""

    return SOURCE_TYPES.get(source_type, SOURCE_TYPES["webpage"])


def clean_source_type(value: Any) -> str:
    """Normalize source type names."""

    source_type = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    source_type = SOURCE_ALIASES.get(source_type, source_type)
    return source_type if source_type in SOURCE_TYPES else "webpage"


def clean_target_type(value: Any) -> str:
    """Normalize target type names."""

    target_type = str(value).strip().lower()
    return target_type if target_type in {"company", "discovery"} else "discovery"


def clean_strings(values: Any) -> list[str]:
    """Return non-empty strings from a list-like value."""

    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def to_int(value: Any, default: int) -> int:
    """Convert a value to int safely."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_url(url: str, source_type: str) -> str:
    """Normalize direct URLs and SEARCH tasks."""

    url = url.strip()
    if not url:
        return "SEARCH:research sources"

    if url.startswith("SEARCH:"):
        query = dedupe_words(url.removeprefix("SEARCH:"))
        return f"SEARCH:{query or 'research sources'}"

    search_query = query_from_search_engine_url(url)
    if search_query:
        return f"SEARCH:{dedupe_words(search_query)}"

    if valid_http_url(url):
        return url

    if url.startswith("www.") and valid_http_url(f"https://{url}"):
        return f"https://{url}"

    if source_type == "search" or " " in url or "." not in url:
        return f"SEARCH:{dedupe_words(url)}"

    return f"SEARCH:{dedupe_words(url)}"


def query_from_search_engine_url(url: str) -> str:
    """Extract q= from Google, Bing, or DuckDuckGo URLs."""

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if not any(domain in host for domain in ["google.", "bing.com", "duckduckgo.com"]):
        return ""
    return parse_qs(parsed.query).get("q", [""])[0].strip()


def dedupe_search(url: str) -> str:
    """Remove duplicate words from SEARCH queries."""

    if not url.startswith("SEARCH:"):
        return url
    return f"SEARCH:{dedupe_words(url.removeprefix('SEARCH:'))}"


def dedupe_words(text: str) -> str:
    """Remove repeated words while keeping order."""

    words = re.findall(r"[A-Za-z0-9+.#-]+", text)
    result = []
    seen = set()

    for word in words:
        key = word.lower()
        if key in seen:
            continue
        result.append(word)
        seen.add(key)

    return " ".join(result).strip()


def infer_source_type(objective: str, task: ResearchTask) -> str:
    """Repair obvious source type mistakes."""

    if task.url.startswith("SEARCH:") or task.source_type == "search":
        return "search"

    text = " ".join([objective, task.query_context, task.extraction_goal, task.url]).lower()
    if has_any(text, ["pricing", "price", "cost", "tier", "token"]):
        return "pricing"
    if has_any(text, ["career", "job", "salary", "training", "benefit"]):
        return "careers"
    if has_any(text, ["docs", "documentation", "api"]):
        return "docs"
    if "arxiv.org" in task.url:
        return "arxiv"
    if "wikipedia.org" in task.url:
        return "wikipedia"

    return clean_source_type(task.source_type)


def source_type_for_question(question: str) -> str:
    """Choose a source type from a sub-question."""

    text = question.lower()
    if has_any(text, ["pricing", "price", "cost", "tier", "token"]):
        return "pricing"
    if has_any(text, ["career", "job", "salary", "training", "benefit"]):
        return "careers"
    if has_any(text, ["docs", "api", "developer"]):
        return "docs"
    if has_any(text, ["review", "feedback", "complaint"]):
        return "reviews"
    if has_any(text, ["news", "launch", "release", "announcement"]):
        return "news"
    return "search"


def search_query_for(task: ResearchTask, source_type: str) -> str:
    """Build a focused SEARCH query from a task."""

    target = "" if task.target_name == "General Research" else task.target_name
    if source_type == "pricing" and target:
        return f"SEARCH:{target} official model pricing"
    if source_type == "careers" and target:
        return f"SEARCH:{target} official careers jobs salary benefits training"
    if source_type == "reviews" and target:
        return f"SEARCH:{target} reviews customer feedback"
    return f"SEARCH:{dedupe_words(' '.join([target, task.query_context, task.extraction_goal]))}"


def valid_task_url(url: str) -> bool:
    """Check whether a task URL can be executed."""

    return (url.startswith("SEARCH:") and bool(url.removeprefix("SEARCH:").strip())) or valid_http_url(url)


def valid_http_url(url: str) -> bool:
    """Check for an http or https URL with a host."""

    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def url_is_alive(url: str) -> bool:
    """Return False only when the URL is confirmed as gone."""

    try:
        response = httpx.head(
            url,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5,
        )
        return response.status_code not in {404, 410}
    except httpx.HTTPError:
        return True


def is_untrusted_url(url: str, source_type: str) -> bool:
    """Avoid scraping weak article URLs directly."""

    if source_type not in {"webpage", "technical_overview", "academic", "benchmarks", "implementation", "news"}:
        return False

    host = urlparse(url).netloc.lower()
    if any(host == bad or host.endswith(f".{bad}") for bad in UNTRUSTED_HOSTS):
        return True
    return not any(host == good or host.endswith(f".{good}") for good in TRUSTED_HOSTS)


def find_known_company(company_name: str) -> Optional[dict[str, Any]]:
    """Find company metadata by display name or alias."""

    company_name = company_name.strip().lower()
    for company in KNOWN_COMPANIES.values():
        names = [company["name"].lower(), *company["aliases"]]
        if company_name in names:
            return company
    return None


def is_known_company_url(url: str, company_name: str) -> bool:
    """Check if a URL belongs to a known company."""

    company = find_known_company(company_name)
    if not company:
        return False

    host = urlparse(url).netloc.lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in company["domains"])


def detect_companies(objective: str) -> list[str]:
    """Detect known companies mentioned in the objective."""

    text = objective.lower()
    companies = []

    for company in KNOWN_COMPANIES.values():
        if any(term_match(text, alias) for alias in company["aliases"]):
            companies.append(company["name"])

    return companies


def default_questions(objective: str) -> list[str]:
    """Create simple fallback sub-questions."""

    if has_any(objective.lower(), ["compare", "vs", "versus", "difference"]):
        return [
            "What are the main features or claims?",
            "What are the pricing, cost, or resource tradeoffs?",
            "What are the strengths and limitations?",
            "Which option is best for each use case?",
        ]

    return [
        f"What is the background of {objective}?",
        f"What evidence or sources explain {objective}?",
        f"What are the practical takeaways for {objective}?",
    ]


def fallback_tasks(objective: str, companies: list[str]) -> list[ResearchTask]:
    """Build tasks without an LLM."""

    if companies:
        tasks = []
        for company in companies:
            for question in default_questions(objective):
                source_type = source_type_for_question(question)
                known = find_known_company(company)
                url = known["sources"].get(source_type, "") if known else ""
                tasks.append(
                    ResearchTask(
                        task_id="",
                        query_context=f"{company}: {question}",
                        url=url or f"SEARCH:{company} {question}",
                        source_type=source_type,
                        priority=1,
                        extraction_goal=goal_for(source_type, question),
                        target_type="company",
                        target_name=company,
                        expected_signals=source_info(source_type)["signals"],
                    )
                )
        return tasks

    return [
        ResearchTask(
            task_id="",
            query_context=question,
            url=f"SEARCH:{question}",
            source_type="search",
            priority=index,
            extraction_goal=goal_for("search", question),
            expected_signals=source_info("search")["signals"],
        )
        for index, question in enumerate(default_questions(objective), 1)
    ]


def task_matches_company_question(task: ResearchTask, company: str, question: str) -> bool:
    """Check if a task already covers this exact company and question."""

    if task.target_name.lower() != company.lower():
        return False

    task_question = task.query_context.lower().removeprefix(f"{company.lower()}:").strip()
    return simplify_text(task_question) == simplify_text(question)


def simplify_text(text: str) -> str:
    """Normalize text before comparing generated questions."""

    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def fix_mode(mode: Any, objective: str, companies: list[str]) -> str:
    """Repair invalid research mode values."""

    mode = str(mode).strip()
    if companies:
        return "competitor_intel"
    if mode in VALID_RESEARCH_MODES:
        return mode
    return infer_mode(objective)


def infer_mode(objective: str) -> str:
    """Infer a mode without an LLM."""

    text = objective.lower()
    if has_any(text, ["architecture", "algorithm", "attention", "paper", "lstm", "rnn", "transformer"]):
        return "technical_deep_dive"
    if has_any(text, ["market", "industry", "landscape", "trend", "competitive"]):
        return "market_research"
    return "knowledge_research"


def fix_output_format(value: Any, objective: str, mode: str) -> str:
    """Repair invalid output format values."""

    value = str(value).strip()
    if value in VALID_OUTPUT_FORMATS:
        return value
    if has_any(objective.lower(), ["compare", "vs", "versus", "difference"]):
        return "comparison_table"
    if mode in {"competitor_intel", "market_research"}:
        return "report"
    return "summary"


def fix_synthesis(instruction: Any, objective: str) -> str:
    """Return useful synthesis instructions."""

    text = str(instruction).strip()
    if text and not is_generic_instruction(text):
        return text

    if has_any(objective.lower(), ["compare", "vs", "versus", "difference"]):
        return (
            f"Create a comparison for: {objective}. Compare features, pricing or cost, "
            "strengths, limitations, use cases, and cite source URLs for major claims."
        )

    return f"Synthesize findings for: {objective}. Cite source URLs and mention uncertainty."


def goal_for(source_type: str, question: str) -> str:
    """Return an extraction goal for a source type."""

    return source_info(source_type)["goal"] if source_type in SOURCE_TYPES else f"Find evidence for: {question}"


def is_generic_instruction(text: str) -> bool:
    """Detect vague synthesis instructions."""

    text = text.lower().strip()
    generic_phrases = [
        "combine evidence",
        "combine the evidence",
        "create a comparison table",
        "provide a comprehensive overview",
        "synthesize findings",
    ]
    return any(text == phrase or text.startswith(f"{phrase}.") for phrase in generic_phrases)


def strip_json_fence(raw: str) -> str:
    """Remove markdown fences around JSON."""

    raw = raw.strip()
    if not raw.startswith("```"):
        return raw

    lines = raw.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def term_match(text: str, keyword: str) -> bool:
    """Match words or phrases without substring mistakes."""

    pattern = r"(?<![a-z0-9])" + re.escape(keyword.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def has_any(text: str, keywords: list[str]) -> bool:
    """Return True when any keyword appears in the text."""

    return any(term_match(text, keyword) for keyword in keywords)
