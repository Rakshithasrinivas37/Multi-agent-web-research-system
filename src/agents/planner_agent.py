"""Planner agent for turning a research objective into executable tasks."""

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
    "webpage": ("Extract key facts, claims, links, and evidence.", ["page title", "key claims", "source links"]),
    "search": ("Find useful source URLs, result titles, and snippets.", ["result titles", "candidate URLs", "snippets"]),
    "wikipedia": ("Extract definitions, background, and references.", ["definitions", "background", "references"]),
    "arxiv": ("Extract paper title, authors, abstract, and method claims.", ["paper title", "authors", "abstract"]),
    "academic": ("Extract paper names, claims, methods, and publication context.", ["papers", "authors", "claims"]),
    "technical_overview": ("Extract concepts, architecture details, tradeoffs, and use cases.", ["concepts", "architecture", "limitations"]),
    "benchmarks": ("Extract datasets, metrics, results, and tradeoffs.", ["metrics", "datasets", "results"]),
    "implementation": ("Extract code examples, framework details, and pitfalls.", ["code examples", "frameworks", "pitfalls"]),
    "news": ("Extract announcements, dates, titles, and summaries.", ["announcements", "dates", "article URLs"]),
    "pricing": ("Extract model names, token costs, tiers, limits, and enterprise notes.", ["model prices", "token units", "tiers", "limits"]),
    "docs": ("Extract API names, examples, limits, and setup guidance.", ["API names", "examples", "limits"]),
    "careers": ("Extract roles, teams, locations, salary hints, benefits, and training.", ["roles", "teams", "locations", "skills"]),
    "reviews": ("Extract ratings, praise, complaints, and feature requests.", ["ratings", "complaints", "praise"]),
}

ALIASES = {
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

MODES = {"competitor_intel", "knowledge_research", "technical_deep_dive", "market_research"}
FORMATS = {"comparison_table", "deep_dive", "summary", "report"}

COMPANIES = {
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

TOPIC_SOURCES = [
    (
        ("attention",),
        [
            ("What did the Transformer paper establish about self-attention?", "https://arxiv.org/abs/1706.03762", "arxiv", "Attention Mechanism"),
            ("How did Bahdanau attention introduce neural machine translation alignment?", "https://arxiv.org/abs/1409.0473", "arxiv", "Bahdanau Attention"),
            ("How does Lilian Weng explain attention and transformer mechanisms?", "https://lilianweng.github.io/posts/2018-06-24-attention/", "webpage", "Attention Mechanism"),
        ],
    ),
    (
        ("transformer", "transformers"),
        [("What did the original Transformer paper introduce?", "https://arxiv.org/abs/1706.03762", "arxiv", "Transformer")],
    ),
    (
        ("lstm",),
        [("How do LSTMs work and why were they introduced?", "https://colah.github.io/posts/2015-08-Understanding-LSTMs", "webpage", "LSTM")],
    ),
    (
        ("rnn",),
        [("What is a recurrent neural network?", "https://en.wikipedia.org/wiki/Recurrent_neural_network", "wikipedia", "RNN")],
    ),
]

TRUSTED_HOSTS = {"arxiv.org", "colah.github.io", "docs.python.org", "github.com", "huggingface.co", "lilianweng.github.io", "pytorch.org", "tensorflow.org", "wikipedia.org"}
UNTRUSTED_HOSTS = {"medium.com", "towardsdatascience.com"}


class PlannerAgent:
    PROMPT = f"""Create a research plan as JSON only.

Schema:
{{
  "research_mode": "competitor_intel|knowledge_research|technical_deep_dive|market_research",
  "companies": ["company names if any"],
  "sub_questions": ["answerable research question"],
  "tasks": [{{
    "query_context": "which question this answers",
    "url": "https://real-url.com OR SEARCH:search query",
    "source_type": "one of {', '.join(SOURCE_TYPES)}",
    "priority": 1,
    "extraction_goal": "what to extract",
    "target_type": "company|discovery",
    "target_name": "company name or General Research",
    "use_playwright": false,
    "expected_signals": ["signals to look for"]
  }}],
  "synthesis_instruction": "specific guidance for the final answer",
  "output_format": "comparison_table|deep_dive|summary|report"
}}

Rules:
- Use competitor_intel for company/product comparisons.
- In competitor_intel, every company must have one task per sub-question.
- Use direct official URLs only when confident; otherwise use SEARCH:.
- SEARCH tasks must set use_playwright=false.
- Prefer official URLs for pricing, docs, and news.
- Keep the JSON compact: 3 to 6 sub_questions, short strings, and no markdown."""

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
        user_request = f"Create a compact research plan for: {objective}"
        messages = [{"role": "user", "content": user_request}]
        last_error: Optional[Exception] = None

        for attempt in range(1, 3):
            response = client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=1400,
                messages=[{"role": "system", "content": self.PROMPT}, *messages],
            )
            raw = (response.choices[0].message.content or "").strip()

            try:
                return self._parse_plan(raw, objective)
            except (json.JSONDecodeError, KeyError, ValueError) as error:
                last_error = error
                messages = [
                    {"role": "user", "content": user_request},
                    {"role": "user", "content": f"The previous response was invalid JSON: {error}. Return shorter valid JSON only."},
                ]
                print(f"[planner_agent] Plan parse failed on attempt {attempt}: {error}")

        raise RuntimeError(f"Groq planner failed after 2 attempts: {last_error}")

    def _parse_plan(self, raw: str, objective: str) -> ResearchPlan:
        data = json.loads(strip_fence(raw))
        companies = string_list(data.get("companies")) or detect_companies(objective)
        questions = string_list(data.get("sub_questions")) or default_questions(objective)
        mode = fix_mode(data.get("research_mode"), objective, companies)
        tasks = [self._task(item, i) for i, item in enumerate(data.get("tasks", []), 1) if isinstance(item, dict)]

        tasks = tasks or fallback_tasks(objective, companies)
        tasks = self._add_topic_sources(objective, tasks)
        tasks = self._repair_tasks(objective, tasks)
        tasks = self._fill_company_coverage(mode, companies, questions, tasks)
        tasks = self._repair_tasks(objective, tasks)
        self._validate_tasks(tasks, mode, companies, questions)

        return ResearchPlan(
            objective=objective,
            research_mode=mode,
            companies=companies,
            sub_questions=questions,
            tasks=sorted(tasks, key=lambda task: task.priority),
            synthesis_instruction=fix_synthesis(data.get("synthesis_instruction"), objective),
            output_format=fix_format(data.get("output_format"), objective, mode),
        )

    def _task(self, item: dict[str, Any], index: int) -> ResearchTask:
        source_type = clean_source(item.get("source_type", "webpage"))
        return ResearchTask(
            task_id=f"task_{index:03d}",
            query_context=str(item.get("query_context", "")).strip(),
            url=normalize_url(str(item.get("url", "")).strip(), source_type),
            source_type=source_type,
            priority=to_int(item.get("priority"), index),
            extraction_goal=str(item.get("extraction_goal", "")).strip(),
            target_type=clean_target(item.get("target_type")),
            target_name=str(item.get("target_name", "General Research")).strip() or "General Research",
            use_playwright=bool(item.get("use_playwright", False)),
            expected_signals=string_list(item.get("expected_signals")),
        )

    def _repair_tasks(self, objective: str, tasks: list[ResearchTask]) -> list[ResearchTask]:
        fixed = []
        for index, task in enumerate(tasks, 1):
            source_type = infer_source(objective, task)
            url = dedupe_search(normalize_url(task.url, source_type))
            url = self._canonical_url(url, source_type, task.target_name)
            url = self._safe_url(url, source_type, task)
            source_type = "search" if url.startswith("SEARCH:") else source_type

            fixed.append(
                replace(
                    task,
                    task_id=f"task_{index:03d}",
                    source_type=source_type,
                    url=url,
                    target_type=clean_target(task.target_type),
                    use_playwright=source_type in {"pricing", "careers", "reviews"} and not url.startswith("SEARCH:"),
                    extraction_goal=task.extraction_goal or goal(source_type),
                    expected_signals=task.expected_signals or signals(source_type),
                )
            )
        return fixed

    def _add_topic_sources(self, objective: str, tasks: list[ResearchTask]) -> list[ResearchTask]:
        text = objective.lower()
        seen = {task.url for task in tasks}
        result = list(tasks)

        for keywords, sources in TOPIC_SOURCES:
            if not any(keyword in text for keyword in keywords):
                continue
            for question, url, source_type, target in sources:
                if url in seen:
                    continue
                result.append(
                    ResearchTask(
                        task_id="",
                        query_context=question,
                        url=url,
                        source_type=source_type,
                        priority=1,
                        extraction_goal=goal(source_type),
                        target_name=target,
                        expected_signals=signals(source_type),
                    )
                )
                seen.add(url)

        return result

    def _fill_company_coverage(
        self,
        mode: str,
        companies: list[str],
        questions: list[str],
        tasks: list[ResearchTask],
    ) -> list[ResearchTask]:
        if mode != "competitor_intel" or not companies or not questions:
            return tasks

        result = list(tasks)
        for company in companies:
            for question in questions:
                if any(covers(task, company, question) for task in result):
                    continue
                source_type = source_for_question(question)
                result.append(
                    ResearchTask(
                        task_id="",
                        query_context=f"{company}: {question}",
                        url=self._company_url(company, source_type) or f"SEARCH:{company} {question}",
                        source_type=source_type,
                        priority=2,
                        extraction_goal=goal(source_type),
                        target_type="company",
                        target_name=company,
                        use_playwright=source_type in {"pricing", "careers", "reviews"},
                        expected_signals=signals(source_type),
                    )
                )
        return result

    def _canonical_url(self, url: str, source_type: str, company: str) -> str:
        if url.startswith("SEARCH:") or source_type not in {"pricing", "docs", "news"}:
            return url
        return self._company_url(company, source_type) if is_company_url(url, company) else url

    def _safe_url(self, url: str, source_type: str, task: ResearchTask) -> str:
        if url.startswith("SEARCH:"):
            return url
        if source_type in {"pricing", "careers", "reviews"} and not is_company_url(url, task.target_name):
            return search_for(task, source_type)
        if is_untrusted(url, source_type):
            return search_for(task, source_type)
        if self.validate_urls and not url_alive(url):
            return self._company_url(task.target_name, source_type) or search_for(task, source_type)
        return url

    def _company_url(self, company_name: str, source_type: str) -> str:
        company = find_company(company_name)
        return company["sources"].get(source_type, "") if company else ""

    def _validate_tasks(self, tasks: list[ResearchTask], mode: str, companies: list[str], questions: list[str]) -> None:
        if len(tasks) < 2:
            raise ValueError("planner returned fewer than two tasks")
        for task in tasks:
            if not task.query_context or not task.url or not task.extraction_goal:
                raise ValueError(f"{task.task_id} is missing required fields")
            if task.source_type not in SOURCE_TYPES:
                raise ValueError(f"{task.task_id} has unsupported source_type {task.source_type!r}")
            if not valid_task_url(task.url):
                raise ValueError(f"{task.task_id} has invalid url {task.url!r}")
        if mode == "competitor_intel":
            for company in companies:
                count = sum(1 for task in tasks if task.target_name.lower() == company.lower())
                if count < len(questions):
                    raise ValueError(f"{company} has only {count}/{len(questions)} required tasks")

    def _fallback_plan(self, objective: str) -> ResearchPlan:
        companies = detect_companies(objective)
        mode = "competitor_intel" if companies else infer_mode(objective)
        questions = default_questions(objective)
        tasks = fallback_tasks(objective, companies)
        tasks = self._add_topic_sources(objective, tasks)
        tasks = self._repair_tasks(objective, tasks)
        tasks = self._fill_company_coverage(mode, companies, questions, tasks)
        tasks = self._repair_tasks(objective, tasks)

        return ResearchPlan(
            objective=objective,
            research_mode=mode,
            companies=companies,
            sub_questions=questions,
            tasks=tasks,
            synthesis_instruction=fix_synthesis("", objective),
            output_format=fix_format("", objective, mode),
        )


def goal(source_type: str) -> str:
    return SOURCE_TYPES.get(source_type, SOURCE_TYPES["webpage"])[0]


def signals(source_type: str) -> list[str]:
    return list(SOURCE_TYPES.get(source_type, SOURCE_TYPES["webpage"])[1])


def clean_source(value: Any) -> str:
    value = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    value = ALIASES.get(value, value)
    return value if value in SOURCE_TYPES else "webpage"


def clean_target(value: Any) -> str:
    value = str(value or "discovery").strip().lower()
    return value if value in {"company", "discovery"} else "discovery"


def string_list(value: Any) -> list[str]:
    return [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else []


def to_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def normalize_url(url: str, source_type: str) -> str:
    url = url.strip()
    if not url:
        return "SEARCH:research sources"
    if url.startswith("SEARCH:"):
        return f"SEARCH:{dedupe_words(url.removeprefix('SEARCH:')) or 'research sources'}"
    query = search_engine_query(url)
    if query:
        return f"SEARCH:{dedupe_words(query)}"
    if valid_http(url):
        return url
    if url.startswith("www.") and valid_http(f"https://{url}"):
        return f"https://{url}"
    return f"SEARCH:{dedupe_words(url)}" if source_type == "search" or " " in url or "." not in url else f"SEARCH:{url}"


def search_engine_query(url: str) -> str:
    parsed = urlparse(url)
    if not any(domain in parsed.netloc.lower() for domain in ("google.", "bing.com", "duckduckgo.com")):
        return ""
    return parse_qs(parsed.query).get("q", [""])[0].strip()


def dedupe_search(url: str) -> str:
    return f"SEARCH:{dedupe_words(url.removeprefix('SEARCH:'))}" if url.startswith("SEARCH:") else url


def dedupe_words(text: str) -> str:
    result, seen = [], set()
    for word in re.findall(r"[A-Za-z0-9+.#-]+", text):
        key = word.lower()
        if key not in seen:
            result.append(word)
            seen.add(key)
    return " ".join(result).strip()


def infer_source(objective: str, task: ResearchTask) -> str:
    if task.url.startswith("SEARCH:") or task.source_type == "search":
        return "search"
    text = " ".join([objective, task.query_context, task.extraction_goal, task.url]).lower()
    checks = [
        ("pricing", ["pricing", "price", "cost", "tier", "token"]),
        ("careers", ["career", "job", "salary", "training", "benefit"]),
        ("docs", ["docs", "documentation", "api"]),
    ]
    for source_type, words in checks:
        if has_any(text, words):
            return source_type
    if "arxiv.org" in task.url:
        return "arxiv"
    if "wikipedia.org" in task.url:
        return "wikipedia"
    return clean_source(task.source_type)


def source_for_question(question: str) -> str:
    text = question.lower()
    checks = [
        ("pricing", ["pricing", "price", "cost", "tier", "token"]),
        ("careers", ["career", "job", "salary", "training", "benefit"]),
        ("docs", ["docs", "api", "developer"]),
        ("reviews", ["review", "feedback", "complaint"]),
        ("news", ["news", "launch", "release", "announcement"]),
    ]
    return next((source_type for source_type, words in checks if has_any(text, words)), "search")


def search_for(task: ResearchTask, source_type: str) -> str:
    target = "" if task.target_name == "General Research" else task.target_name
    queries = {
        "pricing": f"{target} official model pricing",
        "careers": f"{target} official careers jobs salary benefits training",
        "reviews": f"{target} reviews customer feedback",
    }
    return f"SEARCH:{dedupe_words(queries.get(source_type, ' '.join([target, task.query_context, task.extraction_goal])))}"


def valid_task_url(url: str) -> bool:
    return (url.startswith("SEARCH:") and bool(url.removeprefix("SEARCH:").strip())) or valid_http(url)


def valid_http(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def url_alive(url: str) -> bool:
    try:
        response = httpx.head(url, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        return response.status_code not in {404, 410}
    except httpx.HTTPError:
        return True


def is_untrusted(url: str, source_type: str) -> bool:
    if source_type not in {"webpage", "technical_overview", "academic", "benchmarks", "implementation", "news"}:
        return False
    host = urlparse(url).netloc.lower()
    if any(host == bad or host.endswith(f".{bad}") for bad in UNTRUSTED_HOSTS):
        return True
    return not any(host == trusted or host.endswith(f".{trusted}") for trusted in TRUSTED_HOSTS)


def find_company(name: str) -> Optional[dict[str, Any]]:
    name = name.strip().lower()
    return next((company for company in COMPANIES.values() if name in [company["name"].lower(), *company["aliases"]]), None)


def is_company_url(url: str, company_name: str) -> bool:
    company = find_company(company_name)
    host = urlparse(url).netloc.lower()
    return bool(company and any(host == domain or host.endswith(f".{domain}") for domain in company["domains"]))


def detect_companies(objective: str) -> list[str]:
    text = objective.lower()
    return [company["name"] for company in COMPANIES.values() if any(term_match(text, alias) for alias in company["aliases"])]


def default_questions(objective: str) -> list[str]:
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
    questions = default_questions(objective)
    if not companies:
        return [
            ResearchTask("", question, f"SEARCH:{question}", "search", index, goal("search"), expected_signals=signals("search"))
            for index, question in enumerate(questions, 1)
        ]

    tasks = []
    for company in companies:
        known = find_company(company)
        for question in questions:
            source_type = source_for_question(question)
            url = known["sources"].get(source_type, "") if known else ""
            tasks.append(
                ResearchTask(
                    task_id="",
                    query_context=f"{company}: {question}",
                    url=url or f"SEARCH:{company} {question}",
                    source_type=source_type,
                    priority=1,
                    extraction_goal=goal(source_type),
                    target_type="company",
                    target_name=company,
                    expected_signals=signals(source_type),
                )
            )
    return tasks


def covers(task: ResearchTask, company: str, question: str) -> bool:
    if task.target_name.lower() != company.lower():
        return False
    task_question = task.query_context.lower().removeprefix(f"{company.lower()}:").strip()
    return simplify(task_question) == simplify(question)


def simplify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def fix_mode(mode: Any, objective: str, companies: list[str]) -> str:
    mode = str(mode or "").strip()
    if companies:
        return "competitor_intel"
    return mode if mode in MODES else infer_mode(objective)


def infer_mode(objective: str) -> str:
    text = objective.lower()
    if has_any(text, ["architecture", "algorithm", "attention", "paper", "lstm", "rnn", "transformer"]):
        return "technical_deep_dive"
    if has_any(text, ["market", "industry", "landscape", "trend", "competitive"]):
        return "market_research"
    return "knowledge_research"


def fix_format(value: Any, objective: str, mode: str) -> str:
    value = str(value or "").strip()
    if value in FORMATS:
        return value
    if has_any(objective.lower(), ["compare", "vs", "versus", "difference"]):
        return "comparison_table"
    return "report" if mode in {"competitor_intel", "market_research"} else "summary"


def fix_synthesis(value: Any, objective: str) -> str:
    text = str(value or "").strip()
    if text and not generic_instruction(text):
        return text
    if has_any(objective.lower(), ["compare", "vs", "versus", "difference"]):
        return f"Create a comparison for: {objective}. Cover features, cost, strengths, limitations, use cases, and cite URLs."
    return f"Synthesize findings for: {objective}. Cite source URLs and mention uncertainty."


def generic_instruction(text: str) -> bool:
    phrases = {"combine evidence", "combine the evidence", "create a comparison table", "provide a comprehensive overview", "synthesize findings"}
    text = text.lower().strip()
    return any(text == phrase or text.startswith(f"{phrase}.") for phrase in phrases)


def strip_fence(raw: str) -> str:
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
    return re.search(r"(?<![a-z0-9])" + re.escape(keyword.lower()) + r"(?![a-z0-9])", text) is not None


def has_any(text: str, keywords: list[str]) -> bool:
    return any(term_match(text, keyword) for keyword in keywords)
