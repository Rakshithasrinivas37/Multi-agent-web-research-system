import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional
from urllib.parse import quote_plus


@dataclass(frozen=True)
class ResearchTask:
    """One source-specific task that a browser or scraper agent can execute."""

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
    """Structured research plan returned by the planner agent."""

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


SOURCE_TYPES = {
    "overview": {
        "goal": "Extract definitions, scope, key concepts, and background context.",
        "signals": ["definitions", "main concepts", "important subtopics"],
    },
    "authoritative_sources": {
        "goal": "Extract reliable source names, URLs, credibility signals, and primary-source evidence.",
        "signals": ["official pages", "primary sources", "expert sources"],
    },
    "recent_updates": {
        "goal": "Extract dates, new developments, announcements, and changed claims.",
        "signals": ["new developments", "dates", "announcements"],
    },
    "evidence": {
        "goal": "Extract facts, examples, metrics, quotes, and source links.",
        "signals": ["supporting facts", "examples", "metrics", "source links"],
    },
    "synthesis": {
        "goal": "Extract comparisons, tradeoffs, pros, cons, and unresolved questions.",
        "signals": ["comparisons", "pros and cons", "tradeoffs"],
    },
    "technical_overview": {
        "goal": "Extract definitions, architecture components, conceptual differences, and use cases.",
        "signals": ["core concepts", "architectural differences", "strengths and weaknesses"],
    },
    "academic": {
        "goal": "Extract paper titles, authors, publication years, claims, and source links.",
        "signals": ["seminal papers", "method descriptions", "research context"],
    },
    "benchmarks": {
        "goal": "Extract benchmark tasks, metrics, datasets, results, and tradeoff explanations.",
        "signals": ["metrics", "datasets", "accuracy", "latency", "memory tradeoffs"],
    },
    "implementation": {
        "goal": "Extract code references, framework examples, training details, and practical limitations.",
        "signals": ["code examples", "framework implementations", "common pitfalls"],
    },
    "news": {
        "goal": "Extract article titles, dates, categories, summaries, and article URLs.",
        "signals": ["announcements", "partnerships", "release timing"],
    },
}


KNOWN_COMPANIES = {
    "openai": {
        "name": "OpenAI",
        "aliases": ("openai", "chatgpt", "gpt", "sora"),
        "sources": {
            "news": "https://openai.com/news/",
            "pricing": "https://openai.com/chatgpt/pricing/",
            "docs": "https://platform.openai.com/docs",
        },
    },
    "anthropic": {
        "name": "Anthropic",
        "aliases": ("anthropic", "claude"),
        "sources": {
            "news": "https://www.anthropic.com/news",
            "pricing": "https://www.anthropic.com/pricing",
            "docs": "https://docs.anthropic.com",
        },
    },
}


class PlannerAgent:
    """Break any research objective into source-specific subtasks."""

    SYSTEM_PROMPT = """You are an expert planner for a multi-agent web research system.

Create a structured research plan for the user's objective.

Modes:
- competitor_intel: monitor specific companies/products.
- knowledge_research: explain concepts and comparisons.
- technical_deep_dive: papers, internals, architectures, benchmarks.
- market_research: industry trends, landscape, competitors.

Return ONLY valid JSON:
{
  "research_mode": "competitor_intel|knowledge_research|technical_deep_dive|market_research",
  "companies": ["optional company names"],
  "sub_questions": ["specific answerable question"],
  "tasks": [
    {
      "query_context": "which sub-question this answers",
      "url": "https://exact-url.com OR SEARCH:search query",
      "source_type": "webpage|arxiv|wikipedia|search|docs|news",
      "priority": 1,
      "extraction_goal": "precise extraction goal",
      "target_type": "company|discovery",
      "target_name": "company name or General Research",
      "use_playwright": false,
      "expected_signals": ["signal 1", "signal 2"]
    }
  ],
  "synthesis_instruction": "how final answer should combine evidence",
  "output_format": "comparison_table|deep_dive|summary|report"
}

Rules:
- Comparisons should use output_format "comparison_table".
- "How does X work" should use output_format "deep_dive".
- Use SEARCH: when an exact URL is unknown.
- Priority 1 means essential, 2 supporting, 3 optional."""

    def __init__(self, use_llm: bool = True, model: Optional[str] = None) -> None:
        """Initialize the planner with optional Groq LLM planning."""

        self.use_llm = use_llm
        self.model = model or os.environ.get("RESEARCH_PLANNER_MODEL", "llama-3.1-8b-instant")

    def plan(self, objective: str) -> ResearchPlan:
        """Create a research plan using Groq first, then deterministic fallback."""

        if self.use_llm:
            try:
                return self._plan_with_groq(objective)
            except Exception as error:
                print(f"[planner_agent] Groq planner unavailable; using fallback planner: {error}")
        return self._plan_with_rules(objective)

    def _plan_with_groq(self, objective: str) -> ResearchPlan:
        """Use Groq to create a flexible research plan."""

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
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    *messages,
                ],
            )
            raw = (response.choices[0].message.content or "").strip()
            messages.append({"role": "assistant", "content": raw})
            try:
                return self._parse_llm_plan(raw, objective)
            except (json.JSONDecodeError, KeyError, ValueError) as error:
                last_error = error
                messages.append(
                    {
                        "role": "user",
                        "content": f"Fix this error and return only valid JSON: {error}",
                    }
                )
                print(f"[planner_agent] Plan parse failed on attempt {attempt}: {error}")

        raise RuntimeError(f"Groq planner failed after 3 attempts: {last_error}")

    def _parse_llm_plan(self, raw: str, objective: str) -> ResearchPlan:
        """Parse the JSON plan returned by Groq."""

        data = json.loads(self._strip_markdown_fence(raw))
        tasks = [
            ResearchTask(
                task_id=f"task_{index:03d}",
                query_context=task["query_context"],
                url=task["url"],
                source_type=task.get("source_type", "webpage"),
                priority=int(task.get("priority", index)),
                extraction_goal=task["extraction_goal"],
                target_type=task.get("target_type", "discovery"),
                target_name=task.get("target_name", "General Research"),
                use_playwright=bool(task.get("use_playwright", False)),
                expected_signals=list(task.get("expected_signals", [])),
            )
            for index, task in enumerate(data.get("tasks", []), start=1)
        ]
        if not tasks:
            raise ValueError("planner returned no tasks")

        companies = [str(company) for company in data.get("companies", []) if str(company).strip()]
        return ResearchPlan(
            objective=objective,
            research_mode=data["research_mode"],
            companies=companies,
            sub_questions=list(data.get("sub_questions", [])),
            tasks=sorted(tasks, key=lambda task: task.priority),
            synthesis_instruction=data.get("synthesis_instruction", "Synthesize findings clearly."),
            output_format=data.get("output_format", "summary"),
        )

    def _plan_with_rules(self, objective: str) -> ResearchPlan:
        """Fallback planner that works without network access or API keys."""

        companies = self._detect_companies(objective)
        research_mode = self._research_mode(objective, bool(companies))
        source_types = self._source_types(objective, research_mode, bool(companies))

        if companies:
            tasks = self._company_tasks(objective, companies, source_types)
        else:
            tasks = self._discovery_tasks(objective, source_types)

        return ResearchPlan(
            objective=objective,
            research_mode=research_mode,
            companies=[company["name"] for company in companies],
            sub_questions=[task.query_context for task in tasks],
            tasks=tasks,
            synthesis_instruction=self._synthesis_instruction(research_mode),
            output_format=self._output_format(objective, research_mode),
        )

    def _company_tasks(
        self,
        objective: str,
        companies: list[dict[str, Any]],
        source_types: list[str],
    ) -> list[ResearchTask]:
        """Create direct company-source tasks when possible."""

        tasks = []
        task_number = 1
        for company in companies:
            for priority, source_type in enumerate(source_types, start=1):
                url = company["sources"].get(source_type) or self._search_url(
                    f"{company['name']} {objective}",
                    source_type,
                )
                tasks.append(
                    ResearchTask(
                        task_id=f"task_{task_number:03d}",
                        query_context=f"{company['name']}: {self._context(objective, source_type)}",
                        url=url,
                        source_type=source_type,
                        priority=priority,
                        extraction_goal=SOURCE_TYPES.get(source_type, SOURCE_TYPES["overview"])["goal"],
                        target_type="company",
                        target_name=company["name"],
                        use_playwright=source_type in {"pricing", "careers", "reviews"},
                        expected_signals=list(
                            SOURCE_TYPES.get(source_type, SOURCE_TYPES["overview"])["signals"]
                        ),
                    )
                )
                task_number += 1
        return tasks

    def _discovery_tasks(self, objective: str, source_types: list[str]) -> list[ResearchTask]:
        """Create general discovery tasks when no company is named."""

        tasks = []
        for priority, source_type in enumerate(source_types, start=1):
            source = SOURCE_TYPES.get(source_type, SOURCE_TYPES["overview"])
            tasks.append(
                ResearchTask(
                    task_id=f"task_{priority:03d}",
                    query_context=self._context(objective, source_type),
                    url=self._seed_url(objective, source_type),
                    source_type=source_type,
                    priority=priority,
                    extraction_goal=source["goal"],
                    target_type="discovery",
                    target_name="General Research",
                    use_playwright=False,
                    expected_signals=["authoritative source URLs", *list(source["signals"])],
                )
            )
        return tasks

    def _seed_url(self, objective: str, source_type: str) -> str:
        """Return a known source URL or a browser-ready search URL."""

        normalized = objective.lower()
        tokens = set(re.findall(r"[a-z0-9]+", normalized))
        if {"transformers", "transformer", "rnn", "lstm"} & tokens:
            if source_type == "technical_overview":
                return "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)"
            if source_type == "academic":
                return "https://arxiv.org/abs/1706.03762"
        if "rag" in tokens:
            if source_type == "academic":
                return "https://arxiv.org/abs/2005.11401"
            if source_type in {"docs", "implementation"}:
                return "https://python.langchain.com/docs/concepts/rag/"
        return self._search_url(objective, source_type)

    def _search_url(self, objective: str, source_type: str) -> str:
        """Build a search URL for the browsing agent."""

        suffixes = {
            "overview": "overview explanation background guide",
            "authoritative_sources": "official source primary source expert reference",
            "recent_updates": "latest news recent updates announcements",
            "evidence": "examples data metrics evidence facts",
            "synthesis": "comparison analysis pros cons tradeoffs summary",
            "technical_overview": "architecture overview explanation comparison",
            "academic": "research paper survey arxiv",
            "benchmarks": "benchmark comparison performance evaluation",
            "implementation": "implementation tutorial pytorch tensorflow code",
            "news": "latest news announcements",
            "pricing": "pricing plans tiers packaging",
            "docs": "official docs documentation",
        }
        query = quote_plus(f"{objective} {suffixes.get(source_type, 'research sources')}")
        if source_type in {"recent_updates", "news"}:
            return f"https://news.google.com/search?q={query}"
        return f"https://www.google.com/search?q={query}"

    def _research_mode(self, objective: str, company_specific: bool) -> str:
        """Classify the objective."""

        normalized = objective.lower()
        if company_specific:
            return "competitor_intel"
        if self._contains_any(normalized, ["architecture", "algorithm", "attention", "paper", "lstm", "rnn", "transformer"]):
            return "technical_deep_dive"
        if self._contains_any(normalized, ["market", "industry", "landscape", "trend", "competitive", "best"]):
            return "market_research"
        return "knowledge_research"

    def _source_types(self, objective: str, research_mode: str, company_specific: bool) -> list[str]:
        """Choose source categories for the objective."""

        normalized = objective.lower()
        if research_mode == "technical_deep_dive":
            return ["technical_overview", "academic", "benchmarks", "implementation"]
        if research_mode == "market_research":
            return ["recent_updates", "authoritative_sources", "evidence", "synthesis"]
        if company_specific:
            selected = []
            for source_type, keywords in {
                "pricing": ["pricing", "price", "plan", "tier"],
                "docs": ["docs", "api", "developer"],
                "news": ["news", "launch", "release", "announce"],
            }.items():
                if self._contains_any(normalized, keywords):
                    selected.append(source_type)
            return selected or ["news", "pricing", "docs"]
        return ["overview", "authoritative_sources", "recent_updates", "evidence", "synthesis"]

    def _detect_companies(self, objective: str) -> list[dict[str, Any]]:
        """Detect known companies/products mentioned in the objective."""

        normalized = objective.lower()
        return [
            company
            for company in KNOWN_COMPANIES.values()
            if any(self._contains(normalized, alias) for alias in company["aliases"])
        ]

    def _context(self, objective: str, source_type: str) -> str:
        """Create a focused sub-question for a task."""

        templates = {
            "overview": f"What is the scope and background of {objective}?",
            "authoritative_sources": f"Which sources are most authoritative for {objective}?",
            "recent_updates": f"What changed recently about {objective}?",
            "evidence": f"What evidence, examples, or metrics support analysis of {objective}?",
            "synthesis": f"What are the tradeoffs and conclusions for {objective}?",
            "technical_overview": f"What are the core concepts and architecture details behind {objective}?",
            "academic": f"Which papers or scholarly sources best explain {objective}?",
            "benchmarks": f"What benchmark evidence compares performance and tradeoffs for {objective}?",
            "implementation": f"How is {objective} implemented in practice?",
            "news": f"What recent announcements or updates relate to {objective}?",
            "pricing": f"What pricing or packaging information relates to {objective}?",
            "docs": f"What official documentation explains {objective}?",
        }
        return templates.get(source_type, f"What sources help answer {objective}?")

    def _synthesis_instruction(self, research_mode: str) -> str:
        """Tell a later synthesis agent how to combine results."""

        return {
            "competitor_intel": "Compare company signals and cite source evidence.",
            "knowledge_research": "Create a clear learning-oriented answer with cited sources.",
            "technical_deep_dive": "Explain mechanisms, compare tradeoffs, and cite papers or technical sources.",
            "market_research": "Summarize landscape, trends, evidence, and practical implications.",
        }[research_mode]

    def _output_format(self, objective: str, research_mode: str) -> str:
        """Choose a final answer format."""

        normalized = objective.lower()
        if self._contains_any(normalized, ["compare", "versus", "vs", "difference"]):
            return "comparison_table"
        if self._contains_any(normalized, ["how", "explain", "architecture", "mechanism"]):
            return "deep_dive"
        if research_mode in {"competitor_intel", "market_research"}:
            return "report"
        return "summary"

    def _strip_markdown_fence(self, raw: str) -> str:
        """Remove markdown fences around JSON."""

        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            clean = "\n".join(lines)
        return clean.strip()

    def _contains_any(self, text: str, keywords: Iterable[str]) -> bool:
        """Return True if any keyword appears as a term."""

        return any(self._contains(text, keyword) for keyword in keywords)

    def _contains(self, text: str, keyword: str) -> bool:
        """Match full words or phrases without substring false positives."""

        pattern = r"(?<![a-z0-9])" + re.escape(keyword.lower()) + r"(?![a-z0-9])"
        return re.search(pattern, text) is not None
