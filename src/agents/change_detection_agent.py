"""Change detection agent for comparing browser extraction results."""

import difflib
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse

from src.memory.shared_memory import SharedMemory
from src.tools.text_utils import clean_text


class ChangeDetectionAgent:
    """Compares previous and current browser results for meaningful changes."""

    def __init__(self, history_db_path: Union[str, Path] = "data/browser_history.db") -> None:
        self.history_db_path = Path(history_db_path)

    def detect_with_history(
        self,
        objective: str,
        current_results: list[dict[str, Any]],
        research_plan: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        key = objective_key(objective, research_plan)
        previous_results = self.read_previous_results(key, objective)
        diff = self.detect(objective, previous_results, current_results)
        diff["first_run"] = not previous_results
        if not previous_results:
            diff["summary"] = "No previous browser history found for this objective; stored current browser results as the baseline."
            diff["baseline_source_count"] = len(source_records_by_url(current_results))
        diff["history_db_path"] = str(self.history_db_path)
        diff["history_key"] = key
        diff["history_update"] = self.write_current_results(key, objective, previous_results, current_results)
        return diff

    def detect(
        self,
        objective: str,
        previous_results: list[dict[str, Any]],
        current_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return detect_browser_changes(objective, previous_results, current_results)

    def write_to_memory(self, diff: dict[str, Any], memory_path: str) -> None:
        memory = SharedMemory(memory_path)
        memory.write_agent_output("change_detection", {"diff": diff})

    def read_previous_results(self, key: str, objective: str) -> list[dict[str, Any]]:
        self.ensure_history_table()
        lookup_key = self.compatible_history_key(key, objective)
        grouped_results: dict[str, dict[str, Any]] = {}
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT result_json, source_json
                FROM browser_history_sources
                WHERE objective_key = ? AND active = 1
                ORDER BY task_id, url
                """,
                (lookup_key,),
            ).fetchall()

        for result_json, source_json in rows:
            try:
                result = json.loads(result_json)
                source = json.loads(source_json)
            except json.JSONDecodeError:
                continue
            if not isinstance(result, dict) or not isinstance(source, dict):
                continue
            key = clean_text(result.get("task_id")) or clean_text(source.get("url"))
            grouped_results.setdefault(key, {**result, "sources": []})
            grouped_results[key]["sources"].append(source)

        return list(grouped_results.values())

    def compatible_history_key(self, key: str, objective: str) -> str:
        self.ensure_history_table()
        with self.connection() as connection:
            exact_match = connection.execute(
                """
                SELECT 1
                FROM browser_history_sources
                WHERE objective_key = ? AND active = 1
                LIMIT 1
                """,
                (key,),
            ).fetchone()
            if exact_match:
                return key

            rows = connection.execute(
                """
                SELECT objective_key, objective
                FROM browser_history_sources
                WHERE active = 1
                GROUP BY objective_key, objective
                """
            ).fetchall()

        current_terms = set(canonical_objective_identity(objective).split(":"))
        for candidate_key, candidate_objective in rows:
            candidate_terms = set(canonical_objective_identity(candidate_objective).split(":"))
            if current_terms and current_terms == candidate_terms:
                return candidate_key
        return key

    def write_current_results(
        self,
        key: str,
        objective: str,
        previous_results: list[dict[str, Any]],
        current_results: list[dict[str, Any]],
    ) -> dict[str, int]:
        self.ensure_history_table()
        now = datetime.now(timezone.utc).isoformat()
        previous_sources = source_records_by_url(previous_results)
        current_sources = source_records_by_url(current_results)

        added_urls = sorted(set(current_sources) - set(previous_sources))
        changed_urls = sorted(
            url
            for url in set(current_sources) & set(previous_sources)
            if current_sources[url]["content_hash"] != previous_sources[url]["content_hash"]
        )
        removed_urls = sorted(set(previous_sources) - set(current_sources))
        unchanged_count = len(set(current_sources) & set(previous_sources)) - len(changed_urls)

        with self.connection() as connection:
            for url in [*added_urls, *changed_urls]:
                source = current_sources[url]
                connection.execute(
                    """
                    INSERT INTO browser_history_sources (
                        objective_key,
                        objective,
                        url,
                        task_id,
                        content_hash,
                        result_json,
                        source_json,
                        active,
                        updated_at,
                        removed_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, NULL)
                    ON CONFLICT(objective_key, url) DO UPDATE SET
                        objective = excluded.objective,
                        task_id = excluded.task_id,
                        content_hash = excluded.content_hash,
                        result_json = excluded.result_json,
                        source_json = excluded.source_json,
                        active = 1,
                        updated_at = excluded.updated_at,
                        removed_at = NULL
                    """,
                    (
                        key,
                        objective,
                        url,
                        source.get("task_id", ""),
                        source["content_hash"],
                        json.dumps(source.get("result", {}), ensure_ascii=False),
                        json.dumps(source.get("source", {}), ensure_ascii=False),
                        now,
                    ),
                )

            if removed_urls:
                connection.executemany(
                    """
                    UPDATE browser_history_sources
                    SET active = 0, removed_at = ?, updated_at = ?
                    WHERE objective_key = ? AND url = ?
                    """,
                    [(now, now, key, url) for url in removed_urls],
                )

        return {
            "stored_added": len(added_urls),
            "stored_changed": len(changed_urls),
            "marked_removed": len(removed_urls),
            "skipped_unchanged": unchanged_count,
        }

    def ensure_history_table(self) -> None:
        self.history_db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS browser_history_sources (
                    objective TEXT NOT NULL,
                    objective_key TEXT NOT NULL,
                    url TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    source_json TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL,
                    removed_at TEXT,
                    PRIMARY KEY (objective_key, url)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_browser_history_sources_objective_active
                ON browser_history_sources (objective_key, active)
                """
            )

    def connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.history_db_path)


def detect_browser_changes(
    objective: str,
    previous_results: list[dict[str, Any]],
    current_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect source/content changes between two browser result sets."""

    previous_sources = source_records_by_url(previous_results)
    current_sources = source_records_by_url(current_results)
    previous_urls = set(previous_sources)
    current_urls = set(current_sources)

    added_urls = sorted(current_urls - previous_urls)
    removed_urls = sorted(previous_urls - current_urls)
    changed_sources = []
    unchanged_sources = []

    for url in sorted(previous_urls & current_urls):
        previous = previous_sources[url]
        current = current_sources[url]
        if previous["content_hash"] == current["content_hash"]:
            unchanged_sources.append(source_summary(current))
            continue

        changed_sources.append(
            {
                **source_summary(current),
                "previous_content_hash": previous["content_hash"],
                "new_content_hash": current["content_hash"],
                "previous_content_length": previous["content_length"],
                "new_content_length": current["content_length"],
                "important_added_lines": important_diff_lines(
                    objective,
                    previous["content"],
                    current["content"],
                    prefix="+",
                ),
                "important_removed_lines": important_diff_lines(
                    objective,
                    previous["content"],
                    current["content"],
                    prefix="-",
                ),
            }
        )

    status_changes = result_status_changes(previous_results, current_results)
    quality_changes = source_quality_changes(previous_sources, current_sources)

    return {
        "objective": objective,
        "summary": change_summary(len(added_urls), len(removed_urls), len(changed_sources), len(unchanged_sources)),
        "added_sources": [source_summary(current_sources[url]) for url in added_urls],
        "removed_sources": [source_summary(previous_sources[url]) for url in removed_urls],
        "changed_sources": changed_sources,
        "unchanged_count": len(unchanged_sources),
        "status_changes": status_changes,
        "quality_changes": quality_changes,
        "objective_alignment": objective_alignment(objective, current_sources),
    }


def source_records_by_url(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for result in results:
        for source in result.get("sources", []):
            url = normalize_url(clean_text(source.get("url")))
            if not url:
                continue
            content = clean_text(source.get("full_content") or source.get("content_preview") or "")
            result_payload = {key: value for key, value in result.items() if key != "sources"}
            source_payload = {**source, "url": url}
            records[url] = {
                "url": url,
                "title": clean_text(source.get("title")),
                "task_id": clean_text(result.get("task_id")),
                "query_context": clean_text(result.get("query_context")),
                "status": clean_text(result.get("status")),
                "source_type": clean_text(source.get("source_type")),
                "source_quality": clean_text(source.get("source_quality")),
                "content": content,
                "content_hash": hash_text(content),
                "content_length": len(content),
                "result": result_payload,
                "source": source_payload,
            }
    return records


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return url
    return f"{parsed.scheme}://{parsed.netloc.lower()}{parsed.path.rstrip('/')}"


def source_summary(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "url": source["url"],
        "title": source.get("title", ""),
        "task_id": source.get("task_id", ""),
        "query_context": source.get("query_context", ""),
        "source_type": source.get("source_type", ""),
        "source_quality": source.get("source_quality", ""),
        "content_hash": source.get("content_hash", ""),
        "content_length": source.get("content_length", 0),
    }


def important_diff_lines(objective: str, previous: str, current: str, prefix: str, limit: int = 20) -> list[str]:
    diff = difflib.unified_diff(previous.splitlines(), current.splitlines(), lineterm="")
    lines = []
    for line in diff:
        if not line.startswith(prefix) or line.startswith(("+++", "---")):
            continue
        text = clean_text(line[1:])
        if text and important_line(objective, text) and text not in lines:
            lines.append(text[:700])
        if len(lines) >= limit:
            break
    return lines


def important_line(objective: str, line: str) -> bool:
    text = line.lower()
    keywords = {
        "price",
        "pricing",
        "tier",
        "model",
        "benchmark",
        "score",
        "accuracy",
        "bleu",
        "latency",
        "token",
        "limit",
        "feature",
        "launch",
        "deprecated",
        "removed",
        "added",
        "attention",
        "architecture",
        "training",
        "dataset",
        "result",
    }
    objective_terms = {word for word in re_words(objective.lower()) if len(word) > 3}
    return any(keyword in text for keyword in keywords | objective_terms)


def result_status_changes(
    previous_results: list[dict[str, Any]],
    current_results: list[dict[str, Any]],
) -> list[dict[str, str]]:
    previous = {clean_text(item.get("task_id")): clean_text(item.get("status")) for item in previous_results}
    current = {clean_text(item.get("task_id")): clean_text(item.get("status")) for item in current_results}
    changes = []
    for task_id in sorted(set(previous) & set(current)):
        if previous[task_id] != current[task_id]:
            changes.append(
                {
                    "task_id": task_id,
                    "previous_status": previous[task_id],
                    "new_status": current[task_id],
                }
            )
    return changes


def source_quality_changes(
    previous_sources: dict[str, dict[str, Any]],
    current_sources: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    changes = []
    for url in sorted(set(previous_sources) & set(current_sources)):
        previous_quality = previous_sources[url].get("source_quality", "")
        current_quality = current_sources[url].get("source_quality", "")
        if previous_quality != current_quality:
            changes.append(
                {
                    "url": url,
                    "previous_quality": previous_quality,
                    "new_quality": current_quality,
                }
            )
    return changes


def change_summary(added: int, removed: int, changed: int, unchanged: int) -> str:
    if not any((added, removed, changed)):
        return f"No source content changes detected across {unchanged} unchanged sources."
    return f"Detected {added} added, {removed} removed, and {changed} changed sources; {unchanged} sources were unchanged."


def objective_alignment(objective: str, sources: dict[str, dict[str, Any]]) -> str:
    terms = {word for word in re_words(objective.lower()) if len(word) > 3}
    if not terms or not sources:
        return "unknown"
    matches = 0
    for source in sources.values():
        text = f"{source.get('title', '')} {source.get('query_context', '')} {source.get('content', '')[:3000]}".lower()
        if any(term in text for term in terms):
            matches += 1
    ratio = matches / max(1, len(sources))
    if ratio >= 0.75:
        return "strong"
    if ratio >= 0.4:
        return "partial"
    return "weak"


def re_words(text: str) -> list[str]:
    return [word.strip(".,:;!?()[]{}\"'") for word in text.split()]


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def objective_key(objective: str, research_plan: Optional[dict[str, Any]] = None) -> str:
    normalized = canonical_objective_identity(objective, research_plan)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonical_objective_identity(objective: str, research_plan: Optional[dict[str, Any]] = None) -> str:
    plan = research_plan if isinstance(research_plan, dict) else {}
    companies = sorted({canonical_token(company) for company in clean_string_list(plan.get("companies"))})
    company_terms = {term for company in companies for term in company.split("_")}

    terms = canonical_terms(objective)
    terms = [term for term in terms if term not in company_terms]
    identity_parts = [*companies, *terms[:12]]
    if not identity_parts:
        identity_parts = [clean_text(plan.get("research_mode")).lower() or "research"]
    return ":".join(part for part in identity_parts if part)


def clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [clean_text(item) for item in value if clean_text(item)]


def canonical_terms(text: str) -> list[str]:
    stopwords = {
        "about",
        "across",
        "and",
        "are",
        "compare",
        "comparison",
        "different",
        "does",
        "each",
        "for",
        "from",
        "have",
        "how",
        "into",
        "its",
        "main",
        "of",
        "the",
        "their",
        "these",
        "this",
        "to",
        "what",
        "which",
        "with",
        "explain"
    }
    terms = {canonical_token(word) for word in re_words(text.lower())}
    return sorted(term for term in terms if len(term) > 2 and term not in stopwords)


def canonical_token(value: str) -> str:
    token = clean_text(value).lower()
    token = "".join(character if character.isalnum() else "_" for character in token)
    token = "_".join(part for part in token.split("_") if part)
    return singularize_token(token)


def singularize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token
