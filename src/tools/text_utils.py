"""Small text helpers shared by browser tools."""

from typing import Any
import html
import re


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [clean_text(item) for item in value if clean_text(item)]


def clean_content(content: str) -> str:
    text = str(content or "")
    text = re.sub(r"<script.*?</script>", "\n", text, flags=re.I | re.S)
    text = re.sub(r"<style.*?</style>", "\n", text, flags=re.I | re.S)
    text = re.sub(r"</(p|div|section|article|header|footer|main|nav|li|ul|ol|h[1-6]|tr)>", "\n", text, flags=re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</t[dh]>", " | ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)

    lines = []
    for line in text.splitlines():
        cleaned = clean_text(line)
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def title_from_html(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
    return clean_text(match.group(1)) if match else ""
