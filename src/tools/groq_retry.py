"""Retry helpers for Groq chat-completion calls."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.tools.text_utils import clean_text


DEFAULT_GROQ_RETRY_ATTEMPTS = 3
DEFAULT_GROQ_RETRY_SHRINK_FACTOR = 0.75
MIN_RETRIED_MESSAGE_CHARS = 500
MIN_RETRIED_MAX_TOKENS = 80


def create_chat_completion_with_retries(
    client: Any,
    *,
    retry_attempts: int = DEFAULT_GROQ_RETRY_ATTEMPTS,
    shrink_factor: float = DEFAULT_GROQ_RETRY_SHRINK_FACTOR,
    **kwargs: Any,
) -> Any:
    """Call Groq chat completions, shrinking oversized prompts on retry."""

    retry_attempts = max(1, retry_attempts)
    shrink_factor = min(0.95, max(0.25, shrink_factor))
    request = deepcopy(kwargs)
    last_error: Exception | None = None

    for attempt in range(retry_attempts):
        try:
            return client.chat.completions.create(**request)
        except Exception as error:
            last_error = error
            if attempt + 1 >= retry_attempts or not is_groq_request_too_large_error(error):
                raise
            request = shrink_groq_chat_request(request, shrink_factor=shrink_factor)

    if last_error:
        raise last_error
    raise RuntimeError("Groq chat completion retry failed before first attempt")


def is_groq_request_too_large_error(error: Exception) -> bool:
    message = clean_text(error).lower()
    return (
        "request too large" in message
        or "tokens per minute" in message
        or "rate_limit_exceeded" in message
        or "tpm" in message
        or "please reduce your message size" in message
    )


def shrink_groq_chat_request(request: dict[str, Any], shrink_factor: float) -> dict[str, Any]:
    retried = deepcopy(request)
    retried["max_tokens"] = shrink_max_tokens(retried.get("max_tokens"), shrink_factor=shrink_factor)
    retried["messages"] = shrink_largest_user_message(
        retried.get("messages", []),
        shrink_factor=shrink_factor,
    )
    return retried


def shrink_max_tokens(value: Any, shrink_factor: float) -> int:
    try:
        max_tokens = int(value)
    except (TypeError, ValueError):
        max_tokens = MIN_RETRIED_MAX_TOKENS
    return max(MIN_RETRIED_MAX_TOKENS, int(max_tokens * shrink_factor))


def shrink_largest_user_message(messages: Any, shrink_factor: float) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return messages

    retried = deepcopy(messages)
    candidates = [
        (index, len(message.get("content", "")))
        for index, message in enumerate(retried)
        if isinstance(message, dict)
        and message.get("role") == "user"
        and isinstance(message.get("content"), str)
    ]
    if not candidates:
        return retried

    index, length = max(candidates, key=lambda item: item[1])
    if length <= MIN_RETRIED_MESSAGE_CHARS:
        return retried
    target_length = max(MIN_RETRIED_MESSAGE_CHARS, int(length * shrink_factor))
    retried[index]["content"] = truncate_preserving_ends(retried[index]["content"], target_length)
    return retried


def truncate_preserving_ends(text: str, max_chars: int) -> str:
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    notice = "\n\n[...content trimmed for Groq retry...]\n\n"
    keep = max(0, max_chars - len(notice))
    head = keep // 2
    tail = keep - head
    return f"{value[:head].rstrip()}{notice}{value[-tail:].lstrip()}"
