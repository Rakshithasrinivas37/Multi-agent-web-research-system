"""Lightweight progress events for long-running research workflows."""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable


ProgressCallback = Callable[[dict[str, Any]], None]
_progress_callback: ContextVar[ProgressCallback | None] = ContextVar("progress_callback", default=None)


def set_progress_callback(callback: ProgressCallback):
    """Attach a progress callback to the current context."""

    return _progress_callback.set(callback)


def reset_progress_callback(token: Any) -> None:
    """Restore the previous progress callback."""

    _progress_callback.reset(token)


def emit_progress(
    event: str,
    message: str,
    agent: str = "",
    tool: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Emit a JSON-serializable workflow progress event if a callback exists."""

    callback = _progress_callback.get()
    if callback is None:
        return

    callback(
        {
            "event": event,
            "agent": agent,
            "tool": tool,
            "message": message,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
