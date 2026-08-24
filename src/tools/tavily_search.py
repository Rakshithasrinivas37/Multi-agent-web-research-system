"""Tavily search tool."""

import json
import os
import select
import shlex
import subprocess
import time
from itertools import count
from typing import Any

import httpx

from src.tools.text_utils import clean_text


DEFAULT_TAVILY_MCP_COMMAND = "npx"
DEFAULT_TAVILY_MCP_ARGS = "-y tavily-mcp@latest"
DEFAULT_TAVILY_MCP_URL = "https://mcp.tavily.com/mcp/?tavilyApiKey=${TAVILY_API_KEY}"
DEFAULT_TAVILY_MCP_TOOL = "tavily_search"
DEFAULT_TAVILY_MCP_TIMEOUT_SECONDS = 90
MCP_REQUEST_IDS = count(1)


def search_with_tavily(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    query = clean_text(query)
    print(
        "[tavily_search] search requested "
        f"backend={'mcp' if tavily_mcp_enabled() else 'sdk'} max_results={max_results} query={query!r}"
    )
    if tavily_mcp_enabled():
        try:
            results = search_with_tavily_mcp(query, max_results=max_results)
            log_tavily_results("MCP", query, results)
            return results
        except Exception as error:
            if tavily_mcp_required():
                raise RuntimeError(f"Tavily MCP search failed: {error}") from error
            print(f"[tavily_search] Tavily MCP failed; using SDK fallback: {error}")

    results = search_with_tavily_sdk(query, max_results=max_results)
    log_tavily_results("SDK", query, results)
    return results


def search_with_tavily_sdk(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")

    from tavily import TavilyClient

    response = TavilyClient(api_key=api_key).search(
        query=query,
        max_results=max_results,
        search_depth="basic",
    )
    results = response.get("results", [])
    print(
        "[tavily_search] SDK response "
        f"keys={sorted(response.keys()) if isinstance(response, dict) else type(response).__name__} "
        f"result_count={len(results) if isinstance(results, list) else 0}"
    )
    if not results:
        print("[tavily_search] SDK returned no results; check query, API quota/key, and Tavily service response.")
    return results if isinstance(results, list) else []


def search_with_tavily_mcp(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    query = clean_text(query)
    if not query:
        return []

    tool_name = clean_text(os.environ.get("TAVILY_MCP_TOOL")) or DEFAULT_TAVILY_MCP_TOOL
    print(f"[tavily_search] tool call via MCP: {tool_name}")
    result = call_mcp_tool(
        tool_name=tool_name,
        arguments={
            "query": query,
            "max_results": max(1, int(max_results)),
            "search_depth": clean_text(os.environ.get("TAVILY_SEARCH_DEPTH")) or "basic",
        },
        timeout=float(os.environ.get("TAVILY_MCP_TIMEOUT_SECONDS") or DEFAULT_TAVILY_MCP_TIMEOUT_SECONDS),
    )
    log_mcp_result_shape(result)
    return normalize_tavily_mcp_results(result)


def call_mcp_tool(tool_name: str, arguments: dict[str, Any], timeout: float) -> dict[str, Any]:
    transport = clean_text(os.environ.get("TAVILY_MCP_TRANSPORT")).lower()
    if transport in {"http", "remote", "streamable_http"}:
        return call_remote_mcp_tool(tool_name=tool_name, arguments=arguments, timeout=timeout)
    if transport not in {"", "stdio", "local"} and clean_text(os.environ.get("TAVILY_MCP_URL")):
        return call_remote_mcp_tool(tool_name=tool_name, arguments=arguments, timeout=timeout)

    command = clean_text(os.environ.get("TAVILY_MCP_COMMAND")) or DEFAULT_TAVILY_MCP_COMMAND
    args_text = expand_env_vars(clean_text(os.environ.get("TAVILY_MCP_ARGS")) or DEFAULT_TAVILY_MCP_ARGS)
    process = subprocess.Popen(
        [command, *shlex.split(args_text)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
    )
    try:
        send_mcp_message(
            process,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "multi-agent-web-research-system", "version": "0.1.0"},
                },
            },
        )
        initialize_response = read_mcp_response(process, request_id=1, timeout=timeout)
        if initialize_response.get("error"):
            raise RuntimeError(clean_text(initialize_response["error"]))

        send_mcp_message(process, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        send_mcp_message(
            process,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
        )
        tool_response = read_mcp_response(process, request_id=2, timeout=timeout)
        if tool_response.get("error"):
            raise RuntimeError(clean_text(tool_response["error"]))
        print(f"[mcp] tool call completed: {tool_name}")
        return tool_response.get("result", {})
    finally:
        terminate_mcp_process(process)


def call_remote_mcp_tool(tool_name: str, arguments: dict[str, Any], timeout: float) -> dict[str, Any]:
    endpoint = expand_env_vars(clean_text(os.environ.get("TAVILY_MCP_URL")) or DEFAULT_TAVILY_MCP_URL)
    session_id = ""
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        initialize_response = post_mcp_json_rpc(
            client,
            endpoint,
            headers,
            {
                "jsonrpc": "2.0",
                "id": next_mcp_request_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "multi-agent-web-research-system", "version": "0.1.0"},
                },
            },
        )
        session_id = clean_text(initialize_response.get("session_id"))
        if session_id:
            headers["Mcp-Session-Id"] = session_id

        post_mcp_notification(
            client,
            endpoint,
            headers,
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        )
        tool_response = post_mcp_json_rpc(
            client,
            endpoint,
            headers,
            {
                "jsonrpc": "2.0",
                "id": next_mcp_request_id(),
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
        )

    print(f"[mcp] tool call completed: {tool_name}")
    return tool_response.get("result", {})


def post_mcp_json_rpc(
    client: httpx.Client,
    endpoint: str,
    headers: dict[str, str],
    message: dict[str, Any],
) -> dict[str, Any]:
    response = client.post(endpoint, headers=headers, json=message)
    response.raise_for_status()
    parsed = parse_mcp_http_response(response)
    if response.headers.get("mcp-session-id"):
        parsed["session_id"] = response.headers["mcp-session-id"]
    if parsed.get("error"):
        raise RuntimeError(clean_text(parsed["error"]))
    return parsed


def post_mcp_notification(
    client: httpx.Client,
    endpoint: str,
    headers: dict[str, str],
    message: dict[str, Any],
) -> None:
    response = client.post(endpoint, headers=headers, json=message)
    if response.status_code not in {200, 202, 204}:
        response.raise_for_status()


def parse_mcp_http_response(response: httpx.Response) -> dict[str, Any]:
    content_type = clean_text(response.headers.get("content-type")).lower()
    if "text/event-stream" in content_type:
        return parse_mcp_sse_response(response.text)
    return response.json()


def parse_mcp_sse_response(text: str) -> dict[str, Any]:
    for event in text.split("\n\n"):
        data_lines = []
        for line in event.splitlines():
            if line.startswith("data:"):
                data_lines.append(line.partition(":")[2].strip())
        if data_lines:
            return json.loads("\n".join(data_lines))
    raise RuntimeError("MCP HTTP response did not include event data")


def next_mcp_request_id() -> int:
    return next(MCP_REQUEST_IDS)


def send_mcp_message(process: subprocess.Popen[bytes], message: dict[str, Any]) -> None:
    if process.stdin is None:
        raise RuntimeError("MCP process stdin is unavailable")
    body = json.dumps(message, separators=(",", ":")).encode("utf-8")
    process.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    process.stdin.flush()


def read_mcp_response(process: subprocess.Popen[bytes], request_id: int, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + max(1.0, timeout)
    while time.monotonic() < deadline:
        message = read_mcp_message(process, deadline=deadline)
        if message.get("id") == request_id:
            return message
    raise TimeoutError(f"MCP response timed out for request {request_id}")


def read_mcp_message(process: subprocess.Popen[bytes], deadline: float) -> dict[str, Any]:
    if process.stdout is None:
        raise RuntimeError("MCP process stdout is unavailable")

    header_bytes = read_until(process, b"\r\n\r\n", deadline)
    if b"\r\n\r\n" in header_bytes:
        header, body = header_bytes.split(b"\r\n\r\n", 1)
    else:
        header, body = header_bytes.split(b"\n\n", 1)

    content_length = 0
    for line in header.decode("utf-8", errors="replace").splitlines():
        name, _, value = line.partition(":")
        if name.lower() == "content-length":
            content_length = int(value.strip())
            break
    if content_length <= 0:
        raise RuntimeError("MCP response missing Content-Length header")

    while len(body) < content_length:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("MCP response body timed out")
        ready, _, _ = select.select([process.stdout.fileno()], [], [], remaining)
        if not ready:
            raise TimeoutError("MCP response body timed out")
        chunk = os.read(process.stdout.fileno(), content_length - len(body))
        if not chunk:
            raise RuntimeError(mcp_process_closed_error(process))
        body += chunk
    return json.loads(body[:content_length].decode("utf-8"))


def read_until(process: subprocess.Popen[bytes], marker: bytes, deadline: float) -> bytes:
    if process.stdout is None:
        raise RuntimeError("MCP process stdout is unavailable")

    data = b""
    fd = process.stdout.fileno()
    alternate_marker = b"\n\n"
    while marker not in data and alternate_marker not in data:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("MCP response header timed out")
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            raise TimeoutError("MCP response header timed out")
        chunk = os.read(fd, 1)
        if not chunk:
            raise RuntimeError(mcp_process_closed_error(process))
        data += chunk
    return data


def normalize_tavily_mcp_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    if result.get("isError"):
        raise RuntimeError(extract_mcp_text(result) or "Tavily MCP returned an error")

    text_payload = extract_mcp_text(result)
    payload = result.get("structuredContent") or parse_json_text(text_payload)
    if isinstance(payload, dict):
        items = payload.get("results") or payload.get("data") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    normalized = []
    skipped_non_dict = 0
    skipped_missing_url = 0
    for item in items:
        if not isinstance(item, dict):
            skipped_non_dict += 1
            continue
        url = clean_text(item.get("url"))
        if not url:
            skipped_missing_url += 1
            continue
        normalized.append(
            {
                "title": clean_text(item.get("title")),
                "url": url,
                "content": clean_text(item.get("content") or item.get("snippet")),
                "score": item.get("score"),
                "raw_content": clean_text(item.get("raw_content")),
            }
        )
    print(
        "[tavily_search] MCP normalized "
        f"payload={payload_summary(payload)} item_count={len(items)} normalized_count={len(normalized)} "
        f"skipped_non_dict={skipped_non_dict} skipped_missing_url={skipped_missing_url}"
    )
    if not normalized:
        reason = mcp_empty_result_reason(payload, items, skipped_non_dict, skipped_missing_url, text_payload)
        print(f"[tavily_search] MCP returned no usable results: {reason}")
    return normalized


def log_tavily_results(source: str, query: str, results: list[dict[str, Any]]) -> None:
    print(f"[tavily_search] {source} final result_count={len(results)} for query={clean_text(query)!r}")
    if not results:
        print(f"[tavily_search] {source} produced no results for planner/browser to consume.")
        return
    for index, item in enumerate(results, start=1):
        print(
            "[tavily_search] "
            f"{source} result {index}: {clean_text(item.get('title')) or 'Untitled'} | {clean_text(item.get('url'))}"
        )


def log_mcp_result_shape(result: dict[str, Any]) -> None:
    structured = result.get("structuredContent") if isinstance(result, dict) else None
    content = result.get("content") if isinstance(result, dict) else None
    content_types = [
        clean_text(item.get("type"))
        for item in (content or [])
        if isinstance(item, dict) and clean_text(item.get("type"))
    ]
    print(
        "[tavily_search] MCP raw response "
        f"keys={sorted(result.keys()) if isinstance(result, dict) else type(result).__name__} "
        f"structured={payload_summary(structured)} content_items={len(content or [])} content_types={content_types[:5]}"
    )


def payload_summary(payload: Any) -> str:
    if isinstance(payload, dict):
        return f"dict(keys={sorted(payload.keys())[:8]})"
    if isinstance(payload, list):
        return f"list(len={len(payload)})"
    if payload is None:
        return "none"
    return type(payload).__name__


def mcp_empty_result_reason(
    payload: Any,
    items: Any,
    skipped_non_dict: int,
    skipped_missing_url: int,
    text_payload: str,
) -> str:
    if not payload:
        preview = clean_text(text_payload)[:500]
        return f"no structuredContent and text payload was not parsed as JSON; text_preview={preview!r}"
    if isinstance(payload, dict) and not (payload.get("results") or payload.get("data")):
        return f"payload dict has no results/data array; keys={sorted(payload.keys())[:8]}"
    if isinstance(items, list) and not items:
        return "payload had an empty results/data list"
    if skipped_missing_url:
        return "items were present but missing url fields"
    if skipped_non_dict:
        return "items were present but were not JSON objects"
    return "unknown MCP response shape"


def extract_mcp_text(result: dict[str, Any]) -> str:
    parts = []
    for item in result.get("content", []) or []:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(clean_text(item.get("text")))
    return "\n".join(part for part in parts if part)


def parse_json_text(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {}


def expand_env_vars(text: str) -> str:
    value = str(text or "")
    for name, replacement in os.environ.items():
        value = value.replace(f"${{{name}}}", replacement)
        value = value.replace(f"${name}", replacement)
    return value


def terminate_mcp_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()


def mcp_process_closed_error(process: subprocess.Popen[bytes]) -> str:
    error = "MCP process closed stdout"
    if process.stderr is None:
        return error
    ready, _, _ = select.select([process.stderr.fileno()], [], [], 0)
    if not ready:
        return error
    stderr = clean_text(os.read(process.stderr.fileno(), 8000).decode("utf-8", errors="replace"))
    return f"{error}: {stderr}" if stderr else error


def tavily_mcp_enabled() -> bool:
    value = clean_text(os.environ.get("TAVILY_USE_MCP") or os.environ.get("TAVILY_SEARCH_BACKEND")).lower()
    if value in {"0", "false", "no", "sdk"}:
        return False
    return True


def tavily_mcp_required() -> bool:
    return clean_text(os.environ.get("TAVILY_REQUIRE_MCP")).lower() in {"1", "true", "yes"}
