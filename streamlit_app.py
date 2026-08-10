import json
import os
from pathlib import Path
import sys

import httpx
import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


def main() -> None:
    st.set_page_config(page_title="Multi-Agent Web Research", layout="wide")
    st.title("Multi-Agent Web Research")

    configure_sidebar()

    objective = st.text_area(
        "Research objective",
        placeholder="Example: Compare OpenAI, Anthropic, and Google Gemini API pricing and capabilities",
        height=120,
    )

    if st.button("Research", type="primary"):
        run_research(objective)

    render_report_download()


def configure_sidebar() -> None:
    st.sidebar.header("Configuration")
    groq_api_key = st.sidebar.text_input(
        "GROQ API key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
    )
    tavily_api_key = st.sidebar.text_input(
        "Tavily API key",
        value=os.environ.get("TAVILY_API_KEY", ""),
        type="password",
    )

    with st.sidebar.expander("Advanced", expanded=False):
        firecrawl_api_key = st.text_input(
            "Firecrawl API key",
            value=os.environ.get("FIRECRAWL_API_KEY", ""),
            type="password",
        )
        model = st.text_input(
            "Groq model",
            value=os.environ.get("RESEARCH_PLANNER_MODEL", DEFAULT_GROQ_MODEL),
        )
        embedding_model = st.text_input(
            "Embedding model",
            value=os.environ.get("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        )
        embedding_device = st.selectbox(
            "Embedding device",
            options=["auto", "cpu", "mps", "cuda"],
            index=device_index(os.environ.get("RAG_EMBEDDING_DEVICE", "auto")),
        )
        max_concurrency = st.number_input("Max concurrency", min_value=1, max_value=10, value=3)
        backend_url = st.text_input(
            "Backend URL",
            value=os.environ.get("RESEARCH_BACKEND_URL", DEFAULT_BACKEND_URL),
        )

    st.session_state["research_config"] = {
        "groq_api_key": groq_api_key.strip(),
        "tavily_api_key": tavily_api_key.strip(),
        "firecrawl_api_key": firecrawl_api_key.strip(),
        "model": model.strip() or DEFAULT_GROQ_MODEL,
        "embedding_model": embedding_model.strip() or DEFAULT_EMBEDDING_MODEL,
        "embedding_device": embedding_device,
        "max_concurrency": int(max_concurrency),
        "backend_url": backend_url.strip().rstrip("/") or DEFAULT_BACKEND_URL,
    }


def run_research(objective: str) -> None:
    objective = objective.strip()
    config = st.session_state.get("research_config", {})
    errors = validate_inputs(objective, config)
    if errors:
        for error in errors:
            st.error(error)
        return

    clear_previous_report()

    with st.status("Running research workflow...", expanded=True) as status:
        current_agent_slot = st.empty()
        tool_log_slot = st.empty()
        tools_by_agent: dict[str, list[str]] = {}
        payload: dict[str, object] = {}
        try:
            for event in stream_research_backend(objective, config):
                payload = handle_progress_event(event, status, current_agent_slot, tool_log_slot, tools_by_agent, payload)
        except Exception as error:
            status.update(label="Research failed", state="error")
            st.error(str(error))
            return

        report_markdown = str(payload.get("report") or "").strip()
        if not report_markdown:
            status.update(label="Research completed without a report", state="error")
            st.error("The workflow completed, but no report content was returned.")
            return

        st.session_state["generated_report"] = report_markdown
        st.session_state["generated_report_filename"] = payload.get("filename") or "research-report.md"
        status.update(label="Report generated", state="complete")


def stream_research_backend(objective: str, config: dict[str, object]):
    backend_url = str(config["backend_url"])
    request_payload = {
        "objective": objective,
        "groq_api_key": config["groq_api_key"],
        "tavily_api_key": config["tavily_api_key"],
        "firecrawl_api_key": config["firecrawl_api_key"],
        "model": config["model"],
        "embedding_model": config["embedding_model"],
        "embedding_device": config["embedding_device"],
        "max_concurrency": config["max_concurrency"],
    }
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", f"{backend_url}/research/stream", json=request_payload) as response:
            if response.is_error:
                raise RuntimeError(backend_error_message(response))
            for line in response.iter_lines():
                if line:
                    yield json.loads(line)


def handle_progress_event(
    event: dict[str, object],
    status,
    current_agent_slot,
    tool_log_slot,
    tools_by_agent: dict[str, list[str]],
    payload: dict[str, object],
) -> dict[str, object]:
    event_type = str(event.get("event") or "")
    agent = str(event.get("agent") or "")
    tool = str(event.get("tool") or "")
    message = str(event.get("message") or "")

    if event_type == "workflow_started":
        status.update(label="Research workflow started", state="running")
        current_agent_slot.info("Starting research workflow")
    elif event_type == "agent_started":
        status.update(label=f"Running {agent} agent", state="running")
        current_agent_slot.info(f"Current agent: {agent}")
    elif event_type == "agent_completed":
        current_agent_slot.success(message or f"{agent} completed")
    elif event_type == "tool_called":
        agent_name = agent or "workflow"
        tool_text = f"{tool}: {message}" if tool else message
        tools_by_agent.setdefault(agent_name, []).append(tool_text)
        current_agent_slot.info(f"Current agent: {agent_name}")
        render_tool_log(tool_log_slot, tools_by_agent)
    elif event_type == "report":
        payload = event
        current_agent_slot.success("Report generated")
    elif event_type == "workflow_completed":
        status.update(label="Research workflow completed", state="complete")
    elif event_type == "error":
        raise RuntimeError(message or str(event.get("errors") or "Research failed"))

    return payload


def render_tool_log(tool_log_slot, tools_by_agent: dict[str, list[str]]) -> None:
    lines = ["**Tools called**"]
    for agent, tools in tools_by_agent.items():
        lines.append(f"\n**{agent}**")
        for tool_call in tools[-12:]:
            lines.append(f"- `{tool_call}`")
    tool_log_slot.markdown("\n".join(lines))


def backend_error_message(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = response.text
    return f"Backend request failed ({response.status_code}): {detail}"


def render_report_download() -> None:
    report = st.session_state.get("generated_report")
    if not report:
        return

    st.divider()
    st.subheader("Generated Report")
    st.download_button(
        "Download report",
        data=report,
        file_name=st.session_state.get("generated_report_filename", "research-report.md"),
        mime="text/markdown",
        use_container_width=True,
    )
    with st.expander("Preview report", expanded=True):
        st.markdown(report)


def validate_inputs(objective: str, config: dict[str, object]) -> list[str]:
    errors = []
    if not objective:
        errors.append("Enter a research objective.")
    if not config.get("groq_api_key"):
        errors.append("Enter a GROQ API key.")
    if not config.get("tavily_api_key"):
        errors.append("Enter a Tavily API key.")
    return errors


def clear_previous_report() -> None:
    for key in ("generated_report", "generated_report_filename"):
        st.session_state.pop(key, None)


def device_index(value: str) -> int:
    options = ["auto", "cpu", "mps", "cuda"]
    try:
        return options.index((value or "auto").lower())
    except ValueError:
        return 0


if __name__ == "__main__":
    main()
