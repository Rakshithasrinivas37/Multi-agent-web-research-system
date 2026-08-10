import asyncio
import os
from datetime import datetime
from pathlib import Path
import sys

import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.report_agent import slugify_filename
from src.graph.research_workflow import run_research_graph


load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


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

    st.session_state["research_config"] = {
        "groq_api_key": groq_api_key.strip(),
        "tavily_api_key": tavily_api_key.strip(),
        "firecrawl_api_key": firecrawl_api_key.strip(),
        "model": model.strip() or DEFAULT_GROQ_MODEL,
        "embedding_model": embedding_model.strip() or DEFAULT_EMBEDDING_MODEL,
        "embedding_device": embedding_device,
        "max_concurrency": int(max_concurrency),
    }


def run_research(objective: str) -> None:
    objective = objective.strip()
    config = st.session_state.get("research_config", {})
    errors = validate_inputs(objective, config)
    if errors:
        for error in errors:
            st.error(error)
        return

    apply_runtime_config(config)
    clear_previous_report()

    memory_path = PROJECT_ROOT / "data" / "streamlit_shared_memory.json"
    history_db_path = PROJECT_ROOT / "data" / "browser_history.db"
    chroma_path = PROJECT_ROOT / "data" / "chroma"

    with st.status("Running research workflow...", expanded=True) as status:
        st.write("Planning research tasks")
        st.write("Collecting and indexing evidence")
        st.write("Generating cited report")
        try:
            state = asyncio.run(
                run_research_graph(
                    objective=objective,
                    memory_path=str(memory_path),
                    history_db_path=str(history_db_path),
                    chroma_path=str(chroma_path),
                    max_concurrency=config["max_concurrency"],
                    model=config["model"],
                )
            )
        except Exception as error:
            status.update(label="Research failed", state="error")
            st.error(str(error))
            return

        workflow_errors = state.get("errors") or []
        if workflow_errors:
            status.update(label="Research completed with errors", state="error")
            for error in workflow_errors:
                st.error(str(error))
            return

        report_payload = state.get("report") or {}
        report_markdown = str(report_payload.get("report") or "").strip()
        if not report_markdown:
            status.update(label="Research completed without a report", state="error")
            st.error("The workflow completed, but no report content was returned.")
            return

        st.session_state["generated_report"] = report_markdown
        st.session_state["generated_report_filename"] = report_filename(objective)
        status.update(label="Report generated", state="complete")


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


def apply_runtime_config(config: dict[str, object]) -> None:
    os.environ["GROQ_API_KEY"] = str(config["groq_api_key"])
    os.environ["TAVILY_API_KEY"] = str(config["tavily_api_key"])
    if config.get("firecrawl_api_key"):
        os.environ["FIRECRAWL_API_KEY"] = str(config["firecrawl_api_key"])
    os.environ["RESEARCH_PLANNER_MODEL"] = str(config["model"])
    os.environ["RAG_GENERATION_MODEL"] = str(config["model"])
    os.environ["RAG_EMBEDDING_MODEL"] = str(config["embedding_model"])
    os.environ["RAG_EMBEDDING_DEVICE"] = str(config["embedding_device"])


def clear_previous_report() -> None:
    for key in ("generated_report", "generated_report_filename"):
        st.session_state.pop(key, None)


def report_filename(objective: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{slugify_filename(objective)}-{timestamp}.md"


def device_index(value: str) -> int:
    options = ["auto", "cpu", "mps", "cuda"]
    try:
        return options.index((value or "auto").lower())
    except ValueError:
        return 0


if __name__ == "__main__":
    main()
