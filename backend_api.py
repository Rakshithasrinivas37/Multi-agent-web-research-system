import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.report_agent import slugify_filename
from src.graph.research_workflow import run_research_graph
from src.tools.progress import reset_progress_callback, set_progress_callback


load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
ENV_LOCK = asyncio.Lock()

app = FastAPI(title="Multi-Agent Web Research API")


class ResearchRequest(BaseModel):
    objective: str = Field(..., min_length=1)
    groq_api_key: str = Field(..., min_length=1)
    tavily_api_key: str = Field(..., min_length=1)
    firecrawl_api_key: str = ""
    model: str = DEFAULT_GROQ_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_device: str = "auto"
    max_concurrency: int = Field(default=3, ge=1, le=10)


class ResearchResponse(BaseModel):
    objective: str
    report: str
    filename: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    agent_timings: list[dict[str, Any]] = Field(default_factory=list)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    objective = request.objective.strip()
    if not objective:
        raise HTTPException(status_code=400, detail="Research objective is required.")

    env_updates = {
        "GROQ_API_KEY": request.groq_api_key.strip(),
        "TAVILY_API_KEY": request.tavily_api_key.strip(),
        "RESEARCH_PLANNER_MODEL": request.model.strip() or DEFAULT_GROQ_MODEL,
        "RAG_GENERATION_MODEL": request.model.strip() or DEFAULT_GROQ_MODEL,
        "RAG_EMBEDDING_MODEL": request.embedding_model.strip() or DEFAULT_EMBEDDING_MODEL,
        "RAG_EMBEDDING_DEVICE": request.embedding_device.strip() or "auto",
    }
    if request.firecrawl_api_key.strip():
        env_updates["FIRECRAWL_API_KEY"] = request.firecrawl_api_key.strip()

    async with ENV_LOCK:
        async with temporary_env(env_updates):
            state = await run_backend_research(objective, request)

    workflow_errors = state.get("errors") or []
    if workflow_errors:
        raise HTTPException(status_code=500, detail=workflow_errors)

    report_payload = state.get("report") or {}
    report_markdown = str(report_payload.get("report") or "").strip()
    if not report_markdown:
        raise HTTPException(status_code=500, detail="The workflow completed, but no report was generated.")

    return ResearchResponse(
        objective=objective,
        report=report_markdown,
        filename=report_filename(objective),
        sources=report_payload.get("sources") or [],
        agent_timings=state.get("agent_timings") or [],
    )


@app.post("/research/stream")
async def research_stream(request: ResearchRequest) -> StreamingResponse:
    objective = request.objective.strip()
    if not objective:
        raise HTTPException(status_code=400, detail="Research objective is required.")

    return StreamingResponse(
        stream_research_events(objective, request),
        media_type="application/x-ndjson",
    )


async def run_backend_research(objective: str, request: ResearchRequest) -> dict[str, Any]:
    return await run_research_graph(
        objective=objective,
        memory_path=str(PROJECT_ROOT / "data" / "shared_memory.json"),
        history_db_path=str(PROJECT_ROOT / "data" / "browser_history.db"),
        chroma_path=str(PROJECT_ROOT / "data" / "chroma"),
        max_concurrency=request.max_concurrency,
        model=request.model.strip() or DEFAULT_GROQ_MODEL,
    )


async def stream_research_events(objective: str, request: ResearchRequest):
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def push_event(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def runner() -> None:
        env_updates = request_env_updates(request)
        token = set_progress_callback(push_event)
        try:
            push_event({"event": "workflow_started", "message": "Research workflow started", "agent": "", "tool": ""})
            async with ENV_LOCK:
                async with temporary_env(env_updates):
                    state = await run_backend_research(objective, request)

            workflow_errors = state.get("errors") or []
            if workflow_errors:
                push_event({"event": "error", "message": "Research workflow failed", "errors": workflow_errors})
                return

            report_payload = state.get("report") or {}
            report_markdown = str(report_payload.get("report") or "").strip()
            if not report_markdown:
                push_event({"event": "error", "message": "The workflow completed, but no report was generated."})
                return

            push_event(
                {
                    "event": "report",
                    "message": "Report generated",
                    "objective": objective,
                    "report": report_markdown,
                    "filename": report_filename(objective),
                    "sources": report_payload.get("sources") or [],
                    "agent_timings": state.get("agent_timings") or [],
                }
            )
            push_event({"event": "workflow_completed", "message": "Research workflow completed"})
        except Exception as error:
            push_event({"event": "error", "message": str(error)})
        finally:
            reset_progress_callback(token)
            loop.call_soon_threadsafe(queue.put_nowait, None)

    task = asyncio.create_task(runner())
    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield json.dumps(event) + "\n"
    finally:
        await task


def request_env_updates(request: ResearchRequest) -> dict[str, str]:
    env_updates = {
        "GROQ_API_KEY": request.groq_api_key.strip(),
        "TAVILY_API_KEY": request.tavily_api_key.strip(),
        "RESEARCH_PLANNER_MODEL": request.model.strip() or DEFAULT_GROQ_MODEL,
        "RAG_GENERATION_MODEL": request.model.strip() or DEFAULT_GROQ_MODEL,
        "RAG_EMBEDDING_MODEL": request.embedding_model.strip() or DEFAULT_EMBEDDING_MODEL,
        "RAG_EMBEDDING_DEVICE": request.embedding_device.strip() or "auto",
    }
    if request.firecrawl_api_key.strip():
        env_updates["FIRECRAWL_API_KEY"] = request.firecrawl_api_key.strip()
    return env_updates


@asynccontextmanager
async def temporary_env(updates: dict[str, str]):
    previous_values = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def report_filename(objective: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{slugify_filename(objective)}-{timestamp}.md"
