# Multi-Agent Web Research System

A Python-based research assistant that plans web research, gathers evidence from search/pages/PDFs, indexes extracted content into ChromaDB, retrieves relevant context with hybrid RAG, and generates cited reports with Groq-hosted LLMs.

## Features

- LLM-assisted research planning with structured sub-questions and source tasks.
- Parallel browser/search execution with Tavily, Firecrawl, Playwright, HTTPX, and PDF extraction.
- Change detection across research runs using SQLite-backed browser history.
- RAG ingestion into persistent ChromaDB using Sentence Transformers embeddings.
- Hybrid retrieval using semantic search, BM25 keyword search, source authority scoring, URL diversification, and optional cross-encoder reranking.
- Synthesis and final report generation with citations and evidence-gap handling.
- GitHub Actions workflows for CI and RunPod deployment.

## Project Structure

```text
src/
  agents/        Planner, browser, change detection, synthesis, and report agents
  graph/         LangGraph workflow orchestration
  memory/        Shared JSON memory between agents
  rag/           Indexing, retrieval, and generation helpers
  tools/         Tavily, Firecrawl, Playwright, PDF, Groq retry, and text utilities

scripts/
  plan_research.py              Generate a research plan
  run_browser.py                Run browser tasks from a saved plan
  run_research_graph.py         Run the full planner-to-report workflow
  rag_retrieve.py               Query the Chroma RAG store
  evaluate_rag_retrieval.py     Evaluate retrieval quality

tests/          Unit and workflow tests
data/           Local runtime outputs, browser history, and ChromaDB store
```

## Requirements

- Python 3.11
- Node.js 22+ if using Tavily MCP through `npx`
- API keys for the services you enable:
  - `GROQ_API_KEY`
  - `TAVILY_API_KEY`
  - `FIRECRAWL_API_KEY` optional, HTTPX fallback is used when unavailable

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
playwright install chromium
```

Create `.env` in the project root:

```bash
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key

RESEARCH_PLANNER_MODEL=openai/gpt-oss-120b
RAG_GENERATION_MODEL=llama-3.1-8b-instant
RAG_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
RAG_EMBEDDING_DEVICE=auto
```

## Usage

Run the full research workflow:

```bash
python scripts/run_research_graph.py \
  --objective "Compare OpenAI, Anthropic, and Google Gemini API pricing and capabilities"
```

This updates:

```text
data/shared_memory.json
data/browser_history.db
data/chroma/
data/reports/
```

## Workflow

```text
Objective
  -> PlannerAgent creates sub-questions and source tasks
  -> BrowserAgent searches/scrapes sources in parallel
  -> ChangeDetectionAgent compares current and previous runs
  -> RAG indexing chunks and embeds useful source content
  -> SynthesisAgent retrieves evidence and builds report context
  -> ReportAgent generates a cited Markdown report
```

## Deployment

The repository includes:

- `.github/workflows/ci.yml` for pull-request checks.
- `.github/workflows/deploy-runpod.yml` for deploying to a RunPod instance over SSH.

Required GitHub secrets/variables for RunPod deployment include:

```text
RUNPOD_HOST
RUNPOD_PORT
RUNPOD_SSH_PRIVATE_KEY
RUNPOD_DEPLOY_DIR
GROQ_API_KEY
TAVILY_API_KEY
FIRECRAWL_API_KEY
RAG_EMBEDDING_MODEL
RAG_EMBEDDING_DEVICE
```

## Notes

- `data/` contains local runtime artifacts and should be treated as generated state.
- Use `RAG_EMBEDDING_DEVICE=cuda` on GPU machines and `auto` for local development.
- If the embedding model or chunking strategy changes, rebuild the Chroma index.
