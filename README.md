# Multi-agent-web-research-system

## Deploy to RunPod with GitHub Actions

This repo includes a GitHub Actions workflow at `.github/workflows/deploy-runpod.yml`.
It deploys to RunPod host `157.157.221.29` over SSH, installs Python dependencies in a
remote `.venv`, installs Playwright Chromium, writes `.env`, and runs an import smoke test.

Configure these repository secrets in GitHub:

- `RUNPOD_SSH_PRIVATE_KEY`: private SSH key allowed to connect to the RunPod instance
- `RUNPOD_DEPLOY_DIR`: optional remote path, default `/workspace/Multi-agent-web-research-system`
- `GROQ_API_KEY`
- `TAVILY_API_KEY`
- `FIRECRAWL_API_KEY`
- `RAG_EMBEDDING_MODEL`: optional, for example `sentence-transformers/all-MiniLM-L6-v2`

Trigger deployment by pushing to `main` or manually from the GitHub Actions tab.

The workflow connects as SSH user `root` on SSH port `19805`.
