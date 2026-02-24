# Web App Quick Start (FastAPI + React/Vite)

Quick start for the current `quant_engine` web stack. This replaces the old Dash quick-start workflow.

## What Runs

- Backend API: FastAPI app in `api/` (`api/main.py`, launched via `run_server.py` or `qe-server`).
- Frontend UI: React/Vite app in `frontend/` (`frontend/src/*`).

## Prerequisites

- Python 3.10+ (per `pyproject.toml`)
- Node.js/npm (for `frontend/`)
- Optional data/model artifacts in `data/cache/`, `trained_models/`, `results/` for full UI functionality

## Python Installation

Install the API dependencies (minimum web stack):

```bash
pip install -e '.[api]'
```

Optional extras in `pyproject.toml` include `api`, `ml`, `data`, `charts`, `full`, and `dev`.

## Frontend Installation

```bash
cd frontend
npm install
```

Frontend package: `quant-engine-frontend` v2.0.0

## Development Mode (Two Processes)

### Terminal 1: Backend API

```bash
python run_server.py --reload
```

Notes:
- Default backend host/port from `run_server.py`: `0.0.0.0:8000`
- OpenAPI docs available at `http://localhost:8000/docs`

### Terminal 2: Frontend (Vite)

```bash
cd frontend
npm run dev
```

Notes:
- Vite dev server defaults to `http://localhost:5173` (`frontend/vite.config.ts`).
- `/api` requests are proxied to `http://localhost:8000`.

## Production-Like Local Run (Single Process)

Build the frontend and let `run_server.py` serve static assets + API:

```bash
cd frontend && npm run build
cd ..
python run_server.py --static
```

`run_server.py --static` expects `frontend/dist` to exist and mounts `/assets` plus an SPA fallback route.

## Useful URLs

- Frontend (dev): `http://localhost:5173/`
- Backend API docs: `http://localhost:8000/docs`
- Backend OpenAPI JSON: `http://localhost:8000/openapi.json`

## Common First Checks

- `GET /api/health` and `GET /api/health/detailed` respond with `ApiResponse` envelopes.
- `/data-explorer` loads tickers (requires cache metadata/files under `data/cache/`).
- `/signal-desk` and `/backtest-risk` show real results only if `results/predictions_*.csv` / `results/backtest_*` artifacts exist.

## Related Docs

- `docs/operations/CLI_AND_WORKFLOW_RUNBOOK.md`
- `docs/reference/FRONTEND_UI_REFERENCE.md`
- `docs/reference/CONFIG_REFERENCE.md`
