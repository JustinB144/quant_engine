# System Architecture and Flows

## Scope

This document describes the current runtime architecture of `quant_engine` as implemented in source today: FastAPI backend (`api/`) plus React/Vite frontend (`frontend/`), alongside the root CLI orchestration scripts (`run_*.py`).

It intentionally does not describe removed UI stacks (`dash_ui`, tkinter desktop app) except as historical context in archived reports.

## Top-Level Runtime Architecture

```mermaid
flowchart TD
    A[CLI Entrypoints\nrun_train / run_predict / run_backtest / run_autopilot / run_kalshi_event_pipeline] --> B[Core Engine Packages\ndata / features / regime / models / backtest / risk / autopilot / kalshi]
    B --> C[Artifacts & State\ntrained_models/\nresults/\ndata/cache/\ndata/kalshi.duckdb]
    D[FastAPI App\napi/main.py] --> E[Routers\napi/routers/*]
    E --> F[Services & Orchestrator\napi/services/* + api/orchestrator.py]
    F --> B
    F --> C
    E --> G[Job System\napi/jobs/* (SQLite + SSE)]
    G --> F
    H[React Frontend\nfrontend/src/*] -->|HTTP /api + SSE /api/jobs/*/events| D
    I[run_server.py] --> D
    I -->|--static| H
```

## Core Layers (Source-Verified)

- `data/`: provider abstraction, cache IO, WRDS/IBKR/Alpaca-adjacent ingestion helpers, quality checks, survivorship controls, intraday quality/quarantine, cross-source validation. NYSE TAQ Daily Product intraday OHLCV download via `run_wrds_taq_intraday_download.py` (2003-present, all 128 UNIVERSE_INTRADAY tickers, 6 timeframes).
- `features/`: feature/target computation pipeline plus macro/options/intraday/research factors.
- `regime/`: rule/HMM/jump/correlation regime detection and regime feature outputs.
- `models/`: training, prediction, versioning, governance, IV models, retrain triggers.
- `backtest/`: execution-aware trade simulation, validation, advanced validation, optimal execution helpers.
- `risk/`: reusable risk sizing/limits/stops/attribution/covariance/optimization/stress analytics.
- `autopilot/`: strategy discovery, promotion gate, registry, paper trader, autopilot cycle engine.
- `kalshi/`: event-market ingestion/storage/distribution/event features/walk-forward/promotion helpers.

## Backend Web Stack (FastAPI)

### App Lifecycle

- `api/main.py` creates the FastAPI app, registers CORS and error handlers, and lazily mounts routers from `api/routers/__init__.py`.
- Startup lifecycle initializes the async `JobStore` and launches a background retrain-monitor loop that invalidates dashboard caches when models become overdue.
- `api/deps/providers.py` exposes singleton providers for `ApiSettings`, `RuntimeConfig`, `JobStore`, `JobRunner`, and `CacheManager`.

### Routers / Services / Jobs Split

- Routers (`api/routers/*`) are thin: validate input, call services (often via `asyncio.to_thread`), wrap in `ApiResponse`, and submit jobs for long-running compute.
- Services (`api/services/*`) are synchronous adapters around engine packages and result files (`results/`, `trained_models/`, cache files).
- Long-running compute endpoints (`/api/models/train`, `/api/models/predict`, `/api/backtests/run`, `/api/autopilot/run-cycle`) create job records and execute in background threads through `api/jobs/runner.py`.
- `api/orchestrator.py` consolidates data->features->regimes->train/predict/backtest flows for API job executors.
- Current mounted router set also includes `diagnostics`, `regime`, and `risk` routers in addition to the dashboard/data/model/compute pages.

### Caching

- `api/cache/manager.py` is an in-memory per-process TTL cache.
- `api/cache/invalidation.py` provides event-driven invalidation hooks for train/backtest/data-refresh/config changes.
- Caches are used by read-heavy endpoints (dashboard, signals, health, benchmark, model health, feature importance/correlations, data status, selected data explorer reads).

### Jobs & Streaming

- `api/jobs/store.py`: SQLite-backed job persistence (`api_jobs.db` by default, configurable via `QE_API_JOB_DB_PATH`).
- `api/jobs/runner.py`: bounded-concurrency background execution + SSE event fanout.
- `api/routers/jobs.py`: list/get/cancel plus SSE event stream at `/api/jobs/{job_id}/events`.

## Frontend Stack (React/Vite)

- `frontend/src/main.tsx`: React 19 bootstrap with TanStack Query and `BrowserRouter`.
- `frontend/src/App.tsx`: lazy route loading with `AppShell` and `ErrorBoundary` keyed by route path.
- `frontend/src/api/client.ts`: wraps fetch, prefixes `/api`, and expects the shared `ApiResponse` envelope (`ok`, `data`, `meta`, `error`).
- `frontend/src/api/queries/*` and `frontend/src/api/mutations/*`: all page data access lives in query/mutation hooks.
- `frontend/src/hooks/useSSE.ts` + `useJobProgress.ts`: combines SSE and polling for background job progress.
- `frontend/src/store/*`: Zustand stores for UI/filter state.
- The frontend currently consumes a subset of mounted backend routes; some backend diagnostic/regime/risk endpoints are available for future UI expansion and automation.

## Primary End-to-End Flows

### 1. Train Model (CLI)

1. `run_train.py` parses CLI args (universe/horizon/years/feature mode/survivorship/recency).
2. Loads OHLCV via `data.loader` (`load_universe` or survivorship-safe path).
3. Builds features/targets via `features.pipeline.FeaturePipeline.compute_universe`.
4. Detects regimes via `regime.detector.RegimeDetector`.
5. Trains ensemble models via `models.trainer.ModelTrainer.train_ensemble`.
6. Writes artifacts under `trained_models/` and updates version/registry metadata.

### 2. Predict Signals (CLI or API Job)

1. `run_predict.py` or `api/jobs/predict_job.py` -> `api/orchestrator.PipelineOrchestrator.predict`.
2. Loads latest data + features + regimes, resolves model version, runs `models.predictor.EnsemblePredictor`.
3. Writes `results/predictions_{horizon}d.csv`.
4. `/api/signals/latest` serves the persisted predictions via `api/services/results_service.py`.

### 3. Backtest (CLI or API Job)

1. `run_backtest.py` or `api/jobs/backtest_job.py` -> orchestrator backtest path.
2. Uses `backtest.engine.Backtester` and optional validation/advanced validation modules.
3. Writes `results/backtest_{horizon}d_summary.json` + `results/backtest_{horizon}d_trades.csv`.
4. Dashboard/benchmark/backtests endpoints read these artifacts through `api/services/backtest_service.py` and `api/services/data_helpers.py`.

### 4. Autopilot Cycle (CLI or API Job)

1. `run_autopilot.py` or `api/jobs/autopilot_job.py` invokes `autopilot.engine.AutopilotEngine.run_cycle`.
2. Ensures predictor availability (ensemble or heuristic fallback), generates candidate variants, validates, promotes, and runs paper trading.
3. Persists outputs to `results/autopilot/latest_cycle.json`, `strategy_registry.json`, and `paper_state.json`.
4. `/api/autopilot/*` endpoints and the React `/autopilot` page consume those persisted states.

### 5. Kalshi Event Pipeline (CLI)

1. `run_kalshi_event_pipeline.py` orchestrates Kalshi API sync, storage writes, distribution reconstruction, event feature materialization, and optional walk-forward evaluation.
2. Persistent event-time state lives in `data/kalshi.duckdb` (default `KALSHI_DB_PATH`) via `kalshi.storage.EventTimeStore`.
3. Kalshi outputs are currently CLI/artifact-driven; there is no mounted FastAPI Kalshi router in `api/routers/__init__.py`.

### 6. Web App Runtime (Dev / Prod)

- Dev mode: run FastAPI (`python run_server.py`) and Vite (`cd frontend && npm run dev`). Vite proxies `/api` to port 8000.
- Prod-like mode: build frontend (`cd frontend && npm run build`) then `python run_server.py --static` to serve API + static SPA from one process/port.

## Mounted API Endpoint Inventory (Current)

- Mounted router modules: 15 (`jobs`, `system_health`, `dashboard`, `data_explorer`, `model_lab`, `signals`, `backtests`, `benchmark`, `logs`, `autopilot`, `config_mgmt`, `iv_surface`, `regime`, `risk`, `diagnostics`)
- Mounted endpoints: 48
- Newly surfaced endpoint groups relative to earlier docs include:
  - data explorer bars/indicators (`/api/data/ticker/{ticker}/bars`, `/indicators`, `/indicators/batch`)
  - health history (`/api/health/history`)
  - diagnostics (`/api/diagnostics`)
  - regime metadata (`/api/regime/metadata`)
  - risk factor exposures (`/api/risk/factor-exposures`)

For the full source-derived endpoint table (method/path/handler/router), use `docs/reference/SOURCE_API_REFERENCE.md`.

## Artifact and State Boundaries

- `trained_models/`: model artifacts, version metadata, registry/champion records.
- `results/`: predictions, backtests, autopilot reports/state, health/support artifacts.
- `data/cache/`: cached OHLCV files and metadata sidecars consumed by loaders and data status endpoints.
- `data/cache/quarantine/`: quarantined intraday datasets flagged by the intraday quality / cross-source validation workflow.
- `data/kalshi.duckdb`: event-time Kalshi/macroeconomic research store (DuckDB preferred, sqlite fallback supported by `EventTimeStore`).
- `api_jobs.db` (default): async compute job lifecycle store for the web API.

## Architecture Notes / Caveats

- The API is intentionally resilient to missing optional dependencies by lazy-importing routers and services; unavailable routers are skipped with a warning.
- Cache state is in-process memory only; running multiple API processes does not share cache entries.
- `api/services/kalshi_service.py` exists but is not routed, so Kalshi web UI functionality relies on persisted artifacts and autopilot page tabs rather than a dedicated `/api/kalshi/*` contract.
