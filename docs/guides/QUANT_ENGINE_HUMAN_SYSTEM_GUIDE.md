# Quant Engine Human System Guide

Operator-oriented overview of the current Quant Engine system. This guide describes what exists in source now (FastAPI backend + React frontend), how the major components fit together, and how to use the main workflows without reading code first.

## What The System Is

`quant_engine` is a research and operations workspace for:
- equity data ingestion and survivorship-safe loading,
- feature engineering and regime detection,
- model training/prediction/versioning,
- execution-aware backtesting and validation,
- autopilot strategy lifecycle and paper trading,
- Kalshi event-market ingestion/research,
- and web-based observability/controls via a React UI backed by FastAPI.

## Current Runtime Interfaces

### CLI / Scripts
- Root `run_*.py` scripts are the primary orchestration entrypoints for training, prediction, backtesting, autopilot, retraining, cache metadata rehydration, WRDS refresh, server startup, and Kalshi event pipeline runs.
- See `docs/operations/CLI_AND_WORKFLOW_RUNBOOK.md` for exact flags and workflows.

### Web App (Current UI)
- Frontend stack: React/Vite in `frontend/` (not Dash, not tkinter).
- Backend stack: FastAPI in `api/`.
- Dev mode: Vite frontend + FastAPI backend; Prod-like local mode: `run_server.py --static` serves both.

## Major Subsystems

1. Data (`data/`)
Loads and normalizes market data from WRDS/cache/providers, enforces quality checks and survivorship controls.
2. Features (`features/`)
Builds feature panels and targets for training/prediction (core + macro/options/intraday/research factors).
3. Regime (`regime/`)
Computes market regime labels and probabilities used across training, inference, risk, and UI diagnostics.
4. Models (`models/`)
Trains and serves regime-conditional ensembles, maintains versioning and champion governance, provides IV surface models.
5. Backtest + Risk (`backtest/`, `risk/`)
Simulates trades with execution costs and risk controls; runs validation and analytics.
6. Autopilot (`autopilot/`)
Generates candidate strategy variants, applies promotion gate, persists registry, and runs paper trading.
7. Kalshi (`kalshi/`)
Handles event-market ingestion, event-time storage, distribution reconstruction, event features, and walk-forward evaluation.
8. API + Frontend (`api/`, `frontend/`)
Provides operator-facing web UI and API endpoints over persisted artifacts and background jobs.

## Where Results And State Live

- `trained_models/`: model artifacts + registries (versioned and flat artifacts may coexist).
- `results/`: predictions/backtests/autopilot outputs and support metrics files.
- `data/cache/`: local OHLCV cache files and metadata sidecars.
- `data/kalshi.duckdb`: Kalshi event-time storage (default path, configurable in `config.py`).
- `api_jobs.db`: API background job store (default, configurable via `QE_API_JOB_DB_PATH`).

## Web UI Route Guide (Current React App)

### `/` → `DashboardPage`

- Tabs: `Portfolio Overview`, `Regime State`, `Model Performance`, `Feature Importance`, `Trade Log`, `Risk Metrics`
- Main page-local panels/components: `EquityCurveTab`, `FeatureImportanceTab`, `KPIGrid`, `ModelHealthTab`, `RegimeTab`, `RiskTab`, `TradeLogTab`, `useDashboardData`

### `/system-health` → `SystemHealthPage`

- Main page-local panels/components: `HealthCheckPanel`, `HealthRadar`, `HealthScoreCards`
- Data hooks (top-level imports): `useDetailedHealth`

### `/system-logs` → `LogsPage`

- Main page-local panels/components: `LogFilters`, `LogStream`

### `/data-explorer` → `DataExplorerPage`

- Main page-local panels/components: `CacheStatusPanel`, `CandlestickPanel`, `DataQualityReport`, `UniverseSelector`

### `/model-lab` → `ModelLabPage`

- Tabs: `Features`, `Feature Diff`, `Regime`, `Training`
- Main page-local panels/components: `FeatureDiffViewer`, `FeaturesTab`, `RegimeTab`, `TrainingTab`

### `/signal-desk` → `SignalDeskPage`

- Main page-local panels/components: `ConfidenceScatter`, `EnsembleDisagreement`, `SignalControls`, `SignalDistribution`, `SignalRankingsPanel`
- Data hooks (top-level imports): `useLatestSignals`

### `/backtest-risk` → `BacktestPage`

- Main page-local panels/components: `BacktestConfigPanel`, `BacktestResults`, `RunBacktestButton`
- Data hooks (top-level imports): `useConfig`

### `/iv-surface` → `IVSurfacePage`

- Tabs: `SVI Surface`, `Heston Surface`, `Arb-Aware SVI`
- Main page-local panels/components: `ArbAwareSVITab`, `HestonSurfaceTab`, `SVISurfaceTab`

### `/sp-comparison` → `BenchmarkPage`

- Main page-local panels/components: `BenchmarkChartGrid`, `BenchmarkMetricCards`
- Data hooks (top-level imports): `useBenchmarkComparison`, `useBenchmarkEquityCurves`, `useBenchmarkRollingMetrics`

### `/autopilot` → `AutopilotPage`

- Tabs: `Strategy Candidates`, `Paper Trading`, `Live P&L`, `Kalshi Events`, `Lifecycle`
- Main page-local panels/components: `KalshiEventsTab`, `LifecycleTimeline`, `PaperPnLTracker`, `PaperTradingTab`, `StrategyCandidatesTab`
- Data hooks (top-level imports): `useLatestCycle`

## Typical Operator Workflows

### Inspect system health
1. Start the web app (`docs/guides/WEB_APP_QUICK_START.md`).
2. Open `/system-health`.
3. Review domain scores/checks, then drill into `/data-explorer` or `/dashboard` as needed.

### Train / predict / backtest using the UI
1. Use `/model-lab` to submit training or prediction jobs (background jobs).
2. Monitor progress via job panels/SSE updates.
3. Use `/signal-desk` and `/backtest-risk` to inspect resulting artifacts served through the API.

### Run autopilot cycle
1. Use `/autopilot` to trigger a cycle (or run `run_autopilot.py`).
2. Review candidate/paper/lifecycle tabs.
3. Inspect `results/autopilot/*` for persisted source-of-truth state.

### Run Kalshi event research
1. Run `run_kalshi_event_pipeline.py` from CLI (there is no dedicated mounted Kalshi API router today).
2. Inspect generated artifacts/DB state and the autopilot/Kalshi UI tab surfaces that read persisted outputs.

## Important Caveats (Current Source)

- Historical docs/reports may reference removed Dash/tkinter UI stacks; use this guide + source-derived references as current truth.
- The frontend job type definitions currently drift from backend job status names (`pending/completed` vs `queued/succeeded`); the backend models in `api/jobs/models.py` are the canonical API contract.
- `api/services/kalshi_service.py` exists but is not mounted by `api/routers/__init__.py`.

## Where To Go Next

- `docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md` for system internals and flow diagrams
- `docs/reference/FRONTEND_UI_REFERENCE.md` for route/page/hook/component details
- `docs/reference/CONFIG_REFERENCE.md` for configuration and runtime patching
- Package `README.md` files for subsystem deep dives
