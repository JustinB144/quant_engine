# Frontend UI Reference (React/Vite)

Source-derived reference for the active web UI in `frontend/` (current working tree).

## Stack Snapshot

- React + Vite + React Router + TanStack Query
- API client base prefix: `/api` (`frontend/src/api/client.ts`)
- Vite dev proxy: `/api` -> `http://localhost:8000` (`frontend/vite.config.ts`)

## App Bootstrap

- `frontend/src/main.tsx`: app bootstrap + providers + router.
- `frontend/src/App.tsx`: lazy page routes in `AppShell` with page-level `ErrorBoundary` wrappers.
- `frontend/src/components/layout/AppShell.tsx`: shell layout and navigation frame.

## Sidebar Navigation (`frontend/src/components/layout/Sidebar.tsx`)

| Label | Route | Icon |
|---|---|---|
| `Dashboard` | `/` | `ChartLine` |
| `System Health` | `/system-health` | `HeartPulse` |
| `System Logs` | `/system-logs` | `Terminal` |
| `Data Explorer` | `/data-explorer` | `Database` |
| `Model Lab` | `/model-lab` | `FlaskConical` |
| `Signal Desk` | `/signal-desk` | `Signal` |
| `Backtest & Risk` | `/backtest-risk` | `AreaChart` |
| `IV Surface` | `/iv-surface` | `Layers` |
| `S&P Comparison` | `/sp-comparison` | `Scale` |
| `Autopilot & Events` | `/autopilot` | `Bot` |

## Route Map (`frontend/src/App.tsx`)

| Route | Page Component |
|---|---|
| `/` | `DashboardPage` |
| `/system-health` | `SystemHealthPage` |
| `/system-logs` | `LogsPage` |
| `/data-explorer` | `DataExplorerPage` |
| `/model-lab` | `ModelLabPage` |
| `/signal-desk` | `SignalDeskPage` |
| `/backtest-risk` | `BacktestPage` |
| `/iv-surface` | `IVSurfacePage` |
| `/sp-comparison` | `BenchmarkPage` |
| `/autopilot` | `AutopilotPage` |
| `*` | `NotFoundPage` |

## API Endpoint Constants (`frontend/src/api/endpoints.ts`)

| Constant | Kind | Path Expression |
|---|---|---|
| `HEALTH_QUICK` | `const` | `'/health'` |
| `HEALTH_DETAILED` | `const` | `'/health/detailed'` |
| `DASHBOARD_SUMMARY` | `const` | `'/dashboard/summary'` |
| `DASHBOARD_REGIME` | `const` | `'/dashboard/regime'` |
| `DASHBOARD_RETURNS_DISTRIBUTION` | `const` | `'/dashboard/returns-distribution'` |
| `DASHBOARD_ROLLING_RISK` | `const` | `'/dashboard/rolling-risk'` |
| `DASHBOARD_EQUITY` | `const` | `'/dashboard/equity'` |
| `DASHBOARD_ATTRIBUTION` | `const` | `'/dashboard/attribution'` |
| `SIGNALS_LATEST` | `const` | `'/signals/latest'` |
| `BACKTESTS_LATEST` | `const` | `'/backtests/latest'` |
| `BACKTESTS_TRADES` | `const` | `'/backtests/latest/trades'` |
| `BACKTESTS_EQUITY_CURVE` | `const` | `'/backtests/latest/equity-curve'` |
| `BACKTESTS_RUN` | `const` | `'/backtests/run'` |
| `MODELS_VERSIONS` | `const` | `'/models/versions'` |
| `MODELS_HEALTH` | `const` | `'/models/health'` |
| `MODELS_FEATURES` | `const` | `'/models/features/importance'` |
| `MODELS_TRAIN` | `const` | `'/models/train'` |
| `MODELS_PREDICT` | `const` | `'/models/predict'` |
| `MODELS_FEATURE_CORRELATIONS` | `const` | `'/models/features/correlations'` |
| `DATA_UNIVERSE` | `const` | `'/data/universe'` |
| `DATA_STATUS` | `const` | `'/data/status'` |
| `DATA_TICKER` | `fn` | `(ticker: string) => `/data/ticker/${ticker}`` |
| `BENCHMARK_COMPARISON` | `const` | `'/benchmark/comparison'` |
| `BENCHMARK_EQUITY_CURVES` | `const` | `'/benchmark/equity-curves'` |
| `BENCHMARK_ROLLING_METRICS` | `const` | `'/benchmark/rolling-metrics'` |
| `LOGS` | `const` | `'/logs'` |
| `AUTOPILOT_LATEST_CYCLE` | `const` | `'/autopilot/latest-cycle'` |
| `AUTOPILOT_STRATEGIES` | `const` | `'/autopilot/strategies'` |
| `AUTOPILOT_PAPER_STATE` | `const` | `'/autopilot/paper-state'` |
| `AUTOPILOT_RUN_CYCLE` | `const` | `'/autopilot/run-cycle'` |
| `JOBS_LIST` | `const` | `'/jobs'` |
| `JOBS_GET` | `fn` | `(id: string) => `/jobs/${id}`` |
| `JOBS_EVENTS` | `fn` | `(id: string) => `/jobs/${id}/events`` |
| `JOBS_CANCEL` | `fn` | `(id: string) => `/jobs/${id}/cancel`` |
| `CONFIG` | `const` | `'/config'` |
| `CONFIG_STATUS` | `const` | `'/config/status'` |
| `REGIME_METADATA` | `const` | `'/regime/metadata'` |
| `IV_SURFACE_ARB_FREE` | `const` | `'/iv-surface/arb-free-svi'` |
| `DATA_TICKER_BARS` | `fn` | `(ticker: string) => `/data/ticker/${ticker}/bars`` |
| `DATA_TICKER_INDICATORS` | `fn` | `(ticker: string) => `/data/ticker/${ticker}/indicators`` |
| `DATA_TICKER_INDICATORS_BATCH` | `fn` | `(ticker: string) => `/data/ticker/${ticker}/indicators/batch`` |
| `HEALTH_HISTORY` | `const` | `'/health/history'` |

## Query Hooks (`frontend/src/api/queries/*`)

### `frontend/src/api/queries/useAutopilot.ts`
- Hooks: `useLatestCycle`, `usePaperState`, `useStrategies`

### `frontend/src/api/queries/useBacktests.ts`
- Hooks: `useEquityCurve`, `useLatestBacktest`, `useTrades`

### `frontend/src/api/queries/useBenchmark.ts`
- Hooks: `useBenchmarkComparison`, `useBenchmarkEquityCurves`, `useBenchmarkRollingMetrics`

### `frontend/src/api/queries/useConfig.ts`
- Hooks: `useConfig`, `useConfigStatus`

### `frontend/src/api/queries/useDashboard.ts`
- Hooks: `useAttribution`, `useDashboardRegime`, `useDashboardSummary`, `useEquityWithBenchmark`, `useRegimeMetadata`, `useReturnsDistribution`, `useRollingRisk`

### `frontend/src/api/queries/useData.ts`
- Hooks: `useDataStatus`, `useTickerBars`, `useTickerDetail`, `useTickerIndicators`, `useUniverse`

### `frontend/src/api/queries/useHealth.ts`
- Hooks: `useDetailedHealth`, `useHealthHistory`, `useQuickHealth`

### `frontend/src/api/queries/useIVSurface.ts`
- Hooks: `useArbFreeSVI`

### `frontend/src/api/queries/useJobs.ts`
- Hooks: `useJob`, `useJobsList`

### `frontend/src/api/queries/useLogs.ts`
- Hooks: `useLogs`

### `frontend/src/api/queries/useModels.ts`
- Hooks: `useFeatureCorrelations`, `useFeatureImportance`, `useModelHealth`, `useModelVersions`

### `frontend/src/api/queries/useSignals.ts`
- Hooks: `useLatestSignals`

## Mutation Hooks (`frontend/src/api/mutations/*`)

### `frontend/src/api/mutations/useCancelJob.ts`
- Hooks: `useCancelJob`

### `frontend/src/api/mutations/usePatchConfig.ts`
- Hooks: `usePatchConfig`

### `frontend/src/api/mutations/usePredict.ts`
- Hooks: `usePredict`

### `frontend/src/api/mutations/useRunBacktest.ts`
- Hooks: `useRunBacktest`

### `frontend/src/api/mutations/useRunCycle.ts`
- Hooks: `useRunCycle`

### `frontend/src/api/mutations/useTrainModel.ts`
- Hooks: `useTrainModel`

## Async Job Monitoring

- `frontend/src/hooks/useSSE.ts`: SSE helper used for job event streams.
- `frontend/src/hooks/useJobProgress.ts`: combines polling and SSE updates for background jobs.
- `frontend/src/components/job/JobMonitor.tsx`: renders job progress/status/error state.

## Frontend File Inventory (Counts)

- `frontend/src` files: 135
- Pages: 51 files / 4,597 LOC
- Components: 39 files / 2,834 LOC
- Hooks: 7 files / 256 LOC
- Query hooks: 12 files / 494 LOC
- Mutation hooks: 6 files / 95 LOC
- Stores: 3 files / 93 LOC
- Types: 11 files / 493 LOC
