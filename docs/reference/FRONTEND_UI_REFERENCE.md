# Frontend UI Reference (React/Vite)

Source-derived reference for the active web UI in `frontend/` (React 19 + Vite + TanStack Query + React Router).

## Stack Snapshot

- Package: `quant-engine-frontend` v2.0.0
- API client base URL prefix: `/api` (from `frontend/src/api/client.ts`)
- Vite dev server proxy: `/api` -> `http://localhost:8000` (from `frontend/vite.config.ts`)

## App Bootstrap

- `frontend/src/main.tsx`: mounts React app, wraps with `QueryClientProvider` and `BrowserRouter`.
- `frontend/src/App.tsx`: lazy-loads page routes under `AppShell` and page-level `ErrorBoundary` wrappers.
- `frontend/src/components/layout/AppShell.tsx`: application shell (sidebar/status bar/page layout container).

## Sidebar Navigation (Source of User-Facing Page Labels)

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

## Top-Level Page Composition

### `AutopilotPage` (/autopilot)

- File: `frontend/src/pages/AutopilotPage.tsx` (87 lines)
- Tabs: `Strategy Candidates`, `Paper Trading`, `Live P&L`, `Kalshi Events`, `Lifecycle`
- Page-local subcomponents: `KalshiEventsTab`, `LifecycleTimeline`, `PaperPnLTracker`, `PaperTradingTab`, `StrategyCandidatesTab`
- Query hooks imported directly: `useLatestCycle`

### `BacktestPage` (/backtest-risk)

- File: `frontend/src/pages/BacktestPage.tsx` (60 lines)
- Page-local subcomponents: `BacktestConfigPanel`, `BacktestResults`, `RunBacktestButton`
- Query hooks imported directly: `useConfig`

### `BenchmarkPage` (/sp-comparison)

- File: `frontend/src/pages/BenchmarkPage.tsx` (43 lines)
- Page-local subcomponents: `BenchmarkChartGrid`, `BenchmarkMetricCards`
- Query hooks imported directly: `useBenchmarkComparison`, `useBenchmarkEquityCurves`, `useBenchmarkRollingMetrics`

### `DashboardPage` (/)

- File: `frontend/src/pages/DashboardPage.tsx` (80 lines)
- Tabs: `Portfolio Overview`, `Regime State`, `Model Performance`, `Feature Importance`, `Trade Log`, `Risk Metrics`
- Page-local subcomponents: `EquityCurveTab`, `FeatureImportanceTab`, `KPIGrid`, `ModelHealthTab`, `RegimeTab`, `RiskTab`, `TradeLogTab`, `useDashboardData`

### `DataExplorerPage` (/data-explorer)

- File: `frontend/src/pages/DataExplorerPage.tsx` (32 lines)
- Page-local subcomponents: `CacheStatusPanel`, `CandlestickPanel`, `DataQualityReport`, `UniverseSelector`
- Zustand store hooks: `useFilterStore`

### `IVSurfacePage` (/iv-surface)

- File: `frontend/src/pages/IVSurfacePage.tsx` (27 lines)
- Tabs: `SVI Surface`, `Heston Surface`, `Arb-Aware SVI`
- Page-local subcomponents: `ArbAwareSVITab`, `HestonSurfaceTab`, `SVISurfaceTab`

### `LogsPage` (/system-logs)

- File: `frontend/src/pages/LogsPage.tsx` (25 lines)
- Page-local subcomponents: `LogFilters`, `LogStream`

### `ModelLabPage` (/model-lab)

- File: `frontend/src/pages/ModelLabPage.tsx` (30 lines)
- Tabs: `Features`, `Feature Diff`, `Regime`, `Training`
- Page-local subcomponents: `FeatureDiffViewer`, `FeaturesTab`, `RegimeTab`, `TrainingTab`

### `NotFoundPage` (*)

- File: `frontend/src/pages/NotFoundPage.tsx` (25 lines)

### `SignalDeskPage` (/signal-desk)

- File: `frontend/src/pages/SignalDeskPage.tsx` (59 lines)
- Page-local subcomponents: `ConfidenceScatter`, `EnsembleDisagreement`, `SignalControls`, `SignalDistribution`, `SignalRankingsPanel`
- Query hooks imported directly: `useLatestSignals`
- Zustand store hooks: `useFilterStore`

### `SystemHealthPage` (/system-health)

- File: `frontend/src/pages/SystemHealthPage.tsx` (65 lines)
- Page-local subcomponents: `HealthCheckPanel`, `HealthRadar`, `HealthScoreCards`
- Query hooks imported directly: `useDetailedHealth`

## API Endpoint Constants (`frontend/src/api/endpoints.ts`) 

| Constant | Kind | Path Expression |
|---|---|---|
| `HEALTH_QUICK` | `const` | `/health` |
| `HEALTH_DETAILED` | `const` | `/health/detailed` |
| `DASHBOARD_SUMMARY` | `const` | `/dashboard/summary` |
| `DASHBOARD_REGIME` | `const` | `/dashboard/regime` |
| `DASHBOARD_RETURNS_DISTRIBUTION` | `const` | `/dashboard/returns-distribution` |
| `DASHBOARD_ROLLING_RISK` | `const` | `/dashboard/rolling-risk` |
| `DASHBOARD_EQUITY` | `const` | `/dashboard/equity` |
| `DASHBOARD_ATTRIBUTION` | `const` | `/dashboard/attribution` |
| `SIGNALS_LATEST` | `const` | `/signals/latest` |
| `BACKTESTS_LATEST` | `const` | `/backtests/latest` |
| `BACKTESTS_TRADES` | `const` | `/backtests/latest/trades` |
| `BACKTESTS_EQUITY_CURVE` | `const` | `/backtests/latest/equity-curve` |
| `BACKTESTS_RUN` | `const` | `/backtests/run` |
| `MODELS_VERSIONS` | `const` | `/models/versions` |
| `MODELS_HEALTH` | `const` | `/models/health` |
| `MODELS_FEATURES` | `const` | `/models/features/importance` |
| `MODELS_TRAIN` | `const` | `/models/train` |
| `MODELS_PREDICT` | `const` | `/models/predict` |
| `MODELS_FEATURE_CORRELATIONS` | `const` | `/models/features/correlations` |
| `DATA_UNIVERSE` | `const` | `/data/universe` |
| `DATA_STATUS` | `const` | `/data/status` |
| `DATA_TICKER` | `fn` | `/data/ticker/${ticker}` |
| `BENCHMARK_COMPARISON` | `const` | `/benchmark/comparison` |
| `BENCHMARK_EQUITY_CURVES` | `const` | `/benchmark/equity-curves` |
| `BENCHMARK_ROLLING_METRICS` | `const` | `/benchmark/rolling-metrics` |
| `LOGS` | `const` | `/logs` |
| `AUTOPILOT_LATEST_CYCLE` | `const` | `/autopilot/latest-cycle` |
| `AUTOPILOT_STRATEGIES` | `const` | `/autopilot/strategies` |
| `AUTOPILOT_PAPER_STATE` | `const` | `/autopilot/paper-state` |
| `AUTOPILOT_RUN_CYCLE` | `const` | `/autopilot/run-cycle` |
| `JOBS_LIST` | `const` | `/jobs` |
| `JOBS_GET` | `fn` | `/jobs/${id}` |
| `JOBS_EVENTS` | `fn` | `/jobs/${id}/events` |
| `JOBS_CANCEL` | `fn` | `/jobs/${id}/cancel` |
| `CONFIG` | `const` | `/config` |
| `CONFIG_STATUS` | `const` | `/config/status` |
| `IV_SURFACE_ARB_FREE` | `const` | `/iv-surface/arb-free-svi` |

## Query Hooks (`frontend/src/api/queries/*`) 

### `frontend/src/api/queries/useAutopilot.ts`

- Hooks: `useLatestCycle`, `useStrategies`, `usePaperState`
- Endpoint constants imported: `AUTOPILOT_LATEST_CYCLE`, `AUTOPILOT_PAPER_STATE`, `AUTOPILOT_STRATEGIES`

### `frontend/src/api/queries/useBacktests.ts`

- Hooks: `useLatestBacktest`, `useTrades`, `useEquityCurve`
- Endpoint constants imported: `BACKTESTS_EQUITY_CURVE`, `BACKTESTS_LATEST`, `BACKTESTS_TRADES`

### `frontend/src/api/queries/useBenchmark.ts`

- Hooks: `useBenchmarkComparison`, `useBenchmarkEquityCurves`, `useBenchmarkRollingMetrics`
- Endpoint constants imported: `BENCHMARK_COMPARISON`, `BENCHMARK_EQUITY_CURVES`, `BENCHMARK_ROLLING_METRICS`

### `frontend/src/api/queries/useConfig.ts`

- Hooks: `useConfig`, `useConfigStatus`
- Endpoint constants imported: `CONFIG`, `CONFIG_STATUS`

### `frontend/src/api/queries/useDashboard.ts`

- Hooks: `useDashboardSummary`, `useDashboardRegime`, `useReturnsDistribution`, `useRollingRisk`, `useEquityWithBenchmark`, `useAttribution`
- Endpoint constants imported: `DASHBOARD_ATTRIBUTION`, `DASHBOARD_EQUITY`, `DASHBOARD_REGIME`, `DASHBOARD_RETURNS_DISTRIBUTION`, `DASHBOARD_ROLLING_RISK`, `DASHBOARD_SUMMARY`

### `frontend/src/api/queries/useData.ts`

- Hooks: `useUniverse`, `useDataStatus`, `useTickerDetail`
- Endpoint constants imported: `DATA_STATUS`, `DATA_TICKER`, `DATA_UNIVERSE`

### `frontend/src/api/queries/useHealth.ts`

- Hooks: `useQuickHealth`, `useDetailedHealth`
- Endpoint constants imported: `HEALTH_DETAILED`, `HEALTH_QUICK`

### `frontend/src/api/queries/useIVSurface.ts`

- Hooks: `useArbFreeSVI`
- Endpoint constants imported: `IV_SURFACE_ARB_FREE`

### `frontend/src/api/queries/useJobs.ts`

- Hooks: `useJobsList`, `useJob`
- Endpoint constants imported: `JOBS_GET`, `JOBS_LIST`

### `frontend/src/api/queries/useLogs.ts`

- Hooks: `useLogs`
- Endpoint constants imported: `LOGS`

### `frontend/src/api/queries/useModels.ts`

- Hooks: `useModelVersions`, `useModelHealth`, `useFeatureImportance`, `useFeatureCorrelations`
- Endpoint constants imported: `MODELS_FEATURES`, `MODELS_FEATURE_CORRELATIONS`, `MODELS_HEALTH`, `MODELS_VERSIONS`

### `frontend/src/api/queries/useSignals.ts`

- Hooks: `useLatestSignals`
- Endpoint constants imported: `SIGNALS_LATEST`

## Mutation Hooks (`frontend/src/api/mutations/*`) 

### `frontend/src/api/mutations/useCancelJob.ts`

- Hooks: `useCancelJob`
- Endpoint constants imported: `JOBS_CANCEL`

### `frontend/src/api/mutations/usePatchConfig.ts`

- Hooks: `usePatchConfig`
- Endpoint constants imported: `CONFIG`

### `frontend/src/api/mutations/usePredict.ts`

- Hooks: `usePredict`
- Endpoint constants imported: `MODELS_PREDICT`

### `frontend/src/api/mutations/useRunBacktest.ts`

- Hooks: `useRunBacktest`
- Endpoint constants imported: `BACKTESTS_RUN`

### `frontend/src/api/mutations/useRunCycle.ts`

- Hooks: `useRunCycle`
- Endpoint constants imported: `AUTOPILOT_RUN_CYCLE`

### `frontend/src/api/mutations/useTrainModel.ts`

- Hooks: `useTrainModel`
- Endpoint constants imported: `MODELS_TRAIN`

## Async Job Monitoring (Frontend) 

- `frontend/src/hooks/useSSE.ts`: wraps `EventSource` and prefixes URLs with `/api`.
- `frontend/src/hooks/useJobProgress.ts`: combines polling (`useJob`) with SSE (`JOBS_EVENTS(jobId)`).
- `frontend/src/components/job/JobMonitor.tsx`: displays progress, status badge, errors, and completion state.

## Frontend File Inventory

### Pages

- Count: 51
- LOC: 3,916

| File | Lines |
|---|---:|
| `frontend/src/pages/AutopilotPage/KalshiEventsTab.tsx` | 8 |
| `frontend/src/pages/AutopilotPage/LifecycleTimeline.tsx` | 69 |
| `frontend/src/pages/AutopilotPage/PaperPnLTracker.tsx` | 114 |
| `frontend/src/pages/AutopilotPage/PaperTradingTab.tsx` | 47 |
| `frontend/src/pages/AutopilotPage/StrategyCandidatesTab.tsx` | 81 |
| `frontend/src/pages/AutopilotPage.tsx` | 87 |
| `frontend/src/pages/BacktestPage/BacktestConfigPanel.tsx` | 46 |
| `frontend/src/pages/BacktestPage/BacktestResults.tsx` | 149 |
| `frontend/src/pages/BacktestPage/RunBacktestButton.tsx` | 42 |
| `frontend/src/pages/BacktestPage.tsx` | 60 |
| `frontend/src/pages/BenchmarkPage/BenchmarkChartGrid.tsx` | 237 |
| `frontend/src/pages/BenchmarkPage/BenchmarkMetricCards.tsx` | 37 |
| `frontend/src/pages/BenchmarkPage.tsx` | 43 |
| `frontend/src/pages/DashboardPage/EquityCurveTab.tsx` | 152 |
| `frontend/src/pages/DashboardPage/FeatureImportanceTab.tsx` | 64 |
| `frontend/src/pages/DashboardPage/KPIGrid.tsx` | 76 |
| `frontend/src/pages/DashboardPage/ModelHealthTab.tsx` | 74 |
| `frontend/src/pages/DashboardPage/RegimeTab.tsx` | 94 |
| `frontend/src/pages/DashboardPage/RiskTab.tsx` | 255 |
| `frontend/src/pages/DashboardPage/TradeLogTab.tsx` | 30 |
| `frontend/src/pages/DashboardPage/useDashboardData.ts` | 56 |
| `frontend/src/pages/DashboardPage.tsx` | 80 |
| `frontend/src/pages/DataExplorerPage/CacheStatusPanel.tsx` | 180 |
| `frontend/src/pages/DataExplorerPage/CandlestickPanel.tsx` | 27 |
| `frontend/src/pages/DataExplorerPage/DataQualityReport.tsx` | 95 |
| `frontend/src/pages/DataExplorerPage/UniverseSelector.tsx` | 88 |
| `frontend/src/pages/DataExplorerPage.tsx` | 32 |
| `frontend/src/pages/IVSurfacePage/ArbAwareSVITab.tsx` | 132 |
| `frontend/src/pages/IVSurfacePage/HestonSurfaceTab.tsx` | 75 |
| `frontend/src/pages/IVSurfacePage/IVControls.tsx` | 83 |
| `frontend/src/pages/IVSurfacePage/SVISurfaceTab.tsx` | 138 |
| `frontend/src/pages/IVSurfacePage.tsx` | 27 |
| `frontend/src/pages/LogsPage/LogFilters.tsx` | 86 |
| `frontend/src/pages/LogsPage/LogStream.tsx` | 32 |
| `frontend/src/pages/LogsPage.tsx` | 25 |
| `frontend/src/pages/ModelLabPage/FeatureDiffViewer.tsx` | 153 |
| `frontend/src/pages/ModelLabPage/FeaturesTab.tsx` | 80 |
| `frontend/src/pages/ModelLabPage/RegimeTab.tsx` | 67 |
| `frontend/src/pages/ModelLabPage/TrainingTab.tsx` | 122 |
| `frontend/src/pages/ModelLabPage.tsx` | 30 |
| `frontend/src/pages/NotFoundPage.tsx` | 25 |
| `frontend/src/pages/SignalDeskPage/ConfidenceScatter.tsx` | 20 |
| `frontend/src/pages/SignalDeskPage/EnsembleDisagreement.tsx` | 145 |
| `frontend/src/pages/SignalDeskPage/SignalControls.tsx` | 62 |
| `frontend/src/pages/SignalDeskPage/SignalDistribution.tsx` | 14 |
| `frontend/src/pages/SignalDeskPage/SignalRankingsPanel.tsx` | 23 |
| `frontend/src/pages/SignalDeskPage.tsx` | 59 |
| `frontend/src/pages/SystemHealthPage/HealthCheckPanel.tsx` | 81 |
| `frontend/src/pages/SystemHealthPage/HealthRadar.tsx` | 27 |
| `frontend/src/pages/SystemHealthPage/HealthScoreCards.tsx` | 22 |
| `frontend/src/pages/SystemHealthPage.tsx` | 65 |

### Components

- Count: 37
- LOC: 2,474

| File | Lines |
|---|---:|
| `frontend/src/components/charts/AreaChart.tsx` | 67 |
| `frontend/src/components/charts/BarChart.tsx` | 88 |
| `frontend/src/components/charts/CandlestickChart.tsx` | 137 |
| `frontend/src/components/charts/ChartContainer.tsx` | 55 |
| `frontend/src/components/charts/DualAxisChart.tsx` | 64 |
| `frontend/src/components/charts/EquityCurveChart.tsx` | 87 |
| `frontend/src/components/charts/HeatmapChart.tsx` | 85 |
| `frontend/src/components/charts/HistogramChart.tsx` | 102 |
| `frontend/src/components/charts/IVSurfacePlot.tsx` | 75 |
| `frontend/src/components/charts/LineChart.tsx` | 78 |
| `frontend/src/components/charts/RadarChart.tsx` | 48 |
| `frontend/src/components/charts/RegimeTimeline.tsx` | 50 |
| `frontend/src/components/charts/ScatterChart.tsx` | 62 |
| `frontend/src/components/charts/theme.ts` | 81 |
| `frontend/src/components/job/JobMonitor.tsx` | 52 |
| `frontend/src/components/job/JobProgressBar.tsx` | 35 |
| `frontend/src/components/job/JobStatusBadge.tsx` | 37 |
| `frontend/src/components/layout/AppShell.tsx` | 21 |
| `frontend/src/components/layout/PageContainer.tsx` | 5 |
| `frontend/src/components/layout/Sidebar.tsx` | 174 |
| `frontend/src/components/layout/StatusBar.tsx` | 107 |
| `frontend/src/components/tables/DataTable.tsx` | 215 |
| `frontend/src/components/tables/LogTable.tsx` | 58 |
| `frontend/src/components/tables/SignalTable.tsx` | 96 |
| `frontend/src/components/tables/TradeTable.tsx` | 80 |
| `frontend/src/components/tables/tableUtils.ts` | 24 |
| `frontend/src/components/ui/AlertBanner.tsx` | 39 |
| `frontend/src/components/ui/DataProvenanceBadge.tsx` | 61 |
| `frontend/src/components/ui/EmptyState.tsx` | 11 |
| `frontend/src/components/ui/ErrorBoundary.tsx` | 67 |
| `frontend/src/components/ui/ErrorPanel.tsx` | 46 |
| `frontend/src/components/ui/MetricCard.tsx` | 62 |
| `frontend/src/components/ui/PageHeader.tsx` | 26 |
| `frontend/src/components/ui/RegimeBadge.tsx` | 33 |
| `frontend/src/components/ui/SignificanceBadge.tsx` | 90 |
| `frontend/src/components/ui/Spinner.tsx` | 12 |
| `frontend/src/components/ui/TabGroup.tsx` | 44 |

### Hooks

- Count: 7
- LOC: 249

| File | Lines |
|---|---:|
| `frontend/src/hooks/useCSVExport.ts` | 32 |
| `frontend/src/hooks/useClock.ts` | 14 |
| `frontend/src/hooks/useDownsample.ts` | 63 |
| `frontend/src/hooks/useJobProgress.ts` | 61 |
| `frontend/src/hooks/useLogStream.ts` | 15 |
| `frontend/src/hooks/useMediaQuery.ts` | 17 |
| `frontend/src/hooks/useSSE.ts` | 47 |

### Stores

- Count: 3
- LOC: 75

| File | Lines |
|---|---:|
| `frontend/src/store/filterStore.ts` | 19 |
| `frontend/src/store/jobStore.ts` | 45 |
| `frontend/src/store/uiStore.ts` | 11 |

### Types

- Count: 11
- LOC: 434

| File | Lines |
|---|---:|
| `frontend/src/types/api.ts` | 23 |
| `frontend/src/types/autopilot.ts` | 87 |
| `frontend/src/types/backtests.ts` | 61 |
| `frontend/src/types/compute.ts` | 23 |
| `frontend/src/types/config.ts` | 3 |
| `frontend/src/types/dashboard.ts` | 88 |
| `frontend/src/types/data.ts` | 27 |
| `frontend/src/types/health.ts` | 36 |
| `frontend/src/types/jobs.ts` | 26 |
| `frontend/src/types/models.ts` | 40 |
| `frontend/src/types/signals.ts` | 20 |

### Design System Files

- Count: 2
- LOC: 163

| File | Lines |
|---|---:|
| `frontend/src/design-system/index.css` | 102 |
| `frontend/src/design-system/tokens.css` | 61 |

## Known Contract Drift (Source-Verified)

- Backend job statuses in `api/jobs/models.py` are `queued/running/succeeded/failed/cancelled`.
- Frontend job types in `frontend/src/types/jobs.ts` currently model `pending/running/completed/failed/cancelled` and `message` instead of backend `progress_message`.
- Treat backend models as canonical when documenting API payloads; frontend types currently represent a drift that should be reconciled in code.
