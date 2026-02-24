# Frontend Foundation Summary (React/Vite)

Source-verified summary of the active UI foundation in `frontend/`. This replaces the old Dash foundation summary and reflects the current React implementation.

## Stack

- React frontend package: `quant-engine-frontend` v2.0.0
- React Router for page routing (`frontend/src/App.tsx`)
- TanStack Query for data fetching/caching (`frontend/src/main.tsx`, `frontend/src/api/queries/*`)
- Zustand for UI/filter stores (`frontend/src/store/*`)
- Tailwind + custom design tokens/CSS variables (`frontend/src/design-system/*`)
- Charting stack includes ECharts, Plotly, and Lightweight Charts (see `frontend/package.json`)

## File Layout (Current)

### Pages

- Files: 51
- LOC: 3,916

### Components

- Files: 37
- LOC: 2,474

### Hooks

- Files: 7
- LOC: 249

### API Query Hooks

- Files: 12
- LOC: 414

### API Mutation Hooks

- Files: 6
- LOC: 89

### Stores

- Files: 3
- LOC: 75

### Types

- Files: 11
- LOC: 434

### Design System

- Files: 2
- LOC: 163

## Route / Page Composition

### `/autopilot` — `AutopilotPage`

- File: `frontend/src/pages/AutopilotPage.tsx`
- Tabs: `Strategy Candidates`, `Paper Trading`, `Live P&L`, `Kalshi Events`, `Lifecycle`
- Page-local subcomponents: `KalshiEventsTab`, `LifecycleTimeline`, `PaperPnLTracker`, `PaperTradingTab`, `StrategyCandidatesTab`

### `/backtest-risk` — `BacktestPage`

- File: `frontend/src/pages/BacktestPage.tsx`
- Page-local subcomponents: `BacktestConfigPanel`, `BacktestResults`, `RunBacktestButton`

### `/sp-comparison` — `BenchmarkPage`

- File: `frontend/src/pages/BenchmarkPage.tsx`
- Page-local subcomponents: `BenchmarkChartGrid`, `BenchmarkMetricCards`

### `/` — `DashboardPage`

- File: `frontend/src/pages/DashboardPage.tsx`
- Tabs: `Portfolio Overview`, `Regime State`, `Model Performance`, `Feature Importance`, `Trade Log`, `Risk Metrics`
- Page-local subcomponents: `EquityCurveTab`, `FeatureImportanceTab`, `KPIGrid`, `ModelHealthTab`, `RegimeTab`, `RiskTab`, `TradeLogTab`, `useDashboardData`

### `/data-explorer` — `DataExplorerPage`

- File: `frontend/src/pages/DataExplorerPage.tsx`
- Page-local subcomponents: `CacheStatusPanel`, `CandlestickPanel`, `DataQualityReport`, `UniverseSelector`

### `/iv-surface` — `IVSurfacePage`

- File: `frontend/src/pages/IVSurfacePage.tsx`
- Tabs: `SVI Surface`, `Heston Surface`, `Arb-Aware SVI`
- Page-local subcomponents: `ArbAwareSVITab`, `HestonSurfaceTab`, `SVISurfaceTab`

### `/system-logs` — `LogsPage`

- File: `frontend/src/pages/LogsPage.tsx`
- Page-local subcomponents: `LogFilters`, `LogStream`

### `/model-lab` — `ModelLabPage`

- File: `frontend/src/pages/ModelLabPage.tsx`
- Tabs: `Features`, `Feature Diff`, `Regime`, `Training`
- Page-local subcomponents: `FeatureDiffViewer`, `FeaturesTab`, `RegimeTab`, `TrainingTab`

### `*` — `NotFoundPage`

- File: `frontend/src/pages/NotFoundPage.tsx`

### `/signal-desk` — `SignalDeskPage`

- File: `frontend/src/pages/SignalDeskPage.tsx`
- Page-local subcomponents: `ConfidenceScatter`, `EnsembleDisagreement`, `SignalControls`, `SignalDistribution`, `SignalRankingsPanel`

### `/system-health` — `SystemHealthPage`

- File: `frontend/src/pages/SystemHealthPage.tsx`
- Page-local subcomponents: `HealthCheckPanel`, `HealthRadar`, `HealthScoreCards`

## Backend Integration Pattern

- Endpoint paths are centralized in `frontend/src/api/endpoints.ts` (relative paths, no `/api` prefix).
- `frontend/src/api/client.ts` prepends `/api` and unwraps the shared `ApiResponse` envelope.
- Query hooks own read access; mutation hooks submit jobs/config patches and invalidate query caches.
- Job progress UI combines polling + SSE via `useJob` and `useSSE`.
