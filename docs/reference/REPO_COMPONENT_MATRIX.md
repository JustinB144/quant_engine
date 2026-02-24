# Repository Component Matrix

Source-derived inventory of active code components in the current repository state. This document is generated from the filesystem and Python AST (not hand-maintained).

## Snapshot

- Python modules scanned: 196
- FastAPI router modules mounted: 12
- Frontend source files (`frontend/src`): 131 (7,935 LOC)

## Package Summary (Python)

| Package | Modules | Classes | Top-level Functions | LOC |
|---|---:|---:|---:|---:|
| `(root)` | 13 | 13 | 23 | 2,998 |
| `api` | 52 | 56 | 97 | 5,438 |
| `autopilot` | 6 | 9 | 0 | 2,009 |
| `backtest` | 6 | 15 | 16 | 3,677 |
| `data` | 10 | 13 | 58 | 5,310 |
| `features` | 10 | 5 | 47 | 3,747 |
| `indicators` | 2 | 92 | 2 | 2,993 |
| `kalshi` | 25 | 34 | 66 | 6,096 |
| `models` | 14 | 28 | 9 | 5,130 |
| `regime` | 5 | 7 | 5 | 1,624 |
| `risk` | 11 | 14 | 14 | 3,057 |
| `scripts` | 3 | 0 | 11 | 1,056 |
| `tests` | 37 | 67 | 94 | 6,186 |
| `utils` | 2 | 3 | 1 | 444 |

## Root Entrypoints and Core Files

| File | Lines | Classes | Functions | Intent |
|---|---:|---:|---:|---|
| `__init__.py` | 6 | 0 | 0 | Quant Engine - Continuous Feature ML Trading System |
| `config.py` | 464 | 0 | 1 | Central configuration for the quant engine. |
| `config_structured.py` | 235 | 13 | 0 | Structured configuration for the quant engine using typed dataclasses. |
| `reproducibility.py` | 333 | 0 | 6 | Reproducibility locks for run manifests. |
| `run_autopilot.py` | 89 | 0 | 1 | Run one full autopilot cycle: |
| `run_backtest.py` | 428 | 0 | 1 | Backtest the trained model on historical data. |
| `run_kalshi_event_pipeline.py` | 227 | 0 | 2 | Run the integrated Kalshi event-time pipeline inside quant_engine. |
| `run_predict.py` | 202 | 0 | 1 | Generate predictions using trained ensemble model. |
| `run_rehydrate_cache_metadata.py` | 101 | 0 | 2 | Backfill cache metadata sidecars for existing OHLCV cache files. |
| `run_retrain.py` | 290 | 0 | 2 | Retrain the quant engine model — checks triggers and retrains if needed. |
| `run_server.py` | 80 | 0 | 1 | Combined API + frontend static serving entry point. |
| `run_train.py` | 195 | 0 | 1 | Train the regime-conditional ensemble model. |
| `run_wrds_daily_refresh.py` | 348 | 0 | 5 | Re-download all daily OHLCV data from WRDS CRSP to replace old cache files |

## FastAPI Router Matrix

| Router Module | Prefix | Endpoints | Handlers |
|---|---|---:|---|
| `api/routers/autopilot.py` | `/api/autopilot` | 4 | `GET /latest-cycle` → `latest_cycle`, `GET /strategies` → `strategies`, `GET /paper-state` → `paper_state`, `POST /run-cycle` → `run_cycle` |
| `api/routers/backtests.py` | `/api/backtests` | 4 | `GET /latest` → `latest_backtest`, `GET /latest/trades` → `latest_trades`, `GET /latest/equity-curve` → `equity_curve`, `POST /run` → `run_backtest` |
| `api/routers/benchmark.py` | `/api/benchmark` | 3 | `GET /comparison` → `benchmark_comparison`, `GET /equity-curves` → `benchmark_equity_curves`, `GET /rolling-metrics` → `benchmark_rolling_metrics` |
| `api/routers/config_mgmt.py` | `/api/config` | 4 | `GET ` → `get_config`, `GET /validate` → `validate_config_endpoint`, `GET /status` → `get_config_status`, `PATCH ` → `patch_config` |
| `api/routers/dashboard.py` | `/api/dashboard` | 6 | `GET /summary` → `dashboard_summary`, `GET /regime` → `dashboard_regime`, `GET /returns-distribution` → `returns_distribution`, `GET /rolling-risk` → `rolling_risk`, `GET /equity` → `equity_with_benchmark`, `GET /attribution` → `attribution` |
| `api/routers/data_explorer.py` | `/api/data` | 3 | `GET /universe` → `get_universe`, `GET /status` → `get_data_status`, `GET /ticker/{ticker}` → `get_ticker` |
| `api/routers/iv_surface.py` | `/api/iv-surface` | 1 | `GET /arb-free-svi` → `arb_free_svi_surface` |
| `api/routers/jobs.py` | `/api/jobs` | 4 | `GET ` → `list_jobs`, `GET /{job_id}` → `get_job`, `GET /{job_id}/events` → `job_events`, `POST /{job_id}/cancel` → `cancel_job` |
| `api/routers/logs.py` | `/api/logs` | 1 | `GET ` → `get_logs` |
| `api/routers/model_lab.py` | `/api/models` | 6 | `GET /versions` → `list_versions`, `GET /health` → `model_health`, `GET /features/importance` → `feature_importance`, `GET /features/correlations` → `feature_correlations`, `POST /train` → `train_model`, `POST /predict` → `predict_model` |
| `api/routers/signals.py` | `/api/signals` | 1 | `GET /latest` → `latest_signals` |
| `api/routers/system_health.py` | `(mixed absolute paths)` | 4 | `GET /api/health` → `quick_health`, `GET /api/health/detailed` → `detailed_health`, `GET /api/v1/system/model-age` → `model_age`, `GET /api/v1/system/data-mode` → `data_mode` |

## Frontend Source Matrix (`frontend/src`)

### Pages

- Files: 51
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

- Files: 37
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

- Files: 7
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

### API Query Hooks

- Files: 12
- LOC: 414

| File | Lines |
|---|---:|
| `frontend/src/api/queries/useAutopilot.ts` | 28 |
| `frontend/src/api/queries/useBacktests.ts` | 29 |
| `frontend/src/api/queries/useBenchmark.ts` | 75 |
| `frontend/src/api/queries/useConfig.ts` | 20 |
| `frontend/src/api/queries/useDashboard.ts` | 67 |
| `frontend/src/api/queries/useData.ts` | 54 |
| `frontend/src/api/queries/useHealth.ts` | 20 |
| `frontend/src/api/queries/useIVSurface.ts` | 24 |
| `frontend/src/api/queries/useJobs.ts` | 24 |
| `frontend/src/api/queries/useLogs.ts` | 19 |
| `frontend/src/api/queries/useModels.ts` | 42 |
| `frontend/src/api/queries/useSignals.ts` | 12 |

### API Mutation Hooks

- Files: 6
- LOC: 89

| File | Lines |
|---|---:|
| `frontend/src/api/mutations/useCancelJob.ts` | 13 |
| `frontend/src/api/mutations/usePatchConfig.ts` | 14 |
| `frontend/src/api/mutations/usePredict.ts` | 16 |
| `frontend/src/api/mutations/useRunBacktest.ts` | 15 |
| `frontend/src/api/mutations/useRunCycle.ts` | 16 |
| `frontend/src/api/mutations/useTrainModel.ts` | 15 |

### Stores

- Files: 3
- LOC: 75

| File | Lines |
|---|---:|
| `frontend/src/store/filterStore.ts` | 19 |
| `frontend/src/store/jobStore.ts` | 45 |
| `frontend/src/store/uiStore.ts` | 11 |

### Types

- Files: 11
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

## Python Module Index (By Top-Level Package)

### `(root)`

| Module | Lines | Classes | Functions | Intent |
|---|---:|---:|---:|---|
| `__init__.py` | 6 | 0 | 0 | Quant Engine - Continuous Feature ML Trading System |
| `config.py` | 464 | 0 | 1 | Central configuration for the quant engine. |
| `config_structured.py` | 235 | 13 | 0 | Structured configuration for the quant engine using typed dataclasses. |
| `reproducibility.py` | 333 | 0 | 6 | Reproducibility locks for run manifests. |
| `run_autopilot.py` | 89 | 0 | 1 | Run one full autopilot cycle: |
| `run_backtest.py` | 428 | 0 | 1 | Backtest the trained model on historical data. |
| `run_kalshi_event_pipeline.py` | 227 | 0 | 2 | Run the integrated Kalshi event-time pipeline inside quant_engine. |
| `run_predict.py` | 202 | 0 | 1 | Generate predictions using trained ensemble model. |
| `run_rehydrate_cache_metadata.py` | 101 | 0 | 2 | Backfill cache metadata sidecars for existing OHLCV cache files. |
| `run_retrain.py` | 290 | 0 | 2 | Retrain the quant engine model — checks triggers and retrains if needed. |
| `run_server.py` | 80 | 0 | 1 | Combined API + frontend static serving entry point. |
| `run_train.py` | 195 | 0 | 1 | Train the regime-conditional ensemble model. |
| `run_wrds_daily_refresh.py` | 348 | 0 | 5 | Re-download all daily OHLCV data from WRDS CRSP to replace old cache files |

### `api`

| Module | Lines | Classes | Functions | Intent |
|---|---:|---:|---:|---|
| `api/__init__.py` | 1 | 0 | 0 | FastAPI backend for the quant engine. |
| `api/ab_testing.py` | 214 | 3 | 0 | A/B testing framework for strategy evaluation. |
| `api/cache/__init__.py` | 4 | 0 | 0 | TTL cache with event-driven invalidation. |
| `api/cache/invalidation.py` | 32 | 0 | 4 | Event-driven cache invalidation helpers. |
| `api/cache/manager.py` | 62 | 1 | 0 | In-memory TTL cache manager. |
| `api/config.py` | 75 | 2 | 0 | Runtime-adjustable configuration for the API layer. |
| `api/deps/__init__.py` | 16 | 0 | 0 | Dependency injection providers. |
| `api/deps/providers.py` | 54 | 0 | 5 | Singleton dependency providers for FastAPI ``Depends()``. |
| `api/errors.py` | 77 | 5 | 2 | Custom exceptions and FastAPI error handler registration. |
| `api/jobs/__init__.py` | 6 | 0 | 0 | SQLite-backed job queue for long-running compute. |
| `api/jobs/autopilot_job.py` | 31 | 0 | 1 | Autopilot job executor. |
| `api/jobs/backtest_job.py` | 29 | 0 | 1 | Backtest job executor. |
| `api/jobs/models.py` | 32 | 2 | 0 | Job data models. |
| `api/jobs/predict_job.py` | 28 | 0 | 1 | Predict job executor. |
| `api/jobs/runner.py` | 115 | 1 | 0 | Async job runner with concurrency control and SSE event streaming. |
| `api/jobs/store.py` | 145 | 1 | 0 | SQLite-backed persistence for job records. |
| `api/jobs/train_job.py` | 30 | 0 | 1 | Train job executor. |
| `api/main.py` | 148 | 0 | 4 | FastAPI application factory and server entry point. |
| `api/orchestrator.py` | 372 | 2 | 0 | Unified pipeline orchestrator — data -> features -> regimes -> compute. |
| `api/routers/__init__.py` | 39 | 0 | 1 | Route modules — imported lazily by the app factory. |
| `api/routers/autopilot.py` | 83 | 0 | 5 | Autopilot endpoints — cycle reports, strategies, paper state, run-cycle. |
| `api/routers/backtests.py` | 89 | 0 | 5 | Backtest result + compute endpoints. |
| `api/routers/benchmark.py` | 128 | 0 | 6 | Benchmark comparison endpoints. |
| `api/routers/config_mgmt.py` | 146 | 0 | 6 | Runtime config management endpoints. |
| `api/routers/dashboard.py` | 170 | 0 | 10 | Dashboard endpoints — KPIs, regime overview, time series analytics. |
| `api/routers/data_explorer.py` | 56 | 0 | 3 | Data explorer endpoints — universe + per-ticker OHLCV. |
| `api/routers/iv_surface.py` | 71 | 0 | 2 | IV Surface computation endpoints. |
| `api/routers/jobs.py` | 72 | 0 | 5 | Job management endpoints. |
| `api/routers/logs.py` | 38 | 1 | 1 | Log retrieval endpoint. |
| `api/routers/model_lab.py` | 90 | 0 | 6 | Model lab endpoints — versions, health, feature importance, train, predict. |
| `api/routers/signals.py` | 51 | 0 | 2 | Signal / prediction endpoints. |
| `api/routers/system_health.py` | 111 | 0 | 4 | System health endpoints. |
| `api/schemas/__init__.py` | 4 | 0 | 0 | Pydantic schemas for API request/response models. |
| `api/schemas/autopilot.py` | 30 | 3 | 0 | Autopilot schemas. |
| `api/schemas/backtests.py` | 56 | 4 | 0 | Backtest schemas. |
| `api/schemas/compute.py` | 56 | 5 | 0 | Request schemas for compute (POST) endpoints. |
| `api/schemas/dashboard.py` | 38 | 3 | 0 | Dashboard-related schemas. |
| `api/schemas/data_explorer.py` | 37 | 3 | 0 | Data explorer schemas. |
| `api/schemas/envelope.py` | 54 | 2 | 0 | Standard API response envelope with provenance metadata. |
| `api/schemas/model_lab.py` | 43 | 3 | 0 | Model lab schemas. |
| `api/schemas/signals.py` | 26 | 2 | 0 | Signal schemas. |
| `api/schemas/system_health.py` | 47 | 3 | 0 | System health schemas. |
| `api/services/__init__.py` | 20 | 0 | 0 | Engine wrapper services — sync functions returning plain dicts. |
| `api/services/autopilot_service.py` | 62 | 1 | 0 | Wraps autopilot engine and results for API consumption. |
| `api/services/backtest_service.py` | 139 | 1 | 0 | Wraps backtest results for API consumption. |
| `api/services/data_helpers.py` | 1034 | 2 | 22 | Data loading and computation functions extracted from dash_ui/data/loaders.py. |
| `api/services/data_service.py` | 208 | 1 | 0 | Wraps data.loader for API consumption. |
| `api/services/health_service.py` | 686 | 1 | 0 | System health assessment for API consumption. |
| `api/services/kalshi_service.py` | 50 | 1 | 0 | Wraps kalshi.storage for API consumption. |
| `api/services/model_service.py` | 98 | 1 | 0 | Wraps models.* modules for API consumption. |
| `api/services/regime_service.py` | 50 | 1 | 0 | Wraps regime.detector for API consumption. |
| `api/services/results_service.py` | 85 | 1 | 0 | Reads/writes to the results/ directory. |

### `autopilot`

| Module | Lines | Classes | Functions | Intent |
|---|---:|---:|---:|---|
| `autopilot/__init__.py` | 20 | 0 | 0 | Autopilot layer: discovery, promotion, and paper-trading orchestration. |
| `autopilot/engine.py` | 990 | 2 | 0 | End-to-end autopilot cycle: |
| `autopilot/paper_trader.py` | 529 | 1 | 0 | Stateful paper-trading engine for promoted strategies. |
| `autopilot/promotion_gate.py` | 281 | 2 | 0 | Promotion gate for deciding whether a discovered strategy is deployable. |
| `autopilot/registry.py` | 110 | 2 | 0 | Persistent strategy registry for promoted candidates. |
| `autopilot/strategy_discovery.py` | 79 | 2 | 0 | Strategy discovery for execution-layer parameter variants. |

### `backtest`

| Module | Lines | Classes | Functions | Intent |
|---|---:|---:|---:|---|
| `backtest/__init__.py` | 4 | 0 | 0 | Backtesting package exports and namespace initialization. |
| `backtest/advanced_validation.py` | 581 | 5 | 6 | Advanced Validation — Deflated Sharpe, PBO, Monte Carlo, capacity analysis. |
| `backtest/engine.py` | 1869 | 3 | 0 | Backtester — converts model predictions into simulated trades. |
| `backtest/execution.py` | 273 | 2 | 1 | Execution simulator with spread, market impact, and participation limits. |
| `backtest/optimal_execution.py` | 201 | 0 | 2 | Almgren-Chriss (2001) optimal execution model. |
| `backtest/validation.py` | 749 | 5 | 7 | Walk-forward validation and statistical tests. |

### `data`

| Module | Lines | Classes | Functions | Intent |
|---|---:|---:|---:|---|
| `data/__init__.py` | 32 | 0 | 0 | Data subpackage — self-contained data loading, caching, WRDS, and survivorship. |
| `data/alternative.py` | 652 | 1 | 2 | Alternative data framework — WRDS-backed implementation. |
| `data/feature_store.py` | 312 | 1 | 0 | Point-in-time feature store for backtest acceleration. |
| `data/loader.py` | 732 | 0 | 15 | Data loader — self-contained data loading with multiple sources. |
| `data/local_cache.py` | 702 | 0 | 21 | Local data cache for daily OHLCV data. |
| `data/provider_base.py` | 14 | 1 | 0 | Shared provider protocol for pluggable data connectors. |
| `data/provider_registry.py` | 53 | 0 | 5 | Provider registry for unified data-provider access (WRDS, Kalshi, ...). |
| `data/quality.py` | 263 | 1 | 4 | Data quality checks for OHLCV time series. |
| `data/survivorship.py` | 935 | 8 | 5 | Survivorship Bias Controls (Tasks 112-117) |
| `data/wrds_provider.py` | 1615 | 1 | 6 | wrds_provider.py |

### `features`

| Module | Lines | Classes | Functions | Intent |
|---|---:|---:|---:|---|
| `features/__init__.py` | 4 | 0 | 0 | Feature engineering package namespace. |
| `features/harx_spillovers.py` | 242 | 0 | 3 | HARX Volatility Spillover features (Tier 6.1). |
| `features/intraday.py` | 243 | 0 | 2 | Intraday microstructure features from WRDS TAQmsec tick data. |
| `features/lob_features.py` | 311 | 0 | 5 | Markov LOB (Limit Order Book) features from intraday bar data (Tier 6.2). |
| `features/macro.py` | 244 | 1 | 1 | FRED macro indicator features for quant_engine. |
| `features/options_factors.py` | 134 | 0 | 4 | Option surface factor construction from OptionMetrics-enriched daily panels. |
| `features/pipeline.py` | 1272 | 1 | 12 | Feature Pipeline — computes model features from OHLCV data. |
| `features/research_factors.py` | 985 | 1 | 19 | Research-derived factor construction for quant_engine. |
| `features/version.py` | 168 | 2 | 0 | Feature versioning system. |
| `features/wave_flow.py` | 144 | 0 | 1 | Wave-Flow Decomposition for quant_engine. |

### `indicators`

| Module | Lines | Classes | Functions | Intent |
|---|---:|---:|---:|---|
| `indicators/__init__.py` | 89 | 0 | 0 | Quant Engine Indicators — self-contained copy of the technical indicator library. |
| `indicators/indicators.py` | 2904 | 92 | 2 | Technical Indicator Library |

### `kalshi`

| Module | Lines | Classes | Functions | Intent |
|---|---:|---:|---:|---|
| `kalshi/__init__.py` | 58 | 0 | 0 | Kalshi vertical for intraday event-market research. |
| `kalshi/client.py` | 655 | 5 | 1 | Kalshi API client with signed authentication, rate limiting, and endpoint routing. |
| `kalshi/disagreement.py` | 113 | 1 | 2 | Cross-market disagreement engine for Kalshi event features. |
| `kalshi/distribution.py` | 935 | 3 | 23 | Contract -> probability distribution builder for Kalshi markets. |
| `kalshi/events.py` | 517 | 2 | 11 | Event-time joins and as-of feature/label builders for Kalshi-driven research. |
| `kalshi/mapping_store.py` | 75 | 2 | 0 | Versioned event-to-market mapping persistence. |
| `kalshi/microstructure.py` | 127 | 1 | 2 | Market microstructure diagnostics for Kalshi event markets. |
| `kalshi/options.py` | 145 | 0 | 3 | OptionMetrics-style options reference features for Kalshi event disagreement. |
| `kalshi/pipeline.py` | 167 | 1 | 0 | Orchestration helpers for the Kalshi event-market vertical. |
| `kalshi/promotion.py` | 178 | 1 | 2 | Event-strategy promotion helpers for Kalshi walk-forward outputs. |
| `kalshi/provider.py` | 647 | 1 | 3 | Kalshi provider: ingestion + storage + feature-ready retrieval. |
| `kalshi/quality.py` | 206 | 2 | 5 | Quality scoring helpers for Kalshi event-distribution snapshots. |
| `kalshi/regimes.py` | 142 | 1 | 6 | Regime tagging for Kalshi event strategies. |
| `kalshi/router.py` | 102 | 2 | 0 | Routing helpers for live vs historical Kalshi endpoints. |
| `kalshi/storage.py` | 649 | 1 | 0 | Event-time storage layer for Kalshi + macro event research. |
| `kalshi/tests/__init__.py` | 1 | 0 | 0 | Kalshi package-local tests. |
| `kalshi/tests/test_bin_validity.py` | 105 | 1 | 0 | Bin overlap/gap detection test (Instructions I.3). |
| `kalshi/tests/test_distribution.py` | 41 | 1 | 0 | Kalshi test module for distribution behavior and regressions. |
| `kalshi/tests/test_leakage.py` | 46 | 1 | 0 | Kalshi test module for leakage behavior and regressions. |
| `kalshi/tests/test_no_leakage.py` | 117 | 1 | 0 | No-leakage test at panel level (Instructions I.4). |
| `kalshi/tests/test_signature_kat.py` | 141 | 1 | 0 | Known-answer test for Kalshi RSA-PSS SHA256 signature (Instructions A3 + I.1). |
| `kalshi/tests/test_stale_quotes.py` | 152 | 1 | 0 | Stale quote cutoff test (Instructions I.5). |
| `kalshi/tests/test_threshold_direction.py` | 126 | 1 | 0 | Threshold direction correctness test (Instructions I.2). |
| `kalshi/tests/test_walkforward_purge.py` | 159 | 1 | 0 | Walk-forward purge/embargo test (Instructions I.6). |
| `kalshi/walkforward.py` | 492 | 3 | 8 | Walk-forward evaluation for event-centric Kalshi feature panels. |

### `models`

| Module | Lines | Classes | Functions | Intent |
|---|---:|---:|---:|---|
| `models/__init__.py` | 20 | 0 | 0 | Models subpackage — training, prediction, versioning, and retraining triggers. |
| `models/calibration.py` | 327 | 2 | 2 | Confidence Calibration --- Platt scaling and isotonic regression. |
| `models/cross_sectional.py` | 136 | 0 | 1 | Cross-Sectional Ranking Model — rank stocks relative to peers at each date. |
| `models/feature_stability.py` | 313 | 2 | 0 | Feature Stability Monitoring — tracks feature importance rankings across |
| `models/governance.py` | 155 | 2 | 0 | Champion/challenger governance for model versions. |
| `models/iv/__init__.py` | 31 | 0 | 0 | Implied Volatility Surface Models — Heston, SVI, Black-Scholes, and IV Surface. |
| `models/iv/models.py` | 937 | 10 | 1 | Implied Volatility Surface Models. |
| `models/neural_net.py` | 198 | 1 | 0 | Tabular Neural Network — feedforward network for tabular financial data. |
| `models/online_learning.py` | 273 | 2 | 0 | Online learning module for incremental model updates between full retrains. |
| `models/predictor.py` | 404 | 1 | 1 | Model Predictor — loads trained ensemble and generates predictions. |
| `models/retrain_trigger.py` | 296 | 1 | 0 | ML Retraining Trigger Logic |
| `models/trainer.py` | 1598 | 5 | 0 | Model Trainer — trains regime-conditional gradient boosting ensemble. |
| `models/versioning.py` | 207 | 2 | 0 | Model Versioning — timestamped model directories with registry. |
| `models/walk_forward.py` | 235 | 0 | 4 | Walk-Forward Model Selection — expanding-window hyperparameter search |

### `regime`

| Module | Lines | Classes | Functions | Intent |
|---|---:|---:|---:|---|
| `regime/__init__.py` | 17 | 0 | 0 | Regime modeling components. |
| `regime/correlation.py` | 213 | 1 | 0 | Correlation Regime Detection (NEW 11). |
| `regime/detector.py` | 586 | 2 | 1 | Regime detector with two engines: |
| `regime/hmm.py` | 566 | 2 | 4 | Gaussian HMM regime model with sticky transitions and duration smoothing. |
| `regime/jump_model.py` | 242 | 2 | 0 | Statistical Jump Model for regime detection. |

### `risk`

| Module | Lines | Classes | Functions | Intent |
|---|---:|---:|---:|---|
| `risk/__init__.py` | 42 | 0 | 0 | Risk Management Module — Renaissance-grade portfolio risk controls. |
| `risk/attribution.py` | 266 | 0 | 4 | Performance Attribution --- decompose portfolio returns into market, factor, and alpha. |
| `risk/covariance.py` | 249 | 2 | 2 | Covariance estimation utilities for portfolio risk controls. |
| `risk/drawdown.py` | 240 | 3 | 0 | Drawdown Controller — circuit breakers and recovery protocols. |
| `risk/factor_portfolio.py` | 220 | 0 | 2 | Factor-Based Portfolio Construction — factor decomposition and exposure analysis. |
| `risk/metrics.py` | 253 | 2 | 0 | Risk Metrics — VaR, CVaR, tail risk, MAE/MFE, and advanced risk analytics. |
| `risk/portfolio_optimizer.py` | 276 | 0 | 1 | Mean-Variance Portfolio Optimization — turnover-penalised portfolio construction. |
| `risk/portfolio_risk.py` | 329 | 2 | 0 | Portfolio Risk Manager — enforces sector, correlation, and exposure limits. |
| `risk/position_sizer.py` | 564 | 2 | 0 | Position Sizing — Kelly criterion, volatility-scaled, and ATR-based methods. |
| `risk/stop_loss.py` | 255 | 3 | 0 | Stop Loss Manager — regime-aware ATR stops, trailing, time, and regime-change stops. |
| `risk/stress_test.py` | 363 | 0 | 5 | Stress Testing Module --- scenario analysis and historical drawdown replay. |

### `scripts`

| Module | Lines | Classes | Functions | Intent |
|---|---:|---:|---:|---|
| `scripts/generate_types.py` | 146 | 0 | 3 | Generate TypeScript interfaces from Pydantic schemas. |
| `scripts/ibkr_daily_gapfill.py` | 417 | 0 | 4 | IBKR Daily Gap-Fill Downloader for quant_engine cache. |
| `scripts/ibkr_intraday_download.py` | 493 | 0 | 4 | IBKR Intraday Data Downloader for quant_engine cache. |

### `tests`

| Module | Lines | Classes | Functions | Intent |
|---|---:|---:|---:|---|
| `tests/__init__.py` | 6 | 0 | 0 | Project-level test suite package. |
| `tests/api/__init__.py` | 0 | 0 | 0 | No module docstring. |
| `tests/api/test_compute_routers.py` | 65 | 0 | 6 | Tests for POST compute endpoints — verify job creation. |
| `tests/api/test_envelope.py` | 51 | 0 | 6 | Tests for ApiResponse envelope and ResponseMeta. |
| `tests/api/test_integration.py` | 72 | 0 | 4 | Integration tests — full app startup, envelope consistency, config patch. |
| `tests/api/test_jobs.py` | 135 | 0 | 12 | Tests for the job system: store, runner, lifecycle. |
| `tests/api/test_main.py` | 48 | 0 | 5 | Tests for app factory and basic middleware. |
| `tests/api/test_read_routers.py` | 118 | 0 | 14 | Tests for all GET endpoints — verify ApiResponse envelope. |
| `tests/api/test_services.py` | 80 | 0 | 10 | Tests for service wrappers — verify dict outputs. |
| `tests/conftest.py` | 240 | 1 | 9 | Shared test fixtures for the quant_engine test suite. |
| `tests/test_autopilot_predictor_fallback.py` | 58 | 1 | 0 | Test module for autopilot predictor fallback behavior and regressions. |
| `tests/test_backtest_realism.py` | 561 | 7 | 5 | Spec 011 — Backtest Execution Realism & Validation Enforcement tests. |
| `tests/test_cache_metadata_rehydrate.py` | 97 | 1 | 1 | Test module for cache metadata rehydrate behavior and regressions. |
| `tests/test_conceptual_fixes.py` | 324 | 4 | 2 | Tests for Spec 008 — Ensemble & Conceptual Fixes. |
| `tests/test_covariance_estimator.py` | 25 | 1 | 0 | Test module for covariance estimator behavior and regressions. |
| `tests/test_data_diagnostics.py` | 183 | 4 | 0 | Tests for data loading diagnostics (Spec 005). |
| `tests/test_delisting_total_return.py` | 76 | 1 | 0 | Test module for delisting total return behavior and regressions. |
| `tests/test_drawdown_liquidation.py` | 138 | 6 | 0 | Test module for drawdown liquidation behavior and regressions. |
| `tests/test_execution_dynamic_costs.py` | 48 | 1 | 0 | Test module for execution dynamic costs behavior and regressions. |
| `tests/test_feature_fixes.py` | 377 | 7 | 2 | Tests verifying Spec 012 feature engineering fixes: |
| `tests/test_integration.py` | 556 | 4 | 1 | End-to-end integration tests for the quant engine pipeline. |
| `tests/test_iv_arbitrage_builder.py` | 38 | 1 | 0 | Test module for iv arbitrage builder behavior and regressions. |
| `tests/test_kalshi_asof_features.py` | 65 | 1 | 0 | Test module for kalshi asof features behavior and regressions. |
| `tests/test_kalshi_distribution.py` | 114 | 1 | 0 | Test module for kalshi distribution behavior and regressions. |
| `tests/test_kalshi_hardening.py` | 605 | 1 | 0 | Test module for kalshi hardening behavior and regressions. |
| `tests/test_loader_and_predictor.py` | 230 | 3 | 0 | Test module for loader and predictor behavior and regressions. |
| `tests/test_lookahead_detection.py` | 144 | 1 | 1 | Automated lookahead bias detection for the feature pipeline. |
| `tests/test_panel_split.py` | 55 | 1 | 0 | Test module for panel split behavior and regressions. |
| `tests/test_paper_trader_kelly.py` | 128 | 1 | 3 | Test module for paper trader kelly behavior and regressions. |
| `tests/test_position_sizing_overhaul.py` | 414 | 8 | 0 | Comprehensive tests for Spec 009: Kelly Position Sizing Overhaul. |
| `tests/test_promotion_contract.py` | 105 | 1 | 2 | Test module for promotion contract behavior and regressions. |
| `tests/test_provider_registry.py` | 29 | 1 | 0 | Test module for provider registry behavior and regressions. |
| `tests/test_research_factors.py` | 133 | 1 | 1 | Test module for research factors behavior and regressions. |
| `tests/test_survivorship_pit.py` | 74 | 1 | 0 | Test module for survivorship pit behavior and regressions. |
| `tests/test_training_pipeline_fixes.py` | 567 | 5 | 3 | Tests for Spec 013: Model Training Pipeline — CV Fixes, Calibration, and Governance. |
| `tests/test_validation_and_risk_extensions.py` | 107 | 1 | 1 | Test module for validation and risk extensions behavior and regressions. |
| `tests/test_zero_errors.py` | 120 | 1 | 6 | Integration test: common operations must produce zero ERROR-level log entries. |

### `utils`

| Module | Lines | Classes | Functions | Intent |
|---|---:|---:|---:|---|
| `utils/__init__.py` | 5 | 0 | 0 | Utility helpers package namespace. |
| `utils/logging.py` | 439 | 3 | 1 | Structured logging for the quant engine. |
