# Source API Reference

Source-derived Python module inventory for the current repository working tree (including uncommitted changes).
This file is a lookup reference, not a design spec. See architecture docs for runtime behavior and flows.

## Snapshot

- Mounted FastAPI routers: 15
- Mounted FastAPI endpoints: 48
- Root CLI scripts (`run_*.py`): 9
- Utility scripts (`scripts/*.py`): 5

## FastAPI Endpoint Inventory

| Method | Path | Handler | Router |
|---|---|---|---|
| `GET` | `/api/autopilot/latest-cycle` | `latest_cycle` | `api/routers/autopilot.py` |
| `GET` | `/api/autopilot/paper-state` | `paper_state` | `api/routers/autopilot.py` |
| `POST` | `/api/autopilot/run-cycle` | `run_cycle` | `api/routers/autopilot.py` |
| `GET` | `/api/autopilot/strategies` | `strategies` | `api/routers/autopilot.py` |
| `GET` | `/api/backtests/latest` | `latest_backtest` | `api/routers/backtests.py` |
| `GET` | `/api/backtests/latest/equity-curve` | `equity_curve` | `api/routers/backtests.py` |
| `GET` | `/api/backtests/latest/trades` | `latest_trades` | `api/routers/backtests.py` |
| `POST` | `/api/backtests/run` | `run_backtest` | `api/routers/backtests.py` |
| `GET` | `/api/benchmark/comparison` | `benchmark_comparison` | `api/routers/benchmark.py` |
| `GET` | `/api/benchmark/equity-curves` | `benchmark_equity_curves` | `api/routers/benchmark.py` |
| `GET` | `/api/benchmark/rolling-metrics` | `benchmark_rolling_metrics` | `api/routers/benchmark.py` |
| `GET` | `/api/config/` | `get_config` | `api/routers/config_mgmt.py` |
| `PATCH` | `/api/config/` | `patch_config` | `api/routers/config_mgmt.py` |
| `GET` | `/api/config/status` | `get_config_status` | `api/routers/config_mgmt.py` |
| `GET` | `/api/config/validate` | `validate_config_endpoint` | `api/routers/config_mgmt.py` |
| `GET` | `/api/dashboard/attribution` | `attribution` | `api/routers/dashboard.py` |
| `GET` | `/api/dashboard/equity` | `equity_with_benchmark` | `api/routers/dashboard.py` |
| `GET` | `/api/dashboard/regime` | `dashboard_regime` | `api/routers/dashboard.py` |
| `GET` | `/api/dashboard/returns-distribution` | `returns_distribution` | `api/routers/dashboard.py` |
| `GET` | `/api/dashboard/rolling-risk` | `rolling_risk` | `api/routers/dashboard.py` |
| `GET` | `/api/dashboard/summary` | `dashboard_summary` | `api/routers/dashboard.py` |
| `GET` | `/api/data/status` | `get_data_status` | `api/routers/data_explorer.py` |
| `GET` | `/api/data/ticker/{ticker}` | `get_ticker` | `api/routers/data_explorer.py` |
| `GET` | `/api/data/ticker/{ticker}/bars` | `get_ticker_bars` | `api/routers/data_explorer.py` |
| `GET` | `/api/data/ticker/{ticker}/indicators` | `get_ticker_indicators` | `api/routers/data_explorer.py` |
| `POST` | `/api/data/ticker/{ticker}/indicators/batch` | `batch_indicators` | `api/routers/data_explorer.py` |
| `GET` | `/api/data/universe` | `get_universe` | `api/routers/data_explorer.py` |
| `GET` | `/api/diagnostics/` | `get_diagnostics` | `api/routers/diagnostics.py` |
| `GET` | `/api/health` | `quick_health` | `api/routers/system_health.py` |
| `GET` | `/api/health/detailed` | `detailed_health` | `api/routers/system_health.py` |
| `GET` | `/api/health/history` | `health_history` | `api/routers/system_health.py` |
| `GET` | `/api/iv-surface/arb-free-svi` | `arb_free_svi_surface` | `api/routers/iv_surface.py` |
| `GET` | `/api/jobs/` | `list_jobs` | `api/routers/jobs.py` |
| `GET` | `/api/jobs/{job_id}` | `get_job` | `api/routers/jobs.py` |
| `POST` | `/api/jobs/{job_id}/cancel` | `cancel_job` | `api/routers/jobs.py` |
| `GET` | `/api/jobs/{job_id}/events` | `job_events` | `api/routers/jobs.py` |
| `GET` | `/api/logs/` | `get_logs` | `api/routers/logs.py` |
| `GET` | `/api/models/features/correlations` | `feature_correlations` | `api/routers/model_lab.py` |
| `GET` | `/api/models/features/importance` | `feature_importance` | `api/routers/model_lab.py` |
| `GET` | `/api/models/health` | `model_health` | `api/routers/model_lab.py` |
| `POST` | `/api/models/predict` | `predict_model` | `api/routers/model_lab.py` |
| `POST` | `/api/models/train` | `train_model` | `api/routers/model_lab.py` |
| `GET` | `/api/models/versions` | `list_versions` | `api/routers/model_lab.py` |
| `GET` | `/api/regime/metadata` | `regime_metadata` | `api/routers/regime.py` |
| `GET` | `/api/risk/factor-exposures` | `get_factor_exposures` | `api/routers/risk.py` |
| `GET` | `/api/signals/latest` | `latest_signals` | `api/routers/signals.py` |
| `GET` | `/api/v1/system/data-mode` | `data_mode` | `api/routers/system_health.py` |
| `GET` | `/api/v1/system/model-age` | `model_age` | `api/routers/system_health.py` |

## `(root)`

| Module | Lines | Classes | Top-level Functions | Intent |
|---|---|---|---|---|
| `__init__.py` | 7 | 0 | 0 | Quant Engine - Continuous Feature ML Trading System |
| `config.py` | 741 | 0 | 1 | Central configuration for the quant engine. |
| `config_structured.py` | 301 | 18 | 0 | Structured configuration for the quant engine using typed dataclasses. |
| `reproducibility.py` | 334 | 0 | 6 | Reproducibility locks for run manifests. |
| `run_autopilot.py` | 90 | 0 | 1 | Run one full autopilot cycle: |
| `run_backtest.py` | 429 | 0 | 1 | Backtest the trained model on historical data. |
| `run_kalshi_event_pipeline.py` | 228 | 0 | 2 | Run the integrated Kalshi event-time pipeline inside quant_engine. |
| `run_predict.py` | 203 | 0 | 1 | Generate predictions using trained ensemble model. |
| `run_rehydrate_cache_metadata.py` | 102 | 0 | 2 | Backfill cache metadata sidecars for existing OHLCV cache files. |
| `run_retrain.py` | 291 | 0 | 2 | Retrain the quant engine model — checks triggers and retrains if needed. |
| `run_server.py` | 81 | 0 | 1 | Combined API + frontend static serving entry point. |
| `run_train.py` | 196 | 0 | 1 | Train the regime-conditional ensemble model. |
| `run_wrds_daily_refresh.py` | 349 | 0 | 5 | Re-download all daily OHLCV data from WRDS CRSP to replace old cache files |
| `run_wrds_taq_intraday_download.py` | 680 | 0 | 12 | Download NYSE TAQ Daily Product intraday OHLCV for 128 tickers (2003-present) |

### `__init__.py`
- Intent: Quant Engine - Continuous Feature ML Trading System
- LOC: 7
- Classes: none
- Top-level functions: none

### `config.py`
- Intent: Central configuration for the quant engine.
- LOC: 741
- Classes: none
- Top-level functions: `validate_config`

### `config_structured.py`
- Intent: Structured configuration for the quant engine using typed dataclasses.
- LOC: 301
- Classes:
  - `ReturnType` (methods: none)
  - `PriceType` (methods: none)
  - `EntryType` (methods: none)
  - `PreconditionsConfig` (methods: none)
  - `CostStressConfig` (methods: none)
  - `DataConfig` (methods: none)
  - `RegimeConfig` (methods: none)
  - `ModelConfig` (methods: none)
  - `BacktestConfig` (methods: none)
  - `KellyConfig` (methods: none)
  - `DrawdownConfig` (methods: none)
  - `StopLossConfig` (methods: none)
  - `ValidationConfig` (methods: none)
  - `PromotionConfig` (methods: none)
  - `HealthConfig` (methods: none)
  - `PaperTradingConfig` (methods: none)
  - `ExecutionConfig` (methods: none)
  - `SystemConfig` (methods: none)
- Top-level functions: none

### `reproducibility.py`
- Intent: Reproducibility locks for run manifests.
- LOC: 334
- Classes: none
- Top-level functions: `_get_git_commit`, `_dataframe_checksum`, `build_run_manifest`, `write_run_manifest`, `verify_manifest`, `replay_manifest`

### `run_autopilot.py`
- Intent: Run one full autopilot cycle:
- LOC: 90
- Classes: none
- Top-level functions: `main`

### `run_backtest.py`
- Intent: Backtest the trained model on historical data.
- LOC: 429
- Classes: none
- Top-level functions: `main`

### `run_kalshi_event_pipeline.py`
- Intent: Run the integrated Kalshi event-time pipeline inside quant_engine.
- LOC: 228
- Classes: none
- Top-level functions: `_read_df`, `main`

### `run_predict.py`
- Intent: Generate predictions using trained ensemble model.
- LOC: 203
- Classes: none
- Top-level functions: `main`

### `run_rehydrate_cache_metadata.py`
- Intent: Backfill cache metadata sidecars for existing OHLCV cache files.
- LOC: 102
- Classes: none
- Top-level functions: `_parse_root_source`, `main`

### `run_retrain.py`
- Intent: Retrain the quant engine model — checks triggers and retrains if needed.
- LOC: 291
- Classes: none
- Top-level functions: `_check_regime_change_trigger`, `main`

### `run_server.py`
- Intent: Combined API + frontend static serving entry point.
- LOC: 81
- Classes: none
- Top-level functions: `main`

### `run_train.py`
- Intent: Train the regime-conditional ensemble model.
- LOC: 196
- Classes: none
- Top-level functions: `main`

### `run_wrds_daily_refresh.py`
- Intent: Re-download all daily OHLCV data from WRDS CRSP to replace old cache files
- LOC: 349
- Classes: none
- Top-level functions: `_build_ticker_list`, `_verify_file`, `_verify_all`, `_cleanup_old_daily`, `main`

### `run_wrds_taq_intraday_download.py`
- Intent: Download NYSE TAQ Daily Product intraday OHLCV for 128 tickers across 6 timeframes (2003-present)
- LOC: 680
- Classes: none
- Top-level functions: `discover_taq_schema`, `resample_from_1m`, `aggregate_to_1m`, `flush_ticker_data`, `run_download`, `run_verify`, `main`
- Dependencies: `quant_engine.config`, `quant_engine.data.local_cache`, `quant_engine.data.wrds_provider`, `quant_engine.data.intraday_quality`

## `api`

| Module | Lines | Classes | Top-level Functions | Intent |
|---|---|---|---|---|
| `api/__init__.py` | 2 | 0 | 0 | FastAPI backend for the quant engine. |
| `api/ab_testing.py` | 795 | 3 | 0 | A/B testing framework for strategy evaluation. |
| `api/cache/__init__.py` | 5 | 0 | 0 | TTL cache with event-driven invalidation. |
| `api/cache/invalidation.py` | 33 | 0 | 4 | Event-driven cache invalidation helpers. |
| `api/cache/manager.py` | 63 | 1 | 0 | In-memory TTL cache manager. |
| `api/config.py` | 76 | 2 | 0 | Runtime-adjustable configuration for the API layer. |
| `api/deps/__init__.py` | 17 | 0 | 0 | Dependency injection providers. |
| `api/deps/providers.py` | 55 | 0 | 5 | Singleton dependency providers for FastAPI ``Depends()``. |
| `api/errors.py` | 78 | 5 | 2 | Custom exceptions and FastAPI error handler registration. |
| `api/jobs/__init__.py` | 7 | 0 | 0 | SQLite-backed job queue for long-running compute. |
| `api/jobs/autopilot_job.py` | 32 | 0 | 1 | Autopilot job executor. |
| `api/jobs/backtest_job.py` | 30 | 0 | 1 | Backtest job executor. |
| `api/jobs/models.py` | 33 | 2 | 0 | Job data models. |
| `api/jobs/predict_job.py` | 29 | 0 | 1 | Predict job executor. |
| `api/jobs/runner.py` | 116 | 1 | 0 | Async job runner with concurrency control and SSE event streaming. |
| `api/jobs/store.py` | 146 | 1 | 0 | SQLite-backed persistence for job records. |
| `api/jobs/train_job.py` | 31 | 0 | 1 | Train job executor. |
| `api/main.py` | 149 | 0 | 4 | FastAPI application factory and server entry point. |
| `api/orchestrator.py` | 373 | 2 | 0 | Unified pipeline orchestrator — data -> features -> regimes -> compute. |
| `api/routers/__init__.py` | 43 | 0 | 1 | Route modules — imported lazily by the app factory. |
| `api/routers/autopilot.py` | 84 | 0 | 5 | Autopilot endpoints — cycle reports, strategies, paper state, run-cycle. |
| `api/routers/backtests.py` | 90 | 0 | 5 | Backtest result + compute endpoints. |
| `api/routers/benchmark.py` | 129 | 0 | 6 | Benchmark comparison endpoints. |
| `api/routers/config_mgmt.py` | 147 | 0 | 6 | Runtime config management endpoints. |
| `api/routers/dashboard.py` | 171 | 0 | 10 | Dashboard endpoints — KPIs, regime overview, time series analytics. |
| `api/routers/data_explorer.py` | 491 | 0 | 21 | Data explorer endpoints — universe + per-ticker OHLCV + bars + indicators. |
| `api/routers/diagnostics.py` | 66 | 0 | 1 | Diagnostics API router — system self-diagnosis and root-cause analysis. |
| `api/routers/iv_surface.py` | 72 | 0 | 2 | IV Surface computation endpoints. |
| `api/routers/jobs.py` | 73 | 0 | 5 | Job management endpoints. |
| `api/routers/logs.py` | 39 | 1 | 1 | Log retrieval endpoint. |
| `api/routers/model_lab.py` | 91 | 0 | 6 | Model lab endpoints — versions, health, feature importance, train, predict. |
| `api/routers/regime.py` | 134 | 0 | 2 | Regime detection endpoints — metadata, current state, history. |
| `api/routers/risk.py` | 54 | 0 | 1 | Risk API router — factor exposures, diagnostics, and monitoring endpoints. |
| `api/routers/signals.py` | 52 | 0 | 2 | Signal / prediction endpoints. |
| `api/routers/system_health.py` | 122 | 0 | 5 | System health endpoints. |
| `api/schemas/__init__.py` | 5 | 0 | 0 | Pydantic schemas for API request/response models. |
| `api/schemas/autopilot.py` | 31 | 3 | 0 | Autopilot schemas. |
| `api/schemas/backtests.py` | 57 | 4 | 0 | Backtest schemas. |
| `api/schemas/compute.py` | 57 | 5 | 0 | Request schemas for compute (POST) endpoints. |
| `api/schemas/dashboard.py` | 39 | 3 | 0 | Dashboard-related schemas. |
| `api/schemas/data_explorer.py` | 38 | 3 | 0 | Data explorer schemas. |
| `api/schemas/envelope.py` | 55 | 2 | 0 | Standard API response envelope with provenance metadata. |
| `api/schemas/model_lab.py` | 44 | 3 | 0 | Model lab schemas. |
| `api/schemas/signals.py` | 27 | 2 | 0 | Signal schemas. |
| `api/schemas/system_health.py` | 48 | 3 | 0 | System health schemas. |
| `api/services/__init__.py` | 21 | 0 | 0 | Engine wrapper services — sync functions returning plain dicts. |
| `api/services/autopilot_service.py` | 63 | 1 | 0 | Wraps autopilot engine and results for API consumption. |
| `api/services/backtest_service.py` | 140 | 1 | 0 | Wraps backtest results for API consumption. |
| `api/services/data_helpers.py` | 1059 | 2 | 22 | Data loading and computation functions extracted from dash_ui/data/loaders.py. |
| `api/services/data_service.py` | 209 | 1 | 0 | Wraps data.loader for API consumption. |
| `api/services/diagnostics.py` | 289 | 3 | 0 | System Diagnostics — explains WHY performance is degrading. |
| `api/services/health_alerts.py` | 325 | 2 | 2 | Health alert and notification system — Spec 09. |
| `api/services/health_confidence.py` | 315 | 2 | 0 | Health score confidence interval computation — Spec 09. |
| `api/services/health_risk_feedback.py` | 276 | 1 | 2 | Health-to-risk feedback loop — Spec 09. |
| `api/services/health_service.py` | 1954 | 2 | 1 | System health assessment for API consumption. |
| `api/services/kalshi_service.py` | 51 | 1 | 0 | Wraps kalshi.storage for API consumption. |
| `api/services/model_service.py` | 99 | 1 | 0 | Wraps models.* modules for API consumption. |
| `api/services/regime_service.py` | 51 | 1 | 0 | Wraps regime.detector for API consumption. |
| `api/services/results_service.py` | 86 | 1 | 0 | Reads/writes to the results/ directory. |

### `api/__init__.py`
- Intent: FastAPI backend for the quant engine.
- LOC: 2
- Classes: none
- Top-level functions: none

### `api/ab_testing.py`
- Intent: A/B testing framework for strategy evaluation.
- LOC: 795
- Classes:
  - `ABVariant` (methods: `n_trades`, `returns`, `mean_return`, `sharpe`, `sortino`, `max_drawdown`, `win_rate`, `profit_factor`, `turnover`, `total_transaction_costs`)
  - `ABTest` (methods: `record_trade`, `assign_variant`, `get_variant_config`, `compute_required_samples`, `check_early_stopping`, `get_results`, `get_test_report`, `to_dict`)
  - `ABTestRegistry` (methods: `create_test`, `get_test`, `get_active_test`, `list_tests`, `complete_test`, `cancel_test`)
- Top-level functions: none

### `api/cache/__init__.py`
- Intent: TTL cache with event-driven invalidation.
- LOC: 5
- Classes: none
- Top-level functions: none

### `api/cache/invalidation.py`
- Intent: Event-driven cache invalidation helpers.
- LOC: 33
- Classes: none
- Top-level functions: `invalidate_on_train`, `invalidate_on_backtest`, `invalidate_on_data_refresh`, `invalidate_on_config_change`

### `api/cache/manager.py`
- Intent: In-memory TTL cache manager.
- LOC: 63
- Classes:
  - `CacheManager` (methods: `get`, `set`, `invalidate`, `invalidate_pattern`, `invalidate_all`)
- Top-level functions: none

### `api/config.py`
- Intent: Runtime-adjustable configuration for the API layer.
- LOC: 76
- Classes:
  - `ApiSettings` (methods: none)
  - `RuntimeConfig` (methods: `get_adjustable`, `patch`)
- Top-level functions: none

### `api/deps/__init__.py`
- Intent: Dependency injection providers.
- LOC: 17
- Classes: none
- Top-level functions: none

### `api/deps/providers.py`
- Intent: Singleton dependency providers for FastAPI ``Depends()``.
- LOC: 55
- Classes: none
- Top-level functions: `get_settings`, `get_runtime_config`, `get_job_store`, `get_job_runner`, `get_cache`

### `api/errors.py`
- Intent: Custom exceptions and FastAPI error handler registration.
- LOC: 78
- Classes:
  - `DataNotFoundError` (methods: none)
  - `TrainingFailedError` (methods: none)
  - `JobNotFoundError` (methods: none)
  - `ConfigValidationError` (methods: none)
  - `ServiceUnavailableError` (methods: none)
- Top-level functions: `_make_handler`, `register_error_handlers`

### `api/jobs/__init__.py`
- Intent: SQLite-backed job queue for long-running compute.
- LOC: 7
- Classes: none
- Top-level functions: none

### `api/jobs/autopilot_job.py`
- Intent: Autopilot job executor.
- LOC: 32
- Classes: none
- Top-level functions: `execute_autopilot_job`

### `api/jobs/backtest_job.py`
- Intent: Backtest job executor.
- LOC: 30
- Classes: none
- Top-level functions: `execute_backtest_job`

### `api/jobs/models.py`
- Intent: Job data models.
- LOC: 33
- Classes:
  - `JobStatus` (methods: none)
  - `JobRecord` (methods: none)
- Top-level functions: none

### `api/jobs/predict_job.py`
- Intent: Predict job executor.
- LOC: 29
- Classes: none
- Top-level functions: `execute_predict_job`

### `api/jobs/runner.py`
- Intent: Async job runner with concurrency control and SSE event streaming.
- LOC: 116
- Classes:
  - `JobRunner` (methods: `submit`, `cancel`, `subscribe_events`)
- Top-level functions: none

### `api/jobs/store.py`
- Intent: SQLite-backed persistence for job records.
- LOC: 146
- Classes:
  - `JobStore` (methods: `initialize`, `close`, `create_job`, `get_job`, `list_jobs`, `update_status`, `update_progress`, `cancel_job`)
- Top-level functions: none

### `api/jobs/train_job.py`
- Intent: Train job executor.
- LOC: 31
- Classes: none
- Top-level functions: `execute_train_job`

### `api/main.py`
- Intent: FastAPI application factory and server entry point.
- LOC: 149
- Classes: none
- Top-level functions: `_retrain_monitor_loop`, `_lifespan`, `create_app`, `run_server`

### `api/orchestrator.py`
- Intent: Unified pipeline orchestrator — data -> features -> regimes -> compute.
- LOC: 373
- Classes:
  - `PipelineState` (methods: none)
  - `PipelineOrchestrator` (methods: `load_and_prepare`, `train`, `predict`, `backtest`)
- Top-level functions: none

### `api/routers/__init__.py`
- Intent: Route modules — imported lazily by the app factory.
- LOC: 43
- Classes: none
- Top-level functions: `all_routers`

### `api/routers/autopilot.py`
- Intent: Autopilot endpoints — cycle reports, strategies, paper state, run-cycle.
- LOC: 84
- Classes: none
- Top-level functions: `_get_autopilot_meta`, `latest_cycle`, `strategies`, `paper_state`, `run_cycle`

### `api/routers/backtests.py`
- Intent: Backtest result + compute endpoints.
- LOC: 90
- Classes: none
- Top-level functions: `_extract_backtest_meta`, `latest_backtest`, `latest_trades`, `equity_curve`, `run_backtest`

### `api/routers/benchmark.py`
- Intent: Benchmark comparison endpoints.
- LOC: 129
- Classes: none
- Top-level functions: `_compute_comparison`, `_compute_equity_curves`, `_compute_rolling_metrics`, `benchmark_comparison`, `benchmark_equity_curves`, `benchmark_rolling_metrics`

### `api/routers/config_mgmt.py`
- Intent: Runtime config management endpoints.
- LOC: 147
- Classes: none
- Top-level functions: `_annotated`, `_build_config_status`, `get_config`, `validate_config_endpoint`, `get_config_status`, `patch_config`

### `api/routers/dashboard.py`
- Intent: Dashboard endpoints — KPIs, regime overview, time series analytics.
- LOC: 171
- Classes: none
- Top-level functions: `dashboard_summary`, `dashboard_regime`, `_compute_returns_distribution`, `_compute_rolling_risk`, `_compute_equity_with_benchmark`, `_compute_attribution`, `returns_distribution`, `rolling_risk`, `equity_with_benchmark`, `attribution`

### `api/routers/data_explorer.py`
- Intent: Data explorer endpoints — universe + per-ticker OHLCV + bars + indicators.
- LOC: 491
- Classes: none
- Top-level functions: `_find_cached_parquet`, `_available_timeframes`, `_load_bars`, `_compute_sma`, `_compute_ema`, `_compute_bollinger`, `_compute_rsi`, `_compute_macd`, `_compute_atr`, `_compute_stochastic`, `_compute_obv`, `_compute_vwap`, `_compute_adx`, `_parse_indicator_spec`, `_compute_indicators_for_ticker`, `get_universe`, `get_data_status`, `get_ticker`, `get_ticker_bars`, `get_ticker_indicators`, `batch_indicators`

### `api/routers/diagnostics.py`
- Intent: Diagnostics API router — system self-diagnosis and root-cause analysis.
- LOC: 66
- Classes: none
- Top-level functions: `get_diagnostics`

### `api/routers/iv_surface.py`
- Intent: IV Surface computation endpoints.
- LOC: 72
- Classes: none
- Top-level functions: `_compute_arb_free_svi`, `arb_free_svi_surface`

### `api/routers/jobs.py`
- Intent: Job management endpoints.
- LOC: 73
- Classes: none
- Top-level functions: `_not_found`, `list_jobs`, `get_job`, `job_events`, `cancel_job`

### `api/routers/logs.py`
- Intent: Log retrieval endpoint.
- LOC: 39
- Classes:
  - `_BufferHandler` (methods: `emit`)
- Top-level functions: `get_logs`

### `api/routers/model_lab.py`
- Intent: Model lab endpoints — versions, health, feature importance, train, predict.
- LOC: 91
- Classes: none
- Top-level functions: `list_versions`, `model_health`, `feature_importance`, `feature_correlations`, `train_model`, `predict_model`

### `api/routers/regime.py`
- Intent: Regime detection endpoints — metadata, current state, history.
- LOC: 134
- Classes: none
- Top-level functions: `_build_regime_metadata`, `regime_metadata`

### `api/routers/risk.py`
- Intent: Risk API router — factor exposures, diagnostics, and monitoring endpoints.
- LOC: 54
- Classes: none
- Top-level functions: `get_factor_exposures`

### `api/routers/signals.py`
- Intent: Signal / prediction endpoints.
- LOC: 52
- Classes: none
- Top-level functions: `_get_signal_meta_fields`, `latest_signals`

### `api/routers/system_health.py`
- Intent: System health endpoints.
- LOC: 122
- Classes: none
- Top-level functions: `quick_health`, `detailed_health`, `health_history`, `model_age`, `data_mode`

### `api/schemas/__init__.py`
- Intent: Pydantic schemas for API request/response models.
- LOC: 5
- Classes: none
- Top-level functions: none

### `api/schemas/autopilot.py`
- Intent: Autopilot schemas.
- LOC: 31
- Classes:
  - `CycleReport` (methods: none)
  - `StrategyInfo` (methods: none)
  - `PaperState` (methods: none)
- Top-level functions: none

### `api/schemas/backtests.py`
- Intent: Backtest schemas.
- LOC: 57
- Classes:
  - `BacktestSummary` (methods: none)
  - `TradeRecord` (methods: none)
  - `EquityCurvePoint` (methods: none)
  - `RegimeBreakdown` (methods: none)
- Top-level functions: none

### `api/schemas/compute.py`
- Intent: Request schemas for compute (POST) endpoints.
- LOC: 57
- Classes:
  - `TrainRequest` (methods: none)
  - `BacktestRequest` (methods: none)
  - `PredictRequest` (methods: none)
  - `AutopilotRequest` (methods: none)
  - `JobCreatedResponse` (methods: none)
- Top-level functions: none

### `api/schemas/dashboard.py`
- Intent: Dashboard-related schemas.
- LOC: 39
- Classes:
  - `DashboardKPIs` (methods: none)
  - `RegimeInfo` (methods: none)
  - `EquityPoint` (methods: none)
- Top-level functions: none

### `api/schemas/data_explorer.py`
- Intent: Data explorer schemas.
- LOC: 38
- Classes:
  - `UniverseInfo` (methods: none)
  - `OHLCVBar` (methods: none)
  - `TickerDetail` (methods: none)
- Top-level functions: none

### `api/schemas/envelope.py`
- Intent: Standard API response envelope with provenance metadata.
- LOC: 55
- Classes:
  - `ResponseMeta` (methods: none)
  - `ApiResponse` (methods: `success`, `fail`, `from_cached`)
- Top-level functions: none

### `api/schemas/model_lab.py`
- Intent: Model lab schemas.
- LOC: 44
- Classes:
  - `ModelVersionInfo` (methods: none)
  - `ModelHealth` (methods: none)
  - `FeatureImportance` (methods: none)
- Top-level functions: none

### `api/schemas/signals.py`
- Intent: Signal schemas.
- LOC: 27
- Classes:
  - `SignalRow` (methods: none)
  - `SignalsSummary` (methods: none)
- Top-level functions: none

### `api/schemas/system_health.py`
- Intent: System health schemas.
- LOC: 48
- Classes:
  - `QuickStatus` (methods: none)
  - `AlertEvent` (methods: none)
  - `SystemHealthDetail` (methods: none)
- Top-level functions: none

### `api/services/__init__.py`
- Intent: Engine wrapper services — sync functions returning plain dicts.
- LOC: 21
- Classes: none
- Top-level functions: none

### `api/services/autopilot_service.py`
- Intent: Wraps autopilot engine and results for API consumption.
- LOC: 63
- Classes:
  - `AutopilotService` (methods: `get_latest_cycle`, `get_strategy_registry`, `get_paper_state`)
- Top-level functions: none

### `api/services/backtest_service.py`
- Intent: Wraps backtest results for API consumption.
- LOC: 140
- Classes:
  - `BacktestService` (methods: `get_latest_results`, `get_latest_trades`, `get_equity_curve`)
- Top-level functions: none

### `api/services/data_helpers.py`
- Intent: Data loading and computation functions extracted from dash_ui/data/loaders.py.
- LOC: 1059
- Classes:
  - `HealthCheck` (methods: none)
  - `SystemHealthPayload` (methods: none)
- Top-level functions: `load_trades`, `build_portfolio_returns`, `_read_close_returns`, `load_benchmark_returns`, `build_equity_curves`, `compute_rolling_metrics`, `compute_returns_distribution`, `compute_rolling_risk`, `compute_attribution`, `_load_proxy_returns`, `compute_risk_metrics`, `compute_regime_payload`, `compute_model_health`, `load_feature_importance`, `score_to_status`, `collect_health_data`, `_check_data_integrity`, `_check_promotion_contract`, `_check_walkforward`, `_check_execution`, `_check_complexity`, `_check_strengths`

### `api/services/data_service.py`
- Intent: Wraps data.loader for API consumption.
- LOC: 209
- Classes:
  - `DataService` (methods: `load_universe`, `load_single_ticker`, `get_cached_tickers`, `get_universe_info`, `get_cache_status`)
- Top-level functions: none

### `api/services/diagnostics.py`
- Intent: System Diagnostics — explains WHY performance is degrading.
- LOC: 289
- Classes:
  - `DiagnosticFinding` (methods: none)
  - `DiagnosticReport` (methods: none)
  - `SystemDiagnostics` (methods: `diagnose_performance`)
- Top-level functions: none

### `api/services/health_alerts.py`
- Intent: Health alert and notification system — Spec 09.
- LOC: 325
- Classes:
  - `Alert` (methods: `dedup_key`, `to_dict`)
  - `HealthAlertManager` (methods: `check_health_degradation`, `check_domain_failures`, `check_low_confidence`, `process_alerts`)
- Top-level functions: `load_alert_config`, `create_alert_manager`

### `api/services/health_confidence.py`
- Intent: Health score confidence interval computation — Spec 09.
- LOC: 315
- Classes:
  - `ConfidenceResult` (methods: `ci_width`, `is_low_confidence`, `to_dict`)
  - `HealthConfidenceCalculator` (methods: `compute_ci`, `compute_ci_bootstrap`, `compute_ci_normal`, `compute_ci_t`, `compute_ci_binomial`, `propagate_weighted_ci`)
- Top-level functions: none

### `api/services/health_risk_feedback.py`
- Intent: Health-to-risk feedback loop — Spec 09.
- LOC: 276
- Classes:
  - `HealthRiskGate` (methods: `update_health`, `compute_size_multiplier`, `apply_health_gate`, `apply_health_gate_weights`, `should_halt_trading`, `get_status`)
- Top-level functions: `load_health_risk_config`, `create_health_risk_gate`

### `api/services/health_service.py`
- Intent: System health assessment for API consumption.
- LOC: 1954
- Classes:
  - `HealthCheckResult` (methods: `to_dict`)
  - `HealthService` (methods: `get_quick_status`, `get_detailed_health`, `compute_comprehensive_health`, `save_health_snapshot`, `get_health_history`, `get_health_history_with_trends`)
- Top-level functions: `_unavailable`

### `api/services/kalshi_service.py`
- Intent: Wraps kalshi.storage for API consumption.
- LOC: 51
- Classes:
  - `KalshiService` (methods: `get_events`, `get_distributions`)
- Top-level functions: none

### `api/services/model_service.py`
- Intent: Wraps models.* modules for API consumption.
- LOC: 99
- Classes:
  - `ModelService` (methods: `list_versions`, `get_model_health`, `get_feature_importance`, `get_feature_correlations`, `get_champion_info`)
- Top-level functions: none

### `api/services/regime_service.py`
- Intent: Wraps regime.detector for API consumption.
- LOC: 51
- Classes:
  - `RegimeService` (methods: `detect_current_regime`, `get_regime_names`)
- Top-level functions: none

### `api/services/results_service.py`
- Intent: Reads/writes to the results/ directory.
- LOC: 86
- Classes:
  - `ResultsService` (methods: `get_latest_backtest`, `get_latest_predictions`, `list_all_results`)
- Top-level functions: none

## `autopilot`

| Module | Lines | Classes | Top-level Functions | Intent |
|---|---|---|---|---|
| `autopilot/__init__.py` | 23 | 0 | 0 | Autopilot layer: discovery, promotion, and paper-trading orchestration. |
| `autopilot/engine.py` | 1395 | 2 | 0 | End-to-end autopilot cycle: |
| `autopilot/meta_labeler.py` | 457 | 1 | 0 | Meta-labeling model for signal confidence prediction (Spec 04). |
| `autopilot/paper_trader.py` | 1094 | 1 | 0 | Stateful paper-trading engine for promoted strategies. |
| `autopilot/promotion_gate.py` | 328 | 2 | 0 | Promotion gate for deciding whether a discovered strategy is deployable. |
| `autopilot/registry.py` | 111 | 2 | 0 | Persistent strategy registry for promoted candidates. |
| `autopilot/strategy_allocator.py` | 197 | 2 | 0 | Regime-Aware Strategy Allocation — automatically adjust strategy parameters |
| `autopilot/strategy_discovery.py` | 80 | 2 | 0 | Strategy discovery for execution-layer parameter variants. |

### `autopilot/__init__.py`
- Intent: Autopilot layer: discovery, promotion, and paper-trading orchestration.
- LOC: 23
- Classes: none
- Top-level functions: none

### `autopilot/engine.py`
- Intent: End-to-end autopilot cycle:
- LOC: 1395
- Classes:
  - `HeuristicPredictor` (methods: `predict`)
  - `AutopilotEngine` (methods: `run_cycle`)
- Top-level functions: none

### `autopilot/meta_labeler.py`
- Intent: Meta-labeling model for signal confidence prediction (Spec 04).
- LOC: 457
- Classes:
  - `MetaLabelingModel` (methods: `build_meta_features`, `build_labels`, `train`, `predict_confidence`, `save`, `load`, `is_trained`)
- Top-level functions: none

### `autopilot/paper_trader.py`
- Intent: Stateful paper-trading engine for promoted strategies.
- LOC: 1094
- Classes:
  - `PaperTrader` (methods: `run_cycle`)
- Top-level functions: none

### `autopilot/promotion_gate.py`
- Intent: Promotion gate for deciding whether a discovered strategy is deployable.
- LOC: 328
- Classes:
  - `PromotionDecision` (methods: `to_dict`)
  - `PromotionGate` (methods: `evaluate`, `evaluate_event_strategy`, `rank`)
- Top-level functions: none

### `autopilot/registry.py`
- Intent: Persistent strategy registry for promoted candidates.
- LOC: 111
- Classes:
  - `ActiveStrategy` (methods: `to_dict`)
  - `StrategyRegistry` (methods: `get_active`, `apply_promotions`)
- Top-level functions: none

### `autopilot/strategy_allocator.py`
- Intent: Regime-Aware Strategy Allocation — automatically adjust strategy parameters
- LOC: 197
- Classes:
  - `StrategyProfile` (methods: none)
  - `StrategyAllocator` (methods: `get_regime_profile`, `get_all_profiles`, `summarize`)
- Top-level functions: none

### `autopilot/strategy_discovery.py`
- Intent: Strategy discovery for execution-layer parameter variants.
- LOC: 80
- Classes:
  - `StrategyCandidate` (methods: `to_dict`)
  - `StrategyDiscovery` (methods: `generate`)
- Top-level functions: none

## `backtest`

| Module | Lines | Classes | Top-level Functions | Intent |
|---|---|---|---|---|
| `backtest/__init__.py` | 5 | 0 | 0 | Backtesting package exports and namespace initialization. |
| `backtest/adv_tracker.py` | 192 | 1 | 0 | Average Daily Volume (ADV) tracker with EMA smoothing and volume trend analysis. |
| `backtest/advanced_validation.py` | 582 | 5 | 6 | Advanced Validation — Deflated Sharpe, PBO, Monte Carlo, capacity analysis. |
| `backtest/cost_calibrator.py` | 316 | 1 | 0 | Cost model calibrator for per-market-cap-segment impact coefficients. |
| `backtest/cost_stress.py` | 223 | 3 | 0 | Cost stress testing — Truth Layer T5. |
| `backtest/engine.py` | 1957 | 3 | 0 | Backtester — converts model predictions into simulated trades. |
| `backtest/execution.py` | 637 | 2 | 1 | Execution simulator with spread, market impact, and participation limits. |
| `backtest/null_models.py` | 298 | 5 | 2 | Null model baselines — Truth Layer T4. |
| `backtest/optimal_execution.py` | 202 | 0 | 2 | Almgren-Chriss (2001) optimal execution model. |
| `backtest/survivorship_comparison.py` | 165 | 2 | 3 | Survivorship Bias Comparison — quantify the impact of survivorship bias on backtests. |
| `backtest/validation.py` | 1075 | 7 | 12 | Walk-forward validation and statistical tests. |

### `backtest/__init__.py`
- Intent: Backtesting package exports and namespace initialization.
- LOC: 5
- Classes: none
- Top-level functions: none

### `backtest/adv_tracker.py`
- Intent: Average Daily Volume (ADV) tracker with EMA smoothing and volume trend analysis.
- LOC: 192
- Classes:
  - `ADVTracker` (methods: `update`, `update_from_series`, `get_adv`, `get_simple_adv`, `get_volume_trend`, `adjust_participation_limit`, `get_volume_cost_adjustment`, `get_stats`)
- Top-level functions: none

### `backtest/advanced_validation.py`
- Intent: Advanced Validation — Deflated Sharpe, PBO, Monte Carlo, capacity analysis.
- LOC: 582
- Classes:
  - `DeflatedSharpeResult` (methods: none)
  - `PBOResult` (methods: none)
  - `MonteCarloResult` (methods: none)
  - `CapacityResult` (methods: none)
  - `AdvancedValidationReport` (methods: none)
- Top-level functions: `deflated_sharpe_ratio`, `probability_of_backtest_overfitting`, `monte_carlo_validation`, `capacity_analysis`, `run_advanced_validation`, `_print_report`

### `backtest/cost_calibrator.py`
- Intent: Cost model calibrator for per-market-cap-segment impact coefficients.
- LOC: 316
- Classes:
  - `CostCalibrator` (methods: `get_marketcap_segment`, `get_impact_coeff`, `get_impact_coeff_by_segment`, `coefficients`, `record_trade`, `calibrate`, `reset_history`)
- Top-level functions: none

### `backtest/cost_stress.py`
- Intent: Cost stress testing — Truth Layer T5.
- LOC: 223
- Classes:
  - `CostStressPoint` (methods: none)
  - `CostStressResult` (methods: `to_dict`)
  - `CostStressTester` (methods: `run_sweep`, `report`)
- Top-level functions: none

### `backtest/engine.py`
- Intent: Backtester — converts model predictions into simulated trades.
- LOC: 1957
- Classes:
  - `Trade` (methods: none)
  - `BacktestResult` (methods: `summarize_vs_null`)
  - `Backtester` (methods: `run`)
- Top-level functions: none

### `backtest/execution.py`
- Intent: Execution simulator with spread, market impact, and participation limits.
- LOC: 637
- Classes:
  - `ExecutionFill` (methods: none)
  - `ExecutionModel` (methods: `set_base_transaction_cost_bps`, `simulate`)
- Top-level functions: `calibrate_cost_model`

### `backtest/null_models.py`
- Intent: Null model baselines — Truth Layer T4.
- LOC: 298
- Classes:
  - `NullBaselineMetrics` (methods: none)
  - `NullModelResults` (methods: `summary`)
  - `RandomBaseline` (methods: `generate_signals`, `compute_returns`)
  - `ZeroBaseline` (methods: `generate_signals`, `compute_returns`)
  - `MomentumBaseline` (methods: `generate_signals`, `compute_returns`)
- Top-level functions: `_compute_metrics`, `compute_null_baselines`

### `backtest/optimal_execution.py`
- Intent: Almgren-Chriss (2001) optimal execution model.
- LOC: 202
- Classes: none
- Top-level functions: `almgren_chriss_trajectory`, `estimate_execution_cost`

### `backtest/survivorship_comparison.py`
- Intent: Survivorship Bias Comparison — quantify the impact of survivorship bias on backtests.
- LOC: 165
- Classes:
  - `UniverseMetrics` (methods: none)
  - `SurvivorshipComparisonResult` (methods: none)
- Top-level functions: `_extract_metrics`, `compare_survivorship_impact`, `quick_survivorship_check`

### `backtest/validation.py`
- Intent: Walk-forward validation and statistical tests.
- LOC: 1075
- Classes:
  - `WalkForwardFold` (methods: none)
  - `WalkForwardResult` (methods: none)
  - `StatisticalTests` (methods: none)
  - `CPCVResult` (methods: none)
  - `SPAResult` (methods: none)
  - `WalkForwardEmbargoFold` (methods: none)
  - `WalkForwardEmbargoResult` (methods: none)
- Top-level functions: `walk_forward_validate`, `_benjamini_hochberg`, `run_statistical_tests`, `_partition_bounds`, `combinatorial_purged_cv`, `strategy_signal_returns`, `superior_predictive_ability`, `walk_forward_with_embargo`, `rolling_ic`, `detect_ic_decay`, `_sharpe`, `_spearman_ic`

## `data`

| Module | Lines | Classes | Top-level Functions | Intent |
|---|---|---|---|---|
| `data/__init__.py` | 33 | 0 | 0 | Data subpackage — self-contained data loading, caching, WRDS, and survivorship. |
| `data/alternative.py` | 653 | 1 | 2 | Alternative data framework — WRDS-backed implementation. |
| `data/cross_source_validator.py` | 680 | 2 | 0 | Cross-source validation system comparing Alpaca/Alpha Vantage against IBKR. |
| `data/feature_store.py` | 313 | 1 | 0 | Point-in-time feature store for backtest acceleration. |
| `data/intraday_quality.py` | 967 | 2 | 20 | Comprehensive quality gate for intraday OHLCV data. |
| `data/loader.py` | 832 | 0 | 17 | Data loader — self-contained data loading with multiple sources. |
| `data/local_cache.py` | 703 | 0 | 21 | Local data cache for daily OHLCV data. |
| `data/provider_base.py` | 15 | 1 | 0 | Shared provider protocol for pluggable data connectors. |
| `data/provider_registry.py` | 54 | 0 | 5 | Provider registry for unified data-provider access (WRDS, Kalshi, ...). |
| `data/quality.py` | 297 | 1 | 4 | Data quality checks for OHLCV time series. |
| `data/survivorship.py` | 936 | 8 | 5 | Survivorship Bias Controls (Tasks 112-117) |
| `data/wrds_provider.py` | 1616 | 1 | 6 | wrds_provider.py |

### `data/__init__.py`
- Intent: Data subpackage — self-contained data loading, caching, WRDS, and survivorship.
- LOC: 33
- Classes: none
- Top-level functions: none

### `data/alternative.py`
- Intent: Alternative data framework — WRDS-backed implementation.
- LOC: 653
- Classes:
  - `AlternativeDataProvider` (methods: `get_earnings_surprise`, `get_options_flow`, `get_short_interest`, `get_insider_transactions`, `get_institutional_ownership`)
- Top-level functions: `_get_wrds`, `compute_alternative_features`

### `data/cross_source_validator.py`
- Intent: Cross-source validation system comparing Alpaca/Alpha Vantage against IBKR.
- LOC: 680
- Classes:
  - `CrossValidationReport` (methods: none)
  - `CrossSourceValidator` (methods: `validate_ticker`)
- Top-level functions: none

### `data/feature_store.py`
- Intent: Point-in-time feature store for backtest acceleration.
- LOC: 313
- Classes:
  - `FeatureStore` (methods: `save_features`, `load_features`, `list_available`, `invalidate`)
- Top-level functions: none

### `data/intraday_quality.py`
- Intent: Comprehensive quality gate for intraday OHLCV data.
- LOC: 967
- Classes:
  - `CheckResult` (methods: none)
  - `IntradayQualityReport` (methods: `add_check`, `compute_quality_score`)
- Top-level functions: `_get_trading_days`, `_is_in_rth`, `_get_expected_bar_count`, `_check_ohlc_consistency`, `_check_non_negative_volume`, `_check_non_negative_prices`, `_check_timestamp_in_rth`, `_check_extreme_bar_return`, `_check_stale_price`, `_check_zero_volume_liquid`, `_check_missing_bar_ratio`, `_check_duplicate_timestamps`, `_check_monotonic_index`, `_check_overnight_gap`, `_check_volume_distribution`, `_check_split_detection`, `quarantine_ticker`, `write_quality_report`, `read_quality_report`, `validate_intraday_bars`

### `data/loader.py`
- Intent: Data loader — self-contained data loading with multiple sources.
- LOC: 832
- Classes: none
- Top-level functions: `_permno_from_meta`, `_ticker_from_meta`, `_attach_id_attrs`, `_cache_source`, `_get_last_trading_day`, `_trading_days_between`, `_cache_is_usable`, `_cached_universe_subset`, `_normalize_ohlcv`, `_harmonize_return_columns`, `_merge_option_surface_from_prefetch`, `load_ohlcv`, `get_data_provenance`, `get_skip_reasons`, `load_universe`, `load_survivorship_universe`, `load_with_delistings`

### `data/local_cache.py`
- Intent: Local data cache for daily OHLCV data.
- LOC: 703
- Classes: none
- Top-level functions: `_ensure_cache_dir`, `_normalize_ohlcv_columns`, `_to_daily_ohlcv`, `_read_csv_ohlcv`, `_candidate_csv_paths`, `_cache_meta_path`, `_read_cache_meta`, `_write_cache_meta`, `save_ohlcv`, `load_ohlcv_with_meta`, `load_ohlcv`, `load_intraday_ohlcv`, `list_intraday_timeframes`, `list_cached_tickers`, `_daily_cache_files`, `_ticker_from_cache_path`, `_timeframe_from_cache_path`, `_all_cache_files`, `rehydrate_cache_metadata`, `load_ibkr_data`, `cache_universe`

### `data/provider_base.py`
- Intent: Shared provider protocol for pluggable data connectors.
- LOC: 15
- Classes:
  - `DataProvider` (methods: `available`)
- Top-level functions: none

### `data/provider_registry.py`
- Intent: Provider registry for unified data-provider access (WRDS, Kalshi, ...).
- LOC: 54
- Classes: none
- Top-level functions: `_wrds_factory`, `_kalshi_factory`, `get_provider`, `list_providers`, `register_provider`

### `data/quality.py`
- Intent: Data quality checks for OHLCV time series.
- LOC: 297
- Classes:
  - `DataQualityReport` (methods: `to_dict`)
- Top-level functions: `_expected_trading_days`, `assess_ohlcv_quality`, `generate_quality_report`, `flag_degraded_stocks`

### `data/survivorship.py`
- Intent: Survivorship Bias Controls (Tasks 112-117)
- LOC: 936
- Classes:
  - `DelistingReason` (methods: none)
  - `UniverseMember` (methods: `is_active_on`, `to_dict`)
  - `UniverseChange` (methods: `to_dict`)
  - `DelistingEvent` (methods: `to_dict`)
  - `SurvivorshipReport` (methods: `to_dict`)
  - `UniverseHistoryTracker` (methods: `add_member`, `record_change`, `get_universe_on_date`, `get_changes_in_period`, `bulk_load_universe`, `clear_universe`)
  - `DelistingHandler` (methods: `record_delisting`, `preserve_price_history`, `get_dead_company_prices`, `get_delisting_event`, `get_delisting_return`, `is_delisted`, `get_all_delisted_symbols`)
  - `SurvivorshipBiasController` (methods: `get_survivorship_free_universe`, `calculate_bias_impact`, `format_report`)
- Top-level functions: `hydrate_universe_history_from_snapshots`, `hydrate_sp500_history_from_wrds`, `filter_panel_by_point_in_time_universe`, `reconstruct_historical_universe`, `calculate_survivorship_bias_impact`

### `data/wrds_provider.py`
- Intent: wrds_provider.py
- LOC: 1616
- Classes:
  - `WRDSProvider` (methods: `available`, `get_sp500_universe`, `get_sp500_history`, `resolve_permno`, `get_crsp_prices`, `get_crsp_prices_with_delistings`, `get_optionmetrics_link`, `get_option_surface_features`, `get_fundamentals`, `get_earnings_surprises`, `get_institutional_ownership`, `get_taqmsec_ohlcv` (+4 more))
- Top-level functions: `_sanitize_ticker_list`, `_sanitize_permno_list`, `_read_pgpass_password`, `_get_connection`, `get_wrds_provider`, `wrds_available`

## `features`

| Module | Lines | Classes | Top-level Functions | Intent |
|---|---|---|---|---|
| `features/__init__.py` | 5 | 0 | 0 | Feature engineering package namespace. |
| `features/harx_spillovers.py` | 243 | 0 | 3 | HARX Volatility Spillover features (Tier 6.1). |
| `features/intraday.py` | 244 | 0 | 2 | Intraday microstructure features from WRDS TAQmsec tick data. |
| `features/lob_features.py` | 312 | 0 | 5 | Markov LOB (Limit Order Book) features from intraday bar data (Tier 6.2). |
| `features/macro.py` | 245 | 1 | 1 | FRED macro indicator features for quant_engine. |
| `features/options_factors.py` | 135 | 0 | 4 | Option surface factor construction from OptionMetrics-enriched daily panels. |
| `features/pipeline.py` | 1542 | 1 | 13 | Feature Pipeline — computes model features from OHLCV data. |
| `features/research_factors.py` | 986 | 1 | 19 | Research-derived factor construction for quant_engine. |
| `features/version.py` | 169 | 2 | 0 | Feature versioning system. |
| `features/wave_flow.py` | 145 | 0 | 1 | Wave-Flow Decomposition for quant_engine. |

### `features/__init__.py`
- Intent: Feature engineering package namespace.
- LOC: 5
- Classes: none
- Top-level functions: none

### `features/harx_spillovers.py`
- Intent: HARX Volatility Spillover features (Tier 6.1).
- LOC: 243
- Classes: none
- Top-level functions: `_realized_volatility`, `_ols_lstsq`, `compute_harx_spillovers`

### `features/intraday.py`
- Intent: Intraday microstructure features from WRDS TAQmsec tick data.
- LOC: 244
- Classes: none
- Top-level functions: `compute_intraday_features`, `compute_rolling_vwap`

### `features/lob_features.py`
- Intent: Markov LOB (Limit Order Book) features from intraday bar data (Tier 6.2).
- LOC: 312
- Classes: none
- Top-level functions: `_inter_bar_durations`, `_estimate_poisson_lambda`, `_signed_volume`, `compute_lob_features`, `compute_lob_features_batch`

### `features/macro.py`
- Intent: FRED macro indicator features for quant_engine.
- LOC: 245
- Classes:
  - `MacroFeatureProvider` (methods: `get_macro_features`)
- Top-level functions: `_cache_key`

### `features/options_factors.py`
- Intent: Option surface factor construction from OptionMetrics-enriched daily panels.
- LOC: 135
- Classes: none
- Top-level functions: `_pick_numeric`, `_rolling_percentile_rank`, `compute_option_surface_factors`, `compute_iv_shock_features`

### `features/pipeline.py`
- Intent: Feature Pipeline — computes model features from OHLCV data.
- LOC: 1542
- Classes:
  - `FeaturePipeline` (methods: `compute`, `compute_universe`)
- Top-level functions: `get_feature_type`, `_filter_causal_features`, `_build_indicator_set`, `_build_minimal_indicator_set`, `_get_indicators`, `compute_indicator_features`, `compute_raw_features`, `compute_har_volatility_features`, `compute_multiscale_features`, `compute_structural_features`, `compute_interaction_features`, `compute_targets`, `_winsorize_expanding`

### `features/research_factors.py`
- Intent: Research-derived factor construction for quant_engine.
- LOC: 986
- Classes:
  - `ResearchFactorConfig` (methods: none)
- Top-level functions: `_rolling_zscore`, `_safe_pct_change`, `_required_ohlcv`, `compute_order_flow_impact_factors`, `compute_markov_queue_features`, `compute_time_series_momentum_factors`, `compute_vol_scaled_momentum`, `_rolling_levy_area`, `compute_signature_path_features`, `compute_vol_surface_factors`, `compute_single_asset_research_factors`, `_standardize_block`, `_lagged_weight_matrix`, `compute_cross_asset_research_factors`, `_dtw_distance_numpy`, `_dtw_avg_lag_from_path`, `compute_dtw_lead_lag`, `_numpy_order2_signature`, `compute_path_signatures`

### `features/version.py`
- Intent: Feature versioning system.
- LOC: 169
- Classes:
  - `FeatureVersion` (methods: `n_features`, `compute_hash`, `to_dict`, `diff`, `is_compatible`)
  - `FeatureRegistry` (methods: `register`, `get_version`, `get_latest`, `list_versions`, `check_compatibility`)
- Top-level functions: none

### `features/wave_flow.py`
- Intent: Wave-Flow Decomposition for quant_engine.
- LOC: 145
- Classes: none
- Top-level functions: `compute_wave_flow_decomposition`

## `indicators`

| Module | Lines | Classes | Top-level Functions | Intent |
|---|---|---|---|---|
| `indicators/__init__.py` | 90 | 0 | 0 | Quant Engine Indicators — self-contained copy of the technical indicator library. |
| `indicators/eigenvalue.py` | 400 | 1 | 0 | Eigenvalue Spectrum Indicators — portfolio-level systemic risk analysis. |
| `indicators/indicators.py` | 2905 | 92 | 2 | Technical Indicator Library |
| `indicators/ot_divergence.py` | 263 | 1 | 2 | Optimal Transport Indicators — distribution drift detection. |
| `indicators/spectral.py` | 329 | 1 | 0 | Spectral Analysis Indicators — FFT-based frequency decomposition of price series. |
| `indicators/ssa.py` | 322 | 1 | 0 | Singular Spectrum Analysis (SSA) Indicators — non-stationary signal decomposition. |
| `indicators/tail_risk.py` | 241 | 1 | 0 | Tail Risk & Jump Detection Indicators — extreme event analysis. |

### `indicators/__init__.py`
- Intent: Quant Engine Indicators — self-contained copy of the technical indicator library.
- LOC: 90
- Classes: none
- Top-level functions: none

### `indicators/eigenvalue.py`
- Intent: Eigenvalue Spectrum Indicators — portfolio-level systemic risk analysis.
- LOC: 400
- Classes:
  - `EigenvalueAnalyzer` (methods: `compute_eigenvalue_concentration`, `compute_effective_rank`, `compute_avg_correlation_stress`, `compute_spectral_condition_number`, `compute_all`)
- Top-level functions: none

### `indicators/indicators.py`
- Intent: Technical Indicator Library
- LOC: 2905
- Classes:
  - `Indicator` (methods: `name`, `calculate`)
  - `ATR` (methods: `name`, `calculate`)
  - `NATR` (methods: `name`, `calculate`)
  - `BollingerBandWidth` (methods: `name`, `calculate`)
  - `HistoricalVolatility` (methods: `name`, `calculate`)
  - `RSI` (methods: `name`, `calculate`)
  - `MACD` (methods: `name`, `calculate`)
  - `MACDSignal` (methods: `name`, `calculate`)
  - `MACDHistogram` (methods: `name`, `calculate`)
  - `ROC` (methods: `name`, `calculate`)
  - `Stochastic` (methods: `name`, `calculate`)
  - `StochasticD` (methods: `name`, `calculate`)
  - `WilliamsR` (methods: `name`, `calculate`)
  - `CCI` (methods: `name`, `calculate`)
  - `SMA` (methods: `name`, `calculate`)
  - `EMA` (methods: `name`, `calculate`)
  - `PriceVsSMA` (methods: `name`, `calculate`)
  - `SMASlope` (methods: `name`, `calculate`)
  - `ADX` (methods: `name`, `calculate`)
  - `Aroon` (methods: `name`, `calculate`)
  - `VolumeRatio` (methods: `name`, `calculate`)
  - `OBV` (methods: `name`, `calculate`)
  - `OBVSlope` (methods: `name`, `calculate`)
  - `MFI` (methods: `name`, `calculate`)
  - `HigherHighs` (methods: `name`, `calculate`)
  - `LowerLows` (methods: `name`, `calculate`)
  - `CandleBody` (methods: `name`, `calculate`)
  - `CandleDirection` (methods: `name`, `calculate`)
  - `GapPercent` (methods: `name`, `calculate`)
  - `DistanceFromHigh` (methods: `name`, `calculate`)
  - `DistanceFromLow` (methods: `name`, `calculate`)
  - `PricePercentile` (methods: `name`, `calculate`)
  - `BBWidthPercentile` (methods: `name`, `calculate`)
  - `NATRPercentile` (methods: `name`, `calculate`)
  - `VolatilitySqueeze` (methods: `name`, `calculate`)
  - `RVOL` (methods: `name`, `calculate`)
  - `NetVolumeTrend` (methods: `name`, `calculate`)
  - `VolumeForce` (methods: `name`, `calculate`)
  - `AccumulationDistribution` (methods: `name`, `calculate`)
  - `EMAAlignment` (methods: `name`, `calculate`)
  - `TrendStrength` (methods: `name`, `calculate`)
  - `PriceVsEMAStack` (methods: `name`, `calculate`)
  - `PivotHigh` (methods: `name`, `calculate`)
  - `PivotLow` (methods: `name`, `calculate`)
  - `NBarHighBreak` (methods: `name`, `calculate`)
  - `NBarLowBreak` (methods: `name`, `calculate`)
  - `RangeBreakout` (methods: `name`, `calculate`)
  - `ATRTrailingStop` (methods: `name`, `calculate`)
  - `ATRChannel` (methods: `name`, `calculate`)
  - `RiskPerATR` (methods: `name`, `calculate`)
  - `MarketRegime` (methods: `name`, `calculate`)
  - `VolatilityRegime` (methods: `name`, `calculate`)
  - `VWAP` (methods: `name`, `calculate`)
  - `PriceVsVWAP` (methods: `name`, `calculate`)
  - `VWAPBands` (methods: `name`, `calculate`)
  - `AnchoredVWAP` (methods: `name`, `calculate`)
  - `PriceVsAnchoredVWAP` (methods: `name`, `calculate`)
  - `MultiVWAPPosition` (methods: `name`, `calculate`)
  - `ValueAreaHigh` (methods: `name`, `calculate`)
  - `ValueAreaLow` (methods: `name`, `calculate`)
  - `POC` (methods: `name`, `calculate`)
  - `PriceVsPOC` (methods: `name`, `calculate`)
  - `ValueAreaPosition` (methods: `name`, `calculate`)
  - `AboveValueArea` (methods: `name`, `calculate`)
  - `BelowValueArea` (methods: `name`, `calculate`)
  - `Beast666Proximity` (methods: `name`, `calculate`)
  - `Beast666Distance` (methods: `name`, `calculate`)
  - `ParkinsonVolatility` (methods: `name`, `calculate`)
  - `GarmanKlassVolatility` (methods: `name`, `calculate`)
  - `YangZhangVolatility` (methods: `name`, `calculate`)
  - `VolatilityCone` (methods: `name`, `calculate`)
  - `VolOfVol` (methods: `name`, `calculate`)
  - `GARCHVolatility` (methods: `name`, `calculate`)
  - `VolTermStructure` (methods: `name`, `calculate`)
  - `HurstExponent` (methods: `name`, `calculate`)
  - `MeanReversionHalfLife` (methods: `name`, `calculate`)
  - `ZScore` (methods: `name`, `calculate`)
  - `VarianceRatio` (methods: `name`, `calculate`)
  - `Autocorrelation` (methods: `name`, `calculate`)
  - `KalmanTrend` (methods: `name`, `calculate`)
  - `ShannonEntropy` (methods: `name`, `calculate`)
  - `ApproximateEntropy` (methods: `name`, `calculate`)
  - `AmihudIlliquidity` (methods: `name`, `calculate`)
  - `KyleLambda` (methods: `name`, `calculate`)
  - `RollSpread` (methods: `name`, `calculate`)
  - `FractalDimension` (methods: `name`, `calculate`)
  - `DFA` (methods: `name`, `calculate`)
  - `DominantCycle` (methods: `name`, `calculate`)
  - `ReturnSkewness` (methods: `name`, `calculate`)
  - `ReturnKurtosis` (methods: `name`, `calculate`)
  - `CUSUMDetector` (methods: `name`, `calculate`)
  - `RegimePersistence` (methods: `name`, `calculate`)
- Top-level functions: `get_all_indicators`, `create_indicator`

### `indicators/ot_divergence.py`
- Intent: Optimal Transport Indicators — distribution drift detection.
- LOC: 263
- Classes:
  - `OptimalTransportAnalyzer` (methods: `compute_wasserstein_distance`, `compute_sinkhorn_divergence`, `compute_all`)
- Top-level functions: `_logsumexp_rows`, `_logsumexp_cols`

### `indicators/spectral.py`
- Intent: Spectral Analysis Indicators — FFT-based frequency decomposition of price series.
- LOC: 329
- Classes:
  - `SpectralAnalyzer` (methods: `compute_hf_lf_energy`, `compute_spectral_entropy`, `compute_dominant_frequency`, `compute_spectral_bandwidth`, `compute_all`)
- Top-level functions: none

### `indicators/ssa.py`
- Intent: Singular Spectrum Analysis (SSA) Indicators — non-stationary signal decomposition.
- LOC: 322
- Classes:
  - `SSADecomposer` (methods: `compute_trend_strength`, `compute_singular_entropy`, `compute_noise_ratio`, `compute_oscillatory_strength`, `compute_all`)
- Top-level functions: none

### `indicators/tail_risk.py`
- Intent: Tail Risk & Jump Detection Indicators — extreme event analysis.
- LOC: 241
- Classes:
  - `TailRiskAnalyzer` (methods: `compute_jump_intensity`, `compute_expected_shortfall`, `compute_vol_of_vol`, `compute_semi_relative_modulus`, `compute_extreme_return_pct`, `compute_all`)
- Top-level functions: none

## `kalshi`

| Module | Lines | Classes | Top-level Functions | Intent |
|---|---|---|---|---|
| `kalshi/__init__.py` | 59 | 0 | 0 | Kalshi vertical for intraday event-market research. |
| `kalshi/client.py` | 656 | 5 | 1 | Kalshi API client with signed authentication, rate limiting, and endpoint routing. |
| `kalshi/disagreement.py` | 114 | 1 | 2 | Cross-market disagreement engine for Kalshi event features. |
| `kalshi/distribution.py` | 936 | 3 | 23 | Contract -> probability distribution builder for Kalshi markets. |
| `kalshi/events.py` | 518 | 2 | 11 | Event-time joins and as-of feature/label builders for Kalshi-driven research. |
| `kalshi/mapping_store.py` | 76 | 2 | 0 | Versioned event-to-market mapping persistence. |
| `kalshi/microstructure.py` | 128 | 1 | 2 | Market microstructure diagnostics for Kalshi event markets. |
| `kalshi/options.py` | 146 | 0 | 3 | OptionMetrics-style options reference features for Kalshi event disagreement. |
| `kalshi/pipeline.py` | 168 | 1 | 0 | Orchestration helpers for the Kalshi event-market vertical. |
| `kalshi/promotion.py` | 179 | 1 | 2 | Event-strategy promotion helpers for Kalshi walk-forward outputs. |
| `kalshi/provider.py` | 648 | 1 | 3 | Kalshi provider: ingestion + storage + feature-ready retrieval. |
| `kalshi/quality.py` | 207 | 2 | 5 | Quality scoring helpers for Kalshi event-distribution snapshots. |
| `kalshi/regimes.py` | 143 | 1 | 6 | Regime tagging for Kalshi event strategies. |
| `kalshi/router.py` | 103 | 2 | 0 | Routing helpers for live vs historical Kalshi endpoints. |
| `kalshi/storage.py` | 650 | 1 | 0 | Event-time storage layer for Kalshi + macro event research. |
| `kalshi/walkforward.py` | 493 | 3 | 8 | Walk-forward evaluation for event-centric Kalshi feature panels. |

### `kalshi/__init__.py`
- Intent: Kalshi vertical for intraday event-market research.
- LOC: 59
- Classes: none
- Top-level functions: none

### `kalshi/client.py`
- Intent: Kalshi API client with signed authentication, rate limiting, and endpoint routing.
- LOC: 656
- Classes:
  - `RetryPolicy` (methods: none)
  - `RateLimitPolicy` (methods: none)
  - `RequestLimiter` (methods: `acquire`, `update_rate`, `update_from_account_limits`)
  - `KalshiSigner` (methods: `available`, `sign`)
  - `KalshiClient` (methods: `available`, `get`, `paginate`, `get_account_limits`, `fetch_historical_cutoff`, `server_time_utc`, `clock_skew_seconds`, `list_markets`, `list_contracts`, `list_trades`, `list_quotes`)
- Top-level functions: `_normalize_env`

### `kalshi/disagreement.py`
- Intent: Cross-market disagreement engine for Kalshi event features.
- LOC: 114
- Classes:
  - `DisagreementSignals` (methods: none)
- Top-level functions: `compute_disagreement`, `disagreement_as_feature_dict`

### `kalshi/distribution.py`
- Intent: Contract -> probability distribution builder for Kalshi markets.
- LOC: 936
- Classes:
  - `DistributionConfig` (methods: none)
  - `DirectionResult` (methods: none)
  - `BinValidationResult` (methods: none)
- Top-level functions: `_is_tz_aware_datetime`, `_to_utc_timestamp`, `_prob_from_mid`, `_entropy`, `_isotonic_nonincreasing`, `_isotonic_nondecreasing`, `_resolve_threshold_direction`, `_resolve_threshold_direction_with_confidence`, `_validate_bins`, `_tail_thresholds`, `_latest_quotes_asof`, `_normalize_mass`, `_moments`, `_cdf_from_pmf`, `_pmf_on_grid`, `_distribution_distances`, `_tail_probs_from_mass`, `_tail_probs_from_threshold_curve`, `_estimate_liquidity_proxy`, `build_distribution_snapshot`, `_lag_slug`, `_add_distance_features`, `build_distribution_panel`

### `kalshi/events.py`
- Intent: Event-time joins and as-of feature/label builders for Kalshi-driven research.
- LOC: 518
- Classes:
  - `EventTimestampMeta` (methods: none)
  - `EventFeatureConfig` (methods: none)
- Top-level functions: `_to_utc_ts`, `_ensure_asof_before_release`, `asof_join`, `build_event_snapshot_grid`, `_merge_event_market_map`, `_add_revision_speed_features`, `add_reference_disagreement_features`, `build_event_feature_panel`, `build_asset_time_feature_panel`, `build_event_labels`, `build_asset_response_labels`

### `kalshi/mapping_store.py`
- Intent: Versioned event-to-market mapping persistence.
- LOC: 76
- Classes:
  - `EventMarketMappingRecord` (methods: none)
  - `EventMarketMappingStore` (methods: `upsert`, `asof`, `current_version`, `assert_consistent_mapping_version`)
- Top-level functions: none

### `kalshi/microstructure.py`
- Intent: Market microstructure diagnostics for Kalshi event markets.
- LOC: 128
- Classes:
  - `MicrostructureDiagnostics` (methods: none)
- Top-level functions: `compute_microstructure`, `microstructure_as_feature_dict`

### `kalshi/options.py`
- Intent: OptionMetrics-style options reference features for Kalshi event disagreement.
- LOC: 146
- Classes: none
- Top-level functions: `_to_utc_ts`, `build_options_reference_panel`, `add_options_disagreement_features`

### `kalshi/pipeline.py`
- Intent: Orchestration helpers for the Kalshi event-market vertical.
- LOC: 168
- Classes:
  - `KalshiPipeline` (methods: `from_store`, `sync_reference`, `sync_intraday_quotes`, `build_distributions`, `build_event_features`, `run_walkforward`, `evaluate_walkforward_contract`, `evaluate_event_promotion`)
- Top-level functions: none

### `kalshi/promotion.py`
- Intent: Event-strategy promotion helpers for Kalshi walk-forward outputs.
- LOC: 179
- Classes:
  - `EventPromotionConfig` (methods: none)
- Top-level functions: `_to_backtest_result`, `evaluate_event_promotion`

### `kalshi/provider.py`
- Intent: Kalshi provider: ingestion + storage + feature-ready retrieval.
- LOC: 648
- Classes:
  - `KalshiProvider` (methods: `available`, `sync_account_limits`, `refresh_historical_cutoff`, `sync_market_catalog`, `sync_contracts`, `sync_quotes`, `get_markets`, `get_contracts`, `get_quotes`, `get_event_market_map_asof`, `get_macro_events`, `get_event_outcomes` (+4 more))
- Top-level functions: `_to_iso_utc`, `_safe_hash_text`, `_asof_date`

### `kalshi/quality.py`
- Intent: Quality scoring helpers for Kalshi event-distribution snapshots.
- LOC: 207
- Classes:
  - `QualityDimensions` (methods: none)
  - `StalePolicy` (methods: none)
- Top-level functions: `_finite`, `dynamic_stale_cutoff_minutes`, `compute_quality_dimensions`, `passes_hard_gates`, `quality_as_feature_dict`

### `kalshi/regimes.py`
- Intent: Regime tagging for Kalshi event strategies.
- LOC: 143
- Classes:
  - `EventRegimeTag` (methods: none)
- Top-level functions: `classify_inflation_regime`, `classify_policy_regime`, `classify_vol_regime`, `tag_event_regime`, `evaluate_strategy_by_regime`, `regime_stability_score`

### `kalshi/router.py`
- Intent: Routing helpers for live vs historical Kalshi endpoints.
- LOC: 103
- Classes:
  - `RouteDecision` (methods: none)
  - `KalshiDataRouter` (methods: `update_cutoff`, `resolve`)
- Top-level functions: none

### `kalshi/storage.py`
- Intent: Event-time storage layer for Kalshi + macro event research.
- LOC: 650
- Classes:
  - `EventTimeStore` (methods: `init_schema`, `upsert_markets`, `upsert_contracts`, `append_quotes`, `upsert_macro_events`, `upsert_event_outcomes`, `upsert_event_outcomes_first_print`, `upsert_event_outcomes_revised`, `upsert_distributions`, `upsert_event_market_map_versions`, `append_market_specs`, `append_contract_specs` (+8 more))
- Top-level functions: none

### `kalshi/walkforward.py`
- Intent: Walk-forward evaluation for event-centric Kalshi feature panels.
- LOC: 493
- Classes:
  - `EventWalkForwardConfig` (methods: none)
  - `EventWalkForwardFold` (methods: none)
  - `EventWalkForwardResult` (methods: `wf_oos_corr`, `wf_positive_fold_fraction`, `wf_is_oos_gap`, `worst_event_loss`, `to_metrics`)
- Top-level functions: `_bootstrap_mean_ci`, `_event_regime_stability`, `evaluate_event_contract_metrics`, `_corr`, `_fit_ridge`, `_predict`, `_prepare_panel`, `run_event_walkforward`

## `models`

| Module | Lines | Classes | Top-level Functions | Intent |
|---|---|---|---|---|
| `models/__init__.py` | 25 | 0 | 0 | Models subpackage — training, prediction, versioning, and retraining triggers. |
| `models/calibration.py` | 328 | 2 | 2 | Confidence Calibration --- Platt scaling and isotonic regression. |
| `models/conformal.py` | 296 | 3 | 0 | Conformal Prediction — distribution-free prediction intervals. |
| `models/cross_sectional.py` | 137 | 0 | 1 | Cross-Sectional Ranking Model — rank stocks relative to peers at each date. |
| `models/feature_stability.py` | 314 | 2 | 0 | Feature Stability Monitoring — tracks feature importance rankings across |
| `models/governance.py` | 156 | 2 | 0 | Champion/challenger governance for model versions. |
| `models/iv/__init__.py` | 32 | 0 | 0 | Implied Volatility Surface Models — Heston, SVI, Black-Scholes, and IV Surface. |
| `models/iv/models.py` | 938 | 10 | 1 | Implied Volatility Surface Models. |
| `models/neural_net.py` | 199 | 1 | 0 | Tabular Neural Network — feedforward network for tabular financial data. |
| `models/online_learning.py` | 274 | 2 | 0 | Online learning module for incremental model updates between full retrains. |
| `models/predictor.py` | 485 | 1 | 1 | Model Predictor — loads trained ensemble and generates predictions. |
| `models/retrain_trigger.py` | 345 | 1 | 0 | ML Retraining Trigger Logic |
| `models/shift_detection.py` | 323 | 3 | 0 | Distribution Shift Detection — CUSUM and PSI methods. |
| `models/trainer.py` | 1675 | 5 | 0 | Model Trainer — trains regime-conditional gradient boosting ensemble. |
| `models/versioning.py` | 208 | 2 | 0 | Model Versioning — timestamped model directories with registry. |
| `models/walk_forward.py` | 236 | 0 | 4 | Walk-Forward Model Selection — expanding-window hyperparameter search |

### `models/__init__.py`
- Intent: Models subpackage — training, prediction, versioning, and retraining triggers.
- LOC: 25
- Classes: none
- Top-level functions: none

### `models/calibration.py`
- Intent: Confidence Calibration --- Platt scaling and isotonic regression.
- LOC: 328
- Classes:
  - `_LinearRescaler` (methods: `fit`, `transform`)
  - `ConfidenceCalibrator` (methods: `fit`, `transform`, `fit_transform`, `is_fitted`, `backend`)
- Top-level functions: `compute_ece`, `compute_reliability_curve`

### `models/conformal.py`
- Intent: Conformal Prediction — distribution-free prediction intervals.
- LOC: 296
- Classes:
  - `ConformalInterval` (methods: none)
  - `ConformalCalibrationResult` (methods: none)
  - `ConformalPredictor` (methods: `is_calibrated`, `calibrate`, `predict_interval`, `predict_intervals_batch`, `uncertainty_scalars`, `evaluate_coverage`, `to_dict`, `from_dict`)
- Top-level functions: none

### `models/cross_sectional.py`
- Intent: Cross-Sectional Ranking Model — rank stocks relative to peers at each date.
- LOC: 137
- Classes: none
- Top-level functions: `cross_sectional_rank`

### `models/feature_stability.py`
- Intent: Feature Stability Monitoring — tracks feature importance rankings across
- LOC: 314
- Classes:
  - `StabilityReport` (methods: `to_dict`)
  - `FeatureStabilityTracker` (methods: `record_importance`, `check_stability`)
- Top-level functions: none

### `models/governance.py`
- Intent: Champion/challenger governance for model versions.
- LOC: 156
- Classes:
  - `ChampionRecord` (methods: `to_dict`)
  - `ModelGovernance` (methods: `get_champion_version`, `evaluate_and_update`)
- Top-level functions: none

### `models/iv/__init__.py`
- Intent: Implied Volatility Surface Models — Heston, SVI, Black-Scholes, and IV Surface.
- LOC: 32
- Classes: none
- Top-level functions: none

### `models/iv/models.py`
- Intent: Implied Volatility Surface Models.
- LOC: 938
- Classes:
  - `OptionType` (methods: none)
  - `Greeks` (methods: none)
  - `HestonParams` (methods: `validate`)
  - `SVIParams` (methods: none)
  - `BlackScholes` (methods: `price`, `greeks`, `implied_vol`, `iv_surface`)
  - `HestonModel` (methods: `characteristic_function`, `price`, `implied_vol`, `iv_surface`, `calibrate`)
  - `SVIModel` (methods: `total_variance`, `implied_vol`, `iv_surface`, `smile`, `calibrate`, `check_no_butterfly_arbitrage`)
  - `ArbitrageFreeSVIBuilder` (methods: `fit_slice`, `enforce_calendar_monotonicity`, `interpolate_total_variance`, `build_surface`)
  - `IVPoint` (methods: none)
  - `IVSurface` (methods: `add_point`, `add_slice`, `add_surface`, `n_points`, `get_iv`, `get_smile`, `decompose`, `decompose_surface`)
- Top-level functions: `generate_synthetic_market_surface`

### `models/neural_net.py`
- Intent: Tabular Neural Network — feedforward network for tabular financial data.
- LOC: 199
- Classes:
  - `TabularNet` (methods: `fit`, `predict`, `feature_importances_`)
- Top-level functions: none

### `models/online_learning.py`
- Intent: Online learning module for incremental model updates between full retrains.
- LOC: 274
- Classes:
  - `OnlineUpdate` (methods: none)
  - `OnlineLearner` (methods: `add_sample`, `update`, `adjust_prediction`, `should_retrain`, `get_status`, `load_state`)
- Top-level functions: none

### `models/predictor.py`
- Intent: Model Predictor — loads trained ensemble and generates predictions.
- LOC: 485
- Classes:
  - `EnsemblePredictor` (methods: `predict`, `blend_multi_horizon`, `predict_single`)
- Top-level functions: `_prepare_features`

### `models/retrain_trigger.py`
- Intent: ML Retraining Trigger Logic
- LOC: 345
- Classes:
  - `RetrainTrigger` (methods: `add_trade_result`, `check_shift`, `check`, `record_retraining`, `status`)
- Top-level functions: none

### `models/shift_detection.py`
- Intent: Distribution Shift Detection — CUSUM and PSI methods.
- LOC: 323
- Classes:
  - `CUSUMResult` (methods: none)
  - `PSIResult` (methods: none)
  - `DistributionShiftDetector` (methods: `set_reference`, `check_cusum`, `check_psi`, `check_all`)
- Top-level functions: none

### `models/trainer.py`
- Intent: Model Trainer — trains regime-conditional gradient boosting ensemble.
- LOC: 1675
- Classes:
  - `IdentityScaler` (methods: `fit`, `transform`, `fit_transform`, `inverse_transform`)
  - `DiverseEnsemble` (methods: `predict`)
  - `TrainResult` (methods: none)
  - `EnsembleResult` (methods: none)
  - `ModelTrainer` (methods: `train_ensemble`, `compute_shared_features`)
- Top-level functions: none

### `models/versioning.py`
- Intent: Model Versioning — timestamped model directories with registry.
- LOC: 208
- Classes:
  - `ModelVersion` (methods: `to_dict`, `from_dict`)
  - `ModelRegistry` (methods: `latest_version_id`, `get_latest`, `get_version`, `get_version_dir`, `get_latest_dir`, `list_versions`, `create_version_dir`, `register_version`, `rollback`, `prune_old`, `has_versions`)
- Top-level functions: none

### `models/walk_forward.py`
- Intent: Walk-Forward Model Selection — expanding-window hyperparameter search
- LOC: 236
- Classes: none
- Top-level functions: `_spearmanr`, `_expanding_walk_forward_folds`, `_extract_dates`, `walk_forward_select`

## `regime`

| Module | Lines | Classes | Top-level Functions | Intent |
|---|---|---|---|---|
| `regime/__init__.py` | 41 | 0 | 0 | Regime modeling components. |
| `regime/bocpd.py` | 452 | 3 | 0 | Bayesian Online Change-Point Detection (BOCPD) with Gaussian likelihood. |
| `regime/confidence_calibrator.py` | 252 | 1 | 0 | Confidence Calibrator for regime ensemble voting (SPEC_10 T2). |
| `regime/consensus.py` | 274 | 1 | 0 | Cross-Sectional Regime Consensus (SPEC_10 T6). |
| `regime/correlation.py` | 214 | 1 | 0 | Correlation Regime Detection (NEW 11). |
| `regime/detector.py` | 941 | 2 | 2 | Regime detector with multiple engines and structural state layer. |
| `regime/hmm.py` | 662 | 2 | 4 | Gaussian HMM regime model with sticky transitions and duration smoothing. |
| `regime/jump_model.py` | 10 | 0 | 0 | Backward-compatible re-export of legacy Statistical Jump Model. |
| `regime/jump_model_legacy.py` | 243 | 2 | 0 | Statistical Jump Model for regime detection. |
| `regime/jump_model_pypi.py` | 421 | 1 | 0 | PyPI jumpmodels package wrapper for regime detection. |
| `regime/online_update.py` | 246 | 1 | 0 | Online Regime Updating via Forward Algorithm (SPEC_10 T5). |
| `regime/shock_vector.py` | 306 | 2 | 0 | Unified Shock/Structure Vector — version-locked market state representation. |
| `regime/uncertainty_gate.py` | 181 | 1 | 0 | Regime Uncertainty Gate — entropy-based position sizing modifier (SPEC_10 T3). |

### `regime/__init__.py`
- Intent: Regime modeling components.
- LOC: 41
- Classes: none
- Top-level functions: none

### `regime/bocpd.py`
- Intent: Bayesian Online Change-Point Detection (BOCPD) with Gaussian likelihood.
- LOC: 452
- Classes:
  - `BOCPDResult` (methods: none)
  - `BOCPDBatchResult` (methods: none)
  - `BOCPDDetector` (methods: `reset`, `update`, `batch_update`)
- Top-level functions: none

### `regime/confidence_calibrator.py`
- Intent: Confidence Calibrator for regime ensemble voting (SPEC_10 T2).
- LOC: 252
- Classes:
  - `ConfidenceCalibrator` (methods: `fitted`, `component_weights`, `fit`, `calibrate`, `get_component_weight`, `expected_calibration_error`)
- Top-level functions: none

### `regime/consensus.py`
- Intent: Cross-Sectional Regime Consensus (SPEC_10 T6).
- LOC: 274
- Classes:
  - `RegimeConsensus` (methods: `compute_consensus`, `detect_divergence`, `early_warning`, `compute_consensus_series`, `reset_history`)
- Top-level functions: none

### `regime/correlation.py`
- Intent: Correlation Regime Detection (NEW 11).
- LOC: 214
- Classes:
  - `CorrelationRegimeDetector` (methods: `compute_rolling_correlation`, `detect_correlation_spike`, `get_correlation_features`)
- Top-level functions: none

### `regime/detector.py`
- Intent: Regime detector with multiple engines and structural state layer.
- LOC: 941
- Classes:
  - `RegimeOutput` (methods: none)
  - `RegimeDetector` (methods: `detect`, `detect_ensemble`, `calibrate_confidence_weights`, `detect_with_confidence`, `detect_full`, `regime_features`, `get_regime_uncertainty`, `map_raw_states_to_regimes_stable`, `detect_with_shock_context`, `detect_batch_with_shock_context`)
- Top-level functions: `validate_hmm_observation_features`, `detect_regimes_batch`

### `regime/hmm.py`
- Intent: Gaussian HMM regime model with sticky transitions and duration smoothing.
- LOC: 662
- Classes:
  - `HMMFitResult` (methods: none)
  - `GaussianHMM` (methods: `viterbi`, `fit`, `predict_proba`)
- Top-level functions: `_logsumexp`, `select_hmm_states_bic`, `build_hmm_observation_matrix`, `map_raw_states_to_regimes`

### `regime/jump_model.py`
- Intent: Backward-compatible re-export of legacy Statistical Jump Model.
- LOC: 10
- Classes: none
- Top-level functions: none

### `regime/jump_model_legacy.py`
- Intent: Statistical Jump Model for regime detection.
- LOC: 243
- Classes:
  - `JumpModelResult` (methods: none)
  - `StatisticalJumpModel` (methods: `fit`, `compute_jump_penalty_from_data`, `predict`)
- Top-level functions: none

### `regime/jump_model_pypi.py`
- Intent: PyPI jumpmodels package wrapper for regime detection.
- LOC: 421
- Classes:
  - `PyPIJumpModel` (methods: `fit`, `predict_online`, `predict_proba_online`)
- Top-level functions: none

### `regime/online_update.py`
- Intent: Online Regime Updating via Forward Algorithm (SPEC_10 T5).
- LOC: 246
- Classes:
  - `OnlineRegimeUpdater` (methods: `forward_step`, `update_regime_for_security`, `update_batch`, `should_refit`, `reset_security_cache`, `cached_securities`, `get_state_probabilities`)
- Top-level functions: none

### `regime/shock_vector.py`
- Intent: Unified Shock/Structure Vector — version-locked market state representation.
- LOC: 306
- Classes:
  - `ShockVector` (methods: `to_dict`, `from_dict`, `is_shock_event`, `regime_name`)
  - `ShockVectorValidator` (methods: `validate`, `batch_validate`)
- Top-level functions: none

### `regime/uncertainty_gate.py`
- Intent: Regime Uncertainty Gate — entropy-based position sizing modifier (SPEC_10 T3).
- LOC: 181
- Classes:
  - `UncertaintyGate` (methods: `compute_size_multiplier`, `apply_uncertainty_gate`, `should_assume_stress`, `is_uncertain`, `gate_series`)
- Top-level functions: none

## `risk`

| Module | Lines | Classes | Top-level Functions | Intent |
|---|---|---|---|---|
| `risk/__init__.py` | 73 | 0 | 0 | Risk Management Module — Renaissance-grade portfolio risk controls. |
| `risk/attribution.py` | 369 | 0 | 5 | Performance Attribution --- decompose portfolio returns into market, factor, and alpha. |
| `risk/constraint_replay.py` | 198 | 0 | 2 | Constraint Tightening Replay — stress-test portfolios under regime-conditioned constraints. |
| `risk/cost_budget.py` | 222 | 1 | 2 | Transaction Cost Budget Optimization — minimize implementation cost for rebalances. |
| `risk/covariance.py` | 356 | 2 | 2 | Covariance estimation utilities for portfolio risk controls. |
| `risk/drawdown.py` | 261 | 3 | 0 | Drawdown Controller — circuit breakers and recovery protocols. |
| `risk/factor_exposures.py` | 269 | 1 | 0 | Factor Exposure Manager — compute and enforce regime-conditioned factor bounds. |
| `risk/factor_monitor.py` | 301 | 3 | 0 | Factor Exposure Monitoring — track portfolio factor tilts and alert on violations. |
| `risk/factor_portfolio.py` | 221 | 0 | 2 | Factor-Based Portfolio Construction — factor decomposition and exposure analysis. |
| `risk/metrics.py` | 321 | 2 | 0 | Risk Metrics — VaR, CVaR, tail risk, MAE/MFE, and advanced risk analytics. |
| `risk/portfolio_optimizer.py` | 277 | 0 | 1 | Mean-Variance Portfolio Optimization — turnover-penalised portfolio construction. |
| `risk/portfolio_risk.py` | 724 | 3 | 0 | Portfolio Risk Manager — enforces sector, correlation, and exposure limits. |
| `risk/position_sizer.py` | 1115 | 2 | 0 | Position Sizing — Kelly criterion, volatility-scaled, and ATR-based methods. |
| `risk/stop_loss.py` | 294 | 3 | 0 | Stop Loss Manager — regime-aware ATR stops, trailing, time, and regime-change stops. |
| `risk/stress_test.py` | 548 | 0 | 7 | Stress Testing Module --- scenario analysis, correlation stress, and historical drawdown replay. |
| `risk/universe_config.py` | 296 | 2 | 0 | Universe Configuration — centralized sector, liquidity, and borrowability metadata. |

### `risk/__init__.py`
- Intent: Risk Management Module — Renaissance-grade portfolio risk controls.
- LOC: 73
- Classes: none
- Top-level functions: none

### `risk/attribution.py`
- Intent: Performance Attribution --- decompose portfolio returns into market, factor, and alpha.
- LOC: 369
- Classes: none
- Top-level functions: `_estimate_beta`, `_estimate_factor_loadings`, `decompose_returns`, `compute_rolling_attribution`, `compute_attribution_report`

### `risk/constraint_replay.py`
- Intent: Constraint Tightening Replay — stress-test portfolios under regime-conditioned constraints.
- LOC: 198
- Classes: none
- Top-level functions: `replay_with_stress_constraints`, `compute_robustness_score`

### `risk/cost_budget.py`
- Intent: Transaction Cost Budget Optimization — minimize implementation cost for rebalances.
- LOC: 222
- Classes:
  - `RebalanceResult` (methods: none)
- Top-level functions: `estimate_trade_cost_bps`, `optimize_rebalance_cost`

### `risk/covariance.py`
- Intent: Covariance estimation utilities for portfolio risk controls.
- LOC: 356
- Classes:
  - `CovarianceEstimate` (methods: none)
  - `CovarianceEstimator` (methods: `estimate`, `portfolio_volatility`)
- Top-level functions: `compute_regime_covariance`, `get_regime_covariance`

### `risk/drawdown.py`
- Intent: Drawdown Controller — circuit breakers and recovery protocols.
- LOC: 261
- Classes:
  - `DrawdownState` (methods: none)
  - `DrawdownStatus` (methods: none)
  - `DrawdownController` (methods: `update`, `reset`, `get_summary`)
- Top-level functions: none

### `risk/factor_exposures.py`
- Intent: Factor Exposure Manager — compute and enforce regime-conditioned factor bounds.
- LOC: 269
- Classes:
  - `FactorExposureManager` (methods: `compute_exposures`, `is_stress_regime`, `check_factor_bounds`)
- Top-level functions: none

### `risk/factor_monitor.py`
- Intent: Factor Exposure Monitoring — track portfolio factor tilts and alert on violations.
- LOC: 301
- Classes:
  - `FactorExposure` (methods: none)
  - `FactorExposureReport` (methods: none)
  - `FactorExposureMonitor` (methods: `compute_exposures`, `check_limits`, `compute_report`)
- Top-level functions: none

### `risk/factor_portfolio.py`
- Intent: Factor-Based Portfolio Construction — factor decomposition and exposure analysis.
- LOC: 221
- Classes: none
- Top-level functions: `compute_factor_exposures`, `compute_residual_returns`

### `risk/metrics.py`
- Intent: Risk Metrics — VaR, CVaR, tail risk, MAE/MFE, and advanced risk analytics.
- LOC: 321
- Classes:
  - `RiskReport` (methods: none)
  - `RiskMetrics` (methods: `compute_full_report`, `print_report`)
- Top-level functions: none

### `risk/portfolio_optimizer.py`
- Intent: Mean-Variance Portfolio Optimization — turnover-penalised portfolio construction.
- LOC: 277
- Classes: none
- Top-level functions: `optimize_portfolio`

### `risk/portfolio_risk.py`
- Intent: Portfolio Risk Manager — enforces sector, correlation, and exposure limits.
- LOC: 724
- Classes:
  - `RiskCheck` (methods: none)
  - `ConstraintMultiplier` (methods: `is_stress_regime`, `get_multipliers`, `get_multipliers_smoothed`, `reset`)
  - `PortfolioRiskManager` (methods: `check_new_position`, `compute_constraint_utilization`, `invalidate_regime_cov_cache`, `portfolio_summary`)
- Top-level functions: none

### `risk/position_sizer.py`
- Intent: Position Sizing — Kelly criterion, volatility-scaled, and ATR-based methods.
- LOC: 1115
- Classes:
  - `PositionSize` (methods: none)
  - `PositionSizer` (methods: `size_position`, `size_position_paper_trader`, `record_turnover`, `reset_turnover_tracking`, `update_regime_stats`, `update_kelly_bayesian`, `get_bayesian_kelly`, `size_portfolio_aware`, `size_portfolio`, `size_with_backoff`)
- Top-level functions: none

### `risk/stop_loss.py`
- Intent: Stop Loss Manager — regime-aware ATR stops, trailing, time, and regime-change stops.
- LOC: 294
- Classes:
  - `StopReason` (methods: none)
  - `StopResult` (methods: none)
  - `StopLossManager` (methods: `evaluate`, `compute_initial_stop`, `compute_risk_per_share`)
- Top-level functions: none

### `risk/stress_test.py`
- Intent: Stress Testing Module --- scenario analysis, correlation stress, and historical drawdown replay.
- LOC: 548
- Classes: none
- Top-level functions: `_estimate_portfolio_beta`, `_compute_portfolio_vol`, `run_stress_scenarios`, `run_historical_drawdown_test`, `_find_drawdown_episodes`, `correlation_stress_test`, `factor_stress_test`

### `risk/universe_config.py`
- Intent: Universe Configuration — centralized sector, liquidity, and borrowability metadata.
- LOC: 296
- Classes:
  - `ConfigError` (methods: none)
  - `UniverseConfig` (methods: `get_sector`, `get_sector_constituents`, `get_all_sectors`, `get_liquidity_tier`, `is_hard_to_borrow`, `is_restricted`, `constraint_base`, `stress_multipliers`, `factor_limits`, `backoff_policy`, `get_stress_multiplier_set`, `get_factor_bounds` (+1 more))
- Top-level functions: none

## `scripts`

| Module | Lines | Classes | Top-level Functions | Intent |
|---|---|---|---|---|
| `scripts/alpaca_intraday_download.py` | 799 | 0 | 7 | Hybrid Intraday Data Downloader: Alpaca (primary) + IBKR (validation/gap-fill). |
| `scripts/compare_regime_models.py` | 324 | 0 | 3 | A/B comparison: Jump Model (PyPI) vs HMM baseline. |
| `scripts/generate_types.py` | 147 | 0 | 3 | Generate TypeScript interfaces from Pydantic schemas. |
| `scripts/ibkr_daily_gapfill.py` | 418 | 0 | 4 | IBKR Daily Gap-Fill Downloader for quant_engine cache. |
| `scripts/ibkr_intraday_download.py` | 509 | 0 | 4 | IBKR Intraday Data Downloader for quant_engine cache. |

### `scripts/alpaca_intraday_download.py`
- Intent: Hybrid Intraday Data Downloader: Alpaca (primary) + IBKR (validation/gap-fill).
- LOC: 799
- Classes: none
- Top-level functions: `survey_intraday`, `_build_alpaca_client`, `download_alpaca_chunked`, `validate_with_ibkr`, `save_intraday`, `quality_check`, `main`

### `scripts/compare_regime_models.py`
- Intent: A/B comparison: Jump Model (PyPI) vs HMM baseline.
- LOC: 324
- Classes: none
- Top-level functions: `load_representative_features`, `compute_regime_metrics`, `main`

### `scripts/generate_types.py`
- Intent: Generate TypeScript interfaces from Pydantic schemas.
- LOC: 147
- Classes: none
- Top-level functions: `python_type_to_ts`, `model_to_ts`, `main`

### `scripts/ibkr_daily_gapfill.py`
- Intent: IBKR Daily Gap-Fill Downloader for quant_engine cache.
- LOC: 418
- Classes: none
- Top-level functions: `survey_cache`, `download_daily_ibkr`, `merge_and_save`, `main`

### `scripts/ibkr_intraday_download.py`
- Intent: IBKR Intraday Data Downloader for quant_engine cache.
- LOC: 509
- Classes: none
- Top-level functions: `survey_intraday`, `download_intraday_chunked`, `save_intraday`, `main`

## `utils`

| Module | Lines | Classes | Top-level Functions | Intent |
|---|---|---|---|---|
| `utils/__init__.py` | 6 | 0 | 0 | Utility helpers package namespace. |
| `utils/logging.py` | 440 | 3 | 1 | Structured logging for the quant engine. |

### `utils/__init__.py`
- Intent: Utility helpers package namespace.
- LOC: 6
- Classes: none
- Top-level functions: none

### `utils/logging.py`
- Intent: Structured logging for the quant engine.
- LOC: 440
- Classes:
  - `StructuredFormatter` (methods: `format`)
  - `AlertHistory` (methods: `record`, `record_batch`, `query`)
  - `MetricsEmitter` (methods: `emit_cycle_metrics`, `check_alerts`)
- Top-level functions: `get_logger`

## `validation`

| Module | Lines | Classes | Top-level Functions | Intent |
|---|---|---|---|---|
| `validation/__init__.py` | 9 | 0 | 0 | Truth Layer validation — preflight checks for the quant engine. |
| `validation/data_integrity.py` | 115 | 2 | 0 | Data integrity preflight — Truth Layer T2. |
| `validation/feature_redundancy.py` | 115 | 1 | 1 | Feature Redundancy Detection — identifies highly correlated features. |
| `validation/leakage_detection.py` | 194 | 2 | 1 | Leakage detection — Truth Layer T3. |
| `validation/preconditions.py` | 72 | 0 | 2 | Execution contract validation — Truth Layer T1. |

### `validation/__init__.py`
- Intent: Truth Layer validation — preflight checks for the quant engine.
- LOC: 9
- Classes: none
- Top-level functions: none

### `validation/data_integrity.py`
- Intent: Data integrity preflight — Truth Layer T2.
- LOC: 115
- Classes:
  - `DataIntegrityCheckResult` (methods: none)
  - `DataIntegrityValidator` (methods: `validate_universe`)
- Top-level functions: none

### `validation/feature_redundancy.py`
- Intent: Feature Redundancy Detection — identifies highly correlated features.
- LOC: 115
- Classes:
  - `FeatureRedundancyDetector` (methods: `detect_redundant_pairs`, `report`)
- Top-level functions: `validate_structural_feature_composition`

### `validation/leakage_detection.py`
- Intent: Leakage detection — Truth Layer T3.
- LOC: 194
- Classes:
  - `LeakageTestResult` (methods: none)
  - `LeakageDetector` (methods: `test_time_shift_leakage`)
- Top-level functions: `run_leakage_checks`

### `validation/preconditions.py`
- Intent: Execution contract validation — Truth Layer T1.
- LOC: 72
- Classes: none
- Top-level functions: `validate_execution_contract`, `enforce_preconditions`
