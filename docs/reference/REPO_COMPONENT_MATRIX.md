# Repository Component Matrix

Source-derived inventory of active code components in the current repository state (including current working-tree changes).

## Snapshot

- Python modules inventoried in this matrix: 262
- FastAPI router modules mounted: 15
- FastAPI endpoints mounted: 48
- Frontend source files (`frontend/src`): 135

## Package Summary (Python)

| Package | Modules | Classes | Top-level Functions | LOC |
|---|---|---|---|---|
| (root) | 13 | 18 | 23 | 3,352 |
| api | 59 | 65 | 125 | 9,267 |
| autopilot | 8 | 12 | 0 | 3,685 |
| backtest | 11 | 29 | 26 | 5,652 |
| data | 12 | 17 | 80 | 7,099 |
| features | 10 | 5 | 48 | 4,026 |
| indicators | 7 | 97 | 4 | 4,550 |
| kalshi | 16 | 26 | 66 | 5,224 |
| models | 16 | 34 | 9 | 5,971 |
| regime | 13 | 17 | 6 | 4,243 |
| risk | 16 | 22 | 21 | 5,845 |
| scripts | 5 | 0 | 21 | 2,197 |
| tests | 69 | 286 | 164 | 20,539 |
| utils | 2 | 3 | 1 | 446 |
| validation | 5 | 5 | 4 | 505 |

## Root Entrypoints and Core Files

| File | Lines | Classes | Functions | Intent |
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

## FastAPI Router Matrix

| Router Module | Prefix | Endpoints | Handlers |
|---|---|---|---|
| `api/routers/jobs.py` | `/api/jobs` | 4 | GET / -> list_jobs, GET /{job_id} -> get_job, GET /{job_id}/events -> job_events, POST /{job_id}/cancel -> cancel_job |
| `api/routers/system_health.py` | `(mixed absolute paths)` | 5 | GET /api/health -> quick_health, GET /api/health/detailed -> detailed_health, GET /api/health/history -> health_history, GET /api/v1/system/model-age -> model_age, GET /api/v1/system/data-mode -> data_mode |
| `api/routers/dashboard.py` | `/api/dashboard` | 6 | GET /summary -> dashboard_summary, GET /regime -> dashboard_regime, GET /returns-distribution -> returns_distribution, GET /rolling-risk -> rolling_risk, GET /equity -> equity_with_benchmark, GET /attribution -> attribution |
| `api/routers/data_explorer.py` | `/api/data` | 6 | GET /universe -> get_universe, GET /status -> get_data_status, GET /ticker/{ticker} -> get_ticker, GET /ticker/{ticker}/bars -> get_ticker_bars, GET /ticker/{ticker}/indicators -> get_ticker_indicators, POST /ticker/{ticker}/indicators/batch -> batch_indicators |
| `api/routers/model_lab.py` | `/api/models` | 6 | GET /versions -> list_versions, GET /health -> model_health, GET /features/importance -> feature_importance, GET /features/correlations -> feature_correlations, POST /train -> train_model, POST /predict -> predict_model |
| `api/routers/signals.py` | `/api/signals` | 1 | GET /latest -> latest_signals |
| `api/routers/backtests.py` | `/api/backtests` | 4 | GET /latest -> latest_backtest, GET /latest/trades -> latest_trades, GET /latest/equity-curve -> equity_curve, POST /run -> run_backtest |
| `api/routers/benchmark.py` | `/api/benchmark` | 3 | GET /comparison -> benchmark_comparison, GET /equity-curves -> benchmark_equity_curves, GET /rolling-metrics -> benchmark_rolling_metrics |
| `api/routers/logs.py` | `/api/logs` | 1 | GET / -> get_logs |
| `api/routers/autopilot.py` | `/api/autopilot` | 4 | GET /latest-cycle -> latest_cycle, GET /strategies -> strategies, GET /paper-state -> paper_state, POST /run-cycle -> run_cycle |
| `api/routers/config_mgmt.py` | `/api/config` | 4 | GET / -> get_config, GET /validate -> validate_config_endpoint, GET /status -> get_config_status, PATCH / -> patch_config |
| `api/routers/iv_surface.py` | `/api/iv-surface` | 1 | GET /arb-free-svi -> arb_free_svi_surface |
| `api/routers/regime.py` | `/api/regime` | 1 | GET /metadata -> regime_metadata |
| `api/routers/risk.py` | `/api/risk` | 1 | GET /factor-exposures -> get_factor_exposures |
| `api/routers/diagnostics.py` | `/api/diagnostics` | 1 | GET / -> get_diagnostics |

## Frontend Source Matrix (`frontend/src`)

- Files: 135
- Pages: 51 files (4,597 LOC)
- Components: 39 files (2,834 LOC)
- Hooks: 7 files (256 LOC)
- API query hooks: 12 files (494 LOC)
- API mutation hooks: 6 files (95 LOC)
- Stores: 3 files (93 LOC)
- Types: 11 files (493 LOC)
