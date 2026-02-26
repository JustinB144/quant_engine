# CLI and Workflow Runbook

Operator runbook for root `run_*.py` entrypoints plus web app startup (FastAPI + React/Vite). This document is regenerated from current source and should match the CLI parser definitions in each script.

## Web App Startup (Current Stack)

### Development (recommended)

1. Backend API: `python run_server.py --reload` (or `python run_server.py`)
2. Frontend Vite dev server: `cd frontend && npm run dev`
3. Open the Vite URL (default `http://localhost:5173`) — `/api` calls proxy to the backend on port `8000`.

### Production-like local serving (single process)

1. Build frontend: `cd frontend && npm run build`
2. Serve API + static SPA: `python run_server.py --static --host 0.0.0.0 --port 8000`

### Python package script entrypoints (`pyproject.toml`) 

- `qe-server` -> `api.main:run_server`

## Root CLI Entrypoint Summary

| Script | Module Docstring | `add_argument()` Options |
|---|---|---:|
| `run_autopilot.py` | Run one full autopilot cycle: | 11 |
| `run_backtest.py` | Backtest the trained model on historical data. | 16 |
| `run_kalshi_event_pipeline.py` | Run the integrated Kalshi event-time pipeline inside quant_engine. | 21 |
| `run_predict.py` | Generate predictions using trained ensemble model. | 8 |
| `run_rehydrate_cache_metadata.py` | Backfill cache metadata sidecars for existing OHLCV cache files. | 6 |
| `run_retrain.py` | Retrain the quant engine model — checks triggers and retrains if needed. | 12 |
| `run_server.py` | Combined API + frontend static serving entry point. | 5 |
| `run_train.py` | Train the regime-conditional ensemble model. | 10 |
| `run_wrds_daily_refresh.py` | Re-download all daily OHLCV data from WRDS CRSP to replace old cache files | 6 |

## Supplemental Data Maintenance Scripts (`scripts/*.py`)

These are not root `run_*.py` entrypoints but are operationally important for cache quality and intraday refresh workflows:

- `scripts/ibkr_intraday_download.py`: IBKR intraday downloader (now supports per-timeframe default lookbacks, `UNIVERSE_INTRADAY` fallback, and configurable `--pace` pacing control).
- `scripts/alpaca_intraday_download.py`: Alpaca-first intraday downloader with optional IBKR cross-validation / gap-fill path.
- `scripts/ibkr_daily_gapfill.py`: Daily OHLCV gap-fill for stale/missing names in cache.
- `scripts/compare_regime_models.py`: Diagnostic comparison utility for regime model variants.
- `scripts/generate_types.py`: Generate frontend TypeScript interfaces from backend Pydantic schemas.

## Standard Workflow Sequences

### A. Web Monitoring / Inspection Session

1. Start backend + frontend (dev) or `run_server.py --static` (prod-like).
2. Check `/system-health` and `/dashboard` first for broad system state.
3. Use `/data-explorer` for cache freshness/coverage and per-ticker OHLCV quality.
4. Use `/model-lab`, `/signal-desk`, `/backtest-risk`, `/autopilot` for workflow-specific investigation.

### B. Model Development / Evaluation Session

1. `run_train.py` (train versioned models).
2. `run_predict.py` (write prediction artifacts).
3. `run_backtest.py` (write backtest summary/trades, optionally validation/advanced validation).
4. Inspect `results/` and the frontend dashboard/signal/backtest pages via API-backed views.

### C. Autopilot Cycle Session

1. `run_autopilot.py` (or `/api/autopilot/run-cycle` via UI).
2. Review `results/autopilot/latest_cycle.json` and registry/paper state files.
3. Inspect `/autopilot` page tabs for candidates, paper state, P&L, and lifecycle view.

### D. Kalshi Event Research Session

1. `run_kalshi_event_pipeline.py` with the desired sync/build flags.
2. Validate DuckDB/sqlite storage state and distribution/event feature outputs.
3. Review walk-forward and promotion outputs in generated artifacts; consult `docs/reference/KALSHI_STORAGE_SCHEMA_REFERENCE.md` for schema details.

### E. Intraday Data Integrity / Refresh Session

1. Run `scripts/alpaca_intraday_download.py` (primary) or `scripts/ibkr_intraday_download.py` (IBKR-only) for target tickers/timeframes.
2. If using Alpaca validation, review cross-source mismatch/quarantine outputs driven by `data/cross_source_validator.py` and `data/intraday_quality.py`.
3. Confirm quarantined files under `data/cache/quarantine/` and re-run with adjusted pace/source if needed.
4. Verify new bars through `/api/data/ticker/{ticker}/bars` and `/api/data/ticker/{ticker}/indicators`.

## Pre-Run Checklist

- Confirm `config.py` values (or runtime-patched API config values) match the intended experiment/operation.
- Confirm data cache freshness (`/api/data/status` or `Data Explorer`).
- Confirm model artifact availability/version selection for predict/backtest/autopilot workflows.
- Confirm WRDS/Kalshi credentials and feature flags if using live providers.
- For intraday downloads, confirm pacing settings and intraday integrity thresholds in `config.py` (`INTRADAY_*` constants).

## Post-Run Checklist

- Verify generated files in `results/` / `trained_models/` and timestamps.
- For intraday data runs, inspect `data/cache/` plus `data/cache/quarantine/` and associated `.meta.json` sidecars.
- Review drawdown, robustness, and quality metrics (not only total return or headline score).
- For API/UI-triggered jobs, inspect `/api/jobs`, SSE progress, and logs (`/api/logs`) for hidden failures.

## Script Details (Source-Derived Argparse Parse)

### `run_autopilot.py`

- Module doc: Run one full autopilot cycle:
- `argparse` description: `'Run quant_engine autopilot cycle'`
- Detected options: 11

| Options | Default | Action/Type | Help |
|---|---|---|---|
| `--full` | `` | `store_true` | Use full configured universe |
| `--tickers` | `` | `` | Explicit ticker list |
| `--horizon` | `10` | `int` | Holding/prediction horizon |
| `--years` | `15` | `int` | History window for evaluation |
| `--version` | `champion` | `str` | Model version (champion, latest, or explicit version id) |
| `--max-candidates` | `` | `int` | Limit candidate count |
| `--feature-mode` | `AUTOPILOT_FEATURE_MODE` | `` | Feature profile for autopilot evaluation |
| `--no-survivorship` | `` | `store_true` | Opt out of survivorship-bias-free point-in-time universe |
| `--no-walkforward` | `` | `store_true` | Use single-split training instead of rolling walk-forward |
| `--allow-in-sample` | `` | `store_true` | Disable strict out-of-sample filtering |
| `--quiet` | `` | `store_true` |  |

### `run_backtest.py`

- Module doc: Backtest the trained model on historical data.
- `argparse` description: `'Backtest the trained ensemble model'`
- Detected options: 16

| Options | Default | Action/Type | Help |
|---|---|---|---|
| `--full` | `` | `store_true` | Use full universe |
| `--tickers` | `` | `` | Specific tickers |
| `--horizon` | `10` | `int` | Prediction/holding horizon (days) |
| `--no-validate` | `` | `store_true` | Skip walk-forward validation |
| `--years` | `15` | `int` | Years of data for backtest |
| `--feature-mode` | `FEATURE_MODE_DEFAULT` | `` | Feature profile: minimal (~20 indicators), core (reduced complexity), or full |
| `--risk` | `` | `store_true` | Enable risk management (dynamic sizing, stops, drawdown controls) |
| `--advanced` | `` | `store_true` | Run advanced validation (Deflated Sharpe, Monte Carlo, PBO, capacity) |
| `--n-trials` | `1` | `int` | Number of strategy variants tested (for Deflated Sharpe) |
| `--version` | `latest` | `str` | Model version to test (default: latest) |
| `--no-survivorship` | `` | `store_true` | Opt out of survivorship-bias-free universe (use static universe instead) |
| `--allow-in-sample` | `` | `store_true` | Allow scoring dates that overlap model training history |
| `--min-confidence` | `` | `float` | Minimum model confidence to enter (default: config CONFIDENCE_THRESHOLD) |
| `--min-predicted` | `` | `float` | Minimum predicted return to enter (default: config ENTRY_THRESHOLD) |
| `--output` | `` | `str` | Save trade log to CSV |
| `--quiet` | `` | `store_true` |  |

### `run_kalshi_event_pipeline.py`

- Module doc: Run the integrated Kalshi event-time pipeline inside quant_engine.
- `argparse` description: `'Kalshi event-time pipeline'`
- Detected options: 21

| Options | Default | Action/Type | Help |
|---|---|---|---|
| `--db-path` | `str(KALSHI_DB_PATH)` | `str` |  |
| `--backend` | `duckdb` | `` |  |
| `--start-ts` | `` | `str` |  |
| `--end-ts` | `` | `str` |  |
| `--sync-reference` | `` | `store_true` |  |
| `--sync-quotes` | `` | `store_true` |  |
| `--build-distributions` | `` | `store_true` |  |
| `--build-event-features` | `` | `store_true` |  |
| `--event-map` | `` | `str` | CSV/Parquet with event_id/event_type + market_id |
| `--options-reference` | `` | `str` | CSV/Parquet options panel for disagreement features |
| `--options-ts-col` | `ts` | `str` | Timestamp column in options reference panel |
| `--output` | `str(RESULTS_DIR / 'kalshi_event_features.parquet')` | `str` |  |
| `--labels-first-print` | `` | `str` |  |
| `--labels-revised` | `` | `str` |  |
| `--run-walkforward` | `` | `store_true` |  |
| `--disable-promotion-gate` | `` | `store_true` |  |
| `--strategy-id` | `kalshi_event_default` | `str` |  |
| `--build-health-report` | `` | `store_true` | Materialize daily health report aggregates |
| `--print-health-report` | `` | `store_true` | Print latest daily health report table |
| `--health-report-output` | `` | `str` | CSV/Parquet output path for health report |
| `--quiet` | `` | `store_true` |  |

### `run_predict.py`

- Module doc: Generate predictions using trained ensemble model.
- `argparse` description: `'Generate predictions from trained ensemble'`
- Detected options: 8

| Options | Default | Action/Type | Help |
|---|---|---|---|
| `--full` | `` | `store_true` | Use full universe |
| `--tickers` | `` | `` | Specific symbols (resolved to PERMNO) |
| `--horizon` | `10` | `int` | Prediction horizon (days) |
| `--version` | `latest` | `str` | Model version to use |
| `--feature-mode` | `FEATURE_MODE_DEFAULT` | `` | Feature profile: core (reduced complexity) or full |
| `--output` | `` | `str` | Save predictions to CSV |
| `--top` | `20` | `int` | Show top N signals |
| `--quiet` | `` | `store_true` |  |

### `run_rehydrate_cache_metadata.py`

- Module doc: Backfill cache metadata sidecars for existing OHLCV cache files.
- `argparse` description: `'Backfill quant_engine cache metadata sidecars'`
- Detected options: 6

| Options | Default | Action/Type | Help |
|---|---|---|---|
| `--roots` | `` | `` | Optional cache roots to scan. Defaults to standard quant_engine roots. |
| `--root-source` | `[]` | `append` | Root source override in the form '<path>=<source>' (can repeat). |
| `--default-source` | `unknown` | `` | Default source label for unmapped roots. |
| `--force` | `` | `store_true` | Rewrite metadata even when sidecar already exists. |
| `--overwrite-source` | `` | `store_true` | When rewriting metadata, replace existing source labels. |
| `--dry-run` | `` | `store_true` | Scan and report only; do not write metadata. |

### `run_retrain.py`

- Module doc: Retrain the quant engine model — checks triggers and retrains if needed.
- `argparse` description: `'Retrain the quant engine model'`
- Detected options: 12

| Options | Default | Action/Type | Help |
|---|---|---|---|
| `--force` | `` | `store_true` | Force retrain regardless of triggers |
| `--status` | `` | `store_true` | Show retrain status and exit |
| `--versions` | `` | `store_true` | List all model versions |
| `--rollback` | `` | `str` | Rollback to a specific version ID |
| `--survivorship` | `` | `store_true` | Use WRDS survivorship-bias-free universe |
| `--full` | `` | `store_true` | Use full universe |
| `--tickers` | `` | `` | Specific tickers |
| `--horizon` | `[10]` | `int` | Prediction horizons |
| `--years` | `15` | `int` | Years of data |
| `--feature-mode` | `FEATURE_MODE_DEFAULT` | `` | Feature profile: core (reduced complexity) or full |
| `--recency` | `` | `store_true` | Apply exponential recency weighting |
| `--quiet` | `` | `store_true` |  |

### `run_server.py`

- Module doc: Combined API + frontend static serving entry point.
- `argparse` description: `'Quant Engine API Server'`
- Detected options: 5

| Options | Default | Action/Type | Help |
|---|---|---|---|
| `--host` | `0.0.0.0` | `` | Bind address (default: 0.0.0.0) |
| `--port` | `8000` | `int` | Port (default: 8000) |
| `--static` | `` | `store_true` | Serve frontend/dist as static files |
| `--reload` | `` | `store_true` | Enable auto-reload for development |
| `--log-level` | `info` | `` |  |

### `run_train.py`

- Module doc: Train the regime-conditional ensemble model.
- `argparse` description: `'Train regime-conditional gradient boosting ensemble'`
- Detected options: 10

| Options | Default | Action/Type | Help |
|---|---|---|---|
| `--full` | `` | `store_true` | Use full universe |
| `--tickers` | `` | `` | Specific tickers to train on |
| `--horizon` | `[10]` | `int` | Forward return horizons (days) |
| `--years` | `LOOKBACK_YEARS` | `int` | f'Years of historical data (default: {LOOKBACK_YEARS})' |
| `--no-interactions` | `` | `store_true` | Skip interaction features |
| `--feature-mode` | `FEATURE_MODE_DEFAULT` | `` | Feature profile: minimal (~20 indicators), core (reduced complexity), or full |
| `--survivorship` | `` | `store_true` | Use WRDS survivorship-bias-free universe |
| `--recency` | `` | `store_true` | Apply exponential recency weighting |
| `--no-version` | `` | `store_true` | Skip model versioning (save flat) |
| `--quiet` | `` | `store_true` | Minimal output |

### `run_wrds_daily_refresh.py`

- Module doc: Re-download all daily OHLCV data from WRDS CRSP to replace old cache files
- `argparse` description: `'Re-download daily OHLCV from WRDS CRSP to fix O=H=L=C cache files'`
- Detected options: 6

| Options | Default | Action/Type | Help |
|---|---|---|---|
| `--dry-run` | `` | `store_true` | Preview what would be downloaded without downloading |
| `--skip-cleanup` | `` | `store_true` | Download new files but keep old _daily_ files |
| `--tickers` | `` | `str` | Comma-separated ticker list (default: all ~183) |
| `--years` | `20` | `int` | Lookback years (default: 20) |
| `--batch-size` | `50` | `int` | Tickers per WRDS query (default: 50) |
| `--verify-only` | `` | `store_true` | Only run verification on existing _1d.parquet files |
