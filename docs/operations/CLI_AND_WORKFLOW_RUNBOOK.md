cd# CLI and Workflow Runbook

## What This Document Covers

This is the operator runbook for the root `run_*.py` scripts.
It explains what each entry point is for, when to use it, and what to check before/after running it.

## CLI Entry Point Summary

| Script | Purpose (module doc) | `argparse` Options Detected |
|---|---|---:|
| `run_autopilot.py` | Run one full autopilot cycle: | 11 |
| `run_backtest.py` | Backtest the trained model on historical data. | 16 |
| `run_dash.py` | Quant Engine -- Dash Dashboard Launcher. | 3 |
| `run_kalshi_event_pipeline.py` | Run the integrated Kalshi event-time pipeline inside quant_engine. | 21 |
| `run_predict.py` | Generate predictions using trained ensemble model. | 8 |
| `run_rehydrate_cache_metadata.py` | Backfill cache metadata sidecars for existing OHLCV cache files. | 6 |
| `run_retrain.py` | Retrain the quant engine model — checks triggers and retrains if needed. | 12 |
| `run_train.py` | Train the regime-conditional ensemble model. | 10 |
| `run_wrds_daily_refresh.py` | Re-download all daily OHLCV data from WRDS CRSP to replace old cache files | 6 |

## Standard Workflow Sequences

### A. Health / Monitoring Session
1. Launch `run_dash.py`
2. Review `/system-health` in the UI
3. Review `/dashboard` summary cards
4. Use `/data-explorer` if quality or coverage looks off
5. Check `/autopilot` for registry/paper-trading/event workflow status

### B. Model Development Session
1. Validate data quality and availability
2. Run `run_train.py`
3. Run `run_predict.py`
4. Run `run_backtest.py`
5. Inspect results in `results/` and Dash pages (`/signal-desk`, `/backtest-risk`, `/model-lab`)

### C. Strategy Lifecycle Session (Autopilot)
1. Run `run_autopilot.py`
2. Review promotion outcomes and active registry state
3. Review paper-trading state/equity
4. Investigate promotion failures by contract metrics, not only returns

### D. Kalshi Event Research Session
1. Run `run_kalshi_event_pipeline.py`
2. Confirm ingestion/distribution/feature generation outputs
3. Review walk-forward metrics and promotion decision
4. Inspect event visuals via `/autopilot` page in Dash

## Pre-Run Checklist (Applies to Most Workflows)

- Confirm data/cache freshness if results look stale
- Confirm the intended model version/champion (for prediction/backtests)
- Confirm the workflow should operate on current config values (`config.py`)
- Confirm you are using the active UI stack (`run_dash.py`, not legacy UI)

## Post-Run Checklist (Applies to Most Workflows)

- Review generated artifacts/results and timestamps
- Validate metrics against expectations (especially if behavior changed)
- For backtests/autopilot: inspect drawdown and robustness metrics, not just total return
- For Kalshi: inspect quality/coverage and no-leakage assumptions if outputs look anomalous

## Script Details (Static Source Parse)

### `run_autopilot.py`

Purpose: Run one full autopilot cycle:

Argument parser options detected: 11
- `--full`; help: Use full configured universe
- `--tickers`; help: Explicit ticker list
- `--horizon`; help: Holding/prediction horizon; default: `10`
- `--years`; help: History window for evaluation; default: `15`
- `--version`; help: Model version (champion, latest, or explicit version id); default: `champion`
- `--max-candidates`; help: Limit candidate count
- `--feature-mode`; help: Feature profile for autopilot evaluation; default: `AUTOPILOT_FEATURE_MODE`
- `--no-survivorship`; help: Opt out of survivorship-bias-free point-in-time universe
- `--no-walkforward`; help: Use single-split training instead of rolling walk-forward
- `--allow-in-sample`; help: Disable strict out-of-sample filtering
- `--quiet`

### `run_backtest.py`

Purpose: Backtest the trained model on historical data.

Argument parser options detected: 16
- `--full`; help: Use full universe
- `--tickers`; help: Specific tickers
- `--horizon`; help: Prediction/holding horizon (days); default: `10`
- `--no-validate`; help: Skip walk-forward validation
- `--years`; help: Years of data for backtest; default: `15`
- `--feature-mode`; help: Feature profile: core (reduced complexity) or full; default: `FEATURE_MODE_DEFAULT`
- `--risk`; help: Enable risk management (dynamic sizing, stops, drawdown controls)
- `--advanced`; help: Run advanced validation (Deflated Sharpe, Monte Carlo, PBO, capacity)
- `--n-trials`; help: Number of strategy variants tested (for Deflated Sharpe); default: `1`
- `--version`; help: Model version to test (default: latest); default: `latest`
- `--no-survivorship`; help: Opt out of survivorship-bias-free universe (use static universe instead)
- `--allow-in-sample`; help: Allow scoring dates that overlap model training history
- `--min-confidence`; help: Minimum model confidence to enter (default: config CONFIDENCE_THRESHOLD)
- `--min-predicted`; help: Minimum predicted return to enter (default: config ENTRY_THRESHOLD)
- `--output`; help: Save trade log to CSV
- `--quiet`

### `run_dash.py`

Purpose: Quant Engine -- Dash Dashboard Launcher.

Argument parser options detected: 3
- `--port`; help: Server port (default: 8050); default: `8050`
- `--host`; help: Server host (default: 127.0.0.1); default: `127.0.0.1`
- `--no-debug`; help: Disable debug mode

### `run_kalshi_event_pipeline.py`

Purpose: Run the integrated Kalshi event-time pipeline inside quant_engine.

Argument parser options detected: 21
- `--db-path`; default: `str(KALSHI_DB_PATH)`
- `--backend`; default: `duckdb`
- `--start-ts`
- `--end-ts`
- `--sync-reference`
- `--sync-quotes`
- `--build-distributions`
- `--build-event-features`
- `--event-map`; help: CSV/Parquet with event_id/event_type + market_id
- `--options-reference`; help: CSV/Parquet options panel for disagreement features
- `--options-ts-col`; help: Timestamp column in options reference panel; default: `ts`
- `--output`; default: `str(RESULTS_DIR / "kalshi_event_features.parquet")`
- `--labels-first-print`
- `--labels-revised`
- `--run-walkforward`
- `--disable-promotion-gate`
- `--strategy-id`; default: `kalshi_event_default`
- `--build-health-report`; help: Materialize daily health report aggregates
- `--print-health-report`; help: Print latest daily health report table
- `--health-report-output`; help: CSV/Parquet output path for health report
- `--quiet`

### `run_predict.py`

Purpose: Generate predictions using trained ensemble model.

Argument parser options detected: 8
- `--full`; help: Use full universe
- `--tickers`; help: Specific symbols (resolved to PERMNO)
- `--horizon`; help: Prediction horizon (days); default: `10`
- `--version`; help: Model version to use; default: `latest`
- `--feature-mode`; help: Feature profile: core (reduced complexity) or full; default: `FEATURE_MODE_DEFAULT`
- `--output`; help: Save predictions to CSV
- `--top`; help: Show top N signals; default: `20`
- `--quiet`

### `run_rehydrate_cache_metadata.py`

Purpose: Backfill cache metadata sidecars for existing OHLCV cache files.

Argument parser options detected: 6
- `--roots`; help: Optional cache roots to scan. Defaults to standard quant_engine roots.
- `--root-source`; help: Root source override in the form '<path>=<source>' (can repeat).; default: `[]`
- `--default-source`; help: Default source label for unmapped roots.; default: `unknown`
- `--force`; help: Rewrite metadata even when sidecar already exists.
- `--overwrite-source`; help: When rewriting metadata, replace existing source labels.
- `--dry-run`; help: Scan and report only; do not write metadata.

### `run_retrain.py`

Purpose: Retrain the quant engine model — checks triggers and retrains if needed.

Argument parser options detected: 12
- `--force`; help: Force retrain regardless of triggers
- `--status`; help: Show retrain status and exit
- `--versions`; help: List all model versions
- `--rollback`; help: Rollback to a specific version ID
- `--survivorship`; help: Use WRDS survivorship-bias-free universe
- `--full`; help: Use full universe
- `--tickers`; help: Specific tickers
- `--horizon`; help: Prediction horizons; default: `[10]`
- `--years`; help: Years of data; default: `15`
- `--feature-mode`; help: Feature profile: core (reduced complexity) or full; default: `FEATURE_MODE_DEFAULT`
- `--recency`; help: Apply exponential recency weighting
- `--quiet`

### `run_train.py`

Purpose: Train the regime-conditional ensemble model.

Argument parser options detected: 10
- `--full`; help: Use full universe
- `--tickers`; help: Specific tickers to train on
- `--horizon`; help: Forward return horizons (days); default: `[10]`
- `--years`; help: f"Years of historical data (default: {LOOKBACK_YEARS})"; default: `LOOKBACK_YEARS`
- `--no-interactions`; help: Skip interaction features
- `--feature-mode`; help: Feature profile: core (reduced complexity) or full; default: `FEATURE_MODE_DEFAULT`
- `--survivorship`; help: Use WRDS survivorship-bias-free universe
- `--recency`; help: Apply exponential recency weighting
- `--no-version`; help: Skip model versioning (save flat)
- `--quiet`; help: Minimal output

### `run_wrds_daily_refresh.py`

Purpose: Re-download all daily OHLCV data from WRDS CRSP to replace old cache files

Argument parser options detected: 6
- `--dry-run`; help: Preview what would be downloaded without downloading
- `--skip-cleanup`; help: Download new files but keep old _daily_ files
- `--tickers`; help: Comma-separated ticker list (default: all ~183)
- `--years`; help: Lookback years (default: 20); default: `20`
- `--batch-size`; help: Tickers per WRDS query (default: 50); default: `50`
- `--verify-only`; help: Only run verification on existing _1d.parquet files
