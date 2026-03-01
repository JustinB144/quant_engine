  Subsystem 11 — Entry Points & Scripts: Audit Report                                                                                                                                                          
                                                                                                                                                                                                               
  T1: Immutable Audit Baseline                                                                                                                                                                                 
                                                                  
  Line Ledger

  ┌─────┬─────────────────────────────────────────┬───────────────┬────────┬───────┬────────┐
  │  #  │                  File                   │ Spec Baseline │ Actual │ Delta │ Status │
  ├─────┼─────────────────────────────────────────┼───────────────┼────────┼───────┼────────┤
  │ 1   │ run_backtest.py                         │ 464           │ 464    │ 0     │ MATCH  │
  ├─────┼─────────────────────────────────────────┼───────────────┼────────┼───────┼────────┤
  │ 2   │ run_train.py                            │ 227           │ 227    │ 0     │ MATCH  │
  ├─────┼─────────────────────────────────────────┼───────────────┼────────┼───────┼────────┤
  │ 3   │ run_predict.py                          │ 237           │ 237    │ 0     │ MATCH  │
  ├─────┼─────────────────────────────────────────┼───────────────┼────────┼───────┼────────┤
  │ 4   │ run_retrain.py                          │ 322           │ 322    │ 0     │ MATCH  │
  ├─────┼─────────────────────────────────────────┼───────────────┼────────┼───────┼────────┤
  │ 5   │ run_autopilot.py                        │ 129           │ 129    │ 0     │ MATCH  │
  ├─────┼─────────────────────────────────────────┼───────────────┼────────┼───────┼────────┤
  │ 6   │ run_server.py                           │ 94            │ 94     │ 0     │ MATCH  │
  ├─────┼─────────────────────────────────────────┼───────────────┼────────┼───────┼────────┤
  │ 7   │ run_kalshi_event_pipeline.py            │ 227           │ 227    │ 0     │ MATCH  │
  ├─────┼─────────────────────────────────────────┼───────────────┼────────┼───────┼────────┤
  │ 8   │ run_wrds_daily_refresh.py               │ 915           │ 915    │ 0     │ MATCH  │
  ├─────┼─────────────────────────────────────────┼───────────────┼────────┼───────┼────────┤
  │ 9   │ run_rehydrate_cache_metadata.py         │ 101           │ 101    │ 0     │ MATCH  │
  ├─────┼─────────────────────────────────────────┼───────────────┼────────┼───────┼────────┤
  │ 10  │ scripts/alpaca_intraday_download.py     │ 1,202         │ 1,618  │ +416  │ GREW   │
  ├─────┼─────────────────────────────────────────┼───────────────┼────────┼───────┼────────┤
  │ 11  │ scripts/ibkr_intraday_download.py       │ 508           │ 1,022  │ +514  │ GREW   │
  ├─────┼─────────────────────────────────────────┼───────────────┼────────┼───────┼────────┤
  │ 12  │ scripts/ibkr_daily_gapfill.py           │ 417           │ 417    │ 0     │ MATCH  │
  ├─────┼─────────────────────────────────────────┼───────────────┼────────┼───────┼────────┤
  │ 13  │ scripts/compare_regime_models.py        │ 323           │ 323    │ 0     │ MATCH  │
  ├─────┼─────────────────────────────────────────┼───────────────┼────────┼───────┼────────┤
  │ 14  │ scripts/generate_types.py               │ 146           │ 146    │ 0     │ MATCH  │
  ├─────┼─────────────────────────────────────────┼───────────────┼────────┼───────┼────────┤
  │ 15  │ scripts/extract_dependencies.py         │ 600           │ 600    │ 0     │ MATCH  │
  ├─────┼─────────────────────────────────────────┼───────────────┼────────┼───────┼────────┤
  │ 16  │ scripts/generate_interface_contracts.py │ 2,570         │ 2,570  │ 0     │ MATCH  │
  ├─────┼─────────────────────────────────────────┼───────────────┼────────┼───────┼────────┤
  │ 17  │ scripts/hotspot_scoring.py              │ 401           │ 401    │ 0     │ MATCH  │
  ├─────┼─────────────────────────────────────────┼───────────────┼────────┼───────┼────────┤
  │     │ TOTAL                                   │ 8,883         │ 9,813  │ +930  │        │
  └─────┴─────────────────────────────────────────┴───────────────┴────────┴───────┴────────┘

  Growth explanation: Both intraday download scripts grew due to quality-gate and data-validation features added between baseline measurement and audit. Growth is in data utility scripts (non-production
  orchestration), so blast radius is limited.

  File Classification

  ┌───────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬───────┐
  │       Category        │                                                                                 Files                                                                                  │ Count │
  ├───────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┤
  │ Production entry      │ run_backtest.py, run_train.py, run_predict.py, run_retrain.py, run_autopilot.py, run_server.py, run_kalshi_event_pipeline.py, run_wrds_daily_refresh.py,               │ 9     │
  │ points                │ run_rehydrate_cache_metadata.py                                                                                                                                        │       │
  ├───────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┤
  │ Data utility scripts  │ alpaca_intraday_download.py, ibkr_intraday_download.py, ibkr_daily_gapfill.py, compare_regime_models.py, generate_types.py                                             │ 5     │
  ├───────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┤
  │ Audit tooling scripts │ extract_dependencies.py, generate_interface_contracts.py, hotspot_scoring.py                                                                                           │ 3     │
  └───────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴───────┘

  Lines reviewed: 9,813 / 9,813 (100%)

  ---
  T2: Shared Artifact Schema Verification

  Backtest Summary (results/backtest_*d_summary.json)

  Writer: run_backtest.py lines 371-383

  Actual schema (11 fields):
  horizon, total_trades, win_rate, avg_return, sharpe, sortino,
  max_drawdown, profit_factor, annualized_return, trades_per_year,
  regime_breakdown

  Spec claimed 17 fields — actual is 11. The spec count is incorrect.

  Consumer Verification

  ┌───────────────────────┬──────────────┬──────────────────────────────────────────────────────────────────────────────────────┬────────────┐
  │       Consumer        │ Spec Claims  │                                   Actual Behavior                                    │  Verified  │
  ├───────────────────────┼──────────────┼──────────────────────────────────────────────────────────────────────────────────────┼────────────┤
  │ api/orchestrator.py   │ Reads fields │ Passthrough json.load() — returns dict as-is, no field-level access                  │ YES        │
  ├───────────────────────┼──────────────┼──────────────────────────────────────────────────────────────────────────────────────┼────────────┤
  │ api/routes/compute.py │ Reads fields │ Passthrough json.load() — returns dict wrapped in API response                       │ YES        │
  ├───────────────────────┼──────────────┼──────────────────────────────────────────────────────────────────────────────────────┼────────────┤
  │ evaluation/engine.py  │ Reads fields │ Does NOT read backtest summary JSON at all — zero file I/O coupling to this artifact │ SPEC ERROR │
  └───────────────────────┴──────────────┴──────────────────────────────────────────────────────────────────────────────────────┴────────────┘

  Does advanced_validation.py alter the output schema? No. run_backtest.py calls run_advanced_validation() but its return value is only printed/logged — it is not merged into the summary dict.

  ---
  T3: Module Constructor Wiring Pass

  All module constructor calls in all 9 production entry points were verified against actual upstream signatures. Zero parameter mismatches found.

  Verified Wiring Matrix

  ┌─────────────────────────────────┬─────────────────────────────┬───────────────────────────────────────┬───────┐
  │           Entry Point           │        Module Wired         │           Signature Source            │ Match │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_backtest.py                 │ FeaturePipeline             │ features/pipeline.py:995 (9 params)   │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_backtest.py                 │ RegimeDetector              │ regime/detector.py:74 (12 params)     │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_backtest.py                 │ EnsemblePredictor           │ models/predictor.py:66 (3 params)     │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_backtest.py                 │ Backtester                  │ backtest/engine.py:180 (13 params)    │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_backtest.py                 │ walk_forward_validate       │ backtest/validation.py:197 (8 params) │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_backtest.py                 │ run_statistical_tests       │ backtest/statistical_tests.py         │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_backtest.py                 │ combinatorial_purged_cv     │ backtest/statistical_tests.py         │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_backtest.py                 │ strategy_signal_returns     │ backtest/statistical_tests.py         │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_backtest.py                 │ superior_predictive_ability │ backtest/statistical_tests.py         │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_backtest.py                 │ run_advanced_validation     │ backtest/advanced_validation.py:536   │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_train.py                    │ FeaturePipeline             │ features/pipeline.py:995              │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_train.py                    │ RegimeDetector              │ regime/detector.py:74                 │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_train.py                    │ ModelTrainer                │ models/trainer.py:209 (5 params)      │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_train.py                    │ ModelTrainer.train_ensemble │ models/trainer.py:678 (12 params)     │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_train.py                    │ ModelGovernance             │ models/governance.py                  │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_predict.py                  │ FeaturePipeline             │ features/pipeline.py:995              │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_predict.py                  │ RegimeDetector              │ regime/detector.py:74                 │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_predict.py                  │ EnsemblePredictor           │ models/predictor.py:66                │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_retrain.py                  │ FeaturePipeline             │ features/pipeline.py:995              │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_retrain.py                  │ RegimeDetector              │ regime/detector.py:74                 │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_retrain.py                  │ ModelTrainer                │ models/trainer.py:209                 │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_retrain.py                  │ RetrainTrigger              │ models/retrain_trigger.py             │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_autopilot.py                │ AutopilotEngine             │ autopilot/engine.py:150 (12 params)   │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_kalshi_event_pipeline.py    │ 5 Kalshi submodules         │ kalshi/*.py                           │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_wrds_daily_refresh.py       │ WRDSDataManager             │ data/wrds_data.py                     │ YES   │
  ├─────────────────────────────────┼─────────────────────────────┼───────────────────────────────────────┼───────┤
  │ run_rehydrate_cache_metadata.py │ rehydrate_cache_metadata    │ data/local_cache.py:545               │ YES   │
  └─────────────────────────────────┴─────────────────────────────┴───────────────────────────────────────┴───────┘

  API orchestrator consistency: api/orchestrator.py wires the same modules with consistent parameters as the entry points. Verified.

  ---
  T4: CLI Argument Handling & Reproducibility

  CLI Defaults vs Config Constants

  ┌─────────────────┬──────────────────┬───────────────────────────────────┬───────────────────────────┬────────────────────────┐
  │   Entry Point   │     Argument     │              Default              │      Config Constant      │      Consistent?       │
  ├─────────────────┼──────────────────┼───────────────────────────────────┼───────────────────────────┼────────────────────────┤
  │ run_backtest.py │ --years          │ 15 (hardcoded)                    │ LOOKBACK_YEARS (not used) │ NO — P3                │
  ├─────────────────┼──────────────────┼───────────────────────────────────┼───────────────────────────┼────────────────────────┤
  │ run_retrain.py  │ --years          │ 15 (hardcoded)                    │ LOOKBACK_YEARS (not used) │ NO — P3                │
  ├─────────────────┼──────────────────┼───────────────────────────────────┼───────────────────────────┼────────────────────────┤
  │ run_train.py    │ --lookback-years │ LOOKBACK_YEARS                    │ Uses config constant      │ YES                    │
  ├─────────────────┼──────────────────┼───────────────────────────────────┼───────────────────────────┼────────────────────────┤
  │ run_predict.py  │ --feature-mode   │ choices=["core","full"]           │ —                         │ Missing "minimal" — P3 │
  ├─────────────────┼──────────────────┼───────────────────────────────────┼───────────────────────────┼────────────────────────┤
  │ run_retrain.py  │ --feature-mode   │ choices=["core","full"]           │ —                         │ Missing "minimal" — P3 │
  ├─────────────────┼──────────────────┼───────────────────────────────────┼───────────────────────────┼────────────────────────┤
  │ run_backtest.py │ --feature-mode   │ choices=["minimal","core","full"] │ —                         │ Includes all 3         │
  ├─────────────────┼──────────────────┼───────────────────────────────────┼───────────────────────────┼────────────────────────┤
  │ run_train.py    │ --feature-mode   │ choices=["minimal","core","full"] │ —                         │ Includes all 3         │
  └─────────────────┴──────────────────┴───────────────────────────────────┴───────────────────────────┴────────────────────────┘

  Reproducibility Manifest Generation

  ┌─────────────────────────────────┬─────────────────────┬───────────────────────────────────────────────┐
  │           Entry Point           │ Generates Manifest? │                     Notes                     │
  ├─────────────────────────────────┼─────────────────────┼───────────────────────────────────────────────┤
  │ run_backtest.py                 │ YES                 │ Via reproducibility.py                        │
  ├─────────────────────────────────┼─────────────────────┼───────────────────────────────────────────────┤
  │ run_train.py                    │ YES                 │ Via reproducibility.py                        │
  ├─────────────────────────────────┼─────────────────────┼───────────────────────────────────────────────┤
  │ run_predict.py                  │ YES                 │ Via reproducibility.py                        │
  ├─────────────────────────────────┼─────────────────────┼───────────────────────────────────────────────┤
  │ run_retrain.py                  │ YES                 │ Via reproducibility.py                        │
  ├─────────────────────────────────┼─────────────────────┼───────────────────────────────────────────────┤
  │ run_autopilot.py                │ YES                 │ Via reproducibility.py                        │
  ├─────────────────────────────────┼─────────────────────┼───────────────────────────────────────────────┤
  │ run_kalshi_event_pipeline.py    │ NO                  │ No import of reproducibility module           │
  ├─────────────────────────────────┼─────────────────────┼───────────────────────────────────────────────┤
  │ run_wrds_daily_refresh.py       │ NO                  │ No import of reproducibility module           │
  ├─────────────────────────────────┼─────────────────────┼───────────────────────────────────────────────┤
  │ run_rehydrate_cache_metadata.py │ NO                  │ Maintenance utility — acceptable              │
  ├─────────────────────────────────┼─────────────────────┼───────────────────────────────────────────────┤
  │ run_server.py                   │ N/A                 │ Long-running server — manifest not applicable │
  └─────────────────────────────────┴─────────────────────┴───────────────────────────────────────────────┘

  ---
  T5: Data Utility & Audit Tooling Review

  Data Utility Scripts

  ┌─────────────────────────────┬───────────────────────────────────────────────┬──────────────────────────────────────────┬────────────┬──────────────────────────────────────────────────────────────────┐
  │           Script            │                Import Pattern                 │               Cache Target               │   Schema   │                              Issues                              │
  ├─────────────────────────────┼───────────────────────────────────────────────┼──────────────────────────────────────────┼────────────┼──────────────────────────────────────────────────────────────────┤
  │ alpaca_intraday_download.py │ importlib.util to load config.py and          │ DATA_CACHE_ALPACA_DIR                    │ OHLCV +    │ P2: Bypasses quant_engine namespace                              │
  │                             │ local_cache.py directly                       │                                          │ date       │                                                                  │
  ├─────────────────────────────┼───────────────────────────────────────────────┼──────────────────────────────────────────┼────────────┼──────────────────────────────────────────────────────────────────┤
  │ ibkr_intraday_download.py   │ importlib.util to load config.py and          │ DATA_CACHE_IBKR_DIR                      │ OHLCV +    │ P2: Same importlib bypass                                        │
  │                             │ local_cache.py directly                       │                                          │ date       │                                                                  │
  ├─────────────────────────────┼───────────────────────────────────────────────┼──────────────────────────────────────────┼────────────┼──────────────────────────────────────────────────────────────────┤
  │ ibkr_daily_gapfill.py       │ Standard imports from quant_engine.*          │ Merges into existing WRDS cache          │ OHLCV +    │ P2: Imports private _normalize_ohlcv_columns, _write_cache_meta  │
  │                             │                                               │                                          │ date       │ from local_cache.py                                              │
  ├─────────────────────────────┼───────────────────────────────────────────────┼──────────────────────────────────────────┼────────────┼──────────────────────────────────────────────────────────────────┤
  │ compare_regime_models.py    │ Standard imports                              │ Writes                                   │ —          │ Clean. Read-only for model artifacts.                            │
  │                             │                                               │ results/regime_model_comparison.json     │            │                                                                  │
  ├─────────────────────────────┼───────────────────────────────────────────────┼──────────────────────────────────────────┼────────────┼──────────────────────────────────────────────────────────────────┤
  │ generate_types.py           │ Standard imports                              │ Writes frontend/src/types/generated.ts   │ —          │ Clean. Pydantic V2 model_fields API correct.                     │
  └─────────────────────────────┴───────────────────────────────────────────────┴──────────────────────────────────────────┴────────────┴──────────────────────────────────────────────────────────────────┘

  Audit Tooling Scripts

  ┌─────────────────────────────────┬───────────────────────┬─────────────────────────────┬──────────────┐
  │             Script              │ Imports quant_engine? │ Writes to production paths? │   Verdict    │
  ├─────────────────────────────────┼───────────────────────┼─────────────────────────────┼──────────────┤
  │ extract_dependencies.py         │ NO                    │ NO (docs/audit/ only)       │ Pure tooling │
  ├─────────────────────────────────┼───────────────────────┼─────────────────────────────┼──────────────┤
  │ generate_interface_contracts.py │ NO                    │ NO (docs/audit/ only)       │ Pure tooling │
  ├─────────────────────────────────┼───────────────────────┼─────────────────────────────┼──────────────┤
  │ hotspot_scoring.py              │ NO                    │ NO (read-only subprocess)   │ Pure tooling │
  └─────────────────────────────────┴───────────────────────┴─────────────────────────────┴──────────────┘

  All 3 audit scripts confirmed as non-production with no side effects.

  ---
  T6: Findings Table

  ┌─────┬──────────┬─────────────────────────────────────┬─────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┬───────────────────────────────┐
  │ ID  │ Severity │                File                 │                         Finding                         │                       Evidence                       │            Impact             │
  ├─────┼──────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────┤
  │ F1  │ P2       │ Spec                                │ Backtest summary schema claimed as 17 fields; actual is │ run_backtest.py:371-383 — dict literal has exactly   │ Spec documentation error; no  │
  │     │          │                                     │  11                                                     │ 11 keys                                              │ runtime impact                │
  ├─────┼──────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────┤
  │ F2  │ P2       │ Spec                                │ evaluation/engine.py listed as backtest summary         │ Grep for json.load, summary, backtest in             │ Spec documentation error; no  │
  │     │          │                                     │ consumer; it is not                                     │ evaluation/engine.py returns zero hits               │ runtime impact                │
  ├─────┼──────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────┤
  │     │          │                                     │ Uses from api.config import ApiSettings without         │ Line 6-7: bare from api.config import vs. all other  │ Works only when CWD is        │
  │ F3  │ P2       │ run_server.py                       │ sys.path.insert; all other entry points use             │ entry points using from quant_engine.*               │ project root; fragile in      │
  │     │          │                                     │ sys.path.insert(0, ...) + quant_engine.* namespace      │                                                      │ deployment                    │
  ├─────┼──────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────┤
  │     │          │                                     │ Uses importlib.util to load config.py and               │ Lines ~30-50: importlib.util.spec_from_file_location │ Fragile path resolution;      │
  │ F4  │ P2       │ scripts/alpaca_intraday_download.py │ local_cache.py directly, bypassing quant_engine         │  pattern                                             │ breaks if file structure      │
  │     │          │                                     │ namespace                                               │                                                      │ changes                       │
  ├─────┼──────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────┤
  │ F5  │ P2       │ scripts/ibkr_intraday_download.py   │ Same importlib.util bypass as F4                        │ Same pattern as alpaca script                        │ Same fragility                │
  ├─────┼──────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────┤
  │     │          │                                     │ Imports private functions _normalize_ohlcv_columns and  │                                                      │ Coupling to private API; will │
  │ F6  │ P2       │ scripts/ibkr_daily_gapfill.py       │ _write_cache_meta from local_cache.py                   │ Import statement at top of file                      │  break if local_cache.py      │
  │     │          │                                     │                                                         │                                                      │ refactors internals           │
  ├─────┼──────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────┤
  │     │          │                                     │ --feature-mode choices ["core","full"] missing          │ run_predict.py argparse definition vs                │ Inconsistent user experience; │
  │ F7  │ P3       │ run_predict.py                      │ "minimal"                                               │ run_backtest.py/run_train.py which include "minimal" │  "minimal" mode unusable from │
  │     │          │                                     │                                                         │                                                      │  predict CLI                  │
  ├─────┼──────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────┤
  │ F8  │ P3       │ run_retrain.py                      │ Same missing "minimal" in --feature-mode choices        │ Same pattern as F7                                   │ Same inconsistency            │
  ├─────┼──────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────┤
  │ F9  │ P3       │ run_kalshi_event_pipeline.py        │ No reproducibility manifest generation                  │ Grep for reproducibility returns zero hits in this   │ Kalshi pipeline runs are not  │
  │     │          │                                     │                                                         │ file                                                 │ reproducible via manifest     │
  ├─────┼──────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────┤
  │ F10 │ P3       │ run_wrds_daily_refresh.py           │ No reproducibility manifest generation                  │ Grep for reproducibility returns zero hits in this   │ Daily refresh runs are not    │
  │     │          │                                     │                                                         │ file                                                 │ reproducible via manifest     │
  ├─────┼──────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────┤
  │ F11 │ P3       │ run_retrain.py                      │ Accesses private method trigger._save_metadata()        │ Call site in run_retrain.py                          │ Coupling to private API       │
  ├─────┼──────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────┤
  │ F12 │ P3       │ run_backtest.py, run_retrain.py     │ --years default hardcoded to 15 instead of using        │ argparse default vs run_train.py which uses          │ Config drift risk if constant │
  │     │          │                                     │ LOOKBACK_YEARS config constant                          │ LOOKBACK_YEARS                                       │  is changed                   │
  └─────┴──────────┴─────────────────────────────────────┴─────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────┴───────────────────────────────┘

  ---
  Acceptance Criteria Verification

  ┌─────┬──────────────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │  #  │                              Criterion                               │                                                          Status                                                           │
  ├─────┼──────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 1   │ 100% of lines in all 17 files explicitly reviewed                    │ PASS — 9,813/9,813 lines                                                                                                  │
  ├─────┼──────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 2   │ backtest_*d_summary.json schema verified against all downstream      │ PASS — 11-field schema verified; 2 API consumers do passthrough reads; evaluation/engine.py is NOT a consumer (spec error │
  │     │ consumers                                                            │  corrected)                                                                                                               │
  ├─────┼──────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 3   │ All module constructor calls match expected signatures               │ PASS — 23+ signatures verified, zero mismatches                                                                           │
  ├─────┼──────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 4   │ CLI argument defaults consistent with config.py constants            │ PARTIAL — --years hardcoded in 2 entry points (F12, P3); feature-mode inconsistent in 2 entry points (F7/F8, P3)          │
  ├─────┼──────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 5   │ Reproducibility manifests capture all relevant runtime parameters    │ PARTIAL — 2 entry points lack manifests entirely (F9/F10, P3)                                                             │
  ├─────┼──────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 6   │ No unresolved P0/P1 findings                                         │ PASS — zero P0, zero P1                                                                                                   │
  ├─────┼──────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 7   │ Audit tooling scripts confirmed non-production                       │ PASS — all 3 confirmed pure tooling with no side effects                                                                  │
  └─────┴──────────────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Summary

  - P0 findings: 0
  - P1 findings: 0
  - P2 findings: 6 (2 spec documentation errors, 1 inconsistent import pattern in run_server.py, 2 importlib.util namespace bypasses, 1 private function import)
  - P3 findings: 6 (2 missing feature-mode choices, 2 missing reproducibility manifests, 1 private method access, 1 hardcoded default)
  - Total findings: 12

  The subsystem is structurally sound — all module wiring is correct with zero mismatches, and the most critical shared artifact (backtest summary JSON) has a clean 11-field schema with correct consumers.
  The P2 findings are mostly import-pattern fragility in data utility scripts (not production orchestration), and two spec documentation errors that should be corrected. The P3 findings are consistency
  issues that pose no immediate runtime risk.

  Recommended remediations by priority:
  1. Fix spec documentation (F1, F2) — correct field count and consumer list
  2. Standardize run_server.py import pattern (F3) to match other entry points
  3. Add "minimal" to feature-mode choices in run_predict.py and run_retrain.py (F7, F8)
  4. Use LOOKBACK_YEARS config constant for --years default (F12)
  5. Add reproducibility manifests to run_kalshi_event_pipeline.py and run_wrds_daily_refresh.py (F9, F10)
  6. Expose public API in local_cache.py for the functions used by ibkr_daily_gapfill.py (F6) and the method used by run_retrain.py (F11)
  7. The importlib.util patterns (F4, F5) are acceptable for standalone scripts but should be documented as intentional