# `backtest` Package Guide

## Purpose

Backtesting and validation modules used by CLI and API workflows.

## Package Summary

- Modules: 11
- Classes: 29
- Top-level functions: 26
- LOC: 5,652

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
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

## Module Details

### `backtest/__init__.py`
- Intent: Backtesting package exports and namespace initialization.
- Classes: none
- Top-level functions: none

### `backtest/adv_tracker.py`
- Intent: Average Daily Volume (ADV) tracker with EMA smoothing and volume trend analysis.
- Classes:
  - `ADVTracker` (methods: `update`, `update_from_series`, `get_adv`, `get_simple_adv`, `get_volume_trend`, `adjust_participation_limit`, `get_volume_cost_adjustment`, `get_stats`)
- Top-level functions: none

### `backtest/advanced_validation.py`
- Intent: Advanced Validation — Deflated Sharpe, PBO, Monte Carlo, capacity analysis.
- Classes:
  - `DeflatedSharpeResult` (methods: none)
  - `PBOResult` (methods: none)
  - `MonteCarloResult` (methods: none)
  - `CapacityResult` (methods: none)
  - `AdvancedValidationReport` (methods: none)
- Top-level functions: `deflated_sharpe_ratio`, `probability_of_backtest_overfitting`, `monte_carlo_validation`, `capacity_analysis`, `run_advanced_validation`, `_print_report`

### `backtest/cost_calibrator.py`
- Intent: Cost model calibrator for per-market-cap-segment impact coefficients.
- Classes:
  - `CostCalibrator` (methods: `get_marketcap_segment`, `get_impact_coeff`, `get_impact_coeff_by_segment`, `coefficients`, `record_trade`, `calibrate`, `reset_history`)
- Top-level functions: none

### `backtest/cost_stress.py`
- Intent: Cost stress testing — Truth Layer T5.
- Classes:
  - `CostStressPoint` (methods: none)
  - `CostStressResult` (methods: `to_dict`)
  - `CostStressTester` (methods: `run_sweep`, `report`)
- Top-level functions: none

### `backtest/engine.py`
- Intent: Backtester — converts model predictions into simulated trades.
- Classes:
  - `Trade` (methods: none)
  - `BacktestResult` (methods: `summarize_vs_null`)
  - `Backtester` (methods: `run`)
- Top-level functions: none

### `backtest/execution.py`
- Intent: Execution simulator with spread, market impact, and participation limits.
- Classes:
  - `ExecutionFill` (methods: none)
  - `ExecutionModel` (methods: `set_base_transaction_cost_bps`, `simulate`)
- Top-level functions: `calibrate_cost_model`

### `backtest/null_models.py`
- Intent: Null model baselines — Truth Layer T4.
- Classes:
  - `NullBaselineMetrics` (methods: none)
  - `NullModelResults` (methods: `summary`)
  - `RandomBaseline` (methods: `generate_signals`, `compute_returns`)
  - `ZeroBaseline` (methods: `generate_signals`, `compute_returns`)
  - `MomentumBaseline` (methods: `generate_signals`, `compute_returns`)
- Top-level functions: `_compute_metrics`, `compute_null_baselines`

### `backtest/optimal_execution.py`
- Intent: Almgren-Chriss (2001) optimal execution model.
- Classes: none
- Top-level functions: `almgren_chriss_trajectory`, `estimate_execution_cost`

### `backtest/survivorship_comparison.py`
- Intent: Survivorship Bias Comparison — quantify the impact of survivorship bias on backtests.
- Classes:
  - `UniverseMetrics` (methods: none)
  - `SurvivorshipComparisonResult` (methods: none)
- Top-level functions: `_extract_metrics`, `compare_survivorship_impact`, `quick_survivorship_check`

### `backtest/validation.py`
- Intent: Walk-forward validation and statistical tests.
- Classes:
  - `WalkForwardFold` (methods: none)
  - `WalkForwardResult` (methods: none)
  - `StatisticalTests` (methods: none)
  - `CPCVResult` (methods: none)
  - `SPAResult` (methods: none)
  - `WalkForwardEmbargoFold` (methods: none)
  - `WalkForwardEmbargoResult` (methods: none)
- Top-level functions: `walk_forward_validate`, `_benjamini_hochberg`, `run_statistical_tests`, `_partition_bounds`, `combinatorial_purged_cv`, `strategy_signal_returns`, `superior_predictive_ability`, `walk_forward_with_embargo`, `rolling_ic`, `detect_ic_decay`, `_sharpe`, `_spearman_ic`

## Related Docs

- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md`
- `../docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md`
- `../docs/reference/SOURCE_API_REFERENCE.md`
- `../docs/operations/CLI_AND_WORKFLOW_RUNBOOK.md`
