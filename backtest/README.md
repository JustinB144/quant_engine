# `backtest` Package Guide

## Purpose

Signal-to-trade simulation, execution realism, and validation/robustness analytics.

## Package Summary

- Modules: 6
- Classes: 15
- Top-level functions: 16
- LOC: 3,372

## How This Package Fits Into the System

- Consumes prediction panels from `models.predictor` and price data from `data` loaders
- Uses `risk/*` components for risk-managed path
- Feeds promotion checks and UI backtest/risk views

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---:|---:|---:|---|
| `backtest/__init__.py` | 0 | 0 | 0 | No module docstring. |
| `backtest/advanced_validation.py` | 551 | 5 | 6 | Advanced Validation — Deflated Sharpe, PBO, Monte Carlo, capacity analysis. |
| `backtest/engine.py` | 1625 | 3 | 0 | Backtester — converts model predictions into simulated trades. |
| `backtest/execution.py` | 271 | 2 | 1 | Execution simulator with spread, market impact, and participation limits. |
| `backtest/optimal_execution.py` | 199 | 0 | 2 | Almgren-Chriss (2001) optimal execution model. |
| `backtest/validation.py` | 726 | 5 | 7 | Walk-forward validation and statistical tests. |

## Module Details

### `backtest/__init__.py`
- Intent: No module docstring; infer from symbol names below.
- Classes: none
- Top-level functions: none

### `backtest/advanced_validation.py`
- Intent: Advanced Validation — Deflated Sharpe, PBO, Monte Carlo, capacity analysis.
- Classes:
  - `DeflatedSharpeResult`: Result of Deflated Sharpe Ratio test.
  - `PBOResult`: Probability of Backtest Overfitting result.
  - `MonteCarloResult`: Monte Carlo simulation result.
  - `CapacityResult`: Strategy capacity analysis.
  - `AdvancedValidationReport`: Complete advanced validation report.
- Top-level functions: `deflated_sharpe_ratio`, `probability_of_backtest_overfitting`, `monte_carlo_validation`, `capacity_analysis`, `run_advanced_validation`, `_print_report`

### `backtest/engine.py`
- Intent: Backtester — converts model predictions into simulated trades.
- Classes:
  - `Trade`: No class docstring.
  - `BacktestResult`: No class docstring.
  - `Backtester`: Simulates trading from model predictions.
    - Methods: `__init__`, `_init_risk_components`, `_almgren_chriss_cost_bps`, `_simulate_entry`, `_simulate_exit`, `_execution_context`, `_effective_return_series`, `_delisting_adjustment_multiplier`, `_trade_realized_return`, `_is_permno_key`, `_assert_permno_inputs`, `run`, `_process_signals`, `_process_signals_risk_managed`, `_compute_metrics`, `_build_daily_equity`, `_compute_turnover`, `_compute_regime_performance`, `_compute_tca`, `_print_result`, `_empty_result`
- Top-level functions: none

### `backtest/execution.py`
- Intent: Execution simulator with spread, market impact, and participation limits.
- Classes:
  - `ExecutionFill`: No class docstring.
  - `ExecutionModel`: Simple market-impact model for backtests.
    - Methods: `__init__`, `simulate`
- Top-level functions: `calibrate_cost_model`

### `backtest/optimal_execution.py`
- Intent: Almgren-Chriss (2001) optimal execution model.
- Classes: none
- Top-level functions: `almgren_chriss_trajectory`, `estimate_execution_cost`

### `backtest/validation.py`
- Intent: Walk-forward validation and statistical tests.
- Classes:
  - `WalkForwardFold`: No class docstring.
  - `WalkForwardResult`: No class docstring.
  - `StatisticalTests`: No class docstring.
  - `CPCVResult`: No class docstring.
  - `SPAResult`: No class docstring.
- Top-level functions: `walk_forward_validate`, `_benjamini_hochberg`, `run_statistical_tests`, `_partition_bounds`, `combinatorial_purged_cv`, `strategy_signal_returns`, `superior_predictive_ability`



## Related Docs

- `../docs/reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md` (deep system audit)
- `../docs/reference/SOURCE_API_REFERENCE.md` (full API inventory)
- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md` (subsystem interactions)
