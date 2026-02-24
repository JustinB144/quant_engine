# `backtest` Package Guide

## Purpose

Backtesting package exports and namespace initialization.

## Package Summary

- Modules: 6
- Classes: 15
- Top-level functions: 16
- LOC: 3,677

## How This Package Fits Into The System

- Turns model predictions into simulated trades and summary metrics used by both CLI scripts and API endpoints.
- Feeds `results/backtest_*` artifacts read by `api/services/backtest_service.py` and dashboard/benchmark services.
- Validation modules (`validation.py`, `advanced_validation.py`) are reused by promotion gating and risk reviews.

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---:|---:|---:|---|
| `backtest/__init__.py` | 4 | 0 | 0 | Backtesting package exports and namespace initialization. |
| `backtest/advanced_validation.py` | 581 | 5 | 6 | Advanced Validation — Deflated Sharpe, PBO, Monte Carlo, capacity analysis. |
| `backtest/engine.py` | 1869 | 3 | 0 | Backtester — converts model predictions into simulated trades. |
| `backtest/execution.py` | 273 | 2 | 1 | Execution simulator with spread, market impact, and participation limits. |
| `backtest/optimal_execution.py` | 201 | 0 | 2 | Almgren-Chriss (2001) optimal execution model. |
| `backtest/validation.py` | 749 | 5 | 7 | Walk-forward validation and statistical tests. |

## Module Details

### `backtest/__init__.py`
- Intent: Backtesting package exports and namespace initialization.
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
  - `Trade`: Trade record produced by the backtester for one simulated position lifecycle.
  - `BacktestResult`: Aggregate backtest outputs including metrics, curves, and trade history.
  - `Backtester`: Simulates trading from model predictions.
    - Methods: `run`
- Top-level functions: none

### `backtest/execution.py`
- Intent: Execution simulator with spread, market impact, and participation limits.
- Classes:
  - `ExecutionFill`: Simulated execution fill outcome returned by the execution model.
  - `ExecutionModel`: Simple market-impact model for backtests.
    - Methods: `simulate`
- Top-level functions: `calibrate_cost_model`

### `backtest/optimal_execution.py`
- Intent: Almgren-Chriss (2001) optimal execution model.
- Classes: none
- Top-level functions: `almgren_chriss_trajectory`, `estimate_execution_cost`

### `backtest/validation.py`
- Intent: Walk-forward validation and statistical tests.
- Classes:
  - `WalkForwardFold`: Per-fold walk-forward validation metrics for one temporal split.
  - `WalkForwardResult`: Aggregate walk-forward validation summary and overfitting diagnostics.
  - `StatisticalTests`: Bundle of statistical significance tests for prediction quality and signal returns.
  - `CPCVResult`: Combinatorial purged cross-validation summary metrics and pass/fail status.
  - `SPAResult`: Superior Predictive Ability (SPA) test result bundle.
- Top-level functions: `walk_forward_validate`, `_benjamini_hochberg`, `run_statistical_tests`, `_partition_bounds`, `combinatorial_purged_cv`, `strategy_signal_returns`, `superior_predictive_ability`

## Related Docs

- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md` (current runtime architecture)
- `../docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md` (cross-module constraints)
- `../docs/reference/SOURCE_API_REFERENCE.md` (source-derived Python module inventory)
- `../docs/operations/CLI_AND_WORKFLOW_RUNBOOK.md` (entrypoints and workflows)
