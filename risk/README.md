# `risk` Package Guide

## Purpose

Risk Management Module — Renaissance-grade portfolio risk controls.

## Package Summary

- Modules: 11
- Classes: 14
- Top-level functions: 14
- LOC: 3,057

## How This Package Fits Into The System

- Reusable risk controls and analytics (sizing, stop loss, drawdown, portfolio risk, covariance, stress, attribution).
- Backtest and paper-trading flows depend on this package for execution realism and risk-managed behavior.
- Dashboard/benchmark services reuse risk analytics through `api/services/data_helpers.py` wrappers.

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
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

## Module Details

### `risk/__init__.py`
- Intent: Risk Management Module — Renaissance-grade portfolio risk controls.
- Classes: none
- Top-level functions: none

### `risk/attribution.py`
- Intent: Performance Attribution --- decompose portfolio returns into market, factor, and alpha.
- Classes: none
- Top-level functions: `_estimate_beta`, `_estimate_factor_loadings`, `decompose_returns`, `compute_attribution_report`

### `risk/covariance.py`
- Intent: Covariance estimation utilities for portfolio risk controls.
- Classes:
  - `CovarianceEstimate`: Covariance estimation output bundle with metadata about the fit method and sample count.
  - `CovarianceEstimator`: Estimate a robust covariance matrix for asset returns.
    - Methods: `estimate`, `portfolio_volatility`
- Top-level functions: `compute_regime_covariance`, `get_regime_covariance`

### `risk/drawdown.py`
- Intent: Drawdown Controller — circuit breakers and recovery protocols.
- Classes:
  - `DrawdownState`: Discrete drawdown-control states used by the drawdown controller.
  - `DrawdownStatus`: Current drawdown state and action directives.
  - `DrawdownController`: Multi-tier drawdown protection with circuit breakers.
    - Methods: `update`, `reset`, `get_summary`
- Top-level functions: none

### `risk/factor_portfolio.py`
- Intent: Factor-Based Portfolio Construction — factor decomposition and exposure analysis.
- Classes: none
- Top-level functions: `compute_factor_exposures`, `compute_residual_returns`

### `risk/metrics.py`
- Intent: Risk Metrics — VaR, CVaR, tail risk, MAE/MFE, and advanced risk analytics.
- Classes:
  - `RiskReport`: Comprehensive risk metrics report.
  - `RiskMetrics`: Computes comprehensive risk metrics from trade returns and equity curves.
    - Methods: `compute_full_report`, `print_report`
- Top-level functions: none

### `risk/portfolio_optimizer.py`
- Intent: Mean-Variance Portfolio Optimization — turnover-penalised portfolio construction.
- Classes: none
- Top-level functions: `optimize_portfolio`

### `risk/portfolio_risk.py`
- Intent: Portfolio Risk Manager — enforces sector, correlation, and exposure limits.
- Classes:
  - `RiskCheck`: Result of a portfolio risk check.
  - `PortfolioRiskManager`: Enforces portfolio-level risk constraints.
    - Methods: `check_new_position`, `portfolio_summary`
- Top-level functions: none

### `risk/position_sizer.py`
- Intent: Position Sizing — Kelly criterion, volatility-scaled, and ATR-based methods.
- Classes:
  - `PositionSize`: Result of position sizing calculation.
  - `PositionSizer`: Multi-method position sizer with conservative blending.
    - Methods: `size_position`, `update_regime_stats`, `update_kelly_bayesian`, `get_bayesian_kelly`, `size_portfolio_aware`, `size_portfolio`
- Top-level functions: none

### `risk/stop_loss.py`
- Intent: Stop Loss Manager — regime-aware ATR stops, trailing, time, and regime-change stops.
- Classes:
  - `StopReason`: Enumerated reasons a stop-loss evaluation can trigger an exit.
  - `StopResult`: Result of stop-loss evaluation.
  - `StopLossManager`: Multi-strategy stop-loss manager.
    - Methods: `evaluate`, `compute_initial_stop`, `compute_risk_per_share`
- Top-level functions: none

### `risk/stress_test.py`
- Intent: Stress Testing Module --- scenario analysis and historical drawdown replay.
- Classes: none
- Top-level functions: `_estimate_portfolio_beta`, `_compute_portfolio_vol`, `run_stress_scenarios`, `run_historical_drawdown_test`, `_find_drawdown_episodes`

## Related Docs

- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md` (current runtime architecture)
- `../docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md` (cross-module constraints)
- `../docs/reference/SOURCE_API_REFERENCE.md` (source-derived Python module inventory)
