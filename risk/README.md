# `risk` Package Guide

## Purpose

Reusable risk sizing, risk limits, drawdown control, optimization, stress testing, and attribution.

## Package Summary

- Modules: 11
- Classes: 14
- Top-level functions: 14
- LOC: 2,742

## How This Package Fits Into the System

- Used by `backtest.engine` and `autopilot.paper_trader`
- Supports analytics and constraints used in UI and validation

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---:|---:|---:|---|
| `risk/__init__.py` | 42 | 0 | 0 | Risk Management Module — Renaissance-grade portfolio risk controls. |
| `risk/attribution.py` | 266 | 0 | 4 | Performance Attribution --- decompose portfolio returns into market, factor, and alpha. |
| `risk/covariance.py` | 244 | 2 | 2 | Covariance estimation utilities for portfolio risk controls. |
| `risk/drawdown.py` | 233 | 3 | 0 | Drawdown Controller — circuit breakers and recovery protocols. |
| `risk/factor_portfolio.py` | 220 | 0 | 2 | Factor-Based Portfolio Construction — factor decomposition and exposure analysis. |
| `risk/metrics.py` | 251 | 2 | 0 | Risk Metrics — VaR, CVaR, tail risk, MAE/MFE, and advanced risk analytics. |
| `risk/portfolio_optimizer.py` | 255 | 0 | 1 | Mean-Variance Portfolio Optimization — turnover-penalised portfolio construction. |
| `risk/portfolio_risk.py` | 327 | 2 | 0 | Portfolio Risk Manager — enforces sector, correlation, and exposure limits. |
| `risk/position_sizer.py` | 290 | 2 | 0 | Position Sizing — Kelly criterion, volatility-scaled, and ATR-based methods. |
| `risk/stop_loss.py` | 251 | 3 | 0 | Stop Loss Manager — regime-aware ATR stops, trailing, time, and regime-change stops. |
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
  - `CovarianceEstimate`: No class docstring.
  - `CovarianceEstimator`: Estimate a robust covariance matrix for asset returns.
    - Methods: `__init__`, `estimate`, `portfolio_volatility`, `_estimate_values`
- Top-level functions: `compute_regime_covariance`, `get_regime_covariance`

### `risk/drawdown.py`
- Intent: Drawdown Controller — circuit breakers and recovery protocols.
- Classes:
  - `DrawdownState`: No class docstring.
  - `DrawdownStatus`: Current drawdown state and action directives.
  - `DrawdownController`: Multi-tier drawdown protection with circuit breakers.
    - Methods: `__init__`, `update`, `_compute_actions`, `reset`, `get_summary`
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
    - Methods: `__init__`, `compute_full_report`, `_drawdown_analytics`, `_drawdown_analytics_array`, `_compute_mae_mfe`, `_empty_report`, `print_report`
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
    - Methods: `__init__`, `_infer_ticker_from_price_df`, `_resolve_sector`, `check_new_position`, `_check_correlations`, `_estimate_portfolio_beta`, `_estimate_portfolio_vol`, `portfolio_summary`
- Top-level functions: none

### `risk/position_sizer.py`
- Intent: Position Sizing — Kelly criterion, volatility-scaled, and ATR-based methods.
- Classes:
  - `PositionSize`: Result of position sizing calculation.
  - `PositionSizer`: Multi-method position sizer with conservative blending.
    - Methods: `__init__`, `size_position`, `_kelly`, `_vol_scaled`, `_atr_based`, `size_portfolio`
- Top-level functions: none

### `risk/stop_loss.py`
- Intent: Stop Loss Manager — regime-aware ATR stops, trailing, time, and regime-change stops.
- Classes:
  - `StopReason`: No class docstring.
  - `StopResult`: Result of stop-loss evaluation.
  - `StopLossManager`: Multi-strategy stop-loss manager.
    - Methods: `__init__`, `evaluate`, `compute_initial_stop`, `compute_risk_per_share`
- Top-level functions: none

### `risk/stress_test.py`
- Intent: Stress Testing Module --- scenario analysis and historical drawdown replay.
- Classes: none
- Top-level functions: `_estimate_portfolio_beta`, `_compute_portfolio_vol`, `run_stress_scenarios`, `run_historical_drawdown_test`, `_find_drawdown_episodes`



## Related Docs

- `../docs/reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md` (deep system audit)
- `../docs/reference/SOURCE_API_REFERENCE.md` (full API inventory)
- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md` (subsystem interactions)
