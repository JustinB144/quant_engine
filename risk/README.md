# `risk` Package Guide

## Purpose

Reusable risk sizing, constraints, monitoring, and portfolio analytics.

## Package Summary

- Modules: 16
- Classes: 22
- Top-level functions: 21
- LOC: 5,845

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
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

## Module Details

### `risk/__init__.py`
- Intent: Risk Management Module — Renaissance-grade portfolio risk controls.
- Classes: none
- Top-level functions: none

### `risk/attribution.py`
- Intent: Performance Attribution --- decompose portfolio returns into market, factor, and alpha.
- Classes: none
- Top-level functions: `_estimate_beta`, `_estimate_factor_loadings`, `decompose_returns`, `compute_rolling_attribution`, `compute_attribution_report`

### `risk/constraint_replay.py`
- Intent: Constraint Tightening Replay — stress-test portfolios under regime-conditioned constraints.
- Classes: none
- Top-level functions: `replay_with_stress_constraints`, `compute_robustness_score`

### `risk/cost_budget.py`
- Intent: Transaction Cost Budget Optimization — minimize implementation cost for rebalances.
- Classes:
  - `RebalanceResult` (methods: none)
- Top-level functions: `estimate_trade_cost_bps`, `optimize_rebalance_cost`

### `risk/covariance.py`
- Intent: Covariance estimation utilities for portfolio risk controls.
- Classes:
  - `CovarianceEstimate` (methods: none)
  - `CovarianceEstimator` (methods: `estimate`, `portfolio_volatility`)
- Top-level functions: `compute_regime_covariance`, `get_regime_covariance`

### `risk/drawdown.py`
- Intent: Drawdown Controller — circuit breakers and recovery protocols.
- Classes:
  - `DrawdownState` (methods: none)
  - `DrawdownStatus` (methods: none)
  - `DrawdownController` (methods: `update`, `reset`, `get_summary`)
- Top-level functions: none

### `risk/factor_exposures.py`
- Intent: Factor Exposure Manager — compute and enforce regime-conditioned factor bounds.
- Classes:
  - `FactorExposureManager` (methods: `compute_exposures`, `is_stress_regime`, `check_factor_bounds`)
- Top-level functions: none

### `risk/factor_monitor.py`
- Intent: Factor Exposure Monitoring — track portfolio factor tilts and alert on violations.
- Classes:
  - `FactorExposure` (methods: none)
  - `FactorExposureReport` (methods: none)
  - `FactorExposureMonitor` (methods: `compute_exposures`, `check_limits`, `compute_report`)
- Top-level functions: none

### `risk/factor_portfolio.py`
- Intent: Factor-Based Portfolio Construction — factor decomposition and exposure analysis.
- Classes: none
- Top-level functions: `compute_factor_exposures`, `compute_residual_returns`

### `risk/metrics.py`
- Intent: Risk Metrics — VaR, CVaR, tail risk, MAE/MFE, and advanced risk analytics.
- Classes:
  - `RiskReport` (methods: none)
  - `RiskMetrics` (methods: `compute_full_report`, `print_report`)
- Top-level functions: none

### `risk/portfolio_optimizer.py`
- Intent: Mean-Variance Portfolio Optimization — turnover-penalised portfolio construction.
- Classes: none
- Top-level functions: `optimize_portfolio`

### `risk/portfolio_risk.py`
- Intent: Portfolio Risk Manager — enforces sector, correlation, and exposure limits.
- Classes:
  - `RiskCheck` (methods: none)
  - `ConstraintMultiplier` (methods: `is_stress_regime`, `get_multipliers`, `get_multipliers_smoothed`, `reset`)
  - `PortfolioRiskManager` (methods: `check_new_position`, `compute_constraint_utilization`, `invalidate_regime_cov_cache`, `portfolio_summary`)
- Top-level functions: none

### `risk/position_sizer.py`
- Intent: Position Sizing — Kelly criterion, volatility-scaled, and ATR-based methods.
- Classes:
  - `PositionSize` (methods: none)
  - `PositionSizer` (methods: `size_position`, `size_position_paper_trader`, `record_turnover`, `reset_turnover_tracking`, `update_regime_stats`, `update_kelly_bayesian`, `get_bayesian_kelly`, `size_portfolio_aware`, `size_portfolio`, `size_with_backoff`)
- Top-level functions: none

### `risk/stop_loss.py`
- Intent: Stop Loss Manager — regime-aware ATR stops, trailing, time, and regime-change stops.
- Classes:
  - `StopReason` (methods: none)
  - `StopResult` (methods: none)
  - `StopLossManager` (methods: `evaluate`, `compute_initial_stop`, `compute_risk_per_share`)
- Top-level functions: none

### `risk/stress_test.py`
- Intent: Stress Testing Module --- scenario analysis, correlation stress, and historical drawdown replay.
- Classes: none
- Top-level functions: `_estimate_portfolio_beta`, `_compute_portfolio_vol`, `run_stress_scenarios`, `run_historical_drawdown_test`, `_find_drawdown_episodes`, `correlation_stress_test`, `factor_stress_test`

### `risk/universe_config.py`
- Intent: Universe Configuration — centralized sector, liquidity, and borrowability metadata.
- Classes:
  - `ConfigError` (methods: none)
  - `UniverseConfig` (methods: `get_sector`, `get_sector_constituents`, `get_all_sectors`, `get_liquidity_tier`, `is_hard_to_borrow`, `is_restricted`, `constraint_base`, `stress_multipliers`, `factor_limits`, `backoff_policy`, `get_stress_multiplier_set`, `get_factor_bounds` (+1 more))
- Top-level functions: none

## Related Docs

- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md`
- `../docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md`
- `../docs/reference/SOURCE_API_REFERENCE.md`
