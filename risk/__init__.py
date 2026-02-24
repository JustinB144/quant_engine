"""
Risk Management Module — Renaissance-grade portfolio risk controls.

Components:
    - PositionSizer: Kelly criterion, volatility-scaled, ATR-based sizing
    - PortfolioRiskManager: sector/correlation/exposure limits
    - DrawdownController: circuit breakers and recovery protocols
    - RiskMetrics: VaR, CVaR, tail risk, MAE/MFE analysis
    - StopLossManager: ATR stops, trailing, time, regime-change stops
    - FactorPortfolio: factor decomposition and exposure analysis
    - PortfolioOptimizer: mean-variance optimisation with turnover penalty
    - Attribution: performance decomposition (market, factor, alpha)
    - StressTest: scenario analysis and historical drawdown replay
    - FactorExposureMonitor: track unintended factor tilts
    - CostBudget: transaction cost budget optimization
"""
from .position_sizer import PositionSizer
from .portfolio_risk import PortfolioRiskManager
from .drawdown import DrawdownController
from .metrics import RiskMetrics
from .stop_loss import StopLossManager
from .covariance import CovarianceEstimator, compute_regime_covariance, get_regime_covariance
from .factor_portfolio import compute_factor_exposures, compute_residual_returns
from .portfolio_optimizer import optimize_portfolio
from .attribution import decompose_returns, compute_attribution_report, compute_rolling_attribution
from .stress_test import (
    run_stress_scenarios,
    run_historical_drawdown_test,
    correlation_stress_test,
    factor_stress_test,
    CRISIS_SCENARIOS,
)
from .factor_monitor import FactorExposureMonitor
from .cost_budget import optimize_rebalance_cost

__all__ = [
    "PositionSizer",
    "PortfolioRiskManager",
    "DrawdownController",
    "RiskMetrics",
    "StopLossManager",
    "CovarianceEstimator",
    "compute_regime_covariance",
    "get_regime_covariance",
    "compute_factor_exposures",
    "compute_residual_returns",
    "optimize_portfolio",
    "decompose_returns",
    "compute_attribution_report",
    "compute_rolling_attribution",
    "run_stress_scenarios",
    "run_historical_drawdown_test",
    "correlation_stress_test",
    "factor_stress_test",
    "CRISIS_SCENARIOS",
    "FactorExposureMonitor",
    "optimize_rebalance_cost",
]
