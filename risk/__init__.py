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
"""
from .position_sizer import PositionSizer
from .portfolio_risk import PortfolioRiskManager
from .drawdown import DrawdownController
from .metrics import RiskMetrics
from .stop_loss import StopLossManager
from .covariance import CovarianceEstimator, compute_regime_covariance, get_regime_covariance
from .factor_portfolio import compute_factor_exposures, compute_residual_returns
from .portfolio_optimizer import optimize_portfolio
from .attribution import decompose_returns, compute_attribution_report
from .stress_test import run_stress_scenarios, run_historical_drawdown_test

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
    "run_stress_scenarios",
    "run_historical_drawdown_test",
]
