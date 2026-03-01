"""Canonical Sharpe and Sortino ratio calculations.

All risk-adjusted return metrics in the backtest/validation pipeline should
use these functions to ensure consistent conventions.

Convention: ddof=1, configurable Rf, frequency-aware annualization.
"""
import numpy as np

from ..config import RISK_FREE_RATE


def compute_sharpe(
    returns: np.ndarray,
    rf_annual: float = RISK_FREE_RATE,
    frequency: str = "daily",
    annualize: bool = True,
) -> float:
    """Compute Sharpe ratio with consistent conventions.

    Args:
        returns: Array of period returns (daily, per-trade, etc.)
        rf_annual: Annualized risk-free rate (default from config)
        frequency: "daily" (252/yr), "weekly" (52/yr), "monthly" (12/yr),
                   or "per_trade" (requires explicit periods_per_year)
        annualize: Whether to annualize the result

    Returns:
        Sharpe ratio (annualized if annualize=True)
    """
    if len(returns) < 2:
        return 0.0

    periods_map = {"daily": 252, "weekly": 52, "monthly": 12}
    periods_per_year = periods_map.get(frequency, len(returns))

    rf_per_period = rf_annual / periods_per_year
    excess = returns - rf_per_period
    mu = np.mean(excess)
    sigma = np.std(excess, ddof=1)

    if sigma < 1e-12:
        return 0.0

    sharpe = mu / sigma
    if annualize:
        sharpe *= np.sqrt(periods_per_year)
    return float(sharpe)


def compute_sortino(
    returns: np.ndarray,
    rf_annual: float = RISK_FREE_RATE,
    frequency: str = "daily",
    annualize: bool = True,
    target: float = 0.0,
) -> float:
    """Compute Sortino ratio using downside deviation."""
    if len(returns) < 2:
        return 0.0

    periods_map = {"daily": 252, "weekly": 52, "monthly": 12}
    periods_per_year = periods_map.get(frequency, len(returns))

    rf_per_period = rf_annual / periods_per_year
    excess = returns - rf_per_period
    mu = np.mean(excess)
    downside = excess[excess < target]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-12

    if downside_std < 1e-12:
        return 0.0

    sortino = mu / downside_std
    if annualize:
        sortino *= np.sqrt(periods_per_year)
    return float(sortino)
