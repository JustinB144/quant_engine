"""
Almgren-Chriss (2001) optimal execution model.

Computes the trade trajectory that minimizes a combination of expected
execution cost and execution risk (variance of cost) for liquidating or
acquiring a large position over a fixed number of intervals.

Reference:
    Almgren, R. & Chriss, N. (2001). "Optimal execution of portfolio
    transactions." Journal of Risk, 3(2), 5-39.
"""

from __future__ import annotations

import numpy as np

from ..config import ALMGREN_CHRISS_RISK_AVERSION


def almgren_chriss_trajectory(
    total_shares: int,
    n_intervals: int,
    daily_volume: float,
    daily_volatility: float,
    risk_aversion: float = ALMGREN_CHRISS_RISK_AVERSION,
    # TODO: When ALMGREN_CHRISS_ENABLED is activated, add a conversion bridge
    # between AlmgrenChriss per-share impact units and ExecutionModel bps units.
    # See Audit Report 05, F-27 for details.
    temporary_impact: float = 0.01,
    permanent_impact: float = 0.01,
) -> np.ndarray:
    """Compute the optimal execution trajectory using the Almgren-Chriss model.

    The closed-form solution minimises the objective:
        E[cost] + risk_aversion * Var[cost]

    The optimal trade list (shares per interval) is:
        x_j = (total_shares / n) * sinh(kappa * (T - t_j)) / sinh(kappa * T)

    where kappa = sqrt(risk_aversion * sigma^2 / eta), sigma is the per-interval
    volatility, and eta is the temporary impact coefficient.

    Parameters
    ----------
    total_shares : int
        Total number of shares to execute (positive = buy/sell direction
        is handled externally; this is the magnitude).
    n_intervals : int
        Number of trading intervals to split the execution across.
        Must be >= 1.
    daily_volume : float
        Average daily trading volume in shares.  Used for scaling but does
        not directly enter the closed-form; the impact coefficients are
        assumed pre-calibrated.
    daily_volatility : float
        Annualised or daily volatility (should match the unit convention of
        the impact parameters).  Used as sigma in the model.
    risk_aversion : float
        Trader's risk-aversion parameter (lambda).  Higher values produce
        more front-loaded (aggressive) trajectories.
    temporary_impact : float
        Temporary market-impact coefficient (eta).  Cost per share of
        temporary impact scales as eta * (x_j / tau).
    permanent_impact : float
        Permanent market-impact coefficient (gamma).  Not used in the
        trajectory shape (it is a constant cost independent of schedule)
        but included for cost estimation.

    Returns
    -------
    np.ndarray
        Array of shape (n_intervals,) giving the number of shares to trade
        in each interval.  The array sums to *total_shares*.
    """
    if n_intervals < 1:
        raise ValueError("n_intervals must be >= 1")
    if total_shares == 0:
        return np.zeros(n_intervals)

    n = int(n_intervals)
    sigma = float(daily_volatility)
    eta = float(max(temporary_impact, 1e-12))  # avoid division by zero

    # Per-interval time step (fraction of the trading day)
    tau = 1.0 / n

    # kappa controls how aggressively the trajectory is front-loaded
    # kappa = sqrt(risk_aversion * sigma^2 / eta)
    kappa = np.sqrt(float(risk_aversion) * sigma ** 2 / eta)

    T = 1.0  # normalised total time (1 trading day)

    # Time points: t_j for j = 0, 1, ..., n-1
    t = np.linspace(0, T - tau, n)

    # Closed-form trade sizes
    sinh_denom = np.sinh(kappa * T)
    if abs(sinh_denom) < 1e-15:
        # kappa ~ 0 => risk-neutral case => uniform TWAP
        trajectory = np.full(n, float(total_shares) / n)
    else:
        raw = np.sinh(kappa * (T - t)) / sinh_denom
        # Normalise so the trajectory sums exactly to total_shares
        trajectory = raw / raw.sum() * float(total_shares)

    return trajectory


def estimate_execution_cost(
    trajectory: np.ndarray,
    reference_price: float,
    daily_volume: float,
    daily_volatility: float,
    temporary_impact: float = 0.01,
    permanent_impact: float = 0.01,
) -> dict:
    """Estimate the total execution cost for a given trade trajectory.

    Costs are decomposed into:
    - **Permanent impact cost**: proportional to total shares traded and
      permanent impact coefficient (gamma * X^2 / 2).
    - **Temporary impact cost**: proportional to the sum of squared trade
      rates (eta * sum(x_j^2 / tau)).
    - **Timing risk** (standard deviation of cost): sigma * sqrt(sum of
      remaining-inventory-squared * tau).

    Parameters
    ----------
    trajectory : np.ndarray
        Array of trade sizes per interval (output of
        ``almgren_chriss_trajectory``).
    reference_price : float
        Current asset price used to convert share-cost into dollar-cost.
    daily_volume : float
        Average daily trading volume in shares.
    daily_volatility : float
        Daily volatility of the asset.
    temporary_impact : float
        Temporary impact coefficient (eta).
    permanent_impact : float
        Permanent impact coefficient (gamma).

    Returns
    -------
    dict
        ``permanent_cost`` : float — estimated permanent impact cost ($).
        ``temporary_cost`` : float — estimated temporary impact cost ($).
        ``total_expected_cost`` : float — sum of permanent + temporary ($).
        ``timing_risk`` : float — std-dev of execution cost ($).
        ``cost_bps`` : float — total expected cost in basis points of
        notional value.
        ``participation_rate`` : float — average shares-per-interval as a
        fraction of daily_volume.
    """
    n = len(trajectory)
    if n == 0:
        return {
            "permanent_cost": 0.0,
            "temporary_cost": 0.0,
            "total_expected_cost": 0.0,
            "timing_risk": 0.0,
            "cost_bps": 0.0,
            "participation_rate": 0.0,
        }

    tau = 1.0 / n
    total_shares = float(np.sum(trajectory))
    sigma = float(daily_volatility)
    eta = float(temporary_impact)
    gamma = float(permanent_impact)
    px = float(reference_price)

    # Permanent impact cost: gamma * X^2 / 2
    perm_cost_shares = gamma * total_shares ** 2 / 2.0
    perm_cost_dollars = perm_cost_shares * px

    # Temporary impact cost: eta * sum(x_j^2 / tau)
    temp_cost_shares = eta * np.sum(trajectory ** 2) / tau
    temp_cost_dollars = temp_cost_shares * px

    total_cost = perm_cost_dollars + temp_cost_dollars

    # Timing risk: sigma * sqrt(tau * sum(q_j^2)) where q_j = remaining inventory
    # q_j = total_shares - cumsum(trajectory)[j-1]
    cumulative = np.cumsum(trajectory)
    remaining = total_shares - np.concatenate([[0.0], cumulative[:-1]])
    timing_risk = sigma * np.sqrt(tau * np.sum(remaining ** 2)) * px

    # Notional value
    notional = abs(total_shares * px)
    cost_bps = (total_cost / notional * 10_000) if notional > 0 else 0.0

    # Average participation rate
    avg_shares_per_interval = abs(total_shares) / n
    participation = avg_shares_per_interval / max(daily_volume, 1e-9)

    return {
        "permanent_cost": perm_cost_dollars,
        "temporary_cost": temp_cost_dollars,
        "total_expected_cost": total_cost,
        "timing_risk": timing_risk,
        "cost_bps": cost_bps,
        "participation_rate": participation,
    }
