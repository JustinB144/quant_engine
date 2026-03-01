"""
Stress Testing Module --- scenario analysis, correlation stress, and historical drawdown replay.

Provides four main capabilities:

1. **Scenario analysis** (``run_stress_scenarios``): Applies predefined or
   custom macro-economic shocks to a portfolio and estimates the resulting
   loss using factor exposures or simple beta.

2. **Historical drawdown replay** (``run_historical_drawdown_test``): Identifies
   the worst historical drawdown episodes in the returns history and reports
   the portfolio's realised performance during those periods.

3. **Correlation stress test** (``correlation_stress_test``): Models portfolio
   risk when all pairwise correlations spike toward 1.0, as happens during
   crises when diversification fails.

4. **Factor stress test** (``factor_stress_test``): Simulates portfolio impact
   of specific factor shocks using predefined crisis scenarios.

Default built-in scenarios:
    - ``2008_crisis``:   Market -38%, volatility spike +150%
    - ``2020_covid``:    Market -34%, volatility spike +200%
    - ``2022_rates``:    Market -19%, rates shock +250bps
    - ``flash_crash``:   Market -7% in a single day
    - ``stagflation``:   Market -15%, inflation spike +5%
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default stress scenarios
# ---------------------------------------------------------------------------

DEFAULT_SCENARIOS: Dict[str, Dict[str, float]] = {
    "2008_crisis": {
        "market_return": -0.38,
        "volatility_multiplier": 2.5,
        "description_code": 1,  # GFC bear market
    },
    "2020_covid": {
        "market_return": -0.34,
        "volatility_multiplier": 3.0,
        "description_code": 2,  # COVID crash
    },
    "2022_rates": {
        "market_return": -0.19,
        "volatility_multiplier": 1.5,
        "description_code": 3,  # Rate-hiking drawdown
    },
    "flash_crash": {
        "market_return": -0.07,
        "volatility_multiplier": 4.0,
        "description_code": 4,  # Single-day crash
    },
    "stagflation": {
        "market_return": -0.15,
        "volatility_multiplier": 1.8,
        "description_code": 5,  # Persistent inflation + stagnation
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_portfolio_beta(
    portfolio_weights: pd.Series,
    returns_history: pd.DataFrame,
    min_obs: int = 60,
) -> float:
    """Estimate weighted-average beta of the portfolio vs equal-weight market proxy.

    If individual-asset returns are available, each asset's beta vs the
    cross-sectional mean is computed and weight-summed.  If data is
    insufficient, returns 1.0 as a conservative default.
    """
    available = [a for a in portfolio_weights.index if a in returns_history.columns]
    if not available:
        return 1.0

    weights = portfolio_weights.loc[available]
    weights = weights / weights.abs().sum()  # normalise to sum-abs = 1

    asset_returns = returns_history[available].dropna(how="all")
    if len(asset_returns) < min_obs:
        return 1.0

    # Market proxy: equal-weight average of all columns in returns_history
    market = returns_history.mean(axis=1)
    market = market.loc[asset_returns.index]

    market_var = market.var()
    if market_var < 1e-14:
        return 1.0

    weighted_beta = 0.0
    for asset in available:
        asset_ret = asset_returns[asset].dropna()
        common = asset_ret.index.intersection(market.index)
        if len(common) < min_obs:
            # Assume beta = 1 for unknown assets
            weighted_beta += float(weights[asset]) * 1.0
            continue
        cov = np.cov(asset_ret.loc[common].values, market.loc[common].values, ddof=1)[0, 1]
        beta = cov / market_var
        weighted_beta += float(weights[asset]) * beta

    return float(weighted_beta)


def _compute_portfolio_vol(
    portfolio_weights: pd.Series,
    returns_history: pd.DataFrame,
    annual_trading_days: int = 252,
) -> float:
    """Annualized portfolio volatility from historical covariance."""
    available = [a for a in portfolio_weights.index if a in returns_history.columns]
    if not available:
        return 0.20  # Default 20% assumption

    w = portfolio_weights.loc[available].values.astype(float)
    rets = returns_history[available].dropna(how="all")
    if len(rets) < 10:
        return 0.20

    cov = rets.cov().values
    port_var = float(w @ cov @ w.T)
    port_var = max(port_var, 0.0)
    return float(np.sqrt(port_var * annual_trading_days))


# ---------------------------------------------------------------------------
# Main functions
# ---------------------------------------------------------------------------

def run_stress_scenarios(
    portfolio_weights: pd.Series,
    returns_history: pd.DataFrame,
    scenarios: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """Apply stress scenarios to a portfolio and estimate impact.

    Parameters
    ----------
    portfolio_weights : pd.Series
        Asset weights (index = asset identifiers, values = weight fractions).
        Need not sum to 1; they will be normalised internally.
    returns_history : pd.DataFrame
        Historical daily returns for each asset (columns) used to estimate
        betas and volatilities.
    scenarios : dict, optional
        Custom scenarios.  Each key is a scenario name; each value is a dict
        with at least ``"market_return"`` (float, e.g. -0.38 for a -38% shock).
        Optional keys: ``"volatility_multiplier"`` (float, default 1.0).
        If ``None``, the built-in ``DEFAULT_SCENARIOS`` are used.

    Returns
    -------
    pd.DataFrame
        One row per scenario with columns:
        - ``scenario``: Scenario name.
        - ``market_shock``: The assumed market return in the scenario.
        - ``estimated_portfolio_loss``: Estimated portfolio loss (negative).
        - ``portfolio_beta``: Portfolio beta used for the estimate.
        - ``portfolio_vol_annual``: Annualized portfolio volatility.
        - ``vol_adjusted_loss``: Loss adjusted for the vol multiplier.
        - ``var_95_stressed``: Approximate 95% VaR under stressed vol.
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS

    beta = _estimate_portfolio_beta(portfolio_weights, returns_history)
    port_vol = _compute_portfolio_vol(portfolio_weights, returns_history)

    results: List[Dict[str, object]] = []

    for name, params in scenarios.items():
        market_ret = float(params.get("market_return", -0.10))
        vol_mult = float(params.get("volatility_multiplier", 1.0))

        # Core estimate: beta-weighted market return
        estimated_loss = beta * market_ret

        # Vol-adjusted loss: when volatility spikes, realised loss is amplified
        # beyond what beta alone predicts.  Apply a sqrt(vol_mult) amplification
        # to model the fat-tail effect.
        vol_adjusted_loss = estimated_loss * np.sqrt(max(vol_mult, 1.0))

        # Stressed VaR: portfolio vol * vol_multiplier * z_95
        stressed_vol = port_vol * vol_mult
        z_95 = 1.645
        var_95_stressed = -(stressed_vol / np.sqrt(252)) * z_95  # Daily VaR

        results.append({
            "scenario": name,
            "market_shock": market_ret,
            "estimated_portfolio_loss": round(estimated_loss, 6),
            "portfolio_beta": round(beta, 4),
            "portfolio_vol_annual": round(port_vol, 4),
            "vol_adjusted_loss": round(vol_adjusted_loss, 6),
            "var_95_stressed": round(var_95_stressed, 6),
        })

    return pd.DataFrame(results)


def run_historical_drawdown_test(
    portfolio_weights: pd.Series,
    returns_history: pd.DataFrame,
    n_worst: int = 5,
    min_drawdown_pct: float = -0.05,
) -> pd.DataFrame:
    """Replay the worst historical drawdown episodes on the portfolio.

    Identifies the N worst peak-to-trough drawdowns in the returns history
    and computes the portfolio's realised return during each episode.

    Parameters
    ----------
    portfolio_weights : pd.Series
        Asset weights.
    returns_history : pd.DataFrame
        Historical daily asset returns.
    n_worst : int
        Number of worst drawdown episodes to report.
    min_drawdown_pct : float
        Minimum drawdown depth to qualify as an "episode" (e.g., -0.05 = -5%).

    Returns
    -------
    pd.DataFrame
        One row per drawdown episode with columns:
        - ``episode``: Sequential episode index (1-based).
        - ``start_date``: Date the drawdown began (peak).
        - ``trough_date``: Date of the maximum drawdown within the episode.
        - ``end_date``: Date the drawdown recovered (or last date available).
        - ``duration_days``: Number of trading days in the episode.
        - ``market_drawdown``: Drawdown of the equal-weight market proxy.
        - ``portfolio_return``: Realised portfolio return during the episode.
        - ``max_portfolio_drawdown``: Worst intra-episode portfolio drawdown.
    """
    available = [a for a in portfolio_weights.index if a in returns_history.columns]
    if not available or len(returns_history) < 20:
        return pd.DataFrame(columns=[
            "episode", "start_date", "trough_date", "end_date",
            "duration_days", "market_drawdown", "portfolio_return",
            "max_portfolio_drawdown",
        ])

    weights = portfolio_weights.loc[available]
    weights = weights / weights.abs().sum()

    asset_rets = returns_history[available].fillna(0.0)

    # Portfolio daily returns
    port_daily = (asset_rets * weights).sum(axis=1)

    # Market proxy: equal-weight average of all available assets
    market_daily = returns_history.mean(axis=1)

    # Identify drawdown episodes from the market proxy
    episodes = _find_drawdown_episodes(market_daily, min_drawdown_pct)

    # Sort by drawdown depth (most severe first) and take top N
    episodes.sort(key=lambda ep: ep[2])  # ep[2] = max drawdown
    episodes = episodes[:n_worst]

    results: List[Dict[str, object]] = []
    for i, (start_idx, trough_idx, market_dd, end_idx) in enumerate(episodes, 1):
        episode_slice = slice(start_idx, end_idx + 1)
        port_slice = port_daily.iloc[episode_slice]
        duration = len(port_slice)

        # Portfolio return over the episode
        portfolio_return = float((1 + port_slice).prod() - 1)

        # Max portfolio drawdown within the episode
        cum_port = (1 + port_slice).cumprod()
        running_max = cum_port.cummax()
        dd = (cum_port - running_max) / running_max
        max_port_dd = float(dd.min()) if len(dd) > 0 else 0.0

        results.append({
            "episode": i,
            "start_date": str(port_daily.index[start_idx].date())
            if hasattr(port_daily.index[start_idx], "date")
            else str(port_daily.index[start_idx]),
            "trough_date": str(port_daily.index[trough_idx].date())
            if hasattr(port_daily.index[trough_idx], "date")
            else str(port_daily.index[trough_idx]),
            "end_date": str(port_daily.index[end_idx].date())
            if hasattr(port_daily.index[end_idx], "date")
            else str(port_daily.index[end_idx]),
            "duration_days": duration,
            "market_drawdown": round(market_dd, 6),
            "portfolio_return": round(portfolio_return, 6),
            "max_portfolio_drawdown": round(max_port_dd, 6),
        })

    return pd.DataFrame(results)


def _find_drawdown_episodes(
    returns: pd.Series,
    min_drawdown_pct: float = -0.05,
) -> List[Tuple[int, int, float, int]]:
    """Identify non-overlapping drawdown episodes from a return series.

    Returns a list of tuples: (start_index, trough_index, max_drawdown, end_index).
    """
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max

    episodes: List[Tuple[int, int, float, int]] = []
    n = len(drawdown)
    i = 0

    while i < n:
        # Find start of a drawdown
        if drawdown.iloc[i] >= min_drawdown_pct:
            i += 1
            continue

        # We are in a drawdown that exceeds the threshold
        start = i

        # Walk backward to find the peak (start of the drawdown)
        while start > 0 and drawdown.iloc[start - 1] < 0:
            start -= 1

        # Find the trough (lowest point)
        trough = start
        j = start
        while j < n and drawdown.iloc[j] < 0:
            if drawdown.iloc[j] < drawdown.iloc[trough]:
                trough = j
            j += 1

        end = min(j, n - 1)
        max_dd = float(drawdown.iloc[trough])

        episodes.append((start, trough, max_dd, end))

        # Move past this episode
        i = end + 1

    # De-duplicate overlapping episodes (keep the deeper one)
    if len(episodes) <= 1:
        return episodes

    unique: List[Tuple[int, int, float, int]] = [episodes[0]]
    for ep in episodes[1:]:
        prev = unique[-1]
        # Check overlap: if this episode's start is before the previous end
        if ep[0] <= prev[3]:
            # Keep the deeper drawdown
            if ep[2] < prev[2]:
                unique[-1] = ep
        else:
            unique.append(ep)

    return unique


# ---------------------------------------------------------------------------
# Factor-based crisis scenarios
# ---------------------------------------------------------------------------

CRISIS_SCENARIOS: Dict[str, Dict[str, float]] = {
    "equity_crash_2008": {
        "market": -0.40, "size": -0.20, "value": -0.30, "momentum": -0.50,
    },
    "covid_march_2020": {
        "market": -0.35, "volatility": 0.60, "momentum": -0.40,
    },
    "rate_shock": {
        "market": -0.15, "duration": -0.20, "growth": -0.10,
    },
    "liquidity_crisis": {
        "market": -0.20, "size": -0.30, "illiquidity": -0.40,
    },
}


# ---------------------------------------------------------------------------
# Correlation stress test
# ---------------------------------------------------------------------------

def correlation_stress_test(
    portfolio_weights: Dict[str, float],
    covariance: np.ndarray,
    stress_correlation: float = 0.9,
    annual_trading_days: int = 252,
) -> Dict[str, float]:
    """Simulate portfolio risk if all pairwise correlations spike to a stress level.

    During crises, diversification fails as correlations converge toward 1.0.
    This function estimates the portfolio volatility increase and resulting
    tail risk under the stressed correlation regime.

    Parameters
    ----------
    portfolio_weights : Dict[str, float]
        Mapping of asset name to weight.  Need not sum to 1; will be
        normalised internally.
    covariance : np.ndarray
        Current (normal-regime) covariance matrix, shape ``(n, n)``.
        Must be annualised or match the annualisation factor passed.
    stress_correlation : float
        Target pairwise correlation under stress (default 0.9).
    annual_trading_days : int
        Trading days per year for annualisation (default 252).

    Returns
    -------
    Dict[str, float]
        ``normal_portfolio_vol``: annualised portfolio vol under normal correlations.
        ``stress_portfolio_vol``: annualised portfolio vol under stress.
        ``vol_increase_pct``: percentage increase in vol.
        ``stress_var_99_daily``: parametric daily 99% VaR under stress.
        ``max_loss_3sigma``: 3-sigma daily loss estimate under stress.
        ``diversification_benefit_lost``: fraction of diversification benefit erased.
    """
    from scipy.stats import norm as normal_dist

    n = covariance.shape[0]
    w = np.array(list(portfolio_weights.values()), dtype=float)

    if len(w) != n:
        raise ValueError(
            f"Weight vector length ({len(w)}) must match covariance dimension ({n})"
        )

    # Normalise weights to sum to 1
    w_sum = np.abs(w).sum()
    if w_sum > 1e-12:
        w = w / w_sum

    cov = np.asarray(covariance, dtype=float)

    # Extract per-asset vols from the diagonal
    vols = np.sqrt(np.maximum(np.diag(cov), 1e-14))

    # Build stress correlation matrix: all off-diagonal = stress_correlation
    stress_corr = np.full((n, n), stress_correlation)
    np.fill_diagonal(stress_corr, 1.0)

    # Convert to stress covariance
    vol_matrix = np.diag(vols)
    stress_cov = vol_matrix @ stress_corr @ vol_matrix

    # Portfolio vol under both regimes
    normal_var = float(w @ cov @ w)
    stress_var = float(w @ stress_cov @ w)

    normal_vol = float(np.sqrt(max(normal_var, 0.0)))
    stress_vol = float(np.sqrt(max(stress_var, 0.0)))

    # Annualise for reporting
    ann_factor = np.sqrt(annual_trading_days)
    normal_vol_ann = normal_vol * ann_factor
    stress_vol_ann = stress_vol * ann_factor

    vol_increase_pct = (
        (stress_vol_ann / normal_vol_ann - 1.0) * 100
        if normal_vol_ann > 1e-12
        else 0.0
    )

    # 99% daily parametric VaR under stress
    z_99 = normal_dist.ppf(0.01)
    stress_var_99 = float(-z_99 * stress_vol)  # positive = loss magnitude

    # Diversification benefit lost: 1 - (portfolio_vol / weighted_avg_vol)
    weighted_avg_vol = float(np.abs(w) @ vols)
    div_benefit_lost = (
        1.0 - (normal_vol / weighted_avg_vol)
        if weighted_avg_vol > 1e-12
        else 0.0
    )

    return {
        "normal_portfolio_vol": round(normal_vol_ann, 6),
        "stress_portfolio_vol": round(stress_vol_ann, 6),
        "vol_increase_pct": round(vol_increase_pct, 2),
        "stress_var_99_daily": round(stress_var_99, 6),
        "max_loss_3sigma": round(stress_vol * 3.0, 6),
        "max_loss_extreme": round(stress_var_99 * 3.0, 6),
        "diversification_benefit_lost": round(div_benefit_lost, 6),
    }


# ---------------------------------------------------------------------------
# Factor stress test
# ---------------------------------------------------------------------------

def factor_stress_test(
    factor_exposures: Dict[str, float],
    scenarios: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, object]]:
    """Simulate portfolio impact of specific factor shocks.

    For each scenario, the estimated portfolio return is computed as the
    sum of factor exposure times the corresponding factor shock.

    Parameters
    ----------
    factor_exposures : Dict[str, float]
        Current portfolio factor exposures (e.g., ``{"market": 1.1, "size": -0.2}``).
    scenarios : Dict[str, Dict[str, float]], optional
        Factor shocks per scenario.  If ``None``, uses ``CRISIS_SCENARIOS``.

    Returns
    -------
    Dict[str, Dict[str, object]]
        One entry per scenario with keys:
        - ``portfolio_return``: Estimated portfolio return under the scenario.
        - ``factor_contributions``: Per-factor return contribution.
    """
    if scenarios is None:
        scenarios = CRISIS_SCENARIOS

    results: Dict[str, Dict[str, object]] = {}

    for scenario_name, factor_shocks in scenarios.items():
        contributions: Dict[str, float] = {}
        portfolio_impact = 0.0

        for factor, shock in factor_shocks.items():
            exposure = factor_exposures.get(factor, 0.0)
            contrib = exposure * shock
            contributions[factor] = round(contrib, 6)
            portfolio_impact += contrib

        results[scenario_name] = {
            "portfolio_return": round(portfolio_impact, 6),
            "factor_contributions": contributions,
        }

    return results
