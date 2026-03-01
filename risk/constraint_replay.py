"""
Constraint Tightening Replay — stress-test portfolios under regime-conditioned constraints.

Given historical portfolio snapshots and stress scenarios, replays each
snapshot under tightened (stress) constraints and reports which constraints
would have been violated.  This reveals portfolio fragility to regime
shifts before they happen.

Integrates with ``stress_test.py`` scenarios (2008, COVID, 2022, flash
crash, stagflation).

Spec 07, Task 6.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .portfolio_risk import ConstraintMultiplier, PortfolioRiskManager
from .stress_test import DEFAULT_SCENARIOS
from .universe_config import UniverseConfig, ConfigError

logger = logging.getLogger(__name__)


def _estimate_position_beta(
    ticker: str,
    price_data: Dict[str, pd.DataFrame],
    lookback: int = 60,
) -> float:
    """Estimate a single position's beta vs equal-weight market proxy.

    Returns 1.0 as conservative default when data is insufficient.
    """
    if ticker not in price_data:
        return 1.0

    close = price_data[ticker].get("Close")
    if close is None or len(close) < lookback:
        return 1.0

    asset_ret = close.pct_change().iloc[-lookback:].dropna()
    if len(asset_ret) < 20:
        return 1.0

    # Market proxy: equal-weight average of all available tickers
    all_rets = []
    for t, df in price_data.items():
        c = df.get("Close")
        if c is not None and len(c) >= lookback:
            all_rets.append(c.pct_change().iloc[-lookback:])
    if not all_rets:
        return 1.0

    market = pd.concat(all_rets, axis=1).mean(axis=1)
    common = asset_ret.index.intersection(market.index)
    if len(common) < 20:
        return 1.0

    market_var = market.loc[common].var()
    if market_var < 1e-14:
        return 1.0

    cov = asset_ret.loc[common].cov(market.loc[common])
    return float(cov / market_var)


def replay_with_stress_constraints(
    portfolio_history: List[Dict],
    price_data: Dict[str, pd.DataFrame],
    stress_scenarios: Optional[Dict[str, Dict[str, float]]] = None,
    universe_config: Optional[UniverseConfig] = None,
) -> pd.DataFrame:
    """Replay historical portfolios under stress constraints and report violations.

    For each portfolio snapshot in *portfolio_history*, checks what constraints
    would have been violated if stress regime constraints were active.

    Parameters
    ----------
    portfolio_history : list of dict
        Each dict has keys:
            - ``"date"``: date/timestamp of the snapshot
            - ``"positions"``: ``{ticker: weight}`` portfolio weights
    price_data : dict
        ``{ticker: OHLCV DataFrame}`` for all tickers.
    stress_scenarios : dict, optional
        Custom stress scenarios.  If None, uses ``DEFAULT_SCENARIOS``.
    universe_config : UniverseConfig, optional
        Universe configuration for constraint parameters.

    Returns
    -------
    pd.DataFrame
        One row per (date, scenario) with columns:
        - ``date``: snapshot date
        - ``scenario``: scenario name
        - ``sector_cap_util``: max sector utilization under stress
        - ``correlation_util``: max correlation utilization under stress
        - ``gross_util``: gross exposure utilization under stress
        - ``single_name_util``: max single-name utilization
        - ``any_violated``: True if any constraint exceeds 1.0
        - ``max_utilization``: maximum utilization across all constraints
        - ``recommended_backoff``: suggested position scaling factor
    """
    if stress_scenarios is None:
        stress_scenarios = DEFAULT_SCENARIOS

    # Build a stress-mode risk manager
    try:
        config = universe_config or UniverseConfig()
    except ConfigError:
        config = None

    risk_mgr = PortfolioRiskManager(universe_config=config)

    # For stress replay, we use regime=3 (high_volatility) to get stress multipliers
    stress_regime = 3

    results: List[Dict] = []

    for snapshot in portfolio_history:
        date = snapshot.get("date", "unknown")
        positions = snapshot.get("positions", {})

        if not positions:
            continue

        for scenario_name, scenario_params in stress_scenarios.items():
            # Apply scenario shocks before computing utilization
            market_shock = scenario_params.get("market_return", 0.0)
            vol_multiplier = scenario_params.get("volatility_multiplier", 1.0)

            # Apply market shock to position values via beta estimate
            stressed_positions: Dict[str, float] = {}
            for ticker, weight in positions.items():
                beta = _estimate_position_beta(ticker, price_data)
                stressed_weight = weight * (1.0 + market_shock * beta)
                stressed_positions[ticker] = stressed_weight

            # Tighten constraints under high-vol scenario
            stress_tightening = min(1.0, 1.0 / max(vol_multiplier, 1.0))

            # Compute constraint utilization under stress regime with stressed positions
            util = risk_mgr.compute_constraint_utilization(
                positions=stressed_positions,
                price_data=price_data,
                regime=stress_regime,
            )

            # Compute detailed sector breakdown with stressed positions — absolute weights
            sector_weights: Dict[str, float] = {}
            for ticker, weight in stressed_positions.items():
                sector = risk_mgr._resolve_sector(ticker, price_data)
                sector_weights[sector] = sector_weights.get(sector, 0.0) + abs(weight)

            # Get stress-conditioned limits, tightened by vol multiplier
            mults = risk_mgr.multiplier.get_multipliers(stress_regime)
            eff_sector_cap = risk_mgr.max_sector_pct * mults.get("sector_cap", 1.0) * stress_tightening
            eff_gross = risk_mgr.max_gross * mults.get("gross_exposure", 1.0) * stress_tightening

            # Sector utilization — absolute weights
            max_sector_util = 0.0
            if sector_weights and eff_sector_cap > 0:
                max_sector_util = max(abs(v) for v in sector_weights.values()) / eff_sector_cap

            # Gross utilization — sum of absolute weights
            gross = sum(abs(w) for w in stressed_positions.values())
            gross_util = gross / eff_gross if eff_gross > 0 else 0.0

            # Single name — absolute weights, tightened by vol
            eff_single = risk_mgr.max_single * stress_tightening
            single_util = 0.0
            if stressed_positions and eff_single > 0:
                single_util = max(abs(w) for w in stressed_positions.values()) / eff_single

            # Correlation utilization (from overall util)
            corr_util = util.get("correlation", 0.0)

            max_util = max(
                max_sector_util, gross_util, single_util, corr_util, 0.0,
            )
            any_violated = max_util > 1.0

            # Backoff recommendation
            backoff = risk_mgr._compute_backoff_factor(max_util)

            results.append({
                "date": date,
                "scenario": scenario_name,
                "market_shock": market_shock,
                "vol_multiplier": vol_multiplier,
                "sector_cap_util": round(max_sector_util, 4),
                "gross_util": round(gross_util, 4),
                "single_name_util": round(single_util, 4),
                "correlation_util": round(corr_util, 4),
                "any_violated": any_violated,
                "max_utilization": round(max_util, 4),
                "recommended_backoff": round(backoff, 4),
                "n_positions": len(positions),
                "gross_exposure": round(gross, 4),
            })

    if not results:
        return pd.DataFrame(columns=[
            "date", "scenario", "market_shock", "vol_multiplier",
            "sector_cap_util", "gross_util", "single_name_util",
            "correlation_util", "any_violated", "max_utilization",
            "recommended_backoff", "n_positions", "gross_exposure",
        ])

    return pd.DataFrame(results)


def compute_robustness_score(replay_df: pd.DataFrame) -> Dict[str, float]:
    """Compute a robustness score from constraint replay results.

    Returns a dict with:
    - ``overall_score``: 0-1 where 1.0 = no violations across all scenarios
    - ``per_scenario``: {scenario: score} where score = fraction of dates passing
    - ``worst_scenario``: scenario name with most violations
    - ``avg_max_utilization``: average max utilization across all snapshots
    """
    if replay_df.empty:
        return {
            "overall_score": 1.0,
            "per_scenario": {},
            "worst_scenario": "none",
            "avg_max_utilization": 0.0,
        }

    # Overall: fraction of (date, scenario) pairs with no violations
    overall = float((~replay_df["any_violated"]).mean())

    # Per scenario
    per_scenario = {}
    for scenario in replay_df["scenario"].unique():
        mask = replay_df["scenario"] == scenario
        per_scenario[scenario] = float((~replay_df.loc[mask, "any_violated"]).mean())

    # Worst scenario
    worst = min(per_scenario, key=per_scenario.get) if per_scenario else "none"

    return {
        "overall_score": round(overall, 4),
        "per_scenario": per_scenario,
        "worst_scenario": worst,
        "avg_max_utilization": round(float(replay_df["max_utilization"].mean()), 4),
    }
