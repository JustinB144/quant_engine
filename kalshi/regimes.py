"""
Regime tagging for Kalshi event strategies.

Provides macro regime classification for event-centric strategies:
  - Inflation regime (high/low/moderate)
  - Monetary policy regime (tightening/easing/neutral)
  - Volatility regime (low/moderate/high)

Auto-evaluates strategy stability by regime in promotion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class EventRegimeTag:
    """Macro regime labels attached to an event for regime-stability analysis."""
    inflation_regime: str   # "high", "moderate", "low"
    policy_regime: str      # "tightening", "neutral", "easing"
    vol_regime: str         # "low", "moderate", "high"


def classify_inflation_regime(
    cpi_yoy: float,
    high_threshold: float = 3.5,
    low_threshold: float = 2.0,
) -> str:
    """Classify inflation regime from CPI year-over-year."""
    if not np.isfinite(cpi_yoy):
        return "unknown"
    if cpi_yoy >= high_threshold:
        return "high"
    if cpi_yoy <= low_threshold:
        return "low"
    return "moderate"


def classify_policy_regime(
    fed_funds_change_bps: float,
    tightening_threshold_bps: float = 0.0,
    easing_threshold_bps: float = 0.0,
) -> str:
    """Classify monetary policy regime from Fed funds rate changes."""
    if not np.isfinite(fed_funds_change_bps):
        return "unknown"
    if fed_funds_change_bps > tightening_threshold_bps:
        return "tightening"
    if fed_funds_change_bps < easing_threshold_bps:
        return "easing"
    return "neutral"


def classify_vol_regime(
    vix_level: float,
    high_threshold: float = 25.0,
    low_threshold: float = 15.0,
) -> str:
    """Classify volatility regime from VIX level."""
    if not np.isfinite(vix_level):
        return "unknown"
    if vix_level >= high_threshold:
        return "high"
    if vix_level <= low_threshold:
        return "low"
    return "moderate"


def tag_event_regime(
    cpi_yoy: Optional[float] = None,
    fed_funds_change_bps: Optional[float] = None,
    vix_level: Optional[float] = None,
) -> EventRegimeTag:
    """Tag an event with macro regime classifications."""
    # T5: Use explicit None check instead of `or` to preserve valid zero values
    return EventRegimeTag(
        inflation_regime=classify_inflation_regime(cpi_yoy if cpi_yoy is not None else np.nan),
        policy_regime=classify_policy_regime(fed_funds_change_bps if fed_funds_change_bps is not None else np.nan),
        vol_regime=classify_vol_regime(vix_level if vix_level is not None else np.nan),
    )


def evaluate_strategy_by_regime(
    event_returns: np.ndarray,
    regime_tags: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate strategy performance breakdown by regime.

    Returns dict mapping regime_label -> {mean_return, count, win_rate, sharpe_proxy}.
    """
    if len(event_returns) == 0 or len(regime_tags) == 0:
        return {}

    n = min(len(event_returns), len(regime_tags))
    df = pd.DataFrame({
        "ret": np.asarray(event_returns[:n], dtype=float),
        "regime": [str(t) for t in regime_tags[:n]],
    })
    df = df[np.isfinite(df["ret"])]

    result: Dict[str, Dict[str, float]] = {}
    for regime, group in df.groupby("regime"):
        rets = group["ret"].to_numpy(dtype=float)
        if len(rets) == 0:
            continue
        std = float(np.std(rets))
        result[str(regime)] = {
            "mean_return": float(np.mean(rets)),
            "count": float(len(rets)),
            "win_rate": float((rets > 0).mean()),
            "sharpe_proxy": float(np.mean(rets) / max(std, 1e-12)),
        }
    return result


def regime_stability_score(
    breakdown: Dict[str, Dict[str, float]],
) -> float:
    """
    Score strategy stability across regimes (0-1).

    High score = consistent returns across all regimes.
    Low score = regime-dependent performance.
    """
    if not breakdown:
        return np.nan

    means = [v["mean_return"] for v in breakdown.values() if "mean_return" in v]
    if len(means) < 2:
        return 1.0

    overall_std = float(np.std(means))
    overall_mean = float(np.mean([abs(m) for m in means]))
    if overall_mean < 1e-12:
        return 1.0

    dispersion = overall_std / overall_mean
    return float(np.clip(np.exp(-dispersion), 0.0, 1.0))
