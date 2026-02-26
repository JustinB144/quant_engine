"""
Fragility Metrics — PnL concentration, drawdown distribution,
recovery time analysis, and critical slowing down detection.

These metrics reveal how brittle a strategy is:
- High PnL concentration (few trades drive most profit) = fragile.
- Drawdown from a single event vs many small ones = concentration risk.
- Recovery time trending upward = critical slowing down (regime shift warning).

Reference: Scheinkman & Woodford on critical slowing down in complex systems.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..config import (
    EVAL_TOP_N_TRADES,
    EVAL_RECOVERY_WINDOW,
    EVAL_CRITICAL_SLOWING_WINDOW,
    EVAL_CRITICAL_SLOWING_SLOPE_THRESHOLD,
)

logger = logging.getLogger(__name__)


def pnl_concentration(
    trades: List[Dict],
    top_n_list: Optional[List[int]] = None,
) -> Dict:
    """Compute percentage of total PnL attributable to the top N trades.

    Parameters
    ----------
    trades : list[dict]
        Each dict must have a ``"pnl"`` or ``"net_return"`` key.
    top_n_list : list[int], optional
        Top-N values to report (default: [5, 10, 20]).

    Returns
    -------
    dict
        Keys: ``top_{n}_pct`` for each N, ``total_pnl``, ``n_trades``,
        ``herfindahl_index`` (PnL concentration HHI), ``fragile`` (bool).
    """
    if top_n_list is None:
        top_n_list = list(EVAL_TOP_N_TRADES)

    # Extract PnL from trades
    pnls = []
    for t in trades:
        if isinstance(t, dict):
            pnl = t.get("pnl", t.get("net_return", 0.0))
        elif hasattr(t, "pnl"):
            pnl = t.pnl
        elif hasattr(t, "net_return"):
            pnl = t.net_return
        else:
            pnl = 0.0
        pnls.append(float(pnl))

    pnl_arr = np.array(pnls, dtype=float)
    n_trades = len(pnl_arr)
    total_pnl = float(np.sum(pnl_arr))

    if n_trades == 0 or abs(total_pnl) < 1e-12:
        result = {"total_pnl": 0.0, "n_trades": 0, "herfindahl_index": 0.0, "fragile": False}
        for n_val in top_n_list:
            result[f"top_{n_val}_pct"] = 0.0
        return result

    # Sort by absolute PnL (largest contributors first)
    sorted_pnl = np.sort(np.abs(pnl_arr))[::-1]

    result: Dict = {
        "total_pnl": total_pnl,
        "n_trades": n_trades,
    }

    for n_val in top_n_list:
        top_n = min(n_val, n_trades)
        top_sum = float(np.sum(sorted_pnl[:top_n]))
        total_abs = float(np.sum(np.abs(pnl_arr)))
        pct = top_sum / total_abs if total_abs > 0 else 0.0
        result[f"top_{n_val}_pct"] = pct

    # Herfindahl index on PnL shares
    if total_pnl != 0:
        shares = np.abs(pnl_arr) / np.sum(np.abs(pnl_arr))
        hhi = float(np.sum(shares ** 2))
    else:
        hhi = 0.0
    result["herfindahl_index"] = hhi

    # Flag as fragile if top 20 trades drive > 70% of PnL
    max_n = max(top_n_list) if top_n_list else 20
    fragile_key = f"top_{max_n}_pct"
    result["fragile"] = result.get(fragile_key, 0.0) > 0.70

    return result


def drawdown_distribution(
    returns: pd.Series,
    window: int = 252,
) -> Dict:
    """Analyze the distribution of drawdown events.

    Determines whether drawdowns are concentrated in a single event or
    spread across many smaller episodes.

    Parameters
    ----------
    returns : pd.Series
        Daily or per-bar return series.
    window : int
        Lookback window for drawdown analysis.

    Returns
    -------
    dict
        Keys: max_dd, max_dd_single_day, avg_dd_during_episodes, n_episodes,
        dd_concentration (fraction of total DD from worst single episode),
        pct_time_underwater, max_dd_duration.
    """
    ret_arr = returns.values.astype(float)
    ret_arr = ret_arr[np.isfinite(ret_arr)]
    n = len(ret_arr)

    if n < 2:
        return _empty_dd_dist()

    cum_eq = np.cumprod(1 + ret_arr)
    running_max = np.maximum.accumulate(cum_eq)
    drawdowns = (cum_eq - running_max) / np.where(running_max > 0, running_max, 1.0)

    max_dd = float(np.min(drawdowns))

    # Worst single-day contribution
    daily_dd_contrib = np.diff(drawdowns, prepend=0.0)
    max_dd_single_day = float(np.min(daily_dd_contrib))

    # Identify drawdown episodes (contiguous periods where drawdown < 0)
    episodes = _find_dd_episodes(drawdowns)
    n_episodes = len(episodes)

    # Average drawdown depth during episodes
    in_dd = drawdowns[drawdowns < 0]
    avg_dd = float(np.mean(in_dd)) if len(in_dd) > 0 else 0.0

    # Drawdown concentration: worst episode depth / total cumulative depth
    if episodes and max_dd < 0:
        episode_depths = [float(np.min(drawdowns[s:e])) for s, e in episodes]
        worst_episode_dd = min(episode_depths)
        sum_episode_depths = sum(abs(d) for d in episode_depths)
        dd_concentration = abs(worst_episode_dd) / sum_episode_depths if sum_episode_depths > 0 else 0.0
    else:
        dd_concentration = 0.0

    # Percentage of time underwater
    pct_underwater = float((drawdowns < -0.001).sum() / n)

    # Max drawdown duration (bars)
    max_dd_duration = 0
    current_dur = 0
    for d in drawdowns:
        if d < -0.001:
            current_dur += 1
            max_dd_duration = max(max_dd_duration, current_dur)
        else:
            current_dur = 0

    return {
        "max_dd": max_dd,
        "max_dd_single_day": max_dd_single_day,
        "avg_dd_during_episodes": avg_dd,
        "n_episodes": n_episodes,
        "dd_concentration": dd_concentration,
        "pct_time_underwater": pct_underwater,
        "max_dd_duration": max_dd_duration,
    }


def recovery_time_distribution(
    returns: pd.Series,
    lookback: int = EVAL_RECOVERY_WINDOW,
) -> pd.Series:
    """Compute recovery times for each drawdown episode.

    Recovery time = number of bars from the trough of a drawdown to the
    point where the equity returns to its pre-drawdown peak.

    Parameters
    ----------
    returns : pd.Series
        Return series.
    lookback : int
        Minimum drawdown depth (in bars) to count as an episode.
        Episodes shallower than 1% are ignored.

    Returns
    -------
    pd.Series
        Recovery times (one per episode), indexed by trough date.
    """
    ret_arr = returns.values.astype(float)
    ret_arr = ret_arr[np.isfinite(ret_arr)]
    n = len(ret_arr)

    if n < 10:
        return pd.Series(dtype=float, name="recovery_time")

    cum_eq = np.cumprod(1 + ret_arr)
    running_max = np.maximum.accumulate(cum_eq)
    drawdowns = (cum_eq - running_max) / np.where(running_max > 0, running_max, 1.0)

    episodes = _find_dd_episodes(drawdowns, min_depth=0.01)

    recovery_times = []
    trough_indices = []

    for start, end in episodes:
        # Find trough within the episode
        episode_dd = drawdowns[start:end]
        if len(episode_dd) == 0:
            continue
        trough_offset = np.argmin(episode_dd)
        trough_idx = start + trough_offset

        # Recovery = bars from trough to end of episode (or next peak)
        recovery = end - trough_idx
        recovery_times.append(recovery)
        trough_indices.append(trough_idx)

    if not recovery_times:
        return pd.Series(dtype=float, name="recovery_time")

    # Build index from trough dates
    index = returns.index
    trough_dates = [index[min(i, len(index) - 1)] for i in trough_indices]

    return pd.Series(recovery_times, index=trough_dates, dtype=float, name="recovery_time")


def detect_critical_slowing_down(
    recovery_times: pd.Series,
    window: int = EVAL_CRITICAL_SLOWING_WINDOW,
    slope_threshold: float = EVAL_CRITICAL_SLOWING_SLOPE_THRESHOLD,
) -> Tuple[bool, Dict]:
    """Detect critical slowing down from increasing recovery times.

    Recovery time trending upward is a leading indicator of regime shift
    (Scheinkman & Woodford).

    Parameters
    ----------
    recovery_times : pd.Series
        Recovery times from :func:`recovery_time_distribution`.
    window : int
        Rolling window for trend estimation.
    slope_threshold : float
        Slope above this flags critical slowing.

    Returns
    -------
    tuple[bool, dict]
        ``(critical_slowing, info_dict)`` where info_dict has keys:
        ``slope``, ``current_recovery_time``, ``historical_median``,
        ``n_episodes``, ``recent_trend`` (str).
    """
    if len(recovery_times) < 3:
        return False, {
            "slope": 0.0,
            "current_recovery_time": float(recovery_times.iloc[-1]) if len(recovery_times) > 0 else 0.0,
            "historical_median": 0.0,
            "n_episodes": len(recovery_times),
            "recent_trend": "insufficient_data",
        }

    rt_arr = recovery_times.values.astype(float)
    rt_arr = rt_arr[np.isfinite(rt_arr)]
    n = len(rt_arr)

    if n < 3:
        return False, {
            "slope": 0.0,
            "current_recovery_time": float(rt_arr[-1]) if n > 0 else 0.0,
            "historical_median": float(np.median(rt_arr)) if n > 0 else 0.0,
            "n_episodes": n,
            "recent_trend": "insufficient_data",
        }

    # Linear regression of recovery times over episodes
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = float(np.mean(rt_arr))
    slope = float(
        np.sum((x - x_mean) * (rt_arr - y_mean))
        / (np.sum((x - x_mean) ** 2) + 1e-12)
    )

    # Normalize slope by median recovery time to make threshold scale-invariant
    median_rt = float(np.median(rt_arr))
    normalized_slope = slope / (median_rt + 1e-12)

    current_rt = float(rt_arr[-1])
    historical_median = median_rt

    # Recent trend: last few episodes vs historical
    recent_n = min(5, n)
    recent_mean = float(np.mean(rt_arr[-recent_n:]))
    historical_mean = float(np.mean(rt_arr[:-recent_n])) if n > recent_n else recent_mean

    if recent_mean > historical_mean * 1.5:
        recent_trend = "increasing"
    elif recent_mean < historical_mean * 0.75:
        recent_trend = "decreasing"
    else:
        recent_trend = "stable"

    critical_slowing = normalized_slope > slope_threshold or (
        recent_trend == "increasing" and current_rt > median_rt * 2
    )

    # Also detect frequency of consecutive losing days
    return critical_slowing, {
        "slope": slope,
        "normalized_slope": normalized_slope,
        "current_recovery_time": current_rt,
        "historical_median": historical_median,
        "recent_mean": recent_mean,
        "n_episodes": n,
        "recent_trend": recent_trend,
    }


def consecutive_loss_frequency(
    returns: pd.Series,
    window: int = 60,
) -> Dict:
    """Detect increasing frequency of consecutive losing days.

    Parameters
    ----------
    returns : pd.Series
        Per-bar returns.
    window : int
        Rolling window for measuring loss streak frequency.

    Returns
    -------
    dict
        Keys: max_streak, mean_streak, trend_slope, increasing (bool).
    """
    ret_arr = returns.values.astype(float)
    ret_arr = ret_arr[np.isfinite(ret_arr)]
    n = len(ret_arr)

    if n < window:
        return {
            "max_streak": 0,
            "mean_streak": 0.0,
            "trend_slope": 0.0,
            "increasing": False,
        }

    # Find all loss streaks
    streaks = []
    current_streak = 0
    for r in ret_arr:
        if r < 0:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        streaks.append(current_streak)

    if not streaks:
        return {
            "max_streak": 0,
            "mean_streak": 0.0,
            "trend_slope": 0.0,
            "increasing": False,
        }

    streak_arr = np.array(streaks, dtype=float)
    max_streak = int(streak_arr.max())
    mean_streak = float(streak_arr.mean())

    # Trend: are streaks getting longer?
    if len(streak_arr) > 3:
        x = np.arange(len(streak_arr), dtype=float)
        x_mean = x.mean()
        y_mean = streak_arr.mean()
        slope = float(
            np.sum((x - x_mean) * (streak_arr - y_mean))
            / (np.sum((x - x_mean) ** 2) + 1e-12)
        )
    else:
        slope = 0.0

    return {
        "max_streak": max_streak,
        "mean_streak": mean_streak,
        "trend_slope": slope,
        "increasing": slope > 0.05,
    }


# ── Private helpers ──────────────────────────────────────────────────


def _find_dd_episodes(
    drawdowns: np.ndarray,
    min_depth: float = 0.01,
) -> List[Tuple[int, int]]:
    """Find contiguous drawdown episodes from a drawdown array.

    Returns list of (start_index, end_index) tuples.
    """
    n = len(drawdowns)
    episodes = []
    i = 0

    while i < n:
        if drawdowns[i] < -min_depth:
            start = i
            while i < n and drawdowns[i] < 0:
                i += 1
            episodes.append((start, i))
        else:
            i += 1

    return episodes


def _empty_dd_dist() -> Dict:
    """Return empty drawdown distribution result."""
    return {
        "max_dd": 0.0,
        "max_dd_single_day": 0.0,
        "avg_dd_during_episodes": 0.0,
        "n_episodes": 0,
        "dd_concentration": 0.0,
        "pct_time_underwater": 0.0,
        "max_dd_duration": 0,
    }
