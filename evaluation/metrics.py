"""
Evaluation Metrics — per-slice performance metrics and decile spread analysis.

Computes Sharpe, max drawdown, recovery time, and other standard metrics
on arbitrary return slices, with confidence intervals and low-N warnings.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats as sp_stats
except ImportError:
    sp_stats = None

from ..config import EVAL_MIN_SLICE_SAMPLES, EVAL_DECILE_SPREAD_MIN

logger = logging.getLogger(__name__)


def compute_slice_metrics(
    returns: pd.Series,
    predictions: Optional[np.ndarray] = None,
    annual_trading_days: int = 252,
    risk_free_rate: float = 0.04,
) -> Dict:
    """Compute standard performance metrics for a return slice.

    Parameters
    ----------
    returns : pd.Series
        Return series for the slice (already filtered).
    predictions : np.ndarray, optional
        Model predictions aligned to returns (for IC calculation).
    annual_trading_days : int
        Trading days per year for annualization.
    risk_free_rate : float
        Annual risk-free rate for Sharpe calculation.

    Returns
    -------
    dict
        Keys: mean_return, annualized_return, sharpe, max_dd, max_dd_duration,
        recovery_time_mean, n_samples, start_date, end_date, win_rate, ic,
        confidence (str: "high", "medium", or "low").
    """
    n = len(returns)
    if n == 0:
        return _empty_metrics()

    ret_arr = returns.values.astype(float)
    ret_arr = ret_arr[np.isfinite(ret_arr)]
    n = len(ret_arr)
    if n == 0:
        return _empty_metrics()

    mean_ret = float(np.mean(ret_arr))
    std_ret = float(np.std(ret_arr, ddof=1)) if n > 1 else 0.0

    # Annualized return (compounded)
    ann_return = float((1 + mean_ret) ** annual_trading_days - 1)

    # Sharpe ratio
    rf_per_period = risk_free_rate / annual_trading_days
    excess = ret_arr - rf_per_period
    sharpe = 0.0
    if std_ret > 1e-12 and n > 1:
        sharpe = float(np.mean(excess) / std_ret * np.sqrt(annual_trading_days))

    # Sharpe standard error (Lo 2002)
    sharpe_se = 0.0
    if n > 2 and std_ret > 1e-12:
        sharpe_se = float(np.sqrt((1 + 0.5 * sharpe ** 2) / n))

    # Maximum drawdown
    cum_eq = np.cumprod(1 + ret_arr)
    running_max = np.maximum.accumulate(cum_eq)
    drawdowns = (cum_eq - running_max) / np.where(running_max > 0, running_max, 1.0)
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Max drawdown duration (bars)
    max_dd_duration = 0
    current_dd_dur = 0
    for d in drawdowns:
        if d < 0:
            current_dd_dur += 1
            max_dd_duration = max(max_dd_duration, current_dd_dur)
        else:
            current_dd_dur = 0

    # Recovery times
    recovery_times = _compute_recovery_times(drawdowns)
    recovery_time_mean = float(np.mean(recovery_times)) if recovery_times else 0.0

    # Win rate
    win_rate = float((ret_arr > 0).sum() / n) if n > 0 else 0.0

    # Information coefficient (Spearman rank correlation)
    ic = 0.0
    if predictions is not None:
        pred_arr = np.asarray(predictions, dtype=float).ravel()
        if len(pred_arr) == n:
            valid = np.isfinite(pred_arr) & np.isfinite(ret_arr)
            if valid.sum() > 2 and sp_stats is not None:
                corr, _ = sp_stats.spearmanr(pred_arr[valid], ret_arr[valid])
                ic = float(corr) if np.isfinite(corr) else 0.0

    # Confidence level based on sample size
    if n >= 100:
        confidence = "high"
    elif n >= EVAL_MIN_SLICE_SAMPLES:
        confidence = "medium"
    else:
        confidence = "low"

    # Date range
    start_date = str(returns.index[0]) if hasattr(returns, 'index') and n > 0 else "N/A"
    end_date = str(returns.index[-1]) if hasattr(returns, 'index') and n > 0 else "N/A"

    return {
        "mean_return": mean_ret,
        "annualized_return": ann_return,
        "sharpe": sharpe,
        "sharpe_se": sharpe_se,
        "max_dd": max_dd,
        "max_dd_duration": max_dd_duration,
        "recovery_time_mean": recovery_time_mean,
        "n_samples": n,
        "start_date": start_date,
        "end_date": end_date,
        "win_rate": win_rate,
        "ic": ic,
        "confidence": confidence,
        "std_return": std_ret,
    }


def decile_spread(
    predictions: np.ndarray,
    returns: pd.Series,
    n_quantiles: int = 10,
    regime_states: Optional[np.ndarray] = None,
) -> Dict:
    """Compute decile spread (top decile return - bottom decile return).

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions (higher = more bullish).
    returns : pd.Series
        Actual forward returns aligned to predictions.
    n_quantiles : int
        Number of quantile bins (default 10 for deciles).
    regime_states : np.ndarray, optional
        If provided, also compute per-regime decile spreads.

    Returns
    -------
    dict
        Keys: spread, spread_t_stat, spread_pvalue, decile_returns (list),
        monotonicity, per_regime (dict, optional).
    """
    pred_arr = np.asarray(predictions, dtype=float).ravel()
    ret_arr = np.asarray(returns, dtype=float).ravel()

    # Align lengths and filter NaN
    min_len = min(len(pred_arr), len(ret_arr))
    pred_arr = pred_arr[:min_len]
    ret_arr = ret_arr[:min_len]
    valid = np.isfinite(pred_arr) & np.isfinite(ret_arr)
    pred_arr = pred_arr[valid]
    ret_arr = ret_arr[valid]
    n = len(pred_arr)

    if n < n_quantiles * 2:
        return _empty_decile_result(n_quantiles)

    # Rank predictions into quantiles
    ranks = pd.Series(pred_arr).rank(method="first", pct=True)
    bins = np.clip((ranks.values * n_quantiles).astype(int), 0, n_quantiles - 1)

    # Compute mean return per quantile
    decile_returns = []
    decile_counts = []
    for q in range(n_quantiles):
        mask = bins == q
        q_rets = ret_arr[mask]
        if len(q_rets) > 0:
            decile_returns.append(float(np.mean(q_rets)))
            decile_counts.append(int(len(q_rets)))
        else:
            decile_returns.append(0.0)
            decile_counts.append(0)

    # Spread = top decile - bottom decile
    spread = decile_returns[-1] - decile_returns[0]

    # T-test on spread significance
    top_mask = bins == (n_quantiles - 1)
    bottom_mask = bins == 0
    top_rets = ret_arr[top_mask]
    bottom_rets = ret_arr[bottom_mask]

    spread_t_stat = 0.0
    spread_pvalue = 1.0
    if len(top_rets) >= 2 and len(bottom_rets) >= 2 and sp_stats is not None:
        t_stat, p_val = sp_stats.ttest_ind(top_rets, bottom_rets, equal_var=False)
        if np.isfinite(t_stat):
            spread_t_stat = float(t_stat)
            spread_pvalue = float(p_val)

    # Monotonicity: Spearman correlation of quantile rank vs mean return
    monotonicity = 0.0
    if sp_stats is not None and n_quantiles > 2:
        q_ranks = np.arange(n_quantiles)
        valid_q = [i for i in range(n_quantiles) if decile_counts[i] > 0]
        if len(valid_q) > 2:
            corr, _ = sp_stats.spearmanr(
                [q_ranks[i] for i in valid_q],
                [decile_returns[i] for i in valid_q],
            )
            monotonicity = float(corr) if np.isfinite(corr) else 0.0

    result = {
        "spread": spread,
        "spread_t_stat": spread_t_stat,
        "spread_pvalue": spread_pvalue,
        "decile_returns": decile_returns,
        "decile_counts": decile_counts,
        "monotonicity": monotonicity,
        "n_total": n,
        "significant": abs(spread_t_stat) > 1.96,
    }

    # Per-regime decile spread
    if regime_states is not None:
        regime_arr = np.asarray(regime_states, dtype=int).ravel()
        if len(regime_arr) >= min_len:
            regime_arr = regime_arr[:min_len]
            regime_arr = regime_arr[valid]

            per_regime = {}
            for reg_code in sorted(set(regime_arr)):
                reg_mask = regime_arr == reg_code
                if reg_mask.sum() < n_quantiles * 2:
                    continue
                reg_result = decile_spread(
                    pred_arr[reg_mask],
                    pd.Series(ret_arr[reg_mask]),
                    n_quantiles=n_quantiles,
                    regime_states=None,  # Don't recurse
                )
                per_regime[int(reg_code)] = reg_result
            result["per_regime"] = per_regime

    return result


def _compute_recovery_times(drawdowns: np.ndarray) -> List[int]:
    """Extract recovery times (bars from trough to recovery) from drawdown array."""
    recovery_times = []
    i = 0
    n = len(drawdowns)

    while i < n:
        if drawdowns[i] < 0:
            # Start of a drawdown episode
            trough_idx = i
            trough_val = drawdowns[i]

            # Find the trough
            j = i + 1
            while j < n and drawdowns[j] < 0:
                if drawdowns[j] < trough_val:
                    trough_val = drawdowns[j]
                    trough_idx = j
                j += 1

            # Recovery time = bars from trough to recovery
            recovery_bars = j - trough_idx
            recovery_times.append(recovery_bars)
            i = j
        else:
            i += 1

    return recovery_times


def _empty_metrics() -> Dict:
    """Return an empty metrics dict for zero-length slices."""
    return {
        "mean_return": 0.0,
        "annualized_return": 0.0,
        "sharpe": 0.0,
        "sharpe_se": 0.0,
        "max_dd": 0.0,
        "max_dd_duration": 0,
        "recovery_time_mean": 0.0,
        "n_samples": 0,
        "start_date": "N/A",
        "end_date": "N/A",
        "win_rate": 0.0,
        "ic": 0.0,
        "confidence": "low",
        "std_return": 0.0,
    }


def _empty_decile_result(n_quantiles: int) -> Dict:
    """Return an empty decile spread result."""
    return {
        "spread": 0.0,
        "spread_t_stat": 0.0,
        "spread_pvalue": 1.0,
        "decile_returns": [0.0] * n_quantiles,
        "decile_counts": [0] * n_quantiles,
        "monotonicity": 0.0,
        "n_total": 0,
        "significant": False,
    }
