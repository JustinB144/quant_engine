"""
Option surface factor construction from OptionMetrics-enriched daily panels.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _pick_numeric(df: pd.DataFrame, candidates: list[str]) -> Optional[pd.Series]:
    """Internal helper for pick numeric."""
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return None


def _rolling_percentile_rank(
    series: pd.Series,
    window: int = 252,
    min_periods: int = 60,
) -> pd.Series:
    """Internal helper for rolling percentile rank."""
    def _pct_rank(x: np.ndarray) -> float:
        if len(x) == 0:
            return np.nan
        arr = pd.Series(x).dropna().to_numpy(dtype=float)
        if len(arr) == 0:
            return np.nan
        last = arr[-1]
        return float((arr <= last).mean())

    return series.rolling(window=window, min_periods=min_periods).apply(_pct_rank, raw=True)


def compute_option_surface_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute minimal high-signal option surface features.

    Expected input columns (direct or `opt_` prefixed variants):
      iv_atm_30, iv_atm_60, iv_atm_90, iv_put_25d, iv_call_25d
    """
    out = pd.DataFrame(index=df.index).copy(deep=True)

    iv30 = _pick_numeric(df, ["iv_atm_30", "opt_iv_atm_30"])
    iv60 = _pick_numeric(df, ["iv_atm_60", "opt_iv_atm_60"])
    iv90 = _pick_numeric(df, ["iv_atm_90", "opt_iv_atm_90"])
    iv_put_25d = _pick_numeric(df, ["iv_put_25d", "opt_iv_put_25d"])
    iv_call_25d = _pick_numeric(df, ["iv_call_25d", "opt_iv_call_25d"])

    if iv30 is not None:
        out.loc[:, "iv_atm_30"] = iv30
    if iv60 is not None:
        out.loc[:, "iv_atm_60"] = iv60
    if iv90 is not None:
        out.loc[:, "iv_atm_90"] = iv90

    if iv30 is not None and iv90 is not None:
        out.loc[:, "term_slope_30_90"] = iv30 - iv90

    if iv_put_25d is not None and iv_call_25d is not None:
        out.loc[:, "skew_25d"] = iv_put_25d - iv_call_25d
    elif iv_put_25d is not None and iv30 is not None:
        out.loc[:, "skew_25d"] = iv_put_25d - iv30

    if iv_put_25d is not None and iv_call_25d is not None and iv30 is not None:
        out.loc[:, "curvature"] = (iv_put_25d + iv_call_25d) / 2.0 - iv30

    if iv30 is not None and "Close" in df.columns:
        close = pd.to_numeric(df["Close"], errors="coerce")
        realized_vol_30 = close.pct_change().rolling(30, min_periods=20).std() * np.sqrt(252.0)
        out.loc[:, "vrp_30"] = iv30 - realized_vol_30

    if "iv_atm_30" in out.columns:
        out.loc[:, "iv_rank_1y"] = _rolling_percentile_rank(out["iv_atm_30"])
    if "skew_25d" in out.columns:
        out.loc[:, "skew_rank_1y"] = _rolling_percentile_rank(out["skew_25d"])
    if "vrp_30" in out.columns:
        out.loc[:, "vrp_rank_1y"] = _rolling_percentile_rank(out["vrp_30"])

    return out.replace([np.inf, -np.inf], np.nan)


def compute_iv_shock_features(df: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    """
    Event-centric IV shock features (G3).

    Measures IV surface changes around events by computing pre/post deltas:
      delta_atm_iv_pre_post: ATM IV(t+window) - ATM IV(t-window)
      skew_steepening: skew(t+window) - skew(t-window)
      term_structure_kink: change in short_IV / long_IV ratio

    Args:
        df: DataFrame with iv_atm_30, iv_atm_90, skew_25d columns (or opt_ prefixed).
        window: lookback/forward period in rows (default 1 = adjacent bars).
    """
    out = pd.DataFrame(index=df.index)

    iv30 = _pick_numeric(df, ["iv_atm_30", "opt_iv_atm_30"])
    iv90 = _pick_numeric(df, ["iv_atm_90", "opt_iv_atm_90"])
    skew = _pick_numeric(df, ["skew_25d", "opt_skew_25d"])

    if iv30 is not None:
        iv30_pre = iv30.shift(window)
        iv30_post = iv30.shift(-window)
        out["delta_atm_iv_pre_post"] = iv30_post - iv30_pre

    if skew is not None:
        skew_pre = skew.shift(window)
        skew_post = skew.shift(-window)
        out["skew_steepening"] = skew_post - skew_pre

    if iv30 is not None and iv90 is not None:
        ratio = iv30 / iv90.replace(0.0, np.nan)
        ratio_pre = ratio.shift(window)
        ratio_post = ratio.shift(-window)
        out["term_structure_kink"] = ratio_post - ratio_pre

    return out.replace([np.inf, -np.inf], np.nan)
