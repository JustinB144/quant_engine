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


def compute_iv_shock_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Causal IV change features (G3).

    All features are **backward-looking only** — they compare the current
    value to the value ``window`` bars ago.  No future data is used.

    Features produced (all causal):
      delta_atm_iv_change_{w}d : iv30 - iv30.shift(window)
      delta_atm_iv_velocity_{w}d : acceleration of IV change
      skew_change_{w}d : skew - skew.shift(window)
      term_structure_change_{w}d : change in short/long IV ratio

    Args:
        df: DataFrame with iv_atm_30, iv_atm_90, skew_25d columns (or opt_ prefixed).
        window: lookback period in rows (default 5 = 1 trading week).
    """
    out = pd.DataFrame(index=df.index)
    w = int(window)

    iv30 = _pick_numeric(df, ["iv_atm_30", "opt_iv_atm_30"])
    iv90 = _pick_numeric(df, ["iv_atm_90", "opt_iv_atm_90"])
    skew = _pick_numeric(df, ["skew_25d", "opt_skew_25d"])

    if iv30 is not None:
        # Recent IV change: current minus W bars ago (causal)
        delta_atm_iv_change = iv30 - iv30.shift(w)
        out[f"delta_atm_iv_change_{w}d"] = delta_atm_iv_change

        # IV acceleration: change-of-change (causal)
        delta_atm_iv_velocity = delta_atm_iv_change - delta_atm_iv_change.shift(w)
        out[f"delta_atm_iv_velocity_{w}d"] = delta_atm_iv_velocity

    if skew is not None:
        # Skew change: current minus W bars ago (causal)
        out[f"skew_change_{w}d"] = skew - skew.shift(w)

    if iv30 is not None and iv90 is not None:
        # Term structure ratio change (causal)
        term_ratio = iv30 / iv90.replace(0.0, np.nan)
        out[f"term_structure_change_{w}d"] = term_ratio - term_ratio.shift(w)

    return out.replace([np.inf, -np.inf], np.nan)
