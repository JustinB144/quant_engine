"""
OptionMetrics-style options reference features for Kalshi event disagreement.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..features.options_factors import compute_option_surface_factors


_EPS = 1e-12


def _to_utc_ts(series: pd.Series) -> pd.Series:
    """Internal helper for to utc ts."""
    return pd.to_datetime(series, utc=True, errors="coerce")


def build_options_reference_panel(
    options_frame: pd.DataFrame,
    ts_col: str = "ts",
) -> pd.DataFrame:
    """
    Build a normalized options reference panel with:
      iv_atm_30/60/90, term_slope_30_90, skew_25d, vrp_30, ranks
    plus disagreement helpers:
      entropy_proxy, tail_proxy, iv_repricing_speed_per_hour
    """
    if options_frame is None or len(options_frame) == 0:
        return pd.DataFrame()

    raw = options_frame.copy(deep=True)
    if ts_col not in raw.columns:
        if isinstance(raw.index, pd.DatetimeIndex):
            raw = raw.reset_index().rename(columns={raw.index.name or "index": ts_col})
        else:
            raise ValueError(f"options_frame must include {ts_col} or have a DatetimeIndex.")

    raw = raw.assign(**{ts_col: _to_utc_ts(raw[ts_col])})
    raw = raw[raw[ts_col].notna()].sort_values(ts_col).copy()
    if raw.empty:
        return pd.DataFrame()

    raw.index = pd.DatetimeIndex(raw.pop(ts_col), name=ts_col)
    factors = compute_option_surface_factors(raw)
    out = factors.copy(deep=True)
    out = out.replace([np.inf, -np.inf], np.nan)

    if "iv_atm_30" in out.columns:
        iv30 = pd.to_numeric(out["iv_atm_30"], errors="coerce")
        var_30 = np.maximum((iv30 ** 2) * (30.0 / 365.0), _EPS)
        out.loc[:, "entropy_proxy"] = 0.5 * np.log(2.0 * np.pi * np.e * var_30)

        ts = pd.Series(out.index, index=out.index)
        dt_h = ts.diff().dt.total_seconds() / 3600.0
        out.loc[:, "iv_repricing_speed_per_hour"] = iv30.diff() / dt_h.replace(0.0, np.nan)

    if "skew_25d" in out.columns:
        skew = pd.to_numeric(out["skew_25d"], errors="coerce")
        # Heuristic tail proxy: sigmoid of 25d skew. Weak signal placeholder —
        # not derived from actual tail density. Use with caution.
        out.loc[:, "heuristic_tail_proxy"] = 1.0 / (1.0 + np.exp(-skew / 0.10))
        out.loc[:, "tail_proxy_is_heuristic"] = True

    out = out.reset_index().rename(columns={"index": ts_col})
    return out


def add_options_disagreement_features(
    event_panel: pd.DataFrame,
    options_reference: pd.DataFrame,
    options_ts_col: str = "ts",
) -> pd.DataFrame:
    """
    Strict backward as-of join of options reference features into event panel.

    Adds:
      entropy_gap = entropy - opt_entropy_proxy
      tail_gap = tail_p_1 - opt_tail_proxy
      repricing_speed_gap = speed_mean_per_hour - opt_iv_repricing_speed_per_hour
    """
    if event_panel is None or len(event_panel) == 0:
        return event_panel
    if options_reference is None or len(options_reference) == 0:
        return event_panel

    left_had_index = isinstance(event_panel.index, pd.MultiIndex)
    panel = event_panel.reset_index() if left_had_index else event_panel.copy(deep=True)
    if "asof_ts" not in panel.columns:
        raise ValueError("event_panel must include asof_ts (column or index level).")

    ref = options_reference.copy(deep=True)
    if options_ts_col not in ref.columns:
        raise ValueError(f"options_reference missing required timestamp column: {options_ts_col}")

    panel = panel.assign(asof_ts=_to_utc_ts(panel["asof_ts"]))
    panel = panel[panel["asof_ts"].notna()].sort_values("asof_ts").reset_index(drop=True)

    ref = ref.assign(_ts=_to_utc_ts(ref[options_ts_col]))
    ref = ref[ref["_ts"].notna()].sort_values("_ts").reset_index(drop=True)
    ref = ref.drop(columns=[options_ts_col], errors="ignore")

    prefix_cols = {c: f"opt_{c}" for c in ref.columns if c != "_ts"}
    ref = ref.rename(columns=prefix_cols)

    joined = pd.merge_asof(
        panel,
        ref,
        left_on="asof_ts",
        right_on="_ts",
        direction="backward",
        allow_exact_matches=True,
    )
    joined = joined.copy(deep=True)

    if "entropy" in joined.columns and "opt_entropy_proxy" in joined.columns:
        joined.loc[:, "entropy_gap"] = (
            pd.to_numeric(joined["entropy"], errors="coerce")
            - pd.to_numeric(joined["opt_entropy_proxy"], errors="coerce")
        )

    if "tail_p_1" in joined.columns and "opt_heuristic_tail_proxy" in joined.columns:
        joined.loc[:, "tail_gap"] = (
            pd.to_numeric(joined["tail_p_1"], errors="coerce")
            - pd.to_numeric(joined["opt_heuristic_tail_proxy"], errors="coerce")
        )

    if "speed_mean_per_hour" in joined.columns and "opt_iv_repricing_speed_per_hour" in joined.columns:
        joined.loc[:, "repricing_speed_gap"] = (
            pd.to_numeric(joined["speed_mean_per_hour"], errors="coerce")
            - pd.to_numeric(joined["opt_iv_repricing_speed_per_hour"], errors="coerce")
        )

    if "var" in joined.columns and "opt_iv_atm_30" in joined.columns:
        iv = pd.to_numeric(joined["opt_iv_atm_30"], errors="coerce")
        implied_var_30 = (iv ** 2) * (30.0 / 365.0)
        joined.loc[:, "uncertainty_gap"] = pd.to_numeric(joined["var"], errors="coerce") - implied_var_30

    joined = joined.drop(columns=["_ts"], errors="ignore")
    if left_had_index and "event_id" in joined.columns and "asof_ts" in joined.columns:
        joined = joined.set_index(["event_id", "asof_ts"]).sort_index()
    return joined
