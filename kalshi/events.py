"""
Event-time joins and as-of feature/label builders for Kalshi-driven research.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class EventTimestampMeta:
    """Authoritative event timestamp metadata (D2)."""
    event_source: str = ""  # e.g., "BLS_schedule", "Fed_calendar"
    event_known_at_ts: Optional[str] = None  # When the event date was first known
    last_verified_ts: Optional[str] = None  # When the event timing was last confirmed


@dataclass
class EventFeatureConfig:
    """Configuration for event snapshot horizons and event-panel quality filtering."""
    snapshot_horizons: List[str] = field(
        default_factory=lambda: ["7D", "1D", "4h", "1h", "15min", "5min"],
    )
    min_quality_score: float = 0.0


def _to_utc_ts(series: pd.Series) -> pd.Series:
    """Internal helper for to utc ts."""
    return pd.to_datetime(series, utc=True, errors="coerce")


def _ensure_asof_before_release(df: pd.DataFrame) -> None:
    """Internal helper for ensure asof before release."""
    bad = df[df["asof_ts"] >= df["release_ts"]]
    if len(bad) > 0:
        raise ValueError(
            "Leakage guard failed: found asof_ts >= release_ts rows in event feature generation.",
        )


def asof_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    by: Optional[str] = None,
    left_ts_col: str = "asof_ts",
    right_ts_col: str = "ts",
) -> pd.DataFrame:
    """
    Strict backward as-of join (no forward-peeking).
    """
    l = left.copy(deep=True)
    r = right.copy(deep=True)
    l = l.assign(**{left_ts_col: pd.to_datetime(_to_utc_ts(l[left_ts_col]), utc=True, errors="coerce")})
    r = r.assign(**{right_ts_col: pd.to_datetime(_to_utc_ts(r[right_ts_col]), utc=True, errors="coerce")})
    l = l[l[left_ts_col].notna()].copy().sort_values(left_ts_col).reset_index(drop=True)
    r = r[r[right_ts_col].notna()].copy().sort_values(right_ts_col).reset_index(drop=True)

    kwargs = {
        "left": l,
        "right": r,
        "left_on": left_ts_col,
        "right_on": right_ts_col,
        "direction": "backward",
        "allow_exact_matches": True,
    }
    if by:
        l = l.sort_values([by, left_ts_col]).reset_index(drop=True)
        r = r.sort_values([by, right_ts_col]).reset_index(drop=True)
        kwargs["left"] = l
        kwargs["right"] = r
        kwargs["by"] = by

    return pd.merge_asof(**kwargs)


def build_event_snapshot_grid(
    macro_events: pd.DataFrame,
    horizons: Iterable[str],
) -> pd.DataFrame:
    """Build event snapshot grid."""
    e = macro_events.copy(deep=True)
    if "event_id" not in e.columns or "release_ts" not in e.columns:
        raise ValueError("macro_events must contain event_id and release_ts.")
    e = e.assign(release_ts=pd.to_datetime(_to_utc_ts(e["release_ts"]), utc=True, errors="coerce"))
    e = e[e["release_ts"].notna()].copy()

    rows = []
    for _, row in e.iterrows():
        event_id = str(row["event_id"])
        release_ts = pd.Timestamp(row["release_ts"])
        for h in horizons:
            delta = pd.to_timedelta(str(h))
            asof_ts = release_ts - delta
            rows.append(
                {
                    "event_id": event_id,
                    "event_type": str(row.get("event_type", "")),
                    "release_ts": release_ts,
                    "horizon": str(h),
                    "asof_ts": asof_ts,
                },
            )

    out = pd.DataFrame(rows)
    _ensure_asof_before_release(out)
    return out.sort_values(["event_id", "asof_ts"]).reset_index(drop=True)


def _merge_event_market_map(grid: pd.DataFrame, event_market_map: pd.DataFrame) -> pd.DataFrame:
    """Internal helper for merge event market map."""
    mapping = event_market_map.copy(deep=True)
    if mapping.empty:
        return pd.DataFrame()

    if "market_id" not in mapping.columns:
        raise ValueError("event_market_map must include market_id")
    mapping = mapping.assign(market_id=mapping["market_id"].astype(str))

    out = grid.copy(deep=True)
    out = out.assign(event_id=out["event_id"].astype(str), event_type=out["event_type"].astype(str))

    merged: Optional[pd.DataFrame] = None
    if "event_id" in mapping.columns:
        m_id = mapping.copy()
        m_id = m_id.assign(event_id=m_id["event_id"].astype(str))
        merged = out.merge(m_id, on="event_id", how="left", suffixes=("", "_map"))

    if "event_type" in mapping.columns:
        m_ty = mapping.copy()
        m_ty = m_ty.assign(event_type=m_ty["event_type"].astype(str))
        fallback = out.merge(m_ty, on="event_type", how="left", suffixes=("", "_map"))
        if merged is None:
            merged = fallback
        else:
            if "market_id" in fallback.columns:
                merged.loc[:, "market_id"] = merged["market_id"].fillna(fallback["market_id"])

    if merged is None:
        return pd.DataFrame()

    merged = merged[merged["market_id"].notna()].copy()
    return merged


def _add_revision_speed_features(joined: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Internal helper for add revision speed features."""
    out = joined.sort_values(["event_id", "asof_ts"]).copy(deep=True)
    dt_min = out.groupby("event_id")["asof_ts"].diff().dt.total_seconds() / 60.0
    dt_h = dt_min / 60.0

    for col in feature_cols:
        dcol = out.groupby("event_id")[col].diff()
        out.loc[:, f"delta_{col}"] = dcol
        out.loc[:, f"speed_{col}_per_hour"] = dcol / dt_h.replace(0.0, np.nan)

    # Late drift in final hour: relative to the last snapshot beyond 60m.
    if "mean" in out.columns:
        baseline = (
            out[out["minutes_to_release"] > 60]
            .groupby("event_id")["mean"]
            .last()
            .rename("baseline_mean_gt_60m")
        )
        out = out.join(baseline, on="event_id")
        out.loc[:, "late_drift_mean"] = np.where(
            out["minutes_to_release"] <= 60,
            out["mean"] - out["baseline_mean_gt_60m"],
            np.nan,
        )

    if "var" in out.columns and "minutes_to_release" in out.columns:
        # Collapse feature: ratio to earliest horizon variance.
        earliest_var = out.groupby("event_id")["var"].transform("first")
        out.loc[:, "variance_collapse_ratio"] = out["var"] / earliest_var.replace(0.0, np.nan)

    if "entropy" in out.columns and "minutes_to_release" in out.columns:
        earliest_ent = out.groupby("event_id")["entropy"].transform("first")
        out.loc[:, "entropy_collapse_ratio"] = out["entropy"] / earliest_ent.replace(0.0, np.nan)

    return out


def add_reference_disagreement_features(
    event_panel: pd.DataFrame,
    reference_features: pd.DataFrame,
    ts_col: str = "ts",
) -> pd.DataFrame:
    """
    Optional cross-market disagreement block via strict backward as-of join.
    """
    if event_panel is None or len(event_panel) == 0:
        return event_panel
    if reference_features is None or len(reference_features) == 0:
        return event_panel
    if ts_col not in reference_features.columns:
        raise ValueError(f"reference_features missing {ts_col}")

    panel = event_panel.reset_index() if isinstance(event_panel.index, pd.MultiIndex) else event_panel.copy()
    panel = panel.assign(asof_ts=pd.to_datetime(panel["asof_ts"], utc=True, errors="coerce"))

    ref = reference_features.copy()
    ref = ref.assign(ts=pd.to_datetime(ref[ts_col], utc=True, errors="coerce"))
    ref = ref.drop(columns=[ts_col], errors="ignore")

    joined = asof_join(
        left=panel,
        right=ref,
        by=None,
        left_ts_col="asof_ts",
        right_ts_col="ts",
    )

    if "mean" in joined.columns and "ref_mean" in joined.columns:
        joined.loc[:, "gap_mean"] = joined["mean"] - joined["ref_mean"]
    if "var" in joined.columns and "ref_var" in joined.columns:
        joined.loc[:, "gap_var"] = joined["var"] - joined["ref_var"]
    if "tail_p_1" in joined.columns and "ref_tail" in joined.columns:
        joined.loc[:, "gap_tail"] = joined["tail_p_1"] - joined["ref_tail"]

    if "event_id" in joined.columns and "asof_ts" in joined.columns:
        joined = joined.set_index(["event_id", "asof_ts"]).sort_index()
    return joined


def build_event_feature_panel(
    macro_events: pd.DataFrame,
    event_market_map: pd.DataFrame,
    kalshi_distributions: pd.DataFrame,
    config: Optional[EventFeatureConfig] = None,
) -> pd.DataFrame:
    """
    Build event-centric panel indexed by (event_id, asof_ts).

    Inputs:
        macro_events columns: event_id, event_type, release_ts
        event_market_map columns: {event_id|event_type}, market_id
        kalshi_distributions columns: market_id, ts, mean,var,skew,entropy,quality_score,...
    """
    cfg = config or EventFeatureConfig()

    grid = build_event_snapshot_grid(
        macro_events=macro_events,
        horizons=cfg.snapshot_horizons,
    )

    panel = _merge_event_market_map(grid, event_market_map)
    if len(panel) == 0:
        return pd.DataFrame()

    dist = kalshi_distributions.copy(deep=True)
    dist.loc[:, "market_id"] = dist["market_id"].astype(str)
    if "ts" not in dist.columns:
        raise ValueError("kalshi_distributions must contain ts.")
    dist = dist.assign(ts=pd.to_datetime(_to_utc_ts(dist["ts"]), utc=True, errors="coerce"))
    dist = dist[dist["ts"].notna()].copy()

    joined = asof_join(
        left=panel,
        right=dist,
        by="market_id",
        left_ts_col="asof_ts",
        right_ts_col="ts",
    )
    _ensure_asof_before_release(joined)

    if "quality_score" in joined.columns:
        joined = joined[
            pd.to_numeric(joined["quality_score"], errors="coerce").fillna(0.0) >= cfg.min_quality_score
        ]

    feature_cols = [
        c
        for c in [
            "mean",
            "var",
            "skew",
            "entropy",
            "quality_score",
            "coverage_ratio",
            "median_spread",
            "median_quote_age_seconds",
            "volume_oi_proxy",
            "constraint_violation_score",
            "tail_p_1",
            "tail_p_2",
            "tail_p_3",
            "distance_kl_1h",
            "distance_js_1h",
            "distance_wasserstein_1h",
            "distance_kl_1d",
            "distance_js_1d",
            "distance_wasserstein_1d",
        ]
        if c in joined.columns
    ]
    if not feature_cols:
        return pd.DataFrame()

    joined = joined.sort_values(["event_id", "asof_ts"]).copy(deep=True)
    joined.loc[:, "minutes_to_release"] = (
        (joined["release_ts"] - joined["asof_ts"]).dt.total_seconds() / 60.0
    )
    joined = _add_revision_speed_features(joined, feature_cols=feature_cols)

    out_cols = [
        "event_id",
        "asof_ts",
        "release_ts",
        "event_type",
        "market_id",
        "horizon",
        "minutes_to_release",
        *feature_cols,
        *[c for c in joined.columns if c.startswith("delta_") or c.startswith("speed_")],
        *[c for c in ["late_drift_mean", "variance_collapse_ratio", "entropy_collapse_ratio"] if c in joined.columns],
    ]
    out = joined[out_cols].copy()
    out = out.set_index(["event_id", "asof_ts"]).sort_index()
    return out


def build_asset_time_feature_panel(
    asset_frame: pd.DataFrame,
    kalshi_distributions: pd.DataFrame,
    market_id: str,
    asof_col: str = "ts",
) -> pd.DataFrame:
    """
    Optional continuous panel keyed by (asset_id, ts) with strict as-of joins.
    """
    if asof_col not in asset_frame.columns:
        raise ValueError(f"asset_frame missing required timestamp column: {asof_col}")

    asset = asset_frame.copy()
    asset = asset.assign(**{asof_col: _to_utc_ts(asset[asof_col])})
    asset = asset[asset[asof_col].notna()].sort_values(asof_col)

    dist = kalshi_distributions.copy()
    dist = dist.assign(ts=_to_utc_ts(dist["ts"]), market_id=dist["market_id"].astype(str))
    dist = dist[(dist["market_id"] == str(market_id)) & dist["ts"].notna()].copy()
    dist = dist.sort_values("ts")

    if len(dist) == 0 or len(asset) == 0:
        return pd.DataFrame()

    return pd.merge_asof(
        asset,
        dist,
        left_on=asof_col,
        right_on="ts",
        direction="backward",
        allow_exact_matches=True,
    )


def build_event_labels(
    macro_events: pd.DataFrame,
    event_outcomes_first_print: pd.DataFrame,
    event_outcomes_revised: Optional[pd.DataFrame] = None,
    label_mode: str = "first_print",  # "first_print" or "latest"
) -> pd.DataFrame:
    """
    Build event outcome labels with explicit as-of awareness.
    """
    mode = str(label_mode).lower().strip()
    if mode not in {"first_print", "latest"}:
        raise ValueError("label_mode must be 'first_print' or 'latest'.")

    events = macro_events.copy()
    events = events.assign(release_ts=pd.to_datetime(events["release_ts"], utc=True, errors="coerce"))
    events = events[events["event_id"].notna() & events["release_ts"].notna()].copy()

    first = event_outcomes_first_print.copy() if event_outcomes_first_print is not None else pd.DataFrame()
    first = first.assign(
        asof_ts=pd.to_datetime(first.get("asof_ts"), utc=True, errors="coerce"),
        release_ts=pd.to_datetime(first.get("release_ts", first.get("asof_ts")), utc=True, errors="coerce"),
    )
    first = first[first.get("event_id").notna() & first.get("asof_ts").notna()].copy()

    revised = event_outcomes_revised.copy() if event_outcomes_revised is not None else pd.DataFrame()
    if not revised.empty:
        revised = revised.assign(
            asof_ts=pd.to_datetime(revised.get("asof_ts"), utc=True, errors="coerce"),
            release_ts=pd.to_datetime(revised.get("release_ts", revised.get("asof_ts")), utc=True, errors="coerce"),
        )
        revised = revised[revised.get("event_id").notna() & revised.get("asof_ts").notna()].copy()

    if len(events) == 0 or len(first) == 0:
        return pd.DataFrame(columns=["event_id", "label_value", "label_asof_ts", "label_mode", "label_source"])

    rows = []
    for _, e in events.iterrows():
        eid = str(e["event_id"])
        release_ts = pd.Timestamp(e["release_ts"])

        source_df = first
        source_name = "first_print"
        if mode == "latest" and not revised.empty:
            source_df = revised
            source_name = "revised"

        cand = source_df[(source_df["event_id"].astype(str) == eid) & (source_df["asof_ts"] >= release_ts)].copy()
        if cand.empty and mode == "latest":
            cand = first[(first["event_id"].astype(str) == eid) & (first["asof_ts"] >= release_ts)].copy()
            source_name = "first_print_fallback"
        if cand.empty:
            continue

        cand = cand.sort_values("asof_ts")
        pick = cand.iloc[0] if mode == "first_print" else cand.iloc[-1]
        # F2: Track first-print vs revised values and learned_at timestamp
        first_print_value = np.nan
        revised_print_value = np.nan
        first_cand = first[(first["event_id"].astype(str) == eid) & (first["asof_ts"] >= release_ts)].sort_values("asof_ts")
        if not first_cand.empty:
            first_print_value = pd.to_numeric(first_cand.iloc[0].get("realized_value", np.nan), errors="coerce")
        if not revised.empty:
            rev_cand = revised[(revised["event_id"].astype(str) == eid) & (revised["asof_ts"] >= release_ts)].sort_values("asof_ts")
            if not rev_cand.empty:
                revised_print_value = pd.to_numeric(rev_cand.iloc[-1].get("realized_value", np.nan), errors="coerce")

        rows.append(
            {
                "event_id": eid,
                "label_value": pd.to_numeric(pick.get("realized_value", np.nan), errors="coerce"),
                "label_asof_ts": pd.Timestamp(pick["asof_ts"]),
                "label_mode": mode,
                "label_source": source_name,
                "first_print_value": first_print_value,
                "revised_print_value": revised_print_value,
                "learned_at_ts": pd.Timestamp(pick["asof_ts"]),
            },
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.set_index("event_id").sort_index()


def build_asset_response_labels(
    macro_events: pd.DataFrame,
    asset_prices: pd.DataFrame,
    ts_col: str = "ts",
    price_col: Optional[str] = None,
    window_start: str = "1min",
    window_end: str = "60min",
    entry_delay: Optional[str] = None,
    exit_horizon: Optional[str] = None,
    price_source: str = "close",  # close|mid|last
) -> pd.DataFrame:
    """
    Event-to-asset response labels with realistic execution windows.
    """
    events = macro_events.copy()
    px = asset_prices.copy()

    source_key = str(price_source).lower().strip()
    default_cols = {
        "close": "Close",
        "mid": "Mid",
        "last": "Last",
    }
    px_col = price_col or default_cols.get(source_key, "Close")

    if ts_col not in px.columns or px_col not in px.columns:
        raise ValueError(f"asset_prices must contain {ts_col} and {px_col}.")

    events = events.assign(release_ts=pd.to_datetime(events["release_ts"], utc=True, errors="coerce"))
    px = px.assign(
        **{
            ts_col: pd.to_datetime(px[ts_col], utc=True, errors="coerce"),
            px_col: pd.to_numeric(px[px_col], errors="coerce"),
        },
    )

    events = events[events["release_ts"].notna() & events["event_id"].notna()].copy()
    px = px[px[ts_col].notna() & px[px_col].notna()].sort_values(ts_col).copy()
    if len(events) == 0 or len(px) == 0:
        return pd.DataFrame(columns=["event_id", "asset_return", "entry_ts", "exit_ts"])

    w0 = pd.to_timedelta(entry_delay or window_start)
    w1 = pd.to_timedelta(exit_horizon or window_end)

    rows = []
    for _, e in events.iterrows():
        release = pd.Timestamp(e["release_ts"])
        t_entry = release + w0
        t_exit = release + w1

        entry = px[px[ts_col] >= t_entry]
        exit_ = px[px[ts_col] >= t_exit]
        if entry.empty or exit_.empty:
            continue

        p0 = float(entry.iloc[0][px_col])
        p1 = float(exit_.iloc[0][px_col])
        if p0 <= 0:
            continue

        rows.append(
            {
                "event_id": str(e["event_id"]),
                "asset_return": (p1 / p0) - 1.0,
                "entry_ts": pd.Timestamp(entry.iloc[0][ts_col]),
                "exit_ts": pd.Timestamp(exit_.iloc[0][ts_col]),
                "price_source": source_key,
            },
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.set_index("event_id").sort_index()
