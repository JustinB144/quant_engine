"""
Market microstructure diagnostics for Kalshi event markets.

Outputs update frequency, quote staleness distribution, spread dynamics,
and tail liquidity score. Used for market filtering and quality monitoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


_EPS = 1e-12


@dataclass
class MicrostructureDiagnostics:
    """Microstructure summary metrics computed from a quote panel window."""
    update_frequency_per_hour: float
    median_staleness_seconds: float
    p95_staleness_seconds: float
    median_spread: float
    spread_volatility: float
    tail_liquidity_score: float
    active_contract_fraction: float


def compute_microstructure(
    quotes: pd.DataFrame,
    window_hours: float = 4.0,
    asof_ts: Optional[pd.Timestamp] = None,
) -> MicrostructureDiagnostics:
    """
    Compute microstructure diagnostics from a quote panel.

    Args:
        quotes: DataFrame with columns [contract_id, ts, bid, ask, mid, volume].
        window_hours: Lookback window for computing update frequency.
        asof_ts: Reference timestamp (defaults to latest quote time).
    """
    if quotes is None or len(quotes) == 0:
        return MicrostructureDiagnostics(
            update_frequency_per_hour=0.0,
            median_staleness_seconds=np.nan,
            p95_staleness_seconds=np.nan,
            median_spread=np.nan,
            spread_volatility=np.nan,
            tail_liquidity_score=0.0,
            active_contract_fraction=0.0,
        )

    df = quotes.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])

    if asof_ts is None:
        asof_ts = df["ts"].max()
    else:
        asof_ts = pd.Timestamp(asof_ts, tz="UTC")

    window_start = asof_ts - pd.Timedelta(hours=window_hours)
    recent = df[df["ts"] >= window_start]

    # Update frequency
    if len(recent) > 0:
        span_hours = max((recent["ts"].max() - recent["ts"].min()).total_seconds() / 3600.0, _EPS)
        update_frequency = float(len(recent) / span_hours)
    else:
        update_frequency = 0.0

    # Staleness per contract (time since last update as of asof_ts)
    latest_per_contract = df.groupby("contract_id")["ts"].max()
    staleness = (asof_ts - latest_per_contract).dt.total_seconds()
    staleness = staleness[staleness >= 0]

    median_staleness = float(np.median(staleness)) if len(staleness) > 0 else np.nan
    p95_staleness = float(np.percentile(staleness, 95)) if len(staleness) > 0 else np.nan

    # Spread dynamics
    if "bid" in df.columns and "ask" in df.columns:
        bid = pd.to_numeric(df["bid"], errors="coerce")
        ask = pd.to_numeric(df["ask"], errors="coerce")
        mid = (bid + ask) / 2.0
        spread = (ask - bid) / mid.replace(0, np.nan)
        spread = spread[spread.notna() & np.isfinite(spread)]
        median_spread = float(np.median(spread)) if len(spread) > 0 else np.nan
        spread_vol = float(np.std(spread)) if len(spread) > 0 else np.nan
    else:
        median_spread = np.nan
        spread_vol = np.nan

    # Tail liquidity: fraction of contracts with volume > 0
    total_contracts = df["contract_id"].nunique()
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        active = df[vol > 0]["contract_id"].nunique()
    else:
        active = 0

    active_fraction = float(active / max(total_contracts, 1))
    tail_liq_score = float(np.tanh(update_frequency / 10.0) * active_fraction)

    return MicrostructureDiagnostics(
        update_frequency_per_hour=update_frequency,
        median_staleness_seconds=median_staleness,
        p95_staleness_seconds=p95_staleness,
        median_spread=median_spread,
        spread_volatility=spread_vol,
        tail_liquidity_score=float(np.clip(tail_liq_score, 0.0, 1.0)),
        active_contract_fraction=active_fraction,
    )


def microstructure_as_feature_dict(diag: MicrostructureDiagnostics) -> dict:
    """Convert diagnostics to a flat dict for feature panel merging."""
    return {
        "micro_update_freq": diag.update_frequency_per_hour,
        "micro_staleness_median": diag.median_staleness_seconds,
        "micro_staleness_p95": diag.p95_staleness_seconds,
        "micro_spread_median": diag.median_spread,
        "micro_spread_vol": diag.spread_volatility,
        "micro_tail_liquidity": diag.tail_liquidity_score,
        "micro_active_fraction": diag.active_contract_fraction,
    }
