"""
Quality scoring helpers for Kalshi event-distribution snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np


_EPS = 1e-12


@dataclass
class QualityDimensions:
    """Component-level quality metrics for a Kalshi distribution snapshot."""
    coverage_ratio: float
    median_spread: float
    median_quote_age_seconds: float
    volume_oi_proxy: float
    constraint_violation_score: float
    quality_score: float


@dataclass
class StalePolicy:
    """Parameters controlling dynamic stale-quote cutoff schedules for Kalshi snapshots."""
    base_stale_minutes: float = 30.0
    near_event_minutes: float = 30.0
    near_event_stale_minutes: float = 2.0
    far_event_minutes: float = 24.0 * 60.0
    far_event_stale_minutes: float = 60.0
    market_type_multipliers: Dict[str, float] = field(
        default_factory=lambda: {
            "CPI": 0.80,
            "UNEMPLOYMENT": 0.90,
            "FOMC": 0.70,
            "_default": 1.00,
        },
    )
    liquidity_low_threshold: float = 2.0
    liquidity_high_threshold: float = 6.0
    low_liquidity_multiplier: float = 1.35
    high_liquidity_multiplier: float = 0.80
    min_stale_minutes: float = 0.5
    max_stale_minutes: float = 24.0 * 60.0


def _finite(values: Iterable[float]) -> np.ndarray:
    """Return only finite numeric values from an iterable as a NumPy array."""
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def dynamic_stale_cutoff_minutes(
    time_to_event_minutes: Optional[float],
    policy: Optional[StalePolicy] = None,
    market_type: Optional[str] = None,
    liquidity_proxy: Optional[float] = None,
) -> float:
    """
    Dynamic stale-cutoff schedule:
      - tightens near event release
      - adjusts by market type
      - adjusts by liquidity regime
    """
    cfg = policy or StalePolicy()
    if time_to_event_minutes is None or not np.isfinite(float(time_to_event_minutes)):
        base = float(cfg.base_stale_minutes)
    else:
        t = float(max(time_to_event_minutes, 0.0))
        if t <= cfg.near_event_minutes:
            base = float(cfg.near_event_stale_minutes)
        elif t >= cfg.far_event_minutes:
            base = float(cfg.far_event_stale_minutes)
        else:
            span = max(cfg.far_event_minutes - cfg.near_event_minutes, _EPS)
            w = (t - cfg.near_event_minutes) / span
            base = float(
                cfg.near_event_stale_minutes
                + w * (cfg.far_event_stale_minutes - cfg.near_event_stale_minutes),
            )

    # Market-type adjustment.
    market_mult = float(cfg.market_type_multipliers.get("_default", 1.0))
    if market_type:
        key = str(market_type).upper().strip()
        market_mult = float(cfg.market_type_multipliers.get(key, market_mult))

    # Liquidity adjustment: low-liquidity markets tolerate older quotes.
    liq_mult = 1.0
    if liquidity_proxy is not None and np.isfinite(float(liquidity_proxy)):
        liq = float(liquidity_proxy)
        if liq <= float(cfg.liquidity_low_threshold):
            liq_mult = float(cfg.low_liquidity_multiplier)
        elif liq >= float(cfg.liquidity_high_threshold):
            liq_mult = float(cfg.high_liquidity_multiplier)

    out = base * market_mult * liq_mult
    return float(np.clip(out, cfg.min_stale_minutes, cfg.max_stale_minutes))


def compute_quality_dimensions(
    expected_contracts: int,
    observed_contracts: int,
    spreads: Iterable[float],
    quote_ages_seconds: Iterable[float],
    volumes: Optional[Iterable[float]] = None,
    open_interests: Optional[Iterable[float]] = None,
    violation_magnitude: float = 0.0,
) -> QualityDimensions:
    """
    Multi-dimensional quality model for distribution snapshots.
    """
    expected = max(int(expected_contracts), 1)
    observed = max(int(observed_contracts), 0)
    coverage_ratio = float(np.clip(observed / expected, 0.0, 1.0))

    spread_arr = _finite(spreads)
    median_spread = float(np.median(spread_arr)) if spread_arr.size > 0 else np.nan
    spread_score = 0.5
    if np.isfinite(median_spread):
        spread_score = float(np.clip(1.0 - median_spread, 0.0, 1.0))

    age_arr = _finite(quote_ages_seconds)
    median_quote_age_seconds = float(np.median(age_arr)) if age_arr.size > 0 else np.nan
    age_score = 0.5
    if np.isfinite(median_quote_age_seconds):
        # 0s is best, 5m+ decays close to zero.
        age_score = float(np.exp(-median_quote_age_seconds / 300.0))

    v = _finite(volumes or [])
    oi = _finite(open_interests or [])
    liq_base = 0.0
    if v.size > 0:
        liq_base += float(np.log1p(np.nanmedian(v)))
    if oi.size > 0:
        liq_base += float(np.log1p(np.nanmedian(oi)))
    volume_oi_proxy = float(np.tanh(liq_base / 8.0)) if liq_base > 0 else 0.0

    violation_mag = float(max(0.0, violation_magnitude))
    constraint_violation_score = float(np.exp(-violation_mag))

    # Weighted aggregate with explicit component exposure.
    quality_score = float(
        0.35 * coverage_ratio
        + 0.20 * spread_score
        + 0.20 * age_score
        + 0.10 * volume_oi_proxy
        + 0.15 * constraint_violation_score
    )

    return QualityDimensions(
        coverage_ratio=coverage_ratio,
        median_spread=median_spread,
        median_quote_age_seconds=median_quote_age_seconds,
        volume_oi_proxy=volume_oi_proxy,
        constraint_violation_score=constraint_violation_score,
        quality_score=float(np.clip(quality_score, 0.0, 1.0)),
    )


def passes_hard_gates(
    quality: QualityDimensions,
    stale_cutoff_seconds: float = 1800.0,
    min_coverage: float = 0.8,
    max_median_spread: float = 0.25,
) -> bool:
    """
    Hard validity gates (C1).  Must-pass criteria — failing any gate means
    the distribution output should be NaN features, not low-quality numbers.

    Args:
        quality: Computed quality dimensions for the snapshot.
        stale_cutoff_seconds: Maximum acceptable median quote age.
            Callers should set this tighter near event release.
        min_coverage: Minimum fraction of expected contracts observed.
        max_median_spread: Maximum tolerable median relative spread.

    Returns:
        True if all gates pass, False otherwise.
    """
    if quality.coverage_ratio < min_coverage:
        return False
    if np.isfinite(quality.median_quote_age_seconds) and quality.median_quote_age_seconds > stale_cutoff_seconds:
        return False
    if np.isfinite(quality.median_spread) and quality.median_spread >= max_median_spread:
        return False
    return True


def quality_as_feature_dict(quality: QualityDimensions) -> dict:
    """
    Expose soft quality dimensions as separate learnable feature columns (C2).

    Returns a dict suitable for merging into a feature panel.
    """
    return {
        "coverage_ratio": float(quality.coverage_ratio),
        "spread_median": float(quality.median_spread) if np.isfinite(quality.median_spread) else np.nan,
        "quote_age_median": float(quality.median_quote_age_seconds) if np.isfinite(quality.median_quote_age_seconds) else np.nan,
        "constraint_violation_score": float(quality.constraint_violation_score),
        "volume_proxy": float(quality.volume_oi_proxy),
        "oi_proxy": float(quality.volume_oi_proxy),  # combined proxy until separated upstream
    }
