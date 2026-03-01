"""
Contract -> probability distribution builder for Kalshi markets.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .quality import (
    QualityDimensions,
    StalePolicy,
    compute_quality_dimensions,
    dynamic_stale_cutoff_minutes,
    passes_hard_gates,
)

logger = logging.getLogger(__name__)

_EPS = 1e-12

# Canonical empty snapshot template — ensures all return paths produce
# schema-consistent output with the same set of keys.
_EMPTY_SNAPSHOT_TEMPLATE: Dict[str, object] = {
    "mean": np.nan,
    "var": np.nan,
    "skew": np.nan,
    "entropy": np.nan,
    "quality_score": 0.0,
    "coverage_ratio": 0.0,
    "median_spread": np.nan,
    "median_quote_age_seconds": np.nan,
    "volume_oi_proxy": 0.0,
    "constraint_violation_score": 0.0,
    "tail_p_1": np.nan,
    "tail_p_2": np.nan,
    "tail_p_3": np.nan,
    "tail_threshold_1": np.nan,
    "tail_threshold_2": np.nan,
    "tail_threshold_3": np.nan,
    "tail_left_missing": 1,
    "tail_right_missing": 1,
    "mass_missing_estimate": 1.0,
    "moment_truncated": 1,
    "direction_resolved": False,
    "direction_source": "none",
    "direction_confidence": "none",
    "bin_overlap_count": 0,
    "bin_gap_mass_estimate": 0.0,
    "bin_support_is_ordered": 0,
    "isotonic_adjustment_magnitude": 0.0,
    "renormalization_delta": 0.0,
    "violated_constraints_pre": 0,
    "violated_constraints_post": 0,
    "monotonic_violations_pre": 0,
    "monotonic_violation_magnitude": 0.0,
    "renorm_delta": 0.0,
    "isotonic_l1": 0.0,
    "isotonic_l2": 0.0,
    "quality_low": 1,
    "hard_gate_failed": 1,
    "quality_flags": [],
    "_support": [],
    "_mass": [],
}

# T1: Word-boundary regex for threshold direction inference (replaces substring matching)
_ABOVE_PATTERN = re.compile(
    r'\b(>=|above|over|greater\s+than|at\s+least|or\s+higher)\b', re.IGNORECASE
)
_BELOW_PATTERN = re.compile(
    r'\b(<=|below|under|less\s+than|at\s+most|or\s+lower)\b', re.IGNORECASE
)


def _is_tz_aware_datetime(series: pd.Series) -> bool:
    """Return whether tz aware datetime satisfies the expected condition."""
    dtype = getattr(series, "dtype", None)
    return isinstance(dtype, pd.DatetimeTZDtype)


@dataclass
class DistributionConfig:
    """Configuration for Kalshi contract-to-distribution reconstruction and snapshot feature extraction."""
    stale_after_minutes: int = 30
    min_contracts: int = 3
    price_scale: str = "auto"  # "auto", "prob", "cents"

    near_event_minutes: float = 30.0
    near_event_stale_minutes: float = 2.0
    far_event_minutes: float = 24.0 * 60.0
    far_event_stale_minutes: float = 60.0
    stale_market_type_multipliers: Dict[str, float] = field(
        default_factory=lambda: {
            "CPI": 0.80,
            "UNEMPLOYMENT": 0.90,
            "FOMC": 0.70,
            "_default": 1.00,
        },
    )
    stale_liquidity_low_threshold: float = 2.0
    stale_liquidity_high_threshold: float = 6.0
    stale_low_liquidity_multiplier: float = 1.35
    stale_high_liquidity_multiplier: float = 0.80

    tail_thresholds_by_event_type: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "CPI": [3.0, 3.5, 4.0],
            "UNEMPLOYMENT": [4.0, 4.2, 4.5],
            "FOMC": [0.0, 25.0, 50.0],
            "_default": [0.0, 0.5, 1.0],
        },
    )
    distance_lags: List[str] = field(default_factory=lambda: ["1h", "1D"])


def _to_utc_timestamp(value: object) -> pd.Timestamp:
    """Internal helper for to utc timestamp."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _prob_from_mid(mid: float, price_scale: str = "auto") -> float:
    """Internal helper for prob from mid."""
    if not np.isfinite(mid):
        return np.nan
    x = float(mid)
    if price_scale == "cents":
        return np.clip(x / 100.0, 0.0, 1.0)
    if price_scale == "prob":
        return np.clip(x, 0.0, 1.0)
    if x > 1.5:
        return np.clip(x / 100.0, 0.0, 1.0)
    return np.clip(x, 0.0, 1.0)


def _entropy(p: np.ndarray) -> float:
    """Internal helper for entropy."""
    q = np.asarray(p, dtype=float)
    q = q[np.isfinite(q) & (q > 0)]
    if q.size == 0:
        return np.nan
    return float(-(q * np.log(q + _EPS)).sum())


def _isotonic_nonincreasing(y: np.ndarray) -> np.ndarray:
    """
    Pool-adjacent-violators for nonincreasing constraints.
    """
    values = y.astype(float).copy()
    weights = np.ones_like(values)
    i = 0
    while i < len(values) - 1:
        if values[i] < values[i + 1]:
            j = i
            while j >= 0 and values[j] < values[j + 1]:
                total_w = weights[j] + weights[j + 1]
                avg = (weights[j] * values[j] + weights[j + 1] * values[j + 1]) / max(total_w, _EPS)
                values[j] = avg
                values[j + 1] = avg
                weights[j] = total_w
                weights[j + 1] = total_w
                j -= 1
            i = max(j, 0)
        else:
            i += 1
    return np.clip(values, 0.0, 1.0)


def _isotonic_nondecreasing(y: np.ndarray) -> np.ndarray:
    """Internal helper for isotonic nondecreasing."""
    return _isotonic_nonincreasing(y[::-1])[::-1]


@dataclass
class DirectionResult:
    """Result of threshold direction resolution with confidence metadata."""
    direction: Optional[str]  # "ge", "le", or None
    source: str  # "explicit_metadata", "rules_text", or "guess"
    confidence: str  # "high", "medium", or "low"


def _resolve_threshold_direction(row: Mapping[str, object]) -> Optional[str]:
    """
    Resolve threshold contract semantics:
      - "ge": P(X >= threshold)
      - "le": P(X <= threshold)
    """
    result = _resolve_threshold_direction_with_confidence(row)
    return result.direction


def _resolve_threshold_direction_with_confidence(row: Mapping[str, object]) -> DirectionResult:
    """
    Resolve threshold contract semantics with confidence scoring.

    Returns DirectionResult with direction, source, and confidence level.
    When confidence is "low": callers should skip PMF bin conversion and
    compute curve features only; distribution is unsafe for moment extraction.
    """
    # Check explicit metadata first (highest confidence)
    explicit_dir = str(row.get("direction", "")).lower().strip()
    if explicit_dir in ("ge", "gte", ">=", "above", "over"):
        return DirectionResult(direction="ge", source="explicit_metadata", confidence="high")
    if explicit_dir in ("le", "lte", "<=", "below", "under"):
        return DirectionResult(direction="le", source="explicit_metadata", confidence="high")

    payout = str(row.get("payout_structure", "")).lower().strip()
    if payout in ("ge", "gte", ">=", "above", "over"):
        return DirectionResult(direction="ge", source="explicit_metadata", confidence="high")
    if payout in ("le", "lte", "<=", "below", "under"):
        return DirectionResult(direction="le", source="explicit_metadata", confidence="high")

    # Check rules text (medium confidence) — word-boundary regex (T1)
    rules = str(row.get("rules_text", ""))

    if _ABOVE_PATTERN.search(rules):
        return DirectionResult(direction="ge", source="rules_text", confidence="medium")
    if _BELOW_PATTERN.search(rules):
        return DirectionResult(direction="le", source="rules_text", confidence="medium")

    # Guess from title/subtitle (low confidence) — word-boundary regex (T1)
    title_fields = [
        str(row.get("title", "")),
        str(row.get("subtitle", "")),
    ]
    text = " ".join(title_fields)

    if _ABOVE_PATTERN.search(text):
        return DirectionResult(direction="ge", source="guess", confidence="low")
    if _BELOW_PATTERN.search(text):
        return DirectionResult(direction="le", source="guess", confidence="low")

    return DirectionResult(direction=None, source="guess", confidence="low")


@dataclass
class BinValidationResult:
    """Result of bin overlap/gap/ordering validation."""
    bin_overlap_count: int
    bin_gap_mass_estimate: float
    support_is_ordered: bool
    valid: bool


def _validate_bins(contracts: pd.DataFrame) -> BinValidationResult:
    """
    Validate bin structure for non-overlapping, ordered coverage.

    Checks:
      - Bins don't overlap (bin_low[i+1] >= bin_high[i])
      - bin_low < bin_high for each bin
      - Bins cover expected range (or record missing ranges)
    """
    if "bin_low" not in contracts.columns or "bin_high" not in contracts.columns:
        return BinValidationResult(
            bin_overlap_count=0,
            bin_gap_mass_estimate=0.0,
            support_is_ordered=True,
            valid=True,
        )

    df = contracts.copy()
    df = df.assign(
        bl=pd.to_numeric(df["bin_low"], errors="coerce"),
        bh=pd.to_numeric(df["bin_high"], errors="coerce"),
    )
    df = df[df["bl"].notna() & df["bh"].notna()].sort_values("bl")

    if len(df) == 0:
        return BinValidationResult(
            bin_overlap_count=0,
            bin_gap_mass_estimate=0.0,
            support_is_ordered=True,
            valid=True,
        )

    bl = df["bl"].to_numpy(dtype=float)
    bh = df["bh"].to_numpy(dtype=float)

    # Check bin_low < bin_high for each bin
    inverted = int(np.sum(bl >= bh))

    # Check for overlaps: bin_low[i+1] should be >= bin_high[i]
    overlaps = 0
    gap_total = 0.0
    support_range = float(bh[-1] - bl[0]) if len(bl) > 0 else 1.0
    support_range = max(support_range, _EPS)

    for i in range(len(bl) - 1):
        if bl[i + 1] < bh[i]:
            overlaps += 1
        elif bl[i + 1] > bh[i]:
            gap_total += float(bl[i + 1] - bh[i])

    gap_mass_estimate = float(gap_total / support_range) if support_range > 0 else 0.0
    is_ordered = bool(np.all(np.diff(bl) >= 0))

    return BinValidationResult(
        bin_overlap_count=overlaps + inverted,
        bin_gap_mass_estimate=gap_mass_estimate,
        support_is_ordered=is_ordered,
        valid=(overlaps == 0 and inverted == 0 and is_ordered),
    )


def _tail_thresholds(event_type: str, config: DistributionConfig, support: np.ndarray) -> List[float]:
    """Internal helper for tail thresholds."""
    et = str(event_type).upper().strip()
    table = config.tail_thresholds_by_event_type or {}
    if et in table and table[et]:
        return [float(x) for x in table[et][:3]]
    if "_default" in table and table["_default"]:
        return [float(x) for x in table["_default"][:3]]

    s = support[np.isfinite(support)]
    if s.size == 0:
        return [np.nan, np.nan, np.nan]
    q = np.nanpercentile(s, [60, 75, 90])
    return [float(q[0]), float(q[1]), float(q[2])]


def _latest_quotes_asof(
    quotes: pd.DataFrame,
    asof_ts: pd.Timestamp,
    stale_minutes: float,
) -> pd.DataFrame:
    """Internal helper for latest quotes asof."""
    q = quotes.copy(deep=True)
    if "ts" not in q.columns:
        return pd.DataFrame()
    if not _is_tz_aware_datetime(q["ts"]):
        q = q.assign(ts=pd.to_datetime(q["ts"], utc=True, errors="coerce"))
    q = q[q["ts"].notna() & (q["ts"] <= asof_ts)].sort_values(["contract_id", "ts"])
    if q.empty:
        return q
    latest = q.groupby("contract_id", as_index=False).tail(1).copy()
    stale_seconds = max(float(stale_minutes), 0.0) * 60.0
    latest = latest.assign(age_seconds=(asof_ts - latest["ts"]).dt.total_seconds())
    return latest[latest["age_seconds"] <= stale_seconds].copy()


def _normalize_mass(mass: np.ndarray) -> np.ndarray:
    """Internal helper for normalize mass."""
    m = np.asarray(mass, dtype=float)
    m = np.where(np.isfinite(m), np.maximum(m, 0.0), 0.0)
    s = m.sum()
    if s <= _EPS:
        return np.zeros_like(m)
    return m / s


def _moments(support: np.ndarray, mass: np.ndarray) -> Tuple[float, float, float]:
    """Internal helper for moments."""
    x = np.asarray(support, dtype=float)
    p = _normalize_mass(mass)
    valid = np.isfinite(x) & np.isfinite(p)
    if not valid.any():
        return np.nan, np.nan, np.nan
    x = x[valid]
    p = _normalize_mass(p[valid])
    mean = float(np.sum(x * p))
    var = float(np.sum(((x - mean) ** 2) * p))
    if var <= _EPS:
        return mean, var, 0.0
    skew = float(np.sum((((x - mean) / np.sqrt(var)) ** 3) * p))
    return mean, var, skew


def _cdf_from_pmf(support: np.ndarray, mass: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Internal helper for cdf from pmf."""
    s = np.asarray(support, dtype=float)
    p = _normalize_mass(mass)
    order = np.argsort(s)
    s = s[order]
    p = p[order]
    c = np.cumsum(p)
    return s, np.clip(c, 0.0, 1.0)


def _pmf_on_grid(support: np.ndarray, mass: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Internal helper for pmf on grid."""
    s, c = _cdf_from_pmf(support, mass)
    c_interp = np.interp(grid, s, c, left=0.0, right=1.0)
    p_grid = np.diff(np.concatenate([[0.0], c_interp]))
    return _normalize_mass(np.maximum(p_grid, 0.0))


def _distribution_distances(
    support_a: Sequence[float],
    mass_a: Sequence[float],
    support_b: Sequence[float],
    mass_b: Sequence[float],
) -> Tuple[float, float, float]:
    """Internal helper for distribution distances."""
    sa = np.asarray(support_a, dtype=float)
    pa = np.asarray(mass_a, dtype=float)
    sb = np.asarray(support_b, dtype=float)
    pb = np.asarray(mass_b, dtype=float)

    va = np.isfinite(sa) & np.isfinite(pa)
    vb = np.isfinite(sb) & np.isfinite(pb)
    if not va.any() or not vb.any():
        return np.nan, np.nan, np.nan

    sa = sa[va]
    pa = _normalize_mass(pa[va])
    sb = sb[vb]
    pb = _normalize_mass(pb[vb])

    grid = np.array(sorted(set(sa.tolist() + sb.tolist())), dtype=float)
    if grid.size < 2:
        return np.nan, np.nan, np.nan

    p = _pmf_on_grid(sa, pa, grid)
    q = _pmf_on_grid(sb, pb, grid)

    kl = float(np.sum(p * np.log((p + _EPS) / (q + _EPS))))
    m = 0.5 * (p + q)
    js = float(0.5 * np.sum(p * np.log((p + _EPS) / (m + _EPS))) + 0.5 * np.sum(q * np.log((q + _EPS) / (m + _EPS))))

    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    dx = np.diff(grid)
    wass = float(np.sum(np.abs(cdf_p[:-1] - cdf_q[:-1]) * dx))
    return kl, js, wass


def _tail_probs_from_mass(
    support: np.ndarray,
    mass: np.ndarray,
    thresholds: Sequence[float],
) -> List[float]:
    """Internal helper for tail probs from mass."""
    x = np.asarray(support, dtype=float)
    p = _normalize_mass(mass)
    out: List[float] = []
    for t in thresholds:
        if not np.isfinite(t):
            out.append(np.nan)
        else:
            out.append(float(p[x >= float(t)].sum()))
    while len(out) < 3:
        out.append(np.nan)
    return out[:3]


def _tail_probs_from_threshold_curve(
    thresholds_x: np.ndarray,
    prob_curve: np.ndarray,
    direction: str,
    fixed_thresholds: Sequence[float],
) -> List[float]:
    """Internal helper for tail probs from threshold curve."""
    x = np.asarray(thresholds_x, dtype=float)
    p = np.asarray(prob_curve, dtype=float)
    order = np.argsort(x)
    x = x[order]
    p = p[order]
    out: List[float] = []
    for t in fixed_thresholds:
        if not np.isfinite(t):
            out.append(np.nan)
            continue
        p_interp = float(np.interp(float(t), x, p, left=p[0], right=p[-1]))
        if direction == "ge":
            out.append(float(np.clip(p_interp, 0.0, 1.0)))
        else:  # le
            out.append(float(np.clip(1.0 - p_interp, 0.0, 1.0)))
    while len(out) < 3:
        out.append(np.nan)
    return out[:3]


def _estimate_liquidity_proxy(quotes: pd.DataFrame, asof_ts: pd.Timestamp) -> Optional[float]:
    """
    Estimate a stable liquidity proxy from recent quote stream.

    Scale is roughly:
      0-2 very low liquidity,
      2-6 moderate liquidity,
      6+ high liquidity.
    """
    if quotes is None or len(quotes) == 0:
        return None
    q = quotes.copy(deep=True)
    if "ts" not in q.columns:
        return None
    if not _is_tz_aware_datetime(q["ts"]):
        q = q.assign(ts=pd.to_datetime(q["ts"], utc=True, errors="coerce"))
    q = q[q["ts"].notna() & (q["ts"] <= asof_ts)].copy()
    if q.empty:
        return None

    vol = pd.to_numeric(q["volume"], errors="coerce") if "volume" in q.columns else None
    oi = pd.to_numeric(q["oi"], errors="coerce") if "oi" in q.columns else None

    score = 0.0
    n_comp = 0
    if vol is not None and vol.notna().any():
        score += float(np.log1p(np.nanmedian(np.clip(vol.to_numpy(dtype=float), 0.0, np.inf))))
        n_comp += 1
    if oi is not None and oi.notna().any():
        score += float(np.log1p(np.nanmedian(np.clip(oi.to_numpy(dtype=float), 0.0, np.inf))))
        n_comp += 1

    if n_comp == 0:
        return None
    return float(score)


def build_distribution_snapshot(
    contracts: pd.DataFrame,
    quotes: pd.DataFrame,
    asof_ts: pd.Timestamp,
    config: Optional[DistributionConfig] = None,
    event_ts: Optional[pd.Timestamp] = None,
    event_type: str = "",
) -> Dict[str, float]:
    """Build distribution snapshot."""
    cfg = config or DistributionConfig()
    ts = _to_utc_timestamp(asof_ts)
    quotes_parsed = quotes.copy(deep=True)
    if "ts" in quotes_parsed.columns and not _is_tz_aware_datetime(quotes_parsed["ts"]):
        quotes_parsed = quotes_parsed.assign(ts=pd.to_datetime(quotes_parsed["ts"], utc=True, errors="coerce"))

    time_to_event_min = None
    if event_ts is not None:
        et = _to_utc_timestamp(event_ts)
        time_to_event_min = float((et - ts).total_seconds() / 60.0)

    liquidity_proxy = _estimate_liquidity_proxy(quotes_parsed, asof_ts=ts)

    stale_policy = StalePolicy(
        base_stale_minutes=float(cfg.stale_after_minutes),
        near_event_minutes=float(cfg.near_event_minutes),
        near_event_stale_minutes=float(cfg.near_event_stale_minutes),
        far_event_minutes=float(cfg.far_event_minutes),
        far_event_stale_minutes=float(cfg.far_event_stale_minutes),
        market_type_multipliers=dict(cfg.stale_market_type_multipliers),
        liquidity_low_threshold=float(cfg.stale_liquidity_low_threshold),
        liquidity_high_threshold=float(cfg.stale_liquidity_high_threshold),
        low_liquidity_multiplier=float(cfg.stale_low_liquidity_multiplier),
        high_liquidity_multiplier=float(cfg.stale_high_liquidity_multiplier),
    )
    stale_minutes = dynamic_stale_cutoff_minutes(
        time_to_event_minutes=time_to_event_min,
        policy=stale_policy,
        market_type=event_type,
        liquidity_proxy=liquidity_proxy,
    )

    latest = _latest_quotes_asof(
        quotes=quotes_parsed,
        asof_ts=ts,
        stale_minutes=stale_minutes,
    )
    n_total = max(len(contracts), 1)
    if latest.empty:
        result = dict(_EMPTY_SNAPSHOT_TEMPLATE)
        result["renorm_delta"] = 1.0
        return result

    # Validate bins (B2)
    bin_validation = _validate_bins(contracts)

    merged = contracts.merge(
        latest.assign(
            volume=pd.to_numeric(latest.get("volume"), errors="coerce") if "volume" in latest.columns else np.nan,
            oi=pd.to_numeric(latest.get("oi"), errors="coerce") if "oi" in latest.columns else np.nan,
            bid=pd.to_numeric(latest.get("bid"), errors="coerce") if "bid" in latest.columns else np.nan,
            ask=pd.to_numeric(latest.get("ask"), errors="coerce") if "ask" in latest.columns else np.nan,
            mid=pd.to_numeric(latest.get("mid"), errors="coerce") if "mid" in latest.columns else np.nan,
        )[["contract_id", "ts", "age_seconds", "mid", "bid", "ask", "volume", "oi"]],
        on="contract_id",
        how="inner",
    )
    if merged.empty:
        result = dict(_EMPTY_SNAPSHOT_TEMPLATE)
        result["renorm_delta"] = 1.0
        return result

    merged = merged.assign(
        p_raw=pd.to_numeric(merged["mid"], errors="coerce").map(lambda x: _prob_from_mid(x, cfg.price_scale)),
    )
    merged = merged[merged["p_raw"].notna()].copy()
    n_live = len(merged)
    if n_live == 0:
        result = dict(_EMPTY_SNAPSHOT_TEMPLATE)
        result["renorm_delta"] = 1.0
        return result

    spread = np.abs(pd.to_numeric(merged.get("ask"), errors="coerce") - pd.to_numeric(merged.get("bid"), errors="coerce"))
    mid_abs = np.abs(pd.to_numeric(merged.get("mid"), errors="coerce")).replace(0.0, np.nan)
    rel_spread = (spread / mid_abs).replace([np.inf, -np.inf], np.nan)

    monotonic_violations_pre = 0
    monotonic_violation_magnitude = 0.0
    isotonic_l1 = 0.0
    isotonic_l2 = 0.0

    expected_ids = contracts["contract_id"].astype(str).tolist() if "contract_id" in contracts.columns else []
    observed_ids = set(merged["contract_id"].astype(str).tolist())

    support = np.array([], dtype=float)
    mass = np.array([], dtype=float)
    tail_left_missing = 0
    tail_right_missing = 0
    moment_truncated = 0

    has_bins = (
        {"bin_low", "bin_high"}.issubset(set(merged.columns))
        and pd.to_numeric(merged["bin_low"], errors="coerce").notna().any()
        and pd.to_numeric(merged["bin_high"], errors="coerce").notna().any()
    )

    direction_known = True
    if has_bins:
        m = merged.copy()
        p_raw = np.clip(m["p_raw"].to_numpy(dtype=float), 0.0, None)
        renorm_delta = float(abs(p_raw.sum() - 1.0))
        mass = _normalize_mass(p_raw)
        support = (
            pd.to_numeric(m["bin_low"], errors="coerce").to_numpy(dtype=float)
            + pd.to_numeric(m["bin_high"], errors="coerce").to_numpy(dtype=float)
        ) / 2.0

        all_bins = contracts.copy()
        all_bins = all_bins.assign(bin_low_num=pd.to_numeric(all_bins.get("bin_low"), errors="coerce"))
        all_bins = all_bins.sort_values("bin_low_num")
        if len(all_bins) > 0:
            left_id = str(all_bins.iloc[0].get("contract_id", ""))
            right_id = str(all_bins.iloc[-1].get("contract_id", ""))
            tail_left_missing = int(left_id not in observed_ids)
            tail_right_missing = int(right_id not in observed_ids)
        moment_truncated = int(tail_left_missing or tail_right_missing)
    else:
        m = merged.copy()
        m = m.assign(
            threshold_num=pd.to_numeric(m.get("threshold_value"), errors="coerce"),
        )
        m = m[m["threshold_num"].notna()].copy()
        m = m.sort_values("threshold_num")
        renorm_delta = 0.0

        if len(m) == 0:
            support = np.array([], dtype=float)
            mass = np.array([], dtype=float)
            direction_known = False
        else:
            direction = _resolve_threshold_direction(m.iloc[0].to_dict())
            if direction is None:
                direction_known = False
                support = m["threshold_num"].to_numpy(dtype=float)
                mass = _normalize_mass(m["p_raw"].to_numpy(dtype=float))
            else:
                raw_curve = np.clip(m["p_raw"].to_numpy(dtype=float), 0.0, 1.0)
                diffs = np.diff(raw_curve)
                if direction == "ge":
                    violation = np.maximum(diffs, 0.0)
                    monotonic_violations_pre = int(np.sum(violation > 0))
                    monotonic_violation_magnitude = float(np.sum(violation))
                    curve = _isotonic_nonincreasing(raw_curve)
                    mass = np.maximum(0.0, np.concatenate([curve[:-1] - curve[1:], [curve[-1]]]))
                else:  # le
                    violation = np.maximum(-diffs, 0.0)
                    monotonic_violations_pre = int(np.sum(violation > 0))
                    monotonic_violation_magnitude = float(np.sum(violation))
                    curve = _isotonic_nondecreasing(raw_curve)
                    mass = np.maximum(0.0, np.concatenate([[curve[0]], curve[1:] - curve[:-1]]))

                isotonic_l1 = float(np.sum(np.abs(curve - raw_curve)))
                isotonic_l2 = float(np.sqrt(np.sum((curve - raw_curve) ** 2)))
                support = m["threshold_num"].to_numpy(dtype=float)
                mass = _normalize_mass(mass)

                # overwrite p_raw with cleaned curve to support fixed-threshold interpolation.
                m = m.assign(curve=curve)

        all_thr = contracts.copy()
        all_thr = all_thr.assign(threshold_num=pd.to_numeric(all_thr.get("threshold_value"), errors="coerce"))
        all_thr = all_thr[all_thr["threshold_num"].notna()].sort_values("threshold_num")
        if len(all_thr) > 0:
            left_id = str(all_thr.iloc[0].get("contract_id", ""))
            right_id = str(all_thr.iloc[-1].get("contract_id", ""))
            tail_left_missing = int(left_id not in observed_ids)
            tail_right_missing = int(right_id not in observed_ids)
        moment_truncated = int(tail_left_missing or tail_right_missing)

    # T6: Skip moment computation when threshold direction is unresolved
    if not direction_known and not has_bins:
        mean, var, skew = np.nan, np.nan, np.nan
        entropy = np.nan
    else:
        mean, var, skew = _moments(support, mass)
        entropy = _entropy(mass)

    thresholds = _tail_thresholds(event_type=event_type, config=cfg, support=support)
    if has_bins:
        tail_probs = _tail_probs_from_mass(support, mass, thresholds)
    else:
        if direction_known and len(m) > 0 and "curve" in m.columns:
            direction = _resolve_threshold_direction(m.iloc[0].to_dict()) or "ge"
            tail_probs = _tail_probs_from_threshold_curve(
                thresholds_x=m["threshold_num"].to_numpy(dtype=float),
                prob_curve=m["curve"].to_numpy(dtype=float),
                direction=direction,
                fixed_thresholds=thresholds,
            )
        else:
            tail_probs = [np.nan, np.nan, np.nan]

    missing_fraction = float(max(n_total - n_live, 0) / max(n_total, 1))
    mass_missing_estimate = float(np.clip(missing_fraction, 0.0, 1.0))
    if mass_missing_estimate > 0:
        moment_truncated = 1

    violation_for_quality = monotonic_violation_magnitude + renorm_delta
    quality_dims: QualityDimensions = compute_quality_dimensions(
        expected_contracts=n_total,
        observed_contracts=n_live,
        spreads=rel_spread.dropna().to_numpy(dtype=float).tolist(),
        quote_ages_seconds=pd.to_numeric(merged.get("age_seconds"), errors="coerce").dropna().to_numpy(dtype=float).tolist(),
        volumes=pd.to_numeric(merged.get("volume"), errors="coerce").dropna().to_numpy(dtype=float).tolist(),
        open_interests=pd.to_numeric(merged.get("oi"), errors="coerce").dropna().to_numpy(dtype=float).tolist(),
        violation_magnitude=violation_for_quality,
    )

    quality_low = 0
    quality_flags: List[str] = []
    if quality_dims.coverage_ratio < 0.5:
        quality_low = 1
        quality_flags.append("low_coverage")
    if np.isfinite(quality_dims.median_quote_age_seconds) and quality_dims.median_quote_age_seconds > stale_minutes * 60.0:
        quality_low = 1
        quality_flags.append("stale_quotes")
    if n_live < int(cfg.min_contracts):
        quality_low = 1
        quality_flags.append("insufficient_contracts")
    if not direction_known and not has_bins:
        quality_low = 1
        quality_flags.append("unresolved_direction")
    # T8: Enforce bin validation in quality gating
    if not bin_validation.valid:
        quality_low = 1
        quality_flags.append("invalid_bin_structure")

    # T1 (SPEC_39): Wire hard quality gates into snapshot output
    hard_gate_pass = passes_hard_gates(
        quality=quality_dims,
        stale_cutoff_seconds=stale_minutes * 60.0,
    )
    hard_gate_failed = 0
    if not hard_gate_pass:
        mean = np.nan
        var = np.nan
        skew = np.nan
        entropy = np.nan
        quality_low = 1
        hard_gate_failed = 1
        logger.debug(
            "Hard gate failed for snapshot at %s: coverage=%.2f, age=%.0fs, spread=%.3f",
            asof_ts,
            quality_dims.coverage_ratio,
            quality_dims.median_quote_age_seconds,
            quality_dims.median_spread,
        )

    # Resolve direction confidence (B1)
    if not has_bins and len(m) > 0:
        dir_result = _resolve_threshold_direction_with_confidence(m.iloc[0].to_dict())
    else:
        dir_result = DirectionResult(direction=None, source="explicit_metadata" if has_bins else "guess",
                                     confidence="high" if has_bins else "low")

    return {
        "mean": mean,
        "var": var,
        "skew": skew,
        "entropy": entropy,
        "quality_score": float(quality_dims.quality_score),
        "coverage_ratio": float(quality_dims.coverage_ratio),
        "median_spread": float(quality_dims.median_spread),
        "median_quote_age_seconds": float(quality_dims.median_quote_age_seconds),
        "volume_oi_proxy": float(quality_dims.volume_oi_proxy),
        "constraint_violation_score": float(quality_dims.constraint_violation_score),
        "tail_p_1": tail_probs[0],
        "tail_p_2": tail_probs[1],
        "tail_p_3": tail_probs[2],
        "tail_threshold_1": thresholds[0],
        "tail_threshold_2": thresholds[1],
        "tail_threshold_3": thresholds[2],
        "tail_left_missing": int(tail_left_missing),
        "tail_right_missing": int(tail_right_missing),
        "mass_missing_estimate": mass_missing_estimate,
        "moment_truncated": int(moment_truncated),
        # B1: Direction confidence metadata
        "direction_resolved": direction_known or has_bins,  # T6
        "direction_source": dir_result.source,
        "direction_confidence": dir_result.confidence,
        # B2: Bin validity metadata
        "bin_overlap_count": bin_validation.bin_overlap_count,
        "bin_gap_mass_estimate": bin_validation.bin_gap_mass_estimate,
        "bin_support_is_ordered": int(bin_validation.support_is_ordered),
        # B3: Cleaning magnitude as first-class features
        "isotonic_adjustment_magnitude": float(isotonic_l1),
        "renormalization_delta": float(renorm_delta),
        "violated_constraints_pre": int(monotonic_violations_pre),
        "violated_constraints_post": int(
            monotonic_violations_pre if isotonic_l1 < _EPS else 0
        ),
        # Legacy aliases retained for backward compatibility
        "monotonic_violations_pre": int(monotonic_violations_pre),
        "monotonic_violation_magnitude": float(monotonic_violation_magnitude),
        "renorm_delta": float(renorm_delta),
        "isotonic_l1": float(isotonic_l1),
        "isotonic_l2": float(isotonic_l2),
        "quality_low": int(quality_low),
        "hard_gate_failed": int(hard_gate_failed),
        "quality_flags": quality_flags,  # T8
        "_support": support.tolist(),
        "_mass": mass.tolist(),
    }


def _lag_slug(lag: str) -> str:
    """Internal helper for lag slug."""
    return str(lag).replace(" ", "").replace(":", "_")


def _add_distance_features(panel: pd.DataFrame, lags: Sequence[str]) -> pd.DataFrame:
    """Internal helper for add distance features."""
    if panel.empty:
        return panel

    out = panel.sort_values(["market_id", "ts"]).copy()
    out = out.assign(ts=pd.to_datetime(out["ts"], utc=True, errors="coerce"))

    for lag in lags:
        slug = _lag_slug(lag)
        out[f"distance_kl_{slug}"] = np.nan
        out[f"distance_js_{slug}"] = np.nan
        out[f"distance_wasserstein_{slug}"] = np.nan

    for market_id, block in out.groupby("market_id", sort=False):
        idx = block.index.to_numpy()
        times = pd.to_datetime(block["ts"], utc=True, errors="coerce")
        time_index = pd.DatetimeIndex(times)
        supports = block["_support"].tolist()
        masses = block["_mass"].tolist()

        for lag in lags:
            slug = _lag_slug(lag)
            delta = pd.to_timedelta(str(lag))
            for i, ts in enumerate(time_index):
                if pd.isna(ts):
                    continue
                target = pd.Timestamp(ts) - delta
                j = int(time_index.searchsorted(target, side="right") - 1)
                if j < 0:
                    continue
                kl, js, wass = _distribution_distances(
                    support_a=supports[i],
                    mass_a=masses[i],
                    support_b=supports[j],
                    mass_b=masses[j],
                )
                out.loc[idx[i], f"distance_kl_{slug}"] = kl
                out.loc[idx[i], f"distance_js_{slug}"] = js
                out.loc[idx[i], f"distance_wasserstein_{slug}"] = wass

    return out


def build_distribution_panel(
    markets: pd.DataFrame,
    contracts: pd.DataFrame,
    quotes: pd.DataFrame,
    snapshot_times: Iterable[pd.Timestamp],
    config: Optional[DistributionConfig] = None,
) -> pd.DataFrame:
    """
    Build market-level distribution snapshots across times.
    """
    cfg = config or DistributionConfig()
    markets_df = markets.copy()
    contracts_df = contracts.copy()
    quotes_df = quotes.copy()

    if "market_id" not in markets_df.columns:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    market_ids = sorted(set(markets_df["market_id"].astype(str).tolist()))
    for market_id in market_ids:
        market_row = markets_df[markets_df["market_id"].astype(str) == str(market_id)].head(1)
        event_type = str(market_row.iloc[0].get("event_type", "")) if len(market_row) > 0 else ""
        event_ts = None
        if len(market_row) > 0:
            for key in ("settle_ts", "close_ts", "release_ts"):
                raw = market_row.iloc[0].get(key)
                if raw:
                    try:
                        event_ts = _to_utc_timestamp(raw)
                        break
                    except (ValueError, KeyError, TypeError):
                        continue
        spec_version_ts = str(market_row.iloc[0].get("spec_version_ts", "")) if len(market_row) > 0 else ""

        c = contracts_df[contracts_df["market_id"].astype(str) == str(market_id)].copy()
        if c.empty:
            continue
        contract_ids = set(c["contract_id"].astype(str).tolist())
        q = quotes_df[quotes_df["contract_id"].astype(str).isin(contract_ids)].copy()

        for ts in snapshot_times:
            snap_ts = _to_utc_timestamp(ts)
            stats = build_distribution_snapshot(
                contracts=c,
                quotes=q,
                asof_ts=snap_ts,
                config=cfg,
                event_ts=event_ts,
                event_type=event_type,
            )
            rows.append(
                {
                    "market_id": str(market_id),
                    "ts": snap_ts,
                    "spec_version_ts": spec_version_ts,
                    "event_type": event_type,
                    **stats,
                },
            )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    out = _add_distance_features(out, cfg.distance_lags)
    out = out.sort_values(["market_id", "ts"]).reset_index(drop=True)
    return out
