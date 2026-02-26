"""
Data quality checks for OHLCV time series.

Provides per-bar quality assessment (``assess_ohlcv_quality``) and
aggregate dashboard reporting (``generate_quality_report``,
``flag_degraded_stocks``).

Uses ``pandas_market_calendars`` for accurate trading-day counts that
respect exchange holidays, falling back to ``pd.bdate_range`` when the
library is unavailable.
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import pandas_market_calendars as mcal
    _NYSE_CAL = mcal.get_calendar("NYSE")
except ImportError:
    mcal = None
    _NYSE_CAL = None

from ..config import (
    MAX_MISSING_BAR_FRACTION,
    MAX_ZERO_VOLUME_FRACTION,
    MAX_ABS_DAILY_RETURN,
)


def _expected_trading_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Return expected trading days between *start* and *end* (inclusive).

    Uses NYSE calendar when ``pandas_market_calendars`` is installed,
    otherwise falls back to ``pd.bdate_range`` (ignores holidays).
    """
    if _NYSE_CAL is not None:
        schedule = _NYSE_CAL.schedule(start_date=start, end_date=end)
        return mcal.date_range(schedule, frequency="1D").normalize()
    return pd.bdate_range(start=start, end=end)


@dataclass
class DataQualityReport:
    """Structured result of OHLCV quality checks with metrics and warning tags."""
    passed: bool
    metrics: Dict[str, float]
    warnings: List[str]

    def to_dict(self) -> Dict:
        """Serialize DataQualityReport to a dictionary."""
        return asdict(self)


def assess_ohlcv_quality(
    df: pd.DataFrame,
    max_missing_bar_fraction: float = MAX_MISSING_BAR_FRACTION,
    max_zero_volume_fraction: float = MAX_ZERO_VOLUME_FRACTION,
    max_abs_daily_return: float = MAX_ABS_DAILY_RETURN,
    fail_on_error: bool = False,
) -> DataQualityReport:
    """Assess OHLCV data quality and optionally raise on failure.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.
    max_missing_bar_fraction : float
        Maximum allowed fraction of missing trading-day bars.
    max_zero_volume_fraction : float
        Maximum allowed fraction of zero-volume bars.
    max_abs_daily_return : float
        Maximum allowed absolute single-day return.
    fail_on_error : bool
        When True, raise ``ValueError`` if any quality check fails.
        When False (default), return the report with ``passed=False``.

    Returns
    -------
    DataQualityReport
        Structured quality assessment result.

    Raises
    ------
    ValueError
        If ``fail_on_error=True`` and any quality check fails.
    """
    warnings: List[str] = []
    if df is None or len(df) == 0:
        return DataQualityReport(passed=False, metrics={}, warnings=["empty_dataframe"])

    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    idx = pd.to_datetime(df.index)

    # Expected trading-day bars between first and last timestamps.
    if len(idx) > 1:
        expected = _expected_trading_days(idx.min(), idx.max())
        missing_frac = 1.0 - (len(pd.Index(idx.unique())) / max(1, len(expected)))
    else:
        missing_frac = 0.0

    zero_vol_frac = float((volume <= 0).mean()) if len(volume) > 0 else 1.0
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    max_abs_ret = float(ret.abs().max()) if len(ret) > 0 else 0.0
    dup_idx = int(idx.duplicated().sum())
    non_monotonic = int((np.diff(idx.values.astype("datetime64[ns]").astype(np.int64)) < 0).sum())

    if missing_frac > max_missing_bar_fraction:
        warnings.append(f"missing_bars>{max_missing_bar_fraction:.2f}")
    if zero_vol_frac > max_zero_volume_fraction:
        warnings.append(f"zero_volume>{max_zero_volume_fraction:.2f}")
    if max_abs_ret > max_abs_daily_return:
        warnings.append(f"max_abs_return>{max_abs_daily_return:.2f}")
    if dup_idx > 0:
        warnings.append("duplicate_timestamps")
    if non_monotonic > 0:
        warnings.append("non_monotonic_index")

    metrics = {
        "missing_bar_fraction": float(missing_frac),
        "zero_volume_fraction": float(zero_vol_frac),
        "max_abs_daily_return": float(max_abs_ret),
        "duplicate_timestamps": float(dup_idx),
        "non_monotonic_steps": float(non_monotonic),
    }
    report = DataQualityReport(passed=len(warnings) == 0, metrics=metrics, warnings=warnings)

    if fail_on_error and not report.passed:
        raise ValueError(
            f"OHLCV quality check failed: {'; '.join(report.warnings)}"
        )

    return report


# ---------------------------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------------------------

# Default weights for the composite quality score.  Each weight
# corresponds to a penalty source; higher weight = more impact on score.
_DEFAULT_QUALITY_WEIGHTS: Dict[str, float] = {
    "missing_bar_fraction": 0.35,
    "zero_volume_fraction": 0.25,
    "extreme_return_fraction": 0.20,
    "duplicate_fraction": 0.20,
}

# Threshold below which a stock is considered degraded.
_DEGRADED_THRESHOLD: float = 0.80


def generate_quality_report(
    ohlcv_dict: Dict[str, pd.DataFrame],
    *,
    quality_weights: Optional[Dict[str, float]] = None,
    degraded_threshold: float = _DEGRADED_THRESHOLD,
    extreme_return_pct: float = 0.10,
) -> pd.DataFrame:
    """Return a per-stock quality summary DataFrame.

    Parameters
    ----------
    ohlcv_dict:
        Mapping of ``{ticker: ohlcv_dataframe}``.  Each DataFrame must
        have at least ``Close`` and ``Volume`` columns with a
        datetime-like index.
    quality_weights:
        Optional override for the weights used to compute the composite
        ``quality_score``.  Keys must match the default weight dict.
    degraded_threshold:
        Stocks with ``quality_score < degraded_threshold`` are flagged
        ``degraded = True``.
    extreme_return_pct:
        Absolute daily return threshold that counts as "extreme"
        (default 10 %).

    Returns
    -------
    pd.DataFrame
        Indexed by ticker with columns:
        ``missing_bar_fraction``, ``zero_volume_fraction``,
        ``extreme_return_count``, ``duplicate_count``,
        ``quality_score``, ``degraded``.
    """
    weights = quality_weights or _DEFAULT_QUALITY_WEIGHTS

    rows: List[Dict] = []
    for ticker, df in ohlcv_dict.items():
        if df is None or len(df) == 0:
            rows.append({
                "ticker": ticker,
                "missing_bar_fraction": 1.0,
                "zero_volume_fraction": 1.0,
                "extreme_return_count": 0,
                "duplicate_count": 0,
                "quality_score": 0.0,
                "degraded": True,
            })
            continue

        close = df["Close"].astype(float)
        volume = df["Volume"].astype(float)
        idx = pd.to_datetime(df.index)
        n_rows = len(df)

        # --- Missing bar fraction ---
        if n_rows > 1:
            expected = _expected_trading_days(idx.min(), idx.max())
            missing_frac = 1.0 - (len(pd.Index(idx.unique())) / max(1, len(expected)))
        else:
            missing_frac = 0.0

        # --- Zero volume fraction ---
        zero_vol_frac = float((volume <= 0).mean()) if n_rows > 0 else 1.0

        # --- Extreme return count ---
        ret = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        extreme_count = int((ret.abs() > extreme_return_pct).sum())

        # --- Duplicate count ---
        dup_count = int(idx.duplicated().sum())

        # --- Composite quality score (0 = worst, 1 = best) ---
        # Each component is expressed as a penalty fraction in [0, 1].
        extreme_frac = extreme_count / max(1, n_rows)
        dup_frac = dup_count / max(1, n_rows)

        penalties = {
            "missing_bar_fraction": float(np.clip(missing_frac, 0.0, 1.0)),
            "zero_volume_fraction": float(np.clip(zero_vol_frac, 0.0, 1.0)),
            "extreme_return_fraction": float(np.clip(extreme_frac, 0.0, 1.0)),
            "duplicate_fraction": float(np.clip(dup_frac, 0.0, 1.0)),
        }

        weighted_penalty = sum(
            weights.get(k, 0.0) * v for k, v in penalties.items()
        )
        total_weight = sum(weights.get(k, 0.0) for k in penalties)
        if total_weight > 0:
            weighted_penalty /= total_weight

        quality_score = float(np.clip(1.0 - weighted_penalty, 0.0, 1.0))

        rows.append({
            "ticker": ticker,
            "missing_bar_fraction": float(missing_frac),
            "zero_volume_fraction": float(zero_vol_frac),
            "extreme_return_count": extreme_count,
            "duplicate_count": dup_count,
            "quality_score": quality_score,
            "degraded": quality_score < degraded_threshold,
        })

    report = pd.DataFrame(rows)
    if len(report) > 0:
        report = report.set_index("ticker").sort_values("quality_score")
    return report


def flag_degraded_stocks(
    ohlcv_dict: Dict[str, pd.DataFrame],
    *,
    degraded_threshold: float = _DEGRADED_THRESHOLD,
    extreme_return_pct: float = 0.10,
) -> List[str]:
    """Return a list of tickers whose data quality is below threshold.

    This is a convenience wrapper around :func:`generate_quality_report`
    that returns only the ticker symbols flagged as degraded.

    Parameters
    ----------
    ohlcv_dict:
        Mapping of ``{ticker: ohlcv_dataframe}``.
    degraded_threshold:
        Stocks with ``quality_score < degraded_threshold`` are included.
    extreme_return_pct:
        Absolute daily return threshold for extreme return counting.

    Returns
    -------
    list[str]
        Ticker symbols with degraded data quality, sorted alphabetically.
    """
    report = generate_quality_report(
        ohlcv_dict,
        degraded_threshold=degraded_threshold,
        extreme_return_pct=extreme_return_pct,
    )
    if report.empty:
        return []
    return sorted(report.index[report["degraded"]].tolist())

