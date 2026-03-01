"""
Comprehensive quality gate for intraday OHLCV data.

13-point validation system protecting quant trading engines from corrupted data:
- 7 hard rejection checks (bars removed from dataset)
- 3 soft flag checks (bars stay but logged)
- 3 series-level checks (entire dataset validation)

Quality score: 1.0 - (0.6 * hard_rejection_rate + 0.4 * soft_flag_rate)
Quarantine trigger: rejected_bars > 5% OR quality_score < 0.80
"""

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from ..config import DATA_CACHE_DIR, MARKET_CLOSE, MARKET_OPEN
except ImportError:
    # Allow standalone testing
    DATA_CACHE_DIR = Path("data/cache")
    MARKET_OPEN = "09:30"
    MARKET_CLOSE = "16:00"

try:
    import pandas_market_calendars as mcal
    _NYSE_CAL = mcal.get_calendar("NYSE")
except ImportError:
    mcal = None
    _NYSE_CAL = None


logger = logging.getLogger(__name__)


# ============================================================================
# DATACLASSES
# ============================================================================


@dataclass
class CheckResult:
    """Result of a single quality check."""
    check_name: str
    passed: bool
    rejected_count: int = 0
    flagged_count: int = 0
    message: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class IntradayQualityReport:
    """Complete quality assessment for intraday data."""
    ticker: str
    timeframe: str
    source: str
    timestamp: str

    # Input stats
    total_bars_input: int
    date_range: str

    # Check results
    checks: list[CheckResult] = field(default_factory=list)

    # Aggregates
    total_rejected: int = 0
    total_flagged: int = 0
    unique_rejected_bars: int = 0
    unique_flagged_bars: int = 0
    total_bars_output: int = 0
    quality_score: float = 1.0
    quarantine: bool = False
    quarantine_reason: str = ""

    def add_check(self, check: CheckResult) -> None:
        """Add a check result and update aggregates."""
        self.checks.append(check)
        self.total_rejected += check.rejected_count
        self.total_flagged += check.flagged_count

    def compute_quality_score(self, total_bars_output: int) -> None:
        """Compute quality score based on rejection and flag rates.

        Uses deduplicated counts (``unique_rejected_bars`` / ``unique_flagged_bars``)
        for quarantine decisions so bars triggering multiple checks are not
        double-counted.  Per-check breakdowns remain in ``total_rejected`` /
        ``total_flagged`` for diagnostic detail.
        """
        if self.total_bars_input == 0:
            self.quality_score = 0.0
            return

        # Use deduplicated counts for the score and quarantine decisions
        hard_rejection_rate = self.unique_rejected_bars / self.total_bars_input
        soft_flag_rate = self.unique_flagged_bars / self.total_bars_input

        self.quality_score = 1.0 - (0.6 * hard_rejection_rate + 0.4 * soft_flag_rate)
        self.total_bars_output = total_bars_output

        # Determine quarantine status using deduplicated counts
        if hard_rejection_rate > 0.05:
            self.quarantine = True
            self.quarantine_reason = (
                f"Hard rejection rate {hard_rejection_rate:.1%} exceeds 5% threshold"
            )
        elif self.quality_score < 0.80:
            self.quarantine = True
            self.quarantine_reason = (
                f"Quality score {self.quality_score:.2f} below 0.80 threshold"
            )


# ============================================================================
# MARKET CALENDAR HELPERS
# ============================================================================


def _get_trading_days(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DatetimeIndex:
    """Get NYSE trading days in date range using pandas_market_calendars."""
    if _NYSE_CAL is not None:
        try:
            schedule = _NYSE_CAL.schedule(start_date=start_date, end_date=end_date)
            if len(schedule) > 0:
                return mcal.date_range(schedule, frequency="1D").normalize()
            return pd.DatetimeIndex([])
        except Exception as exc:
            logger.warning(
                "Failed to use pandas_market_calendars: %s. Using bdate_range.", exc
            )

    # Fallback to business day range (ignores holidays)
    return pd.bdate_range(start=start_date, end=end_date)


def _is_in_rth(timestamp: pd.Timestamp) -> bool:
    """Check if timestamp is within RTH (09:30-16:00 ET on trading day)."""
    # Check time of day
    ts_time = timestamp.time()
    market_open = datetime.strptime(MARKET_OPEN, "%H:%M").time()
    market_close = datetime.strptime(MARKET_CLOSE, "%H:%M").time()

    if not (market_open <= ts_time < market_close):
        return False

    # Check trading day
    trading_days = _get_trading_days(
        pd.Timestamp(timestamp.date()),
        pd.Timestamp(timestamp.date()),
    )

    return pd.Timestamp(timestamp.date()) in trading_days.normalize()


# ============================================================================
# EXPECTED BAR COUNTS BY TIMEFRAME
# ============================================================================


BARS_PER_DAY = {
    "1m": 390,      # 09:30-16:00 = 6.5 hours = 390 minutes
    "5m": 78,       # 390 / 5
    "15m": 26,      # 390 / 15
    "30m": 13,      # 390 / 30
    "1h": 7,        # 09:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30
    "4h": 2,        # 09:30, 13:30
}


def _get_expected_bar_count(
    trading_days: int,
    timeframe: str,
) -> int:
    """Calculate expected bar count for timeframe."""
    bars_per_day = BARS_PER_DAY.get(timeframe, 0)
    return trading_days * bars_per_day


# ============================================================================
# CHECK 1: OHLC CONSISTENCY (HARD REJECTION)
# ============================================================================


def _check_ohlc_consistency(df: pd.DataFrame) -> Tuple[CheckResult, pd.Index]:
    """
    Check 1: High >= max(Open, Close) AND Low <= min(Open, Close) AND High >= Low.
    Hard rejection check — invalid bars are removed.
    Vectorized for performance on large datasets (10yr 1m = ~1M bars).
    """
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

    # High must be >= max(Open, Close)
    high_ok = h >= pd.concat([o, c], axis=1).max(axis=1)
    # Low must be <= min(Open, Close)
    low_ok = l <= pd.concat([o, c], axis=1).min(axis=1)
    # High must be >= Low
    hl_ok = h >= l

    valid = high_ok & low_ok & hl_ok
    failed_idx = df.index[~valid]

    result = CheckResult(
        check_name="ohlc_consistency",
        passed=len(failed_idx) == 0,
        rejected_count=len(failed_idx),
        message=f"Rejected {len(failed_idx)} bars with invalid OHLC relationships",
    )

    return result, failed_idx


# ============================================================================
# CHECK 2: NON-NEGATIVE VOLUME (HARD REJECTION)
# ============================================================================


def _check_non_negative_volume(df: pd.DataFrame) -> Tuple[CheckResult, pd.Index]:
    """
    Check 2: Volume >= 0.
    Hard rejection check — negative volumes removed.
    """
    failed_idx = df[df["Volume"] < 0].index

    result = CheckResult(
        check_name="non_negative_volume",
        passed=len(failed_idx) == 0,
        rejected_count=len(failed_idx),
        message=f"Rejected {len(failed_idx)} bars with negative volume",
    )

    return result, failed_idx


# ============================================================================
# CHECK 3: NON-NEGATIVE PRICES (HARD REJECTION)
# ============================================================================


def _check_non_negative_prices(df: pd.DataFrame) -> Tuple[CheckResult, pd.Index]:
    """
    Check 3: Open, High, Low, Close all > 0.
    Hard rejection check.
    """
    failed_idx = df[
        (df["Open"] <= 0) |
        (df["High"] <= 0) |
        (df["Low"] <= 0) |
        (df["Close"] <= 0)
    ].index

    result = CheckResult(
        check_name="non_negative_prices",
        passed=len(failed_idx) == 0,
        rejected_count=len(failed_idx),
        message=f"Rejected {len(failed_idx)} bars with zero or negative prices",
    )

    return result, failed_idx


# ============================================================================
# CHECK 4: TIMESTAMP IN RTH (HARD REJECTION)
# ============================================================================


def _check_timestamp_in_rth(df: pd.DataFrame) -> Tuple[CheckResult, pd.Index]:
    """
    Check 4: Bar timestamp must be on NYSE trading day between 09:30-16:00 ET.
    Hard rejection check.
    Vectorized: first filter by time-of-day, then by trading day calendar.
    """
    market_open = datetime.strptime(MARKET_OPEN, "%H:%M").time()
    market_close = datetime.strptime(MARKET_CLOSE, "%H:%M").time()

    idx = df.index

    # Step 1: Vectorized time-of-day check
    times = idx.time
    time_ok = np.array([(market_open <= t < market_close) for t in times])

    # Step 2: Vectorized trading day check — get all valid trading days in range
    if len(idx) > 0:
        trading_days = _get_trading_days(
            pd.Timestamp(idx.min().date()),
            pd.Timestamp(idx.max().date()),
        )
        trading_dates_set = set(trading_days.normalize().date)
        date_ok = np.array([ts.date() in trading_dates_set for ts in idx])
    else:
        date_ok = np.array([], dtype=bool)

    valid = time_ok & date_ok
    failed_idx = df.index[~valid]

    result = CheckResult(
        check_name="timestamp_in_rth",
        passed=len(failed_idx) == 0,
        rejected_count=len(failed_idx),
        message=f"Rejected {len(failed_idx)} bars outside RTH or non-trading days",
    )

    return result, failed_idx


# ============================================================================
# CHECK 5: EXTREME BAR RETURN (SOFT FLAG)
# ============================================================================


def _check_extreme_bar_return(
    df: pd.DataFrame,
    timeframe: str,
) -> Tuple[CheckResult, pd.Index]:
    """
    Check 5: |Close/prev_Close - 1| exceeds threshold. Scale by timeframe.
    Soft flag check — bars stay but are logged.
    """
    thresholds = {
        "1m": 0.15,
        "5m": 0.20,
        "15m": 0.25,
        "30m": 0.30,
        "1h": 0.35,
        "4h": 0.50,
    }
    threshold = thresholds.get(timeframe, 0.25)

    # Calculate returns
    df_copy = df.copy()
    df_copy["prev_close"] = df_copy["Close"].shift(1)
    df_copy["return"] = (df_copy["Close"] / df_copy["prev_close"]) - 1
    df_copy["abs_return"] = df_copy["return"].abs()

    # Flag bars exceeding threshold
    flagged_idx = df_copy[df_copy["abs_return"] > threshold].index.tolist()

    # Remove first bar (no prev_close)
    if len(flagged_idx) > 0 and flagged_idx[0] == df.index[0]:
        flagged_idx = flagged_idx[1:]

    result = CheckResult(
        check_name="extreme_bar_return",
        passed=len(flagged_idx) == 0,
        flagged_count=len(flagged_idx),
        message=(
            f"Flagged {len(flagged_idx)} bars with return > {threshold:.1%} "
            f"(timeframe: {timeframe})"
        ),
        details={"threshold": threshold, "timeframe": timeframe},
    )

    return result, pd.Index(flagged_idx)


# ============================================================================
# CHECK 6: STALE PRICE (SOFT FLAG)
# ============================================================================


def _check_stale_price(df: pd.DataFrame) -> Tuple[CheckResult, pd.Index]:
    """
    Check 6: Close == prev_Close AND High == Low == Open == Close for 3+ consecutive bars.
    Soft flag check — detects frozen/stale data feeds.
    """
    if len(df) < 3:
        return CheckResult(
            check_name="stale_price",
            passed=True,
            message="Dataset too small to check stale prices",
        ), pd.Index([])

    flagged_idx = []

    # Identify flat bars: all four prices identical within the bar
    all_equal = (
        (df["High"] == df["Low"])
        & (df["Open"] == df["Close"])
        & (df["High"] == df["Open"])
    )
    # Also require Close equals previous bar's Close (frozen feed)
    same_as_prev = df["Close"] == df["Close"].shift(1)
    stale = all_equal & same_as_prev

    # Find runs of 3+ consecutive stale bars
    stale_groups = (stale != stale.shift()).cumsum()
    group_sizes = stale.groupby(stale_groups).sum()
    stale_runs = group_sizes[group_sizes >= 3]

    if len(stale_runs) > 0:
        for group_id in stale_runs.index:
            group_mask = stale_groups == group_id
            group_idx = stale[group_mask].index
            # Flag all bars in the stale run (the data is frozen)
            flagged_idx.extend(group_idx.tolist())

    result = CheckResult(
        check_name="stale_price",
        passed=len(flagged_idx) == 0,
        flagged_count=len(flagged_idx),
        message=f"Flagged {len(flagged_idx)} bars in 3+ consecutive stale bar sequences",
    )

    return result, pd.Index(flagged_idx)


# ============================================================================
# CHECK 7: ZERO VOLUME LIQUID (SOFT FLAG)
# ============================================================================


def _check_zero_volume_liquid(df: pd.DataFrame) -> Tuple[CheckResult, pd.Index]:
    """
    Check 7: Volume == 0 during RTH for typically liquid tickers.
    Flag if median daily volume > 1000 shares.
    Soft flag check.  Vectorized for performance on large datasets.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception:
            return CheckResult(
                check_name="zero_volume_liquid",
                passed=True,
                message="Skipped: Could not parse index as datetime",
            ), pd.Index([])

    daily_volumes = df.groupby(df.index.date)["Volume"].sum()
    median_daily_volume = daily_volumes.median()

    # If median daily volume <= 1000, this isn't a liquid name — skip
    if median_daily_volume <= 1000:
        return CheckResult(
            check_name="zero_volume_liquid",
            passed=True,
            message=f"Skipped: Median daily volume {median_daily_volume:.0f} <= 1000",
        ), pd.Index([])

    # Vectorized: find zero-volume bars
    zero_vol = df["Volume"] == 0
    if zero_vol.sum() == 0:
        return CheckResult(
            check_name="zero_volume_liquid",
            passed=True,
            flagged_count=0,
            message=f"No zero-volume bars (median daily volume: {median_daily_volume:.0f})",
            details={"median_daily_volume": float(median_daily_volume)},
        ), pd.Index([])

    # Vectorized RTH time-of-day check (data should already be RTH-filtered
    # by the pipeline, but verify anyway)
    market_open = datetime.strptime(MARKET_OPEN, "%H:%M").time()
    market_close = datetime.strptime(MARKET_CLOSE, "%H:%M").time()
    times = df.index.time
    in_rth = np.array([(market_open <= t < market_close) for t in times])

    flagged_mask = zero_vol.values & in_rth
    flagged_idx = df.index[flagged_mask]

    result = CheckResult(
        check_name="zero_volume_liquid",
        passed=len(flagged_idx) == 0,
        flagged_count=len(flagged_idx),
        message=(
            f"Flagged {len(flagged_idx)} zero-volume bars during RTH "
            f"(median daily volume: {median_daily_volume:.0f})"
        ),
        details={"median_daily_volume": float(median_daily_volume)},
    )

    return result, flagged_idx


# ============================================================================
# CHECK 8: MISSING BAR RATIO (SERIES-LEVEL)
# ============================================================================


def _check_missing_bar_ratio(
    df: pd.DataFrame,
    timeframe: str,
) -> CheckResult:
    """
    Check 8: Expected bars (NYSE calendar + timeframe) vs actual.
    Fail if missing > 5%.
    Series-level check.
    """
    if len(df) == 0:
        return CheckResult(
            check_name="missing_bar_ratio",
            passed=False,
            message="No bars in dataset",
        )

    # Determine date range
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df_idx = pd.to_datetime(df.index)
        except Exception:
            return CheckResult(
                check_name="missing_bar_ratio",
                passed=True,
                message="Skipped: Could not parse index as datetime",
            )
    else:
        df_idx = df.index

    start_date = df_idx[0]
    end_date = df_idx[-1]

    # Get trading days
    trading_days = _get_trading_days(start_date, end_date)
    num_trading_days = len(trading_days)

    # Calculate expected vs actual
    expected = _get_expected_bar_count(num_trading_days, timeframe)
    actual = len(df)
    missing = expected - actual
    missing_ratio = missing / expected if expected > 0 else 0

    passed = missing_ratio <= 0.05

    result = CheckResult(
        check_name="missing_bar_ratio",
        passed=passed,
        message=(
            f"Expected {expected} bars, got {actual} "
            f"({missing_ratio:.1%} missing, threshold: 5%)"
        ),
        details={
            "expected": expected,
            "actual": actual,
            "missing": missing,
            "missing_ratio": float(missing_ratio),
            "trading_days": num_trading_days,
            "timeframe": timeframe,
        },
    )

    return result


# ============================================================================
# CHECK 9: DUPLICATE TIMESTAMPS (SERIES-LEVEL)
# ============================================================================


def _check_duplicate_timestamps(df: pd.DataFrame) -> CheckResult:
    """
    Check 9: No duplicate timestamps after deduplication.
    Series-level check.
    """
    duplicates = df.index.duplicated().sum()
    passed = duplicates == 0

    result = CheckResult(
        check_name="duplicate_timestamps",
        passed=passed,
        message=f"Found {duplicates} duplicate timestamps",
        details={"duplicate_count": duplicates},
    )

    return result


# ============================================================================
# CHECK 10: MONOTONIC INDEX (SERIES-LEVEL)
# ============================================================================


def _check_monotonic_index(df: pd.DataFrame) -> CheckResult:
    """
    Check 10: Timestamps must be strictly increasing after sort.
    Series-level check.
    """
    if len(df) <= 1:
        return CheckResult(
            check_name="monotonic_index",
            passed=True,
            message="Dataset too small to check monotonicity",
        )

    is_monotonic = df.index.is_monotonic_increasing
    passed = is_monotonic

    result = CheckResult(
        check_name="monotonic_index",
        passed=passed,
        message=(
            "Timestamps are strictly increasing"
            if passed
            else "Timestamps are not strictly increasing (out of order)"
        ),
    )

    return result


# ============================================================================
# CHECK 11: OVERNIGHT GAP (SERIES-LEVEL)
# ============================================================================


def _check_overnight_gap(df: pd.DataFrame) -> CheckResult:
    """
    Check 11: First bar of each day's Open within 10% of previous day's last Close.
    Series-level check (informational only, no hard fail).
    """
    if len(df) < 2:
        return CheckResult(
            check_name="overnight_gap",
            passed=True,
            message="Dataset too small to check overnight gaps",
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df_idx = pd.to_datetime(df.index)
        except Exception:
            return CheckResult(
                check_name="overnight_gap",
                passed=True,
                message="Skipped: Could not parse index as datetime",
            )
    else:
        df_idx = df.index

    # Group by date and find first/last of each day
    daily_groups = df.groupby(df_idx.date)

    gap_warnings = []
    prev_close = None
    prev_date = None

    for date, group in daily_groups:
        if prev_close is not None and prev_date is not None:
            # Check if consecutive trading days
            trading_days = _get_trading_days(
                pd.Timestamp(prev_date),
                pd.Timestamp(date),
            )

            is_consecutive = len(trading_days) == 2

            if is_consecutive:
                first_open = group.iloc[0]["Open"]
                gap_pct = abs(first_open / prev_close - 1)

                if gap_pct > 0.10:
                    gap_warnings.append({
                        "prev_date": str(prev_date),
                        "prev_close": float(prev_close),
                        "date": str(date),
                        "first_open": float(first_open),
                        "gap_pct": float(gap_pct),
                    })

        prev_close = group.iloc[-1]["Close"]
        prev_date = date

    result = CheckResult(
        check_name="overnight_gap",
        passed=True,  # Informational, doesn't fail
        message=(
            f"Found {len(gap_warnings)} overnight gaps > 10% "
            f"(expected for splits/news/weekends)"
        ),
        details={"gaps": gap_warnings},
    )

    return result


# ============================================================================
# CHECK 12: VOLUME DISTRIBUTION (SERIES-LEVEL)
# ============================================================================


def _check_volume_distribution(df: pd.DataFrame) -> CheckResult:
    """
    Check 12: Median daily total volume must be > 100 shares.
    Series-level check.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df_idx = pd.to_datetime(df.index)
        except Exception:
            return CheckResult(
                check_name="volume_distribution",
                passed=True,
                message="Skipped: Could not parse index as datetime",
            )
    else:
        df_idx = df.index

    daily_volumes = df.groupby(df_idx.date)["Volume"].sum()
    median_daily = daily_volumes.median()

    passed = median_daily > 100

    result = CheckResult(
        check_name="volume_distribution",
        passed=passed,
        message=(
            f"Median daily volume {median_daily:.0f} shares "
            f"({'OK' if passed else 'below 100 threshold'})"
        ),
        details={"median_daily_volume": float(median_daily)},
    )

    return result


# ============================================================================
# CHECK 13: SPLIT DETECTION (SERIES-LEVEL)
# ============================================================================


def _check_split_detection(df: pd.DataFrame) -> CheckResult:
    """
    Check 13: If bar-to-bar return is within 1% of known split ratio,
    flag as potential unadjusted split.
    Series-level check (informational).
    """
    if len(df) < 2:
        return CheckResult(
            check_name="split_detection",
            passed=True,
            message="Dataset too small to detect splits",
        )

    # Known split ratios and their reverses
    split_ratios = [
        2.0, 0.5,      # 2:1, 1:2
        3.0, 0.333,    # 3:1, 1:3
        4.0, 0.25,     # 4:1, 1:4
        5.0, 0.2,      # 5:1, 1:5
        10.0, 0.1,     # 10:1, 1:10
        1.5, 0.667,    # 3:2, 2:3
        20.0, 0.05,    # 20:1, 1:20
    ]

    prev_close = df["Close"].shift(1)
    ratio = df["Close"] / prev_close
    valid_mask = prev_close.notna()

    split_flags = []

    # Vectorized: check each known ratio
    for split_ratio in split_ratios:
        near_split = (ratio / split_ratio - 1).abs() < 0.01
        hits = df.index[valid_mask & near_split]
        for ts in hits:
            split_flags.append({
                "timestamp": str(ts),
                "prev_close": float(prev_close.loc[ts]),
                "close": float(df.loc[ts, "Close"]),
                "ratio": float(ratio.loc[ts]),
                "suspected_split": f"{split_ratio:.2f}:1",
            })

    result = CheckResult(
        check_name="split_detection",
        passed=True,  # Informational
        message=(
            f"Detected {len(split_flags)} potential unadjusted splits "
            f"(review for data adjustment)"
        ),
        details={"potential_splits": split_flags},
    )

    return result


# ============================================================================
# QUARANTINE FUNCTION
# ============================================================================


def quarantine_ticker(
    ticker: str,
    timeframe: str,
    cache_dir: Path,
    reason: str,
    source_path: Optional[Path] = None,
) -> Path:
    """
    Move bad data to quarantine directory and log the event.

    Quarantined files are moved (not copied) to ``cache_dir/quarantine/``.
    An append-only ``quarantine_log.json`` records every quarantine event
    for audit purposes.

    Args:
        ticker: Ticker symbol.
        timeframe: Timeframe (e.g., ``'1m'``, ``'5m'``).
        cache_dir: Cache directory path (quarantine subdir created inside).
        reason: Human-readable reason for quarantine.
        source_path: Optional path to the data file to move into quarantine.

    Returns:
        Path to the quarantine directory.
    """
    quarantine_dir = Path(cache_dir) / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)

    # Move source file (and sidecars) to quarantine if provided
    moved_files = []
    if source_path is not None and Path(source_path).exists():
        src = Path(source_path)
        dest = quarantine_dir / src.name
        try:
            shutil.move(str(src), str(dest))
            moved_files.append(str(dest))
        except OSError as exc:
            logger.error("Failed to move %s to quarantine: %s", src, exc)

        # Move associated sidecars (.meta.json, .quality.json)
        for suffix in (".meta.json", ".quality.json"):
            sidecar = src.with_suffix(suffix)
            if sidecar.exists():
                try:
                    shutil.move(str(sidecar), str(quarantine_dir / sidecar.name))
                    moved_files.append(str(quarantine_dir / sidecar.name))
                except OSError as exc:
                    logger.error("Failed to move sidecar %s: %s", sidecar, exc)

    # Append to quarantine log (append-only, human-readable JSON-lines)
    log_entry = {
        "timestamp": now.isoformat(),
        "ticker": ticker.upper(),
        "timeframe": timeframe,
        "reason": reason,
        "moved_files": moved_files,
    }

    log_path = quarantine_dir / "quarantine_log.json"
    _append_to_quarantine_log(log_path, log_entry)

    logger.warning("Quarantined %s %s: %s", ticker, timeframe, reason)

    return quarantine_dir


def _append_to_quarantine_log(log_path: Path, entry: dict) -> None:
    """
    Append an entry to the quarantine log.

    The log is stored as a JSON array. On first write, the file is created
    with an initial array. Subsequent writes append to the array.  This is
    kept simple (read-all, append, write-all) because quarantine events are
    rare and the log stays small.
    """
    log_path = Path(log_path)
    entries: List[dict] = []

    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                entries = raw
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read quarantine log, starting fresh: %s", exc)

    entries.append(entry)

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, default=str)
    except OSError as exc:
        logger.error("Failed to write quarantine log: %s", exc)


def list_quarantined(cache_dir: Path) -> List[Dict[str, object]]:
    """
    Return the current quarantine inventory.

    Reads the append-only ``quarantine_log.json`` and returns the list of
    all quarantine events, most recent first.

    Args:
        cache_dir: Cache directory containing the ``quarantine/`` subdirectory.

    Returns:
        List of quarantine log entries (dicts), newest first.
        Empty list if no quarantines have occurred.
    """
    log_path = Path(cache_dir) / "quarantine" / "quarantine_log.json"
    if not log_path.exists():
        return []

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        if isinstance(entries, list):
            return list(reversed(entries))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read quarantine log: %s", exc)

    return []


# ============================================================================
# JSON REPORT I/O
# ============================================================================


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types and pandas Timestamps."""

    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)


def write_quality_report(
    path: Path,
    report: Union[IntradayQualityReport, dict],
) -> None:
    """
    Write quality report to JSON sidecar file.

    Accepts either an ``IntradayQualityReport`` dataclass or a plain dict.
    When a dataclass is provided, its ``CheckResult`` members are serialized
    to dicts automatically.

    Args:
        path: Destination ``.quality.json`` path.
        report: Quality report (dataclass or dict).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(report, dict):
        report_dict = report
    else:
        report_dict = asdict(report)
        # Convert CheckResult dataclasses to dicts
        report_dict["checks"] = [asdict(c) for c in report.checks]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, cls=_NumpyEncoder)

    logger.info("Wrote quality report to %s", path)


def read_quality_report(path: Path) -> dict:
    """
    Read quality report from JSON sidecar file.

    Returns:
        Report as a plain dict (keys match the ``IntradayQualityReport`` fields).
        Callers that need a dataclass can reconstruct via
        ``IntradayQualityReport(**data)`` after popping ``checks``.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ============================================================================
# MAIN VALIDATION FUNCTION
# ============================================================================


def validate_intraday_bars(
    df: pd.DataFrame,
    ticker: str,
    timeframe: str,
    source: str = "unknown",
) -> Tuple[pd.DataFrame, IntradayQualityReport]:
    """
    Run all 13 quality checks on intraday OHLCV data.

    Args:
        df: DataFrame with columns [Open, High, Low, Close, Volume]
            Index should be timestamps.
        ticker: Ticker symbol
        timeframe: Timeframe (e.g., '1m', '5m', '1h')
        source: Data source (e.g., 'alpaca', 'polygon')

    Returns:
        (cleaned_df, quality_report) tuple
        - cleaned_df: DataFrame with hard-rejected bars removed
        - quality_report: IntradayQualityReport with all check results
    """
    logger.info(
        f"Starting validation for {ticker} {timeframe} ({len(df)} bars from {source})"
    )

    # Initialize report
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logger.error(f"Could not parse index as datetime: {e}")
            raise ValueError("DataFrame index must be parseable as datetime")

    date_range = f"{df.index[0].date()} to {df.index[-1].date()}" if len(df) > 0 else "empty"

    report = IntradayQualityReport(
        ticker=ticker,
        timeframe=timeframe,
        source=source,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_bars_input=len(df),
        date_range=date_range,
    )

    # Ensure DataFrame is sorted
    df = df.sort_index()

    # Collect hard rejections
    all_hard_rejected = set()

    # ========== HARD REJECTION CHECKS ==========

    # Check 1: OHLC consistency
    result, failed = _check_ohlc_consistency(df)
    report.add_check(result)
    all_hard_rejected.update(failed)

    # Check 2: Non-negative volume
    result, failed = _check_non_negative_volume(df)
    report.add_check(result)
    all_hard_rejected.update(failed)

    # Check 3: Non-negative prices
    result, failed = _check_non_negative_prices(df)
    report.add_check(result)
    all_hard_rejected.update(failed)

    # Check 4: Timestamp in RTH
    result, failed = _check_timestamp_in_rth(df)
    report.add_check(result)
    all_hard_rejected.update(failed)

    # Remove hard-rejected bars
    df_cleaned = df.drop(all_hard_rejected)
    report.unique_rejected_bars = len(all_hard_rejected)
    logger.info(f"Removed {len(all_hard_rejected)} hard-rejected bars")

    # ========== SOFT FLAG CHECKS ==========

    # Deduplicate soft flags: a bar triggering multiple soft checks is counted once
    all_soft_flagged: set = set()

    # Check 5: Extreme bar return
    result, flagged = _check_extreme_bar_return(df_cleaned, timeframe)
    report.add_check(result)
    all_soft_flagged.update(flagged)

    # Check 6: Stale price
    result, flagged = _check_stale_price(df_cleaned)
    report.add_check(result)
    all_soft_flagged.update(flagged)

    # Check 7: Zero volume liquid
    result, flagged = _check_zero_volume_liquid(df_cleaned)
    report.add_check(result)
    all_soft_flagged.update(flagged)

    # Set deduplicated soft-flag count for quarantine decisions
    report.unique_flagged_bars = len(all_soft_flagged)

    # ========== SERIES-LEVEL CHECKS ==========

    # Check 8: Missing bar ratio
    result = _check_missing_bar_ratio(df_cleaned, timeframe)
    report.add_check(result)

    # Check 9: Duplicate timestamps
    result = _check_duplicate_timestamps(df_cleaned)
    report.add_check(result)

    # Check 10: Monotonic index
    result = _check_monotonic_index(df_cleaned)
    report.add_check(result)

    # Check 11: Overnight gap
    result = _check_overnight_gap(df_cleaned)
    report.add_check(result)

    # Check 12: Volume distribution
    result = _check_volume_distribution(df_cleaned)
    report.add_check(result)

    # Check 13: Split detection
    result = _check_split_detection(df_cleaned)
    report.add_check(result)

    # ========== COMPUTE QUALITY SCORE ==========

    report.compute_quality_score(len(df_cleaned))

    # Log summary
    logger.info(
        f"Validation complete: {len(df_cleaned)}/{len(df)} bars retained, "
        f"quality_score={report.quality_score:.2f}, "
        f"quarantine={report.quarantine}"
    )

    if report.quarantine:
        logger.warning(f"Quarantine triggered: {report.quarantine_reason}")

    return df_cleaned, report
