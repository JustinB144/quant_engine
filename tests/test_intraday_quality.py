"""
Comprehensive tests for the 13-point intraday quality gate.

Tests use synthetic DataFrames — no API calls.

Coverage:
    1. Clean data                → all checks pass
    2. OHLC violation (H < L)    → hard reject
    3. Negative volume           → hard reject
    4. Zero price                → hard reject
    5. Outside RTH               → hard reject
    6. Extreme return            → soft flag
    7. Stale prices (5 identical)→ soft flag
    8. Missing 10% of bars       → hard fail on series check
    9. Duplicate timestamps      → hard fail
   10. Known split (2:1 ratio)   → split detection fires
   11. Mixed corruptions         → correct count of each type
"""

import json
import tempfile
from datetime import datetime, time, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_engine.data.intraday_quality import (
    BARS_PER_DAY,
    CheckResult,
    IntradayQualityReport,
    _check_duplicate_timestamps,
    _check_extreme_bar_return,
    _check_missing_bar_ratio,
    _check_monotonic_index,
    _check_non_negative_prices,
    _check_non_negative_volume,
    _check_ohlc_consistency,
    _check_overnight_gap,
    _check_split_detection,
    _check_stale_price,
    _check_timestamp_in_rth,
    _check_volume_distribution,
    _check_zero_volume_liquid,
    list_quarantined,
    quarantine_ticker,
    read_quality_report,
    validate_intraday_bars,
    write_quality_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_clean_1h(n_days: int = 10, base_price: float = 100.0) -> pd.DataFrame:
    """
    Generate clean 1h intraday data for ``n_days`` business days.

    Returns a DataFrame with 7 bars per day (09:30, 10:30, ..., 15:30)
    on business days, with valid OHLC relationships and positive volume.
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    hours = [time(h, 30) for h in range(9, 16)]  # 09:30..15:30

    timestamps = []
    for d in dates:
        for h in hours:
            timestamps.append(pd.Timestamp(d.year, d.month, d.day, h.hour, h.minute))

    n_bars = len(timestamps)
    close = base_price + np.cumsum(rng.normal(0, 0.3, n_bars))
    close = np.maximum(close, 1.0)
    high = close + rng.uniform(0.05, 0.5, n_bars)
    low = close - rng.uniform(0.05, 0.5, n_bars)
    low = np.maximum(low, 0.01)
    opn = close + rng.normal(0, 0.1, n_bars)
    # Ensure OHLC validity: H >= max(O,C), L <= min(O,C)
    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))
    vol = rng.integers(1000, 100000, n_bars).astype(float)

    df = pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=pd.DatetimeIndex(timestamps),
    )
    return df


def _make_clean_1m(n_days: int = 5, base_price: float = 150.0) -> pd.DataFrame:
    """Generate clean 1m intraday data for ``n_days`` business days."""
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2024-06-03", periods=n_days, freq="B")

    timestamps = []
    for d in dates:
        for minute_offset in range(390):  # 09:30 to 15:59
            h = 9 + (30 + minute_offset) // 60
            m = (30 + minute_offset) % 60
            timestamps.append(pd.Timestamp(d.year, d.month, d.day, h, m))

    n_bars = len(timestamps)
    close = base_price + np.cumsum(rng.normal(0, 0.02, n_bars))
    close = np.maximum(close, 1.0)
    high = close + rng.uniform(0.01, 0.1, n_bars)
    low = close - rng.uniform(0.01, 0.1, n_bars)
    low = np.maximum(low, 0.01)
    opn = close + rng.normal(0, 0.02, n_bars)
    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))
    vol = rng.integers(100, 50000, n_bars).astype(float)

    df = pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=pd.DatetimeIndex(timestamps),
    )
    return df


# ===========================================================================
# Test 1: Clean data — all checks pass
# ===========================================================================


class TestCleanData:
    """Clean data should pass all 13 checks with quality_score near 1.0."""

    def test_clean_1h_passes(self):
        df = _make_clean_1h(n_days=20)
        cleaned, report = validate_intraday_bars(df, "TEST", "1h", source="test")

        assert report.total_rejected == 0
        assert report.quarantine is False
        assert report.quality_score > 0.95
        assert len(cleaned) == len(df)

    def test_clean_1m_passes(self):
        df = _make_clean_1m(n_days=5)
        cleaned, report = validate_intraday_bars(df, "TEST", "1m", source="test")

        assert report.total_rejected == 0
        assert report.quarantine is False
        assert report.quality_score > 0.95


# ===========================================================================
# Test 2: OHLC violation — hard reject
# ===========================================================================


class TestOHLCConsistency:
    """High < Low or other OHLC violations must be hard-rejected."""

    def test_high_less_than_low(self):
        df = _make_clean_1h(n_days=5)
        # Inject: High < Low on 3 bars
        bad_indices = df.index[[5, 15, 25]]
        df.loc[bad_indices, "High"] = df.loc[bad_indices, "Low"] - 1.0

        result, failed = _check_ohlc_consistency(df)

        assert not result.passed
        assert result.rejected_count == 3
        assert set(bad_indices).issubset(set(failed))

    def test_high_less_than_close(self):
        df = _make_clean_1h(n_days=5)
        # Inject: High < Close
        idx = df.index[10]
        df.loc[idx, "High"] = df.loc[idx, "Close"] - 5.0

        result, failed = _check_ohlc_consistency(df)

        assert not result.passed
        assert idx in failed

    def test_full_pipeline_rejects(self):
        df = _make_clean_1h(n_days=5)
        n_original = len(df)
        # Inject bad bars
        df.loc[df.index[0], "High"] = df.loc[df.index[0], "Low"] - 1.0
        df.loc[df.index[1], "High"] = df.loc[df.index[1], "Low"] - 1.0

        cleaned, report = validate_intraday_bars(df, "TEST", "1h")

        assert len(cleaned) == n_original - 2
        ohlc_check = next(c for c in report.checks if c.check_name == "ohlc_consistency")
        assert ohlc_check.rejected_count == 2


# ===========================================================================
# Test 3: Negative volume — hard reject
# ===========================================================================


class TestNegativeVolume:
    """Negative volume bars must be hard-rejected."""

    def test_negative_volume_detected(self):
        df = _make_clean_1h(n_days=5)
        df.loc[df.index[3], "Volume"] = -100
        df.loc[df.index[7], "Volume"] = -1

        result, failed = _check_non_negative_volume(df)

        assert not result.passed
        assert result.rejected_count == 2

    def test_zero_volume_allowed(self):
        """Zero volume is valid (only negative is rejected)."""
        df = _make_clean_1h(n_days=5)
        df.loc[df.index[0], "Volume"] = 0

        result, failed = _check_non_negative_volume(df)

        assert result.passed
        assert result.rejected_count == 0


# ===========================================================================
# Test 4: Zero/negative price — hard reject
# ===========================================================================


class TestNonNegativePrices:
    """Zero or negative prices must be hard-rejected."""

    def test_zero_close(self):
        df = _make_clean_1h(n_days=5)
        df.loc[df.index[10], "Close"] = 0.0

        result, failed = _check_non_negative_prices(df)

        assert not result.passed
        assert result.rejected_count >= 1

    def test_negative_open(self):
        df = _make_clean_1h(n_days=5)
        df.loc[df.index[5], "Open"] = -1.5

        result, failed = _check_non_negative_prices(df)

        assert not result.passed
        assert df.index[5] in failed


# ===========================================================================
# Test 5: Outside RTH — hard reject
# ===========================================================================


class TestTimestampInRTH:
    """Bars outside 09:30-16:00 ET on trading days must be rejected."""

    def test_saturday_bars_rejected(self):
        """Saturday timestamps should fail the trading day check."""
        # 2024-01-06 is a Saturday
        timestamps = [
            pd.Timestamp("2024-01-06 10:00"),
            pd.Timestamp("2024-01-06 11:00"),
        ]
        df = pd.DataFrame(
            {"Open": [100, 100], "High": [101, 101], "Low": [99, 99],
             "Close": [100.5, 100.5], "Volume": [1000, 1000]},
            index=pd.DatetimeIndex(timestamps),
        )

        result, failed = _check_timestamp_in_rth(df)

        assert not result.passed
        assert result.rejected_count == 2

    def test_premarket_bars_rejected(self):
        """Bars before 09:30 ET should be rejected."""
        timestamps = [
            pd.Timestamp("2024-01-02 08:00"),  # Pre-market
            pd.Timestamp("2024-01-02 09:00"),  # Pre-market
            pd.Timestamp("2024-01-02 09:30"),  # RTH — should pass
        ]
        df = pd.DataFrame(
            {"Open": [100] * 3, "High": [101] * 3, "Low": [99] * 3,
             "Close": [100.5] * 3, "Volume": [1000] * 3},
            index=pd.DatetimeIndex(timestamps),
        )

        result, failed = _check_timestamp_in_rth(df)

        # First two should be rejected (before 09:30)
        assert result.rejected_count == 2
        assert timestamps[2] not in failed

    def test_afterhours_rejected(self):
        """Bars at or after 16:00 ET should be rejected."""
        timestamps = [
            pd.Timestamp("2024-01-02 15:59"),  # RTH — pass
            pd.Timestamp("2024-01-02 16:00"),  # After hours — fail
            pd.Timestamp("2024-01-02 17:30"),  # After hours — fail
        ]
        df = pd.DataFrame(
            {"Open": [100] * 3, "High": [101] * 3, "Low": [99] * 3,
             "Close": [100.5] * 3, "Volume": [1000] * 3},
            index=pd.DatetimeIndex(timestamps),
        )

        result, failed = _check_timestamp_in_rth(df)

        assert timestamps[0] not in failed  # 15:59 should pass
        assert timestamps[1] in failed      # 16:00 should fail
        assert timestamps[2] in failed      # 17:30 should fail


# ===========================================================================
# Test 6: Extreme return — soft flag
# ===========================================================================


class TestExtremeReturn:
    """Large bar-to-bar returns should be flagged (not rejected)."""

    def test_50_pct_in_1m_flagged(self):
        """50% return in 1 minute should be flagged (threshold: 15%)."""
        df = _make_clean_1m(n_days=2)
        # Inject a 50% jump
        idx = df.index[100]
        prev_idx = df.index[99]
        df.loc[idx, "Close"] = df.loc[prev_idx, "Close"] * 1.50
        df.loc[idx, "High"] = max(df.loc[idx, "High"], df.loc[idx, "Close"])

        result, flagged = _check_extreme_bar_return(df, "1m")

        assert not result.passed
        assert result.flagged_count >= 1

    def test_small_return_not_flagged(self):
        """Returns under threshold should not be flagged."""
        df = _make_clean_1h(n_days=10)

        result, flagged = _check_extreme_bar_return(df, "1h")

        # Clean data has normal returns — should be zero or very few flags
        assert result.flagged_count == 0

    def test_threshold_scales_by_timeframe(self):
        """4h should have a 50% threshold (more tolerant than 1m's 15%)."""
        df = _make_clean_1h(n_days=5)
        # Inject 20% return — should flag at 1m (15%) but not at 4h (50%)
        idx = df.index[15]
        prev_idx = df.index[14]
        df.loc[idx, "Close"] = df.loc[prev_idx, "Close"] * 1.20
        df.loc[idx, "High"] = max(df.loc[idx, "High"], df.loc[idx, "Close"])

        result_1m, _ = _check_extreme_bar_return(df, "1m")
        result_4h, _ = _check_extreme_bar_return(df, "4h")

        assert result_1m.flagged_count >= 1  # 20% > 15% threshold
        assert result_4h.flagged_count == 0  # 20% < 50% threshold


# ===========================================================================
# Test 7: Stale prices — soft flag
# ===========================================================================


class TestStalePrice:
    """3+ consecutive identical bars (frozen feed) should be flagged."""

    def test_five_identical_bars_flagged(self):
        df = _make_clean_1h(n_days=5)
        # Make 5 consecutive bars identical (stale)
        flat_price = 100.0
        for i in range(10, 15):
            df.loc[df.index[i], "Open"] = flat_price
            df.loc[df.index[i], "High"] = flat_price
            df.loc[df.index[i], "Low"] = flat_price
            df.loc[df.index[i], "Close"] = flat_price

        result, flagged = _check_stale_price(df)

        # Should detect stale bars (at least 3 consecutive meet the criteria)
        # First bar won't match prev_close, so the stale run starts at index[11]
        assert not result.passed
        assert result.flagged_count >= 3

    def test_two_identical_not_flagged(self):
        """Only 2 consecutive flat bars — should NOT flag (need 3+)."""
        df = _make_clean_1h(n_days=5)
        flat_price = 100.0
        for i in range(10, 12):
            df.loc[df.index[i], "Open"] = flat_price
            df.loc[df.index[i], "High"] = flat_price
            df.loc[df.index[i], "Low"] = flat_price
            df.loc[df.index[i], "Close"] = flat_price

        result, flagged = _check_stale_price(df)

        # At most 1 bar could meet Close==prev_Close, so run <3
        assert result.flagged_count < 3

    def test_normal_data_not_stale(self):
        """Normal data should not trigger stale detection."""
        df = _make_clean_1h(n_days=10)

        result, flagged = _check_stale_price(df)

        assert result.flagged_count == 0


# ===========================================================================
# Test 8: Missing bars — series-level hard fail
# ===========================================================================


class TestMissingBarRatio:
    """Missing > 5% of expected bars should fail."""

    def test_10_pct_missing_fails(self):
        """Remove 10% of bars spread across the dataset — should fail 5% threshold."""
        df = _make_clean_1h(n_days=20)
        # Remove bars from the MIDDLE of the dataset so the date range
        # (first to last bar) stays the same, but actual count drops
        rng = np.random.default_rng(123)
        n_bars = len(df)
        n_to_drop = int(n_bars * 0.10)
        # Drop from indices 2..n-2 to preserve first and last timestamps
        drop_candidates = list(range(2, n_bars - 2))
        drop_positions = sorted(rng.choice(drop_candidates, size=n_to_drop, replace=False))
        drop_indices = df.index[drop_positions]
        df_sparse = df.drop(drop_indices)

        result = _check_missing_bar_ratio(df_sparse, "1h")

        # With 10% bars removed but same date span, missing_ratio > 5%
        assert not result.passed
        assert result.details.get("missing_ratio", 0) > 0.05

    def test_full_data_passes(self):
        """Complete data should pass."""
        df = _make_clean_1h(n_days=20)

        result = _check_missing_bar_ratio(df, "1h")

        # May still report some missing due to calendar differences,
        # but should pass the 5% threshold
        assert result.passed or result.details.get("missing_ratio", 0) <= 0.05


# ===========================================================================
# Test 9: Duplicate timestamps — series-level hard fail
# ===========================================================================


class TestDuplicateTimestamps:
    """Duplicate timestamps should be detected."""

    def test_duplicates_detected(self):
        df = _make_clean_1h(n_days=5)
        # Append duplicate of first few bars
        dups = df.iloc[:3].copy()
        df_with_dups = pd.concat([df, dups])

        result = _check_duplicate_timestamps(df_with_dups)

        assert not result.passed
        assert result.details["duplicate_count"] == 3

    def test_no_duplicates_passes(self):
        df = _make_clean_1h(n_days=5)

        result = _check_duplicate_timestamps(df)

        assert result.passed


# ===========================================================================
# Test 10: Split detection — informational flag
# ===========================================================================


class TestSplitDetection:
    """Exact 2:1 or 3:1 price ratios should trigger split detection."""

    def test_2to1_split_detected(self):
        df = _make_clean_1h(n_days=5)
        # Inject exact 2:1 split
        split_idx = df.index[20]
        prev_idx = df.index[19]
        df.loc[split_idx, "Close"] = df.loc[prev_idx, "Close"] * 2.0
        df.loc[split_idx, "Open"] = df.loc[prev_idx, "Close"] * 2.0
        df.loc[split_idx, "High"] = df.loc[split_idx, "Close"] + 0.5
        df.loc[split_idx, "Low"] = df.loc[split_idx, "Close"] - 0.5

        result = _check_split_detection(df)

        assert len(result.details.get("potential_splits", [])) >= 1
        # Check that the 2:1 ratio was identified
        found_2to1 = any(
            "2.00" in s.get("suspected_split", "")
            for s in result.details["potential_splits"]
        )
        assert found_2to1

    def test_reverse_split_detected(self):
        """1:2 reverse split (price halves) should also be detected."""
        df = _make_clean_1h(n_days=5)
        split_idx = df.index[20]
        prev_idx = df.index[19]
        df.loc[split_idx, "Close"] = df.loc[prev_idx, "Close"] * 0.5
        df.loc[split_idx, "Open"] = df.loc[split_idx, "Close"]
        df.loc[split_idx, "Low"] = df.loc[split_idx, "Close"] - 0.5
        df.loc[split_idx, "High"] = df.loc[split_idx, "Close"] + 0.5

        result = _check_split_detection(df)

        assert len(result.details.get("potential_splits", [])) >= 1

    def test_no_split_in_normal_data(self):
        """Normal data should not trigger split detection."""
        df = _make_clean_1h(n_days=10)

        result = _check_split_detection(df)

        assert len(result.details.get("potential_splits", [])) == 0


# ===========================================================================
# Test 11: Mixed corruptions — correct counts
# ===========================================================================


class TestMixedCorruptions:
    """Multiple corruption types should be counted independently."""

    def test_mixed_hard_rejections(self):
        df = _make_clean_1h(n_days=10)
        n_original = len(df)

        # 2 OHLC violations
        df.loc[df.index[5], "High"] = df.loc[df.index[5], "Low"] - 1.0
        df.loc[df.index[6], "High"] = df.loc[df.index[6], "Low"] - 1.0

        # 1 negative volume
        df.loc[df.index[10], "Volume"] = -50

        # 1 zero price
        df.loc[df.index[15], "Close"] = 0.0

        cleaned, report = validate_intraday_bars(df, "MIX", "1h")

        # At least 4 bars rejected (may be more if overlap)
        assert report.total_rejected >= 4
        assert len(cleaned) <= n_original - 4

        # Check each check counted correctly
        ohlc = next(c for c in report.checks if c.check_name == "ohlc_consistency")
        vol = next(c for c in report.checks if c.check_name == "non_negative_volume")
        price = next(c for c in report.checks if c.check_name == "non_negative_prices")

        assert ohlc.rejected_count >= 2
        assert vol.rejected_count >= 1
        assert price.rejected_count >= 1

    def test_quarantine_on_high_rejection(self):
        """More than 5% rejected bars should trigger quarantine."""
        df = _make_clean_1h(n_days=10)
        n_bars = len(df)
        # Corrupt >5% of bars with negative volume
        n_bad = int(n_bars * 0.07)  # 7% corruption
        for i in range(n_bad):
            df.loc[df.index[i], "Volume"] = -1

        cleaned, report = validate_intraday_bars(df, "BAD", "1h")

        assert report.quarantine is True
        assert "5%" in report.quarantine_reason or "threshold" in report.quarantine_reason


# ===========================================================================
# Test: Monotonic index
# ===========================================================================


class TestMonotonicIndex:
    """Timestamps must be strictly increasing."""

    def test_non_monotonic_fails(self):
        df = _make_clean_1h(n_days=5)
        # Swap two timestamps to break monotonicity
        idx_list = list(df.index)
        idx_list[10], idx_list[11] = idx_list[11], idx_list[10]
        df.index = pd.DatetimeIndex(idx_list)

        result = _check_monotonic_index(df)

        assert not result.passed

    def test_monotonic_passes(self):
        df = _make_clean_1h(n_days=5)

        result = _check_monotonic_index(df)

        assert result.passed


# ===========================================================================
# Test: Volume distribution
# ===========================================================================


class TestVolumeDistribution:
    """Median daily volume must be > 100 shares."""

    def test_low_volume_fails(self):
        df = _make_clean_1h(n_days=10)
        # Set all volumes to 1 (below 100 threshold for daily totals = 7*1 = 7)
        df["Volume"] = 1.0

        result = _check_volume_distribution(df)

        assert not result.passed

    def test_normal_volume_passes(self):
        df = _make_clean_1h(n_days=10)  # Has volumes 1000-100000

        result = _check_volume_distribution(df)

        assert result.passed


# ===========================================================================
# Test: Zero volume on liquid name
# ===========================================================================


class TestZeroVolumeLiquid:
    """Zero-volume bars on liquid names during RTH should be flagged."""

    def test_zero_vol_on_liquid_flagged(self):
        df = _make_clean_1h(n_days=10)  # High volume = liquid
        # Set 3 bars to zero volume
        df.loc[df.index[5], "Volume"] = 0
        df.loc[df.index[10], "Volume"] = 0
        df.loc[df.index[15], "Volume"] = 0

        result, flagged = _check_zero_volume_liquid(df)

        assert not result.passed
        assert result.flagged_count == 3

    def test_illiquid_name_skipped(self):
        """Names with low median volume should skip this check."""
        df = _make_clean_1h(n_days=10)
        df["Volume"] = 10.0  # Very low volume = illiquid
        df.loc[df.index[5], "Volume"] = 0  # Zero vol but illiquid

        result, flagged = _check_zero_volume_liquid(df)

        assert result.passed  # Skipped for illiquid names


# ===========================================================================
# Test: Overnight gap (informational)
# ===========================================================================


class TestOvernightGap:
    """Large overnight gaps should be detected but not cause failure."""

    def test_large_gap_detected(self):
        df = _make_clean_1h(n_days=10)
        # Inject 15% overnight gap between day 1 and day 2
        # Find first bar of day 2
        dates = sorted(set(df.index.date))
        if len(dates) >= 2:
            day2_mask = df.index.date == dates[1]
            first_bar_day2 = df.index[day2_mask][0]
            day1_mask = df.index.date == dates[0]
            last_bar_day1 = df.index[day1_mask][-1]
            df.loc[first_bar_day2, "Open"] = df.loc[last_bar_day1, "Close"] * 1.15

        result = _check_overnight_gap(df)

        # Should detect gap but still pass (informational)
        assert result.passed is True
        assert len(result.details.get("gaps", [])) >= 1


# ===========================================================================
# Test: Quality report I/O
# ===========================================================================


class TestQualityReportIO:
    """Test write/read of quality report JSON sidecars."""

    def test_write_and_read_dataclass(self, tmp_path):
        """Write an IntradayQualityReport and read it back."""
        report = IntradayQualityReport(
            ticker="AAPL",
            timeframe="1h",
            source="test",
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_bars_input=1000,
            date_range="2024-01-02 to 2024-12-31",
        )
        report.add_check(CheckResult(
            check_name="ohlc_consistency",
            passed=True,
            rejected_count=0,
            message="All good",
        ))
        report.compute_quality_score(1000)

        path = tmp_path / "test.quality.json"
        write_quality_report(path, report)
        data = read_quality_report(path)

        assert data["ticker"] == "AAPL"
        assert data["quality_score"] == report.quality_score
        assert len(data["checks"]) == 1

    def test_write_and_read_dict(self, tmp_path):
        """Write a plain dict and read it back."""
        report_dict = {"ticker": "MSFT", "passed": True, "quality_score": 0.99}
        path = tmp_path / "test2.quality.json"

        write_quality_report(path, report_dict)
        data = read_quality_report(path)

        assert data["ticker"] == "MSFT"
        assert data["passed"] is True


# ===========================================================================
# Test: Quarantine system
# ===========================================================================


class TestQuarantineSystem:
    """Test quarantine file movement and JSON log."""

    def test_quarantine_creates_log(self, tmp_path):
        """Quarantining a ticker should create an append-only log."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        quarantine_ticker("AAPL", "1m", cache_dir, "Test quarantine")

        log_path = cache_dir / "quarantine" / "quarantine_log.json"
        assert log_path.exists()

        with open(log_path) as f:
            entries = json.load(f)

        assert len(entries) == 1
        assert entries[0]["ticker"] == "AAPL"
        assert entries[0]["reason"] == "Test quarantine"

    def test_quarantine_append_only(self, tmp_path):
        """Multiple quarantines should append, not overwrite."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        quarantine_ticker("AAPL", "1m", cache_dir, "Reason 1")
        quarantine_ticker("MSFT", "5m", cache_dir, "Reason 2")
        quarantine_ticker("GOOGL", "1h", cache_dir, "Reason 3")

        log_path = cache_dir / "quarantine" / "quarantine_log.json"
        with open(log_path) as f:
            entries = json.load(f)

        assert len(entries) == 3
        assert entries[0]["ticker"] == "AAPL"
        assert entries[2]["ticker"] == "GOOGL"

    def test_quarantine_moves_file(self, tmp_path):
        """Quarantining should move data file to quarantine dir."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create a fake data file
        data_file = cache_dir / "AAPL_1min_2024-01-01_2024-12-31.parquet"
        data_file.write_text("fake data")

        meta_file = cache_dir / "AAPL_1min_2024-01-01_2024-12-31.meta.json"
        meta_file.write_text('{"ticker": "AAPL"}')

        quarantine_ticker("AAPL", "1m", cache_dir, "Bad data",
                         source_path=data_file)

        # Original file should be gone
        assert not data_file.exists()
        # File should be in quarantine
        quarantine_path = cache_dir / "quarantine" / data_file.name
        assert quarantine_path.exists()

    def test_list_quarantined(self, tmp_path):
        """list_quarantined should return entries newest-first."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        quarantine_ticker("AAPL", "1m", cache_dir, "Reason 1")
        quarantine_ticker("MSFT", "5m", cache_dir, "Reason 2")

        entries = list_quarantined(cache_dir)

        assert len(entries) == 2
        # Newest first
        assert entries[0]["ticker"] == "MSFT"
        assert entries[1]["ticker"] == "AAPL"

    def test_list_quarantined_empty(self, tmp_path):
        """No quarantines should return empty list."""
        entries = list_quarantined(tmp_path)
        assert entries == []


# ===========================================================================
# Test: Full validate_intraday_bars integration
# ===========================================================================


class TestValidateIntradayBars:
    """Integration test for the full 13-check pipeline."""

    def test_returns_tuple(self):
        df = _make_clean_1h(n_days=5)
        result = validate_intraday_bars(df, "TEST", "1h")

        assert isinstance(result, tuple)
        assert len(result) == 2
        cleaned, report = result
        assert isinstance(cleaned, pd.DataFrame)
        assert isinstance(report, IntradayQualityReport)

    def test_all_13_checks_run(self):
        df = _make_clean_1h(n_days=10)
        _, report = validate_intraday_bars(df, "TEST", "1h")

        check_names = [c.check_name for c in report.checks]
        expected_checks = [
            "ohlc_consistency",
            "non_negative_volume",
            "non_negative_prices",
            "timestamp_in_rth",
            "extreme_bar_return",
            "stale_price",
            "zero_volume_liquid",
            "missing_bar_ratio",
            "duplicate_timestamps",
            "monotonic_index",
            "overnight_gap",
            "volume_distribution",
            "split_detection",
        ]

        for expected in expected_checks:
            assert expected in check_names, f"Missing check: {expected}"

    def test_report_fields(self):
        df = _make_clean_1h(n_days=5)
        _, report = validate_intraday_bars(df, "AAPL", "1h", source="alpaca")

        assert report.ticker == "AAPL"
        assert report.timeframe == "1h"
        assert report.source == "alpaca"
        assert report.total_bars_input == len(df)
        assert report.total_bars_output > 0
        assert 0.0 <= report.quality_score <= 1.0
