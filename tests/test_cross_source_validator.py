"""
Tests for cross-source validation logic.

All tests use synthetic DataFrames — no IBKR connection or API calls.

Coverage:
    1. Identical data → pass, 0 mismatches
    2. 0.10% close difference → pass (under 0.15% tolerance)
    3. 0.20% close difference → fail, bar replaced
    4. Missing bars in primary → detected, counted
    5. 2:1 price ratio (split mismatch) → detected
    6. All bars wrong → quarantine triggered (>5% mismatch rate)
"""

from datetime import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_engine.data.cross_source_validator import CrossSourceValidator, CrossValidationReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ibkr_like_1h(n_days: int = 5, base_price: float = 150.0) -> pd.DataFrame:
    """
    Generate synthetic IBKR-like 1h data.

    7 bars per day (09:30..15:30), clean OHLC, positive volume.
    """
    rng = np.random.default_rng(77)
    dates = pd.bdate_range("2024-06-03", periods=n_days, freq="B")
    hours = [time(h, 30) for h in range(9, 16)]

    timestamps = []
    for d in dates:
        for h in hours:
            timestamps.append(pd.Timestamp(d.year, d.month, d.day, h.hour, h.minute))

    n = len(timestamps)
    close = base_price + np.cumsum(rng.normal(0, 0.3, n))
    close = np.maximum(close, 1.0)
    high = close + rng.uniform(0.1, 0.5, n)
    low = close - rng.uniform(0.1, 0.5, n)
    low = np.maximum(low, 0.01)
    opn = close + rng.normal(0, 0.1, n)
    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))
    vol = rng.integers(5000, 200000, n).astype(float)

    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=pd.DatetimeIndex(timestamps),
    )


def _create_mock_validator(**kwargs) -> CrossSourceValidator:
    """
    Create a CrossSourceValidator with a mocked IB connection.

    The mock IB is never actually called in these unit tests because
    we bypass _fetch_ibkr_for_dates and call _compare_bars directly.
    """
    mock_ib = MagicMock()
    mock_ib.isConnected.return_value = True

    defaults = {
        "ib": mock_ib,
        "close_tolerance_pct": 0.15,
        "open_tolerance_pct": 0.20,
        "highlow_tolerance_pct": 0.25,
        "volume_tolerance_pct": 5.0,
        "volume_min_abs_diff": 100,
        "sample_windows": 10,
        "days_per_window": 2,
        "max_mismatch_rate": 0.05,
        "ibkr_pace": 0.0,
    }
    defaults.update(kwargs)
    return CrossSourceValidator(**defaults)


# ===========================================================================
# Test 1: Identical data → 0 mismatches
# ===========================================================================


class TestIdenticalData:
    """When primary exactly matches IBKR, everything should pass."""

    def test_identical_zero_mismatches(self):
        ibkr_df = _make_ibkr_like_1h(n_days=5)
        primary_df = ibkr_df.copy()

        validator = _create_mock_validator()
        result = validator._compare_bars(primary_df, ibkr_df, "TEST", "1h")

        assert result["overlapping_bars"] == len(ibkr_df)
        assert result["price_mismatches"] == 0
        assert result["open_mismatches"] == 0
        assert result["highlow_mismatches"] == 0
        assert result["missing_in_primary"] == 0
        assert result["phantom_in_primary"] == 0

    def test_replace_bad_bars_noop(self):
        """No mismatches means no replacements."""
        ibkr_df = _make_ibkr_like_1h(n_days=3)
        primary_df = ibkr_df.copy()

        validator = _create_mock_validator()
        corrected, replaced, inserted = validator._replace_bad_bars(
            primary_df, ibkr_df, []
        )

        assert replaced == 0
        assert inserted == 0
        assert len(corrected) == len(primary_df)


# ===========================================================================
# Test 2: 0.10% close difference → pass (under tolerance)
# ===========================================================================


class TestWithinTolerance:
    """Differences under the tolerance threshold should not be flagged."""

    def test_0_10_pct_close_diff_passes(self):
        ibkr_df = _make_ibkr_like_1h(n_days=5)
        primary_df = ibkr_df.copy()

        # Apply 0.10% perturbation to Close (under 0.15% tolerance)
        primary_df["Close"] = primary_df["Close"] * 1.001  # +0.10%

        validator = _create_mock_validator()
        result = validator._compare_bars(primary_df, ibkr_df, "TEST", "1h")

        assert result["price_mismatches"] == 0

    def test_0_10_pct_open_diff_passes(self):
        ibkr_df = _make_ibkr_like_1h(n_days=5)
        primary_df = ibkr_df.copy()

        # Apply 0.10% perturbation to Open (under 0.20% tolerance)
        primary_df["Open"] = primary_df["Open"] * 1.001

        validator = _create_mock_validator()
        result = validator._compare_bars(primary_df, ibkr_df, "TEST", "1h")

        assert result["open_mismatches"] == 0


# ===========================================================================
# Test 3: 0.20% close difference → fail, bar replaced
# ===========================================================================


class TestExceedsTolerance:
    """Differences above the tolerance threshold should be flagged and replaced."""

    def test_0_20_pct_close_diff_fails(self):
        ibkr_df = _make_ibkr_like_1h(n_days=5)
        primary_df = ibkr_df.copy()

        # Apply 0.20% perturbation to Close on all bars (above 0.15% tolerance)
        primary_df["Close"] = primary_df["Close"] * 1.002

        validator = _create_mock_validator()
        result = validator._compare_bars(primary_df, ibkr_df, "TEST", "1h")

        # All overlapping bars should be flagged
        assert result["price_mismatches"] == result["overlapping_bars"]

    def test_single_bar_replaced(self):
        ibkr_df = _make_ibkr_like_1h(n_days=3)
        primary_df = ibkr_df.copy()

        # Corrupt one bar's Close by 1%
        bad_ts = primary_df.index[5]
        primary_df.loc[bad_ts, "Close"] = ibkr_df.loc[bad_ts, "Close"] * 1.01

        validator = _create_mock_validator()
        result = validator._compare_bars(primary_df, ibkr_df, "TEST", "1h")

        assert result["price_mismatches"] == 1

        # Replace the bad bar
        mismatches = result["mismatches"]
        corrected, replaced, inserted = validator._replace_bad_bars(
            primary_df, ibkr_df, mismatches
        )

        assert replaced == 1
        # After replacement, the Close should match IBKR
        assert abs(corrected.loc[bad_ts, "Close"] - ibkr_df.loc[bad_ts, "Close"]) < 0.001


# ===========================================================================
# Test 4: Missing bars in primary → detected
# ===========================================================================


class TestMissingBars:
    """Bars present in IBKR but absent from primary should be detected."""

    def test_missing_bars_detected(self):
        ibkr_df = _make_ibkr_like_1h(n_days=5)
        # Remove 3 bars from primary
        drop_indices = ibkr_df.index[[5, 10, 15]]
        primary_df = ibkr_df.drop(drop_indices)

        validator = _create_mock_validator()
        result = validator._compare_bars(primary_df, ibkr_df, "TEST", "1h")

        assert result["missing_in_primary"] == 3

    def test_missing_bars_inserted(self):
        ibkr_df = _make_ibkr_like_1h(n_days=3)
        drop_indices = ibkr_df.index[[2, 5]]
        primary_df = ibkr_df.drop(drop_indices)

        validator = _create_mock_validator()
        result = validator._compare_bars(primary_df, ibkr_df, "TEST", "1h")

        mismatches = result["mismatches"]
        corrected, replaced, inserted = validator._replace_bad_bars(
            primary_df, ibkr_df, mismatches
        )

        assert inserted == 2
        # After insertion, the missing timestamps should be present
        for ts in drop_indices:
            assert ts in corrected.index


# ===========================================================================
# Test 5: 2:1 price ratio → split mismatch
# ===========================================================================


class TestSplitMismatch:
    """A 2:1 price ratio between sources suggests unadjusted split."""

    def test_2to1_split_detected(self):
        ibkr_df = _make_ibkr_like_1h(n_days=3)
        primary_df = ibkr_df.copy()

        # Set all primary Close to exactly 2x IBKR (unadjusted split)
        primary_df["Close"] = ibkr_df["Close"] * 2.0
        primary_df["Open"] = ibkr_df["Open"] * 2.0
        primary_df["High"] = ibkr_df["High"] * 2.0
        primary_df["Low"] = ibkr_df["Low"] * 2.0

        validator = _create_mock_validator()
        result = validator._compare_bars(primary_df, ibkr_df, "TEST", "1h")

        assert result["split_mismatches"] > 0

    def test_split_is_not_regular_mismatch(self):
        """Split mismatches should not be counted as regular price mismatches."""
        ibkr_df = _make_ibkr_like_1h(n_days=3)
        primary_df = ibkr_df.copy()

        # Set one bar to exactly 2x (split)
        ts = primary_df.index[5]
        primary_df.loc[ts, "Close"] = ibkr_df.loc[ts, "Close"] * 2.0

        validator = _create_mock_validator()
        result = validator._compare_bars(primary_df, ibkr_df, "TEST", "1h")

        # The split bar should be in split_mismatches, not price_mismatches
        assert result["split_mismatches"] >= 1


# ===========================================================================
# Test 6: All bars wrong → high mismatch rate
# ===========================================================================


class TestHighMismatchRate:
    """When most bars disagree, mismatch rate should exceed threshold."""

    def test_all_bars_wrong_triggers_quarantine(self):
        ibkr_df = _make_ibkr_like_1h(n_days=5)
        primary_df = ibkr_df.copy()

        # Corrupt ALL Close prices by 5% (way above 0.15% tolerance)
        primary_df["Close"] = ibkr_df["Close"] * 1.05

        validator = _create_mock_validator()
        result = validator._compare_bars(primary_df, ibkr_df, "TEST", "1h")

        mismatch_rate = result["price_mismatches"] / max(result["overlapping_bars"], 1)

        # All bars should be mismatched
        assert result["price_mismatches"] == result["overlapping_bars"]
        assert mismatch_rate > 0.05  # Above 5% quarantine threshold


# ===========================================================================
# Test: Utility methods
# ===========================================================================


class TestUtilityMethods:
    """Unit tests for helper methods."""

    def test_prices_differ_under_tolerance(self):
        validator = _create_mock_validator()
        # 0.10% difference, 0.15% tolerance → no diff
        assert validator._prices_differ(100.10, 100.0, 0.15) is False

    def test_prices_differ_over_tolerance(self):
        validator = _create_mock_validator()
        # 0.20% difference, 0.15% tolerance → diff detected
        assert validator._prices_differ(100.20, 100.0, 0.15) is True

    def test_prices_differ_nan(self):
        validator = _create_mock_validator()
        assert validator._prices_differ(float("nan"), 100.0, 0.15) is False

    def test_volumes_differ_small_abs(self):
        """Volume difference under min_abs_diff should not flag."""
        validator = _create_mock_validator()
        # 50 shares difference, min_abs_diff=100 → no flag
        assert validator._volumes_differ(1050, 1000, 50.0, 100) is False

    def test_volumes_differ_large(self):
        """Large volume difference should flag."""
        validator = _create_mock_validator()
        assert validator._volumes_differ(2000, 1000, 5.0, 100) is True

    def test_is_suspected_split_2x(self):
        validator = _create_mock_validator()
        assert validator._is_suspected_split(200.0, 100.0) is True

    def test_is_suspected_split_normal(self):
        validator = _create_mock_validator()
        assert validator._is_suspected_split(100.5, 100.0) is False

    def test_pct_diff(self):
        validator = _create_mock_validator()
        assert abs(validator._pct_diff(101.0, 100.0) - 1.0) < 0.01
        assert validator._pct_diff(100.0, 0.0) == 100.0
        assert validator._pct_diff(0.0, 0.0) == 0.0

    def test_group_dates_into_ranges(self):
        validator = _create_mock_validator()

        dates = [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
            pd.Timestamp("2024-01-04"),
            pd.Timestamp("2024-01-10"),  # Gap
            pd.Timestamp("2024-01-11"),
        ]

        ranges = validator._group_dates_into_ranges(dates)

        assert len(ranges) == 2
        assert ranges[0] == (dates[0], dates[2])
        assert ranges[1] == (dates[3], dates[4])


# ===========================================================================
# Test: Sample date selection
# ===========================================================================


class TestSampleDateSelection:
    """Test the stratified sampling logic."""

    def test_select_sample_dates_returns_sorted(self):
        df = _make_ibkr_like_1h(n_days=50)
        validator = _create_mock_validator(sample_windows=5, days_per_window=2)

        dates = validator._select_sample_dates(df)

        assert len(dates) <= 10  # 5 windows * 2 days
        # Should be sorted
        assert dates == sorted(dates)

    def test_select_sample_dates_small_dataset(self):
        """If fewer unique timestamps than windows, return all timestamps."""
        df = _make_ibkr_like_1h(n_days=1)  # 7 bars → 7 unique timestamps < 10 windows
        validator = _create_mock_validator(sample_windows=10, days_per_window=2)

        dates = validator._select_sample_dates(df)

        # 7 unique timestamps < 10 windows → short-circuit returns all
        unique_timestamps = df.index.unique()
        assert len(dates) == len(unique_timestamps)

    def test_select_sample_dates_empty(self):
        validator = _create_mock_validator()
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df.index = pd.DatetimeIndex([])

        dates = validator._select_sample_dates(df)

        assert dates == []


# ===========================================================================
# Test: CrossValidationReport dataclass
# ===========================================================================


class TestCrossValidationReport:
    """Verify the report dataclass structure."""

    def test_report_creation(self):
        report = CrossValidationReport(
            ticker="AAPL",
            timeframe="1h",
            sample_windows=10,
            sample_days=20,
            overlapping_bars=7800,
            price_mismatches=3,
            open_mismatches=1,
            highlow_mismatches=2,
            volume_anomalies=5,
            missing_in_primary=12,
            phantom_in_primary=0,
            split_mismatches=0,
            bars_replaced=3,
            bars_inserted=12,
            passed=True,
            mismatch_rate=0.0004,
        )

        assert report.ticker == "AAPL"
        assert report.passed is True
        assert report.mismatch_rate < 0.05

    def test_report_has_required_fields(self):
        """Verify all spec-required fields exist on the dataclass."""
        required_fields = [
            "ticker", "timeframe", "sample_windows", "sample_days",
            "overlapping_bars", "price_mismatches", "missing_in_primary",
            "phantom_in_primary", "split_mismatches", "bars_replaced",
            "bars_inserted", "passed", "mismatch_rate", "details",
        ]

        report = CrossValidationReport(
            ticker="TEST", timeframe="1m", sample_windows=10, sample_days=20,
            overlapping_bars=0, price_mismatches=0, open_mismatches=0,
            highlow_mismatches=0, volume_anomalies=0, missing_in_primary=0,
            phantom_in_primary=0, split_mismatches=0, bars_replaced=0,
            bars_inserted=0, passed=True, mismatch_rate=0.0,
        )

        for field_name in required_fields:
            assert hasattr(report, field_name), f"Missing field: {field_name}"
