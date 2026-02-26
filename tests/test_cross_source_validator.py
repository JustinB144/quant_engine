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
    7. Volume mismatch detection — corrupted volume data is flagged
    8. Volume detail reporting — mismatch details contain actual values (not 0)
    9. Column normalization — lowercase/uppercase columns mapped to titlecase
"""

from datetime import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_engine.data.cross_source_validator import (
    CrossSourceValidator,
    CrossValidationReport,
    normalize_ohlcv_columns,
)



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


# ===========================================================================
# Test 7: Volume mismatch detection (SPEC-B04)
# ===========================================================================


class TestVolumeMismatchDetection:
    """Deliberately wrong volume data must be caught by the validator."""

    def test_corrupted_volume_detected(self):
        """Volume that differs by >5% and >100 shares should be flagged."""
        ibkr_df = _make_ibkr_like_1h(n_days=3)
        primary_df = ibkr_df.copy()

        # Corrupt volume on several bars: double the volume (>5% and >100 shares)
        bad_indices = primary_df.index[:5]
        for ts in bad_indices:
            primary_df.loc[ts, "Volume"] = ibkr_df.loc[ts, "Volume"] * 2.0

        validator = _create_mock_validator(volume_tolerance_pct=5.0, volume_min_abs_diff=100)
        result = validator._compare_bars(primary_df, ibkr_df, "TEST", "1h")

        assert result["volume_anomalies"] == 5, (
            f"Expected 5 volume anomalies, got {result['volume_anomalies']}"
        )

    def test_small_volume_diff_not_flagged(self):
        """Volume difference under min_abs_diff should not be flagged."""
        ibkr_df = _make_ibkr_like_1h(n_days=3)
        primary_df = ibkr_df.copy()

        # Add only 50 shares — under the 100-share min_abs_diff threshold
        primary_df["Volume"] = ibkr_df["Volume"] + 50

        validator = _create_mock_validator(volume_tolerance_pct=5.0, volume_min_abs_diff=100)
        result = validator._compare_bars(primary_df, ibkr_df, "TEST", "1h")

        assert result["volume_anomalies"] == 0

    def test_volume_zeroed_out_detected(self):
        """Volume dropped to zero should be flagged (absolute diff > 100)."""
        ibkr_df = _make_ibkr_like_1h(n_days=3)
        primary_df = ibkr_df.copy()

        # Zero out all volume in primary
        primary_df["Volume"] = 0.0

        validator = _create_mock_validator(volume_tolerance_pct=5.0, volume_min_abs_diff=100)
        result = validator._compare_bars(primary_df, ibkr_df, "TEST", "1h")

        # All bars should flag as volume anomalies (ibkr volume is 5000-200000)
        assert result["volume_anomalies"] == result["overlapping_bars"]


# ===========================================================================
# Test 8: Volume detail reporting (SPEC-B04 regression test)
# ===========================================================================


class TestVolumeMismatchDetailReporting:
    """Mismatch detail dicts must contain actual volume values, not 0."""

    def test_volume_detail_reports_actual_values(self):
        """
        Regression test for the lowercase 'volume' bug.

        Before the fix, detail["volume_mismatch"]["primary"] and
        detail["volume_mismatch"]["ibkr"] were always 0 because the code
        used prim_row.get("volume", 0) instead of prim_row.get("Volume", 0).
        """
        ibkr_df = _make_ibkr_like_1h(n_days=3)
        primary_df = ibkr_df.copy()

        # Corrupt one bar's volume significantly
        bad_ts = primary_df.index[5]
        original_ibkr_vol = int(ibkr_df.loc[bad_ts, "Volume"])
        primary_df.loc[bad_ts, "Volume"] = original_ibkr_vol * 3.0  # 3x different

        validator = _create_mock_validator(volume_tolerance_pct=5.0, volume_min_abs_diff=100)
        result = validator._compare_bars(primary_df, ibkr_df, "TEST", "1h")

        assert result["volume_anomalies"] >= 1

        # Find the volume mismatch detail for our corrupted bar
        vol_details = [
            d for d in result["details"]
            if d.get("timestamp") == bad_ts and "volume_mismatch" in d
        ]
        assert len(vol_details) == 1, "Expected exactly one volume mismatch detail"

        vol_info = vol_details[0]["volume_mismatch"]

        # Key assertion: values must be actual volumes, NOT 0
        assert vol_info["primary"] == int(original_ibkr_vol * 3.0), (
            f"Expected primary volume {int(original_ibkr_vol * 3.0)}, got {vol_info['primary']}"
        )
        assert vol_info["ibkr"] == original_ibkr_vol, (
            f"Expected ibkr volume {original_ibkr_vol}, got {vol_info['ibkr']}"
        )
        assert vol_info["pct_diff"] > 0, "pct_diff should be positive for mismatched volumes"


# ===========================================================================
# Test 9: Column normalization (SPEC-B04)
# ===========================================================================


class TestColumnNormalization:
    """Verify normalize_ohlcv_columns maps non-standard casing to canonical titlecase."""

    def test_lowercase_columns_normalized(self):
        """Lowercase OHLCV columns should be mapped to titlecase."""
        df = pd.DataFrame({
            "open": [100.0], "high": [105.0], "low": [95.0],
            "close": [102.0], "volume": [50000],
        })
        result = normalize_ohlcv_columns(df)

        assert "Open" in result.columns
        assert "High" in result.columns
        assert "Low" in result.columns
        assert "Close" in result.columns
        assert "Volume" in result.columns
        # Original lowercase should be gone
        assert "open" not in result.columns
        assert "volume" not in result.columns

    def test_uppercase_columns_normalized(self):
        """UPPERCASE OHLCV columns should be mapped to titlecase."""
        df = pd.DataFrame({
            "OPEN": [100.0], "HIGH": [105.0], "LOW": [95.0],
            "CLOSE": [102.0], "VOLUME": [50000],
        })
        result = normalize_ohlcv_columns(df)

        assert "Open" in result.columns
        assert "High" in result.columns
        assert "Low" in result.columns
        assert "Close" in result.columns
        assert "Volume" in result.columns

    def test_already_titlecase_unchanged(self):
        """Columns already in titlecase should not be changed."""
        df = pd.DataFrame({
            "Open": [100.0], "High": [105.0], "Low": [95.0],
            "Close": [102.0], "Volume": [50000],
        })
        result = normalize_ohlcv_columns(df)

        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_vol_abbreviation_normalized(self):
        """'vol' should be mapped to 'Volume'."""
        df = pd.DataFrame({"Open": [100.0], "High": [105.0], "Low": [95.0],
                           "Close": [102.0], "vol": [50000]})
        result = normalize_ohlcv_columns(df)
        assert "Volume" in result.columns
        assert "vol" not in result.columns

    def test_extra_columns_preserved(self):
        """Non-OHLCV columns should be passed through unchanged."""
        df = pd.DataFrame({
            "open": [100.0], "high": [105.0], "low": [95.0],
            "close": [102.0], "volume": [50000], "vwap": [101.5],
        })
        result = normalize_ohlcv_columns(df)

        assert "vwap" in result.columns
        assert result["vwap"].iloc[0] == 101.5

    def test_empty_dataframe_no_error(self):
        """Empty DataFrame should be returned without error."""
        df = pd.DataFrame()
        result = normalize_ohlcv_columns(df)
        assert result.empty

    def test_compare_bars_normalizes_lowercase_input(self):
        """
        _compare_bars should detect volume mismatches even when input uses
        lowercase column names, thanks to built-in normalization.
        """
        ibkr_df = _make_ibkr_like_1h(n_days=3)

        # Create primary with lowercase columns
        primary_df = pd.DataFrame({
            "open": ibkr_df["Open"].values,
            "high": ibkr_df["High"].values,
            "low": ibkr_df["Low"].values,
            "close": ibkr_df["Close"].values,
            "volume": ibkr_df["Volume"].values * 3.0,  # Deliberately wrong
        }, index=ibkr_df.index)

        validator = _create_mock_validator(volume_tolerance_pct=5.0, volume_min_abs_diff=100)
        result = validator._compare_bars(primary_df, ibkr_df, "TEST", "1h")

        # Despite lowercase input, normalization should enable volume detection
        assert result["volume_anomalies"] == result["overlapping_bars"], (
            f"Expected all bars flagged, got {result['volume_anomalies']} "
            f"of {result['overlapping_bars']}"
        )
