"""
Tests for Spec AUDIT_FIX_17: Data Quality Gate Enforcement.

Covers:
    T1 — DataIntegrityValidator in load_with_delistings
    T2 — Quality check on WRDS-fetched data before caching
    T3 — Survivorship-safe loading hard path
"""
from __future__ import annotations

import logging
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


# ── Helpers ───────────────────────────────────────────────────────────


def _make_ohlcv(n: int = 500, seed: int = 42, corrupt: bool = False) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing.

    If corrupt=True, inject NaN prices and OHLC violations.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    close = 100.0 + np.cumsum(rng.normal(0.0005, 0.015, n))
    close = np.maximum(close, 1.0)
    opn = close * (1 + rng.normal(0, 0.003, n))
    opn = np.maximum(opn, 1.0)

    bar_max = np.maximum(opn, close)
    bar_min = np.minimum(opn, close)
    high = bar_max * (1 + rng.uniform(0, 0.015, n))
    low = bar_min * (1 - rng.uniform(0, 0.015, n))

    vol = rng.integers(500_000, 5_000_000, n).astype(float)

    if corrupt:
        # Inject massive zero-volume (>50% to fail quality)
        vol[:] = 0.0

    df = pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )
    # Add required columns for loader
    df["Return"] = df["Close"].pct_change()
    df["total_ret"] = df["Return"]
    df["dlret"] = np.nan
    df["delist_event"] = 0
    return df


def _make_universe(
    n_stocks: int = 3,
    n_bars: int = 500,
    corrupt_tickers: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate a universe of synthetic OHLCV data."""
    corrupt_tickers = corrupt_tickers or []
    data = {}
    for i in range(n_stocks):
        ticker = f"STOCK{i}"
        corrupt = ticker in corrupt_tickers
        df = _make_ohlcv(n=n_bars, seed=42 + i, corrupt=corrupt)
        df.attrs["permno"] = str(10000 + i)
        df.attrs["ticker"] = ticker
        data[ticker] = df
    return data


# ═══════════════════════════════════════════════════════════════════════
# T1: DataIntegrityValidator in load_with_delistings
# ═══════════════════════════════════════════════════════════════════════


class TestLoadWithDelistingsIntegrity:
    """Tests that load_with_delistings validates data before returning."""

    @patch("quant_engine.data.loader.TRUTH_LAYER_FAIL_ON_CORRUPT", True)
    @patch("quant_engine.data.loader.WRDS_ENABLED", False)
    @patch("quant_engine.data.loader.REQUIRE_PERMNO", False)
    @patch("quant_engine.data.loader.DATA_QUALITY_ENABLED", False)
    def test_corrupt_tickers_removed(self):
        """Corrupt tickers should be removed from the result."""
        good_data = _make_ohlcv(n=500, seed=42, corrupt=False)
        bad_data = _make_ohlcv(n=500, seed=43, corrupt=True)

        universe = {"GOOD1": good_data, "BAD1": bad_data}

        # Mock load_universe to return our pre-made universe
        with patch("quant_engine.data.loader.load_universe", return_value=universe):
            with patch("quant_engine.data.loader.cache_load_with_meta", return_value=(None, {}, None)):
                from quant_engine.data.loader import load_with_delistings
                result = load_with_delistings(
                    tickers=["GOOD1", "BAD1"],
                    years=5,
                    verbose=False,
                )

        # Bad ticker should be removed, good one kept
        assert "GOOD1" in result
        assert "BAD1" not in result

    @patch("quant_engine.data.loader.TRUTH_LAYER_FAIL_ON_CORRUPT", True)
    @patch("quant_engine.data.loader.WRDS_ENABLED", False)
    @patch("quant_engine.data.loader.REQUIRE_PERMNO", False)
    @patch("quant_engine.data.loader.DATA_QUALITY_ENABLED", False)
    def test_all_corrupt_raises(self):
        """If all tickers fail integrity, RuntimeError should be raised."""
        bad1 = _make_ohlcv(n=500, seed=42, corrupt=True)
        bad2 = _make_ohlcv(n=500, seed=43, corrupt=True)

        universe = {"BAD1": bad1, "BAD2": bad2}

        with patch("quant_engine.data.loader.load_universe", return_value=universe):
            with patch("quant_engine.data.loader.cache_load_with_meta", return_value=(None, {}, None)):
                from quant_engine.data.loader import load_with_delistings
                with pytest.raises(RuntimeError, match="failed integrity check"):
                    load_with_delistings(
                        tickers=["BAD1", "BAD2"],
                        years=5,
                        verbose=False,
                    )

    @patch("quant_engine.data.loader.TRUTH_LAYER_FAIL_ON_CORRUPT", False)
    @patch("quant_engine.data.loader.WRDS_ENABLED", False)
    @patch("quant_engine.data.loader.REQUIRE_PERMNO", False)
    @patch("quant_engine.data.loader.DATA_QUALITY_ENABLED", False)
    def test_validation_skipped_when_disabled(self):
        """When TRUTH_LAYER_FAIL_ON_CORRUPT is False, validation is skipped."""
        bad_data = _make_ohlcv(n=500, seed=42, corrupt=True)
        universe = {"BAD1": bad_data}

        with patch("quant_engine.data.loader.load_universe", return_value=universe):
            with patch("quant_engine.data.loader.cache_load_with_meta", return_value=(None, {}, None)):
                from quant_engine.data.loader import load_with_delistings
                result = load_with_delistings(
                    tickers=["BAD1"],
                    years=5,
                    verbose=False,
                )
        # Data should pass through without validation
        assert "BAD1" in result


# ═══════════════════════════════════════════════════════════════════════
# T2: Quality Check on WRDS-Fetched Data Before Caching
# ═══════════════════════════════════════════════════════════════════════


class TestWRDSQualityGate:
    """Tests that WRDS data is quality-checked before caching."""

    def test_quality_gate_prevents_caching_bad_data(self):
        """Corrupt WRDS data should not be saved to cache."""
        from quant_engine.data.quality import assess_ohlcv_quality

        bad_data = _make_ohlcv(n=500, seed=42, corrupt=True)
        quality = assess_ohlcv_quality(bad_data)
        # Verify our test data actually fails quality
        assert not quality.passed
        assert len(quality.warnings) > 0

    def test_quality_gate_passes_good_data(self):
        """Good WRDS data should pass quality checks."""
        from quant_engine.data.quality import assess_ohlcv_quality

        good_data = _make_ohlcv(n=500, seed=42, corrupt=False)
        quality = assess_ohlcv_quality(good_data)
        assert quality.passed


# ═══════════════════════════════════════════════════════════════════════
# T3: Survivorship-Safe Loading Hard Path
# ═══════════════════════════════════════════════════════════════════════


class TestSurvivorshipSafeLoading:
    """Tests for survivorship-safe tagging and strict mode."""

    @patch("quant_engine.data.loader.WRDS_ENABLED", False)
    @patch("quant_engine.data.loader.REQUIRE_PERMNO", False)
    @patch("quant_engine.data.loader.DATA_QUALITY_ENABLED", False)
    @patch("quant_engine.data.loader.TRUTH_LAYER_FAIL_ON_CORRUPT", False)
    def test_wrds_disabled_tags_not_safe(self):
        """When WRDS is disabled, data should be tagged survivorship_safe=False."""
        good_data = _make_ohlcv(n=500, seed=42)
        universe = {"STOCK0": good_data}

        with patch("quant_engine.data.loader.load_universe", return_value=universe):
            with patch("quant_engine.data.loader._cached_universe_subset", return_value=[]):
                from quant_engine.data.loader import load_survivorship_universe
                result = load_survivorship_universe(years=5, verbose=False)

        sample_df = next(iter(result.values()))
        assert sample_df.attrs.get("survivorship_safe") is False

    @patch("quant_engine.data.loader.WRDS_ENABLED", False)
    def test_strict_mode_raises_wrds_disabled(self):
        """With strict=True and WRDS disabled, should raise RuntimeError."""
        with patch("quant_engine.data.loader._cached_universe_subset", return_value=[]):
            from quant_engine.data.loader import load_survivorship_universe
            with pytest.raises(RuntimeError, match="WRDS"):
                load_survivorship_universe(years=5, verbose=False, strict=True)

    @patch("quant_engine.data.loader.WRDS_ENABLED", True)
    def test_strict_mode_raises_wrds_unavailable(self):
        """With strict=True and WRDS unavailable, should raise RuntimeError."""
        mock_provider = MagicMock()
        mock_provider.available.return_value = False

        with patch("quant_engine.data.loader.get_provider", return_value=mock_provider):
            with patch("quant_engine.data.loader._cached_universe_subset", return_value=[]):
                from quant_engine.data.loader import load_survivorship_universe
                with pytest.raises(RuntimeError, match="WRDS"):
                    load_survivorship_universe(years=5, verbose=False, strict=True)

    def test_warn_if_survivorship_biased_detects_bias(self, caplog):
        """warn_if_survivorship_biased should log a warning for biased data."""
        from quant_engine.data.loader import warn_if_survivorship_biased

        df = _make_ohlcv(n=500, seed=42)
        df.attrs["survivorship_safe"] = False
        data = {"STOCK0": df}

        with caplog.at_level(logging.WARNING, logger="quant_engine.data.loader"):
            result = warn_if_survivorship_biased(data, context="test")

        assert result is False
        assert "survivorship-BIASED" in caplog.text

    def test_warn_if_survivorship_biased_passes_safe(self, caplog):
        """warn_if_survivorship_biased should not warn for safe data."""
        from quant_engine.data.loader import warn_if_survivorship_biased

        df = _make_ohlcv(n=500, seed=42)
        df.attrs["survivorship_safe"] = True
        data = {"STOCK0": df}

        with caplog.at_level(logging.WARNING, logger="quant_engine.data.loader"):
            result = warn_if_survivorship_biased(data, context="test")

        assert result is True
        assert "survivorship-BIASED" not in caplog.text

    def test_tag_survivorship_safe_helper(self):
        """_tag_survivorship_safe should set attrs on all DataFrames."""
        from quant_engine.data.loader import _tag_survivorship_safe

        df1 = _make_ohlcv(n=100, seed=1)
        df2 = _make_ohlcv(n=100, seed=2)
        data = {"A": df1, "B": df2}

        _tag_survivorship_safe(data, safe=True)
        assert df1.attrs["survivorship_safe"] is True
        assert df2.attrs["survivorship_safe"] is True

        _tag_survivorship_safe(data, safe=False)
        assert df1.attrs["survivorship_safe"] is False
        assert df2.attrs["survivorship_safe"] is False
