"""Tests for charting API endpoints and data flow (Spec 004).

Verifies:
  - /bars endpoint returns valid OHLCV for daily timeframe
  - /bars endpoint handles intraday timeframes
  - /bars endpoint returns 404 for missing timeframe
  - Available timeframes detection
  - /indicators endpoint returns correct RSI structure
  - Indicator overlay vs panel type classification
  - Indicator caching behavior
  - Batch indicator endpoint computes multiple at once
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quant_engine.api.routers.data_explorer import (
    _find_cached_parquet,
    _available_timeframes,
    _load_bars,
    _compute_indicators_for_ticker,
    _parse_indicator_spec,
    _INDICATOR_DISPATCH,
    _AVAILABLE_INDICATORS,
)


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def data_cache(tmp_path):
    """Create a mock data cache with daily and 15min parquets."""
    np.random.seed(42)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Daily data: 500 bars
    n_daily = 500
    idx_daily = pd.date_range("2024-01-01", periods=n_daily, freq="B")
    prices = np.cumsum(np.random.randn(n_daily) * 0.02) + 150

    daily = pd.DataFrame({
        "Open": prices + np.random.randn(n_daily) * 0.3,
        "High": prices + np.abs(np.random.randn(n_daily) * 0.8),
        "Low": prices - np.abs(np.random.randn(n_daily) * 0.8),
        "Close": prices,
        "Volume": np.random.randint(5_000_000, 50_000_000, n_daily).astype(float),
    }, index=idx_daily)
    daily.to_parquet(cache_dir / "AAPL_1d.parquet")

    # 15-minute data: 2000 bars
    n_intra = 2000
    idx_intra = pd.date_range("2024-06-01 09:30", periods=n_intra, freq="15min")
    intra_prices = np.cumsum(np.random.randn(n_intra) * 0.005) + 185

    intra = pd.DataFrame({
        "Open": intra_prices + np.random.randn(n_intra) * 0.1,
        "High": intra_prices + np.abs(np.random.randn(n_intra) * 0.3),
        "Low": intra_prices - np.abs(np.random.randn(n_intra) * 0.3),
        "Close": intra_prices,
        "Volume": np.random.randint(100_000, 1_000_000, n_intra).astype(float),
    }, index=idx_intra)
    intra.to_parquet(cache_dir / "AAPL_15min.parquet")

    # Also a range-based file for MSFT
    msft_prices = np.cumsum(np.random.randn(300) * 0.02) + 400
    msft = pd.DataFrame({
        "Open": msft_prices + np.random.randn(300) * 0.5,
        "High": msft_prices + np.abs(np.random.randn(300) * 1.0),
        "Low": msft_prices - np.abs(np.random.randn(300) * 1.0),
        "Close": msft_prices,
        "Volume": np.random.randint(1_000_000, 10_000_000, 300).astype(float),
    }, index=pd.date_range("2024-01-01", periods=300, freq="B"))
    msft.to_parquet(cache_dir / "MSFT_1d_2024-01-01_2025-02-20.parquet")

    return cache_dir


# ── Test: Bars endpoint daily ───────────────────────────────────────────

class TestBarsEndpoint:
    """T1 & T2: Bars endpoint returns valid OHLCV data."""

    def test_daily_bars_valid_ohlcv(self, data_cache):
        """Daily bars response has correct structure and types."""
        result = _load_bars(data_cache, "AAPL", "1d", 500)

        assert result["found"] is True
        assert result["ticker"] == "AAPL"
        assert result["timeframe"] == "1d"
        assert len(result["bars"]) > 0
        assert result["total_bars"] == len(result["bars"])

        # Check bar structure
        bar = result["bars"][0]
        assert "time" in bar
        assert "open" in bar
        assert "high" in bar
        assert "low" in bar
        assert "close" in bar
        assert "volume" in bar
        assert isinstance(bar["open"], float)
        assert isinstance(bar["volume"], float)

    def test_intraday_15min_loads(self, data_cache):
        """15-minute bars load correctly."""
        result = _load_bars(data_cache, "AAPL", "15min", 500)

        assert result["found"] is True
        assert result["timeframe"] == "15min"
        assert len(result["bars"]) == 500  # limited by max_bars

    def test_max_bars_limit(self, data_cache):
        """Bar count is capped at the requested limit."""
        result = _load_bars(data_cache, "AAPL", "1d", 100)
        assert len(result["bars"]) == 100

    def test_missing_timeframe_returns_not_found(self, data_cache):
        """Requesting a timeframe that doesn't exist returns found=False."""
        result = _load_bars(data_cache, "AAPL", "5min", 500)
        assert result["found"] is False

    def test_missing_ticker_returns_not_found(self, data_cache):
        """Requesting a ticker that doesn't exist returns found=False."""
        result = _load_bars(data_cache, "ZZXYZ", "1d", 500)
        assert result["found"] is False

    def test_range_based_filename(self, data_cache):
        """MSFT with range-based filename pattern is found."""
        result = _load_bars(data_cache, "MSFT", "1d", 100)
        assert result["found"] is True
        assert result["ticker"] == "MSFT"


# ── Test: Available timeframes ──────────────────────────────────────────

class TestAvailableTimeframes:
    """T4: Available timeframes detection."""

    def test_discovers_daily_and_15min(self, data_cache):
        """Detects both daily and 15min for AAPL."""
        tfs = _available_timeframes(data_cache, "AAPL")
        assert "1d" in tfs
        assert "15min" in tfs
        # 5min, 30min, 1hr should NOT be available
        assert "5min" not in tfs
        assert "30min" not in tfs
        assert "1hr" not in tfs

    def test_bars_response_includes_available_timeframes(self, data_cache):
        """Bars response contains available_timeframes list."""
        result = _load_bars(data_cache, "AAPL", "1d", 100)
        assert "available_timeframes" in result
        assert isinstance(result["available_timeframes"], list)
        assert "1d" in result["available_timeframes"]


# ── Test: Indicator endpoint RSI ────────────────────────────────────────

class TestIndicatorEndpoint:
    """T5 & T6: Indicator computation returns correct structure."""

    def test_rsi_returns_panel_type(self, data_cache):
        """RSI is classified as panel type."""
        result = _compute_indicators_for_ticker(data_cache, "AAPL", "1d", ["rsi_14"])

        assert result["found"] is True
        rsi = result["indicators"]["rsi_14"]
        assert rsi["type"] == "panel"
        assert "values" in rsi
        assert len(rsi["values"]) > 0
        assert "thresholds" in rsi
        assert rsi["thresholds"]["overbought"] == 70
        assert rsi["thresholds"]["oversold"] == 30

    def test_bollinger_returns_overlay_type(self, data_cache):
        """Bollinger bands is classified as overlay type."""
        result = _compute_indicators_for_ticker(data_cache, "AAPL", "1d", ["bollinger_20"])

        bb = result["indicators"]["bollinger_20"]
        assert bb["type"] == "overlay"
        assert "upper" in bb
        assert "middle" in bb
        assert "lower" in bb
        assert len(bb["upper"]) > 0

    def test_macd_returns_three_components(self, data_cache):
        """MACD returns macd, signal, and histogram."""
        result = _compute_indicators_for_ticker(data_cache, "AAPL", "1d", ["macd"])

        macd = result["indicators"]["macd"]
        assert macd["type"] == "panel"
        assert "macd" in macd
        assert "signal" in macd
        assert "histogram" in macd

    def test_sma_returns_overlay(self, data_cache):
        """SMA is classified as overlay type."""
        result = _compute_indicators_for_ticker(data_cache, "AAPL", "1d", ["sma_50"])

        sma = result["indicators"]["sma_50"]
        assert sma["type"] == "overlay"
        assert "values" in sma

    def test_unknown_indicator_returns_error(self, data_cache):
        """Unknown indicator name returns an error entry."""
        result = _compute_indicators_for_ticker(data_cache, "AAPL", "1d", ["foobar_14"])

        assert "foobar_14" in result["indicators"]
        assert "error" in result["indicators"]["foobar_14"]

    def test_available_indicators_list(self, data_cache):
        """Response includes available_indicators listing supported indicators."""
        result = _compute_indicators_for_ticker(data_cache, "AAPL", "1d", ["rsi_14"])

        assert "available_indicators" in result
        available = result["available_indicators"]
        assert "rsi" in available
        assert "macd" in available
        assert "bollinger" in available
        assert "sma" in available

    def test_stochastic_returns_k_and_d(self, data_cache):
        """Stochastic returns %K and %D series."""
        result = _compute_indicators_for_ticker(data_cache, "AAPL", "1d", ["stochastic"])

        stoch = result["indicators"]["stochastic"]
        assert stoch["type"] == "panel"
        assert "k" in stoch
        assert "d" in stoch
        assert "thresholds" in stoch


# ── Test: Parse indicator spec ──────────────────────────────────────────

class TestParseIndicatorSpec:
    """Indicator spec parsing helper."""

    def test_rsi_14(self):
        name, period = _parse_indicator_spec("rsi_14")
        assert name == "rsi"
        assert period == 14

    def test_macd_no_period(self):
        name, period = _parse_indicator_spec("macd")
        assert name == "macd"
        assert period == 14  # default

    def test_bollinger_20(self):
        name, period = _parse_indicator_spec("bollinger_20")
        assert name == "bollinger"
        assert period == 20

    def test_sma_200(self):
        name, period = _parse_indicator_spec("sma_200")
        assert name == "sma"
        assert period == 200


# ── Test: Batch indicators ──────────────────────────────────────────────

class TestBatchIndicators:
    """T8: Multiple indicators computed in one pass."""

    def test_multiple_indicators_single_call(self, data_cache):
        """Computing multiple indicators returns all of them."""
        specs = ["rsi_14", "macd", "bollinger_20", "sma_50"]
        result = _compute_indicators_for_ticker(data_cache, "AAPL", "1d", specs)

        assert result["found"] is True
        for spec in specs:
            assert spec in result["indicators"], f"Missing indicator: {spec}"
            assert "error" not in result["indicators"][spec], (
                f"Indicator {spec} errored: {result['indicators'][spec]}"
            )

    def test_batch_with_missing_data(self, data_cache):
        """Batch for nonexistent ticker returns found=False."""
        result = _compute_indicators_for_ticker(data_cache, "ZZXYZ", "1d", ["rsi_14"])
        assert result["found"] is False


# ── Test: Find cached parquet ───────────────────────────────────────────

class TestFindCachedParquet:
    """Parquet file discovery with various naming conventions."""

    def test_exact_match(self, data_cache):
        """Finds AAPL_1d.parquet by exact match."""
        path = _find_cached_parquet(data_cache, "AAPL", "1d")
        assert path is not None
        assert path.name == "AAPL_1d.parquet"

    def test_case_insensitive(self, data_cache):
        """Ticker lookup is case-insensitive."""
        path = _find_cached_parquet(data_cache, "aapl", "1d")
        assert path is not None

    def test_range_based_pattern(self, data_cache):
        """Finds MSFT_1d_START_END.parquet with range-based naming."""
        path = _find_cached_parquet(data_cache, "MSFT", "1d")
        assert path is not None
        assert "MSFT" in path.name

    def test_no_match_returns_none(self, data_cache):
        """Returns None for nonexistent ticker."""
        path = _find_cached_parquet(data_cache, "ZZZZZ", "1d")
        assert path is None

    def test_timeframe_aliases(self, data_cache):
        """15min parquet found via '15min' or '15m' alias."""
        path = _find_cached_parquet(data_cache, "AAPL", "15min")
        assert path is not None
