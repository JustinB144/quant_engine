"""Integration test: common operations must produce zero ERROR-level log entries.

Spec 007 T6 — verify that health checks, regime detection, config validation,
and data loading (when cache exists) run without triggering ERROR-level logs.
WARNING-level entries are acceptable.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class _ErrorCapture(logging.Handler):
    """Handler that records any ERROR or CRITICAL log records."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture()
def error_capture():
    """Attach an error-capturing handler to the root logger for the test."""
    handler = _ErrorCapture()
    root = logging.getLogger()
    root.addHandler(handler)
    yield handler
    root.removeHandler(handler)


def test_config_validation_no_errors(error_capture):
    """validate_config() must not trigger ERROR-level logs."""
    from quant_engine.config import validate_config

    issues = validate_config()
    # validate_config returns issues but should not log ERRORs itself
    assert isinstance(issues, list)
    errors = [r for r in error_capture.records if r.levelno >= logging.ERROR]
    assert errors == [], f"ERROR-level logs during config validation: {[r.getMessage() for r in errors]}"


def test_regime_detection_no_errors(error_capture):
    """Regime detection on synthetic data must not produce ERROR-level logs."""
    from quant_engine.regime.detector import RegimeDetector

    idx = pd.date_range("2020-01-01", periods=600, freq="B")
    rng = np.random.default_rng(42)
    features = pd.DataFrame(
        {
            "Close": np.cumsum(rng.normal(0, 0.02, 600)) + 100,
            "High": np.cumsum(rng.normal(0, 0.02, 600)) + 101,
            "Low": np.cumsum(rng.normal(0, 0.02, 600)) + 99,
            "Open": np.cumsum(rng.normal(0, 0.02, 600)) + 100,
            "Volume": rng.integers(1_000_000, 10_000_000, 600).astype(float),
        },
        index=idx,
    )

    detector = RegimeDetector(method="hmm")
    out = detector.detect_full(features)
    assert not out.probabilities.isna().any().any(), "NaN in probabilities"
    assert not out.regime.isna().any(), "NaN in regime"

    errors = [r for r in error_capture.records if r.levelno >= logging.ERROR]
    assert errors == [], f"ERROR-level logs during regime detection: {[r.getMessage() for r in errors]}"


def test_health_service_no_errors(error_capture):
    """HealthService instantiation must not produce ERROR-level logs."""
    from quant_engine.api.services.health_service import HealthService

    svc = HealthService()
    status = svc.get_quick_status()
    assert "status" in status

    errors = [r for r in error_capture.records if r.levelno >= logging.ERROR]
    assert errors == [], f"ERROR-level logs during health check: {[r.getMessage() for r in errors]}"


def test_data_loading_graceful_degradation(error_capture):
    """Loading a nonexistent ticker must fail gracefully (no ERROR logs, no crash)."""
    from quant_engine.data.loader import load_ohlcv

    result = load_ohlcv("NONEXISTENT_TICKER_XYZ_999", years=1, use_wrds=False)
    assert result is None, "Expected None for nonexistent ticker"

    errors = [r for r in error_capture.records if r.levelno >= logging.ERROR]
    assert errors == [], f"ERROR-level logs during graceful data load: {[r.getMessage() for r in errors]}"


def test_data_loading_single_ticker_if_cache_exists(error_capture):
    """If cache exists, loading a single ticker must not produce ERROR-level logs."""
    from quant_engine.config import DATA_CACHE_DIR

    cache_dir = Path(DATA_CACHE_DIR)
    if not cache_dir.exists() or not any(cache_dir.glob("*.parquet")):
        pytest.skip("No cached data available — skipping cache-dependent test")

    from quant_engine.data.loader import load_ohlcv

    # Try to load the first available cached ticker
    parquets = sorted(cache_dir.glob("*_1d.parquet"))
    if not parquets:
        parquets = sorted(cache_dir.glob("*_daily_*.parquet"))
    if not parquets:
        pytest.skip("No loadable parquet files in cache")

    ticker = parquets[0].stem.split("_")[0]
    result = load_ohlcv(ticker, years=2, use_wrds=False)
    # Result may be None (e.g. PERMNO required), but should not crash or ERROR

    errors = [r for r in error_capture.records if r.levelno >= logging.ERROR]
    assert errors == [], f"ERROR-level logs during cached data load: {[r.getMessage() for r in errors]}"
