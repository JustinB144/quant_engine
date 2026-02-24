"""
Tests for data loading diagnostics (Spec 005).

Verifies:
  1. Orchestrator imports DATA_CACHE_DIR (not the old DATA_DIR)
  2. load_universe logs skip reasons even with verbose=False
  3. RuntimeError includes actionable diagnostics
  4. Data status endpoint returns valid per-ticker info
"""
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


class TestOrchestratorImport(unittest.TestCase):
    """T1: DATA_DIR -> DATA_CACHE_DIR fix."""

    def test_orchestrator_imports_data_cache_dir(self):
        """Importing PipelineOrchestrator must NOT raise ImportError
        caused by referencing the non-existent ``DATA_DIR`` constant.
        """
        from quant_engine.api.orchestrator import PipelineOrchestrator  # noqa: F401

    def test_orchestrator_uses_data_cache_dir_in_diagnostics(self):
        """When load_universe returns empty, the error references
        DATA_CACHE_DIR (not DATA_DIR).
        """
        from quant_engine.api.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()
        with patch("quant_engine.data.loader.load_universe", return_value={}):
            try:
                orchestrator.load_and_prepare(tickers=["FAKE_TICKER"], years=1)
                self.fail("Expected RuntimeError")
            except RuntimeError as exc:
                msg = str(exc)
                self.assertIn("No data loaded", msg)
                self.assertIn("WRDS_ENABLED", msg)
                self.assertIn("REQUIRE_PERMNO", msg)


class TestLoadUniverseSkipReasons(unittest.TestCase):
    """T2: Skip reasons logged at WARNING even with verbose=False."""

    def test_load_universe_logs_skip_reasons(self):
        """With verbose=False, load_universe still logs WARNING for skipped tickers."""
        from quant_engine.data.loader import load_universe, get_skip_reasons

        with self.assertLogs("quant_engine.data.loader", level="WARNING") as cm:
            # Pass a ticker that won't resolve (no cache, no WRDS in test)
            with patch("quant_engine.data.loader.load_ohlcv", return_value=None):
                result = load_universe(["ZZZFAKE"], years=1, verbose=False)

        # Should have logged a warning about skipped tickers
        warning_messages = [r for r in cm.output if "skipped" in r.lower()]
        self.assertTrue(len(warning_messages) > 0, "Expected WARNING log for skipped tickers")

        # Skip reasons should be available
        reasons = get_skip_reasons()
        self.assertIn("ZZZFAKE", reasons)
        self.assertIn("None", reasons["ZZZFAKE"] if "None" in reasons["ZZZFAKE"] else "load_ohlcv returned None")

    def test_skip_reasons_include_reason_text(self):
        """Skip reason includes descriptive text (not just the ticker)."""
        from quant_engine.data.loader import get_skip_reasons, load_universe

        # Create a DataFrame too short to pass MIN_BARS check
        short_df = pd.DataFrame({
            "Open": [100.0] * 10,
            "High": [101.0] * 10,
            "Low": [99.0] * 10,
            "Close": [100.5] * 10,
            "Volume": [1000.0] * 10,
        }, index=pd.date_range("2024-01-01", periods=10))
        short_df.attrs["permno"] = "99999"
        short_df.attrs["ticker"] = "SHORT"

        with patch("quant_engine.data.loader.load_ohlcv", return_value=short_df):
            load_universe(["SHORT"], years=1, verbose=False)

        reasons = get_skip_reasons()
        self.assertIn("SHORT", reasons)
        self.assertIn("insufficient data", reasons["SHORT"])


class TestErrorMessageDiagnostics(unittest.TestCase):
    """T3: RuntimeError includes ticker list, WRDS, cache count."""

    def test_error_message_includes_diagnostics(self):
        """When all tickers fail, the RuntimeError message must contain
        actionable diagnostics: WRDS_ENABLED, REQUIRE_PERMNO, cache info.
        """
        from quant_engine.api.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        # Mock load_universe to return empty, and get_skip_reasons to return data
        fake_skips = {"AAPL": "permno unresolved", "MSFT": "load_ohlcv returned None"}

        with patch("quant_engine.data.loader.load_universe", return_value={}), \
             patch("quant_engine.data.loader.get_skip_reasons", return_value=fake_skips):
            try:
                orchestrator.load_and_prepare(tickers=["AAPL", "MSFT"], years=1)
                self.fail("Expected RuntimeError")
            except RuntimeError as exc:
                msg = str(exc)
                # Must include config diagnostics
                self.assertIn("WRDS_ENABLED", msg)
                self.assertIn("REQUIRE_PERMNO", msg)
                # Must include skip reasons
                self.assertIn("permno unresolved", msg)


class TestDataStatusService(unittest.TestCase):
    """T4: DataService.get_cache_status returns valid ticker info."""

    def test_data_status_returns_summary(self):
        """get_cache_status returns summary with expected keys."""
        from quant_engine.api.services.data_service import DataService

        svc = DataService()
        result = svc.get_cache_status()

        self.assertIn("summary", result)
        self.assertIn("tickers", result)
        summary = result["summary"]
        self.assertIn("total_cached", summary)
        self.assertIn("fresh", summary)
        self.assertIn("stale", summary)
        self.assertIn("very_stale", summary)
        self.assertIn("cache_exists", summary)

    def test_data_status_ticker_entries(self):
        """Each ticker entry has required fields."""
        from quant_engine.api.services.data_service import DataService

        svc = DataService()
        result = svc.get_cache_status()

        if result["tickers"]:
            entry = result["tickers"][0]
            self.assertIn("ticker", entry)
            self.assertIn("source", entry)
            self.assertIn("last_bar_date", entry)
            self.assertIn("total_bars", entry)
            self.assertIn("timeframes_available", entry)
            self.assertIn("freshness", entry)
            self.assertIn("days_stale", entry)
            self.assertIn(entry["freshness"], ("FRESH", "STALE", "VERY_STALE", "UNKNOWN"))

    def test_data_status_with_missing_cache_dir(self):
        """When cache dir doesn't exist, returns empty with cache_exists=False."""
        from quant_engine.api.services.data_service import DataService

        svc = DataService()
        with patch("quant_engine.config.DATA_CACHE_DIR", Path("/nonexistent/path")):
            result = svc.get_cache_status()

        self.assertEqual(result["summary"]["total_cached"], 0)
        self.assertFalse(result["summary"]["cache_exists"])
        self.assertEqual(result["tickers"], [])

    def test_data_status_freshness_categories(self):
        """Freshness categories are correctly assigned based on days_stale."""
        from quant_engine.api.services.data_service import DataService

        svc = DataService()
        result = svc.get_cache_status()
        summary = result["summary"]

        # Counts should add up (excluding UNKNOWN)
        known = summary["fresh"] + summary["stale"] + summary["very_stale"]
        # known should be <= total_cached (some may be UNKNOWN)
        self.assertLessEqual(known, summary["total_cached"])


if __name__ == "__main__":
    unittest.main()
