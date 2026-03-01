"""
Tests for SPEC_AUDIT_FIX_18: Cache Integrity & Reliability Fixes.

Covers:
- T1: Anchored OHLCV column matching (no false matches on substrings)
- T2: Protected metadata fields cannot be overwritten by caller
- T3: Atomic pair write (data + metadata written together)
- T4: Cache error logging at WARNING level
- T5: Intraday fallback directory search
"""

import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from quant_engine.data.local_cache import (
    _OHLCV_PATTERNS,
    _PROTECTED_META_KEYS,
    _atomic_pair_write,
    _build_meta_payload,
    _normalize_ohlcv_columns,
    _write_cache_meta,
    load_intraday_ohlcv,
    load_ohlcv_with_meta,
    save_ohlcv,
)


def _make_ohlcv(n_bars: int = 300, start: str = "2020-01-01") -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame for testing."""
    idx = pd.bdate_range(start, periods=n_bars)
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n_bars))
    close = np.maximum(close, 1.0)
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(100_000, 5_000_000, n_bars).astype(float),
        },
        index=idx,
    )


# -----------------------------------------------------------------------
# T1: Anchored OHLCV column matching
# -----------------------------------------------------------------------


class TestAnchoredColumnMatching(unittest.TestCase):
    """T1: _normalize_ohlcv_columns should not match OHLCV substrings in unrelated columns."""

    def test_standard_columns_mapped(self):
        """Standard column names are correctly mapped."""
        df = pd.DataFrame(
            {"Open": [1], "High": [2], "Low": [3], "Close": [4], "Volume": [5]}
        )
        result = _normalize_ohlcv_columns(df)
        self.assertEqual(list(result.columns), ["Open", "High", "Low", "Close", "Volume"])

    def test_case_insensitive_mapping(self):
        """Case-insensitive column names are correctly mapped."""
        df = pd.DataFrame(
            {"open": [1], "HIGH": [2], "lOw": [3], "CLOSE": [4], "volume": [5]}
        )
        result = _normalize_ohlcv_columns(df)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            self.assertIn(col, result.columns)

    def test_alpha_vantage_numbered_columns(self):
        """Alpha Vantage '1. open' style columns are matched."""
        df = pd.DataFrame(
            {
                "1. open": [1], "2. high": [2], "3. low": [3],
                "4. close": [4], "5. volume": [5],
            }
        )
        result = _normalize_ohlcv_columns(df)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            self.assertIn(col, result.columns)

    def test_adj_columns_mapped(self):
        """Adj-prefixed column names are correctly mapped."""
        df = pd.DataFrame(
            {"adj_open": [1], "adj_high": [2], "adj_low": [3], "adj_close": [4], "volume": [5]}
        )
        result = _normalize_ohlcv_columns(df)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            self.assertIn(col, result.columns)

    def test_following_not_mapped_to_low(self):
        """'following' column must NOT be mapped to 'Low'."""
        df = pd.DataFrame(
            {"Open": [1], "High": [2], "Low": [3], "Close": [4],
             "Volume": [5], "following": [99]}
        )
        result = _normalize_ohlcv_columns(df)
        self.assertIn("following", result.columns)
        # Verify Low is correctly mapped from the actual Low column
        self.assertIn("Low", result.columns)

    def test_shallow_not_mapped_to_low(self):
        """'shallow' column must NOT be mapped to 'Low'."""
        df = pd.DataFrame(
            {"Open": [1], "High": [2], "Low": [3], "Close": [4],
             "Volume": [5], "shallow": [99]}
        )
        result = _normalize_ohlcv_columns(df)
        self.assertIn("shallow", result.columns)

    def test_highlight_not_mapped_to_high(self):
        """'highlight' column must NOT be mapped to 'High'."""
        df = pd.DataFrame(
            {"Open": [1], "High": [2], "Low": [3], "Close": [4],
             "Volume": [5], "highlight": [99]}
        )
        result = _normalize_ohlcv_columns(df)
        self.assertIn("highlight", result.columns)

    def test_reopen_not_mapped_to_open(self):
        """'reopen' column must NOT be mapped to 'Open'."""
        df = pd.DataFrame(
            {"Open": [1], "High": [2], "Low": [3], "Close": [4],
             "Volume": [5], "reopen": [99]}
        )
        result = _normalize_ohlcv_columns(df)
        self.assertIn("reopen", result.columns)

    def test_below_not_mapped_to_low(self):
        """'below' column must NOT be mapped to 'Low'."""
        df = pd.DataFrame(
            {"Open": [1], "High": [2], "Low": [3], "Close": [4],
             "Volume": [5], "below": [99]}
        )
        result = _normalize_ohlcv_columns(df)
        self.assertIn("below", result.columns)

    def test_ambiguous_columns_no_false_matches(self):
        """Multiple ambiguous columns should not produce false OHLCV mappings."""
        df = pd.DataFrame(
            {
                "Open": [1], "High": [2], "Low": [3], "Close": [4], "Volume": [5],
                "following": [6], "shallow": [7], "highlight": [8],
                "reopen": [9], "below": [10], "flow": [11], "enclosure": [12],
            }
        )
        result = _normalize_ohlcv_columns(df)
        # Ensure the real OHLCV columns are mapped
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            self.assertIn(col, result.columns)
        # Ensure ambiguous columns are NOT remapped
        for col in ["following", "shallow", "highlight", "reopen", "below", "flow", "enclosure"]:
            self.assertIn(col, result.columns)

    def test_extra_columns_preserved(self):
        """Return, dlret, delist_event, permno, ticker are correctly mapped."""
        df = pd.DataFrame(
            {
                "Open": [1], "High": [2], "Low": [3], "Close": [4],
                "Volume": [5], "return": [0.01], "dlret": [-0.5],
                "delist_event": [0], "permno": [12345], "tic": ["AAPL"],
            }
        )
        result = _normalize_ohlcv_columns(df)
        self.assertIn("Return", result.columns)
        self.assertIn("dlret", result.columns)
        self.assertIn("delist_event", result.columns)
        self.assertIn("permno", result.columns)
        self.assertIn("ticker", result.columns)

    def test_ohlcv_like_warning_logged(self):
        """Unrecognized OHLCV-like columns should trigger a warning."""
        df = pd.DataFrame(
            {"Open": [1], "High": [2], "Low": [3], "Close": [4],
             "Volume": [5], "preclose": [99]}
        )
        with self.assertLogs("quant_engine.data.local_cache", level="WARNING") as cm:
            _normalize_ohlcv_columns(df)
        self.assertTrue(any("preclose" in msg for msg in cm.output))


# -----------------------------------------------------------------------
# T2: Protected metadata fields
# -----------------------------------------------------------------------


class TestProtectedMetadataFields(unittest.TestCase):
    """T2: Caller metadata cannot overwrite core provenance fields."""

    def test_source_not_overwritten_by_caller(self):
        """Caller meta with 'source' key should be ignored."""
        df = _make_ohlcv()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_ohlcv(
                "AAPL", df, cache_dir=tmp, source="wrds",
                meta={"source": "fake"},
            )
            meta_path = Path(tmp) / "AAPL_1d.meta.json"
            with open(meta_path) as f:
                meta = json.load(f)
            self.assertEqual(meta["source"], "wrds")

    def test_ticker_not_overwritten_by_caller(self):
        """Caller meta with 'ticker' key should be ignored."""
        df = _make_ohlcv()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_ohlcv(
                "AAPL", df, cache_dir=tmp, source="wrds",
                meta={"ticker": "FAKE"},
            )
            meta_path = Path(tmp) / "AAPL_1d.meta.json"
            with open(meta_path) as f:
                meta = json.load(f)
            self.assertEqual(meta["ticker"], "AAPL")

    def test_n_bars_not_overwritten_by_caller(self):
        """Caller meta with 'n_bars' key should be ignored."""
        df = _make_ohlcv(n_bars=300)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_ohlcv(
                "AAPL", df, cache_dir=tmp, source="wrds",
                meta={"n_bars": 999999},
            )
            meta_path = Path(tmp) / "AAPL_1d.meta.json"
            with open(meta_path) as f:
                meta = json.load(f)
            self.assertEqual(meta["n_bars"], 300)

    def test_warning_logged_for_protected_keys(self):
        """A warning should be logged when caller tries to set protected keys."""
        df = _make_ohlcv()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertLogs("quant_engine.data.local_cache", level="WARNING") as cm:
                save_ohlcv(
                    "WARN", df, cache_dir=tmp, source="wrds",
                    meta={"source": "fake", "n_bars": 1},
                )
            self.assertTrue(any("protected keys" in msg for msg in cm.output))

    def test_non_protected_caller_meta_preserved(self):
        """Non-protected caller meta keys (e.g. permno, custom) should be preserved."""
        df = _make_ohlcv()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_ohlcv(
                "AAPL", df, cache_dir=tmp, source="wrds",
                meta={"permno": "12345", "custom_field": "hello"},
            )
            meta_path = Path(tmp) / "AAPL_1d.meta.json"
            with open(meta_path) as f:
                meta = json.load(f)
            self.assertEqual(meta["permno"], "12345")
            self.assertEqual(meta["custom_field"], "hello")

    def test_build_meta_payload_protects_fields(self):
        """_build_meta_payload should filter protected keys."""
        df = _make_ohlcv()
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "TEST_1d.parquet"
            payload = _build_meta_payload(
                data_path, ticker="TEST", df=df, source="wrds",
                meta={"source": "evil", "permno": "99"},
            )
        self.assertEqual(payload["source"], "wrds")
        self.assertEqual(payload["permno"], "99")


# -----------------------------------------------------------------------
# T3: Atomic pair write
# -----------------------------------------------------------------------


class TestAtomicPairWrite(unittest.TestCase):
    """T3: Data + metadata are written as an atomic pair."""

    def test_atomic_pair_write_creates_both_files(self):
        """Both parquet and meta files should be created."""
        df = _make_ohlcv(n_bars=50)
        with tempfile.TemporaryDirectory() as tmp:
            pq = Path(tmp) / "TEST_1d.parquet"
            meta = Path(tmp) / "TEST_1d.meta.json"
            payload = {"ticker": "TEST", "source": "test", "n_bars": 50}
            _atomic_pair_write(pq, meta, df, payload)
            self.assertTrue(pq.exists())
            self.assertTrue(meta.exists())
            loaded = pd.read_parquet(pq)
            self.assertEqual(len(loaded), 50)
            with open(meta) as f:
                meta_data = json.load(f)
            self.assertEqual(meta_data["ticker"], "TEST")

    def test_atomic_pair_write_no_partial_on_failure(self):
        """If parquet write fails, neither file should be created."""
        with tempfile.TemporaryDirectory() as tmp:
            pq = Path(tmp) / "FAIL_1d.parquet"
            meta = Path(tmp) / "FAIL_1d.meta.json"
            # Pass a non-DataFrame to trigger an error
            with self.assertRaises(Exception):
                _atomic_pair_write(pq, meta, "not_a_dataframe", {"ticker": "FAIL"})
            self.assertFalse(pq.exists())
            self.assertFalse(meta.exists())

    def test_save_ohlcv_uses_atomic_pair_write(self):
        """save_ohlcv should produce both parquet and meta atomically."""
        df = _make_ohlcv()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_ohlcv("PAIR", df, cache_dir=tmp, source="test")
            pq = Path(tmp) / "PAIR_1d.parquet"
            meta = Path(tmp) / "PAIR_1d.meta.json"
            self.assertTrue(pq.exists())
            self.assertTrue(meta.exists())

    def test_orphaned_data_logs_warning(self):
        """Loading a parquet with no metadata should log a warning."""
        df = _make_ohlcv()
        with tempfile.TemporaryDirectory() as tmp:
            pq = Path(tmp) / "ORPHAN_1d.parquet"
            df.to_parquet(pq)
            # No metadata sidecar written
            with self.assertLogs("quant_engine.data.local_cache", level="WARNING") as cm:
                loaded, meta, path = load_ohlcv_with_meta("ORPHAN", cache_dir=tmp)
            self.assertIsNotNone(loaded)
            self.assertEqual(meta, {})
            self.assertTrue(any("Orphaned" in msg or "orphan" in msg.lower() for msg in cm.output))


# -----------------------------------------------------------------------
# T4: Cache error logging at WARNING level
# -----------------------------------------------------------------------


class TestCacheErrorLogging(unittest.TestCase):
    """T4: Error conditions are logged at WARNING level."""

    def test_corrupt_parquet_logs_warning(self):
        """Corrupt parquet files should log WARNING, not DEBUG."""
        with tempfile.TemporaryDirectory() as tmp:
            pq = Path(tmp) / "CORRUPT_1d.parquet"
            pq.write_text("this is not a parquet file")
            with self.assertLogs("quant_engine.data.local_cache", level="WARNING") as cm:
                loaded, meta, path = load_ohlcv_with_meta("CORRUPT", cache_dir=tmp)
            self.assertIsNone(loaded)
            self.assertTrue(any("Could not read parquet" in msg for msg in cm.output))

    def test_write_meta_failure_logs_warning(self):
        """Failed metadata writes should log WARNING."""
        df = _make_ohlcv()
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "FAILMETA_1d.parquet"
            # Make the dir read-only so meta write fails
            ro_dir = Path(tmp) / "readonly"
            ro_dir.mkdir()
            data_path_ro = ro_dir / "FAILMETA_1d.parquet"
            os.chmod(str(ro_dir), 0o555)
            try:
                with self.assertLogs("quant_engine.data.local_cache", level="WARNING") as cm:
                    _write_cache_meta(data_path_ro, "FAILMETA", df, "test")
                self.assertTrue(any("Could not write cache meta" in msg for msg in cm.output))
            finally:
                os.chmod(str(ro_dir), 0o755)


# -----------------------------------------------------------------------
# T5: Intraday fallback directory search
# -----------------------------------------------------------------------


class TestIntradayFallbackSearch(unittest.TestCase):
    """T5: Intraday data in fallback 'intraday' subdirectory is discoverable."""

    def test_intraday_data_in_fallback_intraday_subdir(self):
        """Data in a fallback's 'intraday' subdir should be found."""
        df = pd.DataFrame(
            {
                "Open": [100.0] * 100,
                "High": [101.0] * 100,
                "Low": [99.0] * 100,
                "Close": [100.0] * 100,
                "Volume": [1_000_000.0] * 100,
            },
            index=pd.date_range("2024-01-01", periods=100, freq="5min"),
        )
        with tempfile.TemporaryDirectory() as primary, \
             tempfile.TemporaryDirectory() as fallback:
            # Put data in fallback's "intraday" subdir
            intraday_dir = Path(fallback) / "intraday"
            intraday_dir.mkdir()
            parquet_path = intraday_dir / "AAPL_5min_2024-01-01_2024-01-02.parquet"
            df.to_parquet(parquet_path)

            # Patch FALLBACK_SOURCE_DIRS to include our fallback dir
            with patch("quant_engine.data.local_cache.FALLBACK_SOURCE_DIRS", [Path(fallback)]):
                result = load_intraday_ohlcv("AAPL", "5m", cache_dir=primary)
            self.assertIsNotNone(result)
            self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
