"""
Tests for Spec W08: Permanent cache for delisted stock data.

Covers:
- save_ohlcv auto-detection of terminal delisting events
- _cache_is_usable terminal fast-path bypass of staleness checks
- backfill_terminal_metadata scanning and metadata updates
- _build_ticker_list skip-terminal filtering
- Edge cases: missing delist_event column, caller-supplied meta, idempotency
"""

import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from quant_engine.data.local_cache import (
    _read_cache_meta,
    _write_cache_meta,
    backfill_terminal_metadata,
    load_ohlcv_with_meta,
    save_ohlcv,
)
from quant_engine.data.loader import _cache_is_usable


def _make_ohlcv(n_bars=600, start="2020-01-01", with_delist=False, delist_idx=-1):
    """Build a synthetic OHLCV DataFrame, optionally with a terminal delist event."""
    idx = pd.bdate_range(start, periods=n_bars)
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n_bars))
    close = np.maximum(close, 1.0)
    df = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(100_000, 5_000_000, n_bars).astype(float),
            "Return": np.concatenate([[0.0], np.diff(close) / close[:-1]]),
            "total_ret": np.concatenate([[0.0], np.diff(close) / close[:-1]]),
            "delist_event": np.zeros(n_bars, dtype=int),
        },
        index=idx,
    )
    if with_delist:
        df.iloc[delist_idx, df.columns.get_loc("delist_event")] = 1
        df.iloc[delist_idx, df.columns.get_loc("total_ret")] = -0.95
    return df


class TestSaveOhlcvTerminalDetection(unittest.TestCase):
    """Part A: save_ohlcv should auto-detect delist_event and set is_terminal."""

    def test_save_sets_terminal_metadata_for_delisted_stock(self):
        df = _make_ohlcv(with_delist=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_ohlcv("DEAD", df, cache_dir=tmp, source="wrds_delisting")
            meta = _read_cache_meta(path, "DEAD")
        self.assertTrue(meta.get("is_terminal"))
        self.assertEqual(meta.get("terminal_reason"), "delisted")
        self.assertIsNotNone(meta.get("terminal_date"))

    def test_save_does_not_set_terminal_for_active_stock(self):
        df = _make_ohlcv(with_delist=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_ohlcv("LIVE", df, cache_dir=tmp, source="wrds")
            meta = _read_cache_meta(path, "LIVE")
        self.assertNotIn("is_terminal", meta)
        self.assertNotIn("terminal_date", meta)

    def test_save_preserves_caller_meta_alongside_terminal(self):
        df = _make_ohlcv(with_delist=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_ohlcv(
                "DEAD2", df, cache_dir=tmp, source="wrds_delisting",
                meta={"permno": "99999", "ticker": "DEAD2"},
            )
            meta = _read_cache_meta(path, "DEAD2")
        self.assertTrue(meta.get("is_terminal"))
        self.assertEqual(meta.get("permno"), "99999")
        self.assertEqual(meta.get("ticker"), "DEAD2")

    def test_save_uses_setdefault_for_terminal_fields(self):
        """If the caller already provides is_terminal=False, save_ohlcv
        respects the caller's value (setdefault semantics)."""
        df = _make_ohlcv(with_delist=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_ohlcv(
                "OVERRIDE", df, cache_dir=tmp, source="wrds_delisting",
                meta={"is_terminal": False},
            )
            meta = _read_cache_meta(path, "OVERRIDE")
        # Caller explicitly set is_terminal=False — setdefault preserves it
        self.assertFalse(meta.get("is_terminal"))


class TestCacheIsUsableTerminalFastPath(unittest.TestCase):
    """Part B: _cache_is_usable should bypass staleness for terminal entries."""

    def _stale_cached_df(self):
        """Build a cached DataFrame that is stale (last bar > 60 trading days ago)."""
        idx = pd.bdate_range("2020-01-01", periods=600)
        rng = np.random.default_rng(7)
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, 600))
        close = np.maximum(close, 1.0)
        df = pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": rng.integers(100_000, 5_000_000, 600).astype(float),
            },
            index=idx,
        )
        return df

    def test_stale_non_terminal_fails_require_recent(self):
        cached = self._stale_cached_df()
        meta = {"source": "wrds"}
        result = _cache_is_usable(
            cached=cached, meta=meta, years=5,
            require_recent=True, require_trusted=True,
        )
        self.assertFalse(result)

    def test_stale_terminal_meta_bypasses_staleness(self):
        cached = self._stale_cached_df()
        meta = {"source": "wrds", "is_terminal": True}
        result = _cache_is_usable(
            cached=cached, meta=meta, years=5,
            require_recent=True, require_trusted=True,
        )
        self.assertTrue(result)

    def test_stale_terminal_data_bypasses_staleness(self):
        """Even without is_terminal in meta, delist_event==1 in data triggers bypass."""
        cached = self._stale_cached_df()
        cached["delist_event"] = 0
        cached.iloc[-1, cached.columns.get_loc("delist_event")] = 1
        meta = {"source": "wrds"}
        result = _cache_is_usable(
            cached=cached, meta=meta, years=5,
            require_recent=True, require_trusted=True,
        )
        self.assertTrue(result)

    def test_terminal_still_requires_min_bars(self):
        """Terminal fast-path doesn't bypass the MIN_BARS check."""
        idx = pd.bdate_range("2020-01-01", periods=5)
        df = pd.DataFrame(
            {
                "Open": [100.0] * 5,
                "High": [101.0] * 5,
                "Low": [99.0] * 5,
                "Close": [100.0] * 5,
                "Volume": [1_000_000.0] * 5,
            },
            index=idx,
        )
        meta = {"source": "wrds", "is_terminal": True}
        result = _cache_is_usable(
            cached=df, meta=meta, years=1,
            require_recent=True, require_trusted=True,
        )
        self.assertFalse(result)

    def test_terminal_still_requires_trusted_source(self):
        """Terminal entries still need a trusted source when require_trusted=True."""
        cached = self._stale_cached_df()
        meta = {"source": "unknown", "is_terminal": True}
        result = _cache_is_usable(
            cached=cached, meta=meta, years=5,
            require_recent=True, require_trusted=True,
        )
        self.assertFalse(result)


class TestBackfillTerminalMetadata(unittest.TestCase):
    """Part D: backfill_terminal_metadata scans parquets and sets is_terminal."""

    def test_backfill_marks_delisted_parquet(self):
        df = _make_ohlcv(with_delist=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "DEAD_1d.parquet"
            df.to_parquet(path)
            _write_cache_meta(path, ticker="DEAD", df=df, source="wrds_delisting")

            summary = backfill_terminal_metadata(cache_dir=tmp)
            self.assertEqual(summary["scanned"], 1)
            self.assertEqual(summary["updated"], 1)
            self.assertEqual(summary["active"], 0)

            meta = _read_cache_meta(path, "DEAD")
            self.assertTrue(meta.get("is_terminal"))
            self.assertIsNotNone(meta.get("terminal_date"))

    def test_backfill_skips_active_stock(self):
        df = _make_ohlcv(with_delist=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "LIVE_1d.parquet"
            df.to_parquet(path)
            _write_cache_meta(path, ticker="LIVE", df=df, source="wrds")

            summary = backfill_terminal_metadata(cache_dir=tmp)
            self.assertEqual(summary["active"], 1)
            self.assertEqual(summary["updated"], 0)

    def test_backfill_skips_already_terminal(self):
        df = _make_ohlcv(with_delist=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "DEAD_1d.parquet"
            df.to_parquet(path)
            _write_cache_meta(
                path, ticker="DEAD", df=df, source="wrds_delisting",
                meta={"is_terminal": True, "terminal_date": "2022-05-01"},
            )

            summary = backfill_terminal_metadata(cache_dir=tmp)
            self.assertEqual(summary["already_terminal"], 1)
            self.assertEqual(summary["updated"], 0)

    def test_backfill_dry_run_does_not_write(self):
        df = _make_ohlcv(with_delist=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "DEAD_1d.parquet"
            df.to_parquet(path)
            _write_cache_meta(path, ticker="DEAD", df=df, source="wrds")

            summary = backfill_terminal_metadata(cache_dir=tmp, dry_run=True)
            self.assertEqual(summary["updated"], 1)

            # Meta should NOT have is_terminal because dry_run=True
            meta = _read_cache_meta(path, "DEAD")
            self.assertNotEqual(meta.get("is_terminal"), True)

    def test_backfill_idempotent(self):
        """Running backfill twice yields same result."""
        df = _make_ohlcv(with_delist=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "DEAD_1d.parquet"
            df.to_parquet(path)
            _write_cache_meta(path, ticker="DEAD", df=df, source="wrds")

            s1 = backfill_terminal_metadata(cache_dir=tmp)
            s2 = backfill_terminal_metadata(cache_dir=tmp)
            self.assertEqual(s1["updated"], 1)
            self.assertEqual(s2["already_terminal"], 1)
            self.assertEqual(s2["updated"], 0)


class TestBuildTickerListSkipTerminal(unittest.TestCase):
    """Part C: _build_ticker_list excludes terminal tickers when skip_terminal=True."""

    def test_skip_terminal_excludes_delisted_tickers(self):
        """Terminal tickers are excluded from the ticker list."""
        from run_wrds_daily_refresh import _build_ticker_list

        active_df = _make_ohlcv(with_delist=False)
        dead_df = _make_ohlcv(with_delist=True)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Save both active and delisted
            save_ohlcv("ACTIVE", active_df, cache_dir=tmp, source="wrds")
            save_ohlcv("DEAD", dead_df, cache_dir=tmp, source="wrds_delisting")

            with patch("run_wrds_daily_refresh.list_cached_tickers", return_value=["ACTIVE", "DEAD"]):
                with patch("run_wrds_daily_refresh.UNIVERSE_FULL", []):
                    with patch("run_wrds_daily_refresh.BENCHMARK", "SPY"):
                        with patch("run_wrds_daily_refresh.load_ohlcv_with_meta") as mock_load:
                            def _fake_load(ticker, **kw):
                                if ticker == "DEAD":
                                    return None, {"is_terminal": True}, None
                                return None, {"source": "wrds"}, None

                            mock_load.side_effect = _fake_load
                            result = _build_ticker_list(None, skip_terminal=True)

            self.assertNotIn("DEAD", result)
            self.assertIn("ACTIVE", result)

    def test_include_terminal_includes_all(self):
        """With skip_terminal=False, terminal tickers are included."""
        from run_wrds_daily_refresh import _build_ticker_list

        with patch("run_wrds_daily_refresh.list_cached_tickers", return_value=["ACTIVE", "DEAD"]):
            with patch("run_wrds_daily_refresh.UNIVERSE_FULL", []):
                with patch("run_wrds_daily_refresh.BENCHMARK", "SPY"):
                    result = _build_ticker_list(None, skip_terminal=False)

        self.assertIn("ACTIVE", result)
        self.assertIn("DEAD", result)

    def test_explicit_tickers_arg_ignores_skip_terminal(self):
        """When --tickers is provided, skip_terminal is irrelevant."""
        from run_wrds_daily_refresh import _build_ticker_list

        result = _build_ticker_list("DEAD,ALIVE", skip_terminal=True)
        self.assertEqual(result, ["DEAD", "ALIVE"])


class TestTerminalCacheEndToEnd(unittest.TestCase):
    """End-to-end: save -> load -> verify terminal round-trip."""

    def test_save_load_round_trip_preserves_terminal(self):
        df = _make_ohlcv(with_delist=True)
        with tempfile.TemporaryDirectory() as tmp:
            save_ohlcv(
                "BANKRUPT", df, cache_dir=tmp, source="wrds_delisting",
                meta={"permno": "99999", "ticker": "BANKRUPT"},
            )
            loaded_df, meta, loaded_path = load_ohlcv_with_meta("BANKRUPT", cache_dir=tmp)

        self.assertIsNotNone(loaded_df)
        self.assertTrue(meta.get("is_terminal"))
        self.assertEqual(meta.get("terminal_reason"), "delisted")

    def test_terminal_cache_usable_as_trusted_despite_staleness(self):
        """A terminal entry from 2020 should still be usable in 2026."""
        df = _make_ohlcv(n_bars=600, start="2018-01-01", with_delist=True, delist_idx=-1)
        with tempfile.TemporaryDirectory() as tmp:
            save_ohlcv(
                "OLDCORP", df, cache_dir=tmp, source="wrds_delisting",
                meta={"permno": "88888"},
            )
            loaded_df, meta, _ = load_ohlcv_with_meta("OLDCORP", cache_dir=tmp)

        self.assertIsNotNone(loaded_df)
        usable = _cache_is_usable(
            cached=loaded_df, meta=meta, years=5,
            require_recent=True, require_trusted=True,
        )
        self.assertTrue(usable)


if __name__ == "__main__":
    unittest.main()
