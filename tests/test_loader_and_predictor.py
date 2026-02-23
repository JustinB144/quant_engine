"""
Test module for loader and predictor behavior and regressions.
"""

import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from quant_engine.data.loader import (
    load_ohlcv,
    load_survivorship_universe,
    load_with_delistings,
)
from quant_engine.data.local_cache import load_ohlcv as cache_load
from quant_engine.data.local_cache import load_ohlcv_with_meta as cache_load_with_meta
from quant_engine.data.local_cache import save_ohlcv as cache_save
from quant_engine.models.predictor import EnsemblePredictor


class _FakeWRDSProvider:
    """Test double used to isolate behavior in this test module."""
    def available(self):
        return True

    def get_crsp_prices(self, tickers, start_date=None, end_date=None):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        return {
            "10001": pd.DataFrame(
                {
                    "Open": [99.0, 100.0, 101.0],
                    "High": [101.0, 102.0, 103.0],
                    "Low": [98.0, 99.0, 100.0],
                    "Close": [100.0, 101.0, 102.0],
                    "Volume": [1_000.0, 1_100.0, 1_200.0],
                    "Return": [0.01, 0.01, 0.01],
                    "total_ret": [0.01, 0.01, 0.01],
                    "dlret": [None, None, None],
                    "delist_event": [0, 0, 0],
                    "permno": ["10001", "10001", "10001"],
                    "ticker": ["AAPL", "AAPL", "AAPL"],
                },
                index=idx,
            ),
        }

    def get_crsp_prices_with_delistings(self, tickers, start_date=None, end_date=None):
        idx = pd.date_range("2022-01-01", periods=520, freq="D")
        close = [100.0 + i * 0.1 for i in range(520)]
        ret = [0.0] * 519 + [-0.05]
        total_ret = [0.0] * 519 + [((1.0 - 0.05) * (1.0 - 0.4) - 1.0)]
        return {
            "10001": pd.DataFrame(
                {
                    "Open": close,
                    "High": close,
                    "Low": close,
                    "Close": close,
                    "Volume": [1000.0] * 520,
                    "Return": ret,
                    "total_ret": total_ret,
                    "dlret": [None] * 519 + [-0.4],
                    "delist_event": [0] * 519 + [1],
                    "permno": ["10001"] * 520,
                    "ticker": ["AAPL"] * 520,
                },
                index=idx,
            ),
        }

    def get_option_surface_features(self, permnos=None, start_date=None, end_date=None):
        return {}

    def resolve_permno(self, ticker, as_of_date=None):
        return "10001"


class _UnavailableWRDSProvider:
    """Test double representing an unavailable dependency or provider."""
    def available(self):
        return False


class LoaderAndPredictorTests(unittest.TestCase):
    """Test cases covering loader and predictor behavior and system invariants."""
    def test_load_ohlcv_uses_wrds_contract_and_stable_columns(self):
        provider = _FakeWRDSProvider()
        with patch("quant_engine.data.wrds_provider.WRDSProvider", return_value=provider):
            out = load_ohlcv("AAPL", years=1, use_cache=False, use_wrds=True)
        self.assertIsNotNone(out)
        self.assertTrue({"Open", "High", "Low", "Close", "Volume", "Return", "total_ret"}.issubset(set(out.columns)))
        self.assertEqual(out.attrs.get("permno"), "10001")
        self.assertEqual(len(out), 3)

    def test_load_with_delistings_applies_delisting_return(self):
        provider = _FakeWRDSProvider()
        with patch("quant_engine.data.wrds_provider.WRDSProvider", return_value=provider):
            with patch("quant_engine.data.loader.cache_save"):
                out = load_with_delistings(["AAPL"], years=1, verbose=False)
        self.assertIn("10001", out)
        df = out["10001"]
        self.assertAlmostEqual(float(df["Close"].iloc[-1]), 151.9, places=6)
        self.assertAlmostEqual(float(df["total_ret"].iloc[-1]), -0.43, places=6)
        self.assertEqual(int(df["delist_event"].iloc[-1]), 1)

    def test_predictor_explicit_version_does_not_silently_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                EnsemblePredictor(horizon=10, model_dir=tmp, version="20990101_000000")

    def test_cache_load_reads_daily_csv_when_parquet_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = f"{tmp}/AAPL_daily_2015-01-01_2025-12-31.csv"
            pd.DataFrame(
                {
                    "Date": pd.date_range("2024-01-01", periods=5, freq="B"),
                    "Open": [100, 101, 102, 103, 104],
                    "High": [101, 102, 103, 104, 105],
                    "Low": [99, 100, 101, 102, 103],
                    "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
                    "Volume": [1_000, 1_100, 1_200, 1_300, 1_400],
                },
            ).to_csv(csv_path, index=False)

            out = cache_load("AAPL", cache_dir=tmp)
            self.assertIsNotNone(out)
            self.assertEqual(list(out.columns), ["Open", "High", "Low", "Close", "Volume"])
            self.assertEqual(len(out), 5)

    def test_cache_save_falls_back_to_csv_without_parquet_engine(self):
        df = pd.DataFrame(
            {
                "Open": [10.0, 11.0, 12.0],
                "High": [11.0, 12.0, 13.0],
                "Low": [9.0, 10.0, 11.0],
                "Close": [10.5, 11.5, 12.5],
                "Volume": [100.0, 110.0, 120.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="B"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(pd.DataFrame, "to_parquet", side_effect=ImportError("no parquet engine")):
                path = cache_save("MSFT", df, cache_dir=tmp)
            self.assertTrue(str(path).endswith("MSFT_1d.csv"))
            self.assertTrue(path.exists())
            round_trip = cache_load("MSFT", cache_dir=tmp)
            self.assertIsNotNone(round_trip)
            self.assertEqual(len(round_trip), 3)
            _, meta, _ = cache_load_with_meta("MSFT", cache_dir=tmp)
            self.assertEqual(meta.get("source"), "unknown")

    def test_trusted_wrds_cache_short_circuits_live_wrds(self):
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=2000, freq="B")
        cached = pd.DataFrame(
            {
                "Open": [100.0] * len(idx),
                "High": [101.0] * len(idx),
                "Low": [99.0] * len(idx),
                "Close": [100.0] * len(idx),
                "Volume": [1_000_000.0] * len(idx),
            },
            index=idx,
        )
        with patch(
            "quant_engine.data.loader.cache_load_with_meta",
            return_value=(cached, {"source": "wrds", "permno": "10001", "ticker": "AAPL"}, None),
        ):
            with patch("quant_engine.data.wrds_provider.WRDSProvider", side_effect=AssertionError("WRDS should not be called")):
                out = load_ohlcv("AAPL", years=5, use_cache=True, use_wrds=True)
        self.assertIsNotNone(out)
        self.assertEqual(len(out), len(cached))

    def test_untrusted_cache_refreshes_from_wrds_and_sets_wrds_source(self):
        idx_cache = pd.date_range("2019-01-01", periods=800, freq="B")
        cached = pd.DataFrame(
            {
                "Open": [90.0] * len(idx_cache),
                "High": [91.0] * len(idx_cache),
                "Low": [89.0] * len(idx_cache),
                "Close": [90.0] * len(idx_cache),
                "Volume": [900_000.0] * len(idx_cache),
            },
            index=idx_cache,
        )
        provider = _FakeWRDSProvider()
        with patch(
            "quant_engine.data.loader.cache_load_with_meta",
            return_value=(cached, {"source": "ibkr", "ticker": "AAPL"}, None),
        ):
            with patch("quant_engine.data.wrds_provider.WRDSProvider", return_value=provider):
                with patch("quant_engine.data.loader.cache_save") as mock_cache_save:
                    out = load_ohlcv("AAPL", years=1, use_cache=True, use_wrds=True)
        self.assertIsNotNone(out)
        self.assertEqual(len(out), 3)
        self.assertEqual(mock_cache_save.call_args.kwargs.get("source"), "wrds")

    def test_survivorship_fallback_prefers_cached_subset_when_wrds_unavailable(self):
        provider = _UnavailableWRDSProvider()
        long_idx = pd.date_range("2008-01-01", periods=4500, freq="B")
        long_df = pd.DataFrame(
            {
                "Open": [100.0] * len(long_idx),
                "High": [101.0] * len(long_idx),
                "Low": [99.0] * len(long_idx),
                "Close": [100.0] * len(long_idx),
                "Volume": [1_000_000.0] * len(long_idx),
            },
            index=long_idx,
        )

        def _fake_cache_load_with_meta(ticker, cache_dir=None):
            if ticker in {"AAPL", "MSFT"}:
                permno = "10001" if ticker == "AAPL" else "10002"
                return long_df, {"source": "ibkr", "permno": permno, "ticker": ticker}, None
            return None, {}, None

        with patch("quant_engine.data.wrds_provider.WRDSProvider", return_value=provider):
            with patch("quant_engine.data.loader.list_cached_tickers", return_value=["AAPL", "MSFT", "QQQ"]):
                with patch("quant_engine.data.loader.cache_load_with_meta", side_effect=_fake_cache_load_with_meta):
                    with patch("quant_engine.data.loader.load_universe", return_value={}) as mock_load_universe:
                        load_survivorship_universe(years=1, verbose=False)

        args, kwargs = mock_load_universe.call_args
        self.assertEqual(args[0], ["AAPL", "MSFT"])
        self.assertEqual(kwargs["years"], 1)


if __name__ == "__main__":
    unittest.main()
