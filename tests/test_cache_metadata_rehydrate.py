"""
Test module for cache metadata rehydrate behavior and regressions.
"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant_engine.data.local_cache import rehydrate_cache_metadata


def _write_daily_csv(path: Path):
    """Local helper used by the cache metadata rehydrate tests."""
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=800, freq="B")
    df = pd.DataFrame(
        {
            "Date": idx,
            "Open": [100.0] * len(idx),
            "High": [101.0] * len(idx),
            "Low": [99.0] * len(idx),
            "Close": [100.0] * len(idx),
            "Volume": [1_000_000.0] * len(idx),
        },
    )
    df.to_csv(path, index=False)


class CacheMetadataRehydrateTests(unittest.TestCase):
    """Test cases covering cache metadata rehydrate behavior and system invariants."""
    def test_rehydrate_writes_metadata_for_daily_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "AAPL_daily_2015-01-01_2025-12-31.csv"
            _write_daily_csv(data_path)

            summary = rehydrate_cache_metadata(
                cache_roots=[root],
                source_by_root={str(root): "ibkr"},
                default_source="unknown",
                only_missing=True,
            )
            self.assertEqual(summary["written"], 1)

            meta_path = root / "AAPL_1d.meta.json"
            self.assertTrue(meta_path.exists())
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta.get("source"), "ibkr")
            self.assertEqual(meta.get("ticker"), "AAPL")
            self.assertEqual(meta.get("n_bars"), 800)

    def test_rehydrate_only_missing_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "MSFT_daily_2015-01-01_2025-12-31.csv"
            _write_daily_csv(data_path)
            meta_path = root / "MSFT_1d.meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"ticker": "MSFT", "source": "wrds", "n_bars": 1}, f)

            summary = rehydrate_cache_metadata(
                cache_roots=[root],
                source_by_root={str(root): "ibkr"},
                only_missing=True,
            )
            self.assertEqual(summary["skipped_existing"], 1)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta.get("source"), "wrds")

    def test_rehydrate_force_with_overwrite_source_updates_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "NVDA_daily_2015-01-01_2025-12-31.csv"
            _write_daily_csv(data_path)
            meta_path = root / "NVDA_1d.meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"ticker": "NVDA", "source": "ibkr", "n_bars": 1}, f)

            summary = rehydrate_cache_metadata(
                cache_roots=[root],
                source_by_root={str(root): "ibkr"},
                only_missing=False,
                overwrite_source=True,
            )
            self.assertEqual(summary["written"], 1)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta.get("source"), "ibkr")
            self.assertTrue(bool(meta.get("rehydrated")))


if __name__ == "__main__":
    unittest.main()
