"""
Tests for SPEC-D03: Atomic cache writes to prevent race conditions.

Covers:
- local_cache.py: save_ohlcv, _write_cache_meta use atomic temp+replace
- feature_store.py: save_features uses atomic temp+replace
- Concurrent writes to the same key never produce corrupted files
- Failed writes leave no partial files behind
- Temp file cleanup on error
"""

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from quant_engine.data.local_cache import (
    _atomic_replace,
    _write_cache_meta,
    load_ohlcv,
    load_ohlcv_with_meta,
    save_ohlcv,
)
from quant_engine.data.feature_store import (
    FeatureStore,
    _atomic_replace as fs_atomic_replace,
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


def _make_features(n_rows: int = 100) -> pd.DataFrame:
    """Build a synthetic feature DataFrame for testing."""
    idx = pd.bdate_range("2023-01-01", periods=n_rows)
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "momentum_10": rng.normal(0, 1, n_rows),
            "volatility_20": rng.uniform(0.01, 0.05, n_rows),
            "rsi_14": rng.uniform(20, 80, n_rows),
        },
        index=idx,
    )


# -----------------------------------------------------------------------
# Unit tests: _atomic_replace helper
# -----------------------------------------------------------------------


class TestAtomicReplaceLocalCache(unittest.TestCase):
    """Tests for the _atomic_replace helper in local_cache.py."""

    def test_atomic_replace_creates_file(self):
        """Basic: _atomic_replace should create the target file."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "test.json"

            def write(p):
                with open(p, "w") as f:
                    json.dump({"key": "value"}, f)

            _atomic_replace(target, write)
            self.assertTrue(target.exists())
            with open(target) as f:
                self.assertEqual(json.load(f), {"key": "value"})

    def test_atomic_replace_overwrites_existing(self):
        """_atomic_replace should atomically overwrite an existing file."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "test.json"
            with open(target, "w") as f:
                json.dump({"old": True}, f)

            def write(p):
                with open(p, "w") as f:
                    json.dump({"new": True}, f)

            _atomic_replace(target, write)
            with open(target) as f:
                self.assertEqual(json.load(f), {"new": True})

    def test_atomic_replace_cleans_up_on_error(self):
        """On write failure, no temp file should remain and target is untouched."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "test.json"
            # Pre-populate so we can verify it's untouched
            with open(target, "w") as f:
                json.dump({"original": True}, f)

            def bad_write(p):
                raise IOError("simulated write failure")

            with self.assertRaises(IOError):
                _atomic_replace(target, bad_write)

            # Original file untouched
            with open(target) as f:
                self.assertEqual(json.load(f), {"original": True})

            # No temp files left behind
            tmp_files = [f for f in os.listdir(tmp) if f.endswith(".tmp")]
            self.assertEqual(tmp_files, [])

    def test_atomic_replace_no_partial_on_new_file_error(self):
        """If target doesn't exist and write fails, target is never created."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "should_not_exist.json"

            def bad_write(p):
                # Write partial data then fail
                with open(p, "w") as f:
                    f.write('{"partial":')
                raise IOError("mid-write failure")

            with self.assertRaises(IOError):
                _atomic_replace(target, bad_write)

            self.assertFalse(target.exists())
            tmp_files = [f for f in os.listdir(tmp) if f.endswith(".tmp")]
            self.assertEqual(tmp_files, [])


class TestAtomicReplaceFeatureStore(unittest.TestCase):
    """Tests for the _atomic_replace helper in feature_store.py."""

    def test_fs_atomic_replace_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "features.parquet"
            df = _make_features(10)
            fs_atomic_replace(target, lambda p: df.to_parquet(p))
            self.assertTrue(target.exists())
            loaded = pd.read_parquet(target)
            self.assertEqual(len(loaded), 10)

    def test_fs_atomic_replace_cleans_up_on_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "features.parquet"

            def bad_write(p):
                raise RuntimeError("simulated failure")

            with self.assertRaises(RuntimeError):
                fs_atomic_replace(target, bad_write)

            self.assertFalse(target.exists())
            tmp_files = [f for f in os.listdir(tmp) if f.endswith(".tmp")]
            self.assertEqual(tmp_files, [])


# -----------------------------------------------------------------------
# Integration tests: save_ohlcv atomic behavior
# -----------------------------------------------------------------------


class TestSaveOhlcvAtomic(unittest.TestCase):
    """Integration tests for atomic save_ohlcv behavior."""

    def test_save_ohlcv_produces_valid_parquet(self):
        """save_ohlcv should produce a valid, readable parquet file."""
        df = _make_ohlcv()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_ohlcv("AAPL", df, cache_dir=tmp, source="test")
            self.assertTrue(path.exists())
            loaded = pd.read_parquet(path)
            self.assertGreater(len(loaded), 0)

    def test_save_ohlcv_produces_valid_metadata(self):
        """save_ohlcv should produce a valid JSON metadata sidecar."""
        df = _make_ohlcv()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_ohlcv("MSFT", df, cache_dir=tmp, source="test")
            meta_path = Path(tmp) / "MSFT_1d.meta.json"
            self.assertTrue(meta_path.exists())
            with open(meta_path) as f:
                meta = json.load(f)
            self.assertEqual(meta["ticker"], "MSFT")
            self.assertEqual(meta["source"], "test")

    def test_save_ohlcv_no_temp_files_left(self):
        """After a successful save, no .tmp files should remain."""
        df = _make_ohlcv()
        with tempfile.TemporaryDirectory() as tmp:
            save_ohlcv("GOOG", df, cache_dir=tmp, source="test")
            tmp_files = [f for f in os.listdir(tmp) if f.endswith(".tmp")]
            self.assertEqual(tmp_files, [])

    def test_save_ohlcv_round_trip(self):
        """Data saved atomically should round-trip through load_ohlcv_with_meta."""
        df = _make_ohlcv()
        with tempfile.TemporaryDirectory() as tmp:
            save_ohlcv("NVDA", df, cache_dir=tmp, source="wrds")
            loaded, meta, loaded_path = load_ohlcv_with_meta("NVDA", cache_dir=tmp)
            self.assertIsNotNone(loaded)
            self.assertEqual(meta["source"], "wrds")
            self.assertEqual(meta["ticker"], "NVDA")


# -----------------------------------------------------------------------
# Integration tests: FeatureStore atomic behavior
# -----------------------------------------------------------------------


class TestFeatureStoreAtomic(unittest.TestCase):
    """Integration tests for atomic FeatureStore.save_features behavior."""

    def test_save_features_produces_valid_parquet(self):
        """save_features should produce a valid, readable parquet file."""
        features = _make_features()
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(store_dir=Path(tmp))
            pq_path = store.save_features("AAPL", features, "2023-06-01")
            self.assertTrue(pq_path.exists())
            loaded = pd.read_parquet(pq_path)
            self.assertEqual(len(loaded), len(features))

    def test_save_features_produces_valid_metadata(self):
        """save_features should produce a valid JSON metadata sidecar."""
        features = _make_features()
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(store_dir=Path(tmp))
            store.save_features("MSFT", features, "2023-06-01")
            meta_path = Path(tmp) / "MSFT" / "v1" / "features_2023-06-01.meta.json"
            self.assertTrue(meta_path.exists())
            with open(meta_path) as f:
                meta = json.load(f)
            self.assertEqual(meta["permno"], "MSFT")
            self.assertEqual(meta["computed_at"], "2023-06-01")
            self.assertEqual(meta["row_count"], len(features))

    def test_save_features_no_temp_files_left(self):
        """After a successful save, no .tmp files should remain in the version dir."""
        features = _make_features()
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(store_dir=Path(tmp))
            store.save_features("TEST", features, "2023-06-01")
            vdir = Path(tmp) / "TEST" / "v1"
            tmp_files = [f for f in os.listdir(vdir) if f.endswith(".tmp")]
            self.assertEqual(tmp_files, [])

    def test_save_features_round_trip(self):
        """Data saved atomically should round-trip through load_features."""
        features = _make_features()
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(store_dir=Path(tmp))
            store.save_features("AAPL", features, "2023-06-01")
            loaded = store.load_features("AAPL", as_of="2023-12-31")
            self.assertIsNotNone(loaded)
            # Parquet does not preserve DatetimeIndex freq metadata,
            # so compare with check_freq=False.
            pd.testing.assert_frame_equal(loaded, features, check_freq=False)

    def test_save_features_metadata_uses_utc_timezone(self):
        """saved_utc should use timezone-aware UTC (not deprecated utcnow)."""
        features = _make_features(10)
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(store_dir=Path(tmp))
            store.save_features("TZ", features, "2023-06-01")
            meta_path = Path(tmp) / "TZ" / "v1" / "features_2023-06-01.meta.json"
            with open(meta_path) as f:
                meta = json.load(f)
            saved_utc = meta["saved_utc"]
            # Timezone-aware datetime.now(timezone.utc) produces +00:00
            self.assertIn("+00:00", saved_utc)


# -----------------------------------------------------------------------
# Concurrency tests: verify no corruption under concurrent writes
# -----------------------------------------------------------------------


class TestConcurrentCacheWrites(unittest.TestCase):
    """SPEC-D03 verification: concurrent writes should not corrupt cache files."""

    def test_concurrent_save_ohlcv_no_corruption(self):
        """Two threads writing to the same cache key should not produce corruption."""
        n_threads = 4
        n_iterations = 10
        errors = []

        with tempfile.TemporaryDirectory() as tmp:
            def writer(thread_id):
                """Each thread writes slightly different data."""
                for i in range(n_iterations):
                    try:
                        df = _make_ohlcv(n_bars=100 + thread_id + i, start="2020-01-01")
                        save_ohlcv("RACE", df, cache_dir=tmp, source=f"thread_{thread_id}")
                    except Exception as e:
                        errors.append(f"Thread {thread_id} iteration {i}: {e}")

            threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [], f"Errors during concurrent writes: {errors}")

            # Final file should be valid and readable
            path = Path(tmp) / "RACE_1d.parquet"
            self.assertTrue(path.exists(), "Parquet file should exist after concurrent writes")
            loaded = pd.read_parquet(path)
            self.assertGreater(len(loaded), 0, "Parquet file should not be empty")

            # Metadata should also be valid
            meta_path = Path(tmp) / "RACE_1d.meta.json"
            self.assertTrue(meta_path.exists(), "Metadata should exist after concurrent writes")
            with open(meta_path) as f:
                meta = json.load(f)
            self.assertIn("ticker", meta)
            self.assertEqual(meta["ticker"], "RACE")

    def test_concurrent_feature_store_no_corruption(self):
        """Two threads writing to the same feature key should not produce corruption."""
        n_threads = 4
        n_iterations = 10
        errors = []

        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(store_dir=Path(tmp))

            def writer(thread_id):
                for i in range(n_iterations):
                    try:
                        features = _make_features(n_rows=50 + thread_id + i)
                        store.save_features("RACE", features, "2023-06-01")
                    except Exception as e:
                        errors.append(f"Thread {thread_id} iteration {i}: {e}")

            threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [], f"Errors during concurrent writes: {errors}")

            # Final parquet should be valid
            pq_path = Path(tmp) / "RACE" / "v1" / "features_2023-06-01.parquet"
            self.assertTrue(pq_path.exists(), "Parquet should exist after concurrent writes")
            loaded = pd.read_parquet(pq_path)
            self.assertGreater(len(loaded), 0, "Parquet should not be empty")

            # Final metadata should be valid JSON
            meta_path = Path(tmp) / "RACE" / "v1" / "features_2023-06-01.meta.json"
            self.assertTrue(meta_path.exists(), "Metadata should exist after concurrent writes")
            with open(meta_path) as f:
                meta = json.load(f)
            self.assertEqual(meta["permno"], "RACE")

    def test_concurrent_write_meta_no_corruption(self):
        """Concurrent _write_cache_meta calls should not produce invalid JSON."""
        n_threads = 4
        n_iterations = 20
        errors = []

        with tempfile.TemporaryDirectory() as tmp:
            dummy_path = Path(tmp) / "META_1d.parquet"
            df = _make_ohlcv(n_bars=100)

            def writer(thread_id):
                for i in range(n_iterations):
                    try:
                        _write_cache_meta(
                            dummy_path,
                            ticker="META",
                            df=df,
                            source=f"thread_{thread_id}",
                            meta={"iteration": i, "thread": thread_id},
                        )
                    except Exception as e:
                        errors.append(f"Thread {thread_id} iteration {i}: {e}")

            threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [], f"Errors during concurrent meta writes: {errors}")

            # Final JSON should be valid
            meta_path = Path(tmp) / "META_1d.meta.json"
            self.assertTrue(meta_path.exists())
            with open(meta_path) as f:
                meta = json.load(f)
            self.assertEqual(meta["ticker"], "META")

    def test_no_tmp_files_after_concurrent_writes(self):
        """No .tmp files should remain after concurrent writes finish."""
        n_threads = 4
        n_iterations = 10

        with tempfile.TemporaryDirectory() as tmp:
            def writer(thread_id):
                for i in range(n_iterations):
                    df = _make_ohlcv(n_bars=100 + thread_id, start="2020-01-01")
                    save_ohlcv("CLEAN", df, cache_dir=tmp, source="test")

            threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            tmp_files = [f for f in os.listdir(tmp) if ".tmp" in f]
            self.assertEqual(tmp_files, [], f"Leftover temp files: {tmp_files}")


if __name__ == "__main__":
    unittest.main()
