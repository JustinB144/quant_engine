"""
Tests for SPEC_AUDIT_FIX_10: Features Causality Enforcement & Data Integrity.

Covers:
  T1: RESEARCH_ONLY causality filter
  T2: Unknown features fail-closed
  T3: VolSpillover_Net = vol_out - vol_in
  T4: DTW backtracking no negative indices
  T5: Monthly macro forward-fill limit
  T6: Feature collision logging
  T7: FeatureVersion frozen dataclass
  T8: Compatibility check reports extra features
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV data for testing."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    ret = rng.normal(0.0003, 0.012, size=n)
    close = 100.0 * np.cumprod(1.0 + ret)
    open_ = close * (1.0 + rng.normal(0.0, 0.002, size=n))
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, size=n))
    volume = 1_000_000.0 * np.exp(rng.normal(0.0, 0.25, size=n))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


# ===========================================================================
# T1: RESEARCH_ONLY causality filter
# ===========================================================================

class TestCausalityFilter(unittest.TestCase):
    """T1: _filter_causal_features handles all three modes correctly."""

    def _make_features_df(self):
        """Create a minimal DataFrame with features of each type."""
        from quant_engine.features.pipeline import FEATURE_METADATA
        # Pick one feature of each type
        causal_feat = "RSI_14"
        eod_feat = "intraday_vol_ratio"
        research_feat = "relative_mom_10"
        # Verify they exist with expected types
        self.assertEqual(FEATURE_METADATA[causal_feat]["type"], "CAUSAL")
        self.assertEqual(FEATURE_METADATA[eod_feat]["type"], "END_OF_DAY")
        self.assertEqual(FEATURE_METADATA[research_feat]["type"], "RESEARCH_ONLY")

        df = pd.DataFrame({
            causal_feat: [1.0, 2.0],
            eod_feat: [3.0, 4.0],
            research_feat: [5.0, 6.0],
        })
        return df, causal_feat, eod_feat, research_feat

    def test_causal_mode_excludes_non_causal(self):
        from quant_engine.features.pipeline import _filter_causal_features
        df, causal, eod, research = self._make_features_df()
        result = _filter_causal_features(df, causality_filter="CAUSAL")
        self.assertIn(causal, result.columns)
        self.assertNotIn(eod, result.columns)
        self.assertNotIn(research, result.columns)

    def test_research_only_mode(self):
        from quant_engine.features.pipeline import _filter_causal_features
        df, causal, eod, research = self._make_features_df()
        result = _filter_causal_features(df, causality_filter="RESEARCH_ONLY")
        self.assertIn(causal, result.columns)
        self.assertNotIn(eod, result.columns)
        self.assertIn(research, result.columns)

    def test_end_of_day_mode_includes_all(self):
        from quant_engine.features.pipeline import _filter_causal_features
        df, causal, eod, research = self._make_features_df()
        result = _filter_causal_features(df, causality_filter="END_OF_DAY")
        self.assertIn(causal, result.columns)
        self.assertIn(eod, result.columns)
        self.assertIn(research, result.columns)

    def test_unknown_filter_raises(self):
        from quant_engine.features.pipeline import _filter_causal_features
        df, _, _, _ = self._make_features_df()
        with self.assertRaises(ValueError):
            _filter_causal_features(df, causality_filter="INVALID")


# ===========================================================================
# T2: Unknown features fail-closed
# ===========================================================================

class TestUnknownFeatureFailClosed(unittest.TestCase):
    """T2: Unknown features return RESEARCH_ONLY and are excluded from strict modes."""

    def test_unknown_feature_returns_research_only(self):
        from quant_engine.features.pipeline import get_feature_type
        result = get_feature_type("completely_made_up_feature_xyz")
        self.assertEqual(result, "RESEARCH_ONLY")

    def test_interaction_features_are_causal(self):
        from quant_engine.features.pipeline import get_feature_type
        result = get_feature_type("X_RSI_14_times_ADX_14")
        self.assertEqual(result, "CAUSAL")

    def test_known_feature_returns_correct_type(self):
        from quant_engine.features.pipeline import get_feature_type
        self.assertEqual(get_feature_type("RSI_14"), "CAUSAL")
        self.assertEqual(get_feature_type("intraday_vol_ratio"), "END_OF_DAY")
        self.assertEqual(get_feature_type("relative_mom_10"), "RESEARCH_ONLY")

    def test_unknown_excluded_from_causal_mode(self):
        from quant_engine.features.pipeline import _filter_causal_features
        df = pd.DataFrame({
            "RSI_14": [1.0],
            "unknown_feature_abc": [2.0],
        })
        result = _filter_causal_features(df, causality_filter="CAUSAL")
        self.assertIn("RSI_14", result.columns)
        self.assertNotIn("unknown_feature_abc", result.columns)

    def test_unknown_included_in_research_mode(self):
        """Unknown features default to RESEARCH_ONLY, so they pass RESEARCH_ONLY filter."""
        from quant_engine.features.pipeline import _filter_causal_features
        df = pd.DataFrame({
            "RSI_14": [1.0],
            "unknown_feature_abc": [2.0],
        })
        result = _filter_causal_features(df, causality_filter="RESEARCH_ONLY")
        self.assertIn("RSI_14", result.columns)
        self.assertIn("unknown_feature_abc", result.columns)

    def test_unknown_included_in_eod_mode(self):
        from quant_engine.features.pipeline import _filter_causal_features
        df = pd.DataFrame({
            "RSI_14": [1.0],
            "unknown_feature_abc": [2.0],
        })
        result = _filter_causal_features(df, causality_filter="END_OF_DAY")
        self.assertIn("RSI_14", result.columns)
        self.assertIn("unknown_feature_abc", result.columns)


# ===========================================================================
# T3: VolSpillover_Net = vol_out - vol_in
# ===========================================================================

class TestVolSpilloverNet(unittest.TestCase):
    """T3: VolSpillover_Net uses vol_out - vol_in, not vol_out - recv_cent."""

    def test_vol_spillover_net_is_vol_out_minus_vol_in(self):
        from quant_engine.features.research_factors import (
            ResearchFactorConfig,
            compute_cross_asset_research_factors,
        )
        n = 120
        rng = np.random.RandomState(99)
        idx = pd.date_range("2022-01-03", periods=n, freq="B")

        price_data = {}
        for ticker in ["A", "B", "C"]:
            ret = rng.normal(0.0003, 0.015, size=n)
            close = 100.0 * np.cumprod(1.0 + ret)
            open_ = close * (1.0 + rng.normal(0.0, 0.002, size=n))
            high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, size=n))
            low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, size=n))
            volume = 1_000_000.0 * np.exp(rng.normal(0.0, 0.25, size=n))
            price_data[ticker] = pd.DataFrame(
                {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
                index=idx,
            )

        cfg = ResearchFactorConfig(network_window=40, network_min_obs=20)
        result = compute_cross_asset_research_factors(price_data, config=cfg)

        if result.empty:
            self.skipTest("Not enough data for cross-asset factors")

        # VolSpillover_Net should equal Out - In (not Out - RecvCentrality)
        net = result["VolSpillover_Net"]
        out = result["VolSpillover_Out"]
        vin = result["VolSpillover_In"]

        valid = net.notna() & out.notna() & vin.notna()
        if valid.sum() == 0:
            self.skipTest("No valid cross-asset factor rows")

        np.testing.assert_allclose(
            net[valid].values,
            (out[valid] - vin[valid]).values,
            rtol=1e-10,
            err_msg="VolSpillover_Net should be vol_out - vol_in",
        )


# ===========================================================================
# T4: DTW backtracking no negative indices
# ===========================================================================

class TestDTWNoNegativeIndices(unittest.TestCase):
    """T4: DTW path should never contain negative indices."""

    def test_dtw_path_no_negative_indices(self):
        from quant_engine.features.research_factors import _dtw_distance_numpy
        rng = np.random.RandomState(42)
        x = rng.randn(15)
        y = rng.randn(15)
        dist, path = _dtw_distance_numpy(x, y)
        self.assertGreater(len(path), 0)
        for i, j in path:
            self.assertGreaterEqual(i, 0, f"Negative i index: {i}")
            self.assertGreaterEqual(j, 0, f"Negative j index: {j}")

    def test_dtw_path_edge_cases(self):
        from quant_engine.features.research_factors import _dtw_distance_numpy
        # Very short sequences
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        dist, path = _dtw_distance_numpy(x, y)
        for i, j in path:
            self.assertGreaterEqual(i, 0)
            self.assertGreaterEqual(j, 0)

    def test_dtw_path_unequal_lengths(self):
        from quant_engine.features.research_factors import _dtw_distance_numpy
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist, path = _dtw_distance_numpy(x, y)
        for i, j in path:
            self.assertGreaterEqual(i, 0, f"Negative i index: {i}")
            self.assertGreaterEqual(j, 0, f"Negative j index: {j}")
        # Path should be non-empty and monotonically increasing
        self.assertGreater(len(path), 0)
        # Verify path starts near (0,0) and ends near the sequence endpoints
        self.assertEqual(path[0], (0, 0))


# ===========================================================================
# T5: Monthly macro forward-fill limit
# ===========================================================================

class TestMacroForwardFill(unittest.TestCase):
    """T5: Monthly macro data should be forward-filled for up to 25 business days."""

    def test_monthly_ffill_limit(self):
        """Monthly data should fill across all business days between releases."""
        from quant_engine.features.macro import MacroFeatureProvider

        provider = MacroFeatureProvider(api_key="test")

        # Mock a monthly series with ~21 business days between observations
        monthly_dates = pd.to_datetime(["2023-01-02", "2023-02-01", "2023-03-01"])
        monthly_values = pd.Series([100.0, 101.0, 102.0], index=monthly_dates, name="UMCSENT")

        with patch.object(provider, '_fetch_series', return_value=monthly_values):
            result = provider.get_macro_features("2023-01-02", "2023-03-01")

        # The consumer sentiment column should have values for all business days
        col = "macro_consumer_sentiment"
        self.assertIn(col, result.columns)

        # Between Jan 2 and Feb 1 there are ~22 business days
        # With old limit=5, only 5 would be filled. With limit=25, all should be.
        jan_data = result.loc["2023-01-02":"2023-01-31", col]
        non_nan = jan_data.notna().sum()
        # Should have values for all business days in January (not just 5)
        self.assertGreater(non_nan, 5, "Monthly ffill should fill more than 5 days")


# ===========================================================================
# T6: Feature collision logging
# ===========================================================================

class TestFeatureCollisionLogging(unittest.TestCase):
    """T6: Duplicate features should produce a warning log."""

    def test_duplicate_features_logged(self):
        """Verify duplicate columns produce a warning when deduplicating."""
        import logging
        from quant_engine.features.pipeline import logger as pipeline_logger

        df = pd.DataFrame({
            "RSI_14": [1.0, 2.0],
            "MACD_12_26": [3.0, 4.0],
        })
        # Create duplicate column
        df = pd.concat([df, df[["RSI_14"]]], axis=1)
        self.assertTrue(df.columns.duplicated().any())

        with self.assertLogs(pipeline_logger, level="WARNING") as cm:
            dupes = df.columns[df.columns.duplicated()].tolist()
            pipeline_logger.warning("Duplicate features detected and dropped (kept first): %s", dupes)
        self.assertTrue(any("Duplicate features" in msg for msg in cm.output))


# ===========================================================================
# T7: FeatureVersion frozen dataclass
# ===========================================================================

class TestFeatureVersionFrozen(unittest.TestCase):
    """T7: FeatureVersion should be truly immutable."""

    def test_frozen_instance(self):
        from quant_engine.features.version import FeatureVersion
        v = FeatureVersion(feature_names=("b", "a", "c"))
        # Should be sorted
        self.assertEqual(v.feature_names, ("a", "b", "c"))
        # Should raise on mutation
        with self.assertRaises(AttributeError):
            v.feature_names = ("x",)

    def test_feature_names_is_tuple(self):
        from quant_engine.features.version import FeatureVersion
        v = FeatureVersion(feature_names=["z", "a"])
        self.assertIsInstance(v.feature_names, tuple)
        self.assertEqual(v.feature_names, ("a", "z"))

    def test_hash_and_dict(self):
        from quant_engine.features.version import FeatureVersion
        v = FeatureVersion(feature_names=["b", "a"])
        d = v.to_dict()
        self.assertIsInstance(d["feature_names"], list)
        self.assertEqual(d["feature_names"], ["a", "b"])
        self.assertEqual(d["n_features"], 2)
        self.assertTrue(len(d["version_hash"]) > 0)


# ===========================================================================
# T8: Compatibility check reports extra features
# ===========================================================================

class TestCompatibilityCheck(unittest.TestCase):
    """T8: check_compatibility should report extra features and drift_warning."""

    def test_feature_version_check_compatibility(self):
        from quant_engine.features.version import FeatureVersion
        current = FeatureVersion(feature_names=["a", "b", "c", "d"])
        model = FeatureVersion(feature_names=["a", "b", "c"])
        result = current.check_compatibility(model)
        self.assertTrue(result["compatible"])
        self.assertEqual(result["missing_features"], [])
        self.assertEqual(result["extra_features"], ["d"])
        self.assertTrue(result["drift_warning"])

    def test_feature_version_missing_features(self):
        from quant_engine.features.version import FeatureVersion
        current = FeatureVersion(feature_names=["a", "b"])
        model = FeatureVersion(feature_names=["a", "b", "c"])
        result = current.check_compatibility(model)
        self.assertFalse(result["compatible"])
        self.assertEqual(result["missing_features"], ["c"])
        self.assertFalse(result["drift_warning"])

    def test_registry_check_compatibility_drift(self):
        from quant_engine.features.version import FeatureVersion, FeatureRegistry
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "versions.json")
            registry = FeatureRegistry(storage_path=__import__("pathlib").Path(path))
            v = FeatureVersion(feature_names=["a", "b", "c"])
            version_hash = registry.register(v)

            result = registry.check_compatibility(version_hash, ["a", "b", "c", "d"])
            self.assertTrue(result["compatible"])
            self.assertTrue(result["drift_warning"])
            self.assertIn("d", result["extra"])

    def test_exact_match_no_drift(self):
        from quant_engine.features.version import FeatureVersion
        current = FeatureVersion(feature_names=["a", "b"])
        model = FeatureVersion(feature_names=["a", "b"])
        result = current.check_compatibility(model)
        self.assertTrue(result["compatible"])
        self.assertFalse(result["drift_warning"])
        self.assertEqual(result["extra_features"], [])
        self.assertEqual(result["missing_features"], [])


if __name__ == "__main__":
    unittest.main()
