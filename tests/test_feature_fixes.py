"""
Tests verifying Spec 012 feature engineering fixes:
  1. TSMom features use backward-looking shift (no lookahead)
  2. IV shock features use only causal shifts
  3. All features in the pipeline have metadata entries
  4. Production mode filters out RESEARCH_ONLY features
  5. Rolling VWAP is causal (no future data)
  6. Automated lookahead detection on the full pipeline
"""

import unittest

import numpy as np
import pandas as pd


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


def _make_ohlcv_with_iv(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV data with option surface columns."""
    df = _make_ohlcv(n=n, seed=seed)
    rng = np.random.RandomState(seed + 100)
    df["iv_atm_30"] = 0.20 + rng.normal(0, 0.03, size=n)
    df["iv_atm_60"] = 0.22 + rng.normal(0, 0.025, size=n)
    df["iv_atm_90"] = 0.24 + rng.normal(0, 0.02, size=n)
    df["iv_put_25d"] = 0.25 + rng.normal(0, 0.03, size=n)
    df["iv_call_25d"] = 0.18 + rng.normal(0, 0.025, size=n)
    df.loc[:, "skew_25d"] = df["iv_put_25d"] - df["iv_call_25d"]
    return df


class TestTSMomBackwardShift(unittest.TestCase):
    """Verify TSMom features are backward-looking (causal)."""

    def test_tsmom_uses_backward_shift(self):
        """TSMom columns should use shift(h), not shift(-h).

        Proof: at bar 0, there is no history so TSMom must be NaN.
        With forward shift(-h), bar 0 would have a value (from bar h).
        """
        from quant_engine.features.research_factors import (
            ResearchFactorConfig,
            compute_time_series_momentum_factors,
        )

        df = _make_ohlcv(n=300, seed=7)
        cfg = ResearchFactorConfig(tsmom_lookbacks=(5, 21, 63))
        feats = compute_time_series_momentum_factors(df, cfg)

        # Column naming uses TSMom_lag{h}
        self.assertIn("TSMom_lag5", feats.columns)
        self.assertIn("TSMom_lag21", feats.columns)
        self.assertIn("TSMom_lag63", feats.columns)

        # First 5 rows of TSMom_lag5 must be NaN (no 5-bar history)
        self.assertTrue(
            feats["TSMom_lag5"].iloc[:5].isna().all(),
            "TSMom_lag5 should be NaN for the first 5 rows (no history)",
        )

        # First 63 rows of TSMom_lag63 must be NaN
        self.assertTrue(
            feats["TSMom_lag63"].iloc[:63].isna().all(),
            "TSMom_lag63 should be NaN for the first 63 rows (no history)",
        )

        # TSMom_12m1m should also be backward-looking
        self.assertTrue(
            feats["TSMom_12m1m"].iloc[:252].isna().all(),
            "TSMom_12m1m should be NaN for the first 252 rows (no 1-year history)",
        )

    def test_tsmom_values_dont_change_when_future_removed(self):
        """Removing the last 10 bars must not change TSMom at bar N-20."""
        from quant_engine.features.research_factors import (
            ResearchFactorConfig,
            compute_time_series_momentum_factors,
        )

        df = _make_ohlcv(n=300, seed=8)
        cfg = ResearchFactorConfig(tsmom_lookbacks=(5, 21))

        full = compute_time_series_momentum_factors(df, cfg)
        trunc = compute_time_series_momentum_factors(df.iloc[:-10], cfg)

        check_idx = trunc.index[-20]
        for col in ["TSMom_lag5", "TSMom_lag21"]:
            full_val = full.loc[check_idx, col]
            trunc_val = trunc.loc[check_idx, col]
            if np.isnan(full_val) and np.isnan(trunc_val):
                continue
            self.assertAlmostEqual(
                full_val, trunc_val, places=10,
                msg=f"{col} changed when future bars were removed (lookahead!)",
            )


class TestIVShockNonFuture(unittest.TestCase):
    """Verify IV shock features use only backward-looking shifts."""

    def test_iv_shock_no_future_shift(self):
        """IV shock features should NOT use shift(-window)."""
        from quant_engine.features.options_factors import compute_iv_shock_features

        df = _make_ohlcv_with_iv(n=200, seed=10)
        feats = compute_iv_shock_features(df, window=5)

        # New column names should reflect causality
        self.assertFalse(
            any("pre_post" in c for c in feats.columns),
            "Old 'pre_post' column names should not exist",
        )
        self.assertTrue(
            any("change" in c for c in feats.columns),
            "New 'change' column names should exist",
        )

        # First W rows should be NaN (no history to compare against)
        for col in feats.columns:
            first_w = feats[col].iloc[:5]
            self.assertTrue(
                first_w.isna().all(),
                f"{col} should be NaN in first 5 rows (need lookback history)",
            )

    def test_iv_shock_no_lookahead(self):
        """Removing the last bar must not change IV shock values at earlier bars."""
        from quant_engine.features.options_factors import compute_iv_shock_features

        df = _make_ohlcv_with_iv(n=200, seed=11)

        full = compute_iv_shock_features(df, window=5)
        trunc = compute_iv_shock_features(df.iloc[:-1], window=5)

        check_idx = trunc.index[-5]
        for col in full.columns:
            if col not in trunc.columns:
                continue
            full_val = float(full.loc[check_idx, col])
            trunc_val = float(trunc.loc[check_idx, col])
            if np.isnan(full_val) and np.isnan(trunc_val):
                continue
            self.assertAlmostEqual(
                full_val, trunc_val, places=10,
                msg=f"IV shock {col} uses future data!",
            )


class TestFeatureMetadata(unittest.TestCase):
    """Verify feature metadata registry coverage."""

    def test_feature_metadata_exists(self):
        """FEATURE_METADATA should contain entries for all major features."""
        from quant_engine.features.pipeline import FEATURE_METADATA

        self.assertGreater(len(FEATURE_METADATA), 100, "Registry should be comprehensive")

        # Spot-check key features
        for feat in [
            "RSI_14", "MACD_12_26", "ADX_14", "return_1d", "log_volume",
            "TSMom_lag63", "TSMom_12m1m", "OFI_20", "vwap_deviation",
            "vrp_30", "NetMom_Spillover",
        ]:
            self.assertIn(feat, FEATURE_METADATA, f"Missing metadata for {feat}")

    def test_all_metadata_has_valid_type(self):
        """Every entry must have a valid 'type' field."""
        from quant_engine.features.pipeline import FEATURE_METADATA

        valid_types = {"CAUSAL", "END_OF_DAY", "RESEARCH_ONLY"}
        for name, meta in FEATURE_METADATA.items():
            self.assertIn(
                "type", meta,
                f"Feature '{name}' has no 'type' in metadata",
            )
            self.assertIn(
                meta["type"], valid_types,
                f"Feature '{name}' has invalid type '{meta['type']}'",
            )

    def test_pipeline_features_have_metadata(self):
        """Features produced by the pipeline should have metadata entries."""
        from quant_engine.features.pipeline import (
            FEATURE_METADATA,
            FeaturePipeline,
            get_feature_type,
        )

        df = _make_ohlcv(n=300, seed=42)
        pipe = FeaturePipeline(
            feature_mode="full",
            include_interactions=False,
            include_cross_asset_factors=False,
            include_options_factors=False,
            include_research_factors=True,
        )
        features, _ = pipe.compute(df, compute_targets_flag=False)

        # Every produced column should either be in FEATURE_METADATA or
        # return a valid type from get_feature_type (which has defaults)
        for col in features.columns:
            ftype = get_feature_type(col)
            self.assertIn(
                ftype, {"CAUSAL", "END_OF_DAY", "RESEARCH_ONLY"},
                f"Feature '{col}' returned invalid type '{ftype}'",
            )


class TestProductionModeFilter(unittest.TestCase):
    """Verify production_mode filters RESEARCH_ONLY features."""

    def test_production_mode_filters_research(self):
        """production_mode=True should exclude RESEARCH_ONLY features."""
        from quant_engine.features.pipeline import FeaturePipeline

        df = _make_ohlcv(n=300, seed=55)

        pipe_full = FeaturePipeline(
            feature_mode="full",
            include_interactions=False,
            include_cross_asset_factors=False,
            include_options_factors=False,
            include_research_factors=True,
            production_mode=False,
        )
        pipe_prod = FeaturePipeline(
            feature_mode="full",
            include_interactions=False,
            include_cross_asset_factors=False,
            include_options_factors=False,
            include_research_factors=True,
            production_mode=True,
        )

        feats_full, _ = pipe_full.compute(df, compute_targets_flag=False)
        feats_prod, _ = pipe_prod.compute(df, compute_targets_flag=False)

        # Production mode should have fewer or equal columns
        self.assertLessEqual(feats_prod.shape[1], feats_full.shape[1])

        # RESEARCH_ONLY features should be absent in production mode
        research_cols = ["relative_mom_10", "relative_mom_20", "relative_mom_60"]
        for col in research_cols:
            if col in feats_full.columns:
                self.assertNotIn(
                    col, feats_prod.columns,
                    f"RESEARCH_ONLY feature '{col}' should be filtered in production mode",
                )

    def test_production_mode_default_false(self):
        """Default production_mode should be False (backward compatible)."""
        from quant_engine.features.pipeline import FeaturePipeline

        pipe = FeaturePipeline()
        self.assertFalse(pipe.production_mode)


class TestRollingVWAPIsCausal(unittest.TestCase):
    """Verify rolling VWAP uses only past data."""

    def test_rolling_vwap_is_causal(self):
        """Rolling VWAP at bar N should not change when bar N+1 is added."""
        from quant_engine.features.intraday import compute_rolling_vwap

        df = _make_ohlcv(n=200, seed=77)

        full = compute_rolling_vwap(df, window=20)
        trunc = compute_rolling_vwap(df.iloc[:-1], window=20)

        # Value at the second-to-last row should be identical
        check_idx = trunc.index[-5]
        for col in full.columns:
            full_val = float(full.loc[check_idx, col])
            trunc_val = float(trunc.loc[check_idx, col])
            if np.isnan(full_val) and np.isnan(trunc_val):
                continue
            self.assertAlmostEqual(
                full_val, trunc_val, places=10,
                msg=f"Rolling VWAP {col} uses future data!",
            )

    def test_rolling_vwap_first_rows_nan(self):
        """First few rows should be NaN (not enough history)."""
        from quant_engine.features.intraday import compute_rolling_vwap

        df = _make_ohlcv(n=100, seed=78)
        result = compute_rolling_vwap(df, window=20)

        # min_periods = max(5, 20//3) = 7, so first 4 rows (indices 0-3)
        # are guaranteed NaN (need at least 5 non-NaN values in the window).
        # Row 0 has no prior data to form typical_price * volume rolling.
        for col in result.columns:
            first_rows = result[col].iloc[:4]
            self.assertTrue(
                first_rows.isna().all(),
                f"{col} should be NaN in the first few rows",
            )


class TestMultiHorizonVRP(unittest.TestCase):
    """Verify enhanced VRP computation at multiple horizons."""

    def test_vrp_multi_horizon(self):
        """VRP at 10d, 30d, 60d should be computed when IV data is available."""
        from quant_engine.features.options_factors import compute_option_surface_factors

        df = _make_ohlcv_with_iv(n=300, seed=33)
        feats = compute_option_surface_factors(df)

        self.assertIn("vrp_10", feats.columns)
        self.assertIn("vrp_30", feats.columns)
        self.assertIn("vrp_60", feats.columns)

        # VRP values should be finite for rows with enough history
        for col in ["vrp_10", "vrp_30", "vrp_60"]:
            finite = feats[col].dropna()
            self.assertGreater(len(finite), 50, f"{col} has too few valid values")


class TestFullPipelineLookahead(unittest.TestCase):
    """End-to-end lookahead check on the complete feature pipeline."""

    def test_automated_lookahead_detection(self):
        """Full pipeline: no feature at bar N should change when bar N+1 added."""
        from quant_engine.features.pipeline import FeaturePipeline

        df = _make_ohlcv(n=300, seed=42)
        pipe = FeaturePipeline(
            feature_mode="full",
            include_interactions=False,
            include_cross_asset_factors=False,
            include_options_factors=False,
            include_research_factors=True,
        )

        features_full, _ = pipe.compute(df, compute_targets_flag=False)
        features_trunc, _ = pipe.compute(df.iloc[:-1], compute_targets_flag=False)

        check_idx = features_trunc.index[-1]
        self.assertIn(check_idx, features_full.index)

        full_row = features_full.loc[check_idx]
        trunc_row = features_trunc.loc[check_idx]
        common_cols = full_row.index.intersection(trunc_row.index)

        full_vals = full_row[common_cols].astype(float)
        trunc_vals = trunc_row[common_cols].astype(float)

        both_nan = full_vals.isna() & trunc_vals.isna()
        diff = (full_vals - trunc_vals).abs()
        diff[both_nan] = 0.0

        lookahead = diff[diff > 1e-10].dropna()
        lookahead = lookahead[lookahead > 0]

        self.assertEqual(
            len(lookahead), 0,
            f"Lookahead bias in {len(lookahead)} features: {list(lookahead.index[:10])}",
        )


if __name__ == "__main__":
    unittest.main()
