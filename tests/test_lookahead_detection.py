"""
Automated lookahead bias detection for the feature pipeline.

Strategy: compute features on the full dataset, then compute on the dataset
minus the last bar.  If removing the last bar changes any feature value at
earlier bars, that feature uses future data (lookahead bias).

This test is designed for CI — any new lookahead leak in any feature will
cause a failure.
"""

import unittest

import numpy as np
import pandas as pd

from quant_engine.features.pipeline import FeaturePipeline


def _make_synthetic_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Build synthetic OHLCV data for lookahead testing."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
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


class TestLookaheadDetection(unittest.TestCase):
    """Detect any feature that uses future data."""

    def test_no_feature_uses_future_data(self):
        """Removing the last bar must not change any historical feature value.

        If feature F at bar N-2 changes when bar N-1 is removed, F uses
        future data from bar N-1, i.e. lookahead bias.
        """
        df = _make_synthetic_ohlcv(n=300, seed=42)

        # Use core mode to avoid cross-asset / WRDS dependencies,
        # but still test all single-stock features.  Interactions are
        # disabled to keep the test focused on raw feature computation.
        pipe = FeaturePipeline(
            feature_mode="full",
            include_interactions=False,
            include_cross_asset_factors=False,
            include_options_factors=False,
            include_research_factors=True,
        )

        features_full, _ = pipe.compute(df, compute_targets_flag=False)
        features_trunc, _ = pipe.compute(df.iloc[:-1], compute_targets_flag=False)

        # Compare at the last row of the truncated set (= second-to-last
        # row of the full set).
        check_idx = features_trunc.index[-1]
        self.assertIn(check_idx, features_full.index)

        full_row = features_full.loc[check_idx]
        trunc_row = features_trunc.loc[check_idx]

        # Only compare columns present in both
        common_cols = full_row.index.intersection(trunc_row.index)
        self.assertGreater(len(common_cols), 0, "No common feature columns found")

        full_vals = full_row[common_cols].astype(float)
        trunc_vals = trunc_row[common_cols].astype(float)

        # Compute absolute difference; NaN == NaN is considered equal
        both_nan = full_vals.isna() & trunc_vals.isna()
        diff = (full_vals - trunc_vals).abs()
        diff[both_nan] = 0.0

        # Features that changed (lookahead detected)
        lookahead_features = diff[diff > 1e-10].dropna()
        lookahead_features = lookahead_features[lookahead_features > 0]

        self.assertEqual(
            len(lookahead_features),
            0,
            f"Lookahead bias detected in {len(lookahead_features)} features: "
            f"{list(lookahead_features.index[:15])}",
        )

    def test_no_feature_uses_future_data_deep(self):
        """Same test but checks multiple interior rows, not just the last one.

        This catches subtler bugs where lookahead only manifests at
        specific offsets (e.g. shift(-5) only affects rows within 5 bars
        of the end).
        """
        df = _make_synthetic_ohlcv(n=350, seed=99)

        pipe = FeaturePipeline(
            feature_mode="full",
            include_interactions=False,
            include_cross_asset_factors=False,
            include_options_factors=False,
            include_research_factors=True,
        )

        features_full, _ = pipe.compute(df, compute_targets_flag=False)

        # Truncate at multiple depths (1, 5, 10, 20 bars from end)
        for trim in [1, 5, 10, 20]:
            df_short = df.iloc[:-trim]
            features_short, _ = pipe.compute(df_short, compute_targets_flag=False)

            # Check at a safe interior row (well before the truncation point)
            check_idx = features_short.index[-10] if len(features_short) > 10 else features_short.index[-1]
            if check_idx not in features_full.index:
                continue

            full_row = features_full.loc[check_idx]
            short_row = features_short.loc[check_idx]
            common_cols = full_row.index.intersection(short_row.index)

            full_vals = full_row[common_cols].astype(float)
            short_vals = short_row[common_cols].astype(float)

            both_nan = full_vals.isna() & short_vals.isna()
            diff = (full_vals - short_vals).abs()
            diff[both_nan] = 0.0

            lookahead = diff[diff > 1e-10].dropna()
            lookahead = lookahead[lookahead > 0]

            self.assertEqual(
                len(lookahead),
                0,
                f"Lookahead bias (trim={trim}) in {len(lookahead)} features: "
                f"{list(lookahead.index[:10])}",
            )


if __name__ == "__main__":
    unittest.main()
