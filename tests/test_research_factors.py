"""
Test module for research factors behavior and regressions.
"""

import unittest

import numpy as np
import pandas as pd

from quant_engine.features.pipeline import FeaturePipeline
from quant_engine.features.research_factors import (
    ResearchFactorConfig,
    compute_cross_asset_research_factors,
    compute_single_asset_research_factors,
)


def _make_ohlcv(seed: int, periods: int = 420, drift: float = 0.0003) -> pd.DataFrame:
    """Build synthetic test data for the scenarios in this module."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2022-01-03", periods=periods, freq="B")
    ret = drift + rng.normal(0.0, 0.012, size=periods)
    close = 100.0 * np.cumprod(1.0 + ret)
    open_ = close * (1.0 + rng.normal(0.0, 0.0015, size=periods))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0005, 0.006, size=periods))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0005, 0.006, size=periods))
    volume = 1_000_000.0 * np.exp(rng.normal(0.0, 0.25, size=periods))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=idx,
    )


class ResearchFactorTests(unittest.TestCase):
    """Test cases covering research factors behavior and system invariants."""
    def test_single_asset_research_features_exist(self):
        df = _make_ohlcv(seed=7)
        feats = compute_single_asset_research_factors(df)

        required_cols = [
            "OFI_20",
            "LOB_State",
            "TSMom_lag63",
            "SigL1_Return_20",
            "VolSurf_Level",
            "VolSurf_PC1",
        ]
        for col in required_cols:
            self.assertIn(col, feats.columns)

        populated = feats[required_cols].dropna(how="all")
        self.assertGreater(len(populated), 0)

    def test_cross_asset_network_features_shape_and_bounds(self):
        base = _make_ohlcv(seed=1, periods=320)
        idx = base.index
        base_ret = base["Close"].pct_change().fillna(0.0).values

        rng = np.random.RandomState(13)
        r2 = 0.50 * base_ret + rng.normal(0.0, 0.01, size=len(idx))
        r3 = 0.35 * base_ret + rng.normal(0.0, 0.011, size=len(idx))

        c2 = 80.0 * np.cumprod(1.0 + r2)
        c3 = 120.0 * np.cumprod(1.0 + r3)

        d2 = base.copy()
        d2["Close"] = c2
        d2["Open"] = c2 * (1.0 + rng.normal(0.0, 0.001, size=len(idx)))
        d2["High"] = np.maximum(d2["Open"], d2["Close"]) * 1.002
        d2["Low"] = np.minimum(d2["Open"], d2["Close"]) * 0.998

        d3 = base.copy()
        d3["Close"] = c3
        d3["Open"] = c3 * (1.0 + rng.normal(0.0, 0.001, size=len(idx)))
        d3["High"] = np.maximum(d3["Open"], d3["Close"]) * 1.002
        d3["Low"] = np.minimum(d3["Open"], d3["Close"]) * 0.998

        panel = compute_cross_asset_research_factors(
            {"AAPL": base, "MSFT": d2, "NVDA": d3}
        )

        self.assertFalse(panel.empty)
        self.assertTrue(isinstance(panel.index, pd.MultiIndex))
        self.assertIn("NetMom_Spillover", panel.columns)
        self.assertIn("VolSpillover_In", panel.columns)

        finite_density = panel["NetMom_GraphDensity"].dropna()
        self.assertGreater(len(finite_density), 0)
        self.assertTrue((finite_density >= 0.0).all())
        self.assertTrue((finite_density <= 1.0).all())

    def test_cross_asset_factors_are_causally_lagged(self):
        data = {
            "AAPL": _make_ohlcv(seed=10, periods=260),
            "MSFT": _make_ohlcv(seed=11, periods=260),
            "NVDA": _make_ohlcv(seed=12, periods=260),
        }
        lag = 2
        cfg = ResearchFactorConfig(network_window=30, network_min_obs=20, cross_asset_lag_bars=lag)
        panel = compute_cross_asset_research_factors(data, config=cfg)
        self.assertFalse(panel.empty)

        base_col = "NetMom_Spillover"
        for ticker in ["AAPL", "MSFT", "NVDA"]:
            s = panel.loc[ticker, base_col]
            leading = s.iloc[:lag]
            self.assertTrue(leading.isna().all())
            self.assertGreater(int(s.iloc[lag:].notna().sum()), 0)

    def test_pipeline_universe_includes_research_features(self):
        data = {
            "AAPL": _make_ohlcv(seed=5, periods=300),
            "MSFT": _make_ohlcv(seed=6, periods=300),
            "NVDA": _make_ohlcv(seed=8, periods=300),
        }
        pipeline = FeaturePipeline(include_interactions=False, verbose=False)
        features, targets = pipeline.compute_universe(data, verbose=False)

        self.assertIsNotNone(targets)
        self.assertTrue(isinstance(features.index, pd.MultiIndex))
        self.assertIn("OFI_20", features.columns)
        self.assertIn("NetMom_Spillover", features.columns)
        self.assertIn("VolSpillover_In", features.columns)


if __name__ == "__main__":
    unittest.main()
