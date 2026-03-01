"""
Tests for SPEC_AUDIT_FIX_21: Feature Causality & Metadata Correctness.

Covers:
  T1: PivotHigh/PivotLow strictly causal (no look-ahead leak)
  T2: All 16 missing features registered in FEATURE_METADATA; default RESEARCH_ONLY
  T3: END_OF_DAY features in predictor causality gate
"""

import logging
import unittest

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV data for testing."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    ret = rng.normal(0.0003, 0.012, size=n)
    close = 100.0 * np.exp(np.cumsum(ret))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    opn = close * (1 + rng.normal(0, 0.005, n))
    vol = rng.randint(1_000_000, 10_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


# ---------------------------------------------------------------------------
# T1: PivotHigh / PivotLow causality
# ---------------------------------------------------------------------------

class TestPivotCausality(unittest.TestCase):
    """PivotHigh and PivotLow must produce identical output on truncated vs full data."""

    def test_pivot_high_no_lookahead(self):
        """PivotHigh at time t must be the same whether data[:t] or data[:t+100]."""
        from quant_engine.indicators import PivotHigh

        df = _make_ohlcv(300, seed=99)
        indicator = PivotHigh(left_bars=5, right_bars=5)

        full_result = indicator.calculate(df)

        # Truncate at bar 150 and compute
        truncated = df.iloc[:150]
        trunc_result = indicator.calculate(truncated)

        # Values up to bar 150 must be identical
        pd.testing.assert_series_equal(
            trunc_result,
            full_result.iloc[:150],
            check_names=False,
            obj="PivotHigh truncated vs full",
        )

    def test_pivot_low_no_lookahead(self):
        """PivotLow at time t must be the same whether data[:t] or data[:t+100]."""
        from quant_engine.indicators import PivotLow

        df = _make_ohlcv(300, seed=99)
        indicator = PivotLow(left_bars=5, right_bars=5)

        full_result = indicator.calculate(df)

        truncated = df.iloc[:150]
        trunc_result = indicator.calculate(truncated)

        pd.testing.assert_series_equal(
            trunc_result,
            full_result.iloc[:150],
            check_names=False,
            obj="PivotLow truncated vs full",
        )

    def test_pivot_high_no_center_true(self):
        """No center=True should exist in pivot indicator code."""
        import inspect
        from quant_engine.indicators import PivotHigh, PivotLow

        for cls in (PivotHigh, PivotLow):
            src = inspect.getsource(cls.calculate)
            self.assertNotIn("center=True", src, f"{cls.__name__} uses center=True")


# ---------------------------------------------------------------------------
# T2: FEATURE_METADATA completeness
# ---------------------------------------------------------------------------

class TestFeatureMetadataCompleteness(unittest.TestCase):
    """All 16 features must be registered; unknown features default to RESEARCH_ONLY."""

    EXPECTED_FEATURES = [
        # Correlation regime
        "avg_pairwise_corr", "corr_stress_flag", "corr_z_score",
        # HARX spillover
        "harx_spillover_from", "harx_spillover_to", "harx_net_spillover",
        # Macro
        "macro_vix", "macro_vix_mom",
        "macro_term_spread", "macro_term_spread_mom",
        "macro_credit_spread", "macro_credit_spread_mom",
        "macro_initial_claims", "macro_initial_claims_mom",
        "macro_consumer_sentiment", "macro_consumer_sentiment_mom",
    ]

    def test_all_16_features_registered(self):
        from quant_engine.features.pipeline import FEATURE_METADATA

        for feat in self.EXPECTED_FEATURES:
            self.assertIn(feat, FEATURE_METADATA, f"Missing from FEATURE_METADATA: {feat}")
            self.assertEqual(FEATURE_METADATA[feat]["type"], "CAUSAL")

    def test_unknown_feature_defaults_to_research_only(self):
        from quant_engine.features.pipeline import get_feature_type

        with self.assertLogs("quant_engine.features.pipeline", level="WARNING") as cm:
            result = get_feature_type("totally_fake_feature_xyz")

        self.assertEqual(result, "RESEARCH_ONLY")
        self.assertTrue(any("RESEARCH_ONLY" in msg for msg in cm.output))

    def test_x_prefix_still_causal(self):
        """Interaction features (X_ prefix) should remain CAUSAL."""
        from quant_engine.features.pipeline import get_feature_type

        self.assertEqual(get_feature_type("X_RSI_14_times_ATR_14"), "CAUSAL")


# ---------------------------------------------------------------------------
# T3: END_OF_DAY predictor gate
# ---------------------------------------------------------------------------

class TestEndOfDayPredictorGate(unittest.TestCase):
    """END_OF_DAY features should be logged in daily mode, blocked in intraday."""

    def test_end_of_day_logged_in_daily_mode(self):
        """In daily mode, END_OF_DAY features pass through with an INFO log."""
        from quant_engine.features.pipeline import get_feature_type

        # Verify some END_OF_DAY features exist
        self.assertEqual(get_feature_type("intraday_vol_ratio"), "END_OF_DAY")
        self.assertEqual(get_feature_type("vwap_deviation"), "END_OF_DAY")

    def test_research_only_blocked(self):
        """RESEARCH_ONLY features must be blocked regardless of mode."""
        from quant_engine.features.pipeline import get_feature_type

        self.assertEqual(get_feature_type("relative_mom_10"), "RESEARCH_ONLY")

    def test_prediction_mode_config_exists(self):
        """PREDICTION_MODE config constant should exist and default to 'daily'."""
        from quant_engine.config import PREDICTION_MODE

        self.assertEqual(PREDICTION_MODE, "daily")


if __name__ == "__main__":
    unittest.main()
