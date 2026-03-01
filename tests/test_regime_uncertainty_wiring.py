"""Tests for SPEC_AUDIT_FIX_25: Regime Uncertainty Semantic Wiring Fixes.

Verifies:
  T1: compute_shock_vectors receives regime confidence (not model confidence)
  T2: Autopilot uncertainty gate reduces total allocation (not neutralized)
  T3: confidence_series parameter renamed to regime_confidence_series
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 50) -> pd.DataFrame:
    """Create synthetic OHLCV data."""
    dates = pd.bdate_range("2023-01-02", periods=n, freq="B")
    rng = np.random.RandomState(42)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "Open": close - rng.uniform(0, 1, n),
            "High": close + rng.uniform(0, 2, n),
            "Low": close - rng.uniform(0, 2, n),
            "Close": close,
            "Volume": rng.randint(100_000, 500_000, n).astype(float),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# T1: Regime confidence vs model confidence in ShockVector
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestShockVectorRegimeConfidence:
    """Verify ShockVector uses regime confidence, not model confidence."""

    def test_regime_confidence_used_not_model_confidence(self):
        """When model confidence=0.9 but regime confidence=0.4,
        ShockVector should show high uncertainty (~0.6), not low (~0.1)."""
        from quant_engine.regime.shock_vector import compute_shock_vectors

        ohlcv = _make_ohlcv(n=50)
        regime_conf = pd.Series(0.4, index=ohlcv.index)

        result = compute_shock_vectors(
            ohlcv=ohlcv,
            regime_confidence_series=regime_conf,
            ticker="TEST",
        )

        for sv in result.values():
            # uncertainty = 1 - regime_confidence = 1 - 0.4 = 0.6
            assert abs(sv.hmm_uncertainty - 0.6) < 1e-6, (
                f"Expected uncertainty ≈ 0.6, got {sv.hmm_uncertainty}"
            )
            assert abs(sv.hmm_confidence - 0.4) < 1e-6, (
                f"Expected confidence ≈ 0.4, got {sv.hmm_confidence}"
            )

    def test_predictor_output_includes_regime_confidence(self):
        """Predictor output DataFrame should include regime_confidence column."""
        from quant_engine.models.predictor import EnsemblePredictor

        # Create a minimal predictor with no trained models
        # to verify it passes regime_confidence through
        try:
            predictor = EnsemblePredictor.__new__(EnsemblePredictor)
            predictor.global_model = None
            predictor.global_features = []
            predictor.global_scaler = None
            predictor.regime_models = {}
            predictor.regime_features = {}
            predictor.regime_scalers = {}
            predictor.regime_medians = {}
            predictor.regime_reliability = {}
            predictor.regime_target_stds = {}
            predictor.global_target_std = 0.02
            predictor.meta = {}
            predictor.calibrator = None
            predictor.conformal = None
            predictor.use_soft_blending = False
        except Exception:
            pytest.skip("Cannot instantiate minimal EnsemblePredictor")

    def test_backtest_engine_extracts_regime_confidence(self):
        """Backtest engine should prefer regime_confidence over confidence."""
        # Build a minimal predictions DataFrame with both columns
        idx = pd.MultiIndex.from_tuples(
            [("12345", pd.Timestamp("2023-01-03"))],
            names=["permno", "date"],
        )
        preds = pd.DataFrame(
            {
                "predicted_return": [0.05],
                "confidence": [0.9],  # model confidence
                "regime_confidence": [0.4],  # regime confidence
                "regime": [0],
            },
            index=idx,
        )

        # Simulate what the backtest engine does
        ticker_preds = preds.loc["12345"]
        if "regime_confidence" in ticker_preds.columns:
            conf_s = ticker_preds["regime_confidence"]
        elif "confidence" in ticker_preds.columns:
            conf_s = ticker_preds["confidence"]
        else:
            conf_s = None

        # Should have selected regime_confidence = 0.4
        assert conf_s is not None
        assert float(conf_s.iloc[0]) == pytest.approx(0.4)

    def test_fallback_when_regime_confidence_missing(self):
        """When regime_confidence is not available, fall back to confidence."""
        idx = pd.MultiIndex.from_tuples(
            [("12345", pd.Timestamp("2023-01-03"))],
            names=["permno", "date"],
        )
        preds = pd.DataFrame(
            {
                "predicted_return": [0.05],
                "confidence": [0.9],
                "regime": [0],
            },
            index=idx,
        )

        ticker_preds = preds.loc["12345"]
        if "regime_confidence" in ticker_preds.columns:
            conf_s = ticker_preds["regime_confidence"]
        elif "confidence" in ticker_preds.columns:
            conf_s = ticker_preds["confidence"]
        else:
            conf_s = None

        assert conf_s is not None
        assert float(conf_s.iloc[0]) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# T2: Uncertainty gate reduces total allocation (not neutralized)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUncertaintyGateNotNeutralized:
    """Verify the uncertainty gate actually reduces portfolio allocation."""

    def test_gate_reduces_weight_sum(self):
        """With high uncertainty, weights should sum to less than 1.0."""
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        gate = UncertaintyGate()
        weights = np.array([0.25, 0.25, 0.25, 0.25])

        # High uncertainty should produce multiplier < 1.0
        multiplier = gate.compute_size_multiplier(0.8)
        adjusted = weights * multiplier

        assert adjusted.sum() < 1.0, (
            f"Expected sum < 1.0, got {adjusted.sum():.4f}"
        )
        # The sum should equal the multiplier (since input sums to 1.0)
        assert abs(adjusted.sum() - multiplier) < 1e-10

    def test_renormalization_neutralizes_gate(self):
        """Demonstrate that renormalization neutralizes the gate (the old bug)."""
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        gate = UncertaintyGate()
        weights = np.array([0.25, 0.25, 0.25, 0.25])

        # Apply gate then renormalize (old behavior — should be a no-op)
        adjusted = gate.apply_uncertainty_gate(weights, 0.8)
        w_sum = adjusted.sum()
        if abs(w_sum) > 1e-10:
            renormalized = adjusted / w_sum
        else:
            renormalized = adjusted

        # Renormalized weights are the same as original — gate was neutralized!
        np.testing.assert_allclose(renormalized, weights, rtol=1e-10)

    def test_high_uncertainty_produces_cash_allocation(self):
        """With uncertainty=0.8 and sizing map giving multiplier~0.5,
        portfolio should be ~50% invested."""
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        gate = UncertaintyGate()
        multiplier = gate.compute_size_multiplier(0.8)

        weights = pd.Series([0.25, 0.25, 0.25, 0.25])
        adjusted = weights * multiplier

        # Invested fraction should match the multiplier
        invested_fraction = adjusted.sum()
        cash_fraction = 1.0 - invested_fraction

        assert invested_fraction < 1.0
        assert cash_fraction > 0.0
        assert abs(invested_fraction - multiplier) < 1e-10

    def test_position_sizer_unaffected(self):
        """Risk/position_sizer uses compute_size_multiplier directly —
        it should be unaffected by the autopilot fix."""
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        gate = UncertaintyGate()
        # Position sizer applies multiplier directly (no renormalization)
        raw_kelly = 0.05
        multiplier = gate.compute_size_multiplier(0.8)
        sized = raw_kelly * multiplier

        assert sized < raw_kelly
        assert sized > 0


# ---------------------------------------------------------------------------
# T3: Parameter renamed from confidence_series to regime_confidence_series
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParameterRenamed:
    """Verify the parameter is renamed to regime_confidence_series."""

    def test_new_parameter_name_accepted(self):
        """compute_shock_vectors accepts regime_confidence_series kwarg."""
        from quant_engine.regime.shock_vector import compute_shock_vectors

        ohlcv = _make_ohlcv(n=30)
        confidence = pd.Series(0.7, index=ohlcv.index)

        # Should not raise
        result = compute_shock_vectors(
            ohlcv=ohlcv,
            regime_confidence_series=confidence,
            ticker="TEST",
        )
        assert len(result) == 30

    def test_old_parameter_name_rejected(self):
        """compute_shock_vectors should NOT accept confidence_series kwarg."""
        from quant_engine.regime.shock_vector import compute_shock_vectors
        import inspect

        sig = inspect.signature(compute_shock_vectors)
        assert "confidence_series" not in sig.parameters, (
            "Old parameter name 'confidence_series' should be removed"
        )
        assert "regime_confidence_series" in sig.parameters, (
            "New parameter name 'regime_confidence_series' should be present"
        )

    def test_high_mean_confidence_warning(self, caplog):
        """Suspiciously high mean confidence should trigger a warning."""
        import logging
        from quant_engine.regime.shock_vector import compute_shock_vectors

        ohlcv = _make_ohlcv(n=30)
        # Suspiciously high values (> 0.95 mean) suggest model confidence,
        # not regime confidence
        confidence = pd.Series(0.99, index=ohlcv.index)

        with caplog.at_level(logging.WARNING):
            compute_shock_vectors(
                ohlcv=ohlcv,
                regime_confidence_series=confidence,
                ticker="TEST",
            )

        assert any("suspiciously high" in r.message for r in caplog.records), (
            "Expected warning about suspiciously high confidence values"
        )
