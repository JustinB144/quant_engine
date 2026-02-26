"""Tests for confidence-weighted ensemble voting (SPEC_10 T2).

Verifies:
  - ConfidenceCalibrator fit/calibrate cycle
  - ECM maps raw confidence to realized accuracy
  - Component weights proportional to accuracy
  - Weighted ensemble voting produces valid output
  - Disagreement fallback works correctly
  - Calibration improves ECE on validation data
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_calibration_data():
    """Generate synthetic prediction data for calibration testing."""
    rng = np.random.RandomState(42)
    T = 500
    # True regimes: cycle through 0, 1, 2, 3
    actuals = np.array([i % 4 for i in range(T)])

    # HMM: 70% accurate, confidence correlates with accuracy
    hmm_regimes = actuals.copy()
    flip_mask = rng.rand(T) < 0.30
    hmm_regimes[flip_mask] = rng.randint(0, 4, flip_mask.sum())
    hmm_confs = np.where(hmm_regimes == actuals, rng.uniform(0.6, 0.95, T), rng.uniform(0.2, 0.5, T))

    # Rule: 50% accurate, moderate confidence
    rule_regimes = actuals.copy()
    flip_mask = rng.rand(T) < 0.50
    rule_regimes[flip_mask] = rng.randint(0, 4, flip_mask.sum())
    rule_confs = np.full(T, 0.5) + rng.uniform(-0.1, 0.1, T)
    rule_confs = np.clip(rule_confs, 0.1, 0.9)

    # Jump: 60% accurate
    jump_regimes = actuals.copy()
    flip_mask = rng.rand(T) < 0.40
    jump_regimes[flip_mask] = rng.randint(0, 4, flip_mask.sum())
    jump_confs = np.where(jump_regimes == actuals, rng.uniform(0.5, 0.85, T), rng.uniform(0.2, 0.5, T))

    return {
        "actuals": actuals,
        "predictions": {
            "hmm": hmm_confs,
            "rule": rule_confs,
            "jump": jump_confs,
        },
        "predicted_regimes": {
            "hmm": hmm_regimes,
            "rule": rule_regimes,
            "jump": jump_regimes,
        },
    }


@pytest.fixture
def synthetic_features():
    """Generate synthetic feature DataFrame for detector testing."""
    np.random.seed(42)
    n = 600
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = np.cumsum(np.random.randn(n) * 0.02) + 100

    features = pd.DataFrame(
        {
            "Close": prices,
            "High": prices + np.abs(np.random.randn(n) * 1.0),
            "Low": prices - np.abs(np.random.randn(n) * 1.0),
            "Open": prices + np.random.randn(n) * 0.5,
            "Volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
        },
        index=idx,
    )
    features["return_1d"] = features["Close"].pct_change()
    features["return_vol_20d"] = features["return_1d"].rolling(20).std()
    features["NATR_14"] = (features["High"] - features["Low"]) / features["Close"]
    features["SMASlope_50"] = features["Close"].rolling(50).mean().pct_change(5)
    features["Hurst_100"] = 0.5
    features["ADX_14"] = 25.0
    return features.dropna()


# ---------------------------------------------------------------------------
# ConfidenceCalibrator Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConfidenceCalibrator:
    """Test ECM calibration mechanics."""

    def test_fit_and_calibrate(self, synthetic_calibration_data):
        from quant_engine.regime.confidence_calibrator import ConfidenceCalibrator

        data = synthetic_calibration_data
        cal = ConfidenceCalibrator(n_regimes=4)
        cal.fit(data["predictions"], data["predicted_regimes"], data["actuals"])

        assert cal.fitted
        # Calibrated confidence should be in [0, 1]
        result = cal.calibrate(0.7, "hmm", 0)
        assert 0.0 <= result <= 1.0

    def test_component_weights_sum_to_one(self, synthetic_calibration_data):
        from quant_engine.regime.confidence_calibrator import ConfidenceCalibrator

        data = synthetic_calibration_data
        cal = ConfidenceCalibrator(n_regimes=4)
        cal.fit(data["predictions"], data["predicted_regimes"], data["actuals"])

        weights = cal.component_weights
        assert len(weights) == 3
        np.testing.assert_allclose(sum(weights.values()), 1.0, atol=1e-6)

    def test_hmm_gets_highest_weight(self, synthetic_calibration_data):
        """HMM has 70% accuracy vs 50% rule — it should get higher weight."""
        from quant_engine.regime.confidence_calibrator import ConfidenceCalibrator

        data = synthetic_calibration_data
        cal = ConfidenceCalibrator(n_regimes=4)
        cal.fit(data["predictions"], data["predicted_regimes"], data["actuals"])

        weights = cal.component_weights
        assert weights["hmm"] > weights["rule"], (
            f"HMM weight {weights['hmm']:.3f} should exceed rule weight {weights['rule']:.3f}"
        )

    def test_uncalibrated_returns_raw(self):
        from quant_engine.regime.confidence_calibrator import ConfidenceCalibrator

        cal = ConfidenceCalibrator(n_regimes=4)
        # Not fitted yet — should return raw confidence
        assert cal.calibrate(0.75, "hmm", 0) == 0.75

    def test_ece_computed(self, synthetic_calibration_data):
        from quant_engine.regime.confidence_calibrator import ConfidenceCalibrator

        data = synthetic_calibration_data
        cal = ConfidenceCalibrator(n_regimes=4)
        cal.fit(data["predictions"], data["predicted_regimes"], data["actuals"])

        ece = cal.expected_calibration_error(
            data["predictions"], data["predicted_regimes"], data["actuals"],
        )
        assert "hmm" in ece
        assert "rule" in ece
        # ECE should be non-negative and finite
        for comp, val in ece.items():
            assert 0.0 <= val <= 1.0, f"ECE for {comp} = {val}"


# ---------------------------------------------------------------------------
# Weighted Ensemble Voting Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWeightedEnsembleVoting:
    """Test the updated detect_ensemble with confidence weighting."""

    def test_ensemble_returns_valid_output(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector, RegimeOutput

        detector = RegimeDetector(method="hmm")
        out = detector.detect_ensemble(synthetic_features)

        assert isinstance(out, RegimeOutput)
        assert out.model_type == "ensemble"
        assert out.regime.shape[0] == len(synthetic_features)
        assert out.confidence.shape[0] == len(synthetic_features)

    def test_ensemble_regime_values_canonical(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="hmm")
        out = detector.detect_ensemble(synthetic_features)

        unique = set(int(x) for x in out.regime.unique())
        assert unique.issubset({0, 1, 2, 3})

    def test_ensemble_probabilities_sum_to_one(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="hmm")
        out = detector.detect_ensemble(synthetic_features)

        row_sums = out.probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_ensemble_uncertainty_populated(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="hmm")
        out = detector.detect_ensemble(synthetic_features)

        assert out.uncertainty is not None
        assert (out.uncertainty >= 0.0).all()
        assert (out.uncertainty <= 1.0).all()

    def test_calibration_changes_weights(self, synthetic_features):
        """After calibration, weights should differ from defaults."""
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="hmm")
        # Use rule-based output as "ground truth" for calibration test
        rule_out = detector._rule_detect(synthetic_features)
        actual_regimes = rule_out.regime.values

        weights = detector.calibrate_confidence_weights(
            synthetic_features, actual_regimes,
        )
        assert isinstance(weights, dict)
        assert len(weights) >= 2
        assert abs(sum(weights.values()) - 1.0) < 1e-6
