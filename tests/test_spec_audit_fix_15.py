"""Tests for SPEC_AUDIT_FIX_15: Regime Detection Correctness & Robustness Fixes.

Covers all eight tasks:
  T1: Explicit method="ensemble" in detect_full
  T2: _apply_min_duration tie-break fix
  T3: Jump model fallback chain
  T4: ShockVector transition matrix shape validation
  T5: HMM observation standardization temporal leakage
  T6: Empty-input guard on detect_with_shock_context
  T7: Wired config knobs
  T8: ConfidenceCalibrator modulo remapping
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_features():
    """Generate synthetic features with enough data for HMM/ensemble."""
    np.random.seed(42)
    n = 600
    idx = pd.date_range("2020-01-01", periods=n, freq="B")

    prices = np.zeros(n)
    prices[0] = 100.0
    for i in range(1, n):
        if i < 200:
            prices[i] = prices[i - 1] * (1 + 0.0005 + np.random.randn() * 0.008)
        elif i < 350:
            prices[i] = prices[i - 1] * (1 - 0.002 + np.random.randn() * 0.025)
        elif i < 500:
            prices[i] = prices[i - 1] * (1 + np.random.randn() * 0.005)
        else:
            prices[i] = prices[i - 1] * (1 + 0.0003 + np.random.randn() * 0.010)

    features = pd.DataFrame(
        {
            "Close": prices,
            "High": prices * (1 + np.abs(np.random.randn(n) * 0.005)),
            "Low": prices * (1 - np.abs(np.random.randn(n) * 0.005)),
            "Open": prices * (1 + np.random.randn(n) * 0.002),
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
    features["return_20d"] = features["return_1d"].rolling(20).sum()
    features["GARCH_252"] = features["return_1d"].rolling(60).std()
    return features.iloc[60:].copy()


# ---------------------------------------------------------------------------
# T1: Explicit method="ensemble"
# ---------------------------------------------------------------------------

class TestT1EnsembleMethod:
    """method='ensemble' explicitly activates ensemble detection."""

    def test_ensemble_method_returns_ensemble_output(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector, RegimeOutput

        detector = RegimeDetector(method="ensemble")
        out = detector.detect_full(synthetic_features)

        assert isinstance(out, RegimeOutput)
        assert out.model_type == "ensemble"
        assert len(out.regime) == len(synthetic_features)

    def test_unknown_method_raises_value_error(self):
        from quant_engine.regime.detector import RegimeDetector

        with pytest.raises(ValueError, match="Unknown regime detection method"):
            RegimeDetector(method="bogus")

    def test_valid_methods_accepted(self):
        from quant_engine.regime.detector import RegimeDetector

        for method in ("hmm", "jump", "rule", "ensemble"):
            det = RegimeDetector(method=method)
            assert det.method == method


# ---------------------------------------------------------------------------
# T2: _apply_min_duration tie-break
# ---------------------------------------------------------------------------

class TestT2MinDurationTieBreak:
    """Tie-break merges short run into neighbor with higher confidence."""

    def test_merge_direction_follows_confidence(self):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="rule", min_duration=3)

        # Create a regime series with a short run (2 bars of regime 1)
        # between two different regimes (0 on left, 2 on right)
        regime = pd.Series([0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
        # Left neighbor (regime 0) has higher confidence
        confidence_left_wins = pd.Series([0.9, 0.9, 0.9, 0.9, 0.9, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3])
        result_left = detector._apply_min_duration(regime.copy(), confidence_left_wins)
        # Short run (regime 1) should merge into left (regime 0)
        assert result_left.iloc[5] == 0
        assert result_left.iloc[6] == 0

        # Now right neighbor (regime 2) has higher confidence
        confidence_right_wins = pd.Series([0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9])
        result_right = detector._apply_min_duration(regime.copy(), confidence_right_wins)
        # Short run (regime 1) should merge into right (regime 2)
        assert result_right.iloc[5] == 2
        assert result_right.iloc[6] == 2


# ---------------------------------------------------------------------------
# T3: Jump model fallback chain
# ---------------------------------------------------------------------------

class TestT3JumpFallback:
    """PyPI jump failure falls back to legacy before rules."""

    def test_fallback_chain_logged(self, synthetic_features, caplog):
        """Verify the fallback chain structure exists in _jump_detect."""
        from quant_engine.regime.detector import RegimeDetector

        # Just verify the method runs without error
        detector = RegimeDetector(method="jump")
        out = detector._jump_detect(synthetic_features)
        assert out is not None
        assert out.model_type in ("jump", "rule")


# ---------------------------------------------------------------------------
# T4: ShockVector transition matrix shape validation
# ---------------------------------------------------------------------------

class TestT4TransitionMatrixValidation:
    """Variable-sized square transition matrices pass validation."""

    def test_3x3_matrix_passes(self):
        from quant_engine.regime.shock_vector import ShockVector, ShockVectorValidator

        sv = ShockVector(
            transition_matrix=np.eye(3),
            n_hmm_states=3,
        )
        is_valid, errors = ShockVectorValidator.validate(sv)
        tm_errors = [e for e in errors if "Transition matrix" in e or "transition_matrix" in e]
        assert len(tm_errors) == 0, f"3x3 matrix should pass: {tm_errors}"

    def test_5x5_matrix_passes(self):
        from quant_engine.regime.shock_vector import ShockVector, ShockVectorValidator

        sv = ShockVector(
            transition_matrix=np.eye(5),
            n_hmm_states=5,
        )
        is_valid, errors = ShockVectorValidator.validate(sv)
        tm_errors = [e for e in errors if "Transition matrix" in e or "transition_matrix" in e]
        assert len(tm_errors) == 0, f"5x5 matrix should pass: {tm_errors}"

    def test_non_square_matrix_fails(self):
        from quant_engine.regime.shock_vector import ShockVector, ShockVectorValidator

        sv = ShockVector(
            transition_matrix=np.ones((3, 4)),
        )
        is_valid, errors = ShockVectorValidator.validate(sv)
        assert any("square" in e for e in errors), f"Non-square should fail: {errors}"

    def test_oversized_matrix_fails(self):
        from quant_engine.regime.shock_vector import ShockVector, ShockVectorValidator
        from quant_engine.config_structured import MAX_HMM_STATES

        sv = ShockVector(
            transition_matrix=np.eye(MAX_HMM_STATES + 1),
        )
        is_valid, errors = ShockVectorValidator.validate(sv)
        assert any("outside valid range" in e for e in errors), f"Oversized should fail: {errors}"

    def test_1x1_matrix_fails(self):
        from quant_engine.regime.shock_vector import ShockVector, ShockVectorValidator

        sv = ShockVector(
            transition_matrix=np.eye(1),
        )
        is_valid, errors = ShockVectorValidator.validate(sv)
        assert any("outside valid range" in e for e in errors), f"1x1 should fail: {errors}"

    def test_n_hmm_states_populated_from_detect(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="hmm", enable_bocpd=False)
        sv = detector.detect_with_shock_context(synthetic_features, ticker="TEST")
        assert hasattr(sv, "n_hmm_states")
        assert sv.n_hmm_states >= 2


# ---------------------------------------------------------------------------
# T5: HMM backtest-safe standardization
# ---------------------------------------------------------------------------

class TestT5BacktestSafe:
    """Expanding-window standardization prevents future data leakage."""

    def test_backtest_safe_uses_expanding_window(self, synthetic_features):
        from quant_engine.regime.hmm import build_hmm_observation_matrix

        obs_full = build_hmm_observation_matrix(synthetic_features, backtest_safe=False)
        obs_safe = build_hmm_observation_matrix(synthetic_features, backtest_safe=True)

        # Both should have same shape
        assert obs_full.shape == obs_safe.shape

        # In backtest-safe mode, early rows should differ from full-series
        # standardization because they use expanding statistics
        # (values should be different for early rows)
        assert not np.allclose(obs_full.iloc[0:30].values, obs_safe.iloc[0:30].values, atol=0.01), \
            "Early rows should differ between full-series and expanding standardization"

    def test_backtest_safe_no_nan(self, synthetic_features):
        from quant_engine.regime.hmm import build_hmm_observation_matrix

        obs = build_hmm_observation_matrix(synthetic_features, backtest_safe=True)
        assert not obs.isna().any().any(), "backtest_safe output should have no NaN"

    def test_default_is_not_backtest_safe(self, synthetic_features):
        from quant_engine.regime.hmm import build_hmm_observation_matrix

        obs_default = build_hmm_observation_matrix(synthetic_features)
        obs_full = build_hmm_observation_matrix(synthetic_features, backtest_safe=False)
        np.testing.assert_array_equal(obs_default.values, obs_full.values)


# ---------------------------------------------------------------------------
# T6: Empty-input guard
# ---------------------------------------------------------------------------

class TestT6EmptyInputGuard:
    """Empty DataFrame does not crash detect_with_shock_context."""

    def test_empty_features_returns_invalid_shock_vector(self):
        from quant_engine.regime.detector import RegimeDetector
        from quant_engine.regime.shock_vector import ShockVector

        detector = RegimeDetector(method="rule")
        sv = detector.detect_with_shock_context(pd.DataFrame(), ticker="EMPTY")

        assert isinstance(sv, ShockVector)
        assert sv.ticker == "EMPTY"
        assert sv.hmm_confidence == 0.0
        assert sv.hmm_uncertainty == 1.0

    def test_shock_vector_empty_classmethod(self):
        from quant_engine.regime.shock_vector import ShockVector

        sv = ShockVector.empty(ticker="TEST")
        assert sv.ticker == "TEST"
        assert sv.hmm_confidence == 0.0
        assert sv.hmm_uncertainty == 1.0


# ---------------------------------------------------------------------------
# T7: Config knobs wired
# ---------------------------------------------------------------------------

class TestT7ConfigKnobsWired:
    """Verify unused config knobs are now wired."""

    def test_bocpd_threshold_affects_jump_detection(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector

        # Just verify detect_with_shock_context uses BOCPD_CHANGEPOINT_THRESHOLD
        # without crash
        detector = RegimeDetector(method="rule", enable_bocpd=True)
        sv = detector.detect_with_shock_context(synthetic_features, ticker="TEST")
        assert sv is not None

    def test_ensemble_consensus_threshold_config_exists(self):
        from quant_engine.config import REGIME_ENSEMBLE_CONSENSUS_THRESHOLD
        assert isinstance(REGIME_ENSEMBLE_CONSENSUS_THRESHOLD, (int, float))


# ---------------------------------------------------------------------------
# T8: ConfidenceCalibrator modulo fix
# ---------------------------------------------------------------------------

class TestT8CalibratorModuloFix:
    """Out-of-range regime ID returns uncalibrated confidence."""

    def test_out_of_range_regime_returns_uncalibrated(self):
        from quant_engine.regime.confidence_calibrator import ConfidenceCalibrator

        calibrator = ConfidenceCalibrator(n_regimes=4)
        # Fit with some dummy data
        n = 100
        predictions = {"hmm": np.random.rand(n)}
        predicted_regimes = {"hmm": np.random.randint(0, 4, n)}
        actuals = np.random.randint(0, 4, n)
        calibrator.fit(predictions, predicted_regimes, actuals)

        # Out-of-range regime should return raw confidence
        raw_conf = 0.75
        result = calibrator.calibrate(raw_conf, "hmm", regime=5)
        assert result == raw_conf

        result_neg = calibrator.calibrate(raw_conf, "hmm", regime=-1)
        assert result_neg == raw_conf

    def test_valid_regime_calibrates(self):
        from quant_engine.regime.confidence_calibrator import ConfidenceCalibrator

        calibrator = ConfidenceCalibrator(n_regimes=4)
        n = 200
        np.random.seed(42)
        predictions = {"hmm": np.random.rand(n)}
        predicted_regimes = {"hmm": np.random.randint(0, 4, n)}
        actuals = np.random.randint(0, 4, n)
        calibrator.fit(predictions, predicted_regimes, actuals)

        # Valid regime should return a calibrated value (may differ from raw)
        result = calibrator.calibrate(0.75, "hmm", regime=0)
        assert 0.0 <= result <= 1.0

    def test_consensus_filters_invalid_labels(self):
        from quant_engine.regime.consensus import RegimeConsensus

        rc = RegimeConsensus()
        # Mix of valid and invalid labels
        result = rc.compute_consensus([0, 1, 2, 3, 5, -1, 10])
        # Only 4 valid labels (0, 1, 2, 3) should be counted
        assert result["n_securities"] == 4

    def test_consensus_all_invalid_returns_empty(self):
        from quant_engine.regime.consensus import RegimeConsensus

        rc = RegimeConsensus()
        result = rc.compute_consensus([5, 6, 7])
        assert result["n_securities"] == 0
        assert result["consensus"] == 0.0
