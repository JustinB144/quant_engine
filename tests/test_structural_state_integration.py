"""Integration tests for the Structural State Layer (SPEC_03).

Tests cover end-to-end integration of:
  - BOCPD + RegimeDetector (detect_with_shock_context)
  - ShockVector generation from real-ish feature DataFrames
  - Batch ShockVector processing
  - HMM observation matrix feature validation
  - RegimeOutput augmented with BOCPD signals
  - Config constant availability
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_features():
    """Generate synthetic OHLCV features with enough data for regime detection.

    Creates a multi-phase price series with known structure:
      - Phase 1 (0-200): low-vol uptrend (trending bull)
      - Phase 2 (200-350): high-vol crash (high volatility / bear)
      - Phase 3 (350-500): sideways mean-reversion
      - Phase 4 (500-600): moderate uptrend recovery
    """
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
    return features.iloc[50:].copy()


@pytest.fixture
def features_with_return_shift():
    """Features with a clear return shift at bar 200 (for BOCPD testing)."""
    np.random.seed(99)
    n = 400
    idx = pd.date_range("2021-01-01", periods=n, freq="B")

    prices = np.zeros(n)
    prices[0] = 100.0
    for i in range(1, n):
        if i < 200:
            prices[i] = prices[i - 1] * (1 + 0.001 + np.random.randn() * 0.005)
        else:
            prices[i] = prices[i - 1] * (1 - 0.003 + np.random.randn() * 0.020)

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
    return features.iloc[50:].copy()


# ── Config Constants Available ───────────────────────────────────────────


class TestConfigConstants:
    """Verify that BOCPD and ShockVector config constants are accessible."""

    def test_bocpd_config_constants(self):
        from quant_engine.config import (
            BOCPD_ENABLED,
            BOCPD_HAZARD_FUNCTION,
            BOCPD_HAZARD_LAMBDA,
            BOCPD_LIKELIHOOD_TYPE,
            BOCPD_RUNLENGTH_DEPTH,
            BOCPD_CHANGEPOINT_THRESHOLD,
        )

        assert isinstance(BOCPD_ENABLED, bool)
        assert BOCPD_HAZARD_FUNCTION in ("constant", "geometric")
        assert 0 < BOCPD_HAZARD_LAMBDA < 1
        assert BOCPD_LIKELIHOOD_TYPE == "gaussian"
        assert BOCPD_RUNLENGTH_DEPTH >= 2
        assert 0 < BOCPD_CHANGEPOINT_THRESHOLD <= 1

    def test_shock_vector_config_constants(self):
        from quant_engine.config import (
            SHOCK_VECTOR_SCHEMA_VERSION,
            SHOCK_VECTOR_INCLUDE_STRUCTURAL,
        )

        assert SHOCK_VECTOR_SCHEMA_VERSION == "1.0"
        assert isinstance(SHOCK_VECTOR_INCLUDE_STRUCTURAL, bool)


# ── RegimeDetector with BOCPD ────────────────────────────────────────────


class TestRegimeDetectorBOCPD:
    """Test RegimeDetector with BOCPD enabled."""

    def test_detector_initializes_with_bocpd(self):
        from quant_engine.regime.detector import RegimeDetector

        det = RegimeDetector(method="rule", enable_bocpd=True)
        assert det.enable_bocpd is True
        assert det._bocpd is not None

    def test_detector_initializes_without_bocpd(self):
        from quant_engine.regime.detector import RegimeDetector

        det = RegimeDetector(method="rule", enable_bocpd=False)
        assert det.enable_bocpd is False
        assert det._bocpd is None

    def test_detect_still_works_with_bocpd(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector, RegimeOutput

        det = RegimeDetector(method="rule", enable_bocpd=True)
        out = det.detect_full(synthetic_features)

        assert isinstance(out, RegimeOutput)
        assert len(out.regime) == len(synthetic_features)

    def test_detect_still_works_without_bocpd(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector, RegimeOutput

        det = RegimeDetector(method="rule", enable_bocpd=False)
        out = det.detect_full(synthetic_features)

        assert isinstance(out, RegimeOutput)
        assert len(out.regime) == len(synthetic_features)


# ── detect_with_shock_context ────────────────────────────────────────────


class TestDetectWithShockContext:
    """Test RegimeDetector.detect_with_shock_context()."""

    def test_returns_shock_vector(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector
        from quant_engine.regime.shock_vector import ShockVector

        det = RegimeDetector(method="rule", enable_bocpd=True)
        sv = det.detect_with_shock_context(synthetic_features, ticker="AAPL")

        assert isinstance(sv, ShockVector)
        assert sv.ticker == "AAPL"
        assert sv.schema_version == "1.0"
        assert 0 <= sv.hmm_regime <= 3
        assert 0.0 <= sv.hmm_confidence <= 1.0
        assert 0.0 <= sv.bocpd_changepoint_prob <= 1.0
        assert sv.bocpd_runlength >= 0

    def test_returns_shock_vector_without_bocpd(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector
        from quant_engine.regime.shock_vector import ShockVector

        det = RegimeDetector(method="rule", enable_bocpd=False)
        sv = det.detect_with_shock_context(synthetic_features, ticker="MSFT")

        assert isinstance(sv, ShockVector)
        assert sv.bocpd_changepoint_prob == 0.0
        assert sv.bocpd_runlength == 0

    def test_shock_vector_validates(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector
        from quant_engine.regime.shock_vector import ShockVectorValidator

        det = RegimeDetector(method="rule", enable_bocpd=True)
        sv = det.detect_with_shock_context(synthetic_features, ticker="GOOGL")

        is_valid, errors = ShockVectorValidator.validate(sv)
        assert is_valid, f"ShockVector should be valid: {errors}"

    def test_shock_vector_has_model_type(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector

        det = RegimeDetector(method="rule", enable_bocpd=False)
        sv = det.detect_with_shock_context(synthetic_features, ticker="TEST")

        assert sv.ensemble_model_type == "rule"

    def test_jump_detection_on_synthetic_data(self, features_with_return_shift):
        from quant_engine.regime.detector import RegimeDetector

        det = RegimeDetector(method="rule", enable_bocpd=True)
        sv = det.detect_with_shock_context(features_with_return_shift, ticker="CRASH")

        # The last bar may or may not be a jump, but the vector should be valid.
        assert isinstance(sv.jump_detected, bool)
        assert isinstance(sv.jump_magnitude, float)

    def test_bocpd_signals_populated(self, features_with_return_shift):
        from quant_engine.regime.detector import RegimeDetector

        det = RegimeDetector(method="rule", enable_bocpd=True)
        sv = det.detect_with_shock_context(features_with_return_shift, ticker="SIG")

        # With BOCPD enabled and enough data, we should get meaningful signals.
        assert sv.bocpd_changepoint_prob >= 0.0
        # Runlength should be a non-negative integer.
        assert sv.bocpd_runlength >= 0


# ── Batch ShockVector Generation ─────────────────────────────────────────


class TestBatchShockVectorGeneration:
    """Test detect_batch_with_shock_context()."""

    def test_batch_returns_dict(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector
        from quant_engine.regime.shock_vector import ShockVector

        det = RegimeDetector(method="rule", enable_bocpd=True)
        features_by_id = {
            "AAPL": synthetic_features.copy(),
            "MSFT": synthetic_features.copy(),
        }

        result = det.detect_batch_with_shock_context(features_by_id)

        assert isinstance(result, dict)
        assert len(result) == 2
        assert "AAPL" in result
        assert "MSFT" in result

        for ticker, sv in result.items():
            assert isinstance(sv, ShockVector)
            assert sv.ticker == ticker

    def test_batch_all_vectors_valid(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector
        from quant_engine.regime.shock_vector import ShockVectorValidator

        det = RegimeDetector(method="rule", enable_bocpd=True)
        features_by_id = {
            f"STOCK_{i}": synthetic_features.copy() for i in range(5)
        }

        result = det.detect_batch_with_shock_context(features_by_id)
        vectors = list(result.values())
        errors = ShockVectorValidator.batch_validate(vectors)

        assert len(errors) == 0, f"Batch validation errors: {errors}"


# ── HMM Observation Matrix Validation ────────────────────────────────────


class TestHMMObservationMatrixValidation:
    """Test validate_hmm_observation_features()."""

    def test_valid_features(self, synthetic_features):
        from quant_engine.regime.detector import validate_hmm_observation_features

        is_valid, warnings = validate_hmm_observation_features(synthetic_features)
        assert is_valid, f"Core features should be valid: {warnings}"

    def test_missing_core_feature(self, synthetic_features):
        from quant_engine.regime.detector import validate_hmm_observation_features

        # Remove a core feature.
        incomplete = synthetic_features.drop(columns=["return_1d"])
        is_valid, warnings = validate_hmm_observation_features(incomplete)
        assert not is_valid
        assert any("return_1d" in w for w in warnings)

    def test_high_nan_warning(self, synthetic_features):
        from quant_engine.regime.detector import validate_hmm_observation_features

        # Introduce lots of NaN.
        modified = synthetic_features.copy()
        modified.loc[modified.index[:200], "NATR_14"] = np.nan
        is_valid, warnings = validate_hmm_observation_features(modified)

        # Should still be valid (core features present) but warn about NaN.
        assert is_valid
        assert any("HIGH_NAN" in w for w in warnings)

    def test_reports_missing_extended_features(self):
        from quant_engine.regime.detector import validate_hmm_observation_features

        # Minimal features — only core present.
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        minimal = pd.DataFrame(
            {
                "return_1d": np.random.randn(100) * 0.01,
                "return_vol_20d": np.abs(np.random.randn(100)) * 0.02,
                "NATR_14": np.abs(np.random.randn(100)) * 0.01,
                "SMASlope_50": np.random.randn(100) * 0.001,
            },
            index=idx,
        )
        is_valid, warnings = validate_hmm_observation_features(minimal)
        assert is_valid
        assert any("ABSENT" in w for w in warnings)


# ── Package Exports ──────────────────────────────────────────────────────


class TestPackageExports:
    """Verify all new classes are exported from regime package."""

    def test_bocpd_exports(self):
        from quant_engine.regime import BOCPDDetector, BOCPDResult, BOCPDBatchResult
        assert BOCPDDetector is not None
        assert BOCPDResult is not None
        assert BOCPDBatchResult is not None

    def test_shock_vector_exports(self):
        from quant_engine.regime import ShockVector, ShockVectorValidator
        assert ShockVector is not None
        assert ShockVectorValidator is not None

    def test_validation_export(self):
        from quant_engine.regime import validate_hmm_observation_features
        assert callable(validate_hmm_observation_features)


# ── ShockVector to_dict JSON Compatibility ───────────────────────────────


class TestShockVectorJSON:
    """Test that ShockVector.to_dict() produces JSON-serializable output."""

    def test_to_dict_is_json_serializable(self, synthetic_features):
        import json
        from quant_engine.regime.detector import RegimeDetector

        det = RegimeDetector(method="rule", enable_bocpd=True)
        sv = det.detect_with_shock_context(synthetic_features, ticker="JSON_TEST")
        d = sv.to_dict()

        # Should not raise.
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        assert "JSON_TEST" in json_str

    def test_to_dict_excludes_transition_matrix(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector

        det = RegimeDetector(method="rule", enable_bocpd=True)
        sv = det.detect_with_shock_context(synthetic_features, ticker="EXCL")
        d = sv.to_dict()

        assert "transition_matrix" not in d
