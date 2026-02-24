"""Integration tests for the full regime detection pipeline with Jump Model (Spec 001).

Verifies end-to-end compatibility of the PyPI JumpModel with all downstream
consumers: RegimeOutput interface, ensemble voting, regime_features output,
compute_regime_payload, and regime covariance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_features():
    """Generate synthetic OHLCV features with enough data for all regime methods.

    Creates a multi-phase price series with known structure:
      - Phase 1 (0-200): low-vol uptrend (trending bull)
      - Phase 2 (200-350): high-vol crash (high volatility / bear)
      - Phase 3 (350-500): sideways mean-reversion
      - Phase 4 (500-600): moderate uptrend recovery
    """
    np.random.seed(42)
    n = 600
    idx = pd.date_range("2020-01-01", periods=n, freq="B")

    # Multi-phase price
    prices = np.zeros(n)
    prices[0] = 100.0
    for i in range(1, n):
        if i < 200:
            # Low-vol uptrend
            prices[i] = prices[i - 1] * (1 + 0.0005 + np.random.randn() * 0.008)
        elif i < 350:
            # High-vol crash
            prices[i] = prices[i - 1] * (1 - 0.002 + np.random.randn() * 0.025)
        elif i < 500:
            # Sideways mean-reversion
            prices[i] = prices[i - 1] * (1 + np.random.randn() * 0.005)
        else:
            # Moderate recovery
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
    return features.iloc[50:].copy()  # trim warmup


# ── T7.1: detect_full with jump model ────────────────────────────────────


@pytest.mark.integration
class TestDetectFullJumpModel:
    """Full detect_full() with method='jump' on synthetic features."""

    def test_returns_valid_regime_output(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector, RegimeOutput

        detector = RegimeDetector(method="jump")
        out = detector.detect_full(synthetic_features)

        assert isinstance(out, RegimeOutput)
        assert out.model_type == "jump"
        assert out.regime.shape[0] == len(synthetic_features)
        assert out.confidence.shape[0] == len(synthetic_features)
        assert out.probabilities.shape == (len(synthetic_features), 4)

    def test_regime_values_in_canonical_set(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="jump")
        out = detector.detect_full(synthetic_features)

        unique = set(int(x) for x in out.regime.unique())
        assert unique.issubset({0, 1, 2, 3}), f"Unexpected regime values: {unique}"

    def test_probabilities_nan_free(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="jump")
        out = detector.detect_full(synthetic_features)

        assert not out.probabilities.isna().any().any(), "Probabilities contain NaN"

    def test_probabilities_sum_to_one(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="jump")
        out = detector.detect_full(synthetic_features)

        row_sums = out.probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_confidence_in_valid_range(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="jump")
        out = detector.detect_full(synthetic_features)

        assert (out.confidence >= 0.0).all(), "Confidence contains values < 0"
        assert (out.confidence <= 1.0).all(), "Confidence contains values > 1"

    def test_uncertainty_populated(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="jump")
        out = detector.detect_full(synthetic_features)

        assert out.uncertainty is not None
        assert len(out.uncertainty) == len(synthetic_features)
        assert (out.uncertainty >= 0.0).all()
        assert (out.uncertainty <= 1.0).all()

    def test_transition_matrix_none_for_jump(self, synthetic_features):
        """Jump models don't produce a transition matrix natively."""
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="jump")
        out = detector.detect_full(synthetic_features)

        # transition_matrix should be None (handled downstream by np.eye(4) fallback)
        assert out.transition_matrix is None


# ── T7.2: Ensemble includes jump model ────────────────────────────────────


@pytest.mark.integration
class TestEnsembleIncludesJump:
    """Ensemble mode includes jump model vote."""

    def test_ensemble_runs_with_jump(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="hmm")
        out = detector.detect_ensemble(synthetic_features)

        assert out.model_type == "ensemble"
        assert out.regime.shape[0] == len(synthetic_features)

    def test_ensemble_probabilities_valid(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="hmm")
        out = detector.detect_ensemble(synthetic_features)

        assert not out.probabilities.isna().any().any()
        row_sums = out.probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)


# ── T7.3: regime_features output unchanged ─────────────────────────────────


@pytest.mark.integration
class TestRegimeFeaturesUnchanged:
    """regime_features() output has same columns regardless of method."""

    def test_regime_features_columns_match(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector

        # Get columns from rule-based (guaranteed to work)
        det_rule = RegimeDetector(method="rule")
        rf_rule = det_rule.regime_features(synthetic_features)
        rule_cols = set(rf_rule.columns)

        # Get columns from jump model
        det_jump = RegimeDetector(method="jump")
        rf_jump = det_jump.regime_features(synthetic_features)
        jump_cols = set(rf_jump.columns)

        # Must have the same column set
        assert rule_cols == jump_cols, (
            f"Column mismatch: rule has {rule_cols - jump_cols}, "
            f"jump has {jump_cols - rule_cols}"
        )

    def test_regime_features_required_columns(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="jump")
        rf = detector.regime_features(synthetic_features)

        required = {
            "regime", "regime_confidence", "regime_model_type",
            "regime_duration",
            "regime_prob_0", "regime_prob_1", "regime_prob_2", "regime_prob_3",
            "regime_0", "regime_1", "regime_2", "regime_3",
            "regime_transition_prob",
        }
        assert required.issubset(set(rf.columns)), (
            f"Missing columns: {required - set(rf.columns)}"
        )

    def test_regime_features_no_nan(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="jump")
        rf = detector.regime_features(synthetic_features)

        # Check key numeric columns for NaN
        for col in ["regime", "regime_confidence", "regime_duration",
                     "regime_prob_0", "regime_prob_1", "regime_prob_2", "regime_prob_3"]:
            assert not rf[col].isna().any(), f"NaN in {col}"


# ── T7.4: compute_regime_payload compatibility ─────────────────────────────


@pytest.mark.integration
class TestComputeRegimePayloadJump:
    """compute_regime_payload() works with jump model output."""

    def test_payload_with_jump_model_output(self, synthetic_features):
        """Direct test of the regime output processing that
        compute_regime_payload does internally."""
        from quant_engine.regime.detector import RegimeDetector
        from quant_engine.config import REGIME_NAMES

        detector = RegimeDetector(method="jump")
        out = detector.detect_full(synthetic_features)

        # Simulate what compute_regime_payload does
        last_regime = out.regime.iloc[-1]
        current_state = int(last_regime) if pd.notna(last_regime) else 2
        current_probs = out.probabilities.iloc[-1]

        probs_pretty = {}
        for i in range(4):
            col = f"regime_prob_{i}"
            val = current_probs.get(col, 0.0)
            probs_pretty[REGIME_NAMES.get(i, f"Regime {i}")] = (
                float(val) if pd.notna(val) else 0.0
            )

        trans = out.transition_matrix if out.transition_matrix is not None else np.eye(4)
        trans = np.asarray(trans, dtype=float)
        if trans.ndim != 2:
            trans = np.eye(4)

        # Verify no "boolean value of NA is ambiguous" error
        assert isinstance(current_state, int)
        assert current_state in {0, 1, 2, 3}
        assert all(isinstance(v, float) for v in probs_pretty.values())
        assert abs(sum(probs_pretty.values()) - 1.0) < 0.01
        assert trans.shape == (4, 4)


# ── T7.5: HMM still works after changes ─────────────────────────────────


@pytest.mark.integration
class TestHMMStillWorks:
    """HMM-based detection still works after all changes."""

    def test_hmm_detect_full(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector

        # Use _hmm_detect directly to test HMM in isolation
        detector = RegimeDetector(method="hmm")
        out = detector._hmm_detect(synthetic_features)
        assert out.model_type == "hmm"
        assert out.regime.shape[0] == len(synthetic_features)
        assert not out.probabilities.isna().any().any()

    def test_rule_detect_full(self, synthetic_features):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="rule")
        out = detector.detect_full(synthetic_features)
        assert out.model_type == "rule"
        assert out.regime.shape[0] == len(synthetic_features)


# ── T7.6: Legacy jump model fallback ──────────────────────────────────────


@pytest.mark.integration
class TestLegacyFallback:
    """When REGIME_JUMP_USE_PYPI_PACKAGE=False, legacy model is used."""

    def test_legacy_jump_model(self, synthetic_features):
        import quant_engine.config as cfg

        orig = cfg.REGIME_JUMP_USE_PYPI_PACKAGE
        cfg.REGIME_JUMP_USE_PYPI_PACKAGE = False
        try:
            from quant_engine.regime.detector import RegimeDetector

            detector = RegimeDetector(method="jump")
            out = detector.detect_full(synthetic_features)
            assert out.model_type == "jump"
            assert out.regime.shape[0] == len(synthetic_features)
            assert not out.probabilities.isna().any().any()
        finally:
            cfg.REGIME_JUMP_USE_PYPI_PACKAGE = orig
