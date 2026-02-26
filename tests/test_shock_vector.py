"""Unit tests for ShockVector and ShockVectorValidator.

Tests cover:
  - Construction and validation
  - Schema version enforcement
  - Boundary checks for all numeric fields
  - Serialization/deserialization round-trips
  - Shock event detection logic
  - Batch validation
  - Edge cases (empty structural features, None transition matrix)
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest


# ── Construction Tests ───────────────────────────────────────────────────


class TestShockVectorConstruction:
    """Test ShockVector construction and defaults."""

    def test_default_construction(self):
        from quant_engine.regime.shock_vector import ShockVector

        sv = ShockVector()
        assert sv.schema_version == "1.0"
        assert sv.hmm_regime == 0
        assert sv.hmm_confidence == 0.5
        assert sv.hmm_uncertainty == 0.5
        assert sv.bocpd_changepoint_prob == 0.0
        assert sv.bocpd_runlength == 0
        assert sv.jump_detected is False
        assert sv.jump_magnitude == 0.0
        assert sv.structural_features == {}
        assert sv.transition_matrix is None
        assert sv.ensemble_model_type == "hmm"

    def test_full_construction(self):
        from quant_engine.regime.shock_vector import ShockVector

        sv = ShockVector(
            schema_version="1.0",
            timestamp=datetime(2026, 2, 25, 12, 0, 0),
            ticker="AAPL",
            hmm_regime=2,
            hmm_confidence=0.85,
            hmm_uncertainty=0.12,
            bocpd_changepoint_prob=0.72,
            bocpd_runlength=15,
            jump_detected=True,
            jump_magnitude=-0.045,
            structural_features={"spectral_entropy": 0.65, "ssa_trend_strength": 0.23},
            transition_matrix=np.eye(4) * 0.85,
            ensemble_model_type="ensemble",
        )

        assert sv.ticker == "AAPL"
        assert sv.hmm_regime == 2
        assert sv.hmm_confidence == 0.85
        assert sv.bocpd_changepoint_prob == 0.72
        assert sv.jump_detected is True
        assert len(sv.structural_features) == 2

    def test_invalid_schema_version_raises(self):
        from quant_engine.regime.shock_vector import ShockVector
        with pytest.raises(ValueError, match="Unsupported schema version"):
            ShockVector(schema_version="2.0")

    def test_invalid_regime_raises(self):
        from quant_engine.regime.shock_vector import ShockVector
        with pytest.raises(ValueError, match="hmm_regime"):
            ShockVector(hmm_regime=5)

    def test_negative_regime_raises(self):
        from quant_engine.regime.shock_vector import ShockVector
        with pytest.raises(ValueError, match="hmm_regime"):
            ShockVector(hmm_regime=-1)

    def test_confidence_clipped_to_bounds(self):
        from quant_engine.regime.shock_vector import ShockVector

        sv = ShockVector(hmm_confidence=1.5)
        assert sv.hmm_confidence == 1.0

        sv2 = ShockVector(hmm_confidence=-0.3)
        assert sv2.hmm_confidence == 0.0

    def test_changepoint_prob_clipped(self):
        from quant_engine.regime.shock_vector import ShockVector

        sv = ShockVector(bocpd_changepoint_prob=2.0)
        assert sv.bocpd_changepoint_prob == 1.0

    def test_negative_runlength_clipped(self):
        from quant_engine.regime.shock_vector import ShockVector

        sv = ShockVector(bocpd_runlength=-5)
        assert sv.bocpd_runlength == 0


# ── Validation Tests ─────────────────────────────────────────────────────


class TestShockVectorValidator:
    """Test ShockVectorValidator.validate()."""

    def test_valid_shock_vector(self):
        from quant_engine.regime.shock_vector import ShockVector, ShockVectorValidator

        sv = ShockVector(
            hmm_regime=2,
            hmm_confidence=0.8,
            hmm_uncertainty=0.2,
            bocpd_changepoint_prob=0.3,
            bocpd_runlength=50,
            ensemble_model_type="hmm",
        )
        is_valid, errors = ShockVectorValidator.validate(sv)
        assert is_valid, f"Expected valid, got errors: {errors}"
        assert len(errors) == 0

    def test_validates_all_regimes(self):
        from quant_engine.regime.shock_vector import ShockVector, ShockVectorValidator

        for regime in range(4):
            sv = ShockVector(hmm_regime=regime)
            is_valid, errors = ShockVectorValidator.validate(sv)
            assert is_valid, f"Regime {regime} should be valid: {errors}"

    def test_validates_all_model_types(self):
        from quant_engine.regime.shock_vector import ShockVector, ShockVectorValidator

        for mt in ["hmm", "jump", "ensemble", "rule"]:
            sv = ShockVector(ensemble_model_type=mt)
            is_valid, errors = ShockVectorValidator.validate(sv)
            assert is_valid, f"Model type {mt!r} should be valid: {errors}"

    def test_rejects_invalid_model_type(self):
        from quant_engine.regime.shock_vector import ShockVector, ShockVectorValidator

        sv = ShockVector()
        sv.ensemble_model_type = "random_forest"
        is_valid, errors = ShockVectorValidator.validate(sv)
        assert not is_valid
        assert any("ensemble_model_type" in e for e in errors)

    def test_validates_structural_features_types(self):
        from quant_engine.regime.shock_vector import ShockVector, ShockVectorValidator

        sv = ShockVector(structural_features={"good": 1.5, "also_good": 0})
        is_valid, errors = ShockVectorValidator.validate(sv)
        assert is_valid

    def test_rejects_non_numeric_structural_feature(self):
        from quant_engine.regime.shock_vector import ShockVector, ShockVectorValidator

        sv = ShockVector()
        sv.structural_features = {"bad": "not_a_number"}
        is_valid, errors = ShockVectorValidator.validate(sv)
        assert not is_valid
        assert any("structural_features" in e for e in errors)

    def test_rejects_non_finite_structural_feature(self):
        from quant_engine.regime.shock_vector import ShockVector, ShockVectorValidator

        sv = ShockVector(structural_features={"inf_val": float("inf")})
        is_valid, errors = ShockVectorValidator.validate(sv)
        assert not is_valid
        assert any("not finite" in e for e in errors)

    def test_validates_transition_matrix_shape(self):
        from quant_engine.regime.shock_vector import ShockVector, ShockVectorValidator

        sv = ShockVector(transition_matrix=np.eye(4))
        is_valid, errors = ShockVectorValidator.validate(sv)
        assert is_valid

    def test_rejects_wrong_transition_matrix_shape(self):
        from quant_engine.regime.shock_vector import ShockVector, ShockVectorValidator

        sv = ShockVector(transition_matrix=np.eye(3))
        is_valid, errors = ShockVectorValidator.validate(sv)
        assert not is_valid
        assert any("transition_matrix" in e for e in errors)


# ── Batch Validation Tests ───────────────────────────────────────────────


class TestShockVectorBatchValidation:
    """Test ShockVectorValidator.batch_validate()."""

    def test_all_valid(self):
        from quant_engine.regime.shock_vector import ShockVector, ShockVectorValidator

        vectors = [ShockVector(hmm_regime=i) for i in range(4)]
        errors = ShockVectorValidator.batch_validate(vectors)
        assert len(errors) == 0

    def test_some_invalid(self):
        from quant_engine.regime.shock_vector import ShockVector, ShockVectorValidator

        vectors = [
            ShockVector(hmm_regime=0),
            ShockVector(hmm_regime=1),
        ]
        # Manually break one vector's model type.
        vectors[1].ensemble_model_type = "invalid"
        errors = ShockVectorValidator.batch_validate(vectors)
        assert 1 in errors
        assert 0 not in errors

    def test_empty_batch(self):
        from quant_engine.regime.shock_vector import ShockVectorValidator
        errors = ShockVectorValidator.batch_validate([])
        assert len(errors) == 0


# ── Serialization Tests ──────────────────────────────────────────────────


class TestShockVectorSerialization:
    """Test to_dict / from_dict round-trip."""

    def test_to_dict(self):
        from quant_engine.regime.shock_vector import ShockVector

        sv = ShockVector(
            ticker="MSFT",
            hmm_regime=1,
            hmm_confidence=0.9,
            structural_features={"entropy": 0.5},
            transition_matrix=np.eye(4),
        )
        d = sv.to_dict()

        assert isinstance(d, dict)
        assert d["ticker"] == "MSFT"
        assert d["hmm_regime"] == 1
        assert isinstance(d["timestamp"], str)
        assert "transition_matrix" not in d

    def test_from_dict(self):
        from quant_engine.regime.shock_vector import ShockVector

        d = {
            "schema_version": "1.0",
            "timestamp": "2026-02-25T12:00:00",
            "ticker": "GOOGL",
            "hmm_regime": 3,
            "hmm_confidence": 0.7,
            "hmm_uncertainty": 0.3,
            "bocpd_changepoint_prob": 0.6,
            "bocpd_runlength": 10,
            "jump_detected": True,
            "jump_magnitude": -0.05,
            "structural_features": {"key": 1.0},
            "ensemble_model_type": "ensemble",
        }
        sv = ShockVector.from_dict(d)

        assert sv.ticker == "GOOGL"
        assert sv.hmm_regime == 3
        assert sv.timestamp == datetime(2026, 2, 25, 12, 0, 0)
        assert sv.jump_detected is True

    def test_round_trip(self):
        from quant_engine.regime.shock_vector import ShockVector

        original = ShockVector(
            ticker="AAPL",
            hmm_regime=2,
            hmm_confidence=0.88,
            hmm_uncertainty=0.1,
            bocpd_changepoint_prob=0.45,
            bocpd_runlength=30,
            jump_detected=False,
            jump_magnitude=0.01,
            structural_features={"a": 1.0, "b": 2.5},
            ensemble_model_type="jump",
        )

        d = original.to_dict()
        reconstructed = ShockVector.from_dict(d)

        assert reconstructed.ticker == original.ticker
        assert reconstructed.hmm_regime == original.hmm_regime
        assert abs(reconstructed.hmm_confidence - original.hmm_confidence) < 1e-10
        assert abs(reconstructed.bocpd_changepoint_prob - original.bocpd_changepoint_prob) < 1e-10
        assert reconstructed.structural_features == original.structural_features


# ── Shock Event Detection Tests ──────────────────────────────────────────


class TestShockEventDetection:
    """Test ShockVector.is_shock_event()."""

    def test_no_shock(self):
        from quant_engine.regime.shock_vector import ShockVector

        sv = ShockVector(
            bocpd_changepoint_prob=0.1,
            jump_detected=False,
            jump_magnitude=0.005,
        )
        assert not sv.is_shock_event()

    def test_shock_from_jump(self):
        from quant_engine.regime.shock_vector import ShockVector

        sv = ShockVector(
            bocpd_changepoint_prob=0.1,
            jump_detected=True,
            jump_magnitude=0.01,
        )
        assert sv.is_shock_event()

    def test_shock_from_changepoint(self):
        from quant_engine.regime.shock_vector import ShockVector

        sv = ShockVector(
            bocpd_changepoint_prob=0.8,
            jump_detected=False,
            jump_magnitude=0.001,
        )
        assert sv.is_shock_event(changepoint_threshold=0.5)

    def test_shock_from_large_move(self):
        from quant_engine.regime.shock_vector import ShockVector

        sv = ShockVector(
            bocpd_changepoint_prob=0.1,
            jump_detected=False,
            jump_magnitude=-0.05,  # 5% drop
        )
        assert sv.is_shock_event()

    def test_custom_threshold(self):
        from quant_engine.regime.shock_vector import ShockVector

        sv = ShockVector(
            bocpd_changepoint_prob=0.6,
            jump_detected=False,
            jump_magnitude=0.01,
        )
        assert not sv.is_shock_event(changepoint_threshold=0.7)
        assert sv.is_shock_event(changepoint_threshold=0.5)


# ── Regime Name Tests ────────────────────────────────────────────────────


class TestRegimeName:
    """Test ShockVector.regime_name()."""

    def test_all_regime_names(self):
        from quant_engine.regime.shock_vector import ShockVector

        expected = {
            0: "trending_bull",
            1: "trending_bear",
            2: "mean_reverting",
            3: "high_volatility",
        }
        for regime, name in expected.items():
            sv = ShockVector(hmm_regime=regime)
            assert sv.regime_name() == name
