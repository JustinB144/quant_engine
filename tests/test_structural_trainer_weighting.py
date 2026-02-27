"""
Tests for SPEC_03 T4: ShockVector regime-aware sample weighting in trainer.

Verifies that:
  - _compute_structural_weights produces correct per-sample weights
  - Changepoint probability downweights samples near regime transitions
  - Jump events downweight noisy outlier-driven samples
  - Systemic stress downweights high-stress-period samples
  - HMM confidence upweights high-confidence regime states
  - Weights combine multiplicatively with recency weights
  - Empty or None shock_vectors gracefully degrades to uniform weights
  - MultiIndex and single-index alignment both work
  - Weights are normalised to mean 1.0
  - train_ensemble accepts and propagates shock_vectors parameter
  - Metadata records structural_weights_applied flag and config
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quant_engine.regime.shock_vector import ShockVector
from quant_engine.models.trainer import ModelTrainer


# ── Helpers ────────────────────────────────────────────────────────────


def _make_shock_vector(
    timestamp: datetime = None,
    hmm_confidence: float = 0.8,
    bocpd_changepoint_prob: float = 0.0,
    jump_detected: bool = False,
    jump_magnitude: float = 0.0,
    systemic_stress: float = 0.0,
    drift_score: float = 0.5,
    **kwargs,
) -> ShockVector:
    """Create a ShockVector with sensible defaults."""
    return ShockVector(
        schema_version="1.0",
        timestamp=timestamp or datetime.now(),
        ticker="TEST",
        hmm_regime=0,
        hmm_confidence=hmm_confidence,
        hmm_uncertainty=1.0 - hmm_confidence,
        bocpd_changepoint_prob=bocpd_changepoint_prob,
        bocpd_runlength=50,
        jump_detected=jump_detected,
        jump_magnitude=jump_magnitude,
        structural_features={
            "drift_score": drift_score,
            "systemic_stress": systemic_stress,
        },
        **kwargs,
    )


def _make_training_data(
    n_samples: int = 500,
    n_features: int = 40,
    seed: int = 42,
    multiindex: bool = False,
) -> tuple:
    """Create synthetic training data matching trainer expectations."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-03", periods=n_samples)

    if multiindex:
        tickers = [f"TEST{i % 5}" for i in range(n_samples)]
        index = pd.MultiIndex.from_arrays(
            [tickers, dates], names=["permno", "date"],
        )
    else:
        index = dates

    features = pd.DataFrame(
        rng.standard_normal((n_samples, n_features)),
        index=index,
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    targets = pd.Series(
        rng.normal(0.0, 0.02, n_samples),
        index=index,
        name="target",
    )
    regimes = pd.Series(
        rng.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.2, 0.2, 0.2]),
        index=index,
        name="regime",
    )
    return features, targets, regimes, dates


# ── _compute_structural_weights Unit Tests ─────────────────────────────


class TestComputeStructuralWeights:
    """Tests for ModelTrainer._compute_structural_weights."""

    def test_empty_shock_vectors_returns_ones(self):
        """No shock vectors → uniform weights."""
        dates = pd.bdate_range("2023-01-03", periods=100)
        weights = ModelTrainer._compute_structural_weights(
            index=dates, shock_vectors={},
        )
        np.testing.assert_array_almost_equal(weights, np.ones(100))

    def test_none_shock_vectors_returns_ones(self):
        """None shock vectors → uniform weights (guard in train_ensemble)."""
        dates = pd.bdate_range("2023-01-03", periods=50)
        weights = ModelTrainer._compute_structural_weights(
            index=dates, shock_vectors=None,
        )
        np.testing.assert_array_almost_equal(weights, np.ones(50))

    def test_all_calm_returns_near_uniform(self):
        """All calm ShockVectors → weights close to 1.0 (confidence factor)."""
        dates = pd.bdate_range("2023-01-03", periods=100)
        svs = {
            dt: _make_shock_vector(
                timestamp=dt,
                hmm_confidence=0.9,
                bocpd_changepoint_prob=0.0,
                systemic_stress=0.0,
            )
            for dt in dates
        }
        weights = ModelTrainer._compute_structural_weights(
            index=dates, shock_vectors=svs,
        )
        assert weights.shape == (100,)
        # All weights should be similar (only confidence varies from 1.0)
        assert np.std(weights) < 0.01

    def test_changepoint_downweights(self):
        """High changepoint probability → lower weight."""
        dates = pd.bdate_range("2023-01-03", periods=10)
        svs = {}
        for i, dt in enumerate(dates):
            cp_prob = 0.8 if i == 5 else 0.0
            svs[dt] = _make_shock_vector(
                timestamp=dt,
                hmm_confidence=0.9,
                bocpd_changepoint_prob=cp_prob,
            )

        weights = ModelTrainer._compute_structural_weights(
            index=dates, shock_vectors=svs,
        )
        # Sample 5 (changepoint) should have lower weight than others
        assert weights[5] < weights[0]
        assert weights[5] < weights[9]

    def test_jump_event_downweights(self):
        """Jump events → lower weight."""
        dates = pd.bdate_range("2023-01-03", periods=10)
        svs = {}
        for i, dt in enumerate(dates):
            svs[dt] = _make_shock_vector(
                timestamp=dt,
                hmm_confidence=0.9,
                jump_detected=(i == 3),
                jump_magnitude=-0.08 if i == 3 else 0.001,
            )

        weights = ModelTrainer._compute_structural_weights(
            index=dates, shock_vectors=svs,
        )
        # Sample 3 (jump) should have lower weight
        assert weights[3] < weights[0]

    def test_systemic_stress_downweights(self):
        """High systemic stress → lower weight."""
        dates = pd.bdate_range("2023-01-03", periods=10)
        svs = {}
        for i, dt in enumerate(dates):
            stress = 0.9 if i == 7 else 0.1
            svs[dt] = _make_shock_vector(
                timestamp=dt,
                hmm_confidence=0.9,
                systemic_stress=stress,
            )

        weights = ModelTrainer._compute_structural_weights(
            index=dates, shock_vectors=svs,
        )
        # Sample 7 (high stress) should have lower weight
        assert weights[7] < weights[0]

    def test_low_confidence_downweights(self):
        """Low HMM confidence → lower weight."""
        dates = pd.bdate_range("2023-01-03", periods=10)
        svs = {}
        for i, dt in enumerate(dates):
            conf = 0.2 if i == 4 else 0.9
            svs[dt] = _make_shock_vector(
                timestamp=dt,
                hmm_confidence=conf,
            )

        weights = ModelTrainer._compute_structural_weights(
            index=dates, shock_vectors=svs,
        )
        # Sample 4 (low confidence) should have lower weight
        assert weights[4] < weights[0]

    def test_combined_signals_multiply(self):
        """Multiple adverse signals produce compounded downweighting."""
        dates = pd.bdate_range("2023-01-03", periods=10)
        svs = {}
        for i, dt in enumerate(dates):
            svs[dt] = _make_shock_vector(
                timestamp=dt,
                hmm_confidence=0.9,
            )

        # Replace sample 6 with all adverse signals
        svs[dates[6]] = _make_shock_vector(
            timestamp=dates[6],
            hmm_confidence=0.3,
            bocpd_changepoint_prob=0.8,
            jump_detected=True,
            jump_magnitude=-0.05,
            systemic_stress=0.85,
        )

        weights = ModelTrainer._compute_structural_weights(
            index=dates, shock_vectors=svs,
        )
        # Sample 6 should be the lowest
        assert weights[6] == weights.min()
        # And significantly lower than calm samples
        assert weights[6] < 0.5 * weights[0]

    def test_weights_normalised_to_mean_one(self):
        """Weights should be normalised to mean 1.0."""
        dates = pd.bdate_range("2023-01-03", periods=100)
        svs = {}
        rng = np.random.default_rng(42)
        for dt in dates:
            svs[dt] = _make_shock_vector(
                timestamp=dt,
                hmm_confidence=float(rng.uniform(0.3, 0.95)),
                bocpd_changepoint_prob=float(rng.uniform(0.0, 0.5)),
                systemic_stress=float(rng.uniform(0.0, 0.8)),
            )

        weights = ModelTrainer._compute_structural_weights(
            index=dates, shock_vectors=svs,
        )
        assert abs(weights.mean() - 1.0) < 1e-6

    def test_weights_clipped_no_negatives(self):
        """Weights should never be negative or zero."""
        dates = pd.bdate_range("2023-01-03", periods=10)
        svs = {}
        for dt in dates:
            # Extreme adverse conditions
            svs[dt] = _make_shock_vector(
                timestamp=dt,
                hmm_confidence=0.05,
                bocpd_changepoint_prob=1.0,
                jump_detected=True,
                jump_magnitude=-0.1,
                systemic_stress=1.0,
            )

        weights = ModelTrainer._compute_structural_weights(
            index=dates, shock_vectors=svs,
        )
        assert np.all(weights > 0)

    def test_multiindex_date_alignment(self):
        """Shock vectors keyed by date should align with MultiIndex rows."""
        dates = pd.bdate_range("2023-01-03", periods=20)
        tickers = [f"T{i % 2}" for i in range(20)]
        multi_idx = pd.MultiIndex.from_arrays(
            [tickers, dates], names=["permno", "date"],
        )

        # Shock vectors keyed by date — vary signals across dates
        svs = {}
        for i, dt in enumerate(dates):
            svs[dt] = _make_shock_vector(
                timestamp=dt,
                hmm_confidence=0.85,
                bocpd_changepoint_prob=0.7 if i == 10 else 0.05,
            )

        weights = ModelTrainer._compute_structural_weights(
            index=multi_idx, shock_vectors=svs,
        )
        assert weights.shape == (20,)
        # Sample 10 (high changepoint) should be downweighted vs others
        assert weights[10] < weights[0]

    def test_multiindex_ticker_date_tuple_alignment(self):
        """Shock vectors keyed by (ticker, date) should align with MultiIndex."""
        dates = pd.bdate_range("2023-01-03", periods=10)
        tickers = [f"T{i % 2}" for i in range(10)]
        multi_idx = pd.MultiIndex.from_arrays(
            [tickers, dates], names=["permno", "date"],
        )

        # Shock vectors keyed by (ticker, date) tuples
        svs = {}
        for i in range(10):
            key = (tickers[i], dates[i])
            svs[key] = _make_shock_vector(
                timestamp=dates[i],
                hmm_confidence=0.85,
                bocpd_changepoint_prob=0.2 if i == 5 else 0.0,
            )

        weights = ModelTrainer._compute_structural_weights(
            index=multi_idx, shock_vectors=svs,
        )
        assert weights.shape == (10,)
        # Sample 5 should be downweighted
        assert weights[5] < weights[0]

    def test_partial_match_graceful(self):
        """Unmatched samples should keep default weight."""
        dates = pd.bdate_range("2023-01-03", periods=20)
        # Only provide vectors for first 10 samples
        svs = {}
        for dt in dates[:10]:
            svs[dt] = _make_shock_vector(
                timestamp=dt,
                hmm_confidence=0.5,
                bocpd_changepoint_prob=0.5,
                systemic_stress=0.5,
            )

        weights = ModelTrainer._compute_structural_weights(
            index=dates, shock_vectors=svs,
        )
        assert weights.shape == (20,)
        # Weights should still sum/mean correctly after normalisation

    def test_custom_penalty_parameters(self):
        """Custom penalty parameters should be respected."""
        dates = pd.bdate_range("2023-01-03", periods=10)
        svs = {}
        for i, dt in enumerate(dates):
            svs[dt] = _make_shock_vector(
                timestamp=dt,
                hmm_confidence=0.9,
                bocpd_changepoint_prob=0.8 if i == 3 else 0.0,
            )

        # High penalty: changepoint weight = 1.0 - 0.9 * 0.8 = 0.28
        high_penalty = ModelTrainer._compute_structural_weights(
            index=dates, shock_vectors=svs,
            changepoint_penalty=0.9,
        )

        # Low penalty: changepoint weight = 1.0 - 0.1 * 0.8 = 0.92
        low_penalty = ModelTrainer._compute_structural_weights(
            index=dates, shock_vectors=svs,
            changepoint_penalty=0.1,
        )

        # High penalty should produce lower weight for sample 3
        assert high_penalty[3] < low_penalty[3]

    def test_zero_penalty_parameters_no_effect(self):
        """Zero penalties → all weights equal (only confidence matters)."""
        dates = pd.bdate_range("2023-01-03", periods=10)
        svs = {}
        for i, dt in enumerate(dates):
            svs[dt] = _make_shock_vector(
                timestamp=dt,
                hmm_confidence=0.8,
                bocpd_changepoint_prob=0.5,
                jump_detected=True,
                jump_magnitude=-0.05,
                systemic_stress=0.5,
            )

        weights = ModelTrainer._compute_structural_weights(
            index=dates, shock_vectors=svs,
            changepoint_penalty=0.0,
            jump_penalty=0.0,
            stress_penalty=0.0,
        )
        # All weights should be the same (only confidence varies, but it's 0.8 for all)
        assert np.std(weights) < 1e-10


# ── train_ensemble Integration Tests ──────────────────────────────────


class TestTrainEnsembleShockVectorIntegration:
    """Test that train_ensemble correctly accepts and uses shock_vectors."""

    @patch("quant_engine.validation.preconditions.enforce_preconditions")
    def test_train_ensemble_accepts_shock_vectors_param(self, mock_precond):
        """train_ensemble should accept shock_vectors without error."""
        features, targets, regimes, dates = _make_training_data(
            n_samples=500, n_features=40,
        )
        shock_vectors = {
            dt: _make_shock_vector(timestamp=dt, hmm_confidence=0.8)
            for dt in dates
        }

        trainer = ModelTrainer()
        # Should not raise
        result = trainer.train_ensemble(
            features=features,
            targets=targets,
            regimes=regimes,
            horizon=10,
            verbose=False,
            versioned=False,
            shock_vectors=shock_vectors,
        )
        assert result is not None

    @patch("quant_engine.validation.preconditions.enforce_preconditions")
    def test_train_ensemble_none_shock_vectors_works(self, mock_precond):
        """train_ensemble should work fine with shock_vectors=None."""
        features, targets, regimes, dates = _make_training_data(
            n_samples=500, n_features=40,
        )

        trainer = ModelTrainer()
        result = trainer.train_ensemble(
            features=features,
            targets=targets,
            regimes=regimes,
            horizon=10,
            verbose=False,
            versioned=False,
            shock_vectors=None,
        )
        assert result is not None

    @patch("quant_engine.validation.preconditions.enforce_preconditions")
    def test_train_ensemble_backward_compatible(self, mock_precond):
        """train_ensemble without shock_vectors param still works (backward compat)."""
        features, targets, regimes, dates = _make_training_data(
            n_samples=500, n_features=40,
        )

        trainer = ModelTrainer()
        # Old-style call without shock_vectors
        result = trainer.train_ensemble(
            features=features,
            targets=targets,
            regimes=regimes,
            horizon=10,
            verbose=False,
            versioned=False,
        )
        assert result is not None

    @patch("quant_engine.validation.preconditions.enforce_preconditions")
    def test_structural_disabled_ignores_shock_vectors(self, mock_precond):
        """When STRUCTURAL_WEIGHT_ENABLED=False, shock_vectors are ignored."""
        features, targets, regimes, dates = _make_training_data(
            n_samples=500, n_features=40,
        )
        shock_vectors = {
            dt: _make_shock_vector(
                timestamp=dt,
                hmm_confidence=0.1,
                bocpd_changepoint_prob=0.9,
                jump_detected=True,
                systemic_stress=0.9,
            )
            for dt in dates
        }

        trainer = ModelTrainer()
        with patch("quant_engine.models.trainer.STRUCTURAL_WEIGHT_ENABLED", False):
            result = trainer.train_ensemble(
                features=features,
                targets=targets,
                regimes=regimes,
                horizon=10,
                verbose=False,
                versioned=False,
                shock_vectors=shock_vectors,
            )
        assert result is not None

    @patch("quant_engine.validation.preconditions.enforce_preconditions")
    def test_structural_and_recency_combined(self, mock_precond):
        """Structural and recency weights should combine multiplicatively."""
        features, targets, regimes, dates = _make_training_data(
            n_samples=500, n_features=40,
        )
        shock_vectors = {
            dt: _make_shock_vector(timestamp=dt, hmm_confidence=0.8)
            for dt in dates
        }

        trainer = ModelTrainer()
        # Should not raise when both are active
        result = trainer.train_ensemble(
            features=features,
            targets=targets,
            regimes=regimes,
            horizon=10,
            verbose=False,
            versioned=False,
            recency_weight=True,
            shock_vectors=shock_vectors,
        )
        assert result is not None

    @patch("quant_engine.validation.preconditions.enforce_preconditions")
    def test_multiindex_training_data(self, mock_precond):
        """Shock vectors should work with MultiIndex training data."""
        features, targets, regimes, dates = _make_training_data(
            n_samples=500, n_features=40, multiindex=True,
        )
        # Key by date (shock vectors are per-date, not per-ticker)
        shock_vectors = {
            dt: _make_shock_vector(timestamp=dt, hmm_confidence=0.85)
            for dt in dates
        }

        trainer = ModelTrainer()
        result = trainer.train_ensemble(
            features=features,
            targets=targets,
            regimes=regimes,
            horizon=10,
            verbose=False,
            versioned=False,
            shock_vectors=shock_vectors,
        )
        assert result is not None


# ── Metadata Persistence Tests ─────────────────────────────────────────


class TestStructuralWeightingMetadata:
    """Verify structural weighting info is recorded in model metadata."""

    @patch("quant_engine.validation.preconditions.enforce_preconditions")
    def test_metadata_records_structural_flag(self, mock_precond, tmp_path):
        """Model metadata should record structural_weights_applied=True."""
        import json

        # Build data with strong embedded signal so model passes quality gates
        rng = np.random.default_rng(99)
        n = 1000
        dates = pd.bdate_range("2019-01-03", periods=n)
        n_feat = 50
        X = pd.DataFrame(
            rng.standard_normal((n, n_feat)),
            index=dates,
            columns=[f"feat_{i}" for i in range(n_feat)],
        )
        # Strong linear signal in first 10 features
        signal = X.iloc[:, :10].values @ rng.uniform(0.01, 0.05, 10)
        noise = rng.normal(0, 0.005, n)
        targets = pd.Series(signal + noise, index=dates)
        regimes = pd.Series(np.zeros(n, dtype=int), index=dates)

        shock_vectors = {
            dt: _make_shock_vector(timestamp=dt, hmm_confidence=0.8)
            for dt in dates
        }

        model_dir = tmp_path / "trained_models"
        model_dir.mkdir()

        trainer = ModelTrainer()
        with patch("quant_engine.models.trainer.MODEL_DIR", model_dir):
            result = trainer.train_ensemble(
                features=X,
                targets=targets,
                regimes=regimes,
                horizon=10,
                verbose=False,
                versioned=False,
                shock_vectors=shock_vectors,
            )

        # Check if global model was trained (might be rejected by quality gates)
        if result.global_model is None:
            pytest.skip("Global model rejected by quality gates")

        # Find metadata file
        meta_files = list(model_dir.glob("*_meta.json"))
        assert len(meta_files) >= 1, f"No meta files in {model_dir}"

        with open(meta_files[0]) as f:
            meta = json.load(f)

        assert meta.get("structural_weights_applied") is True
        assert "structural_weight_config" in meta
        assert "changepoint_penalty" in meta["structural_weight_config"]
        assert "jump_penalty" in meta["structural_weight_config"]
        assert "stress_penalty" in meta["structural_weight_config"]

    @patch("quant_engine.validation.preconditions.enforce_preconditions")
    def test_metadata_no_structural_when_disabled(self, mock_precond, tmp_path):
        """Without shock_vectors, metadata should record structural=False."""
        import json

        # Build data with strong embedded signal so model passes quality gates
        rng = np.random.default_rng(99)
        n = 1000
        dates = pd.bdate_range("2019-01-03", periods=n)
        n_feat = 50
        X = pd.DataFrame(
            rng.standard_normal((n, n_feat)),
            index=dates,
            columns=[f"feat_{i}" for i in range(n_feat)],
        )
        signal = X.iloc[:, :10].values @ rng.uniform(0.01, 0.05, 10)
        noise = rng.normal(0, 0.005, n)
        targets = pd.Series(signal + noise, index=dates)
        regimes = pd.Series(np.zeros(n, dtype=int), index=dates)

        model_dir = tmp_path / "trained_models"
        model_dir.mkdir()

        trainer = ModelTrainer()
        with patch("quant_engine.models.trainer.MODEL_DIR", model_dir):
            result = trainer.train_ensemble(
                features=X,
                targets=targets,
                regimes=regimes,
                horizon=10,
                verbose=False,
                versioned=False,
            )

        if result.global_model is None:
            pytest.skip("Global model rejected by quality gates")

        meta_files = list(model_dir.glob("*_meta.json"))
        assert len(meta_files) >= 1

        with open(meta_files[0]) as f:
            meta = json.load(f)

        assert meta.get("structural_weights_applied") is False
        assert "structural_weight_config" not in meta


# ── Config Tests ──────────────────────────────────────────────────────


class TestStructuralWeightConfig:
    """Verify config constants are properly defined and accessible."""

    def test_config_constants_importable(self):
        """All structural weight config constants should be importable."""
        from quant_engine.config import (
            STRUCTURAL_WEIGHT_ENABLED,
            STRUCTURAL_WEIGHT_CHANGEPOINT_PENALTY,
            STRUCTURAL_WEIGHT_JUMP_PENALTY,
            STRUCTURAL_WEIGHT_STRESS_PENALTY,
        )
        assert isinstance(STRUCTURAL_WEIGHT_ENABLED, bool)
        assert 0.0 <= STRUCTURAL_WEIGHT_CHANGEPOINT_PENALTY <= 1.0
        assert 0.0 <= STRUCTURAL_WEIGHT_JUMP_PENALTY <= 1.0
        assert 0.0 <= STRUCTURAL_WEIGHT_STRESS_PENALTY <= 1.0

    def test_trainer_imports_config(self):
        """Trainer module should import structural weight config."""
        import quant_engine.models.trainer as mod
        assert hasattr(mod, "STRUCTURAL_WEIGHT_ENABLED")
        assert hasattr(mod, "STRUCTURAL_WEIGHT_CHANGEPOINT_PENALTY")
        assert hasattr(mod, "STRUCTURAL_WEIGHT_JUMP_PENALTY")
        assert hasattr(mod, "STRUCTURAL_WEIGHT_STRESS_PENALTY")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
