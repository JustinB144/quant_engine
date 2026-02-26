"""Tests for evaluation/ml_diagnostics.py — feature drift and ensemble disagreement."""

import numpy as np
import pytest

from quant_engine.evaluation.ml_diagnostics import (
    feature_importance_drift,
    ensemble_disagreement,
)


class TestFeatureImportanceDrift:
    """Test feature importance drift detection."""

    def test_stable_features_no_drift(self):
        """Identical importance across periods => no drift."""
        imp = np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.03, 0.01, 0.01])
        matrices = {
            "2023-01": imp,
            "2023-02": imp,
            "2023-03": imp,
        }
        result = feature_importance_drift(matrices)
        assert result["drift_detected"] is False
        assert result["mean_correlation"] > 0.99
        assert result["n_periods"] == 3

    def test_drifting_features_detected(self):
        """Reversed importance order => drift detected."""
        imp1 = np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.03, 0.01, 0.01])
        imp2 = imp1[::-1]  # Completely reversed
        matrices = {
            "2023-01": imp1,
            "2023-02": imp2,
        }
        result = feature_importance_drift(matrices)
        assert result["drift_detected"] is True
        assert result["min_correlation"] < 0.7

    def test_single_period_no_drift(self):
        """Only one period => no drift can be computed."""
        matrices = {"2023-01": np.ones(10)}
        result = feature_importance_drift(matrices)
        assert result["drift_detected"] is False
        assert result["n_periods"] == 1

    def test_top_k_stability(self):
        """When top features don't change, stability should be high."""
        rng = np.random.RandomState(42)
        base = np.array([0.3, 0.25, 0.2, 0.15, 0.05, 0.03, 0.01, 0.005, 0.003, 0.002])
        matrices = {}
        for i in range(5):
            noise = rng.normal(0, 0.01, len(base))
            matrices[f"2023-{i+1:02d}"] = np.abs(base + noise)

        result = feature_importance_drift(matrices, top_k=3)
        assert result["top_k_stability"] > 0.5

    def test_feature_names_tracked(self):
        imp1 = np.array([0.3, 0.2, 0.1, 0.05, 0.01])
        imp2 = np.array([0.3, 0.19, 0.11, 0.04, 0.01])
        names = ["momentum", "volatility", "trend", "volume", "skew"]

        result = feature_importance_drift(
            {"p1": imp1, "p2": imp2},
            feature_names=names,
        )
        for period, features in result["top_k_features_per_period"].items():
            assert all(isinstance(f, str) for f in features)


class TestEnsembleDisagreement:
    """Test ensemble disagreement measurement."""

    def test_identical_predictions_no_disagreement(self):
        """Same predictions => correlation = 1.0."""
        rng = np.random.RandomState(42)
        preds = rng.normal(0, 1, 200)
        result = ensemble_disagreement({
            "model_a": preds,
            "model_b": preds,
        })
        assert result["mean_correlation"] > 0.99
        assert result["high_disagreement"] is False

    def test_random_predictions_low_correlation(self):
        """Independent random predictions => low correlation."""
        rng = np.random.RandomState(42)
        result = ensemble_disagreement({
            "model_a": rng.normal(0, 1, 500),
            "model_b": rng.normal(0, 1, 500),
            "model_c": rng.normal(0, 1, 500),
        })
        assert result["mean_correlation"] < 0.3
        assert result["high_disagreement"] is True
        assert result["n_models"] == 3

    def test_single_model_no_disagreement(self):
        result = ensemble_disagreement({"only_model": np.ones(100)})
        assert result["high_disagreement"] is False
        assert result["n_models"] == 1

    def test_pairwise_correlations_stored(self):
        rng = np.random.RandomState(42)
        result = ensemble_disagreement({
            "a": rng.normal(0, 1, 100),
            "b": rng.normal(0, 1, 100),
        })
        assert "a_vs_b" in result["pairwise_correlations"]
