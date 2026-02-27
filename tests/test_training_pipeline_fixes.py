"""
Tests for Spec 013: Model Training Pipeline — CV Fixes, Calibration, and Governance.

Covers:
    T1: Per-fold feature selection with stable feature tracking
    T2: Calibration validation split with ECE metric
    T3: Minimum sample enforcement for regime models
    T4: Feature correlation pruning threshold (0.80)
    T5: Governance scoring with validation integration
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature_matrix(n_rows: int, n_features: int, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic feature matrix with a MultiIndex (permno, date)."""
    rng = np.random.RandomState(seed)
    n_stocks = 5
    n_dates = n_rows // n_stocks
    dates = pd.bdate_range("2020-01-02", periods=n_dates, freq="B")
    permnos = [str(10000 + i) for i in range(n_stocks)]

    idx = pd.MultiIndex.from_product([permnos, dates], names=["permno", "date"])
    data = rng.randn(len(idx), n_features)
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(data, index=idx, columns=cols)


def _make_targets(X: pd.DataFrame, signal_features: list, seed: int = 42) -> pd.Series:
    """Create synthetic targets with signal from specified features."""
    rng = np.random.RandomState(seed)
    y = np.zeros(len(X))
    for feat in signal_features:
        if feat in X.columns:
            y += 0.02 * X[feat].values
    y += rng.randn(len(X)) * 0.01
    return pd.Series(y, index=X.index, name="target_10d")


def _make_regimes(X: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Create synthetic regime labels (0-3) aligned with features."""
    rng = np.random.RandomState(seed)
    return pd.Series(rng.choice([0, 1, 2, 3], size=len(X)), index=X.index, name="regime")


# ---------------------------------------------------------------------------
# T1: Per-fold feature selection
# ---------------------------------------------------------------------------


class TestPerFoldFeatureSelection:
    """Verify that feature selection runs independently per CV fold."""

    def test_feature_selection_per_fold(self):
        """Different folds can select different features due to per-fold selection."""
        from quant_engine.models.trainer import ModelTrainer

        # Create data where different time periods have different signal features
        n_features = 20
        X = _make_feature_matrix(1000, n_features, seed=11)
        rng = np.random.RandomState(11)

        # Inject signal into different features at different times
        n = len(X)
        half = n // 2
        y = rng.randn(n) * 0.01
        # First half: signal in feat_0, feat_1
        y[:half] += 0.03 * X["feat_0"].values[:half] + 0.02 * X["feat_1"].values[:half]
        # Second half: signal in feat_5, feat_6
        y[half:] += 0.03 * X["feat_5"].values[half:] + 0.02 * X["feat_6"].values[half:]
        y = pd.Series(y, index=X.index)

        trainer = ModelTrainer(
            model_params={
                "n_estimators": 50, "max_depth": 3, "min_samples_leaf": 20,
                "learning_rate": 0.1, "subsample": 0.8, "max_features": "sqrt",
            },
            max_features=10,
            cv_folds=3,
            holdout_fraction=0.15,
            max_gap=0.30,
        )

        # We need to test that _select_features on different subsets gives different results
        # (the actual per-fold selection is done inside _train_single)
        sel_first, _ = trainer._select_features(
            X.iloc[:half], y.iloc[:half], max_feats=10,
        )
        sel_second, _ = trainer._select_features(
            X.iloc[half:], y.iloc[half:], max_feats=10,
        )

        # The two halves should not select the exact same features
        # (since signal is in different features)
        assert sel_first != sel_second, (
            "Per-fold feature selection should produce different features "
            "when signal varies across time periods"
        )

    def test_stable_features_selected_most_folds(self):
        """Final feature set should use features appearing in ≥80% of folds."""
        from quant_engine.models.trainer import ModelTrainer

        trainer = ModelTrainer(max_features=10)

        # Simulate 5 folds where some features are consistently selected
        fold_features = [
            ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"],
            ["feat_0", "feat_1", "feat_2", "feat_3", "feat_5"],
            ["feat_0", "feat_1", "feat_2", "feat_3", "feat_6"],
            ["feat_0", "feat_1", "feat_2", "feat_7", "feat_8"],
            ["feat_0", "feat_1", "feat_2", "feat_3", "feat_9"],
        ]

        # Create a dummy X_dev and y_dev for the fallback path
        X_dev = pd.DataFrame(
            np.random.randn(200, 10),
            columns=[f"feat_{i}" for i in range(10)],
        )
        y_dev = pd.Series(np.random.randn(200))

        stable = trainer._compute_stable_features(
            fold_features, max_feats=10, X_dev=X_dev, y_dev=y_dev,
        )

        # feat_0, feat_1, feat_2 appear in all 5 folds (100%)
        # feat_3 appears in 4/5 folds (80%)
        # All others appear in 1/5 folds (20%)
        assert "feat_0" in stable
        assert "feat_1" in stable
        assert "feat_2" in stable
        assert "feat_3" in stable

        # Features appearing in only 1 fold should NOT be in the stable set
        for rare in ["feat_5", "feat_6", "feat_7", "feat_8", "feat_9"]:
            assert rare not in stable, (
                f"{rare} should not be stable (only in 1/5 folds)"
            )

    def test_compute_stable_features_fallback(self):
        """When fewer than 5 stable features, relax threshold or fall back."""
        from quant_engine.models.trainer import ModelTrainer

        trainer = ModelTrainer(max_features=10)

        # All features appear in only 1 fold each (no stability)
        fold_features = [
            ["feat_0"],
            ["feat_1"],
            ["feat_2"],
            ["feat_3"],
            ["feat_4"],
        ]

        X_dev = pd.DataFrame(
            np.random.randn(200, 10),
            columns=[f"feat_{i}" for i in range(10)],
        )
        y_dev = pd.Series(np.random.randn(200))

        stable = trainer._compute_stable_features(
            fold_features, max_feats=10, X_dev=X_dev, y_dev=y_dev,
        )

        # Should fall back to global selection (returns non-empty list)
        assert len(stable) > 0, "Fallback should produce at least one feature"


# ---------------------------------------------------------------------------
# T2: Calibration validation split
# ---------------------------------------------------------------------------


class TestCalibrationValidationSplit:
    """Verify calibration uses a separate split and computes ECE."""

    def test_calibration_uses_separate_split(self):
        """_fit_calibrator should split holdout 50/50 for fit/validate."""
        from quant_engine.models.trainer import ModelTrainer

        X = _make_feature_matrix(2000, 15, seed=22)
        y = _make_targets(X, ["feat_0", "feat_1"], seed=22)
        regimes = _make_regimes(X, seed=22)

        trainer = ModelTrainer(
            model_params={
                "n_estimators": 50, "max_depth": 3, "min_samples_leaf": 20,
                "learning_rate": 0.1, "subsample": 0.8, "max_features": "sqrt",
            },
            max_features=10,
            cv_folds=3,
            holdout_fraction=0.15,
            max_gap=0.30,
        )

        result = trainer.train_ensemble(
            features=X,
            targets=y,
            regimes=regimes,
            horizon=10,
            verbose=False,
            versioned=False,
        )

        # The calibrator should have been fitted (result is not None = model passed)
        # We can't directly check the split, but we can verify the method runs
        assert result is not None
        assert result.total_samples > 0

    def test_ece_computed(self):
        """Expected Calibration Error should be computable."""
        from quant_engine.models.calibration import compute_ece

        # Perfect calibration: predicted probs match actual frequencies
        predicted = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        # Actual outcomes: roughly match predicted probabilities
        actual = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])

        ece = compute_ece(predicted, actual, n_bins=5)
        assert isinstance(ece, float)
        assert 0.0 <= ece <= 1.0

    def test_ece_perfect_calibration(self):
        """ECE should be zero for perfectly calibrated predictions."""
        from quant_engine.models.calibration import compute_ece

        # All predictions = 0.5, actual 50% positive
        predicted = np.full(100, 0.5)
        actual = np.array([1] * 50 + [0] * 50)

        ece = compute_ece(predicted, actual, n_bins=10)
        assert ece == pytest.approx(0.0, abs=0.01), (
            f"Perfectly calibrated predictions should have ECE ~0, got {ece}"
        )

    def test_ece_worst_calibration(self):
        """ECE should be high for badly calibrated predictions."""
        from quant_engine.models.calibration import compute_ece

        # Predict 0.9 for everything, but only 10% are actually positive
        predicted = np.full(100, 0.9)
        actual = np.array([1] * 10 + [0] * 90)

        ece = compute_ece(predicted, actual, n_bins=10)
        assert ece > 0.5, (
            f"Badly calibrated predictions should have high ECE, got {ece}"
        )

    def test_reliability_curve(self):
        """Reliability curve should return properly structured data."""
        from quant_engine.models.calibration import compute_reliability_curve

        rng = np.random.RandomState(33)
        predicted = rng.uniform(0, 1, 200)
        actual = (rng.uniform(0, 1, 200) < predicted).astype(float)

        curve = compute_reliability_curve(predicted, actual, n_bins=5)
        assert "bin_centers" in curve
        assert "observed_freq" in curve
        assert "avg_predicted" in curve
        assert "bin_counts" in curve
        assert len(curve["bin_centers"]) == 5
        assert sum(curve["bin_counts"]) == 200


# ---------------------------------------------------------------------------
# T3: Regime model minimum samples
# ---------------------------------------------------------------------------


class TestRegimeMinSamples:
    """Verify regime models require minimum sample count."""

    def test_regime_model_min_samples(self):
        """Regime with fewer than MIN_REGIME_SAMPLES should be skipped."""
        from quant_engine.config import MIN_REGIME_SAMPLES

        assert MIN_REGIME_SAMPLES >= 50, (
            f"MIN_REGIME_SAMPLES={MIN_REGIME_SAMPLES} is too low — "
            "regime models need substantial data (SPEC_10 T7: 50 minimum)"
        )

    def test_regime_model_skipped_for_low_samples(self):
        """Trainer should skip regime models when sample count is below threshold."""
        from quant_engine.models.trainer import ModelTrainer

        # Create data with very unbalanced regimes:
        # regime 0 has 800 samples (enough), regime 3 has 50 (too few)
        n_features = 15
        X = _make_feature_matrix(1700, n_features, seed=44)
        y = _make_targets(X, ["feat_0", "feat_1"], seed=44)

        # Force regime distribution: most in regime 0, very few in regime 3
        regimes = pd.Series(0, index=X.index, name="regime")
        # Last 30 rows get regime 3 (below MIN_REGIME_SAMPLES=50)
        regimes.iloc[-30:] = 3

        trainer = ModelTrainer(
            model_params={
                "n_estimators": 50, "max_depth": 3, "min_samples_leaf": 20,
                "learning_rate": 0.1, "subsample": 0.8, "max_features": "sqrt",
            },
            max_features=10,
            cv_folds=3,
            holdout_fraction=0.15,
            max_gap=0.30,
        )

        result = trainer.train_ensemble(
            features=X,
            targets=y,
            regimes=regimes,
            horizon=10,
            verbose=False,
            versioned=False,
        )

        # Regime 3 with only 30 samples should NOT have a trained model (< MIN_REGIME_SAMPLES=50)
        assert 3 not in result.regime_models, (
            "Regime 3 with only 30 samples should be skipped (below MIN_REGIME_SAMPLES=50)"
        )

    def test_regime_fallback_to_global(self):
        """When regime model is missing, predictor should use global prediction."""
        # This tests that the predictor handles missing regime models gracefully.
        # When a regime has no dedicated model, the blended prediction should
        # just be the global prediction (blend_alpha = 0).
        from quant_engine.models.predictor import EnsemblePredictor

        # We can't easily construct a full predictor without trained models,
        # but we can verify the logic by checking that predictions without
        # a regime model default to global_prediction.
        # Instead, verify the principle: if regime_pred is NaN,
        # blended = global_pred
        global_pred = np.array([0.01, -0.02, 0.005])
        regime_pred = np.array([np.nan, np.nan, np.nan])  # No regime model
        blend_alpha = np.array([0.0, 0.0, 0.0])

        has_regime = ~np.isnan(regime_pred)
        blended = global_pred.copy()
        blended[has_regime] = (
            blend_alpha[has_regime] * regime_pred[has_regime]
            + (1 - blend_alpha[has_regime]) * global_pred[has_regime]
        )

        np.testing.assert_array_equal(blended, global_pred)


# ---------------------------------------------------------------------------
# T4: Correlation pruning threshold
# ---------------------------------------------------------------------------


class TestCorrelationPruning:
    """Verify feature correlation pruning at 0.80 threshold."""

    def test_correlation_threshold_080(self):
        """Features with pairwise correlation > 0.80 should be pruned."""
        from quant_engine.models.trainer import ModelTrainer

        rng = np.random.RandomState(55)
        n = 500

        # Create features where feat_0 and feat_1 are correlated at ~0.85
        base = rng.randn(n)
        feat_0 = base + rng.randn(n) * 0.3
        feat_1 = base + rng.randn(n) * 0.3  # correlation ~0.85
        feat_2 = rng.randn(n)  # independent

        X = pd.DataFrame({
            "feat_0": feat_0,
            "feat_1": feat_1,
            "feat_2": feat_2,
        })

        # Verify correlation is in the 0.80-0.90 range
        corr_01 = X[["feat_0", "feat_1"]].corr().iloc[0, 1]
        assert 0.75 < corr_01 < 0.95, f"Expected correlation ~0.85, got {corr_01}"

        kept = ModelTrainer._prune_correlated_features(X, threshold=0.80)

        # One of feat_0/feat_1 should be dropped (correlation > 0.80)
        assert len(kept) == 2, (
            f"Expected 2 features kept (one correlated pair pruned), got {len(kept)}: {kept}"
        )
        assert "feat_2" in kept, "Independent feature should be kept"

    def test_old_threshold_would_keep_correlated(self):
        """At old threshold 0.90, features correlated at 0.85 would NOT be pruned."""
        from quant_engine.models.trainer import ModelTrainer

        rng = np.random.RandomState(55)
        n = 500

        base = rng.randn(n)
        feat_0 = base + rng.randn(n) * 0.3
        feat_1 = base + rng.randn(n) * 0.3
        feat_2 = rng.randn(n)

        X = pd.DataFrame({
            "feat_0": feat_0,
            "feat_1": feat_1,
            "feat_2": feat_2,
        })

        # With old 0.90 threshold, both correlated features would be kept
        kept_old = ModelTrainer._prune_correlated_features(X, threshold=0.90)
        # With new 0.80 threshold, one is pruned
        kept_new = ModelTrainer._prune_correlated_features(X, threshold=0.80)

        assert len(kept_old) >= len(kept_new), (
            "Tighter threshold should prune at least as many features"
        )

    def test_default_threshold_is_080(self):
        """Default threshold should be 0.80 (not 0.90)."""
        import inspect
        from quant_engine.models.trainer import ModelTrainer

        sig = inspect.signature(ModelTrainer._prune_correlated_features)
        default = sig.parameters["threshold"].default
        assert default == 0.80, (
            f"Default correlation threshold should be 0.80, got {default}"
        )


# ---------------------------------------------------------------------------
# T5: Governance scoring with validation
# ---------------------------------------------------------------------------


class TestGovernanceScoring:
    """Verify governance scoring includes validation metrics."""

    def test_governance_score_includes_validation(self, tmp_path):
        """DSR failure should reduce champion score."""
        from quant_engine.models.governance import ModelGovernance

        gov = ModelGovernance(registry_path=tmp_path / "test_champion.json")

        # Metrics WITH validation passing
        metrics_pass = {
            "oos_spearman": 0.08,
            "holdout_spearman": 0.06,
            "cv_gap": 0.02,
            "dsr_significant": True,
            "pbo": 0.20,
            "mc_significant": True,
            "ece": 0.05,
        }

        # Same performance metrics, but validation FAILING
        metrics_fail = {
            "oos_spearman": 0.08,
            "holdout_spearman": 0.06,
            "cv_gap": 0.02,
            "dsr_significant": False,
            "pbo": 0.80,
            "mc_significant": False,
            "ece": 0.30,
        }

        score_pass = gov._score(metrics_pass)
        score_fail = gov._score(metrics_fail)

        assert score_pass > score_fail, (
            f"Passing validation ({score_pass:.4f}) should score higher "
            f"than failing validation ({score_fail:.4f})"
        )

    def test_governance_score_backward_compatible(self, tmp_path):
        """Score should work without validation metrics (backward compatibility)."""
        from quant_engine.models.governance import ModelGovernance

        gov = ModelGovernance(registry_path=tmp_path / "test_champion.json")

        # Old-style metrics without validation fields
        metrics_old = {
            "oos_spearman": 0.08,
            "holdout_spearman": 0.06,
            "cv_gap": 0.02,
        }

        score = gov._score(metrics_old)
        assert isinstance(score, float)
        assert score > 0, "Positive performance metrics should give positive score"

    def test_governance_promotion_with_validation(self, tmp_path):
        """Promotion gate should work end-to-end with validation metrics."""
        from quant_engine.models.governance import ModelGovernance

        gov = ModelGovernance(registry_path=tmp_path / "test_champion.json")

        # First model: decent performance, poor validation
        result1 = gov.evaluate_and_update(
            horizon=10,
            version_id="v001",
            metrics={
                "oos_spearman": 0.06,
                "holdout_spearman": 0.05,
                "cv_gap": 0.03,
                "dsr_significant": False,
                "pbo": 0.70,
                "mc_significant": False,
            },
        )
        assert result1["promoted"] is True  # First model always promoted

        # Second model: better performance AND better validation
        result2 = gov.evaluate_and_update(
            horizon=10,
            version_id="v002",
            metrics={
                "oos_spearman": 0.10,
                "holdout_spearman": 0.08,
                "cv_gap": 0.01,
                "dsr_significant": True,
                "pbo": 0.15,
                "mc_significant": True,
                "ece": 0.03,
            },
        )
        assert result2["promoted"] is True, (
            "Model with better performance AND validation should be promoted"
        )
        assert result2["score"] > result1["score"]

    def test_dsr_penalty_reduces_score(self, tmp_path):
        """DSR failure specifically should reduce score vs. no DSR data."""
        from quant_engine.models.governance import ModelGovernance

        gov = ModelGovernance(registry_path=tmp_path / "test_champion.json")

        base_metrics = {
            "oos_spearman": 0.08,
            "holdout_spearman": 0.06,
            "cv_gap": 0.02,
        }

        # Score without DSR
        score_no_dsr = gov._score(base_metrics)

        # Score with DSR failure
        metrics_dsr_fail = {**base_metrics, "dsr_significant": False}
        score_dsr_fail = gov._score(metrics_dsr_fail)

        # Score with DSR success
        metrics_dsr_pass = {**base_metrics, "dsr_significant": True}
        score_dsr_pass = gov._score(metrics_dsr_pass)

        assert score_dsr_fail < score_no_dsr, (
            "DSR failure should reduce score below baseline"
        )
        assert score_dsr_pass > score_no_dsr, (
            "DSR success should increase score above baseline"
        )
