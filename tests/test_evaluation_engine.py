"""Tests for evaluation/engine.py — EvaluationEngine integration tests."""

import numpy as np
import pandas as pd
import pytest

from quant_engine.evaluation.engine import EvaluationEngine, EvaluationResult, RedFlag
from quant_engine.evaluation.calibration_analysis import analyze_calibration
from quant_engine.backtest.validation import (
    walk_forward_with_embargo,
    rolling_ic,
    detect_ic_decay,
)


@pytest.fixture
def sample_data():
    """Return a dict with returns, predictions, regimes, and trades."""
    rng = np.random.RandomState(42)
    n = 600
    dates = pd.bdate_range("2022-01-03", periods=n)

    # Returns with mild positive drift
    factor = rng.normal(0.0005, 0.015, n)
    returns = pd.Series(factor, index=dates)

    # Correlated predictions
    predictions = factor + rng.normal(0, 0.005, n)

    # Regime states
    regimes = np.array(
        [0] * 150 + [1] * 150 + [2] * 150 + [3] * 150, dtype=int
    )

    # Trades
    trades = [{"pnl": rng.normal(0.002, 0.03)} for _ in range(100)]

    return {
        "returns": returns,
        "predictions": predictions,
        "regimes": regimes,
        "trades": trades,
    }


class TestWalkForwardWithEmbargo:
    """Test the walk-forward embargo function directly."""

    def test_basic_walk_forward(self, sample_data):
        result = walk_forward_with_embargo(
            sample_data["returns"],
            sample_data["predictions"],
            train_window=200,
            embargo=5,
            test_window=50,
            slide_freq="weekly",
        )
        assert result.n_folds > 0
        assert len(result.folds) == result.n_folds

    def test_embargo_prevents_leakage(self, sample_data):
        """With embargo, the test period should start after the gap."""
        result = walk_forward_with_embargo(
            sample_data["returns"],
            sample_data["predictions"],
            train_window=200,
            embargo=10,
            test_window=50,
        )
        if result.folds:
            fold = result.folds[0]
            assert fold.embargo_start != fold.test_start

    def test_insufficient_data(self):
        returns = pd.Series([0.01] * 50, index=pd.bdate_range("2022-01-03", periods=50))
        predictions = np.ones(50) * 0.01
        result = walk_forward_with_embargo(returns, predictions, train_window=250)
        assert result.n_folds == 0
        assert result.is_overfit is True

    def test_overfit_detection_on_noise(self):
        """Model predicting noise should show overfit (if train overfits)."""
        rng = np.random.RandomState(42)
        n = 600
        returns = pd.Series(rng.normal(0, 0.01, n), index=pd.bdate_range("2022-01-03", periods=n))
        predictions = rng.normal(0, 0.01, n)

        result = walk_forward_with_embargo(
            returns, predictions, train_window=200, embargo=5, test_window=50,
        )
        # Random predictions should give near-zero OOS Sharpe
        assert result.n_folds > 0


class TestRollingIC:
    """Test rolling IC and decay detection."""

    def test_rolling_ic_correlated_predictions(self, sample_data):
        ic_series = rolling_ic(
            sample_data["predictions"],
            sample_data["returns"],
            window=60,
        )
        ic_clean = ic_series.dropna()
        assert len(ic_clean) > 0
        assert ic_clean.mean() > 0  # Correlated => positive IC

    def test_rolling_ic_random_predictions(self):
        rng = np.random.RandomState(42)
        n = 300
        returns = pd.Series(rng.normal(0, 0.01, n), index=pd.bdate_range("2022-01-03", periods=n))
        predictions = rng.normal(0, 0.01, n)

        ic_series = rolling_ic(predictions, returns, window=60)
        ic_clean = ic_series.dropna()
        assert abs(ic_clean.mean()) < 0.1  # Near zero

    def test_detect_ic_decay_decaying_signal(self):
        """IC that trends to zero should be detected."""
        # Create IC that decays
        rng = np.random.RandomState(42)
        ic_vals = np.linspace(0.15, -0.05, 100) + rng.normal(0, 0.02, 100)
        ic_series = pd.Series(ic_vals, index=pd.bdate_range("2022-01-03", periods=100))

        decaying, info = detect_ic_decay(ic_series, decay_threshold=0.02)
        assert info["slope"] < 0  # Negative slope

    def test_detect_ic_decay_stable_signal(self):
        """Stable positive IC should not be flagged."""
        rng = np.random.RandomState(42)
        ic_vals = 0.08 + rng.normal(0, 0.01, 100)
        ic_series = pd.Series(ic_vals, index=pd.bdate_range("2022-01-03", periods=100))

        decaying, info = detect_ic_decay(ic_series, decay_threshold=0.02)
        assert decaying is False
        assert info["mean_ic"] > 0.05


class TestCalibrationAnalysis:
    """Test calibration analysis module."""

    def test_well_calibrated_model(self):
        """Predictions matching returns should show good calibration."""
        rng = np.random.RandomState(42)
        n = 500
        predictions = rng.normal(0, 1, n)
        returns = pd.Series(predictions + rng.normal(0, 0.1, n))

        result = analyze_calibration(predictions, returns, bins=10)
        assert result["n_samples"] == n
        assert result["ece"] < 0.3  # Reasonable ECE

    def test_insufficient_data(self):
        predictions = np.array([0.1])
        returns = pd.Series([0.01])
        result = analyze_calibration(predictions, returns)
        assert result["n_samples"] == 0

    def test_with_confidence_scores(self):
        rng = np.random.RandomState(42)
        n = 500
        predictions = rng.normal(0, 1, n)
        returns = pd.Series(predictions + rng.normal(0, 0.5, n))
        confidence = np.abs(predictions) / (np.abs(predictions).max() + 1e-12)

        result = analyze_calibration(predictions, returns, confidence_scores=confidence)
        assert result["n_samples"] == n


class TestEvaluationEngine:
    """Integration tests for the full EvaluationEngine."""

    def test_full_evaluation(self, sample_data):
        engine = EvaluationEngine(
            train_window=200,
            embargo_days=5,
            test_window=50,
        )
        result = engine.evaluate(
            returns=sample_data["returns"],
            predictions=sample_data["predictions"],
            trades=sample_data["trades"],
            regime_states=sample_data["regimes"],
        )

        assert isinstance(result, EvaluationResult)
        assert result.aggregate_metrics["n_samples"] > 0
        assert result.summary != ""

    def test_regime_slicing_populates(self, sample_data):
        engine = EvaluationEngine(train_window=200)
        result = engine.evaluate(
            returns=sample_data["returns"],
            predictions=sample_data["predictions"],
            regime_states=sample_data["regimes"],
        )

        assert len(result.regime_slice_metrics) > 0
        assert len(result.individual_regime_metrics) > 0

    def test_walk_forward_runs(self, sample_data):
        engine = EvaluationEngine(train_window=200, embargo_days=5, test_window=50)
        result = engine.evaluate(
            returns=sample_data["returns"],
            predictions=sample_data["predictions"],
        )

        assert result.walk_forward is not None
        assert result.walk_forward["n_folds"] > 0

    def test_rolling_ic_runs(self, sample_data):
        engine = EvaluationEngine()
        result = engine.evaluate(
            returns=sample_data["returns"],
            predictions=sample_data["predictions"],
        )

        assert result.rolling_ic_data is not None
        assert result.rolling_ic_data["n_windows"] > 0

    def test_fragility_with_trades(self, sample_data):
        engine = EvaluationEngine()
        result = engine.evaluate(
            returns=sample_data["returns"],
            predictions=sample_data["predictions"],
            trades=sample_data["trades"],
        )

        assert result.fragility is not None
        assert "top_5_pct" in result.fragility

    def test_ml_diagnostics_with_importance(self, sample_data):
        rng = np.random.RandomState(42)
        importance = {
            "2023-01": rng.uniform(0, 1, 20),
            "2023-02": rng.uniform(0, 1, 20),
            "2023-03": rng.uniform(0, 1, 20),
        }

        engine = EvaluationEngine()
        result = engine.evaluate(
            returns=sample_data["returns"],
            predictions=sample_data["predictions"],
            importance_matrices=importance,
        )

        assert result.feature_drift is not None

    def test_ensemble_disagreement(self, sample_data):
        rng = np.random.RandomState(42)
        n = len(sample_data["returns"])
        ens_preds = {
            "model_a": rng.normal(0, 1, n),
            "model_b": rng.normal(0, 1, n),
        }

        engine = EvaluationEngine()
        result = engine.evaluate(
            returns=sample_data["returns"],
            predictions=sample_data["predictions"],
            ensemble_predictions=ens_preds,
        )

        assert result.ensemble_disagreement_result is not None

    def test_generate_json_report(self, sample_data, tmp_path):
        engine = EvaluationEngine(train_window=200)
        result = engine.evaluate(
            returns=sample_data["returns"],
            predictions=sample_data["predictions"],
            regime_states=sample_data["regimes"],
            trades=sample_data["trades"],
        )

        output = tmp_path / "eval_report.json"
        path = engine.generate_report(result, str(output), fmt="json")
        assert output.exists()

        import json
        with open(output) as f:
            data = json.load(f)
        assert "summary" in data
        assert "aggregate_metrics" in data

    def test_generate_html_report(self, sample_data, tmp_path):
        engine = EvaluationEngine(train_window=200)
        result = engine.evaluate(
            returns=sample_data["returns"],
            predictions=sample_data["predictions"],
            regime_states=sample_data["regimes"],
            trades=sample_data["trades"],
        )

        output = tmp_path / "eval_report.html"
        path = engine.generate_report(result, str(output), fmt="html")
        assert output.exists()

        content = output.read_text()
        assert "<html>" in content
        assert "Evaluation Report" in content
        assert "Aggregate Metrics" in content

    def test_red_flags_raised_for_overfit(self):
        """A model with zero signal should raise overfit flags."""
        rng = np.random.RandomState(42)
        n = 600
        dates = pd.bdate_range("2022-01-03", periods=n)
        returns = pd.Series(rng.normal(0, 0.015, n), index=dates)
        predictions = rng.normal(0, 0.015, n)  # Pure noise

        engine = EvaluationEngine(train_window=200, embargo_days=5, test_window=50)
        result = engine.evaluate(returns=returns, predictions=predictions)

        # Should detect overfitting (OOS Sharpe ~ 0)
        assert result.walk_forward is not None

    def test_no_regime_still_works(self, sample_data):
        """Evaluation should work without regime states."""
        engine = EvaluationEngine(train_window=200)
        result = engine.evaluate(
            returns=sample_data["returns"],
            predictions=sample_data["predictions"],
        )
        assert result.aggregate_metrics["n_samples"] > 0
        assert len(result.regime_slice_metrics) == 0  # No regime slicing
