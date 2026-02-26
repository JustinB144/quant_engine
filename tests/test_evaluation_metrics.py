"""Tests for evaluation/metrics.py — slice metrics and decile spread."""

import numpy as np
import pandas as pd
import pytest

from quant_engine.evaluation.metrics import compute_slice_metrics, decile_spread


@pytest.fixture
def positive_returns():
    """Returns with known positive drift."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2022-01-03", periods=500)
    return pd.Series(rng.normal(0.001, 0.01, 500), index=dates)


@pytest.fixture
def negative_returns():
    """Returns with known negative drift."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2022-01-03", periods=500)
    return pd.Series(rng.normal(-0.002, 0.01, 500), index=dates)


class TestComputeSliceMetrics:
    """Test compute_slice_metrics on various return profiles."""

    def test_positive_returns_positive_sharpe(self, positive_returns):
        metrics = compute_slice_metrics(positive_returns)
        assert metrics["sharpe"] > 0
        assert metrics["mean_return"] > 0
        assert metrics["n_samples"] == 500
        assert metrics["confidence"] == "high"

    def test_negative_returns_negative_sharpe(self, negative_returns):
        metrics = compute_slice_metrics(negative_returns)
        assert metrics["sharpe"] < 0
        assert metrics["mean_return"] < 0

    def test_empty_returns(self):
        empty = pd.Series(dtype=float)
        metrics = compute_slice_metrics(empty)
        assert metrics["n_samples"] == 0
        assert metrics["sharpe"] == 0.0
        assert metrics["confidence"] == "low"

    def test_small_sample_low_confidence(self):
        returns = pd.Series([0.01, -0.005, 0.003], index=pd.bdate_range("2022-01-03", periods=3))
        metrics = compute_slice_metrics(returns)
        assert metrics["n_samples"] == 3
        assert metrics["confidence"] == "low"

    def test_max_drawdown_computed(self, positive_returns):
        metrics = compute_slice_metrics(positive_returns)
        assert metrics["max_dd"] <= 0  # Drawdown is negative
        assert metrics["max_dd_duration"] >= 0

    def test_win_rate_range(self, positive_returns):
        metrics = compute_slice_metrics(positive_returns)
        assert 0 <= metrics["win_rate"] <= 1

    def test_ic_with_predictions(self, positive_returns):
        rng = np.random.RandomState(42)
        # Correlated predictions
        predictions = positive_returns.values + rng.normal(0, 0.005, 500)
        metrics = compute_slice_metrics(positive_returns, predictions)
        assert metrics["ic"] > 0  # Should have positive IC since correlated


class TestDecileSpread:
    """Test decile_spread on synthetic data."""

    def test_perfect_predictions_positive_spread(self):
        """When predictions perfectly correlate with returns, spread > 0."""
        rng = np.random.RandomState(42)
        n = 500
        factor = rng.normal(0, 0.01, n)
        predictions = factor + rng.normal(0, 0.001, n)
        returns = pd.Series(factor + rng.normal(0, 0.002, n))

        result = decile_spread(predictions, returns)

        assert result["spread"] > 0
        assert result["n_total"] == n
        assert len(result["decile_returns"]) == 10
        assert result["monotonicity"] > 0.5  # Strong positive monotonicity

    def test_random_predictions_near_zero_spread(self):
        """Random predictions should have spread ~0."""
        rng = np.random.RandomState(42)
        n = 1000
        predictions = rng.normal(0, 1, n)
        returns = pd.Series(rng.normal(0, 0.01, n))

        result = decile_spread(predictions, returns)
        assert abs(result["spread"]) < 0.005  # Near zero

    def test_insufficient_data(self):
        predictions = np.array([0.1, 0.2])
        returns = pd.Series([0.01, -0.01])
        result = decile_spread(predictions, returns)
        assert result["n_total"] == 0
        assert result["spread"] == 0.0

    def test_decile_counts_sum_to_total(self):
        rng = np.random.RandomState(42)
        n = 500
        predictions = rng.normal(0, 1, n)
        returns = pd.Series(rng.normal(0, 0.01, n))

        result = decile_spread(predictions, returns)
        assert sum(result["decile_counts"]) == n

    def test_per_regime_decomposition(self):
        rng = np.random.RandomState(42)
        n = 500
        predictions = rng.normal(0, 1, n)
        returns = pd.Series(rng.normal(0, 0.01, n))
        regimes = np.array([0] * 250 + [3] * 250, dtype=int)

        result = decile_spread(predictions, returns, regime_states=regimes)
        assert "per_regime" in result
        assert len(result["per_regime"]) >= 1
