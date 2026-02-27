"""Tests for SPEC-V01: Uncertainty quantile slicing and regime transition slicing.

Verifies that:
- SliceRegistry.create_uncertainty_slices partitions bars by entropy quantiles
- SliceRegistry.create_transition_slices identifies regime-change proximity
- EvaluationEngine wires both into evaluate() when regime data is provided
- EvaluationResult contains uncertainty_q1..q3, near_transition, stable_regime
- HTML and JSON reports include the new slice sections
"""

import json

import numpy as np
import pandas as pd
import pytest

from quant_engine.evaluation.slicing import PerformanceSlice, SliceRegistry
from quant_engine.evaluation.engine import EvaluationEngine, EvaluationResult


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_returns():
    """500-day return series with mild positive drift."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2022-01-03", periods=500)
    return pd.Series(rng.normal(0.0005, 0.015, 500), index=dates)


@pytest.fixture
def sample_regimes():
    """Regime labels with transitions: 0→1→2→3→0 pattern."""
    return np.array(
        [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [0] * 100, dtype=int
    )


@pytest.fixture
def sample_uncertainty():
    """Uncertainty values spanning [0, 1] with known quantile structure.

    First 167 bars: low uncertainty (0.0–0.3)
    Next 167 bars:  mid uncertainty (0.3–0.6)
    Last 166 bars:  high uncertainty (0.6–1.0)
    """
    rng = np.random.RandomState(42)
    low = rng.uniform(0.0, 0.3, 167)
    mid = rng.uniform(0.3, 0.6, 167)
    high = rng.uniform(0.6, 1.0, 166)
    return np.concatenate([low, mid, high])


@pytest.fixture
def engine_data(sample_returns, sample_regimes, sample_uncertainty):
    """Complete data dict for EvaluationEngine.evaluate()."""
    rng = np.random.RandomState(42)
    n = len(sample_returns)
    predictions = sample_returns.values + rng.normal(0, 0.005, n)
    return {
        "returns": sample_returns,
        "predictions": predictions,
        "regimes": sample_regimes,
        "uncertainty": sample_uncertainty,
    }


# ── Tests: create_uncertainty_slices ────────────────────────────────────


class TestCreateUncertaintySlices:
    """Test SliceRegistry.create_uncertainty_slices()."""

    def test_returns_correct_number_of_slices(self, sample_uncertainty):
        slices = SliceRegistry.create_uncertainty_slices(sample_uncertainty, n_quantiles=3)
        assert len(slices) == 3

    def test_slice_names_follow_convention(self, sample_uncertainty):
        slices = SliceRegistry.create_uncertainty_slices(sample_uncertainty, n_quantiles=3)
        names = [s.name for s in slices]
        assert names == ["uncertainty_q1", "uncertainty_q2", "uncertainty_q3"]

    def test_four_quantile_slices(self, sample_uncertainty):
        slices = SliceRegistry.create_uncertainty_slices(sample_uncertainty, n_quantiles=4)
        assert len(slices) == 4
        assert slices[-1].name == "uncertainty_q4"

    def test_slices_are_performance_slice_instances(self, sample_uncertainty):
        slices = SliceRegistry.create_uncertainty_slices(sample_uncertainty)
        for slc in slices:
            assert isinstance(slc, PerformanceSlice)

    def test_quantiles_cover_all_bars(self, sample_returns, sample_regimes, sample_uncertainty):
        """All bars should appear in exactly one quantile slice."""
        slices = SliceRegistry.create_uncertainty_slices(sample_uncertainty, n_quantiles=3)
        meta = SliceRegistry.build_metadata(
            sample_returns, sample_regimes, uncertainty=sample_uncertainty,
        )

        total_selected = 0
        for slc in slices:
            filtered, info = slc.apply(sample_returns, meta)
            total_selected += info["n_samples"]

        assert total_selected == len(sample_returns)

    def test_low_quantile_has_low_uncertainty(self, sample_returns, sample_regimes, sample_uncertainty):
        """Q1 should contain the lowest-uncertainty bars."""
        slices = SliceRegistry.create_uncertainty_slices(sample_uncertainty, n_quantiles=3)
        meta = SliceRegistry.build_metadata(
            sample_returns, sample_regimes, uncertainty=sample_uncertainty,
        )

        q1 = slices[0]
        mask = np.asarray(q1.condition(meta), dtype=bool)
        q1_uncertainty_mean = sample_uncertainty[mask].mean()

        q3 = slices[2]
        mask3 = np.asarray(q3.condition(meta), dtype=bool)
        q3_uncertainty_mean = sample_uncertainty[mask3].mean()

        assert q1_uncertainty_mean < q3_uncertainty_mean

    def test_high_quantile_has_high_uncertainty(self, sample_returns, sample_regimes, sample_uncertainty):
        """Q3 (high) should have higher mean uncertainty than Q2 (mid)."""
        slices = SliceRegistry.create_uncertainty_slices(sample_uncertainty, n_quantiles=3)
        meta = SliceRegistry.build_metadata(
            sample_returns, sample_regimes, uncertainty=sample_uncertainty,
        )

        q2_mask = np.asarray(slices[1].condition(meta), dtype=bool)
        q3_mask = np.asarray(slices[2].condition(meta), dtype=bool)

        assert sample_uncertainty[q3_mask].mean() > sample_uncertainty[q2_mask].mean()

    def test_uniform_uncertainty_splits_evenly(self):
        """Uniform uncertainty should produce roughly equal-sized quantiles."""
        rng = np.random.RandomState(99)
        unc = rng.uniform(0, 1, 300)
        slices = SliceRegistry.create_uncertainty_slices(unc, n_quantiles=3)

        dates = pd.bdate_range("2022-01-03", periods=300)
        returns = pd.Series(rng.normal(0, 0.01, 300), index=dates)
        regimes = np.full(300, 0, dtype=int)
        meta = SliceRegistry.build_metadata(returns, regimes, uncertainty=unc)

        counts = []
        for slc in slices:
            _, info = slc.apply(returns, meta)
            counts.append(info["n_samples"])

        # Each quantile should have roughly 100 bars (allow ±20)
        for c in counts:
            assert 80 <= c <= 120, f"Quantile count {c} not near 100"

    def test_constant_uncertainty_all_in_one_slice(self):
        """If all uncertainty values are identical, all bars go to a single slice."""
        unc = np.full(200, 0.5)
        slices = SliceRegistry.create_uncertainty_slices(unc, n_quantiles=3)

        dates = pd.bdate_range("2022-01-03", periods=200)
        returns = pd.Series(np.zeros(200), index=dates)
        regimes = np.zeros(200, dtype=int)
        meta = SliceRegistry.build_metadata(returns, regimes, uncertainty=unc)

        total = 0
        for slc in slices:
            _, info = slc.apply(returns, meta)
            total += info["n_samples"]

        # All 200 bars should be accounted for
        assert total == 200

    def test_handles_nan_uncertainty(self, sample_returns, sample_regimes):
        """NaN in uncertainty should be handled gracefully (fillna in metadata)."""
        unc = np.random.RandomState(42).uniform(0, 1, 500)
        unc[10:15] = np.nan

        slices = SliceRegistry.create_uncertainty_slices(unc, n_quantiles=3)
        meta = SliceRegistry.build_metadata(
            sample_returns, sample_regimes, uncertainty=unc,
        )

        total = 0
        for slc in slices:
            _, info = slc.apply(sample_returns, meta)
            total += info["n_samples"]

        # All bars should be assigned (NaN fills to 0.0 in metadata)
        assert total == 500

    def test_uses_metadata_uncertainty_column(self, sample_returns, sample_regimes, sample_uncertainty):
        """Slices should use metadata 'uncertainty' column when present."""
        slices = SliceRegistry.create_uncertainty_slices(sample_uncertainty, n_quantiles=3)

        # Build metadata WITH uncertainty
        meta = SliceRegistry.build_metadata(
            sample_returns, sample_regimes, uncertainty=sample_uncertainty,
        )
        assert "uncertainty" in meta.columns

        # Apply slices — should use metadata column
        total = 0
        for slc in slices:
            _, info = slc.apply(sample_returns, meta)
            total += info["n_samples"]
        assert total == 500


# ── Tests: create_transition_slices ─────────────────────────────────────


class TestCreateTransitionSlices:
    """Test SliceRegistry.create_transition_slices()."""

    def test_returns_two_slices(self, sample_regimes):
        slices = SliceRegistry.create_transition_slices(sample_regimes)
        assert len(slices) == 2

    def test_slice_names(self, sample_regimes):
        slices = SliceRegistry.create_transition_slices(sample_regimes)
        names = [s.name for s in slices]
        assert "near_transition" in names
        assert "stable_regime" in names

    def test_slices_are_exhaustive(self, sample_returns, sample_regimes):
        """near_transition + stable_regime should cover all bars."""
        slices = SliceRegistry.create_transition_slices(sample_regimes)
        meta = SliceRegistry.build_metadata(sample_returns, sample_regimes)

        total = 0
        for slc in slices:
            _, info = slc.apply(sample_returns, meta)
            total += info["n_samples"]

        assert total == len(sample_returns)

    def test_slices_are_mutually_exclusive(self, sample_returns, sample_regimes):
        """A bar cannot be both near_transition and stable_regime."""
        slices = SliceRegistry.create_transition_slices(sample_regimes)
        meta = SliceRegistry.build_metadata(sample_returns, sample_regimes)

        near = slices[0]
        stable = slices[1]

        near_mask = np.asarray(near.condition(meta), dtype=bool)
        stable_mask = np.asarray(stable.condition(meta), dtype=bool)

        # No overlap
        overlap = np.sum(near_mask & stable_mask)
        assert overlap == 0

        # Full coverage
        assert np.sum(near_mask | stable_mask) == len(sample_returns)

    def test_detects_transitions(self, sample_returns, sample_regimes):
        """Regime transitions at indices 100, 200, 300, 400 should be detected."""
        slices = SliceRegistry.create_transition_slices(sample_regimes, window=5)
        meta = SliceRegistry.build_metadata(sample_returns, sample_regimes)

        near_slice = [s for s in slices if s.name == "near_transition"][0]
        _, info = near_slice.apply(sample_returns, meta)

        # With window=5, each of 4 transitions marks ~5 bars each
        assert info["n_samples"] > 0
        assert info["n_samples"] < len(sample_returns)

    def test_no_transitions_all_stable(self, sample_returns):
        """Constant regime should have zero near_transition bars."""
        constant_regimes = np.zeros(len(sample_returns), dtype=int)
        slices = SliceRegistry.create_transition_slices(constant_regimes, window=5)
        meta = SliceRegistry.build_metadata(sample_returns, constant_regimes)

        near_slice = [s for s in slices if s.name == "near_transition"][0]
        _, info = near_slice.apply(sample_returns, meta)
        assert info["n_samples"] == 0

        stable_slice = [s for s in slices if s.name == "stable_regime"][0]
        _, info_s = stable_slice.apply(sample_returns, meta)
        assert info_s["n_samples"] == len(sample_returns)

    def test_larger_window_catches_more_bars(self, sample_returns, sample_regimes):
        """Larger window should mark more bars as near_transition."""
        slices_small = SliceRegistry.create_transition_slices(sample_regimes, window=3)
        slices_large = SliceRegistry.create_transition_slices(sample_regimes, window=10)

        meta = SliceRegistry.build_metadata(sample_returns, sample_regimes)

        near_small = [s for s in slices_small if s.name == "near_transition"][0]
        near_large = [s for s in slices_large if s.name == "near_transition"][0]

        _, info_small = near_small.apply(sample_returns, meta)
        _, info_large = near_large.apply(sample_returns, meta)

        assert info_large["n_samples"] >= info_small["n_samples"]

    def test_every_other_bar_transition(self):
        """Rapidly alternating regimes should mark nearly all bars near_transition."""
        n = 100
        regimes = np.array([i % 2 for i in range(n)], dtype=int)
        dates = pd.bdate_range("2022-01-03", periods=n)
        returns = pd.Series(np.zeros(n), index=dates)

        slices = SliceRegistry.create_transition_slices(regimes, window=3)
        meta = SliceRegistry.build_metadata(returns, regimes)

        near_slice = [s for s in slices if s.name == "near_transition"][0]
        _, info = near_slice.apply(returns, meta)

        # Nearly all bars should be near a transition
        assert info["n_samples"] >= 90


# ── Tests: build_metadata with uncertainty ──────────────────────────────


class TestBuildMetadataUncertainty:
    """Test that build_metadata correctly includes uncertainty."""

    def test_metadata_includes_uncertainty_column(self, sample_returns, sample_regimes, sample_uncertainty):
        meta = SliceRegistry.build_metadata(
            sample_returns, sample_regimes, uncertainty=sample_uncertainty,
        )
        assert "uncertainty" in meta.columns
        assert len(meta["uncertainty"]) == len(sample_returns)

    def test_metadata_without_uncertainty(self, sample_returns, sample_regimes):
        meta = SliceRegistry.build_metadata(sample_returns, sample_regimes)
        assert "uncertainty" not in meta.columns

    def test_uncertainty_length_mismatch_skipped(self, sample_returns, sample_regimes):
        """Mismatched uncertainty length should be skipped with warning."""
        bad_unc = np.ones(50)  # Wrong length
        meta = SliceRegistry.build_metadata(
            sample_returns, sample_regimes, uncertainty=bad_unc,
        )
        assert "uncertainty" not in meta.columns


# ── Tests: EvaluationEngine integration ─────────────────────────────────


class TestEvaluationEngineUncertaintySlicing:
    """Test that EvaluationEngine.evaluate() produces uncertainty and transition slices."""

    def test_uncertainty_slices_populated(self, engine_data):
        engine = EvaluationEngine(train_window=200)
        result = engine.evaluate(
            returns=engine_data["returns"],
            predictions=engine_data["predictions"],
            regime_states=engine_data["regimes"],
            regime_uncertainty=engine_data["uncertainty"],
        )

        assert isinstance(result.uncertainty_slice_metrics, dict)
        assert len(result.uncertainty_slice_metrics) > 0
        assert "uncertainty_q1" in result.uncertainty_slice_metrics
        assert "uncertainty_q2" in result.uncertainty_slice_metrics
        assert "uncertainty_q3" in result.uncertainty_slice_metrics

    def test_transition_slices_populated(self, engine_data):
        engine = EvaluationEngine(train_window=200)
        result = engine.evaluate(
            returns=engine_data["returns"],
            predictions=engine_data["predictions"],
            regime_states=engine_data["regimes"],
        )

        assert isinstance(result.transition_slice_metrics, dict)
        assert len(result.transition_slice_metrics) > 0
        assert "near_transition" in result.transition_slice_metrics
        assert "stable_regime" in result.transition_slice_metrics

    def test_uncertainty_slices_contain_standard_metrics(self, engine_data):
        engine = EvaluationEngine(train_window=200)
        result = engine.evaluate(
            returns=engine_data["returns"],
            predictions=engine_data["predictions"],
            regime_states=engine_data["regimes"],
            regime_uncertainty=engine_data["uncertainty"],
        )

        for name, metrics in result.uncertainty_slice_metrics.items():
            assert "sharpe" in metrics, f"Missing sharpe in {name}"
            assert "mean_return" in metrics, f"Missing mean_return in {name}"
            assert "max_dd" in metrics, f"Missing max_dd in {name}"
            assert "n_samples" in metrics, f"Missing n_samples in {name}"
            assert metrics["n_samples"] > 0

    def test_transition_slices_contain_standard_metrics(self, engine_data):
        engine = EvaluationEngine(train_window=200)
        result = engine.evaluate(
            returns=engine_data["returns"],
            predictions=engine_data["predictions"],
            regime_states=engine_data["regimes"],
        )

        for name, metrics in result.transition_slice_metrics.items():
            assert "sharpe" in metrics, f"Missing sharpe in {name}"
            assert "n_samples" in metrics, f"Missing n_samples in {name}"

    def test_no_regime_states_no_uncertainty_slices(self, engine_data):
        """Without regime_states, uncertainty slices should be empty."""
        engine = EvaluationEngine(train_window=200)
        result = engine.evaluate(
            returns=engine_data["returns"],
            predictions=engine_data["predictions"],
        )

        assert len(result.uncertainty_slice_metrics) == 0
        assert len(result.transition_slice_metrics) == 0

    def test_regime_states_without_uncertainty_no_uncertainty_slices(self, engine_data):
        """With regime_states but no uncertainty, uncertainty slices should be empty."""
        engine = EvaluationEngine(train_window=200)
        result = engine.evaluate(
            returns=engine_data["returns"],
            predictions=engine_data["predictions"],
            regime_states=engine_data["regimes"],
            # No regime_uncertainty
        )

        assert len(result.uncertainty_slice_metrics) == 0
        # Transition slices should still be populated
        assert len(result.transition_slice_metrics) > 0


# ── Tests: Report generation ────────────────────────────────────────────


class TestReportIncludesNewSlices:
    """Test that HTML and JSON reports include the new V01 slice data."""

    def test_json_report_includes_uncertainty_slices(self, engine_data, tmp_path):
        engine = EvaluationEngine(train_window=200)
        result = engine.evaluate(
            returns=engine_data["returns"],
            predictions=engine_data["predictions"],
            regime_states=engine_data["regimes"],
            regime_uncertainty=engine_data["uncertainty"],
        )

        output = tmp_path / "eval_report.json"
        engine.generate_report(result, str(output), fmt="json")

        with open(output) as f:
            data = json.load(f)

        assert "uncertainty_slice_metrics" in data
        assert "transition_slice_metrics" in data
        assert "uncertainty_q1" in data["uncertainty_slice_metrics"]
        assert "near_transition" in data["transition_slice_metrics"]

    def test_html_report_includes_uncertainty_sections(self, engine_data, tmp_path):
        engine = EvaluationEngine(train_window=200)
        result = engine.evaluate(
            returns=engine_data["returns"],
            predictions=engine_data["predictions"],
            regime_states=engine_data["regimes"],
            regime_uncertainty=engine_data["uncertainty"],
        )

        output = tmp_path / "eval_report.html"
        engine.generate_report(result, str(output), fmt="html")

        content = output.read_text()
        assert "Uncertainty Quantile Slices" in content
        assert "Regime Transition Slices" in content
        assert "uncertainty_q1" in content
        assert "near_transition" in content

    def test_json_report_without_uncertainty(self, engine_data, tmp_path):
        """JSON report should include empty dicts when no uncertainty provided."""
        engine = EvaluationEngine(train_window=200)
        result = engine.evaluate(
            returns=engine_data["returns"],
            predictions=engine_data["predictions"],
            regime_states=engine_data["regimes"],
        )

        output = tmp_path / "eval_report.json"
        engine.generate_report(result, str(output), fmt="json")

        with open(output) as f:
            data = json.load(f)

        assert "uncertainty_slice_metrics" in data
        assert data["uncertainty_slice_metrics"] == {}
        # Transition slices should be present even without uncertainty
        assert "transition_slice_metrics" in data
        assert len(data["transition_slice_metrics"]) > 0
