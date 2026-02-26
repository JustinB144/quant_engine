"""Unit tests for BOCPD (Bayesian Online Change-Point Detection).

Tests cover:
  - Synthetic mean-shift detection (the core BOCPD use case)
  - Variance-shift detection
  - Multiple changepoints
  - Edge cases (empty input, single observation, non-finite values)
  - Batch vs sequential consistency
  - Parameter validation
  - Performance characteristics (run-length tracking)

Reference: Adams & MacKay (2007), "Bayesian Online Changepoint Detection"
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def mean_shift_series(rng):
    """Synthetic series: N(0,1) for 100 bars, then N(3,1) for 100 bars."""
    return np.concatenate([
        rng.normal(0.0, 1.0, 100),
        rng.normal(3.0, 1.0, 100),
    ])


@pytest.fixture
def variance_shift_series(rng):
    """Synthetic series: N(0,0.5) for 100 bars, then N(0,3.0) for 100 bars."""
    return np.concatenate([
        rng.normal(0.0, 0.5, 100),
        rng.normal(0.0, 3.0, 100),
    ])


@pytest.fixture
def multi_changepoint_series(rng):
    """Synthetic series with 3 changepoints at bars 80, 160, 240."""
    return np.concatenate([
        rng.normal(0.0, 1.0, 80),   # Segment 1
        rng.normal(3.0, 1.0, 80),   # Segment 2
        rng.normal(-2.0, 0.5, 80),  # Segment 3
        rng.normal(1.0, 2.0, 80),   # Segment 4
    ])


# ── Core BOCPD Detection Tests ──────────────────────────────────────────


class TestBOCPDMeanShift:
    """Test BOCPD detection of mean shifts."""

    def test_detects_mean_shift(self, mean_shift_series):
        from quant_engine.regime.bocpd import BOCPDDetector

        detector = BOCPDDetector(hazard_lambda=1 / 100, max_runlength=200)
        result = detector.batch_update(mean_shift_series)

        # Changepoint probability should spike near bar 100.
        # With hazard_lambda=0.01, baseline P(cp) ≈ 0.01.  A spike above 0.1
        # represents a 10x+ increase — a clear detection signal.
        window = result.changepoint_probs[90:115]
        assert np.max(window) > 0.1, (
            f"Expected changepoint spike near bar 100, max was {np.max(window):.3f}"
        )

    def test_changepoint_probs_bounded(self, mean_shift_series):
        from quant_engine.regime.bocpd import BOCPDDetector

        detector = BOCPDDetector(hazard_lambda=1 / 60)
        result = detector.batch_update(mean_shift_series)

        assert np.all(result.changepoint_probs >= 0.0)
        assert np.all(result.changepoint_probs <= 1.0)

    def test_run_lengths_track_segments(self, mean_shift_series):
        from quant_engine.regime.bocpd import BOCPDDetector

        detector = BOCPDDetector(hazard_lambda=1 / 100, max_runlength=200)
        result = detector.batch_update(mean_shift_series)

        # In the second segment (bars 100-199), run-lengths should generally
        # increase from 0 (reset at changepoint) up to ~100.
        second_segment_rl = result.run_lengths[120:190]
        assert np.mean(second_segment_rl) > 5, (
            "Run-lengths in second segment should be increasing"
        )


class TestBOCPDVarianceShift:
    """Test BOCPD detection of variance shifts."""

    def test_detects_variance_shift(self, variance_shift_series):
        from quant_engine.regime.bocpd import BOCPDDetector

        detector = BOCPDDetector(hazard_lambda=1 / 80, max_runlength=200)
        result = detector.batch_update(variance_shift_series)

        # Variance shifts are harder to detect than mean shifts, but
        # changepoint probability should still elevate near bar 100.
        window = result.changepoint_probs[90:120]
        assert np.max(window) > 0.1, (
            f"Expected some changepoint signal near variance shift, "
            f"max was {np.max(window):.3f}"
        )


class TestBOCPDMultipleChangepoints:
    """Test BOCPD detection of multiple changepoints."""

    def test_detects_multiple_changepoints(self, multi_changepoint_series):
        from quant_engine.regime.bocpd import BOCPDDetector

        detector = BOCPDDetector(hazard_lambda=1 / 60, max_runlength=200)
        result = detector.batch_update(multi_changepoint_series)

        # Check for elevated changepoint probability near each transition.
        for cp_loc in [80, 160, 240]:
            window = result.changepoint_probs[max(0, cp_loc - 10):cp_loc + 15]
            assert np.max(window) > 0.1, (
                f"Expected changepoint signal near bar {cp_loc}, "
                f"max was {np.max(window):.3f}"
            )


# ── Output Shape and Type Tests ──────────────────────────────────────────


class TestBOCPDBatchOutput:
    """Test batch output shapes and types."""

    def test_output_shapes(self, mean_shift_series):
        from quant_engine.regime.bocpd import BOCPDDetector

        detector = BOCPDDetector()
        result = detector.batch_update(mean_shift_series)

        T = len(mean_shift_series)
        assert result.changepoint_probs.shape == (T,)
        assert result.run_lengths.shape == (T,)
        assert result.predicted_means.shape == (T,)
        assert result.predicted_stds.shape == (T,)

    def test_output_types(self, mean_shift_series):
        from quant_engine.regime.bocpd import BOCPDDetector

        detector = BOCPDDetector()
        result = detector.batch_update(mean_shift_series)

        assert result.changepoint_probs.dtype == np.float64
        assert result.run_lengths.dtype == np.intp or result.run_lengths.dtype == np.int64
        assert result.predicted_means.dtype == np.float64
        assert result.predicted_stds.dtype == np.float64

    def test_predicted_stds_positive(self, mean_shift_series):
        from quant_engine.regime.bocpd import BOCPDDetector

        detector = BOCPDDetector()
        result = detector.batch_update(mean_shift_series)

        assert np.all(result.predicted_stds >= 0.0), (
            "Predicted std should be non-negative"
        )

    def test_no_nan_in_output(self, mean_shift_series):
        from quant_engine.regime.bocpd import BOCPDDetector

        detector = BOCPDDetector()
        result = detector.batch_update(mean_shift_series)

        assert not np.any(np.isnan(result.changepoint_probs))
        assert not np.any(np.isnan(result.run_lengths))
        assert not np.any(np.isnan(result.predicted_means))
        assert not np.any(np.isnan(result.predicted_stds))


# ── Single-Step Update Tests ─────────────────────────────────────────────


class TestBOCPDSingleUpdate:
    """Test single-observation update method."""

    def test_first_update(self):
        from quant_engine.regime.bocpd import BOCPDDetector

        detector = BOCPDDetector(max_runlength=50)
        result = detector.update(0.5)

        assert result.run_length_posterior.shape == (50,)
        assert abs(np.sum(result.run_length_posterior) - 1.0) < 1e-6
        assert 0.0 <= result.changepoint_prob <= 1.0

    def test_sequential_updates(self, rng):
        from quant_engine.regime.bocpd import BOCPDDetector

        detector = BOCPDDetector(max_runlength=50)
        data = rng.normal(0.0, 1.0, 20)

        for x in data:
            result = detector.update(x)
            assert abs(np.sum(result.run_length_posterior) - 1.0) < 1e-6

    def test_non_finite_observation_handled(self):
        from quant_engine.regime.bocpd import BOCPDDetector

        detector = BOCPDDetector()
        # First normal observation.
        detector.update(1.0)
        # NaN observation should not crash.
        result = detector.update(float("nan"))
        assert np.isfinite(result.changepoint_prob)

    def test_inf_observation_handled(self):
        from quant_engine.regime.bocpd import BOCPDDetector

        detector = BOCPDDetector()
        detector.update(0.0)
        result = detector.update(float("inf"))
        assert np.isfinite(result.changepoint_prob)


# ── Edge Cases ───────────────────────────────────────────────────────────


class TestBOCPDEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_series(self):
        from quant_engine.regime.bocpd import BOCPDDetector

        detector = BOCPDDetector()
        result = detector.batch_update(np.array([]))

        assert len(result.changepoint_probs) == 0
        assert len(result.run_lengths) == 0

    def test_single_observation(self):
        from quant_engine.regime.bocpd import BOCPDDetector

        detector = BOCPDDetector()
        result = detector.batch_update(np.array([1.0]))

        assert len(result.changepoint_probs) == 1
        assert len(result.run_lengths) == 1

    def test_constant_series(self, rng):
        from quant_engine.regime.bocpd import BOCPDDetector

        detector = BOCPDDetector(hazard_lambda=1 / 60)
        data = np.ones(100) * 5.0
        result = detector.batch_update(data)

        # Constant series: no changepoints expected after initial warmup.
        # Changepoint probability should be low in the latter half.
        assert np.mean(result.changepoint_probs[50:]) < 0.3

    def test_reset_clears_state(self):
        from quant_engine.regime.bocpd import BOCPDDetector

        detector = BOCPDDetector()
        detector.update(1.0)
        detector.update(2.0)
        detector.reset()

        assert detector._posterior is None
        assert detector._t == 0


# ── Parameter Validation ─────────────────────────────────────────────────


class TestBOCPDParameterValidation:
    """Test constructor parameter validation."""

    def test_invalid_hazard_lambda_zero(self):
        from quant_engine.regime.bocpd import BOCPDDetector
        with pytest.raises(ValueError, match="hazard_lambda"):
            BOCPDDetector(hazard_lambda=0.0)

    def test_invalid_hazard_lambda_one(self):
        from quant_engine.regime.bocpd import BOCPDDetector
        with pytest.raises(ValueError, match="hazard_lambda"):
            BOCPDDetector(hazard_lambda=1.0)

    def test_invalid_hazard_lambda_negative(self):
        from quant_engine.regime.bocpd import BOCPDDetector
        with pytest.raises(ValueError, match="hazard_lambda"):
            BOCPDDetector(hazard_lambda=-0.1)

    def test_invalid_max_runlength(self):
        from quant_engine.regime.bocpd import BOCPDDetector
        with pytest.raises(ValueError, match="max_runlength"):
            BOCPDDetector(max_runlength=1)

    def test_invalid_hazard_func(self):
        from quant_engine.regime.bocpd import BOCPDDetector
        with pytest.raises(ValueError, match="hazard_func"):
            BOCPDDetector(hazard_func="exponential")

    def test_geometric_hazard(self, mean_shift_series):
        from quant_engine.regime.bocpd import BOCPDDetector

        detector = BOCPDDetector(hazard_func="geometric", hazard_lambda=1 / 100)
        result = detector.batch_update(mean_shift_series)

        # Should still produce valid output.
        assert np.all(result.changepoint_probs >= 0.0)
        assert np.all(result.changepoint_probs <= 1.0)


# ── Batch vs Sequential Consistency ──────────────────────────────────────


class TestBOCPDConsistency:
    """Test that batch and sequential updates produce identical results."""

    def test_batch_matches_sequential(self, rng):
        from quant_engine.regime.bocpd import BOCPDDetector

        data = rng.normal(0.0, 1.0, 50)

        # Batch mode.
        det_batch = BOCPDDetector(hazard_lambda=1 / 30, max_runlength=100)
        batch_result = det_batch.batch_update(data)

        # Sequential mode.
        det_seq = BOCPDDetector(hazard_lambda=1 / 30, max_runlength=100)
        seq_cp = np.zeros(50)
        seq_rl = np.zeros(50, dtype=int)
        for t, x in enumerate(data):
            r = det_seq.update(x)
            seq_cp[t] = r.changepoint_prob
            seq_rl[t] = r.most_likely_runlength

        np.testing.assert_allclose(batch_result.changepoint_probs, seq_cp, atol=1e-10)
        np.testing.assert_array_equal(batch_result.run_lengths, seq_rl)
