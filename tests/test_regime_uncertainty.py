"""Tests for regime uncertainty integration (SPEC_10 T3).

Verifies:
  - Entropy computation: uniform posterior → max entropy, concentrated → low
  - Size multiplier interpolation from sizing map
  - Uncertainty gate reduces weights correctly
  - Stress assumption triggered at high entropy
  - Gate series produces correct DataFrame
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Entropy computation tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRegimeUncertaintyEntropy:
    """Test entropy computation on regime posteriors."""

    def test_uniform_posterior_max_entropy(self):
        from quant_engine.regime.detector import RegimeDetector

        probs = pd.DataFrame({
            "regime_prob_0": [0.25],
            "regime_prob_1": [0.25],
            "regime_prob_2": [0.25],
            "regime_prob_3": [0.25],
        })
        entropy = RegimeDetector.get_regime_uncertainty(probs)
        # Uniform distribution over 4 regimes → normalized entropy = 1.0
        np.testing.assert_allclose(entropy.values[0], 1.0, atol=0.01)

    def test_concentrated_posterior_low_entropy(self):
        from quant_engine.regime.detector import RegimeDetector

        probs = pd.DataFrame({
            "regime_prob_0": [0.90],
            "regime_prob_1": [0.05],
            "regime_prob_2": [0.03],
            "regime_prob_3": [0.02],
        })
        entropy = RegimeDetector.get_regime_uncertainty(probs)
        # Concentrated → low entropy
        assert entropy.values[0] < 0.5, f"Expected low entropy, got {entropy.values[0]:.3f}"

    def test_degenerate_posterior_zero_entropy(self):
        from quant_engine.regime.detector import RegimeDetector

        probs = pd.DataFrame({
            "regime_prob_0": [1.0],
            "regime_prob_1": [0.0],
            "regime_prob_2": [0.0],
            "regime_prob_3": [0.0],
        })
        entropy = RegimeDetector.get_regime_uncertainty(probs)
        # Degenerate (all mass on one regime) → near-zero entropy
        assert entropy.values[0] < 0.05, f"Expected near-zero entropy, got {entropy.values[0]:.3f}"

    def test_entropy_in_valid_range(self):
        from quant_engine.regime.detector import RegimeDetector

        rng = np.random.RandomState(42)
        n = 100
        raw = rng.dirichlet([1, 1, 1, 1], n)
        probs = pd.DataFrame(
            raw,
            columns=["regime_prob_0", "regime_prob_1", "regime_prob_2", "regime_prob_3"],
        )
        entropy = RegimeDetector.get_regime_uncertainty(probs)
        assert (entropy >= 0.0).all()
        assert (entropy <= 1.0).all()


# ---------------------------------------------------------------------------
# UncertaintyGate tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUncertaintyGate:
    """Test the UncertaintyGate sizing logic."""

    def test_zero_entropy_full_size(self):
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        gate = UncertaintyGate()
        multiplier = gate.compute_size_multiplier(0.0)
        assert multiplier == 1.0

    def test_moderate_entropy_reduces_size(self):
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        gate = UncertaintyGate()
        multiplier = gate.compute_size_multiplier(0.5)
        assert 0.90 <= multiplier <= 0.99, f"Expected ~0.95, got {multiplier}"

    def test_high_entropy_reduces_more(self):
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        gate = UncertaintyGate()
        mid = gate.compute_size_multiplier(0.5)
        high = gate.compute_size_multiplier(1.0)
        assert high < mid, f"High entropy {high} should reduce more than mid {mid}"

    def test_multiplier_never_below_floor(self):
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        gate = UncertaintyGate(min_multiplier=0.80)
        multiplier = gate.compute_size_multiplier(1.0)
        assert multiplier >= 0.80

    def test_apply_uncertainty_gate_reduces_weights(self):
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        gate = UncertaintyGate()
        weights = np.array([0.10, 0.05, 0.08, 0.07])
        adjusted = gate.apply_uncertainty_gate(weights, uncertainty=0.5)
        assert np.all(adjusted <= weights)
        assert np.all(adjusted > 0)

    def test_should_assume_stress_high_entropy(self):
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        gate = UncertaintyGate(stress_threshold=0.80)
        assert gate.should_assume_stress(0.85) is True
        assert gate.should_assume_stress(0.75) is False

    def test_is_uncertain_flag(self):
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        gate = UncertaintyGate(entropy_threshold=0.50)
        assert gate.is_uncertain(0.55) is True
        assert gate.is_uncertain(0.45) is False

    def test_gate_series(self):
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        gate = UncertaintyGate()
        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        uncertainties = pd.Series([0.0, 0.3, 0.5, 0.8, 1.0], index=idx)

        result = gate.gate_series(uncertainties)
        assert "multiplier" in result.columns
        assert "is_uncertain" in result.columns
        assert "assume_stress" in result.columns
        assert len(result) == 5
        # Multipliers should be monotonically non-increasing
        assert result["multiplier"].iloc[0] >= result["multiplier"].iloc[-1]

    def test_custom_sizing_map(self):
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        gate = UncertaintyGate(sizing_map={0.0: 1.0, 0.5: 0.8, 1.0: 0.5}, min_multiplier=0.0)
        assert gate.compute_size_multiplier(0.0) == 1.0
        assert abs(gate.compute_size_multiplier(0.5) - 0.8) < 0.01
        assert abs(gate.compute_size_multiplier(1.0) - 0.5) < 0.01
        # Midpoint should interpolate
        assert 0.8 < gate.compute_size_multiplier(0.25) < 1.0
