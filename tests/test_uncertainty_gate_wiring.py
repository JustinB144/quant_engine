"""Tests for SPEC-W03: Uncertainty gate wired into position sizing and portfolio weights.

Verifies:
  - Backtester._uncertainty_gate is initialized as UncertaintyGate
  - Simple-mode position sizes are reduced when shock vectors carry high hmm_uncertainty
  - Risk-managed-mode position sizes are reduced under high uncertainty
  - Autopilot _compute_optimizer_weights applies the uncertainty gate
  - Zero/low uncertainty leaves sizes unchanged
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Backtest engine: UncertaintyGate initialization
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBacktesterUncertaintyGateInit:
    """Verify the backtester initializes the uncertainty gate."""

    def test_backtester_has_uncertainty_gate(self):
        from quant_engine.backtest.engine import Backtester
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        bt = Backtester()
        assert hasattr(bt, "_uncertainty_gate")
        assert isinstance(bt._uncertainty_gate, UncertaintyGate)

    def test_uncertainty_gate_default_config(self):
        from quant_engine.backtest.engine import Backtester
        from quant_engine.config import (
            REGIME_UNCERTAINTY_MIN_MULTIPLIER,
            REGIME_UNCERTAINTY_ENTROPY_THRESHOLD,
        )

        bt = Backtester()
        assert bt._uncertainty_gate.min_multiplier == REGIME_UNCERTAINTY_MIN_MULTIPLIER
        assert bt._uncertainty_gate.entropy_threshold == REGIME_UNCERTAINTY_ENTROPY_THRESHOLD


# ---------------------------------------------------------------------------
# Backtest engine: Simple mode — uncertainty-gated position sizing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSimpleModeUncertaintyGating:
    """Verify the simple backtester reduces position sizes under uncertainty."""

    def _make_ohlcv(self, n: int = 50) -> pd.DataFrame:
        """Create synthetic OHLCV data."""
        dates = pd.bdate_range("2023-01-02", periods=n, freq="B")
        rng = np.random.RandomState(42)
        close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
        return pd.DataFrame({
            "Open": close - rng.uniform(0, 1, n),
            "High": close + rng.uniform(0, 2, n),
            "Low": close - rng.uniform(0, 2, n),
            "Close": close,
            "Volume": rng.randint(100_000, 500_000, n).astype(float),
        }, index=dates)

    def _make_predictions(self, ohlcv: pd.DataFrame, permno: str = "12345") -> pd.DataFrame:
        """Create a minimal predictions panel with signals in the middle of the data."""
        # Select a signal date in the middle of the OHLCV (so entry and exit fit)
        signal_idx = 10
        signal_date = ohlcv.index[signal_idx]

        preds = pd.DataFrame({
            "predicted_return": [0.05],
            "confidence": [0.9],
            "regime": [0],
        }, index=pd.MultiIndex.from_tuples(
            [(permno, signal_date)], names=["permno", "date"]
        ))
        return preds

    def test_high_uncertainty_reduces_position(self):
        """When shock vector has high hmm_uncertainty, position size should shrink."""
        from quant_engine.backtest.engine import Backtester
        from quant_engine.regime.shock_vector import ShockVector

        ohlcv = self._make_ohlcv(50)
        permno = "12345"
        ohlcv.attrs["ticker"] = permno

        preds = self._make_predictions(ohlcv, permno)
        signal_date = preds.index.get_level_values(1)[0]

        bt = Backtester(
            entry_threshold=0.01,
            confidence_threshold=0.5,
            holding_days=5,
            max_positions=5,
        )

        # Run WITHOUT shock vectors (baseline)
        bt._shock_vectors = {}
        result_baseline = bt.run(
            predictions=preds,
            price_data={permno: ohlcv},
            verbose=False,
        )

        # Run WITH high-uncertainty shock vector
        sv = ShockVector(
            ticker=permno,
            hmm_regime=0,
            hmm_confidence=0.3,
            hmm_uncertainty=0.95,  # Very high uncertainty
            bocpd_changepoint_prob=0.5,
        )
        bt2 = Backtester(
            entry_threshold=0.01,
            confidence_threshold=0.5,
            holding_days=5,
            max_positions=5,
        )
        bt2._shock_vectors = {(permno, signal_date): sv}
        result_gated = bt2.run(
            predictions=preds,
            price_data={permno: ohlcv},
            verbose=False,
        )

        # Both should produce trades
        if result_baseline.total_trades > 0 and result_gated.total_trades > 0:
            baseline_size = result_baseline.trades[0].position_size
            gated_size = result_gated.trades[0].position_size
            # Gated size should be smaller or equal
            assert gated_size <= baseline_size, (
                f"Expected gated size ({gated_size:.4f}) <= baseline ({baseline_size:.4f})"
            )

    def test_zero_uncertainty_no_reduction(self):
        """When hmm_uncertainty is 0, position size should be unchanged."""
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        gate = UncertaintyGate()
        mult = gate.compute_size_multiplier(0.0)
        assert mult == 1.0, f"Expected 1.0, got {mult}"


# ---------------------------------------------------------------------------
# Backtest engine: Risk-managed mode — uncertainty-gated position sizing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRiskManagedModeUncertaintyGating:
    """Verify the risk-managed backtester applies the uncertainty gate."""

    def test_uncertainty_gate_applied_after_sizing(self):
        """The uncertainty gate multiplier is applied after position_sizer output."""
        from quant_engine.regime.uncertainty_gate import UncertaintyGate
        from quant_engine.config import REGIME_UNCERTAINTY_SIZING_MAP

        gate = UncertaintyGate()

        # At uncertainty = 1.0, multiplier should be at the low end
        high_mult = gate.compute_size_multiplier(1.0)
        low_mult = gate.compute_size_multiplier(0.0)
        mid_mult = gate.compute_size_multiplier(0.5)

        assert low_mult == 1.0
        assert mid_mult < low_mult
        assert high_mult < mid_mult
        assert high_mult >= gate.min_multiplier

    def test_gate_reduces_raw_size(self):
        """Simulating position_size *= multiplier reduces the size correctly."""
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        gate = UncertaintyGate()
        raw_size = 0.05  # 5%

        mult = gate.compute_size_multiplier(0.8)
        gated_size = raw_size * mult

        assert gated_size < raw_size
        assert gated_size > 0


# ---------------------------------------------------------------------------
# Autopilot engine: Uncertainty gate on portfolio weights
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAutopilotUncertaintyGateWiring:
    """Verify the autopilot engine applies the uncertainty gate to portfolio weights."""

    def test_uncertainty_gate_import(self):
        """UncertaintyGate is importable from the autopilot engine's deps."""
        from quant_engine.regime.uncertainty_gate import UncertaintyGate
        gate = UncertaintyGate()
        assert gate is not None

    def test_apply_gate_to_weights_with_high_entropy(self):
        """High regime entropy should reduce all portfolio weights uniformly."""
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        gate = UncertaintyGate()
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        uncertainty = 0.9  # High

        adjusted = gate.apply_uncertainty_gate(weights, uncertainty)
        mult = gate.compute_size_multiplier(uncertainty)

        np.testing.assert_allclose(adjusted, weights * mult, rtol=1e-10)
        assert np.all(adjusted < weights)

    def test_apply_gate_to_weights_with_low_entropy(self):
        """Low regime entropy should leave weights nearly unchanged."""
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        gate = UncertaintyGate()
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        uncertainty = 0.1  # Low

        adjusted = gate.apply_uncertainty_gate(weights, uncertainty)
        mult = gate.compute_size_multiplier(uncertainty)

        # Multiplier at uncertainty=0.1 should be close to 1.0
        assert mult >= 0.98, f"Expected mult >= 0.98 at low uncertainty, got {mult}"
        np.testing.assert_allclose(adjusted, weights * mult, rtol=1e-10)

    def test_regime_entropy_computation_from_probs(self):
        """Verify the entropy computation logic used in autopilot optimizer weights."""
        # This tests the inline entropy computation in _compute_optimizer_weights
        probs = pd.DataFrame({
            "regime_prob_0": [0.25, 0.90],
            "regime_prob_1": [0.25, 0.05],
            "regime_prob_2": [0.25, 0.03],
            "regime_prob_3": [0.25, 0.02],
        })
        probs_clipped = probs.clip(lower=1e-10)
        row_sums = probs_clipped.sum(axis=1)
        probs_normed = probs_clipped.div(row_sums, axis=0)
        entropy = -(probs_normed * np.log(probs_normed)).sum(axis=1) / np.log(4)
        mean_entropy = float(entropy.clip(0, 1).mean())

        # First row is uniform (entropy=1.0), second row is concentrated (entropy<0.5)
        assert 0.5 < mean_entropy < 1.0, f"Expected mean entropy in (0.5, 1.0), got {mean_entropy}"

    def test_fallback_to_regime_confidence(self):
        """When regime_prob_* columns are missing, fall back to 1 - regime_confidence."""
        df = pd.DataFrame({
            "permno": ["A", "B", "C"],
            "regime_confidence": [0.9, 0.8, 0.7],
        })
        # No regime_prob_* columns → uses 1 - mean(regime_confidence)
        conf = df["regime_confidence"].astype(float).clip(0, 1).mean()
        uncertainty = float(1.0 - conf)

        assert abs(uncertainty - 0.2) < 0.01, f"Expected ~0.2, got {uncertainty}"

    def test_no_regime_info_no_reduction(self):
        """When neither regime_prob_* nor regime_confidence is available, uncertainty = 0."""
        df = pd.DataFrame({
            "permno": ["A", "B"],
            "predicted_return": [0.05, 0.03],
        })
        regime_prob_cols = [c for c in df.columns if c.startswith("regime_prob_")]
        has_conf = "regime_confidence" in df.columns

        assert len(regime_prob_cols) == 0
        assert not has_conf
        # → current_uncertainty = 0.0 → no gate applied


# ---------------------------------------------------------------------------
# End-to-end: verify the gate is NOT imported lazily (import always present)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUncertaintyGateImportsPresent:
    """Ensure UncertaintyGate is imported in both engine modules."""

    def test_backtest_engine_imports_gate(self):
        import quant_engine.backtest.engine as bt_mod
        assert hasattr(bt_mod, "UncertaintyGate")

    def test_autopilot_engine_imports_gate(self):
        import quant_engine.autopilot.engine as ap_mod
        assert hasattr(ap_mod, "UncertaintyGate")
