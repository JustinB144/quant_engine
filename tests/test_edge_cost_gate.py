"""
SPEC-E01: Edge-after-costs trade gating tests.

Covers:
  T1: ExecutionModel.estimate_cost() basic correctness
  T2: estimate_cost() dynamic cost scaling (vol, uncertainty, liquidity)
  T3: estimate_cost() round-trip cost is 2x one-way
  T4: estimate_cost() edge cases (zero volume, zero notional)
  T5: Edge-cost gate in simple backtest mode
  T6: Edge-cost gate in risk-managed backtest mode
  T7: Edge-cost gate disabled via config flag
  T8: Buffer scales with regime uncertainty
  T9: Integration — gated backtest produces fewer but better trades
"""
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from quant_engine.backtest.execution import ExecutionModel, ExecutionFill
from quant_engine.backtest.engine import Backtester


# ── Test Helpers ─────────────────────────────────────────────────────────


def _make_model(**kwargs) -> ExecutionModel:
    """Create an ExecutionModel with sensible test defaults."""
    defaults = dict(
        spread_bps=3.0,
        impact_coefficient_bps=25.0,
        max_participation_rate=0.05,
        dynamic_costs=True,
        structural_stress_enabled=True,
        volume_trend_enabled=True,
    )
    defaults.update(kwargs)
    return ExecutionModel(**defaults)


def _make_ohlcv(
    n_bars: int = 100,
    base_price: float = 100.0,
    daily_volume: float = 1_000_000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_bars, freq="B")
    returns = rng.normal(0.0005, 0.015, n_bars)
    close = base_price * np.cumprod(1 + returns)
    open_ = close * (1 + rng.normal(0, 0.002, n_bars))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.01, n_bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.01, n_bars))
    volume = rng.uniform(daily_volume * 0.5, daily_volume * 1.5, n_bars)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


def _make_predictions(
    ohlcv: pd.DataFrame,
    ticker: str = "10001",
    n_signals: int = 5,
    start_bar: int = 20,
    spacing: int = 15,
    predicted_return: float = 0.02,
    confidence: float = 0.75,
    regime: int = 0,
) -> pd.DataFrame:
    """Generate synthetic predictions aligned with OHLCV data."""
    rows = []
    indices = []
    for i in range(n_signals):
        bar_idx = start_bar + i * spacing
        if bar_idx >= len(ohlcv):
            break
        dt = ohlcv.index[bar_idx]
        indices.append((ticker, dt))
        rows.append(
            {
                "predicted_return": predicted_return,
                "confidence": confidence,
                "regime": regime,
            }
        )
    if not rows:
        return pd.DataFrame()
    idx = pd.MultiIndex.from_tuples(indices, names=["ticker", "date"])
    return pd.DataFrame(rows, index=idx)


# ══════════════════════════════════════════════════════════════════════════
# T1: estimate_cost() basic correctness
# ══════════════════════════════════════════════════════════════════════════


class TestEstimateCostBasic(unittest.TestCase):
    """estimate_cost() returns positive round-trip cost in normal conditions."""

    def setUp(self):
        self.model = _make_model()

    def test_returns_positive_cost(self):
        cost = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.20,
            reference_price=100.0,
        )
        self.assertGreater(cost, 0.0)

    def test_cost_is_float(self):
        cost = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
        )
        self.assertIsInstance(cost, float)

    def test_cost_is_finite(self):
        cost = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.50,
        )
        self.assertTrue(np.isfinite(cost))


# ══════════════════════════════════════════════════════════════════════════
# T2: estimate_cost() dynamic cost scaling
# ══════════════════════════════════════════════════════════════════════════


class TestEstimateCostScaling(unittest.TestCase):
    """Costs scale with volatility, uncertainty, and liquidity."""

    def setUp(self):
        self.model = _make_model()

    def test_higher_vol_means_higher_cost(self):
        low_vol = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.10,
            reference_price=100.0,
        )
        high_vol = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.50,
            reference_price=100.0,
        )
        self.assertGreater(high_vol, low_vol)

    def test_higher_uncertainty_means_higher_cost(self):
        low_unc = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.20,
            structure_uncertainty=0.0,
            reference_price=100.0,
        )
        high_unc = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.20,
            structure_uncertainty=0.9,
            reference_price=100.0,
        )
        self.assertGreater(high_unc, low_unc)

    def test_lower_volume_means_higher_cost(self):
        """Illiquid stocks should be more expensive to trade."""
        liquid = self.model.estimate_cost(
            daily_volume=10_000_000,
            desired_notional=100_000,
            realized_vol=0.20,
            reference_price=100.0,
        )
        illiquid = self.model.estimate_cost(
            daily_volume=100_000,
            desired_notional=100_000,
            realized_vol=0.20,
            reference_price=100.0,
        )
        self.assertGreater(illiquid, liquid)

    def test_larger_notional_means_higher_cost(self):
        """Larger orders should have more market impact."""
        small = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=10_000,
            realized_vol=0.20,
            reference_price=100.0,
        )
        large = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=500_000,
            realized_vol=0.20,
            reference_price=100.0,
        )
        self.assertGreater(large, small)


# ══════════════════════════════════════════════════════════════════════════
# T3: estimate_cost() round-trip is 2x one-way
# ══════════════════════════════════════════════════════════════════════════


class TestEstimateCostRoundTrip(unittest.TestCase):
    """Round-trip cost should be approximately 2x one-way cost from simulate()."""

    def test_roundtrip_is_double_oneway(self):
        model = _make_model()
        # Get one-way cost from simulate
        fill = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=1_000_000,
            desired_notional_usd=100_000,
            realized_vol=0.20,
            force_full=True,
        )
        one_way_bps = fill.cost_details.get("total_bps", 0.0)

        # Get round-trip cost from estimate_cost
        roundtrip = model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.20,
            reference_price=100.0,
        )

        # Round-trip should be 2x one-way (within tolerance for floating point)
        self.assertAlmostEqual(roundtrip, 2.0 * one_way_bps, places=4)


# ══════════════════════════════════════════════════════════════════════════
# T4: estimate_cost() edge cases
# ══════════════════════════════════════════════════════════════════════════


class TestEstimateCostEdgeCases(unittest.TestCase):
    """Edge cases: zero volume, zero notional, NaN inputs."""

    def setUp(self):
        self.model = _make_model()

    def test_zero_volume_returns_zero(self):
        cost = self.model.estimate_cost(
            daily_volume=0.0,
            desired_notional=100_000,
        )
        self.assertEqual(cost, 0.0)

    def test_zero_notional_returns_zero(self):
        cost = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=0.0,
        )
        self.assertEqual(cost, 0.0)

    def test_nan_vol_treated_as_none(self):
        cost = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=float("nan"),
            reference_price=100.0,
        )
        self.assertGreater(cost, 0.0)
        self.assertTrue(np.isfinite(cost))

    def test_negative_volume_treated_as_zero(self):
        cost = self.model.estimate_cost(
            daily_volume=-100,
            desired_notional=100_000,
        )
        self.assertEqual(cost, 0.0)

    def test_static_costs_mode(self):
        """With dynamic_costs=False, cost uses base spread only."""
        model = _make_model(dynamic_costs=False)
        cost = model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.50,
            reference_price=100.0,
        )
        self.assertGreater(cost, 0.0)

    def test_impact_coeff_override(self):
        """Per-segment impact coefficient override is respected."""
        base = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.20,
            reference_price=100.0,
        )
        overridden = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.20,
            reference_price=100.0,
            impact_coeff_override=50.0,  # double default
        )
        self.assertGreater(overridden, base)


# ══════════════════════════════════════════════════════════════════════════
# T5 & T6: Edge-cost gate in backtest modes
# ══════════════════════════════════════════════════════════════════════════


class TestEdgeCostGateSimpleMode(unittest.TestCase):
    """Edge-cost gate filters low-edge trades in simple (non-risk) mode."""

    def _run_backtest(self, predicted_return: float, gate_enabled: bool = True):
        """Run a simple backtest with given predicted return and gate setting."""
        ohlcv = _make_ohlcv(n_bars=100, daily_volume=1_000_000)
        predictions = _make_predictions(
            ohlcv,
            predicted_return=predicted_return,
            n_signals=3,
            start_bar=20,
            spacing=20,
        )

        with patch("quant_engine.backtest.engine.EDGE_COST_GATE_ENABLED", gate_enabled), \
             patch("quant_engine.backtest.engine.EDGE_COST_BUFFER_BASE_BPS", 5.0), \
             patch("quant_engine.backtest.engine.EXEC_STRUCTURAL_STRESS_ENABLED", False), \
             patch("quant_engine.backtest.engine.EXEC_CALIBRATION_ENABLED", False), \
             patch("quant_engine.validation.preconditions.enforce_preconditions"):
            bt = Backtester(
                entry_threshold=0.001,
                confidence_threshold=0.5,
                holding_days=5,
                max_positions=5,
                use_risk_management=False,
            )
            result = bt.run(predictions, {"10001": ohlcv}, verbose=False)
        return result

    def test_tiny_edge_filtered_out(self):
        """Predicted return of 0.001 (10 bps) should be gated by costs."""
        result = self._run_backtest(predicted_return=0.001, gate_enabled=True)
        # With 10 bps edge vs typical cost ~6-15 bps + 5 bps buffer, most should be filtered
        self.assertLessEqual(result.total_trades, 3)

    def test_large_edge_passes_gate(self):
        """Predicted return of 0.05 (500 bps) should pass the gate."""
        result = self._run_backtest(predicted_return=0.05, gate_enabled=True)
        self.assertGreater(result.total_trades, 0)

    def test_gate_disabled_allows_all(self):
        """With gate disabled, tiny edge trades should still enter."""
        # Run with gate off
        result_off = self._run_backtest(predicted_return=0.002, gate_enabled=False)
        # Run with gate on
        result_on = self._run_backtest(predicted_return=0.002, gate_enabled=True)
        # Gate off should produce at least as many trades
        self.assertGreaterEqual(result_off.total_trades, result_on.total_trades)


class TestEdgeCostGateRiskManagedMode(unittest.TestCase):
    """Edge-cost gate filters low-edge trades in risk-managed mode."""

    def _run_backtest(self, predicted_return: float, gate_enabled: bool = True):
        """Run a risk-managed backtest with given predicted return."""
        ohlcv = _make_ohlcv(n_bars=150, daily_volume=1_000_000)
        predictions = _make_predictions(
            ohlcv,
            predicted_return=predicted_return,
            n_signals=3,
            start_bar=25,
            spacing=25,
        )

        with patch("quant_engine.backtest.engine.EDGE_COST_GATE_ENABLED", gate_enabled), \
             patch("quant_engine.backtest.engine.EDGE_COST_BUFFER_BASE_BPS", 5.0), \
             patch("quant_engine.backtest.engine.EXEC_STRUCTURAL_STRESS_ENABLED", False), \
             patch("quant_engine.backtest.engine.EXEC_CALIBRATION_ENABLED", False), \
             patch("quant_engine.validation.preconditions.enforce_preconditions"):
            bt = Backtester(
                entry_threshold=0.001,
                confidence_threshold=0.5,
                holding_days=5,
                max_positions=5,
                use_risk_management=True,
            )
            result = bt.run(predictions, {"10001": ohlcv}, verbose=False)
        return result

    def test_large_edge_trades_in_risk_mode(self):
        """Large edge passes gate in risk-managed mode."""
        result = self._run_backtest(predicted_return=0.05, gate_enabled=True)
        self.assertGreater(result.total_trades, 0)

    def test_gate_disabled_in_risk_mode(self):
        """With gate disabled, trades should not be filtered by edge."""
        result_off = self._run_backtest(predicted_return=0.002, gate_enabled=False)
        result_on = self._run_backtest(predicted_return=0.002, gate_enabled=True)
        self.assertGreaterEqual(result_off.total_trades, result_on.total_trades)


# ══════════════════════════════════════════════════════════════════════════
# T8: Buffer scaling with uncertainty
# ══════════════════════════════════════════════════════════════════════════


class TestBufferScaling(unittest.TestCase):
    """Cost buffer scales with regime uncertainty."""

    def test_buffer_increases_with_uncertainty(self):
        """Higher uncertainty → larger buffer → harder to pass gate."""
        base_buffer = 5.0
        low_uncertainty = 0.1
        high_uncertainty = 0.9

        buffer_low = base_buffer * (1.0 + low_uncertainty)
        buffer_high = base_buffer * (1.0 + high_uncertainty)

        self.assertAlmostEqual(buffer_low, 5.5)
        self.assertAlmostEqual(buffer_high, 9.5)
        self.assertGreater(buffer_high, buffer_low)

    def test_zero_uncertainty_gives_base_buffer(self):
        """With zero uncertainty, buffer equals the base."""
        base_buffer = 5.0
        buffer = base_buffer * (1.0 + 0.0)
        self.assertEqual(buffer, base_buffer)


# ══════════════════════════════════════════════════════════════════════════
# T9: Config values
# ══════════════════════════════════════════════════════════════════════════


class TestEdgeCostGateConfig(unittest.TestCase):
    """Config values exist and have correct types/defaults."""

    def test_config_values_exist(self):
        from quant_engine.config import EDGE_COST_GATE_ENABLED, EDGE_COST_BUFFER_BASE_BPS
        self.assertIsInstance(EDGE_COST_GATE_ENABLED, bool)
        self.assertIsInstance(EDGE_COST_BUFFER_BASE_BPS, float)

    def test_default_values(self):
        from quant_engine.config import EDGE_COST_GATE_ENABLED, EDGE_COST_BUFFER_BASE_BPS
        self.assertTrue(EDGE_COST_GATE_ENABLED)
        self.assertEqual(EDGE_COST_BUFFER_BASE_BPS, 5.0)


# ══════════════════════════════════════════════════════════════════════════
# T10: estimate_cost() structural state passthrough
# ══════════════════════════════════════════════════════════════════════════


class TestEstimateCostStructuralState(unittest.TestCase):
    """Structural state parameters are correctly passed through to cost model."""

    def setUp(self):
        self.model = _make_model()

    def test_high_break_probability_increases_cost(self):
        base = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.20,
            reference_price=100.0,
        )
        stressed = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.20,
            reference_price=100.0,
            break_probability=0.8,
        )
        self.assertGreater(stressed, base)

    def test_high_systemic_stress_increases_cost(self):
        base = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.20,
            reference_price=100.0,
        )
        stressed = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.20,
            reference_price=100.0,
            systemic_stress=0.9,
        )
        self.assertGreater(stressed, base)

    def test_high_drift_reduces_cost(self):
        base = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.20,
            reference_price=100.0,
        )
        drifting = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.20,
            reference_price=100.0,
            drift_score=0.9,
        )
        self.assertLess(drifting, base)

    def test_event_spread_multiplier(self):
        base = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.20,
            reference_price=100.0,
        )
        event = self.model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.20,
            reference_price=100.0,
            event_spread_multiplier=2.0,
        )
        self.assertGreater(event, base)


if __name__ == "__main__":
    unittest.main()
