"""
Comprehensive tests for Spec 06: Execution Layer Improvements.

Covers:
  - Structural state cost multipliers (T1)
  - ADV tracking and volume trend (T2)
  - Entry/exit urgency differentiation (T3)
  - Cost calibration per market cap segment (T4)
  - No-trade gate (T3)
  - Backward compatibility
  - Integration tests
"""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from quant_engine.backtest.execution import ExecutionModel, ExecutionFill, calibrate_cost_model
from quant_engine.backtest.adv_tracker import ADVTracker
from quant_engine.backtest.cost_calibrator import CostCalibrator


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


def _calm_fill(model: ExecutionModel, **overrides) -> ExecutionFill:
    """Simulate a calm-market buy."""
    params = dict(
        side="buy",
        reference_price=100.0,
        daily_volume=5_000_000.0,
        desired_notional_usd=100_000.0,
        force_full=True,
        realized_vol=0.12,
        overnight_gap=0.001,
        intraday_range=0.01,
    )
    params.update(overrides)
    return model.simulate(**params)


# ══════════════════════════════════════════════════════════════════════════
# T1: Structural State Cost Multipliers
# ══════════════════════════════════════════════════════════════════════════


class TestBreakProbabilityMultiplier(unittest.TestCase):
    """Test break probability → cost multiplier interpolation."""

    def setUp(self):
        self.model = _make_model()

    def test_none_returns_unity(self):
        self.assertEqual(
            self.model._compute_break_probability_mult(None), 1.0
        )

    def test_nan_returns_unity(self):
        self.assertEqual(
            self.model._compute_break_probability_mult(float("nan")), 1.0
        )

    def test_zero_returns_low(self):
        mult = self.model._compute_break_probability_mult(0.0)
        self.assertAlmostEqual(mult, 1.0)

    def test_below_low_threshold(self):
        mult = self.model._compute_break_probability_mult(0.03)
        self.assertAlmostEqual(mult, 1.0)

    def test_at_medium_boundary(self):
        mult = self.model._compute_break_probability_mult(0.05)
        self.assertAlmostEqual(mult, 1.0)

    def test_interpolation_low_to_medium(self):
        mult = self.model._compute_break_probability_mult(0.10)
        # Midpoint between 0.05 and 0.15: should be midpoint of 1.0 and 1.3
        self.assertAlmostEqual(mult, 1.15, places=2)

    def test_at_high_boundary(self):
        mult = self.model._compute_break_probability_mult(0.15)
        self.assertAlmostEqual(mult, 1.3)

    def test_interpolation_medium_to_high(self):
        mult = self.model._compute_break_probability_mult(0.325)
        # Midpoint between 0.15 and 0.50: should be midpoint of 1.3 and 2.0
        self.assertAlmostEqual(mult, 1.65, places=2)

    def test_above_high_threshold(self):
        mult = self.model._compute_break_probability_mult(0.80)
        self.assertAlmostEqual(mult, 2.0)

    def test_clipped_to_one(self):
        mult = self.model._compute_break_probability_mult(1.5)
        self.assertAlmostEqual(mult, 2.0)

    def test_monotonic_increasing(self):
        """Multiplier should increase monotonically with break_prob."""
        probs = [0.0, 0.03, 0.05, 0.10, 0.15, 0.30, 0.50, 0.80, 1.0]
        mults = [self.model._compute_break_probability_mult(p) for p in probs]
        for i in range(1, len(mults)):
            self.assertGreaterEqual(mults[i], mults[i - 1])


class TestStructureUncertaintyMultiplier(unittest.TestCase):

    def setUp(self):
        self.model = _make_model()

    def test_none_returns_unity(self):
        self.assertEqual(
            self.model._compute_structure_uncertainty_mult(None), 1.0
        )

    def test_zero_returns_unity(self):
        self.assertAlmostEqual(
            self.model._compute_structure_uncertainty_mult(0.0), 1.0
        )

    def test_half_uncertainty(self):
        # 0.5 * 0.50 = 0.25 → 1.25
        self.assertAlmostEqual(
            self.model._compute_structure_uncertainty_mult(0.5), 1.25
        )

    def test_full_uncertainty(self):
        # 1.0 * 0.50 = 0.50 → 1.50
        self.assertAlmostEqual(
            self.model._compute_structure_uncertainty_mult(1.0), 1.50
        )

    def test_clipped_above_one(self):
        mult = self.model._compute_structure_uncertainty_mult(2.0)
        self.assertAlmostEqual(mult, 1.50)


class TestDriftScoreMultiplier(unittest.TestCase):

    def setUp(self):
        self.model = _make_model()

    def test_none_returns_unity(self):
        self.assertEqual(self.model._compute_drift_score_mult(None), 1.0)

    def test_zero_returns_unity(self):
        self.assertAlmostEqual(self.model._compute_drift_score_mult(0.0), 1.0)

    def test_half_drift(self):
        # 1.0 - 0.5 * 0.20 = 0.90
        self.assertAlmostEqual(
            self.model._compute_drift_score_mult(0.5), 0.90
        )

    def test_full_drift(self):
        # 1.0 - 1.0 * 0.20 = 0.80
        self.assertAlmostEqual(
            self.model._compute_drift_score_mult(1.0), 0.80
        )

    def test_floor_at_0_70(self):
        # With very high reduction factor, should floor at 0.70
        model = _make_model(drift_score_cost_reduction=0.50)
        mult = model._compute_drift_score_mult(1.0)
        self.assertAlmostEqual(mult, 0.70)

    def test_monotonic_decreasing(self):
        """Drift multiplier should decrease with increasing drift score."""
        scores = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        mults = [self.model._compute_drift_score_mult(s) for s in scores]
        for i in range(1, len(mults)):
            self.assertLessEqual(mults[i], mults[i - 1])


class TestSystemicStressMultiplier(unittest.TestCase):

    def setUp(self):
        self.model = _make_model()

    def test_none_returns_unity(self):
        self.assertEqual(
            self.model._compute_systemic_stress_mult(None), 1.0
        )

    def test_zero_returns_unity(self):
        self.assertAlmostEqual(
            self.model._compute_systemic_stress_mult(0.0), 1.0
        )

    def test_full_stress(self):
        # 1.0 * 0.30 = 0.30 → 1.30
        self.assertAlmostEqual(
            self.model._compute_systemic_stress_mult(1.0), 1.30
        )


class TestCompositeStructuralMultiplier(unittest.TestCase):
    """Test composite structural multiplier computation."""

    def setUp(self):
        self.model = _make_model()

    def test_all_none_returns_unity(self):
        mult, details = self.model._compute_structural_multiplier()
        self.assertAlmostEqual(mult, 1.0)

    def test_disabled_returns_unity(self):
        model = _make_model(structural_stress_enabled=False)
        mult, details = model._compute_structural_multiplier(
            break_probability=0.5,
            structure_uncertainty=0.8,
            systemic_stress=0.9,
        )
        self.assertAlmostEqual(mult, 1.0)
        self.assertEqual(details, {})

    def test_multiplicative_combination(self):
        mult, details = self.model._compute_structural_multiplier(
            break_probability=0.10,
            structure_uncertainty=0.5,
            drift_score=0.0,
            systemic_stress=0.5,
        )
        # break: ~1.15, uncertainty: 1.25, drift: 1.0, stress: 1.15
        # Composite: 1.15 * 1.25 * 1.0 * 1.15 ≈ 1.653
        self.assertGreater(mult, 1.5)
        self.assertLess(mult, 2.0)
        self.assertIn("break_prob_mult", details)
        self.assertIn("composite_structural_mult", details)

    def test_clipped_at_three(self):
        """Extreme inputs should be clipped to 3.0 maximum."""
        mult, _ = self.model._compute_structural_multiplier(
            break_probability=1.0,
            structure_uncertainty=1.0,
            drift_score=0.0,
            systemic_stress=1.0,
        )
        self.assertLessEqual(mult, 3.0)

    def test_clipped_at_floor(self):
        """With strong drift discount, floor is 0.70."""
        mult, _ = self.model._compute_structural_multiplier(
            break_probability=0.0,
            structure_uncertainty=0.0,
            drift_score=1.0,
            systemic_stress=0.0,
        )
        self.assertGreaterEqual(mult, 0.70)


class TestStructuralCostsInSimulate(unittest.TestCase):
    """Test that structural inputs affect fill prices in simulate()."""

    def setUp(self):
        self.model = _make_model()

    def test_high_break_prob_increases_cost(self):
        calm = _calm_fill(self.model, break_probability=0.0)
        stressed = _calm_fill(self.model, break_probability=0.5)
        # Buy side: higher cost → higher fill price
        self.assertGreater(stressed.fill_price, calm.fill_price)

    def test_high_uncertainty_increases_cost(self):
        calm = _calm_fill(self.model, structure_uncertainty=0.0)
        stressed = _calm_fill(self.model, structure_uncertainty=0.8)
        self.assertGreater(stressed.fill_price, calm.fill_price)

    def test_high_drift_decreases_cost(self):
        no_drift = _calm_fill(self.model, drift_score=0.0)
        strong_drift = _calm_fill(self.model, drift_score=0.8)
        self.assertLess(strong_drift.fill_price, no_drift.fill_price)

    def test_high_systemic_stress_increases_cost(self):
        calm = _calm_fill(self.model, systemic_stress=0.0)
        stressed = _calm_fill(self.model, systemic_stress=0.8)
        self.assertGreater(stressed.fill_price, calm.fill_price)

    def test_structural_mult_in_fill(self):
        fill = _calm_fill(
            self.model,
            break_probability=0.3,
            structure_uncertainty=0.5,
        )
        self.assertGreater(fill.structural_mult, 1.0)

    def test_sell_side_structural_cost(self):
        calm = self.model.simulate(
            side="sell", reference_price=100.0, daily_volume=5_000_000.0,
            desired_notional_usd=100_000.0, force_full=True,
            realized_vol=0.12, systemic_stress=0.0,
        )
        stressed = self.model.simulate(
            side="sell", reference_price=100.0, daily_volume=5_000_000.0,
            desired_notional_usd=100_000.0, force_full=True,
            realized_vol=0.12, systemic_stress=0.8,
        )
        # Sell side: higher cost → lower fill price
        self.assertLess(stressed.fill_price, calm.fill_price)


# ══════════════════════════════════════════════════════════════════════════
# T2: ADV Tracking and Volume Trend
# ══════════════════════════════════════════════════════════════════════════


class TestADVTracker(unittest.TestCase):
    """Test ADV computation with EMA smoothing."""

    def setUp(self):
        self.tracker = ADVTracker(lookback_days=20, ema_span=10)

    def test_initial_adv(self):
        self.tracker.update("AAPL", 1_000_000.0)
        self.assertAlmostEqual(self.tracker.get_adv("AAPL"), 1_000_000.0)

    def test_ema_smoothing(self):
        # Feed 10 days of constant volume then a spike
        for _ in range(10):
            self.tracker.update("AAPL", 1_000_000.0)
        adv_before = self.tracker.get_adv("AAPL")
        self.tracker.update("AAPL", 5_000_000.0)  # volume spike
        adv_after = self.tracker.get_adv("AAPL")
        # EMA should respond but not fully jump to 5M
        self.assertGreater(adv_after, adv_before)
        self.assertLess(adv_after, 5_000_000.0)

    def test_volume_trend_above_average(self):
        for _ in range(10):
            self.tracker.update("MSFT", 1_000_000.0)
        # Current volume 2x average
        self.tracker.update("MSFT", 2_000_000.0)
        trend = self.tracker.get_volume_trend("MSFT")
        self.assertGreater(trend, 1.0)

    def test_volume_trend_below_average(self):
        for _ in range(10):
            self.tracker.update("MSFT", 1_000_000.0)
        self.tracker.update("MSFT", 500_000.0)
        trend = self.tracker.get_volume_trend("MSFT")
        self.assertLess(trend, 1.0)

    def test_volume_trend_unknown_symbol(self):
        self.assertAlmostEqual(self.tracker.get_volume_trend("UNKNOWN"), 1.0)

    def test_adjust_participation_limit_high_volume(self):
        for _ in range(10):
            self.tracker.update("AAPL", 1_000_000.0)
        self.tracker.update("AAPL", 2_000_000.0)
        adjusted = self.tracker.adjust_participation_limit("AAPL", 0.02)
        self.assertGreater(adjusted, 0.02)
        self.assertLessEqual(adjusted, 0.04)

    def test_adjust_participation_limit_low_volume(self):
        for _ in range(10):
            self.tracker.update("AAPL", 1_000_000.0)
        self.tracker.update("AAPL", 400_000.0)
        adjusted = self.tracker.adjust_participation_limit("AAPL", 0.02)
        self.assertLess(adjusted, 0.02)
        self.assertGreaterEqual(adjusted, 0.01)

    def test_volume_cost_adjustment_high_volume(self):
        for _ in range(10):
            self.tracker.update("AAPL", 1_000_000.0)
        self.tracker.update("AAPL", 2_000_000.0)
        cost_adj = self.tracker.get_volume_cost_adjustment("AAPL")
        self.assertLess(cost_adj, 1.0)  # discount for high volume

    def test_volume_cost_adjustment_low_volume(self):
        for _ in range(10):
            self.tracker.update("AAPL", 1_000_000.0)
        self.tracker.update("AAPL", 400_000.0)
        cost_adj = self.tracker.get_volume_cost_adjustment("AAPL")
        self.assertGreater(cost_adj, 1.0)  # penalty for low volume

    def test_simple_adv(self):
        for v in [1_000_000.0, 2_000_000.0, 3_000_000.0]:
            self.tracker.update("TEST", v)
        self.assertAlmostEqual(
            self.tracker.get_simple_adv("TEST"), 2_000_000.0
        )

    def test_update_from_series(self):
        volumes = [1e6, 1.1e6, 0.9e6, 1.2e6, 1e6]
        self.tracker.update_from_series("BULK", volumes)
        self.assertGreater(self.tracker.get_adv("BULK"), 0)

    def test_zero_volume_ignored(self):
        self.tracker.update("ZERO", 0.0)
        self.assertAlmostEqual(self.tracker.get_adv("ZERO"), 0.0)

    def test_stats(self):
        self.tracker.update("AAPL", 1_000_000.0)
        stats = self.tracker.get_stats("AAPL")
        self.assertIn("adv_ema", stats)
        self.assertIn("volume_trend", stats)
        self.assertIn("cost_adj", stats)


class TestVolumeTrendInSimulate(unittest.TestCase):
    """Test that volume trend affects participation limits."""

    def test_high_volume_trend_allows_larger_fills(self):
        model = _make_model(max_participation_rate=0.02)
        # With volume_trend=2.0, max participation doubles
        fill_normal = model.simulate(
            side="buy", reference_price=100.0, daily_volume=1_000_000.0,
            desired_notional_usd=5_000_000.0,
            volume_trend=1.0,
        )
        fill_high_vol = model.simulate(
            side="buy", reference_price=100.0, daily_volume=1_000_000.0,
            desired_notional_usd=5_000_000.0,
            volume_trend=2.0,
        )
        self.assertGreaterEqual(
            fill_high_vol.fill_ratio, fill_normal.fill_ratio
        )


# ══════════════════════════════════════════════════════════════════════════
# T3: Urgency Differentiation and No-Trade Gate
# ══════════════════════════════════════════════════════════════════════════


class TestNoTradeGate(unittest.TestCase):
    """Test no-trade gate during extreme stress."""

    def test_entry_blocked_during_extreme_stress(self):
        model = _make_model(no_trade_stress_threshold=0.95)
        fill = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=1_000_000.0,
            desired_notional_usd=100_000.0,
            urgency_type="entry",
            systemic_stress=0.98,
        )
        self.assertTrue(fill.no_trade_blocked)
        self.assertAlmostEqual(fill.fill_ratio, 0.0)

    def test_entry_allowed_below_threshold(self):
        model = _make_model(no_trade_stress_threshold=0.95)
        fill = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=5_000_000.0,
            desired_notional_usd=100_000.0,
            urgency_type="entry",
            systemic_stress=0.80,
        )
        self.assertFalse(fill.no_trade_blocked)
        self.assertGreater(fill.fill_ratio, 0.0)

    def test_exit_not_blocked_during_extreme_stress(self):
        model = _make_model(no_trade_stress_threshold=0.95)
        fill = model.simulate(
            side="sell",
            reference_price=100.0,
            daily_volume=5_000_000.0,
            desired_notional_usd=100_000.0,
            force_full=True,
            urgency_type="exit",
            systemic_stress=0.98,
        )
        # Exits should NOT be blocked
        self.assertFalse(fill.no_trade_blocked)
        self.assertGreater(fill.fill_ratio, 0.0)

    def test_no_urgency_type_no_gate(self):
        model = _make_model(no_trade_stress_threshold=0.95)
        fill = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=5_000_000.0,
            desired_notional_usd=100_000.0,
            systemic_stress=0.98,
        )
        # Without urgency_type="entry", gate doesn't trigger
        self.assertFalse(fill.no_trade_blocked)


class TestUrgencyDifferentiation(unittest.TestCase):
    """Test urgency-based cost acceptance limits."""

    def test_urgency_type_recorded(self):
        model = _make_model()
        fill = _calm_fill(model, urgency_type="entry")
        self.assertEqual(fill.urgency_type, "entry")

        fill2 = _calm_fill(model, urgency_type="exit")
        self.assertEqual(fill2.urgency_type, "exit")

    def test_exit_has_higher_cost_tolerance(self):
        """Exits should tolerate higher costs than entries."""
        model = _make_model(
            exit_urgency_cost_limit_mult=2.0,
            entry_urgency_cost_limit_mult=1.0,
        )
        # These are now properties of the model that affect behavior
        self.assertGreater(
            model.exit_urgency_cost_limit_mult,
            model.entry_urgency_cost_limit_mult,
        )


# ══════════════════════════════════════════════════════════════════════════
# T4: Cost Calibration Per Market Cap Segment
# ══════════════════════════════════════════════════════════════════════════


class TestCostCalibratorSegmentation(unittest.TestCase):

    def setUp(self):
        self.calibrator = CostCalibrator(model_dir=None)

    def test_micro_cap_classification(self):
        self.assertEqual(
            self.calibrator.get_marketcap_segment(100e6), "micro"
        )

    def test_small_cap_classification(self):
        self.assertEqual(
            self.calibrator.get_marketcap_segment(500e6), "small"
        )

    def test_mid_cap_classification(self):
        self.assertEqual(
            self.calibrator.get_marketcap_segment(5e9), "mid"
        )

    def test_large_cap_classification(self):
        self.assertEqual(
            self.calibrator.get_marketcap_segment(50e9), "large"
        )

    def test_invalid_marketcap_defaults_to_mid(self):
        self.assertEqual(
            self.calibrator.get_marketcap_segment(-100), "mid"
        )
        self.assertEqual(
            self.calibrator.get_marketcap_segment(float("nan")), "mid"
        )


class TestCostCalibratorCoefficients(unittest.TestCase):

    def setUp(self):
        self.calibrator = CostCalibrator(model_dir=None)

    def test_default_coefficients(self):
        coeffs = self.calibrator.coefficients
        self.assertAlmostEqual(coeffs["micro"], 40.0)
        self.assertAlmostEqual(coeffs["small"], 30.0)
        self.assertAlmostEqual(coeffs["mid"], 20.0)
        self.assertAlmostEqual(coeffs["large"], 15.0)

    def test_get_impact_coeff_by_marketcap(self):
        self.assertAlmostEqual(
            self.calibrator.get_impact_coeff(100e6), 40.0
        )
        self.assertAlmostEqual(
            self.calibrator.get_impact_coeff(50e9), 15.0
        )


class TestCostCalibratorCalibration(unittest.TestCase):

    def setUp(self):
        self.calibrator = CostCalibrator(
            model_dir=None,
            min_total_trades=10,
            min_segment_trades=5,
            smoothing=0.30,
        )

    def test_insufficient_trades_returns_empty(self):
        for i in range(5):
            self.calibrator.record_trade(
                f"SYM_{i}", 50e9, 0.01, 5.0,
            )
        result = self.calibrator.calibrate()
        self.assertEqual(result, {})

    def test_calibration_updates_coefficients(self):
        # Record 15 large-cap trades with known realized costs
        for i in range(15):
            participation = 0.01
            # Realized cost = spread (3/2=1.5 bps) + impact (coeff * sqrt(0.01))
            # If coeff=10, impact = 10*0.1 = 1.0, total = 2.5 bps
            self.calibrator.record_trade(
                f"AAPL_{i}",
                market_cap=50e9,  # large
                participation_rate=participation,
                realized_cost_bps=2.5,
            )
        result = self.calibrator.calibrate()
        self.assertIn("large", result)
        self.assertEqual(result["large"]["trades_count"], 15)
        # New coeff should be lower than default 15.0 since realized costs are low
        self.assertLess(
            result["large"]["smoothed_coeff"],
            self.calibrator._default["large"],
        )

    def test_smoothing_prevents_wild_swings(self):
        old_coeff = self.calibrator._coefficients["large"]
        # Record trades with moderately high costs (not extreme, to stay in
        # the [5, 100] clip range)
        for i in range(15):
            self.calibrator.record_trade(
                f"SYM_{i}", 50e9, 0.01, 8.0,
            )
        result = self.calibrator.calibrate()
        new_coeff = self.calibrator._coefficients["large"]
        # New coeff should move toward observed but not jump fully
        self.assertGreater(new_coeff, old_coeff)
        # With 30% smoothing, result should match EMA formula (before clip)
        observed = result["large"]["observed_coeff"]
        expected_smoothed = np.clip(
            0.30 * observed + 0.70 * old_coeff, 5.0, 100.0,
        )
        self.assertAlmostEqual(new_coeff, expected_smoothed, places=1)

    def test_reset_history(self):
        self.calibrator.record_trade("SYM", 50e9, 0.01, 5.0)
        self.calibrator.reset_history()
        self.assertEqual(self.calibrator._total_trades, 0)

    def test_record_trade_validation(self):
        # Invalid participation should be ignored
        self.calibrator.record_trade("SYM", 50e9, 0.0, 5.0)
        self.assertEqual(self.calibrator._total_trades, 0)

        # NaN cost should be ignored
        self.calibrator.record_trade("SYM", 50e9, 0.01, float("nan"))
        self.assertEqual(self.calibrator._total_trades, 0)


class TestCostCalibratorInExecution(unittest.TestCase):
    """Test impact_coeff_override parameter in ExecutionModel."""

    def test_override_changes_impact(self):
        model = _make_model(impact_coefficient_bps=25.0)
        fill_default = _calm_fill(model)
        fill_override = _calm_fill(model, impact_coeff_override=5.0)
        # Lower impact coefficient → lower fill price for buy
        self.assertLess(fill_override.fill_price, fill_default.fill_price)


# ══════════════════════════════════════════════════════════════════════════
# Backward Compatibility
# ══════════════════════════════════════════════════════════════════════════


class TestBackwardCompatibility(unittest.TestCase):
    """Ensure existing behavior is preserved when new features are disabled."""

    def test_no_structural_inputs_same_behavior(self):
        model = _make_model(structural_stress_enabled=False)
        fill = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=5_000_000.0,
            desired_notional_usd=100_000.0,
            force_full=True,
            realized_vol=0.12,
            overnight_gap=0.001,
            intraday_range=0.01,
        )
        self.assertAlmostEqual(fill.structural_mult, 1.0)
        self.assertGreater(fill.fill_price, 100.0)

    def test_original_simulate_signature(self):
        """Old-style calls without new params should still work."""
        model = _make_model()
        fill = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=5_000_000.0,
            desired_notional_usd=100_000.0,
            force_full=True,
            realized_vol=0.15,
        )
        self.assertGreater(fill.fill_ratio, 0.0)
        self.assertGreater(fill.fill_price, 0.0)

    def test_original_fill_fields_present(self):
        model = _make_model()
        fill = _calm_fill(model)
        # All original ExecutionFill fields still present
        self.assertIsInstance(fill.fill_price, float)
        self.assertIsInstance(fill.fill_ratio, float)
        self.assertIsInstance(fill.participation_rate, float)
        self.assertIsInstance(fill.impact_bps, float)
        self.assertIsInstance(fill.spread_bps, float)
        self.assertIsInstance(fill.event_spread_multiplier_applied, float)

    def test_zero_desired_returns_zero_fill(self):
        model = _make_model()
        fill = model.simulate(
            side="buy", reference_price=100.0,
            daily_volume=5_000_000.0, desired_notional_usd=0.0,
        )
        self.assertAlmostEqual(fill.fill_ratio, 0.0)

    def test_side_validation(self):
        model = _make_model()
        with self.assertRaises(ValueError):
            model.simulate(
                side="invalid", reference_price=100.0,
                daily_volume=1e6, desired_notional_usd=1e5,
            )


class TestExistingDynamicCosts(unittest.TestCase):
    """Regression: ensure existing dynamic cost behavior still works."""

    def test_dynamic_costs_increase_under_stress(self):
        model = _make_model(structural_stress_enabled=False)
        calm = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=5_000_000.0,
            desired_notional_usd=100_000.0,
            force_full=True,
            realized_vol=0.12,
            overnight_gap=0.001,
            intraday_range=0.01,
        )
        stressed = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=80_000.0,
            desired_notional_usd=100_000.0,
            force_full=True,
            realized_vol=0.55,
            overnight_gap=0.035,
            intraday_range=0.08,
        )
        self.assertGreater(stressed.spread_bps, calm.spread_bps)
        self.assertGreater(stressed.impact_bps, calm.impact_bps)
        self.assertGreater(stressed.fill_price, calm.fill_price)


# ══════════════════════════════════════════════════════════════════════════
# Integration Tests
# ══════════════════════════════════════════════════════════════════════════


class TestIntegrationExecution(unittest.TestCase):
    """End-to-end execution pipeline with all components."""

    def test_full_execution_pipeline(self):
        """Test complete execution with structural state, volume trend,
        urgency, and market cap override."""
        model = _make_model()
        tracker = ADVTracker(lookback_days=20, ema_span=10)
        calibrator = CostCalibrator(model_dir=None)

        # Warm up ADV tracker
        for _ in range(10):
            tracker.update("AAPL", 50_000_000.0)
        tracker.update("AAPL", 40_000_000.0)  # slightly below average

        vol_trend = tracker.get_volume_trend("AAPL")
        impact_coeff = calibrator.get_impact_coeff(200e9)  # large cap

        fill = model.simulate(
            side="buy",
            reference_price=175.0,
            daily_volume=50_000_000.0,
            desired_notional_usd=500_000.0,
            realized_vol=0.18,
            overnight_gap=0.003,
            intraday_range=0.015,
            urgency_type="entry",
            break_probability=0.08,
            structure_uncertainty=0.3,
            drift_score=0.6,
            systemic_stress=0.4,
            volume_trend=vol_trend,
            impact_coeff_override=impact_coeff,
        )

        self.assertGreater(fill.fill_price, 175.0)
        self.assertGreater(fill.fill_ratio, 0.0)
        self.assertGreater(fill.structural_mult, 0.0)
        self.assertEqual(fill.urgency_type, "entry")
        self.assertFalse(fill.no_trade_blocked)

    def test_exit_during_stress(self):
        """Exits should complete even during high stress."""
        model = _make_model()
        fill = model.simulate(
            side="sell",
            reference_price=100.0,
            daily_volume=2_000_000.0,
            desired_notional_usd=200_000.0,
            force_full=True,
            realized_vol=0.45,
            urgency_type="exit",
            break_probability=0.4,
            structure_uncertainty=0.7,
            systemic_stress=0.9,
        )
        self.assertGreater(fill.fill_ratio, 0.0)
        self.assertLess(fill.fill_price, 100.0)

    def test_calibrate_cost_model_legacy(self):
        """Legacy calibrate_cost_model function still works."""
        result = calibrate_cost_model([], pd.DataFrame())
        self.assertIn("spread_bps", result)
        self.assertIn("impact_coeff", result)
        self.assertIn("fill_ratio", result)

    def test_cost_details_populated(self):
        """Verify cost_details dict is populated with useful info."""
        model = _make_model()
        fill = _calm_fill(
            model,
            break_probability=0.1,
            structure_uncertainty=0.3,
        )
        details = fill.cost_details
        self.assertIn("vol_component", details)
        self.assertIn("liquidity_scalar", details)
        self.assertIn("break_prob_mult", details)
        self.assertIn("total_bps", details)
        self.assertGreater(details["total_bps"], 0.0)


# ══════════════════════════════════════════════════════════════════════════
# SPEC-W02: Cost Calibrator Wiring into Backtest Engine
# ══════════════════════════════════════════════════════════════════════════


class TestCostCalibratorWiring(unittest.TestCase):
    """Test that CostCalibrator is wired into backtest execution."""

    def test_different_segments_different_coefficients(self):
        """Micro-cap impact coefficient should be higher than large-cap."""
        calibrator = CostCalibrator(model_dir=None)
        micro_coeff = calibrator.get_impact_coeff(100e6)   # micro
        large_coeff = calibrator.get_impact_coeff(50e9)    # large
        self.assertGreater(micro_coeff, large_coeff)
        self.assertAlmostEqual(micro_coeff, 40.0)
        self.assertAlmostEqual(large_coeff, 15.0)

    def test_get_impact_coeff_by_segment(self):
        """get_impact_coeff_by_segment should return segment coefficients."""
        calibrator = CostCalibrator(model_dir=None)
        self.assertAlmostEqual(calibrator.get_impact_coeff_by_segment("micro"), 40.0)
        self.assertAlmostEqual(calibrator.get_impact_coeff_by_segment("small"), 30.0)
        self.assertAlmostEqual(calibrator.get_impact_coeff_by_segment("mid"), 20.0)
        self.assertAlmostEqual(calibrator.get_impact_coeff_by_segment("large"), 15.0)

    def test_impact_override_micro_vs_large(self):
        """Micro-cap override should produce more impact than large-cap."""
        model = _make_model(impact_coefficient_bps=25.0)
        micro_coeff = 40.0
        large_coeff = 15.0

        fill_micro = _calm_fill(model, impact_coeff_override=micro_coeff)
        fill_large = _calm_fill(model, impact_coeff_override=large_coeff)

        # Micro should have higher impact than large (buy side: higher fill)
        self.assertGreater(fill_micro.impact_bps, fill_large.impact_bps)
        self.assertGreater(fill_micro.fill_price, fill_large.fill_price)

    def test_override_vs_flat_default(self):
        """With override, impact should differ from the flat 25 bps default."""
        model = _make_model(impact_coefficient_bps=25.0)
        fill_default = _calm_fill(model)
        fill_micro = _calm_fill(model, impact_coeff_override=40.0)
        fill_large = _calm_fill(model, impact_coeff_override=15.0)

        # Micro > default > large
        self.assertGreater(fill_micro.fill_price, fill_default.fill_price)
        self.assertLess(fill_large.fill_price, fill_default.fill_price)

    def test_calibrator_record_trade_accumulates(self):
        """record_trade should accumulate observations per segment."""
        calibrator = CostCalibrator(
            model_dir=None,
            min_total_trades=5,
            min_segment_trades=3,
        )
        # Record some large-cap trades
        for i in range(5):
            calibrator.record_trade(f"SYM_{i}", 50e9, 0.01, 5.0)
        self.assertEqual(calibrator._total_trades, 5)
        self.assertEqual(len(calibrator._trade_history["large"]), 5)

    def test_calibrator_custom_coefficients(self):
        """Custom default coefficients should override the hardcoded ones."""
        custom = {"micro": 50.0, "small": 35.0, "mid": 25.0, "large": 10.0}
        calibrator = CostCalibrator(
            default_coefficients=custom,
            model_dir=None,
        )
        self.assertAlmostEqual(calibrator.get_impact_coeff(100e6), 50.0)
        self.assertAlmostEqual(calibrator.get_impact_coeff(50e9), 10.0)

    def test_calibrator_custom_thresholds(self):
        """Custom market cap thresholds should change segment boundaries."""
        thresholds = {"micro": 500e6, "small": 5e9, "mid": 20e9}
        calibrator = CostCalibrator(
            marketcap_thresholds=thresholds,
            model_dir=None,
        )
        # 400M is now micro (below 500M threshold)
        self.assertEqual(calibrator.get_marketcap_segment(400e6), "micro")
        # 3B is now small (500M - 5B)
        self.assertEqual(calibrator.get_marketcap_segment(3e9), "small")

    def test_sell_side_segment_override(self):
        """Sell side should also use segment-specific impact coefficients."""
        model = _make_model(impact_coefficient_bps=25.0)
        fill_micro = model.simulate(
            side="sell", reference_price=100.0, daily_volume=5_000_000.0,
            desired_notional_usd=100_000.0, force_full=True,
            realized_vol=0.12, impact_coeff_override=40.0,
        )
        fill_large = model.simulate(
            side="sell", reference_price=100.0, daily_volume=5_000_000.0,
            desired_notional_usd=100_000.0, force_full=True,
            realized_vol=0.12, impact_coeff_override=15.0,
        )
        # Sell side: more impact → lower fill price
        self.assertLess(fill_micro.fill_price, fill_large.fill_price)


class TestMarketCapEstimation(unittest.TestCase):
    """Test the _estimate_market_cap helper on synthetic OHLCV data."""

    def _make_ohlcv(self, price, volume, n_bars=30):
        """Create synthetic OHLCV DataFrame."""
        dates = pd.bdate_range("2024-01-01", periods=n_bars)
        return pd.DataFrame({
            "Open": price,
            "High": price * 1.01,
            "Low": price * 0.99,
            "Close": price,
            "Volume": volume,
        }, index=dates)

    def test_large_cap_estimation(self):
        """High price × high volume → large cap estimate."""
        from quant_engine.backtest.engine import Backtester
        # $200 stock, 10M shares/day → ADV ~$2B → market_cap ~$400B
        ohlcv = self._make_ohlcv(200.0, 10_000_000.0)
        mcap = Backtester._estimate_market_cap(ohlcv, 25)
        self.assertGreater(mcap, 10e9)

    def test_micro_cap_estimation(self):
        """Low price × low volume → micro cap estimate."""
        from quant_engine.backtest.engine import Backtester
        # $5 stock, 50K shares/day → ADV ~$250K → market_cap ~$50M
        ohlcv = self._make_ohlcv(5.0, 50_000.0)
        mcap = Backtester._estimate_market_cap(ohlcv, 25)
        self.assertLess(mcap, 300e6)

    def test_no_volume_returns_zero(self):
        """Zero volume should return 0 market cap."""
        from quant_engine.backtest.engine import Backtester
        ohlcv = self._make_ohlcv(100.0, 0.0)
        mcap = Backtester._estimate_market_cap(ohlcv, 25)
        self.assertEqual(mcap, 0.0)

    def test_missing_volume_column(self):
        """If no Volume column exists, should return 0."""
        from quant_engine.backtest.engine import Backtester
        dates = pd.bdate_range("2024-01-01", periods=30)
        ohlcv = pd.DataFrame({
            "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0,
        }, index=dates)
        mcap = Backtester._estimate_market_cap(ohlcv, 25)
        self.assertEqual(mcap, 0.0)

    def test_early_bar_still_works(self):
        """Early bar (bar_idx=0) should still produce a valid estimate."""
        from quant_engine.backtest.engine import Backtester
        ohlcv = self._make_ohlcv(100.0, 1_000_000.0)
        mcap = Backtester._estimate_market_cap(ohlcv, 0)
        self.assertGreater(mcap, 0.0)


class TestCostCalibratorWithExecutionPipeline(unittest.TestCase):
    """End-to-end test: calibrator → execution model → verify segment differentiation."""

    def test_full_pipeline_segment_differentiation(self):
        """Complete pipeline: calibrator feeds differentiated coefficients
        to execution model, resulting in different costs per segment."""
        model = _make_model()
        calibrator = CostCalibrator(model_dir=None)

        segments = {
            "micro": 100e6,
            "small": 500e6,
            "mid": 5e9,
            "large": 50e9,
        }

        fills = {}
        for seg_name, mcap in segments.items():
            coeff = calibrator.get_impact_coeff(mcap)
            fill = model.simulate(
                side="buy",
                reference_price=100.0,
                daily_volume=5_000_000.0,
                desired_notional_usd=100_000.0,
                force_full=True,
                realized_vol=0.12,
                overnight_gap=0.001,
                intraday_range=0.01,
                impact_coeff_override=coeff,
            )
            fills[seg_name] = fill

        # Verify ordering: micro should have highest cost, large lowest
        self.assertGreater(fills["micro"].impact_bps, fills["small"].impact_bps)
        self.assertGreater(fills["small"].impact_bps, fills["mid"].impact_bps)
        self.assertGreater(fills["mid"].impact_bps, fills["large"].impact_bps)

    def test_record_and_calibrate_cycle(self):
        """Simulate the record → calibrate cycle that happens during backtest.

        Uses realistic participation rates and cost values that exceed
        the half-spread threshold so calibration produces valid
        coefficient observations.
        """
        calibrator = CostCalibrator(
            model_dir=None,
            min_total_trades=10,
            min_segment_trades=5,
            spread_bps=3.0,
        )

        # Record 15 large-cap trades with realistic participation and
        # costs above the half-spread (1.5 bps) deduction threshold.
        for i in range(15):
            calibrator.record_trade(
                symbol=f"AAPL_{i}",
                market_cap=50e9,
                participation_rate=0.01,    # 1% of daily volume
                realized_cost_bps=5.0,      # 5 bps total cost
            )

        # Calibrate and verify coefficients updated
        result = calibrator.calibrate()
        self.assertIn("large", result)
        self.assertGreater(result["large"]["trades_count"], 0)
        # Observed coeff should be (5.0 - 1.5) / sqrt(0.01) = 3.5/0.1 = 35
        self.assertGreater(result["large"]["observed_coeff"], 0)

    def test_trade_dataclass_has_segment(self):
        """Trade dataclass should have market_cap_segment field."""
        from quant_engine.backtest.engine import Trade
        trade = Trade(
            ticker="TEST", entry_date="2024-01-01", exit_date="2024-01-10",
            entry_price=100.0, exit_price=101.0,
            predicted_return=0.01, actual_return=0.01, net_return=0.01,
            regime=0, confidence=0.8, holding_days=10,
            market_cap_segment="large",
        )
        self.assertEqual(trade.market_cap_segment, "large")

    def test_trade_default_segment_empty(self):
        """Trade should default to empty segment for backward compatibility."""
        from quant_engine.backtest.engine import Trade
        trade = Trade(
            ticker="TEST", entry_date="2024-01-01", exit_date="2024-01-10",
            entry_price=100.0, exit_price=101.0,
            predicted_return=0.01, actual_return=0.01, net_return=0.01,
            regime=0, confidence=0.8, holding_days=10,
        )
        self.assertEqual(trade.market_cap_segment, "")


# ══════════════════════════════════════════════════════════════════════════
# SPEC-E04: Calibration Feedback Loop
# ══════════════════════════════════════════════════════════════════════════


class TestRecordActualFill(unittest.TestCase):
    """Test record_actual_fill stores fill data for feedback comparison."""

    def setUp(self):
        self.calibrator = CostCalibrator(model_dir=None)

    def test_records_valid_fill(self):
        self.calibrator.record_actual_fill(
            symbol="AAPL",
            market_cap=200e9,
            predicted_cost_bps=5.0,
            actual_cost_bps=4.5,
            participation_rate=0.01,
            regime=0,
        )
        self.assertEqual(self.calibrator.feedback_fill_count, 1)

    def test_rejects_zero_participation(self):
        self.calibrator.record_actual_fill(
            symbol="AAPL", market_cap=200e9,
            predicted_cost_bps=5.0, actual_cost_bps=4.5,
            participation_rate=0.0, regime=0,
        )
        self.assertEqual(self.calibrator.feedback_fill_count, 0)

    def test_rejects_nan_costs(self):
        self.calibrator.record_actual_fill(
            symbol="AAPL", market_cap=200e9,
            predicted_cost_bps=float("nan"), actual_cost_bps=4.5,
            participation_rate=0.01, regime=0,
        )
        self.assertEqual(self.calibrator.feedback_fill_count, 0)

        self.calibrator.record_actual_fill(
            symbol="AAPL", market_cap=200e9,
            predicted_cost_bps=5.0, actual_cost_bps=float("nan"),
            participation_rate=0.01, regime=0,
        )
        self.assertEqual(self.calibrator.feedback_fill_count, 0)

    def test_rejects_negative_participation(self):
        self.calibrator.record_actual_fill(
            symbol="AAPL", market_cap=200e9,
            predicted_cost_bps=5.0, actual_cost_bps=4.5,
            participation_rate=-0.01, regime=0,
        )
        self.assertEqual(self.calibrator.feedback_fill_count, 0)

    def test_segment_classification_in_fill(self):
        self.calibrator.record_actual_fill(
            symbol="MICRO", market_cap=100e6,
            predicted_cost_bps=8.0, actual_cost_bps=7.0,
            participation_rate=0.02, regime=2,
        )
        fills = self.calibrator.actual_fills
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["segment"], "micro")

    def test_multiple_fills_accumulated(self):
        for i in range(10):
            self.calibrator.record_actual_fill(
                symbol=f"SYM_{i}", market_cap=50e9,
                predicted_cost_bps=5.0 + i * 0.1,
                actual_cost_bps=4.5 + i * 0.1,
                participation_rate=0.01, regime=i % 4,
            )
        self.assertEqual(self.calibrator.feedback_fill_count, 10)

    def test_fill_stores_timestamp(self):
        self.calibrator.record_actual_fill(
            symbol="AAPL", market_cap=200e9,
            predicted_cost_bps=5.0, actual_cost_bps=4.5,
            participation_rate=0.01, regime=0,
            fill_timestamp="2024-06-01T12:00:00+00:00",
        )
        fills = self.calibrator.actual_fills
        self.assertEqual(fills[0]["timestamp"], "2024-06-01T12:00:00+00:00")

    def test_fill_auto_generates_timestamp(self):
        self.calibrator.record_actual_fill(
            symbol="AAPL", market_cap=200e9,
            predicted_cost_bps=5.0, actual_cost_bps=4.5,
            participation_rate=0.01, regime=0,
        )
        fills = self.calibrator.actual_fills
        # Should have a non-empty ISO timestamp
        self.assertTrue(len(fills[0]["timestamp"]) > 0)


class TestComputeCostSurprise(unittest.TestCase):
    """Test compute_cost_surprise and compute_cost_surprise_by_segment."""

    def setUp(self):
        self.calibrator = CostCalibrator(model_dir=None)

    def test_empty_returns_empty(self):
        self.assertEqual(self.calibrator.compute_cost_surprise(), {})
        self.assertEqual(self.calibrator.compute_cost_surprise_by_segment(), {})

    def test_single_regime_bucket(self):
        # Model over-estimates cost (predicted=5.0, actual=4.0 → surprise=1.0)
        for i in range(5):
            self.calibrator.record_actual_fill(
                symbol=f"SYM_{i}", market_cap=50e9,
                predicted_cost_bps=5.0, actual_cost_bps=4.0,
                participation_rate=0.01, regime=0,
            )
        result = self.calibrator.compute_cost_surprise()
        self.assertIn("0", result)
        self.assertIn("_all", result)
        self.assertAlmostEqual(result["0"]["mean_surprise_bps"], 1.0)
        self.assertAlmostEqual(result["0"]["pct_overestimated"], 1.0)
        self.assertEqual(result["0"]["count"], 5)

    def test_multiple_regime_buckets(self):
        # Regime 0: over-estimate
        for i in range(3):
            self.calibrator.record_actual_fill(
                symbol=f"A_{i}", market_cap=50e9,
                predicted_cost_bps=6.0, actual_cost_bps=4.0,
                participation_rate=0.01, regime=0,
            )
        # Regime 3: under-estimate
        for i in range(3):
            self.calibrator.record_actual_fill(
                symbol=f"B_{i}", market_cap=50e9,
                predicted_cost_bps=3.0, actual_cost_bps=5.0,
                participation_rate=0.01, regime=3,
            )
        result = self.calibrator.compute_cost_surprise()
        self.assertIn("0", result)
        self.assertIn("3", result)
        self.assertAlmostEqual(result["0"]["mean_surprise_bps"], 2.0)
        self.assertAlmostEqual(result["3"]["mean_surprise_bps"], -2.0)
        # Aggregate should be near zero
        self.assertAlmostEqual(result["_all"]["mean_surprise_bps"], 0.0)

    def test_segment_bucketing(self):
        # Large cap fills
        for i in range(3):
            self.calibrator.record_actual_fill(
                symbol=f"L_{i}", market_cap=50e9,
                predicted_cost_bps=5.0, actual_cost_bps=4.0,
                participation_rate=0.01, regime=0,
            )
        # Micro cap fills
        for i in range(3):
            self.calibrator.record_actual_fill(
                symbol=f"M_{i}", market_cap=100e6,
                predicted_cost_bps=10.0, actual_cost_bps=8.0,
                participation_rate=0.02, regime=0,
            )
        result = self.calibrator.compute_cost_surprise_by_segment()
        self.assertIn("large", result)
        self.assertIn("micro", result)
        self.assertAlmostEqual(result["large"]["mean_surprise_bps"], 1.0)
        self.assertAlmostEqual(result["micro"]["mean_surprise_bps"], 2.0)

    def test_std_with_single_fill(self):
        self.calibrator.record_actual_fill(
            symbol="A", market_cap=50e9,
            predicted_cost_bps=5.0, actual_cost_bps=4.0,
            participation_rate=0.01, regime=0,
        )
        result = self.calibrator.compute_cost_surprise()
        self.assertAlmostEqual(result["_all"]["std_surprise_bps"], 0.0)


class TestFeedbackRecalibration(unittest.TestCase):
    """Test run_feedback_recalibration updates coefficients from actual fills."""

    def setUp(self):
        self.calibrator = CostCalibrator(
            model_dir=None,
            min_total_trades=10,
            min_segment_trades=5,
            smoothing=0.30,
            spread_bps=3.0,
        )

    def test_no_fills_returns_empty(self):
        result = self.calibrator.run_feedback_recalibration(force=True)
        self.assertEqual(result, {})

    def test_insufficient_fills_returns_empty(self):
        for i in range(5):
            self.calibrator.record_actual_fill(
                symbol=f"SYM_{i}", market_cap=50e9,
                predicted_cost_bps=5.0, actual_cost_bps=4.0,
                participation_rate=0.01, regime=0,
            )
        result = self.calibrator.run_feedback_recalibration(force=True)
        self.assertEqual(result, {})

    def test_recalibration_updates_coefficients(self):
        old_large = self.calibrator.coefficients["large"]
        # Record fills with actual costs that imply a coefficient different
        # from the default. actual_cost = 5.0 bps, participation = 0.01
        # implied_coeff = (5.0 - 1.5) / sqrt(0.01) = 3.5 / 0.1 = 35.0
        for i in range(15):
            self.calibrator.record_actual_fill(
                symbol=f"AAPL_{i}", market_cap=50e9,
                predicted_cost_bps=5.0, actual_cost_bps=5.0,
                participation_rate=0.01, regime=0,
            )
        result = self.calibrator.run_feedback_recalibration(force=True)
        self.assertIn("large", result)
        new_large = self.calibrator.coefficients["large"]
        # Coefficient should have moved toward 35.0 from 15.0
        self.assertGreater(new_large, old_large)

    def test_smoothing_applied(self):
        # Record fills implying coefficient = 35.0 for large segment
        for i in range(15):
            self.calibrator.record_actual_fill(
                symbol=f"SYM_{i}", market_cap=50e9,
                predicted_cost_bps=5.0, actual_cost_bps=5.0,
                participation_rate=0.01, regime=0,
            )
        old_large = self.calibrator.coefficients["large"]  # 15.0
        result = self.calibrator.run_feedback_recalibration(force=True)
        new_large = self.calibrator.coefficients["large"]
        observed = result["large"]["observed_coeff"]
        expected = np.clip(0.30 * observed + 0.70 * old_large, 5.0, 100.0)
        self.assertAlmostEqual(new_large, expected, places=1)

    def test_interval_check_prevents_early_recalibration(self):
        for i in range(15):
            self.calibrator.record_actual_fill(
                symbol=f"SYM_{i}", market_cap=50e9,
                predicted_cost_bps=5.0, actual_cost_bps=5.0,
                participation_rate=0.01, regime=0,
            )
        # First call with force=True succeeds
        result1 = self.calibrator.run_feedback_recalibration(force=True)
        self.assertIn("large", result1)

        # Second call without force should be skipped (interval not elapsed)
        result2 = self.calibrator.run_feedback_recalibration(force=False)
        self.assertEqual(result2, {})

    def test_force_bypasses_interval(self):
        for i in range(15):
            self.calibrator.record_actual_fill(
                symbol=f"SYM_{i}", market_cap=50e9,
                predicted_cost_bps=5.0, actual_cost_bps=5.0,
                participation_rate=0.01, regime=0,
            )
        self.calibrator.run_feedback_recalibration(force=True)
        # Force should still work even after recent recalibration
        result = self.calibrator.run_feedback_recalibration(force=True)
        self.assertIn("large", result)

    def test_records_recalibration_timestamp(self):
        for i in range(15):
            self.calibrator.record_actual_fill(
                symbol=f"SYM_{i}", market_cap=50e9,
                predicted_cost_bps=5.0, actual_cost_bps=5.0,
                participation_rate=0.01, regime=0,
            )
        self.assertIsNone(self.calibrator.last_feedback_recalibration)
        self.calibrator.run_feedback_recalibration(force=True)
        self.assertIsNotNone(self.calibrator.last_feedback_recalibration)

    def test_multi_segment_recalibration(self):
        # Large-cap fills
        for i in range(10):
            self.calibrator.record_actual_fill(
                symbol=f"L_{i}", market_cap=50e9,
                predicted_cost_bps=5.0, actual_cost_bps=5.0,
                participation_rate=0.01, regime=0,
            )
        # Micro-cap fills
        for i in range(10):
            self.calibrator.record_actual_fill(
                symbol=f"M_{i}", market_cap=100e6,
                predicted_cost_bps=10.0, actual_cost_bps=10.0,
                participation_rate=0.02, regime=0,
            )
        result = self.calibrator.run_feedback_recalibration(force=True)
        self.assertIn("large", result)
        self.assertIn("micro", result)
        # Both segments should have fills_count
        self.assertEqual(result["large"]["fills_count"], 10)
        self.assertEqual(result["micro"]["fills_count"], 10)

    def test_result_includes_surprise_stats(self):
        for i in range(15):
            self.calibrator.record_actual_fill(
                symbol=f"SYM_{i}", market_cap=50e9,
                predicted_cost_bps=5.0 + i * 0.1,
                actual_cost_bps=4.0 + i * 0.1,
                participation_rate=0.01, regime=0,
            )
        result = self.calibrator.run_feedback_recalibration(force=True)
        large = result["large"]
        self.assertIn("mean_surprise_bps", large)
        self.assertIn("median_surprise_bps", large)
        self.assertAlmostEqual(large["mean_surprise_bps"], 1.0)


class TestFeedbackPersistence(unittest.TestCase):
    """Test that feedback fill history persists to and loads from disk."""

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self._feedback_path = Path(self._tmpdir) / "feedback.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_and_load_feedback(self):
        cal1 = CostCalibrator(
            model_dir=None,
            feedback_path=self._feedback_path,
        )
        for i in range(5):
            cal1.record_actual_fill(
                symbol=f"SYM_{i}", market_cap=50e9,
                predicted_cost_bps=5.0, actual_cost_bps=4.0,
                participation_rate=0.01, regime=0,
            )
        cal1._save_feedback_history()
        self.assertTrue(self._feedback_path.exists())

        # Load into fresh calibrator
        cal2 = CostCalibrator(
            model_dir=None,
            feedback_path=self._feedback_path,
        )
        self.assertEqual(cal2.feedback_fill_count, 5)

    def test_recalibration_timestamp_persists(self):
        cal1 = CostCalibrator(
            model_dir=None,
            feedback_path=self._feedback_path,
            min_total_trades=5,
            min_segment_trades=3,
        )
        for i in range(10):
            cal1.record_actual_fill(
                symbol=f"SYM_{i}", market_cap=50e9,
                predicted_cost_bps=5.0, actual_cost_bps=5.0,
                participation_rate=0.01, regime=0,
            )
        cal1.run_feedback_recalibration(force=True)
        self.assertIsNotNone(cal1.last_feedback_recalibration)

        # Load into fresh calibrator
        cal2 = CostCalibrator(
            model_dir=None,
            feedback_path=self._feedback_path,
        )
        self.assertEqual(
            cal2.last_feedback_recalibration,
            cal1.last_feedback_recalibration,
        )

    def test_no_feedback_path_no_save(self):
        cal = CostCalibrator(model_dir=None, feedback_path=None)
        cal.record_actual_fill(
            symbol="SYM", market_cap=50e9,
            predicted_cost_bps=5.0, actual_cost_bps=4.0,
            participation_rate=0.01, regime=0,
        )
        # Should not crash and should not create any file
        cal._save_feedback_history()
        cal._load_feedback_history()


class TestResetFeedbackHistory(unittest.TestCase):
    """Test reset_feedback_history clears fill state."""

    def test_reset_clears_fills_and_timestamp(self):
        cal = CostCalibrator(
            model_dir=None,
            min_total_trades=5,
            min_segment_trades=3,
        )
        for i in range(10):
            cal.record_actual_fill(
                symbol=f"SYM_{i}", market_cap=50e9,
                predicted_cost_bps=5.0, actual_cost_bps=5.0,
                participation_rate=0.01, regime=0,
            )
        cal.run_feedback_recalibration(force=True)
        self.assertGreater(cal.feedback_fill_count, 0)
        self.assertIsNotNone(cal.last_feedback_recalibration)

        cal.reset_feedback_history()
        self.assertEqual(cal.feedback_fill_count, 0)
        self.assertIsNone(cal.last_feedback_recalibration)


class TestCostCalibratorRepr(unittest.TestCase):
    """Test __repr__ includes feedback info."""

    def test_repr_includes_feedback_fills(self):
        cal = CostCalibrator(model_dir=None)
        cal.record_actual_fill(
            symbol="SYM", market_cap=50e9,
            predicted_cost_bps=5.0, actual_cost_bps=4.0,
            participation_rate=0.01, regime=0,
        )
        r = repr(cal)
        self.assertIn("feedback_fills=1", r)


class TestFeedbackEndToEnd(unittest.TestCase):
    """End-to-end: execution model → calibrator → feedback → recalibration."""

    def test_full_feedback_cycle(self):
        """End-to-end: record fills with realistic costs, recalibrate, verify."""
        calibrator = CostCalibrator(
            model_dir=None,
            min_total_trades=10,
            min_segment_trades=5,
            smoothing=0.30,
            spread_bps=3.0,
        )
        old_large = calibrator.coefficients["large"]

        # Record 15 large-cap fills with costs that exceed half-spread
        # so implied coefficients are positive.
        # predicted=5.0, actual=6.0 → under-estimated (surprise < 0)
        # implied_coeff from actual: (6.0 - 1.5) / sqrt(0.01) = 4.5/0.1 = 45.0
        for i in range(15):
            calibrator.record_actual_fill(
                symbol=f"AAPL_{i}",
                market_cap=200e9,  # large cap
                predicted_cost_bps=5.0,
                actual_cost_bps=6.0,
                participation_rate=0.01,
                regime=0,
            )

        self.assertEqual(calibrator.feedback_fill_count, 15)

        # Recalibrate
        result = calibrator.run_feedback_recalibration(force=True)
        self.assertIn("large", result)

        # Cost surprise should be negative (model under-estimated)
        surprise = calibrator.compute_cost_surprise()
        self.assertLess(surprise["_all"]["mean_surprise_bps"], 0.0)

        # Coefficients should have moved toward 45.0 from 15.0
        new_large = calibrator.coefficients["large"]
        self.assertGreater(new_large, old_large)

    def test_well_calibrated_model_surprise_near_zero(self):
        """After calibration, a perfectly calibrated model should have
        cost surprise distribution centered near zero."""
        calibrator = CostCalibrator(
            model_dir=None,
            min_total_trades=10,
            min_segment_trades=5,
        )

        # Record fills where predicted == actual
        for i in range(20):
            calibrator.record_actual_fill(
                symbol=f"SYM_{i}", market_cap=50e9,
                predicted_cost_bps=5.0, actual_cost_bps=5.0,
                participation_rate=0.01, regime=0,
            )

        surprise = calibrator.compute_cost_surprise()
        self.assertAlmostEqual(
            surprise["_all"]["mean_surprise_bps"], 0.0, places=5,
        )
        self.assertAlmostEqual(
            surprise["_all"]["median_surprise_bps"], 0.0, places=5,
        )


if __name__ == "__main__":
    unittest.main()
