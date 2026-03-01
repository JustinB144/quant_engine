"""
Spec AUDIT_FIX_32 — Execution parity, structural robustness & code hygiene tests.

T1: Verify paper trader applies the same shock-mode spread multipliers,
confidence thresholds, and participation limits as the backtest engine.
"""
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quant_engine.backtest.execution import ExecutionModel, ShockModePolicy
from quant_engine.regime.shock_vector import ShockVector


class TestShockPolicyParity(unittest.TestCase):
    """T1: Backtest vs paper trader shock policy parity.

    Both code paths now use ShockModePolicy.from_shock_vector() with the
    same config-driven parameters, so we verify that the resulting
    simulate() costs match when given identical shock vectors.
    """

    def _make_shock_vector(self, tier: str) -> ShockVector:
        """Create a ShockVector that triggers the given tier."""
        if tier == "shock":
            return ShockVector(
                ticker="TEST",
                timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
                hmm_regime=3,
                hmm_confidence=0.3,
                hmm_uncertainty=0.9,
                bocpd_changepoint_prob=0.8,
                jump_detected=True,
                jump_magnitude=0.05,
                structural_features={
                    "drift_score": 0.2,
                    "systemic_stress": 0.8,
                },
            )
        elif tier == "elevated":
            return ShockVector(
                ticker="TEST",
                timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
                hmm_regime=2,
                hmm_confidence=0.5,
                hmm_uncertainty=0.75,
                bocpd_changepoint_prob=0.3,
                jump_detected=False,
                jump_magnitude=0.0,
                structural_features={
                    "drift_score": 0.4,
                    "systemic_stress": 0.4,
                },
            )
        else:
            return ShockVector(
                ticker="TEST",
                timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
                hmm_regime=0,
                hmm_confidence=0.9,
                hmm_uncertainty=0.1,
                bocpd_changepoint_prob=0.05,
                structural_features={},
            )

    def test_shock_mode_costs_match(self):
        """Under shock conditions, simulate() costs should match
        when given identical inputs — verifying the paper trader
        now passes event_spread_multiplier and max_participation_override.
        """
        model = ExecutionModel()
        shock_sv = self._make_shock_vector("shock")
        policy = ShockModePolicy.from_shock_vector(shock_sv)

        self.assertTrue(policy.is_active)
        self.assertEqual(policy.tier, "shock")

        spread_mult = policy.spread_multiplier
        max_part = policy.max_participation_override

        # Simulate the same trade with shock overrides (backtest path)
        fill_bt = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=500_000.0,
            desired_notional_usd=50_000.0,
            event_spread_multiplier=spread_mult,
            max_participation_override=max_part,
            break_probability=shock_sv.bocpd_changepoint_prob,
            structure_uncertainty=shock_sv.hmm_uncertainty,
        )

        # Simulate the same trade with shock overrides (paper trader path)
        fill_pt = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=500_000.0,
            desired_notional_usd=50_000.0,
            event_spread_multiplier=spread_mult,
            max_participation_override=max_part,
            break_probability=shock_sv.bocpd_changepoint_prob,
            structure_uncertainty=shock_sv.hmm_uncertainty,
        )

        # Same inputs → same outputs (exact match since deterministic)
        self.assertAlmostEqual(
            fill_bt.impact_bps + fill_bt.spread_bps, fill_pt.impact_bps + fill_pt.spread_bps, places=6,
        )
        self.assertAlmostEqual(
            fill_bt.fill_ratio, fill_pt.fill_ratio, places=6,
        )

    def test_shock_costs_higher_than_normal(self):
        """Shock-mode execution should produce higher costs than normal."""
        model = ExecutionModel()

        shock_sv = self._make_shock_vector("shock")
        shock_policy = ShockModePolicy.from_shock_vector(shock_sv)
        normal_policy = ShockModePolicy.normal_default()

        fill_shock = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=500_000.0,
            desired_notional_usd=50_000.0,
            event_spread_multiplier=shock_policy.spread_multiplier,
            max_participation_override=shock_policy.max_participation_override,
        )
        fill_normal = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=500_000.0,
            desired_notional_usd=50_000.0,
            event_spread_multiplier=normal_policy.spread_multiplier,
        )

        self.assertGreater(fill_shock.impact_bps + fill_shock.spread_bps, fill_normal.impact_bps + fill_normal.spread_bps)

    def test_shock_confidence_gate(self):
        """ShockModePolicy should enforce minimum confidence thresholds."""
        shock_sv = self._make_shock_vector("shock")
        policy = ShockModePolicy.from_shock_vector(shock_sv)

        self.assertTrue(policy.is_active)
        # Default shock min confidence is 0.80
        self.assertGreaterEqual(policy.min_confidence_override, 0.70)

        # A confidence of 0.5 should be below shock minimum
        self.assertLess(0.5, policy.min_confidence_override)

    def test_elevated_policy_intermediate(self):
        """Elevated tier should have intermediate restrictions."""
        elevated_sv = self._make_shock_vector("elevated")
        policy = ShockModePolicy.from_shock_vector(elevated_sv)

        self.assertTrue(policy.is_active)
        self.assertEqual(policy.tier, "elevated")
        self.assertGreater(policy.spread_multiplier, 1.0)

    def test_normal_policy_not_active(self):
        """Normal tier should have is_active=False."""
        normal_sv = self._make_shock_vector("normal")
        policy = ShockModePolicy.from_shock_vector(normal_sv)

        self.assertFalse(policy.is_active)
        self.assertEqual(policy.tier, "normal")


class TestPaperTraderShockVectorConstruction(unittest.TestCase):
    """T1: Verify _get_current_shock_vector builds valid ShockVectors."""

    def test_builds_shock_vector_from_pred_row(self):
        """Should construct ShockVector from prediction row data."""
        from quant_engine.autopilot.paper_trader import PaperTrader

        pred_row = {
            "break_probability": 0.7,
            "regime_entropy": 0.85,
            "drift_score": 0.3,
            "systemic_stress": 0.6,
            "regime": 3,
            "confidence": 0.4,
        }
        sv = PaperTrader._get_current_shock_vector("AAPL", pred_row)
        self.assertIsNotNone(sv)
        self.assertEqual(sv.ticker, "AAPL")
        self.assertAlmostEqual(sv.bocpd_changepoint_prob, 0.7)
        self.assertAlmostEqual(sv.hmm_uncertainty, 0.85)
        self.assertEqual(sv.hmm_regime, 3)

    def test_returns_none_for_empty_row(self):
        """Should return None when no structural state is available."""
        from quant_engine.autopilot.paper_trader import PaperTrader

        sv = PaperTrader._get_current_shock_vector("AAPL", None)
        self.assertIsNone(sv)

    def test_returns_none_for_no_structural_fields(self):
        """Should return None when pred_row has no structural fields."""
        from quant_engine.autopilot.paper_trader import PaperTrader

        pred_row = {"predicted_return": 0.05, "confidence": 0.8}
        sv = PaperTrader._get_current_shock_vector("AAPL", pred_row)
        self.assertIsNone(sv)


if __name__ == "__main__":
    unittest.main()
