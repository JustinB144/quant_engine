"""
Test module for execution dynamic costs behavior and regressions.
"""

import unittest

from quant_engine.backtest.execution import ExecutionModel


class ExecutionDynamicCostTests(unittest.TestCase):
    """Test cases covering execution dynamic costs behavior and system invariants."""
    def test_dynamic_costs_increase_under_stress(self):
        model = ExecutionModel(
            spread_bps=3.0,
            impact_coefficient_bps=25.0,
            max_participation_rate=0.05,
            dynamic_costs=True,
        )

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


if __name__ == "__main__":
    unittest.main()
