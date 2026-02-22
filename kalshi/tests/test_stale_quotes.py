"""
Stale quote cutoff test (Instructions I.5).

Verifies that dynamic_stale_cutoff_minutes() correctly tightens near events
and loosens far from events, and that StalePolicy parameters work.
"""
import unittest

from quant_engine.kalshi.quality import (
    StalePolicy,
    dynamic_stale_cutoff_minutes,
)


class StaleQuoteCutoffTests(unittest.TestCase):
    """Tests for dynamic stale-cutoff schedule."""

    def test_near_event_tight_cutoff(self):
        """Within near_event_minutes of event, cutoff should be near_event_stale_minutes."""
        policy = StalePolicy(
            near_event_minutes=30.0,
            near_event_stale_minutes=2.0,
            far_event_minutes=1440.0,
            far_event_stale_minutes=60.0,
        )
        cutoff = dynamic_stale_cutoff_minutes(
            time_to_event_minutes=10.0, policy=policy
        )
        self.assertAlmostEqual(cutoff, 2.0, places=1)

    def test_far_event_loose_cutoff(self):
        """Beyond far_event_minutes, cutoff should be far_event_stale_minutes."""
        policy = StalePolicy(
            near_event_minutes=30.0,
            near_event_stale_minutes=2.0,
            far_event_minutes=1440.0,
            far_event_stale_minutes=60.0,
        )
        cutoff = dynamic_stale_cutoff_minutes(
            time_to_event_minutes=2000.0, policy=policy
        )
        self.assertAlmostEqual(cutoff, 60.0, places=1)

    def test_midpoint_interpolation(self):
        """Between near and far, cutoff should interpolate."""
        policy = StalePolicy(
            near_event_minutes=30.0,
            near_event_stale_minutes=2.0,
            far_event_minutes=1440.0,
            far_event_stale_minutes=60.0,
        )
        # Midpoint between 30 and 1440 = 735 minutes
        cutoff = dynamic_stale_cutoff_minutes(
            time_to_event_minutes=735.0, policy=policy
        )
        # Should be between 2 and 60
        self.assertGreater(cutoff, 2.0)
        self.assertLess(cutoff, 60.0)

    def test_cutoff_monotonically_increases_with_distance(self):
        """Cutoff should increase as we get farther from the event."""
        policy = StalePolicy()
        times = [5.0, 30.0, 120.0, 720.0, 1440.0, 2880.0]
        cutoffs = [
            dynamic_stale_cutoff_minutes(t, policy=policy)
            for t in times
        ]
        for i in range(len(cutoffs) - 1):
            self.assertLessEqual(
                cutoffs[i], cutoffs[i + 1],
                f"Cutoff should not decrease: {cutoffs[i]} > {cutoffs[i + 1]} "
                f"at times {times[i]} vs {times[i + 1]}"
            )

    def test_cpi_market_type_multiplier(self):
        """CPI market type should tighten the cutoff (multiplier < 1)."""
        policy = StalePolicy(
            market_type_multipliers={"CPI": 0.80, "_default": 1.00}
        )
        cutoff_default = dynamic_stale_cutoff_minutes(
            time_to_event_minutes=500.0, policy=policy, market_type=None
        )
        cutoff_cpi = dynamic_stale_cutoff_minutes(
            time_to_event_minutes=500.0, policy=policy, market_type="CPI"
        )
        self.assertLess(cutoff_cpi, cutoff_default)

    def test_fomc_market_type_multiplier(self):
        """FOMC should have the tightest multiplier."""
        policy = StalePolicy(
            market_type_multipliers={"CPI": 0.80, "FOMC": 0.70, "_default": 1.00}
        )
        cutoff_fomc = dynamic_stale_cutoff_minutes(
            time_to_event_minutes=500.0, policy=policy, market_type="FOMC"
        )
        cutoff_cpi = dynamic_stale_cutoff_minutes(
            time_to_event_minutes=500.0, policy=policy, market_type="CPI"
        )
        self.assertLess(cutoff_fomc, cutoff_cpi)

    def test_low_liquidity_widens_cutoff(self):
        """Low liquidity should widen the cutoff (multiplier > 1)."""
        policy = StalePolicy(
            liquidity_low_threshold=2.0,
            low_liquidity_multiplier=1.35,
        )
        cutoff_normal = dynamic_stale_cutoff_minutes(
            time_to_event_minutes=500.0, policy=policy, liquidity_proxy=5.0
        )
        cutoff_low_liq = dynamic_stale_cutoff_minutes(
            time_to_event_minutes=500.0, policy=policy, liquidity_proxy=1.0
        )
        self.assertGreater(cutoff_low_liq, cutoff_normal)

    def test_high_liquidity_tightens_cutoff(self):
        """High liquidity should tighten the cutoff (multiplier < 1)."""
        policy = StalePolicy(
            liquidity_high_threshold=6.0,
            high_liquidity_multiplier=0.80,
        )
        cutoff_normal = dynamic_stale_cutoff_minutes(
            time_to_event_minutes=500.0, policy=policy, liquidity_proxy=3.0
        )
        cutoff_high_liq = dynamic_stale_cutoff_minutes(
            time_to_event_minutes=500.0, policy=policy, liquidity_proxy=10.0
        )
        self.assertLess(cutoff_high_liq, cutoff_normal)

    def test_none_time_to_event_uses_base(self):
        """None time_to_event should use base_stale_minutes."""
        policy = StalePolicy(base_stale_minutes=30.0)
        cutoff = dynamic_stale_cutoff_minutes(
            time_to_event_minutes=None, policy=policy
        )
        self.assertAlmostEqual(cutoff, 30.0, places=1)

    def test_cutoff_clamped_to_bounds(self):
        """Cutoff should be clamped between min and max."""
        policy = StalePolicy(
            min_stale_minutes=0.5,
            max_stale_minutes=1440.0,
        )
        # Extreme near-event with very aggressive multipliers
        cutoff = dynamic_stale_cutoff_minutes(
            time_to_event_minutes=0.0, policy=policy
        )
        self.assertGreaterEqual(cutoff, 0.5)
        self.assertLessEqual(cutoff, 1440.0)


if __name__ == "__main__":
    unittest.main()
