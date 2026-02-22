"""
Kalshi test module for distribution behavior and regressions.
"""

import unittest

import pandas as pd

from quant_engine.kalshi.distribution import DistributionConfig, build_distribution_snapshot


class DistributionLocalTests(unittest.TestCase):
    """Test cases covering Kalshi subsystem behavior and safety constraints."""
    def test_bin_distribution_probability_mass_is_normalized(self):
        contracts = pd.DataFrame(
            {
                "contract_id": ["C1", "C2", "C3"],
                "market_id": ["M1", "M1", "M1"],
                "bin_low": [1.0, 2.0, 3.0],
                "bin_high": [2.0, 3.0, 4.0],
            },
        )
        quotes = pd.DataFrame(
            {
                "contract_id": ["C1", "C2", "C3"],
                "ts": ["2025-01-01T10:00:00Z", "2025-01-01T10:00:00Z", "2025-01-01T10:00:00Z"],
                "mid": [0.2, 0.5, 0.3],
            },
        )
        stats = build_distribution_snapshot(
            contracts=contracts,
            quotes=quotes,
            asof_ts=pd.Timestamp("2025-01-01T10:05:00Z"),
            config=DistributionConfig(price_scale="prob"),
            event_type="CPI",
        )
        self.assertAlmostEqual(sum(stats["_mass"]), 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
