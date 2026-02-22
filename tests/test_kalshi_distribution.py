"""
Test module for kalshi distribution behavior and regressions.
"""

import unittest

import pandas as pd

from quant_engine.kalshi.distribution import (
    DistributionConfig,
    build_distribution_panel,
    build_distribution_snapshot,
)


class KalshiDistributionTests(unittest.TestCase):
    """Test cases covering kalshi distribution behavior and regression protections."""
    def test_bin_distribution_normalizes_and_computes_moments(self):
        contracts = pd.DataFrame(
            {
                "contract_id": ["C1", "C2", "C3"],
                "market_id": ["M1", "M1", "M1"],
                "bin_low": [2.0, 3.0, 4.0],
                "bin_high": [3.0, 4.0, 5.0],
            },
        )
        quotes = pd.DataFrame(
            {
                "contract_id": ["C1", "C2", "C3"],
                "ts": [
                    "2025-01-01T13:00:00Z",
                    "2025-01-01T13:00:00Z",
                    "2025-01-01T13:00:00Z",
                ],
                "bid": [20.0, 30.0, 10.0],
                "ask": [22.0, 32.0, 12.0],
                "mid": [21.0, 31.0, 11.0],  # cents
            },
        )
        stats = build_distribution_snapshot(
            contracts=contracts,
            quotes=quotes,
            asof_ts=pd.Timestamp("2025-01-01T13:05:00Z"),
            config=DistributionConfig(price_scale="cents"),
        )
        self.assertTrue(stats["quality_score"] > 0.0)
        self.assertTrue(pd.notna(stats["mean"]))
        self.assertTrue(pd.notna(stats["var"]))
        self.assertTrue(pd.notna(stats["entropy"]))

    def test_threshold_distribution_applies_monotone_constraint(self):
        contracts = pd.DataFrame(
            {
                "contract_id": ["T1", "T2", "T3"],
                "market_id": ["M2", "M2", "M2"],
                "threshold_value": [2.0, 3.0, 4.0],
            },
        )
        quotes = pd.DataFrame(
            {
                "contract_id": ["T1", "T2", "T3"],
                "ts": [
                    "2025-01-01T13:00:00Z",
                    "2025-01-01T13:00:00Z",
                    "2025-01-01T13:00:00Z",
                ],
                # Deliberately non-monotone if interpreted directly.
                "mid": [0.40, 0.65, 0.20],
                "bid": [0.39, 0.64, 0.19],
                "ask": [0.41, 0.66, 0.21],
            },
        )
        stats = build_distribution_snapshot(
            contracts=contracts,
            quotes=quotes,
            asof_ts=pd.Timestamp("2025-01-01T13:05:00Z"),
            config=DistributionConfig(price_scale="prob"),
        )
        self.assertTrue(stats["quality_score"] > 0.0)
        self.assertTrue(pd.notna(stats["mean"]))
        self.assertTrue(pd.notna(stats["var"]))

    def test_distribution_panel_accepts_tz_aware_snapshot_times(self):
        markets = pd.DataFrame({"market_id": ["M1"]})
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
                "ts": ["2025-01-01T13:00:00Z", "2025-01-01T13:00:00Z", "2025-01-01T13:00:00Z"],
                "mid": [0.2, 0.5, 0.3],
                "bid": [0.19, 0.49, 0.29],
                "ask": [0.21, 0.51, 0.31],
            },
        )
        panel = build_distribution_panel(
            markets=markets,
            contracts=contracts,
            quotes=quotes,
            snapshot_times=[pd.Timestamp("2025-01-01T13:05:00Z")],
            config=DistributionConfig(price_scale="prob"),
        )
        self.assertEqual(len(panel), 1)
        self.assertEqual(str(panel.iloc[0]["ts"].tzinfo), "UTC")


if __name__ == "__main__":
    unittest.main()
