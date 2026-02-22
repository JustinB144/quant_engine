"""
Kalshi test module for leakage behavior and regressions.
"""

import unittest

import pandas as pd

from quant_engine.kalshi.events import EventFeatureConfig, build_event_feature_panel


class LeakageLocalTests(unittest.TestCase):
    """Test cases covering Kalshi subsystem behavior and safety constraints."""
    def test_feature_rows_strictly_pre_release(self):
        macro_events = pd.DataFrame(
            {
                "event_id": ["E1"],
                "event_type": ["CPI"],
                "release_ts": ["2025-01-01T13:30:00Z"],
            },
        )
        mapping = pd.DataFrame({"event_type": ["CPI"], "market_id": ["M1"]})
        dists = pd.DataFrame(
            {
                "market_id": ["M1"],
                "ts": ["2025-01-01T13:00:00Z"],
                "mean": [3.0],
                "var": [0.2],
                "skew": [0.0],
                "entropy": [1.0],
                "quality_score": [1.0],
            },
        )
        panel = build_event_feature_panel(
            macro_events=macro_events,
            event_market_map=mapping,
            kalshi_distributions=dists,
            config=EventFeatureConfig(snapshot_horizons=["15m"]),
        )
        self.assertEqual(len(panel), 1)
        row = panel.reset_index().iloc[0]
        self.assertLess(pd.Timestamp(row["asof_ts"]), pd.Timestamp(row["release_ts"]))


if __name__ == "__main__":
    unittest.main()
