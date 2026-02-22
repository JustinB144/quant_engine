"""
Test module for kalshi asof features behavior and regressions.
"""

import unittest

import pandas as pd

from quant_engine.kalshi.events import EventFeatureConfig, build_event_feature_panel


class KalshiAsofFeatureTests(unittest.TestCase):
    """Test cases covering kalshi asof features behavior and regression protections."""
    def test_event_feature_panel_uses_backward_asof_join(self):
        macro_events = pd.DataFrame(
            {
                "event_id": ["E1"],
                "event_type": ["CPI"],
                "release_ts": ["2025-01-01T13:30:00Z"],
            },
        )
        event_market_map = pd.DataFrame({"event_type": ["CPI"], "market_id": ["M1"]})

        distributions = pd.DataFrame(
            {
                "market_id": ["M1", "M1"],
                "ts": ["2025-01-01T13:00:00Z", "2025-01-01T13:31:00Z"],
                "mean": [3.1, 9.9],
                "var": [0.2, 0.2],
                "skew": [0.0, 0.0],
                "entropy": [1.0, 1.0],
                "quality_score": [1.0, 1.0],
            },
        )
        cfg = EventFeatureConfig(snapshot_horizons=["15m"], min_quality_score=0.0)
        panel = build_event_feature_panel(
            macro_events=macro_events,
            event_market_map=event_market_map,
            kalshi_distributions=distributions,
            config=cfg,
        )
        self.assertEqual(len(panel), 1)
        row = panel.reset_index().iloc[0]
        # asof=13:15 should join to 13:00 row, not the 13:31 row.
        self.assertAlmostEqual(float(row["mean"]), 3.1, places=6)
        self.assertLess(pd.Timestamp(row["asof_ts"]), pd.Timestamp(row["release_ts"]))

    def test_event_feature_panel_raises_when_required_columns_missing(self):
        macro_events = pd.DataFrame(
            {"event_id": ["E1"], "event_type": ["CPI"], "release_ts": ["2025-01-01T13:30:00Z"]},
        )
        event_market_map = pd.DataFrame({"event_type": ["CPI"], "market_id": ["M1"]})
        bad = pd.DataFrame({"market_id": ["M1"], "mean": [3.0]})  # missing ts
        with self.assertRaises(ValueError):
            build_event_feature_panel(
                macro_events=macro_events,
                event_market_map=event_market_map,
                kalshi_distributions=bad,
                config=EventFeatureConfig(snapshot_horizons=["15m"]),
            )


if __name__ == "__main__":
    unittest.main()

