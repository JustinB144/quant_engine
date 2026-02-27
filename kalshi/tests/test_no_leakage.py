"""
No-leakage test at panel level (Instructions I.4).

Builds a synthetic panel and asserts asof_ts < release_ts for every row.
This is critical: feature snapshots must never include information from
after the event release.
"""
import unittest

import numpy as np
import pandas as pd

from quant_engine.kalshi.events import EventFeatureConfig, build_event_feature_panel


class NoLeakageTests(unittest.TestCase):
    """Panel-level look-ahead bias detection."""

    def _build_synthetic_panel(self, n_events: int = 5) -> pd.DataFrame:
        """Build a synthetic panel to check for temporal leakage."""
        base_ts = pd.Timestamp("2025-01-01T08:00:00Z")
        macro_events = pd.DataFrame({
            "event_id": [f"E{i}" for i in range(n_events)],
            "event_type": ["CPI"] * n_events,
            "release_ts": [
                (base_ts + pd.Timedelta(days=i * 30, hours=5, minutes=30)).isoformat()
                for i in range(n_events)
            ],
        })

        mapping = pd.DataFrame({
            "event_type": ["CPI"],
            "market_id": ["M1"],
        })

        # Distribution snapshots at various times before each event
        dist_rows = []
        for i in range(n_events):
            release = base_ts + pd.Timedelta(days=i * 30, hours=5, minutes=30)
            for offset_min in [60, 30, 15, 5]:
                ts = release - pd.Timedelta(minutes=offset_min)
                dist_rows.append({
                    "market_id": "M1",
                    "ts": ts.isoformat(),
                    "mean": 3.0 + np.random.randn() * 0.1,
                    "var": 0.2,
                    "skew": 0.0,
                    "entropy": 1.0,
                    "quality_score": 0.9,
                })

        dists = pd.DataFrame(dist_rows)
        return build_event_feature_panel(
            macro_events=macro_events,
            event_market_map=mapping,
            kalshi_distributions=dists,
            config=EventFeatureConfig(snapshot_horizons=["15min", "1h"]),
        )

    def test_all_asof_before_release(self):
        """Every panel row must have asof_ts strictly before release_ts."""
        panel = self._build_synthetic_panel(n_events=5)
        if panel.empty:
            self.skipTest("Panel construction returned empty (dependencies missing)")

        df = panel.reset_index() if isinstance(panel.index, pd.MultiIndex) else panel
        self.assertIn("asof_ts", df.columns)
        self.assertIn("release_ts", df.columns)

        asof = pd.to_datetime(df["asof_ts"], utc=True)
        release = pd.to_datetime(df["release_ts"], utc=True)

        violations = df[asof >= release]
        self.assertEqual(
            len(violations), 0,
            f"Found {len(violations)} rows where asof_ts >= release_ts (look-ahead leak):\n"
            f"{violations[['asof_ts', 'release_ts']].head(10).to_string()}"
        )

    def test_single_event_no_leakage(self):
        """Even a single event must have asof_ts < release_ts."""
        macro_events = pd.DataFrame({
            "event_id": ["E1"],
            "event_type": ["CPI"],
            "release_ts": ["2025-06-15T13:30:00Z"],
        })
        mapping = pd.DataFrame({"event_type": ["CPI"], "market_id": ["M1"]})
        dists = pd.DataFrame({
            "market_id": ["M1"],
            "ts": ["2025-06-15T13:00:00Z"],
            "mean": [3.0],
            "var": [0.2],
            "skew": [0.0],
            "entropy": [1.0],
            "quality_score": [1.0],
        })

        panel = build_event_feature_panel(
            macro_events=macro_events,
            event_market_map=mapping,
            kalshi_distributions=dists,
            config=EventFeatureConfig(snapshot_horizons=["15min"]),
        )
        if panel.empty:
            self.skipTest("Panel empty")

        df = panel.reset_index() if isinstance(panel.index, pd.MultiIndex) else panel
        asof = pd.to_datetime(df["asof_ts"], utc=True)
        release = pd.to_datetime(df["release_ts"], utc=True)
        self.assertTrue(
            (asof < release).all(),
            "asof_ts must be strictly before release_ts for all rows"
        )


if __name__ == "__main__":
    unittest.main()
