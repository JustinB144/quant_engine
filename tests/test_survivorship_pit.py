"""
Test module for survivorship pit behavior and regressions.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant_engine.data.survivorship import (
    filter_panel_by_point_in_time_universe,
    hydrate_universe_history_from_snapshots,
)


class SurvivorshipPointInTimeTests(unittest.TestCase):
    """Test cases covering survivorship pit behavior and system invariants."""
    def test_filter_panel_by_point_in_time_universe(self):
        snapshots = pd.DataFrame(
            {
                "date": [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-04-01",
                    "2024-04-01",
                ],
                "ticker": [
                    "AAA",
                    "BBB",
                    "BBB",
                    "CCC",
                ],
            },
        )

        idx = pd.MultiIndex.from_tuples(
            [
                ("AAA", pd.Timestamp("2024-01-15")),
                ("AAA", pd.Timestamp("2024-04-15")),
                ("BBB", pd.Timestamp("2024-01-15")),
                ("BBB", pd.Timestamp("2024-04-15")),
                ("CCC", pd.Timestamp("2024-01-15")),
                ("CCC", pd.Timestamp("2024-04-15")),
            ],
            names=["ticker", "date"],
        )
        panel = pd.DataFrame({"predicted_return": [1, 2, 3, 4, 5, 6]}, index=idx)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "survivorship.db"
            hydrate_universe_history_from_snapshots(
                snapshots=snapshots,
                universe_name="SP500",
                db_path=str(db_path),
            )
            filtered = filter_panel_by_point_in_time_universe(
                panel=panel,
                universe_name="SP500",
                db_path=str(db_path),
            )

        kept = set(filtered.index.tolist())
        expected = {
            ("AAA", pd.Timestamp("2024-01-15")),
            ("BBB", pd.Timestamp("2024-01-15")),
            ("BBB", pd.Timestamp("2024-04-15")),
            ("CCC", pd.Timestamp("2024-04-15")),
        }
        self.assertEqual(kept, expected)


if __name__ == "__main__":
    unittest.main()
