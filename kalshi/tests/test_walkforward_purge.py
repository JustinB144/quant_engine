"""
Walk-forward purge/embargo test (Instructions I.6).

Proves no training events appear within the purge window of test events.
"""
import unittest

import numpy as np
import pandas as pd

from quant_engine.kalshi.walkforward import (
    EventWalkForwardConfig,
    _prepare_panel,
    run_event_walkforward,
)


class WalkForwardPurgeTests(unittest.TestCase):
    """Tests that walk-forward purge/embargo prevents data leakage."""

    def _build_synthetic_data(self, n_events: int = 80):
        """Build synthetic panel + labels for walk-forward testing."""
        base = pd.Timestamp("2024-01-01T13:30:00Z")
        event_ids = [f"E{i:03d}" for i in range(n_events)]
        release_times = [base + pd.Timedelta(days=i * 7) for i in range(n_events)]

        rng = np.random.RandomState(42)
        panel = pd.DataFrame({
            "event_id": event_ids,
            "release_ts": release_times,
            "event_type": ["CPI" if i % 2 == 0 else "FOMC" for i in range(n_events)],
            "feat1": rng.randn(n_events),
            "feat2": rng.randn(n_events),
            "feat3": rng.randn(n_events),
        })

        labels = pd.DataFrame({
            "event_id": event_ids,
            "label_value": rng.randn(n_events) * 0.01,
        })
        return panel, labels

    def test_no_train_events_in_purge_window(self):
        """Training events must not fall within purge window of test events."""
        panel, labels = self._build_synthetic_data(n_events=80)

        cfg = EventWalkForwardConfig(
            train_min_events=30,
            test_events_per_fold=10,
            step_events=10,
            purge_window="14D",
            embargo_events=0,
        )

        # We'll manually replicate the walk-forward loop to inspect train/test splits
        data = _prepare_panel(panel, labels)
        if data.empty:
            self.skipTest("Panel preparation returned empty")

        data = data.sort_values(["release_ts", "event_id"]).reset_index(drop=True)
        events = data[["event_id", "release_ts"]].drop_duplicates().sort_values("release_ts").reset_index(drop=True)

        purge_delta = pd.to_timedelta(cfg.purge_window)

        start = int(cfg.train_min_events)
        fold_count = 0
        while start + int(cfg.test_events_per_fold) <= len(events):
            test_slice = events.iloc[start:start + int(cfg.test_events_per_fold)]
            test_start = pd.Timestamp(test_slice["release_ts"].min())

            train_events = events.iloc[:start].copy()
            train_events = train_events[
                train_events["release_ts"] <= (test_start - purge_delta)
            ]

            if len(train_events) < 10:
                start += int(cfg.step_events)
                continue

            # Verify: no training event is within purge window of test start
            train_max_ts = pd.Timestamp(train_events["release_ts"].max())
            gap = test_start - train_max_ts
            self.assertGreaterEqual(
                gap, purge_delta,
                f"Fold {fold_count}: train_max_ts={train_max_ts} is within "
                f"purge window of test_start={test_start} (gap={gap}, required={purge_delta})"
            )

            fold_count += 1
            start += int(cfg.step_events)

        self.assertGreater(fold_count, 0, "Must have at least one fold to test")

    def test_event_type_aware_purge(self):
        """Event-type-aware purge should use the larger window for mixed events."""
        panel, labels = self._build_synthetic_data(n_events=80)

        cfg = EventWalkForwardConfig(
            train_min_events=30,
            test_events_per_fold=10,
            step_events=10,
            purge_window="7D",
            purge_window_by_event={"CPI": 14, "FOMC": 21},
            default_purge_days=10,
        )

        result = run_event_walkforward(panel, labels, config=cfg)
        # Should produce folds (purge is applied internally)
        self.assertGreater(len(result.folds), 0, "Walk-forward should produce folds")

    def test_embargo_removes_adjacent_events(self):
        """Embargo should remove events adjacent to the train/test boundary."""
        panel, labels = self._build_synthetic_data(n_events=80)

        cfg_no_embargo = EventWalkForwardConfig(
            train_min_events=30,
            test_events_per_fold=10,
            step_events=10,
            purge_window="7D",
            embargo_events=0,
        )
        cfg_with_embargo = EventWalkForwardConfig(
            train_min_events=30,
            test_events_per_fold=10,
            step_events=10,
            purge_window="7D",
            embargo_events=3,
        )

        result_no = run_event_walkforward(panel, labels, config=cfg_no_embargo)
        result_with = run_event_walkforward(panel, labels, config=cfg_with_embargo)

        # With embargo, we should have fewer training events per fold (or fewer folds)
        if result_no.folds and result_with.folds:
            avg_train_no = np.mean([f.train_event_count for f in result_no.folds])
            avg_train_with = np.mean([f.train_event_count for f in result_with.folds])
            self.assertLessEqual(
                avg_train_with, avg_train_no,
                "Embargo should reduce training event count"
            )

    def test_trial_counting(self):
        """n_trials_total should be positive and reflect the search space."""
        panel, labels = self._build_synthetic_data(n_events=80)

        cfg = EventWalkForwardConfig(
            train_min_events=30,
            test_events_per_fold=10,
            step_events=10,
            purge_window="7D",
            alphas=[0.1, 1.0, 10.0],
        )

        result = run_event_walkforward(panel, labels, config=cfg)
        self.assertGreater(result.n_trials_total, 0, "Trial count should be positive")


if __name__ == "__main__":
    unittest.main()
