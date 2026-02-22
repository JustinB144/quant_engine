"""
Test module for panel split behavior and regressions.
"""

import unittest

import pandas as pd

from quant_engine.models.trainer import ModelTrainer


class PanelSplitTests(unittest.TestCase):
    """Test cases covering panel split behavior and system invariants."""
    def test_holdout_mask_uses_dates_not_raw_rows(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        idx = pd.MultiIndex.from_product([["AAPL", "MSFT"], dates], names=["ticker", "date"])
        dseries = ModelTrainer._extract_dates(idx)

        dev_mask, hold_mask = ModelTrainer._temporal_holdout_masks(
            dates=dseries,
            holdout_fraction=0.20,
            min_dev_rows=1,
            min_hold_rows=1,
        )

        hold_dates = set(dseries[hold_mask].dt.date)
        expected = {dates[-1].date(), dates[-2].date()}
        self.assertEqual(hold_dates, expected)

        dev_dates = set(dseries[dev_mask].dt.date)
        self.assertTrue(all(d < min(hold_dates) for d in dev_dates))

    def test_date_purged_folds_do_not_overlap(self):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        idx = pd.MultiIndex.from_product([["AAPL", "MSFT"], dates], names=["ticker", "date"])
        dseries = ModelTrainer._extract_dates(idx)

        folds = ModelTrainer._date_purged_folds(
            dates=dseries,
            n_folds=3,
            purge_gap=1,
            embargo=1,
        )
        self.assertGreater(len(folds), 0)

        for train_idx, test_idx in folds:
            train_dates = dseries.iloc[train_idx]
            test_dates = dseries.iloc[test_idx]
            self.assertTrue(train_dates.max() < test_dates.min())
            self.assertTrue(set(train_dates).isdisjoint(set(test_dates)))


if __name__ == "__main__":
    unittest.main()

