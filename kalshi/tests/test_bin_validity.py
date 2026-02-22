"""
Bin overlap/gap detection test (Instructions I.3).

Tests _validate_bins() with overlapping, gapped, and clean bin sets.
"""
import unittest

import pandas as pd

from quant_engine.kalshi.distribution import _validate_bins


class BinValidityTests(unittest.TestCase):
    """Tests for bin overlap/gap/ordering validation."""

    def test_clean_bins_valid(self):
        """Contiguous, non-overlapping, ordered bins should pass."""
        contracts = pd.DataFrame({
            "bin_low": [1.0, 2.0, 3.0, 4.0],
            "bin_high": [2.0, 3.0, 4.0, 5.0],
        })
        result = _validate_bins(contracts)
        self.assertTrue(result.valid)
        self.assertEqual(result.bin_overlap_count, 0)
        self.assertAlmostEqual(result.bin_gap_mass_estimate, 0.0)
        self.assertTrue(result.support_is_ordered)

    def test_overlapping_bins_detected(self):
        """Overlapping bins should be flagged."""
        contracts = pd.DataFrame({
            "bin_low": [1.0, 1.5, 3.0],
            "bin_high": [2.0, 3.0, 4.0],
        })
        result = _validate_bins(contracts)
        self.assertFalse(result.valid)
        self.assertGreater(result.bin_overlap_count, 0)

    def test_gapped_bins_detected(self):
        """Gaps between bins should be measured."""
        contracts = pd.DataFrame({
            "bin_low": [1.0, 3.0, 5.0],
            "bin_high": [2.0, 4.0, 6.0],
        })
        result = _validate_bins(contracts)
        self.assertGreater(result.bin_gap_mass_estimate, 0.0)
        # Gaps don't invalidate, but are measured
        self.assertTrue(result.support_is_ordered)

    def test_inverted_bin_detected(self):
        """bin_low >= bin_high should be flagged."""
        contracts = pd.DataFrame({
            "bin_low": [1.0, 3.0, 2.0],
            "bin_high": [2.0, 3.0, 4.0],
        })
        result = _validate_bins(contracts)
        self.assertFalse(result.valid)
        self.assertGreater(result.bin_overlap_count, 0)

    def test_single_bin_valid(self):
        """A single bin should always be valid."""
        contracts = pd.DataFrame({
            "bin_low": [1.0],
            "bin_high": [2.0],
        })
        result = _validate_bins(contracts)
        self.assertTrue(result.valid)
        self.assertEqual(result.bin_overlap_count, 0)

    def test_missing_columns_valid(self):
        """DataFrame without bin columns should return valid (no check needed)."""
        contracts = pd.DataFrame({"contract_id": ["C1", "C2"]})
        result = _validate_bins(contracts)
        self.assertTrue(result.valid)

    def test_empty_dataframe_valid(self):
        """Empty DataFrame should return valid."""
        contracts = pd.DataFrame({"bin_low": [], "bin_high": []})
        result = _validate_bins(contracts)
        self.assertTrue(result.valid)

    def test_unordered_bins_detected(self):
        """Bins in wrong order should have support_is_ordered=False after sorting."""
        # Note: _validate_bins sorts by bin_low, so this tests the sorted result
        contracts = pd.DataFrame({
            "bin_low": [3.0, 1.0, 2.0],
            "bin_high": [4.0, 2.0, 3.0],
        })
        result = _validate_bins(contracts)
        # After sorting by bin_low: [1,2], [2,3], [3,4] — should be valid
        self.assertTrue(result.valid)
        self.assertTrue(result.support_is_ordered)

    def test_severe_overlap(self):
        """Completely overlapping bins should be caught."""
        contracts = pd.DataFrame({
            "bin_low": [1.0, 1.0, 1.0],
            "bin_high": [5.0, 5.0, 5.0],
        })
        result = _validate_bins(contracts)
        self.assertFalse(result.valid)
        self.assertGreater(result.bin_overlap_count, 0)


if __name__ == "__main__":
    unittest.main()
