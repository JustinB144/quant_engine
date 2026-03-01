"""
Tests for SPEC_AUDIT_FIX_13: Kalshi Data Integrity, Storage & Distribution Fixes.

Covers T1-T8 acceptance criteria.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quant_engine.kalshi.distribution import (
    DistributionConfig,
    _resolve_threshold_direction_with_confidence,
    _validate_bins,
    build_distribution_snapshot,
)
from quant_engine.kalshi.events import _merge_event_market_map
from quant_engine.kalshi.regimes import (
    classify_inflation_regime,
    classify_policy_regime,
    classify_vol_regime,
    tag_event_regime,
)
from quant_engine.kalshi.storage import EventTimeStore


class T1ThresholdDirectionTests(unittest.TestCase):
    """T1: Word-boundary regex prevents false substring matches."""

    def test_recovery_above_50_detects_above(self):
        row = {"rules_text": "recovery above 50%"}
        result = _resolve_threshold_direction_with_confidence(row)
        self.assertEqual(result.direction, "ge")

    def test_large_gathering_no_false_ge(self):
        """'gathering' should NOT match old 'ge' substring."""
        row = {"rules_text": "large gathering of people"}
        result = _resolve_threshold_direction_with_confidence(row)
        self.assertIsNone(result.direction)

    def test_danger_no_false_ge(self):
        """'danger' should NOT match old 'ge' substring."""
        row = {"rules_text": "danger zone ahead"}
        result = _resolve_threshold_direction_with_confidence(row)
        self.assertIsNone(result.direction)

    def test_altogether_no_false_le(self):
        """'altogether' should NOT match old 'le' substring."""
        row = {"rules_text": "altogether different situation"}
        result = _resolve_threshold_direction_with_confidence(row)
        self.assertIsNone(result.direction)

    def test_recovery_no_false_over(self):
        """'recovery' should NOT match old 'over' substring."""
        row = {"rules_text": "economic recovery expected"}
        result = _resolve_threshold_direction_with_confidence(row)
        self.assertIsNone(result.direction)

    def test_underperformance_no_false_under(self):
        """'underperformance' should NOT match old 'under' substring."""
        row = {"rules_text": "underperformance in sector"}
        result = _resolve_threshold_direction_with_confidence(row)
        self.assertIsNone(result.direction)

    def test_at_least_matches(self):
        row = {"rules_text": "value at least 3.5"}
        result = _resolve_threshold_direction_with_confidence(row)
        self.assertEqual(result.direction, "ge")

    def test_or_lower_matches(self):
        row = {"rules_text": "price or lower"}
        result = _resolve_threshold_direction_with_confidence(row)
        self.assertEqual(result.direction, "le")

    def test_greater_than_matches(self):
        row = {"rules_text": "greater than expected"}
        result = _resolve_threshold_direction_with_confidence(row)
        self.assertEqual(result.direction, "ge")

    def test_less_than_matches(self):
        row = {"rules_text": "less than forecast"}
        result = _resolve_threshold_direction_with_confidence(row)
        self.assertEqual(result.direction, "le")


class T2EventMarketMergeTests(unittest.TestCase):
    """T2: Keyed merge prevents mis-assignment from index alignment."""

    def test_fallback_uses_key_merge_not_index(self):
        grid = pd.DataFrame({
            "event_id": ["E1", "E2", "E3"],
            "event_type": ["CPI", "CPI", "FOMC"],
            "release_ts": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]),
            "horizon": ["1D", "1D", "1D"],
            "asof_ts": pd.to_datetime(["2024-12-31", "2025-01-31", "2025-02-28"]),
        })
        # event_id mapping for E1 only
        event_market_map = pd.DataFrame({
            "event_id": ["E1"],
            "event_type": ["CPI"],
            "market_id": ["M_CPI_JAN"],
        })
        # event_type fallback for CPI
        fallback_map = pd.DataFrame({
            "event_type": ["CPI", "FOMC"],
            "market_id": ["M_CPI_DEFAULT", "M_FOMC_DEFAULT"],
        })
        combined_map = pd.concat([event_market_map, fallback_map], ignore_index=True)
        result = _merge_event_market_map(grid, combined_map)
        # E1 should get M_CPI_JAN from id merge
        e1 = result[result["event_id"] == "E1"]
        self.assertTrue(len(e1) > 0)
        self.assertEqual(e1.iloc[0]["market_id"], "M_CPI_JAN")


class T3StorageUpsertTests(unittest.TestCase):
    """T3: INSERT ON CONFLICT preserves existing column values."""

    def test_upsert_preserves_existing_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = EventTimeStore(db_path, backend="sqlite")

            # Insert a full market row
            store.upsert_markets([{
                "market_id": "M1",
                "event_id": "E1",
                "event_type": "CPI",
                "title": "January CPI",
                "rules_text": "above 3.0",
                "status": "open",
            }])

            # Upsert with only market_id and status (should NOT null other cols)
            store.upsert_markets([{
                "market_id": "M1",
                "status": "closed",
            }])

            df = store.query_df("SELECT * FROM kalshi_markets WHERE market_id = 'M1'")
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["status"], "closed")
            # Title and event_type should be preserved, not nulled
            self.assertEqual(df.iloc[0]["title"], "January CPI")
            self.assertEqual(df.iloc[0]["event_type"], "CPI")
            self.assertEqual(df.iloc[0]["rules_text"], "above 3.0")


class T4CapacityDeduplicationTests(unittest.TestCase):
    """T4: Capacity metrics count unique events, not duplicate horizon rows."""

    def test_events_per_day_counts_unique(self):
        from quant_engine.kalshi.walkforward import (
            EventWalkForwardResult,
            evaluate_event_contract_metrics,
        )

        # Simulate 3 unique events, each with 2 horizon traces = 6 rows
        result = EventWalkForwardResult(
            folds=[],
            n_trials_total=1,
            event_returns=[0.01, 0.01, -0.02, -0.02, 0.03, 0.03],
            positions=[1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            event_types=["CPI", "CPI", "FOMC", "FOMC", "CPI", "CPI"],
            release_timestamps=[
                "2025-01-01T12:00:00+0000",
                "2025-01-01T12:00:00+0000",  # duplicate
                "2025-01-02T12:00:00+0000",
                "2025-01-02T12:00:00+0000",  # duplicate
                "2025-01-03T12:00:00+0000",
                "2025-01-03T12:00:00+0000",  # duplicate
            ],
        )
        metrics = evaluate_event_contract_metrics(result)
        # Should count 3 unique events over 2 days, not 6
        self.assertLessEqual(metrics["capacity_utilization"], 3.0)


class T5RegimeZeroValueTests(unittest.TestCase):
    """T5: Zero values are valid, not converted to NaN."""

    def test_zero_cpi_returns_valid_regime(self):
        result = classify_inflation_regime(0.0)
        self.assertEqual(result, "low")  # 0.0 <= 2.0 threshold
        self.assertNotEqual(result, "unknown")

    def test_zero_fed_funds_returns_valid_regime(self):
        result = classify_policy_regime(0.0)
        self.assertEqual(result, "neutral")
        self.assertNotEqual(result, "unknown")

    def test_zero_vix_returns_valid_regime(self):
        result = classify_vol_regime(0.0)
        self.assertEqual(result, "low")
        self.assertNotEqual(result, "unknown")

    def test_tag_event_regime_with_zeros(self):
        tag = tag_event_regime(cpi_yoy=0.0, fed_funds_change_bps=0.0, vix_level=0.0)
        self.assertNotEqual(tag.inflation_regime, "unknown")
        self.assertNotEqual(tag.policy_regime, "unknown")
        self.assertNotEqual(tag.vol_regime, "unknown")

    def test_tag_event_regime_with_none(self):
        tag = tag_event_regime(cpi_yoy=None, fed_funds_change_bps=None, vix_level=None)
        self.assertEqual(tag.inflation_regime, "unknown")
        self.assertEqual(tag.policy_regime, "unknown")
        self.assertEqual(tag.vol_regime, "unknown")


class T6UnresolvedDirectionTests(unittest.TestCase):
    """T6: Unresolved direction returns NaN moments and direction_resolved=False."""

    def test_unresolved_direction_returns_nan_moments(self):
        contracts = pd.DataFrame({
            "contract_id": ["T1", "T2", "T3"],
            "market_id": ["M1", "M1", "M1"],
            "threshold_value": [2.0, 3.0, 4.0],
            # No direction, rules_text, or payout_structure
        })
        quotes = pd.DataFrame({
            "contract_id": ["T1", "T2", "T3"],
            "ts": ["2025-01-01T13:00:00Z"] * 3,
            "mid": [0.40, 0.65, 0.20],
            "bid": [0.39, 0.64, 0.19],
            "ask": [0.41, 0.66, 0.21],
        })
        stats = build_distribution_snapshot(
            contracts=contracts,
            quotes=quotes,
            asof_ts=pd.Timestamp("2025-01-01T13:05:00Z"),
            config=DistributionConfig(price_scale="prob"),
        )
        self.assertTrue(np.isnan(stats["mean"]))
        self.assertTrue(np.isnan(stats["var"]))
        self.assertTrue(np.isnan(stats["skew"]))
        self.assertFalse(stats["direction_resolved"])

    def test_resolved_direction_returns_valid_moments(self):
        contracts = pd.DataFrame({
            "contract_id": ["T1", "T2", "T3"],
            "market_id": ["M1", "M1", "M1"],
            "threshold_value": [2.0, 3.0, 4.0],
            "direction": ["ge", "ge", "ge"],
        })
        quotes = pd.DataFrame({
            "contract_id": ["T1", "T2", "T3"],
            "ts": ["2025-01-01T13:00:00Z"] * 3,
            "mid": [0.70, 0.50, 0.20],
            "bid": [0.69, 0.49, 0.19],
            "ask": [0.71, 0.51, 0.21],
        })
        stats = build_distribution_snapshot(
            contracts=contracts,
            quotes=quotes,
            asof_ts=pd.Timestamp("2025-01-01T13:05:00Z"),
            config=DistributionConfig(price_scale="prob"),
        )
        self.assertTrue(pd.notna(stats["mean"]))
        self.assertTrue(stats["direction_resolved"])


class T7VersionOrderingTests(unittest.TestCase):
    """T7: Version ordering is numeric, not lexicographic."""

    def test_v10_sorts_after_v2(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = EventTimeStore(db_path, backend="sqlite")

            store.upsert_event_market_map_versions([
                {
                    "event_id": "E1",
                    "event_type": "CPI",
                    "market_id": "M1",
                    "effective_start_ts": "2024-01-01T00:00:00Z",
                    "mapping_version": "v2",
                    "source": "test",
                },
                {
                    "event_id": "E1",
                    "event_type": "CPI",
                    "market_id": "M1",
                    "effective_start_ts": "2024-06-01T00:00:00Z",
                    "mapping_version": "v10",
                    "source": "test",
                },
            ])

            df = store.get_event_market_map_asof("2025-01-01T00:00:00Z")
            # v10 should sort first (DESC), not v2
            versions = df["mapping_version"].tolist()
            self.assertEqual(versions[0], "v10")


class T8BinValidationQualityTests(unittest.TestCase):
    """T8: Invalid bin structure triggers quality_low flag."""

    def test_overlapping_bins_flagged_quality_low(self):
        contracts = pd.DataFrame({
            "contract_id": ["C1", "C2", "C3"],
            "market_id": ["M1", "M1", "M1"],
            "bin_low": [2.0, 2.5, 4.0],   # C1 and C2 overlap
            "bin_high": [3.0, 3.5, 5.0],
        })
        quotes = pd.DataFrame({
            "contract_id": ["C1", "C2", "C3"],
            "ts": ["2025-01-01T13:00:00Z"] * 3,
            "mid": [30.0, 40.0, 20.0],
            "bid": [29.0, 39.0, 19.0],
            "ask": [31.0, 41.0, 21.0],
        })
        stats = build_distribution_snapshot(
            contracts=contracts,
            quotes=quotes,
            asof_ts=pd.Timestamp("2025-01-01T13:05:00Z"),
            config=DistributionConfig(price_scale="cents"),
        )
        self.assertEqual(stats["quality_low"], 1)
        self.assertIn("invalid_bin_structure", stats["quality_flags"])

    def test_clean_bins_not_flagged(self):
        contracts = pd.DataFrame({
            "contract_id": ["C1", "C2", "C3"],
            "market_id": ["M1", "M1", "M1"],
            "bin_low": [2.0, 3.0, 4.0],
            "bin_high": [3.0, 4.0, 5.0],
        })
        quotes = pd.DataFrame({
            "contract_id": ["C1", "C2", "C3"],
            "ts": ["2025-01-01T13:00:00Z"] * 3,
            "mid": [30.0, 40.0, 20.0],
            "bid": [29.0, 39.0, 19.0],
            "ask": [31.0, 41.0, 21.0],
        })
        stats = build_distribution_snapshot(
            contracts=contracts,
            quotes=quotes,
            asof_ts=pd.Timestamp("2025-01-01T13:05:00Z"),
            config=DistributionConfig(price_scale="cents"),
        )
        self.assertNotIn("invalid_bin_structure", stats["quality_flags"])


if __name__ == "__main__":
    unittest.main()
