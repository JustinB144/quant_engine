"""
Tests for SPEC_AUDIT_FIX_39: Kalshi Quality Gating, Schema Hygiene & Infrastructure.
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
    _EMPTY_SNAPSHOT_TEMPLATE,
    build_distribution_snapshot,
)
from quant_engine.kalshi.mapping_store import EventMarketMappingStore
from quant_engine.kalshi.storage import CURRENT_SCHEMA_VERSION, EventTimeStore


class TestT1HardGateEnforcement(unittest.TestCase):
    """T1: Distributions failing hard gates should have NaN moments."""

    def test_low_coverage_triggers_hard_gate_failure(self):
        """A distribution with coverage=0.5 (below 0.8 gate) returns NaN moments."""
        # 6 contracts defined, only 3 have quotes => coverage=0.5
        contracts = pd.DataFrame(
            {
                "contract_id": ["C1", "C2", "C3", "C4", "C5", "C6"],
                "market_id": ["M1"] * 6,
                "bin_low": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "bin_high": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            },
        )
        quotes = pd.DataFrame(
            {
                "contract_id": ["C1", "C2", "C3"],
                "ts": ["2025-01-01T10:00:00Z"] * 3,
                "mid": [0.3, 0.4, 0.3],
            },
        )
        stats = build_distribution_snapshot(
            contracts=contracts,
            quotes=quotes,
            asof_ts=pd.Timestamp("2025-01-01T10:05:00Z"),
            config=DistributionConfig(price_scale="prob"),
            event_type="CPI",
        )
        self.assertEqual(stats["hard_gate_failed"], 1)
        self.assertTrue(np.isnan(stats["mean"]))
        self.assertTrue(np.isnan(stats["var"]))
        self.assertTrue(np.isnan(stats["skew"]))
        self.assertEqual(stats["quality_low"], 1)

    def test_good_coverage_passes_hard_gate(self):
        """A distribution with full coverage passes the hard gate."""
        contracts = pd.DataFrame(
            {
                "contract_id": ["C1", "C2", "C3"],
                "market_id": ["M1"] * 3,
                "bin_low": [1.0, 2.0, 3.0],
                "bin_high": [2.0, 3.0, 4.0],
            },
        )
        quotes = pd.DataFrame(
            {
                "contract_id": ["C1", "C2", "C3"],
                "ts": ["2025-01-01T10:00:00Z"] * 3,
                "mid": [0.2, 0.5, 0.3],
                "bid": [0.19, 0.49, 0.29],
                "ask": [0.21, 0.51, 0.31],
            },
        )
        stats = build_distribution_snapshot(
            contracts=contracts,
            quotes=quotes,
            asof_ts=pd.Timestamp("2025-01-01T10:05:00Z"),
            config=DistributionConfig(price_scale="prob"),
            event_type="CPI",
        )
        self.assertEqual(stats["hard_gate_failed"], 0)
        self.assertTrue(np.isfinite(stats["mean"]))

    def test_empty_snapshot_template_has_hard_gate_failed(self):
        """The empty snapshot template includes the hard_gate_failed field."""
        self.assertIn("hard_gate_failed", _EMPTY_SNAPSHOT_TEMPLATE)


class TestT3ViolatedConstraintsPost(unittest.TestCase):
    """T3: violated_constraints_post should be 0 after isotonic repair."""

    def test_post_repair_violations_are_zero(self):
        """After isotonic repair, violated_constraints_post should be 0."""
        # Create threshold contracts with monotonicity violations
        contracts = pd.DataFrame(
            {
                "contract_id": ["C1", "C2", "C3", "C4"],
                "market_id": ["M1"] * 4,
                "threshold_value": [1.0, 2.0, 3.0, 4.0],
                "direction": ["ge"] * 4,
            },
        )
        # P(X>=1)=0.8, P(X>=2)=0.9 is a violation (should decrease)
        quotes = pd.DataFrame(
            {
                "contract_id": ["C1", "C2", "C3", "C4"],
                "ts": ["2025-01-01T10:00:00Z"] * 4,
                "mid": [0.80, 0.90, 0.40, 0.10],
            },
        )
        stats = build_distribution_snapshot(
            contracts=contracts,
            quotes=quotes,
            asof_ts=pd.Timestamp("2025-01-01T10:05:00Z"),
            config=DistributionConfig(price_scale="prob"),
        )
        # There should be pre-repair violations
        self.assertGreater(stats["monotonic_violations_pre"], 0)
        # After isotonic repair, post-repair violations should be 0
        self.assertEqual(stats["violated_constraints_post"], 0)

    def test_no_repair_needed_preserves_pre_count(self):
        """When no repair is needed, violated_constraints_post equals pre count."""
        contracts = pd.DataFrame(
            {
                "contract_id": ["C1", "C2", "C3"],
                "market_id": ["M1"] * 3,
                "threshold_value": [1.0, 2.0, 3.0],
                "direction": ["ge"] * 3,
            },
        )
        # Already monotonically decreasing — no repair needed
        quotes = pd.DataFrame(
            {
                "contract_id": ["C1", "C2", "C3"],
                "ts": ["2025-01-01T10:00:00Z"] * 3,
                "mid": [0.90, 0.50, 0.10],
            },
        )
        stats = build_distribution_snapshot(
            contracts=contracts,
            quotes=quotes,
            asof_ts=pd.Timestamp("2025-01-01T10:05:00Z"),
            config=DistributionConfig(price_scale="prob"),
        )
        # No violations pre or post
        self.assertEqual(stats["monotonic_violations_pre"], 0)
        self.assertEqual(stats["violated_constraints_post"], 0)


class TestT4SchemaVersionMigration(unittest.TestCase):
    """T4: Schema version tracking and migration mechanism."""

    def test_new_database_has_schema_version(self):
        """A brand new database should have a _schema_version table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with EventTimeStore(db_path, backend="sqlite") as store:
                df = store.query_df("SELECT * FROM _schema_version")
                self.assertGreater(len(df), 0)
                max_version = int(df["version"].max())
                self.assertEqual(max_version, CURRENT_SCHEMA_VERSION)

    def test_migration_idempotent(self):
        """Re-opening the same database should not re-apply migrations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with EventTimeStore(db_path, backend="sqlite") as store:
                df1 = store.query_df("SELECT * FROM _schema_version")
            with EventTimeStore(db_path, backend="sqlite") as store:
                df2 = store.query_df("SELECT * FROM _schema_version")
            self.assertEqual(len(df1), len(df2))


class TestT5CloseAndContextManager(unittest.TestCase):
    """T5: EventTimeStore resource management."""

    def test_context_manager_closes_connection(self):
        """After exiting context, connections should be None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = EventTimeStore(db_path, backend="sqlite")
            self.assertIsNotNone(store._sqlite_conn)
            store.close()
            self.assertIsNone(store._sqlite_conn)

    def test_with_statement(self):
        """EventTimeStore should work as a context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with EventTimeStore(db_path, backend="sqlite") as store:
                df = store.query_df("SELECT * FROM kalshi_markets")
                self.assertIsNotNone(df)
            # After exiting, connection should be released
            self.assertIsNone(store._sqlite_conn)


class TestT6OutcomeTableDifferentiation(unittest.TestCase):
    """T6: Structural differentiation between outcome tables."""

    def test_first_print_has_print_type_column(self):
        """event_outcomes_first_print should have a print_type column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with EventTimeStore(db_path, backend="sqlite") as store:
                cols = store._table_columns("event_outcomes_first_print")
                self.assertIn("print_type", cols)

    def test_event_outcomes_has_revision_number(self):
        """event_outcomes should have a revision_number column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with EventTimeStore(db_path, backend="sqlite") as store:
                cols = store._table_columns("event_outcomes")
                self.assertIn("revision_number", cols)

    def test_first_print_upsert_adds_print_type(self):
        """upsert_event_outcomes_first_print should auto-set print_type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with EventTimeStore(db_path, backend="sqlite") as store:
                store.upsert_event_outcomes_first_print([
                    {"event_id": "E1", "realized_value": 3.5, "asof_ts": "2025-01-01T00:00:00Z"},
                ])
                df = store.query_df("SELECT * FROM event_outcomes_first_print WHERE event_id = 'E1'")
                self.assertEqual(len(df), 1)
                self.assertEqual(df["print_type"].iloc[0], "first_print")

    def test_first_print_check_constraint(self):
        """Inserting a non-first-print row into first_print table should fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with EventTimeStore(db_path, backend="sqlite") as store:
                with self.assertRaises(Exception):
                    store._execute(
                        "INSERT INTO event_outcomes_first_print "
                        "(event_id, asof_ts, print_type) VALUES (?, ?, ?)",
                        ["E1", "2025-01-01T00:00:00Z", "revised"],
                    )


class TestT7CurrentVersionDeterministic(unittest.TestCase):
    """T7: current_version() returns globally latest version."""

    def test_current_version_returns_latest(self):
        """current_version should return the most recent mapping version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with EventTimeStore(db_path, backend="sqlite") as store:
                mapping_store = EventMarketMappingStore(store)
                # Insert two versions for different events
                store.upsert_event_market_map_versions([
                    {
                        "event_id": "AAA",
                        "market_id": "M1",
                        "effective_start_ts": "2025-01-01T00:00:00Z",
                        "mapping_version": "v1",
                        "source": "test",
                    },
                    {
                        "event_id": "ZZZ",
                        "market_id": "M2",
                        "effective_start_ts": "2025-06-01T00:00:00Z",
                        "mapping_version": "v2",
                        "source": "test",
                    },
                ])
                version = mapping_store.current_version(asof="2025-12-01T00:00:00Z")
                # v2 has the later effective_start_ts, so it should be the current version
                self.assertEqual(version, "v2")

    def test_current_version_empty_store(self):
        """current_version should return 'v1' for an empty store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with EventTimeStore(db_path, backend="sqlite") as store:
                mapping_store = EventMarketMappingStore(store)
                version = mapping_store.current_version()
                self.assertEqual(version, "v1")


if __name__ == "__main__":
    unittest.main()
