"""
Event-time storage layer for Kalshi + macro event research.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# T4 (SPEC_39): Schema version tracking and migration registry.
CURRENT_SCHEMA_VERSION = 2

_MIGRATIONS = {
    # version: (description, sql_statements)
    2: ("Differentiate outcome table schemas", [
        "ALTER TABLE event_outcomes ADD COLUMN revision_number INTEGER DEFAULT 0",
        "ALTER TABLE event_outcomes_first_print ADD COLUMN print_type TEXT DEFAULT 'first_print'",
    ]),
}


_TS_COLUMNS = {
    "ts",
    "asof_ts",
    "open_ts",
    "close_ts",
    "settle_ts",
    "release_ts",
    "effective_ts",
    "effective_start_ts",
    "effective_end_ts",
    "spec_version_ts",
    "version_ts",
    "known_at_ts",
    "inserted_at",
    "ingest_ts",
    "label_asof_ts",
    "learned_at_ts",
}


class EventTimeStore:
    """
    Intraday/event-time storage with a stable schema.

    Backends:
        - duckdb (preferred for research, if installed)
        - sqlite fallback
    """

    def __init__(self, db_path: Path | str, backend: str = "duckdb"):
        """Initialize EventTimeStore."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = str(backend).lower().strip()
        self._duckdb_conn = None
        self._sqlite_conn = None

        if self.backend == "duckdb":
            try:
                import duckdb  # type: ignore

                self._duckdb_conn = duckdb.connect(str(self.db_path))
            except ImportError:
                self.backend = "sqlite"

        if self.backend == "sqlite":
            self._sqlite_conn = sqlite3.connect(str(self.db_path))
            self._sqlite_conn.execute("PRAGMA journal_mode=WAL")
            self._sqlite_conn.execute("PRAGMA synchronous=NORMAL")

        self.init_schema()

    # T5 (SPEC_39): Resource management — close() and context manager support.
    def close(self) -> None:
        """Close the database connection and release locks."""
        if self._duckdb_conn is not None:
            try:
                self._duckdb_conn.close()
            except Exception:
                pass
            self._duckdb_conn = None
        if self._sqlite_conn is not None:
            try:
                self._sqlite_conn.close()
            except Exception:
                pass
            self._sqlite_conn = None

    def __enter__(self) -> "EventTimeStore":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _execute(self, sql: str, params: Optional[Iterable[object]] = None):
        """Internal helper for execute."""
        if self.backend == "duckdb" and self._duckdb_conn is not None:
            if params is None:
                return self._duckdb_conn.execute(sql)
            return self._duckdb_conn.execute(sql, params)

        if self._sqlite_conn is None:
            raise RuntimeError("Store connection is not initialized")

        cur = self._sqlite_conn.cursor()
        if params is None:
            cur.execute(sql)
        else:
            cur.execute(sql, list(params))
        self._sqlite_conn.commit()
        return cur

    def _executemany(self, sql: str, rows: List[Sequence[object]]):
        """Internal helper for executemany."""
        if not rows:
            return
        if self.backend == "duckdb" and self._duckdb_conn is not None:
            self._duckdb_conn.executemany(sql, rows)
            return

        if self._sqlite_conn is None:
            raise RuntimeError("Store connection is not initialized")

        cur = self._sqlite_conn.cursor()
        cur.executemany(sql, rows)
        self._sqlite_conn.commit()

    def _table_columns(self, table: str) -> List[str]:
        """Internal helper for table columns."""
        if self.backend == "duckdb" and self._duckdb_conn is not None:
            df = self._duckdb_conn.execute(f"PRAGMA table_info('{table}')").df()
            if "name" in df.columns:
                return [str(x) for x in df["name"].tolist()]
            return []
        if self._sqlite_conn is None:
            return []
        df = pd.read_sql_query(f"PRAGMA table_info({table})", self._sqlite_conn)
        if "name" not in df.columns:
            return []
        return [str(x) for x in df["name"].tolist()]

    def _get_primary_key_cols(self, table: str) -> List[str]:
        """Return primary key column names for a table (T3)."""
        if self.backend == "duckdb" and self._duckdb_conn is not None:
            df = self._duckdb_conn.execute(f"PRAGMA table_info('{table}')").df()
            if "name" in df.columns and "pk" in df.columns:
                return [str(r["name"]) for _, r in df.iterrows() if int(r["pk"]) > 0]
            return []
        if self._sqlite_conn is None:
            return []
        df = pd.read_sql_query(f"PRAGMA table_info({table})", self._sqlite_conn)
        if "name" not in df.columns or "pk" not in df.columns:
            return []
        return [str(r["name"]) for _, r in df.iterrows() if int(r["pk"]) > 0]

    @staticmethod
    def _norm_ts(value: object) -> Optional[str]:
        """Internal helper for norm ts."""
        if value is None:
            return None
        try:
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts.isoformat()
        except (ValueError, KeyError, TypeError):
            return None

    @staticmethod
    def _clean_value(key: str, value: object) -> object:
        """Internal helper for clean value."""
        if value is None:
            return None
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, sort_keys=True)
        k = str(key).lower().strip()
        if k in _TS_COLUMNS or k.endswith("_ts"):
            return EventTimeStore._norm_ts(value)
        if isinstance(value, bool):
            return int(value)
        return value

    def _insert_or_replace(self, table: str, rows: Iterable[Mapping[str, object]]) -> None:
        """Upsert rows using INSERT ... ON CONFLICT DO UPDATE (T3).

        Preserves existing column values not present in the incoming row,
        unlike INSERT OR REPLACE which nulls out missing columns.
        """
        payload = list(rows)
        if not payload:
            return

        table_cols = self._table_columns(table)
        if not table_cols:
            raise RuntimeError(f"Unknown table or columns unavailable: {table}")

        keys_present = {str(k) for row in payload for k in row.keys()}
        cols = [c for c in table_cols if c in keys_present]
        if not cols:
            return

        placeholders = ", ".join(["?"] * len(cols))
        col_sql = ", ".join(cols)

        # T3: Use ON CONFLICT DO UPDATE to avoid nulling out existing columns
        key_cols = self._get_primary_key_cols(table)
        if key_cols:
            update_cols = [c for c in cols if c not in key_cols]
            conflict_clause = ", ".join(key_cols)
            if update_cols:
                update_clause = ", ".join(f"{c} = excluded.{c}" for c in update_cols)
                sql = (
                    f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders}) "
                    f"ON CONFLICT({conflict_clause}) DO UPDATE SET {update_clause}"
                )
            else:
                # All columns are key columns — nothing to update
                sql = (
                    f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders}) "
                    f"ON CONFLICT({conflict_clause}) DO NOTHING"
                )
        else:
            # Fallback if no PK detected (should not happen with our schema)
            sql = f"INSERT OR REPLACE INTO {table} ({col_sql}) VALUES ({placeholders})"

        rows_out: List[Sequence[object]] = []
        for row in payload:
            rows_out.append(tuple(self._clean_value(c, row.get(c, None)) for c in cols))

        self._executemany(sql, rows_out)

    def init_schema(self):
        # Latest market state.
        """init schema."""
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS kalshi_markets (
                market_id TEXT PRIMARY KEY,
                event_id TEXT,
                event_type TEXT,
                title TEXT,
                rules_text TEXT,
                rules_hash TEXT,
                open_ts TEXT,
                close_ts TEXT,
                settle_ts TEXT,
                status TEXT,
                spec_version_ts TEXT,
                raw_market_json TEXT,
                inserted_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS kalshi_contracts (
                contract_id TEXT PRIMARY KEY,
                market_id TEXT NOT NULL,
                bin_low REAL,
                bin_high REAL,
                threshold_value REAL,
                payout_structure TEXT,
                direction TEXT,
                tick_size REAL,
                fee_bps REAL,
                status TEXT,
                spec_version_ts TEXT,
                raw_contract_json TEXT,
                inserted_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS kalshi_quotes (
                contract_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                bid REAL,
                ask REAL,
                mid REAL,
                last REAL,
                volume REAL,
                oi REAL,
                market_status TEXT,
                PRIMARY KEY (contract_id, ts)
            )
            """,
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS kalshi_fees (
                market_id TEXT,
                contract_id TEXT,
                effective_ts TEXT,
                fee_bps REAL,
                PRIMARY KEY (market_id, contract_id, effective_ts)
            )
            """,
        )

        # Authoritative latest macro calendar.
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS macro_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                release_ts TEXT NOT NULL,
                timezone TEXT,
                source TEXT,
                revision_rules TEXT,
                known_at_ts TEXT,
                version_ts TEXT
            )
            """,
        )

        # Versioned macro calendar snapshots.
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS macro_events_versioned (
                event_id TEXT NOT NULL,
                version_ts TEXT NOT NULL,
                event_type TEXT,
                release_ts TEXT,
                timezone TEXT,
                source TEXT,
                known_at_ts TEXT,
                revision_policy TEXT,
                raw_event_json TEXT,
                PRIMARY KEY (event_id, version_ts)
            )
            """,
        )

        # Legacy unified outcomes (kept for compatibility).
        # T6 (SPEC_39): revision_number differentiates from first_print.
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS event_outcomes (
                event_id TEXT NOT NULL,
                realized_value REAL,
                release_ts TEXT,
                asof_ts TEXT NOT NULL,
                source TEXT,
                learned_at_ts TEXT,
                revision_number INTEGER DEFAULT 0,
                PRIMARY KEY (event_id, asof_ts)
            )
            """,
        )

        # T6 (SPEC_39): print_type with CHECK constraint differentiates from event_outcomes.
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS event_outcomes_first_print (
                event_id TEXT NOT NULL,
                realized_value REAL,
                release_ts TEXT,
                asof_ts TEXT NOT NULL,
                source TEXT,
                learned_at_ts TEXT,
                print_type TEXT NOT NULL DEFAULT 'first_print'
                    CHECK(print_type = 'first_print'),
                PRIMARY KEY (event_id, asof_ts)
            )
            """,
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS event_outcomes_revised (
                event_id TEXT NOT NULL,
                realized_value REAL,
                release_ts TEXT,
                asof_ts TEXT NOT NULL,
                source TEXT,
                learned_at_ts TEXT,
                PRIMARY KEY (event_id, asof_ts)
            )
            """,
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS kalshi_distributions (
                market_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                spec_version_ts TEXT,
                mean REAL,
                var REAL,
                skew REAL,
                entropy REAL,
                quality_score REAL,
                coverage_ratio REAL,
                median_spread REAL,
                median_quote_age_seconds REAL,
                volume_oi_proxy REAL,
                constraint_violation_score REAL,
                tail_p_1 REAL,
                tail_p_2 REAL,
                tail_p_3 REAL,
                tail_threshold_1 REAL,
                tail_threshold_2 REAL,
                tail_threshold_3 REAL,
                tail_left_missing INTEGER,
                tail_right_missing INTEGER,
                mass_missing_estimate REAL,
                moment_truncated INTEGER,
                monotonic_violations_pre INTEGER,
                monotonic_violation_magnitude REAL,
                renorm_delta REAL,
                isotonic_l1 REAL,
                isotonic_l2 REAL,
                distance_kl_1h REAL,
                distance_js_1h REAL,
                distance_wasserstein_1h REAL,
                distance_kl_1d REAL,
                distance_js_1d REAL,
                distance_wasserstein_1d REAL,
                quality_low INTEGER,
                PRIMARY KEY (market_id, ts)
            )
            """,
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS event_market_map_versions (
                event_id TEXT NOT NULL,
                event_type TEXT,
                market_id TEXT NOT NULL,
                effective_start_ts TEXT NOT NULL,
                effective_end_ts TEXT,
                mapping_version TEXT NOT NULL,
                source TEXT,
                inserted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (event_id, market_id, effective_start_ts, mapping_version)
            )
            """,
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS kalshi_market_specs (
                market_id TEXT NOT NULL,
                spec_version_ts TEXT NOT NULL,
                rules_text TEXT,
                rules_hash TEXT,
                raw_market_json TEXT,
                source TEXT,
                PRIMARY KEY (market_id, spec_version_ts)
            )
            """,
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS kalshi_contract_specs (
                contract_id TEXT NOT NULL,
                spec_version_ts TEXT NOT NULL,
                market_id TEXT,
                bin_low REAL,
                bin_high REAL,
                threshold_value REAL,
                payout_structure TEXT,
                direction TEXT,
                status TEXT,
                raw_contract_json TEXT,
                PRIMARY KEY (contract_id, spec_version_ts)
            )
            """,
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS kalshi_data_provenance (
                market_id TEXT NOT NULL,
                asof_date TEXT NOT NULL,
                source_env TEXT,
                endpoint TEXT NOT NULL,
                ingest_ts TEXT,
                records_pulled INTEGER,
                notes TEXT,
                PRIMARY KEY (market_id, asof_date, endpoint)
            )
            """,
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS kalshi_coverage_diagnostics (
                market_id TEXT NOT NULL,
                asof_date TEXT NOT NULL,
                expected_contracts INTEGER,
                observed_contracts INTEGER,
                missing_fraction REAL,
                average_spread REAL,
                median_quote_age_seconds REAL,
                constraint_violations REAL,
                quality_score REAL,
                PRIMARY KEY (market_id, asof_date)
            )
            """,
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS kalshi_ingestion_logs (
                endpoint TEXT NOT NULL,
                asof_date TEXT NOT NULL,
                source_env TEXT,
                ingest_ts TEXT NOT NULL,
                records_pulled INTEGER,
                missing_markets INTEGER,
                missing_contracts INTEGER,
                missing_bins INTEGER,
                p95_quote_age_seconds REAL,
                error_count INTEGER,
                notes TEXT,
                PRIMARY KEY (endpoint, ingest_ts)
            )
            """,
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS kalshi_daily_health_report (
                asof_date TEXT PRIMARY KEY,
                markets_synced INTEGER,
                contracts_synced INTEGER,
                quotes_synced INTEGER,
                avg_missing_fraction REAL,
                avg_quality_score REAL,
                total_constraint_violations REAL,
                p95_quote_age_seconds REAL,
                ingestion_errors INTEGER,
                updated_at TEXT
            )
            """,
        )

        # Ingestion checkpoints for idempotent re-runs (H2).
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS kalshi_ingestion_checkpoints (
                market_id TEXT NOT NULL,
                asof_date TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                last_ingest_ts TEXT NOT NULL,
                records_ingested INTEGER,
                checksum TEXT,
                PRIMARY KEY (market_id, asof_date, endpoint)
            )
            """,
        )

        self._execute("CREATE INDEX IF NOT EXISTS idx_quotes_contract_ts ON kalshi_quotes(contract_id, ts)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_quotes_ts ON kalshi_quotes(ts)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_distributions_market_ts ON kalshi_distributions(market_id, ts)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_events_release ON macro_events(release_ts)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_map_window ON event_market_map_versions(effective_start_ts, effective_end_ts)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_ingestion_logs_date ON kalshi_ingestion_logs(asof_date)")

        # T4 (SPEC_39): Schema version tracking and migration.
        self._init_schema_version_table()
        self._apply_migrations()

    # T4 (SPEC_39): Schema migration infrastructure.
    def _init_schema_version_table(self) -> None:
        """Create the schema version tracking table."""
        self._execute("""
            CREATE TABLE IF NOT EXISTS _schema_version (
                version INTEGER NOT NULL,
                applied_at TEXT NOT NULL,
                description TEXT
            )
        """)

    def _get_schema_version(self) -> int:
        """Return the current schema version from the tracking table."""
        try:
            df = self.query_df("SELECT MAX(version) as v FROM _schema_version")
            if len(df) and df["v"].iloc[0] is not None:
                return int(df["v"].iloc[0])
            return 0
        except Exception:
            return 0

    def _apply_migrations(self) -> None:
        """Apply any pending schema migrations."""
        current = self._get_schema_version()
        if current == 0 and not _MIGRATIONS:
            # Brand-new database: stamp with current version.
            self._execute(
                "INSERT INTO _schema_version (version, applied_at, description) VALUES (?, ?, ?)",
                [CURRENT_SCHEMA_VERSION, datetime.now(timezone.utc).isoformat(), "Initial schema"],
            )
            return
        if current == 0 and _MIGRATIONS:
            # Existing database seeing migrations for the first time — stamp v1.
            self._execute(
                "INSERT INTO _schema_version (version, applied_at, description) VALUES (?, ?, ?)",
                [1, datetime.now(timezone.utc).isoformat(), "Initial schema"],
            )
            current = 1
        for version in sorted(_MIGRATIONS.keys()):
            if version > current:
                desc, statements = _MIGRATIONS[version]
                for stmt in statements:
                    try:
                        self._execute(stmt)
                    except Exception as exc:
                        # Column may already exist (re-open of migrated DB).
                        if "duplicate" in str(exc).lower() or "already" in str(exc).lower():
                            logger.debug("Migration v%d statement skipped (already applied): %s", version, exc)
                        else:
                            raise
                self._execute(
                    "INSERT INTO _schema_version (version, applied_at, description) VALUES (?, ?, ?)",
                    [version, datetime.now(timezone.utc).isoformat(), desc],
                )
                logger.info("Applied schema migration v%d: %s", version, desc)

    def upsert_markets(self, rows: Iterable[Mapping[str, object]]):
        """Upsert markets into storage."""
        self._insert_or_replace("kalshi_markets", rows)

    def upsert_contracts(self, rows: Iterable[Mapping[str, object]]):
        """Upsert contracts into storage."""
        self._insert_or_replace("kalshi_contracts", rows)

    def append_quotes(self, rows: Iterable[Mapping[str, object]]):
        """Append quotes to storage."""
        self._insert_or_replace("kalshi_quotes", rows)

    def upsert_macro_events(self, rows: Iterable[Mapping[str, object]]):
        """Upsert macro events into storage."""
        payload = list(rows)
        if not payload:
            return

        now_iso = datetime.now(timezone.utc).isoformat()
        latest_rows: List[Mapping[str, object]] = []
        version_rows: List[Mapping[str, object]] = []
        for row in payload:
            version_ts = row.get("version_ts") or row.get("known_at_ts") or now_iso
            latest_rows.append(
                {
                    "event_id": row.get("event_id"),
                    "event_type": row.get("event_type"),
                    "release_ts": row.get("release_ts"),
                    "timezone": row.get("timezone", "UTC"),
                    "source": row.get("source", ""),
                    "revision_rules": row.get("revision_rules", row.get("revision_policy", "")),
                    "known_at_ts": row.get("known_at_ts", version_ts),
                    "version_ts": version_ts,
                },
            )
            version_rows.append(
                {
                    "event_id": row.get("event_id"),
                    "version_ts": version_ts,
                    "event_type": row.get("event_type"),
                    "release_ts": row.get("release_ts"),
                    "timezone": row.get("timezone", "UTC"),
                    "source": row.get("source", ""),
                    "known_at_ts": row.get("known_at_ts", version_ts),
                    "revision_policy": row.get("revision_policy", row.get("revision_rules", "")),
                    "raw_event_json": row.get("raw_event_json", row),
                },
            )

        self._insert_or_replace("macro_events", latest_rows)
        self._insert_or_replace("macro_events_versioned", version_rows)

    def upsert_event_outcomes(self, rows: Iterable[Mapping[str, object]]):
        """Upsert event outcomes (latest/revised) into storage."""
        payload = list(rows)
        if not payload:
            return
        self._insert_or_replace("event_outcomes", payload)

    def upsert_event_outcomes_first_print(self, rows: Iterable[Mapping[str, object]]):
        """Upsert first-print event outcomes into storage.

        T6 (SPEC_39): Automatically sets ``print_type='first_print'`` on each row.
        """
        payload = list(rows)
        if not payload:
            return
        enriched = [{**row, "print_type": "first_print"} for row in payload]
        self._insert_or_replace("event_outcomes_first_print", enriched)

    def upsert_event_outcomes_revised(self, rows: Iterable[Mapping[str, object]]):
        """Upsert event outcomes revised into storage."""
        payload = list(rows)
        if not payload:
            return
        self._insert_or_replace("event_outcomes_revised", payload)

    def upsert_distributions(self, rows: Iterable[Mapping[str, object]]):
        """Upsert distributions into storage."""
        self._insert_or_replace("kalshi_distributions", rows)

    def upsert_event_market_map_versions(self, rows: Iterable[Mapping[str, object]]):
        """Upsert event market map versions into storage."""
        self._insert_or_replace("event_market_map_versions", rows)

    def append_market_specs(self, rows: Iterable[Mapping[str, object]]):
        """Append market specs to storage."""
        self._insert_or_replace("kalshi_market_specs", rows)

    def append_contract_specs(self, rows: Iterable[Mapping[str, object]]):
        """Append contract specs to storage."""
        self._insert_or_replace("kalshi_contract_specs", rows)

    def upsert_data_provenance(self, rows: Iterable[Mapping[str, object]]):
        """Upsert data provenance into storage."""
        self._insert_or_replace("kalshi_data_provenance", rows)

    def upsert_coverage_diagnostics(self, rows: Iterable[Mapping[str, object]]):
        """Upsert coverage diagnostics into storage."""
        self._insert_or_replace("kalshi_coverage_diagnostics", rows)

    def upsert_ingestion_logs(self, rows: Iterable[Mapping[str, object]]):
        """Upsert ingestion logs into storage."""
        self._insert_or_replace("kalshi_ingestion_logs", rows)

    def upsert_daily_health_reports(self, rows: Iterable[Mapping[str, object]]):
        """Upsert daily health reports into storage."""
        self._insert_or_replace("kalshi_daily_health_report", rows)

    def upsert_ingestion_checkpoints(self, rows: Iterable[Mapping[str, object]]):
        """Upsert ingestion checkpoints into storage."""
        self._insert_or_replace("kalshi_ingestion_checkpoints", rows)

    def get_ingestion_checkpoint(
        self, market_id: str, asof_date: str, endpoint: str
    ) -> Optional[pd.DataFrame]:
        """Return checkpoint row if this market/date/endpoint was already ingested."""
        sql = """
            SELECT market_id, asof_date, endpoint, last_ingest_ts,
                   records_ingested, checksum
            FROM kalshi_ingestion_checkpoints
            WHERE market_id = ? AND asof_date = ? AND endpoint = ?
        """
        df = self.query_df(sql, params=[market_id, asof_date, endpoint])
        return df if not df.empty else None

    def get_event_market_map_asof(self, asof_ts: object) -> pd.DataFrame:
        """Return event market map asof."""
        asof = self._norm_ts(asof_ts)
        if asof is None:
            return pd.DataFrame()
        # T7: Numeric version ordering — extract integer from 'vN' for correct sort
        sql = """
            SELECT event_id, event_type, market_id, mapping_version, source,
                   effective_start_ts, effective_end_ts
            FROM event_market_map_versions
            WHERE effective_start_ts <= ?
              AND (effective_end_ts IS NULL OR effective_end_ts > ?)
            ORDER BY event_id, market_id,
                     CAST(REPLACE(mapping_version, 'v', '') AS INTEGER) DESC
        """
        return self.query_df(sql, params=[asof, asof])

    def query_df(self, sql: str, params: Optional[Iterable[object]] = None) -> pd.DataFrame:
        """Query dataframe and return the result."""
        if self.backend == "duckdb" and self._duckdb_conn is not None:
            if params is None:
                return self._duckdb_conn.execute(sql).df()
            return self._duckdb_conn.execute(sql, params).df()

        if self._sqlite_conn is None:
            raise RuntimeError("Store connection is not initialized")

        if params is None:
            return pd.read_sql_query(sql, self._sqlite_conn)
        return pd.read_sql_query(sql, self._sqlite_conn, params=list(params))
