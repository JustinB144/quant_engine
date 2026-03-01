# SPEC_AUDIT_FIX_39: Kalshi Quality Gating, Schema Hygiene & Infrastructure

**Priority:** MEDIUM — Dead code quality gates, incorrect diagnostics, missing resource cleanup, and schema fragility.
**Scope:** `kalshi/quality.py`, `kalshi/distribution.py`, `kalshi/walkforward.py`, `kalshi/storage.py`, `kalshi/mapping_store.py`
**Estimated effort:** 4–5 hours
**Depends on:** SPEC_13 (data integrity), SPEC_38 (contracts and security)
**Blocks:** Nothing

---

## Context

Three independent audits of Subsystem 9 (Kalshi / Event Markets) identified medium and low priority findings beyond what SPEC_13 and SPEC_38 cover. These relate to dead quality-gate code that undermines distribution integrity guarantees, incorrect post-repair diagnostic metrics, missing purge gaps in inner validation splits, absent schema migration infrastructure, resource leaks from unclosed database connections, and non-deterministic version resolution. While not causing runtime crashes, these issues degrade observability, weaken research reproducibility, and create maintenance hazards.

### Cross-Audit Traceability

| Finding | Auditor 1 (Claude) | Auditor 2 (Claude) | Auditor 3 (Codex) | Disposition |
|---------|--------------------|--------------------|-------------------|-------------|
| `passes_hard_gates` dead code | K-08 (P2) | — | F-03 (P2) | **NEW → T1** |
| Inner validation split lacks purge | K-10 (P2) | — | — | **NEW → T2** |
| `violated_constraints_post` incorrect | K-14 (P3) | — | F-02 (P2) | **NEW → T3** |
| No schema migration mechanism | — | F-04 (P2) | — | **NEW → T4** |
| No close()/context manager | — | F-05 (P2) | — | **NEW → T5** |
| Outcome table schemas identical | — | F-09 (P3) | — | **NEW → T6** |
| `current_version()` non-deterministic | — | — | F-05 (P3) | **NEW → T7** |

### Deferred Items (P3-LOW / INFO — Document Only)

The following items from the three audits are acknowledged but deferred. They have no functional impact at current scale or are purely cosmetic:

| ID | File | Description | Rationale for Deferral |
|----|------|-------------|----------------------|
| K-11 | storage.py:230 | `kalshi_fees` table has no writer method | Dead schema; no consumer. Remove or implement when fees are needed. |
| K-12 | storage.py:103,109 | Inconsistent PRAGMA quoting between DuckDB/SQLite | Cosmetic; both backends handle correctly. |
| K-13 | provider.py/storage.py | Duplicated timestamp normalization (`_to_iso_utc` / `_norm_ts`) | Technical debt; consolidate during next refactor. |
| K-15 | distribution.py:416-419 | Tail probability convention undocumented | Add docstring when distribution docs are updated. |
| K-16 | distribution.py:448-458 | Liquidity proxy unnormalized (depends on source count) | Correct for single-source; revisit if multi-source is added. |
| K-17 | quality.py:193-206 | `quality_as_feature_dict` dead code with stale TODO | Remove or wire in when quality-as-feature is needed. |
| K-19 | tests/ | `test_leakage.py` / `test_no_leakage.py` near-complete overlap | Consolidate during test suite cleanup. |
| K-04 | tests/ | Distribution test coverage inadequate (1 test for 936 lines) | Add tests incrementally with each distribution fix. |
| Aud2 F-07 | distribution.py:94-115 | PAV algorithm O(n^2) worst case | Negligible for current 5-30 contract scale. |
| Aud2 F-08 | distribution.py:71-82 | `_prob_from_mid` auto-detection threshold fragile at 1.5 | Edge case only; current markets are clearly in 0-1 or 5-95 range. |
| Aud2 F-06 | tests/ | Test coverage gaps for 7 critical paths | Addressed incrementally via SPEC_38/39 acceptance tests. |

---

## Tasks

### T1: Wire `passes_hard_gates` Into Distribution Snapshot Output

**Problem:** `quality.py:164-190` defines `passes_hard_gates()` with strict validity checks (coverage >= 0.8, quote age <= stale cutoff, median spread < 0.25), but it is never called by the distribution builder. `distribution.py:755,772` sets `quality_low=1` for low-quality distributions but still produces numeric moments. The documented contract states that hard-gate failures should produce NaN outputs, but consumers receive full numeric moments regardless of quality gate status.

**Files:** `kalshi/distribution.py`, `kalshi/quality.py`

**Implementation:**
1. In `build_distribution_snapshot()`, after computing quality dimensions, call `passes_hard_gates()`:
   ```python
   from .quality import passes_hard_gates

   # After computing quality_score and quality dimensions:
   hard_gate_pass = passes_hard_gates(
       coverage_ratio=coverage_ratio,
       quote_age_minutes=max_quote_age_minutes,
       stale_cutoff_minutes=stale_cutoff,
       median_spread=median_spread,
   )

   if not hard_gate_pass:
       # NaN-out statistical moments per hard-gate contract
       result["mean"] = np.nan
       result["std"] = np.nan
       result["skew"] = np.nan
       result["kurtosis"] = np.nan
       result["median"] = np.nan
       result["quality_low"] = 1
       result["hard_gate_failed"] = 1
       logger.debug(
           "Hard gate failed for %s at %s: coverage=%.2f, age=%.0fm, spread=%.3f",
           market_id, asof_ts, coverage_ratio, max_quote_age_minutes, median_spread,
       )
   ```
2. Add `hard_gate_failed` field to the snapshot schema (and to the `_EMPTY_SNAPSHOT_TEMPLATE` from SPEC_38 T6).
3. Downstream feature generation should treat `hard_gate_failed=1` rows as missing data (NaN features).
4. Add test: a distribution with coverage=0.5 returns NaN moments after hard-gate enforcement.

**Acceptance:** Distributions that fail hard quality gates have NaN statistical moments. `passes_hard_gates` is called for every distribution snapshot. A `hard_gate_failed` flag is included in the output.

---

### T2: Add Purge Gap to Inner Validation Split

**Problem:** `walkforward.py:394-399` performs an 80/20 temporal split for hyperparameter selection within each training fold, but there is no purge gap between the inner-train and inner-val sets. The last events of inner-train and first events of inner-val may have overlapping information periods, allowing the hyperparameter selection to overfit to temporally adjacent data.

**File:** `kalshi/walkforward.py`

**Implementation:**
1. After computing the inner split point, apply a purge gap:
   ```python
   train_events_sorted = train_events.sort_values("release_ts")
   inner_cut = int(max(5, round(len(train_events_sorted) * 0.8)))

   # Apply purge gap between inner-train and inner-val
   inner_purge_events = max(1, int(cfg.embargo_events) or 1)
   inner_cut_purged = max(inner_cut - inner_purge_events, 5)

   inner_train_ids = set(
       train_events_sorted.iloc[:inner_cut_purged]["event_id"].astype(str).tolist()
   )
   inner_val_ids = set(
       train_events_sorted.iloc[inner_cut:]["event_id"].astype(str).tolist()
   )
   ```
   This drops `inner_purge_events` events between the end of inner-train and start of inner-val.
2. If the purge gap reduces inner-train below the minimum viable size (5), skip the inner validation and use the middle alpha (default regularization):
   ```python
   if len(inner_train_ids) < 5 or not inner_val_ids:
       best_alpha = cfg.alphas[len(cfg.alphas) // 2]  # Default to middle alpha
       logger.debug("Insufficient data for inner validation; using default alpha=%.2f", best_alpha)
   ```
3. Log the purge gap size for each fold.

**Acceptance:** Inner-train and inner-val sets have at least 1 event gap between them. No temporally adjacent events span both sets.

---

### T3: Fix `violated_constraints_post` Metric

**Problem:** `distribution.py:804-806` sets `violated_constraints_post` to `monotonic_violations_pre` whenever `isotonic_l1 >= EPS` (i.e., whenever any repair was applied). This means the "post-repair" metric always equals the pre-repair count, reporting that violations still exist after isotonic repair. The correct behavior is that post-repair violations should always be 0 (isotonic regression produces a monotonic sequence by definition) or should be independently re-counted.

**File:** `kalshi/distribution.py`

**Implementation:**
1. Replace the current logic with a post-repair violation recount:
   ```python
   # After isotonic repair, count actual remaining violations
   if repaired_probs is not None:
       violations_post = 0
       for i in range(len(repaired_probs) - 1):
           if direction == "ge" and repaired_probs[i] < repaired_probs[i + 1]:
               violations_post += 1
           elif direction == "le" and repaired_probs[i] > repaired_probs[i + 1]:
               violations_post += 1
   else:
       violations_post = monotonic_violations_pre  # No repair attempted

   result["violated_constraints_post"] = int(violations_post)
   ```
2. In practice, isotonic regression guarantees monotonicity, so `violations_post` will always be 0 after successful repair. But counting explicitly is correct and self-documenting.
3. Alternatively, if the intent was simply "0 after repair, pre-count if no repair needed":
   ```python
   "violated_constraints_post": int(0 if isotonic_l1 >= _EPS else monotonic_violations_pre),
   ```
   Note: The current code has the condition inverted — it sets to 0 when `isotonic_l1 < _EPS` (no repair needed, which is wrong) and to `monotonic_violations_pre` when repair occurred (also wrong). Fix the logic regardless of approach.

**Acceptance:** After isotonic repair, `violated_constraints_post` reports the actual post-repair violation count (should be 0). When no repair was needed (`isotonic_l1 < EPS`), it reports `monotonic_violations_pre` (which should also be 0 or very small).

---

### T4: Add Schema Migration Mechanism for EventTimeStore

**Problem:** `storage.py:168-496` creates all 18 tables with `CREATE TABLE IF NOT EXISTS`. There is no schema version tracking, no `ALTER TABLE` logic, and no migration framework. The `_insert_or_replace()` method introspects existing table columns and only inserts columns present in both data and table, so adding new columns to schema code silently fails for existing databases. Any database created before a schema change becomes silently incompatible — writes succeed (new columns are skipped) but reads expecting new columns fail or return NULL.

**File:** `kalshi/storage.py`

**Implementation:**
1. Add a `schema_version` metadata table:
   ```python
   def _init_schema_version_table(self):
       self._execute("""
           CREATE TABLE IF NOT EXISTS _schema_version (
               version INTEGER NOT NULL,
               applied_at TEXT NOT NULL,
               description TEXT
           )
       """)
   ```
2. Define the current schema version and a migration registry:
   ```python
   CURRENT_SCHEMA_VERSION = 1

   _MIGRATIONS = {
       # version: (description, sql_statements)
       # Future migrations go here:
       # 2: ("Add ingestion_run_id to quotes", [
       #     "ALTER TABLE kalshi_quotes ADD COLUMN ingestion_run_id TEXT",
       # ]),
   }
   ```
3. In `init_schema()`, after creating tables, check and apply migrations:
   ```python
   def _get_schema_version(self) -> int:
       try:
           df = self.query_df("SELECT MAX(version) as v FROM _schema_version")
           return int(df["v"].iloc[0]) if len(df) and df["v"].iloc[0] is not None else 0
       except Exception:
           return 0

   def _apply_migrations(self):
       current = self._get_schema_version()
       for version in sorted(_MIGRATIONS.keys()):
           if version > current:
               desc, statements = _MIGRATIONS[version]
               for stmt in statements:
                   self._execute(stmt)
               self._execute(
                   "INSERT INTO _schema_version (version, applied_at, description) VALUES (?, ?, ?)",
                   [version, datetime.now().isoformat(), desc],
               )
               logger.info("Applied schema migration v%d: %s", version, desc)
   ```
4. Call `_init_schema_version_table()` and `_apply_migrations()` at the end of `init_schema()`.
5. Set initial version to `CURRENT_SCHEMA_VERSION` for new databases.

**Acceptance:** New databases have a `_schema_version` table with version 1. Existing databases get the version table added on next init. Adding a future migration (e.g., new column) applies automatically on startup.

---

### T5: Add `close()` and Context Manager to EventTimeStore

**Problem:** `storage.py:35-65` opens a DuckDB or SQLite connection in `__init__()` but provides no `close()` method, no `__enter__`/`__exit__`, and no `__del__`. Connections remain open indefinitely. For DuckDB, the connection holds an exclusive file lock, preventing concurrent access. For long-running processes or research notebooks that create multiple stores, this causes resource leaks and file locking errors.

**File:** `kalshi/storage.py`

**Implementation:**
1. Add `close()`, `__enter__`, `__exit__`, and `__del__`:
   ```python
   def close(self):
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

   def __enter__(self):
       return self

   def __exit__(self, exc_type, exc_val, exc_tb):
       self.close()
       return False

   def __del__(self):
       try:
           self.close()
       except Exception:
           pass
   ```
2. Update callers that create `EventTimeStore` instances to use context managers where appropriate. For `kalshi_service.py` (SPEC_38 T1), use:
   ```python
   with EventTimeStore(KALSHI_DB_PATH) as store:
       df = store.query_df(...)
   ```
3. For long-lived instances (e.g., in `KalshiProvider`), document that `close()` should be called in the teardown path.

**Acceptance:** `EventTimeStore` supports `with` statement. After `close()`, the database file lock is released. `__del__` provides safety-net cleanup.

---

### T6: Add Structural Differentiation to Outcome Table Schemas

**Problem:** `storage.py:276-301` — `event_outcomes` and `event_outcomes_first_print` have identical column definitions. Even after fixing the bidirectional upsert (SPEC_38 T2), there is no structural way to enforce that first-print rows are actually first prints. A revised outcome could be inserted into `event_outcomes_first_print` and nothing in the schema prevents it.

**File:** `kalshi/storage.py`

**Implementation:**
1. Add a `print_type` column to `event_outcomes_first_print` with a CHECK constraint:
   ```python
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
   ```
2. Add a `revision_number` column to `event_outcomes`:
   ```python
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
   ```
3. Update `upsert_event_outcomes_first_print()` to include `print_type='first_print'` in the payload.
4. Update `upsert_event_outcomes()` to accept and persist `revision_number` if provided.
5. Register this as a schema migration (coordinate with T4):
   ```python
   _MIGRATIONS = {
       2: ("Differentiate outcome table schemas", [
           "ALTER TABLE event_outcomes ADD COLUMN revision_number INTEGER DEFAULT 0",
           "ALTER TABLE event_outcomes_first_print ADD COLUMN print_type TEXT DEFAULT 'first_print'",
       ]),
   }
   ```

**Acceptance:** `event_outcomes_first_print` has a `print_type` column constrained to `'first_print'`. `event_outcomes` has a `revision_number` column. Inserting a non-first-print row into `event_outcomes_first_print` raises a constraint violation.

---

### T7: Fix `current_version()` Non-Deterministic Result

**Problem:** `mapping_store.py:58` takes `iloc[-1]` from as-of query results, but the query at `storage.py:633` orders by `event_id, market_id, mapping_version DESC` — this is not a global ordering by recency. For datasets with multiple events, `iloc[-1]` returns the last row of the last event/market group alphabetically, not the globally most recent version.

**File:** `kalshi/mapping_store.py`

**Implementation:**
1. Replace `iloc[-1]` with an explicit global max query:
   ```python
   def current_version(self, asof: Optional[str] = None) -> Optional[str]:
       """Return the globally latest mapping version as of the given timestamp."""
       if asof is None:
           asof = datetime.now().isoformat()
       df = self._store.query_df(
           """
           SELECT mapping_version, MAX(effective_start_ts) as latest_ts
           FROM event_market_map_versions
           WHERE effective_start_ts <= ?
             AND (effective_end_ts IS NULL OR effective_end_ts > ?)
           GROUP BY mapping_version
           ORDER BY latest_ts DESC
           LIMIT 1
           """,
           params=[asof, asof],
       )
       if df.empty:
           return None
       return str(df["mapping_version"].iloc[0])
   ```
2. Alternatively, if version strings follow the `v{N}` convention, use numeric extraction:
   ```python
   # After getting the as-of filtered DataFrame:
   df["_version_num"] = df["mapping_version"].str.extract(r"(\d+)").astype(int)
   return str(df.loc[df["_version_num"].idxmax(), "mapping_version"])
   ```
   This coordinates with SPEC_13 T7 (lexicographic version ordering fix).

**Acceptance:** `current_version()` returns the globally most recent mapping version, not an arbitrary row from the last alphabetical group. Result is deterministic regardless of event/market ordering.

---

## Verification

- [ ] Run `pytest kalshi/tests/ -v` — all pass
- [ ] Verify distributions failing hard gates have NaN moments and `hard_gate_failed=1`
- [ ] Verify inner validation split has purge gap between train and val sets
- [ ] Verify `violated_constraints_post` is 0 after successful isotonic repair
- [ ] Verify new databases have `_schema_version` table with correct version
- [ ] Verify `EventTimeStore` works as context manager and releases file lock on close
- [ ] Verify `event_outcomes_first_print` rejects non-first-print rows via CHECK constraint
- [ ] Verify `current_version()` returns globally latest version, not arbitrary last row
