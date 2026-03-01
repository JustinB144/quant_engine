# Audit Report: Subsystem 09 — Kalshi (Event Markets)

> **Status:** Complete
> **Auditor:** Claude Opus 4.6
> **Date:** 2026-02-28
> **Spec:** `docs/audit/subsystem_specs/SPEC_AUDIT_09_KALSHI.md`

---

## Coverage Summary

| Metric | Value |
|---|---|
| Files audited | 25 / 25 |
| Lines reviewed | 6,096 / 6,096 |
| Production files | 16 |
| Test files | 9 |
| Cross-subsystem boundaries verified | 6 / 6 |
| DuckDB tables inventoried | 18 / 18 |
| Indexes inventoried | 6 / 6 |

---

## Task Completion Matrix

| Task | Description | Status |
|---|---|---|
| T1 | Ledger and schema baseline | COMPLETE |
| T2 | Leakage/as-of correctness pass | COMPLETE |
| T3 | Distribution and quote-quality pass | COMPLETE |
| T4 | Walk-forward and promotion contract pass | COMPLETE |
| T5 | Boundary and integration pass | COMPLETE |
| T6 | Findings synthesis and closure | COMPLETE |

---

## Findings (Severity-Ranked)

### F-01 — Bidirectional outcome upsert makes first_print/revised separation unreliable [P1 HIGH]

- **File:** `kalshi/storage.py:550-564`
- **Proof:**
  ```python
  def upsert_event_outcomes(self, rows):
      self._insert_or_replace("event_outcomes", payload)
      self._insert_or_replace("event_outcomes_first_print", payload)  # <-- also writes to first_print

  def upsert_event_outcomes_first_print(self, rows):
      self._insert_or_replace("event_outcomes_first_print", payload)
      self._insert_or_replace("event_outcomes", payload)  # <-- also writes to event_outcomes
  ```
  Both methods write to BOTH `event_outcomes` AND `event_outcomes_first_print` tables. The tables have identical schemas (lines 276-301). This means:
  - Calling `upsert_event_outcomes()` with revised data silently overwrites first_print rows.
  - Calling `upsert_event_outcomes_first_print()` with first-print data silently overwrites the unified outcomes table.
  - The two tables are always identical mirrors — the separation is illusory.
- **Impact:** `build_event_labels()` (`events.py:360-441`) supports `label_mode="first_print"` vs `"latest"` and reads from different tables (lines 399-403). Since both tables contain identical data due to the bidirectional sync, the label mode distinction has no effect. Revision-aware label analysis (first_print_value vs revised_print_value at lines 414-423) will always return the same values for both. This undermines the entire first-print vs revised outcome design for event strategy research.
- **Recommendation:** Remove the cross-table write from each method. `upsert_event_outcomes()` should write ONLY to `event_outcomes`. `upsert_event_outcomes_first_print()` should write ONLY to `event_outcomes_first_print`. The caller should decide which table to populate.

### F-02 — OpenSSL passphrase exposed via command-line arguments [P1 HIGH]

- **File:** `kalshi/client.py:222-223, 241-242`
- **Proof:**
  ```python
  if self.passphrase:
      cmd.extend(["-passin", f"pass:{self.passphrase}"])
  ```
  This passes the private key passphrase as a plaintext command-line argument to `subprocess.run()`. On Unix systems, command-line arguments are visible to all users via `ps aux`, `/proc/PID/cmdline`, or process monitoring tools. The same pattern appears in both the `openssl dgst` path (line 223) and the `openssl pkeyutl` fallback (line 242).
- **Mitigating factors:** This is the OpenSSL subprocess fallback — the primary path uses the `cryptography` library for in-process signing (line 265-270), which does not expose the passphrase. The fallback triggers only when `cryptography` is not installed or raises an exception.
- **Impact:** If the `cryptography` library is unavailable or fails, the passphrase is exposed to the process table. In a shared-host or containerized environment this is a credential leak vector.
- **Recommendation:** Use `-passin stdin` or `-passin fd:N` (pipe the passphrase via stdin or a file descriptor) instead of `-passin pass:`. Alternatively, use `env:VAR_NAME` to pass via environment variable (less ideal but better than command line).

### F-03 — Private key written to temp file during OpenSSL fallback [P2 MEDIUM]

- **File:** `kalshi/client.py:277-287`
- **Proof:**
  ```python
  with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
      tf.write(str(self.private_key_pem))
      tf.flush()
      tmp_path = tf.name
  try:
      raw_sig = self._sign_with_openssl(payload, tmp_path)
  finally:
      try:
          os.unlink(tmp_path)
      except OSError:
          pass
  ```
  When `private_key_path` is empty but `private_key_pem` is set, the PEM key is written to a `NamedTemporaryFile` with `delete=False`. During the window between write and `os.unlink()`, the key file is readable by any process with filesystem access. The temp file uses default permissions (typically `0o600` on Unix, but this varies by umask).
- **Impact:** Brief window of key material exposure on the filesystem. Combined with F-02 (passphrase in args), both the key file and passphrase are simultaneously exposed during the fallback signing window.
- **Recommendation:** Use `tempfile.mkstemp()` with explicit `os.fchmod(fd, 0o600)` and a shorter lifetime, or pass the PEM via stdin to OpenSSL using `-inkey /dev/stdin`.

### F-04 — No schema migration mechanism for EventTimeStore [P2 MEDIUM]

- **File:** `kalshi/storage.py:168-496`
- **Proof:** All 18 tables use `CREATE TABLE IF NOT EXISTS`. There is no schema version tracking table, no `ALTER TABLE` logic, and no migration framework. The `_insert_or_replace()` method (line 143-166) introspects existing table columns at runtime and only inserts columns present in both the data and the table — so adding new columns to the schema code will silently fail for existing databases.

  Example: if a new column `ingestion_run_id` were added to `kalshi_quotes`, existing databases would have the old schema. New inserts would succeed (the new column is silently skipped) but queries expecting the new column would fail or return NULL.
- **Impact:** Any database created before a schema change becomes silently incompatible. The `_insert_or_replace` introspection provides graceful degradation on writes, but reads will fail or return incomplete data. There is no way to know which schema version an existing database has.
- **Recommendation:** Add a `schema_version` metadata table with a version counter. On `init_schema()`, check the version and run `ALTER TABLE ADD COLUMN` migrations as needed. This is especially important because `kalshi_distributions` already has 35 columns and is likely to grow.

### F-05 — No `close()` method or context manager on EventTimeStore [P2 MEDIUM]

- **File:** `kalshi/storage.py:35-65`
- **Proof:** `EventTimeStore.__init__()` opens a DuckDB or SQLite connection and stores it as `self._duckdb_conn` or `self._sqlite_conn`. There is no `close()` method, no `__enter__`/`__exit__`, and no `__del__` cleanup. The connection is never explicitly closed.
- **Impact:** In long-running processes, database connections remain open indefinitely. For DuckDB in particular, the connection holds an exclusive lock on the database file, preventing concurrent access. For SQLite with WAL mode, this is less severe but still a resource leak. In research notebooks or pipeline scripts that create multiple stores, this can cause file locking errors.
- **Recommendation:** Add a `close()` method and `__enter__`/`__exit__` context manager protocol. Consider adding `__del__` as a safety net.

### F-06 — Test coverage gaps for critical paths [P2 MEDIUM]

- **Files:** Multiple test files
- **Proof:** The following critical paths have no test coverage:
  1. **DuckDB backend:** All tests implicitly use SQLite (DuckDB import is conditional). No test verifies DuckDB-specific behavior or SQL dialect compatibility.
  2. **OpenSSL subprocess fallback:** `test_signature_kat.py` tests the `cryptography` library path only. The `_sign_with_openssl()` method (client.py:209-248) and its two fallback paths (`dgst` then `pkeyutl`) are untested.
  3. **`upsert_event_outcomes` cross-table behavior:** No test verifies that first_print and event_outcomes are kept separate (related to F-01).
  4. **`build_event_labels` revision mode:** No test verifies `label_mode="latest"` with revised outcomes or the first_print_fallback path (events.py:406-408).
  5. **`passes_hard_gates()`:** quality.py:164-190 defines hard validity gates (C1) but no test exercises them.
  6. **`quality_as_feature_dict()`:** quality.py:193-206 converts quality to feature columns (C2) but no test exists.
  7. **`assert_consistent_mapping_version()`:** mapping_store.py:66-75 (D1 safety check) has no test.
- **Impact:** Critical contract boundaries (first_print vs revised, hard quality gates, mapping version consistency) lack regression protection. Changes to these paths could silently break correctness.
- **Recommendation:** Add targeted tests for each gap. Priority: items 3 and 4 (directly related to F-01), then items 1 and 5.

### F-07 — `_isotonic_nonincreasing` PAV algorithm has O(n^2) worst case [P3 LOW]

- **File:** `kalshi/distribution.py:94-115`
- **Proof:** The pool-adjacent-violators implementation has a nested while loop:
  ```python
  while i < len(values) - 1:
      if values[i] < values[i + 1]:
          j = i
          while j >= 0 and values[j] < values[j + 1]:  # inner loop walks backwards
              ...
              j -= 1
          i = max(j, 0)
      else:
          i += 1
  ```
  For a fully reversed input array of length n, the inner loop walks back to 0 on each violation, giving O(n^2) time.
- **Impact:** For typical Kalshi markets with 5-30 contracts, this is negligible. For markets with 100+ contracts or in batch processing, it could slow down distribution reconstruction.
- **Recommendation:** No action required for current scale. Document the complexity bound. If Kalshi adds markets with many more contracts, consider switching to the standard O(n) PAV using a stack.

### F-08 — `_prob_from_mid` auto-detection threshold is fragile [P3 LOW]

- **File:** `kalshi/distribution.py:71-82`
- **Proof:**
  ```python
  if x > 1.5:
      return np.clip(x / 100.0, 0.0, 1.0)
  return np.clip(x, 0.0, 1.0)
  ```
  The `"auto"` price scale mode uses `x > 1.5` to distinguish cents (0-100) from probability (0-1). A mid price of exactly 1.5 is treated as probability, while 1.51 is treated as cents. This boundary could misclassify edge cases, particularly for very illiquid contracts with high bid-ask spreads.
- **Impact:** Edge case only — most Kalshi mid prices are clearly in either 0-1 (probability) or 5-95 (cents) ranges. Miscategorization would produce incorrect probabilities for affected contracts.
- **Recommendation:** Consider using per-market scale detection (majority vote across all contracts in a market) rather than per-contract detection. Or require explicit `price_scale` configuration to eliminate ambiguity.

### F-09 — `event_outcomes` and `event_outcomes_first_print` have identical schemas [P3 LOW]

- **File:** `kalshi/storage.py:276-301`
- **Proof:** Both tables have identical column definitions:
  ```sql
  event_id TEXT NOT NULL, realized_value REAL, release_ts TEXT,
  asof_ts TEXT NOT NULL, source TEXT, learned_at_ts TEXT,
  PRIMARY KEY (event_id, asof_ts)
  ```
  There is no `is_first_print` flag, no `revision_number`, and no `supersedes` reference. Combined with F-01 (bidirectional sync), the tables serve no differentiation purpose.
- **Impact:** Even after fixing F-01, the identical schemas mean there's no structural way to enforce that first_print rows are actually first prints. A row with a later `learned_at_ts` could be inserted into `event_outcomes_first_print` and there's nothing in the schema to prevent it.
- **Recommendation:** After fixing F-01, consider adding a `print_number INTEGER DEFAULT 1` or `is_revision BOOLEAN DEFAULT FALSE` column to distinguish rows, or add a check constraint.

---

## T1: Ledger and Schema Baseline

### Table Schema Inventory (18 tables)

| # | Table | Primary Key | Columns | Purpose |
|---|---|---|---|---|
| 1 | `kalshi_markets` | `market_id` | 13 | Latest market state |
| 2 | `kalshi_contracts` | `contract_id` | 13 | Contract specs (bins/thresholds) |
| 3 | `kalshi_quotes` | `(contract_id, ts)` | 8 | Intraday quote time series |
| 4 | `kalshi_fees` | `(market_id, contract_id, effective_ts)` | 4 | Fee schedule history |
| 5 | `macro_events` | `event_id` | 8 | Authoritative macro calendar |
| 6 | `macro_events_versioned` | `(event_id, version_ts)` | 9 | Calendar version snapshots |
| 7 | `event_outcomes` | `(event_id, asof_ts)` | 6 | Legacy unified outcomes |
| 8 | `event_outcomes_first_print` | `(event_id, asof_ts)` | 6 | First-print outcomes |
| 9 | `event_outcomes_revised` | `(event_id, asof_ts)` | 6 | Revised outcomes |
| 10 | `kalshi_distributions` | `(market_id, ts)` | 35 | Distribution snapshots |
| 11 | `event_market_map_versions` | `(event_id, market_id, effective_start_ts, mapping_version)` | 8 | Versioned event-to-market map |
| 12 | `kalshi_market_specs` | `(market_id, spec_version_ts)` | 6 | Market spec version history |
| 13 | `kalshi_contract_specs` | `(contract_id, spec_version_ts)` | 10 | Contract spec version history |
| 14 | `kalshi_data_provenance` | `(market_id, asof_date, endpoint)` | 7 | Ingestion provenance |
| 15 | `kalshi_coverage_diagnostics` | `(market_id, asof_date)` | 8 | Coverage health per market/day |
| 16 | `kalshi_ingestion_logs` | `(endpoint, ingest_ts)` | 11 | Per-endpoint ingestion log |
| 17 | `kalshi_daily_health_report` | `asof_date` | 10 | Daily aggregate health |
| 18 | `kalshi_ingestion_checkpoints` | `(market_id, asof_date, endpoint)` | 6 | Idempotent re-run checkpoints (H2) |

### Index Inventory (6 indexes)

| Index | Table | Columns |
|---|---|---|
| `idx_quotes_contract_ts` | `kalshi_quotes` | `(contract_id, ts)` |
| `idx_quotes_ts` | `kalshi_quotes` | `(ts)` |
| `idx_distributions_market_ts` | `kalshi_distributions` | `(market_id, ts)` |
| `idx_events_release` | `macro_events` | `(release_ts)` |
| `idx_map_window` | `event_market_map_versions` | `(effective_start_ts, effective_end_ts)` |
| `idx_ingestion_logs_date` | `kalshi_ingestion_logs` | `(asof_date)` |

### Storage/Provider API

The `EventTimeStore` exposes 14 write methods and 3 read methods. `KalshiProvider` wraps these with API sync logic and imports 18 `KALSHI_*` config constants from `config.py`. The provider constructs `DistributionConfig` from config constants in `compute_and_store_distributions()` (provider.py:507-519).

**Schema stability verdict:** Schema is stable for current use but lacks migration safety (F-04). No DDL mutations occur at runtime. The `_insert_or_replace` pattern with column introspection provides forward-compatible writes.

---

## T2: Leakage/As-of Correctness Pass

### Leakage Guard Analysis

| Guard | Location | Mechanism | Verdict |
|---|---|---|---|
| `_ensure_asof_before_release()` | events.py:35-41 | Raises `ValueError` if `asof_ts >= release_ts` | PASS — strict, no forward-peeking |
| `asof_join()` | events.py:44-76 | `pd.merge_asof(direction="backward")` | PASS — backward-only join |
| `build_event_snapshot_grid()` | events.py:79-109 | `asof_ts = release_ts - delta` always before release | PASS — horizons are pre-release |
| `build_event_feature_panel()` | events.py:228-322 | Calls `_ensure_asof_before_release(joined)` on output | PASS — post-join guard |
| `build_asset_time_feature_panel()` | events.py:325-356 | `pd.merge_asof(direction="backward")` | PASS — backward-only |
| `add_reference_disagreement_features()` | events.py:186-225 | Uses `asof_join()` with backward direction | PASS — uses shared backward join |
| `add_options_disagreement_features()` | options.py:72-145 | `pd.merge_asof(direction="backward")` | PASS — backward-only |
| `_latest_quotes_asof()` | distribution.py:273-290 | `q["ts"] <= asof_ts` filter | PASS — no future quotes |

### Test Alignment

| Test | What it validates | Verdict |
|---|---|---|
| `test_no_leakage.py` | `asof_ts < release_ts` for all panel rows across 5 events | PASS — 2 tests |
| `test_leakage.py` | Feature rows are strictly pre-release | PASS — 1 test |
| `test_walkforward_purge.py` | Purge/embargo respects temporal boundaries | PASS — 3 tests |

**Leakage verdict:** All feature construction paths use strict backward joins with `direction="backward"`. The `_ensure_asof_before_release()` guard is applied after both grid construction and feature panel merging. No forward-looking data enters the feature pipeline. Tests confirm this behavior.

---

## T3: Distribution and Quote-Quality Pass

### Threshold Direction Resolution

The 3-tier direction resolution (`distribution.py:141-184`) implements:
1. **Tier 1 (high confidence):** Explicit `direction` or `payout_structure` metadata
2. **Tier 2 (medium confidence):** Rules text token matching
3. **Tier 3 (low confidence):** Title/subtitle guessing

When direction is unknown (`confidence="low"`), `quality_low` is set to 1 (`distribution.py:762-763`), flagging the distribution as unreliable. This is the correct behavior.

**Test coverage:** `test_threshold_direction.py` covers 16 cases across all 3 tiers. PASS.

### Monotonicity Repair

- `_isotonic_nonincreasing()` (distribution.py:94-115): Pool-adjacent-violators for GE contracts.
- `_isotonic_nondecreasing()` (distribution.py:118-120): Reversal wrapper for LE contracts.
- Pre-repair violation counts and magnitude are recorded as features (`monotonic_violations_pre`, `monotonic_violation_magnitude`).
- Post-repair metrics (`isotonic_l1`, `isotonic_l2`) quantify the cleaning magnitude.
- B1 (direction confidence), B2 (bin validity), B3 (cleaning magnitude) metadata are all included in the snapshot output (distribution.py:793-809).

**Test coverage:** `test_distribution.py` tests mass normalization. `test_bin_validity.py` tests 8 bin validation cases including overlaps, gaps, inversions, and edge cases. PASS.

### Dynamic Stale Quote Cutoff

`dynamic_stale_cutoff_minutes()` (quality.py:56-101) implements:
- Time-to-event linear interpolation between near (2 min cutoff at ≤30 min to event) and far (60 min cutoff at ≥24h to event)
- Market-type multiplier (CPI 0.80, FOMC 0.70, UNEMPLOYMENT 0.90)
- Liquidity adjustment (low liquidity → 1.35x tolerance, high liquidity → 0.80x)
- Min/max clamping (0.5 min to 1440 min)

**Test coverage:** `test_stale_quotes.py` covers 10 cases: near/far event, interpolation, monotonicity, market types, liquidity, bounds. PASS.

### Quality Model

`compute_quality_dimensions()` (quality.py:104-161) weights:
- 0.35 coverage_ratio + 0.20 spread_score + 0.20 age_score + 0.10 volume_oi_proxy + 0.15 constraint_violation_score

`passes_hard_gates()` (quality.py:164-190) enforces C1 validity checks:
- coverage ≥ 0.8, quote age ≤ stale cutoff, median spread < 0.25

`quality_as_feature_dict()` (quality.py:193-206) exposes soft quality dimensions as C2 feature columns.

**Note:** Hard gates and quality-as-feature functions are untested (see F-06).

**Distribution verdict:** Distribution reconstruction is correct. Monotonicity repair, bin validation, direction resolution, and quality scoring are well-structured. Distance metrics (KL, JS, Wasserstein) are computed correctly via common grid interpolation. The main gap is test coverage for hard gates and quality-to-feature conversion.

---

## T4: Walk-Forward and Promotion Contract Pass

### Walk-Forward Evaluation

`run_event_walkforward()` (walkforward.py:315-492) implements:
- Purge/embargo-aware temporal splits with configurable purge window
- **E3:** Event-type-aware purge windows (`purge_window_by_event` dict, line 364-374)
- Nested alpha selection via ridge regression within training folds
- **E2:** Comprehensive trial counting across 5 dimensions (feature_sets × models × hyperparams × label_windows × markets, line 343-349)
- OOS traces persisted per fold (event_returns, positions, event_types, release_timestamps)
- Surprise-conditional hit rate at top quartile |y| (line 438-445)

`evaluate_event_contract_metrics()` (walkforward.py:147-235) computes:
- Deflated Sharpe Ratio via `deflated_sharpe_ratio()` from `backtest.advanced_validation`
- Monte Carlo validation via `monte_carlo_validation()` from `backtest.advanced_validation`
- Bootstrap mean-return CI (500 iterations, seed=42)
- Regime stability across event types
- Capacity utilization and turnover proxy

**Test coverage:** `test_walkforward_purge.py` covers purge/embargo, event-type-aware purge, and trial counting. PASS.

### Promotion Contract

`evaluate_event_promotion()` (promotion.py:127-177):
1. Creates `StrategyCandidate` from `EventPromotionConfig`
2. Converts event returns to `BacktestResult` via `_to_backtest_result()` (promotion.py:33-124)
3. Calls `PromotionGate.evaluate_event_strategy()` with both backtest-like results and event-specific metrics

**Cross-subsystem contract verification:**

| Import | Source | Target | Match |
|---|---|---|---|
| `PromotionDecision` | `autopilot.promotion_gate` | promotion.py:12 | VERIFIED |
| `PromotionGate` | `autopilot.promotion_gate` | promotion.py:12 | VERIFIED |
| `StrategyCandidate` | `autopilot.strategy_discovery` | promotion.py:13 | VERIFIED — 7 fields match |
| `BacktestResult` | `backtest.engine` | promotion.py:14 | VERIFIED — 15 core fields match `_to_backtest_result` |
| `deflated_sharpe_ratio` | `backtest.advanced_validation` | walkforward.py:12 | VERIFIED — signature matches |
| `monte_carlo_validation` | `backtest.advanced_validation` | walkforward.py:12 | VERIFIED — signature matches |

**`_to_backtest_result` field mapping:** The adapter correctly computes all 15 required `BacktestResult` fields (total_trades, winning_trades, losing_trades, win_rate, avg_return, avg_win, avg_loss, total_return, annualized_return, sharpe_ratio, sortino_ratio, max_drawdown, profit_factor, avg_holding_days, trades_per_year) plus 4 series fields (returns_series, equity_curve, daily_equity, trades).

**Promotion verdict:** Walk-forward and promotion contracts are fully aligned with autopilot and backtest subsystems. No metric-field drift detected. Event-type aware purge and comprehensive trial counting are correctly implemented.

---

## T5: Boundary and Integration Pass

### Cross-Subsystem Boundary Verification

| Boundary ID | Source → Target | Verified |
|---|---|---|
| `kalshi_to_autopilot_9` | kalshi → autopilot (PromotionDecision, PromotionGate, StrategyCandidate) | PASS |
| `kalshi_to_backtest_33` | kalshi → backtest (BacktestResult, deflated_sharpe_ratio, monte_carlo_validation) | PASS |
| `kalshi_to_config_34` | kalshi → config (18 KALSHI_* constants) | PASS |
| `kalshi_to_features_35` | kalshi → features (compute_option_surface_factors) | PASS |
| `api_to_kalshi_44` | api → kalshi (EventTimeStore conditional import) | PASS |
| `data_to_kalshi_32` | data → kalshi (KalshiProvider lazy factory import) | PASS |

### Config Constants

`KalshiProvider` (provider.py:14-33) imports 18 `KALSHI_*` constants:
`KALSHI_API_BASE_URL`, `KALSHI_DISTANCE_LAGS`, `KALSHI_ENV`, `KALSHI_FAR_EVENT_MINUTES`, `KALSHI_FAR_EVENT_STALE_MINUTES`, `KALSHI_HISTORICAL_API_BASE_URL`, `KALSHI_HISTORICAL_CUTOFF_TS`, `KALSHI_NEAR_EVENT_MINUTES`, `KALSHI_NEAR_EVENT_STALE_MINUTES`, `KALSHI_RATE_LIMIT_BURST`, `KALSHI_RATE_LIMIT_RPS`, `KALSHI_STALE_AFTER_MINUTES`, `KALSHI_STALE_HIGH_LIQUIDITY_MULTIPLIER`, `KALSHI_STALE_LIQUIDITY_HIGH_THRESHOLD`, `KALSHI_STALE_LIQUIDITY_LOW_THRESHOLD`, `KALSHI_STALE_LOW_LIQUIDITY_MULTIPLIER`, `KALSHI_STALE_MARKET_TYPE_MULTIPLIERS`, `KALSHI_TAIL_THRESHOLDS`

All 18 verified present in `config.py`. No hardcoded credential values — all API keys resolved from environment variables at runtime (client.py:338-364).

### Inbound Dependencies

- `api/services/kalshi_service.py`: Conditionally imports `EventTimeStore` (lazy, no side effects)
- `data/provider_registry.py`: Lazy factory import of `KalshiProvider` at line 23 (no eager import)

Both inbound consumers use lazy/conditional imports, preventing circular dependencies and import-time side effects. PASS.

### Options Cross-Market Integration

`options.py:11` imports `compute_option_surface_factors` from `features.options_factors`. The function signature matches: `compute_option_surface_factors(df: pd.DataFrame) -> pd.DataFrame`. The options disagreement features (entropy_gap, tail_gap, repricing_speed_gap, uncertainty_gap) are computed via strict backward as-of join. PASS.

**Boundary verdict:** All 6 cross-subsystem boundaries are correctly implemented. No import-time side effects, no circular dependencies, no contract drift. Lazy import patterns in inbound consumers are confirmed.

---

## T6: Synthesis and Verdicts

### Acceptance Criteria

| Criterion | Status |
|---|---|
| 1. 100% line coverage across all 25 Kalshi files | PASS — 6,096/6,096 lines reviewed |
| 2. As-of/leakage contracts validated with test alignment | PASS — all 8 leakage guards verified, 6 tests aligned |
| 3. Storage and distribution schemas confirmed stable | PASS with caveat — stable but no migration mechanism (F-04) |
| 4. Promotion/walk-forward interfaces to core subsystems validated | PASS — all 6 boundaries verified, no drift |

### Defect Matrix

| ID | Severity | Category | File | Remediation |
|---|---|---|---|---|
| F-01 | HIGH | Data Integrity | storage.py:550-564 | Remove cross-table writes from upsert methods |
| F-02 | HIGH | Security | client.py:222-223,241-242 | Use `-passin stdin` or `-passin fd:N` for OpenSSL |
| F-03 | MEDIUM | Security | client.py:277-287 | Secure temp file handling or stdin-based key passing |
| F-04 | MEDIUM | Schema Safety | storage.py:168-496 | Add schema_version table and migration logic |
| F-05 | MEDIUM | Resource Mgmt | storage.py:35-65 | Add `close()` and context manager protocol |
| F-06 | MEDIUM | Test Coverage | Multiple test files | Add tests for 7 identified gaps |
| F-07 | LOW | Performance | distribution.py:94-115 | Document O(n^2) bound; monitor for scale |
| F-08 | LOW | Robustness | distribution.py:71-82 | Consider per-market scale detection |
| F-09 | LOW | Schema Design | storage.py:276-301 | Add structural differentiation after F-01 fix |

### Remediation Priority

1. **Immediate (F-01):** Fix bidirectional outcome sync — this directly affects label correctness for event strategy research.
2. **Short-term (F-02, F-03):** Fix OpenSSL passphrase/key exposure — security concern in the fallback path.
3. **Medium-term (F-04, F-05, F-06):** Add schema migration, connection cleanup, and missing tests.
4. **Low priority (F-07, F-08, F-09):** Document and monitor; no code changes required at current scale.

### Leakage Verdict

**PASS.** All feature construction paths enforce strict backward-looking joins. The `_ensure_asof_before_release()` guard is applied at both grid construction and post-merge. Tests confirm leakage protection. No forward-looking data enters the feature pipeline.

### Schema Compatibility Verdict

**CONDITIONAL PASS.** The schema is stable for current use (no DDL mutations at runtime, column introspection provides forward-compatible writes). However, the absence of a migration mechanism (F-04) means existing databases will silently become incompatible if the schema evolves. Recommend adding migration support before the next schema change.

---

## File Review Ledger

| # | File | Lines | Reviewed | Notes |
|---|---:|---:|---|---|
| 1 | `kalshi/distribution.py` | 935 | YES | Largest file; F-07, F-08 |
| 2 | `kalshi/client.py` | 655 | YES | F-02, F-03 |
| 3 | `kalshi/storage.py` | 649 | YES | F-01, F-04, F-05, F-09 |
| 4 | `kalshi/provider.py` | 647 | YES | 18 config imports verified |
| 5 | `kalshi/events.py` | 517 | YES | Leakage guards verified |
| 6 | `kalshi/walkforward.py` | 492 | YES | E2, E3 implemented correctly |
| 7 | `kalshi/quality.py` | 206 | YES | C1, C2 contracts present |
| 8 | `kalshi/promotion.py` | 177 | YES | BacktestResult adapter verified |
| 9 | `kalshi/pipeline.py` | 167 | YES | Orchestration correct |
| 10 | `kalshi/options.py` | 145 | YES | Backward as-of join verified |
| 11 | `kalshi/microstructure.py` | 127 | YES | Diagnostics only |
| 12 | `kalshi/regimes.py` | 142 | YES | Regime classification |
| 13 | `kalshi/disagreement.py` | 113 | YES | Cross-market signals |
| 14 | `kalshi/mapping_store.py` | 75 | YES | D1 version assertion present |
| 15 | `kalshi/router.py` | 102 | YES | Live/historical routing |
| 16 | `kalshi/__init__.py` | 58 | YES | 24 exports |
| 17 | `kalshi/tests/test_distribution.py` | 41 | YES | Mass normalization |
| 18 | `kalshi/tests/test_bin_validity.py` | 105 | YES | 8 bin validation cases |
| 19 | `kalshi/tests/test_threshold_direction.py` | 126 | YES | 16 direction cases |
| 20 | `kalshi/tests/test_no_leakage.py` | 117 | YES | Pre-release feature check |
| 21 | `kalshi/tests/test_stale_quotes.py` | 152 | YES | 10 stale cutoff cases |
| 22 | `kalshi/tests/test_walkforward_purge.py` | 159 | YES | Purge/embargo/trial counting |
| 23 | `kalshi/tests/test_signature_kat.py` | 141 | YES | RSA-PSS KAT verification |
| 24 | `kalshi/tests/test_leakage.py` | 46 | YES | Strict pre-release check |
| 25 | `kalshi/tests/__init__.py` | 1 | YES | Package marker |
| | **TOTAL** | **6,096** | **25/25** | |
