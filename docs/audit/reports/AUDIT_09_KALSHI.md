# Audit Report: Subsystem 9 -- Kalshi (Event Markets)

> **Date:** 2026-02-28
> **Auditor:** Claude Opus 4.6
> **Spec:** SPEC_AUDIT_09_KALSHI.md
> **Coverage:** 25 files, 6,096 lines (100% line coverage)

---

## Executive Summary

The Kalshi subsystem is well-architected with clean internal layering and correct cross-subsystem boundary contracts. The core event-time join logic is sound -- all joins use `pd.merge_asof(direction="backward")` and the `_ensure_asof_before_release` guard provides a double safety net against forward-looking data leakage.

However, 4 P1-HIGH and 6 P2-MEDIUM findings were identified. The most critical are: (1) the API service layer (`kalshi_service.py`) is completely broken due to wrong `query_df()` API usage, (2) no purge/embargo window exists between features and labels despite config placeholders, (3) event strategies silently bypass stress-regime validation gates, and (4) distribution.py test coverage is severely inadequate for a 936-line module.

**Findings by severity:**
- P0-CRITICAL: 0
- P1-HIGH: 4
- P2-MEDIUM: 6
- P3-LOW: 9
- INFO: 11

---

## T1: Ledger and Schema Baseline

### Schema Inventory

`EventTimeStore` manages **18 tables** in DuckDB with versioning and audit trail:

| # | Table | Primary Key |
|---|-------|-------------|
| 1 | `kalshi_markets` | `market_id` |
| 2 | `kalshi_contracts` | `contract_id` |
| 3 | `kalshi_quotes` | `(contract_id, ts)` |
| 4 | `kalshi_fees` | `(market_id, contract_id, effective_ts)` |
| 5 | `macro_events` | `event_id` |
| 6 | `macro_events_versioned` | `(event_id, version_ts)` |
| 7 | `event_outcomes` | `(event_id, asof_ts)` |
| 8 | `event_outcomes_first_print` | `(event_id, asof_ts)` |
| 9 | `event_outcomes_revised` | `(event_id, asof_ts)` |
| 10 | `kalshi_distributions` | `(market_id, ts)` |
| 11 | `event_market_map_versions` | `(event_id, market_id, effective_start_ts, mapping_version)` |
| 12 | `kalshi_market_specs` | `(market_id, spec_version_ts)` |
| 13 | `kalshi_contract_specs` | `(contract_id, spec_version_ts)` |
| 14 | `kalshi_data_provenance` | `(market_id, asof_date, endpoint)` |
| 15 | `kalshi_coverage_diagnostics` | `(market_id, asof_date)` |
| 16 | `kalshi_ingestion_logs` | `(endpoint, ingest_ts)` |
| 17 | `kalshi_daily_health_report` | `asof_date` |
| 18 | `kalshi_ingestion_checkpoints` | `(market_id, asof_date, endpoint)` |

6 indexes defined for query performance on quotes, distributions, events, and maps.

### Cross-Module Import Summary

| File | Lines | Cross-Module Imports |
|------|-------|---------------------|
| `storage.py` | 649 | duckdb (lazy, try/except) |
| `provider.py` | 647 | 19 KALSHI_* config constants (top-level) |
| `distribution.py` | 935 | `.quality` internal only |
| `events.py` | 517 | None (self-contained) |
| `pipeline.py` | 167 | `autopilot.promotion_gate.PromotionDecision` |
| `promotion.py` | 178 | `autopilot.promotion_gate`, `autopilot.strategy_discovery`, `backtest.engine.BacktestResult` |
| `walkforward.py` | 492 | `backtest.advanced_validation.deflated_sharpe_ratio, monte_carlo_validation` |
| `options.py` | 145 | `features.options_factors.compute_option_surface_factors` |
| `client.py` | 655 | None (self-contained) |
| All others | ~1,711 | None (internal only) |

---

## T2: Leakage/As-of Correctness

### Join Operations -- All Verified Safe

| Join | File:Line | Direction | Guard | Verdict |
|------|-----------|-----------|-------|---------|
| `asof_join` | events.py:44 | `backward` | `_ensure_asof_before_release` | SAFE |
| Distribution join | events.py:260 | `backward` | Grid construction + assertion | SAFE |
| Disagreement join | events.py:208 | `backward` | Backward-only | SAFE |
| Asset time join | events.py:349 | `backward` | Backward-only | SAFE |
| Label construction | events.py:405 | Forward | Intentional (labels, not features) | SAFE |
| Asset response labels | events.py:494 | Forward | Post-release by design | SAFE |

All feature joins are strictly backward-looking. Labels correctly use post-release data.

---

## T3: Distribution and Quote-Quality

### Monotonicity Repair -- Verified Correct

- `_isotonic_nonincreasing` (distribution.py:94): PAV algorithm produces valid nonincreasing sequences
- `_isotonic_nondecreasing` (distribution.py:118): Correctly reverses I/O of nonincreasing
- "ge" direction (survival function) repaired to nonincreasing -- correct
- "le" direction (CDF) repaired to nondecreasing -- correct

### Quality Framework -- Mostly Sound

- Quality score weights sum to 1.0, clipped to [0.0, 1.0]
- Stale cutoff interpolation is correct and monotonic
- Dynamic stale policies correctly adjust by event type, distance, and liquidity

---

## T4: Walk-forward and Promotion Contracts

### Walk-forward Purge -- Correct

- Events sorted by `release_ts`
- Train events filtered to `release_ts <= (test_start - effective_purge)`
- Embargo removes additional trailing train events
- Event-type-aware purge takes max window across test event types

### Promotion Contract -- Functional with Gaps

- `BacktestResult` construction covers all required positional fields
- `PromotionGate.evaluate_event_strategy()` correctly routes with `event_mode=True`
- Event-specific gates (worst_event_loss, surprise_hit_rate) properly applied

---

## T5: Boundary and Integration

### Boundary Contract Verification

| Boundary ID | Status | Notes |
|-------------|--------|-------|
| `kalshi_to_autopilot_9` | **PASS** | PromotionDecision, PromotionGate, StrategyCandidate all verified |
| `kalshi_to_backtest_33` | **PASS** | BacktestResult, deflated_sharpe_ratio, monte_carlo_validation verified |
| `kalshi_to_config_34` | **PASS** | All 19+2 KALSHI_* constants exist in config.py |
| `kalshi_to_features_35` | **PASS** | compute_option_surface_factors exists with correct signature |
| `data_to_kalshi_32` | **PASS** | Conditional factory import works correctly |
| `api_to_kalshi_44` | **FAIL** | Import correct, but query_df() usage is broken |

### Security -- Adequate

- No hardcoded API keys anywhere in subsystem
- Credentials resolved from environment variables with env-specific prefixes
- Rate limiting properly enforced via token-bucket algorithm

---

## Findings Matrix

### P1-HIGH

| ID | File | Lines | Description |
|----|------|-------|-------------|
| **K-01** | `api/services/kalshi_service.py` | 27 | **query_df() called with wrong API**: `store.query_df("kalshi_markets", limit=200)` -- `query_df()` accepts `(sql, params)`, not table names with kwargs. Will raise `TypeError` at runtime. |
| **K-02** | `api/services/kalshi_service.py` | 45 | **query_df() wrong API + SQL injection pattern**: `store.query_df("kalshi_contracts", where=f"event_id = '{market_id}'")` -- same TypeError, plus f-string SQL interpolation of user input creates injection vector. |
| **K-03** | `events.py` / `config.py` | 306-307 | **No purge/embargo window implemented**: `KALSHI_PURGE_WINDOW_BY_EVENT` and `KALSHI_DEFAULT_PURGE_WINDOW` are STATUS: PLACEHOLDER in config -- defined but never imported or enforced. No temporal gap between latest feature snapshot and label beyond structural `asof_ts < release_ts`. |
| **K-04** | `tests/test_distribution.py` | all | **Severely inadequate test coverage**: Only 1 test for a 936-line module. Zero coverage of threshold-mode reconstruction, isotonic repair, distance features, moments, tail probabilities, early-return paths, or panel building. |

### P2-MEDIUM

| ID | File | Lines | Description |
|----|------|-------|-------------|
| **K-05** | `api/services/kalshi_service.py` | 45 | **Wrong table and column**: `get_distributions()` queries `kalshi_contracts` with `event_id` filter, but that table has no `event_id` column. Should query `kalshi_distributions`. |
| **K-06** | `storage.py` | 550-564 | **Bidirectional cross-write**: `upsert_event_outcomes()` and `upsert_event_outcomes_first_print()` both write to BOTH tables, making them always identical. The first-print vs. latest distinction is defeated. |
| **K-07** | `distribution.py` | 510-622 | **Early-return dicts missing 9 fields**: Three fallback paths omit `direction_source`, `direction_confidence`, `bin_overlap_count`, etc. Schema inconsistency causes KeyError or silent NaN in downstream panels. |
| **K-08** | `quality.py` | 164-190 | **`passes_hard_gates` is dead code**: Hard validity gate (C1) defined but never called. Consumers may use garbage distribution moments instead of NaN. |
| **K-09** | `promotion.py` | 33-124 | **Event strategies bypass stress-regime gates**: `_to_backtest_result` omits `regime_performance`, defaulting to `{}`. The `if regime_perf:` guard in PromotionGate silently skips all stress-regime validation (SPEC-V02). |
| **K-10** | `walkforward.py` | 394-399 | **Inner validation split lacks purge**: Hyperparameter selection uses 80/20 temporal split without purge window between inner-train and inner-val. |

### P3-LOW

| ID | File | Lines | Description |
|----|------|-------|-------------|
| K-11 | `storage.py` | 230 | `kalshi_fees` table has no writer method -- dead schema |
| K-12 | `storage.py` | 103, 109 | Inconsistent quoting of table name in PRAGMA between DuckDB/SQLite paths |
| K-13 | `provider.py` / `storage.py` | 40 / 115 | Duplicated `_to_iso_utc` / `_norm_ts` timestamp normalization |
| K-14 | `distribution.py` | 804-806 | `violated_constraints_post` reports pre-repair count after isotonic repair (should always be 0) |
| K-15 | `distribution.py` | 416-419 | Tail probability convention (always upper-tail) undocumented |
| K-16 | `distribution.py` | 448-458 | Liquidity proxy scale depends on data source count (unnormalized) |
| K-17 | `quality.py` | 193-206 | `quality_as_feature_dict` is dead code with stale TODO |
| K-18 | `client.py` | 223, 242 | OpenSSL passphrase exposed via command-line argument (fallback path only) |
| K-19 | `tests/test_leakage.py` / `tests/test_no_leakage.py` | all | Near-complete overlap between test files; redundant coverage |

### INFO

| ID | File | Description |
|----|------|-------------|
| K-I1 | `storage.py` | B1/B2/B3 enrichment columns silently dropped during storage (absent from schema) |
| K-I2 | `storage.py` / `provider.py` / `distribution.py` | Three separate timestamp-to-UTC implementations |
| K-I3 | `distribution.py` | `_support` and `_mass` arrays not persisted; distances cannot be recomputed from stored data |
| K-I4 | `provider.py` | `store_clock_check` does not guard with `available()` before calling client |
| K-I5 | `pipeline.py` | `build_event_features` loads entire distributions table with no time filter |
| K-I6 | `tests/test_no_leakage.py` | No adversarial test with post-release distribution data |
| K-I7 | `tests/test_no_leakage.py` | Silent skipTest on empty panel could mask failures |
| K-I8 | `walkforward.py` | Trial counting uses conservative `max()` of actual vs theoretical -- correct |
| K-I9 | `promotion.py` | `regime_positive_fraction` mapped to `event_regime_stability` -- semantic mismatch, no runtime impact |
| K-I10 | various | Several file line counts differ from spec (files have grown since spec) |
| K-I11 | `storage.py` | `INSERT OR REPLACE` is SQLite syntax; DuckDB supports via compatibility mode |

---

## Recommended Mitigations (Priority Order)

1. **K-01/K-02/K-05 (P1):** Rewrite `kalshi_service.py` to use proper SQL queries with parameterized values:
   ```python
   store.query_df("SELECT * FROM kalshi_markets LIMIT 200")
   store.query_df("SELECT * FROM kalshi_distributions WHERE market_id = ?", params=[market_id])
   ```

2. **K-03 (P1):** Implement purge/embargo windows by importing and enforcing `KALSHI_PURGE_WINDOW_BY_EVENT` and `KALSHI_DEFAULT_PURGE_WINDOW` in `events.py:build_event_snapshot_grid`.

3. **K-04 (P1):** Add integration tests for distribution.py covering threshold-mode, isotonic repair, distance features, moment computation, and early-return paths.

4. **K-09 (P2):** Populate `regime_performance` in `_to_backtest_result` (even if synthetic) OR explicitly skip stress-regime gates in `evaluate_event_strategy` with documented rationale.

5. **K-06 (P2):** Fix `upsert_event_outcomes`/`upsert_event_outcomes_first_print` to write to their respective tables only.

6. **K-07 (P2):** Add missing keys to early-return dicts in `build_distribution_snapshot`.

7. **K-08 (P2):** Wire `passes_hard_gates` into distribution snapshot consumer or remove dead code.

---

## Acceptance Criteria Verification

| Criterion | Status |
|-----------|--------|
| 100% line coverage across all 25 Kalshi files | **PASS** -- all 25 files read line-by-line |
| As-of/leakage contracts validated with test alignment | **PASS** -- all joins verified backward-only; tests exist but purge gap missing |
| Storage and distribution schemas confirmed stable | **PASS with findings** -- schema is stable but K-06, K-07 need attention |
| Promotion/walk-forward interfaces to core subsystems validated | **PASS** -- all boundary contracts verified correct |
