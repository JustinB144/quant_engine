# Audit Report: Subsystem 10 -- API & Frontend

> **Date:** 2026-02-28
> **Auditor:** Claude Opus 4.6
> **Spec:** SPEC_AUDIT_10_API_FRONTEND.md
> **Coverage:** 59 files, 10,188 lines (100% line coverage)

---

## Executive Summary

The API & Frontend subsystem is the consumer gateway for all core engine layers, with 115 outbound dependency edges across 9 modules. The internal layering (routers -> services -> schemas) is cohesive and the global error handler provides a strong safety net. The job system lifecycle is well-designed with clean status transitions and SSE streaming.

However, **2 P0-CRITICAL** and **4 P1-HIGH** findings were identified. The most critical: two routers (`diagnostics.py` and `risk.py`) completely bypass the `ApiResponse` envelope, causing the frontend to always interpret their responses as errors. The Kalshi service layer is broken (wrong `query_df()` API). The health_service.py God Object at 2,929 lines manages 4 SQLite databases with synchronous I/O in an async context.

**Findings by severity:**
- P0-CRITICAL: 2
- P1-HIGH: 4
- P2-MEDIUM: 8
- P3-LOW: 12
- INFO: 12

---

## T1: Ledger and API Contract Baseline

### Envelope Contract

`ApiResponse` (api/schemas/envelope.py:29):
```
ok: bool          -- True for success, False for error
data: Optional[T] -- Generic payload
error: Optional[str] -- Error message on failure
meta: ResponseMeta   -- Timing, warnings, data mode, model version, etc.
```

Factory methods: `ApiResponse.success()`, `ApiResponse.fail()`, `ApiResponse.from_cached()`

### Endpoint Inventory (48 endpoints across 16 routers)

| Router | Endpoints | Envelope Compliant? |
|--------|-----------|---------------------|
| `dashboard.py` | 6 GET | YES (all 6) |
| `system_health.py` | 5 GET | PARTIAL (error paths return ok=True) |
| `data_explorer.py` | 5 GET, 1 POST | YES (all 6) |
| `model_lab.py` | 4 GET, 2 POST | YES (all 6) |
| `autopilot.py` | 3 GET, 1 POST | YES (all 4) |
| `backtests.py` | 3 GET, 1 POST | YES (all 4) |
| `jobs.py` | 3 GET, 1 POST + SSE | YES (SSE correctly uses EventSourceResponse) |
| `diagnostics.py` | 1 GET | **NO -- returns raw Dict** |
| `risk.py` | 1 GET | **NO -- returns raw Dict** |
| `regime.py` | 1 GET | YES |
| `benchmark.py` | 3 GET | YES (all 3) |
| `iv_surface.py` | 1 GET | YES |
| `logs.py` | 1 GET | YES |
| `signals.py` | 1 GET | YES |
| `config_mgmt.py` | 3 GET, 1 PATCH | YES (all 4) |

**Envelope compliance: 44/48 fully compliant, 2 completely non-compliant (P0), 2 partially non-compliant on error paths (P2).**

### Error Handler (api/errors.py)

5 custom exceptions mapped to HTTP status codes (404, 422, 500, 503). Global catch-all returns `ApiResponse.fail("Internal server error")` with 500. Module-alias handling is a solid defensive pattern.

### Router Registration (api/routers/__init__.py)

Fault-tolerant loading: catches any Exception and logs warning, skipping broken routers. App starts even if individual engine subsystems are missing.

---

## T2: Orchestrator and Core-Service Contract Pass

### orchestrator.py (372 lines)

4 pipeline methods: `load_and_prepare()`, `train()`, `predict()`, `backtest()`. All cross-subsystem imports are lazy (inside method bodies). Config constants: `UNIVERSE_FULL`, `UNIVERSE_QUICK`, `WRDS_ENABLED`, `REQUIRE_PERMNO`, `DATA_CACHE_DIR`, `ENTRY_THRESHOLD`, `CONFIDENCE_THRESHOLD`, `REGIME_NAMES`, `RESULTS_DIR`. Service outputs are JSON-serializable dicts suitable for the ApiResponse envelope.

### data_helpers.py (1,058 lines)

~35 distinct config constants referenced -- largest config surface area in the API layer. Provides health check infrastructure (`collect_health_data`), portfolio analytics, and system diagnostics.

### health_service.py (2,929 lines -- LARGEST file in codebase)

**45 method/function definitions**. Manages:
- 19 health checks across 5 weighted domains
- IC tracking (SQLite)
- Disagreement tracking (SQLite)
- Execution quality tracking (SQLite)
- Health history (SQLite)

**Circular dependency exposure to autopilot:**
| Method | Called By | Purpose |
|--------|----------|---------|
| `save_ic_snapshot()` | autopilot/engine.py:1868 | Persist IC metrics after candidate evaluation |
| `save_disagreement_snapshot()` | autopilot/engine.py:1911 | Persist ensemble disagreement after predictions |
| `save_execution_quality_fill()` | autopilot/paper_trader.py:189,532 | Persist execution quality after fills |
| `get_health_history(limit=1)` | autopilot/paper_trader.py | Read latest health score for risk gating |

All imports are lazy (inside function bodies). Runtime-safe but structurally fragile.

---

## T3: Job System Lifecycle Pass

### Canonical Status Set

```
JobStatus: queued | running | succeeded | failed | cancelled
```

### State Transitions (verified correct)

```
queued   -> running    (runner.py:41, semaphore acquired)
running  -> succeeded  (runner.py:56-58, job function completes)
running  -> cancelled  (runner.py:62-64, CancelledError caught)
running  -> failed     (runner.py:66-72, Exception caught)
queued   -> cancelled  (store.py:132-134, pre-execution cancel)
```

No transitions from terminal states. `finally` block always emits `done` SSE event and removes from `_active_tasks`.

### SSE Event Streaming

Correct implementation: `subscribe_events()` creates async queue, initial status event, subsequent events via queue, generator breaks on `done`/`cancelled`, cleanup in finally block.

### Lazy Imports in Job Files

| Job File | Line | Import | Correct? |
|----------|------|--------|----------|
| `autopilot_job.py` | 12 | `AutopilotEngine` (absolute) | YES |
| `backtest_job.py` | 12 | `PipelineOrchestrator` (relative) | YES |
| `train_job.py` | 12 | `PipelineOrchestrator` (relative) | YES |
| `predict_job.py` | 12 | `PipelineOrchestrator` (relative) | YES |

All inside function bodies -- correctly deferred to execution time.

---

## T4: Router Envelope Consistency Pass

See Endpoint Inventory above. Key findings:
- `diagnostics.py` and `risk.py` return raw `Dict` without `ApiResponse` wrapping
- `system_health.py` returns `ok=True` with `"status": "error"` in data payload on exceptions
- Schema models (DashboardKPIs, BacktestSummary, etc.) are defined but never used by any router

---

## T5: Cache and Invalidation Pass

### TTL Configuration

| Domain | TTL (seconds) |
|--------|---------------|
| dashboard | 30 |
| regime | 60 |
| model_health | 120 |
| feature_importance | 300 |
| signals | 30 |
| health | 120 |
| benchmark | 300 |

### Invalidation Hook Wiring

| Hook | Defined | Wired? |
|------|---------|--------|
| `invalidate_on_train` | cache/invalidation.py:7 | YES (main.py, model_lab.py) |
| `invalidate_on_backtest` | cache/invalidation.py:15 | YES (backtests.py) |
| `invalidate_on_config_change` | cache/invalidation.py:30 | YES (config_mgmt.py) |
| `invalidate_on_data_refresh` | cache/invalidation.py:21 | **NEVER CALLED** |

### Shared Artifact Reader Protection

| Service | Missing File? | Corrupt File? | Protected? |
|---------|--------------|---------------|------------|
| AutopilotService | YES (exists check) | YES (try/except) | GOOD |
| BacktestService | YES (exists check) | NO (json.load unprotected) | PARTIAL |
| ResultsService | YES (exists check) | NO (json.load unprotected) | PARTIAL |
| ModelService | NO (delegates to registry) | NO | POOR |
| DataService | YES (cache meta) | YES (try/except) | GOOD |
| KalshiService | N/A | N/A | BROKEN (wrong API) |

---

## T6: Circular Coupling and Boundary Pass

### Circular Dependency Map

```
api/jobs/autopilot_job.py:12 -> autopilot/engine.py (lazy)
autopilot/engine.py:1868     -> api/services/health_service.py (lazy)
autopilot/engine.py:1911     -> api/services/health_service.py (lazy)
autopilot/paper_trader.py:173 -> api/services/health_risk_feedback.py (lazy)
autopilot/paper_trader.py:189 -> api/services/health_service.py (lazy/conditional)
autopilot/paper_trader.py:211 -> api/ab_testing.py (lazy/conditional)
autopilot/paper_trader.py:532 -> api/services/health_service.py (lazy/conditional)
```

All 7 import sites use lazy imports inside method bodies with try/except guards. Runtime-safe (non-SCC). No module-level circular imports exist.

### Boundary Contract Verification

| Boundary ID | Status | Notes |
|-------------|--------|-------|
| `api_to_config_36` | **PASS** | All 82 lazy config imports use correct constant names |
| `api_to_data_37` | **PASS** | load_universe, get_data_provenance, WRDSProvider all verified |
| `api_to_features_38` | **PASS** | FeaturePipeline, RegimeDetector correctly imported |
| `api_to_regime_39` | **PASS** | RegimeDetector correctly imported via orchestrator |
| `api_to_models_40` | **PASS** | ModelRegistry, ModelGovernance, EnsemblePredictor, FeatureStabilityTracker verified |
| `api_to_backtest_41` | **PASS** | Backtester correctly imported via orchestrator |
| `api_to_risk_42` | **PASS** | Risk modules not directly imported (accessed via orchestrator) |
| `api_to_autopilot_43` | **PASS** | AutopilotEngine lazy import in autopilot_job.py correct |
| `api_to_kalshi_44` | **FAIL** | Import correct, but query_df() usage is broken |
| `autopilot_to_api_circular_5` | **PASS** | All 6+1 edges are lazy with try/except guards |

---

## Findings Matrix

### P0-CRITICAL

| ID | File | Lines | Description |
|----|------|-------|-------------|
| **A-01** | `api/routers/diagnostics.py` | 19-65 | **Completely bypasses ApiResponse envelope.** Returns raw `Dict` without `ok`, `data`, `error`, or `meta` fields. Frontend `client.ts` checks `json.ok` which is `undefined` (falsy), so EVERY response from `/api/diagnostics` is interpreted as an error. Internal exceptions caught with bare `except Exception` and returned as raw dicts, bypassing the global error handler. |
| **A-02** | `api/routers/risk.py` | 19-53 | **Completely bypasses ApiResponse envelope.** Same pattern as diagnostics.py. `GET /api/risk/factor-exposures` returns `{"status": "ok", ...}` -- no envelope. Frontend always throws ApiError. |

### P1-HIGH

| ID | File | Lines | Description |
|----|------|-------|-------------|
| **A-03** | `api/routers/diagnostics.py` | all | Does not import `ApiResponse`. Catches exceptions internally as raw dicts, preventing global error handler from wrapping them. |
| **A-04** | `api/routers/risk.py` | all | Does not import `ApiResponse`. Same pattern as A-03. |
| **A-05** | `api/services/kalshi_service.py` | 27 | **query_df() called with wrong API**: `store.query_df("kalshi_markets", limit=200)` -- method accepts `(sql, params)`, not table names with kwargs. Will raise `TypeError` at runtime. |
| **A-06** | `api/services/kalshi_service.py` | 45 | **query_df() wrong API + SQL injection**: `store.query_df("kalshi_contracts", where=f"event_id = '{market_id}'")` -- TypeError, plus f-string interpolation of user input creates injection vector. |

### P2-MEDIUM

| ID | File | Lines | Description |
|----|------|-------|-------------|
| **A-07** | `api/services/health_service.py` | all | **God Object**: 2,929 lines, 45 methods, 4 SQLite databases. Should be decomposed into health checks, IC tracking, disagreement tracking, execution quality tracking, and health history. |
| **A-08** | `api/services/health_service.py` | all save_*/get_* | **Synchronous SQLite I/O in async context**: All SQLite operations use synchronous `sqlite3.connect()`. When called from async FastAPI routes, these block the event loop. Job system uses `aiosqlite` for this reason. |
| **A-09** | `api/jobs/autopilot_job.py` + `autopilot/engine.py` | 12, 1868, 1911 | **Circular dependency api<->autopilot via HealthService**: Works only because all imports are lazy. Adding any top-level import breaks it. Tracking functions should be extracted to a shared module. |
| **A-10** | `api/routers/system_health.py` | 85-88, 118-120 | **Error conditions return ok=True**: `/model-age` and `/data-mode` return `ApiResponse.success(data)` with `"status": "error"` in data on exceptions. Should use `ApiResponse.fail()`. |
| **A-11** | `api/cache/invalidation.py` | 21 | **`invalidate_on_data_refresh` is dead code**: Defined but never called. External data refreshes leave stale cache for up to 300 seconds. |
| **A-12** | `api/services/kalshi_service.py` | 45 | **Wrong table and column**: `get_distributions()` queries `kalshi_contracts` with `event_id` filter, but that table has no `event_id` column. Should query `kalshi_distributions`. |
| **A-13** | `api/services/kalshi_service.py` | 45 | **SQL injection vulnerability**: f-string interpolation of user-supplied `market_id` into SQL without parameterization. |
| **A-14** | All routers / All schemas | -- | **Schema models defined but unused**: DashboardKPIs, BacktestSummary, QuickStatus, etc. are defined in `schemas/` but never imported or used by any router. Routers pass untyped dicts to `ApiResponse.success()`. |

### P3-LOW

| ID | File | Lines | Description |
|----|------|-------|-------------|
| A-15 | `routers/jobs.py` | 20-22 | `_not_found()` returns `JSONResponse` bypassing declared `-> ApiResponse` return type |
| A-16 | `routers/logs.py` | 38 | Does not include `elapsed_ms` in meta, unlike all other endpoints |
| A-17 | `health_service.py` | 1913+ | Class-level mutable path singletons create hidden global state |
| A-18 | `health_service.py` | 123 | Naive `datetime.now()` inconsistent with UTC-aware timestamps elsewhere |
| A-19 | `runner.py` | 83-89 | Job cancellation has theoretical TOCTOU race window |
| A-20 | `runner.py` | 93-111 | SSE subscriber coroutine could leak if job hangs indefinitely |
| A-21 | `runner.py` | 22 | `_active_tasks` dict includes queued-but-not-running tasks |
| A-22 | `backtest_service.py` | 25 | `json.load()` not wrapped in try/except; corrupt JSON produces 500 |
| A-23 | `results_service.py` | 24 | Same as A-22 -- `json.load()` not protected against corrupt JSON |
| A-24 | `model_service.py` | 17-21, 89-98 | `ModelRegistry()` and `ModelGovernance()` called with no error handling |
| A-25 | `health_risk_feedback.py` | 247 | YAML config path uses `__import__("pathlib")` and relative traversal |
| A-26 | `autopilot/paper_trader.py` | 228 | Calls private method `ABTestRegistry._save()` -- fragile coupling |

### INFO

| ID | File | Description |
|----|------|-------------|
| A-I1 | `api/config.py` | Default CORS origin is wildcard `*` -- tighten for production |
| A-I2 | `api/errors.py` | Module-alias error handler is a well-designed defensive pattern |
| A-I3 | `orchestrator.py` | `predict()` and `backtest()` write files as side effects |
| A-I4 | `data_helpers.py` | Hardcoded feature inventory counts may drift from actual pipeline |
| A-I5 | `store.py` | 12-hex-char job ID provides adequate entropy |
| A-I6 | `jobs/__init__.py` | Clean re-export of public API |
| A-I7 | `system_health.py` | `datetime.now()` without timezone vs timezone-aware `train_date` |
| A-I8 | `jobs.py` | SSE endpoint correctly uses `EventSourceResponse` |
| A-I9 | All routers | `ApiResponse` generic parameter `T` never specified -- OpenAPI lacks inner data schema |
| A-I10 | `health_confidence.py` | Hard module-level `from scipy import stats` breaks lazy import pattern |
| A-I11 | `ab_testing.py` | MD5 used for ticker assignment; functional but may fail on FIPS systems |
| A-I12 | Circular dependency | All 7 import sites verified runtime-safe with lazy imports and try/except guards |

---

## Recommended Mitigations (Priority Order)

1. **A-01/A-02/A-03/A-04 (P0/P1):** Wrap `diagnostics.py` and `risk.py` in `ApiResponse` envelope. Import `ApiResponse`, change return type to `-> ApiResponse`, use `ApiResponse.success()` for data and `ApiResponse.fail()` for errors. Remove internal `except Exception` catch-alls or re-raise so the global handler fires.

2. **A-05/A-06/A-12/A-13 (P1/P2):** Rewrite `kalshi_service.py` to use proper SQL queries:
   ```python
   store.query_df("SELECT * FROM kalshi_markets LIMIT 200")
   store.query_df("SELECT * FROM kalshi_distributions WHERE market_id = ?", params=[market_id])
   ```

3. **A-07/A-08 (P2):** Decompose `health_service.py`:
   - Extract tracking stores (IC, disagreement, execution quality, health history) into `api/services/tracking_store.py`
   - Use `aiosqlite` for all SQLite operations called from async context
   - Split health checks by domain into separate files

4. **A-09 (P2):** Extract the tracking interface consumed by autopilot into a shared `quant_engine.tracking` module that neither `api` nor `autopilot` depends on circularly.

5. **A-10 (P2):** Change `/model-age` and `/data-mode` error paths to use `ApiResponse.fail()`.

6. **A-11 (P2):** Wire `invalidate_on_data_refresh` into the data refresh pathway (or invoke it from a startup lifecycle event).

7. **A-14 (P2):** Use typed schema models in router signatures (e.g., `-> ApiResponse[DashboardKPIs]`) for OpenAPI documentation.

---

## Acceptance Criteria Verification

| Criterion | Status |
|-----------|--------|
| Full line coverage across all 59 subsystem files | **PASS** -- all 59 files read line-by-line |
| API envelope/job lifecycle contracts verified | **PASS with P0 findings** -- 44/48 endpoints compliant, 2 non-compliant |
| Cross-subsystem lazy imports validated for graceful degradation | **PASS** -- all lazy imports correctly guarded |
| Circular API-autopilot concern explicitly dispositioned | **PASS** -- all 7 edges verified lazy with try/except; recommendation to extract shared tracking module |

---

## Line Count Verification

Several files have grown since the spec was written. Notable deltas:

| File | Spec Says | Actual | Delta |
|------|-----------|--------|-------|
| `routers/config_mgmt.py` | 29 | 146 | +117 |
| `routers/benchmark.py` | 37 | 128 | +91 |
| `routers/regime.py` | 50 | 133 | +83 |
| `schemas/envelope.py` | 27 | 54 | +27 |
| `routers/signals.py` | 29 | 55 | +26 |
| `routers/dashboard.py` | 127 | 170 | +43 |
| `routers/iv_surface.py` | 33 | 71 | +38 |

All files were read at their current sizes -- no content was missed.
