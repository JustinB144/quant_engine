# Feature Spec: Subsystem Audit Spec — API & Frontend

> **Status:** Draft
> **Author:** Codex
> **Date:** 2026-02-27
> **Estimated effort:** 24-32 hours across 7 tasks

---

## Why

API & Frontend is the largest subsystem by file count and acts as the consumer gateway for all core engine layers. Contract drift here causes system-wide operability issues even when core computations remain correct.

## What

Define complete audit coverage for Subsystem `api_frontend`, including endpoint contracts, envelope/job lifecycle guarantees, lazy-import resilience, and integration correctness with all upstream subsystems.

## Constraints

### Must-haves
- Audit all `59` files (`10,188` lines).
- Validate `ApiResponse` envelope consistency across all routers.
- Validate background job lifecycle contract (`queued/running/succeeded/failed/cancelled`).
- Validate lazy import degradation behavior for all cross-subsystem service calls.

### Must-nots
- No API contract changes during audit.
- No unreviewed cache invalidation side effects.
- No unresolved `P0`/`P1` findings in `api/services/health_service.py`, `api/orchestrator.py`, `api/jobs/autopilot_job.py`, `api/schemas/envelope.py`.

### Out of scope
- Deep frontend visual redesign.
- Non-API entrypoint scripts.

## Current State

Subsystem boundary and dependency data are in [SUBSYSTEM_MAP.json](/Users/justinblaise/Documents/quant_engine/docs/architecture/SUBSYSTEM_MAP.json), [DEPENDENCY_MATRIX.md](/Users/justinblaise/Documents/quant_engine/docs/audit/DEPENDENCY_MATRIX.md), and [SYSTEM_ARCHITECTURE_AND_FLOWS.md](/Users/justinblaise/Documents/quant_engine/docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md).

### Key files
| File | Lines | Def/Class Count | Branch Points |
|---|---:|---:|---:|
| `api/services/health_service.py` |     2929 | 89 | 331 |
| `api/services/data_helpers.py` |     1058 | 22 | 144 |
| `api/ab_testing.py` |      794 | 38 | 73 |
| `api/routers/data_explorer.py` |      490 | 13 | 40 |
| `api/orchestrator.py` |      372 | 6 | 42 |
| `api/services/health_alerts.py` |      324 | 11 | 33 |
| `api/services/health_confidence.py` |      314 | 8 | 17 |
| `api/services/health_risk_feedback.py` |      275 | 8 | 21 |
| `api/services/data_service.py` |      208 | 5 | 18 |
| `api/main.py` |      148 | 2 | 15 |
| `api/jobs/store.py` |      145 | 3 | 14 |
| `api/services/backtest_service.py` |      139 | 4 | 23 |
| `api/services/autopilot_service.py` |      131 | 4 | 17 |
| `api/routers/dashboard.py` |      127 | 6 | 13 |
| `api/routers/system_health.py` |      123 | 6 | 10 |
| `api/jobs/runner.py` |      115 | 3 | 10 |
| `api/routers/model_lab.py` |      113 | 7 | 6 |
| `api/services/results_service.py` |      106 | 4 | 13 |
| `api/routers/autopilot.py` |       96 | 6 | 6 |
| `api/routers/backtests.py` |       94 | 5 | 8 |
| `api/routers/jobs.py` |       93 | 5 | 8 |
| `api/services/diagnostics.py` |       84 | 2 | 6 |
| `api/services/model_service.py` |       81 | 3 | 9 |
| `api/errors.py` |       77 | 7 | 2 |
| `api/config.py` |       75 | 5 | 5 |
| `api/services/regime_service.py` |       67 | 3 | 7 |
| `api/cache/manager.py` |       62 | 7 | 4 |
| `api/routers/diagnostics.py` |       54 | 3 | 4 |
| `api/deps/providers.py` |       54 | 5 | 3 |
| `api/services/kalshi_service.py` |       52 | 3 | 4 |
| `api/routers/regime.py` |       50 | 2 | 6 |
| `api/routers/risk.py` |       44 | 3 | 3 |
| `api/routers/__init__.py` |       42 | 1 | 3 |
| `api/schemas/compute.py` |       42 | 5 | 0 |
| `api/schemas/backtests.py` |       39 | 4 | 0 |
| `api/routers/benchmark.py` |       37 | 1 | 3 |
| `api/schemas/system_health.py` |       36 | 3 | 0 |
| `api/routers/iv_surface.py` |       33 | 1 | 2 |
| `api/routers/logs.py` |       33 | 2 | 3 |
| `api/jobs/models.py` |       32 | 2 | 0 |
| `api/schemas/model_lab.py` |       31 | 2 | 0 |
| `api/jobs/autopilot_job.py` |       31 | 1 | 4 |
| `api/jobs/train_job.py` |       30 | 1 | 0 |
| `api/routers/signals.py` |       29 | 2 | 0 |
| `api/jobs/backtest_job.py` |       29 | 1 | 0 |
| `api/routers/config_mgmt.py` |       29 | 2 | 2 |
| `api/jobs/predict_job.py` |       28 | 1 | 0 |
| `api/schemas/envelope.py` |       27 | 3 | 0 |
| `api/schemas/dashboard.py` |       26 | 3 | 0 |
| `api/schemas/signals.py` |       23 | 2 | 0 |
| `api/schemas/autopilot.py` |       22 | 2 | 0 |
| `api/schemas/data_explorer.py` |       19 | 2 | 0 |
| `api/deps/__init__.py` |       16 | 0 | 0 |
| `api/services/__init__.py` |       15 | 0 | 0 |
| `api/routers/data_explorer.py` |      490 | 13 | 40 |
| `api/cache/invalidation.py` |       32 | 4 | 0 |
| `api/jobs/__init__.py` |        6 | 0 | 0 |
| `api/cache/__init__.py` |        4 | 0 | 0 |
| `api/schemas/__init__.py` |        4 | 0 | 0 |
| `api/__init__.py` |        1 | 0 | 0 |

### Existing patterns to follow
- Router -> service -> orchestrator/core layering.
- Extensive lazy import usage for resilience and optional dependencies.
- Shared response envelope and job status enums are canonical contracts.

### Configuration
- Key boundaries: `api_to_config_36`, `api_to_data_37`, `api_to_features_38`, `api_to_regime_39`, `api_to_models_40`, `api_to_backtest_41`, `api_to_risk_42`, `api_to_autopilot_43`, `api_to_kalshi_44`, `autopilot_to_api_circular_5`.
- API contract invariants: [SYSTEM_CONTRACTS_AND_INVARIANTS.md](/Users/justinblaise/Documents/quant_engine/docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md).

## Tasks

### T1: Ledger and API contract baseline

**What:** Build full line ledger and snapshot API envelope + endpoint inventory.

**Files:**
- `api/schemas/envelope.py`
- `api/main.py`
- `api/routers/__init__.py`

**Implementation notes:**
- Freeze mounted routes and envelope expectations before deeper audit.

**Verify:**
```bash
jq -r '.subsystems.api_frontend.files[]' docs/architecture/SUBSYSTEM_MAP.json | xargs wc -l
```

---

### T2: Orchestrator and core-service contract pass

**What:** Audit orchestrator and major service adapters for upstream contract correctness.

**Files:**
- `api/orchestrator.py`
- `api/services/data_helpers.py`
- `api/services/health_service.py`

**Implementation notes:**
- Validate lazy import safeguards and failure surfaces.
- Confirm service outputs honor downstream schema expectations.

**Verify:**
```bash
rg -n "try:|except|import|ApiResponse|orchestr|health|load_" api/orchestrator.py api/services/data_helpers.py api/services/health_service.py
```

---

### T3: Job system lifecycle pass

**What:** Audit job persistence, execution, status transitions, and SSE event streaming.

**Files:**
- `api/jobs/store.py`
- `api/jobs/runner.py`
- `api/routers/jobs.py`

**Implementation notes:**
- Validate canonical status set and transitions.
- Confirm cancellation and failure propagation semantics.

**Verify:**
```bash
rg -n "queued|running|succeeded|failed|cancelled|events|sse|stream" api/jobs/store.py api/jobs/runner.py api/routers/jobs.py api/jobs/models.py
```

---

### T4: Router envelope consistency pass

**What:** Audit all routers for `ApiResponse` envelope and error/meta consistency.

**Files:**
- `api/routers/*.py`
- `api/schemas/envelope.py`
- `frontend/src/api/client.ts` (read-only contract check)

**Implementation notes:**
- Verify every route response can be parsed by frontend client assumptions.
- Confirm error paths return canonical envelope shape.

**Verify:**
```bash
rg -n "ApiResponse|ok=|error=|meta=|HTTPException" api/routers/*.py api/schemas/envelope.py
```

---

### T5: Cache and invalidation pass

**What:** Audit TTL cache behavior and invalidation hooks tied to train/backtest/data refresh.

**Files:**
- `api/cache/manager.py`
- `api/cache/invalidation.py`
- `api/services/backtest_service.py`

**Implementation notes:**
- Confirm stale cache cannot survive lifecycle events that require refresh.

**Verify:**
```bash
rg -n "cache|ttl|invalidate|refresh|stale" api/cache/manager.py api/cache/invalidation.py api/services/*.py
```

---

### T6: Circular coupling and boundary pass

**What:** Validate API<->autopilot and all cross-subsystem lazy dependency boundaries.

**Files:**
- `api/jobs/autopilot_job.py`
- `api/services/health_service.py`
- `api/services/health_risk_feedback.py`

**Implementation notes:**
- Confirm architectural cycle is non-SCC and runtime-safe.
- Validate assumptions consumed by autopilot paper trader.

**Verify:**
```bash
jq -r '.edges[] | select(.source_module=="api" or .target_module=="api") | [.source_file,.target_file,.import_type] | @tsv' docs/audit/DEPENDENCY_EDGES.json | head -n 180
```

---

### T7: Findings synthesis and release gate

**What:** Publish full findings matrix and required mitigations.

**Files:**
- All subsystem files.

**Implementation notes:**
- Include contract drift section for known frontend/backend job-status mismatch.

**Verify:**
```bash
# Manual gate: 10188/10188 lines reviewed; all HIGH api boundaries dispositioned
```

## Validation

### Acceptance criteria
1. Full line coverage across all 59 subsystem files.
2. API envelope/job lifecycle contracts are verified across routers/services/jobs.
3. Cross-subsystem lazy imports are validated for graceful degradation.
4. Circular API-autopilot architectural concern is explicitly dispositioned.

### Verification steps
```bash
jq -r '.subsystems.api_frontend.files[]' docs/architecture/SUBSYSTEM_MAP.json
jq -r '.edges[] | select(.source_module=="api" or .target_module=="api") | .import_type' docs/audit/DEPENDENCY_EDGES.json | sort | uniq -c
```

### Rollback plan
- Revert this spec file if API subsystem scope changes.

---

## Notes

API & Frontend should be audited after core engine subsystems because it is a broad consumer layer with many lazy cross-subsystem imports.
