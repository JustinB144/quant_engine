# Feature Spec: Subsystem Audit Spec — Autopilot (Strategy Discovery)

> **Status:** Draft
> **Author:** Codex
> **Date:** 2026-02-27
> **Estimated effort:** 16-24 hours across 7 tasks

---

## Why

Autopilot is the integration-dense orchestration layer that couples data, features, regime, models, backtesting, risk, and API health feedback. It is a high-risk zone for contract drift and architectural coupling regressions.

## What

Define a full-line audit spec for Subsystem `autopilot` with explicit contract checks across all upstream dependencies and artifact/state schema guarantees.

## Constraints

### Must-haves
- Audit all `8` files (`4,480` lines).
- Validate promotion-gate correctness (DSR/PBO/CPCV/statistical requirements).
- Validate paper-trader parity with execution/risk contracts.
- Validate lazy API-health coupling behavior and degradation safety.

### Must-nots
- No strategy-selection or threshold retuning during audit.
- No artifact schema drift in `latest_cycle.json`, `strategy_registry.json`, or `paper_state.json`.
- No unresolved `P0`/`P1` on `autopilot/engine.py`, `autopilot/paper_trader.py`, `autopilot/promotion_gate.py`.

### Out of scope
- Frontend page composition.
- Non-autopilot API endpoints unrelated to health/autopilot jobs.

## Current State

Subsystem metadata is in [SUBSYSTEM_MAP.json](/Users/justinblaise/Documents/quant_engine/docs/architecture/SUBSYSTEM_MAP.json) and [SUBSYSTEM_AUDIT_MANIFEST.md](/Users/justinblaise/Documents/quant_engine/docs/architecture/SUBSYSTEM_AUDIT_MANIFEST.md).

### Key files
| File | Lines | Def/Class Count | Branch Points |
|---|---:|---:|---:|
| `autopilot/engine.py` |     1927 | 38 | 222 |
| `autopilot/paper_trader.py` |     1254 | 26 | 110 |
| `autopilot/meta_labeler.py` |      468 | 8 | 26 |
| `autopilot/promotion_gate.py` |      424 | 9 | 60 |
| `autopilot/strategy_allocator.py` |      196 | 6 | 5 |
| `autopilot/registry.py` |      110 | 3 | 4 |
| `autopilot/strategy_discovery.py` |       79 | 2 | 4 |
| `autopilot/__init__.py` |       22 | 0 | 0 |

### Existing patterns to follow
- `autopilot/engine.py` performs broad orchestration with both top-level and lazy imports.
- `paper_trader.py` introduces architectural coupling to API health services.
- Promotion and paper-trading artifacts are shared interfaces with API consumers.

### Configuration
- Key boundaries: `autopilot_to_multi_4`, `autopilot_to_api_circular_5`, `autopilot_to_backtest_31`, `autopilot_to_models_28`, `autopilot_to_risk_30`, `api_to_autopilot_43`, `kalshi_to_autopilot_9`.
- Architectural cycle notes: [SUBSYSTEM_AUDIT_MANIFEST.md](/Users/justinblaise/Documents/quant_engine/docs/architecture/SUBSYSTEM_AUDIT_MANIFEST.md).

## Tasks

### T1: Ledger and shared-artifact schema baseline

**What:** Create full line ledger and snapshot all autopilot output schemas.

**Files:**
- `autopilot/engine.py`
- `autopilot/paper_trader.py`
- `autopilot/registry.py`

**Implementation notes:**
- Capture schema fields for `latest_cycle.json`, `paper_state.json`, `strategy_registry.json`.
- Map each field to API service consumers.

**Verify:**
```bash
jq -r '.subsystems.autopilot.files[]' docs/architecture/SUBSYSTEM_MAP.json | xargs wc -l
```

---

### T2: Orchestration correctness pass

**What:** Audit discovery->validation->promotion->paper-trading control flow.

**Files:**
- `autopilot/engine.py`
- `autopilot/strategy_discovery.py`
- `autopilot/promotion_gate.py`

**Implementation notes:**
- Validate fallback behavior when predictor/model artifacts are unavailable.
- Confirm evaluation metrics consumed by promotion gate are correctly derived.

**Verify:**
```bash
rg -n "run_cycle|discover|promot|fallback|validate|Backtester|ModelTrainer|EnsemblePredictor" autopilot/engine.py autopilot/strategy_discovery.py autopilot/promotion_gate.py
```

---

### T3: Paper-trading execution/risk parity pass

**What:** Audit execution model, sizing, stops, and drawdown controls in paper trading.

**Files:**
- `autopilot/paper_trader.py`
- `backtest/execution.py` (read-only parity check)
- `risk/position_sizer.py` (read-only parity check)

**Implementation notes:**
- Confirm behavior parity with backtest assumptions where required.
- Validate state recovery and persistence reliability.

**Verify:**
```bash
rg -n "ExecutionModel|PositionSizer|StopLoss|Drawdown|paper_state|recover" autopilot/paper_trader.py backtest/execution.py risk/position_sizer.py
```

---

### T4: API coupling and degradation-safety pass

**What:** Audit lazy API imports and resilience when health services are unavailable.

**Files:**
- `autopilot/paper_trader.py`
- `autopilot/engine.py`
- `api/services/health_service.py` (read-only contract check)

**Implementation notes:**
- Validate all lazy imports are guarded and behavior is explicit on import/runtime failures.
- Confirm no hidden hard dependency on API boot order.

**Verify:**
```bash
rg -n "HealthService|health_risk_feedback|ABTest|try:|except" autopilot/paper_trader.py autopilot/engine.py api/services/health_service.py
```

---

### T5: Meta-labeling and allocation pass

**What:** Audit confidence filtering and strategy allocation behavior.

**Files:**
- `autopilot/meta_labeler.py`
- `autopilot/strategy_allocator.py`
- `autopilot/registry.py`

**Implementation notes:**
- Validate model persistence/version assumptions and state transitions.
- Confirm allocation policy handles missing regime/confidence safely.

**Verify:**
```bash
rg -n "meta|confidence|allocate|regime|registry|active" autopilot/meta_labeler.py autopilot/strategy_allocator.py autopilot/registry.py
```

---

### T6: Boundary contract pass

**What:** Validate autopilot contracts with backtest/models/risk/api/kalshi consumers.

**Files:**
- `autopilot/engine.py`
- `autopilot/promotion_gate.py`
- `autopilot/paper_trader.py`

**Implementation notes:**
- Explicitly disposition `autopilot_to_api_circular_5` and `autopilot_to_multi_4`.

**Verify:**
```bash
jq -r '.edges[] | select(.source_module=="autopilot" or .target_module=="autopilot") | [.source_file,.target_file,.import_type] | @tsv' docs/audit/DEPENDENCY_EDGES.json | head -n 140
```

---

### T7: Findings publication and closure

**What:** Publish severity-ranked findings with immediate remediation gates.

**Files:**
- All subsystem files.

**Implementation notes:**
- Include explicit architectural concern section for autopilot<->api coupling.

**Verify:**
```bash
# Manual gate: 4480/4480 lines reviewed; all HIGH autopilot boundaries dispositioned
```

## Validation

### Acceptance criteria
1. Full line coverage across all 8 autopilot files.
2. Promotion, paper-trading, and artifact contracts are validated.
3. API coupling degradation paths are verified.
4. No unresolved `P0`/`P1` findings in hotspot files.

### Verification steps
```bash
jq -r '.subsystems.autopilot.files[]' docs/architecture/SUBSYSTEM_MAP.json
jq -r '.edges[] | select(.source_module=="autopilot" or .target_module=="autopilot") | .import_type' docs/audit/DEPENDENCY_EDGES.json | sort | uniq -c
```

### Rollback plan
- Revert this spec file if autopilot subsystem boundaries are redefined.

---

## Notes

Autopilot should be audited after core subsystems because it composes contracts from nearly every upstream domain.
