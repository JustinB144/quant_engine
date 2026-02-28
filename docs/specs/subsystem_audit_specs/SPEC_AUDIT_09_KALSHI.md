# Feature Spec: Subsystem Audit Spec — Kalshi (Event Markets)

> **Status:** Draft
> **Author:** Codex
> **Date:** 2026-02-27
> **Estimated effort:** 12-16 hours across 6 tasks

---

## Why

Kalshi is a semi-isolated vertical with strict event-time and distribution-quality contracts. Leakage or schema drift here can invalidate event strategy research, promotion outcomes, and API consumers.

## What

Define exhaustive audit coverage for Subsystem `kalshi`, including source and co-located test files, with full line review and contract verification across storage/distribution/event/promotion boundaries.

## Constraints

### Must-haves
- Audit all `25` files (`6,096` lines).
- Validate as-of join correctness and leakage protections.
- Validate `EventTimeStore` schema compatibility and migration safety.
- Validate promotion/walk-forward integration with autopilot/backtest contracts.

### Must-nots
- No schema mutation in storage layer during audit.
- No relaxed leakage checks without explicit risk acceptance.
- No unresolved medium/high findings in provider/distribution/promotion/storage hotspots.

### Out of scope
- Core equity model training logic.
- Frontend-only UX behavior.

## Current State

Subsystem mapping and boundaries are in [SUBSYSTEM_MAP.json](/Users/justinblaise/Documents/quant_engine/docs/architecture/SUBSYSTEM_MAP.json), [SUBSYSTEM_AUDIT_MANIFEST.md](/Users/justinblaise/Documents/quant_engine/docs/architecture/SUBSYSTEM_AUDIT_MANIFEST.md), and [INTERFACE_CONTRACTS.yaml](/Users/justinblaise/Documents/quant_engine/docs/audit/INTERFACE_CONTRACTS.yaml).

### Key files
| File | Lines | Def/Class Count | Branch Points |
|---|---:|---:|---:|
| `kalshi/distribution.py` |      935 | 23 | 83 |
| `kalshi/client.py` |      655 | 24 | 79 |
| `kalshi/storage.py` |      649 | 23 | 37 |
| `kalshi/provider.py` |      647 | 28 | 64 |
| `kalshi/events.py` |      517 | 17 | 48 |
| `kalshi/walkforward.py` |      492 | 17 | 39 |
| `kalshi/quality.py` |      206 | 8 | 15 |
| `kalshi/promotion.py` |      177 | 6 | 11 |
| `kalshi/pipeline.py` |      167 | 4 | 12 |
| `kalshi/microstructure.py` |      145 | 5 | 13 |
| `kalshi/options.py` |      145 | 4 | 15 |
| `kalshi/regimes.py` |      142 | 4 | 15 |
| `kalshi/disagreement.py` |      140 | 5 | 12 |
| `kalshi/mapping_store.py` |      134 | 6 | 5 |
| `kalshi/router.py` |       78 | 2 | 9 |
| `kalshi/tests/test_distribution.py` |       62 | 2 | 4 |
| `kalshi/tests/test_bin_validity.py` |       57 | 2 | 0 |
| `kalshi/tests/test_threshold_direction.py` |       56 | 2 | 0 |
| `kalshi/tests/test_no_leakage.py` |       49 | 2 | 0 |
| `kalshi/tests/test_stale_quotes.py` |       48 | 2 | 0 |
| `kalshi/tests/test_walkforward_purge.py` |       45 | 2 | 0 |
| `kalshi/tests/test_signature_kat.py` |       31 | 2 | 0 |
| `kalshi/tests/test_leakage.py` |       29 | 2 | 0 |
| `kalshi/__init__.py` |       18 | 0 | 0 |
| `kalshi/tests/__init__.py` |        1 | 0 | 0 |

### Existing patterns to follow
- Event-time correctness relies on strict backward-looking joins.
- Distribution repair/quality filtering is a first-class contract.
- Promotion path translates Kalshi metrics into backtest/autopilot-compatible structures.

### Configuration
- Key boundaries: `kalshi_to_autopilot_9`, `kalshi_to_backtest_33`, `kalshi_to_config_34`, `kalshi_to_features_35`, `api_to_kalshi_44`, `data_to_kalshi_32`.
- Kalshi invariants: [SYSTEM_CONTRACTS_AND_INVARIANTS.md](/Users/justinblaise/Documents/quant_engine/docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md).

## Tasks

### T1: Ledger and schema baseline

**What:** Build complete line ledger and schema inventory for storage and output artifacts.

**Files:**
- `kalshi/storage.py`
- `kalshi/provider.py`
- `kalshi/distribution.py`

**Implementation notes:**
- Snapshot table schemas, key columns, and API between storage/provider.

**Verify:**
```bash
jq -r '.subsystems.kalshi.files[]' docs/architecture/SUBSYSTEM_MAP.json | xargs wc -l
```

---

### T2: Leakage/as-of correctness pass

**What:** Audit event feature construction and as-of join semantics.

**Files:**
- `kalshi/events.py`
- `kalshi/pipeline.py`
- `kalshi/tests/test_no_leakage.py`

**Implementation notes:**
- Validate all joins are backward-looking only.
- Confirm tests reflect production join behavior.

**Verify:**
```bash
rg -n "asof|join|timestamp|leak|future|purge|embargo" kalshi/events.py kalshi/pipeline.py kalshi/tests/*.py
```

---

### T3: Distribution and quote-quality pass

**What:** Audit distribution reconstruction, monotonic repairs, and stale quote handling.

**Files:**
- `kalshi/distribution.py`
- `kalshi/quality.py`
- `kalshi/tests/test_distribution.py`

**Implementation notes:**
- Validate threshold-direction logic and repair correctness.
- Confirm quality flags are explicit and consumer-safe.

**Verify:**
```bash
rg -n "monotonic|stale|quality|threshold|repair|probability" kalshi/distribution.py kalshi/quality.py kalshi/tests/test_*.py
```

---

### T4: Walk-forward and promotion contract pass

**What:** Audit event walk-forward validation and promotion conversion to core contracts.

**Files:**
- `kalshi/walkforward.py`
- `kalshi/promotion.py`
- `autopilot/promotion_gate.py` (read-only contract check)

**Implementation notes:**
- Validate compatibility with `BacktestResult` and `PromotionDecision` expectations.
- Confirm no metric-field drift in conversion logic.

**Verify:**
```bash
rg -n "BacktestResult|PromotionDecision|deflated_sharpe|monte_carlo|promotion" kalshi/walkforward.py kalshi/promotion.py autopilot/promotion_gate.py
```

---

### T5: Boundary and integration pass

**What:** Validate cross-subsystem imports and lazy behavior with API/data consumers.

**Files:**
- `kalshi/provider.py`
- `kalshi/options.py`
- `kalshi/storage.py`

**Implementation notes:**
- Confirm config-feature-backtest-autopilot contracts are explicit.
- Validate `api/services/kalshi_service.py` and `data/provider_registry.py` assumptions.

**Verify:**
```bash
jq -r '.edges[] | select(.source_module=="kalshi" or .target_module=="kalshi") | [.source_file,.target_file,.import_type] | @tsv' docs/audit/DEPENDENCY_EDGES.json | head -n 120
```

---

### T6: Findings synthesis and closure

**What:** Publish complete defect matrix and remediation plan.

**Files:**
- All subsystem files.

**Implementation notes:**
- Include explicit leakage and schema compatibility verdicts.

**Verify:**
```bash
# Manual gate: 6096/6096 lines reviewed; all MEDIUM/HIGH kalshi boundaries dispositioned
```

## Validation

### Acceptance criteria
1. 100% line coverage across all 25 Kalshi files.
2. As-of/leakage contracts are validated with test alignment.
3. Storage and distribution schemas are confirmed stable.
4. Promotion/walk-forward interfaces to core subsystems are validated.

### Verification steps
```bash
jq -r '.subsystems.kalshi.files[]' docs/architecture/SUBSYSTEM_MAP.json
jq -r '.edges[] | select(.source_module=="kalshi" or .target_module=="kalshi") | .import_type' docs/audit/DEPENDENCY_EDGES.json | sort | uniq -c
```

### Rollback plan
- Revert this spec file if Kalshi subsystem scope changes.

---

## Notes

Kalshi has lower global blast radius than core equity subsystems but high contract sensitivity within its event-time and promotion pathways.
