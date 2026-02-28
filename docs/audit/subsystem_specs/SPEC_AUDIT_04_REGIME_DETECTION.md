# Feature Spec: Subsystem Audit Spec — Regime Detection

> **Status:** Draft
> **Author:** Codex
> **Date:** 2026-02-27
> **Estimated effort:** 12-16 hours across 6 tasks

---

## Why

Regime outputs drive feature enrichment, position sizing, and structural shock handling. Schema or threshold drift in `ShockVector` and `UncertaintyGate` can silently alter backtests, risk, and autopilot behavior.

## What

Define a complete line-by-line audit spec for Subsystem `regime_detection` that verifies detector correctness, structural schema stability, and downstream contract integrity.

## Constraints

### Must-haves
- Audit all `13` files (`4,420` lines) with full line ledger.
- Validate canonical regime ID/name mapping and confidence semantics.
- Enforce versioned `ShockVector` schema compatibility.
- Validate `UncertaintyGate` thresholds against downstream sizing consumers.

### Must-nots
- No detector parameter or threshold changes during audit.
- No schema changes without migration/versioning plan.
- No unresolved high-risk contract findings.

### Out of scope
- Portfolio/risk optimization internals outside regime interfaces.
- API routing/UI representation changes.

## Current State

Subsystem definition and dependency edges are in [SUBSYSTEM_MAP.json](/Users/justinblaise/Documents/quant_engine/docs/audit/data/SUBSYSTEM_MAP.json) and [SUBSYSTEM_AUDIT_MANIFEST.md](/Users/justinblaise/Documents/quant_engine/docs/audit/data/SUBSYSTEM_AUDIT_MANIFEST.md).

### Key files
| File | Lines | Def/Class Count | Branch Points |
|---|---:|---:|---:|
| `regime/detector.py` |      940 | 21 | 84 |
| `regime/hmm.py` |      661 | 28 | 67 |
| `regime/shock_vector.py` |      494 | 15 | 36 |
| `regime/bocpd.py` |      451 | 6 | 11 |
| `regime/jump_model_pypi.py` |      420 | 8 | 25 |
| `regime/consensus.py` |      273 | 8 | 8 |
| `regime/confidence_calibrator.py` |      251 | 5 | 13 |
| `regime/online_update.py` |      245 | 5 | 10 |
| `regime/jump_model_legacy.py` |      242 | 9 | 15 |
| `regime/correlation.py` |      213 | 3 | 4 |
| `regime/uncertainty_gate.py` |      180 | 3 | 4 |
| `regime/__init__.py` |       30 | 0 | 0 |
| `regime/jump_model.py` |       20 | 0 | 0 |

### Existing patterns to follow
- `RegimeDetector` orchestrates multiple models with fallbacks and consensus logic.
- `ShockVector` is a serialized interface used outside this subsystem.
- Uncertainty gating is a shared behavior contract, not local-only logic.

### Configuration
- Canonical regime invariants: [SYSTEM_CONTRACTS_AND_INVARIANTS.md](/Users/justinblaise/Documents/quant_engine/docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md).
- High-risk boundaries: `backtest_to_regime_shock_1`, `backtest_to_regime_uncertainty_2`, `risk_to_regime_22`, `autopilot_to_regime_29`, `features_to_regime_15` in [INTERFACE_CONTRACTS.yaml](/Users/justinblaise/Documents/quant_engine/docs/audit/data/INTERFACE_CONTRACTS.yaml).

## Tasks

### T1: Ledger and schema baseline

**What:** Establish complete line ledger and structural schema snapshots.

**Files:**
- `regime/detector.py`
- `regime/shock_vector.py`
- `regime/uncertainty_gate.py`

**Implementation notes:**
- Snapshot `ShockVector` fields and serialization methods.
- Record uncertainty threshold defaults and sizing map behavior.

**Verify:**
```bash
jq -r '.subsystems.regime_detection.files[]' docs/audit/data/SUBSYSTEM_MAP.json | xargs wc -l
```

---

### T2: Detector ensemble correctness pass

**What:** Audit HMM/jump/BOCPD/correlation integration and fallback logic.

**Files:**
- `regime/detector.py`
- `regime/hmm.py`
- `regime/bocpd.py`

**Implementation notes:**
- Validate state mapping to `REGIME_NAMES` (0..3).
- Confirm disagreement handling and confidence propagation.

**Verify:**
```bash
rg -n "REGIME_NAMES|consensus|confidence|fallback|detect" regime/detector.py regime/hmm.py regime/bocpd.py
```

---

### T3: ShockVector contract pass

**What:** Validate versioned schema and serialization/deserialization compatibility.

**Files:**
- `regime/shock_vector.py`
- `backtest/execution.py` (read-only consumer check)
- `backtest/engine.py` (read-only consumer check)

**Implementation notes:**
- Ensure `to_dict`/`from_dict` fields are stable and complete.
- Confirm consumer assumptions match schema versions.

**Verify:**
```bash
rg -n "ShockVector|schema_version|to_dict|from_dict" regime/shock_vector.py backtest/execution.py backtest/engine.py
```

---

### T4: UncertaintyGate behavioral pass

**What:** Validate uncertainty thresholds and multiplier semantics used by downstream sizing logic.

**Files:**
- `regime/uncertainty_gate.py`
- `risk/position_sizer.py` (read-only consumer check)
- `autopilot/engine.py` (read-only consumer check)

**Implementation notes:**
- Trace uncertainty inputs and multiplier outputs end-to-end.
- Confirm behavior under missing/edge uncertainty values.

**Verify:**
```bash
rg -n "UncertaintyGate|uncertainty|entropy|sizing_map|multiplier" regime/uncertainty_gate.py risk/position_sizer.py autopilot/engine.py
```

---

### T5: Config and boundary integrity pass

**What:** Validate all config-driven behavior and cross-subsystem boundary assumptions.

**Files:**
- `regime/detector.py`
- `regime/consensus.py`
- `regime/online_update.py`

**Implementation notes:**
- Confirm all lazy config imports are explicit and safe.
- Validate no implicit dependency on unavailable optional packages.

**Verify:**
```bash
jq -r '.edges[] | select(.source_module=="regime" or .target_module=="regime") | [.source_file,.target_file,.import_type] | @tsv' docs/audit/data/DEPENDENCY_EDGES.json | head -n 100
```

---

### T6: Findings consolidation and closure

**What:** Publish severity-ranked findings and required test/risk follow-up.

**Files:**
- All subsystem files.

**Implementation notes:**
- Include explicit compatibility verdict for `ShockVector` and `UncertaintyGate` contracts.

**Verify:**
```bash
# Manual gate: 4420/4420 lines reviewed; all HIGH regime boundaries dispositioned
```

## Validation

### Acceptance criteria
1. Full line coverage across all 13 files.
2. Regime ID/name and confidence contracts are confirmed.
3. `ShockVector` schema compatibility with consumers is confirmed.
4. `UncertaintyGate` behavior is validated across backtest/risk/autopilot consumers.

### Verification steps
```bash
jq -r '.subsystems.regime_detection.files[]' docs/audit/data/SUBSYSTEM_MAP.json
jq -r '.edges[] | select(.source_module=="regime" or .target_module=="regime") | .import_type' docs/audit/data/DEPENDENCY_EDGES.json | sort | uniq -c
```

### Rollback plan
- Revert this spec file if subsystem boundaries or schema ownership change.

---

## Notes

This subsystem has high interface sensitivity despite moderate size because its outputs directly alter execution and sizing logic downstream.
