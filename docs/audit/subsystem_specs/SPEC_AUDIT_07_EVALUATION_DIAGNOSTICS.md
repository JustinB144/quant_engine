# Feature Spec: Subsystem Audit Spec — Evaluation & Diagnostics

> **Status:** Draft
> **Author:** Codex
> **Date:** 2026-02-27
> **Estimated effort:** 6-10 hours across 5 tasks

---

## Why

Evaluation is the decision-quality layer for model/backtest validity. It must preserve time-ordering and correctly interpret calibration/fragility diagnostics without introducing analytical leakage or reporting drift.

## What

Define a complete audit spec for Subsystem `evaluation_diagnostics` covering all lines, dependency contracts to models/backtest, and metric/reporting correctness.

## Constraints

### Must-haves
- Audit all `8` files (`2,816` lines).
- Validate walk-forward/embargo and IC decay diagnostic usage.
- Validate calibration interfaces imported from models.
- Validate red-flag logic and metric thresholds.

### Must-nots
- No metric-definition changes during audit.
- No bypassing time-order or embargo semantics.
- No unresolved medium/high findings on report correctness.

### Out of scope
- Training engine refactors.
- API routing/cache implementation details.

## Current State

Subsystem details and dependency order are defined in [SUBSYSTEM_AUDIT_MANIFEST.md](/Users/justinblaise/Documents/quant_engine/docs/audit/data/SUBSYSTEM_AUDIT_MANIFEST.md).

### Key files
| File | Lines | Def/Class Count | Branch Points |
|---|---:|---:|---:|
| `evaluation/engine.py` |      826 | 15 | 86 |
| `evaluation/fragility.py` |      452 | 8 | 31 |
| `evaluation/slicing.py` |      400 | 8 | 23 |
| `evaluation/visualization.py` |      386 | 7 | 11 |
| `evaluation/metrics.py` |      324 | 8 | 28 |
| `evaluation/ml_diagnostics.py` |      253 | 6 | 18 |
| `evaluation/calibration_analysis.py` |      142 | 3 | 4 |
| `evaluation/__init__.py` |       33 | 0 | 0 |

### Existing patterns to follow
- Evaluation engine orchestrates external metrics via lazy imports from backtest/models.
- Visualization helpers should not alter metric semantics.
- Diagnostic thresholds are config-driven and must remain explicit.

### Configuration
- Relevant boundaries: `evaluation_to_models_backtest_11`, `evaluation_to_config_23`.
- Invariants: [SYSTEM_CONTRACTS_AND_INVARIANTS.md](/Users/justinblaise/Documents/quant_engine/docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md).

## Tasks

### T1: Ledger and metric inventory baseline

**What:** Build full line ledger and metric registry snapshot.

**Files:**
- `evaluation/engine.py`
- `evaluation/metrics.py`
- `evaluation/slicing.py`

**Implementation notes:**
- Map all produced metric names/fields to downstream consumers.

**Verify:**
```bash
jq -r '.subsystems.evaluation_diagnostics.files[]' docs/audit/data/SUBSYSTEM_MAP.json | xargs wc -l
```

---

### T2: Time-ordering and validation contract pass

**What:** Audit usage of walk-forward/embargo/IC diagnostics from backtest validation.

**Files:**
- `evaluation/engine.py`
- `backtest/validation.py` (read-only contract check)
- `evaluation/fragility.py`

**Implementation notes:**
- Confirm no future data access in evaluation windows.
- Validate decay/fragility triggers and edge-case handling.

**Verify:**
```bash
rg -n "walk_forward|embargo|rolling_ic|detect_ic_decay|fragility|drawdown" evaluation/engine.py evaluation/fragility.py backtest/validation.py
```

---

### T3: Calibration and diagnostics pass

**What:** Validate calibration interfaces and ML diagnostics outputs.

**Files:**
- `evaluation/calibration_analysis.py`
- `models/calibration.py` (read-only contract check)
- `evaluation/ml_diagnostics.py`

**Implementation notes:**
- Confirm expected keys/shapes from reliability/ECE functions.
- Validate thresholds and interpretation paths.

**Verify:**
```bash
rg -n "compute_ece|compute_reliability_curve|drift|disagreement|threshold" evaluation/calibration_analysis.py evaluation/ml_diagnostics.py models/calibration.py
```

---

### T4: Visualization and reporting integrity pass

**What:** Audit visualization/report generation for deterministic and accurate representations.

**Files:**
- `evaluation/visualization.py`
- `evaluation/engine.py`
- `evaluation/metrics.py`

**Implementation notes:**
- Confirm visualization functions do not mutate source metric values.
- Validate missing-data and low-sample handling.

**Verify:**
```bash
rg -n "plot|figure|chart|nan|min_samples|insufficient" evaluation/visualization.py evaluation/engine.py evaluation/metrics.py
```

---

### T5: Boundary checks and closure

**What:** Validate all external interfaces and close with severity-ranked findings.

**Files:**
- All subsystem files.

**Implementation notes:**
- Explicitly disposition each contract in `evaluation_to_models_backtest_11`.

**Verify:**
```bash
jq -r '.edges[] | select(.source_module=="evaluation" or .target_module=="evaluation") | [.source_file,.target_file,.import_type] | @tsv' docs/audit/data/DEPENDENCY_EDGES.json
```

## Validation

### Acceptance criteria
1. 100% line coverage across all 8 files.
2. Time-order/embargo-dependent diagnostics are validated.
3. Calibration contracts with models are validated.
4. Evaluation report fields are stable and accurately derived.

### Verification steps
```bash
jq -r '.subsystems.evaluation_diagnostics.files[]' docs/audit/data/SUBSYSTEM_MAP.json
jq -r '.edges[] | select(.source_module=="evaluation" or .target_module=="evaluation") | .import_type' docs/audit/data/DEPENDENCY_EDGES.json | sort | uniq -c
```

### Rollback plan
- Revert this spec document if evaluation subsystem boundaries are revised.

---

## Notes

Although smaller than core execution modules, this subsystem governs trust in model/backtest outcomes and must be audited with the same rigor.
