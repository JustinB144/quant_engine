# Feature Spec: Subsystem Audit Spec — Model Training & Prediction

> **Status:** Draft
> **Author:** Codex
> **Date:** 2026-02-27
> **Estimated effort:** 16-20 hours across 6 tasks

---

## Why

Model training/prediction defines artifact truth and live inference behavior. Contract drift between trainer and predictor or versioning/governance inconsistencies can silently break production signal correctness.

## What

Define complete audit coverage for Subsystem `model_training_prediction` with full line review, artifact schema checks, and cross-subsystem contract validation.

## Constraints

### Must-haves
- Audit all `16` files (`6,153` lines).
- Verify trainer/predictor feature ordering and causality assumptions.
- Validate model versioning/governance registry integrity.
- Validate retrain triggers and distribution-shift sensitivity logic.

### Must-nots
- No retraining or model replacement during audit.
- No unversioned artifact format changes.
- No unresolved high-risk findings in `models/trainer.py`, `models/predictor.py`, `models/versioning.py`.

### Out of scope
- API job orchestration internals.
- Portfolio optimization/risk policy redesign.

## Current State

Subsystem details are defined in [SUBSYSTEM_MAP.json](/Users/justinblaise/Documents/quant_engine/docs/architecture/SUBSYSTEM_MAP.json) and [SUBSYSTEM_AUDIT_MANIFEST.md](/Users/justinblaise/Documents/quant_engine/docs/architecture/SUBSYSTEM_AUDIT_MANIFEST.md).

### Key files
| File | Lines | Def/Class Count | Branch Points |
|---|---:|---:|---:|
| `models/trainer.py` |     1818 | 44 | 195 |
| `models/iv/models.py` |      937 | 35 | 57 |
| `models/predictor.py` |      538 | 19 | 53 |
| `models/retrain_trigger.py` |      344 | 13 | 30 |
| `models/calibration.py` |      327 | 11 | 21 |
| `models/shift_detection.py` |      322 | 9 | 17 |
| `models/feature_stability.py` |      313 | 11 | 13 |
| `models/conformal.py` |      295 | 11 | 16 |
| `models/online_learning.py` |      273 | 9 | 17 |
| `models/walk_forward.py` |      235 | 6 | 16 |
| `models/versioning.py` |      207 | 8 | 16 |
| `models/neural_net.py` |      198 | 4 | 9 |
| `models/governance.py` |      155 | 7 | 11 |
| `models/cross_sectional.py` |      136 | 3 | 5 |
| `models/__init__.py` |       30 | 0 | 0 |
| `models/iv/__init__.py` |       25 | 0 | 0 |

### Existing patterns to follow
- Artifact directories under `trained_models/` are runtime truth.
- Predictor uses feature-type contract from `features/pipeline.py`.
- Governance/versioning are persistence contracts consumed by API and autopilot.

### Configuration
- High-risk boundaries: `models_to_features_6`, `models_to_config_17`, `models_to_validation_18`, `api_to_models_40`, `autopilot_to_models_28`.
- Contract invariants: [SYSTEM_CONTRACTS_AND_INVARIANTS.md](/Users/justinblaise/Documents/quant_engine/docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md).

## Tasks

### T1: Ledger and artifact contract baseline

**What:** Build full line ledger and capture artifact/registry schemas.

**Files:**
- `models/trainer.py`
- `models/predictor.py`
- `models/versioning.py`

**Implementation notes:**
- Snapshot model metadata JSON fields and version lookup behavior.
- Record current assumptions used by API and autopilot consumers.

**Verify:**
```bash
jq -r '.subsystems.model_training_prediction.files[]' docs/architecture/SUBSYSTEM_MAP.json | xargs wc -l
```

---

### T2: Training pipeline correctness pass

**What:** Audit feature matrix assembly, label generation, CV/walk-forward setup, and calibration integration.

**Files:**
- `models/trainer.py`
- `models/walk_forward.py`
- `models/calibration.py`

**Implementation notes:**
- Confirm strict precondition handling and horizon alignment.
- Validate no hidden leakage through fold construction or target prep.

**Verify:**
```bash
rg -n "precondition|walk_forward|fold|horizon|calibration|train_ensemble" models/trainer.py models/walk_forward.py models/calibration.py
```

---

### T3: Prediction and feature-contract pass

**What:** Audit live prediction pathways, feature ordering, and causality filtering.

**Files:**
- `models/predictor.py`
- `features/pipeline.py` (read-only contract check)
- `models/conformal.py`

**Implementation notes:**
- Validate every predictor-required feature maps to metadata/type policy.
- Confirm explicit handling for missing features/artifacts.

**Verify:**
```bash
rg -n "get_feature_type|FEATURE_METADATA|predict|conformal|interval" models/predictor.py features/pipeline.py models/conformal.py
```

---

### T4: Versioning/governance/retrain integrity pass

**What:** Audit registry consistency, champion logic, drift detection, and retrain triggers.

**Files:**
- `models/versioning.py`
- `models/governance.py`
- `models/retrain_trigger.py`

**Implementation notes:**
- Confirm explicit-version requests cannot silently downgrade/fallback.
- Validate registry conflict handling and max-version retention policy.

**Verify:**
```bash
rg -n "version|registry|champion|fallback|trigger|drift|psi|cusum" models/versioning.py models/governance.py models/retrain_trigger.py
```

---

### T5: Specialized model surface pass

**What:** Audit IV models and optional neural/online components for interface stability.

**Files:**
- `models/iv/models.py`
- `models/neural_net.py`
- `models/online_learning.py`

**Implementation notes:**
- Validate method signatures and persistence assumptions expected by consumers.
- Confirm numerical guardrails around IV/arbitrage constraints.

**Verify:**
```bash
rg -n "class |def |arbitrage|surface|save|load|state" models/iv/models.py models/neural_net.py models/online_learning.py
```

---

### T6: Boundary validation and findings closure

**What:** Validate all cross-subsystem contracts and publish severity-ranked findings.

**Files:**
- All subsystem files.

**Implementation notes:**
- Include explicit pass/fail verdict for each high-risk boundary.

**Verify:**
```bash
jq -r '.edges[] | select(.source_module=="models" or .target_module=="models") | [.source_file,.target_file,.import_type] | @tsv' docs/audit/DEPENDENCY_EDGES.json | head -n 120
```

## Validation

### Acceptance criteria
1. 100% of `6,153` lines reviewed.
2. Trainer/predictor feature contracts are fully validated.
3. Artifact versioning/governance integrity is documented and deterministic.
4. All high-risk boundaries touching models are dispositioned.

### Verification steps
```bash
jq -r '.subsystems.model_training_prediction.files[]' docs/architecture/SUBSYSTEM_MAP.json
jq -r '.edges[] | select(.source_module=="models" or .target_module=="models") | .import_type' docs/audit/DEPENDENCY_EDGES.json | sort | uniq -c
```

### Rollback plan
- Revert this spec file if model subsystem boundaries are updated.

---

## Notes

This subsystem sits on the critical contract chain between feature generation and execution outcomes, so contract drift must be treated as high severity.
