# Feature Spec: Subsystem Audit Spec — Feature Engineering

> **Status:** Draft
> **Author:** Codex
> **Date:** 2026-02-27
> **Estimated effort:** 16-24 hours across 6 tasks

---

## Why

Feature Engineering is the highest-scoring module hotspot and directly controls model inputs. Any feature name/order/type drift can silently break causality guarantees and invalidate trained artifacts.

## What

Define an exhaustive audit spec for Subsystem `feature_engineering` with complete line coverage, feature-contract validation, and strict downstream compatibility checks.

## Constraints

### Must-haves
- Review every line across all `17` files (`8,559` lines).
- Validate `FEATURE_METADATA` and `get_feature_type` completeness against emitted features.
- Validate lazy fallback behavior for data/regime/config imports.
- Confirm indicator computations are numerically stable and deterministic.

### Must-nots
- No changes to feature definitions during audit.
- No acceptance of undocumented feature additions/removals.
- No unresolved causality-related `P0`/`P1` findings.

### Out of scope
- Model training policy changes.
- Backtest execution/cost model tuning.

## Current State

Subsystem metadata and dependencies are in [SUBSYSTEM_MAP.json](/Users/justinblaise/Documents/quant_engine/docs/architecture/SUBSYSTEM_MAP.json) and [SUBSYSTEM_AUDIT_MANIFEST.md](/Users/justinblaise/Documents/quant_engine/docs/architecture/SUBSYSTEM_AUDIT_MANIFEST.md).

### Key files
| File | Lines | Def/Class Count | Branch Points |
|---|---:|---:|---:|
| `indicators/indicators.py` |     2904 | 389 | 80 |
| `features/pipeline.py` |     1541 | 17 | 169 |
| `features/research_factors.py` |      985 | 24 | 66 |
| `indicators/eigenvalue.py` |      399 | 13 | 38 |
| `indicators/spectral.py` |      328 | 10 | 16 |
| `features/lob_features.py` |      311 | 6 | 30 |
| `indicators/ssa.py` |      321 | 8 | 12 |
| `indicators/ot_divergence.py` |      262 | 6 | 9 |
| `features/macro.py` |      244 | 5 | 19 |
| `features/intraday.py` |      243 | 7 | 25 |
| `indicators/tail_risk.py` |      240 | 6 | 8 |
| `features/version.py` |      168 | 7 | 12 |
| `features/wave_flow.py` |      144 | 3 | 8 |
| `features/options_factors.py` |      134 | 3 | 20 |
| `features/harx_spillovers.py` |      242 | 3 | 18 |
| `features/__init__.py` |       34 | 0 | 0 |
| `indicators/__init__.py` |       14 | 0 | 0 |

### Existing patterns to follow
- `features/pipeline.py` is an orchestration hub with many lazy imports.
- `indicators/*` are computation providers used almost exclusively by the feature pipeline.
- Feature type metadata is a hard contract for prediction causality filtering.

### Configuration
- Boundary contracts: `features_to_indicators_8`, `models_to_features_6`, `features_to_data_14`, `features_to_regime_15`, `features_to_config_13` in [INTERFACE_CONTRACTS.yaml](/Users/justinblaise/Documents/quant_engine/docs/audit/INTERFACE_CONTRACTS.yaml).
- Hotspots: [HOTSPOT_LIST.md](/Users/justinblaise/Documents/quant_engine/docs/audit/HOTSPOT_LIST.md).

## Tasks

### T1: Ledger and feature-surface baseline

**What:** Capture full file/line ledger plus emitted feature inventory.

**Files:**
- `features/pipeline.py`
- `features/research_factors.py`
- `indicators/indicators.py`

**Implementation notes:**
- Derive canonical feature list and compare with metadata/type registry.
- Mark hotspots for first-pass deep review.

**Verify:**
```bash
jq -r '.subsystems.feature_engineering.files[]' docs/architecture/SUBSYSTEM_MAP.json | xargs wc -l
```

---

### T2: Feature-contract and causality pass

**What:** Audit `FEATURE_METADATA`, `get_feature_type`, and feature emission pathways.

**Files:**
- `features/pipeline.py`
- `features/version.py`
- `features/options_factors.py`

**Implementation notes:**
- Ensure all emitted features have metadata and valid type classification.
- Validate defaults do not silently classify unknown fields as safe when they are not.

**Verify:**
```bash
rg -n "FEATURE_METADATA|get_feature_type|RESEARCH_ONLY|CAUSAL|END_OF_DAY" features/pipeline.py features/version.py
```

---

### T3: Indicator numerical integrity pass

**What:** Audit indicator implementations for stability and edge-case handling.

**Files:**
- `indicators/indicators.py`
- `indicators/spectral.py`
- `indicators/eigenvalue.py`

**Implementation notes:**
- Check division-by-zero guards, NaN handling, and warmup-period semantics.
- Validate output naming consistency consumed by pipeline.

**Verify:**
```bash
rg -n "nan|inf|clip|epsilon|window|min_periods|fillna" indicators/indicators.py indicators/spectral.py indicators/eigenvalue.py
```

---

### T4: Data/regime lazy import correctness pass

**What:** Validate graceful degradation and explicit observability for lazy imports.

**Files:**
- `features/pipeline.py`
- `features/intraday.py`
- `features/macro.py`

**Implementation notes:**
- Trace each lazy import path to fallback behavior.
- Ensure failure paths do not silently produce misleading feature outputs.

**Verify:**
```bash
rg -n "try:|except|lazy|wrds|load_ohlcv|CorrelationRegimeDetector" features/pipeline.py features/intraday.py features/macro.py
```

---

### T5: Boundary and consumer compatibility pass

**What:** Validate compatibility with models/autopilot/kalshi consumers.

**Files:**
- `features/pipeline.py`
- `features/options_factors.py`
- `features/research_factors.py`

**Implementation notes:**
- Confirm contracts used by `models/predictor.py`, `autopilot/engine.py`, and `kalshi/options.py` remain stable.
- Validate no undocumented column/field schema drift.

**Verify:**
```bash
jq -r '.edges[] | select(.target_module=="features" or .source_module=="features") | [.source_file,.target_file,.symbols_imported] | @tsv' docs/audit/DEPENDENCY_EDGES.json | head -n 100
```

---

### T6: Findings synthesis and severity gating

**What:** Complete review closure with defect matrix and remediation order.

**Files:**
- All subsystem files.

**Implementation notes:**
- Every finding must include the exact feature/indicator contract impact.

**Verify:**
```bash
# Manual gate: 8559/8559 lines reviewed; all causality boundaries dispositioned
```

## Validation

### Acceptance criteria
1. 100% of lines in all 17 files are reviewed.
2. Every emitted feature has explicit metadata/type mapping.
3. Indicator outputs and names are stable for downstream consumers.
4. High-risk boundaries (`models_to_features_6`, `features_to_indicators_8`) are explicitly validated.

### Verification steps
```bash
jq -r '.subsystems.feature_engineering.files[]' docs/architecture/SUBSYSTEM_MAP.json
jq -r '.edges[] | select(.source_module=="features" or .target_module=="features" or .target_module=="indicators") | .import_type' docs/audit/DEPENDENCY_EDGES.json | sort | uniq -c
```

### Rollback plan
- Revert this spec file if subsystem boundaries or audit order are revised.

---

## Notes

This subsystem has one of the largest correctness surfaces in the system due to feature count and hidden coupling via lazy imports.
