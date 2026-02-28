# Feature Spec: Subsystem Audit Spec — Data Ingestion & Quality

> **Status:** Draft
> **Author:** Codex
> **Date:** 2026-02-27
> **Estimated effort:** 12-16 hours across 6 tasks

---

## Why

Data ingestion is an upstream correctness gate for features, models, backtests, autopilot, and API workflows. Any leakage, survivorship drift, or cache corruption here contaminates the entire pipeline.

## What

Define a full-line audit spec for Subsystem `data_ingestion_quality` that validates loader/cache/provenance behavior, data-quality enforcement, and cross-subsystem contracts.

## Constraints

### Must-haves
- Review every line in all subsystem files (`19` files, `9,044` lines).
- Validate identity, as-of, survivorship, and cache trust invariants.
- Confirm lazy-provider fallback behavior is explicit and observable.
- Confirm quality checks never silently downgrade hard failures.

### Must-nots
- No production code modification during this audit.
- No assumption of provider availability (WRDS/Alpaca/Alpha Vantage/Kalshi).
- No unresolved leakage/survivorship `P0` findings.

### Out of scope
- Model training logic.
- API response envelope or frontend rendering concerns.

## Current State

Subsystem metadata is defined in [SUBSYSTEM_MAP.json](/Users/justinblaise/Documents/quant_engine/docs/audit/data/SUBSYSTEM_MAP.json) and [SUBSYSTEM_AUDIT_MANIFEST.md](/Users/justinblaise/Documents/quant_engine/docs/audit/data/SUBSYSTEM_AUDIT_MANIFEST.md).

### Key files
| File | Lines | Def/Class Count | Branch Points |
|---|---:|---:|---:|
| `data/wrds_provider.py` |     1620 | 34 | 148 |
| `data/intraday_quality.py` |     1110 | 21 | 70 |
| `data/survivorship.py` |      945 | 24 | 54 |
| `data/alternative.py` |      902 | 23 | 64 |
| `data/loader.py` |      849 | 16 | 130 |
| `data/local_cache.py` |      841 | 24 | 152 |
| `data/cross_source_validator.py` |      791 | 13 | 73 |
| `data/providers/alpha_vantage_provider.py` |      414 | 6 | 30 |
| `data/quality.py` |      385 | 6 | 31 |
| `data/providers/alpaca_provider.py` |      360 | 7 | 21 |
| `data/feature_store.py` |      341 | 10 | 32 |
| `validation/leakage_detection.py` |      193 | 5 | 7 |
| `validation/data_integrity.py` |      114 | 4 | 9 |
| `validation/feature_redundancy.py` |      114 | 4 | 3 |
| `data/provider_registry.py` |       53 | 2 | 7 |
| `data/__init__.py` |       35 | 0 | 0 |
| `data/providers/__init__.py` |       19 | 0 | 0 |
| `validation/__init__.py` |       13 | 0 | 0 |
| `data/provider_base.py` |       13 | 1 | 0 |

### Existing patterns to follow
- Loader hierarchy uses trusted cache then provider fallback with metadata sidecars.
- Quality gates are configurable and may be lazy/conditional.
- Survivorship-safe loading is a hard correctness path, not optional.

### Configuration
- Data quality and cache contracts: [SYSTEM_CONTRACTS_AND_INVARIANTS.md](/Users/justinblaise/Documents/quant_engine/docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md).
- Boundary records: `data_to_validation_10`, `data_to_config_12`, `features_to_data_14`, `data_to_kalshi_32`, `api_to_data_37` in [INTERFACE_CONTRACTS.yaml](/Users/justinblaise/Documents/quant_engine/docs/audit/data/INTERFACE_CONTRACTS.yaml).

## Tasks

### T1: Scope freeze and ledger creation

**What:** Generate file/line ledger and mark risk tiers from hotspot analysis.

**Files:**
- `data/loader.py`
- `data/local_cache.py`
- `data/provider_registry.py`

**Implementation notes:**
- Prioritize hotspot files (`loader`, `local_cache`, `wrds_provider`, `survivorship`).
- Track each 200-line chunk with reviewer status.

**Verify:**
```bash
jq -r '.subsystems.data_ingestion_quality.files[]' docs/audit/data/SUBSYSTEM_MAP.json | xargs wc -l
```

---

### T2: Identity/time/leakage correctness pass

**What:** Audit PERMNO-first identity, as-of logic, and leakage controls.

**Files:**
- `data/loader.py`
- `data/survivorship.py`
- `validation/leakage_detection.py`

**Implementation notes:**
- Verify no forward-looking joins.
- Validate point-in-time universe reconstruction and delisting handling.

**Verify:**
```bash
rg -n "permno|asof|lookahead|leak|survivorship|delist" data/loader.py data/survivorship.py validation/leakage_detection.py
```

---

### T3: Cache/provenance integrity pass

**What:** Audit cache trust hierarchy, atomic writes, and metadata fidelity.

**Files:**
- `data/local_cache.py`
- `data/feature_store.py`
- `data/intraday_quality.py`

**Implementation notes:**
- Confirm stale-data detection and source provenance are explicit.
- Validate quarantine semantics for low-quality intraday data.

**Verify:**
```bash
rg -n "atomic|metadata|stale|trust|quarantine|sidecar" data/local_cache.py data/feature_store.py data/intraday_quality.py
```

---

### T4: Provider fallback and failure semantics pass

**What:** Validate provider registry behavior when optional providers fail.

**Files:**
- `data/provider_registry.py`
- `data/providers/alpaca_provider.py`
- `data/providers/alpha_vantage_provider.py`

**Implementation notes:**
- Confirm fallback behavior is deterministic and surfaced.
- Validate `data/provider_registry.py` lazy Kalshi import path.

**Verify:**
```bash
rg -n "try:|except|get_provider|register_provider|KalshiProvider" data/provider_registry.py data/providers/alpaca_provider.py data/providers/alpha_vantage_provider.py
```

---

### T5: Contract boundary pass

**What:** Validate contracts with features/autopilot/api/validation/kalshi consumers.

**Files:**
- `data/quality.py`
- `validation/data_integrity.py`
- `data/wrds_provider.py`

**Implementation notes:**
- Ensure report schemas and return types consumed downstream are stable.
- Trace each exported loader/quality function to consumer assumptions.

**Verify:**
```bash
jq -r '.edges[] | select(.source_module=="data" or .target_module=="data") | [.source_file,.target_file,.import_type] | @tsv' docs/audit/data/DEPENDENCY_EDGES.json | head -n 80
```

---

### T6: Findings synthesis and risk disposition

**What:** Publish complete defects list and remediation priorities.

**Files:**
- All subsystem files.

**Implementation notes:**
- Findings must include: invariant violated, proof lines, downstream impact, test gap.

**Verify:**
```bash
# Manual gate: 9044/9044 lines reviewed; all high-risk boundaries dispositioned
```

## Validation

### Acceptance criteria
1. 100% of `9,044` lines are reviewed and logged.
2. Survivorship/leakage invariants are explicitly checked and passed or flagged.
3. Provider fallback paths are deterministic and observable.
4. All high-risk data boundaries have pass/fail decisions with evidence.

### Verification steps
```bash
jq -r '.subsystems.data_ingestion_quality.files[]' docs/audit/data/SUBSYSTEM_MAP.json
jq -r '.edges[] | select(.target_module=="data" or .source_module=="data") | .import_type' docs/audit/data/DEPENDENCY_EDGES.json | sort | uniq -c
```

### Rollback plan
- Documentation-only rollback: revert this spec file if scope assumptions change.

---

## Notes

Primary invariants for this subsystem: PERMNO-first identity, as-of joins, provenance/cache trust, and explicit data-quality gating.
