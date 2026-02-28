# Feature Spec: Subsystem Audit Spec — Shared Infrastructure

> **Status:** Draft
> **Author:** Codex
> **Date:** 2026-02-27
> **Estimated effort:** 8-12 hours across 6 tasks

---

## Why

Shared infrastructure is the highest blast-radius subsystem. `config.py` and `config_structured.py` define contracts that every core subsystem consumes, so line-level defects here propagate system-wide.

## What

Produce a complete line-by-line audit spec for Shared Infrastructure that guarantees 100% file coverage, constant-contract validation, and downstream impact checks before any future code change is accepted.

## Constraints

### Must-haves
- Audit every line in all subsystem files (no sampling).
- Verify constant/status integrity (`ACTIVE`/`PLACEHOLDER`/`DEPRECATED`) and typed config parity.
- Validate compatibility with all downstream consumers listed in subsystem dependency maps.
- Record every finding with severity (`P0`-`P3`), evidence, and impacted subsystems.

### Must-nots
- No production code changes during the audit.
- No silent constant renames, default changes, or datatype changes without explicit compatibility notes.
- No closing the audit with unresolved `P0`/`P1` findings.

### Out of scope
- Refactoring unrelated core modules.
- Entry-point/script wiring audits (handled separately from the 10 requested subsystem specs).

## Current State

Shared Infrastructure is Subsystem `shared_infrastructure` in [SUBSYSTEM_MAP.json](/Users/justinblaise/Documents/quant_engine/docs/architecture/SUBSYSTEM_MAP.json), audit order `1`, with `7` files and `2,151` lines.

### Key files
| File | Lines | Def/Class Count | Branch Points |
|---|---:|---:|---:|
| `config.py` |     1020 | 1 | 35 |
| `utils/logging.py` |      439 | 15 | 29 |
| `config_structured.py` |      347 | 20 | 6 |
| `reproducibility.py` |      333 | 6 | 31 |
| `__init__.py` |        6 | 0 | 0 |
| `utils/__init__.py` |        5 | 0 | 0 |
| `config_data/__init__.py` |        1 | 0 | 0 |

### Existing patterns to follow
- `config.py` is the runtime constant surface; `config_structured.py` is the typed source-of-truth.
- Lazy-import patterns are used in consumers to avoid circular imports; this increases runtime-only failure risk.
- Reproducibility manifest generation is an operational contract for `run_*.py` workflows.

### Configuration
- Canonical config invariants: [SYSTEM_CONTRACTS_AND_INVARIANTS.md](/Users/justinblaise/Documents/quant_engine/docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md).
- Contract surface and consumers: [INTERFACE_CONTRACTS.yaml](/Users/justinblaise/Documents/quant_engine/docs/audit/INTERFACE_CONTRACTS.yaml), [DEPENDENCY_MATRIX.md](/Users/justinblaise/Documents/quant_engine/docs/audit/DEPENDENCY_MATRIX.md).

## Tasks

### T1: Build immutable audit baseline

**What:** Freeze scope and derive exact line totals and symbol inventories before review starts.

**Files:**
- `config.py` — constants and helper functions.
- `config_structured.py` — dataclass schema surface.
- `reproducibility.py` — run manifest contract.

**Implementation notes:**
- Generate a line ledger (`file`, `start_line`, `end_line`, `reviewer`, `status`) covering every line.
- Capture exported symbols and compare against current consumers.

**Verify:**
```bash
jq -r '.subsystems.shared_infrastructure.files[]' docs/architecture/SUBSYSTEM_MAP.json | xargs wc -l
```

---

### T2: Constant and typed-config parity pass

**What:** Audit all config constants and their typed equivalents for semantic and type consistency.

**Files:**
- `config.py` — constant values/status tags.
- `config_structured.py` — dataclass defaults and loaders.

**Implementation notes:**
- For each public constant, confirm definition source and downstream usage.
- Flag any constant with ambiguous status or missing typed counterpart.

**Verify:**
```bash
rg -n "STATUS:" config.py
rg -n "class .*Config|def get_config" config_structured.py
```

---

### T3: Cross-subsystem dependency contract audit

**What:** Validate outbound contracts to all dependent subsystems.

**Files:**
- `config.py`
- `utils/logging.py`
- `reproducibility.py`

**Implementation notes:**
- Check boundaries `validation_to_config_7`, `data_to_config_12`, `regime_to_config_16`, `models_to_config_17`, `backtest_to_config_19`, `risk_to_config_21`, `autopilot_to_config_25`, `api_to_config_36`, and `utils_to_config_45`.
- Confirm no consumer assumes stale names or incompatible types.

**Verify:**
```bash
jq -r '.edges[] | select(.target_module=="config") | [.source_file,.source_line,.import_statement] | @tsv' docs/audit/DEPENDENCY_EDGES.json | head -n 40
```

---

### T4: Reproducibility and observability guarantees

**What:** Audit run-manifest and logging side effects for determinism and backward compatibility.

**Files:**
- `reproducibility.py`
- `utils/logging.py`

**Implementation notes:**
- Verify artifact schemas expected by operational workflows are stable.
- Ensure alert history and webhook behavior is explicit and failure-safe.

**Verify:**
```bash
rg -n "manifest|AlertHistory|MetricsEmitter|webhook" reproducibility.py utils/logging.py
```

---

### T5: Line-by-line defect triage

**What:** Complete full-line review and classify all findings.

**Files:**
- All subsystem files.

**Implementation notes:**
- Use chunk size <=200 lines per review unit.
- Require evidence for each finding: exact lines, invariant violated, impacted consumers.

**Verify:**
```bash
# Example completion check: reviewed lines must equal total lines (2151)
```

---

### T6: Final subsystem audit report

**What:** Publish final findings with remediation priorities and risk acceptance decisions.

**Files:**
- Output artifact only (no code edits).

**Implementation notes:**
- Include unresolved risks, required follow-up tests, and migration guidance for any breaking change.

**Verify:**
```bash
# Manual gate: report contains full line ledger + boundary checklist + severity matrix
```

## Validation

### Acceptance criteria
1. 100% of lines in all 7 files are explicitly marked reviewed.
2. Every exported config contract is traced to consumer usage.
3. All high-risk boundaries touching config are evaluated and dispositioned.
4. No unresolved `P0`/`P1` findings remain.

### Verification steps
```bash
jq -r '.subsystems.shared_infrastructure.files[]' docs/architecture/SUBSYSTEM_MAP.json
jq -r '.edges[] | select(.target_module=="config") | .source_module' docs/audit/DEPENDENCY_EDGES.json | sort | uniq -c
```

### Rollback plan
- This spec introduces documentation only. If needed, revert the spec file and regenerate without changing runtime code.

---

## Notes

This spec is grounded in: [SUBSYSTEM_AUDIT_MANIFEST.md](/Users/justinblaise/Documents/quant_engine/docs/architecture/SUBSYSTEM_AUDIT_MANIFEST.md), [HOTSPOT_LIST.md](/Users/justinblaise/Documents/quant_engine/docs/audit/HOTSPOT_LIST.md), and [INTERFACE_CONTRACTS.yaml](/Users/justinblaise/Documents/quant_engine/docs/audit/INTERFACE_CONTRACTS.yaml).
