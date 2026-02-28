# Feature Spec: Subsystem Audit Spec — Entry Points & Scripts

> **Status:** Draft
> **Author:** Codex
> **Date:** 2026-02-28
> **Estimated effort:** 6-8 hours across 6 tasks

---

## Why

Entry Points & Scripts are the top-level orchestration layer that wires all core subsystems together. While they have 0 fan-in (nothing imports from them, so changes cannot cascade), they are the sole writers of critical shared artifacts like `results/backtest_*d_summary.json` and the only place where module constructor signatures, argument types, and pipeline sequencing are exercised end-to-end. A wiring defect here silently corrupts downstream consumers (API, evaluation) without triggering import-time errors.

## What

Produce a complete audit spec for Subsystem `entry_points_scripts` covering all 17 files (8,883 lines), with particular focus on shared artifact schema correctness, module constructor wiring, CLI argument handling, and reproducibility manifest generation.

## Constraints

### Must-haves
- Audit every line in all 17 subsystem files (no sampling).
- Verify module constructor wiring — parameters passed to core module APIs must match expected signatures from prior subsystem audits (Subsystems 1–10).
- Verify shared artifact output schemas — `results/backtest_*d_summary.json` must match the 17-field schema (including `regime_breakdown`) expected by `api/services/backtest_service.py`, `api/services/results_service.py`, and `evaluation/engine.py`.
- Verify CLI argument defaults are sensible and consistent with `config.py` constants.
- Verify reproducibility manifest generation captures all relevant runtime parameters.
- Distinguish production entry points (9 files) from tooling scripts (3 audit scripts) from data utility scripts (5 files).
- Record every finding with severity (`P0`–`P3`), evidence, and impacted consumers.

### Must-nots
- No production code changes during the audit.
- No silent changes to shared artifact schemas without explicit impact analysis.
- No closing the audit with unresolved `P0`/`P1` findings.

### Out of scope
- Deep audit of imported core modules (already covered in Subsystems 1–10).
- Refactoring entry points for code deduplication (document as recommendation only).
- Frontend or API endpoint testing (covered in Subsystem 10).

## Current State

Entry Points & Scripts is Subsystem `entry_points_scripts` in [SUBSYSTEM_MAP.json](/Users/justinblaise/Documents/quant_engine/docs/audit/data/SUBSYSTEM_MAP.json), audit order `11`, with `17` files and `8,883` lines.

**Subsystem characteristics:**
- **Audit priority:** LOW — leaf consumers with 0 fan-in. Changes here cannot cascade to other subsystems.
- **Depends on:** ALL other subsystems (shared_infrastructure, data_ingestion_quality, feature_engineering, regime_detection, backtesting_risk, model_training_prediction, autopilot, kalshi, api_frontend).
- **Depended on by:** Nothing (0 fan-in).
- **3 audit scripts are tooling, not production:** `scripts/extract_dependencies.py`, `scripts/generate_interface_contracts.py`, `scripts/hotspot_scoring.py` — created for this LLM audit workflow.

### Key files

| File | Lines | Def/Class Count | Branch Points |
|---|---:|---:|---:|
| `scripts/generate_interface_contracts.py` | 2570 | 2 | 2 |
| `scripts/alpaca_intraday_download.py` | 1202 | 12 | 255 |
| `run_wrds_daily_refresh.py` | 915 | 13 | 148 |
| `scripts/extract_dependencies.py` | 600 | 9 | 101 |
| `scripts/ibkr_intraday_download.py` | 508 | 12 | 119 |
| `run_backtest.py` | 464 | 1 | 56 |
| `scripts/ibkr_daily_gapfill.py` | 417 | 4 | 53 |
| `scripts/hotspot_scoring.py` | 401 | 8 | 60 |
| `scripts/compare_regime_models.py` | 323 | 3 | 29 |
| `run_retrain.py` | 322 | 2 | 47 |
| `run_predict.py` | 237 | 1 | 35 |
| `run_train.py` | 227 | 1 | 30 |
| `run_kalshi_event_pipeline.py` | 227 | 2 | 25 |
| `scripts/generate_types.py` | 146 | 3 | 26 |
| `run_autopilot.py` | 129 | 1 | 9 |
| `run_rehydrate_cache_metadata.py` | 101 | 2 | 5 |
| `run_server.py` | 94 | 1 | 5 |

**Note:** Line counts for the 14 original files are from MODULE_INVENTORY.yaml (Job 1). The 3 audit scripts (`extract_dependencies.py`, `generate_interface_contracts.py`, `hotspot_scoring.py`) were added after Job 1 and their counts are from the time of Job 5 subsystem clustering. The auditor should verify actual line counts at audit time, as `scripts/alpaca_intraday_download.py` and `scripts/ibkr_intraday_download.py` may have grown since the inventory snapshot.

### File categories

**Production entry points (9 files, ~3,315 lines):**
- `run_backtest.py` — Writes `results/backtest_*d_summary.json` (shared artifact). Most critical entry point.
- `run_train.py` — Wires data loading, feature computation, regime detection, and model training.
- `run_predict.py` — Wires data→features→regime→prediction pipeline.
- `run_retrain.py` — Checks retrain triggers, runs retraining with governance and versioning.
- `run_autopilot.py` — Wires AutopilotEngine for automated lifecycle.
- `run_wrds_daily_refresh.py` — Most complex entry point (915 lines). Daily data refresh pipeline.
- `run_kalshi_event_pipeline.py` — Kalshi event-time pipeline (distribution, events, walkforward, promotion).
- `run_server.py` — API + frontend static serving entry point.
- `run_rehydrate_cache_metadata.py` — Cache metadata backfill utility.

**Data utility scripts (5 files, ~3,650 lines):**
- `scripts/alpaca_intraday_download.py` — Hybrid Alpaca/IBKR intraday data downloader.
- `scripts/ibkr_intraday_download.py` — IBKR intraday data downloader.
- `scripts/ibkr_daily_gapfill.py` — IBKR daily gap-fill downloader.
- `scripts/compare_regime_models.py` — Regime model A/B comparison tool.
- `scripts/generate_types.py` — TypeScript interface generator from Pydantic schemas.

**Audit tooling scripts (3 files, ~3,571 lines):**
- `scripts/extract_dependencies.py` — Audit workflow tooling (not production).
- `scripts/generate_interface_contracts.py` — Audit workflow tooling (not production).
- `scripts/hotspot_scoring.py` — Audit workflow tooling (not production).

### Dependency edges (from DEPENDENCY_EDGES.json)

Key import relationships for production entry points:

| Entry Point | Imports From | Import Type |
|---|---|---|
| `run_backtest.py:29` | `config.py` | top_level |
| `run_backtest.py:35–41` | `data/loader.py`, `data/survivorship.py`, `features/pipeline.py`, `regime/detector.py`, `models/predictor.py`, `backtest/engine.py`, `backtest/validation.py` | top_level |
| `run_backtest.py:48` | `reproducibility.py` | top_level |
| `run_backtest.py:382` | `backtest/advanced_validation.py` | conditional |
| `run_train.py:23–33` | `config.py`, `data/loader.py`, `features/pipeline.py`, `regime/detector.py`, `models/governance.py`, `models/trainer.py`, `models/versioning.py`, `reproducibility.py` | top_level |
| `run_train.py:164` | `config.py` | conditional |
| `run_predict.py:23–31` | `config.py`, `data/loader.py`, `features/pipeline.py`, `regime/detector.py`, `models/predictor.py`, `reproducibility.py` | top_level |
| `run_retrain.py:23–34` | `config.py`, `data/loader.py`, `features/pipeline.py`, `regime/detector.py`, `models/governance.py`, `models/trainer.py`, `models/retrain_trigger.py`, `models/versioning.py`, `reproducibility.py` | top_level |
| `run_autopilot.py:19–23` | `autopilot/engine.py`, `config.py`, `reproducibility.py` | top_level |
| `run_wrds_daily_refresh.py:27–29` | `config.py`, `data/local_cache.py`, `data/survivorship.py` | top_level |
| `run_wrds_daily_refresh.py:406,676,702,746` | `data/wrds_provider.py`, `data/local_cache.py` | conditional |
| `run_server.py:41–42` | `api/config.py`, `api/main.py` | lazy |
| `run_kalshi_event_pipeline.py:11,30–34` | `config.py`, `kalshi/distribution.py`, `kalshi/events.py`, `kalshi/pipeline.py`, `kalshi/promotion.py`, `kalshi/walkforward.py` | top_level |
| `run_rehydrate_cache_metadata.py:17–18` | `config.py`, `data/local_cache.py` | top_level |

### Existing patterns to follow
- All production `run_*.py` files follow a common pattern: imports → CLI argument parsing → main() function → module construction → pipeline execution → reproducibility manifest.
- Lazy imports used only in `run_server.py` (lazy API import for server startup).
- Conditional imports in `run_backtest.py` (advanced validation) and `run_wrds_daily_refresh.py` (WRDS provider).

### Configuration
- Shared artifact contracts: [INTERFACE_CONTRACTS.yaml](/Users/justinblaise/Documents/quant_engine/docs/audit/data/INTERFACE_CONTRACTS.yaml).
- Subsystem dependency maps: [DEPENDENCY_MATRIX.md](/Users/justinblaise/Documents/quant_engine/docs/audit/data/DEPENDENCY_MATRIX.md), [DEPENDENCY_EDGES.json](/Users/justinblaise/Documents/quant_engine/docs/audit/data/DEPENDENCY_EDGES.json).
- Upstream module signatures: verified in Subsystem audits 1–10.

### Carry-forward context from Subsystem 10 audit (transition guide)

From the Subsystem 10→11 transition:
- **API orchestrator wiring** — entry points must pass the same parameters as the API. Cross-reference `api/orchestrator.py` to verify parameter consistency.
- **All upstream module APIs** — entry points wire modules together directly. Key files to cross-reference: `config.py`, `data/loader.py`, `features/pipeline.py`, `regime/detector.py`, `backtest/engine.py`, `models/trainer.py`, `models/predictor.py`, `autopilot/engine.py`.
- **Common failure modes:**
  - Entry point passes wrong parameters to module constructors → TypeError or incorrect behavior at runtime.
  - Entry point writes shared artifacts with incorrect schema → downstream readers (API, evaluation) fail to parse output files.

## Tasks

### T1: Build immutable audit baseline

**What:** Freeze scope and derive exact line totals before review starts.

**Files:**
- All 17 subsystem files.

**Implementation notes:**
- Generate a line ledger (`file`, `start_line`, `end_line`, `reviewer`, `status`) covering every line.
- Verify actual line counts against SUBSYSTEM_MAP.json total of 8,883 lines. Flag any files that have grown since the MODULE_INVENTORY snapshot.
- Classify each file as production (9), data utility (5), or audit tooling (3).

**Verify:**
```bash
jq -r '.subsystems.entry_points_scripts.files[]' docs/audit/data/SUBSYSTEM_MAP.json | xargs wc -l
```

---

### T2: Shared artifact schema verification

**What:** Audit `run_backtest.py` output schema for correctness against all downstream consumers.

**Files:**
- `run_backtest.py` — writes `results/backtest_*d_summary.json`.

**Implementation notes:**
- Trace the exact JSON structure written by `run_backtest.py` and confirm it contains all 17 required fields including `regime_breakdown`.
- Cross-reference against consumer expectations:
  - `api/services/backtest_service.py` (Subsystem 10)
  - `api/services/results_service.py` (Subsystem 10)
  - `evaluation/engine.py` (Subsystem 7)
- Verify no fields are missing, mistyped, or conditionally omitted.
- Check that `run_backtest.py:382` conditional import of `backtest/advanced_validation.py` does not alter the output schema.

**Verify:**
```bash
rg -n "json.dump|json_output|summary_dict|regime_breakdown" run_backtest.py
rg -n "backtest.*summary|json.load|regime_breakdown" api/services/backtest_service.py api/services/results_service.py evaluation/engine.py
```

---

### T3: Module constructor wiring pass

**What:** Verify that every entry point passes correct parameters to core module constructors and functions.

**Files:**
- `run_train.py`
- `run_backtest.py`
- `run_predict.py`
- `run_retrain.py`
- `run_autopilot.py`

**Implementation notes:**
- For each entry point, trace every module constructor call and verify parameter names, types, and ordering match the signatures documented in Subsystem audits 1–10.
- Cross-reference `api/orchestrator.py` (Subsystem 10) to verify entry points pass the same parameters as the API for equivalent operations.
- Pay particular attention to:
  - `PositionSizer.size_position()` — 21 parameters, evolving interface (Subsystem 5).
  - `UncertaintyGate` threshold defaults from config (Subsystem 4).
  - `RegimeDetector` constructor parameters (Subsystem 4).
  - `EnsembleTrainer` and `EnsemblePredictor` construction (Subsystem 6).
- Flag any parameter mismatches as `P1` findings.

**Verify:**
```bash
rg -n "RegimeDetector|EnsembleTrainer|EnsemblePredictor|PositionSizer|AutopilotEngine|BacktestEngine|FeaturePipeline" run_train.py run_backtest.py run_predict.py run_retrain.py run_autopilot.py
```

---

### T4: CLI argument handling and reproducibility pass

**What:** Audit CLI argument parsing, default values, and reproducibility manifest generation.

**Files:**
- `run_train.py`
- `run_backtest.py`
- `run_predict.py`
- `run_retrain.py`
- `run_autopilot.py`
- `run_wrds_daily_refresh.py`
- `run_kalshi_event_pipeline.py`

**Implementation notes:**
- Verify all `argparse` defaults are consistent with `config.py` constants.
- Confirm `reproducibility.py` manifest captures all CLI arguments and runtime state.
- Check `run_wrds_daily_refresh.py` (915 lines, most complex) for correct error handling in the daily data refresh pipeline. This is the entry point most likely to silently corrupt cached data used by features, training, and backtesting.
- Verify `run_kalshi_event_pipeline.py` correctly wires all 5 Kalshi submodules (distribution, events, pipeline, promotion, walkforward).

**Verify:**
```bash
rg -n "argparse|add_argument|parse_args|default=" run_train.py run_backtest.py run_predict.py run_retrain.py run_autopilot.py run_wrds_daily_refresh.py run_kalshi_event_pipeline.py
rg -n "manifest|reproducibility" run_train.py run_backtest.py run_predict.py run_retrain.py run_autopilot.py
```

---

### T5: Data utility and audit tooling review

**What:** Review data utility scripts and audit tooling scripts for correctness and safety.

**Files:**
- `scripts/alpaca_intraday_download.py`
- `scripts/ibkr_intraday_download.py`
- `scripts/ibkr_daily_gapfill.py`
- `scripts/compare_regime_models.py`
- `scripts/generate_types.py`
- `scripts/extract_dependencies.py`
- `scripts/generate_interface_contracts.py`
- `scripts/hotspot_scoring.py`

**Implementation notes:**
- **Data utility scripts**: Verify data download scripts write to expected cache locations (`data/cache/*.parquet`) with correct schema (Open, High, Low, Close, Volume, date). Cross-reference against cache schema documented in INTERFACE_CONTRACTS.yaml.
- **Audit tooling scripts**: Confirm these 3 files are pure tooling with no production side effects. They should not import from or modify production modules in ways that could affect runtime behavior. Lower severity threshold — `P2`/`P3` findings acceptable here.
- **`scripts/generate_types.py`**: Verify TypeScript interface generation correctly introspects Pydantic schemas from `api/schemas/`.
- **`scripts/compare_regime_models.py`**: Verify regime model comparison does not modify trained model artifacts.

**Verify:**
```bash
rg -n "parquet|cache|Open|High|Low|Close|Volume" scripts/alpaca_intraday_download.py scripts/ibkr_intraday_download.py scripts/ibkr_daily_gapfill.py
rg -n "import.*from|from.*import" scripts/extract_dependencies.py scripts/generate_interface_contracts.py scripts/hotspot_scoring.py | head -20
```

---

### T6: Final subsystem audit report

**What:** Publish final findings with remediation priorities and risk acceptance decisions.

**Files:**
- All 17 subsystem files.

**Implementation notes:**
- Include unresolved risks, required follow-up tests, and migration guidance for any breaking change.
- Summarize all wiring mismatches found in T3.
- Summarize all shared artifact schema issues found in T2.
- Note any recommended refactoring (e.g., common CLI pattern extraction, shared argument validation).
- Confirm total line coverage: all 8,883 lines explicitly marked reviewed.

**Verify:**
```bash
# Manual gate: 8883/8883 lines reviewed; all shared artifact schemas verified; all module wiring checked
```

## Validation

### Acceptance criteria
1. 100% of lines in all 17 files are explicitly marked reviewed.
2. `results/backtest_*d_summary.json` schema is verified against all 3 downstream consumers.
3. All module constructor calls in entry points match expected signatures from Subsystem audits 1–10.
4. CLI argument defaults are consistent with `config.py` constants.
5. Reproducibility manifests capture all relevant runtime parameters.
6. No unresolved `P0`/`P1` findings remain.
7. Audit tooling scripts confirmed as non-production with no side effects.

### Verification steps
```bash
jq -r '.subsystems.entry_points_scripts.files[]' docs/audit/data/SUBSYSTEM_MAP.json
jq -r '.edges[] | select(.source_module=="entry_points" or .source_module=="scripts") | [.source_file,.target_file,.import_type] | @tsv' docs/audit/data/DEPENDENCY_EDGES.json | head -n 60
```

### Rollback plan
- This spec introduces documentation only. If needed, revert the spec file and regenerate without changing runtime code.

---

## Notes

- This is the final subsystem in the audit order (Subsystem 11 of 11). All upstream subsystem audits (1–10) should be complete before this audit begins.
- Entry points are leaf consumers with 0 fan-in, so defects found here have limited blast radius (they cannot cascade to other subsystems). However, shared artifact schema defects in `run_backtest.py` can break downstream API and evaluation consumers.
- The 3 audit scripts (`extract_dependencies.py`, `generate_interface_contracts.py`, `hotspot_scoring.py`) were added after Job 1 (Module Inventory) and are tooling for this LLM audit workflow. They are not production code.
- `run_wrds_daily_refresh.py` (915 lines) is the most complex entry point and the most likely source of data corruption. It should receive extra scrutiny during the line-by-line review.

This spec is grounded in: [SUBSYSTEM_AUDIT_MANIFEST.md](/Users/justinblaise/Documents/quant_engine/docs/audit/data/SUBSYSTEM_AUDIT_MANIFEST.md), [HOTSPOT_LIST.md](/Users/justinblaise/Documents/quant_engine/docs/audit/data/HOTSPOT_LIST.md), [INTERFACE_CONTRACTS.yaml](/Users/justinblaise/Documents/quant_engine/docs/audit/data/INTERFACE_CONTRACTS.yaml), and [INPUT_CONTEXT.md](/Users/justinblaise/Documents/quant_engine/docs/audit/INPUT_CONTEXT.md).
