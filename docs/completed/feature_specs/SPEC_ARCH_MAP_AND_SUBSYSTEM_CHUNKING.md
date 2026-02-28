# SPEC: Architecture Dependency Map + Hotspot Analysis & Subsystem Chunking for Audit

**Generated**: 2026-02-27
**Scope**: Two LLM workflow specs for (1) producing an architecture + dependency map with hotspot list, and (2) splitting the repo into audit-friendly subsystem chunks where all co-dependent files live together.
**Verification Method**: Every dependency, file path, and module boundary cited below was verified against actual source code via import tracing (both `from quant_engine.X` and `from ..X` relative import patterns).

---

# PART 1: ARCHITECTURE + DEPENDENCY MAP + HOTSPOT LIST

## Purpose

Produce a machine-readable and human-readable map of the entire quant_engine system that captures:
1. Every module, its responsibilities, and its public interface
2. Every cross-module dependency (with file-level granularity)
3. A ranked "hotspot list" of the most critical and most risky modules/files
4. Interface boundaries and contract surfaces where failures actually propagate

**Why this matters**: Folder-level audits miss the real failure modes. A bug in `config.py` propagates to 13 modules. A change to `regime/shock_vector.py` breaks both `backtest/engine.py` and `backtest/execution.py`. Without a dependency map, auditors don't know which changes are safe and which are load-bearing.

---

## LLM Prompt Spec: Architecture & Dependency Map Generator

### Context to Provide the LLM

The LLM needs access to the full repo. Provide it with:
- The complete file tree (use `find . -name "*.py" -not -path "*__pycache__*" | sort`)
- This spec document
- The existing architecture doc at `docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md`
- The existing contracts doc at `docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md`

### Step-by-Step Instructions for the LLM

#### PHASE 1: Module Inventory

For each top-level Python package directory, produce a module card:

```yaml
module: <directory_name>
path: <relative_path>
purpose: <1-2 sentence description verified from __init__.py and main files>
file_count: <number of .py files>
key_files:
  - path: <relative_path>
    role: <what this file does>
    lines: <approximate line count>
public_interface:
  - <list of classes/functions that OTHER modules import from this module>
internal_only:
  - <list of classes/functions used only within this module>
```

**Modules to inventory** (14 Python packages + root scripts):
- `data/` (including `data/providers/`)
- `features/`
- `indicators/`
- `regime/`
- `models/` (including `models/iv/`)
- `backtest/`
- `risk/`
- `evaluation/`
- `validation/`
- `autopilot/`
- `kalshi/` (including `kalshi/tests/`)
- `api/` (including `api/routers/`, `api/services/`, `api/schemas/`, `api/jobs/`, `api/cache/`, `api/deps/`)
- `utils/`
- `scripts/`
- Root-level files: `config.py`, `config_structured.py`, `reproducibility.py`, `run_*.py`

#### PHASE 2: Dependency Extraction

For every `.py` file in the repo, extract all internal imports. Internal imports use two patterns:
1. **Absolute**: `from quant_engine.MODULE import ...` (used in `run_*.py` entry points)
2. **Relative**: `from ..MODULE import ...` or `from .SUBMODULE import ...` (used within packages)

**Critical**: Many imports are **lazy/conditional** (inside function bodies, behind try/except, or behind if-guards). These MUST be captured separately because they represent optional coupling that can silently break.

Produce a dependency edge list:

```yaml
dependencies:
  - source_file: <file that contains the import>
    source_module: <top-level module of source file>
    target_file: <file being imported from>
    target_module: <top-level module of target file>
    import_type: top_level | lazy | conditional
    symbols_imported: [<list of specific names imported>]
```

#### PHASE 3: Dependency Matrix

Aggregate the edge list into a module-level adjacency matrix:

```
              config  data  features  indicators  regime  models  backtest  risk  evaluation  validation  autopilot  kalshi  api  utils
config          -      -      -         -          -       -        -        -       -           -           -         -      -     -
data            ✓      -      -         -          -       -        -        -       -           ✓           -         ✓      -     -
features        ✓      ✓      -         ✓          ✓       -        -        -       -           -           -         -      -     -
indicators      -      -      -         -          -       -        -        -       -           -           -         -      -     -
regime          ✓      -      -         -          -       -        -        -       -           -           -         -      -     -
models          ✓      -      ✓         -          -       -        -        -       -           ✓           -         -      -     -
backtest        ✓      -      -         -          ✓       -        -        ✓       -           ✓           -         -      -     -
risk            ✓      -      -         -          ✓       -        -        -       -           -           -         -      -     -
evaluation      ✓      -      -         -          -       ✓        ✓        -       -           -           -         -      -     -
validation      ✓      ✓      -         -          -       -        -        -       -           -           -         -      -     -
autopilot       ✓      ✓      ✓         -          ✓       ✓        ✓        ✓       -           -           -         -      ✓     -
kalshi          ✓      -      ✓         -          -       -        ✓        -       -           -           ✓         -      -     -
api             -      -      -         -          -       -        -        -       -           -           -         -      -     -
utils           ✓      -      -         -          -       -        -        -       -           -           -         -      -     -
```

**NOTE TO LLM**: The matrix above is a VERIFIED starting point based on actual import tracing done 2026-02-27. You MUST re-verify every cell by scanning imports yourself — do not trust this matrix blindly. Report any discrepancies.

#### PHASE 4: Hotspot Analysis

Rank every module and every file by risk using these criteria:

**Module-Level Hotspot Scoring** (score each 0-3, sum for total):

| Criterion | 0 | 1 | 2 | 3 |
|-----------|---|---|---|---|
| Fan-in (how many modules depend on it) | 0 | 1-2 | 3-5 | 6+ |
| Fan-out (how many modules it depends on) | 0 | 1-2 | 3-5 | 6+ |
| Contract surface area (shared types, schemas, file formats) | None | Internal only | 1-2 external consumers | 3+ external consumers |
| Change frequency (git log --oneline --since=30.days) | 0-2 commits | 3-10 | 11-20 | 21+ |
| Complexity (lines of code in largest file) | <200 | 200-500 | 500-1000 | 1000+ |
| Test coverage gaps (files with no corresponding test) | All tested | 1-2 gaps | 3-5 gaps | 6+ gaps |

**File-Level Hotspot Scoring** (same criteria applied per-file):

| Criterion | Weight | How to Measure |
|-----------|--------|----------------|
| Cross-module import count (how many other modules import this file) | 3x | Count distinct module directories that import it |
| Is it a contract/interface file (defines types consumed elsewhere) | 2x | Does it export classes/types used in other module signatures |
| Lines of code | 1x | `wc -l` |
| Cyclomatic complexity proxy (number of if/elif/try/except blocks) | 1x | `grep -c "if \|elif \|except \|try:" file` |
| Has lazy/conditional imports from other modules | 2x | Imports inside function bodies |

**VERIFIED HIGH-RISK FILES** (the LLM should confirm these and find others):

1. **`config.py`** — Imported by 13/14 modules. ~1500 lines. 200+ constants. Single point of failure for the entire system. Any constant rename or value change silently propagates everywhere.

2. **`autopilot/engine.py`** — 23+ cross-module imports from 8 different modules (backtest, models, config, data, features, regime, risk, api). The most tightly coupled file in the system. A change in ANY of those 8 modules can break autopilot.

3. **`features/pipeline.py`** — 35+ imports from indicators (21 symbols), config (14 constants), regime, and data. Computes 90+ features. Changes to indicator signatures or config constants break the entire feature layer.

4. **`backtest/engine.py`** — 9 cross-module dependencies (config, regime.shock_vector, regime.uncertainty_gate, validation.preconditions, risk.position_sizer, risk.drawdown, risk.stop_loss, risk.portfolio_risk, risk.metrics). Core simulation loop — errors here invalidate all backtest results.

5. **`autopilot/paper_trader.py`** — Imports from config, backtest (3 files), risk (4 files), AND api.services (3 files). The api→autopilot→api circular reference path runs through this file.

6. **`regime/shock_vector.py`** — Imported by both backtest/engine.py and backtest/execution.py. Defines the ShockVector structural state representation with a version-locked schema. Schema changes here break the backtest layer.

7. **`validation/preconditions.py`** — Imported by both models/trainer.py and backtest/engine.py. Defines the execution contract (RET_TYPE, LABEL_H, etc.). If this file's checks change, both training and backtesting behavior changes.

8. **`models/predictor.py`** — Imports from config AND features.pipeline. The bridge between trained models and predictions. Changes to feature names or ordering break prediction.

9. **`regime/uncertainty_gate.py`** — Imported by backtest/engine.py, autopilot/engine.py, AND risk/position_sizer.py. Controls signal suppression — a threshold change here silently changes position sizing, backtest results, and autopilot promotion.

10. **`data/loader.py`** — Imported by autopilot/engine.py and all run_*.py entry points. Contains lazy import of validation.data_integrity. Primary data ingestion — quality regression here corrupts everything downstream.

#### PHASE 5: Interface Boundary Analysis

For each pair of modules with dependencies, document the **interface contract**:

```yaml
boundary:
  provider: <module that exports>
  consumer: <module that imports>
  contract_type: function_call | class_instantiation | constant_read | type_import | file_schema
  symbols:
    - name: <exported symbol>
      type: <class | function | constant | type>
      stability: stable | evolving | deprecated
      breaking_change_impact: <what breaks if this changes>
  shared_artifacts:
    - path: <file path if they share file-based state>
      format: <json | csv | parquet | sqlite | yaml>
      schema_owner: <which module defines the schema>
```

**VERIFIED SHARED ARTIFACTS** (file-based coupling the LLM must capture):

| Artifact | Writer | Reader(s) | Format |
|----------|--------|-----------|--------|
| `trained_models/*/` | models/trainer.py | models/predictor.py, api/services/model_service.py | joblib + JSON metadata |
| `results/backtest_*d_summary.json` | backtest/engine.py | api/services/backtest_service.py, evaluation/engine.py | JSON |
| `results/backtest_*d_trades.csv` | backtest/engine.py | api/services/backtest_service.py, evaluation/engine.py | CSV |
| `results/predictions_*d.csv` | models/predictor.py, run_predict.py | api/services/results_service.py | CSV |
| `results/autopilot/strategy_registry.json` | autopilot/registry.py | api/services/autopilot_service.py | JSON |
| `results/autopilot/paper_state.json` | autopilot/paper_trader.py | api/services/autopilot_service.py | JSON |
| `results/autopilot/latest_cycle.json` | autopilot/engine.py | api/services/autopilot_service.py | JSON |
| `data/cache/*.parquet` | data/local_cache.py | data/loader.py | Parquet |
| `api_jobs.db` | api/jobs/store.py | api/jobs/runner.py, api/routers/jobs.py | SQLite |
| `config_data/universe.yaml` | run_wrds_daily_refresh.py | config.py, risk/universe_config.py | YAML |
| `data/kalshi.duckdb` | kalshi/storage.py | kalshi/pipeline.py, kalshi/events.py, etc. | DuckDB |

#### PHASE 6: Output Format

Produce three deliverables:

**Deliverable 1: `ARCHITECTURE_MAP.md`**
- Module inventory cards (Phase 1)
- Dependency matrix (Phase 3)
- Mermaid flowchart showing all module-to-module edges
- Shared artifact table

**Deliverable 2: `HOTSPOT_LIST.md`**
- Ranked hotspot table (modules, then files)
- For each hotspot: risk score, justification, what breaks if it changes, recommended audit priority
- "Blast radius" for top 10 files (if this file breaks, what is affected)

**Deliverable 3: `DEPENDENCY_EDGES.json`**
- Machine-readable edge list from Phase 2
- Can be loaded into graph visualization tools (Graphviz, D3, etc.)

---

# PART 2: SUBSYSTEM CHUNKING FOR AUDIT

## Purpose

Reorganize the repo's audit surface into self-contained "subsystem folders" where every file that depends on or is depended upon by other files in the group lives in the same folder. This makes it possible to audit one subsystem completely without needing to cross-reference files scattered across 5 different directories.

**Why this matters**: The current folder structure is organized by technical layer (data, features, models, backtest, risk, etc.). But the actual dependency graph shows that certain files from different layers are tightly coupled and MUST be audited together. For example, `backtest/engine.py` cannot be audited without also reading `risk/position_sizer.py`, `risk/drawdown.py`, `risk/stop_loss.py`, `regime/shock_vector.py`, and `regime/uncertainty_gate.py`. Splitting the audit by folder means the auditor misses these critical interfaces.

**Important constraint**: This spec produces a VIRTUAL grouping for audit purposes. It does NOT propose renaming or moving actual source files (which would break all imports). Instead, it produces an audit manifest that tells the auditor "to audit subsystem X, read these files."

---

## LLM Prompt Spec: Subsystem Chunker

### Context to Provide the LLM

The LLM needs:
- The complete dependency edge list from Part 1 (or access to regenerate it)
- This spec document
- The file tree of the repo

### Step-by-Step Instructions for the LLM

#### PHASE 1: Build the Full Dependency Graph

Using the import analysis from Part 1 (or by re-scanning all imports), build a directed graph where:
- **Nodes** = individual `.py` files
- **Edges** = import relationships (file A imports from file B → edge from A to B)
- **Edge weight** = number of symbols imported (more symbols = tighter coupling)

Also add edges for **shared artifact coupling** (file A writes a JSON file that file B reads → edge between them).

#### PHASE 2: Identify Strongly Connected Components

Run a strongly connected component (SCC) analysis on the graph. Files in the same SCC MUST be in the same subsystem (they form a dependency cycle and cannot be separated).

**KNOWN SCC TO VERIFY**: `autopilot/paper_trader.py` → `api/services/health_service.py` → (potentially back to autopilot). The LLM must confirm whether this is a true cycle or a one-way dependency.

#### PHASE 3: Community Detection / Clustering

After collapsing SCCs into single nodes, cluster the remaining DAG into subsystems using these rules (in priority order):

**Rule 1 — Contract Co-location**: If file A defines a type/class/schema and file B imports that type, they belong in the same subsystem UNLESS the type is a "universal" (imported by 5+ modules — see Rule 5).

**Rule 2 — Shared Artifact Co-location**: If file A writes an artifact and file B reads it, they belong in the same subsystem.

**Rule 3 — Bilateral Coupling**: If module X imports from module Y AND module Y imports from module X (even through different files), all files involved in those cross-imports belong in the same subsystem.

**Rule 4 — Transitive Closure (depth ≤ 2)**: If A→B→C and all three are in different modules, consider grouping them if B is not a "universal" hub file.

**Rule 5 — Universal Hub Exclusion**: Files imported by 5+ modules are "universal" and belong in their own "shared infrastructure" subsystem. They should NOT pull all their consumers into one mega-subsystem. Known universals:
- `config.py` (13 modules)
- `config_structured.py` (used by validation, config.py)
- `reproducibility.py` (used by run_*.py scripts)

**Rule 6 — Subsystem Size Bounds**: Target 5-25 files per subsystem. If a cluster exceeds 25, split along the weakest internal edge. If a cluster is under 5, consider merging with its strongest neighbor.

#### PHASE 4: Produce Subsystem Definitions

**VERIFIED STARTING POINT** — Based on actual dependency analysis, here are the expected subsystems. The LLM MUST verify these by re-tracing imports and adjust as needed:

**Subsystem 1: Shared Infrastructure**
```
config.py
config_structured.py
config_data/universe.yaml
reproducibility.py
utils/logging.py
```
Rationale: These are universal dependencies imported by nearly every module. They must be audited first because changes here propagate everywhere.

**Subsystem 2: Data Ingestion & Quality**
```
data/loader.py
data/local_cache.py
data/provider_base.py
data/provider_registry.py
data/providers/wrds_provider.py
data/providers/alpaca_provider.py
data/providers/alpha_vantage_provider.py
data/quality.py
data/cross_source_validator.py
data/intraday_quality.py
data/survivorship.py
data/alternative.py
data/feature_store.py
validation/data_integrity.py          ← cross-module: imported by data/loader.py
validation/leakage_detection.py
validation/feature_redundancy.py
```
Rationale: `validation/data_integrity.py` is imported by `data/loader.py` — they share a quality-gate contract. The other validation files are also data-quality focused.

**Subsystem 3: Feature Engineering**
```
features/pipeline.py
features/research_factors.py
features/options_factors.py
features/lob_features.py
features/intraday.py
features/macro.py
features/harx_spillovers.py
features/wave_flow.py
features/version.py
indicators/indicators.py
indicators/spectral.py
indicators/eigenvalue.py
indicators/ot_divergence.py
indicators/ssa.py
indicators/tail_risk.py
```
Rationale: `features/pipeline.py` imports 21 symbols from `indicators/`. They are one logical unit. The indicators module has zero other consumers.

**Subsystem 4: Regime Detection**
```
regime/detector.py
regime/hmm.py
regime/jump_model.py
regime/jump_model_legacy.py
regime/jump_model_pypi.py
regime/bocpd.py
regime/correlation.py
regime/confidence_calibrator.py
regime/consensus.py
regime/shock_vector.py
regime/uncertainty_gate.py
regime/online_update.py
```
Rationale: Regime is relatively self-contained (only depends on config). However, `shock_vector.py` and `uncertainty_gate.py` are consumed by backtest and risk — the auditor MUST cross-reference Subsystem 5 after auditing this one.

**Subsystem 5: Backtesting + Risk (Execution Layer)**
```
backtest/engine.py
backtest/validation.py
backtest/advanced_validation.py
backtest/execution.py
backtest/cost_calibrator.py
backtest/cost_stress.py
backtest/optimal_execution.py
backtest/null_models.py
backtest/adv_tracker.py
backtest/survivorship_comparison.py
risk/position_sizer.py
risk/portfolio_risk.py
risk/drawdown.py
risk/stop_loss.py
risk/metrics.py
risk/covariance.py
risk/portfolio_optimizer.py
risk/factor_portfolio.py
risk/factor_exposures.py
risk/factor_monitor.py
risk/attribution.py
risk/stress_test.py
risk/cost_budget.py
risk/constraint_replay.py
risk/universe_config.py
validation/preconditions.py          ← cross-module: imported by backtest/engine.py AND models/trainer.py
regime/shock_vector.py               ← cross-module: imported by backtest/engine.py, backtest/execution.py
regime/uncertainty_gate.py           ← cross-module: imported by backtest/engine.py, risk/position_sizer.py
```
Rationale: Backtest and risk are bilaterally coupled — `backtest/engine.py` imports 5 risk files, and they share the regime interface files. `validation/preconditions.py` is the execution contract enforced by both training and backtesting. Note: this subsystem has ~28 files, slightly above the 25-file target. The LLM should evaluate whether to split backtest-validation from risk-management, but the tight coupling likely argues against splitting.

**Subsystem 6: Model Training & Prediction**
```
models/trainer.py
models/predictor.py
models/versioning.py
models/governance.py
models/calibration.py
models/conformal.py
models/walk_forward.py
models/online_learning.py
models/feature_stability.py
models/shift_detection.py
models/retrain_trigger.py
models/cross_sectional.py
models/neural_net.py
models/iv/models.py
```
Rationale: Models is mostly self-contained. It imports from config, features.pipeline (for feature names), and validation.preconditions. The cross-references to features and validation are noted as audit dependencies.

**Subsystem 7: Evaluation & Diagnostics**
```
evaluation/engine.py
evaluation/metrics.py
evaluation/slicing.py
evaluation/fragility.py
evaluation/calibration_analysis.py
evaluation/ml_diagnostics.py
evaluation/visualization.py
```
Rationale: Evaluation imports from config, models.calibration, and backtest.validation. Relatively self-contained but must be audited after Subsystems 5 and 6.

**Subsystem 8: Autopilot (Strategy Discovery)**
```
autopilot/engine.py
autopilot/strategy_discovery.py
autopilot/promotion_gate.py
autopilot/registry.py
autopilot/paper_trader.py
autopilot/meta_labeler.py
autopilot/strategy_allocator.py
```
Rationale: Autopilot is the most tightly coupled module (imports from 8 other modules). It must be audited LAST because it depends on virtually every other subsystem. The auditor should treat this as an integration audit.

**Subsystem 9: Kalshi (Event Markets)**
```
kalshi/provider.py
kalshi/client.py
kalshi/pipeline.py
kalshi/storage.py
kalshi/events.py
kalshi/distribution.py
kalshi/options.py
kalshi/promotion.py
kalshi/walkforward.py
kalshi/quality.py
kalshi/mapping_store.py
kalshi/microstructure.py
kalshi/disagreement.py
kalshi/regimes.py
kalshi/router.py
kalshi/tests/*
```
Rationale: Kalshi is a semi-isolated vertical. It imports from features.options_factors, autopilot.promotion_gate, and backtest (advanced_validation, engine). Can be audited independently after Subsystems 3 and 5.

**Subsystem 10: API & Frontend**
```
api/main.py
api/config.py
api/errors.py
api/ab_testing.py
api/orchestrator.py
api/deps/providers.py
api/cache/manager.py
api/cache/invalidation.py
api/jobs/store.py
api/jobs/runner.py
api/jobs/models.py
api/jobs/autopilot_job.py
api/jobs/backtest_job.py
api/jobs/predict_job.py
api/jobs/train_job.py
api/routers/* (all 16 routers)
api/services/* (all 13 services)
api/schemas/* (all 9 schemas)
```
Rationale: The API layer is a pure consumer — it imports from core modules but nothing imports from it (except autopilot/paper_trader.py which imports api.services — a coupling concern). Audit this last.

**Subsystem 11: Entry Points & Scripts**
```
run_train.py
run_backtest.py
run_predict.py
run_autopilot.py
run_retrain.py
run_server.py
run_kalshi_event_pipeline.py
run_wrds_daily_refresh.py
run_rehydrate_cache_metadata.py
scripts/alpaca_intraday_download.py
scripts/ibkr_intraday_download.py
scripts/ibkr_daily_gapfill.py
scripts/compare_regime_models.py
scripts/generate_types.py
```
Rationale: These are orchestration entry points. They wire together multiple subsystems. Audit them to verify correct wiring.

#### PHASE 5: Produce Audit Ordering

Based on the dependency DAG between subsystems, produce a recommended audit order:

```
1. Subsystem 1: Shared Infrastructure     (no dependencies — audit first)
2. Subsystem 2: Data Ingestion & Quality   (depends on: 1)
3. Subsystem 3: Feature Engineering        (depends on: 1, 2)
4. Subsystem 4: Regime Detection           (depends on: 1)
5. Subsystem 6: Model Training             (depends on: 1, 3)
6. Subsystem 5: Backtesting + Risk         (depends on: 1, 4)
7. Subsystem 7: Evaluation                 (depends on: 1, 5, 6)
8. Subsystem 8: Autopilot                  (depends on: 1, 2, 3, 4, 5, 6, 10)
9. Subsystem 9: Kalshi                     (depends on: 1, 3, 5, 8)
10. Subsystem 10: API & Frontend           (depends on: all core subsystems)
11. Subsystem 11: Entry Points             (depends on: all subsystems — audit last)
```

#### PHASE 6: Cross-Subsystem Dependency Report

For each pair of subsystems with dependencies, document:

```yaml
cross_subsystem_dependency:
  from_subsystem: <name>
  to_subsystem: <name>
  coupling_files:
    - source: <file in from_subsystem>
      target: <file in to_subsystem>
      symbols: [<what is imported>]
      coupling_strength: tight | moderate | loose
  audit_note: <what the auditor must verify at this boundary>
```

#### PHASE 7: Output Format

Produce two deliverables:

**Deliverable 1: `SUBSYSTEM_AUDIT_MANIFEST.md`**
- Subsystem definitions (name, files, rationale)
- Recommended audit order with dependency justification
- Cross-subsystem boundary table
- For each subsystem: estimated audit effort (hours), key risk areas, what to look for

**Deliverable 2: `SUBSYSTEM_MAP.json`**
- Machine-readable mapping: `{ "subsystem_name": { "files": [...], "depends_on": [...], "audit_order": N } }`
- Can be used to auto-generate audit checklists or symlink folders

---

## Validation Criteria

The LLM's output is correct if and only if:

1. **Completeness**: Every `.py` file in the repo appears in exactly one subsystem
2. **Co-location**: For every import edge A→B where A and B are in different subsystems, the cross-subsystem dependency report documents it
3. **No orphans**: No file is placed in a subsystem where it has zero import relationships with other files in that subsystem (unless it's a leaf utility)
4. **Hotspot coverage**: Every file with fan-in ≥ 3 (imported by 3+ other files across modules) appears in the hotspot list
5. **Cycle detection**: Any true circular dependency is identified and the involved files are co-located
6. **Audit ordering**: The audit order respects the dependency DAG (no subsystem is audited before its dependencies)

---

## Known Limitations & Caveats

1. **config.py is a god object**: It's imported by 13 modules and contains 200+ constants. The dependency map will show it connected to everything. The subsystem chunking puts it in "Shared Infrastructure" but the auditor still needs to understand which constants matter to which subsystem. A future improvement would be to split config.py by subsystem.

2. **Lazy imports hide coupling**: Several files (especially autopilot/engine.py, backtest/engine.py, features/pipeline.py) use lazy imports inside function bodies. These won't appear in simple top-of-file import scans. The LLM MUST scan function bodies for `from` and `import` statements.

3. **File-based coupling is invisible to import analysis**: The shared artifacts table (trained_models/, results/, cache files) represents coupling that doesn't appear in Python imports. The LLM must account for these.

4. **Frontend is TypeScript**: The frontend/ directory uses TypeScript/React and communicates with the API via HTTP. It's not captured by Python import analysis. The API schemas (api/schemas/) are the contract surface between backend and frontend.

5. **The autopilot↔api coupling is a design concern**: `autopilot/paper_trader.py` imports from `api/services/health_service.py`, `api/services/health_risk_feedback.py`, and `api/ab_testing.py`. This means the autopilot module (which the API serves) also depends on the API layer. This is a circular architectural dependency that should be flagged as a high-priority refactoring target.

6. **Test files are not included in subsystem groupings by default**: The `tests/` directory (98 files) and `kalshi/tests/` (8 files) should be mapped to their corresponding subsystems as a secondary pass, but are excluded from the primary chunking to keep subsystems focused on production code.
