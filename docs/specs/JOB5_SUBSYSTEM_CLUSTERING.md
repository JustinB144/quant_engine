# JOB 5: SUBSYSTEM CLUSTERING
## LLM Audit Workflow — Step 5 of 7

**Purpose**: Group all files in the repo into self-contained "subsystem" clusters where co-dependent files live together. This enables auditing the system in manageable chunks where all dependencies and references for a given subsystem are in the same group.

**Estimated effort**: 1 LLM session, ~45 minutes
**Input required**: DEPENDENCY_EDGES.json (Job 2) + INTERFACE_CONTRACTS.yaml (Job 4) + HOTSPOT_LIST.md (Job 3)
**Output**: `SUBSYSTEM_MAP.json` saved to `docs/audit/`
**Depends on**: Jobs 2, 3, 4
**Feeds into**: Job 6 (Audit Ordering), Job 7 (Verification)

---

## WHAT YOU ARE BUILDING

A JSON mapping that assigns every .py file in the repo to exactly one subsystem. Files that depend on each other (import from each other, share artifacts, share types) are grouped together regardless of which folder they physically live in.

**Key constraint**: This is a VIRTUAL grouping for audit purposes. You are NOT moving files or changing imports. You are producing a manifest that says "to audit subsystem X, read these files together."

---

## EXACT INSTRUCTIONS

### Step 1: Load the dependency graph

From DEPENDENCY_EDGES.json, build a directed graph:
- Nodes = individual .py files
- Edges = import relationships (cross-module edges only for clustering; same-module edges help with internal grouping)
- Edge weight = number of symbols imported (from the symbols_imported array length)

Also add edges for shared artifact coupling from INTERFACE_CONTRACTS.yaml:
- If file A writes an artifact and file B reads it → add a bidirectional edge with weight 2

**JOB 4 VERIFIED FINDINGS affecting clustering:**
- Job 4 documented 45 boundaries (19 HIGH, 17 MEDIUM, 9 LOW) across 58 unique module pairs — significantly more than the ~25-30 originally estimated. The clustering should account for ALL 45 boundaries.
- 6 shared artifacts create file-based coupling that isn't visible in import analysis: trained_models/*.pkl (models→predictor, api), trained_models/*_meta.json (models→predictor, api), strategy_registry.json (autopilot→api), data/cache/*.parquet (data→features), backtest_*d_summary.json (backtest→api, evaluation), kalshi.duckdb (kalshi→api). Add bidirectional edges for ALL of these.
- API → config has 82 import edges (the largest cross-module dependency). api/services/health_service.py alone has 25+ config import sites. This reinforces config.py as the supreme hub.
- UncertaintyGate (regime/uncertainty_gate.py) is imported by 3 distinct modules: backtest/engine.py:78, autopilot/engine.py:61, risk/position_sizer.py:27. This is more consumers than originally noted and strengthens the case for it being a cross-subsystem contract file.

### Step 2: Apply clustering rules IN THIS ORDER

**Rule 1 — Same-module files stay together**: All files within the same top-level module directory start in the same cluster. This is the baseline.

**Rule 2 — Universal hub exclusion**: Files imported by 5+ modules go into a "Shared Infrastructure" cluster. They must NOT pull their consumers into one mega-cluster. Known universals (verified by Job 2):
- config.py (161 fan-in edges from ALL 14 modules — supreme hub)
- config_structured.py (imported by validation, derived from config.py)
- reproducibility.py (imported by run_*.py entry points)
- utils/logging.py (imported by config)
- config_data/universe.yaml (read by config.py and risk/universe_config.py)

**Rule 3 — Bilateral coupling merge**: If module X has files that import from module Y AND module Y has files that import from module X, merge the involved files into the same subsystem. Check:
- data ↔ validation: data/loader.py imports validation/data_integrity.py AND validation/data_integrity.py imports data/quality.py → MERGE these files
- data ↔ kalshi: data/provider_registry.py imports kalshi/provider.py (one-way, conditional) → do NOT merge
- autopilot ↔ api: 6 confirmed circular edges (verified by Job 2) — autopilot/paper_trader.py (4 edges) and autopilot/engine.py (2 edges) import from api/services/. The api side does NOT import autopilot code directly, so this is a one-directional code dependency but an architectural circle (api serves autopilot data, autopilot imports api services). Do NOT merge. Flag prominently as coupling concern requiring refactoring.

**Rule 4 — Exclusive consumer merge**: If module Y is ONLY imported by module X (and no other module), consider merging Y into X's subsystem. Check:
- indicators/ is ONLY imported by features/ (6 edges, all from pipeline.py — verified by Job 2) → MERGE indicators into features subsystem
- validation/preconditions.py is imported by models/ AND backtest/ → do NOT merge (shared by 2)

**Rule 4a — Leaf module handling**: Modules with 0 fan-in (nothing imports from them) are pure consumers. Verified leaf modules (from Job 2):
- evaluation/ — 0 fan-in, imports from config, models.calibration, backtest.validation
- utils/ — 0 fan-in, imports from config only
These should remain in their own subsystems but are flagged as LOW priority for audit since changes to them cannot cascade.

**Rule 5 — Strong coupling merge**: If a file imports 3+ symbols from a file in another module, they have strong coupling. Consider merging if the target file isn't a universal hub.
- features/pipeline.py imports 21+ symbols from indicators/ → strong coupling, already merged by Rule 4
- backtest/engine.py imports from 5 risk files → strong coupling, MERGE backtest+risk
- autopilot/engine.py imports from 8 modules → strong coupling but too many to merge, keep as integration layer

**Rule 6 — Cross-module contract files**: Files that define types used across module boundaries should be co-located with their primary consumer (not their definer) if the consumer is more complex:
- regime/shock_vector.py is consumed by backtest/ → include in backtest+risk subsystem
- regime/uncertainty_gate.py is consumed by backtest/, risk/, autopilot/ → include in backtest+risk subsystem (primary consumer)
- validation/preconditions.py is consumed by models/ AND backtest/ → include in backtest+risk subsystem (where it's enforced at runtime)

**Rule 7 — Size bounds**: Target 5-25 files per subsystem. If >25, split along weakest internal edge. If <5, merge with strongest neighbor.

### Step 3: Produce the subsystem assignments

**VERIFIED STARTING POINT** — Based on actual dependency analysis, here are the expected subsystems. You MUST verify these by applying the rules above to the actual dependency data and adjust as needed.

**Subsystem 1: Shared Infrastructure** (5 files)
```
config.py                          (imported by 13 modules)
config_structured.py               (typed config, imported by validation)
config_data/universe.yaml          (YAML config, read by config.py + risk)
reproducibility.py                 (run manifests, imported by run_*.py)
utils/logging.py                   (logging, imports config)
```

**Subsystem 2: Data Ingestion & Quality** (16 files)
```
data/loader.py                     data/local_cache.py
data/provider_base.py              data/provider_registry.py
data/providers/wrds_provider.py    data/providers/alpaca_provider.py
data/providers/alpha_vantage_provider.py
data/quality.py                    data/cross_source_validator.py
data/intraday_quality.py           data/survivorship.py
data/alternative.py                data/feature_store.py
validation/data_integrity.py       (cross-module: bilateral coupling with data/)
validation/leakage_detection.py    (data quality focused)
validation/feature_redundancy.py   (data quality focused)
```

**Subsystem 3: Feature Engineering** (15 files)
```
features/pipeline.py               features/research_factors.py
features/options_factors.py         features/lob_features.py
features/intraday.py               features/macro.py
features/harx_spillovers.py        features/wave_flow.py
features/version.py
indicators/indicators.py           indicators/spectral.py
indicators/eigenvalue.py           indicators/ot_divergence.py
indicators/ssa.py                  indicators/tail_risk.py
```

**Subsystem 4: Regime Detection** (12 files)
```
regime/detector.py                 regime/hmm.py
regime/jump_model.py               regime/jump_model_legacy.py
regime/jump_model_pypi.py          regime/bocpd.py
regime/correlation.py              regime/confidence_calibrator.py
regime/consensus.py                regime/shock_vector.py
regime/uncertainty_gate.py         regime/online_update.py
```
NOTE: shock_vector.py and uncertainty_gate.py are ALSO referenced by Subsystem 5. They appear here as their defining home but are flagged as cross-subsystem contract files.

**Subsystem 5: Backtesting + Risk** (~28 files)
```
backtest/engine.py                 backtest/validation.py
backtest/advanced_validation.py    backtest/execution.py
backtest/cost_calibrator.py        backtest/cost_stress.py
backtest/optimal_execution.py      backtest/null_models.py
backtest/adv_tracker.py            backtest/survivorship_comparison.py
risk/position_sizer.py             risk/portfolio_risk.py
risk/drawdown.py                   risk/stop_loss.py
risk/metrics.py                    risk/covariance.py
risk/portfolio_optimizer.py        risk/factor_portfolio.py
risk/factor_exposures.py           risk/factor_monitor.py
risk/attribution.py                risk/stress_test.py
risk/cost_budget.py                risk/constraint_replay.py
risk/universe_config.py
validation/preconditions.py        (cross-module: execution contract)
regime/shock_vector.py             (cross-module: backtest imports this)
regime/uncertainty_gate.py         (cross-module: backtest+risk import this)
```
NOTE: This subsystem has ~28 files, slightly above 25-file target. The LLM should evaluate whether backtest-core (engine, validation, execution) and risk-core (sizing, stops, drawdown) could be split, but the tight coupling at backtest/engine.py lines 316-320 argues against it.
JOB 4 FINDING: PositionSizer (risk/position_sizer.py) is flagged "evolving" — its size_position() method has 21 parameters with recent uncertainty additions. backtest/engine.py:26 imports 55+ constants from config (largest single import statement in codebase). These facts reinforce keeping backtest+risk merged.

**Subsystem 6: Model Training & Prediction** (14 files)
```
models/trainer.py                  models/predictor.py
models/versioning.py               models/governance.py
models/calibration.py              models/conformal.py
models/walk_forward.py             models/online_learning.py
models/feature_stability.py        models/shift_detection.py
models/retrain_trigger.py          models/cross_sectional.py
models/neural_net.py               models/iv/models.py
```

**Subsystem 7: Evaluation & Diagnostics** (7 files)
```
evaluation/engine.py               evaluation/metrics.py
evaluation/slicing.py              evaluation/fragility.py
evaluation/calibration_analysis.py evaluation/ml_diagnostics.py
evaluation/visualization.py
```

**Subsystem 8: Autopilot** (7 files)
```
autopilot/engine.py                autopilot/strategy_discovery.py
autopilot/promotion_gate.py        autopilot/registry.py
autopilot/paper_trader.py          autopilot/meta_labeler.py
autopilot/strategy_allocator.py
```
JOB 3 FINDINGS: autopilot/engine.py is the primary transitive amplifier (appears in 12 of 15 hotspot blast radii). autopilot/paper_trader.py has 0 cross-module dependents (nothing imports from it) — its risk is entirely from what it consumes (4 circular api edges at lines 173, 189, 211, 532). This means paper_trader.py changes cannot break other subsystems, but changes in 8 other modules can break it.

**Subsystem 9: Kalshi** (~16 files)
```
kalshi/provider.py                 kalshi/client.py
kalshi/pipeline.py                 kalshi/storage.py
kalshi/events.py                   kalshi/distribution.py
kalshi/options.py                  kalshi/promotion.py
kalshi/walkforward.py              kalshi/quality.py
kalshi/mapping_store.py            kalshi/microstructure.py
kalshi/disagreement.py             kalshi/regimes.py
kalshi/router.py
kalshi/tests/*                     (8 test files)
```

**Subsystem 10: API & Frontend** (~59 files)
```
api/main.py                        api/config.py
api/errors.py                      api/ab_testing.py
api/orchestrator.py                api/deps/providers.py
api/cache/manager.py               api/cache/invalidation.py
api/jobs/store.py                  api/jobs/runner.py
api/jobs/models.py                 api/jobs/autopilot_job.py
api/jobs/backtest_job.py           api/jobs/predict_job.py
api/jobs/train_job.py
api/routers/* (16 files)           api/services/* (13 files)
api/schemas/* (9 files)
```

**Subsystem 11: Entry Points & Scripts** (14 files)
```
run_train.py                       run_backtest.py
run_predict.py                     run_autopilot.py
run_retrain.py                     run_server.py
run_kalshi_event_pipeline.py       run_wrds_daily_refresh.py
run_rehydrate_cache_metadata.py
scripts/alpaca_intraday_download.py
scripts/ibkr_intraday_download.py  scripts/ibkr_daily_gapfill.py
scripts/compare_regime_models.py   scripts/generate_types.py
```

### Step 4: Handle files that appear in multiple subsystems

Some files are listed in two subsystems above (shock_vector.py, uncertainty_gate.py, preconditions.py). Each file MUST be assigned to exactly ONE primary subsystem, with a cross-reference note in the other.

**Resolution rule**: Assign the file to the subsystem where it is DEFINED. Add a `cross_references` field listing the other subsystems that depend on it.

### Step 5: Verify completeness

- Count all .py files in each subsystem
- Sum should equal total .py files in repo (excluding tests/, __pycache__, .egg-info)
- No file should appear in zero subsystems
- No file should appear in two subsystems

---

## OUTPUT FORMAT

### SUBSYSTEM_MAP.json

```json
{
  "metadata": {
    "generated": "YYYY-MM-DD",
    "total_subsystems": 11,
    "total_files": "<count>",
    "clustering_rules_applied": ["same_module", "universal_hub_exclusion", "bilateral_merge", "exclusive_consumer_merge", "strong_coupling_merge", "contract_colocation", "size_bounds"]
  },
  "subsystems": {
    "shared_infrastructure": {
      "id": 1,
      "name": "Shared Infrastructure",
      "description": "Universal configuration and utility dependencies imported by nearly every module",
      "files": ["config.py", "config_structured.py", ...],
      "file_count": 5,
      "total_lines": "<sum>",
      "depends_on": [],
      "depended_on_by": ["data_quality", "feature_engineering", "regime_detection", ...],
      "cross_references": {
        "from_other_subsystems": [],
        "to_other_subsystems": ["ALL — config.py is imported by 13 modules"]
      },
      "audit_notes": "Audit first. Changes here propagate everywhere."
    }
  }
}
```

---

## VERIFICATION CHECKLIST

- [ ] Every .py file (excluding tests/) appears in exactly one subsystem
- [ ] Files with bilateral coupling are in the same subsystem
- [ ] Universal hub files (config.py) are NOT pulling all consumers into one cluster
- [ ] Each subsystem has 5-25 files (document any exceptions with justification)
- [ ] Cross-references are documented for files that appear at boundaries
- [ ] The subsystem map is valid JSON
