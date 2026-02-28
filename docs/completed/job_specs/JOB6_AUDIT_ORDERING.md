# JOB 6: AUDIT ORDERING + CROSS-SUBSYSTEM REPORT
## LLM Audit Workflow — Step 6 of 7

**Purpose**: Determine the correct order to audit subsystems (respecting dependencies — never audit a subsystem before the subsystems it depends on) and document every cross-subsystem boundary that an auditor must verify when transitioning between subsystems.

**Estimated effort**: 1 LLM session, ~30 minutes
**Input required**: SUBSYSTEM_MAP.json (Job 5) + DEPENDENCY_EDGES.json (Job 2) + INTERFACE_CONTRACTS.yaml (Job 4)
**Output**: `SUBSYSTEM_AUDIT_MANIFEST.md` saved to `docs/audit/`
**Depends on**: Jobs 2, 4, 5
**Feeds into**: Job 7 (Verification), and the actual audit work

---

## WHAT YOU ARE BUILDING

A comprehensive audit manifest that tells an auditor:
1. What order to audit subsystems in
2. For each subsystem: what files to read, what to look for, estimated time, key risks
3. Between subsystems: what crosses the boundary, what to verify at each transition

---

## EXACT INSTRUCTIONS

### Step 1: Build the subsystem dependency DAG

From SUBSYSTEM_MAP.json, extract the `depends_on` field for each subsystem. Build a directed acyclic graph (DAG) where:
- Nodes = subsystems
- Edges = dependency relationships (subsystem A depends on subsystem B → edge from A to B)

Verify it's actually a DAG (no cycles). If you find a cycle, document it and break it at the weakest edge.

### Step 2: Topological sort for audit order

Perform a topological sort of the DAG. Subsystems with no dependencies come first. When multiple subsystems have the same depth, order by:
1. Risk score (from HOTSPOT_LIST.md — higher risk = audit earlier)
2. File count (fewer files = faster to audit = audit earlier for quick wins)

**VERIFIED AUDIT ORDER** (updated with Job 5 SUBSYSTEM_MAP.json actuals 2026-02-27):

```
ORDER | SUBSYSTEM                    | DEPENDS ON              | FILES | LINES  | EST. HOURS | PRIORITY
------+------------------------------+-------------------------+-------+--------+------------+---------
  1   | Shared Infrastructure        | (none)                  |   7   |  2,151 |   2-3      | CRITICAL
  2   | Data Ingestion & Quality     | 1                       |  19   |  9,044 |   6-8      | HIGH
  3   | Feature Engineering          | 1, 2 (opt: 4)          |  17   |  8,559 |   6-8      | CRITICAL
  4   | Regime Detection             | 1                       |  13   |  4,420 |   5-6      | HIGH
  5   | Backtesting + Risk           | 1, 4                    |  28   | 13,132 |  10-14     | CRITICAL
  6   | Model Training & Prediction  | 1, 3, 5                 |  16   |  6,153 |   5-7      | HIGH
  7   | Evaluation & Diagnostics     | 1, 5, 6                 |   8   |  2,816 |   3-4      | LOW
  8   | Autopilot                    | 1,2,3,4,5,6 (opt:10)   |   8   |  4,480 |   4-6      | CRITICAL
  9   | Kalshi                       | 1, 3, 5, 8              |  16   |  5,208 |   5-7      | MEDIUM
 10   | API & Frontend               | all core subsystems      |  59   | 10,188 |  12-16     | HIGH
 11   | Entry Points & Scripts       | all subsystems           |  17   |  8,883 |   3-4      | LOW
------+------------------------------+-------------------------+-------+--------+------------+---------
TOTAL |                              |                         | 208   | 75,027 |  62-83     |
```

**CHANGES FROM ORIGINAL SPEC (updated by Job 5 output):**
- Total files: 193 → 208 (15 __init__.py files now explicitly counted in subsystem assignments)
- Total lines: 75,027 (first time verified — sum of all subsystem line counts)
- Subsystem 1: 5 → 7 files (added root __init__.py, config_data/__init__.py)
- Subsystem 2: 16 → 19 files (added data/__init__.py, data/providers/__init__.py, validation/__init__.py)
- Subsystem 3: 15 → 17 files (added features/__init__.py, indicators/__init__.py); has optional dependency on regime_detection (features/pipeline.py:1303 lazily imports regime/correlation.py)
- Subsystem 5: depends on 1,4 (not 1,3 — model_training_prediction depends on backtesting_risk, not vice versa)
- Subsystem 6: depends on 1,3,5 (added backtesting_risk dependency — validation/preconditions.py is consumed by models/trainer.py)
- Subsystem 8: 7 → 8 files; autopilot→api is now classified as "optional_depends_on" not hard dependency
- Subsystem 11: 14 → 17 files (added 3 audit scripts: extract_dependencies.py, generate_interface_contracts.py, hotspot_scoring.py)
- Priority column added from SUBSYSTEM_MAP.json audit_priority field

The LLM MUST verify this order by:
1. Loading the actual SUBSYSTEM_MAP.json
2. Checking that every subsystem's dependencies appear EARLIER in the ordering
3. Flagging any ordering violations

**JOB 5 VERIFIED FINDINGS affecting audit order:**
- **Total files: 208 (not 193)** — every __init__.py is now explicitly assigned to a subsystem. The LLM should use the actual SUBSYSTEM_MAP.json file_count values, not the estimates in this spec.
- **Optional dependencies documented**: features→regime (lazy, features work without it), data→kalshi (conditional factory), autopilot→api (6 lazy/conditional circular edges). The audit order treats optional dependencies as non-blocking but they must be flagged in transition guides.
- **API has 115 outbound edges but only 2 modules' fan-in** — lowest fan-in despite highest fan-out. API is a massive consumer that rarely exports, which means API changes have low cascade risk but API is highly fragile to upstream changes.
- **Architectural cycle is NOT a true SCC** — the reverse edge (api→autopilot) goes through api/jobs/autopilot_job.py:12 (job runner), not the service layer. This means the 6 forward edges (autopilot→api services) are the only real concern.
- **data/cache/*.meta.json** — additional shared artifact (metadata sidecars) coupling data subsystem files internally.
- **2 size bound exceptions justified**: backtesting_risk (28 files, +3 over limit) due to 5-way tight coupling; api_frontend (59 files, +34 over limit) due to internal layering cohesion.
- **Refactoring recommendation from Job 5**: extract health risk feedback interface that autopilot depends on, rather than importing from api service layer directly. The auditor should note this as a recommended follow-up.

**JOB 2 VERIFIED FINDINGS affecting audit order:**
- **evaluation and utils are leaf modules (0 fan-in)**: Nothing depends on them, so they can be audited at ANY point without blocking other subsystems. Their current position (7 for evaluation) is based on their own dependencies, not on other subsystems needing them.
- **config.py has 161 fan-in edges**: Subsystem 1 (Shared Infrastructure) is correctly ordered first — every single other subsystem depends on it.
- **autopilot has 39 fan-out edges + 6 circular api edges**: Subsystem 8 (Autopilot) is correctly near the end. The 6 circular edges to api mean the auditor should cross-reference Subsystem 10 (API) DURING the Autopilot audit, even though API is ordered later. Add a note in the Autopilot audit brief to review api/services/health_service.py and api/ab_testing.py as part of the Autopilot audit.
- **indicators is exclusively consumed by features (6 edges)**: Confirms that Subsystem 3 (Feature Engineering, which includes indicators) has no fan-out to other subsystems except through features/pipeline.py.

### Step 3: For each subsystem, produce an audit brief

```yaml
subsystem:
  name: <subsystem name>
  audit_order: <number>
  estimated_hours: <range>

  files_to_read:
    critical:
      - path: <file>
        why: <why this file matters>
        read_first: true | false
    supporting:
      - path: <file>
        why: <context for critical files>

  what_to_look_for:
    - category: <correctness | security | performance | maintainability>
      checks:
        - <specific thing to verify>

  known_risks:
    - risk: <description>
      severity: CRITICAL | HIGH | MEDIUM | LOW
      evidence: <where in the code this was observed>

  dependencies_to_verify:
    - depends_on_subsystem: <name>
      boundary_contracts:
        - symbol: <imported symbol>
          check: <what to verify about this symbol>

  outputs_to_verify:
    - artifact: <file path/pattern>
      consumers: [<subsystems that read this>]
      check: <what to verify about the output>
```

### Step 4: Produce cross-subsystem transition guides

For every pair of adjacent subsystems in the audit order, document what the auditor needs to carry forward:

```yaml
transition:
  from_subsystem: <name just completed>
  to_subsystem: <name starting next>

  carry_forward:
    - context: <what understanding from the previous subsystem is needed>
      relevant_files: [<files in previous subsystem>]

  boundary_checks:
    - boundary_id: <from INTERFACE_CONTRACTS.yaml>
      check: <specific verification>
      files_to_compare:
        - <file from previous subsystem>
        - <file from next subsystem>

  common_failure_modes:
    - failure: <what typically goes wrong at this boundary>
      detection: <how to detect it>
```

### Step 5: Produce the audit brief for each subsystem

**Subsystem 1: Shared Infrastructure** (Audit first — 7 files, 2,151 lines, CRITICAL)
- Critical files: config.py (1,020 lines, 200+ constants), config_structured.py
- JOB 5: 7 files including __init__.py and config_data/__init__.py; config.py hotspot score 16/21
- What to look for: Deprecated/placeholder constants still referenced, TRUTH_LAYER flag consistency, config_structured↔config.py synchronization, STATUS annotations
- Known risks: God object pattern, no typed validation for most constants
- Outputs to verify: Every constant used by downstream modules exists and has correct type
- JOB 4 BOUNDARY DETAIL: config.py has the top 3 largest import statements consuming it: backtest/engine.py:26 (55+ constants), autopilot/paper_trader.py:13 (40+ EXEC_*/PAPER_* constants), autopilot/promotion_gate.py:13 (29+ PROMOTION_* constants). API module alone has 82 import edges to config across 20+ files (all lazy/conditional). Auditor should verify that every config constant referenced in these import statements actually exists and has the expected type.
- JOB 5 SHARED ARTIFACT: config_data/universe.yaml is a non-code artifact read by config.py and risk/universe_config.py — universe membership changes propagate to all data loading, backtesting, and trading.

**Subsystem 2: Data Ingestion & Quality** (19 files, 9,044 lines, HIGH)
- Critical files: data/loader.py (primary ingestion, hotspot 15/21), data/quality.py (OHLCV validation), validation/data_integrity.py (quality gate)
- JOB 5: 19 files (up from 16) — includes data/__init__.py, data/providers/__init__.py, validation/__init__.py. Also includes validation/leakage_detection.py and validation/feature_redundancy.py (data quality focused, no external consumers)
- What to look for: Cache trust/freshness logic (CACHE_MAX_STALENESS_DAYS), WRDS fallback behavior, survivorship filtering correctness, OHLC relationship validation, DataIntegrityValidator conditional import at data/loader.py:567
- Dependencies to verify: config.py constants for paths, thresholds
- Optional dependency: data/provider_registry.py:23 lazily imports kalshi/provider.py (conditional factory — data loading works without Kalshi)
- Outputs: data/cache/*.parquet files consumed by loader AND features/pipeline.py; also data/cache/*.meta.json sidecars for staleness detection

**Subsystem 3: Feature Engineering** (17 files, 8,559 lines, CRITICAL)
- Critical files: features/pipeline.py (1,541 lines, 90+ features, hotspot 16/21), indicators/indicators.py (2,904 lines, 92 indicator subclasses, hotspot 14/21)
- JOB 5: 17 files (up from 15) — includes features/__init__.py, indicators/__init__.py
- Optional dependency: features/pipeline.py:1303 lazily imports regime/correlation.py (CorrelationRegimeDetector) with try/except — features work without it
- What to look for: Feature causality types (CAUSAL vs END_OF_DAY vs RESEARCH_ONLY), indicator signature compatibility, lazy import correctness (18+ lazy imports in pipeline.py)
- Dependencies to verify: All 21+ indicator symbols imported at pipeline.py:21 exist with correct signatures
- JOB 3 TEST COVERAGE WARNING: indicators/ has 7 files and 0 tests. Since indicators has extreme transitive amplification (1 direct dependent → 9 transitive dependents), untested indicator code silently propagates through the entire feature→model→backtest→evaluation chain. Auditor should prioritize verifying indicator correctness manually.
- JOB 4 BOUNDARY DETAIL: 5 advanced indicator analyzers are conditionally imported at specific lines: SpectralAnalyzer:769, SSADecomposer:789, TailRiskAnalyzer:809, OptimalTransportAnalyzer:836, EigenvalueAnalyzer:1337. Each has a compute_all() method returning a dict whose keys become feature column names. Auditor must verify these handle ImportError gracefully AND that return dict keys remain stable (key changes silently break downstream feature columns).

**Subsystem 4: Regime Detection** (13 files, 4,420 lines, HIGH)
- Critical files: regime/detector.py (main detector, hotspot 15/21), regime/shock_vector.py (version-locked schema, hotspot 12/21 +3 contract bonus), regime/uncertainty_gate.py (sizing modifier, hotspot 12/21 +2 contract bonus)
- JOB 5: 13 files (up from 12) — includes regime/__init__.py. UncertaintyGate stability noted as "evolving" in SUBSYSTEM_MAP.
- What to look for: Regime label consistency (0-3 mapping), confidence calibration, ShockVector schema version lock
- Outputs: shock_vector.py and uncertainty_gate.py are consumed by Subsystem 5
- JOB 4 BOUNDARY DETAIL: UncertaintyGate is imported by 3 distinct modules (backtest/engine.py:78, autopilot/engine.py:61, risk/position_sizer.py:27). Threshold defaults come from config via lazy import — verify REGIME_UNCERTAINTY_ENTROPY_THRESHOLD, REGIME_UNCERTAINTY_STRESS_THRESHOLD, REGIME_UNCERTAINTY_SIZING_MAP, REGIME_UNCERTAINTY_MIN_MULTIPLIER are consistent across all 3 consumers. ShockVector has a version-locked schema (schema_version='1.0') with 13 fields including structural_features dict and transition_matrix — consumers must handle missing keys.

**Subsystem 5: Model Training & Prediction** (16 files, 6,153 lines, HIGH)
- Critical files: models/trainer.py (1,818 lines, hotspot 16/21), models/predictor.py (hotspot 12/21)
- JOB 5: 16 files (up from 14) — includes models/__init__.py, models/iv/__init__.py. Now depends on backtesting_risk (validation/preconditions.py consumed at models/trainer.py:219).
- What to look for: Scaler fit inside CV folds (no leakage), purged date-grouped CV, feature name alignment with pipeline, TRUTH_LAYER_ENFORCE_CAUSALITY
- Dependencies: features.pipeline.get_feature_type (for causality filtering), validation.preconditions (execution contract — in backtesting_risk subsystem)
- JOB 4 CRITICAL: get_feature_type() is flagged as "the single most critical correctness boundary" — if FEATURE_METADATA in pipeline.py does not include a feature name, it defaults to 'CAUSAL', allowing RESEARCH_ONLY features to leak forward-looking data into live predictions. Auditor MUST verify FEATURE_METADATA completeness against all computed features.
- JOB 4 SHARED ARTIFACTS: trained_models/ensemble_*d_*.pkl (joblib) and trained_models/ensemble_*d_meta.json are written by models/trainer.py and read by models/predictor.py + api/services/model_service.py. Meta JSON has required fields: global_features, global_feature_medians, global_target_std, regime_models. Schema change breaks EnsemblePredictor._load().

**Subsystem 6: Backtesting + Risk** (28 files, 13,132 lines, CRITICAL — LARGEST, split across 2 audit sessions if needed)
- Critical files: backtest/engine.py (2,488 lines, hotspot 16/21), risk/position_sizer.py (hotspot 10/21, "evolving"), backtest/validation.py (hotspot 11/21), validation/preconditions.py (hotspot 11/21 +3 contract bonus)
- JOB 5: 28 files (includes backtest/__init__.py, risk/__init__.py). Size exception justified: splitting was evaluated but rejected due to 5-way tight coupling at backtest/engine.py:316-320.
- What to look for: Execution realism (costs, fills, participation), risk module integration at engine.py:316-320, regime conditioning of constraints
- Dependencies: regime/shock_vector.py (schema), regime/uncertainty_gate.py (sizing modifier), validation/preconditions.py (execution contract)
- JOB 3 TEST COVERAGE WARNING: validation/ has 5 files and 0 tests. Since validation/preconditions.py enforces execution contracts for models/trainer.py and backtest/engine.py, untested validation code means the safety gates themselves are unverified.
- JOB 4 BOUNDARY DETAIL: All 5 risk classes (PositionSizer, DrawdownController, StopLossManager, PortfolioRiskManager, RiskMetrics) are lazy-imported at backtest/engine.py lines 316-320 inside _init_risk_managers(). Import failures only surface when use_risk_management=True. PositionSizer is flagged "evolving" with 21 parameters including recent uncertainty additions. backtest/engine.py:26 has 55+ config constant imports — the largest import statement in the codebase. Auditor must verify all 5 risk classes can be imported independently and that PositionSizer.size_position() parameter semantics match what the backtester passes.
- JOB 4 SHARED ARTIFACTS: results/backtest_*d_summary.json is written by run_backtest.py and consumed by api/services/backtest_service.py, api/services/results_service.py, and evaluation/engine.py. Schema has 17 fields including regime_breakdown. Key changes break API display and evaluation metrics.

**Subsystem 7: Evaluation & Diagnostics** (8 files, 2,816 lines, LOW)
- Critical files: evaluation/engine.py (red flag detection), evaluation/calibration_analysis.py
- JOB 5: 8 files (up from 7) — includes evaluation/__init__.py. Leaf module with 0 fan-in — changes here CANNOT cascade. Reads backtest_*d_summary.json artifacts (shared artifact coupling). evaluation/calibration_analysis.py lazily imports models/calibration.py.
- What to look for: Red flag thresholds, calibration analysis correctness, IC decay detection, metric statistical correctness
- Dependencies: models.calibration functions, backtest.validation functions

**Subsystem 8: Autopilot** (8 files, 4,480 lines, CRITICAL — HIGHEST COUPLING, integration audit)
- Critical files: autopilot/engine.py (1,927 lines, hotspot 17/21, primary transitive amplifier), autopilot/paper_trader.py (1,254 lines, hotspot 17/21, circular dependency hub)
- JOB 5: 8 files (up from 7) — includes autopilot/__init__.py. autopilot→api classified as "optional_depends_on" (all 6 edges are lazy/conditional). NOT a true SCC — reverse edge goes through api/jobs/autopilot_job.py:12 (job runner), not service layer.
- What to look for: All 23+ cross-module imports are correct, promotion gate thresholds are reasonable, paper trading execution matches backtest execution
- ARCHITECTURAL CONCERN: paper_trader.py imports from api.services — circular dependency (4 lazy edges at lines 173, 189, 211, 532). engine.py has 2 additional circular edges to api at lines 1868, 1911.
- JOB 3 TRANSITIVE AMPLIFIER: autopilot/engine.py appears in 12 of 15 hotspot blast radii. It channels changes from ALL upstream subsystems into the API and run_autopilot.py entry point. Any bug that propagates through autopilot/engine.py has amplified downstream impact.
- JOB 3 ISOLATION NOTE: paper_trader.py has 0 cross-module dependents (nothing imports from it). Its risk is entirely consumption-side — changes in 8 other modules can break it, but paper_trader.py changes cannot break other subsystems.
- CROSS-REFERENCE: Auditor should review api/services/health_service.py and api/ab_testing.py as part of the Autopilot audit due to the 6 circular edges, even though API (Subsystem 10) is ordered later.
- Dependencies: ALL other subsystems

**Subsystem 9: Kalshi** (16 files, 5,208 lines, MEDIUM)
- Critical files: kalshi/pipeline.py, kalshi/storage.py (DuckDB schema), kalshi/walkforward.py
- JOB 5: 16 files (includes kalshi/__init__.py, test files excluded from count). Self-contained vertical with lower blast radius. Has 8 dedicated test files in kalshi/tests/ — the best-tested module in the system.
- What to look for: Event-time join correctness, staleness policies, distribution quality
- Dependencies: features.options_factors, autopilot.promotion_gate, backtest.advanced_validation
- JOB 4 BOUNDARY DETAIL: kalshi.duckdb has 18 tables (kalshi_markets, kalshi_contracts, kalshi_quotes, kalshi_fees, macro_events, macro_events_versioned, event_outcomes, event_outcomes_first_print, event_outcomes_revised, kalshi_distributions, event_market_map_versions, kalshi_market_specs, kalshi_contract_specs, kalshi_data_provenance, kalshi_coverage_diagnostics, kalshi_ingestion_logs, kalshi_daily_health_report, kalshi_ingestion_checkpoints) with versioning and audit trail. DDL changes require migration. Kalshi reuses autopilot's PromotionGate.evaluate_event_strategy() — verify BacktestResult (from backtest/engine.py) is compatible with both standard and event-mode evaluation.

**Subsystem 10: API & Frontend** (59 files, 10,188 lines, HIGH — size exception: 59 files due to internal layering cohesion)
- Critical files: api/orchestrator.py, api/services/health_service.py (2,929 lines — LARGEST file in codebase, hotspot 14/21, "evolving")
- JOB 5: 59 files — size exception justified because all files are internal to api/ with routers→services→schemas layering. API has 115 outbound edges but only 2 modules' fan-in (lowest), meaning API is highly fragile to upstream changes but API changes have low cascade risk. Consider whether health_service.py (2,929 lines) should be split.
- What to look for: Service adapters correctly read artifacts from core modules, job system correctness, cache invalidation logic
- Dependencies: All core modules via services
- JOB 4 BOUNDARY DETAIL: API has 82 import edges to config (ALL lazy/conditional to avoid circular imports). health_service.py alone has 25+ config import sites. API is the largest consumer of models/ (13 edges) and uses lazy imports throughout. Auditor must verify ALL lazy imports resolve at runtime — failures only surface when specific API endpoints are called. HealthService is flagged "evolving" (2,929 lines) and is involved in the circular autopilot→api dependency.

**Subsystem 11: Entry Points & Scripts** (17 files, 8,883 lines, LOW)
- Critical files: run_train.py, run_backtest.py, run_autopilot.py, run_wrds_daily_refresh.py (915 lines — most complex entry point)
- JOB 5: 17 files (up from 14) — added 3 audit scripts (extract_dependencies.py, generate_interface_contracts.py, hotspot_scoring.py). Leaf consumers with 0 fan-in — changes cannot cascade. run_backtest.py writes backtest_*d_summary.json shared artifact.
- What to look for: Correct wiring of modules (do they pass the right parameters?), CLI argument handling, reproducibility manifest generation
- Note: Audit scripts are tooling for this audit workflow and do not affect production.

---

## OUTPUT FORMAT

### SUBSYSTEM_AUDIT_MANIFEST.md

```markdown
# Subsystem Audit Manifest — quant_engine
Generated: YYYY-MM-DD

## Audit Order Summary
[Table from Step 2]

## Subsystem 1: Shared Infrastructure
[Audit brief from Step 3]

### Transition to Subsystem 2
[Transition guide from Step 4]

## Subsystem 2: Data Ingestion & Quality
[Audit brief]
...
```

---

## VERIFICATION CHECKLIST

- [ ] Audit order respects dependency DAG (no subsystem audited before its dependencies)
- [ ] Every subsystem has an audit brief with critical files, what to look for, and known risks
- [ ] Every adjacent pair has a transition guide
- [ ] Hour estimates are realistic (based on file count and complexity)
- [ ] Cross-subsystem boundary checks reference specific files and line numbers
- [ ] The autopilot↔api circular dependency is prominently flagged
