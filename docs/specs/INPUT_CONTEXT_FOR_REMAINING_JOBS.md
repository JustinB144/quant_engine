# Input Context for Remaining Jobs (6, 7)
## Updated: 2026-02-28 after Job 5 completion

This document provides the verified findings from Jobs 1-5 that should be given to the LLM as input context when running Jobs 6 and 7. Copy the relevant section into your prompt along with the job spec file.

---

## COMPLETED JOBS AND THEIR OUTPUTS

| Job | Output File | Location |
|-----|-------------|----------|
| Job 1 — Module Inventory | MODULE_INVENTORY.yaml | docs/audit/ |
| Job 2 — Dependency Extraction | DEPENDENCY_EDGES.json + DEPENDENCY_MATRIX.md | docs/audit/ |
| Job 3 — Hotspot Scoring | HOTSPOT_LIST.md | docs/audit/ |
| Job 4 — Interface Boundaries | INTERFACE_CONTRACTS.yaml | docs/audit/ |
| Job 5 — Subsystem Clustering | SUBSYSTEM_MAP.json | docs/audit/ |

---

## VERIFIED FINDINGS FROM JOB 2 (Dependency Extraction)

These are facts verified against actual source code by the Job 2 LLM:

1. **config.py has exactly 161 fan-in edges from ALL 14 modules** — it is the supreme hub, imported by every single module
2. **autopilot module has exactly 39 fan-out edges across 8 modules** — highest fan-out of any package
3. **evaluation and utils have exactly 0 fan-in edges** — they are pure leaf consumers (nothing imports from them)
4. **indicators module has exactly 6 inbound edges, all from features/** — exclusively consumed by features/pipeline.py
5. **There are exactly 6 circular autopilot→api edges** — 4 from paper_trader.py (lines 173, 189, 211, 532), 2 from engine.py (lines 1868, 1911). All are lazy imports targeting api/services/
6. **Total cross-module edges: 308** (total including same-module: 570)

---

## VERIFIED FINDINGS FROM JOB 3 (Hotspot Scoring)

These are facts verified against actual source code by the Job 3 LLM:

7. **config.py accounts for 52% of ALL cross-module edges** (161/308)
8. **autopilot/engine.py is the primary transitive amplifier** — it appears in 12 of 15 hotspot blast radii
9. **indicators/indicators.py has extreme transitive amplification** — 1 direct dependent (features/pipeline.py) but 9 transitive dependents through the feature→model→backtest→evaluation chain
10. **autopilot/paper_trader.py has 0 cross-module dependents** — nothing imports from it. Its risk is entirely consumption-side.
11. **Test coverage blind spots** — indicators/ (7 files, 0 tests), validation/ (5 files, 0 tests), regime/ (12 files, 0-1 tests)
12. **Module-level hotspot rankings** — features (15/18), autopilot (14/18), data/models/api tied (13/18)
13. **File-level hotspot rankings** — config.py (18), autopilot/engine.py & paper_trader.py (17), features/pipeline.py & backtest/engine.py & models/trainer.py (16)

---

## VERIFIED FINDINGS FROM JOB 4 (Interface Boundaries)

These are facts verified against actual source code by the Job 4 LLM:

14. **45 total boundaries documented** — 19 HIGH risk, 17 MEDIUM risk, 9 LOW risk, across 58 unique module pairs
15. **API → config has 82 import edges** — the largest cross-module dependency. ALL are lazy/conditional. api/services/health_service.py alone has 25+ config import sites.
16. **Largest import statements in the codebase**:
    - backtest/engine.py:26 imports 55+ constants from config (LARGEST)
    - autopilot/paper_trader.py:13 imports 40+ EXEC_*/PAPER_* constants (2nd largest)
    - autopilot/promotion_gate.py:13 imports 29+ PROMOTION_* constants
17. **PositionSizer (risk/position_sizer.py) is "evolving"** — size_position() has 21 parameters with recent uncertainty additions
18. **get_feature_type() is "the single most critical correctness boundary"** — FEATURE_METADATA in features/pipeline.py defaults to 'CAUSAL' for unknown feature names, enabling silent forward-looking data leakage into live predictions
19. **UncertaintyGate is imported by 3 distinct modules** — backtest/engine.py:78, autopilot/engine.py:61, risk/position_sizer.py:27. Threshold defaults come from config via lazy import. Config constants: REGIME_UNCERTAINTY_ENTROPY_THRESHOLD, REGIME_UNCERTAINTY_STRESS_THRESHOLD, REGIME_UNCERTAINTY_SIZING_MAP, REGIME_UNCERTAINTY_MIN_MULTIPLIER
20. **6 shared artifacts create file-based coupling not visible in import analysis**:
    - trained_models/ensemble_*d_*.pkl (joblib) — writer: models/trainer.py, readers: models/predictor.py, api/services/model_service.py
    - trained_models/ensemble_*d_meta.json — writer: models/trainer.py, readers: models/predictor.py, api/services/model_service.py. Required fields: global_features, global_feature_medians, global_target_std, regime_models
    - results/autopilot/strategy_registry.json — writer: autopilot/registry.py, readers: autopilot/registry.py, api/services/autopilot_service.py
    - data/cache/*.parquet — writer: data/local_cache.py, readers: data/loader.py, features/pipeline.py. Schema: Open, High, Low, Close, Volume, date
    - results/backtest_*d_summary.json — writer: run_backtest.py, readers: api/services/backtest_service.py, api/services/results_service.py, evaluation/engine.py. 17 fields including regime_breakdown
    - data/kalshi.duckdb — writer: kalshi/storage.py, readers: kalshi/pipeline.py, kalshi/events.py, api/services/kalshi_service.py. 18 tables with versioning and audit trail
21. **5 advanced indicator analyzers conditionally imported** — SpectralAnalyzer (pipeline.py:769), SSADecomposer (789), TailRiskAnalyzer (809), OptimalTransportAnalyzer (836), EigenvalueAnalyzer (1337). Each has compute_all() returning a dict whose keys become feature column names.
22. **enforce_preconditions() consumed by exactly 2 files** — models/trainer.py:219 (lazy) and backtest/engine.py:198 (lazy), both gated by TRUTH_LAYER_STRICT_PRECONDITIONS
23. **HealthService is "evolving"** — 2,929 lines (largest file in codebase), imported at 5 locations across autopilot/paper_trader.py and autopilot/engine.py. Most fragile component in the circular dependency.
24. **ShockVector has 13 fields** including structural_features (Dict[str,float]) and transition_matrix (Optional[np.ndarray]), version-locked with schema_version='1.0'. to_dict/from_dict contract must remain stable.

---

## VERIFIED FINDINGS FROM JOB 5 (Subsystem Clustering)

These are facts verified by the Job 5 LLM against DEPENDENCY_EDGES.json, INTERFACE_CONTRACTS.yaml, and HOTSPOT_LIST.md:

25. **208 total files across 11 subsystems** (not 193 — __init__.py files explicitly assigned). 0 unassigned, 0 duplicates.
26. **Total codebase: 75,096 lines** — first verified line count sum.
27. **File counts per subsystem**:
    - shared_infrastructure: 7 (2,151 lines, CRITICAL)
    - data_ingestion_quality: 19 (9,106 lines, HIGH)
    - feature_engineering: 17 (8,559 lines, CRITICAL)
    - regime_detection: 13 (4,420 lines, HIGH)
    - backtesting_risk: 28 (13,132 lines, CRITICAL) — size exception: 5-way tight coupling
    - model_training_prediction: 16 (6,153 lines, HIGH)
    - evaluation_diagnostics: 8 (2,816 lines, LOW) — leaf module, 0 fan-in
    - autopilot: 8 (4,480 lines, CRITICAL)
    - kalshi: 16 (5,208 lines, MEDIUM) — 8 test files excluded from count
    - api_frontend: 59 (10,188 lines, HIGH) — size exception: internal layering cohesion
    - entry_points_scripts: 17 (8,883 lines, LOW) — leaf consumers, 0 fan-in
28. **Optional dependencies** (not hard dependencies):
    - features→regime_detection: features/pipeline.py:1303 lazily imports regime/correlation.py (try/except)
    - data→kalshi: data/provider_registry.py:23 lazily imports kalshi/provider.py (conditional factory)
    - autopilot→api_frontend: 6 lazy/conditional circular edges (NOT a true SCC — reverse edge goes through api/jobs/autopilot_job.py:12 job runner, not service layer)
29. **5 cross-subsystem contract files**:
    - regime/shock_vector.py → assigned to regime_detection, consumed by backtesting_risk
    - regime/uncertainty_gate.py → assigned to regime_detection, consumed by backtesting_risk + autopilot (stability: "evolving")
    - regime/correlation.py → assigned to regime_detection, consumed by feature_engineering (lazy)
    - validation/preconditions.py → assigned to backtesting_risk, consumed by model_training_prediction
    - validation/data_integrity.py → assigned to data_ingestion_quality (bilateral merge from validation/)
30. **API has 115 outbound edges but only 2 modules' fan-in** — lowest fan-in despite highest fan-out. Highly fragile to upstream changes but low cascade risk.
31. **Dependency chain corrections from spec**:
    - model_training_prediction depends on backtesting_risk (validation/preconditions.py is consumed by trainer.py:219)
    - backtesting_risk depends on shared_infrastructure + regime_detection (NOT feature_engineering)
32. **data/cache/*.meta.json** — additional shared artifact (metadata sidecars for cache staleness detection) not in original spec
33. **Entry points grew to 17 files** — 3 new audit scripts (extract_dependencies.py, generate_interface_contracts.py, hotspot_scoring.py) — tooling, not production
34. **Refactoring recommendation**: extract health risk feedback interface that autopilot depends on, rather than importing from api service layer directly. Flag this in audit manifest as recommended follow-up.

---

## JOB-SPECIFIC CONTEXT

### For Job 6 (Audit Ordering)

**Required inputs**: SUBSYSTEM_MAP.json (Job 5), DEPENDENCY_EDGES.json (Job 2), INTERFACE_CONTRACTS.yaml (Job 4), plus the verified findings above.

**Key things the LLM must account for**:
- **Use the ACTUAL file counts from SUBSYSTEM_MAP.json** (208 total), not the original spec estimates (193)
- Subsystem 1 audit brief: verify all 55+ constants at backtest/engine.py:26, all 40+ at paper_trader.py:13, all 29+ at promotion_gate.py:13, and all 82 API lazy config imports. Also note config_data/universe.yaml as shared artifact.
- Subsystem 2 audit brief: 19 files now includes 3 validation/ files. Note data/cache/*.meta.json sidecars. Optional dependency on kalshi (conditional factory).
- Subsystem 3 audit brief: verify 5 conditional indicator analyzers handle ImportError, verify compute_all() dict key stability. Optional dependency on regime_detection (lazy). 92 concrete indicator subclasses with 0 tests.
- Subsystem 4 audit brief: verify UncertaintyGate config constants consistent across 3 consumers (stability: "evolving"), verify ShockVector 13-field schema. Note shock_vector.py and uncertainty_gate.py are cross-subsystem contract files.
- Subsystem 5 audit brief: 16 files now includes models/iv/__init__.py. Depends on backtesting_risk (preconditions). verify get_feature_type FEATURE_METADATA completeness (critical data leakage risk), verify model artifact schemas.
- Subsystem 6 audit brief: 28 files, size exception justified. verify all 5 lazy risk class imports at lines 316-320, verify PositionSizer evolving params, verify backtest summary JSON 17-field schema.
- Subsystem 7: LOW priority leaf module, 0 fan-in. 8 files. Reads backtest_*d_summary.json artifacts.
- Subsystem 8 audit brief: 8 files. engine.py as primary transitive amplifier (12/15 blast radii). autopilot→api is optional_depends_on, NOT true SCC. Refactoring recommendation: extract health risk feedback interface.
- Subsystem 9 audit brief: 16 files. verify 18 DuckDB tables, verify event-mode BacktestResult compatibility. Has 8 dedicated test files (best-tested module).
- Subsystem 10 audit brief: 59 files, size exception justified. verify all 82 lazy config imports resolve, note HealthService as evolving/fragile (2,929 lines). Consider splitting health_service.py. 115 outbound edges but only 2 fan-in.
- Subsystem 11: 17 files, LOW priority leaf consumers. 3 audit scripts are tooling, not production.

**Give the LLM**: The Job 6 spec (docs/specs/JOB6_AUDIT_ORDERING.md) + this context section + SUBSYSTEM_MAP.json + DEPENDENCY_EDGES.json + INTERFACE_CONTRACTS.yaml.

---

### For Job 7 (Verification Pass)

**Required inputs**: ALL outputs from Jobs 1-6 + full repo access + the verified findings above.

**Key things the LLM must account for**:
- **31 ground truth checkpoints** (items 1-6 from Job 2, items 7-13 from Job 3, items 14-24 from Job 4, items 25-34 from Job 5). Every single one must be verified against the actual Job outputs.
- **File completeness**: SUBSYSTEM_MAP.json should contain exactly 208 files, matching the actual .py count on disk.
- For Verification 6 (Interface Contract Accuracy), prioritize these 7 boundaries by ID: backtest_to_regime_shock_1, backtest_to_risk_3, autopilot_to_multi_4, features_to_indicators_8, validation_to_config_7, models_to_features_6, autopilot_to_api_circular_5
- Check that SUBSYSTEM_MAP.json accounts for ALL 6 shared artifact coupling edges (plus data/cache/*.meta.json)
- Check that all "evolving" stability flags (PositionSizer, HealthService, UncertaintyGate) are reflected in both SUBSYSTEM_MAP.json and the audit manifest
- Verify optional dependencies (features→regime, data→kalshi, autopilot→api) are correctly classified and do NOT block audit order
- Verify the 5 cross-subsystem contract files are assigned to their defining home with cross-references to consumers
- Verify the 2 size bound exceptions (backtesting_risk: 28, api_frontend: 59) have documented justification
- Verify dependency chain: model_training_prediction depends on backtesting_risk (not vice versa)

**Give the LLM**: The Job 7 spec (docs/specs/JOB7_VERIFICATION_PASS.md) + this context section + ALL Job output files (MODULE_INVENTORY.yaml, DEPENDENCY_EDGES.json, DEPENDENCY_MATRIX.md, HOTSPOT_LIST.md, INTERFACE_CONTRACTS.yaml, SUBSYSTEM_MAP.json, SUBSYSTEM_AUDIT_MANIFEST.md).

---

## EXECUTION ORDER

```
Job 6 (Audit Ordering)        ←  depends on Jobs 2, 4, 5  [NEXT]
    ↓
Job 7 (Verification Pass)     ←  depends on ALL Jobs 1-6
```

Each job should be run in a FRESH LLM session. Give the LLM:
1. The job spec file
2. The relevant section from this context document
3. The required Job output files (listed in each spec's "Input required" field)
4. Full repo access

Do NOT run Job 7 until Job 6 is complete (Job 7 needs SUBSYSTEM_AUDIT_MANIFEST.md).
