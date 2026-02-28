# Input Context — Complete Audit Pipeline
## Updated: 2026-02-28 after Job 7 completion (ALL JOBS COMPLETE)

This document provides the verified findings from all 7 jobs in the LLM audit pipeline. It serves as the authoritative reference for all accumulated ground truth before the actual subsystem audits begin.

---

## COMPLETED JOBS AND THEIR OUTPUTS

| Job | Output File | Location |
|-----|-------------|----------|
| Job 1 — Module Inventory | MODULE_INVENTORY.yaml | docs/audit/data/ |
| Job 2 — Dependency Extraction | DEPENDENCY_EDGES.json + DEPENDENCY_MATRIX.md | docs/audit/data/ |
| Job 3 — Hotspot Scoring | HOTSPOT_LIST.md | docs/audit/data/ |
| Job 4 — Interface Boundaries | INTERFACE_CONTRACTS.yaml | docs/audit/data/ |
| Job 5 — Subsystem Clustering | SUBSYSTEM_MAP.json | docs/audit/data/ |
| Job 6 — Audit Ordering | SUBSYSTEM_AUDIT_MANIFEST.md | docs/audit/data/ |
| Job 7 — Verification Pass | VERIFICATION_REPORT.md | docs/audit/data/ |

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
13. **File-level hotspot rankings** — autopilot/engine.py & paper_trader.py (17), config.py (16, corrected from 18 by Job 7), features/pipeline.py & backtest/engine.py & models/trainer.py (16)

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
26. **Total codebase: 75,027 lines** — corrected from 75,096 by Job 7 (69-line overcount in data_ingestion_quality).
27. **File counts per subsystem**:
    - shared_infrastructure: 7 (2,151 lines, CRITICAL)
    - data_ingestion_quality: 19 (9,044 lines, HIGH) — corrected from 9,106 by Job 7
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

## VERIFIED FINDINGS FROM JOB 6 (Audit Ordering)

These are facts verified by the Job 6 LLM against SUBSYSTEM_MAP.json, DEPENDENCY_EDGES.json, INTERFACE_CONTRACTS.yaml, and HOTSPOT_LIST.md:

35. **Dependency DAG verified as proper DAG with zero cycles** — 11 nodes, 31 hard dependency edges, 3 optional/lazy edges (features→regime, data→kalshi, autopilot→api).
36. **Audit order topological sort uses tiebreaking** — when multiple subsystems share the same topological depth, tiebreak by (1) module-level risk score from HOTSPOT_LIST.md (higher risk first), then (2) file count (fewer files first).
37. **Total estimated audit hours: 62-83** across all 11 subsystems. Individual estimates: Shared Infrastructure 2-3, Data 6-8, Features 6-8, Regime 5-6, Backtesting+Risk 10-14, Models 5-7, Evaluation 3-4, Autopilot 4-6, Kalshi 5-7, API 12-16, Entry Points 3-4.
38. **11 audit briefs produced** — each with structured YAML: files_to_read (critical with read_first ordering + supporting), what_to_look_for (by category), known_risks (with severity and evidence), dependencies_to_verify (with boundary contract IDs), outputs_to_verify (with consumer lists).
39. **10 transition guides produced** — for every adjacent pair (1→2, 2→3, 3→4, 4→5, 5→6, 6→7, 7→8, 8→9, 9→10, 10→11). Each has carry_forward context, boundary_checks (with boundary IDs from INTERFACE_CONTRACTS.yaml), and common_failure_modes.
40. **29 boundary IDs from INTERFACE_CONTRACTS.yaml referenced** across the manifest: data_to_config_12, features_to_data_14, features_to_config_13, features_to_regime_15, regime_to_config_16, backtest_to_regime_shock_1, backtest_to_regime_uncertainty_2, backtest_to_config_19, models_to_validation_18, models_to_features_6, models_to_config_17, evaluation_to_models_backtest_11, autopilot_to_multi_4, autopilot_to_config_25, autopilot_to_data_26, autopilot_to_features_27, autopilot_to_regime_29, autopilot_to_backtest_31, autopilot_to_models_28, kalshi_to_autopilot_9, kalshi_to_config_34, kalshi_to_features_35, kalshi_to_backtest_33, api_to_config_36, api_to_autopilot_43, api_to_kalshi_44, autopilot_to_api_circular_5, api_to_data_37, api_to_models_40.
41. **9 shared artifacts formally tracked in manifest**: config_data/universe.yaml, data/cache/*.parquet, data/cache/*.meta.json, trained_models/ensemble_*d_*.pkl, trained_models/ensemble_*d_meta.json, results/backtest_*d_summary.json, results/autopilot/strategy_registry.json, results/autopilot/paper_state.json, data/kalshi.duckdb.
42. **2 new artifacts formally tracked**: results/autopilot/paper_state.json (Subsystem 8 output, consumed by api/services/autopilot_service.py) and config_data/universe.yaml (Subsystem 1 output, consumed by data_ingestion_quality, backtesting_risk, autopilot, api_frontend). Previously mentioned in earlier jobs but now formally listed as output artifacts with consumers.
43. **Autopilot (Subsystem 8) has cross_reference_during_audit section** — pointing to api/services/health_service.py and api/ab_testing.py for circular dependency review during the autopilot audit despite API being ordered later.
44. **Autopilot dependencies precisely listed as 6 subsystems**: 1 (Shared Infrastructure), 2 (Data), 3 (Features), 4 (Regime), 5 (Backtesting+Risk), 6 (Models), with optional:10 (API).
45. **Kalshi dependencies: 1, 3, 5, 8** — does NOT depend directly on data_ingestion_quality (2), regime_detection (4), model_training_prediction (6), or evaluation_diagnostics (7).
46. **Verification checklist has 14 items (all checked)** — confirming: DAG correctness, all briefs present, all transitions present, realistic hours, boundary checks reference specific files/lines, circular dependency flagged, optional dependencies documented, size exceptions justified, contract files assigned, shared artifacts documented, evolving stability flags reflected, refactoring recommendations noted, test coverage blind spots flagged, file counts match SUBSYSTEM_MAP.json.
47. **Every audit brief specifies read_first ordering** — critical files marked `read_first: true/false` to guide auditor reading sequence.
48. **3 evolving stability flags in audit briefs** — PositionSizer (Subsystem 5), HealthService (Subsystem 10), UncertaintyGate (Subsystem 4).

---

## CORRECTIONS FROM JOB 7 (Verification Pass)

Job 7 performed 6 verification checks (all PASS) and found 3 minor corrections:

49. **config.py hotspot score corrected from 18 to 16** — arithmetic error in original scoring. Fan-in(3×3=9) + Contract(2×1=2) + LOC(1×3=3) + CX(1×0=0) + Lazy(2×0=0) + Artifact(2×1=2) = 16, not 18. This changes the file-level ranking: autopilot/engine.py & paper_trader.py (17) are now scored higher than config.py (16). Rankings unaffected at module level.
50. **data_ingestion_quality total_lines corrected from 9,106 to 9,044** — 62-line overcount in SUBSYSTEM_MAP.json. Total codebase corrected from 75,096 to 75,027 (69-line overcount total; 1-3 line differences in 5 other subsystems from trailing newline methodology). Does not affect file assignments or audit ordering.
51. **Manifest verification checklist count corrected from 15 to 14** — minor counting error in spec. All 14 items are present and accurate.

**Job 7 verification results:**
- File completeness: 208/208 correct, 0 missing/extra/duplicates
- Dependency edges: 20/20 spot-checked correct, all 6 circular edges verified with exact line numbers
- Hotspot scores: 4/5 exact match, 1 corrected (config.py 18→16, ranking unaffected)
- Boundary completeness: 29/29 boundary IDs verified against INTERFACE_CONTRACTS.yaml
- Audit order: 0 hard dependency violations, 3 optional dependencies correctly handled
- Interface contracts: 7/7 highest-risk boundaries match actual source code signatures
- Ground truth: 44/45 items confirmed exactly, 1 minor counting error (item 43: 15→14)

---

## ALL JOBS COMPLETE — PIPELINE STATUS

```
Job 1 (Module Inventory)      ✓ COMPLETE — MODULE_INVENTORY.yaml
Job 2 (Dependency Extraction) ✓ COMPLETE — DEPENDENCY_EDGES.json + DEPENDENCY_MATRIX.md
Job 3 (Hotspot Scoring)       ✓ COMPLETE — HOTSPOT_LIST.md (config.py score corrected to 16)
Job 4 (Interface Boundaries)  ✓ COMPLETE — INTERFACE_CONTRACTS.yaml
Job 5 (Subsystem Clustering)  ✓ COMPLETE — SUBSYSTEM_MAP.json (line counts corrected)
Job 6 (Audit Ordering)        ✓ COMPLETE — SUBSYSTEM_AUDIT_MANIFEST.md (line counts corrected)
Job 7 (Verification Pass)     ✓ COMPLETE — VERIFICATION_REPORT.md (ALL 6 CHECKS PASS)
```

**Next step**: Begin the actual subsystem audits using SUBSYSTEM_AUDIT_MANIFEST.md as the guide. Audit in the order specified (Subsystem 1 → 11), using each subsystem's audit brief and transition guides.
