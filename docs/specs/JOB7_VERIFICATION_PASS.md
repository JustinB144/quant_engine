# JOB 7: VERIFICATION PASS
## LLM Audit Workflow — Step 7 of 7

**Purpose**: A fresh LLM reviews all outputs from Jobs 1-6, cross-checks them against actual source code, and produces a verification report documenting any errors, inconsistencies, or gaps found.

**Estimated effort**: 1 LLM session, ~45 minutes
**Input required**: All outputs from Jobs 1-6 + full repo access
**Output**: `VERIFICATION_REPORT.md` saved to `docs/audit/`
**Depends on**: Jobs 1-6 (all must be complete)
**Feeds into**: Corrections to any Job output, then the actual subsystem audits

---

## WHY THIS JOB EXISTS

Jobs 1-6 were each run by a separate LLM session. Each session had limited context and could have:
- Missed imports (especially lazy ones deep in function bodies)
- Miscounted files or lines
- Assigned a file to the wrong subsystem
- Missed a cross-subsystem boundary
- Scored a hotspot incorrectly

This job catches those errors before the actual audit begins.

---

## EXACT INSTRUCTIONS

### Verification 1: File Completeness

**Check**: Every .py file in the repo appears in exactly one subsystem in SUBSYSTEM_MAP.json

**How**:
1. Run: `find /path/to/quant_engine -name "*.py" -not -path "*__pycache__*" -not -path "*.egg-info*" -not -path "*/node_modules/*" -not -path "*/.venv/*" -not -path "*/tests/*" -not -path "*/.claude/*" | sort`
2. For each file, check it exists in SUBSYSTEM_MAP.json
3. Report any files missing from the map
4. Report any files in the map that don't exist on disk

**Expected**: 208 .py files (including __init__.py, excluding tests/), each in exactly one subsystem (Job 5 verified: 208 files, 0 unassigned, 0 duplicates)

### Verification 2: Dependency Edge Accuracy

**Check**: Spot-check 20 randomly selected edges from DEPENDENCY_EDGES.json against actual source code

**How**:
For each selected edge:
1. Open the source_file at source_line
2. Verify the import_statement matches what's actually there
3. Verify the symbols_imported list is correct
4. Verify the import_type (top_level vs lazy) is correct
5. Verify the target_file/target_module are correct

**Also check**:
- Pick 5 files NOT in the edge list and scan them for cross-module imports that should be there
- Focus on files with known lazy imports: autopilot/engine.py, backtest/engine.py, features/pipeline.py

**Report format**:
```yaml
edge_verification:
  total_checked: 20
  correct: <count>
  errors:
    - edge_id: <from JSON>
      expected: <what the JSON says>
      actual: <what the source code says>
      fix: <what should be corrected>
  missing_edges:
    - source_file: <file>
      line: <line number>
      import: <import statement found>
      reason_missed: <why Job 2 didn't catch this>
```

### Verification 3: Hotspot Score Accuracy

**Check**: Re-score the top 5 hotspots independently and compare to HOTSPOT_LIST.md

**How**:
For each of the top 5 files in HOTSPOT_LIST.md:
1. Count actual fan-in (grep for files that import from this file)
2. Count actual lines (wc -l)
3. Check git log for recent changes (git log --oneline --since=60.days -- FILE)
4. Count if/elif/try/except blocks for complexity
5. Compare your scores to the hotspot list scores

**Report format**:
```yaml
hotspot_verification:
  file: config.py
  hotspot_list_score: <from Job 3>
  reverified_score: <your independent calculation>
  discrepancies:
    - criterion: fan_in
      listed: 13
      actual: <your count>
      source: <how you counted>
```

### Verification 4: Subsystem Boundary Completeness

**Check**: For every pair of subsystems that share dependencies, verify the cross-subsystem boundary is documented in SUBSYSTEM_AUDIT_MANIFEST.md

**How**:
1. From DEPENDENCY_EDGES.json, find all edges where source_module and target_module are in DIFFERENT subsystems
2. For each such edge, verify a boundary check exists in the manifest
3. Report any missing boundary documentation

**Critical boundaries to verify exist**:
- Subsystem 4 (Regime) → Subsystem 6 (Backtest+Risk): shock_vector.py, uncertainty_gate.py
- Subsystem 3 (Features) → Subsystem 6 (Models): get_feature_type
- Subsystem 6 (Backtest+Risk) → Subsystem 4 (Regime): shock_vector import
- Subsystem 8 (Autopilot) → Subsystem 10 (API): paper_trader→health_service
- Subsystem 9 (Kalshi) → Subsystem 8 (Autopilot): promotion_gate imports
- Subsystem 2 (Data) → Subsystem 9 (Kalshi): provider_registry→kalshi.provider

### Verification 5: Audit Order Correctness

**Check**: The audit order in SUBSYSTEM_AUDIT_MANIFEST.md respects the dependency DAG

**How**:
For each subsystem in the order:
1. Check its depends_on list in SUBSYSTEM_MAP.json
2. Verify every dependency appears EARLIER in the audit order
3. If any dependency appears LATER, flag as an ordering violation

**Known pitfall to check**: Subsystem 8 (Autopilot) depends on Subsystem 10 (API) via paper_trader.py→api.services. But API is ordered AFTER autopilot. Verify whether this is correctly handled (it should be, because the dependency is lazy/optional and can be audited as a boundary note rather than a hard prerequisite).

### Verification 6: Interface Contract Accuracy

**Check**: For the 5 highest-risk boundaries in INTERFACE_CONTRACTS.yaml, read the actual source code and verify the documented signatures match

**How**:
For each boundary:
1. Open the provider file at the documented location
2. Read the actual function/class signature
3. Compare to what's documented in the YAML
4. Check parameter types, return types, and method signatures

**Boundaries to prioritize** (boundary IDs from INTERFACE_CONTRACTS.yaml):
1. backtest_to_regime_shock_1: backtest/engine.py → regime/shock_vector.py (ShockVector dataclass — 13 fields, version-locked schema_version='1.0')
2. backtest_to_risk_3: backtest/engine.py → risk/position_sizer.py (PositionSizer — "evolving", 21-param size_position method)
3. autopilot_to_multi_4: autopilot/engine.py → 8 modules (23+ symbols, primary transitive amplifier)
4. features_to_indicators_8: features/pipeline.py → indicators/ (90+ classes top-level + 5 conditional analyzers)
5. validation_to_config_7: validation/preconditions.py → config + config_structured (enforce_preconditions — consumed by 2 files)
6. models_to_features_6: models/predictor.py → features/pipeline.py (get_feature_type — "single most critical correctness boundary")
7. autopilot_to_api_circular_5: autopilot/paper_trader.py → api/services (4 conditional imports — circular dependency)

---

## OUTPUT FORMAT

### VERIFICATION_REPORT.md

```markdown
# Verification Report — quant_engine Audit Artifacts
Generated: YYYY-MM-DD

## Summary
| Check | Status | Issues Found |
|-------|--------|-------------|
| File Completeness | PASS/FAIL | <count> missing, <count> extra |
| Dependency Edges | PASS/FAIL | <count>/<count> correct, <count> errors, <count> missing |
| Hotspot Scores | PASS/FAIL | <count> discrepancies |
| Boundary Completeness | PASS/FAIL | <count> undocumented boundaries |
| Audit Order | PASS/FAIL | <count> ordering violations |
| Interface Contracts | PASS/FAIL | <count> signature mismatches |

## Overall Assessment
<PASS if all checks pass, FAIL if any check fails>
<If FAIL, list which Job outputs need correction>

## Detailed Findings

### 1. File Completeness
[details]

### 2. Dependency Edge Accuracy
[details with specific errors]

### 3. Hotspot Score Accuracy
[details with score comparisons]

### 4. Subsystem Boundary Completeness
[details with missing boundaries]

### 5. Audit Order Correctness
[details with any violations]

### 6. Interface Contract Accuracy
[details with signature mismatches]

## Corrections Required
[Prioritized list of corrections to make to Job 1-6 outputs before proceeding with audit]
```

---

## GROUND TRUTH FROM JOB 2 (verified, use as checkpoints)

These numbers were verified by the completed Job 2 dependency extraction. The verification LLM should confirm these appear correctly in all downstream artifacts:

1. **config.py has exactly 161 fan-in edges from ALL 14 modules** — if HOTSPOT_LIST.md or DEPENDENCY_MATRIX.md shows a different number, that's a bug
2. **autopilot module has exactly 39 fan-out edges across 8 modules** — highest fan-out of any package
3. **evaluation and utils have exactly 0 fan-in edges** — they are pure leaf consumers
4. **indicators module has exactly 6 inbound edges, all from features/** — exclusively consumed by features/pipeline.py
5. **There are exactly 6 circular autopilot→api edges** — 4 from paper_trader.py (lines 173, 189, 211, 532), 2 from engine.py (lines 1868, 1911). All are lazy imports targeting api/services/

## GROUND TRUTH FROM JOB 3 (verified, use as checkpoints)

These numbers were verified by the completed Job 3 hotspot scoring. The verification LLM should confirm these appear correctly in all downstream artifacts:

6. **config.py accounts for 52% of ALL cross-module edges (161/308)** — if the total edge count or percentage differs, investigate
7. **autopilot/engine.py is the primary transitive amplifier** — it appears in 12 of 15 hotspot blast radii. Verify this by checking which hotspot files list autopilot/engine.py as a transitive dependent.
8. **indicators/indicators.py has extreme transitive amplification** — 1 direct dependent (features/pipeline.py) but 9 transitive dependents. A signature change here propagates through the feature→model→backtest→evaluation chain.
9. **autopilot/paper_trader.py has 0 cross-module dependents** — nothing imports from it. Its 4 circular api edges are consumption-side risk only.
10. **Test coverage blind spots** — the following modules have 0 test files: indicators/ (7 files), validation/ (5 files), regime/ (12 files have only 0-1 test files). Verify the SUBSYSTEM_AUDIT_MANIFEST flags these as requiring manual verification during audit.
11. **Module-level hotspot rankings** — features scored highest (15/18), followed by autopilot (14/18), then data/models/api tied at 13/18. File-level: config.py (18), autopilot/engine.py & paper_trader.py (17), features/pipeline.py & backtest/engine.py & models/trainer.py (16).

## GROUND TRUTH FROM JOB 4 (verified, use as checkpoints)

These numbers were verified by the completed Job 4 interface boundary analysis. The verification LLM should confirm these appear correctly in all downstream artifacts:

12. **INTERFACE_CONTRACTS.yaml documents exactly 45 boundaries** — 19 HIGH risk, 17 MEDIUM risk, 9 LOW risk, across 58 unique module pairs. If SUBSYSTEM_MAP.json references boundaries not in this file, or vice versa, that's a gap.
13. **API → config has exactly 82 import edges** — the largest cross-module dependency. All are lazy/conditional. api/services/health_service.py alone has 25+ config import sites. Verify this is reflected in the subsystem map and audit briefs.
14. **backtest/engine.py:26 imports 55+ constants from config** — the largest single import statement in the codebase. autopilot/paper_trader.py:13 imports 40+ constants (second largest). autopilot/promotion_gate.py:13 imports 29+ PROMOTION_* constants.
15. **PositionSizer (risk/position_sizer.py) is flagged "evolving"** — size_position() has 21 parameters with recent uncertainty additions. Verify SUBSYSTEM_AUDIT_MANIFEST flags this as requiring careful review.
16. **get_feature_type() is "the single most critical correctness boundary"** — FEATURE_METADATA default to 'CAUSAL' for unknown feature names enables silent data leakage. Verify the audit manifest prioritizes FEATURE_METADATA completeness verification.
17. **UncertaintyGate is imported by exactly 3 modules** — backtest/engine.py:78, autopilot/engine.py:61, risk/position_sizer.py:27. Threshold defaults come from config via lazy import. Verify all 3 consumers reference consistent config constants.
18. **6 shared artifacts create file-based coupling** — trained_models/*.pkl, trained_models/*_meta.json, strategy_registry.json, data/cache/*.parquet, backtest_*d_summary.json, kalshi.duckdb (18 tables). Verify SUBSYSTEM_MAP.json accounts for artifact-based coupling edges.
19. **5 advanced indicator analyzers are conditionally imported** — SpectralAnalyzer:769, SSADecomposer:789, TailRiskAnalyzer:809, OptimalTransportAnalyzer:836, EigenvalueAnalyzer:1337. Each returns a dict whose keys become feature column names. Verify ImportError handling is documented.
20. **enforce_preconditions() is consumed by exactly 2 files** — models/trainer.py:219 (lazy) and backtest/engine.py:198 (lazy), both gated by TRUTH_LAYER_STRICT_PRECONDITIONS config flag.
21. **HealthService is flagged "evolving"** — 2,929 lines (largest file in codebase), imported at 5 locations across autopilot/paper_trader.py and autopilot/engine.py. This is the most fragile component in the circular dependency.

## GROUND TRUTH FROM JOB 5 (verified, use as checkpoints)

These numbers were verified by the completed Job 5 subsystem clustering. The verification LLM should confirm these appear correctly in all downstream artifacts:

22. **SUBSYSTEM_MAP.json contains exactly 208 files across 11 subsystems** (not 193). The increase is due to __init__.py files being explicitly assigned. Every .py file (excluding tests/, __pycache__, .egg-info) appears in exactly one subsystem with 0 unassigned and 0 duplicate assignments.
23. **Total codebase: 75,096 lines** — first verified line count sum across all subsystems.
24. **File counts per subsystem**: shared_infrastructure=7, data_ingestion_quality=19, feature_engineering=17, regime_detection=13, backtesting_risk=28, model_training_prediction=16, evaluation_diagnostics=8, autopilot=8, kalshi=16, api_frontend=59, entry_points_scripts=17. If SUBSYSTEM_AUDIT_MANIFEST uses different counts, that's a bug.
25. **2 size bound exceptions**: backtesting_risk (28 files, 3 over 25 limit — justified by 5-way tight coupling), api_frontend (59 files, 34 over 25 limit — justified by internal layering cohesion). Both must be documented with justification.
26. **Optional dependencies**: features→regime_detection (pipeline.py:1303 lazy), data→kalshi (provider_registry.py:23 conditional), autopilot→api_frontend (6 lazy/conditional edges). These are classified as "optional_depends_on" in SUBSYSTEM_MAP.json and must NOT block the audit order.
27. **Architectural cycle is NOT a true SCC** — the reverse edge api→autopilot goes through api/jobs/autopilot_job.py:12 (job runner), not the service layer. The clustering decision was to NOT merge autopilot and api_frontend.
28. **5 cross-subsystem contract files**: regime/shock_vector.py (regime→backtesting_risk), regime/uncertainty_gate.py (regime→backtesting_risk+autopilot), regime/correlation.py (regime→feature_engineering, lazy), validation/preconditions.py (backtesting_risk→model_training_prediction), validation/data_integrity.py (data_ingestion_quality, bilateral merge from validation/).
29. **API has exactly 115 outbound edges but only 2 modules' fan-in** — lowest fan-in despite highest fan-out. Verify the audit manifest reflects that API is fragile to upstream changes but has low cascade risk.
30. **Entry points subsystem grew to 17 files** — includes 3 new audit scripts (extract_dependencies.py, generate_interface_contracts.py, hotspot_scoring.py) that are tooling, not production code.
31. **data/cache/*.meta.json** — additional shared artifact (metadata sidecars) not in original spec, now documented in SUBSYSTEM_MAP.json.

---

## IMPORTANT NOTES FOR THE VERIFICATION LLM

1. **You are NOT the same LLM that produced Jobs 1-6.** Approach all outputs with healthy skepticism. Verify, don't trust.

2. **Focus on the HIGHEST RISK items first.** If time is limited, prioritize:
   - Checking that lazy imports weren't missed (these are the most commonly missed)
   - Checking that the autopilot↔api circular dependency is correctly documented with all 6 edges
   - Checking that config.py's blast radius is fully captured (161 edges, 14 modules)
   - Checking that evaluation and utils are correctly flagged as leaf modules

3. **If you find errors, be specific.** Don't say "some edges might be wrong." Say "DEPENDENCY_EDGES.json line 47 says autopilot/engine.py imports from risk.covariance at line 62 but the actual import is at line 1528 and it's lazy, not top_level."

4. **Count files explicitly.** Don't estimate. Run `wc -l` and `find | wc -l` commands.

5. **Read actual signatures.** Don't infer types from names. Open the file and read the `def` or `class` line.

6. **Cross-check ALL ground truth numbers above (Job 2: items 1-5, Job 3: items 6-11, Job 4: items 12-21, Job 5: items 22-31).** If any of the 31 ground truth items don't match what Jobs 3-6 produced, flag it immediately.

---

## VERIFICATION CHECKLIST (META)

- [ ] All 6 verification checks were performed
- [ ] At least 20 dependency edges were spot-checked
- [ ] At least 5 hotspot scores were independently recalculated
- [ ] At least 5 interface contracts were verified against source
- [ ] All critical boundaries (listed above) were checked
- [ ] The report clearly states PASS or FAIL for each check
- [ ] Any FAIL items include specific corrections
