# JOB 3: HOTSPOT SCORING
## LLM Audit Workflow — Step 3 of 7

**Purpose**: Score and rank every module and every file by risk, producing a prioritized "hotspot list" that tells auditors exactly where to focus. A hotspot is a file or module that, if it breaks or changes, causes cascading failures across the system.

**Estimated effort**: 1 LLM session, ~30 minutes
**Input required**: MODULE_INVENTORY.yaml (Job 1) + DEPENDENCY_EDGES.json (Job 2) + git log access
**Output**: `HOTSPOT_LIST.md` saved to `docs/audit/data/`
**Depends on**: Job 1, Job 2
**Feeds into**: Job 5 (Subsystem Clustering), Job 7 (Verification)

---

## WHAT YOU ARE BUILDING

A ranked list of every module and every key file, scored by risk. For each hotspot, you document: what its risk score is, WHY it's risky, what breaks if it changes (blast radius), and what an auditor should prioritize when reviewing it.

---

## EXACT INSTRUCTIONS

### Step 1: Compute module-level scores

For each of the 14 modules + root files, score these 6 criteria on a 0-3 scale:

| Criterion | 0 | 1 | 2 | 3 | How to Measure |
|-----------|---|---|---|---|----------------|
| **Fan-in** (modules that depend on it) | 0 | 1-2 | 3-5 | 6+ | Count columns in DEPENDENCY_MATRIX.md where this module has incoming edges |
| **Fan-out** (modules it depends on) | 0 | 1-2 | 3-5 | 6+ | Count rows in DEPENDENCY_MATRIX.md where this module has outgoing edges |
| **Contract surface** (shared types/schemas/files) | None | Internal only | 1-2 external consumers | 3+ external consumers | Count classes/functions from this module that appear in other modules' import statements |
| **Change frequency** (recent git activity) | 0-2 commits in 60d | 3-10 | 11-20 | 21+ | `git log --oneline --since=60.days -- MODULE/` |
| **Complexity** (size of largest file) | <200 lines | 200-500 | 500-1000 | 1000+ | From MODULE_INVENTORY.yaml |
| **Test coverage gaps** | All files tested | 1-2 untested | 3-5 untested | 6+ untested | Compare module files to test file mapping |

**Total score** = sum of all 6 criteria (max 18).

### Step 2: Compute file-level scores

For every file that has at least ONE cross-module import edge (either as source or target), score these criteria:

| Criterion | Weight | How to Measure |
|-----------|--------|----------------|
| Cross-module fan-in (files in OTHER modules that import this file) | 3x | Count from DEPENDENCY_EDGES.json |
| Is it a contract file (exports types used in other module signatures) | 2x (0 or 1) | Check if exported classes appear in other modules' function signatures |
| Lines of code | 1x (0=<200, 1=200-500, 2=500-1000, 3=1000+) | From MODULE_INVENTORY.yaml |
| Cyclomatic complexity proxy | 1x (0=<200, 1=200-500, 2=500-1000, 3=1000+) | Run: `grep -c "if \|elif \|except \|try:" FILE` |
| Has lazy/conditional cross-module imports | 2x (0 or 1) | From DEPENDENCY_EDGES.json import_type field |
| Is shared artifact writer (writes files other modules read) | 2x (0 or 1) | Check shared artifacts table below |

**Total weighted score** = sum of (value × weight).

### Step 3: Build the blast radius for top 15 files

For each of the 15 highest-scoring files, document:

```yaml
file: <path>
score: <total>
blast_radius:
  direct_dependents:
    - file: <path of file that imports from this one>
      module: <module name>
      symbols_used: [<what they import>]
  transitive_dependents:
    - file: <path of file that depends on a direct dependent>
      module: <module name>
      through: <the intermediate file>
  shared_artifacts_affected:
    - artifact: <path/pattern>
      impact: <what breaks if the artifact schema changes>
  what_breaks: >
    <Plain English description of what goes wrong if this file has a bug or breaking change>
  audit_priority: CRITICAL | HIGH | MEDIUM
  audit_focus: >
    <What specifically the auditor should look for in this file>
```

### Step 4: Identify the verified hotspots

The following files were identified as high-risk through source-verified analysis on 2026-02-27. Your scoring MUST rank these in the top 15. If your scoring doesn't, your criteria weights need adjustment.

**VERIFIED HOTSPOT #1: config.py**
- Fan-in: 14 modules (ALL modules including indicators via transitive), 161 inbound edges total (verified by Job 2)
- Lines: ~1,500
- Constants: 200+
- Risk: Any constant rename or value change silently propagates to ALL 14 modules across 161 import edges. Supreme hub of the entire system.
- Blast radius: ENTIRE SYSTEM
- Audit focus: Check for deprecated/placeholder constants still referenced, verify TRUTH_LAYER flags are consistent, identify which constants are consumed by which modules

**VERIFIED HOTSPOT #2: autopilot/engine.py (+ autopilot module as a whole)**
- Fan-out: 8 modules (backtest, models, config, data, features, regime, risk, api)
- Module-level: autopilot has 39 outbound edges across 8 modules — highest fan-out of ANY package (verified by Job 2)
- Lines: 1,927
- Cross-module imports: 39 edges total across autopilot module (engine.py accounts for ~23 of these)
- Risk: Most tightly coupled module. Changes in ANY of 8 modules can break it. 6 of the 39 edges are the circular autopilot→api dependency.
- Blast radius: Autopilot cycle, strategy discovery, paper trading, health feedback loop
- Audit focus: Verify all lazy imports are guarded, check that cross-module contracts haven't drifted, verify the 6 autopilot→api edges are truly optional (lazy) and won't cause import-time failures

**VERIFIED HOTSPOT #3: features/pipeline.py**
- Fan-out: 4 modules (indicators: 21 symbols, config: 14 constants, regime, data)
- Lines: 1,541
- Cross-module imports: 35+ (top-level: 1 block of 21, lazy: 14+)
- Risk: All 90+ features flow through this file. Indicator signature changes break everything
- Blast radius: Training, prediction, backtesting (all use features)
- Audit focus: Feature causality types (CAUSAL vs END_OF_DAY vs RESEARCH_ONLY), indicator import correctness

**VERIFIED HOTSPOT #4: backtest/engine.py**
- Fan-out: 4 modules (config, regime, validation, risk)
- Lines: 2,488 (LARGEST production file)
- Cross-module imports: 9+ (top-level: 3, lazy: 6+)
- Risk: Core simulation loop. Errors here invalidate ALL backtest results and promotion decisions
- Blast radius: Backtest results, evaluation, autopilot promotion, paper trading
- Audit focus: Execution realism (costs, fills, participation), risk module integration correctness

**VERIFIED HOTSPOT #5: autopilot/paper_trader.py**
- Fan-out: 4 modules (config, backtest, risk, api)
- Lines: 1,254
- ARCHITECTURAL CONCERN: Imports from api.services (health_service, health_risk_feedback, ab_testing) creating circular dependency
- Risk: The autopilot↔api circular reference runs through this file
- Blast radius: Paper trading, health feedback loop, A/B testing
- Audit focus: Verify api imports are truly lazy and won't cause circular import at load time

**VERIFIED HOTSPOT #6: regime/shock_vector.py**
- Fan-in: 2 files in backtest (engine.py:77, execution.py:29)
- Contract: Defines ShockVector dataclass with version-locked schema
- Risk: Schema change here breaks backtest execution layer
- Audit focus: Version lock enforcement, schema compatibility

**VERIFIED HOTSPOT #7: validation/preconditions.py**
- Fan-in: 2 modules (models/trainer.py:219, backtest/engine.py:198)
- Contract: Execution contract (RET_TYPE, LABEL_H, PX_TYPE, ENTRY_PRICE_TYPE)
- Also imports: config_structured.PreconditionsConfig (the ONLY file that uses config_structured)
- Risk: Changes to precondition checks alter both training and backtesting behavior
- Audit focus: Ensure preconditions match between training and backtesting paths

**VERIFIED HOTSPOT #8: models/predictor.py**
- Fan-out: 2 modules (config, features.pipeline)
- Lines: ~500
- Contract: Bridge between trained models and predictions
- Risk: Feature name/ordering mismatches break prediction silently
- Audit focus: Feature alignment with pipeline, model version resolution logic

**VERIFIED HOTSPOT #9: regime/uncertainty_gate.py**
- Fan-in: 3 files across 3 modules (backtest/engine.py:78, autopilot/engine.py:61, risk/position_sizer.py:27)
- Contract: Entropy-based sizing multiplier
- Risk: Threshold changes silently alter position sizing, backtest results, AND autopilot promotion
- Audit focus: Threshold values, multiplier calculation correctness

**VERIFIED HOTSPOT #10: data/loader.py**
- Fan-in: 3+ modules (autopilot, run_*.py scripts, data/__init__.py)
- Lines: ~800
- Lazy imports: validation.data_integrity at line 567
- Risk: Primary data ingestion. Quality regression corrupts everything downstream
- Audit focus: Cache trust logic, WRDS fallback behavior, survivorship filtering

**VERIFIED HOTSPOT #11: api/services/health_service.py**
- Lines: 2,929 (LARGEST file in entire codebase)
- Fan-in from outside api: autopilot/engine.py:1868, :1911, autopilot/paper_trader.py:189, :532
- Risk: Circular dependency hub (api↔autopilot)
- Audit focus: What state does this expose to autopilot? Could changes here break paper trading?

**VERIFIED HOTSPOT #12: indicators/indicators.py**
- Lines: 2,904
- Fan-in: features/pipeline.py imports 21+ symbols at line 21
- Risk: 90+ indicator implementations. Any signature change breaks feature pipeline
- Audit focus: Indicator return types/shapes, parameter defaults

---

## JOB 2 VERIFIED FINDINGS (use these to calibrate your scoring)

The following findings were produced by Job 2 dependency extraction and represent ground truth:

1. **config.py is the supreme hub**: 161 fan-in edges from ALL 14 modules. Fan-in score = 3 (max). This is not "imported by 13 modules" — it's imported by ALL of them.
2. **autopilot has the highest package-level fan-out**: 39 outbound edges across 8 modules. No other module comes close.
3. **evaluation and utils are leaf modules with 0 fan-in**: Nothing imports from them. Fan-in score = 0. These are pure consumers — changes to them cannot break other modules. Audit priority is LOWER than hub modules.
4. **indicators is exclusively imported by features**: 6 edges, all from features/pipeline.py. Confirms indicators is a private dependency of features. Fan-in score = 1 (single consumer).
5. **6 circular autopilot→api edges are confirmed**: autopilot/paper_trader.py (4 edges) and autopilot/engine.py (2 edges) import from api/services/. All are lazy imports. This is the ONLY circular architectural dependency in the system.

**Scoring implications**:
- evaluation and utils should score LOW on fan-in (0) — they are safe to change without cascading risk
- indicators should score LOW on fan-in (1) — but HIGH on contract surface because features/pipeline.py depends heavily on its 90+ indicator signatures
- autopilot should score MAX (3) on fan-out, and the 6 circular edges should add a penalty multiplier to its risk score
- config should score MAX (3) on fan-in with a note that 161 edges makes it categorically different from any other module

---

## SHARED ARTIFACTS TABLE (for scoring criterion)

These files represent file-based coupling between modules:

| Artifact | Writer | Reader(s) | Format | Schema Owner |
|----------|--------|-----------|--------|-------------|
| trained_models/*/ | models/trainer.py | models/predictor.py, api/services/model_service.py | joblib+JSON | models |
| results/backtest_*d_summary.json | backtest/engine.py | api/services/backtest_service.py, evaluation/engine.py | JSON | backtest |
| results/backtest_*d_trades.csv | backtest/engine.py | api/services/backtest_service.py, evaluation/engine.py | CSV | backtest |
| results/predictions_*d.csv | models/predictor.py, run_predict.py | api/services/results_service.py | CSV | models |
| results/autopilot/strategy_registry.json | autopilot/registry.py | api/services/autopilot_service.py | JSON | autopilot |
| results/autopilot/paper_state.json | autopilot/paper_trader.py | api/services/autopilot_service.py | JSON | autopilot |
| results/autopilot/latest_cycle.json | autopilot/engine.py | api/services/autopilot_service.py | JSON | autopilot |
| data/cache/*.parquet | data/local_cache.py | data/loader.py | Parquet | data |
| api_jobs.db | api/jobs/store.py | api/jobs/runner.py, api/routers/jobs.py | SQLite | api |
| config_data/universe.yaml | run_wrds_daily_refresh.py | config.py, risk/universe_config.py | YAML | scripts/config |
| data/kalshi.duckdb | kalshi/storage.py | kalshi/pipeline.py, kalshi/events.py + others | DuckDB | kalshi |

---

## TEST COVERAGE MAPPING (for scoring criterion)

| Module | Test Files | Approximate Coverage |
|--------|-----------|---------------------|
| regime | 12 test files | Good — detection, consensus, uncertainty, shock vectors all tested |
| risk | 15 test files | Good — position sizing, constraints, factors, drawdown all tested |
| backtest | 9 test files | Moderate — engine tested but execution realism coverage varies |
| autopilot | 7 test files | Moderate — promotion and paper trading tested, discovery less so |
| api | 9 test files | Moderate — routers/services tested, orchestrator less so |
| kalshi | 9 test files (in kalshi/tests/) | Good — leakage, distribution, walk-forward all tested |
| models | 6 test files | Moderate — trainer/predictor tested, governance/versioning less so |
| features | 5 test files | Low — pipeline is massive (1,541 lines) with only 5 test files |
| data | 10 test files | Good — quality, cache, survivorship, cross-source all tested |
| evaluation | 5 test files | Moderate — engine, slicing, fragility tested |
| validation | 0 dedicated test files | LOW — preconditions/data_integrity tested indirectly via other tests |
| config | 6 test files | Good — consistency and validation tested |
| indicators | 0 dedicated test files | LOW — tested indirectly through features tests |
| utils | 0 dedicated test files | LOW — minimal module (2 files) |

---

## OUTPUT FORMAT

### HOTSPOT_LIST.md

```markdown
# Hotspot Analysis — quant_engine

## Module Rankings (by total risk score)

| Rank | Module | Score | Fan-in | Fan-out | Contract | Changes | Complexity | Test Gaps | Key Risk |
|------|--------|-------|--------|---------|----------|---------|------------|-----------|----------|
| 1 | ... | /18 | /3 | /3 | /3 | /3 | /3 | /3 | ... |

## File Rankings (top 20 by weighted score)

| Rank | File | Score | Fan-in(3x) | Contract(2x) | LOC(1x) | Complexity(1x) | Lazy(2x) | Artifact(2x) | Key Risk |
|------|------|-------|------------|--------------|---------|----------------|----------|--------------|----------|
| 1 | config.py | ... | ... | ... | ... | ... | ... | ... | God object, 13 consumers |

## Blast Radius Analysis (top 15 files)

### 1. config.py
[blast radius details as specified above]

### 2. autopilot/engine.py
[blast radius details]
...
```

---

## VERIFICATION CHECKLIST

- [ ] All 12 verified hotspots appear in the top 15 file rankings
- [ ] Scores are based on actual data from Jobs 1-2, not estimated
- [ ] Blast radius for each top 15 file lists specific dependent files with line numbers
- [ ] Test coverage gaps are verified (not assumed)
- [ ] Shared artifact coupling is scored (files that write artifacts consumed by other modules)
- [ ] The autopilot↔api circular dependency is flagged prominently
