# JOB 4: INTERFACE BOUNDARY ANALYSIS
## LLM Audit Workflow — Step 4 of 7

**Purpose**: For every cross-module dependency, document the exact contract surface — what symbols are shared, what types flow across the boundary, what file-based artifacts are exchanged, and what would break if either side changed.

**Estimated effort**: 1 LLM session, ~60 minutes (requires reading actual function signatures)
**Input required**: Full repo access + DEPENDENCY_EDGES.json (Job 2)
**Output**: `INTERFACE_CONTRACTS.yaml` saved to `docs/audit/`
**Depends on**: Job 2 (Dependency Extraction)
**Feeds into**: Job 5 (Subsystem Clustering), Job 7 (Verification)

---

## WHAT YOU ARE BUILDING

A YAML file documenting every interface boundary between modules. For each boundary, you capture: which symbols cross it, what types those symbols are (class, function, constant, dataclass), what parameters/return types they have, and what breaks if either side changes.

This is the most labor-intensive job because you must READ ACTUAL FUNCTION SIGNATURES, not just import statements.

---

## EXACT INSTRUCTIONS

### Step 1: Identify all unique module pairs with cross-module edges

From DEPENDENCY_EDGES.json, extract every unique (source_module, target_module) pair where cross_module=true. There should be approximately 25-30 unique pairs.

### Step 2: For each module pair, document the boundary

For each (source_module → target_module) pair, read the actual source code of both files involved and produce:

```yaml
boundary:
  id: "<source>_to_<target>_<N>"
  provider_module: <target_module (the one being imported FROM)>
  consumer_module: <source_module (the one doing the importing)>

  symbols:
    - name: <imported symbol name>
      type: class | function | constant | dataclass | enum
      defined_in: <file path where it's defined>
      imported_by: <file path that imports it>
      import_line: <line number of the import>
      import_type: top_level | lazy | conditional

      # For functions:
      signature: "<full function signature including parameters and return type>"
      parameters:
        - name: <param name>
          type: <type annotation if present, else "untyped">
          required: true | false
      return_type: "<return type annotation if present>"

      # For classes:
      key_methods:
        - name: <method name>
          signature: "<method signature>"
      constructor_params:
        - name: <param name>
          type: <type annotation>

      # For constants:
      value_type: <int | float | str | dict | list | Path | bool>
      current_value: <current value if simple, "complex" if dict/list>

      # For all:
      stability: stable | evolving | deprecated
      breaking_change_impact: >
        <What breaks if this symbol is renamed, removed, or its signature changes>

  shared_artifacts:
    - path: <file path or glob pattern>
      format: json | csv | parquet | sqlite | duckdb | yaml | joblib
      schema_owner: <which module defines the schema>
      writer: <file that writes this artifact>
      readers: [<files that read this artifact>]
      schema_fields: [<key field names in the artifact>]
      breaking_change_impact: >
        <What breaks if the schema changes>

  boundary_risk: HIGH | MEDIUM | LOW
  audit_notes: >
    <Specific things an auditor should check at this boundary>
```

### Step 3: Prioritize boundaries by risk

Rank boundaries by risk. HIGH risk boundaries are those where:
- 3+ symbols cross the boundary
- The symbols include complex types (classes with state, dataclasses with many fields)
- The boundary involves shared file artifacts
- Either side has lazy/conditional imports (fragile coupling)
- Either side is in the hotspot list

**JOB 3 VERIFIED FINDINGS affecting boundary risk:**
- config.py accounts for 52% of ALL cross-module edges (161/308). Any boundary involving config is inherently HIGH risk by volume alone.
- indicators/indicators.py has extreme transitive amplification: 1 direct dependent (features/pipeline.py) but 9 transitive dependents. The features→indicators boundary is a critical bottleneck — a signature change here silently propagates through the entire feature→model→backtest→evaluation chain.
- autopilot/paper_trader.py has 0 cross-module dependents (nothing imports from it). Its 4 circular autopilot→api edges at lines 173, 189, 211, 532 are consumption-side risk only.
- autopilot/engine.py is the primary transitive amplifier for 12 of 15 hotspot blast radii. Every boundary feeding INTO autopilot has amplified risk because autopilot channels changes to the API and run_autopilot.py entry point.

### Step 4: Verify against known critical boundaries

You MUST document these boundaries (verified 2026-02-27). Read the actual files to get signatures.

**CRITICAL BOUNDARY 1: backtest/engine.py → regime/shock_vector.py**
- Symbols: compute_shock_vectors (function), ShockVector (dataclass)
- Import at line 77 of backtest/engine.py
- ShockVector has a version-locked schema — read shock_vector.py to get all fields
- Also imported by: backtest/execution.py:29 (lazy)
- Risk: HIGH — schema change breaks backtest execution

**CRITICAL BOUNDARY 2: backtest/engine.py → regime/uncertainty_gate.py**
- Symbols: UncertaintyGate (class)
- Import at line 78 of backtest/engine.py
- UncertaintyGate.compute_size_multiplier() is the key method — read its signature
- Also imported by: autopilot/engine.py:61, risk/position_sizer.py:27
- Risk: HIGH — threshold change silently alters 3 modules

**CRITICAL BOUNDARY 3: backtest/engine.py → risk/ (5 files)**
- Lazy imports at lines 316-320
- Symbols: PositionSizer, DrawdownController, StopLossManager, PortfolioRiskManager, RiskMetrics
- These are all classes with complex state
- Risk: HIGH — backtester delegates all risk decisions to these classes

**CRITICAL BOUNDARY 4: autopilot/engine.py → (8 modules)**
- 23+ symbols imported from backtest, models, data, features, regime, risk, config, api
- Read autopilot/engine.py lines 20-62 for top-level imports and lines 565-1911 for lazy imports
- Risk: CRITICAL — any change in any of 8 modules can break autopilot

**CRITICAL BOUNDARY 5: autopilot/paper_trader.py → api/services/**
- Lazy imports at lines 173, 189, 211, 532
- Symbols: create_health_risk_gate, HealthService, ABTestRegistry
- Risk: HIGH — circular dependency (api→autopilot→api)

**CRITICAL BOUNDARY 6: models/predictor.py → features/pipeline.py**
- Symbol: get_feature_type (function)
- Import at line 22 of models/predictor.py
- Used for feature causality filtering during prediction
- Risk: HIGH — feature name mismatch breaks prediction silently

**CRITICAL BOUNDARY 7: validation/preconditions.py → config + config_structured**
- Imports from config (line 16) AND config_structured (line 23)
- Only file that imports config_structured.PreconditionsConfig
- Defines enforce_preconditions() used by models/trainer.py and backtest/engine.py
- Risk: HIGH — execution contract enforcement

**CRITICAL BOUNDARY 8: features/pipeline.py → indicators/ (21+ symbols)**
- Top-level import at line 21 imports 21+ indicator classes
- Lazy imports at lines 769, 789, 809, 836, 1337 import specialized analyzers
- Risk: HIGH — indicator signature changes break feature computation

**CRITICAL BOUNDARY 9: kalshi/ → autopilot/ (promotion gate)**
- kalshi/pipeline.py:25 imports PromotionDecision
- kalshi/promotion.py:12-13 imports PromotionDecision, PromotionGate, StrategyCandidate
- Risk: MEDIUM — Kalshi reuses autopilot's promotion logic

**CRITICAL BOUNDARY 10: data/loader.py → validation/data_integrity.py**
- Lazy import at line 567
- Symbol: DataIntegrityValidator
- Risk: MEDIUM — quality gate that blocks corrupt data from entering pipeline

**CRITICAL BOUNDARY 11: evaluation/ → models/ + backtest/**
- evaluation/calibration_analysis.py:92 (lazy) → models.calibration (compute_ece, compute_reliability_curve)
- evaluation/engine.py:272 (lazy) → backtest.validation.walk_forward_with_embargo
- evaluation/engine.py:315 (lazy) → backtest.validation (rolling_ic, detect_ic_decay)
- Risk: MEDIUM — evaluation correctness depends on these functions

---

## SHARED ARTIFACTS DETAIL

For each shared artifact, you MUST read the writer file to determine the actual schema:

**trained_models/*/ (joblib + JSON)**
- Writer: models/trainer.py (save method)
- Readers: models/predictor.py (load), api/services/model_service.py
- Read models/trainer.py save method and models/predictor.py load logic to document the exact artifact structure

**results/backtest_*d_summary.json**
- Writer: backtest/engine.py (compute_metrics method)
- Readers: api/services/backtest_service.py, evaluation/engine.py
- Read the dict structure returned by compute_metrics()

**results/autopilot/strategy_registry.json**
- Writer: autopilot/registry.py
- Reader: api/services/autopilot_service.py
- Read registry.py save/load methods for schema

**data/kalshi.duckdb**
- Writer: kalshi/storage.py (EventTimeStore)
- Readers: kalshi/pipeline.py, kalshi/events.py, and others
- Read storage.py DDL statements for table schema

---

## OUTPUT FORMAT

Save as `docs/audit/INTERFACE_CONTRACTS.yaml`

The YAML file should contain:
```yaml
metadata:
  generated: "YYYY-MM-DD"
  total_boundaries: <count>
  high_risk_boundaries: <count>
  medium_risk_boundaries: <count>
  low_risk_boundaries: <count>

boundaries:
  - <boundary objects as specified above>
```

---

## VERIFICATION CHECKLIST

- [ ] All 11 critical boundaries listed above are documented with actual signatures
- [ ] Every symbol has its actual function signature (read from source, not guessed)
- [ ] Shared artifacts have their actual schema fields documented
- [ ] Boundary risk ratings are justified
- [ ] No boundaries from DEPENDENCY_EDGES.json are missing
- [ ] Lazy imports are flagged with their line numbers
