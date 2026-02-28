# Subsystem Audit Manifest — quant_engine
Generated: 2026-02-27
Job 6 of 7 — Audit Ordering + Cross-Subsystem Report

Source data: SUBSYSTEM_MAP.json (Job 5), DEPENDENCY_EDGES.json (Job 2), INTERFACE_CONTRACTS.yaml (Job 4), HOTSPOT_LIST.md (Job 3)

---

## Dependency DAG Verification

The subsystem dependency graph was verified to be a **proper DAG with zero cycles**.

- **11 nodes** (subsystems), **31 hard dependency edges**, **3 optional/lazy dependency edges**
- Optional edges (do NOT block audit order): features→regime (lazy), data→kalshi (conditional factory), autopilot→api (6 lazy/conditional circular edges)
- The autopilot↔api circular reference is **NOT a true strongly connected component** — the reverse edge (api→autopilot) flows through `api/jobs/autopilot_job.py:12` (job runner), not the service layer

---

## Audit Order Summary

```
ORDER | SUBSYSTEM                    | DEPENDS ON              | FILES | LINES  | EST. HOURS | PRIORITY
------+------------------------------+-------------------------+-------+--------+------------+---------
  1   | Shared Infrastructure        | (none)                  |   7   |  2,151 |   2-3      | CRITICAL
  2   | Data Ingestion & Quality     | 1                       |  19   |  9,044 |   6-8      | HIGH
  3   | Feature Engineering          | 1, 2 (opt: 4)           |  17   |  8,559 |   6-8      | CRITICAL
  4   | Regime Detection             | 1                       |  13   |  4,420 |   5-6      | HIGH
  5   | Backtesting + Risk           | 1, 4                    |  28   | 13,132 |  10-14     | CRITICAL
  6   | Model Training & Prediction  | 1, 3, 5                 |  16   |  6,153 |   5-7      | HIGH
  7   | Evaluation & Diagnostics     | 1, 5, 6                 |   8   |  2,816 |   3-4      | LOW
  8   | Autopilot                    | 1,2,3,4,5,6 (opt:10)    |   8   |  4,480 |   4-6      | CRITICAL
  9   | Kalshi                       | 1, 3, 5, 8              |  16   |  5,208 |   5-7      | MEDIUM
 10   | API & Frontend               | all core subsystems      |  59   | 10,188 |  12-16     | HIGH
 11   | Entry Points & Scripts       | all subsystems           |  17   |  8,883 |   3-4      | LOW
------+------------------------------+-------------------------+-------+--------+------------+---------
TOTAL |                              |                         | 208   | 75,027 |  62-83     |
```

**Ordering methodology**: Topological sort of the dependency DAG. When multiple subsystems share the same topological depth, tiebreaking is by (1) module-level risk score from HOTSPOT_LIST.md (higher risk audited first), then (2) file count (fewer files first for quick wins).

**Ordering verification**: Every subsystem's hard dependencies appear earlier in the ordering. Optional dependencies are flagged in each audit brief but do not block ordering.

---

## Subsystem 1: Shared Infrastructure

```yaml
subsystem:
  name: Shared Infrastructure
  audit_order: 1
  estimated_hours: 2-3

  files_to_read:
    critical:
      - path: config.py
        why: "Supreme hub — 161 fan-in edges from ALL 14 modules, 200+ constants, 52% of all cross-module edges. Hotspot score 16/21."
        read_first: true
      - path: config_structured.py
        why: "Typed dataclass configuration — authoritative source. config.py derives from this via _get_config(). 347 lines."
        read_first: true
    supporting:
      - path: reproducibility.py
        why: "Run manifest generation for entry points — used by run_*.py scripts"
      - path: utils/logging.py
        why: "System-wide logging with alert webhooks — imported by config.py"
      - path: __init__.py
        why: "Package root — verify it doesn't re-export anything unexpected"
      - path: config_data/__init__.py
        why: "Config data package marker"
      - path: utils/__init__.py
        why: "Utils package marker"

  what_to_look_for:
    - category: correctness
      checks:
        - "Verify all 200+ constants have correct types and sensible values"
        - "Verify STATUS annotations (ACTIVE/PLACEHOLDER/DEPRECATED) are accurate — deprecated constants should not be referenced"
        - "Verify TRUTH_LAYER flags are consistent (TRUTH_LAYER_STRICT_PRECONDITIONS, TRUTH_LAYER_ENFORCE_CAUSALITY)"
        - "Verify config_structured.py ↔ config.py synchronization — every structured field should have a flat equivalent"
        - "Verify config_data/universe.yaml membership — changes propagate to ALL data loading, backtesting, and trading"
    - category: maintainability
      checks:
        - "Identify constants that are consumed by no module (dead constants)"
        - "Check for constants with STATUS=PLACEHOLDER that have been implemented but not updated"
        - "Assess God object pattern — config.py is a single point of failure for the entire system"

  known_risks:
    - risk: "God object pattern — 161 inbound edges, any constant rename/type change silently propagates to ALL 14 modules"
      severity: CRITICAL
      evidence: "DEPENDENCY_EDGES.json: 161 cross-module edges to config.py from all 14 modules"
    - risk: "No typed validation for most flat constants — config.py exposes raw values without type guards"
      severity: HIGH
      evidence: "config.py:1-1020 — most constants are bare assignments"
    - risk: "Largest import statements in the codebase consume 55+, 40+, and 29+ constants from config"
      severity: HIGH
      evidence: "backtest/engine.py:26 (55+), autopilot/paper_trader.py:13 (40+), autopilot/promotion_gate.py:13 (29+)"

  dependencies_to_verify: []

  outputs_to_verify:
    - artifact: config_data/universe.yaml
      consumers: [data_ingestion_quality, backtesting_risk, autopilot, api_frontend]
      check: "Universe membership changes propagate to all data loading, backtesting, and trading — verify ticker list is current"
    - artifact: "All 200+ config constants"
      consumers: [ALL subsystems]
      check: "Every constant imported by downstream modules exists, has correct type, and has correct value"
```

### Transition to Subsystem 2

```yaml
transition:
  from_subsystem: Shared Infrastructure
  to_subsystem: Data Ingestion & Quality

  carry_forward:
    - context: "All config constants related to data paths, cache settings, and quality thresholds"
      relevant_files: [config.py, config_structured.py]
    - context: "Universe definitions from config_data/universe.yaml — determines which tickers are loaded"
      relevant_files: [config_data/universe.yaml]

  boundary_checks:
    - boundary_id: data_to_config_12
      check: "Verify all 20+ config constants imported by data/ modules exist and have correct types"
      files_to_compare:
        - config.py
        - data/loader.py
        - data/local_cache.py
        - data/quality.py
        - data/feature_store.py

  common_failure_modes:
    - failure: "Config constant renamed or removed but data module still references old name"
      detection: "Import error at data module load time, or NameError at runtime for lazy imports"
    - failure: "Path constant (DATA_CACHE_DIR, FRAMEWORK_DIR) points to non-existent directory"
      detection: "FileNotFoundError when data loading is attempted"
```

---

## Subsystem 2: Data Ingestion & Quality

```yaml
subsystem:
  name: Data Ingestion & Quality
  audit_order: 2
  estimated_hours: 6-8

  files_to_read:
    critical:
      - path: data/loader.py
        why: "Primary data ingestion point — 849 lines, hotspot 15/21. 11 fan-in edges from 4 consumer modules. Lazy import of DataIntegrityValidator at line 567."
        read_first: true
      - path: data/local_cache.py
        why: "Cache layer with atomic temp-file writes and metadata sidecars — hotspot 14/21. Writes data/cache/*.parquet and *.meta.json consumed by loader and features."
        read_first: true
      - path: data/quality.py
        why: "OHLCV relationship validation — consumed by validation/data_integrity.py"
        read_first: false
      - path: validation/data_integrity.py
        why: "Quality gate — DataIntegrityValidator conditionally imported by data/loader.py:567"
        read_first: false
    supporting:
      - path: data/wrds_provider.py
        why: "WRDS data pipeline — 1,620 lines, hotspot 9/21. Fallback behavior when WRDS unavailable"
      - path: data/provider_base.py
        why: "Base class for all data providers — contract definition"
      - path: data/provider_registry.py
        why: "Provider factory — lazy conditional import of kalshi/provider.py at line 23"
      - path: data/survivorship.py
        why: "Survivorship bias filtering — critical for backtest integrity"
      - path: data/cross_source_validator.py
        why: "Cross-source data validation"
      - path: data/intraday_quality.py
        why: "Intraday OHLCV quality checks"
      - path: data/alternative.py
        why: "Alternative data sources"
      - path: data/feature_store.py
        why: "Feature store integration"
      - path: data/providers/alpaca_provider.py
        why: "Alpaca market data provider"
      - path: data/providers/alpha_vantage_provider.py
        why: "Alpha Vantage market data provider"
      - path: validation/leakage_detection.py
        why: "Data leakage detection — data quality focused"
      - path: validation/feature_redundancy.py
        why: "Feature redundancy detection — data quality focused"
      - path: data/__init__.py
        why: "Package marker — check re-exports"
      - path: data/providers/__init__.py
        why: "Providers package marker"
      - path: validation/__init__.py
        why: "Validation package marker"

  what_to_look_for:
    - category: correctness
      checks:
        - "Cache trust logic — verify CACHE_MAX_STALENESS_DAYS is enforced correctly"
        - "WRDS fallback behavior when connection fails — does it fall back gracefully or silently serve stale data?"
        - "Survivorship filtering correctness — verify point-in-time universe is applied"
        - "OHLCV relationship validation — High >= Open, Close, Low; Low <= Open, Close, High; Volume >= 0"
        - "DataIntegrityValidator conditional import at data/loader.py:567 — verify it works when validation module is available"
    - category: security
      checks:
        - "Cache file permissions — verify parquet files are not world-writable"
        - "WRDS credential handling — verify credentials are not logged or cached in plain text"
    - category: performance
      checks:
        - "Cache atomicity — verify temp-file-then-rename pattern prevents partial reads"
        - "Metadata sidecar schema stability — *.meta.json format changes break cache rehydration"

  known_risks:
    - risk: "Cache trust logic may serve stale data for live trading decisions"
      severity: HIGH
      evidence: "data/local_cache.py — CACHE_MAX_STALENESS_DAYS determines freshness; misconfiguration leads to stale data"
    - risk: "DataIntegrityValidator conditional import may silently skip validation"
      severity: MEDIUM
      evidence: "data/loader.py:567 — conditional import; if validation module fails to import, data quality checks are skipped"
    - risk: "Optional dependency on kalshi/provider.py via conditional factory"
      severity: LOW
      evidence: "data/provider_registry.py:23 — lazily imports kalshi/provider.py; data loading works without it"

  dependencies_to_verify:
    - depends_on_subsystem: Shared Infrastructure
      boundary_contracts:
        - symbol: "DATA_CACHE_DIR, FRAMEWORK_DIR, LOOKBACK_YEARS, MIN_BARS, WRDS_ENABLED, CACHE_*, and 20+ constants"
          check: "All config constants imported by data/ modules exist with correct types (boundary data_to_config_12)"

  outputs_to_verify:
    - artifact: "data/cache/*.parquet"
      consumers: [feature_engineering, api_frontend, entry_points_scripts]
      check: "Schema: Open, High, Low, Close, Volume, date columns. Verify column names and types are consistent."
    - artifact: "data/cache/*.meta.json"
      consumers: [data_ingestion_quality (internal)]
      check: "Metadata sidecar format stability — used for cache staleness detection"
```

### Transition to Subsystem 3

```yaml
transition:
  from_subsystem: Data Ingestion & Quality
  to_subsystem: Feature Engineering

  carry_forward:
    - context: "DataFrame schema from data loading (Open, High, Low, Close, Volume, date columns)"
      relevant_files: [data/loader.py, data/local_cache.py]
    - context: "Data quality validation behavior — which checks are applied before features are computed"
      relevant_files: [data/quality.py, validation/data_integrity.py]
    - context: "Cache structure and parquet file format — features/pipeline.py reads cached data directly"
      relevant_files: [data/local_cache.py]

  boundary_checks:
    - boundary_id: features_to_data_14
      check: "Verify features/pipeline.py imports load_ohlcv, WRDSProvider, load_intraday_ohlcv with correct signatures"
      files_to_compare:
        - data/loader.py
        - data/local_cache.py
        - data/wrds_provider.py
        - features/pipeline.py

  common_failure_modes:
    - failure: "DataFrame column naming mismatch between data loader output and feature pipeline expectations"
      detection: "KeyError or silent NaN propagation when pipeline.py accesses expected column names"
    - failure: "Stale or corrupted cache data fed to feature computation"
      detection: "Unexpected feature values, especially for intraday features from load_intraday_ohlcv"
```

---

## Subsystem 3: Feature Engineering

```yaml
subsystem:
  name: Feature Engineering
  audit_order: 3
  estimated_hours: 6-8

  files_to_read:
    critical:
      - path: features/pipeline.py
        why: "ALL 90+ features flow through this file — 1,541 lines, hotspot 16/21. FEATURE_METADATA is the SINGLE MOST CRITICAL CORRECTNESS BOUNDARY (defaults to CAUSAL for unknown names → silent data leakage)."
        read_first: true
      - path: indicators/indicators.py
        why: "92 concrete indicator subclasses — 2,904 lines (2nd largest production file), hotspot 14/21. 0 dedicated tests. Signature changes silently corrupt feature pipeline."
        read_first: true
    supporting:
      - path: indicators/spectral.py
        why: "SpectralAnalyzer — conditionally imported at pipeline.py:769"
      - path: indicators/ssa.py
        why: "SSADecomposer — conditionally imported at pipeline.py:789"
      - path: indicators/tail_risk.py
        why: "TailRiskAnalyzer — conditionally imported at pipeline.py:809"
      - path: indicators/ot_divergence.py
        why: "OptimalTransportAnalyzer — conditionally imported at pipeline.py:836"
      - path: indicators/eigenvalue.py
        why: "EigenvalueAnalyzer — conditionally imported at pipeline.py:1337"
      - path: features/research_factors.py
        why: "Research-only factors — must be correctly classified as RESEARCH_ONLY in FEATURE_METADATA"
      - path: features/options_factors.py
        why: "Options surface factors — consumed by kalshi/options.py"
      - path: features/intraday.py
        why: "Intraday microstructure features"
      - path: features/macro.py
        why: "Macro factor features"
      - path: features/harx_spillovers.py
        why: "HAR-X volatility spillover features"
      - path: features/wave_flow.py
        why: "Wave flow features"
      - path: features/lob_features.py
        why: "Limit order book features"
      - path: features/version.py
        why: "Feature versioning"
      - path: features/__init__.py
        why: "Package marker"
      - path: indicators/__init__.py
        why: "Re-exports 87 indicator classes to features/pipeline.py:21"

  what_to_look_for:
    - category: correctness
      checks:
        - "CRITICAL: Verify FEATURE_METADATA completeness — every computed feature MUST have an entry. Missing entries default to CAUSAL, enabling RESEARCH_ONLY features to leak forward-looking data into live predictions."
        - "Verify feature causality classifications (CAUSAL vs END_OF_DAY vs RESEARCH_ONLY) are correct for each feature"
        - "Verify all 5 conditional analyzer imports handle ImportError gracefully: SpectralAnalyzer:769, SSADecomposer:789, TailRiskAnalyzer:809, OptimalTransportAnalyzer:836, EigenvalueAnalyzer:1337"
        - "Verify compute_all() return dict keys remain stable — key changes silently break downstream feature columns"
        - "Verify all 87 indicator classes imported at pipeline.py:21 exist with correct compute() signatures"
    - category: correctness
      checks:
        - "Indicator return types/shapes — verify all compute() methods return consistent pd.Series format"
        - "Parameter defaults — check for silent behavior changes in indicator constructors"
    - category: maintainability
      checks:
        - "TEST COVERAGE WARNING: indicators/ has 7 files and 0 dedicated tests. 92 indicator subclasses are tested ONLY indirectly through feature pipeline tests."
        - "Verify INDICATOR_ALIASES for correctness"

  known_risks:
    - risk: "FEATURE_METADATA defaults to CAUSAL for unknown feature names — the single most critical data leakage risk"
      severity: CRITICAL
      evidence: "features/pipeline.py — get_feature_type() returns CAUSAL for any feature name not in FEATURE_METADATA"
    - risk: "92 indicator subclasses with 0 dedicated tests — extreme transitive amplification (1 direct dependent → 9 transitive dependents)"
      severity: HIGH
      evidence: "indicators/indicators.py (2,904 lines) → features/pipeline.py → models → backtest → evaluation chain"
    - risk: "5 advanced indicator analyzers conditionally imported may fail silently or return unstable dict keys"
      severity: HIGH
      evidence: "features/pipeline.py lines 769, 789, 809, 836, 1337 — conditional imports with compute_all() returning dicts"
    - risk: "Optional dependency on regime_detection (lazy import of CorrelationRegimeDetector)"
      severity: LOW
      evidence: "features/pipeline.py:1303 — try/except import of regime/correlation.py"

  dependencies_to_verify:
    - depends_on_subsystem: Shared Infrastructure
      boundary_contracts:
        - symbol: "FORWARD_HORIZONS, BENCHMARK, LOOKBACK_YEARS, INTERACTION_PAIRS, spectral/SSA/jump config, and 15+ constants"
          check: "All config constants imported by features/ exist with correct types (boundary features_to_config_13)"
    - depends_on_subsystem: Data Ingestion & Quality
      boundary_contracts:
        - symbol: "load_ohlcv, WRDSProvider, load_intraday_ohlcv"
          check: "Verify imported data loading functions return expected DataFrame schemas (boundary features_to_data_14)"

  outputs_to_verify:
    - artifact: "FeaturePipeline.compute() output DataFrame"
      consumers: [model_training_prediction, autopilot, api_frontend, entry_points_scripts]
      check: "Feature column names, ordering, and types are consistent between training and prediction paths"
    - artifact: "get_feature_type() causality classifications"
      consumers: [model_training_prediction (models/predictor.py:22)]
      check: "CRITICAL: Every feature name returns the correct causality type (boundary models_to_features_6)"
```

### Transition to Subsystem 4

```yaml
transition:
  from_subsystem: Feature Engineering
  to_subsystem: Regime Detection

  carry_forward:
    - context: "Feature pipeline lazy import of regime/correlation.py — optional dependency direction"
      relevant_files: [features/pipeline.py]
    - context: "FEATURE_METADATA completeness assessment — impacts causality enforcement downstream"
      relevant_files: [features/pipeline.py]

  boundary_checks:
    - boundary_id: features_to_regime_15
      check: "Verify CorrelationRegimeDetector import at features/pipeline.py:1303 is truly optional (try/except)"
      files_to_compare:
        - features/pipeline.py
        - regime/correlation.py

  common_failure_modes:
    - failure: "CorrelationRegimeDetector API change breaks the lazy import in features/pipeline.py"
      detection: "Exception logged but swallowed at pipeline.py:1303 — feature computation completes without correlation regime features"
    - failure: "Regime detection output format changes affect downstream feature-regime integration"
      detection: "Mismatched regime labels between detection and feature conditioning"
```

---

## Subsystem 4: Regime Detection

```yaml
subsystem:
  name: Regime Detection
  audit_order: 4
  estimated_hours: 5-6

  files_to_read:
    critical:
      - path: regime/detector.py
        why: "Main regime orchestrator — 940 lines, hotspot 15/21. Coordinates rule-based, HMM, jump model, and ensemble detection. 10 fan-in edges from 4 consumer modules."
        read_first: true
      - path: regime/shock_vector.py
        why: "Version-locked ShockVector schema with 13 fields — hotspot 12/21 (+3 schema bonus). Cross-subsystem contract file consumed by backtesting_risk."
        read_first: true
      - path: regime/uncertainty_gate.py
        why: "Entropy-based sizing multiplier — hotspot 12/21 (+2 contract bonus). Stability: EVOLVING. Consumed by 3 modules across 2 subsystems (backtest/engine.py:78, autopilot/engine.py:61, risk/position_sizer.py:27)."
        read_first: true
    supporting:
      - path: regime/hmm.py
        why: "HMM-based regime detection — verify state mapping correctness (0-3)"
      - path: regime/jump_model_pypi.py
        why: "Jump model using PyPI implementation"
      - path: regime/jump_model.py
        why: "Core jump model implementation"
      - path: regime/jump_model_legacy.py
        why: "Legacy jump model — check if still referenced"
      - path: regime/bocpd.py
        why: "Bayesian Online Changepoint Detection — sensitivity parameters"
      - path: regime/correlation.py
        why: "CorrelationRegimeDetector — consumed by feature_engineering (lazy)"
      - path: regime/confidence_calibrator.py
        why: "Regime confidence calibration"
      - path: regime/consensus.py
        why: "Ensemble consensus across detection engines"
      - path: regime/online_update.py
        why: "Online regime updates"
      - path: regime/__init__.py
        why: "Package marker"

  what_to_look_for:
    - category: correctness
      checks:
        - "Regime label consistency — verify 0-3 mapping is consistent across all detection engines (rule-based, HMM, jump, ensemble)"
        - "ShockVector schema version lock — verify schema_version='1.0' is enforced at deserialization, not just set at creation"
        - "ShockVector 13 fields — verify structural_features (Dict[str,float]) and transition_matrix (Optional[np.ndarray]) are correctly serialized/deserialized via to_dict/from_dict"
        - "UncertaintyGate config constant consistency — verify REGIME_UNCERTAINTY_ENTROPY_THRESHOLD, REGIME_UNCERTAINTY_STRESS_THRESHOLD, REGIME_UNCERTAINTY_SIZING_MAP, REGIME_UNCERTAINTY_MIN_MULTIPLIER are consistent across all 3 consumers"
        - "HMM state mapping correctness — verify states 0-3 map to meaningful regime labels"
        - "BOCPD changepoint sensitivity — verify hazard parameters produce reasonable changepoint detection"
        - "Ensemble consensus thresholds — verify thresholds are reasonable"
    - category: correctness
      checks:
        - "Confidence calibration — verify calibrated confidence values are well-calibrated (not overconfident)"
        - "Online update behavior — verify regime updates are consistent with batch detection"

  known_risks:
    - risk: "ShockVector schema changes break backtest execution layer and paper trading execution"
      severity: HIGH
      evidence: "regime/shock_vector.py — version-locked schema consumed by backtest/engine.py:77 and backtest/execution.py:29"
    - risk: "UncertaintyGate threshold changes silently alter position sizing across ALL trading paths"
      severity: HIGH
      evidence: "regime/uncertainty_gate.py — consumed by backtest/engine.py:78, autopilot/engine.py:61, risk/position_sizer.py:27"
    - risk: "Regime label inconsistency between detection engines could corrupt downstream conditioning"
      severity: MEDIUM
      evidence: "regime/detector.py orchestrates multiple engines — labels must be consistent"

  dependencies_to_verify:
    - depends_on_subsystem: Shared Infrastructure
      boundary_contracts:
        - symbol: "REGIME_MODEL_TYPE, REGIME_HMM_STATES, BOCPD_*, SHOCK_VECTOR_*, and 25+ constants"
          check: "All config constants imported by regime/ modules exist with correct types (boundary regime_to_config_16)"

  outputs_to_verify:
    - artifact: "ShockVector dataclass"
      consumers: [backtesting_risk (backtest/engine.py:77, backtest/execution.py:29)]
      check: "Schema version lock enforced, 13 fields stable, to_dict/from_dict round-trip correctly (boundary backtest_to_regime_shock_1)"
    - artifact: "UncertaintyGate sizing multiplier"
      consumers: [backtesting_risk (backtest/engine.py:78, risk/position_sizer.py:27), autopilot (autopilot/engine.py:61)]
      check: "Threshold defaults consistent across all 3 consumers, multiplier values are reasonable (boundary backtest_to_regime_uncertainty_2)"
    - artifact: "RegimeDetector output (labels, confidence)"
      consumers: [autopilot, api_frontend, entry_points_scripts]
      check: "Regime labels (0-3) and confidence values are consistent across detection engines"
```

### Transition to Subsystem 5

```yaml
transition:
  from_subsystem: Regime Detection
  to_subsystem: Backtesting + Risk

  carry_forward:
    - context: "ShockVector schema (13 fields, version-locked) — consumed by backtest/engine.py and backtest/execution.py"
      relevant_files: [regime/shock_vector.py]
    - context: "UncertaintyGate threshold values and sizing multiplier behavior — consumed by backtest/engine.py and risk/position_sizer.py"
      relevant_files: [regime/uncertainty_gate.py]
    - context: "Regime label mapping (0-3) — backtest uses regime state for conditioning"
      relevant_files: [regime/detector.py]

  boundary_checks:
    - boundary_id: backtest_to_regime_shock_1
      check: "Verify backtest/engine.py:77 imports compute_shock_vectors and ShockVector with correct signatures. Verify ShockVector fields accessed in backtest match the 13-field schema."
      files_to_compare:
        - regime/shock_vector.py
        - backtest/engine.py
        - backtest/execution.py
    - boundary_id: backtest_to_regime_uncertainty_2
      check: "Verify backtest/engine.py:78 imports UncertaintyGate correctly. Verify threshold constants match the values in regime/uncertainty_gate.py."
      files_to_compare:
        - regime/uncertainty_gate.py
        - backtest/engine.py
        - risk/position_sizer.py

  common_failure_modes:
    - failure: "ShockVector field added/removed but backtest/execution.py still accesses old field name"
      detection: "AttributeError at runtime when ShockVector is used in execution cost calculation"
    - failure: "UncertaintyGate threshold changed in config but one consumer still uses hardcoded default"
      detection: "Inconsistent position sizing between backtest and paper trading paths"
```

---

## Subsystem 5: Backtesting + Risk

```yaml
subsystem:
  name: Backtesting + Risk
  audit_order: 5
  estimated_hours: 10-14

  files_to_read:
    critical:
      - path: backtest/engine.py
        why: "LARGEST production file (2,488 lines), hotspot 16/21. Core simulation loop. 55+ config constant imports at line 26 — the single largest import statement in the codebase. 5 risk classes lazy-imported at lines 316-320."
        read_first: true
      - path: risk/position_sizer.py
        why: "Kelly sizing — 1,254 lines, hotspot 10/21. Stability: EVOLVING with 21 parameters including recent uncertainty additions. Sizing errors directly affect live trading."
        read_first: true
      - path: validation/preconditions.py
        why: "Execution contract — 71 lines, hotspot 11/21 (+3 contract bonus). Cross-subsystem contract file consumed by model_training_prediction (models/trainer.py:219). enforce_preconditions() must match between training and backtesting paths."
        read_first: true
      - path: backtest/validation.py
        why: "Walk-forward validation — 1,074 lines, hotspot 11/21. 4 consumer modules."
        read_first: false
    supporting:
      - path: backtest/execution.py
        why: "Execution simulator with structural costs — 936 lines. Imports ShockVector from regime/"
      - path: backtest/cost_calibrator.py
        why: "Transaction cost calibration"
      - path: backtest/cost_stress.py
        why: "Cost stress testing"
      - path: backtest/optimal_execution.py
        why: "Optimal execution algorithms"
      - path: backtest/null_models.py
        why: "Null model benchmarks for backtest validation"
      - path: backtest/adv_tracker.py
        why: "Average daily volume tracking for participation limits"
      - path: backtest/advanced_validation.py
        why: "Advanced validation — consumed by kalshi/walkforward.py"
      - path: backtest/survivorship_comparison.py
        why: "Survivorship bias comparison"
      - path: risk/portfolio_risk.py
        why: "Portfolio-level risk management"
      - path: risk/drawdown.py
        why: "Drawdown controller"
      - path: risk/stop_loss.py
        why: "Stop loss manager"
      - path: risk/metrics.py
        why: "Risk metrics computation"
      - path: risk/covariance.py
        why: "Covariance estimation"
      - path: risk/portfolio_optimizer.py
        why: "Portfolio optimization"
      - path: risk/factor_portfolio.py
        why: "Factor portfolio construction"
      - path: risk/factor_exposures.py
        why: "Factor exposure analysis"
      - path: risk/factor_monitor.py
        why: "Factor exposure monitoring — consumed by api/services"
      - path: risk/attribution.py
        why: "Performance attribution"
      - path: risk/stress_test.py
        why: "Stress testing"
      - path: risk/cost_budget.py
        why: "Cost budget management"
      - path: risk/constraint_replay.py
        why: "Constraint replay for debugging"
      - path: risk/universe_config.py
        why: "Universe configuration — reads config_data/universe.yaml"
      - path: backtest/__init__.py
        why: "Package marker"
      - path: risk/__init__.py
        why: "Package marker"

  what_to_look_for:
    - category: correctness
      checks:
        - "Execution realism — verify transaction costs, fill assumptions, and participation limits are realistic"
        - "Verify ALL 55+ config constant imports at backtest/engine.py:26 exist and have correct types"
        - "Verify all 5 risk classes can be imported independently at engine.py:316-320 (PositionSizer, DrawdownController, StopLossManager, PortfolioRiskManager, RiskMetrics)"
        - "Verify PositionSizer.size_position() 21 parameters match what the backtester passes"
        - "Verify execution costs match paper_trader.py behavior (consistency between backtest and live)"
        - "Verify enforce_preconditions() behavior matches between training (trainer.py:219) and backtesting (engine.py:198) paths"
        - "Verify TRUTH_LAYER_STRICT_PRECONDITIONS behavior (fail-fast vs warn)"
    - category: correctness
      checks:
        - "ShockVector integration correctness in backtest/execution.py"
        - "UncertaintyGate integration correctness in backtest/engine.py:78 and risk/position_sizer.py:27"
        - "Walk-forward validation correctness — purged dates, embargo gaps"
    - category: maintainability
      checks:
        - "TEST COVERAGE WARNING: validation/ has 5 files and 0 dedicated tests. Execution contracts (preconditions.py) are safety gates that are themselves unverified."
        - "Size exception: 28 files (25 limit) — justified by 5-way tight coupling at backtest/engine.py:316-320"

  known_risks:
    - risk: "Core simulation loop errors invalidate ALL backtest results and ALL promotion decisions"
      severity: CRITICAL
      evidence: "backtest/engine.py (2,488 lines) — single point of failure for backtest integrity"
    - risk: "PositionSizer is EVOLVING with 21 parameters — parameter semantics may drift"
      severity: HIGH
      evidence: "risk/position_sizer.py — recent uncertainty additions to size_position()"
    - risk: "5 risk classes lazy-imported at engine.py:316-320 — import failures only surface when use_risk_management=True"
      severity: HIGH
      evidence: "backtest/engine.py:316-320 — lazy imports inside _init_risk_managers()"
    - risk: "Preconditions divergence between training and backtesting paths"
      severity: HIGH
      evidence: "validation/preconditions.py consumed at models/trainer.py:219 (lazy) and backtest/engine.py:198 (lazy)"
    - risk: "Untested validation code means safety gates themselves are unverified"
      severity: MEDIUM
      evidence: "validation/ has 5 files and 0 dedicated tests"

  dependencies_to_verify:
    - depends_on_subsystem: Shared Infrastructure
      boundary_contracts:
        - symbol: "TRANSACTION_COST_BPS, ENTRY_THRESHOLD, EXEC_*, SHOCK_MODE_*, and 55+ constants"
          check: "All config constants at backtest/engine.py:26 exist with correct types (boundary backtest_to_config_19)"
    - depends_on_subsystem: Regime Detection
      boundary_contracts:
        - symbol: "compute_shock_vectors, ShockVector"
          check: "ShockVector 13-field schema is stable, version lock is enforced (boundary backtest_to_regime_shock_1)"
        - symbol: "UncertaintyGate"
          check: "Threshold constants consistent between regime/ definition and backtest/risk consumers (boundary backtest_to_regime_uncertainty_2)"

  outputs_to_verify:
    - artifact: "results/backtest_*d_summary.json"
      consumers: [api_frontend (api/services/backtest_service.py, api/services/results_service.py), evaluation_diagnostics (evaluation/engine.py)]
      check: "17-field schema including regime_breakdown. Key changes break API display and evaluation metrics."
    - artifact: "BacktestResult dataclass"
      consumers: [autopilot (autopilot/promotion_gate.py:12), kalshi (kalshi/promotion.py:14)]
      check: "BacktestResult fields used by promotion gate and kalshi evaluation must be stable"
    - artifact: "Backtester class"
      consumers: [autopilot (autopilot/engine.py:20), api_frontend (api/orchestrator.py:289)]
      check: "Backtester API (constructor, run methods) must be stable"
    - artifact: "enforce_preconditions()"
      consumers: [model_training_prediction (models/trainer.py:219)]
      check: "CROSS-SUBSYSTEM CONTRACT: precondition behavior must match between training and backtesting paths"
```

### Transition to Subsystem 6

```yaml
transition:
  from_subsystem: Backtesting + Risk
  to_subsystem: Model Training & Prediction

  carry_forward:
    - context: "enforce_preconditions() behavior and config constants (RET_TYPE, LABEL_H, PX_TYPE, ENTRY_PRICE_TYPE, TRUTH_LAYER_STRICT_PRECONDITIONS)"
      relevant_files: [validation/preconditions.py, config.py]
    - context: "BacktestResult schema — models/predictor.py's output feeds into backtesting"
      relevant_files: [backtest/engine.py]
    - context: "Walk-forward validation parameters and behavior"
      relevant_files: [backtest/validation.py]

  boundary_checks:
    - boundary_id: models_to_validation_18
      check: "Verify models/trainer.py:219 imports enforce_preconditions with same semantics as backtest/engine.py:198"
      files_to_compare:
        - validation/preconditions.py
        - models/trainer.py
        - backtest/engine.py
    - boundary_id: models_to_features_6
      check: "Verify models/predictor.py:22 imports get_feature_type correctly — CRITICAL for causality enforcement"
      files_to_compare:
        - features/pipeline.py
        - models/predictor.py

  common_failure_modes:
    - failure: "Preconditions enforce different rules for training vs backtesting — models trained on different assumptions"
      detection: "IS/OOS performance gap diverges unexpectedly; preconditions pass for training but fail for backtesting or vice versa"
    - failure: "Feature alignment mismatch between training and prediction — wrong feature order or missing features"
      detection: "Silent prediction errors — no exception, just incorrect probability values"
```

---

## Subsystem 6: Model Training & Prediction

```yaml
subsystem:
  name: Model Training & Prediction
  audit_order: 6
  estimated_hours: 5-7

  files_to_read:
    critical:
      - path: models/trainer.py
        why: "Training pipeline — 1,818 lines, hotspot 16/21. Writes trained_models/ artifacts. Lazy import of enforce_preconditions at line 219. Anti-overfitting controls (purged CV, embargo, holdout) must be correct."
        read_first: true
      - path: models/predictor.py
        why: "Prediction pipeline — 538 lines, hotspot 12/21. Imports get_feature_type at line 22. Feature alignment between trainer and predictor is critical."
        read_first: true
      - path: models/versioning.py
        why: "Model version registry — 207 lines, hotspot 9/21. Version resolution must be correct."
        read_first: false
    supporting:
      - path: models/governance.py
        why: "Model governance and approval workflows"
      - path: models/calibration.py
        why: "Probability calibration — consumed by evaluation/calibration_analysis.py"
      - path: models/conformal.py
        why: "Conformal prediction intervals"
      - path: models/walk_forward.py
        why: "Walk-forward model training"
      - path: models/online_learning.py
        why: "Online learning updates"
      - path: models/feature_stability.py
        why: "Feature stability tracking — consumed by api/services/health_service.py:1800"
      - path: models/shift_detection.py
        why: "Distribution shift detection"
      - path: models/retrain_trigger.py
        why: "Automated retrain triggers"
      - path: models/cross_sectional.py
        why: "Cross-sectional model ranking"
      - path: models/neural_net.py
        why: "Neural network model"
      - path: models/iv/__init__.py
        why: "Implied volatility package"
      - path: models/iv/models.py
        why: "IV surface models"
      - path: models/__init__.py
        why: "Package marker"

  what_to_look_for:
    - category: correctness
      checks:
        - "CRITICAL: Verify get_feature_type() integration at models/predictor.py:22 — TRUTH_LAYER_ENFORCE_CAUSALITY must actually block non-CAUSAL features during live prediction"
        - "Scaler fit isolation — verify scalers are fit INSIDE CV folds, not outside (data leakage)"
        - "Purged date-grouped CV — verify embargo gaps prevent look-ahead bias"
        - "Feature name alignment between trainer and predictor — mismatches break predictions SILENTLY"
        - "Model artifact schema compatibility between trainer (writer) and predictor+model_service (readers)"
        - "Verify enforce_preconditions() at trainer.py:219 uses same parameters as backtest/engine.py:198"
    - category: correctness
      checks:
        - "Regime blending logic — verify confidence-based blending produces reasonable outputs"
        - "Model version resolution — verify correct model directory is loaded for the right horizon"
    - category: security
      checks:
        - "Trained model artifacts (joblib) — verify deserialization is safe (joblib can execute arbitrary code)"

  known_risks:
    - risk: "Feature alignment mismatch between training and prediction breaks predictions SILENTLY — no error, just wrong numbers"
      severity: CRITICAL
      evidence: "models/trainer.py writes feature ordering; models/predictor.py must match exactly"
    - risk: "get_feature_type() CAUSAL default allows RESEARCH_ONLY features into live predictions"
      severity: CRITICAL
      evidence: "models/predictor.py:22 imports get_feature_type — enforcement depends on FEATURE_METADATA completeness"
    - risk: "Anti-overfitting controls (purged CV, embargo) if incorrect invalidate all downstream results"
      severity: HIGH
      evidence: "models/trainer.py — training pipeline produces models consumed by prediction, backtesting, and autopilot"
    - risk: "Model artifact format changes silently break model loading in predictor and API"
      severity: HIGH
      evidence: "trained_models/ensemble_*d_meta.json required fields: global_features, global_feature_medians, global_target_std, regime_models"

  dependencies_to_verify:
    - depends_on_subsystem: Shared Infrastructure
      boundary_contracts:
        - symbol: "MODEL_DIR, MODEL_PARAMS, MAX_FEATURES_SELECTED, CV_FOLDS, REGIME_NAMES, and 15+ constants"
          check: "All config constants imported by models/ exist with correct types (boundary models_to_config_17)"
    - depends_on_subsystem: Feature Engineering
      boundary_contracts:
        - symbol: "get_feature_type"
          check: "CRITICAL: Verify get_feature_type returns correct causality type for every feature (boundary models_to_features_6)"
    - depends_on_subsystem: Backtesting + Risk
      boundary_contracts:
        - symbol: "enforce_preconditions"
          check: "Verify precondition behavior matches between models/trainer.py:219 and backtest/engine.py:198 (boundary models_to_validation_18)"

  outputs_to_verify:
    - artifact: "trained_models/ensemble_*d_*.pkl"
      consumers: [api_frontend (api/services/model_service.py)]
      check: "Joblib format stability — EnsemblePredictor._load() must handle the serialized format"
    - artifact: "trained_models/ensemble_*d_meta.json"
      consumers: [api_frontend (api/services/model_service.py)]
      check: "Required fields: global_features, global_feature_medians, global_target_std, regime_models. Schema change breaks EnsemblePredictor._load()."
    - artifact: "ModelTrainer class"
      consumers: [autopilot (autopilot/engine.py:59), api_frontend (api/orchestrator.py:161)]
      check: "ModelTrainer API must be stable"
    - artifact: "EnsemblePredictor class"
      consumers: [autopilot (autopilot/engine.py:58), api_frontend (api/orchestrator.py:224)]
      check: "EnsemblePredictor API must be stable — especially predict() signature"
```

### Transition to Subsystem 7

```yaml
transition:
  from_subsystem: Model Training & Prediction
  to_subsystem: Evaluation & Diagnostics

  carry_forward:
    - context: "Model calibration methods — evaluation/calibration_analysis.py imports from models/calibration.py"
      relevant_files: [models/calibration.py]
    - context: "Backtest summary JSON 17-field schema — evaluation/engine.py reads these artifacts"
      relevant_files: [backtest/engine.py]
    - context: "Model performance metrics and training parameters"
      relevant_files: [models/trainer.py, models/predictor.py]

  boundary_checks:
    - boundary_id: evaluation_to_models_backtest_11
      check: "Verify evaluation/ imports (compute_ece, compute_reliability_curve, walk_forward_with_embargo, rolling_ic, detect_ic_decay) exist with correct signatures"
      files_to_compare:
        - models/calibration.py
        - backtest/validation.py
        - evaluation/engine.py
        - evaluation/calibration_analysis.py

  common_failure_modes:
    - failure: "Backtest summary JSON schema change not reflected in evaluation/engine.py reader"
      detection: "KeyError when evaluation attempts to access removed or renamed summary fields"
    - failure: "Model calibration function signature change breaks evaluation/calibration_analysis.py"
      detection: "TypeError at runtime when calibration analysis is executed"
```

---

## Subsystem 7: Evaluation & Diagnostics

```yaml
subsystem:
  name: Evaluation & Diagnostics
  audit_order: 7
  estimated_hours: 3-4

  files_to_read:
    critical:
      - path: evaluation/engine.py
        why: "Red flag detection — 826 lines. Reads backtest_*d_summary.json artifacts (shared artifact coupling). Primary evaluation orchestrator."
        read_first: true
      - path: evaluation/calibration_analysis.py
        why: "Calibration analysis — lazily imports models/calibration.py. Verifies model probability calibration quality."
        read_first: false
    supporting:
      - path: evaluation/metrics.py
        why: "Evaluation metric computations"
      - path: evaluation/slicing.py
        why: "Performance slicing by regime, sector, etc."
      - path: evaluation/fragility.py
        why: "Strategy fragility analysis"
      - path: evaluation/ml_diagnostics.py
        why: "ML model diagnostics"
      - path: evaluation/visualization.py
        why: "Visualization generation"
      - path: evaluation/__init__.py
        why: "Package marker"

  what_to_look_for:
    - category: correctness
      checks:
        - "Red flag thresholds — verify detection thresholds are reasonable and evidence-based"
        - "Calibration analysis correctness — ECE, reliability curves"
        - "IC decay detection — verify statistical method is sound"
        - "Metric statistical correctness — Sharpe, Sortino, max drawdown calculations"
    - category: maintainability
      checks:
        - "Leaf module with 0 fan-in — changes here CANNOT cascade to other subsystems (safe to modify)"

  known_risks:
    - risk: "Evaluation metrics may be statistically incorrect, leading to wrong conclusions about strategy quality"
      severity: MEDIUM
      evidence: "evaluation/metrics.py — Sharpe ratio, drawdown calculations must be verified"
    - risk: "Backtest summary JSON schema changes break evaluation without warning"
      severity: LOW
      evidence: "evaluation/engine.py reads results/backtest_*d_summary.json"

  dependencies_to_verify:
    - depends_on_subsystem: Shared Infrastructure
      boundary_contracts:
        - symbol: "EVAL_WF_*, EVAL_IC_*, EVAL_CALIBRATION_*, EVAL_TOP_N_TRADES, and 15+ constants"
          check: "All config constants imported by evaluation/ exist with correct types (boundary evaluation_to_config_23)"
    - depends_on_subsystem: Backtesting + Risk
      boundary_contracts:
        - symbol: "walk_forward_with_embargo, rolling_ic, detect_ic_decay"
          check: "Backtest validation functions used by evaluation exist with correct signatures"
    - depends_on_subsystem: Model Training & Prediction
      boundary_contracts:
        - symbol: "compute_ece, compute_reliability_curve"
          check: "Model calibration functions used by evaluation/calibration_analysis.py exist with correct signatures"

  outputs_to_verify: []
```

### Transition to Subsystem 8

```yaml
transition:
  from_subsystem: Evaluation & Diagnostics
  to_subsystem: Autopilot

  carry_forward:
    - context: "Evaluation metrics and red flag detection — autopilot uses evaluation results indirectly through promotion decisions"
      relevant_files: [evaluation/engine.py]
    - context: "All upstream subsystem contracts (config, data, features, regime, backtest/risk, models) — autopilot depends on ALL of them"
      relevant_files: [config.py, data/loader.py, features/pipeline.py, regime/detector.py, backtest/engine.py, models/trainer.py, models/predictor.py]

  boundary_checks:
    - boundary_id: autopilot_to_multi_4
      check: "Verify ALL 23+ cross-module imports in autopilot/engine.py resolve correctly with correct signatures. This is the highest fan-out file (39 edges across 8 modules)."
      files_to_compare:
        - autopilot/engine.py
        - backtest/engine.py
        - models/trainer.py
        - models/predictor.py
        - data/loader.py
        - features/pipeline.py
        - regime/detector.py
        - risk/position_sizer.py

  common_failure_modes:
    - failure: "Upstream API change breaks one of autopilot/engine.py's 23+ cross-module imports"
      detection: "ImportError or TypeError when autopilot cycle runs — but only at runtime since many imports are lazy"
    - failure: "Promotion gate thresholds misaligned with evaluation criteria"
      detection: "Strategies that should be promoted aren't, or vice versa"
```

---

## Subsystem 8: Autopilot

```yaml
subsystem:
  name: Autopilot
  audit_order: 8
  estimated_hours: 4-6

  files_to_read:
    critical:
      - path: autopilot/engine.py
        why: "Primary transitive amplifier — 1,927 lines, hotspot 17/21. Appears in 12/15 hotspot blast radii. 23+ cross-module imports across 8 modules. 2 circular edges to api at lines 1868, 1911."
        read_first: true
      - path: autopilot/paper_trader.py
        why: "Circular dependency hub — 1,254 lines, hotspot 17/21. 4 of 6 autopilot→api circular edges at lines 173, 189, 211, 532. 0 cross-module dependents (risk is entirely consumption-side)."
        read_first: true
      - path: autopilot/promotion_gate.py
        why: "Promotion decision logic — imports 29+ PROMOTION_* constants from config. Consumed by kalshi/pipeline.py and kalshi/promotion.py."
        read_first: false
    supporting:
      - path: autopilot/strategy_discovery.py
        why: "Strategy discovery — consumed by kalshi/promotion.py"
      - path: autopilot/registry.py
        why: "Strategy registry — writes results/autopilot/strategy_registry.json (shared artifact consumed by api/services/autopilot_service.py)"
      - path: autopilot/meta_labeler.py
        why: "Meta-labeling for strategy confidence"
      - path: autopilot/strategy_allocator.py
        why: "Strategy allocation and portfolio construction"
      - path: autopilot/__init__.py
        why: "Package marker"

  what_to_look_for:
    - category: correctness
      checks:
        - "CRITICAL: Verify ALL 6 api imports are truly lazy and won't cause circular import at load time"
        - "Circular edge 1: autopilot/paper_trader.py:173 → api.services.health_risk_feedback (lazy)"
        - "Circular edge 2: autopilot/paper_trader.py:189 → api.services.health_service.HealthService (conditional)"
        - "Circular edge 3: autopilot/paper_trader.py:211 → api.ab_testing.ABTestingFramework (conditional)"
        - "Circular edge 4: autopilot/paper_trader.py:532 → api.services.health_service.HealthService (conditional)"
        - "Circular edge 5: autopilot/engine.py:1868 → api.services.health_service.HealthService (lazy)"
        - "Circular edge 6: autopilot/engine.py:1911 → api.services.health_service.HealthService (conditional)"
        - "Verify execution model integration matches backtest/execution.py behavior — cost consistency between backtest and paper trading"
        - "Verify promotion gate thresholds are reasonable and evidence-based"
        - "Verify paper trading Kelly sizing calculations match risk/position_sizer.py behavior"
    - category: correctness
      checks:
        - "Verify all 23+ cross-module imports in engine.py resolve correctly"
        - "Verify meta-labeling integration is correct"
        - "Verify paper state persistence and recovery after crashes"
        - "Verify heuristic predictor fallback path works when trained models are unavailable"

  known_risks:
    - risk: "Primary transitive amplifier — changes in ANY of 8 dependency modules can break the autopilot cycle"
      severity: CRITICAL
      evidence: "autopilot/engine.py appears in 12/15 hotspot blast radii — highest transitive impact"
    - risk: "6 circular edges to api services — any api service change can break paper trading and cycle orchestration at RUNTIME only"
      severity: CRITICAL
      evidence: "4 edges from paper_trader.py (lines 173, 189, 211, 532) + 2 from engine.py (lines 1868, 1911)"
    - risk: "Paper trading execution diverges from backtest execution — cost/fill assumptions may differ"
      severity: HIGH
      evidence: "autopilot/paper_trader.py imports ExecutionModel separately from backtest/execution.py"
    - risk: "Refactoring recommended: extract health risk feedback interface instead of importing from api service layer"
      severity: MEDIUM
      evidence: "Job 5 recommendation — circular dependency should be resolved by interface extraction"

  dependencies_to_verify:
    - depends_on_subsystem: Shared Infrastructure
      boundary_contracts:
        - symbol: "AUTOPILOT_CYCLE_REPORT, PROMOTION_*, PAPER_*, EXEC_*, STRATEGY_REGISTRY_PATH, and 80+ constants"
          check: "All config constants at engine.py:34 and paper_trader.py:13 exist with correct types (boundary autopilot_to_config_25)"
    - depends_on_subsystem: Data Ingestion & Quality
      boundary_contracts:
        - symbol: "load_survivorship_universe, load_universe, filter_panel_by_point_in_time_universe"
          check: "Data loading functions return expected formats (boundary autopilot_to_data_26)"
    - depends_on_subsystem: Feature Engineering
      boundary_contracts:
        - symbol: "FeaturePipeline"
          check: "FeaturePipeline API is stable (boundary autopilot_to_features_27)"
    - depends_on_subsystem: Regime Detection
      boundary_contracts:
        - symbol: "RegimeDetector, UncertaintyGate"
          check: "Regime detection outputs are compatible (boundary autopilot_to_regime_29)"
    - depends_on_subsystem: Backtesting + Risk
      boundary_contracts:
        - symbol: "Backtester, BacktestResult, ExecutionModel, walk_forward_validate, etc."
          check: "Backtest/risk APIs are stable and execution realism matches (boundary autopilot_to_backtest_31)"
    - depends_on_subsystem: Model Training & Prediction
      boundary_contracts:
        - symbol: "ModelTrainer, EnsemblePredictor, cross_sectional_rank"
          check: "Model APIs are stable (boundary autopilot_to_models_28)"

  outputs_to_verify:
    - artifact: "results/autopilot/strategy_registry.json"
      consumers: [api_frontend (api/services/autopilot_service.py)]
      check: "Registry schema stability — changes break API dashboard"
    - artifact: "results/autopilot/paper_state.json"
      consumers: [api_frontend (api/services/autopilot_service.py)]
      check: "Paper state schema stability — changes break API dashboard"
    - artifact: "AutopilotEngine class"
      consumers: [api_frontend (api/jobs/autopilot_job.py:12)]
      check: "AutopilotEngine API must be stable"
    - artifact: "PromotionGate class"
      consumers: [kalshi (kalshi/pipeline.py, kalshi/promotion.py)]
      check: "PromotionGate.evaluate_event_strategy() must handle both standard and event-mode BacktestResult"

  cross_reference_during_audit:
    - file: api/services/health_service.py
      reason: "Circular dependency target — review health risk feedback interface used by paper_trader.py"
    - file: api/ab_testing.py
      reason: "Circular dependency target — ABTestingFramework imported by paper_trader.py:211"
```

### Transition to Subsystem 9

```yaml
transition:
  from_subsystem: Autopilot
  to_subsystem: Kalshi

  carry_forward:
    - context: "PromotionGate contract — kalshi/promotion.py imports PromotionGate and strategy_discovery"
      relevant_files: [autopilot/promotion_gate.py, autopilot/strategy_discovery.py]
    - context: "BacktestResult compatibility — kalshi uses event-mode BacktestResult that must be compatible with standard BacktestResult"
      relevant_files: [backtest/engine.py, autopilot/promotion_gate.py]

  boundary_checks:
    - boundary_id: kalshi_to_autopilot_9
      check: "Verify kalshi/promotion.py imports PromotionDecision, PromotionGate, StrategyCandidate with correct signatures"
      files_to_compare:
        - autopilot/promotion_gate.py
        - autopilot/strategy_discovery.py
        - kalshi/promotion.py
        - kalshi/pipeline.py

  common_failure_modes:
    - failure: "PromotionGate API change breaks kalshi/promotion.py's evaluate_event_strategy() call"
      detection: "TypeError when Kalshi pipeline attempts promotion evaluation"
    - failure: "BacktestResult incompatibility between standard and event-mode evaluation"
      detection: "AttributeError when event-mode BacktestResult is passed to standard promotion gate"
```

---

## Subsystem 9: Kalshi

```yaml
subsystem:
  name: Kalshi
  audit_order: 9
  estimated_hours: 5-7

  files_to_read:
    critical:
      - path: kalshi/pipeline.py
        why: "Main Kalshi event pipeline — orchestrates data ingestion, feature computation, and prediction for event markets"
        read_first: true
      - path: kalshi/storage.py
        why: "DuckDB storage — manages 18 tables with versioning and audit trail. DDL changes require migration."
        read_first: true
      - path: kalshi/walkforward.py
        why: "Walk-forward validation for event strategies — imports from backtest/advanced_validation.py"
        read_first: false
    supporting:
      - path: kalshi/client.py
        why: "Kalshi API client"
      - path: kalshi/events.py
        why: "Event data management — reads from kalshi.duckdb"
      - path: kalshi/distribution.py
        why: "Distribution quality analysis — 935 lines"
      - path: kalshi/options.py
        why: "Options factor integration — imports from features/options_factors.py"
      - path: kalshi/promotion.py
        why: "Event strategy promotion — imports BacktestResult from backtest/ and promotion logic from autopilot/"
      - path: kalshi/quality.py
        why: "Data quality checks for Kalshi data"
      - path: kalshi/provider.py
        why: "KalshiProvider — conditionally imported by data/provider_registry.py:23"
      - path: kalshi/mapping_store.py
        why: "Event-to-market mapping storage"
      - path: kalshi/microstructure.py
        why: "Market microstructure analysis"
      - path: kalshi/disagreement.py
        why: "Market disagreement analysis"
      - path: kalshi/regimes.py
        why: "Event regime detection"
      - path: kalshi/router.py
        why: "Kalshi API router"
      - path: kalshi/__init__.py
        why: "Package marker"

  what_to_look_for:
    - category: correctness
      checks:
        - "Verify DuckDB 18-table schema stability — DDL changes require migration"
        - "Verify event-mode BacktestResult compatibility with standard BacktestResult (from backtest/engine.py)"
        - "Event-time join correctness — verify temporal alignment of event outcomes with market data"
        - "Staleness policies — verify data freshness checks for event data"
        - "Distribution quality — verify distribution fitting methods are statistically sound"
    - category: correctness
      checks:
        - "Verify kalshi/promotion.py correctly calls PromotionGate.evaluate_event_strategy()"
        - "Verify kalshi/options.py correctly calls compute_option_surface_factors()"
        - "Verify kalshi/walkforward.py correctly integrates with backtest/advanced_validation.py"
    - category: security
      checks:
        - "API credentials handling — verify Kalshi API keys are not hardcoded or logged"
        - "Rate limiting — verify KALSHI_RATE_LIMIT_* constants are enforced"

  known_risks:
    - risk: "DuckDB schema changes require coordinated migration across 18 tables"
      severity: MEDIUM
      evidence: "kalshi/storage.py manages 18 tables with versioning and audit trail"
    - risk: "Event-mode BacktestResult incompatibility with standard backtest evaluation"
      severity: MEDIUM
      evidence: "kalshi/promotion.py imports BacktestResult from backtest/engine.py for event evaluation"

  dependencies_to_verify:
    - depends_on_subsystem: Shared Infrastructure
      boundary_contracts:
        - symbol: "KALSHI_API_BASE_URL, KALSHI_ENV, KALSHI_RATE_LIMIT_*, KALSHI_STALE_*, and 18 constants"
          check: "All config constants imported by kalshi/ exist with correct types (boundary kalshi_to_config_34)"
    - depends_on_subsystem: Feature Engineering
      boundary_contracts:
        - symbol: "compute_option_surface_factors"
          check: "Options factor function exists with correct signature (boundary kalshi_to_features_35)"
    - depends_on_subsystem: Backtesting + Risk
      boundary_contracts:
        - symbol: "BacktestResult, deflated_sharpe_ratio, monte_carlo_validation"
          check: "Backtest functions exist with correct signatures (boundary kalshi_to_backtest_33)"
    - depends_on_subsystem: Autopilot
      boundary_contracts:
        - symbol: "PromotionDecision, PromotionGate, StrategyCandidate"
          check: "Promotion gate API is stable for event-mode evaluation (boundary kalshi_to_autopilot_9)"

  outputs_to_verify:
    - artifact: "data/kalshi.duckdb"
      consumers: [api_frontend (api/services/kalshi_service.py)]
      check: "DuckDB schema stability — 18 tables with versioning and audit trail"
    - artifact: "KalshiProvider"
      consumers: [data_ingestion_quality (data/provider_registry.py:23, conditional)]
      check: "Provider interface compatibility — conditional factory import"
```

### Transition to Subsystem 10

```yaml
transition:
  from_subsystem: Kalshi
  to_subsystem: API & Frontend

  carry_forward:
    - context: "Kalshi DuckDB schema — api/services/kalshi_service.py reads from kalshi.duckdb"
      relevant_files: [kalshi/storage.py, kalshi/events.py]
    - context: "ALL upstream subsystem contracts — API imports from every core module"
      relevant_files: [config.py, data/loader.py, features/pipeline.py, regime/detector.py, backtest/engine.py, models/trainer.py, models/predictor.py, autopilot/engine.py]
    - context: "Autopilot circular dependency — api/services/health_service.py is imported by autopilot (already audited)"
      relevant_files: [autopilot/paper_trader.py, autopilot/engine.py]

  boundary_checks:
    - boundary_id: api_to_config_36
      check: "Verify ALL 82 lazy config imports in api/ resolve correctly at runtime"
      files_to_compare:
        - config.py
        - api/services/health_service.py
        - api/orchestrator.py
    - boundary_id: api_to_autopilot_43
      check: "Verify api/jobs/autopilot_job.py:12 lazily imports AutopilotEngine correctly"
      files_to_compare:
        - autopilot/engine.py
        - api/jobs/autopilot_job.py
    - boundary_id: api_to_kalshi_44
      check: "Verify api/services/kalshi_service.py reads kalshi.duckdb with correct schema"
      files_to_compare:
        - kalshi/storage.py
        - api/services/kalshi_service.py
    - boundary_id: autopilot_to_api_circular_5
      check: "CIRCULAR DEPENDENCY: Verify the 6 forward edges (autopilot→api) and the reverse edge (api→autopilot via job runner) are correctly guarded"
      files_to_compare:
        - autopilot/engine.py
        - autopilot/paper_trader.py
        - api/services/health_service.py
        - api/ab_testing.py
        - api/jobs/autopilot_job.py

  common_failure_modes:
    - failure: "Lazy config import in api/ fails at runtime when specific endpoint is called"
      detection: "HTTP 500 error on specific API endpoints — failures only surface when endpoint is accessed"
    - failure: "Upstream model/backtest artifact format change breaks API service reader"
      detection: "API returns empty or malformed data for specific dashboard views"
```

---

## Subsystem 10: API & Frontend

```yaml
subsystem:
  name: API & Frontend
  audit_order: 10
  estimated_hours: 12-16

  files_to_read:
    critical:
      - path: api/orchestrator.py
        why: "Central API orchestrator — lazily imports from 6 subsystems. Coordinates training, prediction, backtesting, and regime detection."
        read_first: true
      - path: api/services/health_service.py
        why: "LARGEST file in codebase (2,929 lines), hotspot 14/21, stability: EVOLVING. 25+ config import sites. Circular dependency target from autopilot. Consider splitting."
        read_first: true
      - path: api/jobs/autopilot_job.py
        why: "Reverse edge of autopilot↔api circular dependency — lazily imports AutopilotEngine at line 12"
        read_first: false
    supporting:
      - path: api/main.py
        why: "FastAPI application entry point"
      - path: api/config.py
        why: "API-specific configuration (ApiSettings, RuntimeConfig)"
      - path: api/errors.py
        why: "Error handling and response formatting"
      - path: api/ab_testing.py
        why: "A/B testing framework — circular dependency target from autopilot/paper_trader.py:211"
      - path: api/deps/providers.py
        why: "Dependency injection providers"
      - path: api/cache/manager.py
        why: "Cache management"
      - path: api/cache/invalidation.py
        why: "Cache invalidation logic"
      - path: api/jobs/store.py
        why: "Job persistence"
      - path: api/jobs/runner.py
        why: "Async job execution"
      - path: api/jobs/backtest_job.py
        why: "Backtest job — lazily imports Backtester"
      - path: api/jobs/predict_job.py
        why: "Prediction job — lazily imports predictor"
      - path: api/jobs/train_job.py
        why: "Training job — lazily imports trainer"
      - path: api/services/autopilot_service.py
        why: "Autopilot service — reads strategy_registry.json and paper_state.json"
      - path: api/services/backtest_service.py
        why: "Backtest service — reads backtest_*d_summary.json artifacts"
      - path: api/services/model_service.py
        why: "Model service — reads trained_models/ artifacts"
      - path: api/services/data_service.py
        why: "Data service"
      - path: api/services/data_helpers.py
        why: "Data helper utilities — imports FeaturePipeline and RegimeDetector"
      - path: api/services/results_service.py
        why: "Results service — reads backtest results and predictions"
      - path: api/services/kalshi_service.py
        why: "Kalshi service — reads kalshi.duckdb"
      - path: api/services/regime_service.py
        why: "Regime service"
      - path: api/services/diagnostics.py
        why: "Diagnostics service"
      - path: api/services/health_alerts.py
        why: "Health alert management"
      - path: api/services/health_confidence.py
        why: "Health confidence scoring"
      - path: api/services/health_risk_feedback.py
        why: "Health risk feedback — circular dependency target from autopilot/paper_trader.py:173"
      - path: "api/routers/*.py (15 files)"
        why: "API route handlers"
      - path: "api/schemas/*.py (9 files)"
        why: "Pydantic schema definitions"
      - path: api/__init__.py
        why: "Package marker"
      - path: api/deps/__init__.py
        why: "Deps package marker"
      - path: api/cache/__init__.py
        why: "Cache package marker"
      - path: api/jobs/__init__.py
        why: "Jobs package marker"
      - path: api/jobs/models.py
        why: "Job data models"
      - path: api/routers/__init__.py
        why: "Routers package marker"
      - path: api/schemas/__init__.py
        why: "Schemas package marker"
      - path: api/services/__init__.py
        why: "Services package marker"

  what_to_look_for:
    - category: correctness
      checks:
        - "CRITICAL: Verify ALL 82 lazy config imports resolve correctly at runtime — failures only surface when specific API endpoints are called"
        - "Verify all shared artifact readers handle missing/corrupt artifacts gracefully (backtest summaries, model files, strategy registry, kalshi.duckdb)"
        - "Verify job system correctness — lazy imports in job files load correct modules"
        - "Verify cache invalidation logic — stale cache data could serve outdated results to dashboard"
        - "Verify the reverse edge of autopilot↔api circular dependency (api/jobs/autopilot_job.py:12) is correctly guarded"
    - category: correctness
      checks:
        - "health_service.py (2,929 lines) — verify health checks produce correct statuses"
        - "health_risk_feedback.py — verify interface consumed by autopilot/paper_trader.py:173 is stable"
        - "ab_testing.py — verify ABTestingFramework interface consumed by autopilot/paper_trader.py:211 is stable"
    - category: maintainability
      checks:
        - "Size exception: 59 files (25 limit) — justified by internal layering cohesion (routers→services→schemas)"
        - "health_service.py at 2,929 lines should be considered for splitting"
        - "115 outbound edges but only 2 modules' fan-in — highly fragile to upstream changes but low cascade risk"

  known_risks:
    - risk: "82 lazy config imports — any config constant rename breaks API endpoints at RUNTIME only"
      severity: HIGH
      evidence: "api/ modules have 82 import edges to config (ALL lazy/conditional to avoid circular imports)"
    - risk: "health_service.py is EVOLVING at 2,929 lines — the most fragile component in the circular dependency"
      severity: HIGH
      evidence: "api/services/health_service.py — largest file in codebase, involved in circular autopilot→api dependency"
    - risk: "API is highly fragile to upstream changes — 115 outbound edges but only 2 modules' fan-in"
      severity: MEDIUM
      evidence: "API consumes from every core module but is rarely imported — upstream changes cascade to API"

  dependencies_to_verify:
    - depends_on_subsystem: Shared Infrastructure
      boundary_contracts:
        - symbol: "82 config import edges across 20+ files"
          check: "All lazy config imports resolve at runtime (boundary api_to_config_36)"
    - depends_on_subsystem: Data Ingestion & Quality
      boundary_contracts:
        - symbol: "load_universe, load_survivorship_universe, get_skip_reasons, get_data_provenance, WRDSProvider"
          check: "Data functions exist with correct signatures (boundary api_to_data_37)"
    - depends_on_subsystem: Model Training & Prediction
      boundary_contracts:
        - symbol: "ModelGovernance, ModelTrainer, ModelRegistry, EnsemblePredictor, FeatureStabilityTracker"
          check: "Model APIs are stable (boundary api_to_models_40)"
    - depends_on_subsystem: Autopilot
      boundary_contracts:
        - symbol: "AutopilotEngine"
          check: "AutopilotEngine API is stable (boundary api_to_autopilot_43)"
    - depends_on_subsystem: Kalshi
      boundary_contracts:
        - symbol: "EventTimeStore"
          check: "Kalshi storage API is stable (boundary api_to_kalshi_44)"

  outputs_to_verify: []
```

### Transition to Subsystem 11

```yaml
transition:
  from_subsystem: API & Frontend
  to_subsystem: Entry Points & Scripts

  carry_forward:
    - context: "API orchestrator wiring — entry points must pass the same parameters as the API"
      relevant_files: [api/orchestrator.py]
    - context: "All upstream module APIs — entry points wire modules together directly"
      relevant_files: [config.py, data/loader.py, features/pipeline.py, regime/detector.py, backtest/engine.py, models/trainer.py, models/predictor.py, autopilot/engine.py]

  boundary_checks: []

  common_failure_modes:
    - failure: "Entry point passes wrong parameters to module constructors"
      detection: "TypeError or incorrect behavior when running entry point scripts"
    - failure: "Entry point writes shared artifacts with incorrect schema"
      detection: "Downstream readers (API, evaluation) fail to parse output files"
```

---

## Subsystem 11: Entry Points & Scripts

```yaml
subsystem:
  name: Entry Points & Scripts
  audit_order: 11
  estimated_hours: 3-4

  files_to_read:
    critical:
      - path: run_backtest.py
        why: "Writes results/backtest_*d_summary.json — shared artifact consumed by API and evaluation. Correct wiring is essential."
        read_first: true
      - path: run_train.py
        why: "Wires data loading, feature computation, and model training together"
        read_first: false
      - path: run_autopilot.py
        why: "Wires AutopilotEngine — entry point for the automated lifecycle"
        read_first: false
      - path: run_wrds_daily_refresh.py
        why: "Most complex entry point (915 lines) — daily data refresh pipeline"
        read_first: false
    supporting:
      - path: run_predict.py
        why: "Prediction entry point"
      - path: run_retrain.py
        why: "Retraining entry point"
      - path: run_server.py
        why: "API server startup"
      - path: run_kalshi_event_pipeline.py
        why: "Kalshi event pipeline entry point"
      - path: run_rehydrate_cache_metadata.py
        why: "Cache metadata rehydration"
      - path: scripts/alpaca_intraday_download.py
        why: "Alpaca intraday data download"
      - path: scripts/ibkr_intraday_download.py
        why: "IBKR intraday data download"
      - path: scripts/ibkr_daily_gapfill.py
        why: "IBKR daily gap fill"
      - path: scripts/compare_regime_models.py
        why: "Regime model comparison — tooling"
      - path: scripts/generate_types.py
        why: "Type generation — tooling"
      - path: scripts/extract_dependencies.py
        why: "Audit script — tooling, not production"
      - path: scripts/generate_interface_contracts.py
        why: "Audit script — tooling, not production"
      - path: scripts/hotspot_scoring.py
        why: "Audit script — tooling, not production"

  what_to_look_for:
    - category: correctness
      checks:
        - "Correct wiring of modules — verify parameters passed to constructors match expected signatures"
        - "CLI argument handling — verify argument parsing is correct and defaults are sensible"
        - "Reproducibility manifest generation — verify run manifests capture all relevant parameters"
        - "run_backtest.py shared artifact output — verify backtest_*d_summary.json schema matches what API and evaluation expect"
    - category: maintainability
      checks:
        - "Audit scripts (extract_dependencies.py, generate_interface_contracts.py, hotspot_scoring.py) are tooling for this audit workflow and do not affect production"
        - "Leaf consumers with 0 fan-in — changes here cannot cascade to other subsystems"

  known_risks:
    - risk: "run_backtest.py writes shared artifacts consumed by API and evaluation — schema changes break downstream"
      severity: MEDIUM
      evidence: "results/backtest_*d_summary.json consumed by api/services/backtest_service.py, api/services/results_service.py, evaluation/engine.py"
    - risk: "run_wrds_daily_refresh.py is the most complex entry point (915 lines) — daily data refresh correctness affects all downstream"
      severity: MEDIUM
      evidence: "Daily data refresh pipeline — errors corrupt cached data used by features, training, and backtesting"

  dependencies_to_verify:
    - depends_on_subsystem: ALL subsystems
      boundary_contracts:
        - symbol: "Module constructor and function signatures"
          check: "Verify each entry point passes correct parameters to module APIs"

  outputs_to_verify:
    - artifact: "results/backtest_*d_summary.json"
      consumers: [api_frontend, evaluation_diagnostics]
      check: "17-field schema including regime_breakdown must be stable"
```

---

## Verification Checklist

- [x] **Audit order respects dependency DAG** — verified programmatically: every subsystem's hard dependencies appear earlier in the ordering. Zero cycles found in the DAG.
- [x] **Every subsystem has an audit brief** — 11 audit briefs with critical files, what to look for, known risks, and estimated hours.
- [x] **Every adjacent pair has a transition guide** — 10 transition guides (1→2, 2→3, 3→4, 4→5, 5→6, 6→7, 7→8, 8→9, 9→10, 10→11) with carry-forward context, boundary checks, and common failure modes.
- [x] **Hour estimates are realistic** — based on file count, line count, and complexity. Total: 62-83 hours across all subsystems.
- [x] **Cross-subsystem boundary checks reference specific files and line numbers** — boundary IDs from INTERFACE_CONTRACTS.yaml are referenced with specific file paths and import line numbers.
- [x] **The autopilot↔api circular dependency is prominently flagged** — documented in Subsystem 8 (Autopilot) with all 6 edges listed with specific line numbers, in Subsystem 10 (API) with the reverse edge, and in the Subsystem 9→10 transition guide.
- [x] **Optional dependencies documented** — features→regime (lazy, pipeline.py:1303), data→kalshi (conditional factory, provider_registry.py:23), autopilot→api (6 lazy/conditional edges). None block audit order.
- [x] **Size bound exceptions justified** — backtesting_risk (28 files, +3 over limit: 5-way tight coupling), api_frontend (59 files, +34 over limit: internal layering cohesion).
- [x] **Cross-subsystem contract files assigned** — regime/shock_vector.py (regime_detection, consumed by backtesting_risk), regime/uncertainty_gate.py (regime_detection, consumed by backtesting_risk + autopilot), regime/correlation.py (regime_detection, consumed by feature_engineering), validation/preconditions.py (backtesting_risk, consumed by model_training_prediction), validation/data_integrity.py (data_ingestion_quality).
- [x] **Shared artifacts documented** — config_data/universe.yaml, data/cache/*.parquet, data/cache/*.meta.json, trained_models/ensemble_*d_*.pkl, trained_models/ensemble_*d_meta.json, results/backtest_*d_summary.json, results/autopilot/strategy_registry.json, results/autopilot/paper_state.json, data/kalshi.duckdb.
- [x] **Evolving stability flags reflected** — PositionSizer (Subsystem 5), HealthService (Subsystem 10), UncertaintyGate (Subsystem 4) all flagged as evolving in their respective audit briefs.
- [x] **Refactoring recommendations noted** — extract health risk feedback interface for autopilot→api dependency (Subsystem 8, known risks); consider splitting health_service.py (Subsystem 10, maintainability checks).
- [x] **Test coverage blind spots flagged** — indicators/ (7 files, 0 tests), validation/ (5 files, 0 tests), and other gaps documented in relevant audit briefs.
- [x] **File counts match SUBSYSTEM_MAP.json** — 208 total files across 11 subsystems, verified against Job 5 output.
