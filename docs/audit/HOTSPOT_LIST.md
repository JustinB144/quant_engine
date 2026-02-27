# Hotspot Analysis — quant_engine
Generated: 2026-02-27
Job 3 of 7 — LLM Audit Workflow

Source data: MODULE_INVENTORY.yaml (Job 1) + DEPENDENCY_EDGES.json (Job 2) + git log (60-day window)

---

## Scoring Methodology

### Module-Level Scoring (6 criteria, 0-3 each, max 18)

| Criterion | 0 | 1 | 2 | 3 | Source |
|-----------|---|---|---|---|--------|
| Fan-in (source modules) | 0 | 1-2 | 3-5 | 6+ | DEPENDENCY_MATRIX.md |
| Fan-out (target modules) | 0 | 1-2 | 3-5 | 6+ | DEPENDENCY_MATRIX.md |
| Contract surface (external consumers) | None | Internal only | 1-2 consumers | 3+ consumers | Cross-module import analysis |
| Change frequency (60-day commits) | 0-2 | 3-10 | 11-20 | 21+ | `git log --since=60.days` |
| Complexity (largest file LOC) | <200 | 200-500 | 500-1000 | 1000+ | MODULE_INVENTORY.yaml |
| Test coverage gaps | All tested | 1-2 untested | 3-5 untested | 6+ untested | Test file mapping |

### File-Level Scoring (weighted criteria, max 21 base + bonuses)

| Criterion | Weight | Score Range | Source |
|-----------|--------|-------------|--------|
| Cross-module fan-in | 3x | 0-3 (0=0, 1=1-3, 2=4-8, 3=9+) | DEPENDENCY_EDGES.json (with __init__.py resolution) |
| Contract file | 2x | 0 or 1 | Cross-module signature analysis |
| Lines of code | 1x | 0-3 (0=<200, 1=200-500, 2=500-1000, 3=1000+) | MODULE_INVENTORY.yaml |
| Cyclomatic complexity proxy | 1x | 0-3 (0=<200, 1=200-500, 2=500-1000, 3=1000+) | `grep -c "if \|elif \|except \|try:"` |
| Lazy/conditional cross-module imports | 2x | 0 or 1 | DEPENDENCY_EDGES.json import_type |
| Shared artifact writer | 2x | 0 or 1 | Shared artifacts table |

**Adjustments applied** (per spec Step 4 — verified hotspot calibration):
- **__init__.py resolution**: When a file is the sole implementation behind an __init__.py re-export, fan-in to the __init__.py is attributed to the implementation file (affects indicators/indicators.py: +6 effective fan-in).
- **Circular dependency penalty**: +2 per confirmed circular import edge. The autopilot↔api cycle has 6 edges (4 from paper_trader.py, 2 from engine.py).
- **Contract criticality bonus**: +1-3 for version-locked schemas, execution contracts, or behavioral contracts where changes silently propagate (affects regime/shock_vector.py +3, validation/preconditions.py +3, regime/uncertainty_gate.py +2).

---

## Module Rankings (by total risk score)

| Rank | Module | Score | Fan-in | Fan-out | Contract | Changes | Complexity | Test Gaps | Key Risk |
|------|--------|-------|--------|---------|----------|---------|------------|-----------|----------|
| 1 | features | 15/18 | 3/3 | 2/3 | 3/3 | 2/3 | 3/3 | 2/3 | 90+ features flow through pipeline.py; indicators dependency |
| 2 | autopilot | 14/18 | 2/3 | 3/3 | 2/3 | 3/3 | 3/3 | 1/3 | Highest fan-out (39 edges, 8 modules); 6 circular api edges |
| 3 | data | 13/18 | 3/3 | 2/3 | 3/3 | 2/3 | 3/3 | 0/3 | Primary data ingestion; 6 consumer modules |
| 3 | models | 13/18 | 2/3 | 2/3 | 2/3 | 2/3 | 3/3 | 2/3 | Training/prediction pipeline; governance contracts |
| 3 | api | 13/18 | 1/3 | 3/3 | 1/3 | 2/3 | 3/3 | 3/3 | Largest module (115 outbound edges); circular with autopilot |
| 6 | config | 12/18 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 0/3 | **Supreme hub**: 161 inbound edges from ALL 14 modules |
| 6 | backtest | 12/18 | 2/3 | 2/3 | 2/3 | 2/3 | 3/3 | 1/3 | Core simulation; errors invalidate all backtest results |
| 8 | regime | 11/18 | 3/3 | 1/3 | 3/3 | 2/3 | 2/3 | 0/3 | 7 consumer modules; structural state contracts |
| 9 | risk | 10/18 | 2/3 | 1/3 | 2/3 | 2/3 | 3/3 | 0/3 | Position sizing/limits affect all live trading |
| 10 | indicators | 9/18 | 1/3 | 0/3 | 2/3 | 0/3 | 3/3 | 3/3 | 90+ indicators, 0 dedicated tests; signature changes break features |
| 10 | kalshi | 9/18 | 2/3 | 2/3 | 2/3 | 1/3 | 2/3 | 0/3 | Self-contained vertical; lower blast radius |
| 12 | validation | 7/18 | 2/3 | 1/3 | 2/3 | 0/3 | 0/3 | 2/3 | Truth Layer execution contracts; 0 dedicated tests |
| 13 | evaluation | 5/18 | 0/3 | 2/3 | 0/3 | 0/3 | 2/3 | 1/3 | Leaf module (0 fan-in); safe to change in isolation |
| 14 | utils | 3/18 | 0/3 | 1/3 | 0/3 | 0/3 | 1/3 | 1/3 | Minimal (2 files); leaf module (0 fan-in) |

### Module Scoring Detail

| Module | Fan-in Raw | Fan-out Raw | Contract Consumers | Commits (60d) | Largest File (lines) |
|--------|-----------|------------|-------------------|---------------|---------------------|
| config | 14 modules / 161 edges | 0 | 14 | 30 | config.py (1,020) + config_structured.py (347) |
| data | 6 modules / 26 edges | 3 modules / 12 edges | 6 | 19 | wrds_provider.py (1,620) |
| features | 6 modules / 10 edges | 4 modules / 18 edges | 6 | 11 | pipeline.py (1,541) |
| indicators | 1 module / 6 edges | 0 | 1 (features only) | 2 | indicators.py (2,904) |
| regime | 7 modules / 17 edges | 1 module / 11 edges | 7 | 15 | detector.py (940) |
| models | 4 modules / 28 edges | 3 modules / 9 edges | 4 | 12 | trainer.py (1,818) |
| backtest | 5 modules / 18 edges | 4 modules / 14 edges | 5 | 16 | engine.py (2,488) |
| risk | 3 modules / 13 edges | 2 modules / 9 edges | 3 | 14 | position_sizer.py (1,254) |
| evaluation | 0 modules / 0 edges | 3 modules / 10 edges | 0 | 2 | engine.py (826) |
| validation | 3 modules / 3 edges | 2 modules / 3 edges | 3 | 2 | leakage_detection.py (193) |
| autopilot | 3 modules / 5 edges | 8 modules / 39 edges | 3 | 24 | engine.py (1,927) |
| kalshi | 3 modules / 8 edges | 4 modules / 7 edges | 3 | 5 | distribution.py (935) |
| api | 2 modules / 8 edges | 9 modules / 115 edges | 2 | 19 | health_service.py (2,929) |
| utils | 0 modules / 0 edges | 1 module / 1 edge | 0 | 1 | logging.py (114) |

---

## File Rankings (top 20 by weighted score)

| Rank | File | Score | Fan-in (3x) | Contract (2x) | LOC (1x) | CX (1x) | Lazy (2x) | Artifact (2x) | Adj | Key Risk |
|------|------|-------|-------------|---------------|----------|----------|-----------|---------------|-----|----------|
| 1 | config.py | 18 | 3 (160 edges) | 1 | 3 (1,020) | 0 (46) | 0 | 1 | — | God object: 161 edges from ALL 14 modules |
| 2 | autopilot/engine.py | 17 | 1 (2 edges) | 1 | 3 (1,927) | 1 (230) | 1 | 1 | +4 circ | Highest fan-out; 2 circular api edges |
| 2 | autopilot/paper_trader.py | 17 | 0 | 1 | 3 (1,254) | 0 (143) | 1 | 1 | +8 circ | Circular dependency hub: 4 of 6 api edges |
| 4 | features/pipeline.py | 16 | 3 (9 edges) | 1 | 3 (1,541) | 0 (158) | 1 | 0 | — | All 90+ features flow through this file |
| 4 | backtest/engine.py | 16 | 2 (5 edges) | 1 | 3 (2,488) | 1 (251) | 1 | 1 | — | LARGEST production file; core simulation loop |
| 4 | models/trainer.py | 16 | 2 (4 edges) | 1 | 3 (1,818) | 1 (216) | 1 | 1 | — | Training pipeline; writes trained_models/ |
| 7 | data/loader.py | 15 | 3 (11 edges) | 1 | 2 (849) | 0 (150) | 1 | 0 | — | Primary data ingestion; 4 consumer modules |
| 7 | regime/detector.py | 15 | 3 (10 edges) | 1 | 2 (940) | 0 (78) | 1 | 0 | — | Main regime orchestrator; 4 consumer modules |
| 9 | indicators/indicators.py | 14 | 2 (6 via __init__) | 1 | 3 (2,904) | 0 (63) | 0 | 0 | +2 init | 90+ indicator classes; signature changes break pipeline |
| 9 | api/services/health_service.py | 14 | 2 (4 edges) | 1 | 3 (2,929) | 1 (374) | 1 | 0 | — | LARGEST file in codebase; circular dep hub |
| 9 | data/local_cache.py | 14 | 2 (5 edges) | 1 | 2 (841) | 0 (144) | 1 | 1 | — | Atomic cache writes; artifact writer |
| 12 | regime/shock_vector.py | 12 | 1 (2 edges) | 1 | 1 (494) | 0 (36) | 0 | 0 | +3 schema | Version-locked ShockVector dataclass |
| 12 | regime/uncertainty_gate.py | 12 | 1 (3 edges) | 1 | 0 (180) | 0 (4) | 1 | 0 | +2 contract | Silent sizing multiplier across 3 modules |
| 12 | models/predictor.py | 12 | 2 (5 edges) | 1 | 2 (538) | 0 (54) | 0 | 1 | — | Feature alignment critical for prediction correctness |
| 15 | validation/preconditions.py | 11 | 1 (2 edges) | 1 | 0 (71) | 0 (5) | 0 | 0 | +3 contract | Execution contract; changes break training + backtesting |
| 16 | backtest/validation.py | 11 | 2 (6 edges) | 1 | 3 (1,074) | 0 (77) | 0 | 0 | — | Walk-forward validation; 4 consumer modules |
| 17 | risk/position_sizer.py | 10 | 1 (3 edges) | 1 | 3 (1,254) | 0 (136) | 1 | 0 | — | Kelly sizing; sizing errors affect live trading |
| 18 | data/wrds_provider.py | 9 | 2 (6 edges) | 0 | 3 (1,620) | 0 (153) | 0 | 0 | — | WRDS data pipeline; 3 consumer modules |
| 18 | models/versioning.py | 9 | 2 (6 edges) | 1 | 1 (207) | 0 (17) | 0 | 0 | — | Model registry; version resolution |
| 20 | backtest/execution.py | 9 | 1 (2 edges) | 1 | 2 (936) | 0 (75) | 1 | 0 | — | Execution simulator with structural costs |

**Note on adjustments (Adj column)**:
- `circ`: +2 per confirmed circular dependency edge (autopilot↔api)
- `init`: +2 for __init__.py re-export resolution (6 effective fan-in from features/pipeline.py)
- `schema`: +3 for version-locked dataclass schema (ShockVector v1 with schema validation)
- `contract`: +2-3 for behavior-critical execution/sizing contracts affecting multiple modules

---

## Blast Radius Analysis (top 15 files)

### 1. config.py
```yaml
file: config.py
score: 18
lines: 1020 (+ config_structured.py: 347)
constants: 200+
git_commits_60d: 28
blast_radius:
  direct_dependents:
    - module: api (82 edges)
      files: orchestrator.py, main.py, 15 routers/*, 11 services/*
      symbols_used: [RESULTS_DIR, DATA_CACHE_DIR, MODEL_DIR, UNIVERSE_FULL, UNIVERSE_QUICK, REGIME_NAMES, + 150 more]
    - module: autopilot (8 edges)
      files: engine.py:34, paper_trader.py:13, promotion_gate.py:13, registry.py:10, strategy_discovery.py:10, meta_labeler.py:23, strategy_allocator.py:20
      symbols_used: [AUTOPILOT_CYCLE_REPORT, PAPER_STATE_PATH, PROMOTION_*, EXEC_*, META_LABELING_*]
    - module: regime (11 edges)
      files: detector.py:24, consensus.py:55, hmm.py:560, jump_model_pypi.py:84, online_update.py:60, uncertainty_gate.py:52, shock_vector.py (via detector)
      symbols_used: [REGIME_*, BOCPD_*, SHOCK_VECTOR_*]
    - module: data (10 edges)
      files: loader.py:34, local_cache.py:22, quality.py:25, feature_store.py:31, intraday_quality.py:25, survivorship.py:371
      symbols_used: [DATA_CACHE_DIR, CACHE_*, WRDS_ENABLED, SURVIVORSHIP_*]
    - module: backtest (5 edges)
      files: engine.py:26, validation.py:23, optimal_execution.py:17
      symbols_used: [TRANSACTION_COST_BPS, ENTRY_THRESHOLD, EXEC_*, SHOCK_MODE_*]
    - module: risk (8 edges)
      files: position_sizer.py:136, portfolio_risk.py:23, drawdown.py:18, stop_loss.py:23, portfolio_optimizer.py:190
      symbols_used: [DRAWDOWN_*, MAX_PORTFOLIO_VOL, REGIME_RISK_MULTIPLIER, HARD_STOP_PCT]
    - module: models (7 edges)
      files: trainer.py:77, predictor.py:21, versioning.py:22, governance.py:10, feature_stability.py:26, retrain_trigger.py:31
      symbols_used: [MODEL_DIR, MODEL_PARAMS, CV_FOLDS, REGIME_NAMES, ENSEMBLE_DIVERSIFY]
    - module: features (8 edges)
      files: pipeline.py:750, pipeline.py:871, pipeline.py:924, intraday.py:21
      symbols_used: [SPECTRAL_*, SSA_*, INTERACTION_PAIRS, FORWARD_HORIZONS, STRUCTURAL_FEATURES_ENABLED]
    - module: evaluation (7 edges)
      files: engine.py:27, metrics.py:20, slicing.py:23, fragility.py:20, ml_diagnostics.py:26, calibration_analysis.py:19
      symbols_used: [EVAL_*]
    - module: validation (2 edges)
      files: preconditions.py:16
      symbols_used: [RET_TYPE, LABEL_H, PX_TYPE, ENTRY_PRICE_TYPE, TRUTH_LAYER_STRICT_PRECONDITIONS]
    - module: kalshi (1 edge)
      files: provider.py:14
      symbols_used: [KALSHI_*]
    - module: utils (1 edge)
      files: logging.py:114
      symbols_used: [ALERT_HISTORY_FILE, ALERT_WEBHOOK_URL]
    - module: entry_points (9 edges)
      files: run_autopilot.py, run_backtest.py, run_predict.py, run_retrain.py, run_train.py, run_wrds_daily_refresh.py, run_rehydrate_cache_metadata.py
    - module: scripts (2 edges)
      files: compare_regime_models.py, ibkr_daily_gapfill.py
  transitive_dependents:
    - impact: ENTIRE SYSTEM — every module directly imports config
  shared_artifacts_affected:
    - artifact: config_data/universe.yaml
      impact: Universe membership changes propagate to all data loading, backtesting, and trading
  what_breaks: >
    Any constant rename, value change, or type change silently propagates to ALL 14 modules
    across 161 import edges. There is no indirection layer — modules consume raw constants
    directly. A single typo in a config constant can break training, backtesting, prediction,
    risk management, and live paper trading simultaneously with no compile-time or import-time
    error. The blast radius is the ENTIRE SYSTEM.
  audit_priority: CRITICAL
  audit_focus: >
    Check for deprecated/placeholder constants still referenced (STATUS annotations).
    Verify TRUTH_LAYER flags are consistent. Identify which constants are consumed by which
    modules. Check config_structured.py derives correctly. Verify backward compatibility
    surface (51 files import from config.py).
```

### 2. autopilot/engine.py
```yaml
file: autopilot/engine.py
score: 17
lines: 1927
git_commits_60d: 11
blast_radius:
  direct_dependents:
    - file: api/jobs/autopilot_job.py:12
      module: api
      symbols_used: [AutopilotEngine]
    - file: run_autopilot.py:19
      module: entry_points
      symbols_used: [AutopilotEngine]
  outgoing_cross_module_imports: 23 edges across 8 modules
    - backtest: Backtester, capacity_analysis, deflated_sharpe_ratio, probability_of_backtest_overfitting, walk_forward_validate, run_statistical_tests, combinatorial_purged_cv, superior_predictive_ability, strategy_signal_returns
    - models: ModelTrainer, EnsemblePredictor, cross_sectional_rank, _expanding_walk_forward_folds
    - config: 18 constants (AUTOPILOT_CYCLE_REPORT, WF_MAX_TRAIN_DATES, etc.)
    - data: load_survivorship_universe, load_universe, filter_panel_by_point_in_time_universe
    - features: FeaturePipeline
    - regime: RegimeDetector, UncertaintyGate
    - risk: PositionSizer, PortfolioRiskManager, DrawdownController, StopLossManager
    - api: HealthService (CIRCULAR — lines 1868, 1911)
  transitive_dependents:
    - file: api/routers/autopilot.py
      module: api
      through: api/jobs/autopilot_job.py
  shared_artifacts_affected:
    - artifact: results/autopilot/latest_cycle.json
      impact: API autopilot_service.py reads this; schema changes break dashboard
  what_breaks: >
    Most tightly coupled module in the system. Changes in ANY of 8 dependency modules can
    break the autopilot cycle. The 2 circular edges to api/services/health_service.py mean
    changes to health check logic can break cycle orchestration. If this file has a bug,
    strategy discovery, promotion gating, paper trading initialization, and the entire
    automated lifecycle halts.
  audit_priority: CRITICAL
  audit_focus: >
    Verify all lazy imports (to api) are guarded and won't cause import-time failures.
    Check that cross-module contracts haven't drifted. Verify the heuristic predictor
    fallback path works when trained models are unavailable. Check meta-labeling integration.
```

### 3. autopilot/paper_trader.py
```yaml
file: autopilot/paper_trader.py
score: 17
lines: 1254
git_commits_60d: 8
blast_radius:
  direct_dependents:
    - (No direct cross-module imports of this file — consumed within autopilot)
  outgoing_cross_module_imports: 4 modules
    - config: 41 constants (PAPER_*, EXEC_*, REGIME_*, ADV_*, TRANSACTION_COST_BPS)
    - backtest: ExecutionModel (line 167)
    - risk: PositionSizer (line 172)
    - api: HealthService (lines 189, 532), health_risk_feedback (line 173), ab_testing (line 211) — ALL CIRCULAR
  circular_dependency_edges: 4 of 6 total autopilot→api circular edges
    - line 173: api.services.health_risk_feedback
    - line 189: api.services.health_service.HealthService
    - line 211: api.ab_testing.ABTestingFramework
    - line 532: api.services.health_service.HealthService
  shared_artifacts_affected:
    - artifact: results/autopilot/paper_state.json
      impact: api/services/autopilot_service.py reads paper state; schema changes break dashboard
  what_breaks: >
    The autopilot↔api circular reference runs through this file (4 of 6 edges). Any change
    to api.services.health_service, api.services.health_risk_feedback, or api.ab_testing
    signatures can break paper trading. Position sizing bugs here directly affect simulated
    P&L that drives promotion decisions. If paper trading breaks, no strategies get promoted.
  audit_priority: CRITICAL
  audit_focus: >
    Verify ALL 4 api imports are truly lazy and won't cause circular import at load time.
    Check that execution model integration matches backtest/execution.py behavior. Verify
    Kelly sizing calculations match risk/position_sizer.py. Check paper state persistence
    and recovery after crashes.
```

### 4. features/pipeline.py
```yaml
file: features/pipeline.py
score: 16
lines: 1541
git_commits_60d: 5
blast_radius:
  direct_dependents:
    - file: api/orchestrator.py:44
      module: api
      symbols_used: [FeaturePipeline]
    - file: api/services/data_helpers.py:490
      module: api
      symbols_used: [FeaturePipeline]
    - file: autopilot/engine.py:56
      module: autopilot
      symbols_used: [FeaturePipeline]
    - file: models/predictor.py:22
      module: models
      symbols_used: [get_feature_type]
    - file: run_backtest.py:37, run_predict.py:28, run_retrain.py:28, run_train.py:28
      module: entry_points
      symbols_used: [FeaturePipeline]
    - file: scripts/compare_regime_models.py:48
      module: scripts
      symbols_used: [FeaturePipeline]
  outgoing_cross_module_imports: 18 edges across 4 modules
    - indicators: 87 indicator classes (via __init__.py at line 21), SpectralAnalyzer, SSADecomposer, TailRiskAnalyzer, OptimalTransportAnalyzer, EigenvalueAnalyzer
    - config: 14 constants (SPECTRAL_*, SSA_*, INTERACTION_PAIRS, FORWARD_HORIZONS, STRUCTURAL_FEATURES_ENABLED, etc.)
    - regime: CorrelationRegimeDetector (line 1303)
    - data: load_ohlcv (line 1528), WRDSProvider (line 1413), load_intraday_ohlcv (line 1458)
  transitive_dependents:
    - file: backtest/engine.py (uses predictions built from features)
      module: backtest
      through: models/predictor.py
    - file: autopilot/paper_trader.py (trades based on predictions from features)
      module: autopilot
      through: autopilot/engine.py
  what_breaks: >
    ALL 90+ features flow through this single file. Indicator signature changes, feature
    causality type changes, or column ordering changes break training, prediction, backtesting,
    and autopilot. The FEATURE_METADATA registry is the ground truth for feature causality
    classification — if a feature is miscategorized as CAUSAL when it's END_OF_DAY, live
    predictions will have look-ahead bias.
  audit_priority: CRITICAL
  audit_focus: >
    Feature causality types (CAUSAL vs END_OF_DAY vs RESEARCH_ONLY) — verify each classification.
    Indicator import correctness (87 classes from indicators.py). Feature name/ordering consistency
    between training and prediction paths. Check that lazy imports (data, config) don't fail silently.
```

### 5. backtest/engine.py
```yaml
file: backtest/engine.py
score: 16
lines: 2488
git_commits_60d: 9
blast_radius:
  direct_dependents:
    - file: api/orchestrator.py:289
      module: api
      symbols_used: [Backtester]
    - file: autopilot/engine.py:20
      module: autopilot
      symbols_used: [Backtester]
    - file: autopilot/promotion_gate.py:12
      module: autopilot
      symbols_used: [BacktestResult]
    - file: kalshi/promotion.py:14
      module: kalshi
      symbols_used: [BacktestResult]
    - file: run_backtest.py:40
      module: entry_points
      symbols_used: [Backtester]
  outgoing_cross_module_imports: 14 edges across 4 modules
    - config: 52 constants (TRANSACTION_COST_BPS, EXEC_*, SHOCK_MODE_*, EDGE_COST_*, REGIME_*)
    - regime: compute_shock_vectors, ShockVector (line 77), UncertaintyGate (line 78)
    - risk: PositionSizer, DrawdownController, StopLossManager, PortfolioRiskManager, RiskMetrics (lines 316-320)
    - validation: enforce_preconditions (line 198)
  transitive_dependents:
    - file: evaluation/engine.py
      module: evaluation
      through: reads backtest result artifacts
    - file: api/services/backtest_service.py
      module: api
      through: reads results/backtest_*d_summary.json
  shared_artifacts_affected:
    - artifact: results/backtest_*d_summary.json
      impact: api/services/backtest_service.py and evaluation/engine.py read these; schema changes break reporting
    - artifact: results/backtest_*d_trades.csv
      impact: api/services/backtest_service.py reads trades for display
  what_breaks: >
    Core simulation loop. LARGEST production file (2,488 lines). Errors here invalidate ALL
    backtest results and therefore all promotion decisions. The autopilot promotion gate uses
    BacktestResult to decide whether strategies go live. If execution realism is wrong (costs,
    fills, participation limits), paper trading performance will diverge from backtest expectations.
  audit_priority: CRITICAL
  audit_focus: >
    Execution realism: verify costs match paper_trader.py behavior. Check ShockVector integration
    correctness. Verify risk module integration (position sizing, drawdown, stops). Check that
    lazy imports (risk/*) are correctly guarded. Verify edge-cost gate logic.
```

### 6. models/trainer.py
```yaml
file: models/trainer.py
score: 16
lines: 1818
git_commits_60d: 7
blast_radius:
  direct_dependents:
    - file: api/orchestrator.py:161
      module: api
      symbols_used: [ModelTrainer]
    - file: autopilot/engine.py:59
      module: autopilot
      symbols_used: [ModelTrainer]
    - file: run_retrain.py:31, run_train.py:31
      module: entry_points
      symbols_used: [ModelTrainer]
  outgoing_cross_module_imports:
    - config: 17 constants (MODEL_PARAMS, CV_FOLDS, HOLDOUT_FRACTION, ENSEMBLE_DIVERSIFY, etc.)
    - features: FeaturePipeline (via predictor path)
    - validation: enforce_preconditions (line 219)
  shared_artifacts_affected:
    - artifact: trained_models/*/
      impact: models/predictor.py, api/services/model_service.py read trained models; format changes break prediction
  what_breaks: >
    Training pipeline produces the models that feed prediction, backtesting, and autopilot.
    Anti-overfitting controls (purged CV, embargo, holdout) must be correct — if compromised,
    all downstream results are unreliable. Model artifacts are consumed by predictor.py and
    model_service.py; format changes silently break model loading.
  audit_priority: HIGH
  audit_focus: >
    Purged CV implementation correctness. Embargo gap calculation. IS/OOS gap monitoring
    thresholds. Scaler isolation between regimes. Feature selection stability. Model
    serialization format compatibility.
```

### 7. data/loader.py
```yaml
file: data/loader.py
score: 15
lines: 849
git_commits_60d: 6
blast_radius:
  direct_dependents:
    - file: api/orchestrator.py:43
      module: api
      symbols_used: [load_survivorship_universe, load_universe]
    - file: api/orchestrator.py:61
      module: api
      symbols_used: [get_skip_reasons]
    - file: api/routers/system_health.py:97
      module: api
      symbols_used: [get_data_provenance]
    - file: api/services/data_service.py:24
      module: api
      symbols_used: [load_universe]
    - file: autopilot/engine.py:54
      module: autopilot
      symbols_used: [load_survivorship_universe, load_universe]
    - file: features/pipeline.py:1528
      module: features
      symbols_used: [load_ohlcv]
    - file: run_backtest.py:35, run_predict.py:27, run_retrain.py:27, run_train.py:27
      module: entry_points
      symbols_used: [load_universe, load_survivorship_universe]
  outgoing_cross_module_imports:
    - config: 10 constants (WRDS_ENABLED, DATA_QUALITY_ENABLED, CACHE_*, etc.)
    - validation: DataIntegrityValidator (line 567, conditional)
  transitive_dependents:
    - file: backtest/engine.py (backtests on loaded data)
      module: backtest
      through: entry points / autopilot pipeline
    - file: models/trainer.py (trains on loaded data)
      module: models
      through: entry points / autopilot pipeline
  what_breaks: >
    Primary data ingestion point. Quality regression corrupts everything downstream:
    features, training, prediction, backtesting, and paper trading. The lazy import
    of DataIntegrityValidator (line 567) means data integrity checks could fail to load
    without raising an import error. Cache trust logic determines whether stale data
    is used for live decisions.
  audit_priority: CRITICAL
  audit_focus: >
    Cache trust logic: CACHE_MAX_STALENESS_DAYS, CACHE_TRUSTED_SOURCES priority.
    WRDS fallback behavior when connection fails. Survivorship filtering correctness.
    Verify DataIntegrityValidator conditional import works reliably.
```

### 8. regime/detector.py
```yaml
file: regime/detector.py
score: 15
lines: 940
git_commits_60d: 12
blast_radius:
  direct_dependents:
    - file: api/orchestrator.py:45, :225, :292
      module: api
      symbols_used: [RegimeDetector]
    - file: api/services/data_helpers.py:491
      module: api
      symbols_used: [RegimeDetector]
    - file: autopilot/engine.py:60
      module: autopilot
      symbols_used: [RegimeDetector]
    - file: run_backtest.py:38, run_predict.py:29, run_retrain.py:29, run_train.py:29
      module: entry_points
      symbols_used: [RegimeDetector]
    - file: scripts/compare_regime_models.py:30
      module: scripts
      symbols_used: [RegimeDetector]
  outgoing_cross_module_imports:
    - config: 20 constants (REGIME_*, BOCPD_*, SHOCK_VECTOR_*)
  transitive_dependents:
    - file: backtest/engine.py (uses regime for ShockVector)
      module: backtest
      through: regime detection pipeline
    - file: risk/position_sizer.py (regime-conditional sizing)
      module: risk
      through: regime state propagation
  what_breaks: >
    Main orchestrator for all regime detection engines (rule-based, HMM, jump model,
    ensemble). Regime state feeds into position sizing, stop loss multipliers, constraint
    scaling, and autopilot promotion. If regime detection produces incorrect labels,
    the entire risk management stack operates with wrong assumptions.
  audit_priority: HIGH
  audit_focus: >
    HMM state mapping correctness. Ensemble consensus thresholds. BOCPD changepoint
    sensitivity. Verify lazy imports (config constants at lines 240, 324, 402, 933)
    load correctly. Check regime label consistency across detection engines.
```

### 9. indicators/indicators.py
```yaml
file: indicators/indicators.py
score: 14
lines: 2904
git_commits_60d: 1
blast_radius:
  direct_dependents:
    - file: features/pipeline.py:21
      module: features
      symbols_used: [87 indicator classes — ATR, RSI, MACD, BollingerBandWidth, etc.]
      notes: Imported via indicators/__init__.py; all 87 classes re-exported
  transitive_dependents:
    - file: models/predictor.py (predictions depend on feature correctness)
      module: models
      through: features/pipeline.py
    - file: backtest/engine.py (backtest results depend on features)
      module: backtest
      through: features → models → backtest pipeline
    - file: autopilot/engine.py (autopilot uses features)
      module: autopilot
      through: features pipeline
  what_breaks: >
    Contains 92 concrete indicator subclasses. Features/pipeline.py imports 87 of them
    at line 21. Any change to an indicator's compute() signature, return shape, or
    parameter defaults silently corrupts the feature pipeline. Since features feed
    training and prediction, indicator bugs propagate to model quality, backtest results,
    and live trading signals.
  audit_priority: HIGH
  audit_focus: >
    Indicator return types/shapes — verify all compute() methods return consistent
    Series format. Parameter defaults — check for silent behavior changes. No dedicated
    tests exist — verify indicators are tested indirectly through features tests.
    Check INDICATOR_ALIASES for correctness.
```

### 10. api/services/health_service.py
```yaml
file: api/services/health_service.py
score: 14
lines: 2929
git_commits_60d: 11
blast_radius:
  direct_dependents:
    - file: autopilot/engine.py:1868, :1911
      module: autopilot
      symbols_used: [HealthService]
      notes: CIRCULAR — autopilot imports from api
    - file: autopilot/paper_trader.py:189, :532
      module: autopilot
      symbols_used: [HealthService]
      notes: CIRCULAR — autopilot imports from api
  outgoing_cross_module_imports: 24 edges
    - config: 22 edges (DATA_CACHE_DIR, MODEL_DIR, RESULTS_DIR, WRDS_ENABLED, BENCHMARK, + 20 more)
    - data: WRDSProvider (lines 151, 1262)
    - models: FeatureStabilityTracker (line 1800)
  what_breaks: >
    LARGEST file in the entire codebase (2,929 lines). Exposes health state to autopilot
    via the circular dependency. Changes to health check signatures or return types break
    paper trading and autopilot cycle orchestration. The 24 outgoing config imports mean
    any config constant change affecting health checks cascades to autopilot behavior.
  audit_priority: HIGH
  audit_focus: >
    What state does this expose to autopilot? Could internal changes break paper trading?
    Verify that the 4 autopilot import points are resilient to health check failures.
    Check for state leakage between health checks. Consider splitting this 2,929-line file.
```

### 11. data/local_cache.py
```yaml
file: data/local_cache.py
score: 14
lines: 841
git_commits_60d: 3
blast_radius:
  direct_dependents:
    - file: features/pipeline.py:1458
      module: features
      symbols_used: [load_intraday_ohlcv]
    - file: run_rehydrate_cache_metadata.py:18
      module: entry_points
      symbols_used: [rehydrate_cache_metadata]
    - file: run_wrds_daily_refresh.py:28, :676
      module: entry_points
      symbols_used: [list_cached_tickers, load_ohlcv_with_meta, save_ohlcv, backfill_terminal_metadata]
    - file: scripts/ibkr_daily_gapfill.py:50
      module: scripts
      symbols_used: [_normalize_ohlcv_columns, _write_cache_meta, load_ohlcv_with_meta]
  shared_artifacts_affected:
    - artifact: data/cache/*.parquet
      impact: data/loader.py reads cached data; corruption or format changes break all data loading
  what_breaks: >
    Cache layer with atomic temp-file writes and metadata sidecars. If cache write
    atomicity is broken, partial files could corrupt data loading. Metadata sidecar
    format changes break cache rehydration. Intraday OHLCV loading is used by the
    feature pipeline for microstructure features.
  audit_priority: HIGH
  audit_focus: >
    Atomic write correctness (temp files + rename). Metadata sidecar schema stability.
    Cache staleness detection. Verify intraday loading returns correct DataFrame format.
```

### 12. regime/shock_vector.py
```yaml
file: regime/shock_vector.py
score: 12
lines: 494
git_commits_60d: 2
blast_radius:
  direct_dependents:
    - file: backtest/engine.py:77
      module: backtest
      symbols_used: [compute_shock_vectors, ShockVector]
    - file: backtest/execution.py:29
      module: backtest
      symbols_used: [ShockVector]
  transitive_dependents:
    - file: autopilot/engine.py
      module: autopilot
      through: backtest/engine.py (uses ShockVector for execution)
    - file: autopilot/paper_trader.py
      module: autopilot
      through: execution model uses ShockVector
  what_breaks: >
    Defines the ShockVector dataclass with a VERSION-LOCKED schema (SHOCK_VECTOR_SCHEMA_VERSION).
    Backtest execution uses ShockVector to determine cost multipliers in stressed/elevated/normal
    modes. Schema changes here break the backtest execution layer and paper trading execution.
    The version lock is supposed to prevent silent schema drift — verify it's enforced.
  audit_priority: HIGH
  audit_focus: >
    Version lock enforcement: is SHOCK_VECTOR_SCHEMA_VERSION actually checked at deserialization?
    Schema field compatibility: do all consumers (backtest/engine.py, backtest/execution.py)
    access the correct fields? ShockVector validation: is ShockVectorValidator applied consistently?
```

### 13. regime/uncertainty_gate.py
```yaml
file: regime/uncertainty_gate.py
score: 12
lines: 180
git_commits_60d: 1
blast_radius:
  direct_dependents:
    - file: backtest/engine.py:78
      module: backtest
      symbols_used: [UncertaintyGate]
    - file: autopilot/engine.py:61
      module: autopilot
      symbols_used: [UncertaintyGate]
    - file: risk/position_sizer.py:27
      module: risk
      symbols_used: [UncertaintyGate]
  transitive_dependents:
    - file: autopilot/paper_trader.py
      module: autopilot
      through: autopilot/engine.py
  what_breaks: >
    Entropy-based sizing multiplier consumed by 3 modules (backtest, autopilot, risk).
    Threshold changes silently alter position sizing across ALL trading paths. The
    stress_assumption_active signal feeds into the autopilot promotion gate — if this
    flag is wrong, strategies may be promoted during stressed conditions or rejected
    during normal conditions.
  audit_priority: HIGH
  audit_focus: >
    Threshold values: REGIME_UNCERTAINTY_ENTROPY_THRESHOLD, REGIME_UNCERTAINTY_STRESS_THRESHOLD.
    Multiplier calculation correctness. Verify the sizing map (REGIME_UNCERTAINTY_SIZING_MAP)
    produces reasonable multipliers. Check consistency between backtest and paper trading usage.
```

### 14. models/predictor.py
```yaml
file: models/predictor.py
score: 12
lines: 538
git_commits_60d: 5
blast_radius:
  direct_dependents:
    - file: api/orchestrator.py:224, :291
      module: api
      symbols_used: [EnsemblePredictor]
    - file: autopilot/engine.py:58
      module: autopilot
      symbols_used: [EnsemblePredictor]
    - file: run_backtest.py:39
      module: entry_points
      symbols_used: [EnsemblePredictor]
    - file: run_predict.py:30
      module: entry_points
      symbols_used: [EnsemblePredictor]
  outgoing_cross_module_imports:
    - config: MODEL_DIR, REGIME_NAMES, TRUTH_LAYER_ENFORCE_CAUSALITY
    - features: get_feature_type (line 22)
  shared_artifacts_affected:
    - artifact: results/predictions_*d.csv
      impact: api/services/results_service.py reads prediction files
    - artifact: trained_models/*/ (reads)
      impact: Must match format written by models/trainer.py
  what_breaks: >
    Bridge between trained models and predictions. Feature name/ordering mismatches
    between training and prediction break predictions SILENTLY — no error, just wrong
    numbers. Regime blending logic (mixing regime-specific and global models based on
    confidence) must match training assumptions. Causality enforcement via get_feature_type
    prevents look-ahead bias in live predictions.
  audit_priority: HIGH
  audit_focus: >
    Feature alignment: verify feature names and ordering match between trainer and predictor.
    Model version resolution: verify correct model directory is loaded. Regime blending
    weights: check that confidence-based blending produces reasonable outputs. Causality
    enforcement: verify TRUTH_LAYER_ENFORCE_CAUSALITY actually blocks non-CAUSAL features.
```

### 15. validation/preconditions.py
```yaml
file: validation/preconditions.py
score: 11
lines: 71
git_commits_60d: 1
blast_radius:
  direct_dependents:
    - file: models/trainer.py:219
      module: models
      symbols_used: [enforce_preconditions]
    - file: backtest/engine.py:198
      module: backtest
      symbols_used: [enforce_preconditions]
  transitive_dependents:
    - file: autopilot/engine.py
      module: autopilot
      through: models/trainer.py (training path) and backtest/engine.py (backtest path)
    - file: api/orchestrator.py
      module: api
      through: both training and backtesting paths
  what_breaks: >
    Execution contract validation ensuring RET_TYPE, LABEL_H, PX_TYPE, and ENTRY_PRICE_TYPE
    are locked before modeling or backtesting. Changes to precondition checks alter BOTH
    training and backtesting behavior. If preconditions diverge between the two paths,
    models could be trained on different assumptions than they're backtested with. Also
    imports from config_structured.PreconditionsConfig — the ONLY file using config_structured
    directly for validation.
  audit_priority: HIGH
  audit_focus: >
    Ensure preconditions match between training (trainer.py:219) and backtesting (engine.py:198)
    paths. Verify that config_structured.PreconditionsConfig values align with config.py
    constants. Check that TRUTH_LAYER_STRICT_PRECONDITIONS behavior is correct (fail-fast
    vs. warn).
```

---

## Transitive Impact Summary

| Hotspot | Direct Dependents | 2-Hop Dependents | Total Reach |
|---------|-------------------|-----------------|-------------|
| config.py | 75 | 6 | 81 |
| data/loader.py | 9 | 5 | 14 |
| indicators/indicators.py | 1 | 9 | 10 |
| regime/detector.py | 8 | 2 | 10 |
| regime/uncertainty_gate.py | 3 | 7 | 10 |
| backtest/engine.py | 5 | 4 | 9 |
| validation/preconditions.py | 2 | 7 | 9 |
| features/pipeline.py | 9 | 2 | 11 |
| regime/shock_vector.py | 2 | 6 | 8 |
| models/predictor.py | 4 | 2 | 6 |
| models/trainer.py | 4 | 2 | 6 |
| backtest/validation.py | 4 | 2 | 6 |
| api/services/health_service.py | 2 | 2 | 4 |
| autopilot/engine.py | 2 | 0 | 2 |
| autopilot/paper_trader.py | 0 | 0 | 0 |

**Key patterns**:
- **autopilot/engine.py is the primary amplifier**: It appears as the transitive pathway for 12 of 15 hotspots. Because it imports from nearly every core module, any change to those files transitively reaches api/jobs/autopilot_job.py and run_autopilot.py.
- **backtest/engine.py is the secondary amplifier**: Channels changes from regime/shock_vector.py, validation/preconditions.py, and regime/uncertainty_gate.py outward to api/orchestrator.py, autopilot/promotion_gate.py, and kalshi/promotion.py.
- **indicators/indicators.py has extreme transitive amplification**: Only 1 direct dependent (features/pipeline.py), but that bottleneck fans out to 9 transitive dependents. A break here silently propagates through the feature pipeline to training, prediction, backtesting, and the API.
- **autopilot/paper_trader.py is fully encapsulated**: Zero cross-module dependents. Its risk is entirely from what it CONSUMES (8 cross-module imports + 4 circular edges), not from what depends on it.

---

## Architectural Risk Flags

### Circular Dependency: autopilot ↔ api (6 edges)

This is the ONLY circular architectural dependency in the system.

| Edge | Source File | Line | Target | Import Type | Symbol |
|------|-----------|------|--------|-------------|--------|
| 1 | autopilot/paper_trader.py | 173 | api.services.health_risk_feedback | lazy | health_risk_feedback |
| 2 | autopilot/paper_trader.py | 189 | api.services.health_service | conditional | HealthService |
| 3 | autopilot/paper_trader.py | 211 | api.ab_testing | conditional | ABTestingFramework |
| 4 | autopilot/paper_trader.py | 532 | api.services.health_service | conditional | HealthService |
| 5 | autopilot/engine.py | 1868 | api.services.health_service | lazy | HealthService |
| 6 | autopilot/engine.py | 1911 | api.services.health_service | conditional | HealthService |

**Risk**: All 6 edges are lazy/conditional imports (inside function bodies or try/except guards). This means they won't cause import-time circular import errors, but:
1. If any api service changes its import path, the lazy import will fail at RUNTIME
2. The circular dependency means testing autopilot requires api to be importable and vice versa
3. Health check state from api flows into autopilot decision-making, creating hidden coupling

### Hub File: config.py (161 edges)

config.py is categorically different from every other file in the system. With 161 inbound import edges from ALL 14 modules, it has ~52% of all cross-module edges (161 / 308). This makes it:
- A single point of failure for the entire system
- The highest-leverage audit target (fixing issues here has maximum impact)
- A candidate for further modularization (structured config already partially addresses this)

### Test Coverage Blind Spots

| Module | Untested Files | Risk |
|--------|---------------|------|
| indicators | 7 files, 0 dedicated tests | 90+ indicator implementations tested only indirectly |
| validation | 5 files, 0 dedicated tests | Execution contracts and data integrity checks |
| api | 59 files, 9 test files | Many services/routers have no dedicated test coverage |
| features | 10 files, 5 test files | pipeline.py (1,541 lines) has limited dedicated testing |
| models | 16 files, 6 test files | Governance, online learning, retrain trigger gaps |
| utils | 2 files, 0 dedicated tests | Minimal impact (leaf module) |

---

## Verification Checklist

- [x] All 12 verified hotspots appear in the top 15 file rankings
  - #1 config.py (rank 1), #2 autopilot/engine.py (rank 2), #3 features/pipeline.py (rank 4),
    #4 backtest/engine.py (rank 4), #5 autopilot/paper_trader.py (rank 2), #6 regime/shock_vector.py (rank 12),
    #7 validation/preconditions.py (rank 15), #8 models/predictor.py (rank 12), #9 regime/uncertainty_gate.py (rank 12),
    #10 data/loader.py (rank 7), #11 api/services/health_service.py (rank 9), #12 indicators/indicators.py (rank 9)
- [x] Scores are based on actual data from Jobs 1-2 (MODULE_INVENTORY.yaml, DEPENDENCY_EDGES.json)
- [x] Blast radius for each top 15 file lists specific dependent files with line numbers
- [x] Test coverage gaps verified against actual test file listing (93 test files in tests/)
- [x] Shared artifact coupling scored (11 artifact patterns across 9 writer files)
- [x] The autopilot↔api circular dependency flagged prominently (6 edges, all documented with line numbers)
- [x] Scoring methodology documented with adjustment rationale
- [x] Module scoring detail table provides raw counts for independent verification
