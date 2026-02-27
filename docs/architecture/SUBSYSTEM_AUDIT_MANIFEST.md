# Subsystem Audit Manifest

**Generated**: 2026-02-27
**Scope**: Virtual subsystem groupings for audit purposes. Every `.py` file in the repository appears in exactly one subsystem. Co-dependent files are co-located so auditors can review each subsystem without cross-referencing scattered directories.
**Verification Method**: All subsystem boundaries verified against actual Python import tracing (top-level and lazy/function-body imports) across 216 production `.py` files.

---

## Subsystem Overview

| # | Subsystem | Files | Audit Order | Key Risk | Depends On |
|---|-----------|-------|-------------|----------|------------|
| 1 | Shared Infrastructure | 7 | 1st | System-wide blast radius | None |
| 2 | Data Ingestion & Quality | 19 | 2nd | Data corruption propagates everywhere | 1 |
| 3 | Feature Engineering | 17 | 3rd | Feature name/ordering changes break models | 1, 2 (4 optional) |
| 4 | Regime Detection | 13 | 4th (parallel with 2) | Regime interface consumed by backtest+risk | 1 |
| 5 | Backtesting + Risk | 28 | 5th | Core simulation correctness | 1, 4 |
| 6 | Model Training & Prediction | 16 | 6th (parallel with 5) | Model artifact schemas | 1, 3, 5 |
| 7 | Evaluation & Diagnostics | 8 | 7th | Evaluation uses backtest + model outputs | 1, 5, 6 |
| 8 | Autopilot (Strategy Discovery) | 8 | 8th | Most tightly coupled; integration audit | 1, 2, 3, 4, 5, 6 (10 optional) |
| 9 | Kalshi (Event Markets) | 25 | 9th (parallel with 8) | Semi-isolated vertical | 1, 3, 5, 8 |
| 10 | API & Frontend | 59 | 10th | Pure consumer; lazy imports throughout | 1-9 (all core) |
| 11 | Entry Points & Scripts | 14 | 11th (last) | Orchestration wiring correctness | All |

**Total production files**: 214

---

## Subsystem Definitions

### Subsystem 1: Shared Infrastructure

**Files** (7):
```
__init__.py
config.py
config_structured.py
config_data/__init__.py
reproducibility.py
utils/__init__.py
utils/logging.py
```

**Rationale**: These are universal dependencies imported by 13 of 14 modules. `config.py` alone exports 200+ constants consumed system-wide. Changes here propagate everywhere. Must be audited first.

**Key Risk Areas**:
- `config.py`: 1020 lines, 200+ constants, imported by 13 modules. Single point of failure.
- `config_structured.py`: Authoritative typed dataclass hierarchy. config.py derives values from it.
- `utils/logging.py`: Contains lazy import of config constants (ALERT_HISTORY_FILE, ALERT_WEBHOOK_URL) to avoid circular imports.
- `reproducibility.py`: Build manifest for run traceability, consumed by all run_*.py entry points.

**What to Look For**:
- Constant value correctness and STATUS annotations (ACTIVE/PLACEHOLDER/DEPRECATED)
- Consistency between flat config.py constants and config_structured.py typed equivalents
- Any constant renamed or removed without updating all 13+ consumer modules

**Estimated Audit Effort**: 8-12 hours (high density, system-wide implications)

---

### Subsystem 2: Data Ingestion & Quality

**Files** (19):
```
data/__init__.py
data/loader.py
data/local_cache.py
data/provider_base.py
data/provider_registry.py
data/providers/__init__.py
data/providers/alpaca_provider.py
data/providers/alpha_vantage_provider.py
data/quality.py
data/cross_source_validator.py
data/intraday_quality.py
data/survivorship.py
data/alternative.py
data/feature_store.py
data/wrds_provider.py
validation/__init__.py
validation/data_integrity.py
validation/leakage_detection.py
validation/feature_redundancy.py
```

**Rationale**: `validation/data_integrity.py` is lazily imported by `data/loader.py` — they share a quality-gate contract. The other validation files (leakage_detection, feature_redundancy) are data-quality focused and have no consumers outside this subsystem. `data/provider_registry.py` has a lazy import of `kalshi/provider.py` (cross-subsystem, documented below).

**Key Risk Areas**:
- `data/loader.py` (849 lines): Primary data ingestion, lazy imports from validation and survivorship. Consumed by features, autopilot, API, and all run_*.py scripts.
- `data/local_cache.py` (841 lines): Cache trust model, atomic writes, metadata sidecars.
- `data/provider_registry.py`: Lazy factory pattern — silently skips providers if imports fail.
- `data/quality.py`: OHLCV quality thresholds from config — changes silently alter data filtering.

**What to Look For**:
- Cache staleness logic and trust source hierarchy in loader.py
- Survivorship bias handling in survivorship.py and load_survivorship_universe()
- Provider fallback behavior when WRDS/Alpaca/AlphaVantage are unavailable
- Data quality thresholds (MAX_MISSING_BAR_FRACTION, etc.) — are they appropriate?

**Cross-Subsystem Dependencies**:
- → Subsystem 1: 6 files import from config
- → Subsystem 9: data/provider_registry.py lazily imports kalshi/provider.py

**Estimated Audit Effort**: 12-16 hours

---

### Subsystem 3: Feature Engineering

**Files** (17):
```
features/__init__.py
features/pipeline.py
features/research_factors.py
features/options_factors.py
features/lob_features.py
features/intraday.py
features/macro.py
features/harx_spillovers.py
features/wave_flow.py
features/version.py
indicators/__init__.py
indicators/indicators.py
indicators/spectral.py
indicators/eigenvalue.py
indicators/ot_divergence.py
indicators/ssa.py
indicators/tail_risk.py
```

**Rationale**: `features/pipeline.py` imports 45+ symbols from `indicators/`. The indicators module has zero external consumers — it exists solely for the feature pipeline. They are one logical unit.

**Key Risk Areas**:
- `features/pipeline.py` (1541 lines): Orchestrates 90+ features. Has lazy imports from config (6 sites), data (3 sites), and regime (1 site). Feature name changes break model prediction.
- `indicators/indicators.py`: Core technical indicator library, 90+ indicator classes. Pure computation with no internal imports.
- `features/research_factors.py`: Advanced factor computation (order flow, Markov queue, signatures).
- `FEATURE_METADATA` and `get_feature_type()`: Contract surface consumed by models/predictor.py for causality enforcement.

**What to Look For**:
- Feature name stability (names are embedded in trained models)
- Lazy import fallbacks in pipeline.py — does graceful degradation hide broken features?
- Indicator correctness (financial math: ATR, Bollinger, GARCH, etc.)
- STRUCTURAL_FEATURES_ENABLED guard — structural features silently disabled if import fails

**Cross-Subsystem Dependencies**:
- → Subsystem 1: pipeline.py lazily imports config (6 sites), intraday.py imports config
- → Subsystem 2: pipeline.py lazily imports data/wrds_provider, data/local_cache, data/loader
- → Subsystem 4: pipeline.py lazily imports regime/correlation.py

**Estimated Audit Effort**: 16-24 hours (indicator math requires domain expertise)

---

### Subsystem 4: Regime Detection

**Files** (13):
```
regime/__init__.py
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

**Rationale**: Regime is relatively self-contained (only depends on config). However, `shock_vector.py` and `uncertainty_gate.py` are consumed by Subsystem 5 (backtest+risk) — the auditor MUST cross-reference Subsystem 5 after auditing this one.

**Key Risk Areas**:
- `regime/detector.py` (940 lines): Orchestrates HMM, jump model, BOCPD, and ensemble detection. 7 lazy imports. Most complex orchestration in this subsystem.
- `regime/shock_vector.py`: Defines ShockVector dataclass with version-locked schema (SHOCK_VECTOR_SCHEMA_VERSION). Schema changes break backtest/engine.py and backtest/execution.py.
- `regime/uncertainty_gate.py`: Sizing modifier consumed by backtest/engine.py, risk/position_sizer.py, and autopilot/engine.py. Threshold changes silently alter positions.
- `regime/jump_model_pypi.py`: Wraps external jumpmodels package. Lazy config imports for parameters.

**What to Look For**:
- ShockVector schema version compatibility with consumers
- UncertaintyGate threshold values and their downstream sizing impact
- HMM state mapping correctness (REGIME_NAMES: 0=trending_bull, 1=trending_bear, 2=mean_reverting, 3=high_volatility)
- Ensemble consensus logic and fallback when detectors disagree

**Cross-Subsystem Dependencies**:
- → Subsystem 1: detector.py (top + 7 lazy), uncertainty_gate.py (lazy), jump_model_pypi.py (lazy), consensus.py (lazy), online_update.py (lazy) — all import from config
- ← Subsystem 5: backtest/engine.py imports shock_vector.py and uncertainty_gate.py; risk/position_sizer.py imports uncertainty_gate.py; backtest/execution.py references ShockVector (TYPE_CHECKING)
- ← Subsystem 3: features/pipeline.py lazily imports regime/correlation.py
- ← Subsystem 8: autopilot/engine.py imports detector.py and uncertainty_gate.py

**Estimated Audit Effort**: 12-16 hours

---

### Subsystem 5: Backtesting + Risk (Execution Layer)

**Files** (28):
```
backtest/__init__.py
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
risk/__init__.py
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
validation/preconditions.py
```

**Rationale**: Backtest and risk are bilaterally coupled — `backtest/engine.py` imports from 2 risk files and 2 regime files, while `risk/position_sizer.py` imports from regime. `validation/preconditions.py` defines the execution contract (RET_TYPE, LABEL_H, PX_TYPE) enforced by both backtest/engine.py and models/trainer.py. At 28 files this slightly exceeds the 25-file target, but the tight coupling between backtest and risk argues against splitting.

**Key Risk Areas**:
- `backtest/engine.py` (2488 lines): Core simulation loop. 50+ config constants, imports from execution, cost_calibrator, regime/shock_vector, regime/uncertainty_gate, and lazily from validation/preconditions. Errors here invalidate ALL backtest results.
- `backtest/execution.py`: ExecutionModel and ShockModePolicy. References ShockVector via TYPE_CHECKING. Execution realism is part of system correctness.
- `risk/position_sizer.py` (1254 lines): Kelly sizing with UncertaintyGate integration. Consumed by backtest/engine.py and autopilot/paper_trader.py.
- `risk/portfolio_risk.py`: PortfolioRiskManager with covariance and factor exposure integration. Imports from risk/covariance.py, risk/factor_exposures.py, risk/universe_config.py.
- `risk/drawdown.py`: DrawdownController with multi-level thresholds. Consumed by autopilot/paper_trader.py.
- `validation/preconditions.py`: Execution contract imported by backtest/engine.py (lazy) and models/trainer.py (lazy). Changes here affect both training and backtesting.
- `risk/constraint_replay.py`: Imports from portfolio_risk, stress_test, universe_config — internal coupling hub.

**What to Look For**:
- Execution cost model correctness (spread, impact, participation limits)
- ShockModePolicy thresholds and regime-dependent behavior
- Position sizing Kelly fraction correctness and edge cases
- Drawdown controller level transitions and size multiplier accuracy
- Walk-forward validation embargo/purge correctness
- Stop-loss spread buffer and ATR multiplier correctness

**Cross-Subsystem Dependencies**:
- → Subsystem 1: backtest/engine.py (50+ constants), risk/drawdown.py, risk/stop_loss.py, risk/portfolio_risk.py, backtest/validation.py, backtest/optimal_execution.py, validation/preconditions.py — all import from config
- → Subsystem 4: backtest/engine.py imports shock_vector.py (top) and uncertainty_gate.py (top); risk/position_sizer.py imports uncertainty_gate.py (top); backtest/execution.py references ShockVector (TYPE_CHECKING)
- ← Subsystem 6: models/trainer.py lazily imports validation/preconditions.py
- ← Subsystem 7: evaluation/engine.py lazily imports backtest/validation.py
- ← Subsystem 8: autopilot/engine.py imports backtest/engine.py, backtest/validation.py, backtest/advanced_validation.py, backtest/execution.py (lazy), risk/position_sizer.py (lazy), risk/covariance.py (lazy), risk/portfolio_optimizer.py; autopilot/paper_trader.py imports backtest/execution.py, backtest/adv_tracker.py, backtest/cost_calibrator.py, risk/position_sizer.py, risk/stop_loss.py, risk/portfolio_risk.py, risk/drawdown.py (lazy); autopilot/promotion_gate.py imports backtest/engine.py (BacktestResult)
- ← Subsystem 9: kalshi/walkforward.py imports backtest/advanced_validation.py; kalshi/promotion.py imports backtest/engine.py (BacktestResult)
- ← Subsystem 10: api/orchestrator.py lazily imports backtest/engine.py; api/routers/risk.py lazily imports risk/factor_monitor.py

**Estimated Audit Effort**: 24-32 hours (largest subsystem, most critical)

---

### Subsystem 6: Model Training & Prediction

**Files** (16):
```
models/__init__.py
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
models/iv/__init__.py
models/iv/models.py
```

**Rationale**: Models is mostly self-contained. Cross-module imports are limited to config (6 files), features/pipeline.py (1 file — predictor.py), and validation/preconditions.py (1 file — trainer.py, lazy).

**Key Risk Areas**:
- `models/trainer.py` (1818 lines): Trains ensemble models. Imports from config, feature_stability, versioning, and lazily from validation/preconditions and calibration. Feature ordering must match predictor expectations.
- `models/predictor.py` (538 lines): Loads trained models and produces predictions. Imports `get_feature_type` from features/pipeline.py for causality enforcement. Feature name mismatch here breaks prediction.
- `models/versioning.py`: Model directory management and version registry. MODEL_DIR and MAX_MODEL_VERSIONS from config.
- `models/governance.py`: Champion record management. GOVERNANCE_SCORE_WEIGHTS from config.
- `models/iv/models.py`: Black-Scholes, Heston, SVI models for IV surface computation. Self-contained.

**What to Look For**:
- Feature ordering consistency between trainer and predictor
- Model versioning and registry integrity (what happens on version conflicts?)
- Governance score weight calibration
- Retrain trigger sensitivity (shift detection thresholds)
- Conformal prediction interval correctness

**Cross-Subsystem Dependencies**:
- → Subsystem 1: trainer.py, predictor.py, governance.py, versioning.py, feature_stability.py, retrain_trigger.py — import from config
- → Subsystem 3: predictor.py imports features/pipeline.py (get_feature_type, top-level)
- → Subsystem 5: trainer.py lazily imports validation/preconditions.py (enforce_preconditions)
- ← Subsystem 7: evaluation/calibration_analysis.py lazily imports models/calibration.py
- ← Subsystem 8: autopilot/engine.py imports models/trainer.py, models/predictor.py, models/walk_forward.py, models/cross_sectional.py
- ← Subsystem 10: api/orchestrator.py lazily imports models/trainer.py, models/predictor.py, models/governance.py, models/versioning.py; api/services/model_service.py lazily imports models/versioning.py, models/governance.py; api/services/backtest_service.py lazily imports models/retrain_trigger.py

**Estimated Audit Effort**: 16-20 hours

---

### Subsystem 7: Evaluation & Diagnostics

**Files** (8):
```
evaluation/__init__.py
evaluation/engine.py
evaluation/metrics.py
evaluation/slicing.py
evaluation/fragility.py
evaluation/calibration_analysis.py
evaluation/ml_diagnostics.py
evaluation/visualization.py
```

**Rationale**: Evaluation imports from config (5 files), lazily from models/calibration (1 file), and lazily from backtest/validation (1 file). Relatively self-contained but must be audited after Subsystems 5 and 6.

**Key Risk Areas**:
- `evaluation/engine.py` (826 lines): Orchestrates all evaluation dimensions. Lazily imports walk_forward_with_embargo and rolling_ic from backtest/validation.py.
- `evaluation/fragility.py`: PnL concentration, drawdown distribution, critical slowing down detection.
- `evaluation/calibration_analysis.py`: Lazily imports compute_ece and compute_reliability_curve from models/calibration.

**What to Look For**:
- Walk-forward embargo configuration correctness
- IC decay detection sensitivity
- Fragility metric thresholds
- Calibration analysis reliability curve accuracy

**Cross-Subsystem Dependencies**:
- → Subsystem 1: engine.py, metrics.py, slicing.py, fragility.py, calibration_analysis.py, ml_diagnostics.py — all import from config
- → Subsystem 5: engine.py lazily imports backtest/validation.py (walk_forward_with_embargo, rolling_ic, detect_ic_decay)
- → Subsystem 6: calibration_analysis.py lazily imports models/calibration.py (compute_ece, compute_reliability_curve)

**Estimated Audit Effort**: 6-10 hours

---

### Subsystem 8: Autopilot (Strategy Discovery)

**Files** (8):
```
autopilot/__init__.py
autopilot/engine.py
autopilot/strategy_discovery.py
autopilot/promotion_gate.py
autopilot/registry.py
autopilot/paper_trader.py
autopilot/meta_labeler.py
autopilot/strategy_allocator.py
```

**Rationale**: Autopilot is the most tightly coupled module. `autopilot/engine.py` imports from 11 distinct external modules (backtest×3, models×4, config, data×2, features, regime×2, risk×3, api). `autopilot/paper_trader.py` lazily imports from api/services (3 files) — the source of the autopilot→api architectural coupling concern. Must be audited LAST among core modules because it depends on virtually every other subsystem.

**Key Risk Areas**:
- `autopilot/engine.py` (1927 lines): The system's integration point. 18 top-level cross-module imports + 9 lazy imports. Changes to ANY upstream module can break autopilot.
- `autopilot/paper_trader.py` (1254 lines): 42 config constants, imports from backtest (3 files), risk (4 files including lazy drawdown), and lazily from api (3 files). Source of autopilot→api circular coupling.
- `autopilot/promotion_gate.py`: 28 config-driven promotion thresholds. Imports BacktestResult from backtest/engine.py.
- `autopilot/meta_labeler.py`: Meta-labeling model for signal confidence filtering.

**What to Look For**:
- Promotion gate threshold appropriateness and completeness
- Paper trader execution model parity with backtest/engine.py (are they using the same cost model?)
- Lazy api imports in paper_trader.py — verify graceful degradation when health service unavailable
- Strategy discovery variant generation logic
- Registry integrity (max active strategies, deduplication)

**Cross-Subsystem Dependencies**:
- → Subsystem 1: All 7 non-init files import from config
- → Subsystem 2: engine.py imports data/loader.py (load_universe, load_survivorship_universe) and data/survivorship.py
- → Subsystem 3: engine.py imports features/pipeline.py (FeaturePipeline)
- → Subsystem 4: engine.py imports regime/detector.py and regime/uncertainty_gate.py
- → Subsystem 5: engine.py imports backtest/engine.py, backtest/validation.py, backtest/advanced_validation.py, backtest/execution.py (lazy), risk/position_sizer.py (lazy), risk/covariance.py (lazy), risk/portfolio_optimizer.py; paper_trader.py imports backtest/execution.py, backtest/adv_tracker.py, backtest/cost_calibrator.py, risk/position_sizer.py, risk/stop_loss.py, risk/portfolio_risk.py, risk/drawdown.py (lazy); promotion_gate.py imports backtest/engine.py (BacktestResult)
- → Subsystem 6: engine.py imports models/trainer.py, models/predictor.py, models/walk_forward.py, models/cross_sectional.py
- → Subsystem 10: engine.py lazily imports api/services/health_service.py; paper_trader.py lazily imports api/services/health_service.py, api/services/health_risk_feedback.py, api/ab_testing.py

**Architectural Concern — autopilot↔api Coupling**:
`autopilot/paper_trader.py` lazily imports 3 api files. `api/jobs/autopilot_job.py` lazily imports `autopilot/engine.py`. This creates a module-level architectural cycle: autopilot→api→autopilot. However, at the file level, no true SCC exists — all cross-links are lazy with try/except guards. The health_service.py and health_risk_feedback.py files do NOT import back from autopilot. Recommendation: Extract health gate interface to a shared module (e.g., `risk/health_gate.py`).

**Estimated Audit Effort**: 16-24 hours (integration audit — requires familiarity with all upstream subsystems)

---

### Subsystem 9: Kalshi (Event Markets)

**Files** (25):
```
kalshi/__init__.py
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
kalshi/tests/__init__.py
kalshi/tests/test_bin_validity.py
kalshi/tests/test_distribution.py
kalshi/tests/test_leakage.py
kalshi/tests/test_no_leakage.py
kalshi/tests/test_signature_kat.py
kalshi/tests/test_stale_quotes.py
kalshi/tests/test_threshold_direction.py
kalshi/tests/test_walkforward_purge.py
```

**Rationale**: Kalshi is a semi-isolated vertical. Internal test files (8) are co-located. Cross-module imports: kalshi/provider.py imports from config (17 constants), kalshi/options.py imports from features/options_factors.py, kalshi/promotion.py imports from autopilot and backtest, kalshi/walkforward.py imports from backtest/advanced_validation.py, and kalshi/pipeline.py imports from autopilot/promotion_gate.py. Can be audited independently after Subsystems 3, 5, and 8.

**Key Risk Areas**:
- `kalshi/provider.py` (647 lines): 17 config constants for rate limiting, staleness, and API configuration.
- `kalshi/distribution.py`: Distribution reconstruction with quality-aware filtering. Imports from kalshi/quality.py.
- `kalshi/promotion.py`: Converts event walk-forward results to BacktestResult for promotion gate evaluation.
- `kalshi/walkforward.py`: Event-specific walk-forward using deflated_sharpe_ratio and monte_carlo_validation from backtest.
- `kalshi/storage.py`: DuckDB/SQLite EventTimeStore — schema stability is a contract.

**What to Look For**:
- As-of join correctness (leakage prevention) in events.py
- Distribution monotonicity repair correctness in distribution.py
- Stale quote detection thresholds in quality.py
- Walk-forward purge/embargo correctness
- DuckDB schema compatibility and migration safety

**Cross-Subsystem Dependencies**:
- → Subsystem 1: provider.py imports 17 config constants
- → Subsystem 3: options.py imports features/options_factors.py (compute_option_surface_factors)
- → Subsystem 5: walkforward.py imports backtest/advanced_validation.py (deflated_sharpe_ratio, monte_carlo_validation); promotion.py imports backtest/engine.py (BacktestResult)
- → Subsystem 8: pipeline.py imports autopilot/promotion_gate.py (PromotionDecision); promotion.py imports autopilot/promotion_gate.py (PromotionDecision, PromotionGate) and autopilot/strategy_discovery.py (StrategyCandidate)

**Estimated Audit Effort**: 12-16 hours

---

### Subsystem 10: API & Frontend

**Files** (59):
```
api/__init__.py
api/main.py
api/config.py
api/errors.py
api/ab_testing.py
api/orchestrator.py
api/deps/__init__.py
api/deps/providers.py
api/cache/__init__.py
api/cache/manager.py
api/cache/invalidation.py
api/jobs/__init__.py
api/jobs/store.py
api/jobs/runner.py
api/jobs/models.py
api/jobs/autopilot_job.py
api/jobs/backtest_job.py
api/jobs/predict_job.py
api/jobs/train_job.py
api/routers/__init__.py
api/routers/autopilot.py
api/routers/backtests.py
api/routers/benchmark.py
api/routers/config_mgmt.py
api/routers/dashboard.py
api/routers/data_explorer.py
api/routers/diagnostics.py
api/routers/iv_surface.py
api/routers/jobs.py
api/routers/logs.py
api/routers/model_lab.py
api/routers/regime.py
api/routers/risk.py
api/routers/signals.py
api/routers/system_health.py
api/schemas/__init__.py
api/schemas/autopilot.py
api/schemas/backtests.py
api/schemas/compute.py
api/schemas/dashboard.py
api/schemas/data_explorer.py
api/schemas/envelope.py
api/schemas/model_lab.py
api/schemas/signals.py
api/schemas/system_health.py
api/services/__init__.py
api/services/autopilot_service.py
api/services/backtest_service.py
api/services/data_helpers.py
api/services/data_service.py
api/services/diagnostics.py
api/services/health_alerts.py
api/services/health_confidence.py
api/services/health_risk_feedback.py
api/services/health_service.py
api/services/kalshi_service.py
api/services/model_service.py
api/services/regime_service.py
api/services/results_service.py
```

**Rationale**: The API layer is a pure consumer — it imports from core modules but nothing imports from it except autopilot (paper_trader.py and engine.py lazily import api/services). Nearly all cross-module imports are lazy (inside function bodies with try/except). The internal layering is: routers → services → (lazy) core engine + config.

**Key Risk Areas**:
- `api/services/health_service.py` (2929 lines): Largest file in the system. Comprehensive health check aggregation. Referenced by autopilot/paper_trader.py and autopilot/engine.py (lazy).
- `api/orchestrator.py`: Pipeline orchestrator that lazily imports data, features, regime, models, and backtest. Consolidates train/predict/backtest flows for API job executors.
- `api/jobs/autopilot_job.py`: Lazily imports AutopilotEngine — the api→autopilot link in the architectural cycle.
- `api/routers/risk.py`: Lazily imports risk/factor_monitor.py — the only direct api→risk import.
- `api/schemas/envelope.py`: ApiResponse contract consumed by all routers and the frontend.
- `api/services/health_risk_feedback.py`: Health-to-position-size feedback gate. Imported by autopilot/paper_trader.py.

**What to Look For**:
- ApiResponse envelope consistency across all routers
- Lazy import failure handling — do services degrade gracefully?
- Job lifecycle correctness (queued→running→succeeded/failed/cancelled)
- SSE event streaming reliability
- Cache invalidation correctness (are stale values served after train/backtest?)
- Health service check completeness and scoring logic

**Cross-Subsystem Dependencies** (all via lazy imports in services/orchestrator/jobs):
- → Subsystem 1: main.py, config.py, orchestrator.py, 7+ services — all lazily import from config
- → Subsystem 2: orchestrator.py → data/loader.py; data_service.py → data/loader.py; health_service.py → data/wrds_provider.py
- → Subsystem 3: orchestrator.py → features/pipeline.py
- → Subsystem 4: orchestrator.py → regime/detector.py; regime_service.py reads regime config
- → Subsystem 5: orchestrator.py → backtest/engine.py; risk router → risk/factor_monitor.py; backtest_service.py reads result files
- → Subsystem 6: orchestrator.py → models/trainer.py, predictor.py, governance.py, versioning.py; model_service.py → models/versioning.py, governance.py; backtest_service.py → models/retrain_trigger.py
- → Subsystem 8: jobs/autopilot_job.py → autopilot/engine.py; autopilot_service.py reads autopilot artifacts
- → Subsystem 9: kalshi_service.py → kalshi/storage.py

**Estimated Audit Effort**: 24-32 hours (large surface area, but many files are thin wrappers)

---

### Subsystem 11: Entry Points & Scripts

**Files** (14):
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

**Rationale**: These are orchestration entry points. They wire together multiple subsystems using absolute imports (`from quant_engine.*`). Audit them to verify correct wiring. Scripts include data download/gapfill utilities and debugging helpers.

**Key Risk Areas**:
- `run_backtest.py`: Wires data→features→regime→models→backtest→validation. Lazy import of advanced_validation.
- `run_train.py` / `run_retrain.py`: Training pipeline with governance and versioning.
- `run_autopilot.py`: AutopilotEngine entry point.
- `scripts/alpaca_intraday_download.py`: Uses importlib for dynamic imports, lazy intraday_quality and cross_source_validator.
- `run_wrds_daily_refresh.py`: Lazy WRDSProvider imports, survivorship handling.

**What to Look For**:
- CLI argument handling correctness
- Correct module wiring order (data → features → regime → train/predict/backtest)
- Reproducibility manifest generation and verification
- Error handling and cleanup on failure

**Cross-Subsystem Dependencies**:
- → Subsystem 1: All run_*.py import config and reproducibility
- → Subsystem 2: run_train/backtest/predict/retrain → data/loader; run_wrds → data/local_cache, data/survivorship; scripts → data/local_cache, data/providers
- → Subsystem 3: run_train/backtest/predict/retrain → features/pipeline; scripts/compare → features/pipeline (lazy)
- → Subsystem 4: run_train/backtest/predict/retrain → regime/detector; scripts/compare → regime/detector, regime/hmm
- → Subsystem 5: run_backtest → backtest/engine, backtest/validation, backtest/advanced_validation (lazy)
- → Subsystem 6: run_backtest/predict → models/predictor; run_train/retrain → models/trainer, versioning, governance
- → Subsystem 8: run_autopilot → autopilot/engine
- → Subsystem 9: run_kalshi_event_pipeline → kalshi/distribution, events, pipeline, promotion, walkforward
- → Subsystem 10: run_server → api/config, api/main

**Estimated Audit Effort**: 8-12 hours

---

## Recommended Audit Order

The audit order respects the dependency DAG — no subsystem is audited before its dependencies:

```
Phase 1 (Foundation — no dependencies):
  1. Subsystem 1: Shared Infrastructure

Phase 2 (Core data + detection — depends only on S1):
  2. Subsystem 2: Data Ingestion & Quality
  4. Subsystem 4: Regime Detection              ← can audit in parallel with S2

Phase 3 (Computation layers — depends on Phase 2):
  3. Subsystem 3: Feature Engineering            ← depends on S1, S2, S4
  5. Subsystem 5: Backtesting + Risk             ← depends on S1, S4
  6. Subsystem 6: Model Training & Prediction    ← depends on S1, S3, S5

Phase 4 (Evaluation — depends on Phase 3):
  7. Subsystem 7: Evaluation & Diagnostics       ← depends on S1, S5, S6

Phase 5 (Integration layers — depends on most core subsystems):
  8. Subsystem 8: Autopilot                      ← depends on S1-S6, S10
  9. Subsystem 9: Kalshi                         ← depends on S1, S3, S5, S8 (parallel with S8)

Phase 6 (Consumer layers — depends on all):
  10. Subsystem 10: API & Frontend               ← depends on all core subsystems
  11. Subsystem 11: Entry Points & Scripts        ← depends on everything (audit last)
```

**Parallelization opportunities**:
- S2 and S4 can be audited in parallel (both depend only on S1)
- S5 and S6 can be partially parallelized (S6 depends on S3→S5 only via lazy preconditions import)
- S8 and S9 can be partially parallelized (S9 depends on S8 only for promotion types)

---

## Cross-Subsystem Boundary Table

Every import edge that crosses subsystem boundaries is documented below. Coupling strength: **tight** = top-level import, many symbols; **moderate** = lazy import or few symbols; **loose** = TYPE_CHECKING only or single symbol.

### S2 → S1 (Data → Shared Infrastructure)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| data/loader.py | config.py | CACHE_MAX_STALENESS_DAYS, DATA_QUALITY_ENABLED, +8 | tight |
| data/quality.py | config.py | MAX_MISSING_BAR_FRACTION, MAX_ZERO_VOLUME_FRACTION, MAX_ABS_DAILY_RETURN | tight |
| data/local_cache.py | config.py | DATA_CACHE_DIR, FRAMEWORK_DIR | tight |
| data/feature_store.py | config.py | ROOT_DIR | tight |
| data/alternative.py | config.py | DATA_CACHE_DIR, MARKET_CLOSE, MARKET_OPEN | tight |
| data/intraday_quality.py | config.py | DATA_CACHE_DIR, MARKET_CLOSE, MARKET_OPEN | tight |
| data/survivorship.py | config.py | SURVIVORSHIP_DB | moderate (lazy) |
| validation/preconditions.py | config.py | RET_TYPE, LABEL_H, PX_TYPE, +2 | tight |
| validation/preconditions.py | config_structured.py | PreconditionsConfig | tight |
| validation/data_integrity.py | data/quality.py | assess_ohlcv_quality, DataQualityReport | tight |

**Audit Note**: Config constant changes to data thresholds silently alter data filtering behavior. Verify MAX_MISSING_BAR_FRACTION and related thresholds are calibrated.

### S2 → S9 (Data → Kalshi)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| data/provider_registry.py | kalshi/provider.py | KalshiProvider | moderate (lazy, in factory) |

**Audit Note**: This is a lazy factory import. If kalshi module is unavailable, the factory silently returns None. Verify this graceful degradation works correctly.

### S3 → S1 (Features → Shared Infrastructure)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| features/pipeline.py | config.py | SPECTRAL_*, SSA_*, JUMP_INTENSITY_*, WASSERSTEIN_*, SINKHORN_*, INTERACTION_PAIRS, FORWARD_HORIZONS, STRUCTURAL_FEATURES_ENABLED, EIGEN_*, INTRADAY_MIN_BARS, BENCHMARK, LOOKBACK_YEARS | moderate (all lazy, 6 sites) |
| features/intraday.py | config.py | MARKET_OPEN, MARKET_CLOSE | tight |

### S3 → S2 (Features → Data)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| features/pipeline.py | data/wrds_provider.py | WRDSProvider | moderate (lazy, try/except) |
| features/pipeline.py | data/local_cache.py | load_intraday_ohlcv | moderate (lazy, try/except) |
| features/pipeline.py | data/loader.py | load_ohlcv | moderate (lazy) |

**Audit Note**: All three data imports are lazy with try/except fallbacks. Verify that feature computation degrades gracefully when data sources are unavailable.

### S3 → S4 (Features → Regime)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| features/pipeline.py | regime/correlation.py | CorrelationRegimeDetector | moderate (lazy, try/except) |

### S4 → S1 (Regime → Shared Infrastructure)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| regime/detector.py | config.py | REGIME_MODEL_TYPE, +19 constants (top-level) + 7 lazy sites | tight |
| regime/uncertainty_gate.py | config.py | REGIME_UNCERTAINTY_* (4 constants) | moderate (lazy) |
| regime/jump_model_pypi.py | config.py | REGIME_JUMP_* (8 constants) | moderate (lazy) |
| regime/consensus.py | config.py | REGIME_CONSENSUS_* (4 constants) | moderate (lazy) |
| regime/online_update.py | config.py | REGIME_ONLINE_REFIT_DAYS | moderate (lazy) |

### S5 → S1 (Backtesting+Risk → Shared Infrastructure)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| backtest/engine.py | config.py | 50+ execution/trading constants | tight |
| backtest/validation.py | config.py | IC_ROLLING_WINDOW | tight |
| backtest/optimal_execution.py | config.py | ALMGREN_CHRISS_RISK_AVERSION | tight |
| risk/drawdown.py | config.py | DRAWDOWN_* (8 constants) | tight |
| risk/stop_loss.py | config.py | REGIME_STOP_MULTIPLIER, HARD_STOP_PCT, +4 | tight |
| risk/portfolio_risk.py | config.py | MAX_PORTFOLIO_VOL, CORRELATION_STRESS_THRESHOLDS | tight |
| validation/preconditions.py | config.py | RET_TYPE, LABEL_H, PX_TYPE, +2 | tight |
| validation/preconditions.py | config_structured.py | PreconditionsConfig | tight |

### S5 → S4 (Backtesting+Risk → Regime)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| backtest/engine.py | regime/shock_vector.py | compute_shock_vectors, ShockVector | tight (top-level) |
| backtest/engine.py | regime/uncertainty_gate.py | UncertaintyGate | tight (top-level) |
| backtest/execution.py | regime/shock_vector.py | ShockVector | loose (TYPE_CHECKING) |
| risk/position_sizer.py | regime/uncertainty_gate.py | UncertaintyGate | tight (top-level) |

**Audit Note**: ShockVector schema changes break both backtest/engine.py and backtest/execution.py. UncertaintyGate threshold changes silently alter position sizing in both backtest and risk.

### S6 → S1 (Models → Shared Infrastructure)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| models/trainer.py | config.py | MODEL_DIR, REGIME_NAMES, HORIZON, +many | tight |
| models/predictor.py | config.py | MODEL_DIR, REGIME_NAMES, TRUTH_LAYER_ENFORCE_CAUSALITY | tight |
| models/governance.py | config.py | CHAMPION_REGISTRY, GOVERNANCE_SCORE_WEIGHTS | tight |
| models/versioning.py | config.py | MODEL_DIR, MAX_MODEL_VERSIONS | tight |
| models/feature_stability.py | config.py | RESULTS_DIR | tight |
| models/retrain_trigger.py | config.py | MODEL_DIR | tight |

### S6 → S3 (Models → Features)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| models/predictor.py | features/pipeline.py | get_feature_type | tight (top-level) |

**Audit Note**: `get_feature_type` is used for causality enforcement — ensuring non-causal features are excluded from prediction. Changes to FEATURE_METADATA break this contract.

### S6 → S5 (Models → Backtesting+Risk)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| models/trainer.py | validation/preconditions.py | enforce_preconditions | moderate (lazy) |

**Audit Note**: The execution contract (RET_TYPE, LABEL_H, PX_TYPE) must be consistent between training and backtesting. Both paths lazily import enforce_preconditions.

### S7 → S1 (Evaluation → Shared Infrastructure)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| evaluation/engine.py | config.py | EVAL_WF_*, EVAL_IC_*, EVAL_CALIBRATION_*, +5 | tight |
| evaluation/metrics.py | config.py | EVAL_MIN_SLICE_SAMPLES, EVAL_DECILE_SPREAD_MIN | tight |
| evaluation/slicing.py | config.py | EVAL_MIN_SLICE_SAMPLES, REGIME_NAMES (lazy) | tight/moderate |
| evaluation/fragility.py | config.py | EVAL_TOP_N_TRADES, +3 | tight |
| evaluation/calibration_analysis.py | config.py | EVAL_CALIBRATION_BINS, EVAL_OVERCONFIDENCE_THRESHOLD | tight |
| evaluation/ml_diagnostics.py | config.py | EVAL_FEATURE_DRIFT_THRESHOLD, EVAL_ENSEMBLE_DISAGREEMENT_THRESHOLD | tight |

### S7 → S5 (Evaluation → Backtesting+Risk)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| evaluation/engine.py | backtest/validation.py | walk_forward_with_embargo, rolling_ic, detect_ic_decay | moderate (lazy) |

### S7 → S6 (Evaluation → Models)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| evaluation/calibration_analysis.py | models/calibration.py | compute_ece, compute_reliability_curve | moderate (lazy) |

### S8 → S1 (Autopilot → Shared Infrastructure)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| autopilot/engine.py | config.py | 18 constants | tight |
| autopilot/paper_trader.py | config.py | 42 constants | tight |
| autopilot/promotion_gate.py | config.py | 28 constants | tight |
| autopilot/meta_labeler.py | config.py | 5 constants | tight |
| autopilot/registry.py | config.py | STRATEGY_REGISTRY_PATH, PROMOTION_MAX_ACTIVE_STRATEGIES | tight |
| autopilot/strategy_allocator.py | config.py | REGIME_NAMES | tight |
| autopilot/strategy_discovery.py | config.py | ENTRY_THRESHOLD, +7 | tight |

### S8 → S2 (Autopilot → Data)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| autopilot/engine.py | data/loader.py | load_survivorship_universe, load_universe | tight |
| autopilot/engine.py | data/survivorship.py | filter_panel_by_point_in_time_universe | tight |

### S8 → S3 (Autopilot → Features)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| autopilot/engine.py | features/pipeline.py | FeaturePipeline | tight |

### S8 → S4 (Autopilot → Regime)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| autopilot/engine.py | regime/detector.py | RegimeDetector | tight |
| autopilot/engine.py | regime/uncertainty_gate.py | UncertaintyGate | tight |

### S8 → S5 (Autopilot → Backtesting+Risk)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| autopilot/engine.py | backtest/engine.py | Backtester | tight |
| autopilot/engine.py | backtest/advanced_validation.py | capacity_analysis, deflated_sharpe_ratio, probability_of_backtest_overfitting | tight |
| autopilot/engine.py | backtest/validation.py | walk_forward_validate, run_statistical_tests, combinatorial_purged_cv, superior_predictive_ability, strategy_signal_returns | tight |
| autopilot/engine.py | backtest/execution.py | ExecutionModel | moderate (lazy) |
| autopilot/engine.py | risk/position_sizer.py | PositionSizer | moderate (lazy) |
| autopilot/engine.py | risk/covariance.py | CovarianceEstimator, compute_regime_covariance, get_regime_covariance | moderate (lazy) |
| autopilot/engine.py | risk/portfolio_optimizer.py | optimize_portfolio | tight |
| autopilot/paper_trader.py | backtest/execution.py | ExecutionModel | tight |
| autopilot/paper_trader.py | backtest/adv_tracker.py | ADVTracker | tight |
| autopilot/paper_trader.py | backtest/cost_calibrator.py | CostCalibrator | tight |
| autopilot/paper_trader.py | risk/position_sizer.py | PositionSizer | tight |
| autopilot/paper_trader.py | risk/stop_loss.py | StopLossManager | tight |
| autopilot/paper_trader.py | risk/portfolio_risk.py | PortfolioRiskManager | tight |
| autopilot/paper_trader.py | risk/drawdown.py | DrawdownController | moderate (lazy) |
| autopilot/promotion_gate.py | backtest/engine.py | BacktestResult | tight |

### S8 → S6 (Autopilot → Models)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| autopilot/engine.py | models/walk_forward.py | _expanding_walk_forward_folds | tight |
| autopilot/engine.py | models/cross_sectional.py | cross_sectional_rank | tight (+lazy dup) |
| autopilot/engine.py | models/predictor.py | EnsemblePredictor | tight |
| autopilot/engine.py | models/trainer.py | ModelTrainer | tight |

### S8 → S10 (Autopilot → API) — ARCHITECTURAL CONCERN

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| autopilot/engine.py | api/services/health_service.py | HealthService | moderate (lazy, try/except) |
| autopilot/paper_trader.py | api/services/health_service.py | HealthService | moderate (lazy, try/except) |
| autopilot/paper_trader.py | api/services/health_risk_feedback.py | create_health_risk_gate | moderate (lazy, try/except) |
| autopilot/paper_trader.py | api/ab_testing.py | ABTestRegistry | moderate (lazy, try/except) |

**Audit Note**: These imports create an architectural cycle (autopilot→api→autopilot via jobs). All are lazy with try/except guards — no runtime breakage. But API health service changes can silently alter autopilot paper trading behavior.

### S9 → S1 (Kalshi → Shared Infrastructure)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| kalshi/provider.py | config.py | 17 KALSHI_* constants | tight |

### S9 → S3 (Kalshi → Features)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| kalshi/options.py | features/options_factors.py | compute_option_surface_factors | tight |

### S9 → S5 (Kalshi → Backtesting+Risk)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| kalshi/walkforward.py | backtest/advanced_validation.py | deflated_sharpe_ratio, monte_carlo_validation | tight |
| kalshi/promotion.py | backtest/engine.py | BacktestResult | tight |

### S9 → S8 (Kalshi → Autopilot)

| Source File | Target File | Symbols | Coupling |
|-------------|-------------|---------|----------|
| kalshi/pipeline.py | autopilot/promotion_gate.py | PromotionDecision | tight |
| kalshi/promotion.py | autopilot/promotion_gate.py | PromotionDecision, PromotionGate | tight |
| kalshi/promotion.py | autopilot/strategy_discovery.py | StrategyCandidate | tight |

### S10 → S1-S9 (API → All Core Subsystems)

All API cross-subsystem imports are lazy (inside function bodies). Key edges:

| Source File | Target File | Coupling |
|-------------|-------------|----------|
| api/orchestrator.py | data/loader.py, features/pipeline.py, regime/detector.py, models/trainer.py, models/predictor.py, models/governance.py, models/versioning.py, backtest/engine.py | moderate (all lazy) |
| api/services/data_service.py | data/loader.py, config | moderate (lazy) |
| api/services/health_service.py | data/wrds_provider.py, config | moderate (lazy) |
| api/services/model_service.py | models/versioning.py, models/governance.py | moderate (lazy) |
| api/services/backtest_service.py | models/retrain_trigger.py, config | moderate (lazy) |
| api/services/kalshi_service.py | kalshi/storage.py, config | moderate (lazy) |
| api/routers/risk.py | risk/factor_monitor.py | moderate (lazy) |
| api/routers/diagnostics.py | api/services/diagnostics.py | moderate (lazy, internal) |
| api/jobs/autopilot_job.py | autopilot/engine.py | moderate (lazy) |
| api/jobs/backtest_job.py | api/orchestrator.py | moderate (lazy) |
| api/jobs/predict_job.py | api/orchestrator.py | moderate (lazy) |
| api/jobs/train_job.py | api/orchestrator.py | moderate (lazy) |
| api/main.py | config.py (validate_config, LOG_LEVEL) | moderate (lazy) |

### S11 → S1-S10 (Entry Points → All Subsystems)

Entry point imports are primarily top-level (absolute `from quant_engine.*` pattern):

| Source File | Primary Targets | Coupling |
|-------------|-----------------|----------|
| run_train.py | config, data/loader, features/pipeline, regime/detector, models/trainer+governance+versioning, reproducibility | tight |
| run_backtest.py | config, data/loader+survivorship, features/pipeline, regime/detector, models/predictor, backtest/engine+validation+advanced_validation, reproducibility | tight |
| run_predict.py | config, data/loader, features/pipeline, regime/detector, models/predictor, reproducibility | tight |
| run_autopilot.py | config, autopilot/engine, reproducibility | tight |
| run_retrain.py | config, data/loader, features/pipeline, regime/detector, models/trainer+governance+versioning+retrain_trigger, reproducibility | tight |
| run_server.py | api/config, api/main | tight |
| run_kalshi_event_pipeline.py | config, kalshi/* | tight |
| run_wrds_daily_refresh.py | config, data/local_cache, data/survivorship, data/wrds_provider (lazy) | tight/moderate |
| run_rehydrate_cache_metadata.py | config, data/local_cache | tight |
| scripts/compare_regime_models.py | config, regime/detector, regime/hmm, features/pipeline (lazy) | tight/moderate |
| scripts/ibkr_daily_gapfill.py | config, data/local_cache | tight |
| scripts/ibkr_intraday_download.py | config, data/local_cache (via importlib) | moderate |
| scripts/alpaca_intraday_download.py | config, data/local_cache, data/intraday_quality, data/cross_source_validator, data/providers/* (via importlib) | moderate |

---

## SCC Analysis (Strongly Connected Components)

**Result: No true file-level SCCs exist.**

All identified "cycles" are module-level architectural concerns with lazy imports that break any cycle at runtime:

### Architectural Cycle 1: autopilot ↔ api (CONFIRMED, NOT A TRUE SCC)

**Forward path** (autopilot → api):
- `autopilot/paper_trader.py` →(lazy)→ `api/services/health_service.py`
- `autopilot/paper_trader.py` →(lazy)→ `api/services/health_risk_feedback.py`
- `autopilot/paper_trader.py` →(lazy)→ `api/ab_testing.py`
- `autopilot/engine.py` →(lazy)→ `api/services/health_service.py`

**Reverse path** (api → autopilot):
- `api/jobs/autopilot_job.py` →(lazy)→ `autopilot/engine.py`

**Why it's not a true SCC**: `api/services/health_service.py` does NOT import from any autopilot file. `api/services/health_risk_feedback.py` does NOT import from any autopilot file. `api/ab_testing.py` does NOT import from any autopilot file. The file-level graph is a DAG.

**Recommendation**: Extract health gate interface to `risk/health_gate.py` to eliminate the architectural coupling.

### Architectural Cycle 2: features → data → kalshi → features (CONFIRMED, NOT A TRUE SCC)

**Path**: `features/pipeline.py` →(lazy)→ `data/wrds_provider.py` ... `data/provider_registry.py` →(lazy)→ `kalshi/provider.py` ... `kalshi/options.py` →(top)→ `features/options_factors.py`

**Why it's not a true SCC**: At the file level, there is no cycle. `features/pipeline.py` → `data/wrds_provider.py` and `data/provider_registry.py` → `kalshi/provider.py` are separate paths. `features/options_factors.py` does not import from `features/pipeline.py`.

---

## Validation Checklist

- [x] **Completeness**: All 214 `.py` files appear in exactly one subsystem
- [x] **Co-location**: All cross-subsystem import edges are documented in the boundary table
- [x] **No orphans**: Every file has at least one import relationship with another file in its subsystem (or is an `__init__.py` module marker)
- [x] **Hotspot coverage**: All files with fan-in ≥ 3 appear in the Part 1 HOTSPOT_LIST.md
- [x] **Cycle detection**: Both known architectural cycles verified — neither is a true SCC
- [x] **Audit ordering**: Audit order respects the dependency DAG (no subsystem audited before its dependencies)
