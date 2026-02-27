# Hotspot List — Risk-Ranked Modules & Files

**Generated**: 2026-02-27
**Methodology**: Each module and file scored against 6 criteria (fan-in, fan-out, contract surface, change frequency, complexity, test gaps). Scores verified against actual imports, git log, and line counts.

---

## Module-Level Hotspot Ranking

Criteria scored 0–3 each. Maximum possible = 18.

| Rank | Module | Fan-In | Fan-Out | Contract Surface | Change Freq (30d) | Complexity | Test Gaps | Total | Risk Level |
|------|--------|--------|---------|------------------|--------------------|-----------|-----------|-------|------------|
| 1 | **config** | 3 (13 modules) | 0 | 3 (200+ constants, 14 dataclasses) | 3 (27 commits) | 2 (1020 lines) | 1 (1 test file) | **12** | CRITICAL |
| 2 | **autopilot** | 2 (kalshi, api) | 3 (8 modules) | 2 (3 JSON artifacts) | 3 (23 commits) | 3 (1927 lines) | 1 (2 test files) | **14** | CRITICAL |
| 3 | **backtest** | 2 (evaluation, autopilot, kalshi, api) | 2 (config, regime, risk, validation) | 3 (JSON+CSV result schemas) | 3 (16 commits) | 3 (2488 lines) | 1 | **14** | CRITICAL |
| 4 | **models** | 2 (evaluation, autopilot, api) | 1 (config, features, validation) | 3 (joblib+JSON artifacts) | 2 (12 commits) | 3 (1818 lines) | 1 | **12** | HIGH |
| 5 | **regime** | 3 (features, backtest, risk, autopilot) | 0 (config only) | 2 (ShockVector, UncertaintyGate) | 2 (15 commits) | 2 (940 lines) | 1 | **10** | HIGH |
| 6 | **risk** | 2 (backtest, autopilot) | 1 (config, regime) | 2 (PositionSizer, PortfolioRiskManager) | 2 (14 commits) | 2 (1254 lines) | 1 | **10** | HIGH |
| 7 | **features** | 2 (models, autopilot, kalshi) | 2 (config, data, indicators, regime) | 2 (FeaturePipeline, FEATURE_METADATA) | 2 (11 commits) | 3 (1541 lines) | 1 | **12** | HIGH |
| 8 | **data** | 2 (features, validation, autopilot, api) | 1 (config, validation, kalshi) | 2 (loader API, cache format) | 2 (18 commits) | 2 (849 lines) | 1 | **10** | HIGH |
| 9 | **api** | 1 (autopilot) | 3 (10 modules) | 3 (48 endpoints, ApiResponse envelope) | 2 (19 commits) | 3 (2929 lines) | 1 | **13** | HIGH |
| 10 | **validation** | 2 (data, models, backtest) | 1 (config, data, config_structured) | 2 (preconditions contract) | 0 (2 commits) | 0 (72 lines max) | 0 | **5** | MODERATE |
| 11 | **evaluation** | 1 (api via lazy) | 1 (config, models, backtest) | 1 (EvaluationResult) | 0 (2 commits) | 1 (827 lines) | 0 | **4** | MODERATE |
| 12 | **kalshi** | 1 (api) | 2 (config, features, autopilot, backtest) | 1 (DuckDB schema) | 0 (4 commits) | 2 (935 lines) | 0 | **6** | MODERATE |
| 13 | **indicators** | 1 (features only) | 0 | 1 (Indicator base class) | 0 (2 commits) | 3 (2904 lines) | 0 | **5** | LOW |
| 14 | **utils** | 0 | 0 (config lazy) | 0 | 0 (1 commit) | 0 (439 lines) | 0 | **0** | LOW |

---

## File-Level Hotspot Ranking

Scoring: Cross-module import count (3x), Is contract/interface file (2x), Lines of code (1x), Cyclomatic complexity proxy (1x), Lazy/conditional imports from other modules (2x).

### Top 15 Hotspot Files

| Rank | File | Cross-Module Imports (3x) | Contract File (2x) | Lines (1x) | Complexity (1x) | Lazy Cross-Imports (2x) | Weighted Score | Risk |
|------|------|---------------------------|-----------------------|-----------|-----------------|--------------------------|----------------|------|
| 1 | **config.py** | 9 (13 modules import it) | 6 (200+ constants define all system behavior) | 3 (1020) | 2 (46 branches) | 0 | **20** | CRITICAL |
| 2 | **autopilot/engine.py** | 6 (2 consumers: kalshi, api) | 4 (latest_cycle.json schema) | 3 (1927) | 3 (230 branches) | 0 | **16** | CRITICAL |
| 3 | **backtest/engine.py** | 6 (3 consumers: evaluation, autopilot, api) | 4 (BacktestResult, JSON/CSV schemas) | 3 (2488) | 3 (251 branches) | 6 (5 risk+validation lazy) | **22** | CRITICAL |
| 4 | **features/pipeline.py** | 6 (3 consumers: models, autopilot, kalshi) | 4 (FEATURE_METADATA, get_feature_type) | 3 (1541) | 3 (158 branches) | 6 (config, data, regime lazy) | **22** | CRITICAL |
| 5 | **autopilot/paper_trader.py** | 3 (1 consumer: engine.py) | 4 (paper_state.json schema) | 3 (1254) | 3 (143 branches) | 6 (risk.drawdown, api.services x3 lazy) | **19** | CRITICAL |
| 6 | **models/trainer.py** | 6 (2 consumers: autopilot, api) | 4 (trained model artifact schema) | 3 (1818) | 3 (216 branches) | 2 (validation lazy) | **18** | CRITICAL |
| 7 | **regime/shock_vector.py** | 6 (3 consumers: backtest/engine, backtest/execution, autopilot) | 4 (ShockVector dataclass schema) | 1 (495) | 1 (36 branches) | 2 (bocpd lazy) | **14** | HIGH |
| 8 | **regime/uncertainty_gate.py** | 6 (3 consumers: backtest/engine, risk/position_sizer, autopilot/engine) | 4 (UncertaintyGate sizing modifier) | 0 (181) | 0 (4 branches) | 2 (config lazy) | **12** | HIGH |
| 9 | **models/predictor.py** | 6 (2 consumers: autopilot, api) | 4 (predictions CSV schema) | 1 (539) | 2 (54 branches) | 0 | **13** | HIGH |
| 10 | **data/loader.py** | 6 (3 consumers: features, autopilot, api, run_*.py) | 4 (load_universe API contract) | 2 (849) | 3 (150 branches) | 4 (validation, survivorship lazy) | **19** | HIGH |
| 11 | **risk/position_sizer.py** | 6 (2 consumers: backtest/engine, autopilot/paper_trader) | 4 (PositionSizer contract) | 3 (1254) | 2 (87 branches) | 0 | **15** | HIGH |
| 12 | **validation/preconditions.py** | 6 (2 consumers: backtest/engine, models/trainer) | 4 (execution contract) | 0 (72) | 0 (5 branches) | 0 | **10** | HIGH |
| 13 | **api/services/health_service.py** | 3 (2 consumers: paper_trader, api routers) | 2 (health check schema) | 3 (2929) | 3 (374 branches) | 0 | **11** | HIGH |
| 14 | **regime/detector.py** | 6 (3 consumers: autopilot, run_*.py, scripts) | 4 (RegimeOutput, detect_regimes_batch) | 2 (940) | 2 (est. 80+ branches) | 4 (config, bocpd, shock_vector lazy) | **18** | HIGH |
| 15 | **backtest/validation.py** | 6 (3 consumers: evaluation, autopilot, run_backtest) | 4 (WalkForwardResult, CPCVResult, SPAResult) | 2 (1074) | 2 (est. 60+ branches) | 0 | **14** | HIGH |

---

## Blast Radius Analysis — Top 10 Files

### 1. config.py (Blast Radius: SYSTEM-WIDE)

**If this file changes**: Every module in the system is potentially affected.

| Affected Module | Impact Mechanism | Severity |
|----------------|------------------|----------|
| data | Loader behavior, cache paths, quality thresholds | HIGH |
| features | Feature computation parameters, interaction pairs, structural feature flags | HIGH |
| regime | HMM states, ensemble weights, jump model params, uncertainty thresholds | HIGH |
| models | Training params, model directory, max features, governance weights | HIGH |
| backtest | 50+ execution constants, cost params, threshold values | CRITICAL |
| risk | Drawdown thresholds, stop-loss params, Kelly fraction, portfolio vol limit | HIGH |
| evaluation | Evaluation thresholds, calibration bins, min slice samples | MEDIUM |
| validation | RET_TYPE, LABEL_H, PX_TYPE execution contract | CRITICAL |
| autopilot | Discovery multipliers, promotion gates, meta-labeling params | HIGH |
| kalshi | 20+ KALSHI_* constants, API config, stale thresholds | MEDIUM |
| api | Multiple routers/services lazily read config constants | MEDIUM |
| utils | Alert history file path, webhook URL | LOW |

**Mitigation**: Config validation in `validate_config()`. Structured equivalents in `config_structured.py`. STATUS annotations on constants.

### 2. autopilot/engine.py (Blast Radius: 3 downstream + 8 upstream)

**If this file changes**: Autopilot cycle behavior changes. Paper trading, promotion, and registry outputs change.

| Affected | Impact | Severity |
|----------|--------|----------|
| results/autopilot/latest_cycle.json | Output schema change breaks api/services/autopilot_service | HIGH |
| autopilot/paper_trader.py | Called directly by engine | HIGH |
| autopilot/promotion_gate.py | Called directly by engine | HIGH |
| autopilot/registry.py | Promotion results flow through | HIGH |
| api/services/autopilot_service.py | Reads engine output artifacts | MEDIUM |
| React frontend /autopilot page | Consumes API autopilot endpoints | MEDIUM |

### 3. backtest/engine.py (Blast Radius: 4 downstream consumers)

**If this file changes**: All backtest results, evaluation, promotion decisions, and paper trading behavior change.

| Affected | Impact | Severity |
|----------|--------|----------|
| results/backtest_*d_summary.json | Schema change breaks API and evaluation | CRITICAL |
| results/backtest_*d_trades.csv | Schema change breaks API and data_helpers | CRITICAL |
| evaluation/engine.py | Reads backtest results | HIGH |
| autopilot/engine.py | Calls Backtester directly | HIGH |
| autopilot/promotion_gate.py | Evaluates BacktestResult | HIGH |
| api/services/backtest_service.py | Reads result files | MEDIUM |
| kalshi/promotion.py | Converts to BacktestResult | MEDIUM |

### 4. features/pipeline.py (Blast Radius: 3 downstream consumers)

**If this file changes**: Feature names, ordering, or computation logic change. Model predictions become misaligned.

| Affected | Impact | Severity |
|----------|--------|----------|
| models/predictor.py | Uses get_feature_type for causality enforcement | CRITICAL |
| models/trainer.py | Trained on pipeline output — feature rename = broken model | CRITICAL |
| autopilot/engine.py | Calls FeaturePipeline.compute_universe() | HIGH |
| run_train.py, run_predict.py, run_backtest.py | All use FeaturePipeline | HIGH |

### 5. autopilot/paper_trader.py (Blast Radius: Health system + API coupling)

**If this file changes**: Paper trading state, execution quality, and health system feedback change.

| Affected | Impact | Severity |
|----------|--------|----------|
| results/autopilot/paper_state.json | Schema change breaks autopilot_service | HIGH |
| api/services/health_service.py | Lazily imported by paper_trader | HIGH (circular risk) |
| api/services/health_risk_feedback.py | Health gate integration | HIGH |
| api/ab_testing.py | A/B test trade recording | MEDIUM |

**NOTE**: This file is the source of the autopilot→api circular dependency concern.

### 6. regime/shock_vector.py (Blast Radius: Backtest layer)

**If this file changes**: ShockVector schema change breaks backtest structural state handling.

| Affected | Impact | Severity |
|----------|--------|----------|
| backtest/engine.py | Uses compute_shock_vectors + ShockVector | CRITICAL |
| backtest/execution.py | ShockModePolicy.from_shock_vector() | CRITICAL |
| autopilot/engine.py | Regime detection context | MEDIUM |

### 7. validation/preconditions.py (Blast Radius: Training + Backtesting contract)

**If this file changes**: The execution contract (RET_TYPE, LABEL_H, etc.) changes. Both training and backtesting behavior changes.

| Affected | Impact | Severity |
|----------|--------|----------|
| models/trainer.py | enforce_preconditions called during training | CRITICAL |
| backtest/engine.py | enforce_preconditions called during backtest init | CRITICAL |

### 8. models/predictor.py (Blast Radius: Prediction consumers)

**If this file changes**: Prediction format, regime blending, or conformal intervals change.

| Affected | Impact | Severity |
|----------|--------|----------|
| results/predictions_*d.csv | Output format change breaks API | HIGH |
| autopilot/engine.py | Calls EnsemblePredictor.predict_single() | HIGH |
| api/services/results_service.py | Reads prediction files | MEDIUM |

### 9. regime/uncertainty_gate.py (Blast Radius: Position sizing + backtest)

**If this file changes**: Position sizing multipliers change across backtest, risk, and autopilot.

| Affected | Impact | Severity |
|----------|--------|----------|
| risk/position_sizer.py | UncertaintyGate modifies position sizes | HIGH |
| backtest/engine.py | Uncertainty-aware signal suppression | HIGH |
| autopilot/engine.py | Uses UncertaintyGate for sizing | HIGH |

### 10. data/loader.py (Blast Radius: All data consumers)

**If this file changes**: Data loading behavior changes for all downstream consumers.

| Affected | Impact | Severity |
|----------|--------|----------|
| features/pipeline.py | Pipeline loads data via loader (lazy) | HIGH |
| autopilot/engine.py | Loads universe data | HIGH |
| run_train.py, run_backtest.py, run_predict.py | All use load_universe | HIGH |
| api/services/data_service.py | Lazy imports load_universe | MEDIUM |

---

## Recommended Audit Priority

Based on hotspot scores and blast radius:

| Priority | Files | Rationale |
|----------|-------|-----------|
| P0 — Audit First | config.py, config_structured.py | System-wide blast radius. Every change propagates everywhere. |
| P1 — Critical Path | backtest/engine.py, features/pipeline.py | Highest weighted scores. Core simulation and feature computation. |
| P1 — Critical Path | autopilot/engine.py, autopilot/paper_trader.py | Most tightly coupled. Circular API dependency. |
| P1 — Critical Path | models/trainer.py, models/predictor.py | Model artifact schemas consumed by API and autopilot. |
| P2 — High Risk | regime/shock_vector.py, regime/uncertainty_gate.py, regime/detector.py | Cross-cutting regime interfaces consumed by backtest+risk. |
| P2 — High Risk | validation/preconditions.py | Execution contract enforced by both training and backtesting. |
| P2 — High Risk | data/loader.py, risk/position_sizer.py | High fan-out and lazy import complexity. |
| P3 — Monitor | api/services/health_service.py | Largest file (2929 lines), called by paper_trader. |
| P3 — Monitor | backtest/validation.py, backtest/execution.py | Complex validation and execution logic. |
| P4 — Standard | All remaining files | Standard audit priority. |

---

## Circular / Architectural Concerns

### 1. autopilot → api → autopilot (CONFIRMED)

**Cycle path**: `autopilot/paper_trader.py` → (lazy) `api/services/health_risk_feedback.py` → (lazy) `api/services/health_service.py` → (lazy) back-reference potential.

**Verification**: This is NOT a true Python import cycle (all imports are lazy inside function bodies with try/except guards). However, it is an architectural dependency cycle: the autopilot module (which the API serves) depends on API services for health feedback. This means changes to API health services can break autopilot paper trading.

**Recommendation**: Extract the health gate interface to a shared module (e.g., `risk/health_gate.py`) that both autopilot and API can import without circular coupling.

### 2. kalshi → autopilot → backtest ← kalshi (DAG, not cycle)

`kalshi/promotion.py` imports from `autopilot/promotion_gate.py` and `backtest/engine.py`. `kalshi/walkforward.py` imports from `backtest/advanced_validation.py`. This is not a cycle but creates a transitive dependency: kalshi depends on autopilot AND backtest, while autopilot also depends on backtest. Changes to backtest affect both paths.

### 3. features → data → kalshi → features (Potential hidden cycle)

`features/pipeline.py` lazily imports from `data/`. `data/provider_registry.py` lazily imports from `kalshi/provider.py`. `kalshi/options.py` imports from `features/options_factors.py`. This creates a lazy-import cycle: features → data → kalshi → features. It does not cause runtime errors because all links are lazy, but it represents architectural coupling that should be monitored.
