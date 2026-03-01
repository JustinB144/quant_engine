# Audit Report — Subsystem 6: Model Training & Prediction

> **Auditor:** Claude Opus 4.6
> **Date:** 2026-02-28
> **Status:** COMPLETE — 16/16 files reviewed, 6,153/6,153 lines covered
> **Spec:** SPEC_AUDIT_06_MODEL_TRAINING_PREDICTION.md

---

## Executive Summary

Subsystem 6 implements the full model training and prediction pipeline: gradient boosting ensemble training with anti-overfitting controls, versioned model persistence, live prediction with regime blending, and supporting infrastructure for calibration, conformal intervals, distribution shift detection, online learning, IV surface modeling, and model governance.

**Overall assessment: SOUND with minor findings.** The core trainer-predictor contract is well-designed with strong anti-leakage measures. No critical bugs or contract violations were found. The subsystem has 5 findings (0 CRITICAL, 2 HIGH, 2 MEDIUM, 1 LOW).

---

## File Ledger

| # | File | Lines | Classes | Key Functions | Reviewed |
|---|------|------:|---------|---------------|----------|
| 1 | `models/trainer.py` | 1,818 | 4 (IdentityScaler, DiverseEnsemble, TrainResult, EnsembleResult, ModelTrainer) | train_ensemble, _train_single, _save, _fit_calibrator | YES |
| 2 | `models/iv/models.py` | 937 | 9 (OptionType, Greeks, HestonParams, SVIParams, BlackScholes, HestonModel, SVIModel, ArbitrageFreeSVIBuilder, IVSurface) | price, greeks, implied_vol, calibrate, build_surface | YES |
| 3 | `models/predictor.py` | 538 | 1 (EnsemblePredictor) | predict, predict_single, blend_multi_horizon, _load | YES |
| 4 | `models/retrain_trigger.py` | 344 | 1 (RetrainTrigger) | check, check_shift, add_trade_result, record_retraining | YES |
| 5 | `models/calibration.py` | 327 | 2 (ConfidenceCalibrator, _LinearRescaler) | fit, transform, compute_ece, compute_reliability_curve | YES |
| 6 | `models/shift_detection.py` | 322 | 1 (DistributionShiftDetector) | check_cusum, check_psi, set_reference | YES |
| 7 | `models/feature_stability.py` | 313 | 2 (StabilityReport, FeatureStabilityTracker) | record_importance, check_stability | YES |
| 8 | `models/conformal.py` | 295 | 3 (ConformalInterval, ConformalCalibrationResult, ConformalPredictor) | calibrate, predict_intervals_batch, uncertainty_scalars | YES |
| 9 | `models/online_learning.py` | 273 | 2 (OnlineUpdate, OnlineLearner) | add_sample, update, adjust_prediction, should_retrain | YES |
| 10 | `models/walk_forward.py` | 235 | 0 | walk_forward_select, _expanding_walk_forward_folds | YES |
| 11 | `models/versioning.py` | 207 | 2 (ModelVersion, ModelRegistry) | register_version, rollback, prune_old, get_latest_dir | YES |
| 12 | `models/neural_net.py` | 198 | 1 (TabularNet) | fit, predict, feature_importances_ | YES |
| 13 | `models/governance.py` | 155 | 2 (ChampionRecord, ModelGovernance) | evaluate_and_update, get_champion_version, _score | YES |
| 14 | `models/cross_sectional.py` | 136 | 0 | cross_sectional_rank | YES |
| 15 | `models/__init__.py` | 24 | 0 | (re-exports) | YES |
| 16 | `models/iv/__init__.py` | 31 | 0 | (re-exports) | YES |
| **TOTAL** | | **6,153** | | | **16/16** |

---

## T1: Ledger and Artifact Contract Baseline

### Model Artifact Schema

**Writer:** `models/trainer.py:_save()` (line 1479)
**Readers:** `models/predictor.py:_load()` (line 121), `api/services/model_service.py`

#### File artifacts per horizon (e.g., horizon=10):
| Artifact | Format | Writer | Reader |
|----------|--------|--------|--------|
| `ensemble_{H}d_global_model.pkl` | joblib | trainer._save:1504 | predictor._load:127 |
| `ensemble_{H}d_global_scaler.pkl` | joblib | trainer._save:1505 | predictor._load:128 |
| `ensemble_{H}d_meta.json` | JSON | trainer._save:1560 | predictor._load:131 |
| `ensemble_{H}d_regime{N}_model.pkl` | joblib | trainer._save:1511 | predictor._load:171 |
| `ensemble_{H}d_regime{N}_scaler.pkl` | joblib | trainer._save:1512 | predictor._load:174 |
| `ensemble_{H}d_calibrator.pkl` | joblib | trainer._fit_calibrator:1686 | predictor._load:145 |
| `ensemble_{H}d_conformal.json` | JSON | (saved externally) | predictor._load:154 |

#### Meta JSON required fields (trainer._save:1515-1531):
```
global_features          -> predictor.global_features (line 133)
global_feature_medians   -> predictor.global_medians (line 136)
global_target_std        -> predictor.global_target_std (line 138)
regime_models            -> predictor iterates (line 168)
  [code].features        -> predictor.regime_features[code] (line 177)
  [code].holdout_corr    -> predictor.regime_reliability[code] (line 179)
  [code].feature_medians -> predictor.regime_medians[code] (line 181)
  [code].target_std      -> predictor.regime_target_stds[code] (line 182)
global_holdout_corr      -> predictor confidence (line 346)
global_cv_corr           -> predictor confidence (line 348)
global_feature_importance -> predictor.predict_single (line 525)
```

**Verdict:** PASS. All fields written by trainer are read by predictor. No missing fields. The `.get()` defaults in predictor (e.g., `global_target_std` defaults to 0.10, `holdout_corr` defaults to 0) provide backward compatibility.

### Version Registry Schema

**File:** `trained_models/registry.json`
```json
{"latest": "20260220_143022", "versions": [{...ModelVersion...}]}
```

**Verdict:** PASS. ModelRegistry._load_registry (versioning.py:73) and _save_registry (versioning.py:80) use consistent JSON format. ModelVersion.from_dict (line 52) filters unknown fields for forward compatibility.

---

## T2: Training Pipeline Correctness Pass

### Anti-Leakage Measures Verified

| Control | Location | Status | Details |
|---------|----------|--------|---------|
| Purged date-grouped CV | trainer._date_purged_folds:326 | **PASS** | Purge gap = horizon days, embargo = horizon/2, operates on unique dates (not rows) |
| Scaler fit inside CV folds | trainer._train_single:1004-1007 | **PASS** | `fold_scaler = StandardScaler(); fold_scaler.fit_transform(X_dev.iloc[train_idx]...)` — separate scaler per fold |
| Holdout split by date | trainer._temporal_holdout_masks:292 | **PASS** | Splits on unique dates, not raw rows, to prevent cross-asset temporal leakage |
| Median imputation from dev-only | trainer._train_single:951-954 | **PASS** | `dev_medians = X.iloc[dev_idx].median()` — holdout medians not used |
| Per-fold feature selection | trainer._train_single:994-998 | **PASS** | Each fold independently selects features from its own training data |
| Stable feature aggregation | trainer._compute_stable_features:508 | **PASS** | Requires features in >= 80% of folds |
| IS/OOS gap monitoring | trainer._train_single:1056-1058 | **PASS** | Computes `cv_gap = mean(IS) - mean(OOS)`, warns if > max_gap |
| Hard block: severe overfit | trainer._train_single:1064-1073 | **PASS** | Rejects model if `cv_gap > max_gap * 3` |
| Hard block: negative R² | trainer._train_single:1120-1127 | **PASS** | Rejects model if holdout R² < 0 |
| Prediction clipping | predictor.predict:239-240 | **PASS** | Clips to ±3*target_std |
| Structural sample weighting | trainer.train_ensemble:726-738 | **PASS** | Downweights changepoints/jumps/stress via ShockVector |

### Preconditions Contract

- **trainer.py:219-220**: `from ..validation.preconditions import enforce_preconditions; enforce_preconditions()` (lazy import in `__init__`)
- **backtest/engine.py:198-199**: Identical pattern — same zero-arg call, same lazy import, same placement in `__init__`
- **Verdict:** PASS. Both consumers call `enforce_preconditions()` identically. The function validates RET_TYPE, LABEL_H, PX_TYPE, ENTRY_PRICE_TYPE against PreconditionsConfig.

### Walk-Forward Model Selection (walk_forward.py)

- Expanding-window folds with purge gap = horizon (line 61)
- Deflated Sharpe Ratio penalty applied to penalize excessive search (line 217-219)
- **Verdict:** PASS. Sound walk-forward methodology with multiple-testing correction.

### Calibration Integration (calibration.py)

- Isotonic regression or Platt scaling with sklearn fallback to linear rescaling
- ECE computation uses proper bin-weighted absolute calibration error (line 228-274)
- Reliability curve computation is standard (line 277-327)
- Trainer fits calibrator on holdout with 50/50 temporal split (trainer:1647-1676) to prevent calibration overfitting
- **Verdict:** PASS.

---

## T3: Prediction and Feature-Contract Pass

### Feature Causality Enforcement

- **predictor.py:22**: `from ..features.pipeline import get_feature_type` — top-level import
- **predictor.py:216-227**: When `TRUTH_LAYER_ENFORCE_CAUSALITY=True`, iterates feature columns, calls `get_feature_type(col)`, and raises `ValueError` if any return `"RESEARCH_ONLY"`
- **get_feature_type()** (pipeline.py:414-427): Returns `FEATURE_METADATA[name]["type"]` if found; defaults to `"CAUSAL"` for unknown names and `X_` prefixed features

**Verdict:** PASS. The contract is correctly enforced. The `"CAUSAL"` default for unknown features is a conscious design choice (fails open). The risk of silent leakage from miscategorized features is mitigated by:
1. predictor._prepare_features (line 30-48) aligns against `expected` columns from trained model metadata, catching truly unknown features
2. The trainer produces the `global_features` list which the predictor uses as the canonical feature set

### Feature Alignment Between Trainer and Predictor

**Trainer writes:** `global_features` = list of selected feature names (trainer._save:1519)
**Predictor reads:** `self.global_features = self.meta["global_features"]` (predictor._load:133)
**Usage:** `_prepare_features(features, self.global_features, self.global_medians)` (predictor.predict:233)

`_prepare_features` (predictor:30-48):
1. Keeps only columns in `expected` that exist in the input DataFrame
2. Fills missing columns with median values
3. Reorders to match `expected` order exactly
4. Fills remaining NaNs with median

**Verdict:** PASS. Feature ordering is deterministic — the predictor enforces the exact column order from training via `X = X[expected]` at line 43.

### Conformal Prediction (conformal.py)

- Finite-sample correction: `q_level = ceil((n+1)*coverage)/n` (line 147) — correct
- Intervals: `[prediction - quantile, prediction + quantile]` — symmetric, standard split conformal
- Integration with position sizing via `uncertainty_scalars()` (line 214-241)
- Serialization via `to_dict()/from_dict()` — clean round-trip
- **Verdict:** PASS. Standard split conformal with correct finite-sample guarantee.

### Regime 2 Suppression

- predictor.predict:411-418: When `regime == 2` (high_vol), sets `confidence = 0.0` and flags `regime_suppressed = True`
- This centralizes the trade gate so both backtester and live trading honor it
- **Verdict:** PASS. Clean centralized gate.

---

## T4: Versioning/Governance/Retrain Integrity Pass

### Model Versioning (versioning.py)

| Check | Status | Details |
|-------|--------|---------|
| Explicit version cannot silently fall back | **PASS** | predictor._resolve_model_dir:112-116: `if self.version != "latest": raise FileNotFoundError(...)` |
| Champion resolution | **PASS** | predictor._resolve_model_dir:80-92: Gets champion_id from governance, verifies directory exists |
| Latest version resolution | **PASS** | predictor._resolve_model_dir:96-103: Checks registry.has_versions(), gets latest dir, verifies existence |
| Backward-compat flat fallback | **PASS** | predictor._resolve_model_dir:118-119: Only when `version="latest"` and no registry |
| Registry conflict handling | **PASS** | versioning.register_version:140-142: Removes existing entry with same ID before inserting |
| Prune never removes latest | **PASS** | versioning.prune_old:191: `if v["version_id"] == latest or len(to_keep) < keep_n` |
| Version directory cleanup | **PASS** | versioning.prune_old:198-200: `shutil.rmtree(version_dir)` for pruned versions |

### Governance (governance.py)

| Check | Status | Details |
|-------|--------|---------|
| Champion scoring formula | **PASS** | Weighted OOS Spearman (1.5) + holdout Spearman (1.0) - CV gap penalty (0.5) + validation bonuses (DSR, PBO, MC, ECE) |
| Relative improvement threshold | **PASS** | evaluate_and_update:125: `score > current_score * (1 + min_relative_improvement)` with default 2% |
| History bounded | **PASS** | evaluate_and_update:142: `payload["history"] = payload["history"][-2000:]` |
| Champion persistence | **PASS** | Writes to `CHAMPION_REGISTRY` path (config-derived) |

### Retrain Triggers (retrain_trigger.py)

| Trigger | Status | Details |
|---------|--------|---------|
| Schedule (30d default) | **PASS** | check:194-203 |
| Trade count (50 default) | **PASS** | check:206-220 |
| Win rate degradation | **PASS** | check:222-233, rolling over lookback |
| OOS Spearman quality | **PASS** | check:235-241 |
| IC drift | **PASS** | check:243-252, rolling mean IC |
| Sharpe degradation | **PASS** | check:254-269, annualized rolling Sharpe |
| Distribution shift (CUSUM/PSI) | **PASS** | check_shift:139-182, delegates to DistributionShiftDetector |

### Distribution Shift Detection (shift_detection.py)

- CUSUM: Two-sided with allowance parameter k=0.5*std (line 164), standard Page formulation
- PSI: Proper symmetric KL-divergence with epsilon smoothing (line 248-251), industry-standard thresholds
- **Verdict:** PASS. Statistically sound implementations.

---

## T5: Specialized Model Surface Pass

### IV Models (iv/models.py — 937 lines)

| Component | Status | Details |
|-----------|--------|---------|
| BlackScholes pricing | **PASS** | Standard BS formula with edge case handling (T<=0, sigma<=0) |
| Greeks computation | **PASS** | Analytical Greeks (delta, gamma, vega, theta, rho) with proper sign conventions |
| Implied vol solver | **PASS** | Brent's method on [1e-6, 10.0] — numerically stable |
| Heston model | **PASS** | Albrecher et al. formulation for numerical stability, Feller condition check |
| SVI parameterization | **PASS** | Standard Gatheral (2004) raw SVI, no-butterfly-arbitrage check |
| ArbitrageFreeSVIBuilder | **PASS** | Weighted least squares with vega/spread weights, calendar monotonicity enforcement, penalty-based arbitrage avoidance |
| IVSurface interpolation | **PASS** | CloughTocher2D with LinearND fallback, log-moneyness coordinates |
| KL-decomposition | **PASS** | Level/slope/curvature extraction via polynomial fit — standard Cont & da Fonseca approach |

**Numerical guardrails:**
- `np.maximum(w, 1e-10)` for positive variance (SVI:318)
- `np.clip(iv_grid + noise, 0.05, 1.0)` for synthetic surface (line 673)
- `np.clip(P1, 0, 1)` and `np.clip(P2, 0, 1)` for Heston probabilities (lines 202-203)
- Put-call parity used for put pricing in Heston (line 212)

### Neural Net (neural_net.py — 198 lines)

- Experimental module, properly gated behind `_TORCH_AVAILABLE`
- Raises `NotImplementedError` when PyTorch not installed (not silent failure)
- Gradient clipping: `clip_grad_norm_(max_norm=1.0)` (line 137)
- `feature_importances_` property for compatibility with ensemble pipeline
- **Verdict:** PASS. Clean interface, proper optional dependency handling.

### Online Learning (online_learning.py — 273 lines)

- Exponential decay running statistics for error tracking
- Multiplicative bias correction with smoothing (0.8 + 0.2 * clipped ratio)
- Feature drift detection via z-score test
- State persistence to JSON
- `should_retrain()` with sensible heuristics (2/3 recent drift, scale > 0.5 from 1.0, error mean > 0.01)
- **Verdict:** PASS. Well-bounded online adjustments, not aggressive.

---

## T6: Boundary Validation and Findings Closure

### Cross-Subsystem Boundary Verdicts

| Boundary ID | Provider → Consumer | Risk | Verdict | Notes |
|-------------|-------------------|------|---------|-------|
| `models_to_features_6` | features → models | HIGH | **PASS** | `get_feature_type()` correctly integrated at predictor.py:22. Causality enforcement blocks RESEARCH_ONLY features when `TRUTH_LAYER_ENFORCE_CAUSALITY=True` (hardcoded True). |
| `models_to_config_17` | config → models | HIGH | **PASS** | All 22 config constants verified present with correct types/values. No stale or missing references. |
| `models_to_validation_18` | validation → models | MEDIUM | **PASS** | `enforce_preconditions()` at trainer.py:219 is identical to backtest/engine.py:198. Zero-arg, lazy import, RuntimeError on failure. |
| `autopilot_to_models_28` | models → autopilot | HIGH | **PASS** | All 4 symbols (ModelTrainer, EnsemblePredictor, cross_sectional_rank, _expanding_walk_forward_folds) are stable exports with correct signatures. |
| `api_to_models_40` | models → api | HIGH | **PASS** | 13 import edges verified. All lazy/conditional. Symbols match actual exports. |

### Shared Artifact Contract Verdicts

| Artifact | Writer → Reader | Verdict | Notes |
|----------|----------------|---------|-------|
| `trained_models/ensemble_*d_*.pkl` | trainer → predictor, model_service | **PASS** | Joblib format consistent. Predictor handles FileNotFoundError per regime (line 183). |
| `trained_models/ensemble_*d_meta.json` | trainer → predictor, model_service | **PASS** | All required fields written and consumed. `.get()` defaults provide backward compat. |
| `trained_models/registry.json` | versioning ↔ versioning, predictor | **PASS** | Consistent JSON schema with forward-compatible field filtering. |
| `trained_models/champion_registry.json` | governance ↔ governance, predictor | **PASS** | Clean read/write cycle with history bounding. |
| `MODEL_DIR/retrain_metadata.json` | retrain_trigger ↔ retrain_trigger | **PASS** | Self-contained persistence with bounded history. |
| `RESULTS_DIR/feature_stability_history.json` | feature_stability ↔ feature_stability, health_service | **PASS** | Clean JSON persistence with version-tolerant loading. |

---

## Findings

### F-01 [HIGH] — `get_feature_type()` defaults unknown features to CAUSAL

**File:** `features/pipeline.py:414-427` (read-only cross-subsystem reference)
**Impact:** If a new feature is added to the pipeline but not registered in `FEATURE_METADATA`, it defaults to `"CAUSAL"` regardless of its actual causality. This means an `END_OF_DAY` or `RESEARCH_ONLY` feature could silently pass the causality gate in predictor.py and enter live predictions with look-ahead data.

**Mitigating factors:**
- The predictor's `_prepare_features()` only uses features listed in `global_features` from the trained model's metadata — truly unknown features never reach the model
- The real risk window is between when a new feature is added and when FEATURE_METADATA is updated

**Recommendation:** Add a warning log in `get_feature_type()` when the default path is taken, so new unregistered features are immediately visible in logs. Consider switching the default to `"END_OF_DAY"` (conservative) rather than `"CAUSAL"` (permissive).

**Note:** This finding belongs to the Feature Engineering subsystem (Subsystem 3) but impacts Subsystem 6 through the `models_to_features_6` boundary.

---

### F-02 [HIGH] — Joblib deserialization of untrusted model artifacts is unsafe

**Files:** `models/predictor.py:127-128, 145, 171-175`
**Impact:** `joblib.load()` can execute arbitrary Python code during deserialization. If the `trained_models/` directory contains a malicious `.pkl` file (from a compromised training run, supply chain attack, or shared filesystem), loading it would execute that code with the process's full permissions.

**Mitigating factors:**
- The model directory is local and controlled by the system operator
- Models are only written by the trainer process
- No external/network model loading is supported

**Recommendation:** Document this as an accepted risk with a warning comment. For defense-in-depth, consider verifying model file checksums or adding a signing mechanism for model artifacts. Not urgent for a single-operator system.

---

### F-03 [MEDIUM] — DiverseEnsemble wraps IdentityScaler, breaking scaler-type assumptions

**File:** `models/trainer.py:1086-1094`
**Impact:** When `ENSEMBLE_DIVERSIFY=True`, the final model is a `DiverseEnsemble` and the returned scaler is an `IdentityScaler` (no-op). The predictor at `predictor.py:235` calls `self.global_scaler.transform(X_global)` which will pass through unchanged. The `DiverseEnsemble.predict()` method handles per-constituent scaling internally. This works correctly but breaks the assumption that `global_scaler` is a `StandardScaler`. If any downstream code inspects scaler attributes (e.g., `mean_`, `scale_`), it will fail with `AttributeError`.

**Mitigating factors:**
- Current predictor code only calls `.transform()` on the scaler, which IdentityScaler supports
- No current consumer inspects scaler internals

**Recommendation:** Add `mean_` and `scale_` stub attributes to `IdentityScaler` to prevent future AttributeError if code introspects the scaler.

---

### F-04 [MEDIUM] — `models/__init__.py` does not re-export ModelTrainer or EnsemblePredictor

**File:** `models/__init__.py:1-24`
**Impact:** The `__init__.py` exports `ModelGovernance`, `ChampionRecord`, `cross_sectional_rank`, `ConfidenceCalibrator`, `TabularNet`, `walk_forward_select`, `FeatureStabilityTracker`, `DistributionShiftDetector`, `ConformalPredictor`. Notably absent: `ModelTrainer`, `EnsemblePredictor`, `ModelRegistry`, `ModelVersion`, `RetrainTrigger`. All 28 inbound consumers import these directly from their respective modules (e.g., `from models.trainer import ModelTrainer`), so this works. But it means the package's public surface via `__init__.py` is inconsistent with its actual public API.

**Mitigating factors:**
- No consumer imports via `from models import ModelTrainer` — they all use the full path
- This is a style/consistency issue, not a bug

**Recommendation:** Consider adding the missing high-use exports to `__init__.py` for consistency, or document the convention that consumers should import from submodules directly.

---

### F-05 [LOW] — `RETRAIN_MAX_DAYS` referenced by api/services/backtest_service.py does not exist in retrain_trigger.py

**File:** `models/retrain_trigger.py`
**Impact:** The DEPENDENCY_EDGES.json records an import of `RETRAIN_MAX_DAYS` from `models/retrain_trigger.py` by `api/services/backtest_service.py`. However, no such constant exists in `retrain_trigger.py`. The default is a constructor parameter `max_days_between_retrain=30`. This is either a phantom edge in the dependency data or the API code may fail at runtime.

**Recommendation:** Verify `api/services/backtest_service.py` to determine if this import actually exists or if it's a dependency extraction artifact. (Deferred to Subsystem 10: API audit.)

---

## Acceptance Criteria Verification

| Criterion | Status | Evidence |
|-----------|--------|----------|
| 100% of 6,153 lines reviewed | **PASS** | All 16 files read in full; `wc -l` confirms 6,153 total |
| Trainer/predictor feature contracts validated | **PASS** | Feature ordering via `global_features` list; causality via `get_feature_type()`; median imputation via `global_feature_medians` |
| Artifact versioning/governance integrity documented | **PASS** | Registry schema, version resolution logic, champion promotion, pruning all verified |
| All high-risk boundaries dispositioned | **PASS** | 5/5 boundaries verified with explicit pass/fail |

---

## Dependency Edge Summary

- **Outbound cross-module:** 9 edges (7 to config, 1 to features/pipeline.py, 1 to validation/preconditions.py)
- **Inbound cross-module:** 28 edges (13 from api, 5 from autopilot, 1 from evaluation, 9 from entry_points)
- **Internal:** 16 edges within models/

---

## Notes for Downstream Audits

### For Subsystem 7 (Evaluation & Diagnostics):
- `evaluation/calibration_analysis.py` imports `compute_ece` and `compute_reliability_curve` from `models/calibration.py`. Signatures are stable.
- Backtest summary JSON 17-field schema must match evaluation reader expectations.

### For Subsystem 8 (Autopilot):
- `autopilot/engine.py` imports `ModelTrainer` (line 59) and `EnsemblePredictor` (line 58) at top-level — changes to these class APIs will break autopilot.
- `_expanding_walk_forward_folds` is a private function imported by autopilot. Consider making it a public API if it's part of the contract.

### For Subsystem 10 (API):
- Verify the `RETRAIN_MAX_DAYS` import from `api/services/backtest_service.py` (F-05).
- 13 lazy/conditional edges from api to models — verify all are runtime-safe.
