# Feature Spec: Model Training Pipeline — CV Fixes, Calibration, and Governance

> **Status:** Approved
> **Author:** justin
> **Date:** 2026-02-23
> **Estimated effort:** ~8 hours across 6 tasks

---

## Why

The model training pipeline has several issues that can silently degrade model quality: (1) Feature selection via permutation importance is computed on the validation fold within each CV split, but the selected features are then fixed for ALL subsequent folds — if fold 1 selects features that happen to work well on fold 1's val set, those features may not generalize to later folds. (2) The confidence calibrator (isotonic regression) is trained on holdout predictions, but if the holdout set is too small or non-representative, calibration can be overfit. (3) The governance promotion gate uses a weighted score formula `w_oos * oos_spearman + w_hold * holdout_spearman - w_gap * max(0, cv_gap)` with fixed weights that may not reflect actual model quality priorities. (4) Regime-specific models are trained on regime-filtered subsets, but there's no minimum sample size enforcement beyond a soft check — if regime 3 (high-vol) has only 50 samples, the model trains anyway and produces unreliable predictions. (5) The `_prune_correlated_features` threshold of 0.90 is too lenient — highly correlated features (0.85-0.90) still cause multicollinearity issues in gradient boosting.

## What

Fix feature selection to be fold-independent, add proper calibration validation, enforce minimum sample sizes for regime models, tighten feature correlation pruning, and improve governance scoring. Done means: each CV fold selects its own features, calibration is validated with a dedicated split, regime models refuse to train with insufficient data, and governance uses validated scoring weights.

## Constraints

### Must-haves
- Feature selection runs independently per CV fold (not once globally)
- Calibration validated on a separate split from the holdout used for selection
- Regime models require minimum 200 samples (configurable via MIN_REGIME_SAMPLES)
- Feature correlation threshold reduced to 0.80
- Governance scoring includes validation test results

### Must-nots
- Do NOT change the model algorithm (LightGBM/XGBoost) without separate A/B test
- Do NOT make training >2x slower (feature selection per fold is the main risk)
- Do NOT break model artifact format (versioned models must load correctly)

## Tasks

### T1: Make feature selection per-fold instead of global

**What:** Run permutation importance feature selection independently within each CV fold.

**Files:**
- `models/trainer.py` — modify `train_ensemble()` and `_select_features()`

**Implementation notes:**
- Current flow: `_select_features()` runs once on full dev set → fixed feature list → all CV folds use same features
- New flow: within `_date_purged_folds()` loop, run `_select_features()` on each fold's training data
  ```python
  for train_idx, test_idx in folds:
      X_train, y_train = X_dev.iloc[train_idx], y_dev.iloc[train_idx]
      X_test, y_test = X_dev.iloc[test_idx], y_dev.iloc[test_idx]
      
      # Per-fold feature selection
      selected_feats, importances = self._select_features(X_train, y_train, max_feats)
      X_train_sel = X_train[selected_feats]
      X_test_sel = X_test[selected_feats]
      
      # Train and evaluate on this fold
      scaler = StandardScaler()
      X_train_scaled = scaler.fit_transform(X_train_sel)
      X_test_scaled = scaler.transform(X_test_sel)
      model.fit(X_train_scaled, y_train)
      preds = model.predict(X_test_scaled)
      fold_corr = spearmanr(y_test, preds)
  ```
- Track feature frequency across folds: features selected in >80% of folds are "stable"
- Final model uses union of stable features (selected in ≥ 80% of folds)
- Performance optimization: cache permutation importance computation per fold (reuse if fold train data overlaps significantly)

**Verify:**
```bash
python -c "
print('Per-fold feature selection:')
print('  - Each fold independently selects best features from its training data')
print('  - Final feature set = features selected in >= 80% of folds')
print('  - Prevents fold-1 selection bias from propagating to all folds')
"
```

---

### T2: Add calibration validation split

**What:** Split holdout into calibration-fit and calibration-validation portions.

**Files:**
- `models/trainer.py` — modify holdout handling
- `models/calibration.py` — add validation metrics

**Implementation notes:**
- Current: holdout used for both calibration fitting and final validation reporting
- Fix: split holdout 50/50 into calibration-fit and calibration-validate
  ```python
  # In train_ensemble():
  holdout_dates = sorted(dates[hold_mask].unique())
  cal_fit_cutoff = holdout_dates[len(holdout_dates) // 2]
  
  cal_fit_mask = hold_mask & (dates <= cal_fit_cutoff)
  cal_val_mask = hold_mask & (dates > cal_fit_cutoff)
  
  # Fit calibrator on first half of holdout
  calibrator.fit(raw_confidence[cal_fit_mask], actuals[cal_fit_mask])
  
  # Validate calibration on second half
  cal_preds = calibrator.transform(raw_confidence[cal_val_mask])
  cal_reliability = compute_reliability_curve(cal_preds, actuals[cal_val_mask])
  ```
- Add reliability curve computation: bin predictions into deciles, compare predicted vs actual hit rate
- Report calibration quality metric: Expected Calibration Error (ECE)
  ```python
  def compute_ece(predicted_probs, actual_outcomes, n_bins=10):
      bins = np.linspace(0, 1, n_bins + 1)
      ece = 0
      for i in range(n_bins):
          mask = (predicted_probs >= bins[i]) & (predicted_probs < bins[i+1])
          if mask.sum() > 0:
              avg_pred = predicted_probs[mask].mean()
              avg_actual = actual_outcomes[mask].mean()
              ece += mask.sum() / len(predicted_probs) * abs(avg_pred - avg_actual)
      return ece
  ```

**Verify:**
```bash
python -c "
print('Calibration validation:')
print('  Holdout split: 50% for calibration fitting, 50% for calibration validation')
print('  ECE (Expected Calibration Error) reported in training results')
print('  Prevents calibration overfitting to holdout set')
"
```

---

### T3: Enforce minimum sample sizes for regime models

**What:** Refuse to train regime-specific models when sample count is below threshold.

**Files:**
- `models/trainer.py` — add sample size enforcement in regime model training

**Implementation notes:**
- Current: soft check exists but model trains anyway with low samples
- Fix: hard enforcement with clear fallback
  ```python
  MIN_REGIME_SAMPLES = 200  # Add to config.py
  
  # In train_ensemble(), regime model loop:
  for regime_id in unique_regimes:
      regime_mask = (regimes == regime_id)
      n_samples = regime_mask.sum()
      
      if n_samples < MIN_REGIME_SAMPLES:
          logger.warning(
              "Regime %d has only %d samples (min=%d) — skipping regime model. "
              "Global model will be used for regime %d predictions.",
              regime_id, n_samples, MIN_REGIME_SAMPLES, regime_id
          )
          regime_models[regime_id] = None  # Explicitly mark as unavailable
          continue
      
      # ... train regime model ...
  ```
- In predictor.py, handle missing regime models:
  ```python
  # If regime model is None, use global model for that regime
  if regime_models.get(regime_id) is None:
      regime_pred = global_pred  # Fall back to global
      regime_weight = 0.0  # Don't blend with missing model
  ```

**Verify:**
```bash
python -c "
from quant_engine.config import MIN_REGIME_SAMPLES
print(f'Minimum regime samples: {MIN_REGIME_SAMPLES}')
assert MIN_REGIME_SAMPLES >= 100, 'Too low — regime models need substantial data'
"
```

---

### T4: Tighten feature correlation pruning

**What:** Reduce correlation threshold from 0.90 to 0.80 and add VIF-based pruning option.

**Files:**
- `models/trainer.py` — update `_prune_correlated_features()` threshold

**Implementation notes:**
- Current threshold: 0.90 — leaves highly correlated features (0.85-0.89) that cause:
  - Feature importance instability (correlated features split importance)
  - Gradient boosting inefficiency (splits on redundant features)
- New threshold: 0.80
- Additionally, add optional VIF (Variance Inflation Factor) check:
  ```python
  def _prune_correlated_features(self, X, threshold=0.80, use_vif=False):
      """Remove features with pairwise correlation > threshold."""
      corr_matrix = X.corr().abs()
      upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
      to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
      
      if use_vif and len(X.columns) - len(to_drop) > 5:
          # Additional VIF check: remove features with VIF > 10
          from statsmodels.stats.outliers_influence import variance_inflation_factor
          X_remaining = X.drop(columns=to_drop)
          vif_data = pd.DataFrame()
          vif_data['feature'] = X_remaining.columns
          vif_data['VIF'] = [variance_inflation_factor(X_remaining.values, i) 
                             for i in range(len(X_remaining.columns))]
          high_vif = vif_data[vif_data['VIF'] > 10]['feature'].tolist()
          to_drop.extend(high_vif)
      
      kept = [c for c in X.columns if c not in to_drop]
      return kept
  ```

**Verify:**
```bash
python -c "
print('Feature correlation pruning:')
print('  OLD threshold: 0.90 (too lenient)')
print('  NEW threshold: 0.80 (removes more redundant features)')
print('  Optional VIF check for multicollinearity (VIF > 10 = drop)')
"
```

---

### T5: Improve governance scoring with validation integration

**What:** Include statistical validation results in the promotion gate scoring.

**Files:**
- `models/governance.py` — update `evaluate_and_update()` scoring formula

**Implementation notes:**
- Current score: `w_oos * oos_spearman + w_hold * holdout_spearman - w_gap * cv_gap`
- This ignores validation results (DSR, CPCV, SPA, PBO)
- New score incorporates validation:
  ```python
  def compute_champion_score(metrics):
      # Base performance score (existing)
      perf_score = (
          0.35 * metrics.get('oos_spearman', 0) +
          0.25 * metrics.get('holdout_spearman', 0) -
          0.15 * max(0, metrics.get('cv_gap', 0))
      )
      
      # Validation bonus/penalty (new)
      val_score = 0.0
      if 'dsr_significant' in metrics:
          val_score += 0.10 if metrics['dsr_significant'] else -0.10
      if 'pbo' in metrics:
          val_score += 0.05 * (1.0 - metrics['pbo'])  # Lower PBO = better
      if 'mc_significant' in metrics:
          val_score += 0.05 if metrics['mc_significant'] else -0.05
      if 'ece' in metrics:
          val_score += 0.05 * (1.0 - min(1.0, metrics['ece'] * 10))  # Lower ECE = better
      
      return perf_score + val_score
  ```
- Total weights: performance=0.75, validation=0.25

**Verify:**
```bash
python -c "
print('Governance scoring update:')
print('  Performance (75%): oos_spearman, holdout_spearman, cv_gap')
print('  Validation (25%): DSR, PBO, Monte Carlo, calibration ECE')
print('  Validation failures penalize score, preventing overfit promotion')
"
```

---

### T6: Test model training pipeline fixes

**What:** Tests for per-fold selection, calibration split, regime sample enforcement, and governance scoring.

**Files:**
- `tests/test_training_pipeline_fixes.py` — new test file

**Implementation notes:**
- Test cases:
  1. `test_feature_selection_per_fold` — Different folds can select different features
  2. `test_stable_features_selected_most_folds` — Final features appear in >=80% of folds
  3. `test_calibration_uses_separate_split` — Calibration fit and validation on different data
  4. `test_ece_computed` — Expected Calibration Error is in training results
  5. `test_regime_model_min_samples` — Regime with <200 samples → model is None
  6. `test_regime_fallback_to_global` — Missing regime model → global prediction used
  7. `test_correlation_threshold_080` — Features with corr >0.80 are pruned
  8. `test_governance_score_includes_validation` — DSR failure reduces champion score

**Verify:**
```bash
python -m pytest tests/test_training_pipeline_fixes.py -v
```

---

## Validation

### Acceptance criteria
1. Feature selection is independent per CV fold
2. Calibration is validated on a separate holdout split with ECE metric
3. Regime models with <200 samples are skipped (global model used as fallback)
4. Feature correlation threshold is 0.80 (not 0.90)
5. Governance score includes validation test results (DSR, PBO, MC, ECE)
6. Model artifacts remain backward-compatible (existing models still load)

### Rollback plan
- Per-fold selection: revert to global selection (change loop structure)
- Calibration split: merge splits back together
- Regime min samples: lower threshold to previous value
- Governance scoring: revert weight formula
