# SPEC_AUDIT_FIX_14: Models Prediction, Calibration & Governance Fixes

**Priority:** CRITICAL/HIGH — Regime suppression targets wrong regime; calibration domain mismatch invalidates confidence.
**Scope:** `models/` — `predictor.py`, `trainer.py`, `walk_forward.py`, `retrain_trigger.py`, `governance.py`, `iv/models.py`
**Estimated effort:** 4 hours
**Depends on:** Nothing
**Blocks:** Nothing

---

## Context

The models subsystem has P0 issues: regime suppression targets regime 2 (mean_reverting) while commenting it as "high_volatility" (which is regime 3 per canonical mapping), confidence calibration trains on prediction magnitude but calibrates a composite confidence score at inference (domain mismatch), and walk-forward selection can hard-fail on NaN inputs. P1 issues include retrain gate ignoring negative OOS Spearman, governance promotion math failing for negative scores, disconnected distribution-shift detection, and Black-Scholes returning 0 for ITM options at zero vol.

---

## Tasks

### T1: Fix Regime Suppression Target

**Problem:** `predictor.py` force-zeros confidence for `regime == 2` while labeling it as "high-volatility", but the canonical mapping in `config.py` defines 2 = mean_reverting, 3 = high_volatility. Other modules (regime detector, risk position sizer) use the canonical mapping.

**File:** `models/predictor.py`

**Implementation:**
1. Determine the intended suppression target:
   - If high-volatility signals should be suppressed (most likely intent per the comment): change to `regime == 3`.
   - If mean-reverting signals should be suppressed: keep `regime == 2` but fix the comment.
2. Based on the comment "suppress high-volatility regime", fix:
   ```python
   # OLD: regime_2_mask = regime_vals == 2  # "high_volatility"
   # NEW:
   regime_suppress_mask = regime_vals == 3  # high_volatility (canonical regime 3)
   ```
3. Add a comment referencing the canonical mapping: `# See config.py REGIME_NAMES: {0: trending_bull, 1: trending_bear, 2: mean_reverting, 3: high_volatility}`.
4. Consider making the suppressed regime configurable: `REGIME_SUPPRESS_ID = 3`.

**Acceptance:** Regime 3 (high_volatility) signals are suppressed. Regime 2 (mean_reverting) signals pass through.

---

### T2: Fix Confidence Calibration Domain Mismatch

**Problem:** `trainer.py` trains the calibrator on normalized `abs(prediction)` (prediction magnitude). `predictor.py` applies the calibrator to a composite confidence score derived from holdout/CV/regime metrics. These are different distributions — the calibration mapping is not valid.

**Files:** `models/trainer.py`, `models/predictor.py`

**Implementation:**
1. Align train and inference domains. Choose one of:
   - **Option A (recommended):** Train calibrator on the same composite confidence score used at inference. Compute the composite score during training using the same formula as `predictor.py`:
     ```python
     composite_conf = compute_composite_confidence(holdout_sharpe, cv_gap, regime_confidence, ...)
     calibrator.fit(composite_conf, binary_labels)
     ```
   - **Option B:** At inference, calibrate `abs(prediction)` (same domain as training), then combine with other confidence signals post-calibration.
2. Document which domain the calibrator operates on.
3. Add assertion in `predictor.py` that calibrator input range matches training range (e.g., both in [0, 1]).

**Acceptance:** Calibrator train and inference inputs come from the same distribution. Calibrated confidence values are statistically meaningful.

---

### T3: Fix walk_forward_select NaN Handling

**Problem:** `walk_forward.py` drops rows with partial NaNs using `any(axis=1)`, then fits GradientBoostingRegressor which raises on remaining NaN. Can also reach `np.nanargmax` with all-NaN scores.

**File:** `models/walk_forward.py`

**Implementation:**
1. After dropping NaN targets, also drop NaN features:
   ```python
   valid = df.dropna(subset=target_cols + feature_cols)
   if len(valid) < min_samples:
       logger.warning("Insufficient non-NaN samples (%d < %d)", len(valid), min_samples)
       return default_model_config
   ```
2. Guard against all-NaN scores:
   ```python
   if np.all(np.isnan(scores)):
       logger.warning("All walk-forward scores are NaN, returning default config")
       return default_model_config
   best_idx = int(np.nanargmax(scores))
   ```
3. Add `min_samples` parameter (default 50) for the minimum viable training set.

**Acceptance:** walk_forward_select with NaN-containing data does not crash. Returns default config with warning when insufficient data.

---

### T4: Fix Retrain Gate Ignoring Negative OOS Spearman

**Problem:** `retrain_trigger.py` requires `oos_spearman > 0` before checking threshold, so very negative OOS quality (indicating a broken model) does not trigger retraining.

**File:** `models/retrain_trigger.py`

**Implementation:**
1. Add explicit check for negative correlation as a retrain trigger:
   ```python
   if oos_spearman < 0:
       logger.warning("OOS Spearman is negative (%.3f), triggering retrain", oos_spearman)
       return True  # Model is actively wrong, must retrain
   if oos_spearman < threshold:
       return True  # Model quality below threshold
   return False
   ```

**Acceptance:** A model with OOS Spearman = -0.3 triggers retraining.

---

### T5: Fix Governance Promotion Math for Negative Scores

**Problem:** `governance.py` computes relative threshold as `current_score * (1 + min_relative_improvement)`. When current_score is negative, this becomes more negative, so a worse challenger can satisfy `score > threshold`.

**File:** `models/governance.py`

**Implementation:**
1. Use absolute improvement for negative scores:
   ```python
   if current_score >= 0:
       threshold = current_score * (1 + min_relative_improvement)
   else:
       # For negative scores, challenger must be better by absolute margin
       threshold = current_score + abs(current_score) * min_relative_improvement
   ```
2. Or use a simpler rule: challenger must beat current score by at least `min_absolute_improvement` (e.g., 0.05 Sharpe units).

**Acceptance:** A challenger with score -0.3 does not beat a current model with score -0.2 when `min_relative_improvement=0.1`.

---

### T6: Wire Distribution-Shift Detection Into Retrain Flow

**Problem:** `retrain_trigger.py` has `check_shift()` but `check()` (the main method called by `run_retrain`) does not call it.

**File:** `models/retrain_trigger.py`

**Implementation:**
1. In `check()`, add shift detection call:
   ```python
   def check(self, ...):
       # Existing checks...
       triggers = []
       if self._check_staleness(): triggers.append("staleness")
       if self._check_performance(): triggers.append("performance")
       if self.check_shift(current_features, reference_features):
           triggers.append("distribution_shift")
       return len(triggers) > 0, triggers
   ```
2. Ensure `check_shift()` has access to current and reference feature distributions. These may need to be passed as parameters or loaded from artifacts.

**Acceptance:** A significant feature distribution shift triggers retraining. `check()` return value includes `"distribution_shift"` when detected.

---

### T7: Fix Black-Scholes Zero-Vol ITM Pricing

**Problem:** `iv/models.py` returns 0.0 for `sigma <= 0` and `T > 0`, but for ITM options the deterministic value should be the discounted intrinsic value.

**File:** `models/iv/models.py`

**Implementation:**
1. For zero vol with T > 0:
   ```python
   if sigma <= 0 and T > 0:
       if option_type == "call":
           return max(0.0, S - K * np.exp(-r * T))  # Discounted intrinsic
       else:
           return max(0.0, K * np.exp(-r * T) - S)
   ```

**Acceptance:** A deep ITM call (S=150, K=100) with sigma=0 returns ~50 * discount factor, not 0.

---

### T8: Fix Causality Enforcement Default

**Problem:** `predictor.py` / `features/pipeline.py` defaults unknown features to `"CAUSAL"` (fail-open). This is also covered in SPEC_10 T2 from the features side, but the predictor has its own check.

**File:** `models/predictor.py`

**Implementation:**
1. In the predictor's truth-layer check, also reject `"UNKNOWN"` features:
   ```python
   blocked_types = {"RESEARCH_ONLY", "UNKNOWN"}
   for feature in input_features:
       if get_feature_type(feature) in blocked_types:
           raise ValueError(f"Feature '{feature}' has type {get_feature_type(feature)}, blocked from prediction")
   ```
2. Coordinate with SPEC_10 T2 which changes the default from `"CAUSAL"` to `"UNKNOWN"`.

**Acceptance:** Predictor rejects unknown/unregistered features. Only explicitly tagged CAUSAL features pass.

---

## Verification

- [ ] Run `pytest tests/ -k "model or predictor or trainer or governance or retrain"` — all pass
- [ ] Verify regime 3 is suppressed, not regime 2
- [ ] Verify calibrator train/infer domains match
- [ ] Verify walk_forward_select with NaN data doesn't crash
- [ ] Verify negative OOS Spearman triggers retrain
- [ ] Verify negative-score governance rejects worse challengers
- [ ] Verify zero-vol ITM call returns intrinsic value
