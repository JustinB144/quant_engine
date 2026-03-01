# SPEC_AUDIT_FIX_15: Regime Detection Correctness & Robustness Fixes

**Priority:** CRITICAL/HIGH — Ensemble silently falls through; HMM standardization leaks future data; ShockVector rejects valid outputs.
**Scope:** `regime/` — `detector.py`, `hmm.py`, `shock_vector.py`, `jump_model_pypi.py`, `confidence_calibrator.py`, `consensus.py`
**Estimated effort:** 4 hours
**Depends on:** Nothing
**Blocks:** Nothing

---

## Context

The regime detection subsystem has five P0 defects: calling `detect_full()` with `method="ensemble"` silently falls through to rule-based detection (the documented ensemble path only activates when `method="hmm"` AND `REGIME_ENSEMBLE_ENABLED=True`); the `_apply_min_duration` tie-break computes identical scores for both neighbors so short runs always bias left; jump model fallback goes to rules instead of legacy jump as documented; ShockVector validation hardcodes a (4,4) transition matrix shape but the HMM supports 2–6 auto-selected states; and HMM observation standardization uses full-series mean/std, leaking future data into historical regime labels. P1 issues include non-incremental BOCPD, missing empty-input guard, unused config knobs, and regime label bound validation gaps.

---

## Tasks

### T1: Support Explicit `method="ensemble"` in detect_full

**Problem:** `detector.py:563` checks `REGIME_ENSEMBLE_ENABLED and self.method == "hmm"` to activate ensemble detection. If a caller passes `method="ensemble"` (documented at line 93), `detect_full()` falls through all `if` branches to `_rule_detect()` at line 569, silently returning rule-based results instead of ensemble.

**File:** `regime/detector.py`

**Implementation:**
1. Add an explicit branch for `method="ensemble"` in `detect_full()`:
   ```python
   def detect_full(self, features: pd.DataFrame) -> RegimeOutput:
       if self.method == "ensemble" or (REGIME_ENSEMBLE_ENABLED and self.method == "hmm"):
           return self.detect_ensemble(features)
       if self.method == "hmm":
           return self._hmm_detect(features)
       if self.method == "jump":
           return self._jump_detect(features)
       return self._rule_detect(features)
   ```
2. Validate `method` in `__init__` against allowed values: `{"hmm", "jump", "rule", "ensemble"}`. Raise `ValueError` for unknown methods.
3. Add a test that explicitly constructs `RegimeDetector(method="ensemble")` and verifies `detect_full()` returns `model_type` containing `"ensemble"`, not `"rule"`.

**Acceptance:** `RegimeDetector(method="ensemble").detect_full(features)` invokes ensemble detection. Unknown methods raise `ValueError`.

---

### T2: Fix _apply_min_duration Confidence Tie-Break

**Problem:** `detector.py:166-170` computes `left_score` and `right_score` for a short regime run. Both use `conf[i:j].mean()` (the mean confidence of the short segment itself), but `left_score` is only nonzero if `left_state == vals[max(0, i-1)]` and `right_score` only if `right_state == vals[min(j, n-1)]`. When both neighbors differ from the short segment (common case), both scores are 0.0, causing the tie-break to always choose left (`left_score >= right_score` is `0.0 >= 0.0 = True`).

**File:** `regime/detector.py`

**Implementation:**
1. Replace the scoring logic with a comparison of neighbor confidence rather than segment self-confidence:
   ```python
   # Score each neighbor by the mean confidence of that neighbor's own run
   # (not the short segment's confidence)
   if left_state is not None:
       # Find the extent of the left neighbor's run
       left_run_start = i - 1
       while left_run_start > 0 and vals[left_run_start - 1] == left_state:
           left_run_start -= 1
       left_score = float(conf[left_run_start:i].mean())
   else:
       left_score = -1.0

   if right_state is not None:
       # Find the extent of the right neighbor's run
       right_run_end = j
       while right_run_end < n - 1 and vals[right_run_end + 1] == right_state:
           right_run_end += 1
       right_score = float(conf[j:right_run_end + 1].mean())
   else:
       right_score = -1.0

   repl = left_state if left_score >= right_score else right_state
   ```
2. This ensures the tie-break merges the short run into whichever adjacent regime has higher mean confidence in its own run, rather than always defaulting to left.

**Acceptance:** A short regime run between two distinct regimes merges into the neighbor with higher mean confidence. Changing confidence values changes the merge direction.

---

### T3: Fix Jump Model Fallback to Match Documentation

**Problem:** `detector.py:332-333` catches jump model failures and falls back to `_rule_detect(features)`. Documentation and code comments (and jump_model_pypi.py:13) state that PyPI jump failures should fall back to the legacy jump model, not rules.

**File:** `regime/detector.py`

**Implementation:**
1. Change the fallback chain in `_jump_detect()`:
   ```python
   try:
       if REGIME_JUMP_USE_PYPI_PACKAGE:
           result = self._jump_detect_pypi(X, features)
       else:
           result = self._jump_detect_legacy(X)
   except (ValueError, RuntimeError, ImportError) as e:
       if REGIME_JUMP_USE_PYPI_PACKAGE:
           logger.warning("PyPI jump model failed (%s), falling back to legacy jump model", e)
           try:
               result = self._jump_detect_legacy(X)
           except (ValueError, RuntimeError) as e2:
               logger.warning("Legacy jump model also failed (%s), falling back to rules", e2)
               return self._rule_detect(features)
       else:
           logger.warning("Legacy jump model failed (%s), falling back to rules", e)
           return self._rule_detect(features)
   ```
2. For the short-series guard (`len(features) < 80`), add a log message noting the fallback to rules due to insufficient data.

**Acceptance:** When PyPI jump model fails, legacy jump model is attempted before falling back to rules. Log messages clearly identify the fallback chain.

---

### T4: Fix ShockVector Transition Matrix Shape Validation

**Problem:** `config_structured.py:117` and `shock_vector.py:267` hardcode the expected transition matrix shape to (4,4), but the HMM supports `n_states` from 2–6 (auto-selectable). When the HMM selects 3 states, `detect_with_shock_context()` (detector.py:804-812) produces a valid (3,3) matrix that ShockVectorValidator marks as invalid, yet the ShockVector is still returned.

**Files:** `regime/shock_vector.py`, `regime/config_structured.py`

**Implementation:**
1. Make the ShockVector validator accept variable-sized square transition matrices:
   ```python
   # In ShockVectorValidator.validate():
   if sv.transition_matrix is not None:
       tm = np.asarray(sv.transition_matrix)
       if tm.ndim != 2 or tm.shape[0] != tm.shape[1]:
           errors.append(f"Transition matrix must be square, got shape {tm.shape}")
       elif tm.shape[0] < 2 or tm.shape[0] > MAX_HMM_STATES:
           errors.append(f"Transition matrix size {tm.shape[0]} outside valid range [2, {MAX_HMM_STATES}]")
   ```
2. Add `MAX_HMM_STATES = 6` constant in `config_structured.py` (matching the HMM auto-select upper bound).
3. Store `n_states` in the ShockVector so downstream consumers know the actual state count:
   ```python
   # In ShockVector dataclass:
   n_hmm_states: int = 4  # Actual number of HMM states used
   ```
4. In `detect_with_shock_context()`, populate `n_hmm_states` from the RegimeOutput.

**Acceptance:** A ShockVector with a (3,3) transition matrix from a 3-state HMM passes validation. A non-square matrix still fails.

---

### T5: Fix HMM Observation Standardization Temporal Leakage

**Problem:** `hmm.py:597-602` standardizes observation columns using full-series `mean()` and `std()`. When this is used for historical regime labeling or backtest workflows, earlier timestamps are normalized using statistics that include future data.

**File:** `regime/hmm.py`

**Implementation:**
1. Add a `backtest_safe` parameter to `build_hmm_observation_matrix()`:
   ```python
   def build_hmm_observation_matrix(
       features: pd.DataFrame,
       backtest_safe: bool = False,
   ) -> pd.DataFrame:
   ```
2. When `backtest_safe=True`, use expanding-window standardization:
   ```python
   if backtest_safe:
       for c in obs.columns:
           expanding_mean = obs[c].expanding(min_periods=20).mean()
           expanding_std = obs[c].expanding(min_periods=20).std()
           expanding_std = expanding_std.replace(0.0, 1e-12)
           obs.loc[:, c] = (obs[c] - expanding_mean) / expanding_std
           # First 19 rows will be NaN — fill with 0
           obs[c] = obs[c].fillna(0.0)
   else:
       # Existing full-series standardization (fine for live inference)
       for c in obs.columns:
           s = obs[c]
           std = float(s.std())
           if std > 1e-12:
               obs.loc[:, c] = (s - s.mean()) / std
           else:
               obs.loc[:, c] = 0.0
   ```
3. In the backtest integration point (where regime detection is called for historical analysis), pass `backtest_safe=True`.
4. Default to `False` to avoid breaking live inference performance.

**Acceptance:** In backtest mode, regime labels at time `t` use only data up to time `t` for standardization. Live inference is unchanged.

---

### T6: Add Empty-Input Guard to detect_with_shock_context

**Problem:** `detector.py:794-796` accesses `features.index[-1]` and `regime_out.regime.iloc[-1]` without checking for empty input. An empty DataFrame causes `IndexError: index -1 is out of bounds`.

**File:** `regime/detector.py`

**Implementation:**
1. Add an early return at the top of `detect_with_shock_context()`:
   ```python
   def detect_with_shock_context(self, features: pd.DataFrame, ticker: str = "") -> "ShockVector":
       if features.empty:
           logger.warning("Empty features passed to detect_with_shock_context for %s", ticker)
           return ShockVector.empty(ticker=ticker)
   ```
2. Add a `ShockVector.empty()` classmethod that returns a default/neutral ShockVector with `is_valid=False`.

**Acceptance:** `detect_with_shock_context(pd.DataFrame())` returns an invalid ShockVector without raising an exception.

---

### T7: Wire Unused Config Knobs or Remove Them

**Problem:** `REGIME_ENSEMBLE_CONSENSUS_THRESHOLD` (detector.py:37), `BOCPD_CHANGEPOINT_THRESHOLD` (detector.py:42), and `REGIME_JUMP_MODE_LOSS_WEIGHT` (jump_model_pypi.py:91) are imported/declared but never used in decision logic.

**Files:** `regime/detector.py`, `regime/jump_model_pypi.py`

**Implementation:**
1. For `REGIME_ENSEMBLE_CONSENSUS_THRESHOLD`: Wire it into `detect_ensemble()` as a minimum agreement threshold. If no method exceeds this confidence, return a low-confidence fallback:
   ```python
   if max_weighted_confidence < REGIME_ENSEMBLE_CONSENSUS_THRESHOLD:
       logger.warning("No method exceeds consensus threshold %.2f", REGIME_ENSEMBLE_CONSENSUS_THRESHOLD)
       # Mark as low-confidence output
       confidence_series *= 0.5
   ```
2. For `BOCPD_CHANGEPOINT_THRESHOLD`: Wire into `detect_with_shock_context()` to threshold the changepoint probability before setting `jump_detected`.
3. For `REGIME_JUMP_MODE_LOSS_WEIGHT`: Either wire into the PyPI jump model's loss function configuration or remove the import and document that this parameter is not supported by the PyPI backend.

**Acceptance:** Each config knob either affects runtime behavior (verified by changing value and observing different output) or is removed with a deprecation note.

---

### T8: Fix ConfidenceCalibrator Modulo Remapping

**Problem:** `confidence_calibrator.py:181` wraps out-of-range regime IDs with modulo (`regime_id % n_regimes`) instead of rejecting them. This silently maps regime 5 → regime 1 (if 4 regimes), producing wrong calibration factors without warning.

**File:** `regime/confidence_calibrator.py`

**Implementation:**
1. Replace modulo with validation:
   ```python
   if regime_id < 0 or regime_id >= self.n_regimes:
       logger.warning("Invalid regime_id %d (expected 0-%d), returning uncalibrated confidence",
                       regime_id, self.n_regimes - 1)
       return raw_confidence  # Pass through uncalibrated
   ```
2. Similarly, in `consensus.py:104-121`, validate regime label bounds in `RegimeConsensus.compute_consensus()`:
   ```python
   for label in regime_labels:
       if label < 0 or label >= self.n_regimes:
           logger.warning("Out-of-range regime label %d in consensus input", label)
   valid_labels = [l for l in regime_labels if 0 <= l < self.n_regimes]
   ```

**Acceptance:** An out-of-range regime ID returns uncalibrated confidence with a warning, not a silently wrong calibration. Consensus computation excludes invalid labels.

---

## Verification

- [ ] Run `pytest tests/ -k "regime"` — all pass
- [ ] Verify `RegimeDetector(method="ensemble").detect_full()` returns ensemble output
- [ ] Verify min-duration tie-break changes direction when confidence changes
- [ ] Verify PyPI jump failure falls back to legacy before rules
- [ ] Verify ShockVector with (3,3) transition matrix passes validation
- [ ] Verify backtest-mode HMM uses expanding-window standardization
- [ ] Verify empty DataFrame input does not crash detect_with_shock_context
