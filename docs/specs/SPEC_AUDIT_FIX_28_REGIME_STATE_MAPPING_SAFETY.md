# SPEC_AUDIT_FIX_28: Regime State Mapping & Online Update Safety

**Priority:** MEDIUM-HIGH — Production code uses unstable heuristic state mapping when a stable Wasserstein-based version exists; online updater can produce out-of-range regime IDs; duration smoothing creates label/probability inconsistency.
**Scope:** `regime/detector.py`, `regime/hmm.py`, `regime/online_update.py`
**Estimated effort:** 3 hours
**Depends on:** Nothing
**Blocks:** Nothing

---

## Context

Three state-assignment correctness issues were identified. First (F-07/P2), `_hmm_detect()` at detector.py:280 calls `map_raw_states_to_regimes()` imported from hmm.py:612-660 — a heuristic that assigns semantic labels based on per-state mean return, NATR, SMA slope, and Hurst exponent. This heuristic is unstable across HMM refits because the EM algorithm's label permutation problem means the same market conditions can produce different raw state assignments. Meanwhile, detector.py:638-711 defines `map_raw_states_to_regimes_stable()`, a Wasserstein distance matching method specifically designed to solve this problem — but it is never called from the production `_hmm_detect` path.

Second (F-08/P2), `OnlineRegimeUpdater.__init__()` at online_update.py:36-65 accepts any `GaussianHMM` instance and caches `n_states` at line 67 without validating it equals 4. The `update_regime_for_security()` method at line 177 returns `int(np.argmax(state_prob))`, which can produce values 4 or 5 when BIC auto-selection (`REGIME_HMM_AUTO_SELECT_STATES=True`) picks more than 4 states. These values are outside the canonical [0,3] range and would trigger `ValueError` in ShockVector's `__post_init__`.

Third (F-12/P3), `GaussianHMM._smooth_duration()` at hmm.py:257-290 modifies `states[i:j]` for short runs but does not update the corresponding rows in `state_probs`. The `fit()` method at hmm.py:342 calls `_smooth_duration(raw_states, probs)`, then at hmm.py:344-348 returns `HMMFitResult(raw_states=raw_states, state_probs=probs)` — smoothed labels with unsmoothed probabilities. detector.py:297 then computes `confidence = probs.max(axis=1)` from the raw probabilities, creating an inconsistency where the confidence doesn't reflect the actual (smoothed) regime label for bars where smoothing changed the assignment.

---

## Tasks

### T1: Switch Production Detection to Wasserstein-Based Stable State Mapping

**Problem:** detector.py:280 calls `map_raw_states_to_regimes(raw_states, features)` — the heuristic version from hmm.py:612-660. The stable version `map_raw_states_to_regimes_stable()` at detector.py:638-711 uses Wasserstein distance matching to reference distributions, solving the EM label permutation problem, but is never used in production.

**File:** `regime/detector.py`

**Implementation:**
1. In `_hmm_detect()`, replace the heuristic call with the stable version:
   ```python
   # BEFORE (line 280):
   mapping = map_raw_states_to_regimes(raw_states, features)

   # AFTER:
   mapping = self.map_raw_states_to_regimes_stable(
       raw_states=raw_states,
       features=features,
       state_probs=raw_probs,
   )
   ```
2. Similarly, update `_jump_detect()` which also calls the heuristic version (verify the line — likely around detector.py:336):
   ```python
   # In _jump_detect():
   mapping = self.map_raw_states_to_regimes_stable(
       raw_states=raw_states,
       features=features,
       state_probs=raw_probs,
   )
   ```
3. Verify that `map_raw_states_to_regimes_stable` at detector.py:638-711:
   - Accepts the same parameters that the callers provide (raw_states, features, state_probs)
   - Returns the same type (a mapping dict `{raw_state: canonical_regime}`)
   - Has reference distributions initialized (check `self._reference_distributions` or similar)
   - Falls back gracefully if reference distributions are not available (first run)
4. If the stable version requires reference distributions that don't exist on first run, add a fallback:
   ```python
   try:
       mapping = self.map_raw_states_to_regimes_stable(...)
   except (ValueError, AttributeError):
       logger.warning("Stable mapping unavailable (first fit?), using heuristic fallback")
       mapping = map_raw_states_to_regimes(raw_states, features)
   ```
5. Consider deprecating the heuristic `map_raw_states_to_regimes()` in hmm.py or marking it as internal-only.

**Acceptance:** After HMM refit on the same data with different random seeds, regime labels remain stable (same market periods get same canonical regime IDs). A test fits the HMM twice with different seeds and verifies label agreement is >95%.

---

### T2: Add State Count Validation to OnlineRegimeUpdater

**Problem:** `online_update.py:36-65` — `__init__` accepts any `GaussianHMM` without checking `n_states`. Line 67 stores `self.n_states = hmm_model.n_states` without validation. Line 177 returns `int(np.argmax(state_prob))` which produces values in `[0, n_states-1]`. If `n_states > 4`, regime IDs 4+ are outside canonical [0,3] and will fail ShockVector validation.

**File:** `regime/online_update.py`

**Implementation:**

**Option A (Recommended): Validate state count in __init__:**
```python
def __init__(self, hmm_model: "GaussianHMM", **kwargs):
    if hmm_model.n_states != 4:
        raise ValueError(
            f"OnlineRegimeUpdater requires exactly 4 HMM states (canonical regime mapping), "
            f"got {hmm_model.n_states}. Use map_raw_states_to_regimes to convert "
            f"auto-selected state counts to canonical [0,3] before online updating."
        )
    # ... existing init code ...
```

**Option B: Apply state mapping inside update_regime_for_security:**
If the system must support variable state counts in online mode, apply `map_raw_states_to_regimes_stable` inside the updater:
```python
def update_regime_for_security(self, ...):
    raw_regime = int(np.argmax(state_prob))

    # If HMM has > 4 states, map to canonical [0, 3]
    if self.n_states != 4 and hasattr(self, '_state_mapping'):
        raw_regime = self._state_mapping.get(raw_regime, 2)  # Default to mean_reverting

    return raw_regime
```

Option A is recommended because it catches the misconfiguration early and forces the caller to handle state mapping before entering online update mode.

**Acceptance:** `OnlineRegimeUpdater(hmm_model_with_6_states)` raises `ValueError` with a clear message. A 4-state HMM works normally. Regime IDs from `update_regime_for_security` are always in [0, 3].

---

### T3: Synchronize State Probabilities After Duration Smoothing

**Problem:** `hmm.py:257-290` — `_smooth_duration()` modifies `states[i:j]` for short runs but returns the original `probs` unchanged. The `fit()` method at hmm.py:342-348 returns `HMMFitResult(raw_states=smoothed_states, state_probs=original_probs)`. Then detector.py:297 computes `confidence = probs.max(axis=1)` from the original (unsmoothed) probabilities. For bars where smoothing changed the regime label, the confidence might peak at a different regime than the one reported.

**File:** `regime/hmm.py`

**Implementation:**
1. After modifying states in `_smooth_duration()`, update the corresponding probability rows to be consistent:
   ```python
   def _smooth_duration(
       self,
       states: np.ndarray,
       probs: np.ndarray,
       min_duration: int,
   ) -> Tuple[np.ndarray, np.ndarray]:
       """Smooth short regime runs, updating both states and probabilities."""
       s = states.copy()
       p = probs.copy()  # NEW: also copy probs
       n = len(s)
       # ... existing loop to find short runs ...

       for i, j, repl in runs_to_replace:
           s[i:j] = repl
           # Update probabilities: set the replacement state's probability
           # to the max of original prob and a minimum threshold
           for row_idx in range(i, j):
               original_max = p[row_idx].max()
               p[row_idx, :] *= 0.5  # Reduce all probabilities
               p[row_idx, repl] = max(original_max, 0.6)  # Boost replacement state
               # Renormalize
               row_sum = p[row_idx].sum()
               if row_sum > 0:
                   p[row_idx] /= row_sum

       return s, p
   ```
2. Update the caller in `fit()` to capture both returns:
   ```python
   # BEFORE (line 342):
   raw_states = self._smooth_duration(raw_states, probs)

   # AFTER:
   raw_states, probs = self._smooth_duration(raw_states, probs, self.min_duration)
   ```
3. This ensures that the `state_probs` returned in `HMMFitResult` are consistent with the smoothed states.

**Alternative (simpler):** Instead of modifying probabilities, recompute confidence after smoothing:
```python
# In detector.py, after applying mapping (around line 296):
# For smoothed bars, set confidence to the probability of the assigned regime
for idx in range(len(probs)):
    assigned_regime = mapped_regimes[idx]
    raw_state_for_regime = reverse_mapping.get(assigned_regime)
    if raw_state_for_regime is not None:
        confidence_val = probs[idx, raw_state_for_regime]
    else:
        confidence_val = probs[idx].max()
    confidence[idx] = confidence_val
```

**Acceptance:** For bars where duration smoothing changes the regime label, the reported confidence reflects the smoothed regime (not the original raw state). A test creates a scenario where smoothing changes a regime and verifies confidence alignment.

---

## Verification

- [ ] Run `pytest tests/ -k "regime or hmm or online"` — all pass
- [ ] Verify stable mapping produces consistent labels across HMM refits with different seeds
- [ ] Verify OnlineRegimeUpdater rejects HMM with n_states != 4
- [ ] Verify confidence is consistent with smoothed regime labels after duration smoothing
- [ ] Verify no regression in backtest results (regime labels should be more stable, not different in expected behavior)
