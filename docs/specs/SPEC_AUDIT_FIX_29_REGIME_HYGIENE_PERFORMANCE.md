# SPEC_AUDIT_FIX_29: Regime Subsystem Hygiene & Performance

**Priority:** LOW-MEDIUM — Dead config import, O(n²) performance issue, unbounded memory growth, and confusing naming.
**Scope:** `regime/jump_model_pypi.py`, `regime/shock_vector.py`, `regime/consensus.py`, `regime/correlation.py`
**Estimated effort:** 2 hours
**Depends on:** Nothing
**Blocks:** Nothing

---

## Context

Four maintenance/hygiene issues were identified. First (F-03/P1), `REGIME_JUMP_MODE_LOSS_WEIGHT` (a float, value 0.1) is imported at jump_model_pypi.py:91 but never used — the constructor at line 81 takes `mode_loss: bool = True`, stores the boolean at line 103, and passes the boolean to the PyPI JumpModel at line 167. The float config is dead code indicating either an incomplete integration or an abandoned feature. This finding enhances what was partially noted in SPEC_15 T7 by identifying the root cause: a type mismatch between the config (float weight) and the parameter (boolean toggle).

Second (F-09/P3), `compute_shock_vectors()` at shock_vector.py:457-461 computes per-bar systemic stress using an expanding percentile loop: for each bar `i`, it slices `rolling_vol[:i+1]` and counts how many elements are below the current value. This is O(n²) — for 2500 bars × 500 tickers, it's ~1.5B comparisons.

Third (F-11/P3), `RegimeConsensus.compute_consensus()` at consensus.py:127 appends to `self._consensus_history` on every call with no size limit. Only `reset_history()` (line 271) or `compute_consensus_series()` (line 244) clear it. Over long-running processes (e.g., daily autopilot iterations over months), memory grows without bound.

Fourth (F-13/P3), `CorrelationRegimeDetector.detect_correlation_spike()` at correlation.py:152 returns a binary {0, 1} series named `corr_regime`. This differs from the canonical 4-regime mapping (0=trending_bull, 1=trending_bear, 2=mean_reverting, 3=high_volatility) used everywhere else. The output is used as a feature column by features/pipeline.py (not as a regime label), so there's no semantic conflict in practice, but the naming `corr_regime` could mislead developers into thinking it follows the canonical convention.

---

## Tasks

### T1: Resolve REGIME_JUMP_MODE_LOSS_WEIGHT Dead Import

**Problem:** jump_model_pypi.py:91 imports `REGIME_JUMP_MODE_LOSS_WEIGHT` (float 0.1) from config. But the constructor at line 81 defines `mode_loss: bool = True`, stores it at line 103 as `self.mode_loss = mode_loss` (boolean), and passes it at line 167 as `mode_loss=self.mode_loss` (boolean). The float config value is never referenced after import. The PyPI `JumpModel` package may or may not support a separate weight parameter — this needs investigation.

**File:** `regime/jump_model_pypi.py`

**Implementation:**
1. Check the PyPI `jumpmodels` package API to determine if `JumpModel` accepts a `mode_loss_weight` parameter:
   ```python
   # In a Python REPL or by reading the jumpmodels source:
   import inspect
   from jumpmodels.jump import JumpModel
   print(inspect.signature(JumpModel.__init__))
   print(inspect.signature(JumpModel.fit))
   ```
2. **If the package supports a weight parameter:** Wire the config value:
   ```python
   def __init__(self, ..., mode_loss: bool = True):
       from config import REGIME_JUMP_MODE_LOSS_WEIGHT
       self.mode_loss = mode_loss
       self.mode_loss_weight = float(REGIME_JUMP_MODE_LOSS_WEIGHT) if mode_loss else 0.0

   # At line 167:
   model = JumpModel(
       ...,
       mode_loss=self.mode_loss,
       mode_loss_weight=self.mode_loss_weight,  # NEW
   )
   ```
3. **If the package does NOT support a weight parameter:** Remove the dead import and mark the config constant as DEPRECATED:
   ```python
   # Remove from jump_model_pypi.py:91 import list:
   # REGIME_JUMP_MODE_LOSS_WEIGHT,  # REMOVED — PyPI JumpModel only accepts boolean toggle

   # In config.py, update the constant's status comment:
   REGIME_JUMP_MODE_LOSS_WEIGHT = 0.1  # STATUS: DEPRECATED — PyPI JumpModel only supports bool toggle
   ```

**Acceptance:** Either the config float controls the mode loss weight in the jump model (if supported), or the dead import is removed and the config is marked deprecated. No imported-but-unused constants remain.

---

### T2: Vectorize Systemic Stress Computation to O(n)

**Problem:** shock_vector.py:457-461 computes an expanding percentile with an explicit Python loop:
```python
for i in range(vol_lookback, n):
    hist = rolling_vol[:i + 1]
    if len(hist) > 1 and np.max(hist) > 1e-10:
        percentile = float(np.sum(hist <= rolling_vol[i])) / len(hist)
        systemic_stress[i] = float(np.clip(percentile, 0.0, 1.0))
```
For each bar, `np.sum(hist <= rolling_vol[i])` compares against all prior bars — O(n²) total.

**File:** `regime/shock_vector.py`

**Implementation:**
Replace the loop with a vectorized expanding rank:
```python
# BEFORE (lines 453-461):
systemic_stress = np.zeros(n)
# ... loop ...

# AFTER:
vol_series = pd.Series(rolling_vol)
# expanding().rank() computes the rank of each element within all elements up to that point
# Dividing by the expanding count gives the percentile
expanding_rank = vol_series.expanding(min_periods=2).rank()
expanding_count = vol_series.expanding(min_periods=2).count()
systemic_stress_series = (expanding_rank / expanding_count).fillna(0.0).clip(0.0, 1.0)

# Zero out the first vol_lookback bars (no valid rolling vol)
systemic_stress_series.iloc[:vol_lookback] = 0.0

# Apply the max-check: if expanding max of rolling_vol is ≤ 1e-10, set to 0
expanding_max = vol_series.expanding(min_periods=1).max()
systemic_stress_series[expanding_max <= 1e-10] = 0.0

systemic_stress = systemic_stress_series.values
```

This is O(n) with pandas internals. For 2500 bars × 500 tickers, this eliminates ~1.5B Python-level comparisons.

**Acceptance:** Systemic stress values are numerically identical (within float tolerance) to the original loop. Backtest results are unchanged. For a 2500-bar series, computation time drops from seconds to milliseconds.

---

### T3: Cap Consensus History Growth

**Problem:** consensus.py:127 — `compute_consensus()` appends to `self._consensus_history` without limit. Over long-running processes, memory grows without bound. Only `reset_history()` (line 271) clears it.

**File:** `regime/consensus.py`

**Implementation:**
1. Add a max history length, derived from the divergence detection window:
   ```python
   def __init__(self, ...):
       # ... existing init ...
       # Cap history to 2x the divergence window — anything older is never accessed
       self._max_history_length = max(
           2 * self._divergence_window,
           100,  # Minimum floor
       )
       self._consensus_history: List[float] = []
   ```
2. After appending in `compute_consensus()`, trim if over capacity:
   ```python
   # Line 127:
   self._consensus_history.append(consensus_value)

   # NEW: Trim to max length
   if len(self._consensus_history) > self._max_history_length:
       # Keep only the most recent entries
       self._consensus_history = self._consensus_history[-self._max_history_length:]
   ```
3. Verify that `detect_divergence()` only accesses the last `self._divergence_window` entries — if so, the trim is safe.

**Acceptance:** After 10,000 calls to `compute_consensus()`, `len(self._consensus_history)` is capped at `_max_history_length`. `detect_divergence()` continues to work correctly.

---

### T4: Rename corr_regime to Avoid Canonical Regime ID Confusion

**Problem:** correlation.py:152 — `detect_correlation_spike()` returns a binary {0, 1} series named `corr_regime`. The canonical regime mapping uses 0-3 with semantic labels (trending_bull, trending_bear, etc.). The name `corr_regime` suggests it follows the canonical convention, but it's actually a binary stress indicator used as a feature column.

**File:** `regime/correlation.py`

**Implementation:**
1. Rename the output column from `corr_regime` to `corr_stress_flag`:
   ```python
   # BEFORE (line 152):
   return (self._avg_corr >= thr).astype(int).rename("corr_regime")

   # AFTER:
   return (self._avg_corr >= thr).astype(int).rename("corr_stress_flag")
   ```
2. Update all consumers that reference `corr_regime` as a column name:
   - Search for `"corr_regime"` across the codebase
   - Update `features/pipeline.py` where it accesses this column
   - Update any test files that reference `corr_regime`
3. If backward compatibility is needed (e.g., cached feature stores), add an alias:
   ```python
   # In features/pipeline.py, where corr features are consumed:
   if "corr_regime" in df.columns and "corr_stress_flag" not in df.columns:
       df = df.rename(columns={"corr_regime": "corr_stress_flag"})
   ```

**Acceptance:** No column named `corr_regime` exists in the feature pipeline output. `corr_stress_flag` clearly communicates its binary stress-indicator nature. Existing tests pass with the renamed column.

---

## Verification

- [ ] Run `pytest tests/ -k "regime or correlation or consensus or jump"` — all pass
- [ ] Verify REGIME_JUMP_MODE_LOSS_WEIGHT is either wired or removed
- [ ] Verify systemic stress computation produces identical results to original loop
- [ ] Verify consensus history is bounded after many calls
- [ ] Verify `corr_stress_flag` column is used everywhere `corr_regime` was
- [ ] Benchmark systemic stress: verify speedup for 2500-bar series
