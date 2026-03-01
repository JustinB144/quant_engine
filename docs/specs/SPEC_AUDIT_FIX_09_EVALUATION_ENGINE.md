# SPEC_AUDIT_FIX_09: Evaluation Engine Correctness & Statistical Fixes

**Priority:** CRITICAL/HIGH — Math errors in evaluation affect all model selection and promotion decisions.
**Scope:** `evaluation/` — `engine.py`, `metrics.py`, `fragility.py`, `calibration_analysis.py`, `slicing.py`
**Estimated effort:** 4–5 hours
**Depends on:** Nothing
**Blocks:** Nothing

---

## Context

The evaluation engine has critical math and logic errors: calibration analysis crashes on mismatched inputs, all subsystem exceptions are silently swallowed (fail-open), regime/uncertainty slices never compute IC despite it being a core metric, the annualization formula overstates returns for volatile strategies, decile binning is imbalanced, PnL concentration misreports trade count, and recovery-time indexing breaks with NaN data.

---

## Tasks

### T1: Fix analyze_calibration Crash on Mismatched Input Length

**Problem:** `calibration_analysis.py:74-77` slices `confidence_scores` then boolean-indexes with a mask from a different-length array, causing `IndexError` before the fallback logic runs.

**File:** `evaluation/calibration_analysis.py`

**Implementation:**
1. Validate input lengths upfront:
   ```python
   if confidence_scores is not None:
       conf_arr = np.asarray(confidence_scores, dtype=float).ravel()
       if len(conf_arr) != len(pred_arr):
           logger.warning("confidence_scores length %d != predictions %d, using prediction magnitude",
                          len(conf_arr), len(pred_arr))
           conf_arr = np.abs(pred_arr) / (np.abs(pred_arr).max() + 1e-12)
       else:
           conf_arr = conf_arr[valid]  # Apply valid mask AFTER length check
   ```
2. Add test: `analyze_calibration(preds=[1,2,3], returns=[1,2,3], confidence_scores=[0.5, 0.6])` does not crash.

**Acceptance:** Mismatched confidence_scores length produces a warning and falls back gracefully, never crashes.

---

### T2: Fix Fail-Open Exception Handling in EvaluationEngine

**Problem:** `engine.py:236,253,308,339,366,454` wraps every subsystem call in try/except that logs a warning and continues. A run can be "PASS" even when critical analyses failed silently.

**File:** `evaluation/engine.py`

**Implementation:**
1. Track failed subsystems:
   ```python
   failed_subsystems = []
   ```
2. In each except block, append to failed_subsystems:
   ```python
   except Exception as exc:
       logger.warning("Subsystem %s failed: %s", subsystem_name, exc)
       failed_subsystems.append(subsystem_name)
   ```
3. Include `failed_subsystems` in the evaluation result.
4. In the overall pass/fail logic (lines 457-468), if any **critical** subsystem failed (walk-forward, calibration, IC), set `overall_pass = False` regardless of red flags:
   ```python
   critical_subsystems = {"walk_forward", "calibration", "ic_analysis"}
   if critical_subsystems & set(failed_subsystems):
       overall_pass = False
       summary_parts.append(f"FAIL (subsystems failed: {failed_subsystems})")
   ```

**Acceptance:** If walk-forward analysis throws an exception, overall result is FAIL, not PASS. `failed_subsystems` list is in the result.

---

### T3: Pass Predictions to Slice Metrics for IC Computation

**Problem:** `engine.py:215,223,233,250` calls `compute_slice_metrics(filtered, None)` for all regime/uncertainty/transition slices, forcing IC to be 0.0 because predictions are not passed.

**File:** `evaluation/engine.py`

**Implementation:**
1. Pass the predictions series to slice calls:
   ```python
   # OLD: metrics = compute_slice_metrics(filtered, None)
   # NEW:
   slice_predictions = predictions[mask] if predictions is not None else None
   metrics = compute_slice_metrics(filtered, slice_predictions)
   ```
2. Ensure `predictions` variable is available in the scope where slicing occurs. It should be the model's prediction series aligned with returns.
3. In `metrics.py`, verify `compute_slice_metrics` correctly computes Spearman IC when predictions are provided.

**Acceptance:** Regime slice metrics include non-zero IC values when predictions are available.

---

### T4: Fix Annualized Return Formula

**Problem:** `metrics.py:65` uses `(1 + mean_return)^252 - 1` which compounds arithmetic mean daily return. For volatile strategies, this overstates annualized return. Should use geometric mean or log-return CAGR.

**File:** `evaluation/metrics.py`

**Implementation:**
1. Replace with geometric compounding:
   ```python
   # Geometric annualized return (CAGR-equivalent):
   cumulative = np.prod(1 + ret_arr)
   n_years = len(ret_arr) / annual_trading_days
   ann_return = float(cumulative ** (1 / n_years) - 1) if n_years > 0 else 0.0
   ```
2. Keep arithmetic annualized return as a separate field for comparison:
   ```python
   ann_return_arithmetic = float((1 + mean_ret) ** annual_trading_days - 1)
   ```
3. Use geometric as the primary reported metric.

**Acceptance:** Annualized return for a volatile strategy (e.g., 0.1% mean daily return with 3% daily vol) is materially lower under geometric vs arithmetic compounding. Both values available.

---

### T5: Fix Decile Binning to Use True Quantiles

**Problem:** `metrics.py:183-184` uses `(rank_pct * n_quantiles).astype(int)` which creates imbalanced bins (edge bins get +/- 1 observation).

**File:** `evaluation/metrics.py`

**Implementation:**
1. Use `pd.qcut` for true quantile binning:
   ```python
   bins = pd.qcut(pred_arr, q=n_quantiles, labels=False, duplicates='drop')
   ```
2. Or use numpy percentile-based binning:
   ```python
   percentiles = np.linspace(0, 100, n_quantiles + 1)
   bin_edges = np.percentile(pred_arr, percentiles)
   bins = np.digitize(pred_arr, bin_edges[1:-1])  # 0 to n_quantiles-1
   ```
3. Verify bin counts are balanced (±1 observation for remainder).

**Acceptance:** Decile bins with 100 perfectly ranked points produce counts of [10, 10, 10, 10, 10, 10, 10, 10, 10, 10].

---

### T6: Fix pnl_concentration Zero-Net-PnL Bug

**Problem:** `fragility.py:69-70` returns `n_trades=0` when `abs(total_pnl) < 1e-12`, even if trades exist.

**File:** `evaluation/fragility.py`

**Implementation:**
1. Separate the empty-trades check from the zero-PnL check:
   ```python
   if n_trades == 0:
       return {"total_pnl": 0.0, "n_trades": 0, ...}
   if abs(total_pnl) < 1e-12:
       return {"total_pnl": float(total_pnl), "n_trades": n_trades,
               "concentration": "undefined (net PnL ≈ 0)", ...}
   ```
2. Report actual n_trades in both cases.

**Acceptance:** `pnl_concentration([{"pnl": 1.0}, {"pnl": -1.0}])` returns `n_trades=2`, not `n_trades=0`.

---

### T7: Fix Recovery-Time NaN Index Misalignment

**Problem:** `fragility.py:209-210,242` filters NaN values from returns (changing array length) but maps trough indices back to the original unfiltered index.

**File:** `evaluation/fragility.py`

**Implementation:**
1. Instead of filtering NaN values, replace them:
   ```python
   ret_arr = returns.values.astype(float)
   ret_arr = np.where(np.isfinite(ret_arr), ret_arr, 0.0)  # Keep index alignment
   ```
2. Or preserve the mapping between filtered and original indices:
   ```python
   valid_mask = np.isfinite(ret_arr)
   valid_indices = np.where(valid_mask)[0]
   filtered = ret_arr[valid_mask]
   # ... compute trough_indices on filtered ...
   original_trough_indices = valid_indices[trough_indices]
   ```

**Acceptance:** Recovery time computation with NaN-containing returns produces correctly aligned trough dates.

---

### T8: Honor Window/Lookback Parameters in Fragility Functions

**Problem:** `fragility.py:108,189,250` accept `window`/`lookback` parameters that are documented but never used in computation.

**File:** `evaluation/fragility.py`

**Implementation:**
1. In `drawdown_distribution()`: use `window` to limit the analysis to the last `window` observations.
2. In `recovery_time_distribution()`: use `lookback` to limit the lookback period.
3. In `tail_risk_contribution()`: use `window` to limit the analysis window.
4. If the parameters are intentionally unused, remove them from the signature and update docstrings.

**Acceptance:** `drawdown_distribution(returns, window=10)` produces different output than `window=500`.

---

### T9: Fix Metadata fillna(0.0) Fabricating States

**Problem:** `slicing.py:399` applies `meta.fillna(0.0)` which converts NaN regime to 0 (trending_bull), NaN uncertainty to 0.0 (certain), fabricating fake states.

**File:** `evaluation/slicing.py`

**Implementation:**
1. Remove blanket `fillna(0.0)`.
2. Handle NaN values per-column with appropriate defaults:
   ```python
   # Regime: NaN → -1 (unknown)
   if 'regime' in meta.columns:
       meta['regime'] = meta['regime'].fillna(-1)
   # Uncertainty: NaN → 1.0 (maximally uncertain)
   if 'uncertainty' in meta.columns:
       meta['uncertainty'] = meta['uncertainty'].fillna(1.0)
   # Cumulative return: NaN → 0.0 (reasonable default)
   if 'cumulative_return' in meta.columns:
       meta['cumulative_return'] = meta['cumulative_return'].fillna(0.0)
   ```
3. In slicing logic, exclude regime=-1 from regime-specific slices.

**Acceptance:** NaN regime values are classified as "unknown", not as "trending_bull". Unknown regime slice is separate.

---

## Verification

- [ ] Run `pytest tests/ -k "evaluation or metrics or fragility or calibration"` — all pass
- [ ] Verify analyze_calibration with mismatched lengths doesn't crash
- [ ] Verify overall FAIL when walk-forward subsystem throws
- [ ] Verify regime slices have non-zero IC when predictions available
- [ ] Verify geometric annualized return < arithmetic for volatile data
- [ ] Verify balanced decile bins with ordered data
- [ ] Verify pnl_concentration reports correct n_trades
