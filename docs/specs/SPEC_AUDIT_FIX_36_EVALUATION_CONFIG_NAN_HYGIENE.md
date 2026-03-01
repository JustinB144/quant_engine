# SPEC_AUDIT_FIX_36: Evaluation Config Wiring, NaN Safety & Hygiene

**Priority:** HIGH — NaN handling silently zeros IC for any slice with missing data; config thresholds are dead code.
**Scope:** `evaluation/metrics.py`, `evaluation/engine.py`, `evaluation/fragility.py`, `evaluation/visualization.py`, `evaluation/slicing.py`, `evaluation/__init__.py`
**Estimated effort:** 4–5 hours
**Depends on:** SPEC_09 (core evaluation fixes must land first)
**Blocks:** Nothing

---

## Context

Two independent audits of Subsystem 7 (Evaluation & Diagnostics) surfaced findings that the existing SPEC_09 does not cover. SPEC_09 addresses core math/logic errors (calibration crash, fail-open exceptions, missing IC in slices, annualization, decile binning, zero-net-PnL, recovery-time NaN, unused params, metadata fillna). This spec covers the *remaining* gaps: NaN-handling in `compute_slice_metrics` that silently zeros IC and misreports date ranges, five config thresholds that are imported but never wired into active logic, NaN propagation in underwater visualization, silent regime fallback on length mismatch, and several LOW/INFO hygiene items.

### Cross-Audit Reconciliation

| Finding | Auditor 1 | Auditor 2 | Existing Spec | Disposition |
|---------|-----------|-----------|---------------|-------------|
| NaN zeros IC / misreports dates | — | F-02 (P1) | Not in SPEC_09 | **NEW → T1** |
| Hardcoded fragility threshold 0.70 | M-1 | F-03 (partial) | SPEC_09 T6 covers n_trades bug, NOT threshold | **NEW → T2** |
| EVAL_IC_DECAY_LOOKBACK not forwarded | M-2 | F-03 (partial) | Not in SPEC_09 | **NEW → T2** |
| EVAL_OVERFIT_GAP_THRESHOLD unused | M-3 | F-03 (partial) | Not in SPEC_09 | **NEW → T2** |
| Underwater visualization NaN | — | F-05 (P2) | Not in SPEC_09 | **NEW → T3** |
| Silent regime fallback on mismatch | L-5 | — | Not in SPEC_09 | **NEW → T4** |
| EVAL_DECILE_SPREAD_MIN unused | L-4 | F-06 (P3) | Not in SPEC_09 | **NEW → T5** |
| consecutive_loss_frequency not exported | L-1 | — | Not in SPEC_09 | **NEW → T6** |
| Redundant .get() in visualization | L-2 | — | Not in SPEC_09 | **NEW → T7** |
| Interface contracts doc stale | I-2 | F-07 (P3) | Not in SPEC_09 | **NEW → T8** |
| Unused params (window/lookback) | — | F-04 (P2) | **SPEC_09 T8** | Already covered |
| PnL concentration zero-net-PnL | — | F-01 (P1) | **SPEC_09 T6** | Already covered |
| scipy fallback degrades IC | I-1 | — | — | Accepted risk (scipy always installed) |
| Trailing return arithmetic sum | I-3 | — | — | Accepted (negligible for 20d window) |
| Transition slices center=True | I-4 | — | — | By design (post-hoc analysis) |
| Five dead imports in engine.py | L-3 | — | — | Resolved by T2 (wiring fixes) |

---

## Tasks

### T1: Fix NaN Handling in compute_slice_metrics (IC and Date Range)

**Problem:** `metrics.py:55-58` filters returns for finites and recomputes `n`, but does NOT apply the same mask to predictions. At line 106, IC is only computed when `len(pred_arr) == n`. If any return is NaN, `n` shrinks but `pred_arr` retains original length → length mismatch → IC defaults to 0.0. Additionally, `start_date`/`end_date` at lines 121-122 use the original `returns.index` instead of the filtered range, misreporting the evaluated period.

**Files:** `evaluation/metrics.py`

**Implementation:**
1. Build a single validity mask over both returns and predictions, then apply it to both:
   ```python
   ret_arr = returns.values.astype(float).ravel()

   # Build unified validity mask
   valid_mask = np.isfinite(ret_arr)

   if predictions is not None:
       pred_arr = np.asarray(predictions, dtype=float).ravel()
       if len(pred_arr) == len(ret_arr):
           valid_mask &= np.isfinite(pred_arr)
           pred_arr = pred_arr[valid_mask]
       else:
           logger.warning(
               "Predictions length %d != returns length %d, IC will be skipped",
               len(pred_arr), len(ret_arr),
           )
           pred_arr = None
   else:
       pred_arr = None

   # Filter returns with the same mask
   ret_arr = ret_arr[valid_mask]
   n = len(ret_arr)

   if n < min_samples:
       return _empty_metrics(returns)
   ```
2. Fix date range to use the filtered index:
   ```python
   # Use filtered index for accurate date reporting
   if hasattr(returns, 'index'):
       filtered_index = returns.index[valid_mask]
       start_date = str(filtered_index[0]) if len(filtered_index) > 0 else "N/A"
       end_date = str(filtered_index[-1]) if len(filtered_index) > 0 else "N/A"
   else:
       start_date = "N/A"
       end_date = "N/A"
   ```
3. IC computation can now simply check `if pred_arr is not None and len(pred_arr) == n` — always true when both arrays were filtered by the same mask.

**Acceptance:** `compute_slice_metrics(returns_with_NaN, predictions)` produces correct non-zero IC when the underlying correlation exists. Reported dates exclude NaN endpoints.

---

### T2: Wire All Config Thresholds Into Active Decision Logic

**Problem:** Five evaluation config constants are imported by `engine.py` but never used in decision logic. Changing them in `config.py` has no effect:

| Constant | Config Value | Where Imported | Why Dead |
|----------|-------------|----------------|----------|
| `EVAL_PNL_CONCENTRATION_THRESHOLD` | 0.70 | engine.py:41 | Hardcoded 0.70 in fragility.py:101 |
| `EVAL_IC_DECAY_LOOKBACK` | 20 | engine.py:34 | Not passed to `detect_ic_decay()` at engine.py:327 |
| `EVAL_OVERFIT_GAP_THRESHOLD` | 0.10 | engine.py:40 | Engine uses only `wf.is_overfit` (backtest internal threshold) |
| `EVAL_RECOVERY_WINDOW` | engine.py:37 | engine.py:37 | Passed to fragility.py but parameter is unused (SPEC_09 T8) |
| `EVAL_CRITICAL_SLOWING_WINDOW` | engine.py:38 | engine.py:38 | Passed to fragility.py but parameter is unused (SPEC_09 T8) |

**Files:** `evaluation/engine.py`, `evaluation/fragility.py`

**Implementation:**
1. **EVAL_PNL_CONCENTRATION_THRESHOLD:** Pass from engine to fragility:
   ```python
   # engine.py — in the fragility analysis call:
   concentration = pnl_concentration(
       trades, top_n=self.top_n_trades,
       fragile_threshold=EVAL_PNL_CONCENTRATION_THRESHOLD,
   )

   # fragility.py — update pnl_concentration signature:
   def pnl_concentration(trades, top_n=10, fragile_threshold=0.70):
       ...
       result["fragile"] = result.get(fragile_key, 0.0) > fragile_threshold
   ```
2. **EVAL_IC_DECAY_LOOKBACK:** Forward to detect_ic_decay:
   ```python
   # engine.py line 327:
   decaying, decay_info = detect_ic_decay(
       ic_series,
       decay_threshold=self.ic_decay_threshold,
       window=EVAL_IC_DECAY_LOOKBACK,
   )
   ```
3. **EVAL_OVERFIT_GAP_THRESHOLD:** Apply as a secondary check alongside `wf.is_overfit`:
   ```python
   # engine.py line 301:
   if wf.is_overfit or wf.mean_overfit_gap > EVAL_OVERFIT_GAP_THRESHOLD:
       red_flags.append(RedFlag(
           category="overfit",
           severity="critical",
           message=f"Walk-forward overfit detected (gap={wf.mean_overfit_gap:.3f}, "
                   f"threshold={EVAL_OVERFIT_GAP_THRESHOLD})",
       ))
   ```
4. **EVAL_RECOVERY_WINDOW / EVAL_CRITICAL_SLOWING_WINDOW:** These will be wired in SPEC_09 T8 (which implements the unused parameters). No duplicate work needed here — just verify they chain correctly after SPEC_09 lands.
5. Remove the five dead imports from engine.py (L-3) since they will now be actively used.

**Acceptance:** Changing `EVAL_PNL_CONCENTRATION_THRESHOLD` in config changes the fragility flag behavior. Changing `EVAL_IC_DECAY_LOOKBACK` changes the decay detection window. Changing `EVAL_OVERFIT_GAP_THRESHOLD` changes when overfit red flags fire.

---

### T3: Fix Underwater Visualization NaN Propagation

**Problem:** `visualization.py:190-193` computes `np.cumprod(1 + ret_arr)` directly from raw returns without NaN filtering. A single NaN propagates through the entire cumulative product, corrupting all subsequent drawdown values.

**File:** `evaluation/visualization.py`

**Implementation:**
1. Filter or fill NaN before cumulative computation:
   ```python
   def plot_underwater(returns, ...):
       ...
       ret_arr = returns.values.astype(float)
       # Forward-fill NaN to preserve index alignment, then mask for display
       valid = np.isfinite(ret_arr)
       if not np.all(valid):
           logger.info(
               "plot_underwater: %d NaN returns forward-filled with 0.0",
               (~valid).sum(),
           )
           ret_arr = np.where(valid, ret_arr, 0.0)

       cum_eq = np.cumprod(1 + ret_arr)
       running_max = np.maximum.accumulate(cum_eq)
       drawdowns = (cum_eq - running_max) / np.where(running_max > 0, running_max, 1.0)
       ...
   ```
2. Alternative: use `dropna()` and preserve date alignment via the index (preferred if the chart should gap over NaN dates):
   ```python
   clean = returns.dropna()
   if len(clean) == 0:
       return {"html": None, "data": {}}
   ret_arr = clean.values.astype(float)
   dates = clean.index
   ```

**Acceptance:** `plot_underwater(pd.Series([0.01, np.nan, 0.02]))` produces a valid drawdown chart, not a NaN-tailed series.

---

### T4: Fix Silent Regime Fallback on Length Mismatch

**Problem:** `slicing.py:356-359` silently fills all bars with regime 2 (mean_reverting) when `regime_states` length doesn't match returns length. No warning is logged. This masks data alignment bugs where the wrong regime array is passed.

**File:** `evaluation/slicing.py`

**Implementation:**
1. Add a warning when the fallback fires:
   ```python
   if len(regime_states) == n:
       meta["regime"] = regime_states
   else:
       logger.warning(
           "regime_states length %d != returns length %d — "
           "defaulting all bars to regime -1 (unknown). "
           "This may indicate a data alignment bug.",
           len(regime_states), n,
       )
       meta["regime"] = np.full(n, -1, dtype=int)  # -1 = unknown, not 2
   ```
2. Change the default from `2` (mean_reverting, a valid regime) to `-1` (unknown). This prevents false regime attribution and makes the issue visible in slice reports as an "Unknown" regime category.
3. In `_build_regime_slices()`, exclude regime -1 from named slices and optionally create an "Unknown" slice:
   ```python
   if code == -1:
       slice_name = "Unknown"
   ```

**Acceptance:** A regime_states/returns length mismatch produces a warning log and creates an "Unknown" regime slice, not a "Mean Reverting" slice.

---

### T5: Wire EVAL_DECILE_SPREAD_MIN Into Red-Flag Logic

**Problem:** `metrics.py:20` imports `EVAL_DECILE_SPREAD_MIN` (0.005) but never uses it. The constant was presumably intended to flag strategies where the decile spread is too small (i.e., the model lacks predictive power across the cross-section). This check is not implemented.

**Files:** `evaluation/metrics.py`, `evaluation/engine.py`

**Implementation:**
1. In `engine.py`, after computing decile spread, add a red flag for insufficient spread:
   ```python
   if decile_result.get("spread", 0.0) < EVAL_DECILE_SPREAD_MIN:
       red_flags.append(RedFlag(
           category="weak_signal",
           severity="warning",
           message=f"Decile spread {decile_result['spread']:.4f} below minimum "
                   f"threshold {EVAL_DECILE_SPREAD_MIN}",
       ))
   ```
2. Move the import from `metrics.py` (where it's unused) to `engine.py` (where it's consumed), or import in both if metrics.py will also use it for validation.
3. Remove the dead import from `metrics.py` if it won't be used there.

**Acceptance:** A strategy with decile spread = 0.002 produces a `weak_signal` warning red flag.

---

### T6: Export consecutive_loss_frequency From `__init__.py`

**Problem:** `fragility.py:consecutive_loss_frequency` is used by `engine.py:420` but is not listed in `evaluation/__init__.py`'s `__all__` or imports. External consumers cannot access it via `from evaluation import consecutive_loss_frequency`.

**File:** `evaluation/__init__.py`

**Implementation:**
1. Add to `__init__.py`:
   ```python
   from .fragility import consecutive_loss_frequency
   ```
2. Add to `__all__`:
   ```python
   __all__ = [
       ...
       "consecutive_loss_frequency",
   ]
   ```

**Acceptance:** `from evaluation import consecutive_loss_frequency` works.

---

### T7: Fix Redundant .get() in plot_walk_forward_folds

**Problem:** `visualization.py:351-352` has redundant nested `.get()` calls where inner and outer keys are identical:
```python
is_sharpes = [f.get("train_sharpe", f.get("train_sharpe", 0.0)) for f in folds]
oos_sharpes = [f.get("test_sharpe", f.get("test_sharpe", 0.0)) for f in folds]
```

**File:** `evaluation/visualization.py`

**Implementation:**
1. Simplify to:
   ```python
   is_sharpes = [f.get("train_sharpe", 0.0) for f in folds]
   oos_sharpes = [f.get("test_sharpe", 0.0) for f in folds]
   ```

**Acceptance:** Functionally identical output, cleaner code.

---

### T8: Update INTERFACE_CONTRACTS.yaml Boundary Documentation

**Problem:** `INTERFACE_CONTRACTS.yaml:1069-1093` lists `evaluation/engine.py` as a reader of `results/backtest_*d_summary.json`. The engine has no file I/O — it receives all data via function parameters. This misleads downstream audits.

**File:** `docs/audit/data/INTERFACE_CONTRACTS.yaml` (or wherever the contracts file lives)

**Implementation:**
1. Remove `evaluation/engine.py` from the readers list for the `backtest_*d_summary.json` artifact.
2. Add a note that evaluation receives data via function parameters from entry points/scripts, not via direct file reads.

**Acceptance:** INTERFACE_CONTRACTS.yaml accurately reflects that evaluation/engine.py has no file reader path.

---

## Verification

- [ ] Run `pytest tests/ -k "evaluation or metrics or fragility or visualization"` — all pass
- [ ] Verify `compute_slice_metrics` with NaN returns produces correct IC when predictions are valid
- [ ] Verify changing `EVAL_PNL_CONCENTRATION_THRESHOLD` to 0.50 changes fragility determination
- [ ] Verify changing `EVAL_IC_DECAY_LOOKBACK` to 40 changes decay detection window
- [ ] Verify `plot_underwater` with NaN returns produces valid drawdown chart
- [ ] Verify regime length mismatch produces warning log and "Unknown" slice
- [ ] Verify `from evaluation import consecutive_loss_frequency` works
