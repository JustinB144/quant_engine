# SPEC_AUDIT_FIX_08: Backtest Validation, Stop-Loss & Execution Edge Fixes

**Priority:** HIGH
**Scope:** `backtest/` — `engine.py`, `advanced_validation.py`, `survivorship_comparison.py`, `cost_calibrator.py`
**Estimated effort:** 3 hours
**Depends on:** SPEC_07 (math fixes land first)
**Blocks:** Nothing

---

## Context

The backtest subsystem has P1-P2 issues in validation logic, stop-loss evaluation, capacity estimation, and exception handling. Stop-loss uses full-bar OHLC (intrabar look-ahead), survivorship check ignores its predictions parameter, capacity uses a single trade for estimation, null/stress result fields are never populated, and exception swallowing hides breakage.

---

## Tasks

### T1: Fix Stop-Loss Intrabar Look-Ahead

**Problem:** `engine.py:1303-1332` evaluates stop-loss using the full bar's High/Low/Close simultaneously. In reality, the high might occur after the stop price was breached, but the simulation treats all OHLC values as available at decision time.

**File:** `backtest/engine.py`

**Implementation:**
1. Add a configurable `stop_loss_bar_model` parameter (default `"conservative"`):
   - `"conservative"`: Assume worst-case bar ordering. For long positions, check if Low < stop_price (stop triggered at stop_price, not close). For short positions, check if High > stop_price.
   - `"close_only"`: Only evaluate stop against Close (simplest, least realistic).
   - `"ohlc_sequential"`: Assume Open→High→Low→Close order (approximation).
2. Replace current logic with the selected model.
3. Document the assumption in the backtest result metadata.

**Acceptance:** Stop-loss trigger price and fill price are consistent with the selected bar model. Conservative mode uses Low (long) or High (short), not Close.

---

### T2: Fix quick_survivorship_check Ignoring Predictions

**Problem:** `survivorship_comparison.py:125,143` accepts a `predictions` parameter but never uses it. The function's docstring claims prediction coverage checks.

**File:** `backtest/survivorship_comparison.py`

**Implementation:**
1. Either implement the prediction coverage check or remove the parameter:
   - **Option A (implement):** Check that predictions cover the same universe as price data. Flag tickers present in prices but missing from predictions (potential survivorship in prediction model).
   - **Option B (remove):** Remove the `predictions` parameter and update the docstring to accurately describe what the function does.
2. Choose Option A — it adds value:
   ```python
   if predictions is not None:
       pred_tickers = set(predictions.columns) if hasattr(predictions, 'columns') else set()
       price_tickers = set(price_data_full.columns) if hasattr(price_data_full, 'columns') else set()
       missing_in_pred = price_tickers - pred_tickers
       result["prediction_coverage_gap"] = list(missing_in_pred)
   ```

**Acceptance:** Passing predictions to `quick_survivorship_check()` produces a `prediction_coverage_gap` field in the result.

---

### T3: Fix Capacity Estimate Using First Trade Only

**Problem:** `advanced_validation.py:426` uses `trades[0].position_size` as the representative trade size for capacity estimation, which is fragile when sizes vary.

**File:** `backtest/advanced_validation.py`

**Implementation:**
1. Replace first-trade size with median trade size:
   ```python
   sizes = [t.position_size for t in trades if hasattr(t, 'position_size') and t.position_size > 0]
   avg_size = float(np.median(sizes)) if sizes else 0.05
   ```
2. Also add max and p75 sizes to the capacity report for robustness.

**Acceptance:** Capacity estimate reflects the distribution of trade sizes, not just the first trade.

---

### T4: Add Missing "_all" Aggregate to Cost Surprise

**Problem:** `cost_calibrator.py:396,409` documents an `"_all"` key in the output but never computes it.

**File:** `backtest/cost_calibrator.py`

**Implementation:**
1. After the segment loop, add:
   ```python
   all_surprises = [s for bucket in segment_buckets.values() for s in bucket]
   if all_surprises:
       results["_all"] = {
           "mean_surprise": float(np.mean(all_surprises)),
           "std_surprise": float(np.std(all_surprises)),
           "count": len(all_surprises),
       }
   ```

**Acceptance:** Output includes `"_all"` key with aggregate statistics across all segments.

---

### T5: Populate Null/Stress Result Fields

**Problem:** `engine.py:141,2019` defines `null_baselines` and `cost_stress_result` fields on `BacktestResult` but they are never populated during the engine run path.

**File:** `backtest/engine.py`

**Implementation:**
1. After computing main backtest metrics, compute null baselines:
   ```python
   from backtest.null_models import compute_null_baselines
   result.null_baselines = compute_null_baselines(trades, returns, n_trials=100)
   ```
2. Compute cost stress results:
   ```python
   from backtest.cost_calibrator import compute_cost_stress
   result.cost_stress_result = compute_cost_stress(trades, cost_multipliers=[1.0, 1.5, 2.0])
   ```
3. If these functions don't exist yet, create stubs that return structured results and mark with TODO for full implementation.
4. Wrap in try/except with logging (these are additive, should not break the main path).

**Acceptance:** `BacktestResult.null_baselines` and `cost_stress_result` are populated (even if with basic implementations). `summarize_vs_null()` returns meaningful data.

---

### T6: Fix Advanced Validation Sentinels

**Problem:** `advanced_validation.py:143` returns `deflated_sharpe=-999` as a sentinel value, and line 266 uses `0` as a sentinel for PBO edge cases.

**File:** `backtest/advanced_validation.py`

**Implementation:**
1. Replace `deflated_sharpe=-999.0` with `deflated_sharpe=float('nan')` and add a `deflated_sharpe_valid: bool = False` field.
2. Replace PBO sentinel `0` with `float('nan')` and add `pbo_valid: bool`.
3. Downstream consumers should check the `_valid` flag before using the value.

**Acceptance:** No magic number sentinels. Invalid values are `NaN` with explicit validity flags.

---

### T7: Fix Exception Swallowing in Shock Vector Precompute

**Problem:** `engine.py:883` catches all exceptions from shock vector computation with a bare `except Exception`, silently downgrading to no-structural-state execution.

**File:** `backtest/engine.py`

**Implementation:**
1. Narrow the exception catch to expected failures:
   ```python
   except (ValueError, RuntimeError, KeyError) as exc:
       logger.warning("Shock vector computation failed for %s: %s", ticker, exc)
       shock = None
   ```
2. Let unexpected exceptions (TypeError, AttributeError, etc.) propagate so they are visible.
3. Add the failed ticker to a `_shock_failures` list for post-run diagnostics.

**Acceptance:** Expected failures are handled gracefully. Unexpected exceptions propagate. Failure list is available in backtest results.

---

## Verification

- [ ] Run `pytest tests/ -k "backtest or validation or survivorship"` — all pass
- [ ] Verify stop-loss uses conservative bar model by default
- [ ] Verify capacity estimate uses median trade size
- [ ] Verify null_baselines and cost_stress_result are populated in results
- [ ] Verify no -999 sentinel values in validation output
