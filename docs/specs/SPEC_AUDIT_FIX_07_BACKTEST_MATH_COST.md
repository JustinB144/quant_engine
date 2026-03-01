# SPEC_AUDIT_FIX_07: Backtest Math Consistency & Cost Calibration Fixes

**Priority:** CRITICAL — P0/P1 math errors can materially overstate backtest results.
**Scope:** `backtest/` — `engine.py`, `cost_calibrator.py`, `execution.py`, `optimal_execution.py`
**Estimated effort:** 4–5 hours
**Depends on:** Nothing
**Blocks:** SPEC_08 (validation fixes build on corrected math)

---

## Context

The backtest engine has fundamental math inconsistencies: portfolio returns mix two different return models (per-trade full-capital vs position-weighted daily), cost calibration ingests impact-only costs but the calibrator assumes total slippage, TCA compares incompatible quantities, execution urgency pullback doesn't recompute derived metrics, and the edge-cost gate uses pre-adjustment position sizes.

---

## Tasks

### T1: Fix Portfolio Return Model Inconsistency

**Problem:** `engine.py:1889-1903` — In simple mode, trade returns are treated as full-capital returns (`t.net_return`), but the equity drawdown curve at `:1943` is computed from position-weighted daily returns. This mixes two return models, potentially overstating Sharpe while computing drawdown on a different basis.

**File:** `backtest/engine.py`

**Implementation:**
1. Standardize on **position-weighted returns** throughout:
   ```python
   # Weighted return for each trade:
   weighted_return = t.net_return * t.position_size
   ```
2. Use the same weighted returns for both metric computation (Sharpe, total return, etc.) AND equity curve construction.
3. In `_compute_metrics()`, ensure `returns` array and `_build_daily_equity()` use the same return series.
4. Add a comment documenting the return convention: "All returns are portfolio-weighted: trade_return × position_fraction."
5. If per-trade unit returns are needed for diagnostics, compute them separately and label clearly.

**Acceptance:** Sharpe ratio and max drawdown are computed from the same return series. No mixed return models.

---

### T2: Fix Cost Calibration Input Mismatch

**Problem:** `engine.py:487` records `impact_bps` (impact-only) as `realized_cost_bps`, but `cost_calibrator.py:175,238` expects total slippage (spread + impact) and subtracts half-spread from the input, biasing calibrated coefficients downward.

**Files:** `backtest/engine.py`, `backtest/cost_calibrator.py`

**Implementation:**
1. In `engine.py:487`, pass total slippage to the calibrator:
   ```python
   total_slippage = fill.spread_bps + fill.impact_bps
   self._cost_calibrator.record_trade(
       ...,
       realized_cost_bps=float(total_slippage),
       ...
   )
   ```
2. Verify `cost_calibrator.py:238` decomposition logic is now consistent with the input.
3. Update docstrings in both files to explicitly document: "realized_cost_bps includes both spread and market impact."
4. Add assertion in `record_trade()`: `assert realized_cost_bps >= 0 or allow_negative_cost, "Expected non-negative total slippage"`.

**Acceptance:** Cost calibrator receives total slippage. Decomposition into spread + impact components produces sensible values.

---

### T3: Fix TCA Predicted vs Realized Quantity Mismatch

**Problem:** `engine.py:2294-2315` compares `entry_impact_bps` (impact-only prediction) against total realized slippage (spread + impact from fill-vs-reference), making correlation/RMSE metrics biased by construction.

**File:** `backtest/engine.py`

**Implementation:**
1. Compare like-with-like. Choose one of:
   - **Option A:** Compare total predicted cost (spread + impact) vs total realized slippage.
   - **Option B:** Compare impact-only predicted vs impact-only realized (requires separating spread from realized slippage).
2. Option A is simpler:
   ```python
   entry_predicted = t.entry_spread_bps + t.entry_impact_bps  # total predicted
   entry_realized = (t.entry_price - t.entry_reference_price) / t.entry_reference_price * 10_000  # total realized
   ```
3. Apply the same fix to exit-side TCA.
4. Update TCA report labels to clarify what's being compared.

**Acceptance:** TCA correlation/RMSE reflect comparison of same-domain quantities. Labels clearly state "total cost" or "impact only."

---

### T4: Recompute Participation/Impact After Urgency Pullback

**Problem:** `execution.py:658,759,788` — When urgency pullback reduces fill quantity, `participation_rate` and `impact_bps` are NOT recomputed with the reduced size. Downstream calibration and diagnostics use stale values.

**File:** `backtest/execution.py`

**Implementation:**
1. After urgency-based fill reduction (line 763), recompute:
   ```python
   if reduction_factor < 1.0:
       fill_notional = desired * fill_ratio
       participation = fill_notional / max(daily_dollar_volume, 1e-9)
       impact_bps = self._compute_impact(participation, volatility, ...)
   ```
2. Use the recomputed values in the returned `ExecutionFill` object.
3. Add a field `urgency_reduced: bool` to `ExecutionFill` for diagnostics.

**Acceptance:** When urgency reduces fill by 50%, participation and impact reflect the reduced size, not the original.

---

### T5: Fix Negative Spread Inference in Cost Calibration

**Problem:** `execution.py:901,924` — `calibrate_cost_model()` can produce negative spread estimates when slippage data includes negative values. No floor check on the calibrated median.

**File:** `backtest/execution.py`

**Implementation:**
1. After computing `est_spread` at line 901, add floor:
   ```python
   est_spread = max(0.0, est_spread)
   ```
2. After computing calibrated median at line 924, add floor:
   ```python
   calibrated_spread = max(0.0, float(np.median(spreads_bps)))
   ```
3. Log a warning when negative spread would have been inferred: `"Negative spread estimate clipped to 0: original={est_spread:.2f} bps"`.

**Acceptance:** `calibrate_cost_model()` never returns negative spread. Warning logged when clipping occurs.

---

### T6: Fix Edge-Cost Gate to Use Adjusted Position Size

**Problem:** `engine.py:1052,1106,1631,1739` — The edge-cost profitability gate evaluates whether a trade is worth its costs using `position_size_pct` (the base size), but the actual executed size may be smaller after uncertainty/risk adjustments.

**File:** `backtest/engine.py`

**Implementation:**
1. Compute the adjusted size BEFORE the cost gate:
   ```python
   adjusted_size = self.position_size_pct
   if shock is not None:
       adjusted_size *= size_mult
   # Apply other adjustments (uncertainty, regime, etc.)
   ```
2. Pass `adjusted_size` to the cost gate calculation instead of `self.position_size_pct`.
3. This ensures the gate evaluates the ACTUAL trade that will be executed, not a hypothetical larger one.

**Acceptance:** A trade with `position_size_pct=5%` but `size_mult=0.3` (so adjusted=1.5%) passes the cost gate if 1.5% is profitable after costs, even if 5% would not be.

---

### T7: Fix Package Import Contract

**Problem:** `config.py:29` uses `from config_structured import ...` (absolute import) which fails under package import (`quant_engine.config`). This blocks test execution.

**File:** `config.py` (root level)

**Implementation:**
1. Change to relative import or conditional import:
   ```python
   try:
       from .config_structured import get_config as _get_config
   except ImportError:
       from config_structured import get_config as _get_config
   ```
2. Or add `quant_engine` to `sys.path` in the test configuration.
3. Verify `pytest tests/` can collect and run backtest tests.

**Acceptance:** `python -c "from quant_engine.config import validate_config"` succeeds. Test collection does not fail with ModuleNotFoundError.

---

## Verification

- [ ] Run `pytest tests/ -k "backtest or cost_calibrat"` — all pass
- [ ] Verify portfolio Sharpe and drawdown use same return series
- [ ] Verify cost calibrator receives total slippage (spread + impact)
- [ ] Verify TCA compares like-with-like quantities
- [ ] Verify urgency pullback updates participation and impact
- [ ] Verify no negative spread values in calibration output
