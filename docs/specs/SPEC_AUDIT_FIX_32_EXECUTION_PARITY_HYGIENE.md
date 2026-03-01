# SPEC_AUDIT_FIX_32: Execution Parity, Structural Robustness & Code Hygiene

**Priority:** MEDIUM — P2/P3 issues that improve execution fidelity, reduce maintenance risk, and close testing gaps.
**Scope:** `autopilot/paper_trader.py`, `backtest/engine.py`, `backtest/execution.py`, `backtest/cost_calibrator.py`, `backtest/survivorship_comparison.py`, `risk/position_sizer.py`, `risk/factor_portfolio.py`, `risk/stop_loss.py`, `risk/attribution.py`, `risk/factor_monitor.py`, `validation/preconditions.py`, `config_data/universe.yaml`
**Estimated effort:** 5–6 hours
**Depends on:** SPEC_30 (contract fixes), SPEC_31 (math fixes)
**Blocks:** Nothing
**Audit sources:** Audit 1 F-11, F-14–F-16, F-19–F-22, F-24–F-55; Audit 2 F-05

---

## Context

After addressing the critical contract breaks (SPEC_30) and math errors (SPEC_31), the remaining findings fall into three categories: (1) execution parity drift between backtest and paper trader under shock conditions, (2) structural robustness issues that can cause silent degradation (fragile lookups, unbounded history growth, assertion gaps), and (3) code hygiene that reduces maintenance risk (duplication, dead code, misleading labels). This spec groups them for efficient remediation.

---

## Tasks

### T1: Fix Backtest vs Paper Trader Shock Policy Parity [P2 — Audit 2 F-05]

**Problem:** The backtest engine applies shock-mode policy overrides (confidence thresholds, participation limits, spread multipliers via `event_spread_multiplier`) when entering/exiting during stress/transition regimes. The paper trader in `autopilot/paper_trader.py` uses `ExecutionModel.simulate()` but does NOT pass `event_spread_multiplier` from shock policy, does not enforce shock-mode confidence minimums, and does not apply shock-mode participation overrides.

**Evidence (verified against source):**
- `backtest/engine.py` extracts `_shock_spread_mult` from `ShockModePolicy` and passes it to `simulate(event_spread_multiplier=...)`
- `autopilot/paper_trader.py:902` (exit) and `:1125` (entry) call `simulate()` without `event_spread_multiplier`
- Paper trader has no `ShockModePolicy` computation at all

**Files:** `autopilot/paper_trader.py`

**Implementation:**

1. Add shock vector computation to the paper trader's execution path. Import `compute_shock_vectors` and `ShockModePolicy`:
   ```python
   from regime.shock_vector import compute_shock_vectors, ShockVector
   from backtest.execution import ShockModePolicy
   ```

2. Before entry execution, compute the shock policy for the current market state:
   ```python
   shock_vector = self._get_current_shock_vector(ticker, market_data)
   shock_policy = ShockModePolicy.from_shock_vector(shock_vector) if shock_vector else None

   shock_spread_mult = (
       shock_policy.spread_multiplier
       if shock_policy is not None and shock_policy.is_active else 1.0
   )
   ```

3. Pass the spread multiplier to `simulate()`:
   ```python
   entry_fill = self._execution_model.simulate(
       side="buy",
       ...,
       event_spread_multiplier=shock_spread_mult,
       ...
   )
   ```

4. Apply shock confidence minimum (skip entry if confidence < policy minimum):
   ```python
   if shock_policy and shock_policy.is_active:
       if confidence < shock_policy.min_confidence:
           logger.info("Skipping entry for %s: confidence %.2f < shock minimum %.2f",
                       ticker, confidence, shock_policy.min_confidence)
           continue
   ```

5. Apply the same logic to exit paths at line 902.

6. Add a parity test that runs the same trade through both backtest and paper trader under shock conditions and verifies costs are within 5% of each other.

**Acceptance:** Paper trader applies the same shock-mode spread multipliers, confidence thresholds, and participation limits as the backtest engine. Under a stress regime, paper trading costs increase equivalently to backtest costs.

---

### T2: Fix ShockVector Lookup Fragility in Engine [P2 — Audit 1 F-11]

**Problem:** `backtest/engine.py:408-409,541-542` uses `ohlcv.attrs.get("ticker", ...)` to look up shock vectors. If `ohlcv.attrs` is not set (common for plain DataFrames), the lookup key is an empty string and shock vectors are never found. The main signal loop uses `permno` correctly, but execution-level structural cost conditioning is fragile.

**File:** `backtest/engine.py`

**Implementation:**

1. Use the `permno` parameter already available in the function scope instead of relying on `ohlcv.attrs`:
   ```python
   # Was: ticker_id = ohlcv.attrs.get("ticker", ohlcv.attrs.get("permno", ""))
   # Now: Use the permno/ticker already resolved in the calling context
   ticker_id = permno if permno else ticker
   shock = self._shock_vectors.get(ticker_id) if self._shock_vectors else None
   ```

2. Add a debug log when shock vector lookup fails unexpectedly:
   ```python
   if self._shock_vectors and shock is None:
       logger.debug("No shock vector found for %s; structural cost multipliers not applied", ticker_id)
   ```

**Acceptance:** Shock vectors are found for all tickers that have them, regardless of whether `ohlcv.attrs` is set. No silent degradation.

---

### T3: Fix Kelly Division-by-Zero Warning [P2 — Audit 1 F-14]

**Problem:** `risk/position_sizer.py:710-712` computes `b = abs(avg_win / avg_loss)`, which is 0 when `avg_win=0`. Then `(p*b - q)/b` divides by zero, producing `-inf`. The result is caught by the `kelly <= 0` check, but generates a runtime warning that pollutes logs.

**File:** `risk/position_sizer.py`

**Implementation:**

1. Add an early return when `avg_win` is 0 (no winning trades → Kelly = 0):
   ```python
   if avg_win <= 0 or avg_loss >= 0:
       return 0.0  # No edge to exploit
   ```

2. This eliminates the runtime warning without changing the behavior (the downstream check already handles it).

**Acceptance:** No `RuntimeWarning: divide by zero` in logs when `avg_win=0`. Kelly returns 0 cleanly.

---

### T4: Fix Bayesian Kelly Counter Overwrite [P2 — Audit 1 F-15]

**Problem:** `risk/position_sizer.py:918-919` replaces global Bayesian counters (`_bayesian_wins`, `_bayesian_losses`) instead of accumulating. Per-regime counters use `+=`. If called with partial trade batches, the global posterior reflects only the latest batch.

**File:** `risk/position_sizer.py`

**Implementation:**

1. Change to accumulation, matching the per-regime pattern:
   ```python
   # Was:
   self._bayesian_wins = int((returns > 0).sum())
   self._bayesian_losses = int((returns <= 0).sum())

   # Now:
   self._bayesian_wins += int((returns > 0).sum())
   self._bayesian_losses += int((returns <= 0).sum())
   ```

2. Verify that the initial values (from `KELLY_BAYESIAN_ALPHA` and `KELLY_BAYESIAN_BETA` priors) are set in `__init__` and accumulated from there.

**Acceptance:** Calling `update_kelly_bayesian()` three times with 10 trades each produces the same posterior as calling it once with 30 trades.

---

### T5: Fix Cost Calibration Stress Multiplier Decomposition [P2 — Audit 1 F-16]

**Problem:** `backtest/cost_calibrator.py:238` decomposes realized cost using `net_impact = max(0.0, cost - 0.5 * self._spread_bps)` with the BASE spread. During stressed periods, actual spreads include structural/event multipliers. The decomposition over-attributes to impact, inflating calibrated coefficients.

**File:** `backtest/cost_calibrator.py`

**Implementation:**

1. Pass the effective spread (including multipliers) to the decomposition:
   ```python
   def record_trade(self, ..., effective_spread_bps: Optional[float] = None):
       """Record a trade for calibration.

       Args:
           effective_spread_bps: The actual spread including all multipliers.
               If None, uses base spread (backward-compatible).
       """
       spread = effective_spread_bps if effective_spread_bps is not None else 0.5 * self._spread_bps
       net_impact = max(0.0, realized_cost_bps - spread)
   ```

2. Update `backtest/engine.py` callers to pass `fill.spread_bps` as the effective spread.

**Acceptance:** Calibrated impact coefficients are stable across stressed and normal periods. The base spread is not used as a proxy for the effective spread during stressed conditions.

---

### T6: Fix Timezone-Naive Comparison in Cost Calibrator [P2 — Audit 1 F-20]

**Problem:** `backtest/cost_calibrator.py:456-461` subtracts a potentially timezone-naive `datetime.fromisoformat()` from `datetime.now(timezone.utc)`. In Python 3.12+, this raises `TypeError`.

**File:** `backtest/cost_calibrator.py`

**Implementation:**

1. Ensure stored timestamps include timezone info:
   ```python
   from datetime import datetime, timezone

   # When storing:
   timestamp = datetime.now(timezone.utc).isoformat()

   # When loading:
   stored_dt = datetime.fromisoformat(stored_timestamp)
   if stored_dt.tzinfo is None:
       stored_dt = stored_dt.replace(tzinfo=timezone.utc)

   elapsed = datetime.now(timezone.utc) - stored_dt
   ```

**Acceptance:** Feedback recalibration works in Python 3.12+ with no `TypeError`.

---

### T7: Extract Duplicated Edge-Cost Gate Logic [P2 — Audit 1 F-24]

**Problem:** `backtest/engine.py:1045-1101` and `:1625-1680` contain ~110 lines of identical edge-cost gate logic duplicated between simple and risk-managed signal processing. Changes to one copy must be manually mirrored.

**File:** `backtest/engine.py`

**Implementation:**

1. Extract the common logic into a private method:
   ```python
   def _evaluate_edge_cost_gate(
       self,
       predicted_return: float,
       position_size_pct: float,
       shock_policy: Optional[ShockModePolicy],
       uncertainty: float,
       regime: int,
       ticker: str,
       ...
   ) -> tuple[bool, str]:
       """Evaluate whether a trade's predicted edge exceeds expected costs.

       Returns:
           (should_trade, reason)
       """
       # ... extracted gate logic ...
   ```

2. Call from both `_process_signals()` and `_process_signals_risk_managed()`:
   ```python
   passes_gate, gate_reason = self._evaluate_edge_cost_gate(...)
   if not passes_gate:
       skipped_edge.append(gate_reason)
       continue
   ```

3. Also extract the shock mode policy check and logging (~40 lines) into a helper.

**Acceptance:** Edge-cost gate logic exists in exactly one place. Both signal processing paths produce identical gate decisions for identical inputs.

---

### T8: Fix ADV Tracker Double-Counting on Residual Exit [P2 — Audit 1 F-25]

**Problem:** `backtest/engine.py` updates the ADV tracker during both entry and exit simulations. For residual positions that exit across multiple bars, volume is counted once per bar per exit attempt, inflating ADV estimates.

**File:** `backtest/engine.py`

**Implementation:**

1. Track which (ticker, date) combinations have already been counted for ADV:
   ```python
   # In __init__:
   self._adv_counted: set[tuple[str, str]] = set()

   # Before ADV update:
   date_key = (ticker, str(bar_date))
   if date_key not in self._adv_counted:
       self._adv_tracker.update(ticker, volume, bar_date)
       self._adv_counted.add(date_key)
   ```

2. Clear the set at the start of each new trading day (not per-bar).

**Acceptance:** ADV estimates are not inflated by multi-bar residual unwinding. Volume is counted at most once per ticker per day.

---

### T9: Fix Duplicate Ticker in universe.yaml [P3 — Audit 1 F-26]

**Problem:** `config_data/universe.yaml:49,73` lists NKE twice in the `consumer` sector.

**File:** `config_data/universe.yaml`

**Implementation:**

1. Remove the duplicate NKE entry (keep one, remove the other).
2. Add a validation check in `risk/universe_config.py` that warns on duplicate tickers.

**Acceptance:** `NKE` appears exactly once in universe.yaml. Sector weight calculations are not affected by duplicates.

---

### T10: Fix Stop Loss Long-Only Assumption [P3 — Audit 1 F-55]

**Problem:** `risk/stop_loss.py:138` computes `unrealized = (current_price - entry_price) / entry_price`, which is wrong for short positions (profit when price falls).

**File:** `risk/stop_loss.py`

**Implementation:**

1. Accept a `side` parameter:
   ```python
   def check_stops(self, entry_price, current_price, side="long", ...):
       if side == "short":
           unrealized = (entry_price - current_price) / entry_price
       else:
           unrealized = (current_price - entry_price) / entry_price
   ```

2. If the system is currently long-only, add the parameter with `side="long"` default and a TODO comment for future short support.

**Acceptance:** For short positions, unrealized PnL is positive when price drops. Stops trigger correctly for both long and short positions.

---

### T11: Fix Survivorship Comparison Absolute Thresholds [P2 — Audit 1 F-29]

**Problem:** `backtest/survivorship_comparison.py:159-163` uses absolute counts (>10 = HIGH, >3 = MEDIUM) regardless of universe size. Dropping 11 from 5000 is not HIGH risk.

**File:** `backtest/survivorship_comparison.py`

**Implementation:**

1. Use percentage-based thresholds:
   ```python
   pct_dropped = dropped_count / max(universe_size, 1)
   if pct_dropped > 0.05:  # >5% of universe
       bias_risk = "HIGH"
   elif pct_dropped > 0.01:  # >1%
       bias_risk = "MEDIUM"
   else:
       bias_risk = "LOW"
   ```

**Acceptance:** Dropping 11 tickers from 5000 (0.22%) is rated LOW, not HIGH.

---

### T12: Fix BH FDR Threshold Floor [P2 — Audit 1 F-28]

**Problem:** `backtest/validation.py:497` uses `max(fdr_threshold, 0.001)` as a floor when BH says nothing is significant. This lets strategies pass at p < 0.001 even when strict FDR control rejects everything.

**File:** `backtest/validation.py`

**Implementation:**

1. When BH rejects everything, respect that decision:
   ```python
   if fdr_threshold <= 0:
       # BH says nothing is significant — do not apply a floor
       passes_fdr = False
   else:
       passes_fdr = any(p < fdr_threshold for p in p_values)
   ```

2. Add a config flag `VALIDATION_FDR_FLOOR_ENABLED = False` with a STATUS annotation marking the old behavior as deprecated.

**Acceptance:** When BH procedure determines no strategy is significant, the FDR test correctly reports failure. No artificial floor bypasses the correction.

---

### T13: Add Preconditions Warn-Only Mode [P3 — Audit 1 F-54]

**Problem:** When `TRUTH_LAYER_STRICT_PRECONDITIONS=False`, validation is completely bypassed with no logging.

**File:** `validation/preconditions.py`

**Implementation:**

1. When strict mode is off, still run validation but log warnings instead of raising:
   ```python
   if not TRUTH_LAYER_STRICT_PRECONDITIONS:
       try:
           _validate_preconditions()
       except RuntimeError as e:
           logger.warning("Precondition violation (non-strict mode): %s", e)
       return  # Don't raise
   ```

**Acceptance:** With strict=False, precondition violations are logged as warnings, not silently swallowed.

---

### T14: Prune Unbounded History Lists [P3 — Audit 1 F-37, F-52]

**Problem:** `risk/position_sizer.py:677` (`_turnover_history`) and `backtest/cost_calibrator.py:158-189` (trade/feedback history) grow without bound.

**Files:** `risk/position_sizer.py`, `backtest/cost_calibrator.py`

**Implementation:**

1. Add maximum length enforcement:
   ```python
   # position_sizer.py
   MAX_TURNOVER_HISTORY = 504  # ~2 years of trading days
   self._turnover_history.append(turnover)
   if len(self._turnover_history) > MAX_TURNOVER_HISTORY:
       self._turnover_history = self._turnover_history[-MAX_TURNOVER_HISTORY:]

   # cost_calibrator.py
   MAX_CALIBRATION_HISTORY = 5000
   self._trade_history.append(trade)
   if len(self._trade_history) > MAX_CALIBRATION_HISTORY:
       self._trade_history = self._trade_history[-MAX_CALIBRATION_HISTORY:]
   ```

**Acceptance:** History lists are bounded. Memory usage does not grow linearly with backtest duration.

---

### T15: Fix Factor Portfolio Numerical Stability [P2 — Audit 1 F-19]

**Problem:** `risk/factor_portfolio.py:111-115` uses `np.linalg.inv(X.T @ X)` which is numerically unstable for ill-conditioned factor matrices, despite a `1e-8 * I` ridge penalty.

**File:** `risk/factor_portfolio.py`

**Implementation:**

1. Replace with `np.linalg.lstsq`:
   ```python
   # Was:
   # beta = np.linalg.inv(X.T @ X + 1e-8 * np.eye(X.shape[1])) @ X.T @ y

   # Now:
   beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
   ```

2. This is numerically stable and handles rank-deficient matrices correctly.

**Acceptance:** Factor exposure estimates are stable for correlated factor matrices. No `LinAlgError` on ill-conditioned inputs.

---

## Verification

- [ ] Run `pytest tests/ -k "backtest or risk or autopilot"` — all pass
- [ ] Paper trader applies shock spread multipliers (compare with backtest for same trade)
- [ ] Edge-cost gate logic has no duplication (single method, called from both paths)
- [ ] No `RuntimeWarning: divide by zero` from Kelly with avg_win=0
- [ ] `universe.yaml` has no duplicate tickers
- [ ] Precondition violations logged as warnings when strict=False

---

*Generated from cross-audit reconciliation — 2026-02-28*
