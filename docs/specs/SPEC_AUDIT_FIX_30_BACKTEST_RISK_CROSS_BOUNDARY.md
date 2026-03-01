# SPEC_AUDIT_FIX_30: Backtest/Risk Cross-Boundary Contract & Critical Math Fixes

**Priority:** CRITICAL — P0/P1 contract breaks and dimensional errors that cause runtime crashes and silently invalidate risk controls.
**Scope:** `autopilot/engine.py`, `risk/position_sizer.py`, `api/orchestrator.py`, `api/services/data_helpers.py`, `risk/cost_budget.py`, `risk/portfolio_risk.py`, `risk/metrics.py`, `run_backtest.py`
**Estimated effort:** 6–8 hours
**Depends on:** Nothing (these are independent contract/math fixes)
**Blocks:** SPEC_31 (parity and execution fixes assume corrected contracts), SPEC_07 T2 (cost calibration depends on correct cost budget)
**Audit sources:** Audit Report 05 (both auditors), INTERFACE_CONTRACTS.yaml boundaries `autopilot_to_backtest_31`, `api_to_backtest_41`, `evaluation_to_models_backtest_11`

---

## Context

Two independent audits of the backtesting_risk subsystem identified critical cross-boundary failures. The autopilot module calls `backtest/validation.py` with a stale API (missing required parameter, wrong field names), the turnover-budget governor compares fractions against a percentage constant, the API equity-curve endpoint crashes on orchestrator-produced trade CSVs, the declared summary.json schema is not met by writers, the cost budget module has a dimensional error making market impact non-functional, gross exposure uses algebraic sum instead of absolute sum, and the Calmar ratio sign is inverted for losing strategies.

These findings were independently confirmed by source inspection. **Audit 1 F-01 (ShockModePolicy spread_multiplier not applied) was NOT confirmed** — source inspection shows `spread_multiplier` IS extracted from the policy and passed as `event_spread_multiplier` to `ExecutionModel.simulate()` in both `_simulate_entry()` and `_simulate_exit()`. That finding is excluded from this spec.

---

## Tasks

### T1: Fix Autopilot → Backtest Validation Contract Break [P0 CRITICAL]

**Problem:** `autopilot/engine.py` calls `run_statistical_tests()`, `combinatorial_purged_cv()`, and `superior_predictive_ability()` from `backtest/validation.py` using stale signatures and nonexistent field names. At runtime, this produces `TypeError` (missing required `trade_returns` parameter) and `AttributeError` (accessing `.overall_pass`, `.is_significant`, `.rejects_null` which do not exist — the actual field is `.passes` on all three result types).

**Evidence (verified against source):**
- `autopilot/engine.py:881` calls `run_statistical_tests(predictions, actuals, entry_threshold=...)` — missing required parameter `trade_returns: np.ndarray`
- `autopilot/engine.py:886` accesses `stat_tests_result.overall_pass` — field is `StatisticalTests.passes` (`backtest/validation.py:141`)
- `autopilot/engine.py:902` accesses `cpcv_result.is_significant` — field is `CPCVResult.passes` (`backtest/validation.py:167`)
- `autopilot/engine.py:915` accesses `spa_result.rejects_null` — field is `SPAResult.passes` (`backtest/validation.py:185`)
- Boundary `autopilot_to_backtest_31` in INTERFACE_CONTRACTS.yaml is rated HIGH risk with explicit audit note to verify lazy import contracts.

**Files:** `autopilot/engine.py`

**Implementation:**

1. Fix the `run_statistical_tests` call to provide the required `trade_returns` parameter:
   ```python
   # Compute trade_returns from the backtest result trades
   trade_returns = np.array([t.net_return for t in result.trades]) if result.trades else np.array([])

   stat_tests_result = run_statistical_tests(
       predictions=pred_series,
       actuals=actual_series,
       trade_returns=trade_returns,
       entry_threshold=c.entry_threshold,
       holding_days=c.holding_days,
   )
   ```

2. Fix all three field name accesses:
   ```python
   # Was: stat_tests_result.overall_pass
   stat_tests_pass = bool(stat_tests_result.passes)

   # Was: cpcv_result.is_significant
   cpcv_pass = bool(cpcv_result.passes)

   # Was: spa_result.rejects_null
   spa_pass = bool(spa_result.passes)
   ```

3. Fix any additional stale field accesses around these call sites. Specifically check for:
   - `stat_tests_result.mean_test_corr` — verify this exists on `StatisticalTests` or replace with the correct field from `StatisticalTests.details`
   - Any other references to the old API shape in the ~70 lines surrounding these calls

4. Add a regression test in `tests/` that imports both `autopilot.engine.AutopilotEngine` and `backtest.validation.run_statistical_tests` and verifies the call signature matches:
   ```python
   import inspect
   sig = inspect.signature(run_statistical_tests)
   assert "trade_returns" in sig.parameters
   ```

**Acceptance:** `AutopilotEngine.run_cycle()` completes without `TypeError` or `AttributeError` when backtest validation is invoked. The `autopilot_to_backtest_31` boundary is compatible.

---

### T2: Fix Turnover-Budget Unit Mismatch [P1 HIGH]

**Problem:** `risk/position_sizer.py:642-654` computes annualized turnover as a fractional ratio (e.g., 12.6 for 1260% annualized turnover) but compares it against `MAX_ANNUALIZED_TURNOVER` which is `500` (config.py:392, documented as "500% annualized turnover warning threshold"). The comparison `position_turnover > remaining_budget` compares a fraction (e.g., 0.05) against 500 minus a fraction, meaning the constraint essentially never binds. The turnover-budget governor is materially weaker than its configured intent.

**Evidence (verified against source):**
- `risk/position_sizer.py:644`: `annualized_turnover = (total_turnover / days_elapsed) * 252` — produces a fraction
- `risk/position_sizer.py:654`: `remaining_budget = self._max_annualized_turnover - annualized_turnover` — subtracts fraction from 500
- `config.py:389`: `MAX_ANNUALIZED_TURNOVER = 500.0` documented as "500% annualized turnover warning threshold"

**File:** `risk/position_sizer.py`

**Implementation:**

1. Normalize the threshold to the same unit as the metric. The cleanest fix is to convert the config constant from percentage to fraction at the point of consumption:
   ```python
   # In PositionSizer.__init__ or wherever _max_annualized_turnover is set:
   self._max_annualized_turnover = MAX_ANNUALIZED_TURNOVER / 100.0  # Convert from % to fraction (500% → 5.0)
   ```

2. Add a comment at the conversion site documenting the unit convention:
   ```python
   # MAX_ANNUALIZED_TURNOVER is configured in percent (e.g., 500 = 500%).
   # Internal turnover tracking uses fractions (e.g., 5.0 = 500%).
   ```

3. Do NOT change config.py — the percentage convention there is documented and used in logging. Only convert at the consumption site.

4. Add a unit test that verifies the turnover budget actually constrains sizing:
   ```python
   def test_turnover_budget_constrains_sizing():
       sizer = PositionSizer(...)
       # Simulate enough history to exceed budget
       for _ in range(100):
           sizer._turnover_history.append(0.10)  # 10% daily turnover
       # Annualized = 0.10 * 252 = 25.2 (2520%)
       # With MAX_ANNUALIZED_TURNOVER=500% (5.0 as fraction), budget is exhausted
       result = sizer._apply_turnover_budget(0.05, "TEST", {...}, 1_000_000)
       assert result < 0.05  # Should be constrained
   ```

**Acceptance:** With `MAX_ANNUALIZED_TURNOVER=500` and annualized turnover exceeding 500%, new position sizing is constrained. Unit test proves the constraint binds.

---

### T3: Fix API Equity-Curve Crash on Orchestrator-Produced Trade CSV [P1 HIGH]

**Problem:** Two linked defects cause the `/api/backtests/latest/equity-curve` endpoint to crash with `AttributeError` when consuming trade CSVs produced by `api/orchestrator.py`:

1. `api/orchestrator.py:334-347` writes trade data without the `position_size` field.
2. `api/services/data_helpers.py:50-51` falls back to scalar `1.0` when `position_size` is missing, then calls `pd.to_numeric(...).fillna(...)` on it, which produces undefined behavior for scalar inputs.

**Evidence (verified against source):**
- `api/orchestrator.py` trade_data dict has 11 fields, no `position_size`
- `api/services/data_helpers.py:50`: `weights = valid["position_size"] if "position_size" in valid.columns else 1.0`
- `api/services/data_helpers.py:51`: `weights = pd.to_numeric(weights, errors="coerce").fillna(0.05).clip(...)` — when `weights=1.0`, `pd.to_numeric(1.0)` returns a numpy scalar, and `.fillna()` on a scalar raises `AttributeError`

**Files:** `api/orchestrator.py`, `api/services/data_helpers.py`

**Implementation:**

1. Fix `api/orchestrator.py` to include `position_size` in the trade CSV output:
   ```python
   trade_data.append({
       "permno": t.ticker,
       "entry_date": t.entry_date,
       "exit_date": t.exit_date,
       "predicted_return": t.predicted_return,
       "actual_return": t.actual_return,
       "net_return": t.net_return,
       "regime": REGIME_NAMES.get(t.regime, f"regime_{t.regime}"),
       "confidence": t.confidence,
       "holding_days": t.holding_days,
       "exit_reason": t.exit_reason,
       "position_size": t.position_size,  # ADD THIS
   })
   ```

2. Fix `api/services/data_helpers.py` to handle the fallback defensively regardless:
   ```python
   if "position_size" in valid.columns:
       weights = pd.to_numeric(valid["position_size"], errors="coerce").fillna(0.05).clip(lower=0.0, upper=1.0)
   else:
       # Uniform weighting when position_size is absent
       weights = pd.Series(0.05, index=valid.index)
   ```

3. Add a contract test that validates the orchestrator trade CSV schema includes all fields expected by data_helpers:
   ```python
   REQUIRED_TRADE_CSV_FIELDS = {"net_return", "entry_date", "exit_date", "position_size"}
   # Assert orchestrator output dict keys are a superset of required fields
   ```

**Acceptance:** `/api/backtests/latest/equity-curve` returns valid data for trade CSVs produced by both `run_backtest.py` and `api/orchestrator.py`. No `AttributeError` on scalar fallback.

---

### T4: Fix Summary.json Contract — Emit All Declared Fields [P1 HIGH]

**Problem:** `INTERFACE_CONTRACTS.yaml:1072-1089` declares a 17-field schema for `results/backtest_*d_summary.json`, but both writers (`run_backtest.py:427` and `api/orchestrator.py:349`) emit only 11 fields. The 6 missing fields are: `winning_trades`, `losing_trades`, `avg_win`, `avg_loss`, `total_return`, `avg_holding_days`.

**Evidence (verified against source):**
- Both writers emit: `horizon`, `total_trades`, `win_rate`, `avg_return`, `sharpe`, `sortino`, `max_drawdown`, `profit_factor`, `annualized_return`, `trades_per_year`, `regime_breakdown`
- Missing: `winning_trades`, `losing_trades`, `avg_win`, `avg_loss`, `total_return`, `avg_holding_days`
- All 6 missing fields are available on the `BacktestResult` dataclass

**Files:** `run_backtest.py`, `api/orchestrator.py`

**Implementation:**

1. Create a shared helper function to build the summary dict (eliminates the duplication between the two writers):
   ```python
   # In backtest/engine.py or a new backtest/schemas.py:
   def backtest_result_to_summary_dict(result: BacktestResult) -> dict:
       """Build the canonical summary.json dict from a BacktestResult.

       Schema contract: docs/audit/data/INTERFACE_CONTRACTS.yaml boundary
       evaluation_to_models_backtest_11, fields 1072-1089.
       """
       trades = result.trades or []
       winning = [t for t in trades if t.net_return > 0]
       losing = [t for t in trades if t.net_return <= 0]

       return {
           "horizon": result.holding_days,
           "total_trades": result.total_trades,
           "winning_trades": len(winning),
           "losing_trades": len(losing),
           "win_rate": result.win_rate,
           "avg_return": result.avg_return,
           "avg_win": float(np.mean([t.net_return for t in winning])) if winning else 0.0,
           "avg_loss": float(np.mean([t.net_return for t in losing])) if losing else 0.0,
           "total_return": result.annualized_return,  # Or compute cumulative total
           "annualized_return": result.annualized_return,
           "sharpe": result.sharpe_ratio,
           "sortino": result.sortino_ratio,
           "max_drawdown": result.max_drawdown,
           "profit_factor": result.profit_factor,
           "avg_holding_days": float(np.mean([t.holding_days for t in trades])) if trades else 0.0,
           "trades_per_year": result.trades_per_year,
           "regime_breakdown": result.regime_breakdown,
       }
   ```

2. Replace the inline dict construction in both `run_backtest.py` and `api/orchestrator.py` with a call to this function.

3. Add a schema assertion test that validates summary.json contains all 17 declared fields:
   ```python
   DECLARED_SUMMARY_FIELDS = {
       "horizon", "total_trades", "winning_trades", "losing_trades",
       "win_rate", "avg_return", "avg_win", "avg_loss", "total_return",
       "annualized_return", "sharpe", "sortino", "max_drawdown",
       "profit_factor", "avg_holding_days", "trades_per_year", "regime_breakdown",
   }

   def test_summary_schema_contract():
       result = backtest_result_to_summary_dict(mock_backtest_result)
       assert set(result.keys()) == DECLARED_SUMMARY_FIELDS
   ```

**Acceptance:** Both writers emit all 17 fields declared in INTERFACE_CONTRACTS.yaml. Schema assertion test passes. Boundary `evaluation_to_models_backtest_11` shared artifact contract is satisfied.

---

### T5: Fix Cost Budget Dimensional Inconsistency [P0 — Audit 1 F-02]

**Problem:** `risk/cost_budget.py:68` computes participation rate as `abs(trade_size_weight) / daily_dollar_volume`, dividing a dimensionless portfolio-weight fraction (e.g., 0.05) by a dollar volume (e.g., $25M). The result (~2×10⁻⁹) is not a valid participation rate. The correct formula requires `portfolio_value_usd`: `participation = (trade_size_weight × portfolio_value_usd) / daily_dollar_volume`.

**Evidence (verified against source):**
- `risk/cost_budget.py:68`: `participation = abs(trade_size_weight) / max(daily_volume, 1e-12)`
- `trade_size_weight` is documented as a portfolio fraction
- `daily_volume` is daily dollar volume

**File:** `risk/cost_budget.py`

**Implementation:**

1. Add `portfolio_value_usd` as a required parameter to `estimate_trade_cost_bps()`:
   ```python
   def estimate_trade_cost_bps(
       self,
       trade_size_weight: float,
       daily_volume: float,
       spread_bps: float = 3.0,
       impact_coeff_bps: float = 25.0,
       portfolio_value_usd: float = 1_000_000.0,  # Required for correct participation
   ) -> float:
   ```

2. Fix the participation rate calculation:
   ```python
   trade_dollar_value = abs(trade_size_weight) * portfolio_value_usd
   participation = trade_dollar_value / max(daily_volume, 1e-12)
   participation = min(participation, 1.0)  # Cap at 100%
   ```

3. Propagate `portfolio_value_usd` to `optimize_rebalance_cost()` and any other callers.

4. Update the return expression to use `trade_dollar_value / portfolio_value_usd` (i.e., `abs(trade_size_weight)`) as the final multiplier, keeping the cost in bps space.

5. Add a unit test verifying dimensional correctness:
   ```python
   def test_cost_budget_participation_rate():
       cb = CostBudget()
       # 5% portfolio weight, $1M portfolio, $25M daily volume
       cost = cb.estimate_trade_cost_bps(0.05, 25_000_000, portfolio_value_usd=1_000_000)
       # participation = $50K / $25M = 0.002 → impact = 25 * sqrt(0.002) ≈ 1.12 bps
       assert cost > 0.5  # Must be material, not ~0
   ```

**Acceptance:** Market impact in cost budget is non-trivially positive for realistic trade sizes. Cost budget optimization produces meaningfully different results for large vs small trades.

---

### T6: Fix Gross Exposure Constraint — Use Absolute Sum [P1 — Audit 1 F-07]

**Problem:** `risk/portfolio_risk.py:361` computes gross exposure as `sum(proposed_positions.values())`, which is algebraic (net) sum. For long-short portfolios (e.g., +50% long, -50% short), this reports 0% gross exposure, making the constraint meaningless.

**Evidence (verified against source):**
- `risk/portfolio_risk.py:361`: `gross = sum(proposed_positions.values())`
- Correct: `gross = sum(abs(v) for v in proposed_positions.values())`

**File:** `risk/portfolio_risk.py`

**Implementation:**

1. Change the gross exposure calculation:
   ```python
   gross = sum(abs(v) for v in proposed_positions.values())
   ```

2. Add a comment distinguishing gross from net:
   ```python
   # Gross exposure = sum of absolute position weights (long + |short|).
   # Net exposure = algebraic sum (long - |short|).
   gross = sum(abs(v) for v in proposed_positions.values())
   ```

3. Verify no downstream code relies on the old (incorrect) net-as-gross behavior. Check `check_portfolio_constraints()` return values and any callers.

4. Add a test with a long-short portfolio:
   ```python
   def test_gross_exposure_long_short():
       mgr = PortfolioRiskManager(...)
       positions = {"AAPL": 0.5, "MSFT": -0.3, "GOOGL": 0.4}
       result = mgr.check_portfolio_constraints(positions, ...)
       # Gross should be 1.2 (0.5 + 0.3 + 0.4), not 0.6 (0.5 - 0.3 + 0.4)
       assert result["gross_exposure"] == pytest.approx(1.2)
   ```

**Acceptance:** Gross exposure constraint correctly constrains long-short portfolios. A 50% long / 50% short portfolio reports 100% gross exposure, not 0%.

---

### T7: Fix Calmar Ratio Sign Inversion [P1 — Audit 1 F-03]

**Problem:** `risk/metrics.py:136` uses `calmar = abs(ann_return / dd_metrics["max_drawdown"])`. Since `max_drawdown` is negative, a losing strategy with negative `ann_return` produces `abs(negative / negative) = abs(positive) = positive`. Losing strategies appear profitable by Calmar ratio.

**Evidence (verified against source):**
- `risk/metrics.py:136`: `calmar = abs(ann_return / dd_metrics["max_drawdown"])`
- Correct: `calmar = ann_return / abs(dd_metrics["max_drawdown"])`

**File:** `risk/metrics.py`

**Implementation:**

1. Fix the Calmar ratio:
   ```python
   calmar = ann_return / abs(dd_metrics["max_drawdown"]) if dd_metrics["max_drawdown"] != 0 else 0.0
   ```

2. This preserves the sign of `ann_return` while normalizing by the magnitude of drawdown. A losing strategy (-10% return, -20% drawdown) correctly reports Calmar = -0.5, not +0.5.

3. Add a test for both profitable and unprofitable strategies:
   ```python
   def test_calmar_ratio_sign():
       # Profitable: 20% return, -10% drawdown → Calmar = 2.0
       report_profit = compute_risk_report(returns_profitable)
       assert report_profit.calmar_ratio > 0

       # Unprofitable: -10% return, -20% drawdown → Calmar = -0.5
       report_loss = compute_risk_report(returns_unprofitable)
       assert report_loss.calmar_ratio < 0
   ```

**Acceptance:** Calmar ratio is negative for losing strategies. Promotion gates and evaluation consumers can correctly distinguish profitable from unprofitable strategies.

---

## Cross-Audit Reconciliation Notes

### Audit 1 F-01 (ShockModePolicy spread_multiplier not applied) — EXCLUDED

Source inspection confirmed that `spread_multiplier` IS extracted from `ShockModePolicy` and passed to `ExecutionModel.simulate()` as the `event_spread_multiplier` parameter in both `_simulate_entry()` and `_simulate_exit()` within `backtest/engine.py`. The extraction pattern is:
```python
_shock_spread_mult = (
    shock_policy.spread_multiplier
    if shock_policy is not None and shock_policy.is_active else 1.0
)
```
This finding is not valid for the current source tree and is excluded from remediation.

### Overlap with Existing SPEC_07 and SPEC_08

- **SPEC_07 T4 (urgency pullback recompute):** Addresses Audit 1 F-04. No overlap with this spec.
- **SPEC_07 T2 (cost calibration input):** Audit 1 F-16 (stress multiplier decomposition) is a related but distinct issue. SPEC_07 T2 fixes the input; this spec's T5 fixes the dimensional error.
- **SPEC_08 T5 (null/stress result fields):** Audit 2 F-04 (summary.json fields) is about the *summary dict schema*, not the BacktestResult fields. Addressed here in T4.

---

## Verification

- [ ] Run `pytest tests/ -k "autopilot"` — no TypeError or AttributeError in validation calls
- [ ] Run `pytest tests/ -k "position_sizer or turnover"` — turnover budget constrains sizing
- [ ] Verify `/api/backtests/latest/equity-curve` returns valid JSON for orchestrator-produced trades
- [ ] Verify `summary.json` contains all 17 declared fields
- [ ] Verify `estimate_trade_cost_bps()` returns material cost values for realistic inputs
- [ ] Verify gross exposure = 1.2 for a {+0.5, -0.3, +0.4} portfolio
- [ ] Verify Calmar ratio < 0 for a strategy with negative annual return

---

*Generated from cross-audit reconciliation — 2026-02-28*
*Audit 1: 55 findings (2 P0, 8 P1, 19 P2, 26 P3)*
*Audit 2: 5 findings (1 P0, 3 P1, 1 P2)*
*Source verification: 8 findings checked, 7 confirmed, 1 rejected (Audit 1 F-01)*
