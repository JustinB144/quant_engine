# SPEC_AUDIT_FIX_31: Risk Math, Sharpe Consistency & Constraint Correctness

**Priority:** HIGH — P1/P2 math errors that weaken risk controls, bias metrics, or introduce look-ahead.
**Scope:** `backtest/engine.py`, `backtest/validation.py`, `backtest/advanced_validation.py`, `backtest/null_models.py`, `backtest/cost_stress.py`, `risk/position_sizer.py`, `risk/drawdown.py`, `risk/covariance.py`, `risk/factor_exposures.py`, `risk/portfolio_risk.py`, `risk/portfolio_optimizer.py`, `risk/stop_loss.py`, `risk/constraint_replay.py`, `risk/stress_test.py`
**Estimated effort:** 6–8 hours
**Depends on:** SPEC_30 (critical contracts must be fixed first)
**Blocks:** Nothing
**Audit sources:** Audit 1 F-05 through F-29, Audit 2 F-02 (related constraint issue)

---

## Context

Both audits identified multiple math and logic issues in the risk and validation modules. The most impactful are: 5 inconsistent Sharpe ratio conventions across validation files (undermining statistical rigor), hardcoded risk-free rate in 4+ locations, cost stress annualization error, covariance look-ahead bias from `bfill()`, portfolio optimizer re-normalization violating position limits, constraint replay not applying stress scenarios, and multiple smaller metric/constraint bugs.

---

## Tasks

### T1: Centralize Sharpe Ratio Calculation [P1 — Audit 1 F-05, F-06]

**Problem:** At least 5 different Sharpe conventions are used across the codebase:
1. `validation.py:313` — ddof=0, no Rf, no annualization (fold Sharpe)
2. `validation.py:447` — numpy std, Rf=0.04×holding/252, no annualization
3. `validation.py:1057` — ddof=1, Rf=0.04/252, annualized √252
4. `advanced_validation.py:253` — pandas std (ddof=1), no Rf, no annualization
5. `null_models.py:235` — std, no Rf, annualized √252

Additionally, `engine.py:1921,2201`, `validation.py:1057`, and `advanced_validation.py:335` all hardcode `rf=0.04`.

**Files:** New file `backtest/sharpe_utils.py`, plus `backtest/engine.py`, `backtest/validation.py`, `backtest/advanced_validation.py`, `backtest/null_models.py`, `config.py`

**Implementation:**

1. Add `RISK_FREE_RATE = 0.04` to `config.py` with STATUS annotation:
   ```python
   RISK_FREE_RATE = 0.04  # STATUS: ACTIVE — backtest/sharpe_utils.py; annualized risk-free rate for Sharpe/Sortino
   ```

2. Create `backtest/sharpe_utils.py` with a single canonical implementation:
   ```python
   """Canonical Sharpe and Sortino ratio calculations.

   All risk-adjusted return metrics in the backtest/validation pipeline should
   use these functions to ensure consistent conventions.

   Convention: ddof=1, configurable Rf, frequency-aware annualization.
   """
   import numpy as np
   from config import RISK_FREE_RATE


   def compute_sharpe(
       returns: np.ndarray,
       rf_annual: float = RISK_FREE_RATE,
       frequency: str = "daily",
       annualize: bool = True,
   ) -> float:
       """Compute Sharpe ratio with consistent conventions.

       Args:
           returns: Array of period returns (daily, per-trade, etc.)
           rf_annual: Annualized risk-free rate (default from config)
           frequency: "daily" (252/yr), "weekly" (52/yr), "monthly" (12/yr),
                      or "per_trade" (requires explicit periods_per_year)
           annualize: Whether to annualize the result

       Returns:
           Sharpe ratio (annualized if annualize=True)
       """
       if len(returns) < 2:
           return 0.0

       periods_map = {"daily": 252, "weekly": 52, "monthly": 12}
       periods_per_year = periods_map.get(frequency, len(returns))

       rf_per_period = rf_annual / periods_per_year
       excess = returns - rf_per_period
       mu = np.mean(excess)
       sigma = np.std(excess, ddof=1)

       if sigma < 1e-12:
           return 0.0

       sharpe = mu / sigma
       if annualize:
           sharpe *= np.sqrt(periods_per_year)
       return float(sharpe)


   def compute_sortino(
       returns: np.ndarray,
       rf_annual: float = RISK_FREE_RATE,
       frequency: str = "daily",
       annualize: bool = True,
       target: float = 0.0,
   ) -> float:
       """Compute Sortino ratio using downside deviation."""
       if len(returns) < 2:
           return 0.0

       periods_map = {"daily": 252, "weekly": 52, "monthly": 12}
       periods_per_year = periods_map.get(frequency, len(returns))

       rf_per_period = rf_annual / periods_per_year
       excess = returns - rf_per_period
       mu = np.mean(excess)
       downside = excess[excess < target]
       downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-12

       if downside_std < 1e-12:
           return 0.0

       sortino = mu / downside_std
       if annualize:
           sortino *= np.sqrt(periods_per_year)
       return float(sortino)
   ```

3. Replace all 5+ Sharpe implementations with calls to the canonical utility:
   - `engine.py:1921,2201` → `compute_sharpe(daily_returns)`
   - `validation.py:313` → `compute_sharpe(fold_returns, annualize=False)` (fold-level)
   - `validation.py:447` → `compute_sharpe(trade_returns, frequency="per_trade")`
   - `validation.py:1057` → `compute_sharpe(daily_returns)`
   - `advanced_validation.py:253` → `compute_sharpe(period_returns, annualize=False)`
   - `advanced_validation.py:335` → use `RISK_FREE_RATE` from config
   - `null_models.py:235` → `compute_sharpe(daily_returns)`

4. Remove all hardcoded `rf_annual = 0.04` and `rf = 0.04` occurrences.

**Acceptance:** All Sharpe/Sortino calculations route through `sharpe_utils.py`. Changing `RISK_FREE_RATE` in config propagates everywhere. `grep -rn "0\.04" backtest/ risk/` shows no hardcoded risk-free rates.

---

### T2: Fix Cost Stress Annualization [P1 — Audit 1 F-08]

**Problem:** `backtest/cost_stress.py:128` uses `√252` to annualize Sharpe from per-trade returns. If average holding period is 10 days, there are ~25 trades/year, and the correct factor is `√25`, not `√252`. The Sharpe is overstated by ~3.16×.

**File:** `backtest/cost_stress.py`

**Implementation:**

1. Use the canonical `compute_sharpe()` from T1 with frequency="per_trade":
   ```python
   from backtest.sharpe_utils import compute_sharpe

   # Estimate trades per year from actual data
   if len(gross_returns) >= 2 and holding_days_avg > 0:
       trades_per_year = 252.0 / holding_days_avg
   else:
       trades_per_year = 252.0  # Fallback to daily assumption

   sharpe = compute_sharpe(
       gross_returns,
       frequency="per_trade",
       annualize=True,
   )
   ```

   Note: This requires `compute_sharpe` to accept `periods_per_year` override or to compute it from the data.

2. Alternative approach if holding days aren't readily available:
   ```python
   # Derive trades_per_year from actual trade count and date range
   if len(gross_returns) > 1:
       date_range_years = (max_exit_date - min_entry_date).days / 365.25
       trades_per_year = len(gross_returns) / max(date_range_years, 0.01)
   ```

**Acceptance:** `CostStressResult.breakeven_multiplier` reflects correct annualization. A strategy with 25 trades/year no longer has its Sharpe inflated by 3×.

---

### T3: Fix float("inf") in CostStressResult [P1 — Audit 1 F-10]

**Problem:** `backtest/cost_stress.py:40` sets `breakeven_multiplier: float = float("inf")` when the strategy is profitable at all cost levels. `json.dumps()` raises `ValueError` on `inf`.

**File:** `backtest/cost_stress.py`

**Implementation:**

1. Replace `float("inf")` with `None`:
   ```python
   @dataclass
   class CostStressResult:
       ...
       breakeven_multiplier: Optional[float] = None  # None = profitable at all tested levels
   ```

2. Update all consumers to check for `None`:
   ```python
   if result.breakeven_multiplier is not None:
       # Strategy breaks even at this cost multiplier
   else:
       # Strategy is profitable at all tested cost levels
   ```

3. Add a JSON serialization test.

**Acceptance:** `json.dumps(asdict(cost_stress_result))` succeeds without error.

---

### T4: Fix Covariance bfill() Look-Ahead Bias [P2 — Audit 1 F-17]

**Problem:** `risk/covariance.py:96` uses `.ffill().bfill()`. The `bfill()` uses future data to fill missing returns at the start of the series, introducing look-ahead bias for assets with late start dates.

**File:** `risk/covariance.py`

**Implementation:**

1. Remove `bfill()` and use `ffill()` only, with NaN handling via dropping:
   ```python
   clean = clean.ffill().dropna(how="any")
   ```

2. Add a minimum observation threshold to avoid covariance estimates from sparse data:
   ```python
   if len(clean) < min_observations:
       logger.warning("Insufficient overlap (%d < %d) for covariance; using diagonal", len(clean), min_observations)
       return np.diag(clean.var().values)  # Diagonal fallback
   ```

3. Document the forward-fill convention:
   ```python
   # Forward-fill only (no bfill) to prevent look-ahead bias.
   # Assets with missing data at series start are dropped, not back-filled.
   ```

**Acceptance:** `bfill()` is removed from covariance estimation. Covariance matrices at time t never use data from t+1.

---

### T5: Fix Constraint Replay Not Applying Stress Scenarios [P1 — Audit 1 F-09]

**Problem:** `risk/constraint_replay.py:88-94` iterates stress scenarios (`market_return`, `volatility_multiplier`) but never applies them to the data. `compute_constraint_utilization()` receives unmodified data for every scenario, making all scenarios identical.

**File:** `risk/constraint_replay.py`

**Implementation:**

1. Apply scenario shocks before computing utilization:
   ```python
   for scenario in stress_scenarios:
       # Apply market return shock to portfolio returns
       shocked_returns = returns + scenario.get("market_return", 0.0)

       # Apply volatility multiplier to position weights
       vol_mult = scenario.get("volatility_multiplier", 1.0)
       shocked_positions = {k: v * vol_mult for k, v in positions.items()}

       utilization = compute_constraint_utilization(
           shocked_positions, shocked_returns, constraints
       )
       results[scenario["name"]] = utilization
   ```

2. Document what each scenario parameter does in the function docstring.

3. Add a test verifying different scenarios produce different results:
   ```python
   def test_constraint_replay_scenarios_differ():
       results = replay_constraints(positions, returns, scenarios=[
           {"name": "normal", "market_return": 0.0, "volatility_multiplier": 1.0},
           {"name": "stress", "market_return": -0.05, "volatility_multiplier": 2.0},
       ])
       assert results["normal"] != results["stress"]
   ```

**Acceptance:** Different stress scenarios produce different constraint utilization results. A stress scenario with `volatility_multiplier=2.0` shows tighter constraint utilization than normal.

---

### T6: Fix Portfolio Optimizer Re-Normalization Violating Position Limits [P2 — Audit 1 F-23]

**Problem:** `risk/portfolio_optimizer.py:266-277` clips small weights to 0 and renormalizes remaining weights to sum to 1. This can push individual weights above `max_position`.

**File:** `risk/portfolio_optimizer.py`

**Implementation:**

1. After renormalization, re-clip any weights that now exceed the maximum:
   ```python
   # Clip small weights
   weights[weights < min_weight] = 0.0

   # Renormalize
   total = weights.sum()
   if total > 0:
       weights = weights / total

   # Re-clip any weights that now exceed max_position after renormalization
   for _ in range(5):  # Iterate to convergence (usually 1-2 passes)
       violations = weights > max_position
       if not violations.any():
           break
       weights[violations] = max_position
       # Redistribute excess to non-capped weights
       excess = 1.0 - weights.sum()
       non_capped = ~violations & (weights > 0)
       if non_capped.any():
           weights[non_capped] += excess * (weights[non_capped] / weights[non_capped].sum())
   ```

2. Add assertion at the end: `assert all(weights <= max_position + 1e-8)`.

**Acceptance:** After optimization and normalization, no individual weight exceeds `max_position`. Test with a 3-asset portfolio where normalization would push weights above the limit.

---

### T7: Fix Weekly PnL Partial History in Drawdown Controller [P2 — Audit 1 F-13]

**Problem:** `risk/drawdown.py:115` uses only today's PnL when fewer than 5 days of history exist, ignoring accumulated losses from previous days.

**File:** `risk/drawdown.py`

**Implementation:**

1. Use `sum(history[-5:])` regardless of length:
   ```python
   # Weekly PnL = sum of last 5 trading days (or fewer if less history)
   weekly_pnl = sum(self._daily_pnl_history[-5:]) if self._daily_pnl_history else daily_pnl
   ```

2. This correctly accumulates 3 days of -1.5% into weekly_pnl = -4.5%, triggering the weekly limit.

**Acceptance:** 3 consecutive -1.5% days produce weekly_pnl = -4.5%, exceeding the -5% weekly limit and triggering the correct drawdown tier.

---

### T8: Fix Factor Exposures Beta Variance Mismatch [P2 — Audit 1 F-18]

**Problem:** `risk/factor_exposures.py:258-268` computes `bench_var` over the full benchmark slice but covariance only over overlapping dates. Mismatched denominators bias beta for assets with data gaps.

**File:** `risk/factor_exposures.py`

**Implementation:**

1. Compute both covariance and variance over the same overlapping date range:
   ```python
   # Align asset and benchmark to common dates
   common = asset_returns.dropna().index.intersection(bench_returns.dropna().index)
   if len(common) < 20:
       return 1.0  # Default beta

   asset_aligned = asset_returns.loc[common]
   bench_aligned = bench_returns.loc[common]

   bench_var = bench_aligned.var()
   cov = asset_aligned.cov(bench_aligned)
   beta = cov / bench_var if bench_var > 1e-12 else 1.0
   ```

**Acceptance:** Beta is computed from the same date range for both asset and benchmark. Assets with data gaps do not have systematically biased betas.

---

### T9: Fix Portfolio Beta Not Normalized by Total Weight [P2 — Audit 1 F-12]

**Problem:** `risk/portfolio_risk.py:815` accumulates `weighted_beta` but doesn't divide by `total_weight` when some tickers are missing.

**File:** `risk/portfolio_risk.py`

**Implementation:**

1. Normalize weighted beta by total contributing weight:
   ```python
   if total_weight > 0:
       portfolio_beta = weighted_beta / total_weight
   else:
       portfolio_beta = 1.0  # Default market-neutral assumption
   ```

**Acceptance:** A portfolio where 50% of weight has beta data returns a correctly scaled portfolio beta, not a half-weighted one.

---

### T10: Fix Stress Test max_loss_3sigma Mislabeled [P2 — Audit 1 F-22]

**Problem:** `risk/stress_test.py:495` labels `stress_var_99 * 3.0` as "max_loss_3sigma". The 99% VaR is already ~2.326σ. Multiplying by 3 gives ~7σ, not 3σ.

**File:** `risk/stress_test.py`

**Implementation:**

1. Rename the field and fix the calculation:
   ```python
   # Option A: Correct the label
   "max_loss_7sigma_approx": round(stress_var_99 * 3.0, 6),

   # Option B: Correct the calculation to be actual 3-sigma
   "max_loss_3sigma": round(portfolio_std * 3.0, 6),
   ```

2. Prefer Option B — rename to use actual 3σ from portfolio standard deviation, and add the extreme scenario as a separate field.

**Acceptance:** The `max_loss_3sigma` field represents an actual 3σ loss (mean - 3×std), not a ~7σ loss.

---

### T11: Fix Hard Stop Ignoring Spread Buffer [P2 — Audit 1 F-21]

**Problem:** `risk/stop_loss.py:152-153` evaluates the hard stop using percentage PnL but does not account for the spread buffer applied to other stop types (ATR, trailing). This means hard stops fire prematurely relative to actual fill prices.

**File:** `risk/stop_loss.py`

**Implementation:**

1. Apply the same spread buffer to the hard stop threshold:
   ```python
   hard_stop_adjusted = self.hard_stop - (self.spread_buffer_bps / 10_000)
   if unrealized <= hard_stop_adjusted:
       return StopResult(triggered=True, stop_type="hard", ...)
   ```

**Acceptance:** Hard stop accounts for spread buffer, consistent with ATR and trailing stops.

---

## Verification

- [ ] Run `pytest tests/ -k "risk or backtest or validation"` — all pass
- [ ] `grep -rn "0\.04" backtest/ risk/` shows no hardcoded risk-free rates (all use RISK_FREE_RATE)
- [ ] `grep -rn "bfill" risk/covariance.py` returns no hits
- [ ] Constraint replay produces different results for different stress scenarios
- [ ] Portfolio optimizer respects max_position after normalization
- [ ] Weekly PnL accumulates correctly in first week of trading

---

*Generated from cross-audit reconciliation — 2026-02-28*
