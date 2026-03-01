# Audit Report: Subsystem 05 — Backtesting + Risk

> **Status:** Complete
> **Auditor:** Claude (Opus 4.6)
> **Date:** 2026-02-28
> **Spec:** `docs/audit/subsystem_specs/SPEC_AUDIT_05_BACKTESTING_RISK.md`

---

## Executive Summary

All **28 files** and **~13,132 lines** in the `backtesting_risk` subsystem have been reviewed line-by-line. This is the largest and most behavior-critical subsystem — errors here invalidate all historical performance claims and promotion decisions. The subsystem is architecturally sound with comprehensive execution realism modeling, multi-tier risk management, and robust statistical validation. However, **2 P0 critical findings**, **8 P1 high findings**, **19 P2 medium findings**, and **26 P3 low findings** were identified. The P0 findings relate to a ShockModePolicy spread multiplier that is computed but never applied, and a cost model dimensional inconsistency that renders market impact estimation non-functional in the cost budget module.

---

## Scope & Ledger (T1)

### File Coverage

| # | File | Lines | Risk Tier | Reviewed |
|---|---|---:|---|---|
| 1 | `backtest/engine.py` | 2,488 | CRITICAL (16/21) | Yes |
| 2 | `risk/position_sizer.py` | 1,254 | HIGH (10/21) | Yes |
| 3 | `backtest/validation.py` | 1,074 | HIGH (11/21) | Yes |
| 4 | `backtest/execution.py` | 936 | MEDIUM (9/21) | Yes |
| 5 | `risk/portfolio_risk.py` | 885 | MEDIUM | Yes |
| 6 | `backtest/cost_calibrator.py` | 685 | MEDIUM | Yes |
| 7 | `backtest/advanced_validation.py` | 665 | MEDIUM | Yes |
| 8 | `risk/stress_test.py` | 547 | MEDIUM | Yes |
| 9 | `risk/attribution.py` | 368 | LOW | Yes |
| 10 | `risk/covariance.py` | 355 | LOW | Yes |
| 11 | `risk/metrics.py` | 320 | LOW | Yes |
| 12 | `risk/factor_monitor.py` | 300 | LOW | Yes |
| 13 | `risk/universe_config.py` | 295 | LOW | Yes |
| 14 | `risk/stop_loss.py` | 293 | LOW | Yes |
| 15 | `risk/portfolio_optimizer.py` | 279 | LOW | Yes |
| 16 | `risk/factor_exposures.py` | 268 | LOW | Yes |
| 17 | `risk/drawdown.py` | 260 | MEDIUM | Yes |
| 18 | `backtest/null_models.py` | 296 | LOW | Yes |
| 19 | `backtest/cost_stress.py` | 221 | LOW | Yes |
| 20 | `risk/factor_portfolio.py` | 220 | LOW | Yes |
| 21 | `risk/cost_budget.py` | 221 | LOW | Yes |
| 22 | `backtest/optimal_execution.py` | 200 | LOW | Yes |
| 23 | `risk/constraint_replay.py` | 197 | LOW | Yes |
| 24 | `backtest/adv_tracker.py` | 190 | LOW | Yes |
| 25 | `backtest/survivorship_comparison.py` | 163 | LOW | Yes |
| 26 | `validation/preconditions.py` | 71 | HIGH (11/21) | Yes |
| 27 | `backtest/__init__.py` | 15 | LOW | Yes |
| 28 | `risk/__init__.py` | 11 | LOW | Yes |

**Total: ~13,132 lines reviewed.** 28/28 files — 100% coverage. Matches spec expectation.

### Downstream Consumer Files Reviewed (Read-Only)

| File | Why |
|---|---|
| `models/trainer.py` | `enforce_preconditions()` consumer (line 219) |
| `autopilot/promotion_gate.py` | `BacktestResult` consumer (line 12) |
| `kalshi/promotion.py` | `BacktestResult` consumer (line 14) |
| `evaluation/engine.py` | `walk_forward_with_embargo`, `rolling_ic`, `detect_ic_decay` consumer |
| `api/orchestrator.py` | `Backtester` consumer (line 289) |
| `api/services/backtest_service.py` | `summary.json` reader |
| `api/services/results_service.py` | `summary.json` / `predictions.csv` reader |

---

## Invariant Verification Summary

| Invariant | Status | Evidence |
|---|---|---|
| Execution realism assumptions explicit | **PASS** | 54 config constants imported at engine.py:26-72; dynamic costs, shock mode, edge-cost gate all config-driven |
| BacktestResult schema stability | **PASS** | 31 fields (15 required + 16 defaulted), all consumer accesses verified against dataclass. No undocumented drift. |
| enforce_preconditions() contract symmetry | **PASS** | Both backtest/engine.py:198 and models/trainer.py:219 use identical lazy import + zero-arg call pattern, gated by TRUTH_LAYER_STRICT_PRECONDITIONS |
| Risk class lazy imports viable | **PASS** | All 5 classes (PositionSizer:316, DrawdownController:317, StopLossManager:318, PortfolioRiskManager:319, RiskMetrics:320) import successfully when use_risk_management=True |
| ShockVector integration correct | **PARTIAL** | ShockVector used correctly for shock-tier determination (is_shock_event, hmm_uncertainty). However, ShockModePolicy.spread_multiplier is computed but never consumed — see F-01. |
| UncertaintyGate consistency | **PASS** | All 3 consumers instantiate `UncertaintyGate()` with no args, using identical config defaults |
| Output artifact schemas verified | **PASS** | summary.json (11 keys via orchestrator.py:349-363), trades.csv (Trade dataclass: 17 fields), BacktestResult consumed by promotion_gate.py and kalshi/promotion.py — all field accesses verified |

---

## Findings

### P0 — Critical (Must Fix Before Production Use)

#### F-01: ShockModePolicy spread_multiplier computed but never applied
- **File:** `backtest/execution.py:76,529` + `backtest/engine.py:287-312`
- **Description:** `ShockModePolicy.from_shock_vector()` computes a `spread_multiplier` field (1.0x normal, 1.5x elevated, 2.0x shock). The engine passes `max_participation_override` from the policy to `ExecutionModel.simulate()`, but `simulate()` has no parameter for `spread_multiplier`. The shock-tier spread widening is **never actually applied** to execution costs.
- **Impact:** During shock events, participation limits are correctly tightened (0.5% vs 2%), but the spread widening that should make trading more expensive (2x spread during shocks) is silently dropped. This means backtest P&L during market stress is **overly optimistic** — trading costs are understated during the exact periods where they are highest.
- **Root Cause:** API gap between `ShockModePolicy` (which produces `spread_multiplier`) and `ExecutionModel.simulate()` (which has no mechanism to consume it). The `event_spread_multiplier` parameter in `simulate()` exists but is only used for explicit event windows, not for shock-mode spread widening.
- **Recommendation:** Pass `shock_policy.spread_multiplier` as `event_spread_multiplier` to `ExecutionModel.simulate()` in both `_simulate_entry()` and `_simulate_exit()`, or add a dedicated `shock_spread_multiplier` parameter.

#### F-02: Cost budget dimensional inconsistency makes market impact non-functional
- **File:** `risk/cost_budget.py:68`
- **Description:** `estimate_trade_cost_bps()` computes participation rate as `abs(trade_size_weight) / daily_dollar_volume`, dividing a dimensionless portfolio weight fraction (e.g., 0.05) by a dollar volume (e.g., $1M). The result (~5×10⁻⁸) is not a valid participation rate. The correct formula requires the portfolio's dollar value: `participation = (trade_size_weight * portfolio_value) / daily_dollar_volume`.
- **Impact:** Market impact in the cost budget module is effectively zero for any realistic portfolio. The cost model degrades to half-spread-only, making the "cost budget optimization" meaningless — it will always approve all trades since estimated costs are near-zero.
- **Recommendation:** Add `portfolio_value_usd` as a required parameter to `estimate_trade_cost_bps()` and `optimize_rebalance_cost()`.

---

### P1 — High (Should Fix)

#### F-03: Calmar ratio uses abs(), masking negative returns
- **File:** `risk/metrics.py:136`
- **Description:** `calmar = abs(ann_return / dd_metrics["max_drawdown"])`. Since `max_drawdown` is negative, `abs(negative_return / negative_drawdown) = abs(positive) = positive`. A losing strategy appears to have a positive Calmar ratio.
- **Impact:** Downstream evaluation consuming the `RiskReport.calmar_ratio` field cannot distinguish profitable from unprofitable strategies by Calmar alone.
- **Recommendation:** Change to `calmar = ann_return / abs(dd_metrics["max_drawdown"])`.

#### F-04: Urgency cost-limit reduction does not recalculate fill price
- **File:** `backtest/execution.py:758-783`
- **Description:** When the urgency cost-limit check triggers a fill reduction, `fill_ratio` and `fill_notional` are reduced, but `impact_bps`, `total_bps`, and `fill_price` are NOT recalculated. The returned `ExecutionFill` reflects the original (higher) slippage, not the reduced fill.
- **Impact:** Trades that trigger urgency-based fill reduction report higher costs than actually incurred. This causes backtest P&L to be slightly pessimistic for these trades, and the `tca_report` metrics will overstate slippage.
- **Recommendation:** Recalculate `participation_rate`, `impact_bps`, and `fill_price` after reducing `fill_notional`.

#### F-05: Hardcoded risk-free rate of 4%
- **File:** `backtest/engine.py:1921,2201`
- **Description:** `rf_annual = 0.04` is hardcoded in two places. This value is used for Sharpe ratio computation.
- **Impact:** When interest rates change, all Sharpe ratios in backtest results will be computed against a stale risk-free rate, affecting promotion gate decisions. The rate also appears hardcoded in `validation.py:1057` (`_sharpe` helper) and `advanced_validation.py:335` (Monte Carlo). There is no single configurable source.
- **Recommendation:** Add `RISK_FREE_RATE` to config.py and propagate to all Sharpe calculations.

#### F-06: Inconsistent Sharpe ratio calculations across validation files
- **File:** `backtest/validation.py`, `backtest/advanced_validation.py`, `backtest/null_models.py`
- **Description:** At least 5 different Sharpe ratio calculation conventions are used:
  - `validation.py:313` — ddof=0, no Rf, no annualization (fold Sharpe)
  - `validation.py:447` — numpy std, Rf=0.04×holding/252, no annualization
  - `validation.py:1057` — ddof=1, Rf=0.04/252, annualized sqrt(252)
  - `advanced_validation.py:253` — pandas std (ddof=1), no Rf, no annualization
  - `null_models.py:235` — std, no Rf, annualized sqrt(252)
- **Impact:** A strategy may "pass" one Sharpe-based gate but "fail" another purely due to calculation method differences. This undermines the statistical rigor of the validation pipeline.
- **Recommendation:** Create a single canonical `compute_sharpe()` utility and use it throughout.

#### F-07: Gross exposure computed as sum (not sum of absolutes)
- **File:** `risk/portfolio_risk.py:361`
- **Description:** `gross = sum(proposed_positions.values())` computes net exposure, not gross. A portfolio with +50% long and -50% short shows 0% gross exposure.
- **Impact:** For any portfolio with short positions, gross exposure constraints are unenforced. The constraint check at `if gross > eff_gross` will never trigger for hedged portfolios.
- **Recommendation:** Change to `gross = sum(abs(v) for v in proposed_positions.values())`.

#### F-08: Cost stress test uses sqrt(252) annualization on per-trade returns
- **File:** `backtest/cost_stress.py:128`
- **Description:** `sharpe = (mean_ret / max(std_ret, 1e-10)) * np.sqrt(252)`. The input `gross_returns` are per-trade returns (not daily), but the annualization factor assumes daily frequency.
- **Impact:** If average holding period is 10 days, the Sharpe is overstated by a factor of ~sqrt(252/25.2) = ~3.16x. Breakeven cost estimates derived from this Sharpe are therefore also wrong.
- **Recommendation:** Use `sqrt(trades_per_year)` for annualization, where `trades_per_year` is derived from the actual trading frequency.

#### F-09: Constraint replay stress scenarios are not applied
- **File:** `risk/constraint_replay.py:88-94`
- **Description:** Stress scenario parameters (`market_return`, `volatility_multiplier`) are iterated but never used to modify portfolio returns or price data. `compute_constraint_utilization()` is called with the same unmodified data for every scenario.
- **Impact:** The per-scenario breakdown in the replay report is meaningless — all scenarios produce identical constraint utilization results.
- **Recommendation:** Apply scenario shocks (return adjustments, volatility scaling) to the price data or position weights before computing utilization.

#### F-10: float("inf") in CostStressResult breaks JSON serialization
- **File:** `backtest/cost_stress.py:40`
- **Description:** `breakeven_multiplier: float = float("inf")` is set when the strategy is profitable at all tested cost levels. `json.dumps()` will raise `ValueError` on this value.
- **Impact:** Any downstream JSON serialization of the cost stress result (API responses, result persistence) will fail for strategies that never cross the breakeven threshold.
- **Recommendation:** Use `None` or a sentinel value (e.g., `999.0`) instead of `float("inf")`.

---

### P2 — Medium (Should Address)

#### F-11: ShockVector lookup may fail in _simulate_entry/_simulate_exit
- **File:** `backtest/engine.py:408-409,541-542`
- **Description:** Inside `_simulate_entry()` and `_simulate_exit()`, the shock vector lookup uses `ohlcv.attrs.get("ticker", ...)` as the key. If `ohlcv.attrs` is not set (common for plain DataFrames), `ticker_id` will be empty string, and shock vectors will never be found. The main signal processing paths use `permno` for lookup correctly, but the structural state conditioning within the execution model depends on this fragile `attrs`-based lookup.
- **Impact:** Structural state multipliers (break probability, uncertainty, drift, systemic stress) may not be applied to execution costs in some code paths.

#### F-12: Portfolio beta not normalized by total_weight
- **File:** `risk/portfolio_risk.py:815`
- **Description:** `weighted_beta` accumulates `weight * beta` but is returned without dividing by `total_weight`. When some tickers are missing from `price_data`, the beta is systematically under-estimated.
- **Impact:** Beta exposure constraint checks may pass when they should fail, allowing excessive market exposure.

#### F-13: Weekly PnL ignores partial history in drawdown controller
- **File:** `risk/drawdown.py:115`
- **Description:** When fewer than 5 days of history exist, weekly PnL defaults to just today's daily PnL, ignoring accumulated losses from previous days.
- **Impact:** In the first week of trading, the weekly loss limit can be exceeded without triggering the circuit breaker (3 days of -1.5% each = -4.5%, but weekly PnL only shows -1.5%).

#### F-14: Kelly division-by-zero when avg_win = 0
- **File:** `risk/position_sizer.py:710-712`
- **Description:** `b = abs(avg_win / avg_loss)` produces `b=0` when `avg_win=0`, then `(p*b - q)/b` is a division by zero. Python produces `-inf`, caught by the `kelly <= 0` check, but generates a runtime warning.
- **Impact:** Cosmetic (caught downstream), but the warning noise could mask real issues in production logs.

#### F-15: update_kelly_bayesian overwrites global counters instead of accumulating
- **File:** `risk/position_sizer.py:918-919`
- **Description:** `self._bayesian_wins = int((returns > 0).sum())` replaces the global counter each call. Per-regime counters use `+=` (accumulate). If called with partial trade batches, global posterior reflects only the latest batch.
- **Impact:** Bayesian Kelly estimates may be systematically wrong if `update_kelly_bayesian()` is called incrementally.

#### F-16: Cost calibration decomposition ignores stress multipliers
- **File:** `backtest/cost_calibrator.py:238`
- **Description:** `net_impact = max(0.0, cost - 0.5 * self._spread_bps)` uses the base spread to decompose realized cost. During stressed periods, actual spreads are wider due to structural/event multipliers, causing the decomposition to over-attribute cost to impact coefficients.
- **Impact:** Calibrated impact coefficients are systematically inflated during stress periods, making normal-period cost estimates too conservative.

#### F-17: Covariance bfill introduces look-ahead bias
- **File:** `risk/covariance.py:96`
- **Description:** `clean = clean.ffill().bfill().dropna(how="any")`. Back-filling uses future values to fill gaps.
- **Impact:** Covariance estimates at time t may use return data from t+1, introducing subtle look-ahead bias. Impact is small for occasional gaps but significant for thinly-traded assets.

#### F-18: Factor exposures beta variance mismatch
- **File:** `risk/factor_exposures.py:258-268`
- **Description:** `bench_var` is computed over the full `bench_returns[-lookback:]` slice, but covariance is computed only over dates where both asset and benchmark data exist. Mismatched denominators produce biased beta.
- **Impact:** Beta estimates for assets with data gaps are systematically biased (understated).

#### F-19: Factor portfolio uses np.linalg.inv instead of lstsq
- **File:** `risk/factor_portfolio.py:111-115`
- **Description:** Uses normal equations with `np.linalg.inv(X.T @ X)` for OLS, which is numerically unstable for ill-conditioned factor matrices. Ridge penalty `1e-8 * I` partially mitigates.
- **Impact:** Factor exposure estimates may be unreliable for correlated factors.

#### F-20: Timezone-naive datetime comparison in cost calibrator
- **File:** `backtest/cost_calibrator.py:456-461`
- **Description:** `datetime.fromisoformat(stored_timestamp)` may produce a naive datetime, then subtracted from `datetime.now(timezone.utc)`. In Python 3.12, this raises `TypeError`.
- **Impact:** Feedback recalibration interval check may crash if stored timestamps lack timezone info.

#### F-21: Hard stop ignores spread buffer
- **File:** `risk/stop_loss.py:152-153`
- **Description:** Hard stop trigger uses percentage PnL (`unrealized <= self.hard_stop`) but computes an unused spread-adjusted `hard_stop_price`. Unlike ATR and trailing stops, the hard stop does not account for the spread buffer.
- **Impact:** Hard stops can fire prematurely relative to actual fill prices, generating exits that would not have been triggered with spread-adjusted thresholds.

#### F-22: Stress test max_loss_3sigma mislabeled
- **File:** `risk/stress_test.py:495`
- **Description:** `"max_loss_3sigma": round(stress_var_99 * 3.0, 6)`. The 99% VaR is already ~2.326 sigma. Multiplying by 3 gives ~7-sigma, not 3-sigma as labeled.
- **Impact:** Downstream consumers interpreting this as a 3-sigma loss will dramatically overestimate worst-case risk.

#### F-23: Portfolio optimizer re-normalization can violate position limits
- **File:** `risk/portfolio_optimizer.py:266-277`
- **Description:** After optimization, small weights are clipped to 0 and remaining weights renormalized to sum to 1. This can push individual weights above `max_position`.
- **Impact:** The optimized portfolio may violate the position size constraint it was supposed to enforce.

#### F-24: Edge-cost gate logic duplicated ~110 lines
- **File:** `backtest/engine.py:1045-1101,1625-1680`
- **Description:** Identical edge-cost gate logic is duplicated verbatim between `_process_signals()` (simple mode) and `_process_signals_risk_managed()`.
- **Impact:** Maintenance risk — changes to one copy must be manually mirrored. Shock mode policy check and logging is also duplicated (~40 more lines).

#### F-25: ADV tracker double-counting on residual exit bars
- **File:** `backtest/engine.py` (lines 401, 533)
- **Description:** The ADV tracker is updated during both entry and exit simulations. For residual positions that exit across multiple bars, volume is counted multiple times per bar.
- **Impact:** Inflated ADV estimates during residual unwinding, leading to more permissive participation limits.

#### F-26: Duplicate ticker (NKE) in universe.yaml
- **File:** `config_data/universe.yaml:49,73`
- **Description:** NKE appears twice in the `consumer` sector. `_sector_to_tickers` will contain NKE twice, potentially causing double-counting in sector weight calculations.
- **Impact:** Consumer sector weight could be overcounted by one position's worth.

#### F-27: Almgren-Chriss impact coefficient units incompatible with ExecutionModel
- **File:** `backtest/optimal_execution.py:26 vs execution.py:181`
- **Description:** Almgren-Chriss uses `temporary_impact` in per-share units (default 0.01). ExecutionModel uses `impact_coefficient_bps` in bps per sqrt(participation) (default 25.0). No conversion bridge exists.
- **Impact:** The two execution models cannot share calibrated parameters. Users may pass Almgren-Chriss parameters to ExecutionModel (or vice versa) and get nonsensical results.

#### F-28: BH FDR threshold floor weakens multiple testing correction
- **File:** `backtest/validation.py:497`
- **Description:** When BH procedure says "nothing is significant" (threshold=0), the code uses `max(fdr_threshold, 0.001)` as a floor. This allows passage at p < 0.001 even when strict FDR control would reject everything.
- **Impact:** Strategies can pass the FDR-corrected statistical test even when BH says they shouldn't, if any individual p-value is < 0.001.

#### F-29: Survivorship comparison uses absolute thresholds for bias risk
- **File:** `backtest/survivorship_comparison.py:159-163`
- **Description:** Bias risk thresholds are absolute counts (>10 = HIGH, >3 = MEDIUM) regardless of universe size. Dropping 11 tickers from a 5000-ticker universe is flagged "HIGH."
- **Impact:** Misleading bias risk assessments for large universes.

---

### P3 — Low (Monitor / Defer)

#### F-30: Unused import ALMGREN_CHRISS_RISK_AVERSION
- **File:** `backtest/engine.py:36`
- **Description:** Imported from config but never referenced.

#### F-31: Dead branch in regime lookup building
- **File:** `backtest/engine.py:843-846`
- **Description:** Both branches of `if isinstance(idx, tuple)` conditional do the identical thing.

#### F-32: risk_report and null_baselines typed as Optional[object]
- **File:** `backtest/engine.py:132,141`
- **Description:** Too-loose typing loses IDE support and type safety.

#### F-33: Relative path for CostCalibrator model_dir
- **File:** `backtest/engine.py:259`
- **Description:** `model_dir=Path("trained_models")` depends on working directory.

#### F-34: iterrows() for regime lookup building
- **File:** `backtest/engine.py:842`
- **Description:** `iterrows()` is extremely slow for large DataFrames. A vectorized approach would be faster.

#### F-35: `field` imported but unused in position_sizer.py
- **File:** `risk/position_sizer.py:20`

#### F-36: No constructor input validation in PositionSizer
- **File:** `risk/position_sizer.py:69-81`
- **Description:** Does not validate `target_portfolio_vol > 0`, `max_position_pct > min_position_pct`, etc.

#### F-37: Turnover history grows unbounded
- **File:** `risk/position_sizer.py:677`
- **Description:** `_turnover_history` list grows forever. No automatic pruning.

#### F-38: Double uncertainty reduction (regime_uncertainty + regime_entropy)
- **File:** `risk/position_sizer.py:264-270,525-592`
- **Description:** `regime_uncertainty` reduces Kelly by up to 20% via UncertaintyGate, `regime_entropy` reduces composite by up to 30% via `_compute_uncertainty_scale()`. If both are high, Kelly is reduced ~44%. May be intentional but is not documented.

#### F-39: size_portfolio() ignores regime and uncertainty parameters
- **File:** `risk/position_sizer.py:1166-1177`
- **Description:** Bulk sizing calls `size_position()` without passing regime, uncertainty, or equity parameters.

#### F-40: Sortino vs Sharpe target inconsistency
- **File:** `backtest/engine.py:1931`
- **Description:** Sortino uses target=0, Sharpe uses risk-free rate. These measure excess return differently.

#### F-41: No logging in factor_monitor.py, factor_portfolio.py, cost_budget.py

#### F-42: Attribution.py labeled "Brinson-style" but implements factor regression decomposition
- **File:** `risk/attribution.py:4`

#### F-43: Beta fallback comment says "market-neutral" but value is 1.0 (market-tracking)
- **File:** `risk/attribution.py:36`

#### F-44: Value factor proxy uses negative trailing return (contrarian, not value)
- **File:** `risk/factor_exposures.py:112-117`

#### F-45: Factor monitor z-scores computed within portfolio, not vs broader universe
- **File:** `risk/factor_monitor.py:177-211`

#### F-46: Factor monitor timestamp field always empty
- **File:** `risk/factor_monitor.py:44,265-282`

#### F-47: No transaction costs in null model returns
- **File:** `backtest/null_models.py:123-127`
- **Description:** Null models trade every bar with zero costs, making them easier to beat.

#### F-48: Null models not integrated into validation pass/fail gating
- **File:** `backtest/null_models.py`
- **Description:** `compute_null_baselines` exists but is not wired into the statistical validation pipeline.

#### F-49: SPA test default n_bootstraps=400 provides coarse p-value resolution
- **File:** `backtest/validation.py:719`

#### F-50: Duplicated scipy fallback code in validation.py and advanced_validation.py
- **File:** `backtest/validation.py:27-107`, `backtest/advanced_validation.py:21-34`

#### F-51: Dead parameter `predictions` in quick_survivorship_check
- **File:** `backtest/survivorship_comparison.py:128`

#### F-52: Trade and feedback history grow unboundedly in CostCalibrator
- **File:** `backtest/cost_calibrator.py:158-189,280-329`

#### F-53: Zero-volume days silently dropped in ADV tracker
- **File:** `backtest/adv_tracker.py:65-67`

#### F-54: No warn-only mode in enforce_preconditions
- **File:** `validation/preconditions.py:60-71`
- **Description:** When `TRUTH_LAYER_STRICT_PRECONDITIONS=False`, validation is completely bypassed with no logging.

#### F-55: Stop loss assumes long-only positions
- **File:** `risk/stop_loss.py:138`
- **Description:** `unrealized = (current_price - entry_price) / entry_price` is wrong for short positions.

---

## Boundary Compatibility (T6)

### Cross-Subsystem Contract Verification

| Boundary ID | Status | Detail |
|---|---|---|
| `backtest_to_regime_shock_1` | **PARTIAL** | `compute_shock_vectors` signature matches. ShockVector 13-field schema matches. **But** `ShockModePolicy.spread_multiplier` not consumed (F-01). |
| `backtest_to_regime_uncertainty_2` | **PASS** | UncertaintyGate instantiated identically (no-args) across all 3 consumers. Config constants consistent. |
| `backtest_to_risk_3` | **PASS** | PositionSizer.size_position() called with 13 of 20 params from engine.py:1721-1735; remaining use defaults. Return type PositionSize correctly consumed. |
| `backtest_to_config_19` | **PASS** | All 54 config constants at engine.py:26-72 exist and are correctly typed. |
| `backtest_to_validation_20` | **PASS** | `enforce_preconditions()` at engine.py:198 uses identical semantics as models/trainer.py:219. |
| `models_to_validation_18` | **PASS** | Both paths call `enforce_preconditions()` with zero args inside `__init__()`, gated by same flag. |

### Consumer Contract Verification

| Consumer | Import | Status | Notes |
|---|---|---|---|
| `autopilot/promotion_gate.py:12` | `BacktestResult` | **PASS** | Accesses 8 fields (total_trades, win_rate, sharpe_ratio, profit_factor, max_drawdown, annualized_return, trades_per_year, regime_performance) — all exist. |
| `kalshi/promotion.py:14` | `BacktestResult` | **PASS** | Constructs BacktestResult with all 15 required fields. regime_performance defaults to `{}`, correctly handled by event_mode. |
| `evaluation/engine.py:272` | `walk_forward_with_embargo` | **PASS** | Call signature matches. Return fields (n_folds, folds, etc.) correctly consumed. |
| `api/orchestrator.py:289` | `Backtester` | **PASS** | Constructor args (holding_days, use_risk_management) valid. All result field accesses match dataclass. |
| `api/services/backtest_service.py` | Reads JSON | **PASS** | Pass-through reader; no specific key validation. |
| `api/services/results_service.py` | Reads JSON | **PASS** | Defensive column checks (`.get()`, `if X not in df.columns`). |

### JSON Schema Key Naming Note

`api/orchestrator.py:354-355` writes `"sharpe"` and `"sortino"` (abbreviated keys) to `summary.json`, while `BacktestResult` uses `sharpe_ratio` and `sortino_ratio`. This is not currently a contract violation (services are pass-through readers), but represents a latent inconsistency that could bite future consumers expecting field-name parity.

---

## BacktestResult Schema (T1)

### Trade Dataclass (17 fields)

| Field | Type | Default |
|---|---|---|
| `ticker` | str | required |
| `entry_date` | str | required |
| `exit_date` | str | required |
| `entry_price` | float | required |
| `exit_price` | float | required |
| `predicted_return` | float | required |
| `actual_return` | float | required |
| `net_return` | float | required |
| `regime` | int | required |
| `confidence` | float | required |
| `holding_days` | int | required |
| `position_size` | float | required |
| `exit_reason` | str | required |
| `fill_ratio` | float | 1.0 |
| `entry_impact_bps` | float | 0.0 |
| `exit_impact_bps` | float | 0.0 |
| `entry_reference_price` | float | 0.0 |
| `exit_reference_price` | float | 0.0 |
| `market_cap_segment` | str | "" |

### BacktestResult Dataclass (31 fields)

15 required positional fields + 16 defaulted fields. Key fields consumed by promotion gates:
- `total_trades`, `win_rate`, `sharpe_ratio`, `profit_factor`, `max_drawdown`, `annualized_return`, `trades_per_year`, `regime_performance`

Truth Layer extensions: `null_baselines`, `cost_stress_result` (both `Optional[object]`)

---

## Execution Realism Assessment (T2)

### Cost Model Structure (6 factors)

1. **Half spread** — base 3 bps, dynamically scaled by vol/gap/range
2. **Square-root market impact** — `impact_coeff * sqrt(participation_rate)`
3. **Volatility-dependent spread widening** — vol_spread_beta, gap_spread_beta, range_spread_beta
4. **Liquidity-dependent cost scaling** — `sqrt(dollar_volume_ref / daily_dollar_volume)`, clipped [0.70, 3.00]
5. **Structural state multiplier** — break probability, regime uncertainty, drift, systemic stress
6. **Event-window spread blowout** — external event windows

### Shock Mode (3-tier)

| Tier | Max Participation | Spread Mult | Min Confidence |
|---|---|---|---|
| Shock | 0.5% | 2.0x (NOT APPLIED — F-01) | 80% |
| Elevated | 1.0% | 1.5x (NOT APPLIED — F-01) | 65% |
| Normal | 2.0% | 1.0x | 50% |

### Participation Limits

- Base: `EXEC_MAX_PARTICIPATION` (default 2%)
- ADV-adjusted via volume trend ratio (clipped [0.5x, 2.0x])
- Almgren-Chriss upgrade above `ALMGREN_CHRISS_ADV_THRESHOLD`
- Minimum fill ratio: 20% (trades below this are dropped)

### Edge-Cost Gate

`predicted_edge_bps <= expected_cost_bps + cost_buffer_bps` → skip trade. Buffer includes `EDGE_COST_BUFFER_BASE_BPS * (1 + uncertainty)`.

---

## Risk Manager Assessment (T3)

### PositionSizer (EVOLVING — 20 parameters)

- **Kelly formula:** `f* = (p*b - q) / b` — mathematically correct
- **Half-Kelly applied:** `kelly_frac = 0.5`
- **Small-sample penalty:** Linear from 0 at 0 trades to full at 50 trades
- **Regime-conditional blend weights:** Kelly/vol/ATR blending varies by regime (CRITICAL regime: Kelly weight drops to 0.05)
- **Uncertainty scaling:** Composite confidence from signal_uncertainty (0.40w), regime_entropy (0.30w), drift_score (0.30w) → 0-30% reduction
- **UncertaintyGate:** Applied only to Kelly component (max 20% reduction)
- **Drawdown governor:** Exponential backoff below -5% portfolio drawdown

### DrawdownController

- 5 states: NORMAL → WARNING → CAUTION → CRITICAL → RECOVERY
- Thresholds: -5% (warning), -10% (caution, no new entries), -15% (critical, force liquidate)
- Recovery: Quadratic ramp over configurable period
- Daily and weekly loss limits enforced

### StopLossManager (6 stop types)

Hard stop (-8%), ATR stop (2x ATR), Trailing stop (1.5x ATR after +2% gain), Time stop (30 days), Regime change, Profit target (optional)

### PortfolioRiskManager (7 constraints)

Single-name (10%), Gross exposure (100%), Sector (40%), Pairwise correlation (0.85), Beta (1.5), Portfolio vol (config), Factor exposures (regime-conditioned bounds from universe.yaml)

---

## Statistical Validation Assessment (T4)

### Walk-Forward Validation
- **Purge gap:** Correctly removes `purge_gap` samples before test start
- **Embargo:** Correctly skips first `embargo` samples of test set
- **No look-ahead bias detected** — training always precedes test, purge/embargo properly implemented

### CPCV (Combinatorial Purged Cross-Validation)
- Evaluates pre-existing predictions (does NOT retrain per combination)
- Purge applied before test partitions, embargo after
- Pass criteria: median OOS corr > 0, >55% combinations positive, mean OOS return > 0

### SPA (Superior Predictive Ability)
- Block bootstrap with circular blocks (correct, Politis & Romano 1994)
- Default 400 bootstraps (coarse but configurable)
- Single-strategy variant (not multi-strategy White/Hansen)

### Deflated Sharpe Ratio
- Bailey & Lopez de Prado (2014) formula — verified correct
- Euler-Mascheroni correction for expected maximum Sharpe

### PBO (Probability of Backtest Overfitting)
- CSCV-based implementation
- Overfit threshold: pbo > 0.45
- `pbo_logit > 0` check is redundant (equivalent to pbo > 0.5)

---

## Shared Contract Assessment (T5)

### enforce_preconditions() — PASS

| Property | backtest/engine.py:198 | models/trainer.py:219 |
|---|---|---|
| Import type | Lazy (inside `__init__`) | Lazy (inside `__init__`) |
| Call signature | `enforce_preconditions()` (zero-arg) | `enforce_preconditions()` (zero-arg) |
| Gating flag | `TRUTH_LAYER_STRICT_PRECONDITIONS` | `TRUTH_LAYER_STRICT_PRECONDITIONS` |
| Failure mode | `RuntimeError` | `RuntimeError` |

The contract is **perfectly symmetric** between the two consumers. Both import the same function, call it the same way, and are gated by the same config flag. The function validates `PreconditionsConfig(ret_type=RET_TYPE, label_h=LABEL_H, px_type=PX_TYPE, entry_price_type=ENTRY_PRICE_TYPE)` via `config_structured.py` dataclass validation.

---

## Hardcoded Values Inventory

Notable hardcoded values that should be configurable (not exhaustive):

| File | Line | Value | Description |
|---|---|---|---|
| `engine.py` | 188 | `0.0005` | Legacy slippage percentage (5 bps) |
| `engine.py` | 323-324 | `2x, 0.25x` | Max/min position pct multipliers |
| `engine.py` | 328-331 | `3x, 2.0, 1.5` | Max holding, ATR stop, trailing stop multipliers |
| `engine.py` | 1248 | `20` | Max bars for residual unwinding |
| `engine.py` | 1700 | `0.02` | ATR fallback (2% of price) |
| `engine.py` | 1709 | `0.50, 0.02, -0.02` | Default Kelly params (insufficient history) |
| `engine.py` | 1921,2201 | `0.04` | Risk-free rate (4%) |
| `position_sizer.py` | 720 | `50` | Trades for full Kelly penalty removal |
| `position_sizer.py` | 791 | `2.0` | Drawdown governor curvature |
| `execution.py` | 254 | `20.0` | Base transaction cost bps |
| `execution.py` | 356 | `[0.70, 3.0]` | Structural multiplier clip range |
| `cost_stress.py` | 128 | `252` | Annualization (assumes daily) |
| `cost_calibrator.py` | 69 | `40,30,20,15` | Default impact coefficients by market cap |

---

## Test Coverage Assessment

| Module | Test Files | Coverage Assessment |
|---|---|---|
| `backtest/` | Some tests exist | Engine has moderate test coverage |
| `risk/` | Some tests exist | Position sizer, drawdown have tests; factor modules under-tested |
| `validation/` | **0 dedicated tests** | **CRITICAL GAP** — execution contracts (preconditions.py) are safety gates that are themselves unverified |

---

## Summary Statistics

| Severity | Count | Status |
|---|---|---|
| P0 Critical | 2 | F-01, F-02 |
| P1 High | 8 | F-03 through F-10 |
| P2 Medium | 19 | F-11 through F-29 |
| P3 Low | 26 | F-30 through F-55 |
| **Total** | **55** | |

### Critical Hotspot Disposition

| File | Hotspot Score | Disposition |
|---|---|---|
| `backtest/engine.py` | 16/21 | **1 P0, 2 P1, 3 P2, 5 P3** — largest and most complex file. F-01 (spread multiplier gap) is the most impactful finding. |
| `risk/position_sizer.py` | 10/21 | **0 P0, 0 P1, 2 P2, 5 P3** — Kelly formula correct. EVOLVING status justified (20 params). Double uncertainty reduction should be documented. |
| `validation/preconditions.py` | 11/21 | **0 P0, 0 P1, 0 P2, 1 P3** — Cross-subsystem contract is perfectly symmetric. Well-implemented but lacks a warn-only mode. |
| `backtest/validation.py` | 11/21 | **0 P0, 1 P1, 2 P2, 4 P3** — Statistical tests are mathematically sound. Sharpe inconsistency (F-06) is the main concern. |
| `backtest/execution.py` | 9/21 | **1 P0, 1 P1, 0 P2, 0 P3** — F-01 and F-04 both originate here. The execution model itself is comprehensive and well-designed. |
| `risk/portfolio_risk.py` | N/A | **0 P0, 1 P1, 1 P2, 0 P3** — Gross exposure bug (F-07) is significant for long-short portfolios. |
| `risk/drawdown.py` | N/A | **0 P0, 0 P1, 1 P2, 0 P3** — Weekly PnL partial history bug (F-13). Overall design is solid. |
| `risk/metrics.py` | N/A | **0 P0, 1 P1, 0 P2, 1 P3** — Calmar ratio sign bug (F-03). Missing Sharpe/Sortino from RiskReport is a gap. |

---

## Recommendations (Priority Order)

### Must-Fix Before Production

1. **F-01:** Wire `ShockModePolicy.spread_multiplier` into `ExecutionModel.simulate()` — backtest P&L during market stress is currently overly optimistic.
2. **F-02:** Add `portfolio_value_usd` parameter to `cost_budget.py` — the cost budget optimizer is non-functional without it.

### Should-Fix

3. **F-03:** Fix Calmar ratio sign — `ann_return / abs(max_drawdown)` instead of `abs(ann_return / max_drawdown)`.
4. **F-05/F-06:** Centralize Sharpe ratio calculation — create a single canonical `compute_sharpe(returns, rf, frequency)` utility.
5. **F-07:** Fix gross exposure to use `sum(abs(v))` for long-short support.
6. **F-08:** Fix cost stress annualization to use `sqrt(trades_per_year)`.
7. **F-10:** Replace `float("inf")` with JSON-serializable sentinel.

### Should-Address

8. **F-04:** Recalculate fill price after urgency-based fill reduction.
9. **F-09:** Apply stress scenario parameters to price data in constraint replay.
10. **F-13:** Fix weekly PnL to use `sum(history[-5:])` regardless of length.
11. **F-17:** Remove `bfill()` from covariance estimation to prevent look-ahead bias.

### Testing Gaps to Close

12. Add dedicated tests for `validation/preconditions.py` (cross-subsystem contract).
13. Add integration tests verifying ShockVector → ShockModePolicy → ExecutionModel pipeline end-to-end.
14. Add parametric tests for Kelly edge cases (avg_win=0, extreme win rates).

---

*Generated by Claude (Opus 4.6) — 2026-02-28*
*All 28 files (13,132 lines) reviewed. All HIGH boundaries dispositioned.*
