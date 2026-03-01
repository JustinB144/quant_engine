# SPEC_AUDIT_FIX_16: Risk Management Constraint & Position Sizing Fixes

**Priority:** CRITICAL/HIGH — Short positions bypass all core risk constraints; drawdown escalation can suppress liquidation; stress replay ignores scenario severity.
**Scope:** `risk/` — `portfolio_risk.py`, `drawdown.py`, `constraint_replay.py`, `portfolio_optimizer.py`, `stress_test.py`, `stop_loss.py`, `factor_exposures.py`, `position_sizer.py`
**Estimated effort:** 5–6 hours
**Depends on:** Nothing
**Blocks:** Nothing

---

## Context

The risk subsystem has five Critical defects: single-name, gross, and sector constraint checks use signed values so short positions can bypass all limits (a -50% position passes a 30% cap); drawdown daily/weekly branches execute before the critical-tier check, so a catastrophic -20% day enters CAUTION not CRITICAL; `replay_with_stress_constraints` computes utilization without applying scenario shock parameters, making scenario severity cosmetic; the optimizer fallback returns unconstrained equal weights that can violate `max_position`; and `correlation_stress_test` assumes weight vector order matches covariance matrix column order without validating asset mapping. High-risk issues include missing correlation utilization, un-normalized portfolio beta for net-short books, unused hard-stop spread buffer, absolute-weight factor exposure aggregation, Bayesian Kelly double-counting, and no clean no-trade path in position sizer.

---

## Tasks

### T1: Fix Short Position Bypass of Risk Constraints

**Problem:** `portfolio_risk.py:353-356` checks `position_size > eff_single` on the signed value. A short position of -0.50 (50% short) passes because `-0.50 < 0.30`. Similarly, `gross = sum(proposed_positions.values())` at line 361 nets longs and shorts rather than summing absolute values. Sector checks at lines 481-496 in `compute_constraint_utilization()` have the same signed-value issue. `constraint_replay.py:110-119` compounds this.

**Files:** `risk/portfolio_risk.py`, `risk/constraint_replay.py`

**Implementation:**
1. In `check_risk_constraints()`, use absolute values for single-name and gross:
   ```python
   # Single-name check — use absolute position size
   abs_position = abs(position_size)
   if abs_position > eff_single:
       violations.append(f"Position size {abs_position:.1%} > max {eff_single:.1%}")
   if eff_single > 0:
       constraint_util["single_name"] = abs_position / eff_single

   # Gross exposure check — sum of absolute weights
   gross = sum(abs(w) for w in proposed_positions.values())
   ```
2. In `compute_constraint_utilization()`, apply the same absolute-value treatment:
   ```python
   gross = sum(abs(w) for w in positions.values())
   # Single name
   if positions and eff_single > 0:
       single_util = max(abs(w) for w in positions.values()) / eff_single
   ```
3. For sector checks, sum absolute weights per sector:
   ```python
   sector_weights[sector] = sector_weights.get(sector, 0.0) + abs(weight)
   ```
4. In `constraint_replay.py:110-119`, apply matching absolute-value fixes:
   ```python
   max_sector_util = max(abs(v) for v in sector_weights.values()) / eff_sector_cap
   gross = sum(abs(w) for w in positions.values())
   single_util = max(abs(w) for w in positions.values()) / risk_mgr.max_single
   ```
5. Add a net exposure metric separately for informational purposes: `constraint_util["net_exposure"] = sum(positions.values())`.

**Acceptance:** A portfolio with positions `{"A": 0.25, "B": -0.50}` reports gross=0.75, single_name=0.50. A -50% short position triggers a single-name violation against a 30% cap.

---

### T2: Fix Drawdown Escalation Order

**Problem:** `drawdown.py:121-128` checks daily/weekly limits before cumulative drawdown tiers. A -20% single-day loss that breaches `daily_limit` enters CAUTION (line 123), even if cumulative drawdown also breaches `critical_thresh`. The daily limit check shadows the more severe CRITICAL state.

**File:** `risk/drawdown.py`

**Implementation:**
1. Reorder the state determination to check the most severe state first:
   ```python
   # Determine state — most severe first
   if dd <= self.critical_thresh:
       new_state = DrawdownState.CRITICAL
       messages.append(f"CRITICAL: Drawdown {dd:.1%} breached {self.critical_thresh:.1%}")
   elif daily < self.daily_limit and daily < self.critical_thresh:
       # Daily loss breaches limit AND is catastrophic — CRITICAL
       new_state = DrawdownState.CRITICAL
       messages.append(f"CRITICAL: Daily loss {daily:.1%} breached limit {self.daily_limit:.1%}")
   elif daily < self.daily_limit:
       new_state = DrawdownState.CAUTION
       messages.append(f"Daily loss {daily:.1%} breached limit {self.daily_limit:.1%}")
   elif weekly < self.weekly_limit:
       new_state = DrawdownState.CAUTION
       messages.append(f"Weekly loss {weekly:.1%} breached limit {self.weekly_limit:.1%}")
   elif dd <= self.caution_thresh:
       new_state = DrawdownState.CAUTION
       messages.append(f"CAUTION: Drawdown {dd:.1%} breached {self.caution_thresh:.1%}")
   elif dd <= self.warning_thresh:
       new_state = DrawdownState.WARNING
       messages.append(f"WARNING: Drawdown {dd:.1%} breached {self.warning_thresh:.1%}")
   elif prev_state in (DrawdownState.CAUTION, DrawdownState.CRITICAL):
       new_state = DrawdownState.RECOVERY
   ```
2. The key principle: always take the maximum severity across all breach types. If cumulative drawdown is CRITICAL but daily is only CAUTION-level, result is CRITICAL.
3. Alternatively, compute all breach levels independently and take `max()`:
   ```python
   severities = []
   if dd <= self.critical_thresh: severities.append(DrawdownState.CRITICAL)
   if daily < self.daily_limit: severities.append(DrawdownState.CAUTION)
   if weekly < self.weekly_limit: severities.append(DrawdownState.CAUTION)
   if dd <= self.caution_thresh: severities.append(DrawdownState.CAUTION)
   if dd <= self.warning_thresh: severities.append(DrawdownState.WARNING)
   new_state = max(severities, key=lambda s: _severity_order[s]) if severities else DrawdownState.NORMAL
   ```

**Acceptance:** A -20% daily loss that also breaches the critical cumulative threshold enters CRITICAL, not CAUTION. All breach reasons appear in messages.

---

### T3: Fix Stress Replay to Apply Scenario Parameters

**Problem:** `constraint_replay.py:88-133` iterates over scenarios but `compute_constraint_utilization()` at line 90-93 is called with the same positions and `stress_regime` regardless of `scenario_params`. The `market_shock` and `vol_multiplier` from line 133-134 are only written to output metadata, not applied to the constraint math. All scenarios produce identical utilization numbers.

**File:** `risk/constraint_replay.py`

**Implementation:**
1. Apply scenario shocks to positions before computing utilization:
   ```python
   for scenario_name, scenario_params in stress_scenarios.items():
       market_shock = scenario_params.get("market_return", 0.0)
       vol_multiplier = scenario_params.get("volatility_multiplier", 1.0)

       # Apply market shock to position values
       stressed_positions = {}
       for ticker, weight in positions.items():
           # Scale position by (1 + market_shock * beta)
           beta = _estimate_position_beta(ticker, price_data)
           stressed_weight = weight * (1.0 + market_shock * beta)
           stressed_positions[ticker] = stressed_weight

       # Tighten constraints under high-vol scenario
       stress_tightening = min(1.0, 1.0 / max(vol_multiplier, 1.0))

       util = risk_mgr.compute_constraint_utilization(
           positions=stressed_positions,
           price_data=price_data,
           regime=stress_regime,
       )
   ```
2. Pass `vol_multiplier` to affect the effective limits (tighter constraints under stress):
   ```python
   # Effective limits tightened by vol multiplier
   eff_sector_cap = base_sector_cap * stress_tightening
   eff_gross = base_gross * stress_tightening
   ```
3. Update the utilization computations to use stressed positions and tightened limits.

**Acceptance:** A scenario with `market_return=-0.38` produces different utilization numbers than one with `market_return=-0.07`. A high-vol scenario tightens effective limits.

---

### T4: Fix Optimizer Fallback to Respect Constraints

**Problem:** `portfolio_optimizer.py:257-262` falls back to equal weights (`1/n`) when SLSQP fails. For `n=3` assets, each gets 33.3%. If `max_position=0.25`, the fallback violates it. The fallback also ignores min_weight, sector limits, and any other constraints.

**File:** `risk/portfolio_optimizer.py`

**Implementation:**
1. After generating equal weights, clip to constraint bounds:
   ```python
   if not result.success:
       logger.warning(
           "Portfolio optimizer did not converge: %s. Falling back to constrained equal weight.",
           result.message,
       )
       equal_w = np.full(n, 1.0 / n)
       # Clip to max_position constraint
       if max_position is not None and max_position > 0:
           equal_w = np.minimum(equal_w, max_position)
       # Clip to min_weight
       if min_weight is not None:
           equal_w = np.maximum(equal_w, min_weight)
       # Re-normalize to sum to 1
       w_sum = equal_w.sum()
       if w_sum > 1e-10:
           equal_w = equal_w / w_sum
       # Final clip again after normalization (normalization can push above max)
       if max_position is not None and max_position > 0:
           for _ in range(5):  # Iterative projection
               equal_w = np.minimum(equal_w, max_position)
               w_sum = equal_w.sum()
               if w_sum > 1e-10:
                   equal_w = equal_w / w_sum
               if np.all(equal_w <= max_position + 1e-8):
                   break
       return pd.Series(equal_w, index=assets, name="weight")
   ```
2. Add a `fallback_used=True` flag to the return metadata so callers know optimization didn't converge.

**Acceptance:** For 3 assets with `max_position=0.25`, the fallback returns weights ≤0.25 each (not 0.333). The returned weights sum to 1.0.

---

### T5: Fix Correlation Stress Test Weight-Covariance Alignment

**Problem:** `stress_test.py:435` creates the weight vector as `np.array(list(portfolio_weights.values()))`, relying on dict insertion order matching the covariance matrix column order. Line 437 validates only length, not asset mapping. If covariance and weights have different asset orderings, the stress test silently computes wrong risk numbers.

**File:** `risk/stress_test.py`

**Implementation:**
1. Accept both weights and covariance with explicit index alignment:
   ```python
   def correlation_stress_test(
       portfolio_weights: pd.Series,   # Changed: require pd.Series with asset index
       covariance: pd.DataFrame,       # Changed: require pd.DataFrame with asset columns
       ...
   ) -> dict:
       # Align weight vector to covariance column order
       common_assets = covariance.columns.intersection(portfolio_weights.index)
       if len(common_assets) < len(portfolio_weights):
           missing = set(portfolio_weights.index) - set(common_assets)
           logger.warning("Assets in weights but not in covariance: %s", missing)

       cov_aligned = covariance.loc[common_assets, common_assets]
       w = portfolio_weights.reindex(common_assets, fill_value=0.0).values
       cov = cov_aligned.values
   ```
2. If callers pass dict + ndarray (backward compat), convert to pd.Series/DataFrame with explicit index:
   ```python
   if isinstance(portfolio_weights, dict):
       portfolio_weights = pd.Series(portfolio_weights)
   if isinstance(covariance, np.ndarray):
       logger.warning("Passing ndarray covariance without asset labels is unsafe; assuming order matches weights")
       covariance = pd.DataFrame(covariance, index=portfolio_weights.index, columns=portfolio_weights.index)
   ```

**Acceptance:** Passing weights with order `["A", "B", "C"]` and covariance with order `["C", "A", "B"]` produces the same result as aligned inputs. A test with deliberately misaligned inputs validates correctness.

---

### T6: Wire Correlation Utilization Into Constraint System

**Problem:** `portfolio_risk.py:460` `compute_constraint_utilization()` computes gross, single-name, and sector utilizations but never produces a `"correlation"` key. `constraint_replay.py:122` reads `util.get("correlation", 0.0)`, which always returns 0.0.

**File:** `risk/portfolio_risk.py`

**Implementation:**
1. In `compute_constraint_utilization()`, compute and return correlation utilization:
   ```python
   # Correlation utilization
   avg_corr = self._compute_avg_pairwise_correlation(positions, price_data)
   if self.max_corr > 0:
       util["correlation"] = avg_corr / self.max_corr
   else:
       util["correlation"] = 0.0
   ```
2. Verify that `_compute_avg_pairwise_correlation` is accessible from this method (it's in the same class — confirmed at line 730+).

**Acceptance:** `compute_constraint_utilization()` returns a dict with a `"correlation"` key reflecting actual portfolio correlation relative to the limit. A portfolio with avg_corr=0.6 and max_corr=0.7 reports utilization of ~0.86.

---

### T7: Fix Portfolio Beta Normalization for Net-Short Books

**Problem:** `portfolio_risk.py:740-755` computes portfolio beta as a weighted average but doesn't normalize by total weight. For a net-short portfolio where weights sum to -0.2, the beta is over-scaled. The formula should use absolute-weight normalization or total-net-weight normalization.

**File:** `risk/portfolio_risk.py`

**Implementation:**
1. Normalize beta by sum of absolute weights:
   ```python
   total_abs_weight = sum(abs(w) for w in positions.values())
   if total_abs_weight > 1e-8:
       portfolio_beta = sum(
           (abs(w) / total_abs_weight) * _estimate_beta(ticker, price_data)
           * np.sign(w)  # Short positions contribute negative beta
           for ticker, w in positions.items()
       )
   else:
       portfolio_beta = 0.0
   ```
2. This ensures: a 50% long (beta=1.2) + 50% short (beta=0.8) = 0.5*1.2 - 0.5*0.8 = 0.20 net beta, properly normalized.

**Acceptance:** A portfolio with equal long and short positions in similar-beta stocks reports near-zero net beta, not a collapsed or over-scaled value.

---

### T8: Wire Hard-Stop Spread Buffer Into Trigger Condition

**Problem:** `stop_loss.py:152` computes `hard_stop_price = entry_price * (1 + self.hard_stop) - spread_buf`, but line 153 checks `unrealized <= self.hard_stop` (comparing raw return against raw threshold). The spread-buffered `hard_stop_price` is returned for informational purposes but not used in the trigger decision.

**File:** `risk/stop_loss.py`

**Implementation:**
1. Derive the effective threshold from the spread-buffered price:
   ```python
   # Hard stop (spread-adjusted)
   hard_stop_price = entry_price * (1 + self.hard_stop) - spread_buf
   # Effective threshold accounts for spread cost
   effective_hard_stop = (hard_stop_price / entry_price) - 1.0
   if unrealized <= effective_hard_stop:
       return StopResult(
           should_exit=True,
           reason=StopReason.HARD_STOP,
           stop_price=hard_stop_price,
           ...
       )
   ```
2. Alternatively (simpler), compute unrealized relative to the spread-buffered price:
   ```python
   current_price = entry_price * (1 + unrealized)
   if current_price <= hard_stop_price:
       return StopResult(should_exit=True, ...)
   ```

**Acceptance:** A position with entry_price=100, hard_stop=-0.10, and spread_buf=0.50 triggers at price 89.50 (not 90.00). The spread buffer widens the stop.

---

### T9: Fix Factor Exposure Aggregation for Short Positions

**Problem:** `factor_exposures.py:139-148` uses `abs(weight)` for normalization (`abs_weights`, `norm_weights`) and then computes `np.average(betas, weights=norm_weights)`. Since all weights are absolute, a 50% short position contributes positive beta weight, making the portfolio appear long-biased when it's actually hedged.

**File:** `risk/factor_exposures.py`

**Implementation:**
1. Use signed weights for beta aggregation, absolute weights only for normalization denominator:
   ```python
   signed_weights = np.array([p["weight"] for p in position_data])
   abs_weights = np.abs(signed_weights)
   total_abs_weight = abs_weights.sum()
   if total_abs_weight < 1e-8:
       return exposures

   # Beta: signed weight contribution (shorts contribute negative beta)
   betas = np.array([p["beta"] for p in position_data])
   exposures["beta"] = float(np.sum(signed_weights * betas) / total_abs_weight)
   ```
2. For non-directional factors (size, vol, value), absolute weights remain appropriate.

**Acceptance:** A portfolio with 50% long (beta=1.2) and 50% short (beta=1.0) reports net beta exposure of 0.10 (not 1.10).

---

### T10: Fix Bayesian Kelly Double-Counting

**Problem:** `position_sizer.py:929` uses `+=` to accumulate regime wins/losses from a trade history DataFrame. If `update_kelly_bayesian()` is called repeatedly over the full trade history (not just new trades), all prior trades are double-counted, inflating the posterior and creating overconfident Kelly fractions.

**File:** `risk/position_sizer.py`

**Implementation:**
1. Replace incremental accumulation with full replacement when processing a complete history:
   ```python
   def update_kelly_bayesian(self, trades: pd.DataFrame, regime_col: str = "regime",
                              incremental: bool = False):
       """Update Bayesian Kelly prior from trade results.

       Parameters
       ----------
       incremental : bool
           If True, add to existing counts (caller guarantees no overlap).
           If False (default), reset counts and recompute from full history.
       """
       if not incremental:
           # Reset counters before recomputing
           for regime_id in self._bayesian_regime:
               self._bayesian_regime[regime_id] = {"wins": 0, "losses": 0}

       # Existing accumulation logic...
   ```
2. Update all callers to pass `incremental=True` only when providing exclusively new trades.
3. Log total counts after update for observability.

**Acceptance:** Calling `update_kelly_bayesian(full_history)` twice produces the same posterior as calling it once. Counts do not double.

---

### T11: Fix Diversification Benefit Lost Formula

**Problem:** `stress_test.py:482-488` computes `diversification_benefit_lost = 1 - (normal_vol / weighted_avg_vol)`. This is the normal diversification benefit, not the benefit *lost* under stress. The "lost" metric should compare normal vs stress diversification.

**File:** `risk/stress_test.py`

**Implementation:**
1. Compute benefit under both regimes and report the difference:
   ```python
   # Normal diversification benefit
   normal_div_benefit = 1.0 - (normal_vol / weighted_avg_vol) if weighted_avg_vol > 1e-12 else 0.0

   # Stress diversification benefit
   stress_weighted_avg_vol = float(np.abs(w) @ (vols * np.sqrt(vol_multiplier_factor)))
   stress_div_benefit = 1.0 - (stress_vol / stress_weighted_avg_vol) if stress_weighted_avg_vol > 1e-12 else 0.0

   # Benefit lost = how much diversification shrinks under stress
   div_benefit_lost = max(0.0, normal_div_benefit - stress_div_benefit)
   ```
2. Return both `normal_diversification_benefit` and `diversification_benefit_lost` for clarity.

**Acceptance:** A portfolio that is 40% diversified normally but only 10% under stress reports `diversification_benefit_lost ≈ 0.30`.

---

## Verification

- [ ] Run `pytest tests/ -k "risk or portfolio or drawdown or stress or position_sizer or constraint"` — all pass
- [ ] Verify short positions trigger constraint violations (signed positions)
- [ ] Verify -20% daily loss reaching critical drawdown enters CRITICAL, not CAUTION
- [ ] Verify different stress scenarios produce different utilization numbers
- [ ] Verify optimizer fallback respects max_position constraint
- [ ] Verify correlation stress test with misaligned inputs produces correct results
- [ ] Verify correlation utilization key is present in constraint output
- [ ] Verify hard-stop spread buffer affects trigger condition
