# SPEC_AUDIT_FIX_05: Autopilot Critical Logic & Data Integrity Fixes

**Priority:** CRITICAL — P0 bugs affecting position sizing, A/B testing validity, and signal integrity.
**Scope:** `autopilot/` — `engine.py`, `paper_trader.py`, `meta_labeler.py`
**Estimated effort:** 4–5 hours
**Depends on:** Nothing
**Blocks:** SPEC_06 (reliability fixes build on corrected logic)

---

## Context

The autopilot subsystem has four P0 defects: drawdown state is mutated N times per cycle (once per candidate) instead of once, portfolio optimizer weights are computed but never consumed by execution sizing, A/B test Kelly overrides are calculated but never applied to the sizer, and meta-label retraining leaks across assets due to global concatenation of rolling features.

---

## Tasks

### T1: Fix Drawdown State Mutation During Sizing

**Problem:** `paper_trader.py:709` calls `self._dd_controller.update(0.0)` inside `_position_size_pct()`, which is called once per candidate entry. The `update()` method in `drawdown.py:102-104` increments `_total_days`, appends to `state_history`, and potentially transitions the drawdown state. This means drawdown regime/timers drift based on how many symbols were evaluated, not actual time/PnL.

**Files:** `autopilot/paper_trader.py`, `risk/drawdown.py`

**Implementation:**
1. Add a read-only property to `DrawdownController`:
   ```python
   @property
   def current_drawdown_ratio(self) -> float:
       """Read current drawdown without mutating state."""
       if self.peak_equity <= 0:
           return 0.0
       return (self.peak_equity - self.current_equity) / self.peak_equity
   ```
2. In `paper_trader.py:709`, replace:
   ```python
   # OLD: dd_ratio = float(self._dd_controller.update(0.0).current_drawdown)
   # NEW:
   dd_ratio = self._dd_controller.current_drawdown_ratio
   ```
3. Ensure `_dd_controller.update(actual_daily_pnl)` is called exactly **once per cycle** at the end of the cycle (after all trades are evaluated), with the actual realized PnL for that cycle.
4. Verify this single update point already exists in the cycle orchestration. If not, add it.

**Acceptance:** Running a cycle with 20 candidate symbols calls `update()` exactly once (not 20 times). Drawdown state_history length matches number of trading days, not number of symbols × days.

---

### T2: Wire Portfolio Optimizer Output Into Execution Sizing

**Problem:** `engine.py:1713-1732` computes `optimizer_weight` and adds it to the predictions DataFrame, but `paper_trader.py` never reads this column. The carefully computed mean-variance optimal weights are discarded entirely.

**Files:** `autopilot/engine.py`, `autopilot/paper_trader.py`

**Implementation:**
1. Add config constant `OPTIMIZER_BLEND_WEIGHT = 0.4` (0.0 = pure Kelly, 1.0 = pure optimizer).
2. In `engine.py`, ensure `optimizer_weight` column is included in the data passed to paper_trader for each candidate.
3. In `paper_trader.py:_position_size_pct()`, after computing Kelly-based `base_size`:
   ```python
   optimizer_weight = candidate.get("optimizer_weight", None)
   if optimizer_weight is not None and np.isfinite(optimizer_weight) and optimizer_weight > 0:
       blend = config.OPTIMIZER_BLEND_WEIGHT
       base_size = (1 - blend) * base_size + blend * optimizer_weight
   ```
4. Clip final blended size to `[min_position, max_position]`.
5. Log when blending occurs for observability.

**Acceptance:** When optimizer weights are available, position sizes reflect a blend of Kelly and optimizer output. When optimizer weights are NaN/missing, pure Kelly sizing is used (backward compatible).

---

### T3: Fix A/B Test Kelly Override Not Applied

**Problem:** `paper_trader.py:1031` computes `eff_kelly_fraction` from variant config at lines 1052-1054, but `_position_size_pct()` at line 707 always uses `self.kelly_fraction` (the base value), never the variant-specific override.

**File:** `autopilot/paper_trader.py`

**Implementation:**
1. Store the effective Kelly fraction as an instance variable before the sizing loop:
   ```python
   # Before the sizing loop in the A/B test path:
   original_kelly = self.kelly_fraction
   self.kelly_fraction = eff_kelly_fraction
   ```
2. After the sizing loop completes, restore:
   ```python
   self.kelly_fraction = original_kelly
   ```
3. Alternatively (cleaner): pass `kelly_fraction` as a parameter to `_position_size_pct()` instead of reading from `self`:
   ```python
   def _position_size_pct(self, ..., kelly_fraction: Optional[float] = None):
       kf = kelly_fraction if kelly_fraction is not None else self.kelly_fraction
   ```
   Then call with `self._position_size_pct(..., kelly_fraction=eff_kelly_fraction)`.

**Acceptance:** An A/B test variant with `kelly_fraction=0.25` produces different position sizes than the control with `kelly_fraction=0.50`. Test by running both variants on the same signal set and verifying size differences.

---

### T4: Fix Meta-Label Cross-Asset Leakage

**Problem:** `engine.py:1125-1138` concatenates all assets' signals/returns/regimes into one global series, then passes to `meta_labeler.py:122-136` which computes rolling features (signal autocorrelation, volatility, streaks) on the combined series. Boundary rows between assets use adjacent-asset history, contaminating the training signal.

**Files:** `autopilot/engine.py`, `autopilot/meta_labeler.py`

**Implementation:**
1. **Option A (recommended): Per-asset meta-labeler training.** Replace the global concatenation with a per-asset loop:
   ```python
   for permno in permnos:
       asset_signals = signals_for_permno[permno]
       asset_returns = returns_for_permno[permno]
       asset_features = self.meta_labeler.build_meta_features(asset_signals, asset_returns, ...)
       self.meta_labeler.train(asset_features, asset_labels)
   ```
   Store a single meta-labeler model trained on the union of per-asset features (but built per-asset so rolling windows don't cross boundaries).

2. **Option B (simpler): NaN row separators.** Insert a row of NaN between each asset's data in the concatenated frame:
   ```python
   all_signals.append(pd.Series([np.nan], index=["__separator__"]))
   ```
   Then in `build_meta_features()`, ensure `min_periods` on all rolling windows equals the window size, so NaN separators cause rolling features to return NaN at boundaries. Drop NaN rows before training.

3. Choose Option A unless it causes significant runtime overhead. Document the choice.

**Acceptance:** Meta-labeler features for asset A's first row do not depend on asset B's last row. Rolling autocorrelation at asset boundaries returns NaN or is computed only within-asset.

---

## Verification

- [ ] Run `pytest tests/test_system_innovations.py -v` — all pass
- [ ] Run `pytest tests/ -k "autopilot or paper_trader or meta_label"` — all pass
- [ ] Verify drawdown update count matches cycle count, not candidate count
- [ ] Verify A/B Kelly override produces different sizes
- [ ] Verify optimizer weights affect position sizing when available
- [ ] Verify meta-label features at asset boundaries are NaN or within-asset only
