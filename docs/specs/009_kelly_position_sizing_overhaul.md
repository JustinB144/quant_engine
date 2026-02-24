# Feature Spec: Kelly Criterion & Position Sizing Overhaul

> **Status:** Approved
> **Author:** justin
> **Date:** 2026-02-23
> **Estimated effort:** ~12 hours across 8 tasks

---

## Why

The position sizing system has several formula-level bugs and architectural gaps: (1) The Kelly formula `f* = (p*b - q) / b` doesn't handle negative edge correctly — when `p*b < q`, it returns `min_position` instead of 0, meaning the system still takes positions on negative-edge signals. (2) The drawdown governor uses a linear ramp `1 - |dd|/|max_dd|` which reduces too slowly at first and too aggressively near the limit — an exponential or convex curve would be more appropriate. (3) The Bayesian win-rate update is global (not per-regime), so bull-market wins dilute bear-market priors. (4) Regime stats in the PositionSizer are hardcoded defaults that are never updated from actual trade history. (5) The confidence scalar formula `0.5 + confidence` allows a 1.5x multiplier at confidence=1.0, which can amplify oversized positions. (6) The portfolio-level Kelly (correlations between positions) is not implemented — each position is sized independently.

## What

Fix the Kelly formula, replace the linear drawdown governor with a convex curve, make Bayesian updates per-regime, wire regime stats to actual trade history, cap the confidence scalar, and add correlation-aware portfolio-level position sizing. Done means: position sizes are mathematically correct, regime-adaptive, and respect portfolio-level risk constraints.

## Constraints

### Must-haves
- Kelly returns 0 (not min_position) for negative-edge signals
- Drawdown governor uses convex scaling (exponential or quadratic)
- Bayesian win-rate tracked per-regime with separate Beta priors
- Regime stats updated from actual trade history after each backtest
- Confidence scalar capped at 1.0 (not 1.5)
- Portfolio-level position sizing considers inter-position correlations

### Must-nots
- Do NOT change the blend_weights (kelly=0.3, vol_scaled=0.4, atr_based=0.3) without backtesting
- Do NOT remove the half-Kelly default (it's a valid conservative choice)
- Do NOT make position sizing slower than 10ms per position

## Current State

### Key files
| File | Role | Issues |
|------|------|--------|
| `risk/position_sizer.py` | Core sizing with Kelly + vol + ATR blend | Kelly formula bug, linear drawdown governor, hardcoded regime stats |
| `autopilot/paper_trader.py` | Live paper trading with Kelly sizing | Doesn't pass drawdown state to sizer, doesn't update regime stats |
| `risk/drawdown.py` | DrawdownController with state machine | Linear recovery ramp, no hysteresis, not wired to paper trader |
| `risk/portfolio_optimizer.py` | Mean-variance optimizer | No portfolio-level Kelly, no cash allocation |

## Tasks

### T1: Fix Kelly formula for negative edge

**What:** Return 0.0 (not min_position) when the edge is negative, and add a small-sample penalty.

**Files:**
- `risk/position_sizer.py` — fix `_kelly()` method (lines 190-208)

**Implementation notes:**
- Current bug (line 202): `if kelly <= 0: return self.min_position_pct` — this means negative-edge signals still get min_position allocation
- Fix:
  ```python
  def _kelly(self, win_rate, avg_win, avg_loss):
      if avg_loss >= -1e-9 or win_rate <= 0.0 or win_rate >= 1.0:
          return 0.0  # Not min_position — refuse to size
      
      b = abs(avg_win / avg_loss)
      p, q = win_rate, 1.0 - win_rate
      kelly = (p * b - q) / b
      
      if kelly <= 0:
          return 0.0  # Negative edge — do not trade
      
      # Small-sample penalty: reduce Kelly when n_trades is low
      # Shrink toward 0 as n_trades → 0
      n_trades = getattr(self, '_current_n_trades', 100)
      sample_penalty = min(1.0, n_trades / 50.0)  # Full Kelly only after 50 trades
      kelly *= sample_penalty
      
      return min(kelly, self.max_position_pct)
  ```
- Pass `n_trades` context through `size_position()` method

**Verify:**
```bash
python -c "
from quant_engine.risk.position_sizer import PositionSizer
ps = PositionSizer()
# Negative edge: win_rate=0.3, avg_win=0.02, avg_loss=-0.03 → edge < 0
size = ps._kelly(0.3, 0.02, -0.03)
assert size == 0.0, f'Should be 0 for negative edge, got {size}'
# Positive edge: should return > 0
size = ps._kelly(0.55, 0.03, -0.02)
assert size > 0.0, f'Should be > 0 for positive edge, got {size}'
print('Kelly formula fixed')
"
```

---

### T2: Replace linear drawdown governor with convex curve

**What:** Use exponential decay instead of linear ramp so sizing reduces slowly at first and aggressively near the limit.

**Files:**
- `risk/position_sizer.py` — fix `_apply_drawdown_governor()` (lines 250-266)

**Implementation notes:**
- Current (linear): `scale = 1.0 - (|dd| / |max_dd|)` — at 50% of max_dd, sizing is at 50%
- Target (exponential): `scale = exp(-k * (|dd| / |max_dd|))` where k controls steepness
  - At 50% of max_dd: sizing ≈ 72% (k=2) vs 50% (linear) — less aggressive early
  - At 90% of max_dd: sizing ≈ 16% (k=2) vs 10% (linear) — more aggressive late
- Implementation:
  ```python
  import math
  
  def _apply_drawdown_governor(self, kelly_fraction, current_drawdown, max_allowed_dd=-0.20):
      if current_drawdown >= 0:
          return kelly_fraction
      
      dd_ratio = abs(current_drawdown) / abs(max_allowed_dd)
      
      if dd_ratio >= 1.0:
          return 0.0
      
      # Convex scaling: slow reduction early, aggressive near limit
      k = 3.0  # Steepness parameter
      scale = math.exp(-k * dd_ratio)
      
      return kelly_fraction * scale
  ```

**Verify:**
```bash
python -c "
from quant_engine.risk.position_sizer import PositionSizer
ps = PositionSizer()
# At 50% of max drawdown, should still have >50% sizing
frac = ps._apply_drawdown_governor(0.5, -0.10, -0.20)
assert frac > 0.25, f'At 50% DD, frac should be >0.25, got {frac}'
# At 90% of max drawdown, should be very small
frac = ps._apply_drawdown_governor(0.5, -0.18, -0.20)
assert frac < 0.10, f'At 90% DD, frac should be <0.10, got {frac}'
print('Convex drawdown governor working')
"
```

---

### T3: Implement per-regime Bayesian win-rate tracking

**What:** Maintain separate Beta priors for each regime so bull-market wins don't inflate bear-market priors.

**Files:**
- `risk/position_sizer.py` — modify `update_kelly_bayesian()` and `get_bayesian_kelly()` (lines 289-312)

**Implementation notes:**
- Current: single global `_bayesian_wins` and `_bayesian_losses` counters
- Fix: dict of per-regime counters
  ```python
  def __init__(self, ...):
      # Per-regime Bayesian priors: {regime_id: (alpha, beta)}
      self._bayesian_regime = {
          0: {'wins': 0, 'losses': 0},  # trending_bull
          1: {'wins': 0, 'losses': 0},  # trending_bear
          2: {'wins': 0, 'losses': 0},  # mean_reverting
          3: {'wins': 0, 'losses': 0},  # high_volatility
      }
      # Global fallback (still useful for cold-start)
      self._bayesian_wins = 0
      self._bayesian_losses = 0
  
  def update_kelly_bayesian(self, trade_df, regime_col='regime'):
      """Update per-regime Bayesian counters from trade history."""
      for regime_id in self._bayesian_regime:
          regime_trades = trade_df[trade_df[regime_col] == regime_id]
          if len(regime_trades) > 0:
              wins = (regime_trades['net_return'] > 0).sum()
              losses = (regime_trades['net_return'] <= 0).sum()
              self._bayesian_regime[regime_id]['wins'] += wins
              self._bayesian_regime[regime_id]['losses'] += losses
      # Also update global
      total_wins = (trade_df['net_return'] > 0).sum()
      total_losses = (trade_df['net_return'] <= 0).sum()
      self._bayesian_wins += total_wins
      self._bayesian_losses += total_losses
  
  def get_bayesian_kelly(self, avg_win, avg_loss, regime=None):
      """Get Kelly fraction using regime-specific Bayesian posterior."""
      if regime is not None and regime in self._bayesian_regime:
          stats = self._bayesian_regime[regime]
          wins, losses = stats['wins'], stats['losses']
          if wins + losses >= 10:  # Enough regime-specific data
              posterior_wr = (self.bayesian_alpha + wins) / (
                  self.bayesian_alpha + self.bayesian_beta + wins + losses
              )
              return self._kelly(posterior_wr, avg_win, avg_loss) * self.kelly_fraction
      
      # Fallback to global posterior
      posterior_wr = (self.bayesian_alpha + self._bayesian_wins) / (
          self.bayesian_alpha + self.bayesian_beta + self._bayesian_wins + self._bayesian_losses
      )
      return self._kelly(posterior_wr, avg_win, avg_loss) * self.kelly_fraction
  ```

**Verify:**
```bash
python -c "
import pandas as pd
from quant_engine.risk.position_sizer import PositionSizer
ps = PositionSizer()
# Simulate regime-specific trades
trades = pd.DataFrame({
    'net_return': [0.02, 0.03, -0.01, 0.01, -0.02, -0.03, -0.01, -0.02, -0.01, -0.02,
                   0.01, 0.02, 0.03, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01],
    'regime': [0]*5 + [3]*5 + [0]*10  # Bull regime has more wins, HV has more losses
})
ps.update_kelly_bayesian(trades)
bull_kelly = ps.get_bayesian_kelly(0.02, -0.02, regime=0)
hv_kelly = ps.get_bayesian_kelly(0.02, -0.02, regime=3)
print(f'Bull Kelly: {bull_kelly:.4f}, HV Kelly: {hv_kelly:.4f}')
assert bull_kelly > hv_kelly, 'Bull regime should have higher Kelly than HV'
print('Per-regime Bayesian working')
"
```

---

### T4: Wire regime stats to actual trade history

**What:** Replace hardcoded regime_stats defaults with live trade history updates.

**Files:**
- `risk/position_sizer.py` — fix `update_regime_stats()` (lines 268-287) and constructor defaults
- `autopilot/paper_trader.py` — call `update_regime_stats()` after each cycle

**Implementation notes:**
- Current issue: `regime_stats` defaults (line 87-92) are hardcoded and `update_regime_stats()` is never called from paper_trader
- Fix in paper_trader.py `run_cycle()`:
  ```python
  # After processing exits, update regime stats on the sizer
  if hasattr(self, '_position_sizer') and len(state['trades']) >= 20:
      trade_df = pd.DataFrame(state['trades'][-self.kelly_lookback_trades:])
      if 'regime' in trade_df.columns and 'net_return' not in trade_df.columns:
          trade_df['net_return'] = (trade_df['exit_price'] - trade_df['entry_price']) / trade_df['entry_price']
      self._position_sizer.update_regime_stats(trade_df)
      self._position_sizer.update_kelly_bayesian(trade_df)
  ```
- Ensure `regime` is stored in paper_trader trade records (currently not stored)

**Verify:**
```bash
python -c "
from quant_engine.risk.position_sizer import PositionSizer
import pandas as pd
ps = PositionSizer()
# Check defaults exist
assert 'trending_bull' in ps.regime_stats
# Update with fake trades
trades = pd.DataFrame({
    'net_return': [0.03, 0.02, -0.01, 0.04, 0.01, 0.02, -0.02, 0.03, 0.01, 0.02],
    'regime': [0]*10
})
ps.update_regime_stats(trades)
assert ps.regime_stats['trending_bull']['n_trades'] == 10
print('Regime stats updated from trades')
"
```

---

### T5: Cap confidence scalar and fix multiplier range

**What:** Change confidence scalar from `0.5 + confidence` (range 0.5-1.5) to a capped range of 0.5-1.0.

**Files:**
- `risk/position_sizer.py` — fix confidence scalar in `size_position()` (around line 145)

**Implementation notes:**
- Current (line ~145): `confidence_scalar = 0.5 + confidence` — allows 1.5x at confidence=1.0
- This means a max-confidence signal gets 50% MORE than the blend calculation, which can overshoot risk limits
- Fix:
  ```python
  # Confidence scales position DOWN from blend, never UP
  # Range: [0.5, 1.0] — full confidence = 100% of blend, low confidence = 50%
  confidence_scalar = 0.5 + 0.5 * min(1.0, max(0.0, confidence))
  ```
- This ensures confidence only reduces positions, never amplifies beyond the calculated blend

**Verify:**
```bash
python -c "
# Old: 0.5 + 1.0 = 1.5 (amplifies)
# New: 0.5 + 0.5*1.0 = 1.0 (neutral at max confidence)
old_scalar = 0.5 + 1.0  # confidence=1.0
new_scalar = 0.5 + 0.5 * min(1.0, max(0.0, 1.0))
print(f'Old scalar at conf=1.0: {old_scalar} (amplifies by 50%!)')
print(f'New scalar at conf=1.0: {new_scalar} (neutral)')
print(f'New scalar at conf=0.5: {0.5 + 0.5*0.5} (reduces by 25%)')
print(f'New scalar at conf=0.0: {0.5 + 0.5*0.0} (reduces by 50%)')
"
```

---

### T6: Add portfolio-level position sizing with correlation adjustment

**What:** Adjust individual position sizes based on correlation with existing portfolio positions.

**Files:**
- `risk/position_sizer.py` — add `size_portfolio_aware()` method
- `risk/covariance.py` — add helper for marginal risk contribution

**Implementation notes:**
- New method that wraps `size_position()`:
  ```python
  def size_portfolio_aware(self, ticker, win_rate, avg_win, avg_loss, realized_vol, atr, 
                           price, existing_positions, returns_data, **kwargs):
      """Size position considering correlation with existing portfolio."""
      base_size = self.size_position(ticker, win_rate, avg_win, avg_loss, realized_vol, 
                                      atr, price, **kwargs).final_size
      
      if not existing_positions or returns_data is None:
          return base_size
      
      # Compute average correlation with existing positions
      ticker_returns = returns_data.get(ticker)
      if ticker_returns is None:
          return base_size
      
      correlations = []
      for pos_ticker, pos_weight in existing_positions.items():
          pos_returns = returns_data.get(pos_ticker)
          if pos_returns is not None:
              corr = ticker_returns.corr(pos_returns)
              if not pd.isna(corr):
                  correlations.append(corr * pos_weight)
      
      if not correlations:
          return base_size
      
      # Weighted average correlation with portfolio
      avg_corr = sum(correlations) / sum(abs(w) for _, w in existing_positions.items())
      
      # Reduce size when highly correlated with existing positions
      # corr=0 → full size, corr=1 → 50% size
      correlation_penalty = 1.0 - 0.5 * max(0, avg_corr)
      
      return base_size * correlation_penalty
  ```

**Verify:**
```bash
python -c "
print('Portfolio-aware sizing: high-corr positions get reduced allocation')
print('corr=0.0 → 100% of base size')
print('corr=0.5 → 75% of base size')
print('corr=0.9 → 55% of base size')
for corr in [0.0, 0.3, 0.5, 0.7, 0.9]:
    penalty = 1.0 - 0.5 * max(0, corr)
    print(f'  corr={corr} → {penalty:.0%} of base size')
"
```

---

### T7: Wire drawdown controller into paper trader

**What:** Integrate DrawdownController into PaperTrader so position sizes respect drawdown state.

**Files:**
- `autopilot/paper_trader.py` — add DrawdownController integration in `run_cycle()`

**Implementation notes:**
- Current gap: PaperTrader has no drawdown tracking — it only checks signal strength for exits
- Fix: Initialize DrawdownController in constructor, call `update()` each cycle
  ```python
  # In __init__:
  from quant_engine.risk.drawdown import DrawdownController
  self.dd_controller = DrawdownController(
      initial_equity=initial_capital,
  )
  
  # In run_cycle(), after computing daily PnL:
  daily_pnl = (current_equity - prev_equity) / prev_equity
  dd_status = self.dd_controller.update(daily_pnl)
  
  # Apply drawdown multiplier to all new position sizes:
  position_size_pct *= dd_status.size_multiplier
  
  # Block new entries if drawdown controller says no:
  if not dd_status.allow_new_entries:
      candidates = []  # Skip all entries this cycle
  
  # Force liquidate if critical:
  if dd_status.force_liquidate:
      # Exit all positions immediately
      for pos in state['positions']:
          # ... execute exit ...
  ```

**Verify:**
```bash
python -c "
from quant_engine.risk.drawdown import DrawdownController
dc = DrawdownController()
# Simulate 5% daily loss
status = dc.update(-0.05)
print(f'After -5% day: state={status.state.name}, size_mult={status.size_multiplier}, allow_new={status.allow_new_entries}')
assert status.size_multiplier < 1.0, 'Should reduce sizing after large loss'
print('Drawdown controller integration verified')
"
```

---

### T8: Test all position sizing fixes

**What:** Comprehensive tests for Kelly formula, drawdown governor, Bayesian updates, and portfolio-aware sizing.

**Files:**
- `tests/test_position_sizing_overhaul.py` — new test file

**Implementation notes:**
- Test cases:
  1. `test_kelly_negative_edge_returns_zero` — Kelly with negative edge → 0.0
  2. `test_kelly_positive_edge_returns_positive` — Kelly with positive edge → >0
  3. `test_kelly_small_sample_penalty` — Low n_trades → reduced Kelly
  4. `test_drawdown_governor_convex` — At 50% DD, sizing > 50% of full
  5. `test_drawdown_governor_zero_at_max` — At 100% DD, sizing = 0
  6. `test_bayesian_per_regime` — Bull regime posterior > bear regime posterior
  7. `test_confidence_scalar_capped` — Max confidence → scalar = 1.0 (not 1.5)
  8. `test_portfolio_correlation_penalty` — High-corr position gets reduced size
  9. `test_regime_stats_update_from_trades` — Stats update after trade DF passed
  10. `test_drawdown_controller_blocks_entries` — Critical drawdown → no new entries

**Verify:**
```bash
python -m pytest tests/test_position_sizing_overhaul.py -v
```

---

## Validation

### Acceptance criteria
1. Kelly returns 0.0 for negative-edge signals (not min_position)
2. Drawdown governor curve is convex (more lenient early, aggressive late)
3. Bayesian win-rate is tracked per-regime with separate posteriors
4. Confidence scalar range is [0.5, 1.0], not [0.5, 1.5]
5. Portfolio-level sizing reduces allocation for correlated positions
6. DrawdownController is wired into paper trader with entry blocking and forced liquidation

### Rollback plan
- Kelly formula fix: revert to old `return min_position` (one-line change)
- Drawdown governor: change `exp(-k*x)` back to `1-x`
- Bayesian: fall back to global counters
- Confidence scalar: revert formula
