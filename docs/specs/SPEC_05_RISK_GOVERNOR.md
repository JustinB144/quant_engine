# Feature Spec: Risk Governor + Kelly Unification + Uncertainty-Aware Sizing

> **Status:** Draft
> **Author:** Claude Opus 4
> **Date:** 2026-02-26
> **Estimated effort:** 110 hours across 8 tasks

---

## Why

The codebase already implements regime-conditional Kelly sizing, Bayesian updating, drawdown governance, and a composite blend in `position_sizer.py`. However, three significant gaps remain:

1. **Dual sizing pathways**: The paper trader has its own `_position_size_pct()` method that duplicates logic from PositionSizer but doesn't call it. This creates maintenance burden and inconsistency.
2. **Missing uncertainty-aware inputs**: The sizing model doesn't incorporate structured uncertainty (signal uncertainty, regime entropy, drift likelihood), only historical volatility and drawdown state.
3. **Missing budget constraints**: No shock budget to reserve capital for tail events, no turnover budget enforcement to limit rebalancing costs, and no explicit portfolio concentration limits that interact with Kelly sizing.

Additionally, the composite blend weights (Kelly=0.3, vol=0.4, ATR=0.3) are hardcoded rather than configurable and regime-conditional. The Kelly Bayesian prior uses Beta(2,2) which may overregularize with small samples.

This spec unifies sizing, adds uncertainty-aware features, and implements budget constraints while preserving the existing Bayesian Kelly and drawdown governance.

---

## What

### Part 1: Unified Position Sizing Architecture
Consolidate paper trader's `_position_size_pct()` and PositionSizer into a single unified interface. Paper trader calls PositionSizer for all sizing decisions, eliminating code duplication.

### Part 2: Uncertainty-Aware Sizing
Extend PositionSizer to accept structured uncertainty inputs: signal_uncertainty (Bayesian posterior std), regime_entropy (measure of regime state confidence), drift_score (0-1 indicating price trend strength). Use these to scale position size downward when uncertainty is high.

### Part 3: Shock Budget + Turnover Budget + Concentration Limits
Implement three new budget constraints:
- **Shock budget**: Reserve X% of capital for tail events (e.g., hold 5% cash, never size positions >95% notional)
- **Turnover budget**: Enforce MAX_ANNUALIZED_TURNOVER by computing forward-looking turnover and reducing position sizes if budget would be exceeded
- **Concentration limit**: Hard cap on single-position notional as % of portfolio, with Kelly sizing respected but never exceeding cap

### Part 4: Configurable Composite Blend Weights
Make the blend weights (Kelly fraction, volatility scaling, ATR-based) configurable per regime. Allow both static weights and regime-conditional weights.

---

## Constraints

### Must-haves
- PositionSizer unified interface used by both autopilot engine and paper trader
- Uncertainty inputs (signal_uncertainty, regime_entropy, drift_score) integrated into size_position
- Shock budget enforced: reserve SHOCK_BUDGET_PCT of capital, never exceed (1 - SHOCK_BUDGET_PCT) notional
- Turnover budget enforced: track portfolio turnover, scale down sizes if forward turnover exceeds MAX_ANNUALIZED_TURNOVER
- Concentration limit enforced: single position <= CONCENTRATION_LIMIT_PCT notional
- Blend weights configurable per regime (BLEND_WEIGHTS_BY_REGIME dict in config)
- Kelly Beta prior parameterized in config (default Beta(2,2), configurable)
- Backward compatibility: existing paper trader functionality unchanged when uncertainty features disabled
- All budgets tracked and logged per cycle

### Must-nots
- Do not change existing drawdown governor (ExpCurve shape, recovery ramp quadratic) - keep as-is
- Do not remove Bayesian Kelly updating; keep Beta-Binomial model intact
- Do not break existing portfolio optimizer integration in PositionSizer.size_portfolio_aware
- Do not apply uncertainty scaling to turnover budget (only to position size)
- Do not use uncertainty inputs for risk limits (use only for position sizing adjustment)

### Out of scope
- Dynamic rebalancing of blend weights based on recent performance
- Multi-period shock budget allocation (static per configuration)
- Portfolio-level Kelly sizing (only single-position Kelly, then constrained by budgets)
- Real-time turnover forecast (use static forecast model)

---

## Current State

### Key files
- `/quant_engine/position_sizer.py` (565 lines): Has size_position(), size_portfolio_aware(), size_portfolio(). Already has regime-conditional Kelly, Bayesian updating, drawdown governor, composite blend (hardcoded 0.3/0.4/0.3).
- `/quant_engine/paper_trader.py` (817 lines): Has _position_size_pct() which computes Kelly with regime multiplier, ATR stops, drawdown checks. Duplicates sizing logic.
- `/quant_engine/drawdown_controller.py` (261 lines): 5 states with quadratic recovery ramp. Keep unchanged.
- `/quant_engine/config.py`: MAX_ANNUALIZED_TURNOVER=500, TRANSACTION_COST_BPS=20. Missing SHOCK_BUDGET_PCT, CONCENTRATION_LIMIT_PCT, BLEND_WEIGHTS_BY_REGIME, Kelly prior params.

### Existing patterns to follow
1. PositionSizer.size_position(signal, volatility, regime_state, recent_performance) pattern - extend to include uncertainty params
2. Regime state constants: NORMAL, WARNING, CAUTION, CRITICAL, RECOVERY (from DrawdownController)
3. Kelly sizing uses win_rate, profit_factor from recent trade history
4. Composite blend in PositionSizer already blends Kelly, vol_scaled, atr_based sizes
5. Portfolio sizing uses mean-variance optimization with turnover penalty
6. Configuration inheritance: all params in config.py, engine passes to PositionSizer constructor

### Configuration
```python
# Current config.py (relevant sections)
MAX_ANNUALIZED_TURNOVER = 500
TRANSACTION_COST_BPS = 20
EXEC_MAX_PARTICIPATION = 0.02

# Existing position sizer params (implied)
KELLY_FRACTION = 1.0  # Not explicitly in current code
DRAWDOWN_GOVERNOR_ENABLED = True

# New config entries needed:
# Shock budget
SHOCK_BUDGET_PCT = 0.05  # Reserve 5% of capital for tail events

# Concentration limit
CONCENTRATION_LIMIT_PCT = 0.20  # Max single position is 20% of notional

# Turnover budget
TURNOVER_BUDGET_ENFORCEMENT = True
TURNOVER_BUDGET_LOOKBACK_DAYS = 252  # Use annualized turnover

# Blend weights - static (regime-insensitive)
BLEND_WEIGHTS_STATIC = {
    'kelly': 0.30,
    'vol_scaled': 0.40,
    'atr_based': 0.30
}

# Blend weights - regime-conditional
BLEND_WEIGHTS_BY_REGIME = {
    'NORMAL': {'kelly': 0.35, 'vol_scaled': 0.35, 'atr_based': 0.30},
    'WARNING': {'kelly': 0.25, 'vol_scaled': 0.45, 'atr_based': 0.30},
    'CAUTION': {'kelly': 0.15, 'vol_scaled': 0.50, 'atr_based': 0.35},
    'CRITICAL': {'kelly': 0.05, 'vol_scaled': 0.50, 'atr_based': 0.45},
    'RECOVERY': {'kelly': 0.20, 'vol_scaled': 0.40, 'atr_based': 0.40}
}

# Kelly Bayesian prior
KELLY_BAYESIAN_PRIOR_ALPHA = 2  # Beta(alpha, beta) prior
KELLY_BAYESIAN_PRIOR_BETA = 2
KELLY_MIN_SAMPLES_FOR_UPDATE = 10

# Uncertainty scaling
UNCERTAINTY_SCALING_ENABLED = True
UNCERTAINTY_SIGNAL_WEIGHT = 0.40  # Weight of signal_uncertainty in composite
UNCERTAINTY_REGIME_WEIGHT = 0.30  # Weight of regime_entropy
UNCERTAINTY_DRIFT_WEIGHT = 0.30   # Weight of drift_score (inverted: high drift = high confidence)
UNCERTAINTY_REDUCTION_FACTOR = 0.30  # Max reduction from base size due to uncertainty (30% reduction max)
```

---

## Tasks

### T1: Extract Paper Trader Sizing Logic into PositionSizer Interface

**What:** Create comprehensive unified interface in PositionSizer that encapsulates all sizing logic currently in paper_trader._position_size_pct(). This interface should accept paper trader inputs (kelly_history, atr, regime_state, dd_ratio) and produce position size. Update paper trader to call this interface instead of doing sizing inline.

**Files:**
- `/quant_engine/position_sizer.py` (add `size_position_paper_trader()` method, ~50 lines)
- `/quant_engine/paper_trader.py` (replace _position_size_pct() body with call to PositionSizer, ~30 lines)

**Implementation notes:**
- Add new method to PositionSizer class:
  ```python
  def size_position_paper_trader(
      self,
      kelly_history: List[float],  # Recent P&L win rates
      atr: float,  # Current ATR
      regime_state: str,  # NORMAL, WARNING, CAUTION, CRITICAL, RECOVERY
      dd_ratio: float,  # Current drawdown as % of peak equity
      signal_magnitude: float = None,  # Optional z-score of signal
      position_notional_before: float = 0.0  # Current position size for turnover calc
  ) -> float:
      """Compute position size as % of account equity for paper trader usage."""
      # This method wraps existing size_position() and _apply_*() methods
      # Returns % of equity to allocate to this position
  ```
- Method should internally:
  1. Compute win_rate from kelly_history (with minimum sample size check)
  2. Call _compute_kelly_fraction(win_rate) to get Kelly estimate
  3. Compute vol_scaled size based on ATR and asset volatility
  4. Compute atr_based size based on ATR ratio to historical mean
  5. Blend three sizes using regime-conditional weights
  6. Apply drawdown governor (_apply_drawdown_governor)
  7. Apply turnover constraint
  8. Apply concentration limit
  9. Return final size
- Ensure this method uses same logic as existing size_position() to avoid divergence
- Add logging: input parameters and output size for debugging

**Verify:**
- Unit test: size_position_paper_trader() produces reasonable output for various inputs
- Regression test: paper trader sizing output unchanged before/after refactoring
- Integration test: paper trader calls new interface without exceptions

---

### T2: Implement Uncertainty-Aware Sizing Inputs

**What:** Extend PositionSizer.size_position() to accept three new optional parameters: signal_uncertainty (0-1 confidence of primary signal), regime_entropy (0-1 measure of regime state uncertainty), drift_score (0-1 trend strength). Use these to scale position size downward when uncertainty is high.

**Files:**
- `/quant_engine/position_sizer.py` (modify `size_position()` signature and add `_compute_uncertainty_scale()` method, ~60 lines)
- `/quant_engine/config.py` (add UNCERTAINTY_* parameters)
- `/quant_engine/autopilot/engine.py` (pass uncertainty inputs to position sizer, ~10 lines)

**Implementation notes:**
- Modify size_position() signature:
  ```python
  def size_position(
      self,
      signal: float,
      volatility: float,
      regime_state: str = 'NORMAL',
      recent_performance: dict = None,
      signal_uncertainty: float = None,  # 0-1, 0=low conf, 1=high conf
      regime_entropy: float = None,       # 0-1, 0=low entropy (certain), 1=high entropy
      drift_score: float = None           # 0-1, 0=no trend, 1=strong trend
  ) -> float:
  ```
- Add new method `_compute_uncertainty_scale()`:
  ```python
  def _compute_uncertainty_scale(
      self,
      signal_uncertainty: float,
      regime_entropy: float,
      drift_score: float
  ) -> float:
      """Compute scaling factor [0, 1] based on uncertainty components.

      Returns 1.0 when all uncertainty inputs are high confidence.
      Returns lower value when any uncertainty is high.
      """
      if signal_uncertainty is None or regime_entropy is None or drift_score is None:
          return 1.0  # No uncertainty adjustment if inputs missing

      # Invert inputs to get confidence metrics
      signal_confidence = 1.0 - signal_uncertainty
      regime_confidence = 1.0 - regime_entropy
      drift_confidence = drift_score  # High drift = high confidence in position direction

      # Weighted composite confidence
      composite_confidence = (
          self.config['UNCERTAINTY_SIGNAL_WEIGHT'] * signal_confidence +
          self.config['UNCERTAINTY_REGIME_WEIGHT'] * regime_confidence +
          self.config['UNCERTAINTY_DRIFT_WEIGHT'] * drift_confidence
      )

      # Map confidence [0, 1] to size scale [1 - max_reduction, 1.0]
      max_reduction = self.config['UNCERTAINTY_REDUCTION_FACTOR']
      scale = 1.0 - (max_reduction * (1.0 - composite_confidence))

      return np.clip(scale, 1.0 - max_reduction, 1.0)
  ```
- In size_position(), after computing base size (blend of Kelly, vol, ATR), apply uncertainty scale:
  ```python
  base_size = self._blend_sizes(kelly_size, vol_size, atr_size, regime_state)
  uncertainty_scale = self._compute_uncertainty_scale(signal_uncertainty, regime_entropy, drift_score)
  adjusted_size = base_size * uncertainty_scale
  ```
- Add detailed logging: input uncertainties, composite confidence, scale factor, size before/after adjustment
- If any uncertainty input is NaN or out of range [0, 1], log warning and treat as missing (scale = 1.0)

**Verify:**
- Unit test: _compute_uncertainty_scale() produces [1 - max_reduction, 1.0] output
- Unit test: size with high uncertainty < size with low uncertainty (same other params)
- Integration test: size_position() with uncertainty inputs produces smaller sizes as expected
- Regression test: size_position() without uncertainty inputs unchanged (backward compatible)

---

### T3: Implement Shock Budget Constraint

**What:** Add shock budget constraint to PositionSizer that reserves SHOCK_BUDGET_PCT of capital for tail events. No position can exceed (1 - SHOCK_BUDGET_PCT) of portfolio notional. Enforce in size_position() after other adjustments.

**Files:**
- `/quant_engine/position_sizer.py` (add `_apply_shock_budget()` method and integrate into size_position(), ~40 lines)
- `/quant_engine/config.py` (add SHOCK_BUDGET_PCT=0.05)

**Implementation notes:**
- Add method:
  ```python
  def _apply_shock_budget(self, position_size: float, portfolio_equity: float) -> float:
      """Ensure position doesn't exceed (1 - shock_budget_pct) of portfolio."""
      max_position_notional = portfolio_equity * (1.0 - self.config['SHOCK_BUDGET_PCT'])
      position_notional = position_size * portfolio_equity

      if position_notional > max_position_notional:
          reduced_size = max_position_notional / portfolio_equity
          self.logger.warning(
              f"Position size {position_size:.4f} exceeds shock budget. "
              f"Reduced to {reduced_size:.4f}."
          )
          return reduced_size

      return position_size
  ```
- In size_position(), call after all other adjustments:
  ```python
  # ... existing sizing logic ...
  size = self._apply_drawdown_governor(size, regime_state)
  size = self._apply_shock_budget(size, portfolio_equity)
  size = self._apply_concentration_limit(size, portfolio_equity, symbol)
  return size
  ```
- Log total notional allocated vs. reserved shock budget
- Shock budget is portfolio-level, not per-position: total positions should not exceed max_position_notional

**Verify:**
- Unit test: position size capped by shock budget
- Unit test: with SHOCK_BUDGET_PCT=0.10, max position size = 90% of equity
- Integration test: portfolio respects shock budget across multiple positions

---

### T4: Implement Turnover Budget Constraint

**What:** Add turnover budget enforcement to PositionSizer. Track portfolio turnover (sum of abs changes in position weights), compute forward-looking turnover for proposed new positions, and scale down sizes if total turnover would exceed MAX_ANNUALIZED_TURNOVER.

**Files:**
- `/quant_engine/position_sizer.py` (add `_apply_turnover_budget()` method, ~70 lines)
- `/quant_engine/paper_trader.py` or new `/quant_engine/turnover_tracker.py` (track portfolio weights history)
- `/quant_engine/config.py` (verify TURNOVER_BUDGET_ENFORCEMENT, MAX_ANNUALIZED_TURNOVER=500)

**Implementation notes:**
- Add method to PositionSizer:
  ```python
  def _apply_turnover_budget(
      self,
      proposed_size: float,
      symbol: str,
      portfolio_equity: float,
      current_positions: dict,  # {symbol: notional} dict
      dates_in_period: int  # Trading days in lookback window
  ) -> float:
      """Scale down position size if forward turnover would exceed budget."""

      if not self.config.get('TURNOVER_BUDGET_ENFORCEMENT', False):
          return proposed_size

      # Compute current turnover (annualized)
      total_turnover_ytd = sum(
          abs(pos_history.get('turnover_ytd', 0))
          for pos_history in self.position_history.values()
      )
      annualized_turnover = (total_turnover_ytd / dates_in_period) * 252

      # Compute forward turnover from this position change
      current_notional = current_positions.get(symbol, 0)
      proposed_notional = proposed_size * portfolio_equity
      position_turnover = abs(proposed_notional - current_notional) / portfolio_equity

      # Check remaining budget
      remaining_budget = self.config['MAX_ANNUALIZED_TURNOVER'] - annualized_turnover

      if position_turnover > remaining_budget:
          # Scale down position to fit in remaining budget
          if remaining_budget > 0:
              scale_factor = remaining_budget / position_turnover
              reduced_size = proposed_size * scale_factor
              self.logger.warning(
                  f"Turnover budget exceeded for {symbol}. "
                  f"Scaled from {proposed_size:.4f} to {reduced_size:.4f}."
              )
              return reduced_size
          else:
              self.logger.warning(f"Turnover budget exhausted. Not increasing {symbol}.")
              return 0.0

      return proposed_size
  ```
- Integrate into size_position():
  ```python
  size = self._apply_turnover_budget(
      size, symbol, portfolio_equity,
      current_positions, dates_in_period
  )
  ```
- Add turnover tracking struct in PositionSizer to store per-position ytd turnover
- Log: current annualized turnover, remaining budget, forward turnover from this position

**Verify:**
- Unit test: position size scaled down when turnover budget exceeded
- Unit test: multiple positions respect cumulative turnover budget
- Integration test: turnover tracking accurate across multiple positions and periods

---

### T5: Implement Concentration Limit Constraint

**What:** Add hard cap on single-position notional as % of portfolio. No position can exceed CONCENTRATION_LIMIT_PCT of portfolio value. Enforce after other constraints.

**Files:**
- `/quant_engine/position_sizer.py` (add `_apply_concentration_limit()` method, ~30 lines)
- `/quant_engine/config.py` (add CONCENTRATION_LIMIT_PCT=0.20)

**Implementation notes:**
- Add method:
  ```python
  def _apply_concentration_limit(
      self,
      position_size: float,
      portfolio_equity: float,
      symbol: str = None
  ) -> float:
      """Ensure position doesn't exceed concentration limit as % of portfolio."""
      position_notional = position_size * portfolio_equity
      max_notional = portfolio_equity * self.config['CONCENTRATION_LIMIT_PCT']

      if position_notional > max_notional:
          reduced_size = max_notional / portfolio_equity
          self.logger.warning(
              f"Position {symbol} exceeds concentration limit {self.config['CONCENTRATION_LIMIT_PCT']:.1%}. "
              f"Capped at {reduced_size:.4f}."
          )
          return reduced_size

      return position_size
  ```
- Call in size_position() after shock budget:
  ```python
  size = self._apply_concentration_limit(size, portfolio_equity, symbol)
  ```
- Configuration should allow different limits for different regime states (optional)
- Log: position size, limit, and whether capped

**Verify:**
- Unit test: position size capped by concentration limit
- Unit test: with CONCENTRATION_LIMIT_PCT=0.20, no position exceeds 20% notional
- Integration test: portfolio respects concentration limit

---

### T6: Make Blend Weights Regime-Conditional and Configurable

**What:** Replace hardcoded blend weights (Kelly=0.3, vol=0.4, ATR=0.3) with regime-conditional configuration. Add BLEND_WEIGHTS_BY_REGIME dict to config, use it in PositionSizer._blend_sizes(). Support both static and dynamic blend weights.

**Files:**
- `/quant_engine/position_sizer.py` (modify `_blend_sizes()` method, ~30 lines)
- `/quant_engine/config.py` (add BLEND_WEIGHTS_BY_REGIME dict and BLEND_WEIGHTS_STATIC)

**Implementation notes:**
- Modify _blend_sizes() signature:
  ```python
  def _blend_sizes(
      self,
      kelly_size: float,
      vol_size: float,
      atr_size: float,
      regime_state: str = 'NORMAL'
  ) -> float:
      """Blend three sizing methods using regime-conditional weights."""

      # Get regime-specific weights (default to NORMAL if regime not configured)
      if self.config.get('BLEND_WEIGHTS_BY_REGIME'):
          weights = self.config['BLEND_WEIGHTS_BY_REGIME'].get(regime_state,
                      self.config['BLEND_WEIGHTS_BY_REGIME'].get('NORMAL'))
      else:
          weights = self.config.get('BLEND_WEIGHTS_STATIC',
                      {'kelly': 0.30, 'vol_scaled': 0.40, 'atr_based': 0.30})

      # Validate weights sum to 1.0
      weight_sum = weights['kelly'] + weights['vol_scaled'] + weights['atr_based']
      if abs(weight_sum - 1.0) > 0.01:
          self.logger.warning(f"Blend weights don't sum to 1.0 for regime {regime_state}. Normalizing.")
          weights = {k: v / weight_sum for k, v in weights.items()}

      blended = (
          weights['kelly'] * kelly_size +
          weights['vol_scaled'] * vol_size +
          weights['atr_based'] * atr_size
      )

      self.logger.debug(
          f"Blended size for {regime_state}: kelly={kelly_size:.4f} (w={weights['kelly']:.2f}), "
          f"vol={vol_size:.4f} (w={weights['vol_scaled']:.2f}), "
          f"atr={atr_size:.4f} (w={weights['atr_based']:.2f}) = {blended:.4f}"
      )

      return blended
  ```
- Config should have both options:
  - BLEND_WEIGHTS_STATIC: always use these weights (simple, predictable)
  - BLEND_WEIGHTS_BY_REGIME: use different weights per regime (more sophisticated)
- If both are specified, use BY_REGIME (it takes precedence)
- Log blend weights being used for each position size computation

**Verify:**
- Unit test: blend weights applied correctly per regime
- Unit test: different regimes produce different blends
- Integration test: changing BLEND_WEIGHTS_BY_REGIME affects sizing output

---

### T7: Parameterize Kelly Bayesian Prior and Update Sample Size Threshold

**What:** Make Kelly Bayesian prior parameters (alpha, beta in Beta distribution) configurable in config. Add minimum sample size threshold for updating Kelly estimate. With small sample size, prior dominates less.

**Files:**
- `/quant_engine/position_sizer.py` (modify `update_kelly_bayesian()` method, ~20 lines)
- `/quant_engine/config.py` (add KELLY_BAYESIAN_PRIOR_ALPHA, KELLY_BAYESIAN_PRIOR_BETA, KELLY_MIN_SAMPLES_FOR_UPDATE)

**Implementation notes:**
- Modify update_kelly_bayesian():
  ```python
  def update_kelly_bayesian(self, win_rate: float, num_trades: int) -> float:
      """Update Kelly estimate using Bayesian Beta-Binomial model.

      Prior: Beta(alpha, beta) with default (2, 2) (uninformed uniform-ish prior)
      Likelihood: Binomial(num_trades, win_rate)
      Posterior: Beta(alpha + wins, beta + losses)
      """

      # Check minimum sample size
      min_samples = self.config.get('KELLY_MIN_SAMPLES_FOR_UPDATE', 10)
      if num_trades < min_samples:
          self.logger.debug(f"Insufficient trades ({num_trades} < {min_samples}). Using prior estimate.")
          alpha = self.config['KELLY_BAYESIAN_PRIOR_ALPHA']
          beta = self.config['KELLY_BAYESIAN_PRIOR_BETA']
          return (alpha - 1) / (alpha + beta - 2) if (alpha + beta) > 2 else 0.5

      # Update with data
      alpha = self.config['KELLY_BAYESIAN_PRIOR_ALPHA']
      beta = self.config['KELLY_BAYESIAN_PRIOR_BETA']
      wins = int(win_rate * num_trades)
      losses = num_trades - wins

      posterior_alpha = alpha + wins
      posterior_beta = beta + losses

      # Maximum a posteriori estimate
      kelly_estimate = (posterior_alpha - 1) / (posterior_alpha + posterior_beta - 2)
      kelly_estimate = np.clip(kelly_estimate, 0.0, 1.0)

      self.logger.debug(
          f"Kelly Bayesian update: {num_trades} trades, {wins} wins. "
          f"Prior Beta({alpha}, {beta}), Posterior Beta({posterior_alpha}, {posterior_beta}), "
          f"Estimate: {kelly_estimate:.4f}"
      )

      return kelly_estimate
  ```
- Default configuration:
  ```python
  KELLY_BAYESIAN_PRIOR_ALPHA = 2  # Uninformed, slightly favors 0.5
  KELLY_BAYESIAN_PRIOR_BETA = 2
  KELLY_MIN_SAMPLES_FOR_UPDATE = 10
  ```
- Log: prior parameters, sample size, posterior estimate, degree of regularization

**Verify:**
- Unit test: Kelly estimate with small sample size dominated by prior
- Unit test: Kelly estimate with large sample size converges toward empirical win rate
- Integration test: different prior parameters produce different Kelly estimates

---

### T8: Consolidate Paper Trader and Autopilot to Use Unified PositionSizer

**What:** Update paper_trader.py to remove _position_size_pct() method and use PositionSizer interface for all sizing. Update autopilot/engine.py to call PositionSizer.size_position() with all uncertainty inputs. Verify both paths produce consistent sizing.

**Files:**
- `/quant_engine/paper_trader.py` (replace _position_size_pct() with PositionSizer calls, ~80 lines)
- `/quant_engine/autopilot/engine.py` (add uncertainty input computation and pass to PositionSizer, ~40 lines)
- `/quant_engine/tests/test_position_sizing_unification.py` (new integration tests, ~200 lines)

**Implementation notes:**
- In paper_trader.py, replace _position_size_pct() body with:
  ```python
  def _position_size_pct(self, symbol, kelly_history, atr, regime_state):
      """Compute position size as % of equity using unified PositionSizer."""
      return self.position_sizer.size_position_paper_trader(
          kelly_history=kelly_history,
          atr=atr,
          regime_state=regime_state,
          dd_ratio=self.dd_controller.current_drawdown_ratio,
          position_notional_before=self.positions[symbol].notional if symbol in self.positions else 0.0
      )
  ```
- In autopilot/engine.py, add uncertainty computation in _evaluate_candidates():
  ```python
  # Compute uncertainty inputs for sizing
  signal_uncertainty = ...  # From meta-labeling confidence or signal volatility
  regime_entropy = ...      # From regime state; CRITICAL -> high entropy
  drift_score = ...         # From momentum indicator

  # Pass to position sizer
  position_size = self.position_sizer.size_position(
      signal=signal,
      volatility=asset_volatility,
      regime_state=regime_state,
      recent_performance={'win_rate': ..., 'profit_factor': ...},
      signal_uncertainty=signal_uncertainty,
      regime_entropy=regime_entropy,
      drift_score=drift_score
  )
  ```
- Create comprehensive integration tests comparing old vs new sizing output
- Run backtest with unified sizing, verify P&L unchanged (within 0.1%)
- Log: both paths being used, any divergences detected

**Verify:**
- Unit test: paper_trader sizing calls unified interface correctly
- Integration test: autopilot engine sizing passes uncertainty inputs
- Regression test: full backtest with unified sizing produces same P&L as before (within tolerance)
- Backtest test: risk metrics (max DD, Sharpe) unchanged

---

### T9: Write Comprehensive Documentation and Tests

**What:** Write detailed design documentation explaining unified PositionSizer architecture, uncertainty-aware sizing, budget constraints, and blend weight configuration. Write extensive integration and unit tests.

**Files:**
- `/quant_engine/docs/DESIGN_POSITION_SIZING.md` (new, ~150 lines)
- `/quant_engine/tests/test_position_sizing_unification.py` (new, ~400 lines)
- `/quant_engine/tests/test_uncertainty_aware_sizing.py` (new, ~300 lines)
- `/quant_engine/tests/test_budget_constraints.py` (new, ~300 lines)

**Implementation notes:**
- DESIGN_POSITION_SIZING.md should cover:
  - High-level architecture: three sizing methods (Kelly, vol, ATR) blended with regime weights
  - Uncertainty-aware scaling: how signal_uncertainty, regime_entropy, drift_score affect sizing
  - Budget constraints: shock budget, turnover budget, concentration limit flow
  - Kelly Bayesian updating: Beta-Binomial model, prior hyperparameters, sample size effects
  - Configuration guide: how to set blend weights, budgets, uncertainty scaling
  - Troubleshooting: how to debug sizing issues, interpret logs
- Test files should have >90% coverage of PositionSizer class
- Tests use synthetic data with known properties (constant vol, known win rate, etc.)
- Include performance benchmarks: sizing computation time for large portfolios

**Verify:**
- All tests pass with >90% coverage
- Documentation reviewed and complete

---

## Validation

### Acceptance criteria
1. Unified PositionSizer interface used by both paper trader and autopilot engine
2. Uncertainty inputs integrated into size_position() with scaling factor applied correctly
3. Shock budget enforced: no position exceeds (1 - SHOCK_BUDGET_PCT) of portfolio
4. Turnover budget enforced: forward turnover tracked and sizes scaled down if budget exceeded
5. Concentration limit enforced: no single position exceeds CONCENTRATION_LIMIT_PCT
6. Blend weights regime-conditional: different regimes produce different sizing
7. Kelly Bayesian prior parameterized and min sample size threshold functional
8. Paper trader sizing consolidated with autopilot (no more _position_size_pct duplicate logic)
9. Backward compatibility: all existing tests pass, P&L unchanged (within 0.1%)
10. Comprehensive logging and documentation complete

### Verification steps
1. Run unit tests: `pytest tests/test_position_sizing_unification.py -v`
2. Run uncertainty sizing tests: `pytest tests/test_uncertainty_aware_sizing.py -v`
3. Run budget constraint tests: `pytest tests/test_budget_constraints.py -v`
4. Run full autopilot with unified sizing, verify logs show all three components (uncertainty, budgets, blend weights)
5. Run backtest: compare old sizing vs unified sizing, verify P&L within 0.1%
6. Verify all four budget constraints enforced: log shock budget remaining, turnover budget remaining, concentration limit breaches
7. Verify blend weights change per regime: log blend weights for each position in different regimes
8. Verify uncertainty scaling applied: log uncertainty inputs and scale factor for several positions
9. Review logs for correct Kelly Bayesian updates with small/large sample sizes
10. Verify documentation complete and accurate

### Rollback plan
- If unified sizing breaks paper trader, revert to old _position_size_pct() method
- If uncertainty-aware sizing causes over-sizing, disable UNCERTAINTY_SCALING_ENABLED=False
- If shock budget too restrictive, increase SHOCK_BUDGET_PCT
- If turnover budget constraint too limiting, increase MAX_ANNUALIZED_TURNOVER or disable TURNOVER_BUDGET_ENFORCEMENT
- If concentration limit too tight, increase CONCENTRATION_LIMIT_PCT
- Keep backup of pre-unification position_sizer.py for A/B testing

---

## Notes

- Shock budget is separate from margin/leverage: shock budget is a capital reserve, not a leverage constraint
- Turnover budget is annualized; daily turnover should be roughly MAX_ANNUALIZED_TURNOVER / 252
- Concentration limit applies to notional gross exposure; can be set independently of net delta concentration
- Uncertainty-aware sizing can be disabled by passing None for uncertainty inputs or setting UNCERTAINTY_SCALING_ENABLED=False
- Blend weights should be tuned per strategy and market regime; defaults (0.30/0.40/0.30) are starting point
- Kelly Bayesian prior with Beta(2, 2) is weakly informative; Beta(1, 1) is uniform, Beta(5, 5) is informative
- Regime-conditional blend weights are optional; if not configured, falls back to BLEND_WEIGHTS_STATIC
- Paper trader sizing and autopilot engine sizing should be identical if called with same inputs; verify in tests
