# Feature Spec: Execution Layer Improvements + Structural State-Aware Costs

> **Status:** Draft
> **Author:** Claude Opus 4
> **Date:** 2026-02-26
> **Estimated effort:** 95 hours across 7 tasks

---

## Why

The execution layer in `execution.py` already implements dynamic costs with volatility, gap, and range stress components, plus liquidity scalar and participation limits. However, three significant gaps remain:

1. **No structural state inputs**: Dynamic costs don't incorporate market structure inputs (break probability, structure uncertainty, drift score, systemic stress). Costs scale with volatility but not with market regime quality.
2. **No explicit ADV computation**: Participation limits use daily volume directly, but there's no separate ADV (Average Daily Volume) calculation or volume trend analysis.
3. **No entry/exit urgency differentiation**: Exit urgency during stops should accept higher costs than discretionary entry urgency. Currently, all orders are treated identically.

Additionally, the impact coefficient (EXEC_IMPACT_COEFF_BPS=25) is likely miscalibrated for different market cap segments. The liquidity scalar clips to [0.70, 3.00], which creates a hard floor (even very liquid stocks pay 70% of base costs). There's no automated no-trade gate during extreme stress, and the calibrate_cost_model() function exists but is never called in production.

This spec addresses structural state-aware costs, ADV computation, entry/exit urgency differentiation, and cost model calibration.

---

## What

### Part 1: Structural State-Aware Dynamic Costs
Extend the dynamic cost model to incorporate market structure inputs: break_probability (likelihood of flash crash), structure_uncertainty (regime state entropy), drift_score (price trend strength), systemic_stress (VIX percentile). These inputs scale cost multipliers up during stress periods.

### Part 2: Explicit ADV Computation and Trend Analysis
Create separate ADV computation module that tracks daily volumes and computes exponential moving average (EMA) of volume. Use volume trend to adjust participation limits and cost multipliers (higher volumes = lower costs).

### Part 3: Entry/Exit Urgency Differentiation
Add urgency_type parameter to execution model distinguishing entry vs exit. Exits during drawdown/stops accept higher costs (e.g., up to 100% of base cost multiplied by urgency factor). Discretionary entries have lower urgency and use tighter cost limits.

### Part 4: Cost Model Calibration and Market Cap Segmentation
Implement calibrate_cost_model() that adjusts impact coefficients per market cap segment (micro, small, mid, large cap) based on realized impact from historical trades. Store calibrated coefficients and use them in production.

---

## Constraints

### Must-haves
- Dynamic cost model extended to accept structural state inputs (break_prob, structure_uncertainty, drift_score, systemic_stress)
- Structural inputs scale cost multipliers: higher stress = higher multiplier (1.0 baseline, up to 3.0 in extreme stress)
- Explicit ADV computation with EMA smoothing (20-day EMA default)
- Volume trend adjustment: low volume days increase costs, high volume days decrease costs
- Entry vs exit urgency differentiation with separate cost acceptance limits
- Cost model calibration per market cap segment (micro, small, mid, large cap)
- No-trade gate during extreme stress (VIX > 95th percentile) for low-urgency orders
- All structural inputs optional; cost model works without them (backward compatible)
- Cost multipliers tracked and logged per execution

### Must-nots
- Do not modify existing sqrt(participation) impact formula (Almgren-Chriss model remains)
- Do not break existing spread, impact, participation limit components
- Do not change how fill_ratio is computed; keep [0, 1] scale unchanged
- Do not apply structural stress costs to portfolio optimizer (sizing only, not optimization)
- Do not store calibrated coefficients in database; use model files (joblib) in `/models/execution/`

### Out of scope
- Real-time cost prediction (use static calibrated coefficients)
- Multi-asset execution optimization (single-asset execution model only)
- Slicing algorithms for large orders (participation limit enforces position sizing upstream)
- Market impact decay modeling (assume immediate impact)

---

## Current State

### Key files
- `/quant_engine/execution.py` (274 lines): Has dynamic_costs() with vol/gap/range stress, liquidity_scalar, participation limits. Uses sqrt(participation) for impact.
- `/quant_engine/config.py`: EXEC_IMPACT_COEFF_BPS=25, EXEC_MAX_PARTICIPATION=0.02, EXEC_SPREAD_BPS=3, TRANSACTION_COST_BPS=20
- `/quant_engine/paper_trader.py` (817 lines): Calls execution.py to compute costs for position sizing
- `/quant_engine/position_sizer.py` (565 lines): Uses execution costs in portfolio optimization indirectly via position sizing

### Existing patterns to follow
1. Dynamic cost computation: base_cost + stress_multiplier * vol_component + gap_component + range_component
2. Config structure: all cost parameters in config.py, engine passes to execution module
3. Liquidity scalar clipping: [0.70, 3.00] range for cost multiplier
4. Participation limit: ADV-based, default 0.02 (2% of ADV per execution)
5. Fill ratio computation: assumed fill as % of target size, minimum 0.20 (20% guaranteed fill)
6. Spread and impact modeled separately: spread is symmetric, impact scales with volume fraction

### Configuration
```python
# Current config.py (relevant sections)
EXEC_SPREAD_BPS = 3  # Spread cost
EXEC_MAX_PARTICIPATION = 0.02  # 2% of ADV per execution
EXEC_IMPACT_COEFF_BPS = 25  # Almgren-Chriss impact coefficient
TRANSACTION_COST_BPS = 20  # Total transaction cost baseline
EXEC_STRESS_VOL_THRESHOLD = 2.5  # Vol multiplier trigger
EXEC_LIQUIDITY_SCALAR_MIN = 0.70
EXEC_LIQUIDITY_SCALAR_MAX = 3.00

# New config entries needed:
# Structural state-aware costs
EXEC_STRUCTURAL_STRESS_ENABLED = True
EXEC_BREAK_PROB_COST_MULT = {
    'low': 1.0,    # break_prob < 0.05
    'medium': 1.3, # 0.05 <= break_prob < 0.15
    'high': 2.0    # break_prob >= 0.15
}
EXEC_STRUCTURE_UNCERTAINTY_COST_MULT = 0.50  # +50% for each 0.1 increase in uncertainty
EXEC_DRIFT_SCORE_COST_REDUCTION = 0.20  # -20% for each 0.1 increase in drift (high conviction trend)
EXEC_SYSTEMIC_STRESS_COST_MULT = 0.30  # +30% for each 0.1 increase in systemic stress

# ADV computation
ADV_LOOKBACK_DAYS = 20  # Window for ADV calculation
ADV_EMA_SPAN = 20  # EMA smoothing parameter (higher = slower response)
EXEC_VOLUME_TREND_ENABLED = True
EXEC_LOW_VOLUME_COST_MULT = 1.5  # +50% on days with below-average volume

# Entry vs exit urgency
EXEC_EXIT_URGENCY_COST_LIMIT_MULT = 1.5  # Exits can tolerate 1.5x higher costs
EXEC_ENTRY_URGENCY_COST_LIMIT_MULT = 1.0  # Entries use standard cost limits
EXEC_STRESS_PULLBACK_MIN_SIZE = 0.10  # Reduce order size by 10% per urgency level

# Cost model calibration
EXEC_CALIBRATION_ENABLED = True
EXEC_CALIBRATION_RETRAIN_FREQ_DAYS = 30
EXEC_COST_IMPACT_COEFF_BY_MARKETCAP = {
    'micro': 40,   # Market cap < 300M
    'small': 30,   # 300M - 2B
    'mid': 20,     # 2B - 10B
    'large': 15    # > 10B
}
EXEC_NO_TRADE_STRESS_THRESHOLD = 0.95  # VIX percentile; above this, skip low-urgency orders
```

---

## Tasks

### T1: Extend Dynamic Cost Model with Structural State Inputs

**What:** Modify `execution.py.dynamic_costs()` to accept four new optional parameters: break_probability (0-1), structure_uncertainty (0-1), drift_score (0-1), systemic_stress (0-1). Implement multiplier functions that scale cost upward based on these inputs.

**Files:**
- `/quant_engine/execution.py` (modify `dynamic_costs()` signature and implementation, ~80 lines)
- `/quant_engine/config.py` (add EXEC_STRUCTURAL_STRESS_* parameters)

**Implementation notes:**
- Modify dynamic_costs() signature:
  ```python
  def dynamic_costs(
      self,
      volatility: float,
      participation_ratio: float,
      daily_volume: float,
      gap_percent: float = 0.0,
      range_percent: float = 0.0,
      break_probability: float = None,  # 0-1, likelihood of flash crash
      structure_uncertainty: float = None,  # 0-1, regime state entropy
      drift_score: float = None,  # 0-1, trend strength (high = confident direction)
      systemic_stress: float = None  # 0-1, normalized VIX percentile
  ) -> float:
  ```
- Implement multiplier functions:
  ```python
  def _compute_break_probability_mult(self, break_prob: float) -> float:
      """Scale cost multiplier based on flash crash risk."""
      if break_prob is None:
          return 1.0

      if break_prob < 0.05:
          return self.config['EXEC_BREAK_PROB_COST_MULT']['low']
      elif break_prob < 0.15:
          mult_low = self.config['EXEC_BREAK_PROB_COST_MULT']['low']
          mult_med = self.config['EXEC_BREAK_PROB_COST_MULT']['medium']
          # Interpolate between low and medium
          interp_factor = (break_prob - 0.05) / (0.15 - 0.05)
          return mult_low + (mult_med - mult_low) * interp_factor
      else:
          mult_med = self.config['EXEC_BREAK_PROB_COST_MULT']['medium']
          mult_high = self.config['EXEC_BREAK_PROB_COST_MULT']['high']
          # Interpolate between medium and high, capped at high
          interp_factor = min((break_prob - 0.15) / (0.50 - 0.15), 1.0)
          return mult_med + (mult_high - mult_med) * interp_factor

  def _compute_structure_uncertainty_mult(self, uncertainty: float) -> float:
      """Scale cost multiplier based on regime state uncertainty."""
      if uncertainty is None:
          return 1.0

      cost_mult_per_unit = self.config['EXEC_STRUCTURE_UNCERTAINTY_COST_MULT']
      return 1.0 + (uncertainty * cost_mult_per_unit)

  def _compute_drift_score_mult(self, drift_score: float) -> float:
      """Reduce cost multiplier based on trend strength (high drift = high confidence)."""
      if drift_score is None:
          return 1.0

      cost_reduction_per_unit = self.config['EXEC_DRIFT_SCORE_COST_REDUCTION']
      return 1.0 - (drift_score * cost_reduction_per_unit)

  def _compute_systemic_stress_mult(self, systemic_stress: float) -> float:
      """Scale cost multiplier based on systemic stress (e.g., VIX percentile)."""
      if systemic_stress is None:
          return 1.0

      cost_mult_per_unit = self.config['EXEC_SYSTEMIC_STRESS_COST_MULT']
      return 1.0 + (systemic_stress * cost_mult_per_unit)
  ```
- In dynamic_costs(), compute composite structural multiplier:
  ```python
  # Existing components
  vol_mult = 1.0 + (volatility / baseline_vol - 1.0) * vol_stress_weight
  gap_mult = 1.0 + (gap_percent / 0.02) * gap_weight  # Existing logic
  range_mult = 1.0 + (range_percent / 0.05) * range_weight  # Existing logic

  # New structural components
  break_mult = self._compute_break_probability_mult(break_probability)
  uncertainty_mult = self._compute_structure_uncertainty_mult(structure_uncertainty)
  drift_mult = self._compute_drift_score_mult(drift_score)
  stress_mult = self._compute_systemic_stress_mult(systemic_stress)

  # Composite structural multiplier (multiplicative, not additive)
  structural_mult = break_mult * uncertainty_mult * drift_mult * stress_mult

  # Apply to base cost
  total_mult = np.clip(vol_mult * gap_mult * range_mult * structural_mult, 1.0, 3.0)
  cost_bps = base_cost_bps * total_mult
  ```
- Log: all component multipliers and composite multiplier
- Structural inputs are optional; if None, multiplier = 1.0 (no effect)

**Verify:**
- Unit test: each multiplier function produces expected output for boundary values
- Unit test: composite multiplier clips to [1.0, 3.0]
- Integration test: high break_probability increases costs
- Integration test: high structure_uncertainty increases costs
- Integration test: high drift_score decreases costs (more confident entry)
- Integration test: high systemic_stress increases costs

---

### T2: Implement Explicit ADV Computation and Volume Trend Analysis

**What:** Create new module `execution/adv_tracker.py` that computes Average Daily Volume (ADV) with exponential moving average. Track volume trends and adjust participation limits and cost multipliers based on volume conditions.

**Files:**
- `/quant_engine/execution/adv_tracker.py` (new, ~200 lines)
- `/quant_engine/execution.py` (integrate ADV tracker, ~20 lines)
- `/quant_engine/config.py` (add ADV_* parameters)

**Implementation notes:**
- Create ADVTracker class:
  ```python
  class ADVTracker:
      def __init__(self, config):
          self.lookback_days = config['ADV_LOOKBACK_DAYS']
          self.ema_span = config['ADV_EMA_SPAN']
          self.volume_history = {}  # {symbol: [volumes]}
          self.ema_history = {}     # {symbol: ema_value}
          self.volume_trend = {}    # {symbol: trend_percentile}

      def update(self, symbol: str, daily_volume: float, date: pd.Timestamp):
          """Update ADV estimates with latest daily volume."""
          if symbol not in self.volume_history:
              self.volume_history[symbol] = []
              self.ema_history[symbol] = daily_volume

          self.volume_history[symbol].append(daily_volume)
          if len(self.volume_history[symbol]) > self.lookback_days:
              self.volume_history[symbol].pop(0)

          # Compute EMA
          alpha = 2.0 / (self.ema_span + 1)
          self.ema_history[symbol] = (
              alpha * daily_volume +
              (1 - alpha) * self.ema_history[symbol]
          )

          # Compute volume trend (percentile of current volume vs EMA)
          trend = min(daily_volume / self.ema_history[symbol], 2.0)  # Cap at 2.0x
          self.volume_trend[symbol] = trend

      def get_adv(self, symbol: str) -> float:
          """Get current ADV estimate."""
          return self.ema_history.get(symbol, 0.0)

      def get_volume_trend(self, symbol: str) -> float:
          """Get volume trend multiplier [0.5, 2.0]."""
          return np.clip(self.volume_trend.get(symbol, 1.0), 0.5, 2.0)

      def adjust_participation_limit(self, symbol: str, base_limit: float) -> float:
          """Adjust participation limit based on volume trend."""
          trend = self.get_volume_trend(symbol)
          # High volume days allow higher participation
          adjusted_limit = base_limit * trend
          return np.clip(adjusted_limit, base_limit * 0.5, base_limit * 2.0)
  ```
- In execution.py, create ADVTracker instance in __init__:
  ```python
  self.adv_tracker = ADVTracker(config)
  ```
- In execution methods, call:
  ```python
  self.adv_tracker.update(symbol, daily_volume, date)
  adjusted_participation_limit = self.adv_tracker.adjust_participation_limit(symbol, base_limit)
  ```
- Volume trend adjustment: high volume (trend > 1.0) reduces costs, low volume (trend < 1.0) increases costs
- Log: ADV estimate, volume trend, adjusted participation limit

**Verify:**
- Unit test: ADV computed correctly with EMA smoothing
- Unit test: volume trend adjustment within [0.5, 2.0] range
- Unit test: high volume decreases participation limit, low volume increases it
- Integration test: participation limit adapts to volume conditions

---

### T3: Add Entry/Exit Urgency Differentiation and Cost Acceptance Limits

**What:** Add urgency_type parameter to execution model distinguishing 'entry' (discretionary, low urgency) vs 'exit' (forced, high urgency). Exits tolerate higher costs per EXEC_EXIT_URGENCY_COST_LIMIT_MULT. Implement no-trade gate for low-urgency orders during extreme stress.

**Files:**
- `/quant_engine/execution.py` (add `execute_order()` method with urgency param, ~60 lines)
- `/quant_engine/config.py` (add EXEC_EXIT_URGENCY_COST_LIMIT_MULT, EXEC_ENTRY_URGENCY_COST_LIMIT_MULT, EXEC_STRESS_PULLBACK_MIN_SIZE, EXEC_NO_TRADE_STRESS_THRESHOLD)

**Implementation notes:**
- Add new method execute_order():
  ```python
  def execute_order(
      self,
      symbol: str,
      size: float,  # % of equity or shares
      urgency_type: str = 'entry',  # 'entry' or 'exit'
      current_price: float = None,
      volatility: float = None,
      daily_volume: float = None,
      vix_percentile: float = None,  # VIX percentile [0, 1]
      break_probability: float = None,
      structure_uncertainty: float = None,
      drift_score: float = None,
      systemic_stress: float = None
  ) -> dict:
      """Execute order with urgency-aware cost acceptance and no-trade gate."""

      # No-trade gate: skip low-urgency orders during extreme stress
      if (urgency_type == 'entry' and
          vix_percentile is not None and
          vix_percentile > self.config['EXEC_NO_TRADE_STRESS_THRESHOLD']):
          self.logger.warning(
              f"Skipping entry order for {symbol} due to extreme stress "
              f"(VIX percentile {vix_percentile:.2f})"
          )
          return {'executed': False, 'reason': 'extreme_stress', 'size': 0.0}

      # Compute expected costs
      costs_bps = self.dynamic_costs(
          volatility=volatility,
          participation_ratio=size / daily_volume if daily_volume > 0 else 0.0,
          daily_volume=daily_volume,
          break_probability=break_probability,
          structure_uncertainty=structure_uncertainty,
          drift_score=drift_score,
          systemic_stress=systemic_stress
      )

      # Check cost acceptance limit based on urgency
      if urgency_type == 'exit':
          max_acceptable_cost = self.config['TRANSACTION_COST_BPS'] * self.config['EXEC_EXIT_URGENCY_COST_LIMIT_MULT']
      else:
          max_acceptable_cost = self.config['TRANSACTION_COST_BPS'] * self.config['EXEC_ENTRY_URGENCY_COST_LIMIT_MULT']

      if costs_bps > max_acceptable_cost:
          # Reduce order size to fit cost limit
          reduction_factor = max_acceptable_cost / costs_bps
          adjusted_size = size * reduction_factor
          # Apply minimum size pullback
          pullback = self.config['EXEC_STRESS_PULLBACK_MIN_SIZE']
          adjusted_size = max(adjusted_size, size * (1.0 - pullback))
          self.logger.warning(
              f"Order cost {costs_bps:.2f} bps exceeds limit {max_acceptable_cost:.2f} bps. "
              f"Reduced size from {size:.4f} to {adjusted_size:.4f}."
          )
          size = adjusted_size

      # Execute and return filled size
      fill_ratio = self._compute_fill_ratio(size, daily_volume)
      filled_size = size * fill_ratio

      return {
          'executed': True,
          'target_size': size,
          'filled_size': filled_size,
          'fill_ratio': fill_ratio,
          'cost_bps': costs_bps,
          'acceptable_cost_limit': max_acceptable_cost,
          'urgency_type': urgency_type
      }
  ```
- Log: cost comparison, urgency type, size reduction decision, final fill

**Verify:**
- Unit test: exit orders have higher cost acceptance limit than entry
- Unit test: no-trade gate blocks low-urgency orders during high stress
- Unit test: high stress causes size reduction for entry orders
- Integration test: execution handles all combination of parameters

---

### T4: Implement Cost Model Calibration Per Market Cap Segment

**What:** Create calibration module that adjusts EXEC_IMPACT_COEFF_BPS per market cap segment (micro, small, mid, large cap) based on realized impact from historical trades. Calibration runs weekly, stores coefficients to model files, and uses them in production.

**Files:**
- `/quant_engine/execution/cost_calibrator.py` (new, ~250 lines)
- `/quant_engine/execution.py` (integrate cost calibrator, ~20 lines)
- `/quant_engine/config.py` (add EXEC_CALIBRATION_* and EXEC_COST_IMPACT_COEFF_BY_MARKETCAP)

**Implementation notes:**
- Create CostCalibrator class:
  ```python
  class CostCalibrator:
      def __init__(self, config):
          self.config = config
          self.trade_history = []  # Store (symbol, size, participation, realized_cost)
          self.calibrated_coeff = config['EXEC_COST_IMPACT_COEFF_BY_MARKETCAP'].copy()
          self.last_calibration_date = None

      def record_trade(
          self,
          symbol: str,
          market_cap: float,
          size: float,
          participation_ratio: float,
          expected_cost_bps: float,
          realized_cost_bps: float
      ):
          """Record historical trade for calibration."""
          marketcap_segment = self._get_marketcap_segment(market_cap)
          self.trade_history.append({
              'symbol': symbol,
              'marketcap_segment': marketcap_segment,
              'size': size,
              'participation_ratio': participation_ratio,
              'expected_cost': expected_cost_bps,
              'realized_cost': realized_cost_bps,
              'date': pd.Timestamp.now()
          })

      def _get_marketcap_segment(self, market_cap: float) -> str:
          """Classify market cap into segment."""
          if market_cap < 300e6:
              return 'micro'
          elif market_cap < 2e9:
              return 'small'
          elif market_cap < 10e9:
              return 'mid'
          else:
              return 'large'

      def calibrate(self) -> dict:
          """Calibrate impact coefficients per segment using realized data."""
          if len(self.trade_history) < 100:
              self.logger.warning(f"Insufficient trades ({len(self.trade_history)}) for calibration")
              return {}

          calibration_results = {}

          for segment in ['micro', 'small', 'mid', 'large']:
              segment_trades = [t for t in self.trade_history
                               if t['marketcap_segment'] == segment]

              if len(segment_trades) < 20:
                  self.logger.warning(f"Insufficient {segment} cap trades for calibration")
                  continue

              # Compute realized impact per participation unit
              # impact = (realized_cost - spread) / participation_ratio^2
              # Solve for impact_coeff: realized_cost = spread + impact_coeff * sqrt(participation)
              impacts = []
              for trade in segment_trades:
                  if trade['participation_ratio'] > 0:
                      # Remove spread component (assume 3 bps)
                      net_cost = trade['realized_cost'] - self.config['EXEC_SPREAD_BPS']
                      # Solve for coefficient
                      coeff = net_cost / np.sqrt(trade['participation_ratio'])
                      impacts.append(coeff)

              if impacts:
                  # Use median impact coefficient (robust to outliers)
                  new_coeff = np.median(impacts)
                  old_coeff = self.calibrated_coeff.get(segment, 25)

                  # Apply 30% smoothing to avoid wild swings
                  smoothed_coeff = 0.70 * old_coeff + 0.30 * new_coeff

                  self.calibrated_coeff[segment] = smoothed_coeff
                  calibration_results[segment] = {
                      'trades_count': len(segment_trades),
                      'old_coeff': old_coeff,
                      'new_coeff': new_coeff,
                      'smoothed_coeff': smoothed_coeff
                  }

                  self.logger.info(
                      f"{segment} cap calibration: {old_coeff:.1f} -> {new_coeff:.1f} "
                      f"(smoothed {smoothed_coeff:.1f}) based on {len(segment_trades)} trades"
                  )

          # Save calibrated coefficients
          self._save_calibration()
          self.last_calibration_date = pd.Timestamp.now()

          return calibration_results

      def _save_calibration(self):
          """Save calibrated coefficients to model file."""
          model_path = Path('/models/execution/cost_calibration_current.joblib')
          model_path.parent.mkdir(parents=True, exist_ok=True)
          joblib.dump(self.calibrated_coeff, model_path)
          self.logger.info(f"Saved cost calibration to {model_path}")

      def get_impact_coeff(self, market_cap: float) -> float:
          """Get calibrated impact coefficient for market cap."""
          segment = self._get_marketcap_segment(market_cap)
          return self.calibrated_coeff.get(segment, 25.0)
  ```
- In execution.py, use calibrated coefficients:
  ```python
  def dynamic_costs(self, ..., market_cap: float = None, ...):
      # Get calibrated impact coefficient
      impact_coeff = self.cost_calibrator.get_impact_coeff(market_cap)
      # Use in impact calculation instead of hardcoded 25
  ```
- Calibration runs weekly (similar to meta-labeling retraining in SPEC_04)
- Log: calibration results per segment, trades count, coefficient changes

**Verify:**
- Unit test: market cap segments classified correctly
- Unit test: impact coefficients updated correctly from realized data
- Unit test: smoothing applied to new coefficients (30% old, 70% new)
- Integration test: calibrated coefficients saved and loaded correctly

---

### T5: Integrate All Execution Layer Improvements into Paper Trader

**What:** Update paper_trader.py to call the new execute_order() method with all available inputs: urgency_type, vix_percentile, structural state inputs. Log all execution decisions for monitoring.

**Files:**
- `/quant_engine/paper_trader.py` (modify position execution and entry/exit logic, ~50 lines)
- `/quant_engine/autopilot/engine.py` (pass structural state inputs to paper trader, ~20 lines)

**Implementation notes:**
- In paper_trader.py, for entry execution:
  ```python
  # Compute structural state inputs (similar to sizing in SPEC_05)
  break_prob = self._compute_break_probability()
  structure_uncertainty = regime_state_to_entropy[regime_state]
  drift_score = self._compute_drift_score()
  systemic_stress = self._compute_systemic_stress(vix_percentile)

  # Execute entry with all inputs
  result = self.execution_model.execute_order(
      symbol=symbol,
      size=position_size,
      urgency_type='entry',
      current_price=current_price,
      volatility=volatility,
      daily_volume=daily_volume,
      vix_percentile=vix_percentile,
      break_probability=break_prob,
      structure_uncertainty=structure_uncertainty,
      drift_score=drift_score,
      systemic_stress=systemic_stress
  )

  # Handle result
  if result['executed']:
      filled_size = result['filled_size']
      # Update position with filled size
  else:
      self.logger.warning(f"Entry order rejected for {symbol}: {result['reason']}")
  ```
- For exit execution (stop loss, drawdown):
  ```python
  result = self.execution_model.execute_order(
      symbol=symbol,
      size=position_size,
      urgency_type='exit',  # Higher cost tolerance for stops
      ...
  )
  ```
- Log: execution decision, cost comparison, urgency type

**Verify:**
- Integration test: paper trader calls execute_order with all parameters
- Integration test: entry and exit execution handled correctly
- Backtest test: filled sizes reflect cost-based rejections/reductions

---

### T6: Add Comprehensive Execution Layer Logging and Monitoring

**What:** Implement detailed logging for all execution components: structural state multipliers, ADV adjustments, urgency-based cost acceptance, calibration results. Log to both stdout and structured JSON files for analysis.

**Files:**
- `/quant_engine/execution.py` (add logging throughout, ~40 lines)
- `/quant_engine/execution/adv_tracker.py` (add logging, ~15 lines)
- `/quant_engine/execution/cost_calibrator.py` (add logging, ~20 lines)

**Implementation notes:**
- Log at INFO level for each execution:
  - Base and structural cost multipliers (break_prob, uncertainty, drift, stress)
  - Final composite multiplier and cost in bps
  - ADV estimate and volume trend adjustment
  - Urgency type and cost acceptance limit
  - Size reduction due to high costs
  - Fill ratio and filled size
- Log at DEBUG level:
  - All intermediate calculations
  - Cost calibration per segment
  - ADV EMA updates
- Store structured JSON logs for analysis:
  ```json
  {
    "timestamp": "2026-02-26T10:30:45",
    "symbol": "AAPL",
    "execution_type": "entry",
    "costs": {
      "base_bps": 23,
      "vol_mult": 1.05,
      "break_prob_mult": 1.2,
      "uncertainty_mult": 1.15,
      "drift_mult": 0.95,
      "stress_mult": 1.1,
      "composite_mult": 1.535,
      "final_cost_bps": 35.3
    },
    "adv": {
      "estimate": 5000000,
      "volume_trend": 1.2,
      "adjusted_participation_limit": 0.024
    },
    "urgency": {
      "type": "entry",
      "cost_limit": 20,
      "cost_exceeded": true,
      "size_reduction": 0.30
    },
    "execution": {
      "target_size": 0.02,
      "fill_ratio": 0.95,
      "filled_size": 0.019
    }
  }
  ```

**Verify:**
- Integration test: logs generated with correct format
- Log analysis: verify cost multipliers reasonable (1.0-3.0 range)
- Log analysis: verify ADV trends make sense (volume trend around 1.0 mean)

---

### T7: Write Integration Tests and Documentation

**What:** Write comprehensive tests covering: structural cost multipliers, ADV computation, entry/exit urgency, cost calibration, no-trade gate, and end-to-end execution flow. Write design guide explaining execution layer architecture.

**Files:**
- `/quant_engine/tests/test_execution_layer.py` (new, ~500 lines)
- `/quant_engine/docs/DESIGN_EXECUTION.md` (new, ~120 lines)

**Implementation notes:**
- Test file structure:
  - TestStructuralCosts: test each multiplier function and composite
  - TestADVTracking: test ADV computation and volume trend
  - TestUrgencyDifferentiation: test entry vs exit cost acceptance
  - TestCostCalibration: test calibration logic and coefficient updates
  - TestNoTradeGate: test stress threshold blocking
  - TestIntegrationExecution: full execution pipeline with all components
- Tests use synthetic OHLCV data with controlled costs and volumes
- Documentation includes:
  - Execution layer architecture overview
  - Structural state inputs explanation and sources
  - ADV computation and volume trend tuning
  - Entry vs exit urgency guidance
  - Cost calibration methodology and results interpretation
  - Troubleshooting guide (when costs are too high, why orders are skipped, etc.)
  - Performance metrics (execution speed, cost prediction accuracy)

**Verify:**
- All tests pass with >90% coverage
- Documentation reviewed and complete

---

## Validation

### Acceptance criteria
1. Structural state-aware cost model extended with all four inputs (break_prob, uncertainty, drift, stress)
2. Structural multipliers computed correctly for each input, composite in [1.0, 3.0]
3. Explicit ADV computation implemented with EMA smoothing and volume trend tracking
4. Entry vs exit urgency differentiation functional with separate cost acceptance limits
5. No-trade gate blocks low-urgency orders during extreme stress (VIX > 95th percentile)
6. Cost model calibration per market cap segment (micro, small, mid, large) implemented
7. Calibrated coefficients saved to model files and used in production
8. Paper trader integrated with all new execution features
9. Comprehensive logging of all execution components implemented
10. Integration tests pass with >90% coverage, documentation complete
11. Backward compatibility: execution without structural inputs works as before

### Verification steps
1. Run unit tests: `pytest tests/test_execution_layer.py -v`
2. Run integration test: `pytest tests/test_execution_layer.py::TestIntegrationExecution -v`
3. Run execution module with synthetic data, verify structural multipliers computed correctly
4. Verify ADV tracking updates correctly with volume data, EMA computed accurately
5. Verify entry orders rejected and exit orders accepted with different cost limits
6. Verify no-trade gate triggered at appropriate stress levels
7. Run cost calibration on historical trade data, verify coefficients updated by segment
8. Run backtest with full execution layer, verify execution costs reasonable (20-50 bps typical)
9. Check logs for all execution decisions, cost multipliers, ADV adjustments
10. Verify structural inputs optional: execution works without them (backward compatible)

### Rollback plan
- If structural cost multipliers cause over-rejection of orders, disable EXEC_STRUCTURAL_STRESS_ENABLED=False
- If ADV tracking causes volatility, disable EXEC_VOLUME_TREND_ENABLED=False
- If no-trade gate too restrictive, increase EXEC_NO_TRADE_STRESS_THRESHOLD
- If urgency-based cost limits cause issues, set EXIT_URGENCY_COST_LIMIT_MULT=1.0 (same as entry)
- If cost calibration produces unreasonable coefficients, disable EXEC_CALIBRATION_ENABLED=False
- Keep backup of pre-calibration impact coefficients for comparison

---

## Notes

- Structural state inputs (break_prob, uncertainty, drift, stress) come from external sources: regime analyzer, drift indicator, stress metric
- ADV computation uses EMA for stability; volume spike on single day has limited effect due to smoothing
- Cost calibration is robust to outliers: uses median impact coefficient, applies smoothing to avoid wild swings
- No-trade gate only blocks low-urgency (entry) orders; exits are always allowed (emergency override)
- Calibrated impact coefficients are saved to model files; retraining doesn't affect live production until explicitly loaded
- Structural cost multipliers are multiplicative (break_mult * uncertainty_mult * drift_mult * stress_mult), not additive, to avoid unrealistic stacking
- Entry/exit urgency model allows flexible cost acceptance: exits during stops can tolerate 1.5-2.0x normal costs
- Market cap segments can be customized per broker/exchange by adjusting EXEC_COST_IMPACT_COEFF_BY_MARKETCAP thresholds
