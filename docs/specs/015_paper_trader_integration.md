# Feature Spec: Paper Trader Integration — Wire All Risk Components

> **Status:** Approved
> **Author:** justin
> **Date:** 2026-02-23
> **Estimated effort:** ~8 hours across 6 tasks

---

## Why

The paper trader operates as a standalone component that doesn't use most of the risk management infrastructure: (1) DrawdownController is never instantiated or called — the paper trader has no drawdown protection, meaning it keeps entering new positions even during significant equity decline. (2) StopLossManager is never used — exits are only triggered by signal decay or time-out, not by ATR stops, trailing stops, or hard stops. (3) PortfolioRiskManager is never called — there's no check for sector concentration, correlation, or portfolio volatility before entering new positions. (4) The position sizer is called but regime is not passed from the signal data — it defaults to None, meaning regime-conditional sizing never activates. (5) Trade records don't store entry regime, preventing post-hoc regime analysis of paper trading performance. (6) There's no equity curve tracking — you can't compute drawdown, Sharpe, or any risk metric from paper trading state.

## What

Wire DrawdownController, StopLossManager, and PortfolioRiskManager into the paper trader's run_cycle(). Store regime in trade records. Track equity curve for risk metrics. Done means: the paper trader uses all the same risk components as the backtester's risk-managed mode.

## Constraints

### Must-haves
- DrawdownController.update() called each cycle with daily PnL
- StopLossManager.evaluate() called for every open position
- PortfolioRiskManager.check_new_position() called before each entry
- Regime stored in position and trade records
- Equity curve tracked in state for risk metric computation

### Must-nots
- Do NOT change the paper trader's basic entry/exit logic (signal threshold, confidence threshold)
- Do NOT make run_cycle() take >5 seconds (risk checks must be fast)
- Do NOT break existing paper trader state files (migration, not replacement)

## Tasks

### T1: Wire DrawdownController into paper trader

**What:** Initialize DrawdownController in constructor and call update() each cycle.

**Files:**
- `autopilot/paper_trader.py` — add DrawdownController integration

**Implementation notes:**
```python
# In __init__:
from quant_engine.risk.drawdown import DrawdownController
self.dd_controller = DrawdownController(initial_equity=initial_capital)

# In run_cycle(), after computing current equity:
prev_equity = state.get('prev_equity', self.initial_capital)
current_equity = state['cash'] + sum(
    pos['shares'] * self._get_current_price(pos['ticker'], price_data)
    for pos in state['positions']
)
daily_pnl = (current_equity - prev_equity) / prev_equity if prev_equity > 0 else 0

dd_status = self.dd_controller.update(daily_pnl)
state['prev_equity'] = current_equity

# Apply drawdown controls:
# 1. Scale new position sizes
position_size_pct *= dd_status.size_multiplier

# 2. Block new entries if drawdown is too deep
if not dd_status.allow_new_entries:
    logger.warning("Drawdown controller blocking new entries: state=%s, dd=%.2f%%", 
                   dd_status.state.name, dd_status.current_drawdown * 100)
    candidates = []

# 3. Force liquidate if critical
if dd_status.force_liquidate:
    logger.error("CRITICAL drawdown — force liquidating all positions")
    for pos in list(state['positions']):
        # Execute immediate exit
        ...
```

**Verify:**
```bash
python -c "
from quant_engine.risk.drawdown import DrawdownController
dc = DrawdownController(initial_equity=100000)
# Normal day
s1 = dc.update(0.01)
print(f'After +1%: state={s1.state.name}, allow_new={s1.allow_new_entries}')
# Bad day
s2 = dc.update(-0.04)
print(f'After -4%: state={s2.state.name}, allow_new={s2.allow_new_entries}')
"
```

---

### T2: Wire StopLossManager into position evaluation

**What:** Replace simple signal-decay/timeout exit logic with proper stop loss evaluation.

**Files:**
- `autopilot/paper_trader.py` — add StopLossManager to exit evaluation

**Implementation notes:**
- Current exit logic (lines ~296-341): only checks weak signal and timeout
- New: evaluate stops FIRST, then check signal decay as secondary
```python
# In run_cycle(), position evaluation phase:
from quant_engine.risk.stop_loss import StopLossManager

stop_mgr = StopLossManager()

for pos in state['positions']:
    current_price = self._get_current_price(pos['ticker'], price_data)
    highest_price = max(pos.get('highest_price', pos['entry_price']), current_price)
    pos['highest_price'] = highest_price  # Track for trailing stop
    
    # Compute ATR from price data
    ticker_data = price_data.get(pos['ticker'])
    atr = self._compute_atr(ticker_data, lookback=14)
    
    # Evaluate stop loss
    stop_result = stop_mgr.evaluate(
        entry_price=pos['entry_price'],
        current_price=current_price,
        highest_price=highest_price,
        atr=atr,
        bars_held=pos['holding_days'],
        entry_regime=pos.get('entry_regime', 0),
        current_regime=self._get_current_regime(pos['ticker']),
    )
    
    if stop_result.should_exit:
        # Execute stop-loss exit
        reason = stop_result.reason.name  # 'ATR_STOP', 'TRAILING_STOP', etc.
        self._execute_exit(state, pos, current_price, reason)
        continue
    
    # Secondary: check signal decay (existing logic)
    # ... weak_signal and timed_out checks ...
```

**Verify:**
```bash
python -c "
from quant_engine.risk.stop_loss import StopLossManager
sm = StopLossManager()
# Test hard stop: 10% loss
result = sm.evaluate(
    entry_price=100, current_price=91, highest_price=105,
    atr=2.0, bars_held=5, entry_regime=0, current_regime=0
)
print(f'Hard stop at -9%: should_exit={result.should_exit}, reason={result.reason.name}')
"
```

---

### T3: Wire PortfolioRiskManager into entry evaluation

**What:** Check portfolio-level risk constraints before entering new positions.

**Files:**
- `autopilot/paper_trader.py` — add PortfolioRiskManager to entry evaluation

**Implementation notes:**
```python
# In __init__:
from quant_engine.risk.portfolio_risk import PortfolioRiskManager
self.risk_mgr = PortfolioRiskManager()

# In run_cycle(), before entering a new position:
for candidate in candidates:
    ticker = candidate['ticker']
    
    # Build current portfolio weights
    portfolio_weights = {}
    for pos in state['positions']:
        pos_value = pos['shares'] * self._get_current_price(pos['ticker'], price_data)
        portfolio_weights[pos['ticker']] = pos_value / current_equity
    
    # Check if adding this position passes risk checks
    risk_check = self.risk_mgr.check_new_position(
        ticker=ticker,
        proposed_weight=position_size_pct,
        current_weights=portfolio_weights,
        returns_data=self._get_returns_data(price_data),
    )
    
    if not risk_check.passed:
        logger.info("Risk check blocked entry for %s: %s", ticker, risk_check.violations)
        continue  # Skip this candidate
    
    # Proceed with entry...
```

**Verify:**
```bash
python -c "
from quant_engine.risk.portfolio_risk import PortfolioRiskManager
rm = PortfolioRiskManager(max_corr_between=0.85)
print(f'Max correlation: {rm.max_corr_between}')
print(f'Max sector pct: {rm.max_sector_pct}')
print(f'Max single name: {rm.max_single_name_pct}')
print('Risk checks: correlation, sector, single-name, beta, volatility')
"
```

---

### T4: Pass regime through signal data and store in trade records

**What:** Extract regime from predictions and store it in position/trade records.

**Files:**
- `autopilot/paper_trader.py` — add regime handling

**Implementation notes:**
- Current: `_position_size_pct()` accepts `regime` parameter but it's always None from run_cycle
- Fix: extract regime from latest_predictions
```python
# In run_cycle(), when processing candidates:
for candidate in candidates:
    permno = candidate['permno']
    ticker = candidate['ticker']
    
    # Extract regime from predictions
    pred_row = latest_predictions[latest_predictions['permno'] == permno].iloc[0]
    regime = int(pred_row.get('regime', 0))
    confidence = float(pred_row.get('confidence', 0.5))
    
    # Pass regime to position sizer
    position_size_pct = self._position_size_pct(
        state, strategy_id, base_size, max_holding_days,
        confidence=confidence,
        regime=regime,  # NOW POPULATED (was None)
        ticker=ticker,
        as_of=as_of,
        price_data=price_data,
    )
    
    # Store regime in position record
    position = {
        'strategy_id': strategy_id,
        'permno': permno,
        'ticker': ticker,
        'entry_price': entry_price,
        'entry_date': as_of_str,
        'entry_regime': regime,  # NEW FIELD
        'confidence': confidence,  # NEW FIELD
        'shares': shares,
        'holding_days': 0,
        'highest_price': entry_price,  # NEW: for trailing stop
        ...
    }
    
    # Store regime in trade record on exit
    trade = {
        ...
        'entry_regime': pos.get('entry_regime', 0),  # NEW FIELD
        'exit_regime': current_regime,  # NEW FIELD
        'regime_changed': pos.get('entry_regime', 0) != current_regime,  # NEW FIELD
    }
```

**Verify:**
```bash
python -c "
print('Regime tracking in paper trader:')
print('  Position: entry_regime stored at entry time')
print('  Trade: entry_regime + exit_regime + regime_changed flag')
print('  Sizing: regime passed to PositionSizer for regime-conditional sizing')
print('  Stops: entry_regime passed to StopLossManager for regime-adjusted stops')
"
```

---

### T5: Add equity curve tracking and risk metrics to state

**What:** Track daily equity values and compute running risk metrics.

**Files:**
- `autopilot/paper_trader.py` — add equity curve to state

**Implementation notes:**
```python
# In run_cycle(), after computing current equity:
if 'equity_curve' not in state:
    state['equity_curve'] = []

state['equity_curve'].append({
    'date': as_of_str,
    'equity': current_equity,
    'cash': state['cash'],
    'positions_value': current_equity - state['cash'],
    'n_positions': len(state['positions']),
    'daily_pnl': daily_pnl,
    'drawdown': dd_status.current_drawdown if dd_status else 0,
    'dd_state': dd_status.state.name if dd_status else 'NORMAL',
})

# Cap equity curve to last 2520 entries (10 years of daily data)
if len(state['equity_curve']) > 2520:
    state['equity_curve'] = state['equity_curve'][-2520:]

# Add summary risk metrics to run_cycle return value
if len(state['equity_curve']) >= 20:
    equity_series = pd.Series([e['equity'] for e in state['equity_curve']])
    returns_series = equity_series.pct_change().dropna()
    
    summary['risk_metrics'] = {
        'sharpe': returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0,
        'max_drawdown': dd_status.current_drawdown if dd_status else 0,
        'volatility': returns_series.std() * np.sqrt(252),
        'calmar': (returns_series.mean() * 252) / abs(dd_status.current_drawdown) if dd_status and dd_status.current_drawdown < 0 else 0,
        'win_rate': sum(1 for r in returns_series if r > 0) / len(returns_series),
    }
```

**Verify:**
```bash
python -c "
print('Equity curve tracking:')
print('  Daily: equity, cash, positions_value, n_positions, daily_pnl, drawdown')
print('  Risk metrics: Sharpe, max_drawdown, volatility, Calmar, win_rate')
print('  Capped at 2520 entries (10 years)')
"
```

---

### T6: Test paper trader integration

**What:** Tests for drawdown protection, stop losses, risk checks, regime tracking, and equity curve.

**Files:**
- `tests/test_paper_trader_integration.py` — new test file

**Implementation notes:**
- Test cases:
  1. `test_drawdown_blocks_entries` — After significant losses, new entries are blocked
  2. `test_drawdown_force_liquidate` — Critical drawdown → all positions closed
  3. `test_stop_loss_triggers_exit` — Hard stop at -8% triggers exit
  4. `test_trailing_stop_triggers` — Trailing stop after position was in profit
  5. `test_risk_manager_blocks_correlated` — Highly correlated position rejected
  6. `test_regime_stored_in_position` — Position record has entry_regime field
  7. `test_regime_stored_in_trade` — Trade record has entry_regime and exit_regime
  8. `test_regime_passed_to_sizer` — Position sizer receives non-None regime
  9. `test_equity_curve_tracked` — State has equity_curve with daily entries
  10. `test_risk_metrics_computed` — Summary includes Sharpe, drawdown, volatility

**Verify:**
```bash
python -m pytest tests/test_paper_trader_integration.py -v
```

---

## Validation

### Acceptance criteria
1. DrawdownController called every cycle — sizing reduced during drawdowns, entries blocked during CAUTION/CRITICAL
2. StopLossManager evaluates every open position — ATR stops, trailing stops, hard stops all functional
3. PortfolioRiskManager checks every new entry — correlated/concentrated positions rejected
4. Regime stored in every position and trade record
5. Equity curve tracked with at least 20 days of history after 20 cycles
6. Risk metrics (Sharpe, drawdown, vol) available in run_cycle summary

### Rollback plan
- DrawdownController: remove update() call, set size_multiplier=1.0, allow_new=True
- StopLossManager: remove evaluate() call, revert to signal-decay-only exits
- PortfolioRiskManager: remove check_new_position() call, accept all entries
- All changes are additive — removing them restores original behavior
