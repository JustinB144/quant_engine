# Feature Spec: Backtest Execution Realism & Validation Enforcement

> **Status:** Approved
> **Author:** justin
> **Date:** 2026-02-23
> **Estimated effort:** ~10 hours across 7 tasks

---

## Why

The backtest engine has several execution realism issues that inflate apparent performance: (1) Entry uses Close-on-signal-day pricing when it should use next-bar Open — the `_simulate_entry()` at `signal_idx + 1` correctly uses the Open of the next bar in `_process_signals()` but `_process_signals_risk_managed()` uses `signal_idx` directly, creating an inconsistency. (2) Almgren-Chriss `risk_aversion=1e-6` is nearly risk-neutral, meaning the optimal execution model barely penalizes market impact — real institutional values are 1e-3 to 1e-1. (3) Exit simulation uses `force_full=True` which ignores volume constraints — a $500K position can't exit in one bar of an illiquid stock. (4) Validation tests (CPCV, SPA, Deflated Sharpe, PBO) are computed but NOT enforced in autopilot — the promotion gate can pass overfitted strategies if `REQUIRE_STATISTICAL_VALIDATION=False` (the default). (5) DSR p-value is one-sided, so a negative-Sharpe strategy can pass. (6) PBO measures "degradation rate" rather than the actual probability of backtest overfitting.

## What

Fix execution realism so backtests don't overstate returns, and enforce validation so only statistically validated strategies get promoted. Done means: entry/exit timing is consistent and conservative, Almgren-Chriss uses realistic parameters, exits respect volume constraints, and autopilot requires CPCV/SPA/DSR validation for promotion.

## Constraints

### Must-haves
- Consistent entry timing: always next-bar Open after signal
- Almgren-Chriss risk_aversion calibrated from market data (not 1e-6)
- Exit simulation respects volume: partial fills over multiple bars if needed
- Validation required for autopilot promotion (not optional)
- DSR two-sided or at least reject negative Sharpe

### Must-nots
- Do NOT make execution simulation 10x slower (keep per-trade <1ms)
- Do NOT change simple mode (_process_signals) — it's for quick research
- Do NOT remove the ability to run backtests without validation (research use case)

## Tasks

### T1: Fix entry timing consistency in risk-managed mode

**What:** Ensure `_process_signals_risk_managed()` uses next-bar Open for entry, matching simple mode.

**Files:**
- `backtest/engine.py` — fix signal processing in `_process_signals_risk_managed()` (lines 903-1024)

**Implementation notes:**
- Current issue: risk-managed mode processes signals on the signal date itself
- Should advance entry_idx by 1 bar to avoid using same-bar close information
- The signal is generated from data available at close of bar t, so entry should be at open of bar t+1
- Fix: `entry_idx = signal_idx + 1` (matching simple mode at line 620)
- Edge case: if entry_idx >= len(data), skip the signal (end of data)

**Verify:**
```bash
python -c "
# Verify that entry uses next-bar Open, not same-bar Close
print('Entry timing: signal at bar t close → entry at bar t+1 open')
print('This prevents using same-bar information for entry pricing')
"
```

---

### T2: Calibrate Almgren-Chriss risk aversion from market data

**What:** Replace hardcoded `risk_aversion=1e-6` with empirically calibrated values.

**Files:**
- `backtest/execution.py` — update default parameters
- `config.py` — add ALMGREN_CHRISS_RISK_AVERSION config constant

**Implementation notes:**
- Current: `risk_aversion=1e-6` — this makes the optimizer nearly risk-neutral, meaning it doesn't penalize market impact
- Academic literature suggests:
  - Passive institutional: 1e-3 to 1e-2
  - Active institutional: 1e-2 to 1e-1
  - Aggressive: 1e-1 to 1.0
- Default should be 1e-2 (moderate institutional)
- Add to config.py:
  ```python
  # ACTIVE — Almgren-Chriss optimal execution risk aversion.
  # Higher values = more conservative execution (split orders more).
  # Range: 1e-3 (passive) to 1e-1 (aggressive risk aversion).
  # Default 0.01 matches moderate institutional execution.
  ALMGREN_CHRISS_RISK_AVERSION = 0.01
  ```
- Also fix impact coefficient: current default 25 bps may be too low for small-cap
  - Make it dependent on dollar volume: lower volume → higher impact

**Verify:**
```bash
python -c "
from quant_engine.config import ALMGREN_CHRISS_RISK_AVERSION
print(f'AC risk aversion: {ALMGREN_CHRISS_RISK_AVERSION}')
assert ALMGREN_CHRISS_RISK_AVERSION >= 1e-3, 'Too low — nearly risk-neutral'
assert ALMGREN_CHRISS_RISK_AVERSION <= 1.0, 'Too high — would never trade'
print('Almgren-Chriss calibrated')
"
```

---

### T3: Fix exit simulation to respect volume constraints

**What:** Remove `force_full=True` default on exits and implement multi-bar exit when position exceeds daily volume capacity.

**Files:**
- `backtest/engine.py` — fix `_simulate_exit()` (lines 301-359)

**Implementation notes:**
- Current: `force_full=True` on exits means even illiquid positions exit at 100% fill
- Real constraint: if position_notional > max_participation * daily_dollar_volume, you can't exit in one bar
- Fix:
  ```python
  def _simulate_exit(self, data, exit_idx, shares, entry_price, position_size, regime=None):
      # Calculate how much we can realistically exit this bar
      daily_volume = data['Volume'].iloc[exit_idx] * data['Close'].iloc[exit_idx]
      max_exit_notional = daily_volume * self.exec_model.max_participation_rate
      position_notional = shares * data['Close'].iloc[exit_idx]
      
      if position_notional > max_exit_notional:
          # Partial fill — can only exit what the market absorbs
          fill_ratio = max_exit_notional / position_notional
          logger.debug("Partial exit: %.0f%% of position (volume constraint)", fill_ratio * 100)
      else:
          fill_ratio = 1.0
      
      result = self.exec_model.simulate(
          side='sell',
          reference_price=data['Close'].iloc[exit_idx],
          daily_volume=daily_volume,
          desired_notional_usd=position_notional * fill_ratio,
          force_full=False,  # Respect volume constraints
          ...
      )
      return result
  ```
- For multi-bar exits: track remaining shares and exit over subsequent bars
- This is a significant change — add a `residual_positions` tracker in risk-managed mode

**Verify:**
```bash
python -c "
# Conceptual check: large position in illiquid stock should partial fill
print('Exit realism: position_notional > max_participation * daily_volume → partial fill')
print('Example: $500K position, $2M daily volume, 2% participation = $40K/bar exit')
print('Would take ~13 bars to fully exit')
"
```

---

### T4: Enforce validation in autopilot promotion gate

**What:** Make statistical validation mandatory for strategy promotion, not optional.

**Files:**
- `backtest/advanced_validation.py` — update `run_advanced_validation()` (lines 431-509)
- `autopilot/` — find promotion gate and enforce validation

**Implementation notes:**
- Current: `REQUIRE_STATISTICAL_VALIDATION` config flag defaults to False
- Fix: Change default to True, and add minimum validation requirements:
  ```python
  # In promotion gate logic:
  def evaluate_for_promotion(backtest_result, validation_result):
      # Mandatory checks (cannot be disabled):
      if validation_result is None:
          return {"promoted": False, "reason": "No validation run"}
      
      # 1. Deflated Sharpe must be significant
      if not validation_result.dsr.is_significant:
          return {"promoted": False, "reason": f"DSR p={validation_result.dsr.p_value:.3f} > 0.05"}
      
      # 2. Monte Carlo must pass
      if not validation_result.mc.is_significant:
          return {"promoted": False, "reason": f"MC p={validation_result.mc.p_value:.3f} > 0.05"}
      
      # 3. PBO must not indicate overfitting
      if validation_result.pbo and validation_result.pbo.is_overfit:
          return {"promoted": False, "reason": f"PBO={validation_result.pbo.pbo:.2f} > 0.5"}
      
      # 4. Sharpe must be positive (prevent negative-Sharpe promotion)
      if validation_result.dsr.observed_sharpe <= 0:
          return {"promoted": False, "reason": f"Sharpe={validation_result.dsr.observed_sharpe:.3f} <= 0"}
      
      return {"promoted": True, "reason": "All validation checks passed"}
  ```

**Verify:**
```bash
python -c "
print('Validation enforcement: DSR + MC + PBO all required for promotion')
print('Negative Sharpe strategies always rejected')
print('No more REQUIRE_STATISTICAL_VALIDATION=False bypass')
"
```

---

### T5: Fix Deflated Sharpe Ratio to reject negative Sharpe

**What:** Make DSR two-sided and explicitly reject negative-Sharpe strategies.

**Files:**
- `backtest/advanced_validation.py` — fix `deflated_sharpe_ratio()` (lines 94-160)

**Implementation notes:**
- Current: one-sided p-value `1 - norm.cdf(deflated)` — negative Sharpe can pass if deflated < 0 and p > 0.5
- Fix: add explicit check
  ```python
  def deflated_sharpe_ratio(returns, n_trials, ...):
      # ... existing calculation ...
      
      # Reject negative Sharpe outright — no statistical test needed
      if observed_sharpe <= 0:
          return DeflatedSharpeResult(
              observed_sharpe=observed_sharpe,
              deflated_sharpe=-999,
              expected_max_sharpe=expected_max,
              n_trials=n_trials,
              p_value=1.0,
              is_significant=False,
          )
      
      # ... rest of existing calculation ...
  ```

**Verify:**
```bash
python -c "
import numpy as np
from quant_engine.backtest.advanced_validation import deflated_sharpe_ratio
# Negative returns should always fail
neg_returns = np.random.randn(500) * 0.01 - 0.001  # Slightly negative mean
result = deflated_sharpe_ratio(neg_returns, n_trials=10)
assert not result.is_significant, 'Negative Sharpe should not pass DSR'
print(f'Negative Sharpe DSR: significant={result.is_significant}, p={result.p_value}')
"
```

---

### T6: Fix PBO methodology

**What:** Ensure PBO measures actual probability of overfitting, not just degradation rate.

**Files:**
- `backtest/advanced_validation.py` — fix `probability_of_backtest_overfitting()` (lines 163-259)

**Implementation notes:**
- Current PBO implementation uses CSCV but measures "degradation_rate" (fraction where IS-best underperforms)
- This is actually correct for PBO — the degradation rate IS the PBO metric (Bailey et al. 2017)
- However, the threshold is wrong: `is_overfit = pbo > 0.5` is too lenient
  - PBO > 0.5 means >50% chance the best IS strategy underperforms OOS
  - Academic standard: PBO > 0.4 should flag concern, PBO > 0.5 should reject
- Fix: tighten threshold and add PBO logit distribution
  ```python
  # PBO logit: logit(pbo) for confidence interval
  # logit > 0 means more likely than not that strategy is overfit
  pbo_logit = np.log(pbo / (1 - pbo)) if 0 < pbo < 1 else float('inf')
  is_overfit = pbo > 0.45 or pbo_logit > 0
  ```
- Also fix: minimum combinations (current max=100 is too low for reliable PBO)
  - Increase to max_combinations=200 for better statistical power

**Verify:**
```bash
python -c "
print('PBO thresholds:')
print('  PBO < 0.35: PASS (low overfitting risk)')
print('  PBO 0.35-0.45: WARN (moderate risk)')
print('  PBO > 0.45: FAIL (high overfitting risk)')
"
```

---

### T7: Test backtest execution realism fixes

**What:** Tests for entry timing, exit volume constraints, validation enforcement, and DSR/PBO fixes.

**Files:**
- `tests/test_backtest_realism.py` — new test file

**Implementation notes:**
- Test cases:
  1. `test_entry_timing_uses_next_bar_open` — Both simple and risk-managed modes use t+1 Open
  2. `test_exit_respects_volume` — Large position in illiquid stock → fill_ratio < 1.0
  3. `test_almgren_chriss_risk_aversion` — Default risk_aversion is 0.01, not 1e-6
  4. `test_negative_sharpe_fails_dsr` — Negative Sharpe → DSR p=1.0, is_significant=False
  5. `test_pbo_rejects_overfit` — PBO > 0.45 → is_overfit=True
  6. `test_validation_required_for_promotion` — No validation → promotion rejected
  7. `test_partial_exit_multi_bar` — Residual shares tracked across bars

**Verify:**
```bash
python -m pytest tests/test_backtest_realism.py -v
```

---

## Validation

### Acceptance criteria
1. Entry timing: always next-bar Open in both simple and risk-managed modes
2. Almgren-Chriss risk_aversion ≥ 0.001 (not 1e-6)
3. Exits respect volume: fill_ratio < 1.0 when position > participation * daily_volume
4. Negative Sharpe strategies always fail DSR
5. PBO threshold tightened to 0.45
6. Validation mandatory for autopilot promotion

### Rollback plan
- Entry timing: revert signal_idx+1 to signal_idx (one-line change)
- AC risk aversion: revert config constant
- Exit volume: add force_full=True parameter back
- Validation enforcement: re-add config flag to disable
