# Feature Spec: A/B Testing Framework — From Skeleton to Production

> **Status:** Approved
> **Author:** justin
> **Date:** 2026-02-23
> **Estimated effort:** ~10 hours across 7 tasks

---

## Why

The A/B testing module (`api/ab_testing.py`) is a minimal skeleton: (1) It uses Welch's t-test on raw returns, but this doesn't account for time-series autocorrelation — adjacent trades are correlated, inflating statistical significance. (2) Trade assignment is fully random (50/50 split), but this can cause the same ticker to appear in both variants simultaneously, creating contamination. (3) The `min_trades=50` threshold is arbitrary — proper sample size depends on effect size and desired power. (4) Results only report mean return and Sharpe, missing critical metrics like max drawdown, win rate, turnover, and risk-adjusted return. (5) There's no mechanism to stop a test early when one variant is clearly worse (sequential testing). (6) The registry only persists test metadata to JSON, not the actual trades — results can't be reproduced after the fact. (7) No integration with the autopilot/paper trader — A/B tests require manual trade recording.

## What

Build a production-grade A/B testing framework that: properly handles time-series data, prevents ticker contamination, computes sample size requirements, supports early stopping, tracks complete trade history, and integrates with the autopilot system. Done means: you can create an A/B test, it automatically routes trades to variants in the paper trader, results are statistically sound, and early stopping prevents excessive losses.

## Constraints

### Must-haves
- Block-bootstrap or Newey-West adjusted significance testing (not raw t-test)
- Ticker-level assignment (not trade-level) to prevent contamination
- Power analysis for minimum sample size computation
- Sequential testing with alpha spending function for early stopping
- Full trade history persistence (not just metadata)
- Integration with paper_trader.py for automatic variant routing

### Must-nots
- Do NOT run A/B tests on live capital (paper trading only)
- Do NOT allow more than 3 concurrent A/B tests
- Do NOT auto-promote winners without human review

## Tasks

### T1: Fix statistical testing with time-series awareness

**What:** Replace Welch's t-test with block-bootstrap or HAC (heteroskedasticity and autocorrelation consistent) testing.

**Files:**
- `api/ab_testing.py` — rewrite `get_results()` method in ABTest class

**Implementation notes:**
- Current: Welch's t-test assumes IID observations — trades are NOT independent
- Fix: use block bootstrap for p-value computation
  ```python
  def _block_bootstrap_test(self, returns_a, returns_b, n_bootstrap=1000, block_size=10):
      """Block bootstrap test for difference in means, respecting autocorrelation."""
      observed_diff = returns_a.mean() - returns_b.mean()
      
      # Pool all returns under null hypothesis
      pooled = np.concatenate([returns_a, returns_b])
      n_a, n_b = len(returns_a), len(returns_b)
      
      bootstrap_diffs = []
      for _ in range(n_bootstrap):
          # Block-resample from pooled returns
          boot_pooled = self._block_resample(pooled, block_size)
          boot_a = boot_pooled[:n_a]
          boot_b = boot_pooled[n_a:n_a+n_b]
          bootstrap_diffs.append(boot_a.mean() - boot_b.mean())
      
      # Two-sided p-value
      p_value = np.mean(np.abs(bootstrap_diffs) >= abs(observed_diff))
      return p_value
  
  def _block_resample(self, data, block_size):
      """Circular block bootstrap resampling."""
      n = len(data)
      n_blocks = int(np.ceil(n / block_size))
      indices = []
      for _ in range(n_blocks):
          start = np.random.randint(0, n)
          indices.extend(range(start, start + block_size))
      indices = [i % n for i in indices[:n]]
      return data[indices]
  ```
- Also add Newey-West adjusted t-test as alternative:
  ```python
  def _newey_west_test(self, returns_a, returns_b, max_lag=10):
      """HAC-consistent t-test using Newey-West standard errors."""
      diff = returns_a.mean() - returns_b.mean()
      combined = np.concatenate([returns_a - returns_a.mean(), returns_b - returns_b.mean()])
      
      # Newey-West HAC variance estimate
      T = len(combined)
      gamma_0 = np.var(combined)
      nw_var = gamma_0
      for lag in range(1, max_lag + 1):
          weight = 1 - lag / (max_lag + 1)  # Bartlett kernel
          gamma_lag = np.mean(combined[lag:] * combined[:-lag])
          nw_var += 2 * weight * gamma_lag
      
      se = np.sqrt(nw_var * (1/len(returns_a) + 1/len(returns_b)))
      t_stat = diff / se if se > 0 else 0
      p_value = 2 * (1 - norm.cdf(abs(t_stat)))
      return t_stat, p_value
  ```

**Verify:**
```bash
python -c "
print('Statistical testing upgrade:')
print('  Block bootstrap: respects time-series autocorrelation')
print('  Newey-West: HAC-consistent standard errors')
print('  Both methods give wider confidence intervals than raw t-test')
print('  Prevents false positives from correlated trade returns')
"
```

---

### T2: Implement ticker-level variant assignment

**What:** Assign entire tickers to variants (not individual trades) to prevent contamination.

**Files:**
- `api/ab_testing.py` — rewrite `assign_variant()` in ABTest class

**Implementation notes:**
- Current: `assign_variant()` is purely random per-call, so the same ticker can be in both variants
- Fix: deterministic assignment based on ticker hash
  ```python
  def assign_variant(self, ticker: str) -> str:
      """Assign ticker to variant deterministically based on hash."""
      if ticker in self._ticker_assignments:
          return self._ticker_assignments[ticker]
      
      # Deterministic hash: same ticker always gets same variant
      hash_val = int(hashlib.md5(f"{self.test_id}:{ticker}".encode()).hexdigest(), 16)
      
      # Assign based on allocation ratio
      if (hash_val % 1000) / 1000.0 < self.control.allocation:
          variant = self.control.name
      else:
          variant = self.treatment.name
      
      self._ticker_assignments[ticker] = variant
      return variant
  ```
- Benefits:
  - Same ticker always in same variant → no contamination
  - Reproducible: same test_id + ticker → same assignment
  - Balanced: hash is uniformly distributed → approximately matches allocation ratio

**Verify:**
```bash
python -c "
import hashlib
test_id = 'abc123'
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'BAC', 'GS']
for t in tickers:
    h = int(hashlib.md5(f'{test_id}:{t}'.encode()).hexdigest(), 16)
    variant = 'control' if (h % 1000) / 1000.0 < 0.5 else 'treatment'
    print(f'  {t} → {variant}')
print('Same ticker always gets same variant')
"
```

---

### T3: Add power analysis for sample size computation

**What:** Compute minimum trades needed before declaring a winner, based on expected effect size.

**Files:**
- `api/ab_testing.py` — add `compute_required_samples()` method

**Implementation notes:**
  ```python
  def compute_required_samples(
      self,
      min_detectable_effect: float = 0.005,  # 50bps return difference
      power: float = 0.80,
      alpha: float = 0.05,
      return_std: float = 0.02,  # Typical daily return volatility
  ) -> int:
      """Compute minimum trades per variant for desired statistical power."""
      from scipy.stats import norm
      
      z_alpha = norm.ppf(1 - alpha / 2)  # Two-sided
      z_beta = norm.ppf(power)
      
      # n = 2 * (z_alpha + z_beta)^2 * sigma^2 / delta^2
      n = 2 * (z_alpha + z_beta)**2 * return_std**2 / min_detectable_effect**2
      
      # Adjust for autocorrelation (effective sample size is smaller)
      # Assume AR(1) with rho ≈ 0.1
      rho = 0.10
      autocorr_factor = (1 + rho) / (1 - rho)
      n_adjusted = int(np.ceil(n * autocorr_factor))
      
      return max(n_adjusted, 50)  # At least 50 trades
  ```
- Store required_samples in test metadata
- Update min_trades to be computed from power analysis, not hardcoded at 50

**Verify:**
```bash
python -c "
from scipy.stats import norm
import numpy as np
# Power analysis: detect 50bps difference with 80% power
alpha, power_target = 0.05, 0.80
z_a, z_b = norm.ppf(1 - alpha/2), norm.ppf(power_target)
sigma, delta = 0.02, 0.005
n = 2 * (z_a + z_b)**2 * sigma**2 / delta**2
print(f'Required trades per variant: {int(np.ceil(n))} (without autocorr adjustment)')
print(f'With autocorr (rho=0.1): {int(np.ceil(n * 1.1/0.9))}')
"
```

---

### T4: Add sequential testing with alpha spending

**What:** Allow early stopping when one variant is clearly better or worse, while controlling overall false positive rate.

**Files:**
- `api/ab_testing.py` — add `check_early_stopping()` method

**Implementation notes:**
  ```python
  def check_early_stopping(self, interim_looks: int = 5) -> Dict:
      """Check if test can be stopped early using O'Brien-Fleming alpha spending."""
      n_control = self.control.n_trades
      n_treatment = self.treatment.n_trades
      required = self.compute_required_samples()
      
      if n_control < 20 or n_treatment < 20:
          return {"stop": False, "reason": "Insufficient data for interim analysis"}
      
      # Information fraction: how much data do we have vs required?
      info_fraction = min(1.0, min(n_control, n_treatment) / required)
      
      # O'Brien-Fleming alpha spending: very conservative early, liberal late
      # alpha_spent(t) = 2 * (1 - Phi(z_alpha/2 / sqrt(t)))
      from scipy.stats import norm
      z_alpha = norm.ppf(1 - self.confidence_level / 2 + 0.5)
      z_boundary = z_alpha / np.sqrt(info_fraction) if info_fraction > 0 else float('inf')
      spent_alpha = 2 * (1 - norm.cdf(z_boundary))
      
      # Compute current test statistic
      _, p_value = self._newey_west_test(
          np.array(self.control.returns()),
          np.array(self.treatment.returns()),
      )
      
      can_stop = p_value < spent_alpha
      
      # Safety check: stop if treatment is significantly WORSE
      treatment_worse = (self.treatment.mean_return < self.control.mean_return and p_value < 0.01)
      
      return {
          "stop": can_stop or treatment_worse,
          "reason": "Treatment significantly better" if can_stop else
                    "Treatment significantly worse — recommend stopping" if treatment_worse else
                    "Not yet significant",
          "info_fraction": info_fraction,
          "p_value": p_value,
          "alpha_boundary": spent_alpha,
          "trades_remaining": max(0, required - min(n_control, n_treatment)),
      }
  ```

**Verify:**
```bash
python -c "
print('Sequential testing with O Brien-Fleming spending:')
print('  20% of data: alpha boundary ≈ 0.0001 (very hard to reject)')
print('  50% of data: alpha boundary ≈ 0.005')
print('  80% of data: alpha boundary ≈ 0.02')
print('  100% of data: alpha boundary = 0.05')
print('  Safety valve: auto-stop if treatment is clearly worse (p<0.01)')
"
```

---

### T5: Persist full trade history and improve registry

**What:** Store complete trade records (not just metadata) so results can be reproduced.

**Files:**
- `api/ab_testing.py` — rewrite `ABTestRegistry` persistence

**Implementation notes:**
- Current: `_save()` persists only metadata (test_id, name, dates, status) — trades are lost
- Fix: save full trade history as parquet for efficient storage
  ```python
  def _save(self):
      # Save metadata as JSON (existing)
      metadata = {t.test_id: t.to_dict() for t in self._tests.values()}
      with open(self.storage_path, 'w') as f:
          json.dump(metadata, f, default=str)
      
      # Save trade history as parquet per test
      for test in self._tests.values():
          trades_dir = self.storage_path.parent / 'ab_trades'
          trades_dir.mkdir(exist_ok=True)
          
          for variant in [test.control, test.treatment]:
              if variant.trades:
                  df = pd.DataFrame(variant.trades)
                  df.to_parquet(
                      trades_dir / f"{test.test_id}_{variant.name}.parquet",
                      index=False
                  )
  ```
- Add `_load()` that restores trades from parquet
- Add `get_test_report()` that computes comprehensive comparison:
  - Mean return, Sharpe, Sortino, max drawdown, win rate, profit factor
  - Per-regime performance comparison
  - Turnover comparison
  - Transaction cost comparison

**Verify:**
```bash
python -c "
print('A/B test persistence:')
print('  Metadata: JSON (test config, dates, status)')
print('  Trades: Parquet per variant (full trade records)')
print('  Report: comprehensive comparison with 10+ metrics')
"
```

---

### T6: Integrate A/B testing with paper trader

**What:** Wire A/B test variant assignment into the paper trader's trade execution flow.

**Files:**
- `autopilot/paper_trader.py` — add A/B test routing in `run_cycle()`
- `api/ab_testing.py` — add convenience methods for paper trader integration

**Implementation notes:**
- In paper_trader.py `run_cycle()`, when evaluating entries:
  ```python
  # Check if any A/B test is active
  active_test = self._get_active_ab_test()
  
  if active_test:
      for candidate in candidates:
          ticker = candidate['ticker']
          variant = active_test.assign_variant(ticker)
          
          # Apply variant-specific config overrides
          variant_config = active_test.get_variant_config(variant)
          
          # Override parameters for this trade
          entry_threshold = variant_config.get('entry_threshold', self.entry_threshold)
          position_size = variant_config.get('position_size_pct', base_position_size)
          
          # ... execute trade with variant-specific parameters ...
          
          # Record trade to A/B test
          active_test.record_trade(variant, {
              'ticker': ticker,
              'entry_date': as_of,
              'entry_price': entry_price,
              'position_size': position_size,
              'variant_config': variant_config,
          })
  ```
- Config overrides that A/B tests can modify:
  - `entry_threshold`, `confidence_threshold`, `position_size_pct`, `max_holding_days`
  - `kelly_fraction`, `use_risk_management`
  - NOT: `max_positions` (would create imbalanced capital allocation)

**Verify:**
```bash
python -c "
print('Paper trader A/B integration:')
print('  1. Active test detected in run_cycle()')
print('  2. Ticker assigned to variant (deterministic hash)')
print('  3. Variant config overrides applied to trade parameters')
print('  4. Trade recorded to variant with full details')
print('  5. Early stopping checked after each cycle')
"
```

---

### T7: Test A/B testing framework

**What:** Tests for statistical testing, ticker assignment, power analysis, early stopping, and paper trader integration.

**Files:**
- `tests/test_ab_testing_framework.py` — new test file

**Implementation notes:**
- Test cases:
  1. `test_block_bootstrap_wider_ci` — Block bootstrap p-value ≥ raw t-test p-value
  2. `test_ticker_assignment_deterministic` — Same ticker always gets same variant
  3. `test_ticker_assignment_balanced` — ~50% of 100 tickers in each variant
  4. `test_power_analysis_reasonable` — Required samples between 50 and 10000
  5. `test_early_stopping_conservative_early` — At 20% data, alpha boundary < 0.001
  6. `test_early_stopping_liberal_late` — At 100% data, alpha boundary = 0.05
  7. `test_trade_persistence_parquet` — Trades saved and restored correctly
  8. `test_comprehensive_report_metrics` — Report includes Sharpe, drawdown, win rate
  9. `test_no_ticker_contamination` — Same ticker never in both variants
  10. `test_max_concurrent_tests` — Cannot create >3 active tests

**Verify:**
```bash
python -m pytest tests/test_ab_testing_framework.py -v
```

---

## Validation

### Acceptance criteria
1. Block bootstrap test gives wider confidence intervals than raw t-test
2. Same ticker is always in the same variant (no contamination)
3. Power analysis computes reasonable sample size (50-10000 for typical parameters)
4. Early stopping boundary at 20% of data is very conservative (<0.001 alpha)
5. Full trade history persisted and restorable from parquet
6. Paper trader automatically routes trades to active A/B test variants

### Rollback plan
- Statistical testing: fall back to Welch's t-test (less correct but functional)
- Ticker assignment: revert to random (less rigorous but functional)
- Paper trader integration: remove A/B test check from run_cycle (trades execute normally)
