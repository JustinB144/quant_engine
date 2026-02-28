# Feature Spec: Risk System Improvements — Covariance, Stress Testing, Attribution, Metrics

> **Status:** Approved
> **Author:** justin
> **Date:** 2026-02-23
> **Estimated effort:** ~12 hours across 8 tasks

---

## Why

The risk management system has several gaps: (1) CovarianceEstimator uses static Ledoit-Wolf shrinkage with a fixed 0.15 shrinkage intensity — it should use the data-driven shrinkage from sklearn's LedoitWolf which estimates optimal shrinkage. (2) There's no exponential weighting for covariance — recent correlations matter more than year-old correlations, especially during regime changes. (3) RiskMetrics computes VaR/CVaR using historical percentiles, which underestimates tail risk during calm periods — should offer parametric and Monte Carlo alternatives. (4) The stress testing module (if it exists) doesn't model correlation breakdown — correlations spike toward 1.0 during crises, which is when diversification fails most. (5) Portfolio attribution doesn't exist — there's no way to decompose P&L into alpha, factor exposure, and residual components. (6) The drawdown controller's recovery ramp from 0.25 to 1.0 is linear — it should be slower at the start of recovery (more cautious). (7) Stop loss doesn't account for bid-ask spread — the stop price should be adjusted for expected spread to avoid false triggers on spread-widening events.

## What

Upgrade covariance estimation with dynamic shrinkage and exponential weighting, add parametric/Monte Carlo VaR, implement correlation-stress scenarios, add Brinson-style attribution, improve drawdown recovery, and add spread-aware stops. Done means: risk estimates are more accurate, stress tests model realistic crisis dynamics, and portfolio performance is fully decomposable.

## Constraints

### Must-haves
- Covariance: data-driven shrinkage via sklearn LedoitWolf (not fixed 0.15)
- Covariance: exponential weighting option with configurable half-life
- VaR: parametric (normal) and historical, clearly labeled
- Stress test: correlation spike scenario (all correlations → 0.9)
- Attribution: decompose returns into market, factor, and alpha components
- Stop loss: bid-ask spread buffer on stop prices

### Must-nots
- Do NOT remove Ledoit-Wolf as an option (keep as fallback)
- Do NOT make covariance estimation take >1 second for 100 assets
- Do NOT add complex attribution models that require factor model estimation (start simple)

## Tasks

### T1: Upgrade covariance estimation with dynamic shrinkage

**What:** Use sklearn's data-driven shrinkage intensity instead of fixed 0.15.

**Files:**
- `risk/covariance.py` — update `_estimate_values()` (lines 79-117)

**Implementation notes:**
- Current (lines 91-101): tries sklearn LedoitWolf but then applies ADDITIONAL fixed shrinkage on top
  ```python
  # Current (double-shrinks):
  lw = LedoitWolf().fit(returns)
  cov = lw.covariance_  # Already shrunk by sklearn
  cov = (1 - self.shrinkage) * cov + self.shrinkage * np.diag(np.diag(cov))  # Shrinks AGAIN
  ```
- Fix: use sklearn's optimal shrinkage, don't add extra
  ```python
  def _estimate_values(self, returns):
      if self.method == 'ledoit_wolf':
          try:
              lw = LedoitWolf().fit(returns)
              cov = lw.covariance_
              shrinkage_used = lw.shrinkage_  # Optimal data-driven shrinkage
              logger.debug("Ledoit-Wolf shrinkage: %.4f", shrinkage_used)
          except Exception:
              cov = np.cov(returns.T)
              shrinkage_used = self.shrinkage  # Fallback to manual
              cov = (1 - shrinkage_used) * cov + shrinkage_used * np.diag(np.diag(cov))
      
      # PSD enforcement
      eigvals, eigvecs = np.linalg.eigh(cov)
      eigvals = np.maximum(eigvals, 1e-10)
      cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
      
      return cov * self.annualization
  ```

**Verify:**
```bash
python -c "
from quant_engine.risk.covariance import CovarianceEstimator
import numpy as np, pandas as pd
np.random.seed(42)
returns = pd.DataFrame(np.random.randn(252, 10) * 0.01, columns=[f'A{i}' for i in range(10)])
est = CovarianceEstimator(method='ledoit_wolf')
result = est.estimate(returns)
print(f'Covariance shape: {result.covariance.shape}')
print(f'All eigenvalues positive: {np.all(np.linalg.eigvalsh(result.covariance) > 0)}')
"
```

---

### T2: Add exponential weighting option for covariance

**What:** Weight recent observations more heavily, with configurable half-life.

**Files:**
- `risk/covariance.py` — add `exponential_weighted` method option

**Implementation notes:**
```python
def _estimate_values(self, returns):
    if self.method == 'ewma':
        # Exponentially weighted covariance
        half_life = getattr(self, 'half_life', 60)  # Default 60-day half-life
        decay = 1 - np.log(2) / half_life
        
        # Compute EWMA covariance using pandas
        ewma_cov = returns.ewm(halflife=half_life).cov().iloc[-len(returns.columns):]
        cov = ewma_cov.values
        
        # PSD enforcement
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-10)
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        return cov * self.annualization
```
- Add `half_life` parameter to constructor
- Add to config.py: `COVARIANCE_HALF_LIFE = 60`
- Recommendation: use EWMA for risk management (more responsive to regime changes), Ledoit-Wolf for optimization (more stable)

**Verify:**
```bash
python -c "
import numpy as np, pandas as pd
np.random.seed(42)
# Simulate regime change: first 200 bars low vol, last 52 bars high vol
returns = pd.DataFrame(
    np.vstack([np.random.randn(200, 5) * 0.01, np.random.randn(52, 5) * 0.03]),
    columns=[f'A{i}' for i in range(5)]
)
# EWMA should show higher recent vol than equal-weight
ewma_cov = returns.ewm(halflife=60).cov().iloc[-5:]
equal_cov = returns.cov()
print(f'Equal-weight vol: {np.sqrt(np.diag(equal_cov.values) * 252)[:3]}')
print(f'EWMA vol (recent): should be higher due to regime change')
"
```

---

### T3: Add parametric VaR alongside historical

**What:** Compute parametric (normal) VaR and Monte Carlo VaR in addition to historical.

**Files:**
- `risk/metrics.py` — add VaR methods to RiskMetrics

**Implementation notes:**
```python
def compute_full_report(self, returns, holding_days=10, trades=None, method='historical'):
    # Historical VaR (existing)
    var_95_hist = np.percentile(returns, 5)
    var_99_hist = np.percentile(returns, 1)
    cvar_95_hist = returns[returns <= var_95_hist].mean() if len(returns[returns <= var_95_hist]) > 0 else var_95_hist
    
    # Parametric VaR (assume normal distribution)
    from scipy.stats import norm
    mu, sigma = returns.mean(), returns.std()
    var_95_param = mu + sigma * norm.ppf(0.05)
    var_99_param = mu + sigma * norm.ppf(0.01)
    
    # Cornish-Fisher VaR (adjusts for skewness and kurtosis)
    skew = returns.skew() if hasattr(returns, 'skew') else 0
    kurt = returns.kurtosis() if hasattr(returns, 'kurtosis') else 0
    z_95 = norm.ppf(0.05)
    z_cf = z_95 + (z_95**2 - 1)/6 * skew + (z_95**3 - 3*z_95)/24 * kurt - (2*z_95**3 - 5*z_95)/36 * skew**2
    var_95_cf = mu + sigma * z_cf
    
    # Add to report
    report.var_95_historical = var_95_hist
    report.var_95_parametric = var_95_param
    report.var_95_cornish_fisher = var_95_cf
    report.var_method_used = method
```

**Verify:**
```bash
python -c "
import numpy as np
from scipy.stats import norm
# Fat-tailed returns: historical VaR should be worse than parametric
np.random.seed(42)
returns = np.random.standard_t(df=3, size=1000) * 0.01  # t-distribution (fat tails)
var_hist = np.percentile(returns, 5)
var_param = returns.mean() + returns.std() * norm.ppf(0.05)
print(f'Historical VaR (5%): {var_hist:.4f}')
print(f'Parametric VaR (5%): {var_param:.4f}')
print(f'Historical should be more negative (captures fat tails)')
"
```

---

### T4: Add correlation-stress scenario to stress testing

**What:** Model portfolio risk if all pairwise correlations spike to stress level.

**Files:**
- `risk/stress_test.py` — create or update stress testing module

**Implementation notes:**
```python
def correlation_stress_test(portfolio_weights, covariance, stress_correlation=0.9):
    """
    Simulate portfolio risk if all pairwise correlations spike to stress level.
    During crises, diversification fails as correlations converge to ~1.0.
    """
    n = len(portfolio_weights)
    vols = np.sqrt(np.diag(covariance))
    
    # Build stress correlation matrix: all off-diagonal = stress_correlation
    stress_corr = np.full((n, n), stress_correlation)
    np.fill_diagonal(stress_corr, 1.0)
    
    # Convert back to covariance
    vol_matrix = np.diag(vols)
    stress_cov = vol_matrix @ stress_corr @ vol_matrix
    
    # Compute portfolio vol under stress
    w = np.array(list(portfolio_weights.values()))
    normal_vol = np.sqrt(w @ covariance @ w * 252)
    stress_vol = np.sqrt(w @ stress_cov @ w * 252)
    
    # Compute stress VaR (99%)
    from scipy.stats import norm
    stress_var_99 = -norm.ppf(0.01) * stress_vol / np.sqrt(252)
    
    return {
        'normal_portfolio_vol': normal_vol,
        'stress_portfolio_vol': stress_vol,
        'vol_increase_pct': (stress_vol / normal_vol - 1) * 100,
        'stress_var_99_daily': stress_var_99,
        'max_loss_estimate': stress_var_99 * 3,  # 3-sigma event
        'diversification_benefit_lost': 1 - (normal_vol / (w @ vols)),
    }

def factor_stress_test(portfolio_weights, factor_exposures, scenarios):
    """
    Simulate portfolio impact of specific factor shocks.
    """
    results = {}
    for scenario_name, factor_shocks in scenarios.items():
        portfolio_impact = sum(
            factor_exposures.get(factor, 0) * shock
            for factor, shock in factor_shocks.items()
        )
        results[scenario_name] = {
            'portfolio_return': portfolio_impact,
            'factor_contributions': {
                f: factor_exposures.get(f, 0) * s
                for f, s in factor_shocks.items()
            }
        }
    return results

# Predefined scenarios:
CRISIS_SCENARIOS = {
    'equity_crash_2008': {'market': -0.40, 'size': -0.20, 'value': -0.30, 'momentum': -0.50},
    'covid_march_2020': {'market': -0.35, 'volatility': 0.60, 'momentum': -0.40},
    'rate_shock': {'market': -0.15, 'duration': -0.20, 'growth': -0.10},
    'liquidity_crisis': {'market': -0.20, 'size': -0.30, 'illiquidity': -0.40},
}
```

**Verify:**
```bash
python -c "
import numpy as np
# 5-asset portfolio, equal weight
n = 5
w = np.ones(n) / n
# Normal correlations: 0.3
corr_normal = np.full((n,n), 0.3); np.fill_diagonal(corr_normal, 1.0)
vols = np.ones(n) * 0.20
vol_mat = np.diag(vols)
cov_normal = vol_mat @ corr_normal @ vol_mat
# Stress correlations: 0.9
corr_stress = np.full((n,n), 0.9); np.fill_diagonal(corr_stress, 1.0)
cov_stress = vol_mat @ corr_stress @ vol_mat
normal_vol = np.sqrt(w @ cov_normal @ w * 252)
stress_vol = np.sqrt(w @ cov_stress @ w * 252)
print(f'Normal portfolio vol: {normal_vol:.1%}')
print(f'Stress portfolio vol: {stress_vol:.1%}')
print(f'Vol increase: {(stress_vol/normal_vol - 1)*100:.0f}%')
"
```

---

### T5: Add simple portfolio attribution

**What:** Decompose portfolio returns into market (beta), factor, and alpha (residual) components.

**Files:**
- `risk/attribution.py` — create new module

**Implementation notes:**
```python
def compute_return_attribution(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    factor_returns: Optional[pd.DataFrame] = None,
    window: int = 60,
) -> pd.DataFrame:
    """
    Decompose portfolio return into:
    - Market component (beta * benchmark_return)
    - Factor component (sum of factor_beta_i * factor_return_i)
    - Alpha (residual = portfolio - market - factor)
    
    Uses rolling regression to allow time-varying exposures.
    """
    results = []
    
    for end_idx in range(window, len(portfolio_returns)):
        start_idx = end_idx - window
        y = portfolio_returns.iloc[start_idx:end_idx]
        X = benchmark_returns.iloc[start_idx:end_idx]
        
        if factor_returns is not None:
            X = pd.concat([X.rename('market'), factor_returns.iloc[start_idx:end_idx]], axis=1)
        else:
            X = X.to_frame('market')
        
        # OLS regression
        X_with_const = np.column_stack([np.ones(len(X)), X.values])
        try:
            beta = np.linalg.lstsq(X_with_const, y.values, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(X_with_const.shape[1])
        
        # Decompose current-period return
        current_return = portfolio_returns.iloc[end_idx]
        market_component = beta[1] * benchmark_returns.iloc[end_idx]
        
        factor_component = 0
        if factor_returns is not None and len(beta) > 2:
            for i, col in enumerate(factor_returns.columns):
                factor_component += beta[i+2] * factor_returns[col].iloc[end_idx]
        
        alpha_component = current_return - market_component - factor_component
        
        results.append({
            'date': portfolio_returns.index[end_idx],
            'total_return': current_return,
            'market_return': market_component,
            'factor_return': factor_component,
            'alpha_return': alpha_component,
            'beta': beta[1],
        })
    
    return pd.DataFrame(results).set_index('date')
```

**Verify:**
```bash
python -c "
print('Portfolio attribution:')
print('  Market component: beta * benchmark_return')
print('  Factor component: sum(factor_beta_i * factor_return_i)')
print('  Alpha: residual (what the model actually adds)')
print('  Rolling regression: time-varying exposures (60-day window)')
"
```

---

### T6: Add bid-ask spread buffer to stop loss

**What:** Adjust stop prices to account for expected bid-ask spread, preventing false triggers.

**Files:**
- `risk/stop_loss.py` — add spread buffer to `evaluate()` and `compute_initial_stop()`

**Implementation notes:**
- Current: stop triggers when `current_price <= stop_price`
- Problem: during spread-widening events, the mid-price may briefly touch the stop level without a real trade occurring there
- Fix: add spread buffer
  ```python
  def evaluate(self, entry_price, current_price, highest_price, atr, bars_held,
               entry_regime, current_regime, spread_bps=3.0):
      """
      spread_bps: estimated bid-ask spread in basis points.
      Stop prices are adjusted downward by half-spread to avoid false triggers.
      """
      # Spread buffer: half the spread (stop would fill at bid, not mid)
      spread_buffer = entry_price * (spread_bps / 10000.0) * 0.5
      
      # Adjust all stop prices down by spread buffer
      hard_stop_price = entry_price * (1 + self.hard_stop_pct) - spread_buffer
      atr_stop_price = entry_price - (effective_atr_mult * atr) - spread_buffer
      trailing_stop_price = highest_price - (effective_trail_mult * atr) - spread_buffer
      
      # ... rest of evaluation using adjusted stop prices ...
  ```
- Default spread_bps=3.0 (typical for liquid large-cap US equities)
- Make configurable via config.py: `STOP_LOSS_SPREAD_BUFFER_BPS = 3.0`

**Verify:**
```bash
python -c "
# Without buffer: stop at exactly 92.00
# With buffer (3bps): stop at 92.00 - 0.015 = 91.985
entry = 100.0
stop_pct = -0.08  # -8% hard stop
spread_bps = 3.0
no_buffer = entry * (1 + stop_pct)
with_buffer = no_buffer - entry * (spread_bps / 10000) * 0.5
print(f'Hard stop without buffer: {no_buffer:.3f}')
print(f'Hard stop with 3bps buffer: {with_buffer:.3f}')
print(f'Difference: {(no_buffer - with_buffer):.3f} ({(no_buffer-with_buffer)/entry*10000:.1f} bps)')
"
```

---

### T7: Improve drawdown recovery ramp

**What:** Make recovery ramp concave (cautious start, faster finish) instead of linear.

**Files:**
- `risk/drawdown.py` — update `_compute_actions()` RECOVERY state (lines 180-198)

**Implementation notes:**
- Current (linear): `progress = days_in_state / recovery_days; scale = 0.25 + 0.75 * progress`
- Fix (concave): slower start, faster end
  ```python
  # In RECOVERY state:
  progress = min(1.0, self._total_days_in_state / self.recovery_days)
  
  # Concave ramp: sqrt gives slow start, fast finish
  # At 25% of recovery: scale = 0.25 + 0.75*sqrt(0.25) = 0.625 (vs 0.4375 linear)
  # Actually we want SLOW start: use square
  # At 25% of recovery: scale = 0.25 + 0.75*0.25^2 = 0.297 (very cautious)
  # At 50% of recovery: scale = 0.25 + 0.75*0.50^2 = 0.4375
  # At 75% of recovery: scale = 0.25 + 0.75*0.75^2 = 0.672
  # At 100% recovery: scale = 0.25 + 0.75*1.0 = 1.0
  
  scale = 0.25 + 0.75 * (progress ** 2)  # Quadratic: cautious early recovery
  
  size_multiplier = scale
  allow_new_entries = progress >= 0.3  # Only allow new entries after 30% of recovery
  ```
- Rationale: after a large drawdown, the system should be very cautious initially to prevent immediate re-drawdown, then gradually return to full sizing

**Verify:**
```bash
python -c "
# Compare linear vs quadratic recovery
for pct in [0.1, 0.25, 0.5, 0.75, 1.0]:
    linear = 0.25 + 0.75 * pct
    quadratic = 0.25 + 0.75 * (pct ** 2)
    print(f'Recovery {pct:.0%}: linear={linear:.3f}, quadratic={quadratic:.3f}')
"
```

---

### T8: Test risk system improvements

**What:** Tests for covariance, VaR, stress testing, attribution, and stop loss improvements.

**Files:**
- `tests/test_risk_improvements.py` — new test file

**Implementation notes:**
- Test cases:
  1. `test_ledoit_wolf_data_driven_shrinkage` — Shrinkage intensity comes from data, not hardcoded
  2. `test_ewma_covariance_recent_emphasis` — Recent vol regime change reflected in EWMA
  3. `test_parametric_var_vs_historical` — Both methods computed, parametric underestimates fat tails
  4. `test_cornish_fisher_var` — CF adjustment for non-normal returns
  5. `test_correlation_stress_increases_vol` — Stress vol > normal vol for diversified portfolio
  6. `test_attribution_sums_to_total` — market + factor + alpha = total return
  7. `test_stop_loss_spread_buffer` — Stop with buffer triggers at lower price than without
  8. `test_recovery_ramp_concave` — Quadratic recovery slower at start than linear
  9. `test_factor_stress_scenarios` — Predefined scenarios produce reasonable impacts

**Verify:**
```bash
python -m pytest tests/test_risk_improvements.py -v
```

---

## Validation

### Acceptance criteria
1. Covariance uses data-driven shrinkage (sklearn LedoitWolf.shrinkage_ attribute)
2. EWMA option available with configurable half-life
3. VaR reported as both historical and parametric (Cornish-Fisher for non-normal)
4. Correlation stress test shows vol increase for diversified portfolios
5. Attribution: market + factor + alpha ≈ total return (within 1bp)
6. Stop loss spread buffer prevents false triggers (adjustable via config)
7. Recovery ramp is concave (quadratic), not linear

### Rollback plan
- Covariance: revert to fixed shrinkage (one parameter change)
- VaR: keep historical as primary, parametric as supplementary
- Stress test: additive module, remove if issues
- Attribution: additive module, remove if issues
- Stop buffer: set STOP_LOSS_SPREAD_BUFFER_BPS = 0 to disable
