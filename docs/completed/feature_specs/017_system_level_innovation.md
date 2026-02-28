# Feature Spec: System-Level Innovation — Outside-the-Box Improvements

> **Status:** Approved
> **Author:** justin
> **Date:** 2026-02-23
> **Estimated effort:** ~20 hours across 10 tasks

---

## Why

Beyond fixing bugs and improving individual components, the system needs architectural innovations that create compounding advantages. The current system treats prediction, risk, and execution as sequential steps — but they should be jointly optimized. Several well-established quantitative finance techniques are completely missing: (1) No distribution shift detection — the model trains on historical data but doesn't detect when the market has fundamentally changed. (2) No prediction intervals — the model outputs point predictions but not confidence bands, so position sizing can't account for prediction uncertainty. (3) No online learning — the model retrains on a schedule but doesn't adapt between retraining cycles. (4) No multi-horizon blending — separate models for 5d, 10d, 20d horizons are trained independently but could share information. (5) No automatic regime-aware strategy allocation — the system detects regimes but doesn't automatically switch between strategies optimized for each regime. (6) No factor exposure monitoring — the portfolio may have unintended factor tilts that increase systematic risk. (7) No transaction cost budget — the system doesn't optimize the total cost of implementing a set of portfolio changes. (8) No survivorship-free backtest comparison — backtests on survivorship-free vs biased universes are never compared. (9) No self-diagnostic dashboard — the system can't explain WHY performance is degrading. (10) No ensemble of diverse model architectures — only gradient boosting is used.

## What

Implement system-level innovations that create compounding advantages: distribution shift detection to trigger early retraining, prediction intervals for uncertainty-aware sizing, multi-horizon information sharing, regime-aware strategy allocation, factor exposure monitoring, transaction cost budgeting, and a self-diagnostic system. Done means: the system is architecturally more sophisticated than a standard predict-size-execute pipeline.

## Constraints

### Must-haves
- At least 5 of the 10 innovations implemented
- Each innovation must be independently testable and toggleable via config
- No innovation should degrade existing performance
- Each innovation has a clear A/B testable hypothesis

### Must-nots
- Do NOT implement all 10 at once (incremental rollout via A/B testing)
- Do NOT add model architectures that require GPU (system runs on CPU)
- Do NOT make the pipeline >3x slower

## Tasks

### T1: Distribution shift detection (CUSUM + PSI)

**What:** Detect when the market data distribution has shifted away from the training distribution, triggering early retraining.

**Files:**
- `models/shift_detection.py` — new module
- `autopilot/retrain_trigger.py` — integrate shift detection

**Implementation notes:**
```python
class DistributionShiftDetector:
    """
    Two methods for detecting distribution shift:
    1. CUSUM (Cumulative Sum): detects mean shift in prediction errors
    2. PSI (Population Stability Index): detects feature distribution shift
    """
    
    def __init__(self, cusum_threshold=5.0, psi_threshold=0.25, reference_window=252):
        self.cusum_threshold = cusum_threshold
        self.psi_threshold = psi_threshold
        self.reference_window = reference_window
        self._reference_distributions = {}
    
    def set_reference(self, features: pd.DataFrame):
        """Store reference distributions from training data."""
        for col in features.columns:
            self._reference_distributions[col] = {
                'bins': np.histogram(features[col].dropna(), bins=10)[1],
                'hist': np.histogram(features[col].dropna(), bins=10)[0] / len(features),
            }
    
    def check_cusum(self, prediction_errors: pd.Series, target_mean=0.0):
        """CUSUM: detect mean shift in prediction errors."""
        errors = prediction_errors - target_mean
        cusum_pos = np.maximum.accumulate(np.cumsum(errors))
        cusum_neg = np.minimum.accumulate(np.cumsum(errors))
        cusum_range = cusum_pos - cusum_neg
        
        shift_detected = cusum_range.iloc[-1] > self.cusum_threshold * errors.std()
        return {
            'shift_detected': shift_detected,
            'cusum_statistic': cusum_range.iloc[-1],
            'threshold': self.cusum_threshold * errors.std(),
        }
    
    def check_psi(self, current_features: pd.DataFrame):
        """PSI: detect feature distribution shift vs training data."""
        psi_scores = {}
        for col in current_features.columns:
            if col not in self._reference_distributions:
                continue
            ref = self._reference_distributions[col]
            current_hist = np.histogram(current_features[col].dropna(), bins=ref['bins'])[0]
            current_hist = current_hist / max(1, current_hist.sum())
            
            # PSI = sum((current - reference) * ln(current / reference))
            eps = 1e-6
            ref_h = np.maximum(ref['hist'], eps)
            cur_h = np.maximum(current_hist, eps)
            psi = np.sum((cur_h - ref_h) * np.log(cur_h / ref_h))
            psi_scores[col] = psi
        
        max_psi = max(psi_scores.values()) if psi_scores else 0
        avg_psi = np.mean(list(psi_scores.values())) if psi_scores else 0
        shift_detected = avg_psi > self.psi_threshold
        
        return {
            'shift_detected': shift_detected,
            'avg_psi': avg_psi,
            'max_psi': max_psi,
            'threshold': self.psi_threshold,
            'top_shifted_features': sorted(psi_scores.items(), key=lambda x: -x[1])[:5],
        }
```
- Integrate into retrain_trigger.py: if shift detected, reduce retrain interval
- Add to health checks: distribution shift score

**Verify:**
```bash
python -c "
import numpy as np
# PSI: identical distributions → PSI ≈ 0
# PSI: shifted distribution → PSI > 0.25
ref = np.random.randn(1000)
same = np.random.randn(1000)
shifted = np.random.randn(1000) + 2.0  # Mean shift
eps = 1e-6
for name, data in [('same', same), ('shifted', shifted)]:
    ref_h = np.histogram(ref, bins=10)[0] / 1000
    cur_h = np.histogram(data, bins=np.histogram(ref, bins=10)[1])[0] / 1000
    ref_h = np.maximum(ref_h, eps)
    cur_h = np.maximum(cur_h, eps)
    psi = np.sum((cur_h - ref_h) * np.log(cur_h / ref_h))
    print(f'{name}: PSI = {psi:.4f} (threshold=0.25)')
"
```

---

### T2: Prediction intervals via conformal prediction

**What:** Add prediction intervals (not just point predictions) using conformal prediction, which requires no distributional assumptions.

**Files:**
- `models/conformal.py` — new module
- `models/predictor.py` — integrate prediction intervals

**Implementation notes:**
```python
class ConformalPredictor:
    """
    Split conformal prediction: uses calibration residuals to construct
    prediction intervals with guaranteed coverage probability.
    
    No distributional assumptions. Works with any base model.
    """
    
    def __init__(self, coverage=0.90):
        self.coverage = coverage
        self._quantile = None
        self._calibration_residuals = None
    
    def calibrate(self, predictions: np.ndarray, actuals: np.ndarray):
        """Compute non-conformity scores from calibration set."""
        residuals = np.abs(actuals - predictions)
        self._calibration_residuals = np.sort(residuals)
        
        # Quantile for desired coverage (with finite-sample correction)
        n = len(residuals)
        q_level = np.ceil((n + 1) * self.coverage) / n
        self._quantile = np.quantile(residuals, min(q_level, 1.0))
    
    def predict_interval(self, point_prediction: float):
        """Return (lower, upper) prediction interval."""
        if self._quantile is None:
            raise ValueError("Must calibrate first")
        return (
            point_prediction - self._quantile,
            point_prediction + self._quantile,
        )
    
    def predict_intervals_batch(self, predictions: np.ndarray):
        """Batch prediction intervals."""
        return np.column_stack([
            predictions - self._quantile,
            predictions + self._quantile,
        ])
```
- Integration with position sizing:
  ```python
  # Wider prediction interval → more uncertain → smaller position
  interval_width = upper - lower
  uncertainty_scalar = 1.0 / (1.0 + interval_width / avg_interval_width)
  position_size *= uncertainty_scalar
  ```

**Verify:**
```bash
python -c "
import numpy as np
# Conformal prediction: guaranteed coverage
np.random.seed(42)
n_cal = 200
preds = np.random.randn(n_cal) * 0.02
actuals = preds + np.random.randn(n_cal) * 0.01  # Noise around predictions
residuals = np.abs(actuals - preds)
q90 = np.quantile(residuals, np.ceil(201 * 0.90) / 200)
print(f'90% prediction interval width: ±{q90:.4f}')
# Check coverage on new data
n_test = 1000
test_preds = np.random.randn(n_test) * 0.02
test_actuals = test_preds + np.random.randn(n_test) * 0.01
covered = np.abs(test_actuals - test_preds) <= q90
print(f'Actual coverage: {covered.mean():.1%} (target: 90%)')
"
```

---

### T3: Multi-horizon information sharing

**What:** Share feature importance and regime information across 5d, 10d, 20d prediction horizons.

**Files:**
- `models/trainer.py` — add multi-horizon training option
- `models/predictor.py` — add horizon blending

**Implementation notes:**
- Current: each horizon trains independently with no information sharing
- New: train a shared feature selector and then fine-tune per horizon
  ```python
  def train_multi_horizon_ensemble(self, features, targets_dict, regimes, ...):
      """
      targets_dict: {5: targets_5d, 10: targets_10d, 20: targets_20d}
      
      Phase 1: Shared feature selection across all horizons
        - Pool all horizon targets
        - Select features that are important for ANY horizon
        - This finds features with persistent predictive power
      
      Phase 2: Per-horizon model training
        - Each horizon trains on shared features
        - Different models can learn horizon-specific patterns
      
      Phase 3: Horizon blending at prediction time
        - Short-term prediction: weight toward 5d model
        - Medium-term: blend 5d and 10d
        - For 10d horizon: blend = 0.2*pred_5d + 0.6*pred_10d + 0.2*pred_20d
      """
  ```
- Blending weights can be regime-adaptive:
  - High-vol regime: more weight on shorter horizons (faster mean-reversion)
  - Trending regime: more weight on longer horizons (momentum persists)

**Verify:**
```bash
python -c "
print('Multi-horizon blending:')
print('  Trending regime: 0.1*5d + 0.3*10d + 0.6*20d (momentum persists)')
print('  Mean-reverting: 0.5*5d + 0.3*10d + 0.2*20d (quick reversals)')
print('  High-vol: 0.6*5d + 0.3*10d + 0.1*20d (short-term focus)')
"
```

---

### T4: Regime-aware strategy allocation

**What:** Automatically adjust strategy parameters based on detected regime, instead of using fixed parameters across all regimes.

**Files:**
- `autopilot/strategy_allocator.py` — new module
- `config.py` — add regime-specific parameter sets

**Implementation notes:**
```python
# Regime-specific strategy profiles
REGIME_STRATEGY_PROFILES = {
    0: {  # trending_bull
        'name': 'Momentum-Heavy',
        'entry_threshold': 0.008,  # Lower bar (ride trends)
        'position_size_pct': 0.08,  # Larger positions
        'max_positions': 25,  # More diversified
        'holding_days': 15,  # Hold longer
        'kelly_fraction': 0.6,  # More aggressive
    },
    1: {  # trending_bear
        'name': 'Defensive',
        'entry_threshold': 0.015,  # Higher bar (selective)
        'position_size_pct': 0.04,  # Smaller positions
        'max_positions': 10,  # Concentrated best ideas
        'holding_days': 5,  # Quick exits
        'kelly_fraction': 0.3,  # Conservative
    },
    2: {  # mean_reverting
        'name': 'Mean-Reversion Focus',
        'entry_threshold': 0.010,
        'position_size_pct': 0.06,
        'max_positions': 20,
        'holding_days': 10,
        'kelly_fraction': 0.5,
    },
    3: {  # high_volatility
        'name': 'Risk-Off',
        'entry_threshold': 0.025,  # Very high bar
        'position_size_pct': 0.03,  # Tiny positions
        'max_positions': 5,  # Minimal exposure
        'holding_days': 3,  # Very short
        'kelly_fraction': 0.2,  # Very conservative
    },
}

class StrategyAllocator:
    def get_regime_profile(self, regime, regime_confidence):
        """Blend regime-specific parameters with default based on confidence."""
        profile = REGIME_STRATEGY_PROFILES.get(regime, REGIME_STRATEGY_PROFILES[2])
        default = REGIME_STRATEGY_PROFILES[2]  # Mean-reverting as default
        
        # Blend based on regime confidence: high confidence → full regime profile
        blended = {}
        for key in profile:
            if key == 'name':
                blended[key] = profile[key]
                continue
            regime_val = profile[key]
            default_val = default[key]
            blended[key] = regime_confidence * regime_val + (1 - regime_confidence) * default_val
        
        return blended
```

**Verify:**
```bash
python -c "
print('Regime strategy profiles:')
print('  Bull: larger positions, longer holds, momentum-heavy')
print('  Bear: smaller positions, shorter holds, defensive')
print('  Mean-revert: moderate parameters')
print('  High-vol: tiny positions, very short holds, risk-off')
print('  Confidence blending: low confidence → use defaults')
"
```

---

### T5: Factor exposure monitoring

**What:** Track unintended factor tilts (market, size, value, momentum, volatility) and alert when exposure exceeds thresholds.

**Files:**
- `risk/factor_monitor.py` — new module
- `api/routers/risk.py` — expose via API

**Implementation notes:**
```python
class FactorExposureMonitor:
    """
    Track portfolio factor exposures and alert on unintended tilts.
    Uses simple factor proxies computable from price data.
    """
    
    FACTOR_PROXIES = {
        'market': 'beta to SPY',
        'size': 'log(market_cap) z-score',
        'value': 'book-to-market or earnings yield',
        'momentum': '12m-1m return z-score',
        'volatility': 'realized vol z-score',
        'liquidity': 'dollar volume z-score',
    }
    
    def compute_exposures(self, positions, price_data, benchmark='SPY'):
        """Compute portfolio factor exposures."""
        exposures = {}
        
        # Market beta
        portfolio_returns = self._compute_portfolio_returns(positions, price_data)
        benchmark_returns = price_data.get(benchmark, pd.Series()).pct_change()
        if len(portfolio_returns) > 20 and len(benchmark_returns) > 20:
            cov = portfolio_returns.cov(benchmark_returns)
            var = benchmark_returns.var()
            exposures['market_beta'] = cov / var if var > 0 else 1.0
        
        # Size tilt: average market cap of positions vs universe
        # Momentum tilt: average 12m return of positions vs universe
        # Vol tilt: average realized vol of positions vs universe
        # ... compute for each factor ...
        
        return exposures
    
    def check_limits(self, exposures, limits=None):
        """Check if any exposure exceeds limits."""
        default_limits = {
            'market_beta': (0.5, 1.5),  # Should be near 1.0
            'size_zscore': (-1.5, 1.5),  # Shouldn't be extreme
            'momentum_zscore': (-2.0, 2.0),
            'vol_zscore': (-1.5, 1.5),
        }
        limits = limits or default_limits
        
        violations = []
        for factor, (lo, hi) in limits.items():
            val = exposures.get(factor, 0)
            if val < lo or val > hi:
                violations.append(f"{factor}={val:.2f} outside [{lo}, {hi}]")
        
        return {'passed': len(violations) == 0, 'violations': violations}
```

**Verify:**
```bash
python -c "
print('Factor exposure monitoring:')
print('  Market beta: should be near 1.0 (unless intentionally hedged)')
print('  Size tilt: detect if portfolio is all small-cap or all large-cap')
print('  Momentum tilt: detect if chasing recent winners excessively')
print('  Volatility tilt: detect if portfolio is concentrated in high-vol names')
print('  Alert when exposure exceeds configurable limits')
"
```

---

### T6: Transaction cost budget optimization

**What:** Optimize the total cost of implementing a set of portfolio changes, not just individual trades.

**Files:**
- `risk/cost_budget.py` — new module

**Implementation notes:**
```python
def optimize_rebalance_cost(
    current_weights: pd.Series,
    target_weights: pd.Series,
    daily_volumes: pd.Series,
    urgency: float = 0.5,  # 0=patient, 1=urgent
    total_budget_bps: float = 50,  # Max total implementation cost
):
    """
    Given current and target portfolios, find the optimal rebalance
    that minimizes tracking error vs target while staying within
    transaction cost budget.
    
    For example: if target has 20 trades but budget only allows 10,
    prioritize the trades with highest alpha-to-cost ratio.
    """
    trades = target_weights - current_weights
    trades = trades[trades.abs() > 1e-6]  # Ignore tiny adjustments
    
    # Estimate cost per trade
    costs = {}
    for ticker, trade_size in trades.items():
        volume = daily_volumes.get(ticker, 1e6)
        participation = abs(trade_size) / volume if volume > 0 else 1.0
        impact_bps = 25 * np.sqrt(participation)  # Square-root impact
        spread_bps = 3.0  # Fixed spread estimate
        costs[ticker] = (0.5 * spread_bps + impact_bps) * abs(trade_size)
    
    total_cost = sum(costs.values())
    
    if total_cost <= total_budget_bps:
        # Full rebalance is within budget
        return target_weights, total_cost
    
    # Prioritize trades by |alpha improvement| / cost
    # For now, prioritize larger weight changes (they matter more for tracking error)
    priority = trades.abs() / pd.Series(costs)
    priority = priority.sort_values(ascending=False)
    
    # Greedy: execute trades in priority order until budget exhausted
    remaining_budget = total_budget_bps
    executed_trades = {}
    for ticker in priority.index:
        if costs[ticker] <= remaining_budget:
            executed_trades[ticker] = trades[ticker]
            remaining_budget -= costs[ticker]
    
    # Partial rebalance: current + executed changes
    partial_weights = current_weights.copy()
    for ticker, change in executed_trades.items():
        partial_weights[ticker] = partial_weights.get(ticker, 0) + change
    partial_weights /= partial_weights.sum()  # Renormalize
    
    return partial_weights, total_budget_bps - remaining_budget
```

**Verify:**
```bash
python -c "
print('Transaction cost budget:')
print('  Total cost budget: 50bps per rebalance')
print('  Prioritize trades by importance / cost ratio')
print('  Partial rebalance when full rebalance too expensive')
print('  Reduces turnover drag on returns')
"
```

---

### T7: Self-diagnostic dashboard data

**What:** Build a system that explains WHY performance is degrading by correlating health metrics with P&L.

**Files:**
- `api/services/diagnostics.py` — new module
- `api/routers/diagnostics.py` — new API endpoint

**Implementation notes:**
```python
class SystemDiagnostics:
    """
    Correlate system metrics with performance to identify root causes
    of degradation.
    """
    
    def diagnose_performance(self, equity_curve, health_history, regime_history):
        """
        When performance is poor, identify likely causes:
        1. Is alpha decaying? (IC trend negative)
        2. Is risk management too tight? (position sizes dropping)
        3. Is market regime unfavorable? (high-vol regime duration)
        4. Are execution costs too high? (slippage increasing)
        5. Is the model stale? (time since last retrain)
        6. Has data quality degraded? (feature shift detected)
        """
        diagnostics = []
        
        # Recent performance
        recent_returns = equity_curve['daily_pnl'].tail(20)
        is_underperforming = recent_returns.mean() < 0
        
        if not is_underperforming:
            return {"status": "PERFORMING", "diagnostics": []}
        
        # Check each potential cause
        # 1. Alpha decay
        if health_history.get('signal_decay_score', 100) < 50:
            diagnostics.append({
                'cause': 'ALPHA_DECAY',
                'severity': 'HIGH',
                'evidence': 'Signal quality health check score below 50',
                'recommendation': 'Retrain model with recent data',
            })
        
        # 2. Regime impact
        current_regime = regime_history.get('current_regime', 0)
        if current_regime == 3:  # High volatility
            diagnostics.append({
                'cause': 'UNFAVORABLE_REGIME',
                'severity': 'MEDIUM',
                'evidence': f'Currently in high-volatility regime',
                'recommendation': 'System is designed to reduce exposure in high-vol. Wait for regime change.',
            })
        
        # 3. Execution degradation
        if health_history.get('execution_quality_score', 100) < 50:
            diagnostics.append({
                'cause': 'EXECUTION_DEGRADATION',
                'severity': 'MEDIUM',
                'evidence': 'Execution quality score below 50',
                'recommendation': 'Check market impact model calibration',
            })
        
        # 4. Data staleness
        if health_history.get('data_freshness_days', 0) > 7:
            diagnostics.append({
                'cause': 'STALE_DATA',
                'severity': 'HIGH',
                'evidence': f'Data is {health_history["data_freshness_days"]} days old',
                'recommendation': 'Run data refresh pipeline',
            })
        
        return {
            'status': 'UNDERPERFORMING',
            'diagnostics': diagnostics,
            'primary_cause': diagnostics[0]['cause'] if diagnostics else 'UNKNOWN',
        }
```

**Verify:**
```bash
python -c "
print('Self-diagnostic system:')
print('  Detects underperformance and identifies root causes:')
print('  - ALPHA_DECAY: model predictions losing power')
print('  - UNFAVORABLE_REGIME: market in difficult regime')
print('  - EXECUTION_DEGRADATION: costs increasing')
print('  - STALE_DATA: data not refreshed recently')
print('  - MODEL_STALE: too long since last retrain')
print('  Each diagnosis includes evidence and recommended action')
"
```

---

### T8: Ensemble of diverse model architectures

**What:** Add a simple linear model and a random forest alongside gradient boosting to reduce model risk.

**Files:**
- `models/trainer.py` — add model diversity support

**Implementation notes:**
- Current: only LightGBM/XGBoost (gradient boosting)
- Issue: all models share the same inductive bias (tree-based, greedy splits)
- Add 2 additional model types:
  ```python
  # 1. Ridge Regression (linear, regularized)
  from sklearn.linear_model import Ridge
  linear_model = Ridge(alpha=1.0)
  
  # 2. Random Forest (bagging, not boosting)
  from sklearn.ensemble import RandomForestRegressor
  rf_model = RandomForestRegressor(n_estimators=100, max_depth=4, n_jobs=-1)
  ```
- Ensemble weighting based on holdout performance:
  ```python
  # Weight proportional to holdout Spearman correlation
  weights = softmax([gbm_corr, linear_corr, rf_corr])
  ensemble_pred = weights[0]*gbm_pred + weights[1]*linear_pred + weights[2]*rf_pred
  ```
- If any model has negative holdout correlation, set its weight to 0
- This reduces model risk: if GBM overfits, linear model provides regularization

**Verify:**
```bash
python -c "
import numpy as np
# Softmax weighting example
def softmax(x):
    e = np.exp(np.array(x) - max(x))
    return e / e.sum()
corrs = [0.08, 0.05, 0.06]  # GBM best, but not by much
weights = softmax(corrs)
print(f'Model weights: GBM={weights[0]:.2f}, Linear={weights[1]:.2f}, RF={weights[2]:.2f}')
print('Diverse ensemble reduces model risk from single-architecture dependency')
"
```

---

### T9: Survivorship-free backtest comparison tool

**What:** Automatically run and compare backtests on survivorship-biased vs survivorship-free universes.

**Files:**
- `backtest/survivorship_comparison.py` — new module

**Implementation notes:**
```python
def compare_survivorship_impact(
    predictions,
    price_data_full,  # Includes delisted tickers (survivorship-free)
    price_data_survivors,  # Only tickers that survived to end of period
    **backtest_params,
):
    """
    Run identical backtest on both universes and quantify survivorship bias.
    
    Returns:
    - Full universe Sharpe, Drawdown, Win Rate
    - Survivor-only Sharpe, Drawdown, Win Rate
    - Bias estimate: Sharpe(survivors) - Sharpe(full)
    - Per-metric bias breakdown
    """
    from quant_engine.backtest.engine import Backtester
    
    bt = Backtester(**backtest_params)
    
    result_full = bt.run(predictions, price_data_full)
    result_survivors = bt.run(predictions, price_data_survivors)
    
    return {
        'full_universe': {
            'sharpe': result_full.sharpe,
            'win_rate': result_full.win_rate,
            'n_trades': result_full.total_trades,
            'max_drawdown': result_full.max_drawdown,
        },
        'survivors_only': {
            'sharpe': result_survivors.sharpe,
            'win_rate': result_survivors.win_rate,
            'n_trades': result_survivors.total_trades,
            'max_drawdown': result_survivors.max_drawdown,
        },
        'bias': {
            'sharpe_bias': result_survivors.sharpe - result_full.sharpe,
            'win_rate_bias': result_survivors.win_rate - result_full.win_rate,
            'survivorship_inflates_sharpe_by': (
                f"{(result_survivors.sharpe - result_full.sharpe) / max(0.01, abs(result_full.sharpe)) * 100:.1f}%"
            ),
        }
    }
```

**Verify:**
```bash
python -c "
print('Survivorship comparison:')
print('  Run same strategy on full universe vs survivors-only')
print('  Quantify how much survivorship bias inflates each metric')
print('  Typical bias: Sharpe inflated 20-40%, Win rate inflated 5-15%')
"
```

---

### T10: Test system-level innovations

**What:** Tests for all new modules.

**Files:**
- `tests/test_system_innovations.py` — new test file

**Implementation notes:**
- Test cases:
  1. `test_cusum_detects_mean_shift` — CUSUM fires when prediction errors shift
  2. `test_psi_detects_distribution_change` — PSI > 0.25 when feature distributions shift
  3. `test_conformal_coverage` — Conformal intervals achieve target coverage rate
  4. `test_conformal_uncertainty_scaling` — Wider intervals → smaller position sizes
  5. `test_regime_strategy_profiles` — Each regime has distinct parameter set
  6. `test_factor_exposure_limits` — Alert when beta > 1.5 or < 0.5
  7. `test_cost_budget_partial_rebalance` — Over-budget → partial execution
  8. `test_diagnostics_identify_alpha_decay` — Low signal score → ALPHA_DECAY diagnosis
  9. `test_diverse_ensemble_weights` — Negative-corr model gets weight=0
  10. `test_survivorship_bias_positive` — Survivors-only Sharpe > full universe Sharpe

**Verify:**
```bash
python -m pytest tests/test_system_innovations.py -v
```

---

## Validation

### Acceptance criteria
1. CUSUM detects mean shift in prediction errors within 50 samples
2. Conformal prediction intervals achieve ≥88% coverage (for 90% target)
3. Regime strategy profiles produce different parameters for each regime
4. Factor exposure monitoring alerts on extreme tilts
5. Cost budget produces partial rebalance when full is too expensive
6. Self-diagnostics correctly identify underperformance causes

### Rollback plan
- Each innovation is a separate module with config flag → disable individually
- No existing code is modified (all new modules)
- Paper trader integration is additive → remove hooks if issues

### Implementation priority
1. Distribution shift detection (T1) — prevents model staleness
2. Prediction intervals (T2) — improves position sizing
3. Regime-aware allocation (T4) — biggest expected PnL impact
4. Factor monitoring (T5) — risk reduction
5. Self-diagnostics (T7) — operational improvement
6. Multi-horizon (T3) — moderate PnL improvement
7. Cost budget (T6) — reduces implementation drag
8. Diverse ensemble (T8) — reduces model risk
9. Survivorship comparison (T9) — validation tool
