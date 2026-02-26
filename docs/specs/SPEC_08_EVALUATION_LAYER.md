# Feature Spec: Evaluation Layer (Truth Engine)

> **Status:** Draft
> **Author:** Claude Opus 4
> **Date:** 2026-02-26
> **Estimated effort:** 140 hours across 8 tasks

---

## Why

Performance evaluation in `/mnt/quant_engine/engine/validation.py` and `/mnt/quant_engine/engine/advanced_validation.py` computes aggregate metrics (Sharpe, returns, maximum drawdown) but **ignores regime structure and shock periods**. All performance is pooled; no decomposition of returns by market regime or stress scenario. This creates blind spots:

- A strategy that performs well in bull markets but catastrophically in bear markets appears mediocre in aggregate.
- ML overfitting is hidden by aggregate metrics; rolling information coefficient (IC) rolling window exists (60 days) but is not surfaced.
- Fragility diagnostics (% PnL from top trades, drawdown distribution, recovery time) are absent, hiding tail dependence.
- Calibration of confidence intervals is not validated; overconfident predictions are not detected.
- No "critical slowing down" detection (increasing recovery time as warning signal).

Furthermore, the proposed slicing framework (Section 8, improvement docs) suggests 15 dimensions; with only ~50 observations per slice, statistics are unreliable. We need to prioritize 5–6 most informative slices and make others optional.

This spec implements **regime-aware slicing, ML diagnostics, fragility analysis, and calibration validation** within the existing validation framework.

---

## What

Implement an **Evaluation Layer (Truth Engine)** that:

1. **Slices performance by market regime and stress scenarios:**
   - Decompose returns, Sharpe, max DD, recovery time per regime (normal vs. stress).
   - Identify strategy regimes (e.g., "beta-capture" vs. "alpha-generation" vs. "drawdown" regime).
   - Report: "Returns +8% in normal, -12% in stress" to highlight regime dependence.

2. **Implements ML diagnostics:**
   - Rolling information coefficient (IC) with decay detection (IC falling below threshold).
   - Decile spread (top decile return - bottom decile return) as predictive power metric.
   - Calibration curve (predicted rank vs. actual rank) to detect overconfidence.
   - Feature importance drift (top features changing between retraining periods).

3. **Computes fragility metrics:**
   - % PnL from top N trades (concentration of returns).
   - Drawdown distribution (is max DD from one large drop or many small ones?).
   - Recovery time distribution (time to recover from DD, trending upward = danger signal).
   - Underwater plot statistics (% time in drawdown, max drawdown duration).

4. **Performs walk-forward evaluation with embargo:**
   - Train on [t-250 days, t] data.
   - Embargo: 5-day gap (no data leakage).
   - Evaluate on [t+5, t+60] test period.
   - Repeat: slide window weekly.
   - Report: in-sample vs. out-of-sample performance (overfitting gap).

5. **Implements regime/shock slicing** with statistical confidence:
   - Focus on 5 primary slices: (1) Normal, (2) High Vol, (3) Crash, (4) Recovery, (5) Trendless.
   - Optional secondary slices (sector, factor, time-of-day, day-of-week).
   - Report 95% CI around metrics in each slice; flag slice if N < 20 observations.

6. **Detects critical slowing down:**
   - Recovery time increasing over rolling 3-month windows (sign of regime shift).
   - Frequency of consecutive losing days increasing (hardening of downside).

---

## Constraints

### Must-haves

- Slicing by regime requires mapping returns to regime_state from `/mnt/quant_engine/engine/regime/detector.py`.
- Walk-forward uses embargo (5-day gap minimum) to prevent data leakage.
- Calibration curve uses `ConfidenceCalibrator` from `/mnt/quant_engine/engine/models/calibration.py`, not a custom implementation.
- Rolling IC computed with window=60 days minimum; flag if < 20 observations.
- Recovery time trends computed over 3-month rolling window (60 days).
- All slices report sample size N and flag if N < 20 (low confidence).

### Must-nots

- **Do not** propose 15 slicing dimensions; focus on 5 primary slices.
- **Do not** pool performance across regimes; always report per-regime.
- **Do not** compute aggregate Sharpe if regime Sharpes differ by >50% (report regime breakdown instead).
- **Do not** create new calibration logic; import and use existing ConfidenceCalibrator.
- **Do not** compute walk-forward without embargo; data leakage is a critical issue.
- **Do not** use AD (Anderson-Darling) test without checking sample size; many hypothesis tests break for small N.

### Out of scope

- Cross-sectional performance (comparing strategy to benchmark or peer strategies). This is risk attribution.
- Factor decomposition of returns using Fama-French models. This is factor analysis, not evaluation.
- Capacity analysis with realistic slippage curves. Capacity is addressed in advanced_validation.py.
- Parameter sensitivity analysis. This is hyperparameter search, not validation.
- Monte Carlo return distribution (seed variants). This is in advanced_validation.py.

---

## Current State

### Key files

- **`/mnt/quant_engine/engine/validation.py`** (comprehensive): Walk-forward, statistical tests (t-test, KS), CPCV, SPA. `IC_ROLLING_WINDOW=60`.
- **`/mnt/quant_engine/engine/advanced_validation.py`**: DSR (depth of sleep / robustness), PBO (probability of backtest overfitting), Monte Carlo, capacity.
- **`/mnt/quant_engine/engine/regime/detector.py`**: Detects regime, exposes `regime_state` with `.regime` (0–3).
- **`/mnt/quant_engine/engine/models/calibration.py`**: `ConfidenceCalibrator` class for calibration curve.
- **`/mnt/quant_engine/engine/backtest/paper_trader.py`**: Records trade-by-trade PnL, positions, dates.
- **`/mnt/quant_engine/engine/stress_test.py`** (548 lines): 5 macro scenarios. Can be reused to label stress periods.

### Existing patterns to follow

- Validation function signature: `func(returns: pd.Series, predictions: np.ndarray, ...) -> Dict[str, float]` (return dict of metrics).
- Slicing: take Boolean mask `is_regime_x` and filter returns/predictions. Return metrics separately per slice.
- Logging: use `logging` module with INFO for summaries, DEBUG for details.
- Testing: parametrized tests over multiple datasets (synthetic, real stock, crypto).

### Configuration

**New config section in `/mnt/quant_engine/config/validation.yaml`:**

```yaml
walk_forward:
  training_window: 250  # days
  embargo_days: 5
  test_window: 60  # days
  slide_frequency: "weekly"  # "daily" or "weekly"

regime_slicing:
  primary_slices:
    - {name: "Normal", regime_ids: [0, 1]}
    - {name: "High Vol", regime_ids: [2]}
    - {name: "Crash", regex: "2008|COVID|Flash Crash"}
    - {name: "Recovery", regex: "post_crisis_[0-9]+"}
    - {name: "Trendless", volatility_condition: "vol > 1.5x median"}

  secondary_slices:
    - {name: "Sector", dimension: "sector", optional: true}
    - {name: "Factor", dimension: "factor", optional: true}
    - {name: "Weekday", dimension: "day_of_week", optional: true}

ml_diagnostics:
  ic_rolling_window: 60
  ic_decay_threshold: 0.3  # warn if IC < 0.3
  decile_spread_min: 0.02  # expected min spread for good predictor
  calibration_bins: 10

fragility_metrics:
  top_n_trades: [5, 10, 20]  # report % PnL from top N
  recovery_window: 60  # days, for recovery time stats
  critical_slowing_window: 60  # days, for trend detection

output:
  format: "html"  # or "json", "csv"
  include_plots: true
  include_regime_slices: true
  include_fragility: true
```

---

## Tasks

### T1: Regime-Based Slicing Framework

**What:** Create a `PerformanceSlice` class and registry that slices returns/predictions by regime and custom conditions. Implement 5 primary slices: Normal, High Vol, Crash, Recovery, Trendless.

**Files:**
- Create `/mnt/quant_engine/engine/evaluation/slicing.py` (new):
  - Class `PerformanceSlice`:
    - Attributes: `name: str`, `condition: Callable[[pd.DataFrame], np.ndarray]`, `min_samples: int = 20`.
    - Method `apply(returns: pd.Series, metadata: pd.DataFrame) -> Tuple[pd.Series, Dict]`:
      - Returns: filtered returns + metadata dict {name, n_samples, n_days, date_range}.
  - Class `SliceRegistry`:
    - Static method `create_regime_slices(regime_states: np.ndarray) -> List[PerformanceSlice]`:
      - Primary slices: Normal (regime 0–1), High Vol (regime 2), Crash (regime 3 + large drawdown), Recovery (regime 3 with positive return), Trendless.
      - Example: `regime_0_1 = lambda meta: (meta['regime'] == 0) | (meta['regime'] == 1)`.
    - Static method `create_secondary_slices(optional=True) -> List[PerformanceSlice]` (stub for extension).
- Create `/mnt/quant_engine/engine/evaluation/metrics.py` (new):
  - Function `compute_slice_metrics(returns: pd.Series, predictions: np.ndarray, slice_mask: np.ndarray) -> Dict`:
    - Return: {mean_return, sharpe, max_dd, recovery_time, n_samples, start_date, end_date}.
  - Handle edge cases: if n_samples < 20, report metric with caveat {metric: value, confidence: "low", n_samples: N}.
- Tests: `/mnt/quant_engine/tests/test_slicing.py`.

**Implementation notes:**
- Slice conditions are callables that take metadata DataFrame (with regime, volatility, drawdown, date columns) and return Boolean mask.
- Crash slice = regime 3 AND (cumulative return < -10%). Recovery slice = regime 3 AND (return last 20 days > 5%).
- Trendless slice = regime 0 but volatility > 1.5x rolling median.
- Edge case: if a slice has < 20 observations, don't report metric, flag low confidence.

**Verify:**
- Create synthetic returns with labeled regimes; apply slices and verify regime slices partition returns correctly.
- Verify Crash slice catches 2008 Sept–Dec period.
- Test edge cases: empty slice, single-observation slice.

---

### T2: Walk-Forward with Embargo and Overfitting Detection

**What:** Implement walk-forward evaluation with 5-day embargo gap. Compute in-sample vs. out-of-sample metrics to measure overfitting.

**Files:**
- Modify `/mnt/quant_engine/engine/validation.py`:
  - Add function `walk_forward_with_embargo(returns: pd.Series, predictions: np.ndarray, train_window: int = 250, embargo: int = 5, test_window: int = 60, slide_freq: str = "weekly") -> pd.DataFrame`:
    - Returns DataFrame with columns: fold, train_start, train_end, embargo_start, embargo_end, test_start, test_end, in_sample_metrics, out_of_sample_metrics, overfit_gap.
    - For each fold: train [t, t+train_window], embargo [t+train_window, t+train_window+embargo], test [t+train_window+embargo, t+train_window+embargo+test_window].
  - Helper: `compute_fold_metrics(returns_in_sample, preds_in_sample, returns_test, preds_test) -> Dict`:
    - Return {train_sharpe, test_sharpe, train_ic, test_ic, overfit_gap_sharpe, overfit_gap_ic}.
- Tests: `/mnt/quant_engine/tests/test_walk_forward_embargo.py`.

**Implementation notes:**
- Embargo gap prevents lookahead bias. If model trained on data[t-250:t], it cannot be tested on data[t:t+5] (too recent, information leakage).
- Slide frequency: "weekly" = slide by 5 days, "daily" = slide by 1 day (slower).
- Overfit gap = in_sample_metric - out_of_sample_metric. Gap > 0.1 (10%) signals overfitting.
- Return multiple folds; user can aggregate or analyze trend.

**Verify:**
- Run walk-forward on synthetic data with known overfitting (model trains on noise); verify out_of_sample_sharpe << in_sample_sharpe.
- Verify embargo gap catches data leakage if embargo=0 (in-sample performance should be much better).

---

### T3: Rolling Information Coefficient and Decay Detection

**What:** Compute rolling IC (rank correlation of predictions vs. actual returns) with decay detection. Warn if IC falls below threshold or trends downward.

**Files:**
- Modify `/mnt/quant_engine/engine/validation.py`:
  - Function `rolling_ic(predictions: np.ndarray, returns: pd.Series, window: int = 60) -> pd.Series`:
    - For each day t, compute IC on [t-window:t] using rank correlation (Spearman).
    - Return Series indexed by date with IC values.
  - Function `detect_ic_decay(ic_series: pd.Series, decay_threshold: float = 0.3, window: int = 20) -> Tuple[bool, Dict]`:
    - Check if IC < decay_threshold for last `window` days (at least 80% of days).
    - Fit linear trend to IC series; flag if slope < -0.001 (declining IC).
    - Return (decaying, {current_ic, mean_ic, slope, days_below_threshold}).
- Tests: `/mnt/quant_engine/tests/test_ic_decay.py`.

**Implementation notes:**
- IC = Spearman rank correlation of (prediction ranks, actual return ranks). Range [-1, 1]; 0 = no predictive power.
- Decay threshold = 0.3 is reasonable for equity alpha (Grinold & Kahn recommend > 0.05 for value).
- Trend: linear regression of IC over last 60 days; slope < -0.001 means decaying.
- Edge case: if window < 20 observations, flag low confidence.

**Verify:**
- Generate synthetic predictions with decaying IC (IC starts at 0.5, decays to 0.1). Verify `detect_ic_decay()` flags decay.
- Real backtest: compute rolling IC on 2-year window, verify IC trends are reasonable (should oscillate, not monotonically decline).

---

### T4: Decile Spread and Predictive Power

**What:** Compute decile spread (top 10% securities return - bottom 10% return) as a metric of predictive power. Include per-regime decomposition.

**Files:**
- Modify `/mnt/quant_engine/engine/evaluation/metrics.py`:
  - Function `decile_spread(predictions: np.ndarray, returns: pd.Series, date_index: pd.DatetimeIndex = None) -> Dict`:
    - Rank predictions into deciles (0–9).
    - For each decile, compute mean return.
    - Decile spread = mean_return[decile_9] - mean_return[decile_0].
    - Return {spread, spread_t_stat, spread_pvalue, decile_returns: [r0, r1, ..., r9]}.
  - If date_index provided, compute rolling decile spread over windows.
  - Per-regime: call decile_spread on each regime slice separately.
- Tests: `/mnt/quant_engine/tests/test_decile_spread.py`.

**Implementation notes:**
- Decile spread is a common metric in ML for predictions (ML-for-trading literature).
- Can include long-short spread (buy top decile, short bottom decile) to isolate alpha.
- t-stat = spread / std_error; if |t| > 1.96, spread is significant at 95%.
- Edge case: if decile has < 5 samples, mark low confidence.

**Verify:**
- Test on synthetic data: predictions = true factor + noise. Verify spread > 0 and correlates with factor signal.
- Test on real backtest: compute decile spread per sector, per regime, verify it makes sense.

---

### T5: Calibration Curve and Overconfidence Detection

**What:** Plot calibration curve (predicted rank percentile vs. actual rank percentile) and compute calibration error. Use existing `ConfidenceCalibrator` from `/mnt/quant_engine/engine/models/calibration.py`.

**Files:**
- Create `/mnt/quant_engine/engine/evaluation/calibration_analysis.py` (new):
  - Function `analyze_calibration(predictions: np.ndarray, returns: pd.Series, bins: int = 10) -> Dict`:
    - Instantiate `ConfidenceCalibrator(n_bins=bins)`.
    - Call `calibrator.calibrate(predictions, returns)`.
    - Compute calibration error (mean squared error of calibration curve vs. diagonal).
    - Return {calibration_error, plot_data: {predicted_percentiles, actual_percentiles}, overconfident: bool}.
  - Overconfident = True if max(|predicted_percentile - actual_percentile|) > 0.2 (20% gap).
- Modify `/mnt/quant_engine/engine/evaluation/metrics.py` to call `analyze_calibration()`.
- Tests: `/mnt/quant_engine/tests/test_calibration_analysis.py`.

**Implementation notes:**
- Do NOT implement new calibration logic. Import `ConfidenceCalibrator` from models/calibration.py.
- Calibration curve tells you: "if model says prediction rank is 80th percentile, does it actually land in 80th percentile?"
- Perfect calibration: predicted_percentile == actual_percentile (diagonal line).
- Overconfident: predicted_percentile > actual_percentile (model thinks it's more confident than warranted).

**Verify:**
- Test on synthetic data with perfect confidence (predictions = true returns + no noise). Verify calibration_error ~0.
- Test on synthetic data with overconfident predictions (noise added). Verify overconfident = True and error > 0.1.

---

### T6: Fragility Metrics (PnL Concentration, Drawdown Distribution, Recovery Time Trend)

**What:** Compute fragility metrics: % PnL from top N trades, drawdown distribution, recovery time distribution, and trend detection for critical slowing down.

**Files:**
- Create `/mnt/quant_engine/engine/evaluation/fragility.py` (new):
  - Function `pnl_concentration(trades: List[Dict], top_n_list: List[int] = [5, 10, 20]) -> Dict`:
    - Input: list of trades with {date, pnl}.
    - For each top_n, compute % of total PnL from top N trades.
    - Return {top_5_pct: 0.45, top_10_pct: 0.62, top_20_pct: 0.80} (example: 45% from top 5 trades).
    - High concentration (> 70% from top 20) signals fragility.
  - Function `drawdown_distribution(returns: pd.Series, window: int = 252) -> Dict`:
    - Compute drawdown (underwater curve) for each 252-day rolling window.
    - Fit distribution: is drawdown from single large drop or many small ones?
    - Return {max_dd_single_day: 0.05, avg_dd_daily: 0.002, dd_from_single_event_pct: 0.70}.
  - Function `recovery_time_distribution(returns: pd.Series, lookback: int = 60) -> pd.Series`:
    - For each drawdown event, compute time to recover (return to previous peak).
    - Return Series of recovery times.
  - Function `detect_critical_slowing_down(recovery_times: pd.Series, window: int = 60) -> Tuple[bool, Dict]`:
    - Fit linear trend to recovery_times over rolling `window`-day windows.
    - Flag if slope > 0.05 (recovery time increasing; danger signal).
    - Return (critical_slowing, {slope, current_recovery_time, historical_median_recovery_time}).
- Tests: `/mnt/quant_engine/tests/test_fragility.py`.

**Implementation notes:**
- PnL concentration: high concentration (> 70% from top 20) is bad (strategy depends on few lucky trades).
- Drawdown distribution: use underwater plot; compute variance of drawdown depths.
- Recovery time: from trough (lowest point in DD) to recovery (return to previous high).
- Critical slowing: recovery time trending upward = warning of regime change (Scheinkman & Woodford).

**Verify:**
- Synthetic data: all returns from single large trade; verify top_5_pct = 100%.
- Real data: compute recovery times during 2008 crisis; verify trend is upward (recovery getting slower).

---

### T7: ML Diagnostics Suite (Feature Importance Drift, Ensemble Disagreement)

**What:** Compute feature importance drift (most important features changing between retraining periods) and ensemble disagreement (if multiple models, how much do they disagree?).

**Files:**
- Create `/mnt/quant_engine/engine/evaluation/ml_diagnostics.py` (new):
  - Function `feature_importance_drift(importance_matrices: Dict[str, np.ndarray]) -> Dict`:
    - Input: Dict of {date: importance_array} from trainer.py retraining logs.
    - For consecutive retraining periods, compute Spearman correlation of importance rankings.
    - Flag if correlation < 0.7 (significant drift).
    - Return {drift_detected, correlations_per_period, top_10_features_change_count}.
  - Function `ensemble_disagreement(predictions: Dict[str, np.ndarray]) -> Dict`:
    - Input: Dict of {model_name: predictions} from multiple models.
    - Compute pairwise rank correlation of predictions.
    - High disagreement (< 0.5 correlation) signals ensemble conflict.
    - Return {mean_correlation, min_correlation, disagreement_pairs}.
- Tests: `/mnt/quant_engine/tests/test_ml_diagnostics.py`.

**Implementation notes:**
- Feature importance drift: rank top 10 features by importance in each period. Spearman rank correlation tells you if ranking changed.
- Ensemble disagreement: if you have 3 models (HMM, jump, rule-based), compute pairwise correlations.
- Low correlation (< 0.5) is a sign of model disagreement; high correlation (> 0.8) means consensus.

**Verify:**
- Synthetic data: feature importance stable across periods. Verify correlation > 0.8.
- Synthetic data: add noise to one model's features. Verify disagreement detected.

---

### T8: Comprehensive Evaluation Report with Visualization

**What:** Orchestrate all evaluation functions into a single `EvaluationEngine` class that generates a comprehensive HTML report with slicing, walk-forward, ML diagnostics, fragility metrics, and visualizations.

**Files:**
- Create `/mnt/quant_engine/engine/evaluation/engine.py` (new):
  - Class `EvaluationEngine`:
    - `__init__(config: Dict)`: Load config from `/mnt/quant_engine/config/validation.yaml`.
    - `evaluate(returns: pd.Series, predictions: np.ndarray, trades: List[Dict], metadata: pd.DataFrame, regime_states: np.ndarray) -> Dict`:
      - Orchestrate all evaluation functions.
      - Call: walk_forward, regime slicing, rolling IC, decile spread, calibration, fragility, ML diagnostics.
      - Return comprehensive results dict.
    - `generate_report(results: Dict, output_path: str, format: str = "html")`:
      - Generate HTML report with sections: summary, regime slicing, walk-forward overfitting, ML diagnostics, fragility, calibration.
      - Include plots: regime slices Sharpe comparison, rolling IC, decile spread, underwater plot, recovery time histogram, calibration curve.
      - Export to HTML or JSON.
- Create `/mnt/quant_engine/engine/evaluation/__init__.py` to export main classes.
- Create visualization functions in `/mnt/quant_engine/engine/evaluation/visualization.py`:
  - Functions: `plot_regime_slices()`, `plot_rolling_ic()`, `plot_decile_spread()`, `plot_underwater()`, `plot_recovery_distribution()`, `plot_calibration_curve()`.
- Tests: `/mnt/quant_engine/tests/test_evaluation_engine.py`.
- Documentation: `/mnt/quant_engine/docs/evaluation_guide.md`.

**Implementation notes:**
- Report structure: summary (aggregate metrics + red flags) → regime slicing → walk-forward analysis → ML diagnostics → fragility analysis → detailed tables.
- Red flags: regime Sharpe differ > 50%, IC decaying, PnL concentration > 70%, overfit gap > 10%, calibration error > 0.15, critical slowing detected.
- Visualization: use matplotlib or plotly for interactive plots.
- JSON export for programmatic consumption (e.g., feed to risk dashboard).

**Verify:**
- Run end-to-end on 2-year backtest. Generate HTML report. Verify all sections populated.
- Verify red flags are raised for obvious overfitting (in_sample_sharpe >> out_of_sample_sharpe).
- Verify plots render correctly.

---

## Validation

### Acceptance criteria

1. **Regime slicing:** `PerformanceSlice` and `SliceRegistry` implemented. 5 primary slices (Normal, High Vol, Crash, Recovery, Trendless) partition returns correctly.
2. **Walk-forward with embargo:** 5-day embargo gap implemented. `overfit_gap_sharpe` computed correctly. Detected when model overfits (in_sample >> out_of_sample).
3. **Rolling IC and decay detection:** `rolling_ic()` computed with 60-day window. `detect_ic_decay()` flags if IC < 0.3 or declining.
4. **Decile spread:** Computed per regime. Spread metric and t-stat report predictive power correctly.
5. **Calibration analysis:** Uses `ConfidenceCalibrator` from models/calibration.py. Overconfidence detected if max gap > 0.2.
6. **Fragility metrics:** PnL concentration, drawdown distribution, recovery time distribution computed. Critical slowing detected if recovery time trending upward.
7. **ML diagnostics:** Feature importance drift and ensemble disagreement computed if applicable.
8. **Evaluation engine:** `EvaluationEngine.evaluate()` returns comprehensive results dict. `generate_report()` produces HTML with all sections.
9. **Tests pass:** All unit tests pass. Coverage > 85%. Integration test on 2-year backtest runs without error.
10. **Documentation:** `/mnt/quant_engine/docs/evaluation_guide.md` is thorough with examples.

### Verification steps

1. Create synthetic regime-labeled returns. Apply regime slices; verify returns partition correctly.
2. Run walk-forward on real backtest data. Verify embargo gap is respected and overfit_gap_sharpe is reasonable (typically 0–5%).
3. Compute rolling IC on real backtest; verify IC trend is reasonable (oscillates, doesn't monotonically decline).
4. Compute decile spread; verify spread > 0 and correlates with model confidence.
5. Build calibration curve; verify it's close to diagonal (well-calibrated) or biased (over/underconfident).
6. Compute recovery times; verify they're positive and reasonable (typically 10–50 days).
7. Run `detect_critical_slowing_down()` on synthetic recovery times with increasing trend; verify flag.
8. Run `EvaluationEngine.evaluate()` on 2-year backtest. Verify report generation succeeds.
9. Inspect HTML report; verify all sections populated, plots render, red flags raised if overfitting present.
10. Run pytest on test suite. Verify coverage > 85%.

### Rollback plan

- **If regime detection fails:** Fall back to time-based slicing (e.g., 2008 vs. 2009+ vs. 2020+).
- **If walk-forward is slow:** Reduce slide frequency from "daily" to "weekly".
- **If ConfidenceCalibrator import fails:** Implement simple calibration (bin predictions, compute empirical calibration).
- **If HTML report generation fails:** Fall back to JSON export.
- **If visualization library missing:** Use matplotlib as fallback (no interactive plots).
- **If tests fail:** Revert to previous version of validation.py, advanced_validation.py.

---

## Notes

- **Slicing reduces sample size:** With 5 primary slices and ~250 trading days/year, each slice has ~50 observations. This is marginal for statistical significance. Always report confidence intervals and flag low-N slices.
- **Walk-forward is computationally expensive:** Full walk-forward with daily sliding on 5-year backtest requires 5*252 = 1260 folds. Use weekly sliding (1260/5 = 252 folds) in production.
- **Critical slowing down as regime warning:** Increasing recovery time (documented in nonlinear dynamics, Scheinkman & Woodford) is a leading indicator of regime shift. Monitor but do not over-interpret (can have spurious trends).
- **Calibration requires predictions with explicit confidence:** If model only outputs point predictions (ranks), calibration curve is less meaningful. Ensure trainer outputs prediction scores (probabilities or distances).
- **Feature importance drift is model-dependent:** Only meaningful if trainer logs feature importance per retraining. If not, skip this check.
- **Fragility metrics are most useful for position-level analysis:** Aggregating to portfolio level (all trades pooled) can hide sector/factor concentration. Consider computing per-sector fragility in future.
