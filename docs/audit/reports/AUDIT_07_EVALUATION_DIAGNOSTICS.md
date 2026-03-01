# Audit Report — Subsystem 7: Evaluation & Diagnostics

> **Status:** COMPLETE
> **Auditor:** Claude Opus 4.6
> **Date:** 2026-02-28
> **Spec:** SPEC_AUDIT_07_EVALUATION_DIAGNOSTICS.md

---

## Summary

| Metric | Value |
|--------|-------|
| Files audited | 8 / 8 (100%) |
| Lines audited | 2,816 / 2,816 (100%) |
| Line count verified | Exact match to SUBSYSTEM_MAP.json |
| Fan-in verified | 0 (leaf module confirmed — no production code imports from evaluation/) |
| Cross-module edges (outbound) | 10 (7 to config, 2 to backtest/validation, 1 to models/calibration) |
| Boundary contracts verified | 2 / 2 (evaluation_to_models_backtest_11, evaluation_to_config_23) |
| Findings | 3 MEDIUM, 5 LOW, 4 INFO |

**Overall assessment: PASS** — No HIGH or CRITICAL findings. The subsystem is well-structured, metrics are statistically sound, and all external contracts match. Three MEDIUM findings involve dead config constants that have no functional impact but represent config-code desynchronization.

---

## T1: Ledger and Metric Inventory Baseline

### File Ledger

| File | Lines | Defs/Classes | Config Imports | Cross-Module Imports |
|------|------:|:-------------|:---------------|:---------------------|
| `evaluation/__init__.py` | 33 | 0 defs, 0 classes | 0 | 0 (intra-module only) |
| `evaluation/engine.py` | 826 | 6 defs, 3 classes | 15 constants | 2 (backtest/validation.py) |
| `evaluation/metrics.py` | 324 | 5 defs, 0 classes | 2 constants | 0 |
| `evaluation/slicing.py` | 400 | 5 static methods, 2 classes | 1 constant + 1 lazy (REGIME_NAMES) | 0 |
| `evaluation/fragility.py` | 452 | 6 defs, 0 classes | 4 constants | 0 |
| `evaluation/calibration_analysis.py` | 142 | 2 defs, 0 classes | 2 constants | 1 (models/calibration.py, lazy) |
| `evaluation/ml_diagnostics.py` | 253 | 4 defs, 0 classes | 2 constants | 0 |
| `evaluation/visualization.py` | 386 | 7 defs, 0 classes | 0 | 0 |
| **Total** | **2,816** | | **26 config refs** | **3 cross-module** |

### Metric Registry

All metrics produced by this subsystem and their sources:

| Metric | Produced By | Key/Field Name | Consumers |
|--------|------------|----------------|-----------|
| Sharpe Ratio | `metrics.py:compute_slice_metrics` | `sharpe` | engine.py, HTML report |
| Sharpe SE | `metrics.py:compute_slice_metrics` | `sharpe_se` | HTML report |
| Mean Return | `metrics.py:compute_slice_metrics` | `mean_return` | HTML report |
| Annualized Return | `metrics.py:compute_slice_metrics` | `annualized_return` | HTML report |
| Max Drawdown | `metrics.py:compute_slice_metrics` | `max_dd` | HTML report |
| Max DD Duration | `metrics.py:compute_slice_metrics` | `max_dd_duration` | HTML report |
| Recovery Time Mean | `metrics.py:compute_slice_metrics` | `recovery_time_mean` | HTML report |
| Win Rate | `metrics.py:compute_slice_metrics` | `win_rate` | HTML report |
| Information Coefficient | `metrics.py:compute_slice_metrics` | `ic` | engine.py summary, HTML report |
| Confidence Level | `metrics.py:compute_slice_metrics` | `confidence` | HTML report (low-conf styling) |
| Decile Spread | `metrics.py:decile_spread` | `spread` | HTML report |
| Spread T-Stat | `metrics.py:decile_spread` | `spread_t_stat` | HTML report |
| Spread P-Value | `metrics.py:decile_spread` | `spread_pvalue` | HTML report |
| Monotonicity | `metrics.py:decile_spread` | `monotonicity` | HTML report |
| PnL Concentration | `fragility.py:pnl_concentration` | `top_{n}_pct` | engine.py red flags |
| Herfindahl Index | `fragility.py:pnl_concentration` | `herfindahl_index` | HTML report |
| Drawdown Distribution | `fragility.py:drawdown_distribution` | 7 fields | HTML report |
| Recovery Time Distribution | `fragility.py:recovery_time_distribution` | pd.Series | engine.py, critical slowing |
| Critical Slowing Down | `fragility.py:detect_critical_slowing_down` | `critical_slowing` | engine.py red flags |
| Loss Streaks | `fragility.py:consecutive_loss_frequency` | 4 fields | engine.py (stored, not red-flagged) |
| Feature Importance Drift | `ml_diagnostics.py:feature_importance_drift` | `drift_detected` | engine.py red flags |
| Ensemble Disagreement | `ml_diagnostics.py:ensemble_disagreement` | `high_disagreement` | engine.py red flags |
| ECE | `calibration_analysis.py:analyze_calibration` | `ece` | HTML report |
| Calibration Error | `calibration_analysis.py:analyze_calibration` | `calibration_error` | engine.py red flags |
| Overconfidence | `calibration_analysis.py:analyze_calibration` | `overconfident` | HTML report |
| Walk-Forward Results | `backtest/validation.py` (imported) | `WalkForwardEmbargoResult` | engine.py red flags, HTML report |
| Rolling IC | `backtest/validation.py` (imported) | `pd.Series` | engine.py, HTML report |
| IC Decay | `backtest/validation.py` (imported) | `tuple[bool, dict]` | engine.py red flags |

### Red Flag Categories (7 total)

| Category | Severity | Trigger Condition | Source |
|----------|----------|------------------|--------|
| `regime` | warning | Sharpe range / max(abs(Sharpe)) > EVAL_REGIME_SHARPE_DIVERGENCE (0.5) | engine.py:261 |
| `overfit` | **critical** | `wf.is_overfit == True` from walk_forward_with_embargo | engine.py:302 |
| `ic_decay` | warning | `detect_ic_decay()` returns `decaying=True` | engine.py:333 |
| `calibration` | warning | `calibration_error > EVAL_CALIBRATION_ERROR_THRESHOLD` (0.15) | engine.py:359 |
| `fragility` | warning | `pnl_concentration.fragile == True` (hardcoded 0.70 threshold) | engine.py:379 |
| `slowing` | **critical** | `detect_critical_slowing_down()` returns `critical=True` | engine.py:410 |
| `feature_drift` | warning | `feature_importance_drift.drift_detected == True` | engine.py:434 |
| `ensemble_disagreement` | warning | `ensemble_disagreement.high_disagreement == True` | engine.py:447 |

### Public API (__init__.py exports)

```python
__all__ = [
    "PerformanceSlice", "SliceRegistry",         # slicing.py
    "compute_slice_metrics", "decile_spread",     # metrics.py
    "pnl_concentration", "drawdown_distribution", # fragility.py
    "recovery_time_distribution", "detect_critical_slowing_down",
    "feature_importance_drift", "ensemble_disagreement",  # ml_diagnostics.py
    "analyze_calibration",                        # calibration_analysis.py
    "EvaluationEngine",                           # engine.py
]
```

**Note:** `consecutive_loss_frequency` (fragility.py) is used by engine.py but NOT exported from `__init__.py`. See Finding L-1.

---

## T2: Time-Ordering and Validation Contract Pass

### Walk-Forward with Embargo (engine.py:269-309)

**Contract:** `backtest.validation.walk_forward_with_embargo`

| Aspect | Status | Evidence |
|--------|--------|----------|
| Import type | Lazy (inside try/except) | engine.py:272 |
| Signature match | PASS | `walk_forward_with_embargo(returns, predictions, train_window, embargo, test_window, slide_freq)` — all 6 parameters used correctly |
| Return shape match | PASS | Accesses `wf.n_folds`, `wf.mean_train_sharpe`, `wf.mean_test_sharpe`, `wf.mean_overfit_gap`, `wf.is_overfit`, `wf.warnings`, and fold fields — all exist on `WalkForwardEmbargoResult` / `WalkForwardEmbargoFold` |
| Embargo semantics | PASS | Embargo gap is correctly forwarded via `self.embargo_days` (default: 5 days from `EVAL_WF_EMBARGO_DAYS`) |
| Error handling | PASS | Wrapped in try/except, failure logged, evaluation continues with `wf_result = None` |

### Rolling IC and Decay (engine.py:311-340)

**Contract:** `backtest.validation.rolling_ic`, `backtest.validation.detect_ic_decay`

| Aspect | Status | Evidence |
|--------|--------|----------|
| Import type | Lazy (inside try/except) | engine.py:315 |
| `rolling_ic` signature | PASS | Called as `rolling_ic(pred_arr, returns, window=self.ic_window)` — matches `def rolling_ic(predictions, returns, window=60)` |
| `detect_ic_decay` signature | PASS | Called as `detect_ic_decay(ic_series, decay_threshold=self.ic_decay_threshold)` — matches `def detect_ic_decay(ic_series, decay_threshold=0.02, window=20)` |
| Return shape | PASS | `(decaying: bool, decay_info: dict)` — correctly destructured at line 327 |
| No future data | PASS | rolling_ic uses rolling Spearman — backward-looking only |

**Note:** EVAL_IC_DECAY_LOOKBACK (20) is imported but never passed to `detect_ic_decay()`'s `window` parameter. See Finding M-2.

### Fragility Analysis (engine.py:369-422)

| Aspect | Status | Evidence |
|--------|--------|----------|
| `pnl_concentration` | PASS | Uses trades list, no time-order concern |
| `drawdown_distribution` | PASS | Uses cumulative product — backward-looking only |
| `recovery_time_distribution` | PASS | Computes from drawdown trough to recovery — backward-looking only |
| `detect_critical_slowing_down` | PASS | Linear regression on recovery time series — backward-looking only |
| `consecutive_loss_frequency` | PASS | Iterates returns sequentially — backward-looking only |

### Transition Slicing (slicing.py:273-324)

| Aspect | Status | Details |
|--------|--------|---------|
| `center=True` rolling | INFO | Uses centered rolling window — bars BEFORE a regime transition are included in "near_transition". This is post-hoc evaluation (not used for trading decisions), so forward-looking information is acceptable for analytical decomposition. |

### Metadata Construction (slicing.py:326-400)

| Column | Computation | Time-Safe? |
|--------|-------------|:----------:|
| `regime` | Direct assignment from regime_states | PASS |
| `cumulative_return` | `(1+returns).cumprod() - 1` | PASS (backward) |
| `drawdown` | `(cum_eq - running_max) / running_max` | PASS (backward) |
| `trailing_return_20d` | `returns.rolling(20).sum()` | PASS (backward) |
| `volatility` | `returns.rolling(20).std() * sqrt(252)` | PASS (backward) |
| `volatility_median` | `vol.rolling(252).median()` | PASS (backward) |
| `uncertainty` | Direct assignment | PASS |

**Verdict: No future data leakage in any evaluation computation.**

---

## T3: Calibration and Diagnostics Pass

### Calibration Analysis (calibration_analysis.py)

**Contract:** `models.calibration.compute_ece`, `models.calibration.compute_reliability_curve`

| Aspect | Status | Evidence |
|--------|--------|----------|
| Import type | Lazy (line 92, inside function body) | PASS |
| `compute_ece` signature | PASS | Called as `compute_ece(conf_arr, outcomes, n_bins=bins)` — matches `def compute_ece(predicted_probs, actual_outcomes, n_bins=10) -> float` |
| `compute_reliability_curve` signature | PASS | Called as `compute_reliability_curve(conf_arr, outcomes, n_bins=bins)` — matches `def compute_reliability_curve(predicted_probs, actual_outcomes, n_bins=10) -> dict` |
| Return dict keys consumed | PASS | `reliability["avg_predicted"]`, `reliability["observed_freq"]`, `reliability["bin_counts"]` — all documented return keys |
| Calibration error (MSE) | PASS | `np.mean(gaps ** 2)` where `gaps = abs(avg_pred - obs_freq)` — standard MSE of calibration gap |
| Overconfidence detection | PASS | `max_gap > overconfidence_threshold` (default 0.20 from EVAL_OVERCONFIDENCE_THRESHOLD) |
| Low-N handling | PASS | Returns empty result if `n < bins * 2` (line 69) |
| NaN handling | PASS | Filters `np.isfinite(pred_arr) & np.isfinite(ret_arr)` at line 64 |

**Confidence score handling (lines 72-89):**
- If `confidence_scores` provided: binary outcome (direction correctness), confidence = provided scores
- If `confidence_scores` not provided: outcome = return percentile, confidence = prediction percentile
- Fallback at line 78: if confidence_scores don't align after filtering, falls back to prediction magnitude

### ML Diagnostics (ml_diagnostics.py)

| Function | Statistical Method | Correctness |
|----------|-------------------|:-----------:|
| `feature_importance_drift` | Spearman rank correlation between consecutive importance vectors | PASS |
| `ensemble_disagreement` | Pairwise Spearman rank correlation among ensemble members | PASS |

**Thresholds verified:**

| Threshold | Config Constant | Value | Used At |
|-----------|----------------|-------|---------|
| Feature drift | `EVAL_FEATURE_DRIFT_THRESHOLD` | 0.7 | ml_diagnostics.py:138 |
| Ensemble disagreement | `EVAL_ENSEMBLE_DISAGREEMENT_THRESHOLD` | 0.5 | ml_diagnostics.py:153 |

**Edge cases:**
- `feature_importance_drift`: handles < 2 periods (returns no drift), length mismatches (logs warning, correlation = 0.0), missing scipy (fallback `_manual_spearman`)
- `ensemble_disagreement`: handles < 2 models (returns no disagreement), length mismatches (truncates), missing scipy (fallback `_manual_spearman`)
- `_manual_spearman` (line 242): correct manual implementation using rank correlation formula

---

## T4: Visualization and Reporting Integrity Pass

### Visualization Functions (visualization.py)

| Function | Mutates Source Data? | Missing Data Handling | Low-Sample Handling |
|----------|:-------------------:|:---------------------:|:-------------------:|
| `plot_regime_slices` | No (reads .get("sharpe")) | Returns `html: None` if no names | PASS |
| `plot_rolling_ic` | No (.dropna() on copy) | Returns `html: None` if empty | PASS |
| `plot_decile_spread` | No (reads list values) | Returns `html: None` if n_q=0 | PASS |
| `plot_underwater` | No (computes from values) | Returns `html: None` if len=0 | PASS |
| `plot_recovery_distribution` | No (.dropna() on copy) | Returns `html: None` if empty | PASS |
| `plot_calibration_curve` | No (reads dict values) | Filters NaN bins (line 294) | PASS |
| `plot_walk_forward_folds` | No (reads .get()) | Returns `html: None` if no folds | PASS |

**All visualization functions:**
- Return `{"html": ..., "data": ...}` — consistent contract
- Never modify input data
- Gracefully degrade when plotly is unavailable (`_HAS_PLOTLY = False`)
- Include raw `data` dict for JSON export regardless of plotly availability

**Finding L-2:** `plot_walk_forward_folds` at line 351 has redundant `.get()` — `f.get("train_sharpe", f.get("train_sharpe", 0.0))` where inner and outer keys are identical. Harmless but should be `f.get("train_sharpe", 0.0)`.

### Report Generation (engine.py:493-827)

| Aspect | Status | Evidence |
|--------|--------|----------|
| JSON serialization | PASS | Uses `_strip_non_serializable` recursively + `default=str` fallback |
| HTML template | PASS | Uses `.format()` with `{content}` placeholder, CSS properly escaped with `{{` |
| Metric display accuracy | PASS | All f-string format specifiers match metric semantics (e.g., `.2%` for drawdowns, `.4f` for IC) |
| Red flag rendering | PASS | Critical/warning icons and CSS classes properly applied |
| Directory creation | PASS | `path.parent.mkdir(parents=True, exist_ok=True)` at line 516 |

**`_strip_non_serializable` (engine.py:812-826):**
- Handles: `None`, `dict`, `list`, `tuple`, `np.integer`, `np.int64`, `np.floating`, `np.float64`, `np.ndarray`
- Does NOT handle: `pd.Timestamp`, `pd.Series`, `pd.DataFrame` — covered by `json.dump(default=str)` fallback
- Adequate for all evaluation output types

---

## T5: Boundary Checks and Closure

### Boundary: evaluation_to_models_backtest_11

| Symbol | Provider | Import Line | Type | Signature Match | Status |
|--------|----------|:-----------:|:----:|:---------------:|:------:|
| `compute_ece` | models/calibration.py | 92 | lazy | PASS | VERIFIED |
| `compute_reliability_curve` | models/calibration.py | 92 | lazy | PASS | VERIFIED |
| `walk_forward_with_embargo` | backtest/validation.py | 272 | conditional | PASS | VERIFIED |
| `rolling_ic` | backtest/validation.py | 315 | conditional | PASS | VERIFIED |
| `detect_ic_decay` | backtest/validation.py | 315 | conditional | PASS | VERIFIED |

**Shared artifact:** `results/backtest_*d_summary.json` is listed as consumed by evaluation/engine.py in INTERFACE_CONTRACTS.yaml. However, **evaluation/engine.py does NOT read any files** — it receives all data via function parameters. See Finding I-2.

### Boundary: evaluation_to_config_23

| Importing File | Constants Imported | Import Type | All Exist? |
|---------------|-------------------|:-----------:|:----------:|
| engine.py:27 | 15 EVAL_* constants | top_level | PASS |
| calibration_analysis.py:19 | EVAL_CALIBRATION_BINS, EVAL_OVERCONFIDENCE_THRESHOLD | top_level | PASS |
| fragility.py:20 | EVAL_TOP_N_TRADES, EVAL_RECOVERY_WINDOW, EVAL_CRITICAL_SLOWING_WINDOW, EVAL_CRITICAL_SLOWING_SLOPE_THRESHOLD | top_level | PASS |
| metrics.py:20 | EVAL_MIN_SLICE_SAMPLES, EVAL_DECILE_SPREAD_MIN | top_level | PASS |
| ml_diagnostics.py:26 | EVAL_FEATURE_DRIFT_THRESHOLD, EVAL_ENSEMBLE_DISAGREEMENT_THRESHOLD | top_level | PASS |
| slicing.py:23 | EVAL_MIN_SLICE_SAMPLES | top_level | PASS |
| slicing.py:193 | REGIME_NAMES | lazy | PASS |

**All 22 unique config constants verified to exist in config.py with STATUS: ACTIVE.**

### Fan-In Verification

Confirmed 0 fan-in from production code. Only consumers of evaluation/ are:
- `tests/test_evaluation_*.py` (5 test files)
- `tests/test_spec_v01_uncertainty_slicing.py`
- `tests/test_package_installation.py` (smoke test)

No production module imports from evaluation. **Leaf module status confirmed.**

### Cross-Module Dependency Edge Summary

| # | Source File | Target File | Symbols | Type |
|---|-----------|------------|---------|------|
| 1 | engine.py:27 | config.py | 15 EVAL_* constants | top_level |
| 2 | engine.py:272 | backtest/validation.py | walk_forward_with_embargo | conditional |
| 3 | engine.py:315 | backtest/validation.py | rolling_ic, detect_ic_decay | conditional |
| 4 | calibration_analysis.py:19 | config.py | 2 EVAL_* constants | top_level |
| 5 | calibration_analysis.py:92 | models/calibration.py | compute_ece, compute_reliability_curve | lazy |
| 6 | fragility.py:20 | config.py | 4 EVAL_* constants | top_level |
| 7 | metrics.py:20 | config.py | 2 EVAL_* constants | top_level |
| 8 | ml_diagnostics.py:26 | config.py | 2 EVAL_* constants | top_level |
| 9 | slicing.py:23 | config.py | 1 constant | top_level |
| 10 | slicing.py:193 | config.py | REGIME_NAMES | lazy |

**Total: 10 cross-module edges (7 to config, 2 to backtest, 1 to models) — matches DEPENDENCY_EDGES.json.**

---

## Findings

### MEDIUM Severity

#### M-1: Hardcoded fragility threshold bypasses config

**File:** `fragility.py:101`
**Evidence:**
```python
result["fragile"] = result.get(fragile_key, 0.0) > 0.70
```
`EVAL_PNL_CONCENTRATION_THRESHOLD` is defined in config.py as `0.70` and imported by engine.py but **never used**. The threshold is hardcoded in fragility.py. Changing the config value has no effect.

**Impact:** Config desynchronization — operators cannot tune the fragility threshold via config.
**Recommendation:** Pass the threshold as a parameter from engine.py, or import and use `EVAL_PNL_CONCENTRATION_THRESHOLD` in fragility.py.

---

#### M-2: EVAL_IC_DECAY_LOOKBACK imported but never forwarded

**File:** `engine.py:34` (import), `engine.py:327` (call site)
**Evidence:**
```python
# Imported at line 34:
EVAL_IC_DECAY_LOOKBACK,  # value = 20

# Called at line 327-328 WITHOUT the window parameter:
decaying, decay_info = detect_ic_decay(
    ic_series, decay_threshold=self.ic_decay_threshold,
)
# detect_ic_decay's window parameter defaults to 20
```
The config constant matches the function default today, but if `EVAL_IC_DECAY_LOOKBACK` is changed in config, the change has no effect — the function will still use its default of 20.

**Impact:** Config change silently ineffective.
**Recommendation:** Forward the constant: `detect_ic_decay(ic_series, decay_threshold=..., window=EVAL_IC_DECAY_LOOKBACK)`, or remove the import.

---

#### M-3: EVAL_OVERFIT_GAP_THRESHOLD imported but never applied

**File:** `engine.py:40` (import), `engine.py:301-307` (overfit detection)
**Evidence:**
```python
# Imported at line 40:
EVAL_OVERFIT_GAP_THRESHOLD,  # value = 0.10

# Overfit detection at line 301-302 uses only wf.is_overfit:
if wf.is_overfit:
    red_flags.append(RedFlag(category="overfit", ...))
```
The overfit gap threshold is defined in evaluation config but the engine delegates overfit detection entirely to the `walk_forward_with_embargo` function in backtest/validation.py, which uses its own internal threshold. The evaluation-level threshold is dead code.

**Impact:** Evaluation config has no independent control over overfitting sensitivity.
**Recommendation:** Either apply `EVAL_OVERFIT_GAP_THRESHOLD` as a secondary check (e.g., `wf.mean_overfit_gap > EVAL_OVERFIT_GAP_THRESHOLD`), or remove the import.

---

### LOW Severity

#### L-1: `consecutive_loss_frequency` not exported from `__init__.py`

**File:** `evaluation/__init__.py`
**Evidence:** The function is defined in `fragility.py`, imported and used by `engine.py:420`, but is not listed in `__init__.py`'s `__all__` or imports. External consumers cannot access it via `from evaluation import consecutive_loss_frequency`.
**Impact:** Inconsistent public API surface. Minor — function is used internally.

---

#### L-2: Redundant `.get()` in `plot_walk_forward_folds`

**File:** `visualization.py:351-352`
**Evidence:**
```python
is_sharpes = [f.get("train_sharpe", f.get("train_sharpe", 0.0)) for f in folds]
oos_sharpes = [f.get("test_sharpe", f.get("test_sharpe", 0.0)) for f in folds]
```
The inner and outer `.get()` keys are identical, so the inner call provides no different fallback. Should be `f.get("train_sharpe", 0.0)`.
**Impact:** No functional impact, purely cosmetic.

---

#### L-3: Five unused config imports in engine.py

**File:** `engine.py:34-42`
**Evidence:** The following constants are imported but never referenced in the function body:
- `EVAL_IC_DECAY_LOOKBACK` (line 34) — see M-2
- `EVAL_RECOVERY_WINDOW` (line 37) — used only by fragility.py directly
- `EVAL_CRITICAL_SLOWING_WINDOW` (line 38) — used only by fragility.py directly
- `EVAL_OVERFIT_GAP_THRESHOLD` (line 40) — see M-3
- `EVAL_PNL_CONCENTRATION_THRESHOLD` (line 41) — see M-1

**Impact:** Dead imports. No runtime effect.

---

#### L-4: `EVAL_DECILE_SPREAD_MIN` imported but unused in metrics.py

**File:** `metrics.py:20`
**Evidence:**
```python
from ..config import EVAL_MIN_SLICE_SAMPLES, EVAL_DECILE_SPREAD_MIN
```
`EVAL_DECILE_SPREAD_MIN` (0.005) is imported but never referenced in any function in metrics.py. Presumably intended for validating the spread magnitude but not implemented.
**Impact:** Dead import. The minimum expected spread for a good predictor is not enforced.

---

#### L-5: Silent regime fallback on length mismatch

**File:** `slicing.py:356-359`
**Evidence:**
```python
if len(regime_states) == n:
    meta["regime"] = regime_states
else:
    meta["regime"] = np.full(n, 2, dtype=int)  # Default to mean_reverting
```
If regime_states has a different length than returns, all bars silently default to regime 2 (mean_reverting). No warning is logged. This could mask data alignment bugs where the wrong regime array is passed.
**Impact:** Silent failure mode — all regime-based slicing would show performance concentrated in "Mean Reverting" with no indication of the error.

---

### INFO

#### I-1: scipy fallback degrades IC silently

**File:** `metrics.py:15-18, 108-110`
**Evidence:** If scipy is not installed, `sp_stats = None` and IC calculation is silently skipped (returns 0.0 with no warning). All other scipy-dependent computations (decile_spread t-test, monotonicity) similarly degrade.
**Impact:** Functional degradation without notification. In practice, scipy is always installed in this codebase.

---

#### I-2: Shared artifact claim inaccurate for evaluation/engine.py

**File:** INTERFACE_CONTRACTS.yaml line 1071
**Evidence:** Lists `evaluation/engine.py` as a reader of `results/backtest_*d_summary.json`. However, evaluation/engine.py contains **no file I/O** — it receives all data via function parameters to `evaluate()`. The actual readers are entry points or scripts that read the JSON and pass data to the engine.
**Impact:** Documentation inaccuracy in INTERFACE_CONTRACTS.yaml. No functional impact.

---

#### I-3: Trailing return uses arithmetic sum

**File:** `slicing.py:371-372`
**Evidence:** `trail_20 = returns.rolling(20, min_periods=1).sum()` — this is an arithmetic sum of daily returns, not compound trailing return `(1+r).cumprod()-1`. For daily equity returns, the difference is negligible (< 0.01% for typical values).
**Impact:** Negligible approximation error. Standard practice for short windows.

---

#### I-4: Transition slices use centered (look-ahead) window

**File:** `slicing.py:298-302`
**Evidence:** `changes.rolling(window, center=True, min_periods=1).max()` — marks bars BEFORE a regime transition as "near_transition". This is a forward-looking operation.
**Impact:** None — this is post-hoc evaluation of completed backtests, not live decision-making. The centered window is intentional for analytical decomposition of performance around regime boundaries.

---

## Metric Correctness Verification

### Sharpe Ratio (metrics.py:67-72)

```python
rf_per_period = risk_free_rate / annual_trading_days   # 0.04/252
excess = ret_arr - rf_per_period
sharpe = np.mean(excess) / std_ret * np.sqrt(annual_trading_days)
```

- **Formula:** Annualized Sharpe = (mean excess return / std of returns) * sqrt(252)
- **Risk-free adjustment:** Daily rate = annual rate / 252 (simple, not compound — standard practice)
- **Standard deviation:** Uses `ddof=1` (sample std) — correct
- **Annualization:** Multiplies by sqrt(252) — standard assumption of IID returns
- **Verdict:** CORRECT

### Sharpe SE / Lo (2002) (metrics.py:74-77)

```python
sharpe_se = np.sqrt((1 + 0.5 * sharpe ** 2) / n)
```

- **Formula:** SE(SR) = sqrt((1 + SR^2/2) / n) under IID normality
- **Reference:** Lo (2002), "The Statistics of Sharpe Ratios"
- **Verdict:** CORRECT (under normality assumption)

### Max Drawdown (metrics.py:79-83)

```python
cum_eq = np.cumprod(1 + ret_arr)
running_max = np.maximum.accumulate(cum_eq)
drawdowns = (cum_eq - running_max) / np.where(running_max > 0, running_max, 1.0)
max_dd = np.min(drawdowns)
```

- **Formula:** DD = (equity - peak) / peak, max_dd = min of all drawdowns
- **Division safety:** Uses `np.where(running_max > 0, ...)` to avoid division by zero
- **Verdict:** CORRECT

### Critical Slowing Down (fragility.py:248-337)

- **Method:** Linear regression of recovery times over episodes + normalized slope test
- **Normalization:** `slope / (median_rt + 1e-12)` — scale-invariant
- **Dual trigger:** `normalized_slope > threshold OR (recent_trend == "increasing" AND current_rt > 2 * median_rt)`
- **Verdict:** CORRECT — reasonable heuristic aligned with Scheinkman & Woodford framework

### ECE and Calibration (calibration_analysis.py)

- **ECE:** Delegated to `models/calibration.py:compute_ece` — verified correct implementation
- **Calibration error:** MSE of (predicted - observed) across non-empty bins — correct
- **Overconfidence:** max_gap > threshold — correct
- **Verdict:** CORRECT

---

## Acceptance Criteria Disposition

| # | Criterion | Status | Evidence |
|---|-----------|:------:|----------|
| 1 | 100% line coverage across all 8 files | PASS | All 2,816 lines across 8 files read and audited |
| 2 | Time-order/embargo diagnostics validated | PASS | All computations backward-looking; embargo correctly forwarded; centered window in transition slices is intentional for post-hoc analysis |
| 3 | Calibration contracts with models validated | PASS | compute_ece and compute_reliability_curve signatures, parameters, and return shapes verified against models/calibration.py |
| 4 | Evaluation report fields stable and accurately derived | PASS | All metrics verified; visualization functions do not mutate data; JSON serialization handles all types |

---

## Carry-Forward to Subsystem 8 (Autopilot)

Per the transition guide:

- **Evaluation metrics and red flag detection** — autopilot uses evaluation results indirectly through promotion decisions. No direct code dependency from autopilot → evaluation (confirmed 0 fan-in).
- **All upstream subsystem contracts** (config, data, features, regime, backtest/risk, models) carry forward to autopilot, which depends on ALL of them.
- **Boundary to check:** `autopilot_to_multi_4` — verify all 23+ cross-module imports in autopilot/engine.py.
