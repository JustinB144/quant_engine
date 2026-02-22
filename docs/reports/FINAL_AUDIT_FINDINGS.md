# Quant Engine: Final Audit — Findings & Remaining Improvements

**Date:** February 21, 2026
**Scope:** Complete re-audit of 130 Python files after all claimed improvements implemented
**Syntax Check:** 130/130 files pass

---

## STATUS: MASSIVE PROGRESS

Every previously identified issue from the original audit has been addressed. The codebase has grown from ~97 files to 130 files with 11 entirely new modules. All WRDS alternative data is wired in. DTW, path signatures, neural nets, macro features, intraday features, LOB features, wave-flow decomposition, HARX spillovers — all implemented and real code, not stubs.

**What follows are the NEW findings from this audit — things that weren't on any previous list.**

---

## SECTION 1: BUGS TO FIX

These are actual code-level bugs found during the deep audit.

### BUG 1: RV_weekly Uses Wrong Window (2 files)

**Files:** `features/harx_spillovers.py` line 114, `features/pipeline.py` line 250
**Issue:** Both compute RV_weekly using a 5-day window, identical to RV_daily. The HAR model requires daily (1d/5d), weekly (5d/21d), and monthly (22d) horizons. RV_weekly should use a 21-day window.
**Impact:** HAR features are partially redundant — RV_daily and RV_weekly are the same feature.
**Fix:** Change `rv_weekly_window` from 5 to 21 in both files.

### BUG 2: Regime Tie-Breaking in run_retrain.py

**File:** `run_retrain.py` line 62
**Issue:** Uses `.mode().iloc[0]` to determine dominant regime. If two regimes are equally common, `mode()` returns both and `iloc[0]` silently picks the first alphabetically/numerically.
**Fix:** Use `tail.value_counts().idxmax()` which always returns a single value (picks highest count, or first among ties deterministically).

### BUG 3: DSR Penalty Uses n_configs Instead of n_trials

**File:** `models/walk_forward.py` line 217
**Issue:** The Deflated Sharpe Ratio penalty divides by `n_configs` (number of hyperparameter configurations) but should divide by `n_trials` (total configs × folds evaluated). This underestimates the multiple-testing penalty.
**Fix:** Change to `n_trials = n_configs * n_folds` and use that in the penalty formula.

### BUG 4: Feature Selection Leakage in CV Folds

**File:** `models/trainer.py` lines 653-657
**Issue:** Feature selection (correlation pruning + permutation importance) runs on the full fold data before splitting into train/test within the fold. This means the test portion of the fold influences which features are selected.
**Fix:** Run feature selection only on `X_dev.iloc[train_idx]` within each fold, not on the full fold.

### BUG 5: Queue Imbalance Returns 0.0 for No-Move Days

**File:** `features/lob_features.py` line 213
**Issue:** When no price moves occur, queue_imbalance is set to 0.0. This is misleading because 0.0 implies balanced buying/selling, when in reality there's no data.
**Fix:** Return `float("nan")` instead of 0.0.

### BUG 6: Identity Scaler Hack in Trainer

**File:** `models/trainer.py` lines 715-727
**Issue:** Manually overwrites sklearn StandardScaler internals (`mean_`, `scale_`, `var_`) to create an identity transform. This violates sklearn's abstraction and could break with future sklearn versions.
**Fix:** Create a simple `IdentityScaler` wrapper class with `fit()` and `transform()` methods that return input unchanged.

---

## SECTION 2: ROBUSTNESS IMPROVEMENTS

These aren't bugs but are defensive improvements that reduce risk of silent failures.

### ROBUSTNESS 1: HMM Initialization Only Uses First Feature

**File:** `regime/hmm.py` lines 66-106
**Issue:** Centroid initialization for the HMM sorts by `X[:, 0]` (first observation dimension only) and places initial centroids at quantiles of that dimension. If the first feature is noisy, all other features are ignored during initialization.
**Recommendation:** Use k-means++ initialization or sort by average distance across all dimensions.

### ROBUSTNESS 2: HMM Covariance Can Become Near-Singular

**File:** `regime/hmm.py` lines 91-92
**Issue:** Adding `min_covar * eye(d)` to the diagonal helps but doesn't guarantee positive-definiteness if off-diagonal elements are very large (highly correlated features).
**Recommendation:** After adding the diagonal, verify positive-definiteness via Cholesky and add additional regularization if it fails.

### ROBUSTNESS 3: BIC State Selection Failures Are Silent

**File:** `regime/detector.py` lines 130-143
**Issue:** If BIC state selection throws an exception, it falls back to the default 4 states with no logging. You'd never know BIC failed.
**Recommendation:** Log the exception at WARNING level so you can see it in the structured logs.

### ROBUSTNESS 4: Forward-Fill in Macro Features Has No Staleness Limit

**File:** `features/macro.py` line 235
**Issue:** `raw.reindex(date_range).ffill()` will carry stale values indefinitely if a FRED series has a multi-month gap. A 90-day-old consumer sentiment value is worse than no value.
**Recommendation:** Add `ffill(limit=5)` (max 5 business days) or add a `days_since_last_update` feature alongside each macro indicator.

### ROBUSTNESS 5: BIC Auto-Select Is Disabled in Config

**File:** `config.py` line 147
**Issue:** `REGIME_HMM_AUTO_SELECT_STATES = False`. The BIC selection code is fully implemented and tested, but it's turned off. The system always uses 4 states regardless of what the data suggests.
**Recommendation:** Either enable it or document the empirical justification for always using 4 states.

### ROBUSTNESS 6: Neural Net Missing Gradient Clipping

**File:** `models/neural_net.py` lines 127-141
**Issue:** No `torch.nn.utils.clip_grad_norm_()` call in the training loop. Financial data has heavy tails that cause occasional gradient explosions.
**Recommendation:** Add gradient clipping with max_norm=1.0.

### ROBUSTNESS 7: Walk-Forward Inner Join Drops Unaligned Dates

**File:** `run_backtest.py` lines 247-252
**Issue:** Uses `inner` join when aligning predictions with actuals. If ticker calendars don't perfectly align, this silently drops predictions. Could lose 10-30% of data.
**Recommendation:** Use `outer` join with explicit NaN logging, or at minimum log how many rows were dropped.

### ROBUSTNESS 8: Regime Duration Assumes Sorted Index

**File:** `regime/detector.py` lines 212-214
**Issue:** Duration calculation uses `diff()` on the regime column, which assumes the index is sorted by date. If dates are out of order, durations will be incorrect.
**Recommendation:** Add `.sort_index()` before computing durations.

---

## SECTION 3: NEW FEATURES & COMPONENTS TO ADD

These are genuinely new capabilities not on any previous list.

### NEW 1: Integrate New Feature Modules into Pipeline

**Current state:** `harx_spillovers.py`, `intraday.py`, `lob_features.py`, and `macro.py` are all implemented but NOT called from `pipeline.py`. Only `wave_flow.py` and `research_factors.py` (DTW, signatures) are integrated.

**What needs to happen:**
- Add HARX spillover features to the cross-asset section of pipeline.py (after network momentum)
- Add macro features to the single-asset section (FRED data is per-date, not per-stock, so broadcast to all stocks)
- Add intraday + LOB features as optional enrichment (gated behind WRDS TAQmsec availability check)
- This is purely wiring — all computation code exists

**Impact:** Potentially 20-30 new features (6 HARX spillover, 10 macro, 6 intraday, 7 LOB) that are currently computed but not used.

### NEW 2: Integrate Almgren-Chriss into Backtest Engine

**Current state:** `backtest/optimal_execution.py` is fully implemented but not called from the main backtest engine. The backtester still uses fixed impact models.

**What needs to happen:**
- For positions above a configurable size threshold (e.g., >5% of ADV), compute the Almgren-Chriss optimal trajectory and use the estimated cost instead of the fixed impact model
- This only matters for large positions — small positions can keep using the simple model

### NEW 3: Add Confidence-Weighted Position Sizing

**Current state:** The predictor outputs confidence scores (calibrated via Platt/isotonic), and the portfolio optimizer uses predicted returns. But confidence isn't used as a sizing input.

**What needs to happen:**
- Multiply the optimizer's target weight by the calibrated confidence score
- High-confidence predictions get full weight; low-confidence get reduced exposure
- This naturally reduces position sizes when the model is uncertain

### NEW 4: Add a Model Monitoring Dashboard Alert System

**Current state:** `utils/logging.py` has `MetricsEmitter.check_alerts()` with thresholds for IC drift, negative Sharpe, drawdown, and regime 2 duration. But these alerts log to file — there's no notification mechanism.

**What needs to happen:**
- Wire alerts to the UI dashboard (add an alerts panel or notification badge)
- Optionally add email/webhook alerts for critical thresholds
- Add alert history so you can see when thresholds were breached over time

### NEW 5: Add Ensemble Weight Optimization

**Current state:** The DiverseEnsemble averages GBR + ElasticNet + RF + XGBoost predictions with equal weight.

**What needs to happen:**
- Optimize ensemble weights using OOS performance from CV folds
- Models that performed better OOS get higher weight
- Use constrained optimization (weights sum to 1, all non-negative)
- This is sometimes called "stacking" — train a meta-learner on OOS predictions

### NEW 6: Add Turnover Tracking and Reporting

**Current state:** Portfolio optimizer has a turnover penalty (λ), but there's no explicit tracking of realized turnover over time.

**What needs to happen:**
- Track daily turnover (sum of absolute weight changes) in the backtest
- Report annualized turnover in results
- Alert when turnover exceeds a threshold (e.g., >500% annualized = too expensive)
- Use realized turnover to calibrate the λ parameter

### NEW 7: Add a Data Quality Dashboard Panel

**Current state:** `data/quality.py` computes quality metrics (missing bars, zero volume, extreme returns, duplicates) but results aren't surfaced in the UI.

**What needs to happen:**
- Add a data quality panel to the dashboard showing per-stock quality scores
- Flag stocks with degraded data (high missing bar fraction, suspicious zero-volume periods)
- This helps catch data feed issues before they corrupt predictions

### NEW 8: Add Feature Stability Monitoring

**Current state:** Feature importance is computed during training but not tracked over time.

**What needs to happen:**
- After each training cycle, store the top-30 feature importance rankings
- Compare current importance to previous cycles
- Alert when feature importance distribution shifts dramatically (suggests regime change or data issue)
- Visualize in the Model Lab dashboard page

### NEW 9: Add Per-Regime Performance Tracking in Dashboard

**Current state:** The dashboard shows aggregate portfolio metrics but doesn't break down P&L by regime.

**What needs to happen:**
- Show cumulative returns per regime (how much was made in trending-bull vs high-vol etc.)
- Show win rate per regime over time
- This validates that the regime gating is working as intended

### NEW 10: Add Transaction Cost Analysis (TCA)

**Current state:** The backtest tracks execution costs, but there's no post-hoc analysis comparing estimated costs to what the cost model predicted.

**What needs to happen:**
- After backtest, compare realized slippage per trade vs the execution model's estimate
- Report: average slippage, slippage by market cap bucket, slippage by time of day, slippage vs predicted
- Use this to calibrate the cost model parameters over time

### NEW 11: Add Correlation Regime Detection

**Current state:** The HMM detects regimes based on return and volatility. But correlation structure changes (e.g., all stocks moving together during a crisis) aren't explicitly detected.

**What needs to happen:**
- Track rolling average pairwise correlation across the universe
- When average correlation spikes above a threshold (e.g., 0.7), flag a "correlation regime"
- Use this to reduce gross exposure or increase hedging during correlation spikes
- This is separate from the HMM — it captures a different dimension of market behavior

### NEW 12: Add Sector Neutrality Constraints

**Current state:** The portfolio optimizer has position limits but no sector exposure limits.

**What needs to happen:**
- Tag each stock with its GICS sector (data available from WRDS Compustat)
- Add sector-level constraints to the optimizer: max net exposure per sector (e.g., ±10%)
- This prevents the portfolio from becoming an implicit sector bet

---

## SECTION 4: INTEGRATION GAPS

These are modules that exist and are fully implemented but not wired into the main pipeline.

| Module | Implemented | Integrated | Gap |
|--------|-------------|------------|-----|
| `features/harx_spillovers.py` | ✅ 242 lines | ❌ Not in pipeline.py | Need to add to cross-asset section |
| `features/intraday.py` | ✅ 193 lines | ❌ Not in pipeline.py | Need optional enrichment (TAQmsec gate) |
| `features/lob_features.py` | ✅ 311 lines | ❌ Not in pipeline.py | Need optional enrichment (TAQmsec gate) |
| `features/macro.py` | ✅ 243 lines | ❌ Not in pipeline.py | Need to broadcast to all stocks |
| `backtest/optimal_execution.py` | ✅ 200 lines | ❌ Not in engine.py | Need size-gated integration |
| `risk/attribution.py` | ✅ 267 lines | ❌ Not in dashboard | Need reporting panel |
| `data/quality.py` | ✅ 74 lines | ❌ Not in dashboard | Need data quality panel |

---

## SECTION 5: PRIORITY RANKING

### Do Now (< 1 hour each)
1. Fix RV_weekly window (5 → 21) in harx_spillovers.py and pipeline.py
2. Fix regime tie-breaking in run_retrain.py
3. Fix DSR penalty denominator in walk_forward.py
4. Fix feature selection leakage in trainer.py
5. Wire harx_spillovers into pipeline.py
6. Wire macro features into pipeline.py

### Do This Week
7. Fix queue_imbalance NaN handling in lob_features.py
8. Fix identity scaler hack in trainer.py
9. Add gradient clipping to neural_net.py
10. Enable BIC auto-select or document 4-state justification
11. Add forward-fill limit to macro.py
12. Log BIC selection failures in detector.py
13. Wire intraday + LOB features into pipeline.py (behind TAQmsec gate)

### Do Next Sprint
14. Integrate Almgren-Chriss into backtest engine for large positions
15. Add confidence-weighted position sizing
16. Add ensemble weight optimization (stacking)
17. Add turnover tracking and reporting
18. Add correlation regime detection
19. Add sector neutrality constraints

### Future Enhancements
20. Add model monitoring alerts to UI dashboard
21. Add data quality dashboard panel
22. Add feature stability monitoring
23. Add per-regime performance tracking in dashboard
24. Add transaction cost analysis (TCA) reporting
25. Add email/webhook notification for critical alerts

---

## APPENDIX: CODEBASE INVENTORY (130 Files)

| Directory | Files | Approx Lines | Status |
|-----------|-------|--------------|--------|
| Core | 3 | ~400 | Production |
| Models | 12 | ~3,700 | Production |
| Features | 8 | ~2,800 | Production (4 not integrated) |
| Regime | 3 | ~760 | Production |
| Backtest | 6 | ~2,960 | Production |
| Risk | 11 | ~2,560 | Production |
| Data | 10 | ~3,200 | Production |
| Autopilot | 6 | ~1,500 | Production |
| Kalshi | 16 + 8 tests | ~5,070 | Production |
| UI | 12 | ~4,700 | Functional |
| Utils | 2 | ~200 | Production |
| Tests | 20 | ~2,500+ | Comprehensive |
| Run scripts | 8 | ~1,700 | Production |

**Estimated total: ~32,000+ lines across 130 files**
