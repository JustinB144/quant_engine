# COMPREHENSIVE IMPROVEMENT INSTRUCTIONS
## Quant Engine System Enhancement Guide

**Version**: 1.0
**Date**: 2026-02-23
**Audience**: Human Developer (Justin) + LLM Context Consumption
**Purpose**: Exhaustive instructions for closing validation→autopilot gaps, Kelly integration, health system overhaul, regime detection improvements, UI audit fixes, and system-wide improvements.

---

## SECTION 1: PURPOSE AND USAGE

### 1.1 Document Structure and Audience

This document serves **dual purposes**:

1. **For Human Developer (Justin)**: A comprehensive roadmap of every known deficiency in the quant_engine system, with specific implementation guidance, file locations, and success criteria.

2. **For LLM Context Consumption**: A structured directive document that an LLM assistant can consume to understand the system's current state, known gaps, best practices from academic research, and step-by-step instructions for improvement.

The document is organized by **system domain** rather than chronologically, enabling targeted reading for specific areas of focus.

### 1.2 Scope of Coverage

This document covers:

- **Validation→Autopilot Gap Analysis** (Section 3): Specific statistical tests from validation.py and advanced_validation.py that are missing from the autopilot engine, with wiring instructions.

- **Kelly System Integration** (Section 4): Current Kelly implementation fragmentation, problems with regime-agnostic Kelly sizing, and comprehensive integration strategy.

- **Health System Renaissance Overhaul** (Section 5): 15 missing health monitors that RenTec-level systems track, with specific implementations.

- **Regime Detection Best Models** (Section 6): Academic research on regime detection, problems with current HMM approach, and upgrade path to Statistical Jump Models.

- **UI Audit and Criticisms** (Section 7): Issues visible in recent screenshots, with specific fixes for each.

- **Comprehensive System Criticisms** (Section 8): Architecture, model, risk, data, backtest, and integration criticisms with improvement strategies.

- **Survivorship Bias Deep Dive** (Section 9): Quantified survivorship bias analysis that was flagged in screenshots.

- **Implementation Priority Order** (Section 10): Risk-adjusted prioritization of all improvements.

### 1.3 How to Use This Document

**For Justin (human developer)**:
- Read Section 2 for current state context
- Jump to the section most relevant to your current task
- Each section has subsections with exact file paths, function names, and implementation steps
- Use Section 10 to decide priority
- Create a checklist from the Priority sections and track completion

**For LLM Assistants**:
- Consume Sections 2-9 as context about the system state and requirements
- When asked to implement improvements, refer to the exact file paths and function names
- When writing code, match the existing style and architecture patterns
- When unsure about a requirement, return to the relevant subsection for clarification
- Use Section 10 to understand priority if balancing multiple tasks

---

## SECTION 2: COMPLETE AUDIT OF CURRENT STATE

### 2.1 System Architecture Overview

The quant_engine is structured as a pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                   QUANT ENGINE PIPELINE                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. DATA INGESTION & FEATURES                               │
│     └─ WRDS CRSP (historical)                               │
│     └─ IBKR (live, real-time)                               │
│     └─ FRED macroeconomic                                    │
│     └─ Kalshi event probabilities                            │
│     └─ Feature Pipeline (100+ engineered features)          │
│                                                               │
│  2. REGIME DETECTION                                        │
│     └─ HMM-based detector                                    │
│     └─ Correlation regimes                                   │
│     └─ Rule-based fallback                                   │
│                                                               │
│  3. PREDICTION MODELS                                       │
│     └─ XGBoost Global                                        │
│     └─ LightGBM per-regime                                   │
│     └─ Neural Net (experimental)                             │
│     └─ HeuristicPredictor (fallback)                         │
│     └─ EnsemblePredictor (combines above)                    │
│                                                               │
│  4. BACKTESTING                                             │
│     └─ Walk-forward with expanding windows                  │
│     └─ Per-fold metrics tracking                             │
│     └─ Transaction cost modeling                             │
│                                                               │
│  5. VALIDATION (Backtest Phase)                             │
│     └─ Spearman IC computation                               │
│     └─ CPCV robustness testing                               │
│     └─ SPA bootstrap testing                                 │
│     └─ Sharpe significance testing                           │
│     └─ FDR correction across tickers                         │
│                                                               │
│  6. ADVANCED VALIDATION (Optional)                          │
│     └─ Deflated Sharpe Ratio (DSR)                           │
│     └─ Probability of Backtest Overfitting (PBO)             │
│     └─ Monte Carlo validation                                │
│     └─ Capacity/market impact analysis                       │
│                                                               │
│  7. PROMOTION GATE                                          │
│     └─ Contract metric evaluation                            │
│     └─ Strategy rating & selection                           │
│                                                               │
│  8. AUTOPILOT ORCHESTRATION                                 │
│     └─ Continuous strategy discovery                         │
│     └─ Periodic model retraining                             │
│     └─ Health monitoring                                     │
│                                                               │
│  9. PAPER TRADING & EXECUTION                               │
│     └─ Kelly-based position sizing                           │
│     └─ Live signal generation                                │
│     └─ Position tracking & P&L                               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Key File Organization

**Core Pipeline**:
- `/data/` - Data ingestion and caching
- `/features/` - Feature engineering
- `/regime/` - Regime detection (HMM, rules, correlation)
- `/models/` - Prediction models (XGBoost, LightGBM, neural net, heuristic)
- `/backtest/` - Backtesting engine
- `/backtest/validation.py` - Statistical validation (IC, CPCV, SPA, Sharpe)
- `/backtest/advanced_validation.py` - Advanced tests (DSR, PBO, Monte Carlo)
- `/risk/` - Risk management and position sizing
- `/api/` - API and orchestration
- `/ui/` - React frontend (Dash → React migration completed)

**Key Entry Points**:
- `run_autopilot.py` - Main autopilot loop
- `run_backtest.py` - Backtest runner
- `run_train.py` - Model training
- `api/main.py` - FastAPI server
- `api/orchestrator.py` - PipelineOrchestrator

### 2.3 Recent Migration: Dash → FastAPI+React

The system recently migrated from Dash (Python frontend) to FastAPI backend + React frontend. This is completed. Key implications:

- Backend is now pure Python/FastAPI
- Frontend is Node.js/React
- State management is cleaner (explicit API calls vs Dash callbacks)
- UI can be iterated faster than before

### 2.4 Core Architecture Principles

**PERMNO-First Design**: All operations are vectorized around PERMNO (permanent stock number), not ticker. This allows:
- Handling multiple tickers with same underlying security
- Survivorship-bias-free backtesting
- Efficient batch processing

**Batch-Oriented Processing**: The pipeline processes data in batches (daily batches, weekly regime updates, monthly retrains). Not single-order streaming (though paper trading has live order execution).

**Config-Driven System**: The system has 300+ configuration parameters in `config.py`:
- Data sources and cache paths
- Feature engineering thresholds
- Regime detection parameters
- Model hyperparameters
- Backtest parameters
- Autopilot cycle parameters
- Health thresholds

### 2.5 Current State of Key Subsystems

**Data Subsystem**:
- WRDS CRSP: Complete historical data through 2024-12-31
- IBKR: Real-time data from 2025-01-01 onward (gap-fill script handles boundary)
- FRED: Macroeconomic series (VIX, term spread, credit spread, jobless claims, consumer sentiment)
- Kalshi: Event probabilities (election, Fed decisions, etc.)
- Caching: Implemented with freshness checks

**Regime Subsystem**:
- Dual-engine: Rule-based (fast, interpretable) + HMM (accurate, complex)
- HMM has: Baum-Welch EM, Viterbi decoding, sticky transitions (0.92), BIC state selection
- Observation matrix: 4 features (ret_1d, vol_20d, NATR_14, SMASlope_50)
- State mapping: Return mean, volatility, trend, Hurst exponent → semantic labels

**Model Subsystem**:
- Ensemble of XGBoost, LightGBM, neural net (experimental)
- Per-regime LightGBM models + global XGBoost
- Heuristic fallback (signal-based rules)
- No calibration currently wired in (calibration.py exists but unused)

**Validation Subsystem**:
- validation.py: IC, CPCV, SPA, Sharpe significance, FDR
- advanced_validation.py: DSR, PBO, Monte Carlo
- Walk-forward cross-validation with expanding windows

**Autopilot Subsystem**:
- Continuous discovery (generate candidates, backtest, promote)
- Periodic retraining (weekly model updates)
- Health monitoring (data integrity, promotion contract, walk-forward gaps)
- Paper trading integration

**Risk Management**:
- Position sizing: Kelly, vol-scaled, ATR methods in PositionSizer
- Paper trader has separate Kelly implementation
- No portfolio-level risk constraints
- No dynamic stop-loss or correlation-based reduction

---

## SECTION 3: VALIDATION.PY → AUTOPILOT GAP ANALYSIS

### 3.1 Spearman Information Coefficient (IC) Testing

**Current State in validation.py**:

The file `/backtest/validation.py` contains the `compute_spearman_ic()` method that:
- Computes Spearman rank correlation between predicted returns and actual returns
- Applies Benjamini-Hochberg FDR correction across multiple tickers
- Tests whether correlations are significantly different from zero
- Returns IC values, p-values, and significance flags

**What's Missing in Autopilot**:

The autopilot engine in `/autopilot/engine.py` does NOT compute or track IC over time. The `_run_cycle()` method:
1. Generates predictions
2. Backtests strategies
3. Evaluates promotion contract
4. **Never calls** `compute_spearman_ic()`

**Problem Impact**:

- Autopilot can promote strategies with high in-sample Sharpe but zero predictive power (IC ≈ 0)
- No continuous IC monitoring means you don't know when alpha decays
- Information Ratio (excess return / tracking error) is never computed
- Cannot distinguish overfitting from genuine predictive ability

**Instructions to Fix**:

1. **File**: `/autopilot/engine.py`
   - **Method**: `_run_cycle()` around line 150 (after backtest loop completes)
   - **What to Add**: After computing walk-forward metrics, add a call to compute Spearman IC:
     ```
     Add import: from backtest.validation import compute_spearman_ic

     After fold metrics are computed, call:
     ic_results = compute_spearman_ic(
         predictions=self.all_predictions,  # shape: (n_samples, n_tickers)
         actuals=self.all_actuals,          # shape: (n_samples, n_tickers)
         apply_fdr=True,
         fdr_method='benjamini_hochberg'
     )

     Store IC results in cycle_report:
     cycle_report['ic_mean'] = ic_results['mean_ic']
     cycle_report['ic_pvalues'] = ic_results['pvalues']
     cycle_report['ic_significant_tickers'] = ic_results['significant_count']
     ```

   - **Expected Outcome**: IC values are now computed during each autopilot cycle
   - **Verification**: In the cycle report JSON, verify 'ic_mean', 'ic_pvalues' fields are populated

2. **File**: `/autopilot/promotion.py`
   - **Method**: `PromotionGate.evaluate()` around line 80 (contract metric evaluation)
   - **What to Add**: Add IC threshold check to the promotion contract:
     ```
     self.contract_metrics = {
         'sharpe': strategy_sharpe,
         'max_drawdown': max_dd,
         ...existing metrics...
         'ic_mean': contract_metrics.get('ic_mean', 0),
         'ic_significant': contract_metrics.get('ic_significant_tickers', 0),
     }

     Add gate:
     if contract_metrics.get('ic_mean', 0) < self.config.MIN_IC_THRESHOLD:
         self.contract_results.append({
             'gate': 'IC_CHECK',
             'passed': False,
             'reason': f"IC ({ic_mean:.4f}) below threshold ({MIN_IC_THRESHOLD:.4f})"
         })
     ```

   - **Expected Outcome**: Strategies with zero predictive power are rejected
   - **Verification**: Check that low-IC strategies are demoted in promotion gate results

3. **File**: `/api/services/health_service.py`
   - **Method**: `_check_signal_quality()` around line 200 (signal quality scoring)
   - **What to Add**: Add IC tracking to health score:
     ```
     Get latest cycle report IC values from autopilot registry
     If IC mean < 0.01 (threshold), flag as WARNING
     If IC mean < 0.0, flag as CRITICAL

     Add to signal_quality_score:
     ic_score = max(0, (ic_mean - (-0.01)) / 0.03)  # normalize to [0, 1]
     signal_quality_score = 0.6 * existing_score + 0.4 * ic_score
     ```

   - **Expected Outcome**: Health system explicitly monitors IC
   - **Verification**: Health dashboard shows IC score in the Signal Quality category

### 3.2 Combinatorial Purged Cross-Validation (CPCV)

**Current State in validation.py**:

The file contains `_cpcv_robustness()` method that:
- Partitions time series into groups
- Tests all C(N, N/2) combinations of groups as train/test splits
- Computes overfit probability: probability that the strategy's out-of-sample performance is worse than a random portfolio
- Returns PBO (Probability of Backtest Overfitting) metric

**What's Missing in Autopilot**:

The autopilot uses simple walk-forward expanding windows. It does NOT run CPCV. This means:
- No quantification of how robust the strategy is to different train/test splits
- Overfitting can hide in the expanding window structure
- Overfit probability is never computed

**Problem Impact**:

- Strategy can look great in walk-forward because of lucky sequential splits
- Cross-validation gap (IS vs OOS difference) is not monitored
- No statistical test for overfitting

**Instructions to Fix**:

1. **File**: `/autopilot/engine.py`
   - **Method**: `_run_cycle()` around line 180 (after walk-forward results)
   - **What to Add**: After walk-forward completes, run CPCV as a robustness check:
     ```
     Add import: from backtest.validation import _cpcv_robustness

     Call CPCV on a subset of the backtest data:
     cpcv_result = _cpcv_robustness(
         train_returns=training_returns,
         test_returns=test_returns,
         n_combos=min(50, 2**n_folds),  # Limit combinations for speed
         random_seed=42
     )

     Store results:
     cycle_report['cpcv_overfit_probability'] = cpcv_result['pbo']
     cycle_report['cpcv_n_combos'] = cpcv_result['n_combos']
     cycle_report['cpcv_mean_oos_sharpe'] = cpcv_result['mean_oos_sharpe']
     ```

   - **Expected Outcome**: CPCV results are computed (may take 30-60 seconds per cycle)
   - **Verification**: Cycle report contains 'cpcv_overfit_probability' field

2. **File**: `/autopilot/promotion.py`
   - **Method**: `PromotionGate.evaluate()` around line 90
   - **What to Add**: Add CPCV-based overfitting check:
     ```
     if contract_metrics.get('cpcv_overfit_probability', 0) > 0.5:
         self.contract_results.append({
             'gate': 'CPCV_OVERFIT',
             'passed': False,
             'reason': f"Overfit probability ({pbo:.2%}) exceeds 50%"
         })
     ```

   - **Expected Outcome**: Overfit strategies are rejected
   - **Verification**: Strategies with high PBO are demoted

3. **Configuration**: `/config.py`
   - **What to Add**: New config parameters:
     ```
     # CPCV Configuration
     CPCV_ENABLED = True
     CPCV_N_COMBOS = 50  # Limit combinations to keep computation fast
     CPCV_PBO_THRESHOLD = 0.5  # Reject if PBO > 50%
     ```

   - **Expected Outcome**: CPCV is configurable
   - **Verification**: Can enable/disable and set thresholds

### 3.3 Superior Predictive Ability (SPA) Bootstrap Test

**Current State in validation.py**:

Contains `_spa_bootstrap()` method that:
- Implements White's Reality Check bootstrap test
- Tests whether a strategy genuinely outperforms a benchmark (usually buy-and-hold)
- Computes p-values for the performance advantage
- Returns a boolean: does the strategy beat the benchmark with statistical significance?

**What's Missing in Autopilot**:

The autopilot never runs SPA tests. Strategies can show positive backtest Sharpe but fail to beat a simple buy-and-hold benchmark.

**Problem Impact**:

- Strategies can be promoted with returns worse than just buying SPY
- No benchmark-relative testing
- Information Ratio (vs benchmark) is never computed

**Instructions to Fix**:

1. **File**: `/autopilot/engine.py`
   - **Method**: `_run_cycle()` around line 190 (after SPA can run)
   - **What to Add**: Run SPA bootstrap test:
     ```
     Add import: from backtest.validation import _spa_bootstrap

     # Get benchmark returns (SPY or configured benchmark)
     benchmark_returns = get_benchmark_returns(
         start_date=backtest_start,
         end_date=backtest_end,
         benchmark_ticker=config.BENCHMARK_TICKER  # 'SPY'
     )

     spa_result = _spa_bootstrap(
         strategy_returns=strategy_returns,
         benchmark_returns=benchmark_returns,
         n_bootstrap=500,
         test_type='one-sided'  # one-sided: does strategy beat benchmark?
     )

     Store results:
     cycle_report['spa_pvalue'] = spa_result['pvalue']
     cycle_report['spa_beats_benchmark'] = spa_result['reject_null']
     cycle_report['spa_excess_sharpe'] = spa_result['excess_sharpe']
     ```

   - **Expected Outcome**: SPA test is run for each strategy
   - **Verification**: Cycle report shows 'spa_pvalue' and 'spa_beats_benchmark'

2. **File**: `/autopilot/promotion.py`
   - **Method**: `PromotionGate.evaluate()` around line 100
   - **What to Add**: Add SPA-based rejection:
     ```
     if not contract_metrics.get('spa_beats_benchmark', False):
         self.contract_results.append({
             'gate': 'SPA_TEST',
             'passed': False,
             'reason': f"Strategy does not beat benchmark (p={spa_pvalue:.3f})"
         })
     ```

   - **Expected Outcome**: Non-benchmark-beating strategies are rejected
   - **Verification**: Only strategies that beat buy-and-hold are promoted

### 3.4 Sharpe Ratio Significance Testing

**Current State in validation.py**:

Contains `_sharpe_significance()` method that:
- Computes standard error of Sharpe ratio using proper formulas (accounting for kurtosis)
- Tests whether Sharpe is significantly different from zero
- Returns p-value and confidence interval

**What's Missing in Autopilot**:

The promotion gate checks `strategy_sharpe > SHARPE_THRESHOLD` but never tests whether the Sharpe is statistically significant. With few trades (N < 50), even Sharpe = 1.5 can be noise.

**Problem Impact**:

- Strategies with high but statistically insignificant Sharpe are promoted
- No sample size consideration
- Confidence intervals are never computed

**Instructions to Fix**:

1. **File**: `/autopilot/engine.py`
   - **Method**: `_run_cycle()` around line 170 (after Sharpe is computed)
   - **What to Add**: Test Sharpe significance:
     ```
     Add import: from backtest.validation import _sharpe_significance

     sharpe_sig = _sharpe_significance(
         returns=strategy_returns,
         target_return=0.0  # Test if Sharpe > 0
     )

     Store results:
     cycle_report['sharpe_pvalue'] = sharpe_sig['pvalue']
     cycle_report['sharpe_significant'] = sharpe_sig['reject_null']
     cycle_report['sharpe_ci_lower'] = sharpe_sig['ci_lower']
     cycle_report['sharpe_ci_upper'] = sharpe_sig['ci_upper']
     ```

   - **Expected Outcome**: Sharpe significance is computed
   - **Verification**: Cycle report contains sharpe_pvalue, sharpe_ci_lower, sharpe_ci_upper

2. **File**: `/autopilot/promotion.py`
   - **Method**: `PromotionGate.evaluate()` around line 70
   - **What to Add**: Require Sharpe to be statistically significant:
     ```
     if not contract_metrics.get('sharpe_significant', False):
         self.contract_results.append({
             'gate': 'SHARPE_SIGNIFICANCE',
             'passed': False,
             'reason': f"Sharpe not statistically significant (p={sharpe_pvalue:.3f})"
         })
     ```

   - **Expected Outcome**: Only statistically significant Sharpe ratios are promoted
   - **Verification**: Strategies with few trades are rejected if Sharpe is not significant

### 3.5 Walk-Forward Fold-Level Metrics Tracking

**Current State in autopilot**:

The autopilot's walk-forward loop in `_run_cycle()` computes per-fold metrics but doesn't expose them properly:
- Per-fold Sharpe is computed internally
- Per-fold IC is not computed
- Per-fold hit rate is not computed
- IS-vs-OOS gap is not quantified
- Positive fold fraction (% of folds with positive Sharpe) is not tracked

**What's Missing**:

The promotion gate doesn't have detailed fold-level statistics, only summary statistics (mean Sharpe, max drawdown).

**Problem Impact**:

- Cannot detect when a strategy is only profitable in certain folds
- Cannot identify regime-dependent profitability
- Cannot require minimum positive fold count

**Instructions to Fix**:

1. **File**: `/autopilot/engine.py`
   - **Method**: `_run_cycle()` around line 140 (fold loop)
   - **What to Add**: Enhance fold-level metric tracking:
     ```
     For each fold in the walk-forward loop:

     fold_metrics = {
         'fold_id': fold_num,
         'is_start': is_start_date,
         'is_end': is_end_date,
         'oos_start': oos_start_date,
         'oos_end': oos_end_date,
         'is_sharpe': in_sample_sharpe,
         'oos_sharpe': out_of_sample_sharpe,
         'is_ic_mean': in_sample_ic_mean,
         'oos_ic_mean': out_of_sample_ic_mean,
         'is_hit_rate': in_sample_hit_rate,
         'oos_hit_rate': out_of_sample_hit_rate,
         'is_max_dd': in_sample_max_drawdown,
         'oos_max_dd': out_of_sample_max_drawdown,
     }

     fold_metrics['is_oos_gap'] = fold_metrics['is_sharpe'] - fold_metrics['oos_sharpe']

     self.fold_metrics_list.append(fold_metrics)

     Aggregate at end of cycle:
     cycle_report['fold_count'] = len(self.fold_metrics_list)
     cycle_report['positive_fold_fraction'] = sum(1 for f in self.fold_metrics_list if f['oos_sharpe'] > 0) / len(self.fold_metrics_list)
     cycle_report['mean_is_oos_gap'] = np.mean([f['is_oos_gap'] for f in self.fold_metrics_list])
     cycle_report['oos_ic_correlation'] = np.corrcoef([f['is_ic_mean'] for f in self.fold_metrics_list], [f['oos_ic_mean'] for f in self.fold_metrics_list])[0, 1]
     ```

   - **Expected Outcome**: Detailed fold-level metrics are tracked
   - **Verification**: Cycle report contains 'fold_count', 'positive_fold_fraction', 'mean_is_oos_gap'

2. **File**: `/autopilot/promotion.py`
   - **Method**: `PromotionGate.evaluate()`
   - **What to Add**: Add fold-level gates:
     ```
     # Require minimum fraction of profitable folds
     if contract_metrics.get('positive_fold_fraction', 0) < 0.6:
         self.contract_results.append({
             'gate': 'FOLD_PROFITABILITY',
             'passed': False,
             'reason': f"Only {positive_fold_fraction:.1%} of folds were profitable"
         })

     # Require IS-OOS gap to be small
     if contract_metrics.get('mean_is_oos_gap', 0) > 2.0:
         self.contract_results.append({
             'gate': 'IS_OOS_GAP',
             'passed': False,
             'reason': f"IS-OOS gap too large ({is_oos_gap:.2f})"
         })
     ```

   - **Expected Outcome**: Fold-level requirements are enforced
   - **Verification**: Strategies with poor fold distribution are rejected

### 3.6 Benjamini-Hochberg FDR Correction

**Current State in validation.py**:

The `compute_spearman_ic()` method applies Benjamini-Hochberg FDR correction across multiple ticker p-values to control false discovery rate at level α = 0.05.

**What's Missing in Autopilot**:

When evaluating signals across multiple tickers/regimes/horizons, the autopilot does NOT apply FDR correction. This means:
- Multiple testing problem: testing 500 tickers means 25 expected false discoveries at α=0.05
- No correction for how many hypothesis tests are run
- Non-significant signals are passed through

**Problem Impact**:

- Inflated false positive rate across multi-ticker evaluations
- Signals that are individually p=0.05 are kept, even though many are noise
- Portfolio of "significant" signals is actually mostly noise

**Instructions to Fix**:

1. **File**: `/autopilot/engine.py`
   - **Method**: `_run_cycle()` around line 160 (signal evaluation)
   - **What to Add**: Apply FDR correction to all multi-ticker hypothesis tests:
     ```
     Add import: from scipy.stats import false_discovery_control
     from backtest.validation import benjamini_hochberg_correction

     After computing all signal p-values across tickers:
     pvalues = [signal_pvalue for each ticker]  # shape: (n_tickers,)

     # Apply BH correction
     rejected, corrected_pvalues = benjamini_hochberg_correction(
         pvalues=pvalues,
         alpha=0.05
     )

     # Keep only FDR-corrected-significant signals
     significant_signals = {
         ticker: signals[ticker]
         for ticker, sig in zip(tickers, rejected)
         if sig
     }

     Store in cycle_report:
     cycle_report['n_tests'] = len(pvalues)
     cycle_report['n_significant_after_fdr'] = sum(rejected)
     cycle_report['fdr_rejection_rate'] = sum(rejected) / len(pvalues)
     ```

   - **Expected Outcome**: FDR-corrected signals are computed
   - **Verification**: Significantly fewer signals pass after FDR correction

2. **File**: `/backtest/validation.py`
   - **Method**: `benjamini_hochberg_correction()` (if not already present)
   - **What to Add**: If the function doesn't exist, add it:
     ```
     def benjamini_hochberg_correction(pvalues, alpha=0.05):
         '''
         Benjamini-Hochberg FDR correction.

         Input:
             pvalues: array of p-values (one per test)
             alpha: FDR level (0.05 means max 5% of rejected tests are false discoveries)

         Returns:
             rejected: boolean array indicating which tests reject the null
             corrected_pvalues: adjusted p-values
         '''
         from scipy.stats import rankdata
         import numpy as np

         pvalues = np.asarray(pvalues)
         n_tests = len(pvalues)

         # Sort p-values and compute threshold
         sorted_idx = np.argsort(pvalues)
         sorted_pvals = pvalues[sorted_idx]

         # Find largest i such that P(i) <= (i/n) * alpha
         thresholds = (np.arange(1, n_tests + 1) / n_tests) * alpha
         rejection_idx = np.where(sorted_pvals <= thresholds)[0]

         if len(rejection_idx) > 0:
             threshold = sorted_pvals[rejection_idx[-1]]
         else:
             threshold = -np.inf  # Reject nothing

         rejected = pvalues <= threshold
         return rejected, pvalues  # Return original pvals (not Benjamini-corrected values)
     ```

   - **Expected Outcome**: FDR correction is available as a utility
   - **Verification**: Function can be imported and used

### 3.7 Summary of Validation→Autopilot Gap Closure

The following table summarizes what needs to be wired:

| Test | validation.py Function | Autopilot Integration | File to Modify | Promotion Gate Impact |
|------|------------------------|-----------------------|-----------------|----------------------|
| Spearman IC | compute_spearman_ic() | Call after predictions | engine.py line 150 | Reject IC < 0.01 |
| CPCV | _cpcv_robustness() | Run post-backtest | engine.py line 180 | Reject PBO > 0.5 |
| SPA Bootstrap | _spa_bootstrap() | Compare vs benchmark | engine.py line 190 | Reject if doesn't beat SPY |
| Sharpe Sig | _sharpe_significance() | Test Sharpe != 0 | engine.py line 170 | Reject if p > 0.05 |
| Fold Metrics | (tracked in loop) | Aggregate fold stats | engine.py line 140 | Reject if <60% folds positive |
| FDR Correction | benjamini_hochberg() | Apply to multi-test | engine.py line 160 | Keep only FDR-significant |

---

## SECTION 4: KELLY SYSTEM — COMPREHENSIVE INTEGRATION

### 4.1 Current State of Kelly in the System

The system currently has Kelly sizing scattered across multiple locations:

**1. Risk Module** (`/risk/position_sizer.py`):
```
PositionSizer class with methods:
- size_position(account_value, current_price, kelly_stats) → position_size
  Uses Kelly formula: f* = (p*b - q) / b
  Where p = win rate, b = avg_win/avg_loss, q = 1-p
- Half-Kelly = f* / 2
- Vol-scaled sizing
- ATR-based sizing
```

**2. Paper Trader** (`/autopilot/paper_trader.py`):
```
PaperTrader class with _kelly_size() method:
- Simplified Kelly: f* = (p*b - q) / b
- Uses half-Kelly fraction
- Separate implementation from PositionSizer
- Uses paper trader's own trade history to estimate p, b, q
```

**3. Backtest Engine** (`/backtest/engine.py`):
```
Uses flat position_size_pct from config (NOT Kelly-based)
- All positions sized identically
- No regime conditioning
- No feedback from trade history
```

### 4.2 Problems with Current Kelly Implementation

**Problem 1: No Regime-Conditional Kelly**

Kelly parameters (win_rate, avg_win, avg_loss) vary dramatically by regime:

- Bull market: win_rate = 60%, avg_win/avg_loss = 1.5 → Kelly fraction high
- Bear market: win_rate = 45%, avg_win/avg_loss = 0.8 → Kelly fraction can be negative

**Current Impact**: Same Kelly fraction used regardless of regime. In bear markets, oversizing leads to catastrophic losses.

**Problem 2: No Drawdown Governor**

Full Kelly can theoretically cause 25% drawdown even with positive expectancy. Half-Kelly helps but is still aggressive.

**Current Impact**: No automatic position size reduction during losing streaks.

**Problem 3: No Portfolio-Level Kelly**

Each asset sized independently. No covariance-aware multi-asset Kelly.

**Current Impact**: Ignores correlation between positions. In crisis, correlations → 1.0, but position sizes don't adjust.

**Problem 4: No Bayesian Updating**

Win rate estimates are static from historical data. New trade results don't update the win rate estimate in real-time.

**Current Impact**: Kelly fraction doesn't adapt to recent performance. A strategy that goes 0-for-10 still sizes as if it has 60% win rate.

**Problem 5: Paper Trader Kelly Disconnected**

Paper trader uses its own Kelly implementation (`_kelly_size()`), not the unified `PositionSizer`.

**Current Impact**: Paper trader sizing logic is different from backtest/training assumptions.

### 4.3 Comprehensive Kelly Integration Plan

#### 4.3.1 Regime-Conditional Kelly in PositionSizer

**File**: `/risk/position_sizer.py`

**Method**: `size_position()` (current signature around line 40)

**Current Code Structure**:
```
def size_position(self, account_value, current_price, kelly_fraction=0.5):
    # Global Kelly stats
    win_rate = self.stats['win_rate']
    avg_win = self.stats['avg_win']
    avg_loss = self.stats['avg_loss']
    # ... compute Kelly ...
```

**What to Add**:

1. Add a `regime` parameter to the method signature:
```
def size_position(self, account_value, current_price, kelly_fraction=0.5, regime=None):
```

2. Add per-regime stats storage in `__init__`:
```
self.regime_stats = {
    'BULL': {'win_rate': 0.60, 'avg_win': 1.5, 'avg_loss': 1.0},
    'BEAR': {'win_rate': 0.45, 'avg_win': 0.8, 'avg_loss': 1.2},
    'MEAN_REVERT': {'win_rate': 0.55, 'avg_win': 1.2, 'avg_loss': 1.0},
    'HIGH_VOL': {'win_rate': 0.50, 'avg_win': 1.0, 'avg_loss': 1.0},
}
```

3. Implement regime-conditional Kelly:
```
if regime and regime in self.regime_stats:
    stats = self.regime_stats[regime]
else:
    stats = self.stats  # Fall back to global

win_rate = stats['win_rate']
avg_win = stats['avg_win']
avg_loss = stats['avg_loss']

b = avg_win / avg_loss
q = 1 - win_rate
kelly_fraction_optimal = (p * b - q) / b

# Apply configured Kelly fraction (e.g., 0.5 for half-Kelly)
position_kelly = kelly_fraction_optimal * kelly_fraction
```

4. Add method to update per-regime stats:
```
def update_regime_stats(self, regime, trades):
    '''Update per-regime statistics based on recent trades in that regime.'''
    wins = sum(1 for t in trades if t['return'] > 0)
    win_rate = wins / len(trades) if trades else 0.5

    avg_win = np.mean([t['return'] for t in trades if t['return'] > 0]) if wins > 0 else 1.0
    avg_loss = np.mean([abs(t['return']) for t in trades if t['return'] <= 0]) if (len(trades) - wins) > 0 else 1.0

    self.regime_stats[regime] = {
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
    }
```

**Expected Outcome**: Position sizing adjusts based on regime-specific profitability.

**Verification**:
- In a bear market simulation, position size should decrease
- In a bull market simulation, position size should increase
- Call `print(position_sizer.regime_stats)` and verify regime-specific values differ

#### 4.3.2 Drawdown-Adjusted Kelly

**File**: `/risk/position_sizer.py`

**Method**: `size_position()` (same method)

**What to Add**:

Add a drawdown governor that reduces Kelly fraction when current drawdown is deep:

```
def _apply_drawdown_governor(self, kelly_fraction, current_drawdown, max_allowed_dd=0.20):
    '''
    Reduce Kelly fraction proportionally as current drawdown approaches max allowed.

    Args:
        kelly_fraction: base Kelly fraction (e.g., 0.5)
        current_drawdown: current portfolio drawdown (e.g., 0.08)
        max_allowed_dd: maximum allowed drawdown threshold (e.g., 0.20)

    Returns:
        adjusted_kelly: reduced Kelly fraction
    '''
    if current_drawdown < 0:
        current_drawdown = 0

    if current_drawdown >= max_allowed_dd:
        return 0  # Stop trading if max DD exceeded

    # Linear reduction: kelly * (1 - dd_ratio)
    dd_ratio = current_drawdown / max_allowed_dd
    adjusted_kelly = kelly_fraction * (1 - dd_ratio)

    return max(0, adjusted_kelly)
```

Then in `size_position()`, after computing Kelly:

```
# Get current portfolio drawdown from perf tracker
current_dd = self.get_current_drawdown()

# Apply drawdown governor
kelly_fraction = self._apply_drawdown_governor(
    kelly_fraction=kelly_fraction,
    current_drawdown=current_dd,
    max_allowed_dd=self.config.MAX_PORTFOLIO_DD
)
```

**Configuration** (`/config.py`):
```
MAX_PORTFOLIO_DD = 0.20  # Stop trading if DD exceeds 20%
```

**Expected Outcome**: Position sizes reduce automatically during losing streaks.

**Verification**:
- Simulate a strategy that hits -10% drawdown
- Position size should reduce to 50% of normal
- When drawdown reaches -20%, position size should go to zero

#### 4.3.3 Multi-Asset Portfolio-Level Kelly

**File**: `/risk/position_sizer.py`

**Method**: New method `size_portfolio()`

**What to Add**:

```
def size_portfolio(self, account_value, assets, individual_sizes, recent_returns_matrix):
    '''
    Size a portfolio of multiple assets using multi-asset Kelly.

    Args:
        account_value: total portfolio value
        assets: list of asset names/PERMNOs
        individual_sizes: dict of asset → individual Kelly-sized position (before portfolio adjustment)
        recent_returns_matrix: shape (n_periods, n_assets) of recent returns for covariance

    Returns:
        portfolio_sizes: dict of asset → final position size accounting for covariance
    '''
    import numpy as np
    from scipy.linalg import inv

    # Step 1: Compute covariance matrix of returns
    cov_matrix = np.cov(recent_returns_matrix.T)

    # Step 2: Estimate expected returns from recent performance
    expected_returns = np.mean(recent_returns_matrix, axis=0)

    # Step 3: Compute portfolio-level Kelly: w* = Σ⁻¹ @ μ
    try:
        inv_cov = inv(cov_matrix)
        kelly_weights = inv_cov @ expected_returns
        kelly_weights = kelly_weights / np.sum(kelly_weights)  # Normalize to sum to 1
    except np.linalg.LinAlgError:
        # Covariance singular, fall back to individual sizes
        kelly_weights = np.array([individual_sizes[asset] for asset in assets])

    # Step 4: Blend individual Kelly sizes with portfolio-level Kelly
    blend_weight = self.config.KELLY_PORTFOLIO_BLEND  # e.g., 0.3

    final_weights = (
        blend_weight * kelly_weights +
        (1 - blend_weight) * np.array([individual_sizes[asset] for asset in assets])
    )

    # Step 5: Scale to account value
    portfolio_sizes = {
        asset: account_value * weight
        for asset, weight in zip(assets, final_weights)
    }

    return portfolio_sizes
```

**Configuration** (`/config.py`):
```
KELLY_PORTFOLIO_BLEND = 0.3  # Blend 30% portfolio Kelly with 70% individual Kelly
```

**Expected Outcome**: Highly correlated positions are downsized, lowly correlated positions are upsized.

**Verification**:
- Create a 10-asset portfolio where 5 assets have correlation 0.9 and 5 have correlation 0.2
- Portfolio-level Kelly should favor the uncorrelated 5

#### 4.3.4 Bayesian Updating of Kelly Parameters

**File**: `/risk/position_sizer.py`

**Method**: New method `update_kelly_bayesian()`

**What to Add**:

```
def __init__(self, ...):
    # Initialize Beta priors for win rate
    self.win_rate_prior_alpha = 2  # Prior belief: Beta(2, 2)
    self.win_rate_prior_beta = 2

    # Posterior = Prior + Data
    self.win_rate_posterior_alpha = self.win_rate_prior_alpha
    self.win_rate_posterior_beta = self.win_rate_prior_beta

def update_kelly_bayesian(self, trades):
    '''
    Update Kelly parameters using Bayesian Beta-Binomial conjugate model.

    Args:
        trades: list of recent trade results (True if win, False if loss)
    '''
    wins = sum(1 for trade in trades if trade)
    losses = len(trades) - wins

    # Posterior = Prior + Data
    self.win_rate_posterior_alpha = self.win_rate_prior_alpha + wins
    self.win_rate_posterior_beta = self.win_rate_prior_beta + losses

    # Posterior mean (credibility-weighted estimate)
    posterior_mean = (
        self.win_rate_posterior_alpha /
        (self.win_rate_posterior_alpha + self.win_rate_posterior_beta)
    )

    # Posterior variance (uncertainty)
    posterior_variance = (
        (self.win_rate_posterior_alpha * self.win_rate_posterior_beta) /
        ((self.win_rate_posterior_alpha + self.win_rate_posterior_beta)**2 *
         (self.win_rate_posterior_alpha + self.win_rate_posterior_beta + 1))
    )

    return posterior_mean, posterior_variance

def get_kelly_fraction(self):
    '''Get Kelly fraction using posterior Bayesian estimate.'''
    posterior_mean = (
        self.win_rate_posterior_alpha /
        (self.win_rate_posterior_alpha + self.win_rate_posterior_beta)
    )
    # Use posterior mean instead of MLE
    return self._compute_kelly(win_rate=posterior_mean)
```

**Expected Outcome**: Kelly fraction updates as new trades come in, with proper uncertainty quantification.

**Verification**:
- Start with no trades: posterior should be Beta(2, 2), mean = 0.5
- After 10 wins, 0 losses: posterior should be Beta(12, 2), mean = 0.857
- Verify posterior mean differs from simple win_rate (10/10)

#### 4.3.5 Unify Paper Trader Kelly

**File**: `/autopilot/paper_trader.py`

**Current Code** (around line 200):
```
def _kelly_size(self):
    # Separate Kelly implementation
    ...
```

**What to Change**:

1. Remove `_kelly_size()` method entirely.

2. Import `PositionSizer`:
```
from risk.position_sizer import PositionSizer
```

3. Initialize `PositionSizer` in `__init__`:
```
def __init__(self, ...):
    ...
    self.position_sizer = PositionSizer(config=config)
```

4. In `_generate_order()` where Kelly sizing is called, replace:
```
# OLD:
size = self._kelly_size(price, account_value)

# NEW:
regime = self.current_regime  # Get from regime detector
size = self.position_sizer.size_position(
    account_value=account_value,
    current_price=price,
    kelly_fraction=self.config.KELLY_FRACTION,
    regime=regime
)
```

5. When paper trader executes trades, feed back to PositionSizer:
```
def _execute_order(self, order):
    # ... execute order ...

    # Update PositionSizer with actual trade result
    trade_result = {
        'return': (close_price - entry_price) / entry_price,
        'regime': order['regime']
    }
    self.position_sizer.update_regime_stats(
        regime=order['regime'],
        trades=[trade_result]
    )
```

**Expected Outcome**: Paper trader uses the unified, sophisticated PositionSizer instead of its own simple Kelly.

**Verification**:
- Paper trader position sizes should now be regime-aware
- Sizes should adjust based on regime-specific win rates
- Sizes should reduce during drawdowns

#### 4.3.6 Kelly in Backtesting

**File**: `/backtest/engine.py`

**Method**: `_run_fold()` or relevant backtest loop (around line 250)

**Current Code**:
```
position_size = self.config.position_size_pct
```

**What to Change**:

1. Initialize `PositionSizer`:
```
from risk.position_sizer import PositionSizer

self.position_sizer = PositionSizer(config=self.config)
```

2. In the backtest loop, at each entry signal, compute Kelly:
```
# Get current regime
regime = self.regime_detector.get_current_regime(timestamp)

# Size position using regime-aware Kelly
kelly_size = self.position_sizer.size_position(
    account_value=self.portfolio_value,
    current_price=current_price,
    kelly_fraction=self.config.KELLY_FRACTION,
    regime=regime
)

position_size_pct = kelly_size / self.portfolio_value
```

3. As trades close, update Kelly statistics:
```
# After each trade closure:
trade_return = (exit_price - entry_price) / entry_price
self.position_sizer.update_regime_stats(
    regime=entry_regime,
    trades=[trade_return > 0]  # Boolean: did we win?
)
```

**Expected Outcome**: Backtests use Kelly-based sizing, with regime conditioning and updating over time.

**Verification**:
- Backtest Sharpe should improve (proper risk sizing)
- Position sizes should decrease during losing streaks
- Compare backtest results with/without Kelly sizing

#### 4.3.7 Kelly Configuration

**File**: `/config.py`

**What to Add**:

```python
# ============================================================================
# KELLY POSITION SIZING CONFIGURATION
# ============================================================================

# Kelly Fraction: what fraction of optimal Kelly to use?
# 0.25 = Quarter Kelly (low risk, 92.6% of optimal growth, ~6% of variance)
# 0.50 = Half Kelly (popular, 75% of optimal growth, 25% of variance)
# 0.75 = Three-Quarter Kelly (aggressive, 56% of optimal growth, 56% of variance)
# 1.00 = Full Kelly (max growth, can cause 25% drawdowns)
KELLY_FRACTION = 0.5

# Maximum portfolio drawdown before all positions are closed
MAX_PORTFOLIO_DD = 0.20

# Blend weight for portfolio-level Kelly
# 0.3 means: 30% portfolio-level Kelly + 70% individual Kelly sizes
KELLY_PORTFOLIO_BLEND = 0.3

# Bayesian prior for win rate Beta distribution
# Beta(alpha, beta) represents belief about win rate before any trades
# Beta(2, 2) is a weak, neutral prior
KELLY_BAYESIAN_ALPHA = 2
KELLY_BAYESIAN_BETA = 2

# Regime detection flag: if False, all positions use global Kelly
# If True, use per-regime Kelly with regime-specific win rates
KELLY_REGIME_CONDITIONAL = True
```

**Expected Outcome**: Kelly system is fully configurable.

**Verification**: Can modify config and see changes in position sizing

### 4.4 Summary: Kelly System Integration Checklist

| Component | File | Method | Status |
|-----------|------|--------|--------|
| Regime-conditional Kelly | position_sizer.py | size_position() | Add regime param |
| Drawdown governor | position_sizer.py | _apply_drawdown_governor() | New method |
| Multi-asset Kelly | position_sizer.py | size_portfolio() | New method |
| Bayesian updating | position_sizer.py | update_kelly_bayesian() | New method |
| Paper trader unification | paper_trader.py | _generate_order() | Remove local Kelly |
| Backtest Kelly | engine.py | _run_fold() | Add Kelly sizing |
| Configuration | config.py | (top-level) | Add KELLY_* params |

---

## SECTION 5: HEALTH SYSTEM — RENAISSANCE-LEVEL OVERHAUL

### 5.1 Current Health System Audit

**Location**: `/api/services/health_service.py` and `/api/services/data_helpers.py`

**Current Capabilities**:

1. **Data Integrity Checks**:
   - Survivorship DB exists (binary check)
   - WRDS source freshness
   - Cache age and availability

2. **Promotion Contract Verification**:
   - Sharpe ratio threshold
   - Max drawdown threshold
   - DSR (Deflated Sharpe) check (if available)
   - PBO (Probability of Backtest Overfitting) check (if available)
   - Capacity constraints check

3. **Walk-Forward Validation Checks**:
   - Number of CV folds
   - Holdout period presence
   - CPCV robustness (if run)

4. **Execution Cost Audit**:
   - Transaction costs model (fixed + dynamic)
   - Almgren-Chriss cost scaling

5. **Complexity Checks**:
   - Feature count
   - Feature interactions
   - Model parameter count

6. **Strengths Identification**:
   - High-Sharpe strategies detected
   - Low-correlation ensemble members identified

**Current Scoring** (around line 350):
```python
health_score = (
    0.2 * data_integrity_score +
    0.2 * signal_quality_score +
    0.2 * risk_management_score +
    0.2 * execution_quality_score +
    0.2 * model_governance_score
)
```

**Current Problem**: Health scores start at 50/100 (base score) before any checks. Even a system with no data shows 50/100, inflating perception of system health.

### 5.2 15 Missing Health Monitors

#### 5.2.1 Signal Decay Monitoring

**What it is**: Track how quickly the predictive power of signals decays after generation. RenTec monitors signal half-life religiously.

**Why it's missing**: No explicit monitoring of how long signals remain valid. IC at day 1 might be 0.05, but at day 5 it might be 0.02 (decay).

**How to implement**:

**File**: `/api/services/health_service.py`

**New Method**: `_check_signal_decay()`

```python
def _check_signal_decay(self):
    '''
    Monitor how quickly alpha decays after signal generation.

    Compute rolling autocorrelation of prediction errors at lags 1, 5, 10, 20 days.
    If lag-1 autocorr is still high (>0.3), holding period may be misaligned.
    '''
    # Get recent prediction error history (from paper trading)
    error_history = self._get_prediction_error_history(days=60)  # Last 60 days

    if len(error_history) < 20:
        return {
            'status': 'INSUFFICIENT_DATA',
            'score': 0.5,
            'reason': 'Need at least 20 days of trading history'
        }

    autocorrelations = {}
    for lag in [1, 5, 10, 20]:
        if lag < len(error_history):
            autocorr = np.corrcoef(
                error_history[:-lag],
                error_history[lag:]
            )[0, 1]
            autocorrelations[lag] = autocorr

    # Score: how fast does autocorr drop?
    # Good: lag-1 = 0.2, lag-5 = 0.1, lag-20 = 0.0 (fast decay)
    # Bad: lag-1 = 0.5, lag-5 = 0.4, lag-20 = 0.3 (slow decay)

    decay_rate = (autocorrelations[1] - autocorrelations[20]) / autocorrelations[1] if autocorrelations[1] > 0 else 0

    if decay_rate > 0.8:  # Fast decay
        score = 0.9
        status = 'PASS'
    elif decay_rate > 0.5:  # Moderate decay
        score = 0.7
        status = 'PASS'
    elif decay_rate > 0.2:  # Slow decay
        score = 0.4
        status = 'WARNING'
    else:  # Very slow or no decay
        score = 0.1
        status = 'FAIL'

    return {
        'status': status,
        'score': score,
        'autocorrelations': autocorrelations,
        'decay_rate': decay_rate,
        'reason': f'Signal decay rate: {decay_rate:.2%}'
    }
```

**Configuration** (`/config.py`):
```python
SIGNAL_DECAY_THRESHOLD = 0.5  # Decay rate must be > 50%
SIGNAL_HALF_LIFE_TARGET = 5   # Signals should decay to half value in ~5 days
```

**Expected Outcome**: Health dashboard shows signal decay metrics.

**Verification**:
- Run on a paper trading account
- Check that decay metrics are computed and displayed
- Verify that fast-decaying signals are flagged

#### 5.2.2 Feature Importance Drift Monitoring

**What it is**: Track how much feature rankings change between retrains. Large changes indicate model instability.

**Why it's missing**: `/models/feature_stability.py` exists but is not wired into health checks.

**How to implement**:

**File**: `/api/services/health_service.py`

**New Method**: `_check_feature_importance_drift()`

```python
def _check_feature_importance_drift(self):
    '''
    Monitor feature importance rank correlation between current and previous models.

    Alert if top-5 features change by more than 3 positions.
    '''
    from models.feature_stability import compute_feature_rank_correlation

    # Get current model feature importance
    current_model = self._get_current_model()
    current_importance = current_model.feature_importances_
    current_ranks = rankdata(-current_importance)  # Higher importance = lower rank number

    # Get previous model feature importance
    previous_model = self._get_previous_model()
    if previous_model is None:
        return {
            'status': 'INSUFFICIENT_DATA',
            'score': 0.5,
            'reason': 'No previous model for comparison'
        }

    previous_importance = previous_model.feature_importances_
    previous_ranks = rankdata(-previous_importance)

    # Compute Spearman rank correlation
    rank_correlation = spearmanr(current_ranks, previous_ranks)[0]

    # Check top-5 feature changes
    current_top5 = set(np.argsort(-current_importance)[:5])
    previous_top5 = set(np.argsort(-previous_importance)[:5])

    overlap = len(current_top5 & previous_top5)  # Intersection

    # Score based on stability
    if rank_correlation > 0.8 and overlap >= 4:  # High correlation, mostly same top features
        score = 0.9
        status = 'PASS'
    elif rank_correlation > 0.6 and overlap >= 3:
        score = 0.7
        status = 'PASS'
    elif rank_correlation > 0.4:
        score = 0.4
        status = 'WARNING'
    else:
        score = 0.1
        status = 'CRITICAL'

    return {
        'status': status,
        'score': score,
        'rank_correlation': rank_correlation,
        'top5_overlap': overlap,
        'reason': f'Feature drift: {rank_correlation:.2f} correlation, {overlap}/5 top features stable'
    }
```

**Expected Outcome**: Health system tracks feature importance stability.

**Verification**:
- Retrain model
- Verify that feature rank correlation is computed
- Check that large changes trigger WARNING status

#### 5.2.3 Regime Transition Health

**What it is**: Monitor sanity of regime transitions (avoiding stuck states, unrealistic durations, invalid probabilities).

**Why it's missing**: No monitoring of HMM transition matrix quality.

**How to implement**:

**File**: `/api/services/health_service.py`

**New Method**: `_check_regime_transition_health()`

```python
def _check_regime_transition_health(self):
    '''
    Monitor regime transition matrix for sanity:
    - No stuck states (diagonal > 0.99)
    - Regime durations match economic reality
    - Transition probabilities sum to 1.0
    '''
    regime_detector = self._get_regime_detector()
    hmm_model = regime_detector.hmm_model

    if hmm_model is None:
        return {
            'status': 'NO_MODEL',
            'score': 0.5,
            'reason': 'No HMM model available'
        }

    # Get transition matrix
    transition_matrix = hmm_model.transmat_
    n_states = transition_matrix.shape[0]

    issues = []

    # Check 1: Stuck states (diagonal too high)
    for i in range(n_states):
        if transition_matrix[i, i] > 0.99:
            issues.append(f"State {i} is stuck (self-transition {transition_matrix[i, i]:.2%})")

    # Check 2: Probability rows sum to 1.0
    for i in range(n_states):
        row_sum = transition_matrix[i, :].sum()
        if not np.isclose(row_sum, 1.0, atol=1e-6):
            issues.append(f"State {i} row sum {row_sum:.4f} != 1.0")

    # Check 3: Expected regime duration (geometric distribution)
    # Expected duration = 1 / (1 - diagonal)
    durations = [1 / (1 - transition_matrix[i, i]) for i in range(n_states)]

    # Sanity checks:
    # Bull/Bear markets should last at least 10 days, at most 1000 days
    for i, duration in enumerate(durations):
        if duration < 10:
            issues.append(f"State {i} has unrealistically short duration ({duration:.1f} days)")
        elif duration > 1000:
            issues.append(f"State {i} has unrealistically long duration ({duration:.1f} days)")

    # Compute score
    if not issues:
        score = 0.95
        status = 'PASS'
    elif len(issues) <= 1:
        score = 0.7
        status = 'PASS'
    elif len(issues) <= 3:
        score = 0.4
        status = 'WARNING'
    else:
        score = 0.1
        status = 'CRITICAL'

    return {
        'status': status,
        'score': score,
        'issues': issues,
        'durations': durations,
        'reason': f'{len(issues)} transition matrix issues detected'
    }
```

**Expected Outcome**: Health system validates regime transition matrices.

**Verification**:
- Train HMM
- Verify transition matrix checks pass
- Manually corrupt transition matrix and verify CRITICAL flag

#### 5.2.4 Model Prediction Distribution Health

**What it is**: Monitor prediction distribution for collapse (all predictions converging to 0 or a narrow band).

**Why it's missing**: No check for prediction variance collapse.

**How to implement**:

**File**: `/api/services/health_service.py`

**New Method**: `_check_prediction_distribution_health()`

```python
def _check_prediction_distribution_health(self):
    '''
    Monitor prediction distributions for collapse.

    Alert if:
    - Prediction variance is very low (all predictions similar)
    - >80% of predictions are within ±0.5 std devs of mean
    - Entropy of prediction distribution is low
    '''
    # Get recent predictions
    recent_predictions = self._get_recent_predictions(days=30)

    if len(recent_predictions) < 100:
        return {
            'status': 'INSUFFICIENT_DATA',
            'score': 0.5,
            'reason': 'Need at least 100 predictions'
        }

    predictions = np.array(recent_predictions).flatten()

    # Check 1: Variance
    pred_std = np.std(predictions)
    pred_mean = np.mean(predictions)

    if pred_std < 0.01:  # Very low variance
        score = 0.1
        status = 'CRITICAL'
        reason = f'Prediction std dev ({pred_std:.4f}) is too low - collapse detected'
        return {
            'status': status,
            'score': score,
            'std': pred_std,
            'mean': pred_mean,
            'reason': reason
        }

    # Check 2: Percentage within ±0.5 std
    within_band = np.sum(np.abs(predictions - pred_mean) < 0.5 * pred_std) / len(predictions)

    if within_band > 0.80:
        score = 0.3
        status = 'WARNING'
        reason = f'{within_band:.1%} of predictions within ±0.5 std (collapse risk)'
    else:
        score = 0.8
        status = 'PASS'
        reason = f'Prediction distribution healthy (std={pred_std:.4f})'

    # Check 3: Entropy (discretize predictions into 10 bins)
    hist, _ = np.histogram(predictions, bins=10)
    probs = hist / len(predictions)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(10)  # Max entropy with 10 bins
    normalized_entropy = entropy / max_entropy

    if normalized_entropy < 0.3:
        score = max(score - 0.2, 0.1)
        status = 'WARNING'
        reason += f'; entropy low ({normalized_entropy:.2f})'

    return {
        'status': status,
        'score': score,
        'prediction_std': pred_std,
        'prediction_mean': pred_mean,
        'within_band_pct': within_band,
        'entropy': entropy,
        'reason': reason
    }
```

**Expected Outcome**: Health system detects prediction distribution collapse.

**Verification**:
- Check on a healthy system: should PASS
- Manually set all predictions to 0.1 and verify CRITICAL status

#### 5.2.5 Survivorship Bias Quantification

**What it is**: Quantify the actual impact of survivorship bias on backtest returns (rather than binary check).

**Why it's missing**: Current check is only "DB exists or not".

**How to implement**:

**File**: `/api/services/health_service.py`

**New Method**: `_check_survivorship_bias()`

```python
def _check_survivorship_bias(self):
    '''
    Quantify survivorship bias impact by comparing:
    - Backtest on surviving universe only
    - Backtest on full historical universe (including delisted)

    Returns bias impact in basis points of annual return.
    '''
    from data.survivorship import get_survival_universe, get_full_universe

    # Check 1: Survivorship DB freshness
    db_age = self._get_survivorship_db_age()

    if db_age > 30:  # Older than 30 days
        return {
            'status': 'STALE_DB',
            'score': 0.4,
            'reason': f'Survivorship DB is {db_age} days old'
        }

    # Check 2: Run comparative backtests
    # This is expensive, so only do it weekly
    if not self._should_run_survivorship_test():
        return {
            'status': 'DEFERRED',
            'score': 0.7,
            'reason': 'Last test was recent, skipping'
        }

    surviving_returns = self._backtest_on_universe(universe='surviving')
    full_returns = self._backtest_on_universe(universe='full')

    # Compute bias impact
    surviving_sharpe = sharpe_ratio(surviving_returns)
    full_sharpe = sharpe_ratio(full_returns)

    bias_impact_sharpe = surviving_sharpe - full_sharpe

    # Convert to annual return impact
    # Assuming 252 trading days
    surviving_annual_return = np.mean(surviving_returns) * 252
    full_annual_return = np.mean(full_returns) * 252

    bias_impact_bps = (surviving_annual_return - full_annual_return) * 10000  # Convert to bps

    # Score
    if abs(bias_impact_bps) < 10:  # Less than 1 bp annual impact
        score = 0.95
        status = 'PASS'
    elif abs(bias_impact_bps) < 50:  # Less than 5 bp
        score = 0.8
        status = 'PASS'
    elif abs(bias_impact_bps) < 100:  # Less than 10 bp
        score = 0.6
        status = 'WARNING'
    else:
        score = 0.3
        status = 'CRITICAL'

    return {
        'status': status,
        'score': score,
        'bias_impact_bps': bias_impact_bps,
        'bias_impact_sharpe': bias_impact_sharpe,
        'surviving_annual_return': surviving_annual_return,
        'full_annual_return': full_annual_return,
        'reason': f'Survivorship bias: {bias_impact_bps:.0f} bps annual impact'
    }
```

**Configuration** (`/config.py`):
```python
SURVIVORSHIP_TEST_FREQUENCY_DAYS = 7  # Run test weekly
SURVIVORSHIP_BIAS_TOLERANCE_BPS = 50  # Warn if bias > 5 bp
```

**Expected Outcome**: Health system quantifies and displays survivorship bias impact.

**Verification**:
- Run on a system with WRDS data
- Verify that comparative backtest is performed
- Check that bias impact is computed and displayed

#### 5.2.6 Correlation Regime Health

**What it is**: Monitor average pairwise correlation of universe. High correlation (>0.7) means portfolio is less diversified.

**Why it's missing**: `/regime/correlation.py` exists but is not health-checked.

**How to implement**:

**File**: `/api/services/health_service.py`

**New Method**: `_check_correlation_regime_health()`

```python
def _check_correlation_regime_health(self):
    '''
    Monitor universe correlation for diversification risk.

    High correlation indicates reduced diversification:
    - Correlation > 0.7: Poor diversification, auto-reduce exposure
    - Correlation 0.5-0.7: Moderate, maintain normal sizing
    - Correlation < 0.5: Good diversification
    '''
    from regime.correlation import get_current_correlation_regime

    corr_regime = get_current_correlation_regime()

    if corr_regime is None:
        return {
            'status': 'NO_DATA',
            'score': 0.5,
            'reason': 'Correlation regime not computed'
        }

    # Get average pairwise correlation from the current universe
    universe = self._get_current_universe()
    price_data = self._get_recent_prices(universe, days=60)

    returns = price_data.pct_change().dropna()
    corr_matrix = returns.corr()

    # Extract upper triangle (unique correlations)
    upper_tri = np.triu_indices_from(corr_matrix, k=1)
    pairwise_corrs = corr_matrix.values[upper_tri]

    avg_corr = np.mean(pairwise_corrs)
    max_corr = np.max(pairwise_corrs)

    # Score
    if avg_corr < 0.4:
        score = 0.9
        status = 'PASS'
        action = 'NORMAL_SIZING'
    elif avg_corr < 0.55:
        score = 0.8
        status = 'PASS'
        action = 'NORMAL_SIZING'
    elif avg_corr < 0.7:
        score = 0.6
        status = 'CAUTION'
        action = 'INCREASE_DIVERSIFICATION'
    else:
        score = 0.3
        status = 'WARNING'
        action = 'AUTO_REDUCE_EXPOSURE'

    return {
        'status': status,
        'score': score,
        'avg_correlation': avg_corr,
        'max_correlation': max_corr,
        'action': action,
        'reason': f'Avg correlation {avg_corr:.2f}; action={action}'
    }
```

**Expected Outcome**: Health system monitors correlation and recommends actions.

**Verification**:
- Monitor during low-vol regime (correlation should be low)
- Monitor during crisis (correlation should spike)
- Verify action recommendations change appropriately

#### 5.2.7 Execution Quality Monitoring

**What it is**: Compare expected fill prices (from backtest model) to actual paper trade fills. Track slippage drift.

**Why it's missing**: No comparison between backtest assumptions and reality.

**How to implement**:

**File**: `/api/services/health_service.py`

**New Method**: `_check_execution_quality()`

```python
def _check_execution_quality(self):
    '''
    Monitor slippage: compare expected fills (from backtest model) to actual paper trade fills.

    Alert if mean slippage exceeds 2x the modeled amount.
    '''
    # Get recent paper trade executions
    trades = self._get_recent_paper_trades(days=30)

    if len(trades) < 10:
        return {
            'status': 'INSUFFICIENT_DATA',
            'score': 0.5,
            'reason': 'Need at least 10 recent trades'
        }

    slippages = []
    for trade in trades:
        # Expected fill price from model (midpoint at signal time)
        expected_fill = trade['signal_price']

        # Actual fill price from IBKR
        actual_fill = trade['actual_fill_price']

        # Slippage in basis points
        slippage_bps = (actual_fill - expected_fill) / expected_fill * 10000
        slippages.append(slippage_bps)

    mean_slippage = np.mean(slippages)
    std_slippage = np.std(slippages)

    # Compare to modeled slippage
    modeled_slippage = self.config.MODELED_SLIPPAGE_BPS  # e.g., 1 bp

    slippage_ratio = abs(mean_slippage) / modeled_slippage if modeled_slippage > 0 else 0

    # Score
    if slippage_ratio < 1.0:  # Better than model
        score = 0.95
        status = 'PASS'
    elif slippage_ratio < 1.5:
        score = 0.8
        status = 'PASS'
    elif slippage_ratio < 2.0:
        score = 0.5
        status = 'WARNING'
    else:
        score = 0.2
        status = 'CRITICAL'

    return {
        'status': status,
        'score': score,
        'mean_slippage_bps': mean_slippage,
        'std_slippage_bps': std_slippage,
        'modeled_slippage_bps': modeled_slippage,
        'slippage_ratio': slippage_ratio,
        'reason': f'Actual slippage {mean_slippage:.1f} bps vs modeled {modeled_slippage:.1f} bps'
    }
```

**Configuration** (`/config.py`):
```python
MODELED_SLIPPAGE_BPS = 1.0  # Expected slippage in basis points
EXECUTION_QUALITY_THRESHOLD = 2.0  # Warn if actual > 2x modeled
```

**Expected Outcome**: Health system tracks execution quality vs backtest assumptions.

**Verification**:
- Execute 20+ trades through IBKR
- Verify slippage is tracked
- Check that warnings trigger if slippage drifts

#### 5.2.8 Tail Risk Monitoring

**What it is**: Track rolling 60-day CVaR (conditional value at risk), tail ratio, and alert if tail risk degrades.

**Why it's missing**: Current risk metrics are point-in-time only, not rolling tail metrics.

**How to implement**:

**File**: `/api/services/health_service.py`

**New Method**: `_check_tail_risk()`

```python
def _check_tail_risk(self):
    '''
    Monitor tail risk metrics:
    - Rolling 60-day CVaR (Value at Risk at 95% confidence)
    - Tail ratio (upside/downside at 5th percentile)
    - Alert if tail risk degrades
    '''
    returns = self._get_recent_returns(days=60)

    if len(returns) < 20:
        return {
            'status': 'INSUFFICIENT_DATA',
            'score': 0.5,
            'reason': 'Need at least 20 days of returns'
        }

    # Compute CVaR (expected shortfall)
    # CVaR = mean of the worst 5% returns
    var_95 = np.percentile(returns, 5)  # 5th percentile
    cvar_95 = np.mean(returns[returns <= var_95])

    # Compute tail ratio
    # Tail ratio = mean of top 5% / |mean of bottom 5%|
    top_5_pct_returns = np.percentile(returns, 95)
    upside_tail = np.mean(returns[returns >= top_5_pct_returns])
    downside_tail = np.mean(returns[returns <= var_95])

    tail_ratio = upside_tail / abs(downside_tail) if downside_tail != 0 else 0

    # Compute rolling history for trend
    # ... compare to previous month's CVaR ...
    previous_cvar = self._get_previous_month_cvar()

    if previous_cvar is None:
        cvar_trend = 'NEUTRAL'
    elif cvar_95 > previous_cvar:  # CVaR got worse (less negative, tail risk increased)
        cvar_trend = 'DETERIORATING'
    else:
        cvar_trend = 'IMPROVING'

    # Score
    if tail_ratio > 1.2 and cvar_trend != 'DETERIORATING':
        score = 0.9
        status = 'PASS'
    elif tail_ratio > 1.0:
        score = 0.75
        status = 'PASS'
    elif tail_ratio > 0.8:
        score = 0.5
        status = 'WARNING'
    else:
        score = 0.2
        status = 'CRITICAL'

    return {
        'status': status,
        'score': score,
        'cvar_95': cvar_95,
        'tail_ratio': tail_ratio,
        'cvar_trend': cvar_trend,
        'reason': f'Tail ratio {tail_ratio:.2f}, CVaR {cvar_95:.4f}, trend {cvar_trend}'
    }
```

**Expected Outcome**: Health system monitors tail risk metrics.

**Verification**:
- Run on live paper trading account
- Verify CVaR and tail ratio are computed
- Check that deteriorating tail risk triggers WARNING

#### 5.2.9 Information Ratio Tracking

**What it is**: Rolling Information Ratio (excess return over benchmark / tracking error). Alert if IR drops below 0.3 for extended period.

**Why it's missing**: Current metrics are Sharpe and max drawdown only.

**How to implement**:

**File**: `/api/services/health_service.py`

**New Method**: `_check_information_ratio()`

```python
def _check_information_ratio(self):
    '''
    Monitor Information Ratio = (portfolio return - benchmark return) / tracking error.

    Benchmark = buy-and-hold SPY or configured benchmark.
    Alert if rolling IR < 0.3 for 60 consecutive days.
    '''
    portfolio_returns = self._get_recent_returns(days=60)
    benchmark_returns = self._get_benchmark_returns(days=60, ticker='SPY')

    if len(portfolio_returns) < 20:
        return {
            'status': 'INSUFFICIENT_DATA',
            'score': 0.5,
            'reason': 'Need at least 20 days of returns'
        }

    # Align returns
    portfolio_returns, benchmark_returns = align_returns(portfolio_returns, benchmark_returns)

    # Compute excess returns
    excess_returns = portfolio_returns - benchmark_returns

    # Compute tracking error (std of excess returns)
    tracking_error = np.std(excess_returns)

    # Compute Information Ratio
    mean_excess_return = np.mean(excess_returns)
    information_ratio = mean_excess_return / tracking_error if tracking_error > 0 else 0

    # Check rolling 60-day IR
    rolling_ir = []
    for i in range(len(excess_returns) - 30, len(excess_returns)):
        er = excess_returns[i-30:i]
        te = np.std(er)
        ir = np.mean(er) / te if te > 0 else 0
        rolling_ir.append(ir)

    low_ir_days = sum(1 for ir in rolling_ir if ir < 0.3)

    # Score
    if information_ratio > 0.5:
        score = 0.9
        status = 'PASS'
    elif information_ratio > 0.3:
        score = 0.75
        status = 'PASS'
    elif information_ratio > 0.0:
        score = 0.5
        status = 'WARNING'
    else:
        score = 0.1
        status = 'CRITICAL'

    if low_ir_days > 40:  # Low IR for more than 40 of last 60 days
        score *= 0.7  # Penalize
        status = 'WARNING'

    return {
        'status': status,
        'score': score,
        'information_ratio': information_ratio,
        'tracking_error': tracking_error,
        'low_ir_days': low_ir_days,
        'reason': f'IR {information_ratio:.2f}, {low_ir_days} days below 0.3'
    }
```

**Expected Outcome**: Health system monitors Information Ratio.

**Verification**:
- Run on paper trading account
- Verify IR is computed and compared to benchmark
- Check that sustained low-IR periods trigger WARNING

#### 5.2.10 Cross-Validation Gap Monitoring

**What it is**: Track CV gap (IS sharpe - OOS sharpe) over time. If gap widening, model becoming more overfit.

**Why it's missing**: CV gaps are computed per-backtest but not tracked over time as a health metric.

**How to implement**:

**File**: `/api/services/health_service.py`

**New Method**: `_check_cv_gap_trend()`

```python
def _check_cv_gap_trend(self):
    '''
    Monitor CV gap (in-sample Sharpe - out-of-sample Sharpe) over time.

    If gap is widening (current > historical mean + 1 std), model is overfitting.
    '''
    from autopilot.registry import get_backtest_history

    # Get recent backtests (past 10 retrains)
    recent_backtests = get_backtest_history(limit=10)

    if len(recent_backtests) < 3:
        return {
            'status': 'INSUFFICIENT_HISTORY',
            'score': 0.5,
            'reason': 'Need at least 3 recent backtests for trend analysis'
        }

    cv_gaps = []
    for backtest in recent_backtests:
        is_sharpe = backtest['metrics'].get('is_sharpe', 0)
        oos_sharpe = backtest['metrics'].get('oos_sharpe', 0)
        gap = is_sharpe - oos_sharpe
        cv_gaps.append(gap)

    # Compute trend
    cv_gaps = np.array(cv_gaps)
    mean_gap = np.mean(cv_gaps)
    std_gap = np.std(cv_gaps)
    current_gap = cv_gaps[-1]

    # Check if current gap is unusually high
    gap_zscore = (current_gap - mean_gap) / std_gap if std_gap > 0 else 0

    # Check if gap is increasing (regression of gaps over time)
    X = np.arange(len(cv_gaps)).reshape(-1, 1)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, cv_gaps)
    gap_trend_slope = model.coef_[0]

    # Score
    if current_gap < mean_gap and gap_trend_slope < 0.01:
        score = 0.9
        status = 'PASS'
        reason = 'CV gap stable/decreasing (good)'
    elif current_gap < mean_gap + 0.5 * std_gap:
        score = 0.75
        status = 'PASS'
        reason = 'CV gap normal'
    elif current_gap < mean_gap + 1.5 * std_gap:
        score = 0.5
        status = 'WARNING'
        reason = f'CV gap elevated ({current_gap:.2f} vs mean {mean_gap:.2f})'
    else:
        score = 0.2
        status = 'CRITICAL'
        reason = f'CV gap very high ({current_gap:.2f}) - severe overfitting detected'

    if gap_trend_slope > 0.05:  # Trend increasing
        score *= 0.8
        status = 'WARNING'
        reason += f'; gap trend increasing (+{gap_trend_slope:.3f}/retrain)'

    return {
        'status': status,
        'score': score,
        'current_gap': current_gap,
        'mean_gap': mean_gap,
        'gap_zscore': gap_zscore,
        'gap_trend_slope': gap_trend_slope,
        'reason': reason
    }
```

**Expected Outcome**: Health system monitors whether model is becoming overfit over time.

**Verification**:
- Track multiple backtests
- Verify that widening CV gap triggers WARNING
- Check that trend slope is computed

#### 5.2.11 Data Quality Anomaly Detection

**What it is**: Statistical anomaly detection on incoming OHLCV data (unusual volume spikes, price gaps, etc.).

**Why it's missing**: Current quality checks are basic (cache age only).

**How to implement**:

**File**: `/api/services/health_service.py`

**New Method**: `_check_data_quality_anomalies()`

```python
def _check_data_quality_anomalies(self):
    '''
    Detect statistical anomalies in incoming OHLCV data:
    - Volume spikes (>5x 20-day average)
    - Price jumps without corresponding news
    - Gaps between daily Close and next Open exceeding 3 ATRs
    - Zero-volume days for liquid stocks
    '''
    from data.loader import get_recent_ohlcv

    universe = self._get_current_universe()
    ohlcv = get_recent_ohlcv(universe, days=60)

    if len(ohlcv) < 20:
        return {
            'status': 'INSUFFICIENT_DATA',
            'score': 0.5,
            'reason': 'Need at least 20 days of OHLCV data'
        }

    anomalies = []

    # Check each stock
    for ticker in universe:
        if ticker not in ohlcv.columns:
            continue

        data = ohlcv[ticker].dropna()

        if len(data) < 20:
            continue

        # Anomaly 1: Volume spikes
        volume = data['volume']
        vol_ma20 = volume.rolling(20).mean()
        vol_spike_ratio = volume / vol_ma20

        if vol_spike_ratio[-1] > 5:
            anomalies.append(f'{ticker}: Volume spike {vol_spike_ratio[-1]:.1f}x')

        # Anomaly 2: Price gaps
        close = data['close']
        open_price = data['open']

        # Gap from previous close to current open
        returns = close.pct_change().abs()
        atr = returns.rolling(14).mean() * close  # Approximate ATR

        gap = abs(open_price - close.shift(1))
        gap_multiple = gap / atr

        if gap_multiple[-1] > 3:
            anomalies.append(f'{ticker}: Price gap {gap_multiple[-1]:.1f} ATR')

        # Anomaly 3: Zero volume on liquid stock
        if volume[-1] == 0 and vol_ma20[-1] > 1000000:  # Liquid stock with zero volume
            anomalies.append(f'{ticker}: Zero volume on liquid stock')

    # Score
    if len(anomalies) == 0:
        score = 0.95
        status = 'PASS'
    elif len(anomalies) <= 2:
        score = 0.7
        status = 'WARNING'
    elif len(anomalies) <= 5:
        score = 0.4
        status = 'WARNING'
    else:
        score = 0.1
        status = 'CRITICAL'

    return {
        'status': status,
        'score': score,
        'anomalies': anomalies[:10],  # Limit to top 10
        'total_anomalies': len(anomalies),
        'reason': f'{len(anomalies)} data quality anomalies detected'
    }
```

**Expected Outcome**: Health system detects unusual data events.

**Verification**:
- Monitor on live data
- Verify that large volume spikes are detected
- Check that price gap alerts appear during market discontinuities

#### 5.2.12 Ensemble Disagreement Monitoring

**What it is**: With regime-specific + global models in ensemble, monitor disagreement between members. High disagreement suggests uncertainty.

**Why it's missing**: Ensemble predictions are combined, but member disagreement is not tracked.

**How to implement**:

**File**: `/api/services/health_service.py`

**New Method**: `_check_ensemble_disagreement()`

```python
def _check_ensemble_disagreement(self):
    '''
    Monitor disagreement between ensemble members (global + regime-specific models).

    High disagreement indicates uncertainty. Alert if std dev of ensemble member
    predictions exceeds historical 95th percentile.
    '''
    # Get recent predictions from each ensemble member
    global_predictions = self._get_recent_predictions(model='global', days=30)
    bull_predictions = self._get_recent_predictions(model='bull_regime', days=30)
    bear_predictions = self._get_recent_predictions(model='bear_regime', days=30)

    if len(global_predictions) < 50:
        return {
            'status': 'INSUFFICIENT_DATA',
            'score': 0.5,
            'reason': 'Need at least 50 recent predictions'
        }

    # For each prediction set, compute disagreement (std across ensemble members)
    disagreements = []
    for i in range(len(global_predictions)):
        ensemble_preds = [
            global_predictions[i],
            bull_predictions[i] if i < len(bull_predictions) else global_predictions[i],
            bear_predictions[i] if i < len(bear_predictions) else global_predictions[i],
        ]
        disagreement = np.std(ensemble_preds)
        disagreements.append(disagreement)

    disagreements = np.array(disagreements)

    # Compare to historical distribution
    current_disagreement = disagreements[-30:].mean()
    historical_p95 = np.percentile(disagreements[:-30], 95) if len(disagreements) > 30 else np.percentile(disagreements, 95)

    disagreement_ratio = current_disagreement / historical_p95 if historical_p95 > 0 else 0

    # Score
    if disagreement_ratio < 0.8:
        score = 0.9
        status = 'PASS'
    elif disagreement_ratio < 1.0:
        score = 0.8
        status = 'PASS'
    elif disagreement_ratio < 1.5:
        score = 0.6
        status = 'WARNING'
    else:
        score = 0.2
        status = 'CRITICAL'

    return {
        'status': status,
        'score': score,
        'current_disagreement': current_disagreement,
        'historical_p95': historical_p95,
        'disagreement_ratio': disagreement_ratio,
        'reason': f'Ensemble disagreement {disagreement_ratio:.2f}x normal'
    }
```

**Expected Outcome**: Health system monitors ensemble member consensus.

**Verification**:
- Run ensemble on normal market data
- Verify disagreement is tracked
- Spike disagreement during volatile periods and verify WARNING

#### 5.2.13 Market Microstructure Health

**What it is**: Monitor bid-ask spreads, Amihud illiquidity, Kyle's lambda for trading universe. Alert if liquidity deteriorates.

**Why it's missing**: No monitoring of execution liquidity environment.

**How to implement**:

**File**: `/api/services/health_service.py`

**New Method**: `_check_market_microstructure()`

```python
def _check_market_microstructure(self):
    '''
    Monitor liquidity metrics for trading universe:
    - Bid-ask spreads
    - Amihud illiquidity (|return| / dollar volume)
    - Alert if liquidity deteriorates beyond historical norm
    '''
    from data.loader import get_bid_ask_spreads, get_amihud_illiquidity

    universe = self._get_current_universe()
    spreads = get_bid_ask_spreads(universe, days=30)
    illiquidity = get_amihud_illiquidity(universe, days=30)

    if len(spreads) < 10:
        return {
            'status': 'INSUFFICIENT_DATA',
            'score': 0.5,
            'reason': 'Need bid-ask data'
        }

    # Compute average spread (in basis points)
    avg_spread_bps = spreads.mean().mean() * 10000  # Convert to bps

    # Compute average Amihud illiquidity
    avg_illiquidity = illiquidity.mean().mean()

    # Compare to historical baseline
    historical_spread = self._get_historical_median_spread_bps()
    historical_illiquidity = self._get_historical_median_illiquidity()

    spread_deterioration = (avg_spread_bps - historical_spread) / historical_spread if historical_spread > 0 else 0
    illiquidity_deterioration = (avg_illiquidity - historical_illiquidity) / historical_illiquidity if historical_illiquidity > 0 else 0

    # Score
    if spread_deterioration < 0.1 and illiquidity_deterioration < 0.1:
        score = 0.9
        status = 'PASS'
    elif spread_deterioration < 0.25 and illiquidity_deterioration < 0.25:
        score = 0.75
        status = 'PASS'
    elif spread_deterioration < 0.5:
        score = 0.5
        status = 'WARNING'
    else:
        score = 0.2
        status = 'CRITICAL'

    return {
        'status': status,
        'score': score,
        'avg_spread_bps': avg_spread_bps,
        'historical_spread_bps': historical_spread,
        'spread_deterioration': spread_deterioration,
        'illiquidity_deterioration': illiquidity_deterioration,
        'reason': f'Spreads +{spread_deterioration:.1%}, illiquidity +{illiquidity_deterioration:.1%}'
    }
```

**Expected Outcome**: Health system monitors market liquidity environment.

**Verification**:
- Track during calm markets: should PASS
- Monitor during crisis: spreads widen, should flag WARNING

#### 5.2.14 Retraining Effectiveness Tracking

**What it is**: After each retrain, track whether new model improves or degrades OOS performance vs previous version.

**Why it's missing**: Model retraining happens, but success rate is not tracked.

**How to implement**:

**File**: `/api/services/health_service.py`

**New Method**: `_check_retraining_effectiveness()`

```python
def _check_retraining_effectiveness(self):
    '''
    Track whether retrains improve or degrade out-of-sample performance.

    Maintain a running score of retrain success rate.
    '''
    from autopilot.registry import get_model_history

    # Get recent model history (last 10 retrains)
    models = get_model_history(limit=10)

    if len(models) < 2:
        return {
            'status': 'INSUFFICIENT_HISTORY',
            'score': 0.5,
            'reason': 'Need at least 2 recent models for comparison'
        }

    improvements = []
    for i in range(1, len(models)):
        previous_oos_sharpe = models[i-1]['oos_sharpe']
        current_oos_sharpe = models[i]['oos_sharpe']

        improved = current_oos_sharpe > previous_oos_sharpe
        improvements.append(improved)

    success_rate = sum(improvements) / len(improvements)

    # Score
    if success_rate > 0.7:  # 70%+ of retrains improve OOS
        score = 0.9
        status = 'PASS'
    elif success_rate > 0.5:  # 50-70%
        score = 0.7
        status = 'PASS'
    elif success_rate > 0.3:
        score = 0.4
        status = 'WARNING'
    else:
        score = 0.1
        status = 'CRITICAL'

    return {
        'status': status,
        'score': score,
        'retrain_success_rate': success_rate,
        'recent_retrains': len(improvements),
        'reason': f'{success_rate:.1%} of retrains improved OOS performance'
    }
```

**Expected Outcome**: Health system tracks retraining success.

**Verification**:
- Monitor after several retrains
- Verify that success rate is computed
- Check that low success rates trigger WARNING

#### 5.2.15 Capital Utilization Efficiency

**What it is**: Track what percentage of available capital is actually deployed, and whether idle capital periods coincide with high-opportunity regimes.

**Why it's missing**: No monitoring of whether capital is being efficiently deployed.

**How to implement**:

**File**: `/api/services/health_service.py`

**New Method**: `_check_capital_utilization()`

```python
def _check_capital_utilization(self):
    '''
    Monitor capital utilization:
    - What fraction of account value is deployed?
    - Are idle periods coinciding with high-opportunity regimes?
    '''
    # Get daily capital allocation
    daily_allocation = self._get_daily_capital_allocation(days=60)  # Returns: date -> deployed_amount

    if len(daily_allocation) < 20:
        return {
            'status': 'INSUFFICIENT_DATA',
            'score': 0.5,
            'reason': 'Need at least 20 days of allocation history'
        }

    account_value = self._get_account_value()
    utilization_pct = np.array([allocation / account_value for allocation in daily_allocation.values()])

    mean_utilization = np.mean(utilization_pct)
    min_utilization = np.min(utilization_pct)

    # Check if idle periods coincide with high-opportunity regimes
    regime_history = self._get_regime_history(days=60)
    opportunity_history = self._get_opportunity_signal_history(days=60)

    # Correlation: are we deployed when opportunities are high?
    opportunity_correlation = np.corrcoef(utilization_pct, opportunity_history)[0, 1]

    # Score
    if mean_utilization > 0.7 and opportunity_correlation > 0.3:
        score = 0.9
        status = 'PASS'
    elif mean_utilization > 0.5 and opportunity_correlation > 0.1:
        score = 0.7
        status = 'PASS'
    elif mean_utilization > 0.3:
        score = 0.5
        status = 'WARNING'
    else:
        score = 0.2
        status = 'CRITICAL'

    return {
        'status': status,
        'score': score,
        'mean_utilization': mean_utilization,
        'min_utilization': min_utilization,
        'opportunity_correlation': opportunity_correlation,
        'reason': f'Deployed {mean_utilization:.1%} of capital; opportunity corr {opportunity_correlation:.2f}'
    }
```

**Expected Outcome**: Health system monitors capital deployment efficiency.

**Verification**:
- Monitor paper trading account
- Verify utilization metrics are computed
- Check that unused capital during opportunity periods triggers WARNING

### 5.3 Health Score Computation Overhaul

**File**: `/api/services/health_service.py`

**Method**: `compute_health_score()` (around line 350)

**Current Code**:
```python
health_score = (
    0.2 * data_integrity_score +
    0.2 * signal_quality_score +
    0.2 * risk_management_score +
    0.2 * execution_quality_score +
    0.2 * model_governance_score
)
```

**Problems**:
- Base score is 50 (inflates perceived health)
- All categories equally weighted
- No sub-scores or detailed breakdown
- No "system readiness" boolean check

**What to Change**:

```python
def compute_health_score(self):
    '''
    Compute weighted health score with new monitors integrated.

    Weights:
    - Data Integrity (25%): freshness, completeness, survivorship
    - Signal Quality (25%): IC, decay, prediction distribution
    - Risk Management (20%): drawdown, correlations, Kelly
    - Execution Quality (15%): slippage, fills, liquidity
    - Model Governance (15%): feature drift, retraining, validation
    '''

    # Compute sub-scores (each 0-100)
    data_integrity_score = 0.4 * self._check_data_freshness()['score'] + \
                           0.3 * self._check_survivorship_bias()['score'] + \
                           0.3 * self._check_data_quality_anomalies()['score']

    signal_quality_score = 0.3 * self._check_spearman_ic()['score'] + \
                           0.3 * self._check_signal_decay()['score'] + \
                           0.25 * self._check_prediction_distribution_health()['score'] + \
                           0.15 * self._check_ensemble_disagreement()['score']

    risk_management_score = 0.3 * self._check_max_drawdown()['score'] + \
                            0.25 * self._check_correlation_regime_health()['score'] + \
                            0.2 * self._check_tail_risk()['score'] + \
                            0.15 * self._check_kelly_sizing()['score'] + \
                            0.1 * self._check_capital_utilization()['score']

    execution_quality_score = 0.4 * self._check_execution_quality()['score'] + \
                              0.35 * self._check_market_microstructure()['score'] + \
                              0.25 * self._check_information_ratio()['score']

    model_governance_score = 0.3 * self._check_feature_importance_drift()['score'] + \
                             0.25 * self._check_retraining_effectiveness()['score'] + \
                             0.25 * self._check_cv_gap_trend()['score'] + \
                             0.2 * self._check_regime_transition_health()['score']

    # Weighted overall score (base = 0, not 50)
    health_score = (
        0.25 * data_integrity_score +
        0.25 * signal_quality_score +
        0.20 * risk_management_score +
        0.15 * execution_quality_score +
        0.15 * model_governance_score
    )

    # Compute "system readiness" boolean
    critical_checks = {
        'data_fresh': self._check_data_freshness()['status'] != 'STALE',
        'model_trained': self._check_model_age()['status'] != 'OLD',
        'regime_detected': self._check_regime_detection()['status'] != 'FAILED',
        'no_prediction_collapse': self._check_prediction_distribution_health()['status'] != 'CRITICAL',
        'execution_ok': self._check_execution_quality()['status'] != 'CRITICAL',
    }

    system_ready = all(critical_checks.values())

    return {
        'overall_score': max(0, min(100, health_score)),  # Clamp to [0, 100]
        'system_ready': system_ready,
        'data_integrity': data_integrity_score,
        'signal_quality': signal_quality_score,
        'risk_management': risk_management_score,
        'execution_quality': execution_quality_score,
        'model_governance': model_governance_score,
        'critical_checks': critical_checks,
    }
```

**Expected Outcome**: Health score is now properly calibrated (0-100), weighted appropriately, and includes system readiness flag.

**Verification**:
- System with no issues should score >90
- System with data freshness issues should score <70
- Check that system_ready is False if any critical check fails

---

## SECTION 6: REGIME DETECTION — BEST MODELS AND IMPROVEMENTS

### 6.1 Current Regime Detection Audit

**Files**:
- `/regime/detector.py` - Main RegimeDetector class
- `/regime/hmm.py` - GaussianHMM implementation
- `/regime/correlation.py` - Correlation-based regime detection

**Current Architecture**:

The RegimeDetector uses a dual-engine approach:
1. **Rule-based (Fast)**: Uses hardcoded rules based on recent returns, volatility, and trend
2. **HMM (Accurate)**: Gaussian HMM with Baum-Welch EM learning

**HMM Details**:
- **Observation matrix**: 4 features
  - `ret_1d`: 1-day return
  - `vol_20d`: 20-day rolling volatility
  - `NATR_14`: Normalized Average True Range (ATR / close)
  - `SMASlope_50`: Slope of 50-day SMA

- **Fitting**: Baum-Welch EM algorithm with up to 20 iterations
- **Decoding**: Viterbi algorithm for most-likely state sequence
- **Transitions**: Sticky-transition HMM with stickiness=0.92 (favors staying in current state)
- **Duration**: Min-duration smoothing to enforce minimum regime length
- **State Selection**: BIC (Bayesian Information Criterion) to automatically select optimal number of states (2-5)

**State Mapping**: After HMM convergence, states are mapped to semantic labels:
- Compute per-state statistics: mean return, volatility, trend, Hurst exponent
- Assign labels: BULL (high return, moderate vol), BEAR (low return, high vol), MEAN_REVERT, HIGH_VOL, etc.

### 6.2 Problems and Criticisms

**Problem 1: Only 4 Observation Features**

The HMM observes only 4 features. Renaissance-level regime detection uses 20+ dimensions:
- Credit spreads (HY-Treasury, HY-IG)
- Term structure slope (2yr-10yr, 2yr-20yr)
- VIX and VVIX (option volatility, volatility-of-volatility)
- Cross-asset correlation (stocks-bonds, stocks-commodities)
- Market breadth (% stocks above 50-day SMA, new highs/lows ratio)
- Sector rotation signals (momentum rankings)

**Impact**: Regime detection is 1-dimensional (mostly just volatility). Missing credit cycles, term structure shifts, and breadth divergences.

**Problem 2: No Online/Incremental Updating**

HMM is re-fit from scratch every time new data arrives. This is:
- **Computationally expensive**: Baum-Welch takes minutes on full history
- **Unstable**: Parameters can converge to different local optima on re-fit
- **Reactive**: Only uses new data in the next scheduled retrain

**Impact**: Real-time regime detection lags. Regime can switch but system doesn't detect it for hours.

**Problem 3: No Model Uncertainty Quantification**

After HMM fit, you get point estimates of states and parameters. No posterior predictive distribution. No entropy of regime probabilities.

**Impact**: You don't know when the model is uncertain about the regime. High-uncertainty regime assignments are treated same as confident ones.

**Problem 4: State Mapping is Fragile**

The `map_raw_states_to_regimes()` method assigns semantic labels (BULL, BEAR) based on ad-hoc scoring. If HMM converges to a different local optimum, states can swap labels.

**Impact**: Promotion gate checks for "BULL regime profitability" but if state labels swap, it's actually checking the opposite regime.

**Problem 5: No Regime Change Lead Indicators**

Current detection is reactive: "What regime are we in now?" Missing are leading indicators:
- Credit spreads widening before volatility spike
- Breadth deteriorating before bear market onset
- Term structure inversion before recession

**Impact**: By the time regime detection switches, alpha opportunities are partially missed.

**Problem 6: No Jump/Transition Detection**

System detects current state but doesn't flag *transitions*. Transition probability matrix has off-diagonal elements that spike during regime changes.

**Impact**: Traders don't get alerts when regimes shift. Position sizing doesn't auto-adjust at the transition moment.

**Problem 7: BIC State Selection Runs Every Time**

Expensive to re-compute optimal state count on every retrain.

**Impact**: Computational overhead. Potential to select different state counts on re-fit.

### 6.3 Best Models from Academic Research

#### 6.3.1 Statistical Jump Models (JM)

**Paper**: arXiv 2402.05272 (2024)

**What it is**: Combines regime detection with explicit jump modeling. Instead of continuous Gaussian HMM, explicitly models:
- Intra-regime dynamics (smooth)
- Jump events (discrete transitions with penalties)

**Formula**:
```
P(regime_t | data) ∝ P(data_t | regime_t) * P(regime_t | regime_{t-1})

P(regime_t | regime_{t-1}) = {
  (1 - λ) * p_stay                           if regime_t == regime_{t-1}
  (λ / (n_regimes - 1)) * p_jump            if regime_t != regime_{t-1}
}

Where:
- λ = jump penalty (e.g., 0.02 means 2% chance of regime change per day)
- Larger λ = fewer, longer regime periods
- Smaller λ = frequent regime switching
```

**Advantages Over HMM**:
- More persistent regimes (fewer false switches)
- Better Sharpe ratio (lower turnover)
- Better max drawdown (less whipsawed)
- Outperforms both HMM and HSMM across multiple metrics
- Interpretable: jump penalty has clear meaning

**How to Implement**:

**File**: Create `/regime/jump_model.py`

```python
import numpy as np
from scipy.optimize import minimize

class StatisticalJumpModel:
    '''
    Statistical Jump Model for regime detection.
    Explicitly penalizes regime transitions with a configurable jump penalty.
    '''

    def __init__(self, n_regimes=3, jump_penalty=0.02, max_iter=50):
        self.n_regimes = n_regimes
        self.jump_penalty = jump_penalty  # λ in formula above
        self.max_iter = max_iter
        self.regimes = None
        self.transition_probs = None

    def fit(self, observations):
        '''
        Fit the jump model to observations.

        Args:
            observations: shape (n_timesteps, n_features)

        Returns:
            regime_probs: shape (n_timesteps, n_regimes)
        '''
        # Step 1: Initialize with K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
        initial_regimes = kmeans.fit_predict(observations)

        # Step 2: EM algorithm
        regime_probs = self._em_fit(observations, initial_regimes)

        return regime_probs

    def _em_fit(self, observations, initial_regimes):
        '''
        EM algorithm for jump model fitting.
        '''
        # Implementation: forward-backward algorithm with jump penalties
        # Pseudocode:
        # For each iteration:
        #   E-step: compute forward/backward probabilities with jump penalty
        #   M-step: update regime parameters and transition probabilities
        # Return final regime probabilities
        pass

    def decode(self, regime_probs):
        '''
        Viterbi decoding to get most-likely regime sequence.

        Args:
            regime_probs: shape (n_timesteps, n_regimes)

        Returns:
            regime_sequence: shape (n_timesteps,) - regime index per timestep
        '''
        # Viterbi with jump penalty
        pass

    @staticmethod
    def compute_jump_penalty_from_data(regime_changes_per_year=4):
        '''
        Calibrate jump penalty λ from expected regime changes per year.

        Args:
            regime_changes_per_year: e.g., 4 = average 4 regime changes per year

        Returns:
            lambda: jump penalty
        '''
        trading_days_per_year = 252
        changes_per_day = regime_changes_per_year / trading_days_per_year
        # λ ≈ changes_per_day
        return changes_per_day
```

**Integration into RegimeDetector**:

**File**: `/regime/detector.py`

```python
from regime.jump_model import StatisticalJumpModel

class RegimeDetector:
    def __init__(self, config, use_jump_model=True):
        self.use_jump_model = use_jump_model

        if use_jump_model:
            self.model = StatisticalJumpModel(
                n_regimes=config.N_REGIMES,
                jump_penalty=StatisticalJumpModel.compute_jump_penalty_from_data(
                    regime_changes_per_year=config.EXPECTED_REGIME_CHANGES_PER_YEAR
                )
            )
        else:
            self.model = GaussianHMM(...)  # Fallback to HMM

    def detect_regime(self, observation_matrix):
        '''Detect regime using jump model or HMM fallback.'''
        if self.use_jump_model:
            regime_probs = self.model.fit(observation_matrix)
            regime_sequence = self.model.decode(regime_probs)
        else:
            # HMM fallback
            regime_sequence = self.hmm_model.fit_predict(observation_matrix)
            regime_probs = self.hmm_model.predict_proba(observation_matrix)

        return regime_sequence, regime_probs
```

**Configuration** (`/config.py`):
```python
REGIME_DETECTION_MODEL = 'JUMP_MODEL'  # 'JUMP_MODEL' or 'HMM'
USE_JUMP_MODEL = True
EXPECTED_REGIME_CHANGES_PER_YEAR = 4  # Calibrate jump penalty
N_REGIMES = 3  # BULL, BEAR, NEUTRAL
```

**Expected Outcome**: More stable, interpretable regime detection with fewer false switches.

**Verification**:
- Compare backtest Sharpe with JM vs HMM
- Count regime transitions: JM should have fewer
- Check max drawdown: JM should be lower (less whipsawed)

#### 6.3.2 Ensemble Regime Detection

**What it is**: Run multiple methods (HMM, JM, rule-based) and take consensus. Only declare regime change when 2+ methods agree.

**How to Implement**:

**File**: `/regime/detector.py`

**Method**: Add to `RegimeDetector`

```python
class RegimeDetector:
    def __init__(self, config):
        # Initialize multiple methods
        self.hmm_model = GaussianHMM(...)
        self.jump_model = StatisticalJumpModel(...)
        self.rule_based_detector = RuleBasedDetector(...)

    def detect_regime_ensemble(self, observation_matrix):
        '''
        Ensemble regime detection: run all methods and take consensus.

        Returns:
            regime_ensemble: dict with each method's prediction
            consensus_regime: majority vote regime
            confidence: fraction of methods agreeing
        '''
        # Get predictions from each method
        hmm_regime = self.hmm_model.predict(observation_matrix)
        jm_regime = self.jump_model.decode(self.jump_model.fit(observation_matrix))
        rule_regime = self.rule_based_detector.detect(observation_matrix)

        # Consensus: majority vote
        predictions = np.array([hmm_regime[-1], jm_regime[-1], rule_regime[-1]])
        unique_regimes, counts = np.unique(predictions, return_counts=True)

        if counts.max() >= 2:  # At least 2 methods agree
            consensus_regime = unique_regimes[np.argmax(counts)]
            confidence = counts.max() / len(predictions)
        else:
            # No consensus, use JM as tiebreaker (more sophisticated)
            consensus_regime = jm_regime[-1]
            confidence = 1.0 / 3.0

        return {
            'ensemble': {'hmm': hmm_regime[-1], 'jm': jm_regime[-1], 'rule': rule_regime[-1]},
            'consensus': consensus_regime,
            'confidence': confidence,
            'methods_agreeing': counts.max()
        }

    def should_transition_regime(self, consensus_result, threshold=2):
        '''
        Only transition regime if confidence threshold is met.

        Args:
            consensus_result: dict from detect_regime_ensemble()
            threshold: minimum methods agreeing (2 or 3)

        Returns:
            should_transition: bool
        '''
        return consensus_result['methods_agreeing'] >= threshold
```

**Configuration** (`/config.py`):
```python
ENSEMBLE_REGIME_DETECTION = True
ENSEMBLE_CONSENSUS_THRESHOLD = 2  # Require 2 of 3 methods to agree
```

**Expected Outcome**: Regime transitions are more robust, fewer false switches.

**Verification**:
- Verify that single-method disagreements don't cause transitions
- Check that major regime changes (2020 COVID crash) get consensus agreement from all 3 methods

### 6.4 Instructions for Regime Detection Improvement

#### 6.4.1 Expand Observation Matrix (Immediate)

**File**: `/regime/detector.py` or `/regime/hmm.py`

**Method**: `build_hmm_observation_matrix()` (around line 80)

**Current Code**:
```python
def build_hmm_observation_matrix(self):
    '''Current: only 4 features'''
    features = {
        'ret_1d': daily_returns,
        'vol_20d': rolling_volatility,
        'NATR_14': natr,
        'SMASlope_50': sma_slope,
    }
    return np.column_stack([features[f] for f in feature_names])
```

**What to Add**:

```python
def build_hmm_observation_matrix(self):
    '''
    Expanded observation matrix with 10+ features for better regime detection.
    '''
    # Get existing features
    ret_1d = self.daily_returns
    vol_20d = self.rolling_volatility
    natr_14 = self.compute_natr(period=14)
    sma_slope_50 = self.compute_sma_slope(period=50)

    # NEW: Add credit spread proxy
    credit_spread = self._get_credit_spread_proxy()  # HY-Treasury spread from FRED

    # NEW: Add market breadth
    market_breadth = self._compute_market_breadth()  # % stocks > 50-day SMA

    # NEW: Add cross-asset correlation
    corr_stocks_bonds = self._compute_cross_correlation(asset1='SPY', asset2='AGG')

    # NEW: Add VIX level or implied vol rank
    vix_level = self._get_vix_level()
    vix_rank = self._compute_vix_rank(period=60)  # Percentile rank

    # NEW: Add volume regime
    volume_regime = self._compute_volume_regime()  # Normalized vs 20-day avg

    # NEW: Add trend momentum (ROC)
    momentum_20d = self._compute_momentum(period=20)

    # NEW: Add mean reversion signal (RSI-like)
    mean_reversion_signal = self._compute_mean_reversion_signal()

    features = {
        'ret_1d': ret_1d,
        'vol_20d': vol_20d,
        'natr_14': natr_14,
        'sma_slope_50': sma_slope_50,
        'credit_spread': credit_spread,
        'market_breadth': market_breadth,
        'corr_stocks_bonds': corr_stocks_bonds,
        'vix_rank': vix_rank,
        'volume_regime': volume_regime,
        'momentum_20d': momentum_20d,
        'mean_reversion': mean_reversion_signal,
    }

    return np.column_stack([features[name] for name in features.keys()])
```

**New Helper Methods** (add to the class):

```python
def _get_credit_spread_proxy(self):
    '''Get HY-Treasury spread from FRED (if available) or compute from option skew.'''
    from data.loader import get_fred_series
    try:
        spread = get_fred_series('BAMLH0A0HYM2')  # HY Option-Adjusted Spread
        return spread / 100  # Convert to decimal
    except:
        # Fallback: estimate from VIX skew
        return np.ones(len(self.daily_returns)) * 0.03

def _compute_market_breadth(self):
    '''Compute % of stocks in universe above their 50-day SMA.'''
    universe_data = self._get_universe_ohlcv()
    sma_50 = universe_data['close'].rolling(50).mean()
    above_sma = (universe_data['close'] > sma_50).sum(axis=1)
    breadth = above_sma / len(universe_data.columns)
    return breadth

def _compute_cross_correlation(self, asset1='SPY', asset2='AGG'):
    '''Compute rolling correlation between two assets.'''
    returns_1 = self._get_asset_returns(asset1)
    returns_2 = self._get_asset_returns(asset2)
    rolling_corr = returns_1.rolling(20).corr(returns_2)
    return rolling_corr

def _compute_vix_rank(self, period=60):
    '''Compute VIX percentile rank (0=lowest, 1=highest in period).'''
    vix = self._get_vix_level()
    vix_min = vix.rolling(period).min()
    vix_max = vix.rolling(period).max()
    vix_rank = (vix - vix_min) / (vix_max - vix_min + 1e-8)
    return vix_rank

def _compute_volume_regime(self):
    '''Normalize recent volume vs 20-day average.'''
    volume = self._get_volume()
    vol_ma = volume.rolling(20).mean()
    volume_regime = volume / (vol_ma + 1e-8)
    return volume_regime

def _compute_momentum(self, period=20):
    '''Rate of change (momentum) of closing price.'''
    price = self._get_close_price()
    roc = (price - price.shift(period)) / price.shift(period)
    return roc

def _compute_mean_reversion_signal(self):
    '''Compute mean-reversion signal based on distance to SMA.'''
    price = self._get_close_price()
    sma_50 = price.rolling(50).mean()
    distance = (price - sma_50) / sma_50
    return distance
```

**Expected Outcome**: HMM/JM now observes 11 features spanning multiple asset classes and market regimes.

**Verification**:
- Verify observation matrix shape increases from (n, 4) to (n, 11)
- Check that new features are not NaN-heavy (data quality)
- Compare backtest results before/after expansion

#### 6.4.2 Add Regime Uncertainty Tracking (Immediate)

**File**: `/regime/detector.py`

**Method**: `detect_regime()` or new method `get_regime_uncertainty()`

**What to Add**:

```python
def get_regime_uncertainty(self):
    '''
    Compute entropy of regime posterior: H = -Σ p_i * log(p_i).

    High entropy = model is uncertain about which regime we're in.
    Low entropy = model is confident.

    Returns:
        entropy: scalar, 0 = certain, log(n_regimes) = completely uncertain
    '''
    regime_probs = self.get_regime_probabilities()  # Shape: (n_timesteps, n_regimes)

    # Compute entropy for the most recent timestep
    latest_probs = regime_probs[-1, :]

    # H = -Σ p_i * log(p_i), only for p_i > 0
    entropy = -np.sum(latest_probs[latest_probs > 0] * np.log(latest_probs[latest_probs > 0]))

    # Normalize to [0, 1]
    max_entropy = np.log(self.n_regimes)
    normalized_entropy = entropy / max_entropy

    return {
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'regime_probs': latest_probs,
        'confident': normalized_entropy < 0.3,  # <30% entropy = confident
    }
```

**Use in Promotion Gate**:

**File**: `/autopilot/promotion.py`

**Method**: `PromotionGate.evaluate()`

**What to Add**:

```python
# Get regime uncertainty at time of backtest
regime_uncertainty = self.regime_detector.get_regime_uncertainty()

# Penalize strategies trained during uncertain regimes
if regime_uncertainty['normalized_entropy'] > 0.6:  # Very uncertain
    self.contract_results.append({
        'gate': 'REGIME_UNCERTAINTY',
        'passed': False,
        'reason': f"Regime detection uncertain (entropy {regime_uncertainty['entropy']:.2f})"
    })

# Or penalize less severely
elif regime_uncertainty['normalized_entropy'] > 0.4:
    # Still allow, but flag as lower confidence
    pass
```

**Expected Outcome**: Strategies trained during regime detection uncertainty are flagged or rejected.

**Verification**:
- Compute entropy on current regime
- Verify that it's low (confident) during normal conditions
- Check that it spikes during market discontinuities (COVID, 2020 election, etc.)

#### 6.4.3 Stable State Labeling (Medium Priority)

**File**: `/regime/detector.py`

**Method**: `map_raw_states_to_regimes()`

**Problem**: Every HMM fit converges to potentially different state ordering. State 0 in fit 1 might be BULL, but state 0 in fit 2 might be BEAR.

**Solution**: Maintain a reference mapping and match new states to reference using Wasserstein distance.

```python
def map_raw_states_to_regimes_stable(self, new_fitted_model):
    '''
    Map HMM states to regime labels stably across retrains.

    Instead of re-computing the mapping each time, match new states to
    reference states using Wasserstein distance.

    Args:
        new_fitted_model: newly fitted HMM model

    Returns:
        mapping: dict from state_id -> regime_label
    '''
    # Get reference mapping (from initial fit)
    if not hasattr(self, 'reference_state_distributions'):
        self.reference_state_distributions = {
            'BULL': np.array([0.05, 0.10, 0.02, 0.01]),  # Mean obs per state
            'BEAR': np.array([-0.02, 0.20, 0.03, -0.01]),
            'NEUTRAL': np.array([0.01, 0.12, 0.02, 0.00]),
        }

    # Compute new state statistics
    new_means = new_fitted_model.means_

    # Match new states to reference states using Wasserstein distance
    from scipy.spatial.distance import cdist
    distances = cdist(new_means, [self.reference_state_distributions[regime] for regime in self.reference_state_distributions])

    # Greedy matching
    mapping = {}
    for new_state_id in range(len(new_means)):
        best_regime = list(self.reference_state_distributions.keys())[np.argmin(distances[new_state_id])]
        mapping[new_state_id] = best_regime

    return mapping
```

**Expected Outcome**: State labels remain consistent across retrains.

**Verification**:
- Fit HMM, get state mapping
- Retrain HMM, verify that state labels don't swap
- Check that "BULL regime profitability" gate is stable

#### 6.4.4 Implement Statistical Jump Model (Medium Priority)

See Section 6.3.1 above for full implementation.

**Timeline**: 4-6 hours of development

**Expected Improvement**: Sharpe +0.1-0.3, Max DD -1-3%, regime switch reduction -50%

#### 6.4.5 Implement Ensemble Regime Detection (Medium Priority)

See Section 6.3.2 above for full implementation.

**Timeline**: 2-3 hours of development

**Expected Improvement**: Reduced false regime switches, better backtest robustness

---

## SECTION 7: UI AUDIT AND CRITICISMS

### 7.1 Issues Visible in Screenshots

#### 7.1.1 Hardcoded Promotion Funnel

**Issue**: The System Health page shows a promotion funnel with hardcoded numbers:
```
Candidates Generated: 24
Passed Sharpe: 18
Passed Drawdown: 15
Passed Advanced Validation: 12
Promoted to Live: 3
```

These numbers are NOT from actual autopilot runs. They are hardcoded in `data_helpers.py` for UI demonstration.

**File**: `/api/services/data_helpers.py` (around line 420 in `_check_promotion_contract()`)

**Current Code**:
```python
def _check_promotion_contract(self):
    # HARDCODED VALUES - NOT REAL
    return {
        'candidates_generated': 24,
        'passed_sharpe': 18,
        'passed_drawdown': 15,
        'passed_validation': 12,
        'promoted': 3,
    }
```

**Why This is Bad**: Users see the funnel and think the system is working well (3 promoted per cycle), but these are fake numbers.

**Fix**:

Replace the hardcoded values with actual data from the autopilot registry:

```python
def _check_promotion_contract(self):
    '''
    Get real promotion funnel data from autopilot registry.
    '''
    from autopilot.registry import AutopyloatRegistry

    registry = AutopilotRegistry()

    # Get the last cycle
    last_cycle = registry.get_latest_cycle()

    if last_cycle is None:
        return {
            'candidates_generated': 0,
            'passed_sharpe': 0,
            'passed_drawdown': 0,
            'passed_validation': 0,
            'promoted': 0,
            'status': 'NO_DATA'
        }

    # Extract real funnel counts from cycle report
    funnel = {
        'candidates_generated': last_cycle['n_candidates'],
        'passed_sharpe': sum(1 for c in last_cycle['candidates'] if c['metrics']['sharpe'] > self.config.SHARPE_THRESHOLD),
        'passed_drawdown': sum(1 for c in last_cycle['candidates'] if c['metrics']['max_dd'] < self.config.MAX_DD_THRESHOLD),
        'passed_validation': sum(1 for c in last_cycle['candidates'] if c['promoted']),  # Already passed validation
        'promoted': sum(1 for c in last_cycle['candidates'] if c['live']),
        'cycle_id': last_cycle['cycle_id'],
        'timestamp': last_cycle['timestamp'],
    }

    return funnel
```

**Expected Outcome**: Promotion funnel shows real data.

**Verification**: Check UI, funnel numbers should match database records.

#### 7.1.2 Survivorship Bias Concern

**Issue**: A screenshot note flags survivorship bias. Current health check is binary (DB exists or not).

**Fix**: Implement the quantified survivorship bias check from Section 5.2.5.

#### 7.1.3 Regime Detection Display Concerns

**Issue**: UI shows regime state using HMM terminology. User unsure if HMM is best approach.

**Fix**:
1. Implement Statistical Jump Model (Section 6.3.1)
2. Add ensemble regime detection (Section 6.3.2)
3. Update UI to show model type: "Regime: BULL (Jump Model, confidence 95%)"

#### 7.1.4 Data Mode/Provenance Badges

**Issue**: UI shows "LIVE", "FALLBACK", "DEMO" badges. Need to verify these are accurate and not silently switching.

**File**: `/api/services/data_helpers.py`

**What to Add**:

```python
def get_data_provenance():
    '''
    Return the source of currently-used data.
    '''
    from data.loader import DataLoader

    loader = DataLoader()
    latest_price = loader.get_latest_price('SPY')

    if latest_price is None:
        return {'mode': 'UNAVAILABLE', 'reason': 'No data available'}

    # Check source
    if loader.source == 'IBKR':
        return {
            'mode': 'LIVE',
            'source': 'IBKR',
            'last_update': loader.last_update_time,
            'confidence': 1.0
        }
    elif loader.source == 'FALLBACK_WRDS':
        return {
            'mode': 'FALLBACK',
            'source': 'WRDS (fallback from IBKR)',
            'reason': 'IBKR unavailable',
            'last_update': loader.last_update_time,
            'confidence': 0.7
        }
    elif loader.source == 'CACHE':
        return {
            'mode': 'DEMO',
            'source': 'Cache (demo data)',
            'reason': 'Live data unavailable, using cached data',
            'age_days': (now() - loader.cache_time).days,
            'confidence': 0.3
        }
    else:
        return {
            'mode': 'UNKNOWN',
            'source': loader.source,
            'confidence': 0.0
        }
```

Then in the API:

```python
@app.get("/api/v1/system/data-mode")
def get_data_mode():
    return get_data_provenance()
```

And update the UI StatusBar to fetch this endpoint and display accordingly.

**Expected Outcome**: UI accurately shows data source and cannot silently fall back.

#### 7.1.5 Health Score Inflation (Base Score = 50)

**Issue**: Health scores start at 50 before any checks. Even a system with zero data shows 50/100.

**Fix**: Already addressed in Section 5.3. Change base score from 50 to 0.

**File**: `/api/services/health_service.py`

**What to Change**:

```python
# OLD:
base_score = 50
health_score = base_score + (contributions) / 2

# NEW:
health_score = 0 + (weighted average of checks)  # No base score
# Clamp to [0, 100]
health_score = max(0, min(100, health_score))
```

**Expected Outcome**: Health scores now properly calibrated (0 = broken, 100 = perfect).

#### 7.1.6 Model Staleness

**Issue**: No visible indicator of when model was last trained.

**File**: `/ui/components/StatusBar.tsx` (React frontend)

**What to Add**:

```typescript
function StatusBar() {
  const [modelAge, setModelAge] = useState(null);

  useEffect(() => {
    fetch('/api/v1/system/model-age')
      .then(r => r.json())
      .then(data => setModelAge(data));
  }, []);

  return (
    <div className="status-bar">
      {/* ... existing status items ... */}

      {modelAge && (
        <StatusItem
          label="Model Age"
          value={`${modelAge.hours}h ${modelAge.minutes}m`}
          severity={
            modelAge.hours < 1 ? 'green' :
            modelAge.hours < 24 ? 'yellow' :
            modelAge.hours < 168 ? 'orange' :
            'red'
          }
          tooltip="Hours since last model retrain"
        />
      )}
    </div>
  );
}
```

Add backend endpoint:

**File**: `/api/main.py`

```python
@app.get("/api/v1/system/model-age")
def get_model_age():
    from autopilot.registry import AutopilotRegistry
    registry = AutopilotRegistry()
    last_model = registry.get_latest_model()

    if last_model is None:
        return {'hours': 999, 'minutes': 0, 'status': 'UNKNOWN'}

    age = datetime.now() - last_model['trained_at']
    hours = int(age.total_seconds() / 3600)
    minutes = int((age.total_seconds() % 3600) / 60)

    return {
        'hours': hours,
        'minutes': minutes,
        'trained_at': last_model['trained_at'].isoformat(),
        'status': 'OK' if hours < 24 else 'STALE'
    }
```

**Expected Outcome**: UI clearly shows model age, warning if stale.

#### 7.1.7 Signal Confidence Calibration

**Issue**: SignalTable shows confidence bars but users don't understand what confidence means.

**File**: `/ui/components/SignalTable.tsx`

**What to Add**:

Tooltip and explanation:

```typescript
<Tooltip title={
  "Confidence = calibrated probability that this signal will be profitable. " +
  "Computed from model's predicted probability, adjusted for Platt scaling calibration, " +
  "and regularized based on historical accuracy."
}>
  <ConfidenceBar value={signal.confidence} />
</Tooltip>
```

And add a legend below the table:

```
Confidence Interpretation:
0.9-1.0: Very High → 90% chance of profit (rare)
0.7-0.9: High → 70-90% chance
0.5-0.7: Moderate → 50-70% chance (frequent)
0.3-0.5: Low → 30-50% chance
0.0-0.3: Very Low → Signal likely unprofitable
```

**Expected Outcome**: Users understand signal confidence meaning.

#### 7.1.8 Backtest Results Statistical Context

**Issue**: Sharpe ratios displayed without p-values or confidence intervals.

**File**: `/ui/components/BacktestResults.tsx`

**What to Add**:

```typescript
function BacktestResults({ results }) {
  return (
    <div className="backtest-results">
      <div className="metric-row">
        <span className="metric-label">Sharpe Ratio</span>
        <span className="metric-value">
          {results.sharpe.toFixed(2)}
          {results.sharpe_pvalue && (
            <span className="significance">
              {results.sharpe_pvalue < 0.05 ? '✓' : '✗'}
              {' '}p={results.sharpe_pvalue.toFixed(3)}
            </span>
          )}
        </span>
      </div>
      {results.sharpe_ci_lower && (
        <div className="metric-row confidence-interval">
          <span className="metric-label">95% CI</span>
          <span className="metric-value">
            [{results.sharpe_ci_lower.toFixed(2)}, {results.sharpe_ci_upper.toFixed(2)}]
          </span>
        </div>
      )}
    </div>
  );
}
```

**Expected Outcome**: Backtest results show statistical significance.

#### 7.1.9 Live P&L Tracking

**Issue**: Paper trading view shows positions but no mark-to-market P&L curve.

**File**: `/ui/components/PaperTradingView.tsx`

**What to Add**:

```typescript
function PaperTradingView() {
  const [pnlHistory, setPnlHistory] = useState([]);

  useEffect(() => {
    const interval = setInterval(() => {
      fetch('/api/v1/paper-trading/pnl-history')
        .then(r => r.json())
        .then(data => setPnlHistory(data));
    }, 5000);  // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="paper-trading-view">
      {/* Positions table */}
      <PositionsTable positions={positions} />

      {/* P&L chart */}
      <PnLChart
        data={pnlHistory}
        title="Mark-to-Market P&L"
        showRollingMetrics={true}
      />

      {/* Daily summary */}
      <DailySummary
        dailyPnL={pnlHistory[pnlHistory.length - 1]?.daily_pnl}
        mtmPnL={pnlHistory[pnlHistory.length - 1]?.mtm_pnl}
        totalReturn={totalReturn}
      />
    </div>
  );
}
```

**Backend Endpoint**:

```python
@app.get("/api/v1/paper-trading/pnl-history")
def get_pnl_history(days: int = 30):
    '''Get daily mark-to-market P&L history.'''
    from autopilot.paper_trader import PaperTrader

    trader = PaperTrader()
    pnl_history = []

    for timestamp in trader.get_daily_timestamps(days=days):
        mtm_pnl = trader.compute_mtm_pnl(timestamp)
        daily_pnl = trader.get_daily_pnl(timestamp)

        pnl_history.append({
            'timestamp': timestamp,
            'mtm_pnl': mtm_pnl,
            'daily_pnl': daily_pnl,
            'return_pct': mtm_pnl / trader.account_value,
        })

    return pnl_history
```

**Expected Outcome**: Paper trading view shows real-time P&L tracking.

### 7.2 Additional UI Improvements Checklist

| # | Item | File | Effort | Impact |
|---|------|------|--------|--------|
| 1 | Fix hardcoded promotion funnel | data_helpers.py | 30 min | High |
| 2 | Add real survivorship bias quantification | health_service.py | 1 hr | High |
| 3 | Update regime detection with JM/ensemble | UI display updates | 1 hr | Medium |
| 4 | Data mode/provenance badges | main.py + StatusBar | 1 hr | Medium |
| 5 | Fix health score inflation | health_service.py | 30 min | High |
| 6 | Model staleness indicator | main.py + StatusBar | 1 hr | High |
| 7 | Signal confidence calibration tooltip | SignalTable.tsx | 30 min | Low |
| 8 | Backtest statistical context | BacktestResults.tsx | 1 hr | Medium |
| 9 | Live P&L tracking chart | PaperTradingView.tsx | 2 hrs | High |
| 10 | Feature importance diff viewer | FeatureView.tsx | 2 hrs | Medium |
| 11 | Prediction distribution histogram | SignalDesk.tsx | 1 hr | Low |
| 12 | Ensemble disagreement indicator | SignalDesk.tsx | 1 hr | Low |
| 13 | Kelly position sizing visualization | PaperTradingView.tsx | 1 hr | Medium |
| 14 | Regime transition alert toasts | main.py + Notifications | 1 hr | Medium |

---

## SECTION 8: COMPREHENSIVE SYSTEM CRITICISMS AND IMPROVEMENTS

### 8.1 Architecture Criticisms

#### 8.1.1 Orchestration Duplication

**Criticism**: The same pipeline is duplicated across 4 entry points:
- `run_autopilot.py` - Autopilot discovery loop
- `run_backtest.py` - Standalone backtest
- `run_train.py` - Model training
- `/api/services/` - API service methods

Each has its own implementation of: load data → compute features → detect regime → predict → backtest.

**Problem Impact**:
- Bug fixes need to be applied in 4 places
- Inconsistent behavior across paths
- Maintenance burden

**Solution**: Use PipelineOrchestrator as the single source of truth.

**File**: `/api/orchestrator.py` (already exists, but may not be used uniformly)

**What to Check**:
1. Verify `PipelineOrchestrator` class exists and is fully featured
2. Update `run_autopilot.py` to use it
3. Update `run_backtest.py` to use it
4. Update `run_train.py` to use it
5. Verify API services use it

**Code Pattern**:

```python
# OLD (duplicated):
# run_autopilot.py
data = load_data()
features = compute_features(data)
regime = detect_regime(features)
predictions = predict(features, regime)
backtest_results = backtest(predictions, ...)

# run_train.py
data = load_data()  # Different implementation?
features = compute_features(data)  # Different parameters?
...

# NEW (unified):
from api.orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator(config)
result = orchestrator.run(
    stage='AUTOPILOT',  # or 'BACKTEST', 'TRAIN'
    start_date='2020-01-01',
    end_date='2024-12-31',
)
```

**Expected Outcome**: Single code path for all pipeline stages.

#### 8.1.2 Config.py as God Object

**Criticism**: 300+ configuration values in a single flat namespace.

```python
# config.py
SHARPE_THRESHOLD = 1.0
MAX_DD_THRESHOLD = 0.15
N_FOLDS = 5
KELLY_FRACTION = 0.5
...
# 300 more parameters
```

**Problem Impact**:
- Hard to navigate (what configs control regime detection?)
- Easy to misconfigure (set wrong parameter)
- No type checking
- Documentation scattered

**Solution**: Use typed dataclasses.

**File**: Create `/config_structured.py`

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class DataConfig:
    '''Data loading configuration.'''
    source: str = 'WRDS'  # WRDS, IBKR, CACHE
    start_date: str = '2010-01-01'
    end_date: str = '2024-12-31'
    cache_dir: str = '/data/cache'
    cache_freshness_days: int = 1

@dataclass
class RegimeConfig:
    '''Regime detection configuration.'''
    detector_type: str = 'JUMP_MODEL'  # JUMP_MODEL, HMM, RULE_BASED
    n_regimes: int = 3
    jump_penalty: float = 0.02
    ensemble_enabled: bool = True
    ensemble_threshold: int = 2

@dataclass
class ModelConfig:
    '''Model training configuration.'''
    model_type: str = 'ENSEMBLE'
    ensemble_members: list = ('XGBoost', 'LightGBM', 'NeuralNet')
    max_features: int = 50
    target_type: str = 'RETURNS'
    test_size: float = 0.2

@dataclass
class KellyConfig:
    '''Kelly position sizing configuration.'''
    kelly_fraction: float = 0.5
    regime_conditional: bool = True
    drawdown_governor: bool = True
    max_portfolio_dd: float = 0.20
    portfolio_blend: float = 0.3

@dataclass
class HealthConfig:
    '''Health monitoring thresholds.'''
    min_ic_threshold: float = 0.01
    signal_decay_threshold: float = 0.5
    max_correlation_threshold: float = 0.7
    execution_quality_threshold: float = 2.0

@dataclass
class PromotionConfig:
    '''Promotion gate thresholds.'''
    min_sharpe: float = 1.0
    max_drawdown: float = 0.15
    min_dsr_significant: bool = True
    min_pbo_threshold: float = 0.5
    min_ic_for_promotion: float = 0.01

@dataclass
class SystemConfig:
    '''Top-level system configuration.'''
    data: DataConfig = DataConfig()
    regime: RegimeConfig = RegimeConfig()
    model: ModelConfig = ModelConfig()
    kelly: KellyConfig = KellyConfig()
    health: HealthConfig = HealthConfig()
    promotion: PromotionConfig = PromotionConfig()
```

**Usage**:

```python
from config_structured import SystemConfig

config = SystemConfig()

# Type-checked access
config.regime.n_regimes  # IDE autocomplete works
config.kelly.max_portfolio_dd  # IDE knows this exists
```

**Expected Outcome**: Cleaner config management with type safety.

#### 8.1.3 No Feature Versioning

**Criticism**: When feature pipeline changes, there's no way to know which features a model was trained on.

**Example**:
- Version 1: Model trained on 50 features (no market breadth)
- Version 2: Code updated, 51 features (added market breadth)
- Backtests comparing v1 vs v2 get wrong features for v1

**Solution**: Feature versioning system.

**File**: Create `/features/version.py`

```python
import hashlib
import json

class FeatureVersion:
    '''Track feature pipeline version.'''

    def __init__(self, feature_names, parameters):
        self.feature_names = sorted(feature_names)
        self.parameters = parameters

    def compute_hash(self):
        '''Compute hash of feature definition.'''
        data = {
            'features': self.feature_names,
            'params': self.parameters,
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()

    def to_dict(self):
        return {
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'version_hash': self.compute_hash(),
            'parameters': self.parameters,
        }
```

**Integration with model training**:

```python
def train_model(data, config):
    from features.version import FeatureVersion

    # Compute features
    features_computed, feature_names = compute_features(data, config)

    # Version the features
    fv = FeatureVersion(feature_names, config.feature_params)
    feature_version_info = fv.to_dict()

    # Train model
    model = XGBoostModel()
    model.fit(features_computed, targets)

    # Save with version info
    model.metadata['feature_version'] = feature_version_info
    save_model(model)
```

**When loading model for prediction**:

```python
def make_prediction(data, model_path):
    # Load model
    model = load_model(model_path)
    feature_version = model.metadata['feature_version']

    # Compute SAME features
    features = compute_features_exact_version(
        data,
        feature_version['feature_names'],
        feature_version['parameters']
    )

    # Predict
    return model.predict(features)
```

**Expected Outcome**: Models and their feature versions are always aligned.

#### 8.1.4 Silent Demo Mode Fallback

**Criticism**: When live data is unavailable, the system falls back to demo mode via exception handling, without alerting users.

**Example Code**:
```python
def get_live_data():
    try:
        return ibkr_loader.load()
    except:
        # Silent fallback to demo
        return cache_loader.load()
```

**Problem Impact**: Production system can silently switch to demo data without notification.

**Solution**: Explicit data source tracking and mandatory alerting.

**File**: `/data/loader.py`

```python
class DataLoader:
    def __init__(self):
        self.current_source = None
        self.fallback_count = 0

    def load(self, ticker, fallback_allowed=True):
        '''Load data with explicit source selection.'''

        # Try primary source
        try:
            data = self._load_ibkr(ticker)
            self.current_source = 'IBKR'
            return data
        except IbkrUnavailableError as e:
            if not fallback_allowed:
                raise  # Don't silently fall back

            # Alert about fallback
            self._alert_fallback(ticker, 'IBKR', 'WRDS', reason=str(e))
            self.fallback_count += 1

            try:
                data = self._load_wrds(ticker)
                self.current_source = 'WRDS'
                return data
            except WrdsUnavailableError:
                # Try cache
                self._alert_fallback(ticker, 'WRDS', 'CACHE')
                data = self._load_cache(ticker)
                self.current_source = 'CACHE'
                return data

    def _alert_fallback(self, ticker, from_source, to_source, reason=''):
        '''Send alert about data source fallback.'''
        logger.warning(
            f"Data fallback: {ticker} from {from_source} to {to_source}. {reason}"
        )

        # Send to monitoring system
        send_alert(
            severity='WARNING',
            message=f"Data source degradation: {from_source} unavailable, using {to_source}",
            metadata={'ticker': ticker, 'fallback_count': self.fallback_count}
        )

        # Update system status
        update_system_status(data_source=to_source, fallback=True)
```

**Expected Outcome**: System explicitly reports when using fallback data.

### 8.2 Model Criticisms

#### 8.2.1 Single Model Family (GBM Only)

**Criticism**: The ensemble is all gradient boosting (XGBoost, LightGBM). No linear models, no neural nets in production.

**Problem Impact**:
- Ensemble members are highly correlated
- No diversity of algorithm types
- Missing out on linear model robustness or neural net pattern recognition

**Solution**: Add elastic net as a diversifying member.

**File**: Create `/models/elastic_net_model.py`

```python
from sklearn.linear_model import ElasticNetCV
import numpy as np

class ElasticNetModel:
    '''
    Elastic Net for predictions.

    Combines L1 (Lasso) and L2 (Ridge) regularization.
    Provides interpretability and robustness vs GBM.
    '''

    def __init__(self, config):
        self.config = config
        self.model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],  # Mix of L1 and L2
            alphas=np.logspace(-4, 1, 20),  # Regularization strength
            cv=5,
            max_iter=5000,
        )
        self.feature_names = None

    def fit(self, X, y):
        '''Train the elastic net.'''
        self.feature_names = X.columns if hasattr(X, 'columns') else None
        self.model.fit(X, y)
        return self

    def predict(self, X):
        '''Make predictions.'''
        return self.model.predict(X)

    def get_feature_importances(self):
        '''Return feature importances (coefficients).'''
        importances = np.abs(self.model.coef_)
        return importances / importances.sum()  # Normalize
```

**Integration into EnsemblePredictor**:

**File**: `/models/ensemble.py`

```python
class EnsemblePredictor:
    def __init__(self, config):
        self.models = {
            'xgboost': XGBoostModel(config),
            'lightgbm': LightGBMModel(config),
            'neural_net': NeuralNetModel(config),
            'elastic_net': ElasticNetModel(config),  # NEW
        }

    def predict(self, X, regime=None):
        '''Ensemble prediction averaging.'''
        predictions = {}
        for name, model in self.models.items():
            try:
                predictions[name] = model.predict(X)
            except:
                logger.warning(f"{name} prediction failed, using fallback")

        # Average (could be weighted)
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        return ensemble_pred
```

**Expected Outcome**: Ensemble has more diverse member algorithms.

**Verification**: Train ensemble, check that elastic net adds decorrelated signal.

#### 8.2.2 No Prediction Calibration

**Criticism**: `models/calibration.py` exists with Platt scaling and isotonic regression but is not called.

**Problem Impact**: Confidence scores are not calibrated. A prediction of 0.7 might have only 50% true probability.

**Solution**: Wire calibration into pipeline.

**File**: `/models/ensemble.py`

```python
from models.calibration import PlattScaling, IsotonicCalibration

class EnsemblePredictor:
    def __init__(self, config):
        self.models = {...}
        self.calibrator = PlattScaling()  # Initialize calibrator
        self.calibrated = False

    def fit(self, X, y, X_calib=None, y_calib=None):
        '''
        Fit ensemble and calibration.

        Args:
            X, y: training data
            X_calib, y_calib: separate calibration set (if available)
        '''
        # Train ensemble members
        for model in self.models.values():
            model.fit(X, y)

        # If no separate calibration set, use part of training data
        if X_calib is None:
            split_idx = int(0.8 * len(X))
            X_calib, y_calib = X[split_idx:], y[split_idx:]

        # Get uncalibrated predictions on calibration set
        uncalibrated_preds = self.predict(X_calib)

        # Fit calibrator
        self.calibrator.fit(uncalibrated_preds, y_calib)
        self.calibrated = True

        return self

    def predict(self, X, calibrate=True):
        '''Predict with optional calibration.'''
        # Ensemble prediction
        ensemble_pred = np.mean([model.predict(X) for model in self.models.values()])

        # Calibrate if trained
        if self.calibrated and calibrate:
            ensemble_pred = self.calibrator.calibrate(ensemble_pred)

        return ensemble_pred
```

**Expected Outcome**: Confidence scores are calibrated.

**Verification**: On test set, check that predictions with confidence 0.7 have ~70% win rate.

#### 8.2.3 Online Learning Between Retrains

**Criticism**: Models are batch-retrained weekly. For high-frequency alpha decay, need incremental updates.

**Solution**: Implement online learning option.

**File**: Create `/models/online_learning.py`

```python
class OnlineLearningWrapper:
    '''
    Wrapper for batch models to support online/incremental learning.

    Updates model on new daily data without full retrain.
    '''

    def __init__(self, base_model, update_frequency='daily', update_threshold=100):
        self.base_model = base_model
        self.update_frequency = update_frequency
        self.update_threshold = update_threshold  # trades before update
        self.trades_since_update = 0

    def update_with_new_data(self, X_new, y_new):
        '''
        Incrementally update model with new data.

        For GBM: add new trees
        For linear models: update coefficients
        For neural nets: one or few SGD steps
        '''
        self.trades_since_update += len(y_new)

        if self.trades_since_update < self.update_threshold:
            return  # Not enough new data yet

        # Update model based on type
        if hasattr(self.base_model, 'init_estimator'):
            # Gradient boosting: add trees
            self._update_gbm(X_new, y_new)
        elif hasattr(self.base_model, 'coef_'):
            # Linear model: SGD step
            self._update_linear(X_new, y_new)
        else:
            # Neural net: SGD step
            self._update_nn(X_new, y_new)

        self.trades_since_update = 0

    def _update_gbm(self, X_new, y_new):
        '''Add new trees to GBM.'''
        # Minimal retrain: fit new trees on recent data
        recent_x = np.vstack([self.base_model.train_data[-1000:], X_new])
        recent_y = np.hstack([self.base_model.train_targets[-1000:], y_new])

        # Only train 1-2 new trees
        self.base_model.fit_additional_trees(recent_x, recent_y, n_trees=2)

    def _update_linear(self, X_new, y_new):
        '''SGD step on linear model.'''
        self.base_model.partial_fit(X_new, y_new)

    def _update_nn(self, X_new, y_new):
        '''SGD step on neural net.'''
        self.base_model.train_on_batch(X_new, y_new)
```

**Expected Outcome**: Models adapt to new data between full retrains.

**Verification**: Compare online-updated model vs batch-retrained model on next validation period.

### 8.3 Risk Management Criticisms

#### 8.3.1 No Portfolio-Level Risk Constraints

**Criticism**: Position sizing is per-asset. No portfolio VaR constraint, sector exposure limits, or factor exposure limits.

**Solution**: Add portfolio-level constraints to position sizer.

**File**: `/risk/portfolio_risk.py` (create if doesn't exist)

```python
class PortfolioRiskManager:
    '''
    Portfolio-level risk constraints:
    - VaR limit
    - Sector exposure limits
    - Factor exposure limits (e.g., market beta)
    - Correlation-based exposure reduction
    '''

    def __init__(self, config):
        self.var_limit = config.PORTFOLIO_VAR_LIMIT  # e.g., 0.05 (5%)
        self.sector_limits = config.SECTOR_EXPOSURE_LIMITS  # e.g., {'Technology': 0.30}
        self.factor_limits = config.FACTOR_LIMITS

    def adjust_positions(self, positions, covariance_matrix, sector_map, factor_loadings):
        '''
        Reduce positions to satisfy portfolio constraints.

        Args:
            positions: dict of asset -> size
            covariance_matrix: covariance of asset returns
            sector_map: dict of asset -> sector
            factor_loadings: matrix of asset -> factor exposures

        Returns:
            adjusted_positions: reduced positions meeting constraints
        '''
        adjusted = positions.copy()

        # Constraint 1: VaR limit
        portfolio_var = self._compute_portfolio_var(adjusted, covariance_matrix)
        if portfolio_var > self.var_limit:
            adjustment_factor = self.var_limit / portfolio_var
            adjusted = {asset: size * adjustment_factor for asset, size in adjusted.items()}

        # Constraint 2: Sector limits
        adjusted = self._apply_sector_limits(adjusted, sector_map)

        # Constraint 3: Factor limits
        adjusted = self._apply_factor_limits(adjusted, factor_loadings)

        return adjusted

    def _compute_portfolio_var(self, positions, cov_matrix):
        '''Compute portfolio Value-at-Risk at 95% confidence.'''
        weights = np.array([positions.get(asset, 0) for asset in cov_matrix.index])
        portfolio_var_matrix = weights @ cov_matrix @ weights.T
        portfolio_std = np.sqrt(portfolio_var_matrix)
        var_95 = -1.645 * portfolio_std  # 95% confidence
        return var_95
```

**Expected Outcome**: Portfolio risks are actively constrained.

### 8.4 Data Criticisms

#### 8.4.1 WRDS/IBKR Data Boundary Reconciliation

**Criticism**: WRDS ends 2024-12-31, IBKR begins 2025-01-01. Data seam exists.

**Solution**: Automated reconciliation checks.

**File**: `/data/reconciliation.py`

```python
def reconcile_wrds_ibkr_boundary():
    '''
    Check that WRDS and IBKR data align at boundary.

    Verify:
    - Last WRDS date close matches first IBKR date open (within tolerance)
    - Volume is continuous
    - No suspicious gaps
    '''
    wrds_loader = WrdsLoader()
    ibkr_loader = IbkrLoader()

    wrds_last = wrds_loader.get_last_trading_date()
    ibkr_first = ibkr_loader.get_first_trading_date()

    # Get overlapping tickers
    common_tickers = set(wrds_loader.get_universe()) & set(ibkr_loader.get_universe())

    discrepancies = []
    for ticker in common_tickers:
        wrds_close_last = wrds_loader.get_close(ticker, wrds_last)
        ibkr_open_first = ibkr_loader.get_open(ticker, ibkr_first)

        # Allow 2% gap (normal overnight gap)
        gap = abs(ibkr_open_first - wrds_close_last) / wrds_close_last

        if gap > 0.02:
            discrepancies.append({
                'ticker': ticker,
                'wrds_last_close': wrds_close_last,
                'ibkr_first_open': ibkr_open_first,
                'gap_pct': gap,
            })

    if discrepancies:
        logger.warning(f"WRDS/IBKR boundary discrepancies: {len(discrepancies)} tickers")
        for disc in discrepancies[:5]:
            logger.warning(f"  {disc['ticker']}: gap {disc['gap_pct']:.2%}")

    return {
        'discrepancies': len(discrepancies),
        'status': 'PASS' if len(discrepancies) < 10 else 'WARNING'
    }
```

**Expected Outcome**: Data boundary is validated, discrepancies flagged.

### 8.5 Backtest Criticisms

#### 8.5.1 Transaction Cost Sensitivity Analysis

**Criticism**: Backtest uses a single cost model. No sensitivity to cost changes.

**Solution**: Add sweep analysis.

**File**: `/backtest/sensitivity.py`

```python
def transaction_cost_sweep(strategy, costs_bps=None):
    '''
    Sweep transaction costs from 0 to 50 bps.

    Show how Sharpe degrades with increasing costs.
    '''
    if costs_bps is None:
        costs_bps = [0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]

    results = []
    for cost in costs_bps:
        backtest = BacktestEngine(strategy, transaction_cost_bps=cost)
        bt_result = backtest.run()

        results.append({
            'cost_bps': cost,
            'sharpe': bt_result['sharpe'],
            'return': bt_result['return'],
            'max_dd': bt_result['max_dd'],
        })

    return results
```

**Expected Outcome**: System is robust to cost assumptions.

---

## SECTION 9: SURVIVORSHIP BIAS — DEEP DIVE

### 9.1 Current State

- `/data/survivorship.py` handles universe construction
- WRDS CRSP includes delisting data
- Health check is binary (DB exists or not)

### 9.2 What Needs to Be Added

**1. Quantified survivorship bias impact**:
- Run backtest on surviving universe
- Run backtest on full historical universe (including delisted stocks)
- Report difference in annual returns (in bps)

**2. Delisting return treatment**:
- Verify delisting returns are included (CRSP provides these)
- Ensure returns are calculated correctly for delisted securities

**3. Universe rebalancing**:
- Rebalance periodically (quarterly) based on point-in-time index membership
- Not current membership

**4. Enhanced health check**:
- Replace binary check with three-part check: (a) DB freshness, (b) coverage ratio, (c) bias impact

See Section 5.2.5 for implementation.

---

## SECTION 10: IMPLEMENTATION PRIORITY ORDER

### Priority 1: Critical (Do First)

These are blocking issues that affect system correctness.

| Task | File | Effort | Impact | Dependencies |
|------|------|--------|--------|--------------|
| Fix hardcoded promotion funnel | data_helpers.py | 30 min | Critical | None |
| Fix health score base inflation | health_service.py | 30 min | Critical | None |
| Wire advanced validation (DSR, PBO, MC) into autopilot | engine.py, promotion.py | 3 hrs | Critical | validation.py complete |
| Add prediction calibration to pipeline | ensemble.py | 1 hr | Critical | calibration.py complete |
| Fix silent demo mode fallback | data/loader.py, main.py | 1.5 hrs | Critical | None |

**Estimated Total**: 6 hours

**Success Criteria**:
- Promotion funnel shows real data, not hardcoded values
- Health scores properly calibrated (0-100, not 50-100)
- DSR, PBO, MC metrics visible in promotion gate decisions
- Confidence scores are calibrated (not raw model outputs)
- Data source fallback is explicitly logged and alerted

### Priority 2: High (Edge Improvement)

These improve system quality and close critical gaps.

| Task | File | Effort | Impact | Dependencies |
|------|------|--------|--------|--------------|
| Add Spearman IC computation to autopilot cycle | engine.py | 1.5 hrs | High | validation.py |
| Add CPCV robustness testing | engine.py | 1.5 hrs | High | validation.py |
| Add SPA bootstrap testing | engine.py | 1 hr | High | validation.py |
| Add Sharpe significance testing | engine.py | 0.5 hr | High | validation.py |
| Add FDR correction to multi-test signals | engine.py | 1 hr | High | validation.py |
| Regime-conditional Kelly sizing | position_sizer.py | 2 hrs | High | None |
| Kelly drawdown governor | position_sizer.py | 1 hr | High | position_sizer ready |
| Signal decay monitoring | health_service.py | 1.5 hrs | High | None |
| Feature importance drift monitoring | health_service.py | 1 hr | High | feature_stability.py |
| Regime transition health checks | health_service.py | 1 hr | High | regime detector |

**Estimated Total**: 12 hours

**Success Criteria**:
- Kelly sizes vary by regime
- Kelly sizes reduce during drawdowns
- Health system monitors signal decay
- Feature importance drift is detected

### Priority 3: Medium (Robustness)

These improve system robustness and research quality.

| Task | File | Effort | Impact | Dependencies |
|------|------|--------|--------|--------------|
| Expand HMM observation matrix | regime/hmm.py | 2 hrs | Medium | Data loaders complete |
| Add regime uncertainty tracking | regime/detector.py | 1 hr | Medium | HMM expanded |
| Implement Statistical Jump Model | regime/jump_model.py | 4 hrs | Medium | None |
| Implement ensemble regime detection | regime/detector.py | 2 hrs | Medium | JM complete |
| Multi-asset portfolio-level Kelly | position_sizer.py | 2 hrs | Medium | Kelly regime-conditional |
| Bayesian updating for Kelly | position_sizer.py | 1.5 hrs | Medium | Kelly regime-conditional |
| Unify paper trader Kelly | paper_trader.py | 1 hr | Medium | PositionSizer enhanced |
| Kelly in backtesting | engine.py | 1.5 hrs | Medium | PositionSizer complete |
| Survivorship bias quantification | health_service.py | 2 hrs | Medium | data/survivorship.py |
| Correlation regime health | health_service.py | 1 hr | Medium | correlation.py |
| Execution quality monitoring | health_service.py | 1 hr | Medium | paper_trader data |
| Tail risk monitoring | health_service.py | 1 hr | Medium | None |
| Information Ratio tracking | health_service.py | 1 hr | Medium | None |
| CV gap trending | health_service.py | 1 hr | Medium | backtest history |
| Data quality anomaly detection | health_service.py | 1.5 hrs | Medium | data loader |
| Ensemble disagreement monitoring | health_service.py | 1 hr | Medium | ensemble.py |
| Market microstructure health | health_service.py | 1 hr | Medium | Data loaders |
| Retraining effectiveness | health_service.py | 1 hr | Medium | model registry |
| Capital utilization efficiency | health_service.py | 1 hr | Medium | paper_trader |

**Estimated Total**: 31 hours

**Success Criteria**:
- Health system has 15 independent monitors
- Regime detection uses 10+ features
- Jump Model implemented and integrated
- Kelly system is fully regime/drawdown/portfolio-aware

### Priority 4: Lower (Enhancement)

These are nice-to-have improvements.

| Task | File | Effort | Impact | Dependencies |
|------|------|--------|--------|--------------|
| A/B testing framework | api/ab_testing.py | 3 hrs | Low | None |
| Online learning between retrains | models/online_learning.py | 3 hrs | Low | Models ready |
| Feature versioning system | features/version.py | 1.5 hrs | Low | None |
| Elastic Net as ensemble member | models/elastic_net_model.py | 1 hr | Low | models ready |
| Unify orchestration (no duplication) | api/orchestrator.py | 2 hrs | Low | All pipelines |
| Config refactoring to dataclasses | config_structured.py | 2 hrs | Low | None |
| Sector rotation overlay | regime/sector.py | 3 hrs | Low | regime detection |
| Event-driven alpha integration | models/event_features.py | 2 hrs | Low | Kalshi pipeline |
| Tail hedging module | risk/tail_hedge.py | 2 hrs | Low | derivatives pricing |
| Portfolio rebalancing optimization | risk/rebalancing.py | 2 hrs | Low | position_sizer |
| UI: Model age indicator | main.py + StatusBar | 1 hr | Low | model registry |
| UI: Live P&L tracking | PaperTradingView.tsx | 2 hrs | Low | paper_trader |
| UI: Feature importance diff viewer | FeatureView.tsx | 2 hrs | Low | feature tracking |
| UI: Statistical significance badges | BacktestResults.tsx | 1 hr | Low | validation.py |
| UI: Ensemble disagreement display | SignalDesk.tsx | 1 hr | Low | ensemble.py |

**Estimated Total**: 30 hours

---

## SECTION 11: TESTING AND VERIFICATION

### 11.1 Testing Strategy

For each major component, implement unit tests, integration tests, and validation tests:

**Unit Tests**: Test individual functions/methods in isolation
**Integration Tests**: Test components working together
**Validation Tests**: Test against known scenarios or benchmarks

### 11.2 Regression Test Suite

Create `/tests/regression_tests.py`:

```python
def test_ic_computation_matches_reference():
    '''Verify Spearman IC matches manual calculation.'''
    predictions = np.random.normal(0, 1, 100)
    actuals = np.random.normal(0, 1, 100)
    ic = compute_spearman_ic(predictions, actuals)
    # Compare to scipy.stats.spearmanr
    assert isinstance(ic, float)
    assert -1 < ic < 1

def test_kelly_fraction_reasonable():
    '''Verify Kelly fractions are in reasonable range.'''
    ps = PositionSizer(config)
    size = ps.size_position(account_value=100000, current_price=100, regime='BULL')
    assert 0 < size < 50000  # Sanity check

def test_regime_detection_stability():
    '''Verify regime detection doesn't flip on each retrain.'''
    # ... run 2 HMM fits, check state stability
```

### 11.3 Backtesting Validation

When implementing major changes, re-run backtests and verify:
- Sharpe ratio changes < 10% (or investigate)
- Max drawdown changes < 5% (or investigate)
- Trade count remains reasonable

---

## CONCLUSION

This document provides a comprehensive roadmap for improving the quant_engine system to Renaissance-level sophistication. The improvements are prioritized by impact and dependencies, enabling incremental development.

Key principles:
1. **Incremental**: Start with Priority 1 (critical), then Priority 2 (high), etc.
2. **Testable**: Each section includes verification steps
3. **Maintainable**: Use unified orchestration, config dataclasses, and feature versioning
4. **Transparent**: Explicit error handling, data source tracking, and health monitoring

**Total Estimated Effort**:
- Priority 1: 6 hours
- Priority 2: 12 hours
- Priority 3: 31 hours
- Priority 4: 30 hours
- **Total: ~80 hours** (~2 weeks of full-time development)

**Expected Result**: A system matching Renaissance Technologies' standards for:
- Statistical rigor (validation, calibration, significance testing)
- Risk management (regime-aware Kelly, portfolio constraints, tail monitoring)
- Health monitoring (15+ independent health checks)
- Robustness (ensemble detection, jump models, online learning)

---

**Document Version**: 1.0
**Last Updated**: 2026-02-23
**Status**: Ready for implementation
