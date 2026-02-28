# Feature Spec: Signal Enhancement + Meta-Labeling + Fold-Level Validation

> **Status:** Draft
> **Author:** Claude Opus 4
> **Date:** 2026-02-26
> **Estimated effort:** 120 hours across 8 tasks

---

## Why

The current autopilot engine calls statistical validation tests (IC, CPCV, SPA) in `_evaluate_candidates()`, but these validations operate at the summary level only. The signal selection still uses a fixed threshold after cross-sectional z-scoring rather than adaptive top-K selection. Meta-labeling—a technique that trains a secondary classifier to predict when the primary signal is reliable—does not exist in the pipeline. Additionally, fold-level metric granularity is lost in the promotion gate, preventing regime-specific signal quality assessment.

This spec addresses three distinct gaps:
1. **Cross-sectional top-K selection** instead of fixed threshold
2. **Meta-labeling pipeline** for signal reliability prediction
3. **Fold-level metric tracking** through the validation and promotion gates

These changes improve signal robustness by adapting selection criteria to market regime and explicitly modeling signal confidence.

---

## What

### Part 1: Cross-Sectional Top-K Selection
Modify the autopilot signal selection to rank candidates by z-scored signal strength and select the top-K by quantile, rather than applying a static threshold. This ensures the engine always has a target universe size and adapts to changing signal dispersion across market conditions.

### Part 2: Meta-Labeling Framework
Implement a meta-labeling layer that trains a secondary classifier (gradient boosting) on whether the primary signal is correct. Features include signal magnitude, recent signal volatility, regime indicators, and return volatility. This allows filtering low-confidence signals even when primary signal is strong.

### Part 3: Fold-Level Metrics in Promotion Gate
Extend the promotion gate to track metrics per walk-forward fold (training period, test period), not just aggregated statistics. This reveals whether signal quality degrades in specific regimes or forward dates. Composite score now includes fold consistency penalty.

---

## Constraints

### Must-haves
- Cross-sectional top-K selection with configurable quantile (default 0.70)
- Meta-labeling model training on holdout data with 5-fold cross-validation
- Fold-level IC, Sharpe, win_rate, profit_factor storage in validation output
- Fold consistency metric (std dev of fold Sharpe scores) in promotion gate
- Integration with existing walk_forward_validate() output
- Backward compatibility: if meta-labeling disabled, signal behavior unchanged
- Fallback to HeuristicPredictor if meta-labeling model fails

### Must-nots
- Do not retrain meta-labeling on live trading data (only on historical walk-forward folds)
- Do not apply meta-labeling threshold that eliminates >90% of signals in any regime
- Do not modify the existing validation.py statistical tests (IC, CPCV, SPA remain unchanged)
- Do not create new database tables; store meta-labeling predictions in existing signal cache
- Do not break the existing promotion gate score formula; add fold_consistency as new optional component

### Out of scope
- Stacking multiple meta-labeling models (single XGBoost only)
- Real-time meta-labeling retraining (weekly batch only)
- Portfolio-level meta-labeling (per-signal only)
- Alternative thresholding approaches (top-K quantile only, no percentile rank)

---

## Current State

### Key files
- `/quant_engine/autopilot/engine.py` (991 lines): `_evaluate_candidates()` calls validation tests but uses fixed signal threshold
- `/quant_engine/validation.py`: Has `walk_forward_validate()` returning (pred, label, metrics). No fold-level metrics exposed
- `/quant_engine/promotion_gate.py` (282 lines): `CompositeScorer.score()` aggregates metrics across all folds; no per-fold storage
- `/quant_engine/predictor.py`: HeuristicPredictor defined; no meta-labeling layer
- `/quant_engine/config.py`: Signal thresholds hardcoded (SIGNAL_Z_THRESHOLD=1.5)

### Existing patterns to follow
1. The validation.py uses walk_forward_validate() which returns dict with keys 'ic', 'sharpe', 'win_rate', 'profit_factor', 'pbo'
2. The promotion_gate.py has CompositeScorer with weight_* parameters for each metric
3. The autopilot.engine.py uses _walk_forward_predictions() to orchestrate validation
4. Model serialization pattern: joblib dump to `/models/candidates/{model_id}/` directory
5. Configuration inheritance: engine loads settings from config.py and allows runtime overrides

### Configuration
```python
# Current config.py (relevant sections)
SIGNAL_Z_THRESHOLD = 1.5  # Static threshold
EXEC_MAX_PARTICIPATION = 0.02
MAX_ANNUALIZED_TURNOVER = 500

# New config entries needed:
SIGNAL_TOPK_QUANTILE = 0.70  # Select top 70% by z-scored signal
META_LABELING_ENABLED = True
META_LABELING_RETRAIN_FREQ_DAYS = 7
META_LABELING_FOLD_COUNT = 5
META_LABELING_MIN_SAMPLES = 500
META_LABELING_XGB_PARAMS = {
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
META_LABELING_CONFIDENCE_THRESHOLD = 0.55
FOLD_CONSISTENCY_PENALTY_WEIGHT = 0.15
```

---

## Tasks

### T1: Implement Cross-Sectional Top-K Selection

**What:** Modify `autopilot/engine.py._evaluate_candidates()` to rank signals by cross-sectional z-score and select top-K by quantile instead of applying fixed threshold. Replace `signal > SIGNAL_Z_THRESHOLD` logic with quantile-based filtering.

**Files:**
- `/quant_engine/autopilot/engine.py` (modify `_evaluate_candidates()` method, ~50 lines)
- `/quant_engine/config.py` (add `SIGNAL_TOPK_QUANTILE`, `SIGNAL_Z_THRESHOLD` marked deprecated)

**Implementation notes:**
- In `_evaluate_candidates()`, after computing cross-sectional z-scores for signal, compute `threshold = np.quantile(signals, 1 - SIGNAL_TOPK_QUANTILE)` where quantile is measured across all candidates in the universe
- Keep fallback to HeuristicPredictor unchanged; only filter which candidates to pass to validation
- Log the derived threshold value and count of selected candidates per cycle for monitoring
- Ensure regime state affects quantile selection (e.g., during CRITICAL drawdown, use 0.50 quantile to reduce position count)
- Test with synthetic z-scores to verify quantile computation is correct

**Verify:**
- Unit test: signal quantile selection produces expected candidate count within ±5%
- Integration test: run full autopilot cycle, verify candidate count changes with signal dispersion
- Regression test: compare old threshold vs new quantile with historical data, signal count stability over time

---

### T2: Build Meta-Labeling Model Training Pipeline

**What:** Create new module `autopilot/meta_labeler.py` with class `MetaLabelingModel` that trains an XGBoost classifier to predict whether primary signal is correct. Training uses walk-forward fold data with features: signal magnitude, signal volatility, regime state, return volatility, price momentum.

**Files:**
- `/quant_engine/autopilot/meta_labeler.py` (new, ~300 lines)
- `/quant_engine/config.py` (add META_LABELING_* parameters)
- `/quant_engine/autopilot/engine.py` (modify `_evaluate_candidates()` to call meta_labeler, ~20 lines)

**Implementation notes:**
- MetaLabelingModel class has methods:
  - `__init__(config)`: store XGB params, confidence threshold
  - `build_meta_features(signals, returns, volatility, regime_states) -> DataFrame`: create 10-15 features per sample
    - signal_magnitude (absolute value of z-score)
    - signal_volatility (rolling std of signal over last 20 days)
    - regime_state (one-hot: NORMAL, WARNING, CAUTION, CRITICAL, RECOVERY)
    - return_volatility (realized vol of asset returns, last 20 days)
    - price_momentum (10-day return)
    - signal_autocorr (lag-1 autocorrelation of signal)
    - max_dd_realized (max drawdown of this asset in last 60 days)
  - `train(fold_signals, fold_returns, fold_labels, fold_regimes) -> fit XGBoost model`: use stratified 5-fold CV on fold data, return best estimator
  - `predict_confidence(meta_features) -> np.array`: return P(signal_correct) for each sample
  - `save(filepath)` and `load(filepath)`: use joblib for serialization
- Training happens on each walk-forward fold's training data, not on live data
- Use early stopping with 10% validation split within fold training
- Store feature importance for monitoring
- Handle class imbalance: if signal correct/incorrect classes are imbalanced (>70/30 split), apply class_weight='balanced' to XGB

**Verify:**
- Unit test: meta features computed correctly for edge cases (no variance, all zeros, NaN handling)
- Unit test: XGB training produces model with reasonable feature importance (no single feature >50%)
- Integration test: meta-labeling confidence correlates with realized signal accuracy (correlation >0.40)
- Backtest test: apply meta-labeling filter to historical signals, verify it removes low-accuracy signals

---

### T3: Add Meta-Labeling Filtering to Signal Pipeline

**What:** Integrate meta-labeling predictions into `autopilot/engine.py._evaluate_candidates()` to filter candidates. After computing top-K signals, pass meta-features through meta-labeling model, apply confidence threshold, and only validate candidates with confidence > threshold.

**Files:**
- `/quant_engine/autopilot/engine.py` (modify `_evaluate_candidates()`, ~40 lines)
- `/quant_engine/autopilot/meta_labeler.py` (add logging and edge case handling)

**Implementation notes:**
- After ranking candidates by z-score and applying top-K quantile, build meta_features for those candidates
- Call `meta_labeler.predict_confidence(meta_features)` to get array of confidences
- Filter candidates: `mask = confidences > META_LABELING_CONFIDENCE_THRESHOLD`
- Log filtering stats: how many candidates removed, confidence distribution percentiles (10th, 50th, 90th)
- If meta-labeling is disabled in config, skip filtering (but still rank by top-K)
- If meta_labeler.predict_confidence() raises exception, log error and skip filtering (fail-open)
- Only update meta_labeler model during weekly retraining cycle, not during daily predict cycles

**Verify:**
- Unit test: filtering logic preserves high-confidence candidates and removes low-confidence ones
- Integration test: confidence threshold applied correctly, signal count reduced by expected amount
- Regression test: no candidates incorrectly filtered due to NaN or out-of-range confidences

---

### T4: Extend Validation Output to Include Per-Fold Metrics

**What:** Modify `validation.py.walk_forward_validate()` to track metrics per fold (training period, test period), not just aggregate. Return new dict structure with fold_metrics list. Each fold_metric contains: fold_id, train_start, train_end, test_start, test_end, ic, sharpe, win_rate, profit_factor, pbo, sample_count.

**Files:**
- `/quant_engine/validation.py` (modify `walk_forward_validate()` return structure, ~30 lines)
- `/quant_engine/autopilot/engine.py` (update code that consumes validation output, ~10 lines)

**Implementation notes:**
- Modify walk_forward_validate() to yield intermediate results for each fold (keep existing iteration)
- For each fold, store test period metrics in a list before aggregating
- Return dict structure:
  ```python
  {
    'metrics_summary': {  # Existing aggregate metrics
      'ic': ..., 'sharpe': ..., 'win_rate': ..., ...
    },
    'fold_metrics': [  # New per-fold detail
      {
        'fold_id': 0,
        'train_start': pd.Timestamp(...),
        'train_end': pd.Timestamp(...),
        'test_start': pd.Timestamp(...),
        'test_end': pd.Timestamp(...),
        'ic': 0.15,
        'sharpe': 1.20,
        'win_rate': 0.55,
        'profit_factor': 1.45,
        'pbo': 0.65,
        'sample_count': 250
      },
      { 'fold_id': 1, ... },
      ...
    ]
  }
  ```
- Ensure backward compatibility: existing code accessing metrics_summary still works
- Add assertion: len(fold_metrics) matches number of walk-forward splits

**Verify:**
- Unit test: walk_forward_validate() returns both metrics_summary and fold_metrics
- Unit test: fold_metrics is list with correct length and all required keys
- Regression test: aggregate metrics_summary matches previous behavior

---

### T5: Compute Fold Consistency Metric

**What:** Add new metric to promotion_gate.py: fold_consistency = 1 - (std_dev_of_fold_sharpes / mean_sharpe). This penalizes strategies where signal quality varies wildly across folds (e.g., good in bull markets, poor in bear). Integrate into CompositeScorer.

**Files:**
- `/quant_engine/promotion_gate.py` (add `_compute_fold_consistency()` method and update CompositeScorer, ~40 lines)
- `/quant_engine/config.py` (add `FOLD_CONSISTENCY_PENALTY_WEIGHT = 0.15`)

**Implementation notes:**
- Add method `CompositeScorer._compute_fold_consistency(fold_metrics) -> float`:
  - Extract Sharpe ratio from each fold
  - Compute mean_sharpe = np.mean([fm['sharpe'] for fm in fold_metrics])
  - Compute std_sharpe = np.std([fm['sharpe'] for fm in fold_metrics])
  - If mean_sharpe <= 0, return 0 (inconsistent is meaningless for unprofitable signals)
  - Otherwise return 1.0 - (std_sharpe / abs(mean_sharpe))
  - Clip result to [0, 1]
- In CompositeScorer.score(), add line:
  - `fold_consistency = self._compute_fold_consistency(fold_metrics)` if fold_metrics provided else 1.0
- Update composite score formula:
  ```
  score = (
    self.weight_sharpe * sharpe_score +
    self.weight_win_rate * win_rate_score +
    ... (existing weights) ...
    + self.weight_fold_consistency * fold_consistency
  )
  ```
- Default weight_fold_consistency = FOLD_CONSISTENCY_PENALTY_WEIGHT (0.15)
- Log fold consistency per candidate for monitoring

**Verify:**
- Unit test: fold consistency 1.0 when all folds have same Sharpe
- Unit test: fold consistency 0.0 when Sharpe varies wildly
- Integration test: candidate with consistent folds scores higher than candidate with inconsistent folds (same aggregate Sharpe)
- Backtest test: strategies with high fold consistency outperform those with low consistency in walk-forward

---

### T6: Add Meta-Labeling Model Retraining Schedule

**What:** Add weekly retraining job to autopilot cycle that retrains the meta-labeling model on the most recent walk-forward fold data. Use configuration to set retraining frequency. Handle model versioning and fallback if retraining fails.

**Files:**
- `/quant_engine/autopilot/engine.py` (add `_retrain_meta_labeler()` method, ~40 lines)
- `/quant_engine/autopilot/meta_labeler.py` (add version tracking and fallback logic)
- `/quant_engine/config.py` (add retraining schedule parameters)

**Implementation notes:**
- In `run_cycle()`, after promotion decision, check if retraining is needed:
  - `days_since_training = (datetime.now() - meta_labeler.last_train_time).days`
  - If `days_since_training >= META_LABELING_RETRAIN_FREQ_DAYS`, call `_retrain_meta_labeler()`
- _retrain_meta_labeler() method:
  - Collect walk-forward test data from most recent N folds (e.g., last 3 folds)
  - Aggregate signals and labels across folds
  - Check sample count >= META_LABELING_MIN_SAMPLES (default 500), else skip retraining
  - Call meta_labeler.train() with aggregated data
  - Save new model to versioned path: `/models/meta_labeler/meta_labeler_v{timestamp}.joblib`
  - Keep pointer to latest version: `/models/meta_labeler/meta_labeler_current.joblib` (symlink)
  - Update `meta_labeler.last_train_time`
  - Log retraining results: old vs new model accuracy, feature importance changes
- Fallback: if retraining fails (exception), log error, keep using old model, do not raise exception
- Add health check: verify latest meta_labeler model produces reasonable confidence scores before loading

**Verify:**
- Unit test: retraining schedule computed correctly
- Integration test: meta-labeler model updated on schedule without breaking signal pipeline
- Fallback test: if retraining fails, old model still used and no exception propagated

---

### T7: Add Monitoring and Logging for Meta-Labeling

**What:** Add comprehensive logging for meta-labeling pipeline: signal quantile threshold, meta-labeling confidence distribution, filtering impact, fold consistency scores. Log to both stdout and `/logs/meta_labeling_{date}.log`.

**Files:**
- `/quant_engine/autopilot/engine.py` (add logging statements in `_evaluate_candidates()`, ~15 lines)
- `/quant_engine/autopilot/meta_labeler.py` (add logging in predict_confidence, train methods, ~20 lines)
- `/quant_engine/promotion_gate.py` (add logging for fold_consistency, ~10 lines)

**Implementation notes:**
- Log at INFO level for each autopilot cycle:
  - Quantile threshold computed and candidate count selected
  - Meta-labeling confidence stats: min, p10, p50, p90, max
  - Candidates filtered due to low confidence: count and percentage
  - Fold consistency scores per candidate
- Log at DEBUG level:
  - Top-3 signal z-scores before and after filtering
  - Feature importance from latest meta-labeling model
  - Training samples used in retraining
- Store structured logs (JSON format) for analysis:
  ```json
  {
    "cycle_date": "2026-02-26",
    "signal_topk_quantile": 0.70,
    "signal_quantile_threshold": 1.25,
    "candidates_before_filtering": 150,
    "candidates_after_topk": 105,
    "candidates_after_meta_labeling": 85,
    "meta_labeling_confidences": {
      "min": 0.45,
      "p10": 0.52,
      "p50": 0.68,
      "p90": 0.82,
      "max": 0.98
    },
    "fold_consistency_scores": {
      "mean": 0.72,
      "std": 0.15,
      "min": 0.35,
      "max": 0.95
    }
  }
  ```

**Verify:**
- Integration test: logs generated with correct format and content
- Log analysis: verify meta-labeling confidence distribution is reasonable (not skewed to 0 or 1)

---

### T8: Write Integration Tests and Documentation

**What:** Write comprehensive integration tests covering: top-K selection, meta-labeling filtering, fold metrics tracking, fold consistency scoring, and end-to-end autopilot cycle with all three components enabled. Write developer guide explaining meta-labeling design, hyperparameter tuning, and troubleshooting.

**Files:**
- `/quant_engine/tests/test_signal_meta_labeling.py` (new, ~400 lines)
- `/quant_engine/docs/DESIGN_META_LABELING.md` (new, ~100 lines)

**Implementation notes:**
- Test file structure:
  - TestCrossectionalTopK: test quantile selection with synthetic signals
  - TestMetaLabelingModel: test training, prediction, confidence calibration
  - TestFoldMetrics: test per-fold metric tracking in walk_forward_validate
  - TestFoldConsistency: test consistency score computation
  - TestIntegration: full autopilot cycle with all components
- Tests use 1-year synthetic OHLCV data, 100 symbols, with known signal pattern
- Documentation includes:
  - Design rationale for each component
  - XGB hyperparameter guidance (max_depth sensitivity, learning_rate tuning)
  - How to debug low meta-labeling accuracy (feature importance analysis)
  - How to adjust SIGNAL_TOPK_QUANTILE for different universes
  - Performance impact metrics (time, memory, accuracy improvement)

**Verify:**
- All tests pass with >95% coverage
- Documentation reviewed and approved

---

## Validation

### Acceptance criteria
1. Cross-sectional top-K selection implemented and candidate count adapts to signal dispersion
2. Meta-labeling model trains successfully on walk-forward folds with >55% accuracy in holdout validation
3. Meta-labeling confidence correlates with realized signal accuracy (correlation >= 0.40)
4. Fold-level metrics tracked and returned from walk_forward_validate()
5. Fold consistency metric computed and integrated into promotion gate score
6. Meta-labeling retraining scheduled weekly without breaking signal pipeline
7. Comprehensive logging of all three components implemented and validated
8. Integration tests pass with >95% coverage, documentation complete
9. No regression in existing autopilot behavior when components disabled in config
10. Backward compatibility: old code consuming validation output still works

### Verification steps
1. Run unit tests: `pytest tests/test_signal_meta_labeling.py -v`
2. Run integration test on 1-year synthetic data: `pytest tests/test_signal_meta_labeling.py::TestIntegration -v`
3. Run autopilot cycle with all three components enabled, verify logs output correctly
4. Verify meta-labeling confidence distribution is reasonable (not collapsed to single value)
5. Verify fold consistency scores are in [0, 1] and vary across candidates
6. Run backtest comparing old (threshold-based) signal selection vs new (top-K + meta-labeling)
7. Check logs for any errors or warnings related to meta-labeling
8. Verify weekly retraining executes without exception, model versions updated

### Rollback plan
- If meta-labeling accuracy degrades in production, disable META_LABELING_ENABLED=False
- If top-K selection causes too much volatility in candidate count, revert to fixed threshold (set SIGNAL_Z_THRESHOLD and remove top-K logic)
- If fold consistency metric causes unexpected promotion gate behavior, set FOLD_CONSISTENCY_PENALTY_WEIGHT=0.0
- Keep previous promotion gate model binary for A/B testing during rollout
- Monitor candidate count, signal accuracy, and fold consistency scores for 4 weeks post-deployment

---

## Notes

- Meta-labeling is optional: if disabled, signal selection reverts to fixed threshold (backward compatible)
- Top-K selection is adaptive: quantile adjusts with market regime (CRITICAL drawdown uses 0.50 quantile)
- Fold consistency metric only applies if fold_metrics available; otherwise ignored (backward compatible)
- Meta-labeling retraining is asynchronous: if it fails, old model continues; no exception raised
- Feature importance from meta-labeling model should be monitored: if single feature >50%, retrain with regularization increased
- Cross-sectional top-K selection assumes multiple assets/signals; single-asset strategies default to all candidates
- Fold metrics require at least 3 folds for meaningful consistency score; shorter histories use aggregate metric only
