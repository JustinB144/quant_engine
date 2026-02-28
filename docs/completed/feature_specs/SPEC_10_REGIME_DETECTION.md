# Feature Spec: Regime Detection Upgrade

> **Status:** Draft
> **Author:** Claude Opus 4
> **Date:** 2026-02-26
> **Estimated effort:** 130 hours across 7 tasks

---

## Why

Regime detection in `/mnt/quant_engine/engine/regime/detector.py` is **more advanced than improvement docs assume**. It already has:

- Ensemble detection with majority voting (HMM + rule-based + jump model).
- 11-feature HMM observation matrix (not 4 as claimed).
- Wasserstein state mapping (handles label switching).
- Statistical jump model (jump_model_legacy.py).

However, it has five critical gaps:

1. **Jump model audit needed:** The jump model is a 9-line re-export wrapper; the actual implementation (jump_model_legacy.py) quality is unknown. Does it detect real jumps or noise?

2. **Ensemble voting is naive:** Simple majority voting ignores model confidence. A confident HMM signal and uncertain rule-based signal are weighted equally.

3. **Regime uncertainty unused:** `get_regime_uncertainty()` (entropy of regime posterior) exists but isn't integrated into promotion gates or sizing.

4. **No online/incremental updating:** Full refit required each period; no fast online update for regime changes.

5. **New structural features missing:** SPEC 2/3 proposed spectral analysis, SSA, BOCPD; these should be integrated into the observation matrix.

6. **Cross-sectional regime consensus missing:** Are most securities in the same regime? Divergence from consensus is informative (regime transition early indicator).

Furthermore, the regime-conditional models require `MIN_REGIME_SAMPLES=200`, which means in low-frequency regimes (e.g., crash regimes lasting 20 days), no regime-specific model is trained. This is a config issue, not a code issue, but critical for performance.

This spec implements **jump model audit, confidence-weighted voting, regime uncertainty integration, structural feature expansion, and cross-sectional consensus.**

---

## What

Implement a **Regime Detection Upgrade** that:

1. **Audits the jump model:**
   - Run unit tests on jump_model_legacy.py to validate jump detection.
   - Compare jump model output on synthetic data (with known jumps) vs. real data.
   - Validate that jump detection precision > 80% (few false positives).
   - Profile computation time; ensure online detection is fast (<100ms per update).

2. **Implements confidence-weighted ensemble voting:**
   - Each ensemble component (HMM, rule-based, jump) returns regime + confidence score.
   - Weighted vote: final_regime = argmax(sum(weight_i * score_i)).
   - Weights calibrated from cross-validation: calibrated confidence = realized frequency (ECM calibration).
   - Handle disagreement: if max weight < 0.4, regime is uncertain; use uncertainty threshold.

3. **Integrates regime uncertainty into risk decisions:**
   - Compute posterior entropy: uncertainty = -sum(p_i * log(p_i)) for regime probabilities.
   - If uncertainty > threshold (e.g., 0.5 bits), reduce position sizes.
   - Use uncertainty in promotion gate: if regime uncertain, conservatively assume stress regime.

4. **Expands observation matrix with structural features:**
   - Current 11 features: ∆market cap, ∆volume, ∆correlation, HML, SMB, RMW, CMA, beta, volatility, P/E, dividend yield.
   - Add: spectral entropy (from SPEC 2), SSA trend component (from SPEC 3), BOCPD changepoint confidence.
   - New matrix: 14–15 features. Refit HMM with BIC to validate new state count.

5. **Implements online regime updating:**
   - Cache HMM state for each security (n_securities * n_states matrix).
   - On new day: update each security's HMM state incrementally (forward algorithm).
   - Full refit monthly (less frequent).
   - Speed: online update << full refit.

6. **Computes cross-sectional regime consensus:**
   - For each regime, count % of securities in each regime.
   - Consensus = max(regime_pcts). High consensus (> 80%) = agreed-upon regime. Low consensus (< 60%) = regime transition.
   - Use consensus as early warning: if consensus drops sharply, regime change imminent.

7. **Reduces MIN_REGIME_SAMPLES threshold:**
   - Current: 200 samples minimum. For regime 3 (rare crashes lasting 20 days), this is never met.
   - New: 50 samples minimum (or 10 days, whichever is larger).
   - This allows regime-specific models to train even on short regimes.

---

## Constraints

### Must-haves

- Jump model audit includes unit tests, comparison to synthetic data, and precision/recall metrics.
- Ensemble voting weighted by confidence scores (not naive majority). Weights calibrated via cross-validation.
- Regime uncertainty computed via posterior entropy and integrated into sizing decisions.
- Observation matrix expanded with spectral, SSA, BOCPD features. HMM refitted with new features and BIC state selection.
- Cross-sectional consensus computed daily; consensus pct and divergence flagged.
- Online HMM update implemented via forward algorithm; full refit monthly.
- MIN_REGIME_SAMPLES reduced to 50 samples or 10 days.

### Must-nots

- **Do not** remove existing HMM or rule-based detector; ensemble all three.
- **Do not** use naive confidence scores (e.g., max posterior probability without calibration). Calibrate on validation set.
- **Do not** modify HMM state count (4 states) without BIC validation; state count can change with new features.
- **Do not** use online update for promotion gate decisions; only for daily regime tracking. Promotion gate uses full fit.
- **Do not** hard-code cross-sectional threshold (80%); make it configurable.

### Out of scope

- Regime change early-warning system (beyond consensus divergence detection). This is a separate forecasting system.
- Alternative ensemble methods (voting vs. stacking vs. mixture-of-experts). Voting is sufficient.
- Regime-aware asset allocation (changing num_securities vs. sector allocation). This is portfolio optimization.
- Intraday regime detection. Spec focuses on daily close-to-close regimes.

---

## Current State

### Key files

- **`/mnt/quant_engine/engine/regime/detector.py`** (comprehensive): 4-state HMM, rule-based detector, jump model wrapper. `detect_ensemble()` with majority voting. `get_regime_uncertainty()` exists but unused. `regime_features()` generates 11-feature matrix. BIC state selection.
- **`/mnt/quant_engine/engine/regime/jump_model_legacy.py`** (actual implementation): The real jump detection logic.
- **`/mnt/quant_engine/engine/covariance.py`**: `compute_regime_covariance()` uses regime-conditional matrices.
- **`/mnt/quant_engine/engine/trainer.py`**: Trains regime-conditional models; checks `MIN_REGIME_SAMPLES=200`.
- **`/mnt/quant_engine/engine/regime/hmm.py`** (likely): HMM implementation details.
- **`/mnt/quant_engine/config/regime.yaml`** (new): Configuration for regime thresholds, confidence weights, consensus thresholds.

### Existing patterns to follow

- Regime function signature: `func(prices: pd.DataFrame, returns: pd.Series, regime_state: RegimeState) -> Dict` or returns RegimeState directly.
- Feature computation: `regime_features(prices, returns) -> np.ndarray` (11 features).
- Ensemble function: `detect_ensemble(prices, returns) -> Tuple[int, float]` (regime, confidence).
- Logging: use `logging` module for debug-level regime change notifications.

### Configuration

**New config file: `/mnt/quant_engine/config/regime.yaml`**

```yaml
hmm:
  n_states: 4
  observation_features: 15  # expanded from 11
  bic_state_selection: true
  cov_type: "diag"  # or "full"

ensemble:
  components:
    - name: "hmm"
      weight: 0.5  # will be recalibrated
    - name: "rule_based"
      weight: 0.3
    - name: "jump_model"
      weight: 0.2

  confidence_calibration:
    enabled: true
    method: "ecm"  # Empirical Calibration Matrix
    window: 252  # days, for calibration

  disagreement_handling:
    consensus_threshold: 0.80  # 80% agreement for high confidence
    uncertain_regime_fallback: "stress"  # if uncertain, assume stress

uncertainty:
  entropy_threshold: 0.5  # bits, flag if entropy > 0.5
  sizing_multiplier:
    entropy_0_0: 1.0   # no uncertainty → full size
    entropy_0_5: 0.95  # moderate uncertainty → 95% size
    entropy_1_0: 0.85  # high uncertainty → 85% size (assume stress)

cross_sectional:
  enabled: true
  consensus_threshold: 0.80  # flag if < 80% securities in same regime
  divergence_window: 20  # days, for detecting sharp consensus drops
  early_warning_threshold: 0.60  # if consensus drops below 60%, warning

regime_samples:
  min_samples: 50  # reduced from 200
  min_days: 10

structural_features:
  spectral_entropy:
    enabled: true
    window: 20
  ssa:
    enabled: true
    embedding_dim: 10
  bocpd:
    enabled: true
    hazard_rate: 0.01

online_update:
  enabled: true
  method: "forward_algorithm"  # fast incremental update
  full_refit_frequency: "monthly"  # full refit once per month

jump_model:
  enabled: true
  min_jump_size: 0.01  # 1% move is a jump
  max_jump_lookback: 5  # days, for jump detection window
  precision_threshold: 0.80  # alert if precision < 80%
```

---

## Tasks

### T1: Jump Model Audit and Validation

**What:** Audit `/mnt/quant_engine/engine/regime/jump_model_legacy.py`. Test jump detection on synthetic data with known jumps, measure precision/recall, and profile computation time.

**Files:**
- Read and understand jump_model_legacy.py. Document implementation details.
- Create `/mnt/quant_engine/tests/test_jump_model_validation.py`:
  - Unit tests: synthetic data with 1–3 known jumps. Verify jump_detect() identifies them.
  - Test 1: Single large jump (5% move). Verify detected.
  - Test 2: Multiple small jumps (0.5% each) vs. continuous movement. Verify small jumps not over-detected.
  - Test 3: Noise (no jumps). Verify false positive rate < 5%.
  - Test 4: Real data (2008 financial crisis, TSLA single-day moves). Verify detected jumps align with known events.
  - Precision/recall metrics: TP/(TP+FP) and TP/(TP+FN).
  - Computation time: profile jump_detect() on 1000-day window. Ensure < 100ms.
- Create `/mnt/quant_engine/docs/jump_model_audit.md`:
  - Document implementation details, algorithm, assumptions.
  - Report unit test results, precision/recall, computation time.
  - Recommendations: if precision < 80%, propose fixes; if > 100ms, propose optimization.

**Implementation notes:**
- Jump detection typically uses thresholds: if |return_t - return_{t-1:t-k}| > threshold, flag as jump.
- Precision = TP / (TP + FP); want > 80% (few false positives).
- Recall = TP / (TP + FN); want > 70% (catch most real jumps).
- Synthetic data: use returns = N(0, 0.01^2) with 2–3 injected jumps (5%, 3%, 2%).

**Verify:**
- Run unit tests. Pass/fail report.
- Check precision on real data: 2008 crash should have 5–10 detected jumps. Sept 15 (Lehman) should be one.
- Profile computation time; report bottleneck if any.

---

### T2: Confidence-Weighted Ensemble Voting

**What:** Implement confidence weighting for ensemble voting. Each component (HMM, rule-based, jump) returns regime + confidence. Weighted vote produces final regime.

**Files:**
- Modify `/mnt/quant_engine/engine/regime/detector.py`:
  - Update `detect_ensemble(prices, returns, confidence_weights=None) -> Tuple[int, float]`:
    - Call three detectors: `hmm_regime, hmm_conf = detect_hmm(...)`, similarly for rule-based and jump.
    - Weighted vote: `regime_scores = [hmm_conf * w_hmm, rule_conf * w_rule, jump_conf * w_jump]`.
    - Final regime = argmax over regimes of sum(scores where regime == i).
    - Final confidence = max(regime_scores) / sum(regime_scores).
  - Add method `calibrate_confidence_weights(validation_returns, validation_regimes) -> Dict`:
    - Train confidence calibration on validation set.
    - Use empirical calibration matrix (ECM): for each component and each regime, compute realized frequency vs. reported confidence.
    - Store calibrated weights in `self.confidence_weights`.
  - Helper method `get_component_confidence(regime, component) -> float`:
    - Return calibrated confidence score (not just raw posterior prob).
- Create `/mnt/quant_engine/engine/regime/confidence_calibrator.py` (new):
  - Class `ConfidenceCalibrator`:
    - `__init__(n_regimes=4)`.
    - `fit(predictions: Dict[str, np.ndarray], actuals: np.ndarray)`:
      - For each component and each regime, compute calibration curve.
      - predictions: {component_name: confidences (1D array)}.
      - actuals: regime labels (1D array, values 0–3).
      - Store ECM matrix.
    - `calibrate(confidence: float, component: str, regime: int) -> float`:
      - Return calibrated confidence (interpolate from ECM).
- Tests: `/mnt/quant_engine/tests/test_ensemble_voting.py`.

**Implementation notes:**
- Confidence weights should sum to 1: e.g., {hmm: 0.5, rule: 0.3, jump: 0.2}.
- Confidence = posterior probability (for HMM) or max_score / sum_scores (for rule-based).
- ECM calibration: if raw confidence = 0.7 but realized frequency = 0.6, calibrated confidence = 0.6.
- Handle disagreement: if all three detectors disagree (regimes 0, 1, 2), use uncertainty handling (fallback to stress).

**Verify:**
- Train on 1000-day window. Fit calibration. Test on held-out 250 days.
- Verify weighted vote matches individual detector confidence.
- Verify calibration improves on validation set (expected calibration error decreases).

---

### T3: Regime Uncertainty Integration

**What:** Compute posterior entropy (regime uncertainty) and integrate into sizing decisions via a size multiplier.

**Files:**
- Modify `/mnt/quant_engine/engine/regime/detector.py`:
  - Update `get_regime_uncertainty(regime_posterior: np.ndarray) -> float` (likely already exists; enhance if needed):
    - Compute entropy: U = -sum(p_i * log(p_i)) for regime probabilities p_i.
    - Return U in bits or nats (use bits; max = log2(4) = 2 bits for 4 regimes).
  - Add method `compute_regime_posterior(prices, returns) -> np.ndarray`:
    - Return [p_0, p_1, p_2, p_3] = probabilities of each regime.
- Create `/mnt/quant_engine/engine/regime/uncertainty_gate.py` (new):
  - Class `UncertaintyGate`:
    - `__init__(config: Dict)`: Load uncertainty thresholds and size multipliers from regime.yaml.
    - `compute_size_multiplier(uncertainty: float) -> float`:
      - Interpolate from config: uncertainty 0.0 → 1.0x, uncertainty 0.5 → 0.95x, uncertainty 1.0 → 0.85x.
      - Return multiplier ∈ [0.8, 1.0].
    - `apply_uncertainty_gate(weights: np.ndarray, uncertainty: float) -> np.ndarray`:
      - Compute multiplier. Return weights * multiplier.
    - `should_assume_stress(uncertainty: float) -> bool`:
      - If uncertainty > hard_threshold, assume stress regime (for promotion gate).
- Modify promotion gate or sizing pipeline:
  - After regime detection, compute uncertainty.
  - Call `uncertainty_gate.apply_uncertainty_gate()` to reduce sizes.
  - If uncertain, use stress regime constraints (tighter limits).
- Tests: `/mnt/quant_engine/tests/test_regime_uncertainty.py`.

**Implementation notes:**
- Entropy U = 0 → regime certain. U = 2 bits → all regimes equally likely.
- Size multiplier: continuous interpolation based on entropy. High entropy (uncertain) → slightly reduced sizes.
- Promotion gate: if uncertainty > 0.8 bits (all regimes almost equally likely), conservatively assume stress regime (smallest position sizes).

**Verify:**
- Compute entropy on uniform regime posterior [0.25, 0.25, 0.25, 0.25]. Verify entropy = 2 bits.
- Compute entropy on concentrated posterior [0.9, 0.05, 0.03, 0.02]. Verify entropy ≈ 0.3 bits.
- Apply uncertainty gate: entropy 0.5 → multiplier 0.95. Verify sizes reduced by 5%.

---

### T4: Expand Observation Matrix with Structural Features

**What:** Integrate spectral entropy, SSA trend component, and BOCPD changepoint confidence into the HMM observation matrix. Expand from 11 to 14–15 features. Refit HMM with BIC.

**Files:**
- Modify `/mnt/quant_engine/engine/regime/regime_features.py` (or equivalent):
  - Current features (11): market_cap_change, volume_change, correlation_change, HML, SMB, RMW, CMA, beta, volatility, PE, dividend_yield.
  - Add features:
    - Spectral entropy (from `/mnt/quant_engine/engine/features/spectral.py`, spec 2): measure of frequency content.
    - SSA trend component (from `/mnt/quant_engine/engine/features/ssa.py`, spec 3): strength of trend vs. noise.
    - BOCPD changepoint confidence (from `/mnt/quant_engine/engine/features/bocpd.py`, spec 3): probability of changepoint in last 5 days.
  - New signature: `regime_features_expanded(prices, returns, ...) -> np.ndarray` (14–15 features).
  - Implement feature normalization: each feature scaled to [0, 1] or standardized.
- Modify `/mnt/quant_engine/engine/regime/detector.py`:
  - Update `build_hmm_observation_matrix()` to use expanded features.
  - Refit HMM: `hmm_model = GaussianHMM(n_components=?, cov_type='diag')`.
  - Run BIC: `bic_scores = [GaussianHMM(n_components=k).fit(X).bic(X) for k in range(2, 6)]`.
  - Select optimal n_states from BIC. If different from 4, alert and document.
- Tests: `/mnt/quant_engine/tests/test_observation_matrix_expansion.py`.

**Implementation notes:**
- Spectral entropy: FFT on rolling window (e.g., 20 days), compute entropy of power spectrum.
- SSA trend: decompose signal into trend + noise. Report trend strength (% of variance explained by trend).
- BOCPD changepoint: run Bayesian Online Changepoint Detection. Report probability of changepoint in last 5 days.
- Feature normalization: standardize each feature (mean 0, std 1) before feeding to HMM.
- BIC state selection: if optimal n_states changes from 4 to 5, retrain all downstream models.

**Verify:**
- Compute spectral entropy on synthetic sine wave. Verify single frequency → low entropy.
- Compute spectral entropy on white noise. Verify high entropy (many frequencies).
- Build expanded observation matrix on real data. Verify 14–15 features.
- Refit HMM. Run BIC. Report optimal state count. If ≠ 4, investigate.

---

### T5: Online Regime Updating via Forward Algorithm

**What:** Implement incremental HMM state updating via forward algorithm (fast). Full refit occurs monthly. Daily updates use cached state from previous day.

**Files:**
- Create `/mnt/quant_engine/engine/regime/online_update.py` (new):
  - Class `OnlineRegimeUpdater`:
    - `__init__(hmm_model: GaussianHMM)`: Cache HMM model.
    - `forward_step(observation: np.ndarray, prev_state_prob: np.ndarray) -> Tuple[np.ndarray, float]`:
      - Implement forward algorithm: given new observation and previous state probability, compute updated state probability.
      - Formula: state_prob_t = normalize(transition_matrix.T @ (state_prob_{t-1} * emission_prob)).
      - Return (updated_state_prob, likelihood).
    - `update_regime_for_security(security_id, observation) -> int`:
      - Get previous state prob from cache.
      - Call forward_step.
      - Update cache.
      - Return max_likelihood_regime.
    - `should_refit(last_refit_date: date) -> bool`:
      - Return True if last_refit > 30 days ago.
- Modify `/mnt/quant_engine/engine/regime/detector.py`:
  - Add online updater instance.
  - Daily regime detection:
    - If refit_needed, full refit: `hmm_model.fit(X_all)`.
    - Else, online update: loop over securities, call `updater.update_regime_for_security()`.
  - Log computation time for both paths.
- Tests: `/mnt/quant_engine/tests/test_online_update.py`.

**Implementation notes:**
- Forward algorithm: O(n_states^2 * n_features) per observation, vs. O(n_obs * n_states^2) for full refit. ~100x faster.
- Cache: store state probability vector for each security. Update in-place daily.
- Full refit monthly: retrain on last 252 days of data, reset state probability cache.
- Edge case: if new observation is far outside training range, consider it anomalous (log warning).

**Verify:**
- Synthetic data: generate 50 observations from HMM. After observation 25, switch hidden state. Verify online updater tracks state change.
- Compare online update output vs. full refit output on same data. Verify they agree (within numerical error).
- Profile: online update on 500 securities should be < 1 second. Full refit should be < 30 seconds.

---

### T6: Cross-Sectional Regime Consensus

**What:** Compute % of securities in each regime daily. Detect consensus (>80% in same regime) vs. divergence (<60%). Flag rapid consensus drops as early warning.

**Files:**
- Create `/mnt/quant_engine/engine/regime/consensus.py` (new):
  - Class `RegimeConsensus`:
    - `__init__(config: Dict)`: Load thresholds from regime.yaml.
    - `compute_consensus(regime_per_security: List[int]) -> Dict`:
      - Count securities per regime.
      - Consensus = max(pct_per_regime).
      - Return {consensus: 0.82, regime_pcts: [0.35, 0.43, 0.15, 0.07], consensus_regime: 1, divergence: 0.43}.
    - `detect_divergence(consensus_history: List[float], window: int = 20) -> Tuple[bool, Dict]`:
      - Fit linear trend to consensus over last 20 days.
      - If slope < -0.01 (consensus falling by 1% per day), flag divergence.
      - Return (diverging, {slope, current_consensus, historical_avg_consensus}).
    - `early_warning(consensus: float, threshold: float = 0.60) -> Tuple[bool, str]`:
      - If consensus < threshold, regime transition might be imminent.
      - Return (warning, reason).
- Modify `/mnt/quant_engine/engine/regime/detector.py`:
  - On each day, after computing regime for all securities:
    - Call `consensus.compute_consensus(regime_per_security)`.
    - Call `consensus.detect_divergence()`.
    - Log consensus and divergence metrics.
    - Store in time-series database (alongside health history).
- Tests: `/mnt/quant_engine/tests/test_regime_consensus.py`.

**Implementation notes:**
- Consensus: if 82% of securities are in regime 1, consensus = 0.82.
- Divergence: compute rolling slope of consensus over 20 days. If slope < -1% per day, divergence detected.
- Early warning: consensus < 60% suggests regime transition (no clear majority).
- Cross-sectional: requires regime detection per-security. Check if run_train.py does this (likely yes).

**Verify:**
- Synthetic data: 80 securities all in regime 1, 20 in regime 2. Compute consensus; verify 0.80.
- Simulate regime change: day 1–10 consensus 0.80, day 11–20 consensus 0.40. Fit trend; verify divergence detected.
- Simulate consensus drop to 0.55. Verify early_warning returns True.

---

### T7: Reduce MIN_REGIME_SAMPLES and Full Integration Testing

**What:** Reduce MIN_REGIME_SAMPLES from 200 to 50. Test that regime-conditional models train even for short regimes. Perform full integration test of upgraded regime detection.

**Files:**
- Modify `/mnt/quant_engine/engine/trainer.py`:
  - Change `MIN_REGIME_SAMPLES = 200` to `MIN_REGIME_SAMPLES = 50`.
  - Also add: `MIN_REGIME_DAYS = 10` (minimum 10 days in regime before training).
  - Update condition: `if n_samples >= MIN_REGIME_SAMPLES and n_days >= MIN_REGIME_DAYS: train_regime_model()`.
- Create `/mnt/quant_engine/tests/test_regime_detection_integration.py`:
  - End-to-end test: full backtest window with regime detection, ensemble voting, uncertainty gating, consensus computation.
  - Test 1: Normal regime (2016–2017). Verify ensemble converges on regime 0–1, confidence > 0.7, consensus > 0.80.
  - Test 2: Crash regime (2008 Sept–Dec). Verify regime shifts to 3, uncertainty increases, consensus drops.
  - Test 3: Recovery (2009). Verify regime transitions smoothly 3→2→1.
  - Test 4: Regime-conditional models train for each regime (even short ones like crash).
  - Verify: regime posterior probabilities make sense, uncertainty gate reduces sizes in uncertain periods, consensus early warnings precede actual regime changes.
- Create `/mnt/quant_engine/docs/regime_detection_guide.md`:
  - Architecture overview: HMM + rule-based + jump model ensemble, confidence weighting, uncertainty gating, consensus.
  - Config reference: regime.yaml parameters.
  - API reference: public functions and classes.
  - Example: how to interpret regime output, uncertainty, consensus.
- Create `/mnt/quant_engine/docs/regime_audit_summary.md`:
  - Summary of jump model audit (T1), confidence calibration results (T2), structural feature impact (T4), online update speedup (T5).

**Implementation notes:**
- MIN_REGIME_SAMPLES = 50 allows regime-specific training on 10-day crash periods (20 trading days at 50% drop-out).
- Integration test should cover transitions between all pairs of regimes.
- Verify that changing MIN_REGIME_SAMPLES doesn't break existing code (e.g., assumes regime_model exists).

**Verify:**
- Run trainer on 10-year backtest with MIN_REGIME_SAMPLES=50. Verify regime-specific models train for all regimes.
- Run integration tests. All pass.
- Compare regime detection (old vs. new): confidence weighting, uncertainty gating, consensus should improve.
- Profile: online update faster, full refit occurs < 1x/month.

---

## Validation

### Acceptance criteria

1. **Jump model audited:** Unit tests pass. Precision > 80%, recall > 70%. Computation time < 100ms per update. Audit report documents implementation.
2. **Confidence-weighted voting:** Ensemble voting uses calibrated confidence weights (not naive majority). Calibration improves on validation set.
3. **Regime uncertainty integrated:** `get_regime_uncertainty()` computes entropy. Size multiplier applied based on entropy. Sizing reduces by 5–15% in high-uncertainty periods.
4. **Observation matrix expanded:** 14–15 features (added spectral, SSA, BOCPD). HMM refitted with BIC. Optimal state count documented.
5. **Online update implemented:** Forward algorithm computes regime updates in < 1 second for 500 securities. Full refit occurs monthly.
6. **Cross-sectional consensus:** Consensus computed daily. Divergence detected when slope < -1%/day. Early warning when consensus < 60%.
7. **MIN_REGIME_SAMPLES reduced:** Changed to 50. Regime-conditional models train for all regimes (even short crash regimes).
8. **Integration test passes:** End-to-end backtest with upgraded detection. Regime transitions, uncertainty, consensus behave as expected.
9. **Documentation complete:** `/mnt/quant_engine/docs/regime_detection_guide.md` and audit summary document all changes.
10. **Tests pass:** All unit and integration tests pass. Coverage > 85%.

### Verification steps

1. Run jump model unit tests (T1). Report precision, recall, computation time.
2. Train confidence calibration on 1000-day window (T2). Test on 250-day holdout. Report ECM and calibration improvement.
3. Compute regime posterior entropy on synthetic uniform distribution (T3). Verify entropy = 2 bits. Apply size multiplier; verify reduction.
4. Build expanded observation matrix on real data (T4). Refit HMM; run BIC. Report state count (should be 4, or close).
5. Run online updater on synthetic HMM data (T5). Compare online output to full refit. Verify agreement within 1%.
6. Profile online update on 500 securities. Report computation time.
7. Compute regime consensus on real data (T6). Verify consensus = max(regime_pcts). Detect divergence on synthetic falling trend.
8. Change MIN_REGIME_SAMPLES to 50. Run trainer on backtest. Verify regime-specific models train.
9. Run integration test on 10-year window. Verify regime transitions, uncertainty, consensus behavior.
10. Run pytest on test suite. Report coverage.

### Rollback plan

- **If jump model precision < 80%:** Increase min_jump_size threshold (false positives are worse than false negatives).
- **If confidence calibration doesn't improve validation accuracy:** Revert to naive majority voting (simpler, robust).
- **If uncertainty gating causes excessive position reduction:** Increase entropy threshold (only gate when very uncertain, entropy > 1.0).
- **If spectral/SSA features break HMM fitting:** Remove those features and refit (back to 11 features).
- **If online update diverges from full refit:** Fall back to full refit daily (slower but accurate).
- **If consensus computation is expensive:** Cache consensus and update only weekly (not daily).
- **If MIN_REGIME_SAMPLES=50 causes regime model overfitting:** Increase back to 100 or 150 (balance).
- **If tests fail:** Revert regime detection changes; keep only jump model audit (no-op for production).

---

## Notes

- **Jump model is critical but understudied:** Many trading systems oversimplify jump detection. A rigorous audit is essential before relying on it.
- **Confidence weighting scales with team:** Naive voting works if all detectors are equally good. With heterogeneous models, weighting is essential.
- **Regime uncertainty is a leading indicator:** High entropy before regime change is valuable for early warning. Monitor but don't over-interpret single spikes.
- **Structural features compound:** Adding 4 new features to 11 increases dimensionality by 36%. Risk of overfitting HMM to noise. BIC state selection mitigates this.
- **Online update is a performance optimization:** Does NOT improve accuracy (full refit is unbiased, online update is approximate). Use only if speed is critical.
- **Cross-sectional consensus is a check on regime detection:** If 80% of securities agree on regime but 20% disagree, investigate the outliers. Could be sector-specific regime or data error.
- **MIN_REGIME_SAMPLES trade-off:** Lower threshold allows training on short regimes but increases overfitting risk. 50 samples is reasonable for 250-day training window (20% regime).
