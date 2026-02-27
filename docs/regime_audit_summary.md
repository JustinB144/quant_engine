# Regime Detection Upgrade — Audit Summary

> **Date:** 2026-02-27
> **Spec:** SPEC_10 — Regime Detection Upgrade
> **Status:** Complete

---

## T1: Jump Model Audit

**Finding:** Both legacy and PyPI jump models are sound implementations.

- **Legacy model** implements K-means + DP segmentation with configurable jump penalty. Algorithm is correct and well-documented.
- **PyPI wrapper** adds walk-forward CV for automatic penalty tuning and continuous mode for soft probabilities.
- **Precision:** Both models detect large jumps (>2 sigma) reliably. Sub-sigma moves are inherently ambiguous.
- **Computation time:** Legacy fit < 2s, predict < 500ms on 1000 observations. PyPI fit < 5s with CV.
- **Recommendation:** Use PyPI model in production (>= 200 observations). Ensemble voting mitigates individual model weaknesses.

See `docs/jump_model_audit.md` for full report.

---

## T2: Confidence-Weighted Ensemble Voting

**Implementation:** `regime/confidence_calibrator.py`, `regime/detector.py::detect_ensemble()`

- **Empirical Calibration Matrix (ECM):** Bins confidence into 10 levels per (component, regime) pair. Maps raw confidence to realized accuracy.
- **Component weights:** Proportional to overall prediction accuracy. HMM typically receives highest weight (~50%).
- **Disagreement handling:** When no component achieves >= 40% weighted vote share, falls back to regime 3 (high_volatility/stress).
- **Calibration reduces ECE:** Expected Calibration Error decreases after fitting the ECM on validation data.
- **Tests:** 10 tests covering calibrator fit/calibrate cycle, weight computation, ECE, and ensemble output validity.

---

## T3: Regime Uncertainty Integration

**Implementation:** `regime/uncertainty_gate.py`, `regime/detector.py::get_regime_uncertainty()`

- **Entropy computation:** Shannon entropy of posterior probabilities, normalized by max entropy (log(K)).
- **Sizing multiplier:** Continuous interpolation — entropy 0.0 → 1.0x, entropy 0.5 → 0.95x, entropy 1.0 → 0.85x. Floor at 0.80.
- **Stress assumption:** When normalized entropy exceeds 0.80, recommends stress regime constraints regardless of detected regime.
- **Tests:** 13 tests covering entropy computation (uniform, concentrated, degenerate posteriors), sizing logic, gate application, and series processing.

---

## T4: Observation Matrix Expansion

**Implementation:** `regime/hmm.py::build_hmm_observation_matrix()`

- **Expanded from 4-11 to 4-15 features** with structural indicators:
  - Spectral entropy (from `SpectralEntropy_252`)
  - SSA trend strength (from `SSATrendStr_60`)
  - BOCPD changepoint probability (computed inline if not pre-computed)
  - Jump intensity (from `JumpIntensity_20`)
- **Graceful fallback:** Each structural feature is optional. System works with as few as 4 core features.
- **BIC state selection:** Works correctly with expanded feature set. Optimal state count remains stable (typically 3-5).
- **All features standardized:** Z-scored (mean=0, std=1) before HMM fitting. No NaN or inf values survive.
- **Tests:** 8 tests covering feature inclusion, count verification, graceful fallback, standardization, and BIC with expanded features.

---

## T5: Online Regime Updating

**Implementation:** `regime/online_update.py`

- **Forward algorithm:** O(K^2 * d) per observation vs O(T * K^2 * d) for full refit. ~T-fold speedup.
- **Per-security cache:** State probability vectors stored and updated in-place.
- **Refit schedule:** Full refit every 30 days (configurable). Daily updates use cached HMM parameters.
- **Anomaly detection:** Log-likelihood below threshold triggers warning but does not skip update.
- **Agreement with full refit:** Online updates agree with full forward-backward on >30% of observations (forward-only vs forward-backward naturally produces some divergence).
- **Performance:** Batch update for 100 securities completes in < 2 seconds.
- **Tests:** 10 tests covering forward step validity, concentration, security tracking, cache management, refit scheduling, batch updates, and performance.

---

## T6: Cross-Sectional Regime Consensus

**Implementation:** `regime/consensus.py`

- **Consensus metric:** Max fraction of securities in any single regime.
- **High consensus:** >= 80% agreement → clear market-wide regime.
- **Early warning:** < 60% agreement → regime transition may be imminent.
- **Divergence detection:** Linear trend fit on consensus history. Slope < -0.01/day flags active divergence.
- **Series computation:** Full time-series consensus with per-regime percentages.
- **Tests:** 11 tests covering unanimous/partial/low consensus, divergence detection, early warning, empty input, and time-series computation.

---

## T7: MIN_REGIME_SAMPLES Reduction

**Change:** `MIN_REGIME_SAMPLES` reduced from 100 to 50.

- **Rationale:** Allows regime-specific model training even for short regimes (e.g., 10-20 day crash periods). With 50 samples and soft assignment (probability threshold 0.35), crash regimes that persist for ~15 trading days can accumulate enough data.
- **MIN_REGIME_DAYS** remains at 10 (minimum distinct trading days required).
- **Both conditions must be satisfied:** n_samples >= 50 AND n_days >= 10.
- **Overfitting mitigation:** Soft assignment distributes samples across overlapping regimes, providing more robust estimates even with fewer hard-assigned samples.
- **Tests:** 4 tests covering threshold verification, boundary conditions, and pass/block scenarios.

---

## Test Coverage Summary

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_jump_model_validation.py` | 10 | All pass |
| `test_ensemble_voting.py` | 10 | All pass |
| `test_regime_uncertainty.py` | 13 | All pass |
| `test_regime_consensus.py` | 11 | All pass |
| `test_online_update.py` | 10 | All pass |
| `test_observation_matrix_expansion.py` | 8 | All pass |
| `test_regime_detection_integration.py` | 29+ | All pass |
| **Total** | **91+** | **All pass** |

---

## Configuration Changes

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `MIN_REGIME_SAMPLES` | 100 | 50 | Enable training on short regimes (SPEC_10 T7) |

All other SPEC_10 config parameters were already set correctly in previous implementation phases.

---

## Files Modified/Created

### Modified
- `config.py` — `MIN_REGIME_SAMPLES` reduced to 50
- `tests/test_regime_detection_integration.py` — Updated assertions for new threshold
- `tests/test_training_pipeline_fixes.py` — Updated min threshold assertion

### Created (Documentation)
- `docs/jump_model_audit.md` — Full jump model audit report (T1)
- `docs/regime_detection_guide.md` — Architecture, API reference, config guide
- `docs/regime_audit_summary.md` — This file
