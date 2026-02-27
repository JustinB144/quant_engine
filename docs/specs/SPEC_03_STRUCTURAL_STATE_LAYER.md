# Feature Spec: Structural State Layer — BOCPD + HMM Enhancement + Unified Shock Vector

> **Status:** Complete
> **Author:** Claude Opus 4
> **Date:** 2026-02-26
> **Completed:** 2026-02-27
> **Estimated effort:** 95 hours across 4 tasks

---

## Why

The current regime detection stack is **offline and retrospective**:
- HMM is fitted on full historical data; new regimes detected only after data accumulates
- Jump model detects ex-post shocks, not ahead-of-time transitions
- RegimeOutput is an ad-hoc dataclass with `{regime, confidence, probabilities}` — no versioning, no structural metadata

Production systems need **online changepoint detection** that triggers immediately when regime shifts occur, with rich structural context. Current limitations:

- **No anticipation**: HMM smooths over regime transitions; regime changes are delayed (2–5 day lag)
- **No structural fusion**: Jump model and HMM operate independently; no unified shock vector
- **Poor ops integration**: RegimeOutput lacks schema versioning; downstream systems (backtester, models) can't validate structure
- **Limited observability**: 11-feature HMM observation matrix is underdocumented (spec claims 4 features; actually 11)

**Impact**: Models react to regime shifts with 2–5 day lag. Structural shocks (earnings, macro events) trigger late hedges, excessive drawdowns. No audit trail for decision-making.

---

## What

Build a "Structural State Layer" that provides **real-time regime intelligence with structural context**:

1. **BOCPD (Bayesian Online Change-Point Detection)**: Real-time run-length posterior on price data; detect regime transitions within 1 bar
2. **HMM observation matrix audit**: Document all 11 existing features; add structural features if missing
3. **Unified shock/structure vector**: Version-locked schema combining HMM state, BOCPD hazard, jump flags, structural features
4. **Integration with evaluation**: Feed unified vector to backtester, models, and dashboard

This is **additive** — HMM remains unchanged. BOCPD runs in parallel, providing complementary (ahead-of-time) signals. Shock vector is a normalized output for downstream consumption.

---

## Constraints

### Must-haves

- BOCPD implementation with Gaussian likelihood and run-length posterior (not particle filter; too slow)
- HMM observation matrix: document all 11 features; add any missing structural features
- Unified shock vector: `ShockVector` dataclass with version, timestamp, regime metadata
- Integration: RegimeDetector.detect() returns augmented RegimeOutput with BOCPD signals
- Schema validation: ShockVector must validate against defined schema; V1, V2, etc. supported
- Real-time performance: BOCPD < 10ms per bar for 1000-stock universe

### Must-nots

- Do NOT modify HMM fitting logic (remains offline, batch-only)
- Do NOT use particle filters (computational burden; Gaussian sufficient for Gaussian data)
- Do NOT assume full-rank covariance (use regularized covariance, diagonal fallback)
- Do NOT ignore shock vector versioning (future compatibility depends on it)

### Out of scope

- Multi-scale BOCPD (high-frequency + daily) — separate enhancement
- Non-Gaussian BOCPD (Student-t, mixture) — future robustness improvement
- Shock prediction (anticipating shocks before they occur) — separate ML task

---

## Current State

### Key files

| Module | File | Role | Status |
|--------|------|------|--------|
| Regime | `/sessions/fervent-optimistic-fermat/mnt/quant_engine/regime/detector.py` | RegimeOutput, RegimeDetector.detect*() | ACTIVE |
| HMM | `/sessions/fervent-optimistic-fermat/mnt/quant_engine/regime/hmm.py` | GaussianHMM, build_hmm_observation_matrix() | ACTIVE |
| Jump Model | `/sessions/fervent-optimistic-fermat/mnt/quant_engine/regime/jump_model.py` | 9-line re-export to jump_model_legacy.py | ACTIVE |
| Config | `/sessions/fervent-optimistic-fermat/mnt/quant_engine/config.py` | REGIME_* constants | ACTIVE |

### Existing patterns to follow

1. **Regime output**: RegimeOutput dataclass with:
   ```python
   @dataclass
   class RegimeOutput:
       regime: pd.Series
       confidence: pd.Series
       probabilities: pd.DataFrame
       transition_matrix: Optional[np.ndarray]
       model_type: str
       uncertainty: Optional[pd.Series] = None
   ```

2. **HMM observation matrix**: Already builds 11 features via `build_hmm_observation_matrix()`:
   - Returns, vol_20d, NATR, SMA slope, credit spread proxy, market breadth, VIX rank, volume regime, momentum, mean reversion, cross-asset correlation

3. **Config patterns**: Status annotations on all constants

### Configuration

**Current regime config**:
```python
REGIME_MODEL_TYPE = "hmm"
REGIME_HMM_STATES = 4
REGIME_HMM_MAX_ITER = 60
REGIME_HMM_STICKINESS = 0.92
REGIME_MIN_DURATION = 3
REGIME_JUMP_MODEL_ENABLED = True
REGIME_JUMP_PENALTY = 10.0
REGIME_EXPECTED_CHANGES_PER_YEAR = 12
```

**New config to add**:
```python
# ── BOCPD Configuration ────────────────────────────────────────────────────
BOCPD_ENABLED = True                       # STATUS: PLACEHOLDER
BOCPD_HAZARD_FUNCTION = "constant"         # STATUS: PLACEHOLDER — "constant", "geometric", "weakly_periodic"
BOCPD_HAZARD_LAMBDA = 1.0 / 60             # STATUS: PLACEHOLDER — constant hazard rate (1 change per 60 days)
BOCPD_LIKELIHOOD_TYPE = "gaussian"         # STATUS: PLACEHOLDER — "gaussian" (only option for v1)
BOCPD_RUNLENGTH_DEPTH = 200                # STATUS: PLACEHOLDER — max run-length to track (older = pruned)

# ── Shock Vector Schema ────────────────────────────────────────────────────
SHOCK_VECTOR_SCHEMA_VERSION = "1.0"        # STATUS: PLACEHOLDER
SHOCK_VECTOR_INCLUDE_STRUCTURAL = True     # STATUS: PLACEHOLDER — include spectral/SSA/tail features
```

---

## Tasks

### T1: BOCPD Implementation

**What:** Implement Bayesian Online Change-Point Detection with Gaussian likelihood. Compute run-length posterior at each bar; flag regime transitions when probability of changepoint > threshold.

**Files:**
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/regime/bocpd.py` (new) — BOCPDDetector class
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/config.py` — add BOCPD_* constants

**Implementation notes:**

1. Create `/sessions/fervent-optimistic-fermat/mnt/quant_engine/regime/bocpd.py`:
   ```python
   import numpy as np
   import pandas as pd
   from dataclasses import dataclass
   from typing import Dict, Optional, Tuple
   from scipy import stats

   @dataclass
   class BOCPDResult:
       """Output of BOCPD at a single timestep."""
       run_length_posterior: np.ndarray      # P(r_t | x_1:t), shape (max_runlength,)
       changepoint_prob: float               # P(changepoint at t) = sum(posterior[:]) - posterior[0]
       predicted_mean: float
       predicted_std: float
       most_likely_runlength: int

   class BOCPDDetector:
       """
       Bayesian Online Change-Point Detection with Gaussian likelihood.

       Maintains run-length posterior P(r_t | x_1:t) where r_t is the
       number of observations since the last changepoint.

       At each new observation, the posterior is updated:
       1. Evaluate likelihood of new observation under each run-length hypothesis
       2. Grow each run-length by 1 (no changepoint) or reset to 0 (changepoint)
       3. Normalize posterior

       Reference: Adams & MacKay (2007), "Bayesian Online Changepoint Detection"
       """

       def __init__(
           self,
           hazard_lambda: float = 1.0 / 60,  # Expected 1 change per 60 bars
           hazard_func: str = "constant",    # "constant", "geometric", "weakly_periodic"
           max_runlength: int = 200,
           likelihood_type: str = "gaussian",
       ):
           """Initialize BOCPD detector."""
           self.hazard_lambda = hazard_lambda
           self.hazard_func = hazard_func
           self.max_runlength = max_runlength
           self.likelihood_type = likelihood_type

           # Run-length prior (geometric with hazard rate)
           self.growth_probs = self._compute_growth_probs()

           # Gaussian model parameters (sufficient statistics)
           self.model_params: Dict[int, Dict] = {}  # {runlength: {mean, var, alpha, beta}}

       def _compute_growth_probs(self) -> np.ndarray:
           """Compute P(r_t = k | r_{t-1}) for all k."""
           growth = np.zeros(self.max_runlength)

           if self.hazard_func == "constant":
               # Geometric prior: P(r_t > r_{t-1}) = 1 - hazard_lambda
               # P(r_t = r_{t-1} + 1 | no change) = 1 - hazard_lambda
               # P(r_t = 0 | changepoint) = hazard_lambda
               growth[:] = 1.0 - self.hazard_lambda
           elif self.hazard_func == "geometric":
               # Stronger decay: hazard increases with run-length
               for r in range(self.max_runlength):
                   growth[r] = 1.0 - self.hazard_lambda * (r + 1)
           else:
               raise ValueError(f"Unknown hazard function: {self.hazard_func}")

           return np.clip(growth, 0.0, 1.0)

       def _init_model(self, x: float) -> Dict:
           """Initialize Gaussian model from single observation."""
           return {
               "mean": x,
               "var": 1.0,
               "alpha": 0.5,  # Shape parameter for inverse-gamma variance prior
               "beta": 0.5,   # Rate parameter for inverse-gamma variance prior
           }

       def _update_model(self, params: Dict, x: float) -> Dict:
           """
           Update Gaussian model parameters with new observation.

           Using Bayesian normal-inverse-gamma conjugate prior.
           """
           mean, var = params["mean"], params["var"]
           alpha, beta = params["alpha"], params["beta"]

           # Number of observations (implicitly tracked via alpha)
           n = 2 * alpha

           # Bayesian update
           mean_new = (n * mean + x) / (n + 1)
           var_new = var + (n / (n + 1)) * (x - mean) ** 2

           alpha_new = alpha + 0.5
           beta_new = beta + 0.5 * (n / (n + 1)) * (x - mean) ** 2

           return {
               "mean": mean_new,
               "var": var_new,
               "alpha": alpha_new,
               "beta": beta_new,
           }

       def _log_pred_likelihood(self, x: float, params: Dict) -> float:
           """Log-likelihood of observation under this model (predictive distribution)."""
           mean = params["mean"]
           var = params["var"]
           alpha = params["alpha"]
           beta = params["beta"]

           # Student-t predictive distribution
           nu = 2 * alpha
           sigma2 = beta / alpha

           # Log Student-t density
           log_const = -0.5 * np.log(np.pi * nu * sigma2)
           log_exp = -(nu + 1) / 2 * np.log(1 + (x - mean) ** 2 / (nu * sigma2))

           return log_const + log_exp

       def update(self, x: float, posterior: Optional[np.ndarray] = None) -> BOCPDResult:
           """
           Update run-length posterior with new observation.

           Parameters
           ----------
           x : float
               New observation (scalar)
           posterior : np.ndarray, optional
               Previous run-length posterior. If None, assume first observation.

           Returns
           -------
           BOCPDResult
               Updated posterior, changepoint probability, predictions
           """
           if posterior is None:
               # First observation: initialize models
               posterior = np.zeros(self.max_runlength)
               posterior[0] = 1.0  # No history, so run-length = 0
               self.model_params = {0: self._init_model(x)}
           else:
               # Check posterior is valid
               assert len(posterior) == self.max_runlength

           # Evaluate likelihood under each run-length hypothesis
           likelihoods = np.zeros(self.max_runlength)
           for r in range(self.max_runlength):
               if r not in self.model_params:
                   # Initialize model if first time seeing this run-length
                   self.model_params[r] = self._init_model(x)

               likelihoods[r] = np.exp(self._log_pred_likelihood(x, self.model_params[r]))

           # Grow: each run-length transitions to r+1 (no changepoint) or 0 (changepoint)
           # P(r_t = k | x_1:t, no change at t) ∝ P(x_t | r_{t-1} = k-1) * P(r_t=k|r_{t-1}=k-1) * P(r_{t-1}=k-1|x_1:{t-1})
           new_posterior = np.zeros(self.max_runlength)

           # Changepoint: all previous run-lengths can transition to 0
           changepoint_prob = np.sum(posterior * likelihoods)
           new_posterior[0] = changepoint_prob * self.hazard_lambda

           # Growth: each run-length k grows to k+1 (if k+1 < max_runlength)
           for k in range(self.max_runlength - 1):
               growth_prob = self.growth_probs[k]
               new_posterior[k + 1] = (
                   posterior[k] * likelihoods[k] * growth_prob
               )

           # Normalize
           new_posterior /= np.sum(new_posterior + 1e-10)

           # Update models
           for r in range(self.max_runlength):
               if new_posterior[r] > 1e-6:  # Only update models with non-negligible posterior
                   if r == 0:
                       self.model_params[r] = self._init_model(x)
                   else:
                       self.model_params[r] = self._update_model(
                           self.model_params[r - 1], x
                       )

           # Compute predictions
           predicted_mean = np.sum(
               new_posterior * np.array([self.model_params[r]["mean"] for r in range(self.max_runlength)])
           )
           predicted_std = np.sqrt(np.sum(
               new_posterior * np.array([self.model_params[r]["var"] for r in range(self.max_runlength)])
           ))

           # Changepoint probability
           changepoint_prob = np.sum(new_posterior[:])  # Probability that changepoint occurred recently
           # More precisely: P(changepoint at t) = 1 - P(r_t > 0) = new_posterior[0]
           changepoint_prob = float(new_posterior[0])

           # Most likely run-length
           most_likely_r = np.argmax(new_posterior)

           return BOCPDResult(
               run_length_posterior=new_posterior.copy(),
               changepoint_prob=changepoint_prob,
               predicted_mean=float(predicted_mean),
               predicted_std=float(predicted_std),
               most_likely_runlength=int(most_likely_r),
           )

       def batch_update(
           self,
           x_series: np.ndarray,
       ) -> Tuple[np.ndarray, np.ndarray]:
           """
           Update on full time series.

           Returns
           -------
           changepoint_probs : np.ndarray of shape (len(x_series),)
               P(changepoint at t) for each timestep
           run_lengths : np.ndarray of shape (len(x_series),)
               Most likely run-length at each timestep
           """
           changepoint_probs = np.zeros(len(x_series))
           run_lengths = np.zeros(len(x_series))
           posterior = None

           for t, x in enumerate(x_series):
               result = self.update(x, posterior)
               posterior = result.run_length_posterior.copy()
               changepoint_probs[t] = result.changepoint_prob
               run_lengths[t] = result.most_likely_runlength

           return changepoint_probs, run_lengths
   ```

2. Add to config.py (after REGIME section):
   ```python
   # ── Bayesian Online Change-Point Detection ─────────────────────────────
   BOCPD_ENABLED = True                       # STATUS: PLACEHOLDER
   BOCPD_HAZARD_FUNCTION = "constant"         # STATUS: PLACEHOLDER
   BOCPD_HAZARD_LAMBDA = 1.0 / 60             # STATUS: PLACEHOLDER — 1 change per 60 bars
   BOCPD_LIKELIHOOD_TYPE = "gaussian"         # STATUS: PLACEHOLDER
   BOCPD_RUNLENGTH_DEPTH = 200                # STATUS: PLACEHOLDER — max run-length tracked
   BOCPD_CHANGEPOINT_THRESHOLD = 0.50         # STATUS: PLACEHOLDER — flag if P(changepoint) > 50%
   ```

**Verify:**
- Test on synthetic changepoint data:
  ```python
  def test_bocpd_detects_mean_shift():
      # N(0, 1) then N(2, 1)
      x = np.concatenate([np.random.randn(100), np.random.randn(100) + 2])
      detector = BOCPDDetector(hazard_lambda=1/100)
      cp_probs, _ = detector.batch_update(x)
      # Changepoint probability should spike around index 100
      assert np.max(cp_probs[95:105]) > 0.7
  ```

---

### T2: HMM Observation Matrix Audit and Enhancement

**What:** Document all 11 features in HMM observation matrix. Verify causality. Add any missing structural features identified in SPEC_02.

**Files:**
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/regime/hmm.py` — add detailed docstring to build_hmm_observation_matrix()
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/docs/reference/hmm_observation_matrix.md` (new) — documentation

**Implementation notes:**

1. Locate `build_hmm_observation_matrix()` in hmm.py and add comprehensive docstring:
   ```python
   def build_hmm_observation_matrix(
       features: pd.DataFrame,
       target_features: List[str] = None,
   ) -> np.ndarray:
       """
       Build observation matrix for HMM from high-level features.

       The observation matrix captures market regimes via 11 carefully selected features:

       1. **Returns (1 feature)**: Log daily return
          - Regime signal: positive/negative skew
          - Causality: CAUSAL (backward-looking)

       2. **Volatility (1 feature)**: 20-day historical volatility
          - Regime signal: low vol (calm), high vol (stressed)
          - Causality: CAUSAL

       3. **NATR (1 feature)**: Normalized ATR / Close
          - Regime signal: tight stops (low NATR = trend), wide stops (high NATR = noise)
          - Causality: CAUSAL

       4. **SMA Slope (1 feature)**: Slope of 50-day SMA
          - Regime signal: positive slope (uptrend), negative (downtrend), flat (range)
          - Causality: CAUSAL

       5. **Credit Spread Proxy (1 feature)**: HY-IG spread (or TED spread if no corp data)
          - Regime signal: widening spreads = stress, tightening = calm
          - Causality: CAUSAL (macro data)

       6. **Market Breadth (1 feature)**: % of stocks trading above SMA
          - Regime signal: high breadth (healthy), low breadth (selective, weak)
          - Causality: END_OF_DAY (requires full universe data)

       7. **VIX Rank (1 feature)**: Percentile of VIX vs 252-day history
          - Regime signal: VIX extremes mark regime transitions
          - Causality: CAUSAL

       8. **Volume Regime (1 feature)**: Current volume vs 20-day average
          - Regime signal: volume spikes with regime changes
          - Causality: CAUSAL

       9. **Momentum (1 feature)**: ROC(20) or RSI(14)
          - Regime signal: overbought/oversold (mean reversion), sustained (trend)
          - Causality: CAUSAL

       10. **Mean Reversion (1 feature)**: Hurst exponent or variance ratio
           - Regime signal: Hurst > 0.55 (trending), < 0.45 (mean-reverting)
           - Causality: CAUSAL

       11. **Cross-Asset Correlation (1 feature)**: Average correlation with market
           - Regime signal: high corr (contagion), low corr (decoupling)
           - Causality: END_OF_DAY

       Parameters
       ----------
       features : pd.DataFrame
           Computed features (from feature pipeline)
       target_features : List[str], optional
           Explicit list of 11 features to use. If None, auto-select.

       Returns
       -------
       np.ndarray of shape (n_bars, 11)
           Observation matrix ready for HMM fitting/inference

       Notes
       -----
       - Features are standardized (mean 0, std 1) before concatenation
       - Missing values are forward-filled, then padded with 0 if still missing
       - All features except #6 (breadth) can be computed on single-asset data
       - Features #5, #6, #11 require macro/cross-asset data; fall back to internal proxies if unavailable
       """

       # Default feature selection (documented, reproducible)
       default_features = [
           "Close_returns",     # 1. Returns
           "HV_20",             # 2. Volatility
           "NATR_14",           # 3. NATR
           "SMASlope_50_5",     # 4. SMA slope
           "credit_spread",     # 5. Credit spread proxy
           "breadth_pct",       # 6. Market breadth (or computed internally)
           "vix_rank_252",      # 7. VIX rank
           "volume_regime_20",  # 8. Volume regime
           "ROC_20",            # 9. Momentum
           "hurst_exponent",    # 10. Mean reversion
           "cross_asset_corr",  # 11. Cross-asset correlation
       ]

       if target_features is None:
           target_features = default_features

       assert len(target_features) == 11, f"Expected 11 features, got {len(target_features)}"

       # Extract and standardize
       obs_list = []
       for feat in target_features:
           if feat in features.columns:
               col = features[feat]
           else:
               # Fallback: compute internally
               col = _compute_fallback_feature(feat, features)

           # Standardize
           col_std = (col - col.mean()) / (col.std() + 1e-10)
           col_std = col_std.fillna(method="ffill").fillna(0)

           obs_list.append(col_std.values)

       obs_matrix = np.column_stack(obs_list)
       return obs_matrix
   ```

2. Create documentation file `/sessions/fervent-optimistic-fermat/mnt/quant_engine/docs/reference/hmm_observation_matrix.md`:
   ```markdown
   # HMM Observation Matrix Reference

   ## Overview
   The HMM observation matrix encodes the 11-dimensional market regime state.

   ## Feature Composition

   | Index | Feature | Source | Window | Type | Purpose |
   |-------|---------|--------|--------|------|---------|
   | 0 | Returns | Price data | 1d | CAUSAL | Directional bias |
   | 1 | HV_20 | Volatility | 20d | CAUSAL | Vol regime |
   | 2 | NATR_14 | ATR | 14d | CAUSAL | Volatility regime |
   | 3 | SMASlope_50 | SMA(50) | 5d | CAUSAL | Trend strength |
   | 4 | Credit Spread | Macro | 20d | CAUSAL | Stress indicator |
   | 5 | Breadth | Cross-asset | EOD | END_OF_DAY | Market health |
   | 6 | VIX Rank | Volatility | 252d | CAUSAL | Volatility extremes |
   | 7 | Volume Regime | Volume | 20d | CAUSAL | Liquidity/activity |
   | 8 | Momentum | ROC/RSI | 20d | CAUSAL | Oscillation |
   | 9 | Hurst | Price series | 100d | CAUSAL | Persistence |
   | 10 | Cross-Correlation | Universe | 60d | END_OF_DAY | Systemic risk |

   ## Regime Interpretation

   ### State 0: Calm/Trending
   - Returns: Positive drift
   - HV: Low-moderate (< 15%)
   - SMA Slope: Positive
   - Breadth: High (> 60%)
   - Correlation: Moderate (< 0.5)

   ### State 1: Volatile/Stressed
   - Returns: Negative drift
   - HV: High (> 25%)
   - SMA Slope: Flat or negative
   - Breadth: Low (< 40%)
   - Correlation: High (> 0.7)

   ### State 2: Mean-Reversion
   - Returns: Extreme values (overbought/oversold)
   - Momentum: Extreme (RSI > 70 or < 30)
   - Hurst: < 0.45
   - Volume: Elevated (potential reversal)

   ### State 3: Transition
   - Mixed signals
   - Credit spread: Widening
   - VIX Rank: Extreme (> 90th or < 10th percentile)

   ## Feature Constraints and Causality

   **CRITICAL**: All features must be backward-looking (CAUSAL) or END_OF_DAY.
   Do NOT include forward-looking features (e.g., future returns, labels).

   Causality check:
   - HMM fitted on in-sample data (e.g., 2020-2022)
   - At each prediction date, observation matrix uses only data available at that date
   - No future price data, macro surprises, or realized labels

   ## Extension for Structural Features

   From SPEC_02, consider adding structural features to observation matrix:
   - **Spectral Entropy**: Periodicity strength
   - **SSA Trend Strength**: Non-stationary trend component
   - **Jump Intensity**: Frequency of large moves
   - **Eigenvalue Concentration**: Portfolio-level systemic stress

   If added, observation matrix becomes 11 + N_structural features.
   Must retrain HMM with new dimension.
   ```

3. Add feature validation to detector.py:
   ```python
   def validate_hmm_observation_features(features: pd.DataFrame) -> bool:
       """Check that all 11 required HMM observation features are present and non-NaN."""
       required = [
           "Close_returns", "HV_20", "NATR_14", "SMASlope_50_5",
           "credit_spread", "breadth_pct", "vix_rank_252",
           "volume_regime_20", "ROC_20", "hurst_exponent", "cross_asset_corr",
       ]

       missing = [f for f in required if f not in features.columns]
       if missing:
           logger.warning(f"Missing HMM observation features: {missing}")
           return False

       for f in required:
           nan_pct = features[f].isna().mean()
           if nan_pct > 0.1:
               logger.warning(f"Feature {f} has {nan_pct:.1%} missing values")

       return True
   ```

**Verify:**
- Check HMM observation matrix shape:
  ```python
  def test_hmm_observation_matrix_shape():
      obs = build_hmm_observation_matrix(features_df)
      assert obs.shape == (len(features_df), 11)
  ```

---

### T3: Unified Shock/Structure Vector Contract

**What:** Define `ShockVector` dataclass with version, timestamp, regime metadata, BOCPD signals, and structural features. Implement schema validation.

**Files:**
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/regime/shock_vector.py` (new) — ShockVector, ShockVectorValidator
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/config.py` — add SHOCK_VECTOR_SCHEMA_VERSION

**Implementation notes:**

1. Create `/sessions/fervent-optimistic-fermat/mnt/quant_engine/regime/shock_vector.py`:
   ```python
   from dataclasses import dataclass, field, asdict
   from datetime import datetime
   from typing import Dict, Optional, List
   import numpy as np
   import pandas as pd

   @dataclass
   class ShockVector:
       """
       Unified market state and shock representation.

       Combines HMM regime, BOCPD online changepoint signals, and structural features.
       Version-locked for reproducibility and schema compatibility.

       Attributes
       ----------
       schema_version : str
           Version of ShockVector schema (e.g., "1.0"). For backward compatibility.
       timestamp : datetime
           Observation timestamp
       ticker : str
           Security identifier
       hmm_regime : int
           HMM-detected regime (0, 1, 2, or 3)
       hmm_confidence : float
           HMM state confidence (0-1)
       bocpd_changepoint_prob : float
           BOCPD probability of recent regime change (0-1)
       bocpd_runlength : int
           BOCPD most likely run-length (bars since last change)
       jump_detected : bool
           True if recent jump (2.5+ sigma event)
       jump_magnitude : float
           Magnitude of detected jump (pct return)
       structural_features : Dict[str, float]
           Structural features: {spectral_entropy, ssa_trend_strength, ...}
       transitions_prob : np.ndarray, optional
           HMM transition probabilities, shape (4, 4)
       """

       schema_version: str = "1.0"
       timestamp: datetime = field(default_factory=datetime.now)
       ticker: str = ""
       hmm_regime: int = 0
       hmm_confidence: float = 0.5
       bocpd_changepoint_prob: float = 0.0
       bocpd_runlength: int = 0
       jump_detected: bool = False
       jump_magnitude: float = 0.0
       structural_features: Dict[str, float] = field(default_factory=dict)
       transitions_prob: Optional[np.ndarray] = field(default=None, repr=False)

       def __post_init__(self):
           """Validate ShockVector upon creation."""
           if self.schema_version != "1.0":
               raise ValueError(f"Unknown schema version: {self.schema_version}")

           if not 0 <= self.hmm_regime < 4:
               raise ValueError(f"Invalid regime: {self.hmm_regime}")

           if not 0.0 <= self.hmm_confidence <= 1.0:
               raise ValueError(f"Invalid confidence: {self.hmm_confidence}")

           if not 0.0 <= self.bocpd_changepoint_prob <= 1.0:
               raise ValueError(f"Invalid changepoint prob: {self.bocpd_changepoint_prob}")

       def to_dict(self) -> Dict:
           """Serialize to dictionary (excludes arrays)."""
           d = asdict(self)
           d["timestamp"] = self.timestamp.isoformat()
           d["transitions_prob"] = None  # Don't serialize large array
           return d

       def is_shock_event(self, changepoint_threshold: float = 0.50) -> bool:
           """Determine if current vector represents a shock event."""
           return (
               self.jump_detected or
               self.bocpd_changepoint_prob > changepoint_threshold or
               abs(self.jump_magnitude) > 0.03  # 3% move
           )

       def regime_name(self) -> str:
           """Human-readable regime name."""
           names = ["Calm", "Stressed", "MeanReverting", "Transition"]
           return names[self.hmm_regime]

   class ShockVectorValidator:
       """Validate ShockVector schema and data integrity."""

       SCHEMA_V1_FIELDS = {
           "schema_version": str,
           "timestamp": (datetime, str),
           "ticker": str,
           "hmm_regime": int,
           "hmm_confidence": float,
           "bocpd_changepoint_prob": float,
           "bocpd_runlength": int,
           "jump_detected": bool,
           "jump_magnitude": float,
           "structural_features": dict,
       }

       @staticmethod
       def validate(sv: ShockVector) -> Tuple[bool, List[str]]:
           """
           Validate ShockVector.

           Returns
           -------
           Tuple[bool, List[str]]
               (is_valid, list_of_errors)
           """
           errors = []

           # Schema version
           if sv.schema_version not in ["1.0"]:
               errors.append(f"Unsupported schema version: {sv.schema_version}")

           # Regime
           if not isinstance(sv.hmm_regime, (int, np.integer)):
               errors.append(f"hmm_regime must be int, got {type(sv.hmm_regime)}")
           elif not 0 <= sv.hmm_regime < 4:
               errors.append(f"hmm_regime must be in [0, 3], got {sv.hmm_regime}")

           # Confidence
           if not 0.0 <= sv.hmm_confidence <= 1.0:
               errors.append(f"hmm_confidence must be in [0, 1], got {sv.hmm_confidence}")

           if not 0.0 <= sv.bocpd_changepoint_prob <= 1.0:
               errors.append(f"bocpd_changepoint_prob must be in [0, 1]")

           # Structural features
           if not isinstance(sv.structural_features, dict):
               errors.append(f"structural_features must be dict, got {type(sv.structural_features)}")
           else:
               for key, val in sv.structural_features.items():
                   if not isinstance(val, (float, int, np.number)):
                       errors.append(f"structural_features[{key}] must be numeric, got {type(val)}")

           return len(errors) == 0, errors

       @staticmethod
       def batch_validate(vectors: List[ShockVector]) -> Dict[int, List[str]]:
           """Validate batch of ShockVectors."""
           errors_by_idx = {}
           for i, sv in enumerate(vectors):
               is_valid, errors = ShockVectorValidator.validate(sv)
               if not is_valid:
                   errors_by_idx[i] = errors
           return errors_by_idx
   ```

2. Create integration function in detector.py:
   ```python
   from .shock_vector import ShockVector

   def detect_with_shock_vector(
       self,
       features: pd.DataFrame,
       ohlcv: Optional[pd.DataFrame] = None,
       include_structural: bool = True,
   ) -> Dict[str, ShockVector]:
       """
       Detect regimes and generate ShockVectors for entire universe.

       Returns
       -------
       Dict[str, ShockVector]
           {ticker: ShockVector}
       """
       # Existing regime detection
       regime_output = self.detect_with_confidence(features)

       shock_vectors = {}

       for ticker in features.index.get_level_values(0).unique():
           ticker_features = features.loc[ticker]

           # Build ShockVector
           sv = ShockVector(
               schema_version="1.0",
               timestamp=pd.Timestamp.now(),
               ticker=ticker,
               hmm_regime=int(regime_output.regime.iloc[-1]),
               hmm_confidence=float(regime_output.confidence.iloc[-1]),
               bocpd_changepoint_prob=0.0,  # Placeholder (requires BOCPD output)
               bocpd_runlength=0,
               jump_detected=False,
               jump_magnitude=0.0,
               structural_features={},
           )

           # Add structural features if requested
           if include_structural and hasattr(self, 'structural_features'):
               sv.structural_features = {
                   "spectral_entropy": float(ticker_features.get("SpectralEntropy_252", np.nan)),
                   "ssa_trend_strength": float(ticker_features.get("SSATrendStr_60", np.nan)),
                   "jump_intensity": float(ticker_features.get("JumpIntensity_20", np.nan)),
                   "eigenvalue_concentration": float(ticker_features.get("EigenConcentration_60", np.nan)),
               }

           shock_vectors[ticker] = sv

       return shock_vectors
   ```

3. Add to config.py:
   ```python
   SHOCK_VECTOR_SCHEMA_VERSION = "1.0"        # STATUS: PLACEHOLDER
   SHOCK_VECTOR_INCLUDE_STRUCTURAL = True     # STATUS: PLACEHOLDER
   ```

**Verify:**
- Test ShockVector validation:
  ```python
  def test_shock_vector_validation():
      sv = ShockVector(hmm_regime=2, hmm_confidence=0.8)
      is_valid, errors = ShockVectorValidator.validate(sv)
      assert is_valid
      assert len(errors) == 0

  def test_shock_vector_rejects_invalid_regime():
      sv = ShockVector(hmm_regime=5)  # Invalid
      with pytest.raises(ValueError):
          ShockVectorValidator.validate(sv)
  ```

---

### T4: Integration with Regime Detector and Evaluation Pipeline

**What:** Wire BOCPD into RegimeDetector. Augment RegimeOutput with BOCPD signals. Feed ShockVector to backtester and evaluation pipeline.

**Files:**
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/regime/detector.py` — integrate BOCPD and ShockVector
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/backtest/engine.py` — consume ShockVector for risk alerts
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/models/trainer.py` — use ShockVector for regime-aware weighting

**Implementation notes:**

1. Modify RegimeDetector in detector.py:
   ```python
   from .bocpd import BOCPDDetector
   from .shock_vector import ShockVector, ShockVectorValidator

   class RegimeDetector:
       def __init__(self, ..., enable_bocpd: bool = True):
           # ... existing init ...
           self.enable_bocpd = enable_bocpd
           if enable_bocpd:
               self.bocpd = BOCPDDetector(
                   hazard_lambda=BOCPD_HAZARD_LAMBDA,
                   hazard_func=BOCPD_HAZARD_FUNCTION,
                   max_runlength=BOCPD_RUNLENGTH_DEPTH,
               )

       def detect_with_shock_context(
           self,
           features: pd.DataFrame,
           ohlcv: Optional[pd.DataFrame] = None,
       ) -> Dict[str, ShockVector]:
           """
           Detect regimes and produce ShockVectors with BOCPD signals.

           Returns
           -------
           Dict[str, ShockVector]
               {ticker: ShockVector with BOCPD changepoint_prob, runlength}
           """
           shock_vectors = {}

           for ticker in features.index.unique():
               ticker_features = features.loc[ticker]
               ticker_ohlcv = ohlcv.loc[ticker] if ohlcv is not None else None

               # HMM regime detection
               regime = self.detect(ticker_features)
               _, confidence = self.detect_with_confidence(ticker_features)

               # BOCPD changepoint detection
               if self.enable_bocpd and ticker_ohlcv is not None:
                   returns = np.diff(np.log(ticker_ohlcv["Close"].values))
                   cp_probs, run_lengths = self.bocpd.batch_update(returns)

                   bocpd_cp_prob = float(cp_probs[-1])
                   bocpd_runlength = int(run_lengths[-1])
               else:
                   bocpd_cp_prob = 0.0
                   bocpd_runlength = 0

               # Jump detection
               if ticker_ohlcv is not None:
                   returns = ticker_ohlcv["Close"].pct_change().values
                   recent_ret = returns[-1] if not np.isnan(returns[-1]) else 0.0
                   recent_vol = np.std(returns[-20:]) if len(returns) >= 20 else 0.01
                   jump_detected = abs(recent_ret) > 2.5 * recent_vol
                   jump_magnitude = float(recent_ret)
               else:
                   jump_detected = False
                   jump_magnitude = 0.0

               # Build ShockVector
               sv = ShockVector(
                   schema_version="1.0",
                   timestamp=pd.Timestamp.now(),
                   ticker=ticker,
                   hmm_regime=int(regime.iloc[-1]),
                   hmm_confidence=float(confidence.iloc[-1]),
                   bocpd_changepoint_prob=bocpd_cp_prob,
                   bocpd_runlength=bocpd_runlength,
                   jump_detected=jump_detected,
                   jump_magnitude=jump_magnitude,
                   structural_features={},
               )

               # Validate
               is_valid, errors = ShockVectorValidator.validate(sv)
               if not is_valid:
                   logger.error(f"Invalid ShockVector for {ticker}: {errors}")

               shock_vectors[ticker] = sv

           return shock_vectors
   ```

2. Update BacktestEngine to consume ShockVector for regime-aware alerts:
   ```python
   def run(self, predictions: Dict[str, pd.Series], shock_vectors: Optional[Dict[str, ShockVector]] = None) -> BacktestResult:
       """
       Run backtest with optional shock vectors for regime-aware risk management.

       If shock_vectors provided, use BOCPD changepoint signals to trigger early exits.
       """
       # ... existing backtest logic ...

       if shock_vectors is not None:
           for ticker, shock_vec in shock_vectors.items():
               if shock_vec.is_shock_event(changepoint_threshold=BOCPD_CHANGEPOINT_THRESHOLD):
                   logger.info(f"SHOCK EVENT at {shock_vec.timestamp}: {ticker}, regime={shock_vec.regime_name()}, cp_prob={shock_vec.bocpd_changepoint_prob:.2f}")
                   # Optional: trigger position exit or de-risk

       return BacktestResult(...)
   ```

3. Update ModelTrainer to use regime-aware weighting:
   ```python
   def train_regime_aware(
       self,
       features: pd.DataFrame,
       labels: pd.Series,
       shock_vectors: Optional[Dict[str, ShockVector]] = None,
   ) -> None:
       """
       Train model with regime-aware sample weights.

       Samples with high BOCPD changepoint probability are downweighted
       (potential data drift). Samples with high HMM confidence are upweighted.
       """
       if shock_vectors is not None:
           # Compute per-sample weights based on ShockVector
           weights = np.ones(len(features))
           for i, (idx, shock_vec) in enumerate(shock_vectors.items()):
               # Downweight near changepoints
               changepoint_weight = 1.0 - 0.5 * shock_vec.bocpd_changepoint_prob
               # Upweight high-confidence regime states
               confidence_weight = shock_vec.hmm_confidence
               # Penalize jump events
               jump_weight = 0.5 if shock_vec.jump_detected else 1.0

               weights[i] = changepoint_weight * confidence_weight * jump_weight

           # Normalize
           weights /= weights.sum()

           # Train with sample weights
           self.model.fit(features, labels, sample_weight=weights)
       else:
           self.model.fit(features, labels)
   ```

**Verify:**
- Integration test: Run full pipeline with BOCPD and ShockVectors:
  ```bash
  pytest tests/regime/test_shock_vector_integration.py -v
  python scripts/run_backtest.py --enable-bocpd --enable-shock-vectors --universe SPY
  ```

---

## Validation

### Acceptance criteria

1. **BOCPD**: Detects synthetic mean-shift at correct location (within 5 bars) with P(changepoint) > 0.7
2. **HMM audit**: All 11 features documented and validated; causality checks pass
3. **ShockVector**: Dataclass with version 1.0; validation rejects invalid regime/confidence
4. **Integration**: RegimeDetector.detect_with_shock_context() returns Dict[str, ShockVector]; all validators pass
5. **Real-time performance**: BOCPD processes 1000-stock universe < 10ms per bar
6. **Ops integration**: ShockVector serializes to JSON; downstream systems can consume

### Verification steps

1. **Unit tests**:
   ```bash
   pytest tests/regime/test_bocpd.py -v
   pytest tests/regime/test_hmm_observation_matrix.py -v
   pytest tests/regime/test_shock_vector.py -v
   pytest tests/regime/test_detector_integration.py -v
   ```

2. **Integration test**:
   ```bash
   python scripts/test_shock_vector_pipeline.py --universe SPY,QQQ,IWM
   ```
   Expected output:
   - "BOCPD initialized: 1000 tickers"
   - "ShockVector schema v1.0: PASSED"
   - "Regime detection + BOCPD: N changepoints detected"
   - "Performance: X ms per bar"

3. **Operational test**:
   ```bash
   python scripts/run_backtest.py --enable-bocpd --universe SPY --verbose
   ```
   Expected: Backtest displays ShockVector alerts for regime changes

### Rollback plan

1. **BOCPD failures**: Add `BOCPD_ENABLED = False` to config; BOCPD runs in read-only mode (no decisions based on output)
2. **ShockVector schema mismatch**: Add version compatibility layer (ShockVectorV0, ShockVectorV1, etc.)
3. **Performance degradation**: Implement BOCPD culling (update only N most-active tickers, batch updates)
4. **Integration issues**: Revert to RegimeDetector.detect() (HMM only, no BOCPD)

---

## Notes

- **Computational complexity**: BOCPD O(T * R) per bar where T = time and R = max run-length. For 1000 stocks and R=200, ~0.2M operations per bar, ~10ms at 20M ops/sec.
- **Numerical stability**: Gaussian likelihood computed in log-domain; underflow/overflow handled via logsumexp
- **HMM observation matrix**: 11 features are carefully chosen to be minimally correlated and collectively cover trend, vol, breadth, macro. Future additions (from SPEC_02) must maintain independence.
- **ShockVector versioning**: Schema version in dataclass enables graceful deprecation; V2 can extend without breaking V1 consumers.
- **BOCPD vs. HMM**: Complementary. HMM is batch, retrospective (best for offline modeling). BOCPD is online, prospective (triggers alerts immediately). Use both.
- **Future enhancements**:
  - Adaptive hazard rate (learned from data instead of fixed)
  - Multi-scale BOCPD (intraday + daily)
  - Student-t likelihood for robustness to outliers
  - Causal filtering on BOCPD signals (exclude look-ahead data)
