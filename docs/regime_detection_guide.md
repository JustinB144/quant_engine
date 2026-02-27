# Regime Detection Guide

> **Date:** 2026-02-27
> **Spec:** SPEC_10 — Regime Detection Upgrade

---

## Architecture Overview

The regime detection system classifies market conditions into one of four canonical regimes using an ensemble of three independent detectors, with confidence-weighted voting, uncertainty gating, and cross-sectional consensus analysis.

### Canonical Regimes

| Code | Name | Characteristics |
|------|------|-----------------|
| 0 | `trending_bull` | Positive returns, positive momentum, low volatility, uptrend |
| 1 | `trending_bear` | Negative returns, negative momentum, moderate volatility, downtrend |
| 2 | `mean_reverting` | Near-zero returns, low volatility, flat trend |
| 3 | `high_volatility` | Extreme returns, high volatility, elevated stress indicators |

### Detection Engines

1. **Rule-Based** (`_rule_detect`): Deterministic thresholds on Hurst exponent, ADX, volatility. Fast, interpretable, but lacks probabilistic output.

2. **Gaussian HMM** (`_hmm_detect`): Sticky-transition HMM with full covariance, BIC state selection (2-6 states), Viterbi decoding, and duration smoothing. Provides posterior probabilities.

3. **Statistical Jump Model** (`_jump_detect`): PyPI `jumpmodels` package with walk-forward CV for penalty selection. Explicitly penalizes regime transitions. Falls back to in-house implementation for short series.

4. **Ensemble** (`detect_ensemble`): Confidence-weighted voting across all three engines. Default weights: HMM=0.5, rule=0.3, jump=0.2. Calibrated via ECM when validation data is available.

### System Flow

```
Features → Observation Matrix (4-15 features)
                ↓
    ┌───────────┼───────────┐
    ↓           ↓           ↓
  Rule-Based   HMM      Jump Model
    ↓           ↓           ↓
    └───────────┼───────────┘
                ↓
    Confidence-Weighted Voting
                ↓
    Regime + Posterior Probabilities
                ↓
    ┌───────────┼───────────┐
    ↓           ↓           ↓
  Uncertainty  Sizing   Consensus
    Gate       Adjust   Analysis
```

---

## Key Components

### Observation Matrix (`build_hmm_observation_matrix`)

Constructs up to 15 features from the feature pipeline:

**Core (4, always available):**
- `ret_1d` — 1-day return
- `vol_20d` — 20-day realized volatility
- `natr` — Normalized Average True Range (14-bar)
- `trend` — 50-day SMA slope

**Extended (up to 7, graceful fallback):**
- `credit_spread_proxy` — GARCH/short-vol ratio
- `market_breadth` — Fraction of positive recent returns
- `vix_rank` — Volatility percentile (252-bar)
- `volume_regime` — Volume z-score
- `momentum_20d` — 20-day momentum
- `mean_reversion` — 0.5 - Hurst exponent
- `cross_correlation` — Lagged autocorrelation

**Structural (up to 4, when `REGIME_EXPANDED_FEATURES_ENABLED=True`):**
- `spectral_entropy` — Shannon entropy of power spectrum
- `ssa_trend_strength` — SSA first singular value variance fraction
- `bocpd_changepoint` — BOCPD changepoint probability (rolling 5-bar max)
- `jump_intensity` — Jump detection intensity

All features are z-scored (mean=0, std=1) before feeding to the HMM.

### Confidence Calibrator (`ConfidenceCalibrator`)

Implements Empirical Calibration Matrix (ECM) calibration:
- Bins confidence into 10 levels per (component, regime) pair
- Maps raw confidence to realized accuracy
- Corrects overconfident/underconfident components
- Component weights proportional to overall accuracy

**Usage:**
```python
detector = RegimeDetector()
weights = detector.calibrate_confidence_weights(features, actual_regimes)
# Subsequent calls to detect_ensemble() use calibrated weights
```

### Uncertainty Gate (`UncertaintyGate`)

Computes sizing multiplier from regime posterior entropy:
- Entropy = 0 (certain) → multiplier = 1.0 (full size)
- Entropy = 0.5 (moderate) → multiplier = 0.95
- Entropy = 1.0 (maximum) → multiplier = 0.85

When entropy exceeds the stress threshold (0.80), recommends assuming high_volatility regime for risk limits.

**Usage:**
```python
gate = UncertaintyGate()
multiplier = gate.compute_size_multiplier(uncertainty=0.6)
adjusted_weights = gate.apply_uncertainty_gate(weights, uncertainty=0.6)
should_stress = gate.should_assume_stress(uncertainty=0.9)
```

### Cross-Sectional Consensus (`RegimeConsensus`)

Measures agreement across securities about the current market regime:
- Consensus = max fraction of securities in any single regime
- High consensus (>= 80%): clear market-wide regime
- Early warning (< 60%): potential regime transition

Divergence detection fits a linear trend to consensus history. Falling consensus (slope < -0.01/day) signals regime instability.

**Usage:**
```python
rc = RegimeConsensus()
result = rc.compute_consensus(regime_per_security)
diverging, details = rc.detect_divergence(consensus_history)
warning, reason = rc.early_warning(consensus=0.55)
```

### Online Regime Updater (`OnlineRegimeUpdater`)

Incremental HMM state updates via the forward algorithm:
- O(K^2 * d) per observation vs O(T * K^2 * d) for full refit
- Approximately T-fold speedup for daily updates
- Full refit every 30 days (configurable)
- Per-security state probability cache

**Usage:**
```python
updater = OnlineRegimeUpdater(fitted_hmm_model)
regime, state_prob = updater.update_regime_for_security("AAPL", observation)
results = updater.update_batch(security_observations)
if updater.should_refit(last_refit_date):
    # Trigger full refit
```

### Shock Vector (`ShockVector`)

Unified, version-locked market state representation combining:
- HMM regime, confidence, uncertainty
- BOCPD changepoint probability and run-length
- Jump detection flag and magnitude
- Structural features (spectral entropy, SSA, etc.)

---

## Configuration Reference

All config constants are in `config.py` with `STATUS: ACTIVE` annotations.

### Core Detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `REGIME_MODEL_TYPE` | `"jump"` | Primary detection method |
| `REGIME_HMM_STATES` | `4` | Number of hidden states |
| `REGIME_HMM_STICKINESS` | `0.92` | Diagonal prior bias |
| `REGIME_MIN_DURATION` | `3` | Minimum regime duration (bars) |
| `REGIME_HMM_COVARIANCE_TYPE` | `"full"` | Covariance structure |
| `REGIME_HMM_AUTO_SELECT_STATES` | `True` | Use BIC to select optimal states |

### Ensemble Voting

| Parameter | Default | Description |
|-----------|---------|-------------|
| `REGIME_ENSEMBLE_ENABLED` | `True` | Enable ensemble voting |
| `REGIME_ENSEMBLE_DEFAULT_WEIGHTS` | `{hmm: 0.5, rule: 0.3, jump: 0.2}` | Component weights |
| `REGIME_ENSEMBLE_DISAGREEMENT_THRESHOLD` | `0.40` | Min vote share to confirm regime |
| `REGIME_ENSEMBLE_UNCERTAIN_FALLBACK` | `3` | Fallback regime on disagreement |

### Uncertainty Gating

| Parameter | Default | Description |
|-----------|---------|-------------|
| `REGIME_UNCERTAINTY_ENTROPY_THRESHOLD` | `0.50` | Uncertainty flag threshold |
| `REGIME_UNCERTAINTY_STRESS_THRESHOLD` | `0.80` | Assume stress above this |
| `REGIME_UNCERTAINTY_SIZING_MAP` | `{0.0: 1.0, 0.5: 0.95, 1.0: 0.85}` | Entropy → multiplier |
| `REGIME_UNCERTAINTY_MIN_MULTIPLIER` | `0.80` | Floor for sizing multiplier |

### Consensus

| Parameter | Default | Description |
|-----------|---------|-------------|
| `REGIME_CONSENSUS_THRESHOLD` | `0.80` | High consensus threshold |
| `REGIME_CONSENSUS_EARLY_WARNING` | `0.60` | Early warning threshold |
| `REGIME_CONSENSUS_DIVERGENCE_WINDOW` | `20` | Lookback for trend fitting |
| `REGIME_CONSENSUS_DIVERGENCE_SLOPE` | `-0.01` | Slope threshold for divergence |

### Online Updating

| Parameter | Default | Description |
|-----------|---------|-------------|
| `REGIME_ONLINE_UPDATE_ENABLED` | `True` | Enable incremental updates |
| `REGIME_ONLINE_REFIT_DAYS` | `30` | Days between full refits |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_REGIME_SAMPLES` | `50` | Minimum training samples per regime |
| `MIN_REGIME_DAYS` | `10` | Minimum days in regime |
| `REGIME_EXPANDED_FEATURES_ENABLED` | `True` | Include structural features |

---

## Interpreting Regime Output

### RegimeOutput

```python
@dataclass
class RegimeOutput:
    regime: pd.Series          # Integer regime label per bar
    confidence: pd.Series      # Confidence score [0, 1]
    probabilities: pd.DataFrame  # Posterior probabilities per regime
    transition_matrix: np.ndarray  # HMM transition matrix (K×K)
    model_type: str            # "hmm", "rule", "jump", or "ensemble"
    uncertainty: pd.Series     # Normalized entropy [0, 1]
```

### Decision Making

1. **High confidence (>0.7), low uncertainty (<0.3):** Trust the regime label. Use full position sizes.
2. **Moderate confidence (0.4-0.7), moderate uncertainty (0.3-0.5):** Regime is likely correct but hedging is prudent. Sizing reduced by 5%.
3. **Low confidence (<0.4), high uncertainty (>0.5):** Regime unclear. Sizing reduced by 5-15%. Consider using stress regime constraints.
4. **Very high uncertainty (>0.8):** Assume stress regime regardless of label. Apply tightest risk limits.

### Consensus Signals

- **Consensus > 80%:** Market-wide agreement on regime. Strong signal.
- **Consensus 60-80%:** Mixed signals. Some securities diverging.
- **Consensus < 60%:** Potential regime transition. Monitor closely.
- **Falling consensus (slope < -1%/day):** Active regime transition underway.
