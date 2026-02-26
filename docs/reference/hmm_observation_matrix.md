# HMM Observation Matrix Reference

> **Source:** `regime/hmm.py :: build_hmm_observation_matrix()`
> **Last Updated:** 2026-02-25
> **Spec:** SPEC_03 Structural State Layer

---

## Overview

The HMM observation matrix encodes the multi-dimensional market regime state used by `GaussianHMM` for regime inference.  It consists of 4 core features (always available) and up to 7 extended features (gracefully degraded when source columns are absent).

---

## Feature Composition

| Index | Feature | Source Column(s) | Window | Causality | Purpose |
|-------|---------|-----------------|--------|-----------|---------|
| 0 | `ret_1d` | `return_1d` | 1 bar | CAUSAL | Directional bias |
| 1 | `vol_20d` | `return_vol_20d` | 20 bars | CAUSAL | Volatility regime |
| 2 | `natr` | `NATR_14` | 14 bars | CAUSAL | Normalized range (noise level) |
| 3 | `trend` | `SMASlope_50` | 50 bars | CAUSAL | Trend strength/direction |
| 4 | `credit_spread_proxy` | `GARCH_252`, `return_vol_20d` | 252 bars | CAUSAL | Stress indicator (long/short vol ratio) |
| 5 | `market_breadth` | `return_1d` | 20 bars | CAUSAL | Fraction of recent returns > 0 |
| 6 | `vix_rank` | `return_vol_20d` | 252 bars | CAUSAL | Percentile rank of realized vol |
| 7 | `volume_regime` | `Volume` | 60 bars | CAUSAL | Z-score of volume vs trailing mean |
| 8 | `momentum_20d` | `return_20d` or `Close` | 20 bars | CAUSAL | 20-day price momentum |
| 9 | `mean_reversion` | `Hurst_100` | 100 bars | CAUSAL | 0.5 - Hurst (positive = mean-reverting) |
| 10 | `cross_correlation` | `AutoCorr_20_1` | 20 bars | CAUSAL | Lagged return autocorrelation |

---

## Regime Interpretation

### Regime 0: Trending Bull
- **Returns:** Positive drift
- **Volatility:** Low-moderate (HV < 15% annualized)
- **SMA Slope:** Positive
- **Breadth:** High (> 60% of returns positive)
- **Credit Spread Proxy:** Low (< 0.5)
- **Momentum:** Positive
- **Correlation:** Moderate

### Regime 1: Trending Bear
- **Returns:** Negative drift
- **Volatility:** Moderate-high (HV 15-25%)
- **SMA Slope:** Negative
- **Breadth:** Low (< 40% of returns positive)
- **Credit Spread Proxy:** Rising
- **Momentum:** Negative
- **Correlation:** Elevated

### Regime 2: Mean Reverting
- **Returns:** Near-zero, oscillating
- **Volatility:** Low
- **SMA Slope:** Flat (near zero)
- **Hurst Exponent:** < 0.45 (mean-reverting)
- **Breadth:** Mixed
- **Autocorrelation:** Low or negative

### Regime 3: High Volatility
- **Returns:** Extreme values (both directions)
- **Volatility:** High (HV > 25%)
- **NATR:** Elevated (wide daily ranges)
- **VIX Rank:** > 80th percentile
- **Volume:** Spike (z-score > 1)
- **Credit Spread Proxy:** High (> 1.0)

---

## Causality Constraints

**All features are backward-looking (CAUSAL).**

At each observation date `t`, the observation matrix uses only data available at `t`:
- No future prices, returns, or labels
- No forward-looking macro surprises or earnings
- All rolling windows look backward from `t`

The `market_breadth` feature (index 5) is computed as the fraction of recent *single-asset* returns that are positive — it does not require cross-sectional universe data.

---

## Preprocessing

1. **Missing values:** Forward-fill → backward-fill → fill with 0.0
2. **Infinite values:** Replaced with NaN before filling
3. **Standardization:** Each column z-scored (mean 0, std 1) for numerical stability
4. **Zero-variance:** Columns with std < 1e-12 are set to constant 0.0

---

## Extension Points

The observation matrix is designed for extensibility.  Future structural features from SPEC_02 may include:

| Feature | Source | Window | Purpose |
|---------|--------|--------|---------|
| Spectral Entropy | FFT of returns | 252 bars | Periodicity strength |
| SSA Trend Strength | SSA decomposition | 60 bars | Non-stationary trend |
| Jump Intensity | Return distribution | 20 bars | Frequency of large moves |
| Eigenvalue Concentration | Correlation matrix | 60 bars | Systemic stress |

If added, HMM must be retrained with the expanded dimensionality.  The BIC state selection in `select_hmm_states_bic()` will automatically adapt to the new feature count.
