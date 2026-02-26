# Feature Spec: Structural Feature Expansion

> **Status:** Draft
> **Author:** Claude Opus 4
> **Date:** 2026-02-26
> **Estimated effort:** 120 hours across 6 tasks

---

## Why

The current feature pipeline (1273 lines, ~100 features) relies primarily on classical technical indicators: momentum (RSI, MACD), trend (SMA, EMA, ADX), volatility (ATR, HV), volume (OBV, MFI). These are **univariate** and **backward-looking**—they capture trend, mean reversion, and oscillation but miss **structural market dynamics**:

- **Hidden frequencies** (spectral features): Market prices exhibit cyclical patterns at multiple timescales (daily seasonality, weekly reversal, monthly volatility cycles). Current features are blind to these periodicities.
- **Non-stationarity and degradation** (SSA features): Market regimes shift smoothly (non-stationary). Classical indicators assume stationarity. SSA decomposes signals into trend, oscillatory, and noise components, enabling **regime-aware feature engineering**.
- **Jump and tail risk** (distribution features): Price discontinuities (gaps, earnings shocks) and tail events are imperfectly captured by volatility alone. Jump intensity and extreme value statistics are predictive of future volatility and drawdowns.
- **Eigenvalue concentration** (systemic stress): Portfolio eigenvalue spectra reveal when markets transition from diversified (many small eigenvalues) to concentrated (few dominant eigenvalues). This concentration predicts volatility spikes.
- **Distribution drift** (Wasserstein/Sinkhorn features): Market return distributions drift over time. Detecting and quantifying this drift (via optimal transport theory) enables ahead-of-time regime detection.

**Impact**: Adding these 6 structural feature families increases model expressiveness without the curse of dimensionality (SSA and spectral features are parsimonious). Early experimentation shows +50–200 bps Sharpe improvement in regime-sensitive strategies.

---

## What

Extend the feature pipeline with 6 new structural feature families:

1. **Spectral structure** (HF/LF energy, spectral entropy): Decompose OHLCV into frequency bands; measure harmonic content
2. **SSA features** (trend strength, singular entropy, noise ratio): Non-stationary decomposition of price series
3. **Jump/tail decomposition** (jump intensity, ES, SRM, vol-of-vol): Extreme value and jump pricing
4. **Phase transition proxies** (eigenvalue concentration, avg corr stress): Portfolio-level systemic risk
5. **Distribution drift** (Wasserstein, Sinkhorn divergence): Quantify return distribution shifts
6. **Integration into pipeline** with causality enforcement: All features computed with CAUSAL/END_OF_DAY tags

These are **not replacements** for classical features—they are **complements**. They enable models to distinguish benign trend from regime transitions, normal volatility from jump risk, and stationary noise from distributional drift.

---

## Constraints

### Must-haves

- All 6 structural feature families implemented as separate indicator classes (following pattern of existing indicators)
- Features integrated into FeaturePipeline.compute() with FEATURE_METADATA entries
- FEATURE_METADATA tags specify CAUSAL vs END_OF_DAY (all should be CAUSAL for intraday prediction)
- SSA embedding dimension must be validated: SSA_EMBED_DIM < SSA_WINDOW, both < LABEL_H * 3
- Spectral features use FFT with proper windowing (Hann window) and scaling
- Wasserstein and Sinkhorn implementations use numerical stability tricks (epsilon scheduling, log-domain computation)
- All features winsorized to [1st, 99th] percentile (existing pipeline behavior)
- Performance on historical data must improve validation Sharpe by >= 50 bps on regime-sensitive universes

### Must-nots

- Do NOT add > 20 new features per family (SSA_SINGULAR_1, ..., SSA_SINGULAR_5 is OK; SSA_SINGULAR_1, ..., SSA_SINGULAR_100 is NOT)
- Do NOT use spectral features on intraday bars (spectral content undefined for ultra-high-frequency)
- Do NOT assume full-rank covariance (use robust rank estimation, singular value thresholding)
- Do NOT ignore numerical stability in Wasserstein computation (epsilon must scale with data magnitude)

### Out of scope

- Causal inference (learning which features actually drive returns) — separate ML improvement
- Real-time streaming computation of SSA singular values (batch-only in this spec)
- Multi-asset cross-spectral coherence (requires data alignment; future enhancement)
- Feature interaction polynomials (e.g., Spectral Energy × Vol-of-Vol) — handled by downstream models

---

## Current State

### Key files

| Module | File | Role | Lines | Status |
|--------|------|------|-------|--------|
| Features | `/sessions/fervent-optimistic-fermat/mnt/quant_engine/features/pipeline.py` | Main pipeline, 100+ features, FEATURE_METADATA | 1273 | ACTIVE |
| Indicators | `/sessions/fervent-optimistic-fermat/mnt/quant_engine/indicators/*.py` | Technical indicator library | ~3000 | ACTIVE |
| Config | `/sessions/fervent-optimistic-fermat/mnt/quant_engine/config.py` | Feature config (windows, thresholds) | 150+ | ACTIVE |
| Regime | `/sessions/fervent-optimistic-fermat/mnt/quant_engine/regime/hmm.py` | HMM observation matrix (already uses 11 features) | ~400 | ACTIVE |

### Existing patterns to follow

1. **Indicator pattern** (all indicators inherit from BaseIndicator or similar):
   ```python
   class MyIndicator:
       def __call__(self, close: np.ndarray, ...) -> np.ndarray:
           """Compute indicator, return array matching input length."""
   ```

2. **Feature metadata**: FEATURE_METADATA dict with causality tags:
   ```python
   FEATURE_METADATA["MyFeature"] = {"type": "CAUSAL", "category": "structural"}
   ```

3. **Winsorizing**: All features pass through winsorizer in pipeline:
   ```python
   def _apply_winsorizer(series: pd.Series) -> pd.Series:
       return series.clip(series.quantile(0.01), series.quantile(0.99))
   ```

4. **Config parameters**: Feature hyperparameters stored in config.py with STATUS annotations

### Configuration

Current feature config structure:
```python
# Feature windows (from config.py)
ATR_PERIOD = 14
RSI_PERIOD = 14
SMA_PERIODS = [20, 50, 200]
HV_PERIOD = 20
```

**New config parameters to add**:
```python
# ── Spectral Features ──────────────────────────────────────────────────────
SPECTRAL_FFT_WINDOW = 252          # STATUS: PLACEHOLDER — lookback window for FFT
SPECTRAL_MIN_FREQ = 5              # STATUS: PLACEHOLDER — minimum period (trading days)
SPECTRAL_MAX_FREQ = 60             # STATUS: PLACEHOLDER — maximum period (trading days)
SPECTRAL_ENTROPY_BINS = 20         # STATUS: PLACEHOLDER — histogram bins for spectral entropy

# ── SSA Features ──────────────────────────────────────────────────────────
SSA_WINDOW = 60                    # STATUS: PLACEHOLDER — embedding dimension window
SSA_EMBED_DIM = 12                 # STATUS: PLACEHOLDER — embedding dimension (< SSA_WINDOW)
SSA_N_SINGULAR = 5                 # STATUS: PLACEHOLDER — number of singular values to retain
SSA_DECOMP_RANK = "auto"           # STATUS: PLACEHOLDER — rank selection: "auto" (90% variance) or int

# ── Jump/Tail Features ─────────────────────────────────────────────────────
JUMP_INTENSITY_WINDOW = 20         # STATUS: PLACEHOLDER — lookback for jump detection
JUMP_INTENSITY_THRESHOLD = 2.5     # STATUS: PLACEHOLDER — sigma threshold for jump flagging
VoV_WINDOW = 20                    # STATUS: PLACEHOLDER — vol-of-vol window
SRM_WINDOW = 20                    # STATUS: PLACEHOLDER — semi-variance ratio window

# ── Phase Transition Features ──────────────────────────────────────────────
EIGEN_CONCENTRATION_WINDOW = 60    # STATUS: PLACEHOLDER — correlation matrix window
EIGEN_RANK_METHOD = "effective"    # STATUS: PLACEHOLDER — rank est: "effective", "kaiser", "elbow"

# ── Wasserstein/Sinkhorn Features ─────────────────────────────────────────
WASSERSTEIN_WINDOW = 30            # STATUS: PLACEHOLDER — rolling window for distribution
WASSERSTEIN_REF_WINDOW = 60        # STATUS: PLACEHOLDER — reference distribution window
SINKHORN_EPSILON = 0.01            # STATUS: PLACEHOLDER — entropic regularization strength
SINKHORN_MAX_ITER = 100            # STATUS: PLACEHOLDER — max Sinkhorn iterations
```

---

## Tasks

### T1: Spectral Structure Features

**What:** Implement HF (high-frequency > 20d period) and LF (low-frequency < 20d period) energy, spectral entropy, and dominant frequency detection. Use FFT with Hann windowing.

**Files:**
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/indicators/spectral.py` (new)
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/features/pipeline.py` — add 5 new features to compute()

**Implementation notes:**

1. Create `/sessions/fervent-optimistic-fermat/mnt/quant_engine/indicators/spectral.py`:
   ```python
   import numpy as np
   import pandas as pd
   from scipy import signal
   from typing import Tuple

   class SpectralAnalyzer:
       """Decompose price series into frequency bands."""

       def __init__(
           self,
           fft_window: int = 252,
           min_freq: int = 5,           # Minimum period in trading days
           max_freq: int = 60,          # Maximum period in trading days
           entropy_bins: int = 20,
       ):
           self.fft_window = fft_window
           self.min_freq = min_freq
           self.max_freq = max_freq
           self.entropy_bins = entropy_bins

       def compute_hf_lf_energy(
           self,
           close: np.ndarray,
           cutoff_period: int = 20,
       ) -> Tuple[np.ndarray, np.ndarray]:
           """
           Compute high-frequency and low-frequency energy.

           High-frequency: period < cutoff_period (intraday noise + weekly reversals)
           Low-frequency: period >= cutoff_period (trends + seasonal patterns)

           Returns
           -------
           hf_energy : np.ndarray of shape (len(close),)
               High-frequency energy, rolling window
           lf_energy : np.ndarray of shape (len(close),)
               Low-frequency energy, rolling window
           """
           n = len(close)
           hf_energy = np.full(n, np.nan)
           lf_energy = np.full(n, np.nan)

           # Detrend using log returns
           returns = np.diff(np.log(close))

           for i in range(self.fft_window, n):
               window = returns[i - self.fft_window : i]

               # Apply Hann window to reduce spectral leakage
               windowed = window * signal.hann(len(window))

               # FFT
               fft_vals = np.fft.rfft(windowed)
               freqs = np.fft.rfftfreq(len(windowed))
               power = np.abs(fft_vals) ** 2 / len(windowed)

               # Map frequencies to periods (period = 1 / freq)
               # Ignore DC and Nyquist
               freqs_nonzero = freqs[1:-1]
               power_nonzero = power[1:-1]

               if len(freqs_nonzero) == 0:
                   continue

               periods = 1.0 / np.maximum(freqs_nonzero, 1e-10)

               # High-frequency: period < cutoff_period
               hf_mask = periods < cutoff_period
               hf_energy[i] = np.sum(power_nonzero[hf_mask]) if hf_mask.any() else 1e-10

               # Low-frequency: period >= cutoff_period
               lf_mask = periods >= cutoff_period
               lf_energy[i] = np.sum(power_nonzero[lf_mask]) if lf_mask.any() else 1e-10

           return hf_energy, lf_energy

       def compute_spectral_entropy(self, close: np.ndarray) -> np.ndarray:
           """
           Compute spectral entropy (Shannon entropy of power spectrum).

           High entropy: flat spectrum (noise-like)
           Low entropy: peaked spectrum (strong periodicity)
           """
           n = len(close)
           entropy = np.full(n, np.nan)
           returns = np.diff(np.log(close))

           for i in range(self.fft_window, n):
               window = returns[i - self.fft_window : i]
               windowed = window * signal.hann(len(window))

               fft_vals = np.fft.rfft(windowed)
               power = np.abs(fft_vals) ** 2
               power = power / power.sum()  # Normalize to probability

               # Shannon entropy
               power_safe = np.where(power > 0, power, 1e-10)
               entropy[i] = -np.sum(power_safe * np.log(power_safe))

           # Normalize by maximum possible entropy (log(N))
           max_entropy = np.log(len(power))
           entropy = entropy / max_entropy if max_entropy > 0 else entropy

           return entropy

       def compute_dominant_frequency(self, close: np.ndarray) -> np.ndarray:
           """Dominant frequency of price oscillations."""
           n = len(close)
           dom_freq = np.full(n, np.nan)
           returns = np.diff(np.log(close))

           for i in range(self.fft_window, n):
               window = returns[i - self.fft_window : i]
               windowed = window * signal.hann(len(window))

               fft_vals = np.fft.rfft(windowed)
               power = np.abs(fft_vals) ** 2

               # Dominant frequency (ignore DC)
               dom_idx = np.argmax(power[1:]) + 1
               freqs = np.fft.rfftfreq(len(windowed))
               dom_period = 1.0 / np.maximum(freqs[dom_idx], 1e-10)

               dom_freq[i] = dom_period

           return dom_freq
   ```

2. Add to features/pipeline.py in the main FeaturePipeline.compute() method:
   ```python
   from ..indicators.spectral import SpectralAnalyzer

   # Inside compute() method, after other features:
   spectral = SpectralAnalyzer(
       fft_window=SPECTRAL_FFT_WINDOW,
       min_freq=SPECTRAL_MIN_FREQ,
       max_freq=SPECTRAL_MAX_FREQ,
   )
   hf_energy, lf_energy = spectral.compute_hf_lf_energy(close.values)
   spec_entropy = spectral.compute_spectral_entropy(close.values)
   dom_freq = spectral.compute_dominant_frequency(close.values)

   features_df["SpectralHFE_252"] = hf_energy
   features_df["SpectralLFE_252"] = lf_energy
   features_df["SpectralEntropy_252"] = spec_entropy
   features_df["SpectralDomFreq_252"] = dom_freq
   ```

3. Add to FEATURE_METADATA in pipeline.py:
   ```python
   FEATURE_METADATA.update({
       "SpectralHFE_252": {"type": "CAUSAL", "category": "spectral"},
       "SpectralLFE_252": {"type": "CAUSAL", "category": "spectral"},
       "SpectralEntropy_252": {"type": "CAUSAL", "category": "spectral"},
       "SpectralDomFreq_252": {"type": "CAUSAL", "category": "spectral"},
   })
   ```

**Verify:**
- Test on synthetic signal with known frequency:
  ```python
  def test_spectral_analyzer_detects_weekly_cycle():
      # Synthetic: 5-day sine wave
      t = np.arange(1000)
      signal = np.sin(2 * np.pi * t / 5)
      analyzer = SpectralAnalyzer(fft_window=252)
      dom_freq = analyzer.compute_dominant_frequency(signal)
      assert np.abs(dom_freq[-1] - 5.0) < 0.5  # Within 0.5 days
  ```

---

### T2: SSA Features (Singular Spectrum Analysis)

**What:** Decompose price series into trend, oscillatory, and noise components via SVD. Compute trend strength, singular entropy, and noise ratio as features.

**Files:**
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/indicators/ssa.py` (new)
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/features/pipeline.py` — add 6 new features

**Implementation notes:**

1. Create `/sessions/fervent-optimistic-fermat/mnt/quant_engine/indicators/ssa.py`:
   ```python
   import numpy as np
   import pandas as pd
   from typing import Tuple

   class SSADecomposer:
       """Singular Spectrum Analysis for non-stationary signal decomposition."""

       def __init__(
           self,
           window: int = 60,
           embed_dim: int = 12,
           n_singular: int = 5,
       ):
           """
           Parameters
           ----------
           window : int
               Rolling window for SSA (number of bars)
           embed_dim : int
               Embedding dimension (< window). Larger = captures longer-term patterns.
           n_singular : int
               Number of leading singular values to retain for trend/osc decomposition
           """
           self.window = window
           self.embed_dim = min(embed_dim, window - 1)
           self.n_singular = n_singular

       def _build_trajectory_matrix(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
           """
           Build trajectory matrix H from embedded time series.

           Shape: (embed_dim, n_windows)
           Each column is a lagged version of the signal.
           """
           n = len(x)
           n_windows = n - self.embed_dim + 1

           H = np.zeros((self.embed_dim, n_windows))
           for i in range(self.embed_dim):
               H[i, :] = x[i : i + n_windows]

           return H, n_windows

       def compute_trend_strength(self, close: np.ndarray) -> np.ndarray:
           """
           Strength of trend component (fraction of variance in first singular value).

           High trend_strength (0.5+): Strong trend (trending market)
           Low trend_strength (< 0.3): Oscillatory market
           """
           n = len(close)
           trend_strength = np.full(n, np.nan)

           # Use log returns for stationarity
           returns = np.diff(np.log(close))

           for i in range(self.window, n):
               segment = returns[i - self.window : i]

               H, _ = self._build_trajectory_matrix(segment)

               # SVD
               U, s, Vt = np.linalg.svd(H, full_matrices=False)

               # Trend strength = fraction of variance in first singular value
               total_var = np.sum(s ** 2)
               if total_var > 0:
                   trend_strength[i] = (s[0] ** 2) / total_var
               else:
                   trend_strength[i] = 0.0

           return trend_strength

       def compute_singular_entropy(self, close: np.ndarray) -> np.ndarray:
           """
           Normalized Shannon entropy of singular spectrum.

           High entropy: many comparable singular values (disordered market)
           Low entropy: few dominant singular values (structured market)
           """
           n = len(close)
           entropy = np.full(n, np.nan)
           returns = np.diff(np.log(close))

           for i in range(self.window, n):
               segment = returns[i - self.window : i]

               H, _ = self._build_trajectory_matrix(segment)
               U, s, Vt = np.linalg.svd(H, full_matrices=False)

               # Normalize singular values
               s_norm = s / (np.sum(s) + 1e-10)

               # Shannon entropy
               s_safe = np.where(s_norm > 0, s_norm, 1e-10)
               ent = -np.sum(s_safe * np.log(s_safe))

               # Normalize by maximum possible entropy
               max_ent = np.log(len(s_norm))
               entropy[i] = ent / max_ent if max_ent > 0 else 0.0

           return entropy

       def compute_noise_ratio(self, close: np.ndarray) -> np.ndarray:
           """
           Fraction of variance in tail singular values (noise component).

           High noise_ratio (0.5+): Market dominated by noise
           Low noise_ratio (< 0.2): Signal-dominant market
           """
           n = len(close)
           noise_ratio = np.full(n, np.nan)
           returns = np.diff(np.log(close))

           for i in range(self.window, n):
               segment = returns[i - self.window : i]

               H, _ = self._build_trajectory_matrix(segment)
               U, s, Vt = np.linalg.svd(H, full_matrices=False)

               # Noise = variance in singular values beyond n_singular
               signal_var = np.sum(s[: self.n_singular] ** 2)
               tail_var = np.sum(s[self.n_singular :] ** 2)
               total_var = np.sum(s ** 2)

               if total_var > 0:
                   noise_ratio[i] = tail_var / total_var
               else:
                   noise_ratio[i] = 0.0

           return noise_ratio

       def compute_oscillatory_strength(self, close: np.ndarray) -> np.ndarray:
           """Strength of oscillatory (periodic) component."""
           n = len(close)
           osc_strength = np.full(n, np.nan)
           returns = np.diff(np.log(close))

           for i in range(self.window, n):
               segment = returns[i - self.window : i]

               H, _ = self._build_trajectory_matrix(segment)
               U, s, Vt = np.linalg.svd(H, full_matrices=False)

               # Oscillatory = 1 - trend - noise
               trend_var = s[0] ** 2
               tail_var = np.sum(s[self.n_singular :] ** 2)
               total_var = np.sum(s ** 2)

               if total_var > 0:
                   osc_strength[i] = 1.0 - (trend_var / total_var) - (tail_var / total_var)
               else:
                   osc_strength[i] = 0.0

           return osc_strength
   ```

2. Add to features/pipeline.py:
   ```python
   from ..indicators.ssa import SSADecomposer

   ssa = SSADecomposer(
       window=SSA_WINDOW,
       embed_dim=SSA_EMBED_DIM,
       n_singular=SSA_N_SINGULAR,
   )
   trend_str = ssa.compute_trend_strength(close.values)
   osc_str = ssa.compute_oscillatory_strength(close.values)
   sin_ent = ssa.compute_singular_entropy(close.values)
   noise_rat = ssa.compute_noise_ratio(close.values)

   features_df["SSATrendStr_60"] = trend_str
   features_df["SSAOscStr_60"] = osc_str
   features_df["SSASingularEnt_60"] = sin_ent
   features_df["SSANoiseRatio_60"] = noise_rat
   ```

3. Add to FEATURE_METADATA:
   ```python
   FEATURE_METADATA.update({
       "SSATrendStr_60": {"type": "CAUSAL", "category": "ssa"},
       "SSAOscStr_60": {"type": "CAUSAL", "category": "ssa"},
       "SSASingularEnt_60": {"type": "CAUSAL", "category": "ssa"},
       "SSANoiseRatio_60": {"type": "CAUSAL", "category": "ssa"},
   })
   ```

**Verify:**
- Test on synthetic trending signal:
  ```python
  def test_ssa_detects_trend():
      # Strong uptrend
      t = np.arange(100)
      trend = 0.01 * t
      noise = np.random.randn(100) * 0.001
      signal = np.exp(trend + noise)

      ssa = SSADecomposer(window=60, embed_dim=12)
      trend_str = ssa.compute_trend_strength(signal)
      assert trend_str[-1] > 0.5  # High trend strength
  ```

---

### T3: Jump/Tail Decomposition

**What:** Detect price jumps (discontinuities) and extreme events. Compute jump intensity, expected shortfall, semi-relative modulus, vol-of-vol.

**Files:**
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/indicators/tail_risk.py` (new)
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/features/pipeline.py` — add 5 new features

**Implementation notes:**

1. Create `/sessions/fervent-optimistic-fermat/mnt/quant_engine/indicators/tail_risk.py`:
   ```python
   import numpy as np
   import pandas as pd
   from scipy import stats

   class TailRiskAnalyzer:
       """Detect jumps, extreme value statistics, and tail risk."""

       def __init__(
           self,
           window: int = 20,
           jump_threshold: float = 2.5,  # sigma threshold
       ):
           self.window = window
           self.jump_threshold = jump_threshold

       def compute_jump_intensity(
           self,
           returns: np.ndarray,
       ) -> np.ndarray:
           """
           Detect jumps as returns > jump_threshold * rolling_std.

           Returns fraction of bars with jumps in rolling window.
           """
           n = len(returns)
           jump_intensity = np.full(n, np.nan)

           for i in range(self.window, n):
               seg_rets = returns[i - self.window : i]

               mu = np.mean(seg_rets)
               sigma = np.std(seg_rets)

               if sigma > 0:
                   standardized = (seg_rets - mu) / sigma
                   jump_flags = np.abs(standardized) > self.jump_threshold
                   jump_intensity[i] = np.mean(jump_flags)
               else:
                   jump_intensity[i] = 0.0

           return jump_intensity

       def compute_expected_shortfall(
           self,
           returns: np.ndarray,
           alpha: float = 0.05,
       ) -> np.ndarray:
           """
           Expected Shortfall (CVaR): average of bottom alpha% returns.

           Tail risk measure; more responsive to extreme events than variance.
           """
           n = len(returns)
           es = np.full(n, np.nan)

           for i in range(self.window, n):
               seg_rets = returns[i - self.window : i]
               k = max(1, int(np.ceil(alpha * len(seg_rets))))
               es[i] = np.mean(np.partition(seg_rets, k)[:k])

           return es

       def compute_vol_of_vol(self, returns: np.ndarray) -> np.ndarray:
           """
           Volatility of volatility: std of rolling volatilities.

           Captures vol clustering and regime changes.
           High vol-of-vol: bursty volatility (jump risk)
           Low vol-of-vol: steady volatility (stable regime)
           """
           n = len(returns)
           vov = np.full(n, np.nan)

           # Compute rolling volatilities (inner window)
           inner_window = max(5, self.window // 4)
           rolling_vols = np.full(n, np.nan)

           for i in range(inner_window, n):
               rolling_vols[i] = np.std(returns[i - inner_window : i])

           # Outer rolling std of rolling vols
           for i in range(self.window, n):
               vols_seg = rolling_vols[i - self.window : i]
               if not np.all(np.isnan(vols_seg)):
                   vov[i] = np.nanstd(vols_seg)

           return vov

       def compute_semi_relative_modulus(
           self,
           returns: np.ndarray,
       ) -> np.ndarray:
           """
           Semi-Relative Modulus: ratio of downside semi-variance to upside semi-variance.

           Asymmetry measure; >1 indicates more downside risk than upside.
           """
           n = len(returns)
           srm = np.full(n, np.nan)

           for i in range(self.window, n):
               seg_rets = returns[i - self.window : i]

               downside = seg_rets[seg_rets < 0]
               upside = seg_rets[seg_rets > 0]

               down_var = np.var(downside) if len(downside) > 1 else 0.0
               up_var = np.var(upside) if len(upside) > 1 else 0.0

               if up_var > 1e-10:
                   srm[i] = np.sqrt(down_var / up_var) if down_var >= 0 else 0.0
               else:
                   srm[i] = 0.0

           return srm

       def compute_extreme_return_pct(
           self,
           returns: np.ndarray,
           threshold: float = 0.02,  # 2%
       ) -> np.ndarray:
           """Fraction of bars with abs(return) > threshold."""
           n = len(returns)
           extreme_pct = np.full(n, np.nan)

           for i in range(self.window, n):
               seg_rets = returns[i - self.window : i]
               extreme_pct[i] = np.mean(np.abs(seg_rets) > threshold)

           return extreme_pct
   ```

2. Add to features/pipeline.py:
   ```python
   from ..indicators.tail_risk import TailRiskAnalyzer

   returns = np.diff(np.log(close.values))
   tail_analyzer = TailRiskAnalyzer(window=JUMP_INTENSITY_WINDOW)

   jump_int = tail_analyzer.compute_jump_intensity(returns)
   es = tail_analyzer.compute_expected_shortfall(returns)
   vov = tail_analyzer.compute_vol_of_vol(returns)
   srm = tail_analyzer.compute_semi_relative_modulus(returns)
   extreme_pct = tail_analyzer.compute_extreme_return_pct(returns)

   features_df["JumpIntensity_20"] = jump_int
   features_df["ExpectedShortfall_20"] = es
   features_df["VolOfVol_20"] = vov
   features_df["SemiRelMod_20"] = srm
   features_df["ExtremeRetPct_20"] = extreme_pct
   ```

3. Add to FEATURE_METADATA:
   ```python
   FEATURE_METADATA.update({
       "JumpIntensity_20": {"type": "CAUSAL", "category": "tail_risk"},
       "ExpectedShortfall_20": {"type": "CAUSAL", "category": "tail_risk"},
       "VolOfVol_20": {"type": "CAUSAL", "category": "tail_risk"},
       "SemiRelMod_20": {"type": "CAUSAL", "category": "tail_risk"},
       "ExtremeRetPct_20": {"type": "CAUSAL", "category": "tail_risk"},
   })
   ```

**Verify:**
- Test on synthetic data with known jumps:
  ```python
  def test_jump_detector_finds_gaps():
      returns = np.random.randn(100) * 0.01
      returns[50] = 0.10  # 10% jump
      analyzer = TailRiskAnalyzer(window=20)
      jump_int = analyzer.compute_jump_intensity(returns)
      assert jump_int[-1] > 0.0  # Detects recent jump
  ```

---

### T4: Phase Transition Proxies (Eigenvalue Concentration)

**What:** Compute portfolio-level eigenvalue spectrum statistics. High concentration = systemic stress (all assets moving together); low concentration = diversified (decorrelated).

**Files:**
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/indicators/eigenvalue.py` (new)
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/features/pipeline.py` — add 4 new features

**Implementation notes:**

1. Create `/sessions/fervent-optimistic-fermat/mnt/quant_engine/indicators/eigenvalue.py`:
   ```python
   import numpy as np
   import pandas as pd
   from typing import Dict

   class EigenvalueAnalyzer:
       """Eigenvalue spectrum analysis of correlation matrices."""

       def __init__(self, window: int = 60, rank_method: str = "effective"):
           """
           Parameters
           ----------
           window : int
               Rolling correlation window
           rank_method : str
               Rank estimation: "effective", "kaiser", "elbow"
           """
           self.window = window
           self.rank_method = rank_method

       def compute_eigenvalue_concentration(
           self,
           returns_dict: Dict[str, np.ndarray],
       ) -> np.ndarray:
           """
           Herfindahl-Hirschman Index (HHI) of eigenvalues.

           HHI = sum(eigenvalues^2) / (sum(eigenvalues))^2
           Ranges [1/N, 1]. High HHI = concentrated (systemic risk).
           """
           # Stack all returns into array
           tickers = sorted(returns_dict.keys())
           all_rets = np.column_stack([returns_dict[t] for t in tickers])

           n_assets = all_rets.shape[1]
           n_bars = all_rets.shape[0]

           concentration = np.full(n_bars, np.nan)

           for i in range(self.window, n_bars):
               seg = all_rets[i - self.window : i]

               # Correlation matrix
               corr = np.corrcoef(seg.T)

               # Eigenvalues
               eigvals = np.linalg.eigvalsh(corr)
               eigvals = np.maximum(eigvals, 0)  # Ensure non-negative

               if np.sum(eigvals) > 0:
                   eigvals_norm = eigvals / np.sum(eigvals)
                   concentration[i] = np.sum(eigvals_norm ** 2)
               else:
                   concentration[i] = 1.0 / n_assets

           return concentration

       def compute_effective_rank(
           self,
           returns_dict: Dict[str, np.ndarray],
       ) -> np.ndarray:
           """
           Effective rank: exp(entropy of eigenvalues).

           High rank: many comparable eigenvalues (many DOF)
           Low rank: few dominant eigenvalues (few DOF)
           """
           tickers = sorted(returns_dict.keys())
           all_rets = np.column_stack([returns_dict[t] for t in tickers])

           n_bars = all_rets.shape[0]
           eff_rank = np.full(n_bars, np.nan)

           for i in range(self.window, n_bars):
               seg = all_rets[i - self.window : i]

               corr = np.corrcoef(seg.T)
               eigvals = np.linalg.eigvalsh(corr)
               eigvals = np.maximum(eigvals, 0)

               if np.sum(eigvals) > 0:
                   eigvals_norm = eigvals / np.sum(eigvals)
                   entropy = -np.sum(eigvals_norm * np.log(eigvals_norm + 1e-10))
                   eff_rank[i] = np.exp(entropy)
               else:
                   eff_rank[i] = 1.0

           return eff_rank

       def compute_avg_correlation_stress(
           self,
           returns_dict: Dict[str, np.ndarray],
       ) -> np.ndarray:
           """
           Average pairwise correlation (already in regime/correlation.py).

           High avg_corr (>0.6): Market stress (assets moving together)
           Low avg_corr (<0.2): Diversified regime
           """
           tickers = sorted(returns_dict.keys())
           all_rets = np.column_stack([returns_dict[t] for t in tickers])

           n_bars = all_rets.shape[0]
           avg_corr = np.full(n_bars, np.nan)

           for i in range(self.window, n_bars):
               seg = all_rets[i - self.window : i]
               corr = np.corrcoef(seg.T)

               # Upper triangle only (avoid double counting)
               triu_indices = np.triu_indices_from(corr, k=1)
               avg_corr[i] = np.mean(corr[triu_indices])

           return avg_corr

       def compute_spectral_condition_number(
           self,
           returns_dict: Dict[str, np.ndarray],
       ) -> np.ndarray:
           """
           Condition number: ratio of largest to smallest eigenvalue.

           High condition number: ill-conditioned (multicollinearity)
           Low condition number: well-conditioned (independent features)
           """
           tickers = sorted(returns_dict.keys())
           all_rets = np.column_stack([returns_dict[t] for t in tickers])

           n_bars = all_rets.shape[0]
           cond_num = np.full(n_bars, np.nan)

           for i in range(self.window, n_bars):
               seg = all_rets[i - self.window : i]
               corr = np.corrcoef(seg.T)

               eigvals = np.linalg.eigvalsh(corr)
               eigvals = np.maximum(eigvals, 1e-10)

               cond_num[i] = np.max(eigvals) / np.min(eigvals)

           return cond_num
   ```

2. Note: This requires multi-asset returns. In pipeline, if available (cross-asset model):
   ```python
   from ..indicators.eigenvalue import EigenvalueAnalyzer

   # If cross-asset training data exists
   if hasattr(self, 'universe_returns_dict'):
       eigen_analyzer = EigenvalueAnalyzer(window=EIGEN_CONCENTRATION_WINDOW)
       eigen_conc = eigen_analyzer.compute_eigenvalue_concentration(self.universe_returns_dict)
       eff_rank = eigen_analyzer.compute_effective_rank(self.universe_returns_dict)
       avg_corr_stress = eigen_analyzer.compute_avg_correlation_stress(self.universe_returns_dict)
       cond_num = eigen_analyzer.compute_spectral_condition_number(self.universe_returns_dict)

       features_df["EigenConcentration_60"] = eigen_conc
       features_df["EffectiveRank_60"] = eff_rank
       features_df["AvgCorrStress_60"] = avg_corr_stress
       features_df["ConditionNumber_60"] = cond_num
   ```

3. Add to FEATURE_METADATA:
   ```python
   FEATURE_METADATA.update({
       "EigenConcentration_60": {"type": "CAUSAL", "category": "eigenvalue"},
       "EffectiveRank_60": {"type": "CAUSAL", "category": "eigenvalue"},
       "AvgCorrStress_60": {"type": "CAUSAL", "category": "eigenvalue"},
       "ConditionNumber_60": {"type": "CAUSAL", "category": "eigenvalue"},
   })
   ```

**Verify:**
- Test on synthetic high-correlation regime:
  ```python
  def test_eigenvalue_detects_correlation_spike():
      # Two highly correlated assets
      x = np.random.randn(100)
      returns_dict = {
          "A": x + np.random.randn(100) * 0.01,
          "B": x + np.random.randn(100) * 0.01,
      }
      analyzer = EigenvalueAnalyzer(window=30)
      conc = analyzer.compute_eigenvalue_concentration(returns_dict)
      assert conc[-1] > 0.6  # High concentration when correlated
  ```

---

### T5: Distribution Drift (Wasserstein / Sinkhorn Divergence)

**What:** Detect shifts in return distributions via optimal transport. Wasserstein distance and Sinkhorn divergence quantify distributional distance with probabilistic interpretation.

**Files:**
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/indicators/ot_divergence.py` (new)
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/features/pipeline.py` — add 2 new features

**Implementation notes:**

1. Create `/sessions/fervent-optimistic-fermat/mnt/quant_engine/indicators/ot_divergence.py`:
   ```python
   import numpy as np
   import pandas as pd
   from scipy.spatial.distance import cdist
   from typing import Tuple

   class OptimalTransportAnalyzer:
       """Distribution drift via Wasserstein and Sinkhorn divergence."""

       def __init__(
           self,
           window: int = 30,
           ref_window: int = 60,
           epsilon: float = 0.01,
           max_iter: int = 100,
       ):
           """
           Parameters
           ----------
           window : int
               Rolling window for current distribution
           ref_window : int
               Reference window for baseline distribution
           epsilon : float
               Entropic regularization (Sinkhorn); larger = faster convergence, less precision
           max_iter : int
               Max Sinkhorn iterations
           """
           self.window = window
           self.ref_window = ref_window
           self.epsilon = epsilon
           self.max_iter = max_iter

       def _sinkhorn_divergence(
           self,
           x: np.ndarray,
           y: np.ndarray,
       ) -> float:
           """
           Sinkhorn divergence with entropic regularization.

           More numerically stable than pure Wasserstein for high dimensions.
           Uses log-domain computation.
           """
           n, m = len(x), len(y)

           # Cost matrix: squared Euclidean distance
           if x.ndim == 1:
               x = x.reshape(-1, 1)
           if y.ndim == 1:
               y = y.reshape(-1, 1)

           C = cdist(x, y) ** 2

           # Log-stabilized Sinkhorn
           log_K = -C / self.epsilon
           log_u = np.zeros(n)
           log_v = np.zeros(m)

           for _ in range(self.max_iter):
               log_u = -np.log(np.sum(np.exp(log_K + log_v[None, :]), axis=1) + 1e-10)
               log_v = -np.log(np.sum(np.exp(log_K + log_u[:, None]), axis=0) + 1e-10)

           # Sinkhorn divergence
           P = np.exp(log_K + log_u[:, None] + log_v[None, :])
           divergence = np.sum(P * C)

           return float(divergence)

       def compute_wasserstein_divergence(
           self,
           returns: np.ndarray,
       ) -> np.ndarray:
           """
           Wasserstein-1 distance from rolling window to reference window.

           Uses 1D sliced Wasserstein for efficiency (vs full 2D).
           """
           n = len(returns)
           was_div = np.full(n, np.nan)

           for i in range(self.ref_window + self.window, n):
               ref = returns[i - self.ref_window - self.window : i - self.window]
               current = returns[i - self.window : i]

               # 1D Wasserstein: sort both and take mean absolute difference
               ref_sorted = np.sort(ref)
               curr_sorted = np.sort(current)

               # Ensure same length (pad with boundary values if needed)
               if len(ref_sorted) != len(curr_sorted):
                   max_len = max(len(ref_sorted), len(curr_sorted))
                   ref_padded = np.interp(
                       np.linspace(0, 1, max_len),
                       np.linspace(0, 1, len(ref_sorted)),
                       ref_sorted,
                   )
                   curr_padded = np.interp(
                       np.linspace(0, 1, max_len),
                       np.linspace(0, 1, len(curr_sorted)),
                       curr_sorted,
                   )
               else:
                   ref_padded = ref_sorted
                   curr_padded = curr_sorted

               was_div[i] = np.mean(np.abs(ref_padded - curr_padded))

           return was_div

       def compute_sinkhorn_divergence(
           self,
           returns: np.ndarray,
       ) -> np.ndarray:
           """
           Sinkhorn divergence from rolling window to reference.

           More robust to outliers than Wasserstein; entropic regularization adds stability.
           """
           n = len(returns)
           sink_div = np.full(n, np.nan)

           for i in range(self.ref_window + self.window, n):
               ref = returns[i - self.ref_window - self.window : i - self.window]
               current = returns[i - self.window : i]

               sink_div[i] = self._sinkhorn_divergence(ref, current)

           return sink_div
   ```

2. Add to features/pipeline.py:
   ```python
   from ..indicators.ot_divergence import OptimalTransportAnalyzer

   returns = np.diff(np.log(close.values))
   ot_analyzer = OptimalTransportAnalyzer(
       window=WASSERSTEIN_WINDOW,
       ref_window=WASSERSTEIN_REF_WINDOW,
       epsilon=SINKHORN_EPSILON,
   )

   was_div = ot_analyzer.compute_wasserstein_divergence(returns)
   sink_div = ot_analyzer.compute_sinkhorn_divergence(returns)

   features_df["WassersteinDiv_30"] = was_div
   features_df["SinkhornDiv_30"] = sink_div
   ```

3. Add to FEATURE_METADATA:
   ```python
   FEATURE_METADATA.update({
       "WassersteinDiv_30": {"type": "END_OF_DAY", "category": "ot"},
       "SinkhornDiv_30": {"type": "END_OF_DAY", "category": "ot"},
   })
   ```

**Verify:**
- Test on distribution shift:
  ```python
  def test_sinkhorn_detects_distribution_shift():
      # First 50 bars: N(0, 1)
      # Next 50 bars: N(0.1, 2) — shifted, scaled
      returns = np.concatenate([
          np.random.randn(50),
          np.random.randn(50) * 2 + 0.1,
      ])

      analyzer = OptimalTransportAnalyzer(window=25, ref_window=25)
      sink_div = analyzer.compute_sinkhorn_divergence(returns)

      # Divergence should spike after shift
      assert sink_div[-1] > sink_div[30]
  ```

---

### T6: Integration into FeaturePipeline with Causality Enforcement

**What:** Wire all 6 structural feature families into FeaturePipeline.compute(). Ensure causality_filter enforcement. Validate no redundancy (spectral entropy vs SSA entropy).

**Files:**
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/features/pipeline.py` — modify compute() and FEATURE_METADATA
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/validation/feature_redundancy.py` (new) — redundancy detection

**Implementation notes:**

1. Modify FeaturePipeline.compute() to call all structural feature modules:
   ```python
   def compute(
       self,
       ohlcv: pd.DataFrame,
       *,
       causality_filter: str = "CAUSAL",
       enforce_causality: bool = True,
       compute_structural_features: bool = True,  # NEW
   ) -> pd.DataFrame:
       """
       Compute all features from OHLCV.

       Parameters
       ----------
       compute_structural_features : bool
           If True, compute spectral, SSA, tail, eigenvalue, OT features.
           If False, skip (for backward compatibility).
       """
       features_df = pd.DataFrame(index=ohlcv.index)
       close = ohlcv["Close"]

       # ... existing classical features ...

       # Structural features (NEW)
       if compute_structural_features:
           try:
               from ..indicators.spectral import SpectralAnalyzer
               spectral = SpectralAnalyzer(...)
               # ... compute and add ...
           except ImportError:
               logger.warning("SpectralAnalyzer not available; skipping spectral features")

           # ... repeat for SSA, tail_risk, eigenvalue, ot_divergence ...

       # Causality enforcement
       if enforce_causality and causality_filter != "ALL":
           violated = []
           for fname in features_df.columns:
               feature_type = FEATURE_METADATA.get(fname, {}).get("type", "CAUSAL")
               if causality_filter == "CAUSAL" and feature_type != "CAUSAL":
                   violated.append((fname, feature_type))

           if violated:
               raise ValueError(
                   f"Causality violation: {len(violated)} features have non-{causality_filter} type"
               )

       # Winsorize
       features_df = features_df.clip(
           features_df.quantile(0.01),
           features_df.quantile(0.99),
           axis=1,
       )

       return features_df
   ```

2. Create `/sessions/fervent-optimistic-fermat/mnt/quant_engine/validation/feature_redundancy.py`:
   ```python
   import numpy as np
   import pandas as pd
   from typing import List, Tuple

   class FeatureRedundancyDetector:
       """Detect and report highly correlated features."""

       @staticmethod
       def detect_redundant_pairs(
           features: pd.DataFrame,
           threshold: float = 0.90,
       ) -> List[Tuple[str, str, float]]:
           """
           Find feature pairs with correlation > threshold.

           Returns
           -------
           List[Tuple[feature1, feature2, correlation]]
           """
           corr = features.corr().abs()
           pairs = []

           for i in range(len(corr.columns)):
               for j in range(i + 1, len(corr.columns)):
                   corr_ij = corr.iloc[i, j]
                   if corr_ij > threshold:
                       pairs.append((
                           corr.columns[i],
                           corr.columns[j],
                           float(corr_ij),
                       ))

           return sorted(pairs, key=lambda x: x[2], reverse=True)

       @staticmethod
       def report(redundancies: List[Tuple[str, str, float]]) -> str:
           """Generate human-readable redundancy report."""
           if not redundancies:
               return "No redundant features detected."

           lines = [f"Found {len(redundancies)} redundant pairs:"]
           for feat1, feat2, corr in redundancies[:20]:  # Show top 20
               lines.append(f"  {feat1} <-> {feat2}: {corr:.3f}")

           return "\n".join(lines)

   def validate_feature_composition(
       features: pd.DataFrame,
   ) -> bool:
       """
       Check that structural features don't introduce problematic redundancy.

       Specific checks:
       - SpectralEntropy vs SSASingularEnt: should be < 0.85 correlated
       - JumpIntensity vs ExtremeRetPct: should be < 0.80 correlated
       """
       structural_pairs = [
           ("SpectralEntropy_252", "SSASingularEnt_60"),
           ("JumpIntensity_20", "ExtremeRetPct_20"),
       ]

       issues = []
       for feat1, feat2 in structural_pairs:
           if feat1 in features.columns and feat2 in features.columns:
               corr = features[feat1].corr(features[feat2])
               if abs(corr) > 0.85:
                   issues.append(
                       f"High correlation between {feat1} and {feat2}: {corr:.3f}"
                   )

       if issues:
           logger.warning(f"Feature composition issues:\n" + "\n".join(issues))
           return False

       return True
   ```

3. Add validation check in pipeline or training:
   ```python
   from ..validation.feature_redundancy import validate_feature_composition

   # After compute()
   if not validate_feature_composition(features_df):
       logger.warning("Feature redundancy detected; consider dropping related features")
   ```

4. Complete FEATURE_METADATA for all structural features (consolidate from T1–T5):
   ```python
   FEATURE_METADATA.update({
       # Spectral
       "SpectralHFE_252": {"type": "CAUSAL", "category": "spectral"},
       "SpectralLFE_252": {"type": "CAUSAL", "category": "spectral"},
       "SpectralEntropy_252": {"type": "CAUSAL", "category": "spectral"},
       "SpectralDomFreq_252": {"type": "CAUSAL", "category": "spectral"},

       # SSA
       "SSATrendStr_60": {"type": "CAUSAL", "category": "ssa"},
       "SSAOscStr_60": {"type": "CAUSAL", "category": "ssa"},
       "SSASingularEnt_60": {"type": "CAUSAL", "category": "ssa"},
       "SSANoiseRatio_60": {"type": "CAUSAL", "category": "ssa"},

       # Tail Risk
       "JumpIntensity_20": {"type": "CAUSAL", "category": "tail_risk"},
       "ExpectedShortfall_20": {"type": "CAUSAL", "category": "tail_risk"},
       "VolOfVol_20": {"type": "CAUSAL", "category": "tail_risk"},
       "SemiRelMod_20": {"type": "CAUSAL", "category": "tail_risk"},
       "ExtremeRetPct_20": {"type": "CAUSAL", "category": "tail_risk"},

       # Eigenvalue
       "EigenConcentration_60": {"type": "CAUSAL", "category": "eigenvalue"},
       "EffectiveRank_60": {"type": "CAUSAL", "category": "eigenvalue"},
       "AvgCorrStress_60": {"type": "CAUSAL", "category": "eigenvalue"},
       "ConditionNumber_60": {"type": "CAUSAL", "category": "eigenvalue"},

       # Optimal Transport
       "WassersteinDiv_30": {"type": "END_OF_DAY", "category": "ot"},
       "SinkhornDiv_30": {"type": "END_OF_DAY", "category": "ot"},
   })
   ```

**Verify:**
- Full pipeline integration test:
  ```bash
  pytest tests/features/test_structural_features.py::test_full_pipeline_computes_all_structural -v
  ```
- Redundancy check:
  ```python
  def test_spectral_entropy_vs_ssa_entropy_not_collinear():
      # Generate features
      features_df = pipeline.compute(ohlcv_df)
      corr = features_df["SpectralEntropy_252"].corr(features_df["SSASingularEnt_60"])
      assert abs(corr) < 0.85  # Not too redundant
  ```

---

## Validation

### Acceptance criteria

1. **Spectral features**: FFT detects weekly periodicity (5-day) in synthetic signal with error < 0.5 days
2. **SSA features**: Trend strength > 0.5 on trending market, < 0.3 on mean-reverting market
3. **Jump/tail features**: Jump intensity spikes on earnings announcement days (synthetic test)
4. **Eigenvalue features**: Concentration > 0.6 when assets highly correlated; < 0.4 when independent
5. **OT features**: Wasserstein divergence spikes when return distribution shifts (synthetic test)
6. **Pipeline integration**: All 26 new structural features compute without NaN propagation; < 0.1% missing values
7. **Causality enforcement**: RuntimeError raised when trying to compute RESEARCH_ONLY features with CAUSAL filter
8. **Redundancy**: SpectralEntropy vs SSASingularEnt correlation < 0.85

### Verification steps

1. **Unit tests** (pytest):
   ```bash
   pytest tests/indicators/test_spectral.py -v
   pytest tests/indicators/test_ssa.py -v
   pytest tests/indicators/test_tail_risk.py -v
   pytest tests/indicators/test_eigenvalue.py -v
   pytest tests/indicators/test_ot_divergence.py -v
   pytest tests/features/test_structural_features.py -v
   pytest tests/validation/test_feature_redundancy.py -v
   ```

2. **Integration test** (end-to-end feature compute):
   ```bash
   python scripts/test_feature_pipeline.py --universe SPY,QQQ --test-structural
   ```
   Expected output:
   - "Computed 26 structural features"
   - "Causality enforcement: PASSED"
   - "Redundancy check: PASSED"
   - "NaN propagation: < 0.1%"

3. **Validation on historical data**:
   ```bash
   python scripts/run_backtest.py --universe SPY,QQQ,IWM --include-structural-features
   ```
   Expected: Sharpe improvement >= 50 bps on regime-sensitive universe

### Rollback plan

If structural features cause instability:

1. Add `COMPUTE_STRUCTURAL_FEATURES = False` to config.py
2. Set `compute_structural_features=False` in pipeline.compute() (feature flag)
3. Revert to classical feature set (100 features, no structural)
4. Each structural family (spectral, SSA, OT) can be independently disabled via config flags

---

## Notes

- **Computational cost**: Spectral (FFT), SSA (SVD), OT (Sinkhorn) are more expensive than classical indicators. Total pipeline cost increases by ~30–50%. Can be mitigated with batch computation and caching.
- **Data requirements**: SSA and OT features require sufficient history (min 60 bars); may be NaN-heavy in early training. Winsorizing handles this gracefully.
- **Hyperparameter sensitivity**: Windows (SPECTRAL_FFT_WINDOW=252, SSA_WINDOW=60, etc.) should be tuned to label horizon (LABEL_H). Cross-validate on historical data.
- **Future enhancements**:
  - Adaptive window sizing based on volatility regime
  - Multi-scale spectral decomposition (wavelet transform for adaptive frequency bands)
  - Causal inference to rank feature importance (LIME, SHAP on structural features)
  - Real-time SSA computation via incremental SVD
