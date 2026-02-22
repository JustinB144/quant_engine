"""
Wave-Flow Decomposition for quant_engine.

Physics-inspired decomposition of returns into:
  - **Flow** (secular trend / drift): long-window moving average of returns.
  - **Wave** (oscillatory / mean-reverting): residual after removing the flow.

Spectral analysis (FFT) characterises the wave component, producing features
that connect directly to regime detection: trending regimes exhibit strong flow
while mean-reverting regimes exhibit strong waves.

Reference: 9781107669666 (Bouchaud & Potters — *Theory of Financial Risk and
Derivative Pricing*).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

_EPS = 1e-12


def compute_wave_flow_decomposition(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 120,
    regime_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Decompose the return series into flow (secular trend) and wave (oscillatory)
    components and extract spectral features from the wave.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with at least a 'Close' column.
    short_window : int
        Window used for rolling spectral analysis of the wave component.
    long_window : int
        Window for the moving-average that defines the flow (drift) component.
    regime_threshold : float
        Threshold on ``wave_flow_ratio`` above which the regime is classified
        as mean-reverting (wave-dominated).

    Returns
    -------
    pd.DataFrame
        Columns: flow_component, wave_component, dominant_frequency,
        spectral_entropy, wave_amplitude, flow_strength, wave_flow_ratio,
        wave_regime_indicator.
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    returns = close.pct_change()

    out = pd.DataFrame(index=df.index)

    # --- Flow component: long-window moving average of returns (drift) ---
    flow = returns.rolling(long_window, min_periods=max(10, long_window // 3)).mean()
    out["flow_component"] = flow

    # --- Wave component: residual after removing the flow ---
    wave = returns - flow
    out["wave_component"] = wave

    # --- Rolling spectral analysis on the wave component ---
    wave_vals = wave.values.astype(float)
    n = len(wave_vals)

    dominant_freq = np.full(n, np.nan, dtype=float)
    spectral_ent = np.full(n, np.nan, dtype=float)
    wave_amp = np.full(n, np.nan, dtype=float)
    flow_str = np.full(n, np.nan, dtype=float)
    wf_ratio = np.full(n, np.nan, dtype=float)
    regime_ind = np.full(n, np.nan, dtype=float)

    flow_vals = flow.values.astype(float)
    min_periods_spectral = max(10, short_window // 2)

    for end in range(short_window - 1, n):
        start = end - short_window + 1
        w_block = wave_vals[start : end + 1]
        f_block = flow_vals[start : end + 1]

        valid_w = np.isfinite(w_block)
        valid_f = np.isfinite(f_block)

        if valid_w.sum() < min_periods_spectral:
            continue

        w_clean = w_block.copy()
        w_clean[~valid_w] = 0.0  # zero-fill NaNs for FFT

        # Wave amplitude: std of wave component in this window
        w_std = float(np.nanstd(w_clean))
        wave_amp[end] = w_std

        # Flow strength: absolute mean of flow in this window
        f_clean = f_block.copy()
        f_clean[~valid_f] = 0.0
        f_mean_abs = float(np.abs(np.nanmean(f_clean)))
        flow_str[end] = f_mean_abs

        # Wave-flow ratio
        ratio = w_std / max(f_mean_abs, 1e-10)
        wf_ratio[end] = ratio

        # Regime indicator: 1 if wave-dominated (mean-reverting), 0 otherwise
        regime_ind[end] = 1.0 if ratio > regime_threshold else 0.0

        # --- FFT spectral analysis ---
        fft_vals = np.fft.rfft(w_clean)
        power = np.abs(fft_vals) ** 2

        # Exclude DC component (index 0) for frequency analysis
        if len(power) > 1:
            power_no_dc = power[1:]
            freqs = np.fft.rfftfreq(len(w_clean))[1:]

            total_power = power_no_dc.sum()
            if total_power > _EPS:
                # Dominant frequency: frequency with highest power
                max_idx = int(np.argmax(power_no_dc))
                dominant_freq[end] = float(freqs[max_idx])

                # Spectral entropy: normalized Shannon entropy of power spectrum
                p_norm = power_no_dc / total_power
                # Clip to avoid log(0)
                p_norm = np.clip(p_norm, _EPS, None)
                entropy = -np.sum(p_norm * np.log(p_norm))
                # Normalize by maximum possible entropy (uniform distribution)
                max_entropy = np.log(len(p_norm)) if len(p_norm) > 1 else 1.0
                spectral_ent[end] = entropy / max(max_entropy, _EPS)

    out["dominant_frequency"] = dominant_freq
    out["spectral_entropy"] = spectral_ent
    out["wave_amplitude"] = wave_amp
    out["flow_strength"] = flow_str
    out["wave_flow_ratio"] = wf_ratio
    out["wave_regime_indicator"] = regime_ind

    return out.replace([np.inf, -np.inf], np.nan)
