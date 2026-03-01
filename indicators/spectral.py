"""
Spectral Analysis Indicators — FFT-based frequency decomposition of price series.

Decomposes return series into frequency bands to detect cyclical patterns,
measure harmonic content, and identify dominant periodicities. Uses Hann
windowing to reduce spectral leakage.

All features are CAUSAL — they use only past and current data.
"""

import numpy as np


class SpectralAnalyzer:
    """Decompose price series into frequency bands via FFT.

    Produces four features:
        - HF energy: power in high-frequency band (period < cutoff_period)
        - LF energy: power in low-frequency band (period >= cutoff_period)
        - Spectral entropy: Shannon entropy of normalized power spectrum
        - Dominant frequency: period of the strongest spectral peak

    Parameters
    ----------
    fft_window : int
        Number of bars for rolling FFT computation (default 252 ~ 1 year).
    cutoff_period : int
        Period in trading days separating HF from LF bands (default 20).
    """

    def __init__(
        self,
        fft_window: int = 252,
        cutoff_period: int = 20,
    ):
        if fft_window < 20:
            raise ValueError(f"fft_window must be >= 20, got {fft_window}")
        self.fft_window = fft_window
        self.cutoff_period = cutoff_period

    def _hann_window(self, n: int) -> np.ndarray:
        """Hann window to reduce spectral leakage."""
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))

    def _compute_power_spectrum(self, window: np.ndarray):
        """Compute normalized power spectrum of a windowed signal.

        Returns
        -------
        freqs : np.ndarray
            Frequency bins (cycles per bar).
        power : np.ndarray
            Power spectral density at each frequency.
        """
        n = len(window)
        hann = self._hann_window(n)
        windowed = window * hann

        fft_vals = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(n)
        # Parseval-correct normalization: divide by window length and
        # compensate for Hann window energy loss.
        hann_energy = np.mean(hann ** 2)
        power = (np.abs(fft_vals) ** 2) / (n * hann_energy + 1e-30)

        return freqs, power

    def compute_hf_lf_energy(self, close: np.ndarray):
        """High-frequency and low-frequency energy decomposition.

        HF energy captures intraday noise and short-term reversals
        (period < cutoff_period). LF energy captures trends and seasonal
        patterns (period >= cutoff_period).

        Parameters
        ----------
        close : np.ndarray
            Close prices.

        Returns
        -------
        hf_energy : np.ndarray
            High-frequency energy (same length as close).
        lf_energy : np.ndarray
            Low-frequency energy (same length as close).
        """
        n = len(close)
        hf_energy = np.full(n, np.nan)
        lf_energy = np.full(n, np.nan)

        # Detrend using log returns (ensures stationarity for FFT)
        log_prices = np.log(np.maximum(close, 1e-10))
        returns = np.diff(log_prices)

        for i in range(self.fft_window, n):
            window = returns[i - self.fft_window: i]

            freqs, power = self._compute_power_spectrum(window)

            # Skip DC component (index 0) and Nyquist
            if len(freqs) < 3:
                continue
            freqs_inner = freqs[1:-1]
            power_inner = power[1:-1]

            if len(freqs_inner) == 0:
                continue

            # Convert frequencies to periods (trading days per cycle)
            periods = 1.0 / np.maximum(freqs_inner, 1e-10)

            # High-frequency: period < cutoff_period (fast oscillations)
            hf_mask = periods < self.cutoff_period
            hf_energy[i] = np.sum(power_inner[hf_mask]) if hf_mask.any() else 0.0

            # Low-frequency: period >= cutoff_period (trends, seasons)
            lf_mask = periods >= self.cutoff_period
            lf_energy[i] = np.sum(power_inner[lf_mask]) if lf_mask.any() else 0.0

        return hf_energy, lf_energy

    def compute_spectral_entropy(self, close: np.ndarray) -> np.ndarray:
        """Shannon entropy of the normalized power spectrum.

        High entropy indicates a flat spectrum (noise-like, no dominant
        frequency). Low entropy indicates a peaked spectrum (strong
        periodicity, predictable dynamics).

        Parameters
        ----------
        close : np.ndarray
            Close prices.

        Returns
        -------
        entropy : np.ndarray
            Normalized spectral entropy in [0, 1].
        """
        n = len(close)
        entropy = np.full(n, np.nan)

        log_prices = np.log(np.maximum(close, 1e-10))
        returns = np.diff(log_prices)

        for i in range(self.fft_window, n):
            window = returns[i - self.fft_window: i]

            freqs, power = self._compute_power_spectrum(window)

            # Use all non-DC components for entropy
            power_pos = power[1:]
            total_power = np.sum(power_pos)
            if total_power < 1e-30 or len(power_pos) < 2:
                continue

            # Normalize to probability distribution
            p = power_pos / total_power

            # Shannon entropy with numerical safety
            p_safe = np.where(p > 1e-30, p, 1e-30)
            raw_entropy = -np.sum(p_safe * np.log(p_safe))

            # Normalize by maximum possible entropy (uniform distribution)
            max_entropy = np.log(len(p_safe))
            entropy[i] = raw_entropy / max_entropy if max_entropy > 0 else 0.0

        return entropy

    def compute_dominant_frequency(self, close: np.ndarray) -> np.ndarray:
        """Period (in trading days) of the dominant spectral peak.

        Identifies the strongest periodicity in the price oscillations.
        A period of 5 indicates weekly cycles; 21 indicates monthly cycles.

        Parameters
        ----------
        close : np.ndarray
            Close prices.

        Returns
        -------
        dom_period : np.ndarray
            Dominant period in trading days.
        """
        n = len(close)
        dom_period = np.full(n, np.nan)

        log_prices = np.log(np.maximum(close, 1e-10))
        returns = np.diff(log_prices)

        for i in range(self.fft_window, n):
            window = returns[i - self.fft_window: i]

            freqs, power = self._compute_power_spectrum(window)

            # Ignore DC (index 0); find dominant among non-DC components
            if len(power) < 3:
                continue
            power_no_dc = power[1:]
            freqs_no_dc = freqs[1:]

            dom_idx = np.argmax(power_no_dc)
            freq = freqs_no_dc[dom_idx]
            if freq > 1e-10:
                dom_period[i] = 1.0 / freq

        return dom_period

    def compute_spectral_bandwidth(self, close: np.ndarray) -> np.ndarray:
        """Spectral bandwidth — standard deviation of the frequency distribution.

        Measures how spread out the spectral energy is across frequencies.
        Narrow bandwidth indicates a concentrated, pure-tone signal; wide
        bandwidth indicates broadband noise.

        Parameters
        ----------
        close : np.ndarray
            Close prices.

        Returns
        -------
        bandwidth : np.ndarray
            Spectral bandwidth (frequency-domain std).
        """
        n = len(close)
        bandwidth = np.full(n, np.nan)

        log_prices = np.log(np.maximum(close, 1e-10))
        returns = np.diff(log_prices)

        for i in range(self.fft_window, n):
            window = returns[i - self.fft_window: i]

            freqs, power = self._compute_power_spectrum(window)

            power_pos = power[1:]
            freqs_pos = freqs[1:]
            total_power = np.sum(power_pos)
            if total_power < 1e-30 or len(power_pos) < 2:
                continue

            p = power_pos / total_power

            # Spectral centroid (weighted mean frequency)
            centroid = np.sum(p * freqs_pos)

            # Spectral bandwidth (weighted std of frequency)
            variance = np.sum(p * (freqs_pos - centroid) ** 2)
            bandwidth[i] = np.sqrt(max(variance, 0.0))

        return bandwidth

    def compute_all(self, close: np.ndarray) -> dict:
        """Compute all spectral features in a single pass.

        More efficient than calling individual methods since FFT
        is computed once per window position.

        Parameters
        ----------
        close : np.ndarray
            Close prices.

        Returns
        -------
        dict
            Keys: 'hf_energy', 'lf_energy', 'spectral_entropy',
                  'dominant_period', 'spectral_bandwidth'
        """
        n = len(close)
        hf_energy = np.full(n, np.nan)
        lf_energy = np.full(n, np.nan)
        entropy = np.full(n, np.nan)
        dom_period = np.full(n, np.nan)
        bandwidth = np.full(n, np.nan)

        log_prices = np.log(np.maximum(close, 1e-10))
        returns = np.diff(log_prices)

        for i in range(self.fft_window, n):
            window = returns[i - self.fft_window: i]

            freqs, power = self._compute_power_spectrum(window)

            if len(freqs) < 3:
                continue

            # --- HF/LF energy ---
            freqs_inner = freqs[1:-1]
            power_inner = power[1:-1]
            if len(freqs_inner) > 0:
                periods = 1.0 / np.maximum(freqs_inner, 1e-10)
                hf_mask = periods < self.cutoff_period
                lf_mask = periods >= self.cutoff_period
                hf_energy[i] = np.sum(power_inner[hf_mask]) if hf_mask.any() else 0.0
                lf_energy[i] = np.sum(power_inner[lf_mask]) if lf_mask.any() else 0.0

            # --- Spectral entropy ---
            power_no_dc = power[1:]
            freqs_no_dc = freqs[1:]
            total_power = np.sum(power_no_dc)
            if total_power > 1e-30 and len(power_no_dc) >= 2:
                p = power_no_dc / total_power
                p_safe = np.where(p > 1e-30, p, 1e-30)
                raw_ent = -np.sum(p_safe * np.log(p_safe))
                max_ent = np.log(len(p_safe))
                entropy[i] = raw_ent / max_ent if max_ent > 0 else 0.0

                # --- Dominant frequency ---
                dom_idx = np.argmax(power_no_dc)
                freq = freqs_no_dc[dom_idx]
                if freq > 1e-10:
                    dom_period[i] = 1.0 / freq

                # --- Spectral bandwidth ---
                centroid = np.sum(p * freqs_no_dc)
                variance = np.sum(p * (freqs_no_dc - centroid) ** 2)
                bandwidth[i] = np.sqrt(max(variance, 0.0))

        return {
            "hf_energy": hf_energy,
            "lf_energy": lf_energy,
            "spectral_entropy": entropy,
            "dominant_period": dom_period,
            "spectral_bandwidth": bandwidth,
        }
