"""
Singular Spectrum Analysis (SSA) Indicators — non-stationary signal decomposition.

Decomposes price returns into trend, oscillatory, and noise components via
SVD of the Hankel (trajectory) matrix. Captures regime-aware structural
dynamics that classical indicators miss.

All features are CAUSAL — they use only past and current data.
"""

import numpy as np
import pandas as pd


class SSADecomposer:
    """Singular Spectrum Analysis for non-stationary signal decomposition.

    Produces four features:
        - Trend strength: fraction of variance in first singular value
        - Oscillatory strength: fraction of variance in middle singular values
        - Singular entropy: Shannon entropy of normalized singular spectrum
        - Noise ratio: fraction of variance in tail singular values

    Parameters
    ----------
    window : int
        Rolling lookback window for SSA (number of bars).
    embed_dim : int
        Embedding dimension for trajectory matrix. Must be < window.
        Larger values capture longer-term patterns but require more data.
    n_singular : int
        Number of leading singular values considered as "signal"
        (trend + oscillatory). Remainder is treated as noise.
    """

    def __init__(
        self,
        window: int = 60,
        embed_dim: int = 12,
        n_singular: int = 5,
    ):
        if embed_dim >= window:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be < window ({window})"
            )
        if n_singular < 1:
            raise ValueError(f"n_singular must be >= 1, got {n_singular}")
        self.window = window
        self.embed_dim = min(embed_dim, window - 1)
        self.n_singular = n_singular

    def _build_trajectory_matrix(self, x: np.ndarray) -> np.ndarray:
        """Build Hankel (trajectory) matrix from time series segment.

        Each column is a lagged version of the input signal, creating
        an embedding of the dynamics in delay-coordinate space.

        Parameters
        ----------
        x : np.ndarray
            1-D time series segment of length >= embed_dim.

        Returns
        -------
        H : np.ndarray of shape (embed_dim, n_windows)
            Trajectory matrix where n_windows = len(x) - embed_dim + 1.
        """
        n = len(x)
        n_windows = n - self.embed_dim + 1
        if n_windows < 1:
            return np.empty((self.embed_dim, 0))

        # Construct trajectory matrix via stride tricks for efficiency
        H = np.zeros((self.embed_dim, n_windows))
        for k in range(self.embed_dim):
            H[k, :] = x[k: k + n_windows]
        return H

    def _svd_safe(self, H: np.ndarray):
        """SVD with graceful fallback for degenerate matrices.

        Returns
        -------
        s : np.ndarray
            Singular values (descending).
        """
        if H.shape[1] == 0:
            return np.zeros(min(H.shape))
        try:
            _, s, _ = np.linalg.svd(H, full_matrices=False)
            return s
        except np.linalg.LinAlgError:
            return np.zeros(min(H.shape))

    def compute_trend_strength(self, close: np.ndarray) -> np.ndarray:
        """Fraction of variance captured by the first singular value.

        High values (> 0.5) indicate strong trend dominance.
        Low values (< 0.3) indicate oscillatory or noisy conditions.

        Parameters
        ----------
        close : np.ndarray
            Close prices.

        Returns
        -------
        trend_strength : np.ndarray
            Values in [0, 1], same length as close.
        """
        n = len(close)
        trend_strength = np.full(n, np.nan)

        log_prices = np.log(np.maximum(close, 1e-10))
        returns = np.diff(log_prices)

        for i in range(self.window, n):
            segment = returns[i - self.window: i]
            H = self._build_trajectory_matrix(segment)
            s = self._svd_safe(H)

            total_var = np.sum(s ** 2)
            if total_var > 1e-30:
                trend_strength[i] = (s[0] ** 2) / total_var
            else:
                trend_strength[i] = 0.0

        return trend_strength

    def compute_singular_entropy(self, close: np.ndarray) -> np.ndarray:
        """Normalized Shannon entropy of the singular value spectrum.

        High entropy indicates many comparable singular values (disordered,
        noisy market). Low entropy indicates few dominant singular values
        (structured, predictable market).

        Parameters
        ----------
        close : np.ndarray
            Close prices.

        Returns
        -------
        entropy : np.ndarray
            Normalized entropy in [0, 1].
        """
        n = len(close)
        entropy = np.full(n, np.nan)

        log_prices = np.log(np.maximum(close, 1e-10))
        returns = np.diff(log_prices)

        for i in range(self.window, n):
            segment = returns[i - self.window: i]
            H = self._build_trajectory_matrix(segment)
            s = self._svd_safe(H)

            total_s = np.sum(s)
            if total_s < 1e-30 or len(s) < 2:
                continue

            # Normalize singular values to probability-like weights
            s_norm = s / total_s
            s_safe = np.where(s_norm > 1e-30, s_norm, 1e-30)
            raw_entropy = -np.sum(s_safe * np.log(s_safe))

            # Normalize by max possible entropy
            max_entropy = np.log(len(s_safe))
            entropy[i] = raw_entropy / max_entropy if max_entropy > 0 else 0.0

        return entropy

    def compute_noise_ratio(self, close: np.ndarray) -> np.ndarray:
        """Fraction of variance in tail singular values (noise component).

        Singular values beyond n_singular are treated as noise.
        High noise ratio (> 0.5) indicates noise-dominated dynamics.
        Low noise ratio (< 0.2) indicates signal-dominant dynamics.

        Parameters
        ----------
        close : np.ndarray
            Close prices.

        Returns
        -------
        noise_ratio : np.ndarray
            Values in [0, 1].
        """
        n = len(close)
        noise_ratio = np.full(n, np.nan)

        log_prices = np.log(np.maximum(close, 1e-10))
        returns = np.diff(log_prices)

        for i in range(self.window, n):
            segment = returns[i - self.window: i]
            H = self._build_trajectory_matrix(segment)
            s = self._svd_safe(H)

            total_var = np.sum(s ** 2)
            if total_var < 1e-30:
                noise_ratio[i] = 0.0
                continue

            # Noise = variance in singular values beyond n_singular
            k = min(self.n_singular, len(s))
            tail_var = np.sum(s[k:] ** 2)
            noise_ratio[i] = tail_var / total_var

        return noise_ratio

    def compute_oscillatory_strength(self, close: np.ndarray) -> np.ndarray:
        """Fraction of variance in middle singular values (oscillatory component).

        Oscillatory strength = 1 - trend_strength - noise_ratio.
        High values indicate dominant periodic/cyclical behavior.

        Parameters
        ----------
        close : np.ndarray
            Close prices.

        Returns
        -------
        osc_strength : np.ndarray
            Values in [0, 1].
        """
        n = len(close)
        osc_strength = np.full(n, np.nan)

        log_prices = np.log(np.maximum(close, 1e-10))
        returns = np.diff(log_prices)

        for i in range(self.window, n):
            segment = returns[i - self.window: i]
            H = self._build_trajectory_matrix(segment)
            s = self._svd_safe(H)

            total_var = np.sum(s ** 2)
            if total_var < 1e-30:
                osc_strength[i] = 0.0
                continue

            trend_var = s[0] ** 2
            k = min(self.n_singular, len(s))
            tail_var = np.sum(s[k:] ** 2)
            osc_strength[i] = 1.0 - (trend_var / total_var) - (tail_var / total_var)
            # Clamp to [0, 1] for numerical safety
            osc_strength[i] = max(0.0, min(1.0, osc_strength[i]))

        return osc_strength

    def compute_all(self, close: np.ndarray) -> dict:
        """Compute all SSA features in a single pass.

        More efficient than calling individual methods since SVD is
        computed once per window position.

        Parameters
        ----------
        close : np.ndarray
            Close prices.

        Returns
        -------
        dict
            Keys: 'trend_strength', 'oscillatory_strength',
                  'singular_entropy', 'noise_ratio'
        """
        n = len(close)
        trend_strength = np.full(n, np.nan)
        osc_strength = np.full(n, np.nan)
        entropy = np.full(n, np.nan)
        noise_ratio = np.full(n, np.nan)

        log_prices = np.log(np.maximum(close, 1e-10))
        returns = np.diff(log_prices)

        for i in range(self.window, n):
            segment = returns[i - self.window: i]
            H = self._build_trajectory_matrix(segment)
            s = self._svd_safe(H)

            total_var = np.sum(s ** 2)
            total_s = np.sum(s)

            if total_var < 1e-30:
                trend_strength[i] = 0.0
                osc_strength[i] = 0.0
                noise_ratio[i] = 0.0
                continue

            # Trend strength
            tv = (s[0] ** 2) / total_var
            trend_strength[i] = tv

            # Noise ratio
            k = min(self.n_singular, len(s))
            tail_var = np.sum(s[k:] ** 2)
            nr = tail_var / total_var
            noise_ratio[i] = nr

            # Oscillatory strength
            osc = max(0.0, min(1.0, 1.0 - tv - nr))
            osc_strength[i] = osc

            # Singular entropy
            if total_s > 1e-30 and len(s) >= 2:
                s_norm = s / total_s
                s_safe = np.where(s_norm > 1e-30, s_norm, 1e-30)
                raw_ent = -np.sum(s_safe * np.log(s_safe))
                max_ent = np.log(len(s_safe))
                entropy[i] = raw_ent / max_ent if max_ent > 0 else 0.0

        return {
            "trend_strength": trend_strength,
            "oscillatory_strength": osc_strength,
            "singular_entropy": entropy,
            "noise_ratio": noise_ratio,
        }
