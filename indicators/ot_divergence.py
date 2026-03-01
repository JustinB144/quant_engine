"""
Optimal Transport Indicators — distribution drift detection.

Detects shifts in return distributions using Wasserstein distance and
Sinkhorn divergence. These features quantify how much the recent return
distribution has diverged from the reference distribution, enabling
ahead-of-time regime detection.

All features are CAUSAL — they use only past and current data.
"""

import numpy as np


class OptimalTransportAnalyzer:
    """Distribution drift detection via Wasserstein and Sinkhorn divergence.

    Compares a rolling "current" distribution to a rolling "reference"
    distribution. Spikes in divergence indicate distributional shifts
    (regime changes, volatility transitions).

    Produces two features:
        - Wasserstein-1 distance: L1 optimal transport cost
        - Sinkhorn divergence: entropically regularized transport cost

    Parameters
    ----------
    window : int
        Size of the "current" distribution window (recent bars).
    ref_window : int
        Size of the "reference" distribution window (baseline, immediately
        preceding the current window).
    epsilon : float
        Entropic regularization for Sinkhorn algorithm. Larger values
        give faster convergence but less precise transport plan.
    max_iter : int
        Maximum Sinkhorn iterations.
    """

    def __init__(
        self,
        window: int = 30,
        ref_window: int = 60,
        epsilon: float = 0.01,
        max_iter: int = 100,
    ):
        if window < 5:
            raise ValueError(f"window must be >= 5, got {window}")
        if ref_window < window:
            raise ValueError(
                f"ref_window ({ref_window}) must be >= window ({window})"
            )
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self.window = window
        self.ref_window = ref_window
        self.epsilon = epsilon
        self.max_iter = max_iter

    def _sinkhorn_divergence_1d(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Compute Sinkhorn divergence between two 1-D distributions.

        Uses log-domain stabilization to prevent numerical overflow/underflow.
        The Sinkhorn divergence is defined as:
            S(P, Q) = OT_ε(P, Q) - 0.5 * OT_ε(P, P) - 0.5 * OT_ε(Q, Q)
        which removes the entropic bias and is non-negative.

        Parameters
        ----------
        x : np.ndarray
            Samples from distribution P.
        y : np.ndarray
            Samples from distribution Q.

        Returns
        -------
        float
            Sinkhorn divergence (non-negative).
        """
        n, m = len(x), len(y)
        if n == 0 or m == 0:
            return 0.0

        # Cost matrix: squared L2 distance (1-D)
        x_col = x.reshape(-1, 1)
        y_row = y.reshape(1, -1)
        C = (x_col - y_row) ** 2

        # Adaptive epsilon scaling: scale with data magnitude
        data_scale = max(np.std(x), np.std(y), 1e-10)
        eps = self.epsilon * (data_scale ** 2)

        # Log-stabilized Sinkhorn iterations
        log_K = -C / eps
        log_a = -np.log(n) * np.ones(n)  # Uniform weights (log)
        log_b = -np.log(m) * np.ones(m)  # Uniform weights (log)

        log_u = np.zeros(n)
        log_v = np.zeros(m)

        for _ in range(self.max_iter):
            # Log-sum-exp for numerical stability
            log_u_new = log_a - _logsumexp_rows(log_K + log_v[None, :])
            log_v_new = log_b - _logsumexp_cols(log_K + log_u_new[:, None])
            log_u = log_u_new
            log_v = log_v_new

        # Transport plan
        log_P = log_u[:, None] + log_K + log_v[None, :]
        P = np.exp(log_P)

        # OT_ε(P, Q) = <P, C>
        ot_pq = np.sum(P * C)

        # Self-transport terms for debiasing (simplified 1-D)
        # For uniform marginals, OT_ε(P, P) uses x-to-x cost
        C_xx = (x_col - x_col.T) ** 2
        log_K_xx = -C_xx / eps
        log_u_xx = np.zeros(n)
        log_v_xx = np.zeros(n)
        log_a_xx = -np.log(n) * np.ones(n)
        for _ in range(self.max_iter // 2):
            log_u_xx = log_a_xx - _logsumexp_rows(log_K_xx + log_v_xx[None, :])
            log_v_xx = log_a_xx - _logsumexp_cols(log_K_xx + log_u_xx[:, None])
        P_xx = np.exp(log_u_xx[:, None] + log_K_xx + log_v_xx[None, :])
        ot_pp = np.sum(P_xx * C_xx)

        C_yy = (y_row.T - y_row) ** 2
        log_K_yy = -C_yy / eps
        log_u_yy = np.zeros(m)
        log_v_yy = np.zeros(m)
        log_b_yy = -np.log(m) * np.ones(m)
        for _ in range(self.max_iter // 2):
            log_u_yy = log_b_yy - _logsumexp_rows(log_K_yy + log_v_yy[None, :])
            log_v_yy = log_b_yy - _logsumexp_cols(log_K_yy + log_u_yy[:, None])
        P_yy = np.exp(log_u_yy[:, None] + log_K_yy + log_v_yy[None, :])
        ot_qq = np.sum(P_yy * C_yy)

        # Sinkhorn divergence (debiased, non-negative)
        divergence = max(0.0, ot_pq - 0.5 * ot_pp - 0.5 * ot_qq)

        return float(divergence)

    def compute_wasserstein_distance(self, returns: np.ndarray) -> np.ndarray:
        """Wasserstein-1 distance between current and reference distributions.

        For 1-D distributions, W₁ = ∫|F(x) - G(x)|dx where F, G are CDFs.
        This is computed efficiently as the mean absolute difference between
        sorted samples (quantile-quantile matching).

        Parameters
        ----------
        returns : np.ndarray
            Log returns (or simple returns).

        Returns
        -------
        was_dist : np.ndarray
            Wasserstein-1 distance (same length as returns).
        """
        n = len(returns)
        was_dist = np.full(n, np.nan)
        total_lookback = self.ref_window + self.window

        for i in range(total_lookback, n):
            ref = returns[i - total_lookback: i - self.window]
            current = returns[i - self.window: i]

            # Sort both distributions
            ref_sorted = np.sort(ref)
            curr_sorted = np.sort(current)

            # Interpolate to common grid for quantile comparison
            n_grid = max(len(ref_sorted), len(curr_sorted))
            ref_quantiles = np.interp(
                np.linspace(0, 1, n_grid),
                np.linspace(0, 1, len(ref_sorted)),
                ref_sorted,
            )
            curr_quantiles = np.interp(
                np.linspace(0, 1, n_grid),
                np.linspace(0, 1, len(curr_sorted)),
                curr_sorted,
            )

            # W₁ = mean |Q_ref(p) - Q_curr(p)| over p ∈ [0,1]
            was_dist[i] = np.mean(np.abs(ref_quantiles - curr_quantiles))

        return was_dist

    def compute_sinkhorn_divergence(self, returns: np.ndarray) -> np.ndarray:
        """Sinkhorn divergence between current and reference distributions.

        Entropically regularized optimal transport cost with debiasing.
        More robust to outliers than Wasserstein and provides smoother
        gradients for optimization.

        Parameters
        ----------
        returns : np.ndarray
            Log returns (or simple returns).

        Returns
        -------
        sink_div : np.ndarray
            Sinkhorn divergence (non-negative, same length as returns).
        """
        n = len(returns)
        sink_div = np.full(n, np.nan)
        total_lookback = self.ref_window + self.window

        for i in range(total_lookback, n):
            ref = returns[i - total_lookback: i - self.window]
            current = returns[i - self.window: i]

            try:
                sink_div[i] = self._sinkhorn_divergence_1d(ref, current)
            except (FloatingPointError, ValueError, OverflowError):
                # Numerical failure — skip this bar
                continue

        return sink_div

    def compute_all(self, returns: np.ndarray) -> dict:
        """Compute all optimal transport features.

        Parameters
        ----------
        returns : np.ndarray
            Log returns (or simple returns).

        Returns
        -------
        dict
            Keys: 'wasserstein_distance', 'sinkhorn_divergence'
        """
        return {
            "wasserstein_distance": self.compute_wasserstein_distance(returns),
            "sinkhorn_divergence": self.compute_sinkhorn_divergence(returns),
        }


def _logsumexp_rows(log_matrix: np.ndarray) -> np.ndarray:
    """Numerically stable log-sum-exp along rows (axis=1)."""
    max_vals = np.max(log_matrix, axis=1, keepdims=True)
    # Prevent -inf propagation
    max_vals = np.where(np.isfinite(max_vals), max_vals, 0.0)
    return (max_vals.ravel() +
            np.log(np.sum(np.exp(log_matrix - max_vals), axis=1) + 1e-30))


def _logsumexp_cols(log_matrix: np.ndarray) -> np.ndarray:
    """Numerically stable log-sum-exp along columns (axis=0)."""
    max_vals = np.max(log_matrix, axis=0, keepdims=True)
    max_vals = np.where(np.isfinite(max_vals), max_vals, 0.0)
    return (max_vals.ravel() +
            np.log(np.sum(np.exp(log_matrix - max_vals), axis=0) + 1e-30))
