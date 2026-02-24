"""
Statistical Jump Model for regime detection.

Reference: arXiv 2402.05272 (2024)

Combines regime detection with explicit jump modeling. Instead of continuous
Gaussian HMM, explicitly models:
- Intra-regime dynamics (smooth)
- Jump events (discrete transitions with penalties)

The jump penalty λ controls regime persistence: larger λ → fewer, longer regimes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class JumpModelResult:
    """Result from fitting a Statistical Jump Model."""

    regime_sequence: np.ndarray  # (T,) int — most likely regime per timestep
    regime_probs: np.ndarray  # (T, K) float — posterior probability per regime
    centroids: np.ndarray  # (K, D) float — regime centroids
    jump_penalty: float
    n_iterations: int
    converged: bool


class StatisticalJumpModel:
    """Statistical Jump Model for regime detection.

    Explicitly penalizes regime transitions with a configurable jump penalty.
    Uses a K-means-like algorithm with an L1 penalty on regime changes
    (segmentation via dynamic programming).

    Parameters
    ----------
    n_regimes : int
        Number of regimes to detect (default 4).
    jump_penalty : float
        Penalty for switching regimes. Higher → fewer transitions.
        Typical values: 0.01–0.10 (calibrate from expected transitions/year).
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on centroid movement.
    """

    def __init__(
        self,
        n_regimes: int = 4,
        jump_penalty: float = 0.02,
        max_iter: int = 50,
        tol: float = 1e-6,
    ):
        self.n_regimes = n_regimes
        self.jump_penalty = jump_penalty
        self.max_iter = max_iter
        self.tol = tol

        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

    def fit(self, observations: np.ndarray) -> JumpModelResult:
        """Fit the jump model to observation matrix.

        Parameters
        ----------
        observations : ndarray of shape (T, D)
            T timesteps, D features. Should be standardized.

        Returns
        -------
        JumpModelResult with regime assignments, probabilities, and centroids.
        """
        X = np.asarray(observations, dtype=np.float64)
        T, D = X.shape
        K = self.n_regimes

        if T < K * 2:
            raise ValueError(f"Need at least {K * 2} observations, got {T}")

        # Step 1: Initialize centroids with K-means
        centroids = self._kmeans_init(X, K)

        converged = False
        n_iter = 0

        for n_iter in range(1, self.max_iter + 1):
            # E-step: optimal segmentation via dynamic programming
            labels = self._dp_segment(X, centroids)

            # M-step: update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(K):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    # Dead cluster — reinitialize from random data point
                    new_centroids[k] = X[np.random.randint(T)]

            # Check convergence
            shift = np.max(np.abs(new_centroids - centroids))
            centroids = new_centroids

            if shift < self.tol:
                converged = True
                break

        self.centroids_ = centroids
        self.labels_ = labels

        # Compute soft probabilities from distances to centroids
        regime_probs = self._compute_probs(X, centroids)

        return JumpModelResult(
            regime_sequence=labels,
            regime_probs=regime_probs,
            centroids=centroids,
            jump_penalty=self.jump_penalty,
            n_iterations=n_iter,
            converged=converged,
        )

    def _kmeans_init(self, X: np.ndarray, K: int) -> np.ndarray:
        """Initialize centroids using K-means++."""
        try:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=K, n_init=3, random_state=42, max_iter=20)
            km.fit(X)
            return km.cluster_centers_.copy()
        except ImportError:
            # Fallback: pick K evenly-spaced points from the data
            indices = np.linspace(0, len(X) - 1, K, dtype=int)
            return X[indices].copy()

    def _dp_segment(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Optimal segmentation via dynamic programming with jump penalty.

        Solves:
            min_{s_1..s_T} sum_t ||x_t - mu_{s_t}||^2 + lambda * sum_t 1(s_t != s_{t-1})

        This is the core of the Statistical Jump Model — a Viterbi-like
        forward pass with an additive penalty for regime switches.
        """
        T = X.shape[0]
        K = centroids.shape[0]
        lam = self.jump_penalty

        # Cost of assigning each timestep to each regime: (T, K)
        # Use squared Euclidean distance
        costs = np.zeros((T, K), dtype=np.float64)
        for k in range(K):
            diff = X - centroids[k]
            costs[:, k] = np.sum(diff ** 2, axis=1)

        # Forward DP
        V = np.full((T, K), np.inf, dtype=np.float64)  # min cost to reach (t, k)
        backptr = np.zeros((T, K), dtype=np.int32)

        # Base case
        V[0, :] = costs[0, :]

        for t in range(1, T):
            for k in range(K):
                # Cost of staying vs jumping from each previous state
                stay_or_jump = V[t - 1, :].copy()
                for j in range(K):
                    if j != k:
                        stay_or_jump[j] += lam
                best_prev = np.argmin(stay_or_jump)
                V[t, k] = stay_or_jump[best_prev] + costs[t, k]
                backptr[t, k] = best_prev

        # Backtrack
        labels = np.zeros(T, dtype=np.int32)
        labels[-1] = np.argmin(V[-1, :])
        for t in range(T - 2, -1, -1):
            labels[t] = backptr[t + 1, labels[t + 1]]

        return labels

    def _compute_probs(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute soft regime probabilities from distances to centroids.

        Uses a softmax over negative squared distances for interpretable probs.
        """
        T = X.shape[0]
        K = centroids.shape[0]
        dist_sq = np.zeros((T, K), dtype=np.float64)
        for k in range(K):
            diff = X - centroids[k]
            dist_sq[:, k] = np.sum(diff ** 2, axis=1)

        # Softmax: prob_k = exp(-dist_k / tau) / sum_j exp(-dist_j / tau)
        # Use median distance as temperature scale
        tau = max(np.median(dist_sq), 1e-10)
        log_probs = -dist_sq / tau
        # Numerical stability
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)

        return probs

    @staticmethod
    def compute_jump_penalty_from_data(regime_changes_per_year: float = 4.0) -> float:
        """Calibrate jump penalty from expected regime changes per year.

        Parameters
        ----------
        regime_changes_per_year : float
            Expected number of regime transitions per year. Typical: 3-6.

        Returns
        -------
        float : jump penalty λ (higher → fewer transitions)
        """
        trading_days = 252
        # Average segment length in days
        avg_segment = trading_days / max(regime_changes_per_year, 0.1)
        # Jump penalty scales with average segment duration
        # Empirically, penalty ≈ 0.5 * (avg_segment / trading_days) works well
        return 0.5 * avg_segment / trading_days

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Assign regimes to new observations using fitted centroids.

        Uses the same DP segmentation with stored centroids (no re-fitting).
        """
        if self.centroids_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._dp_segment(observations, self.centroids_)
