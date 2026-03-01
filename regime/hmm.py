"""
Gaussian HMM regime model with sticky transitions and duration smoothing.

This is a lightweight in-repo implementation (no external HMM dependency).
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Internal helper for logsumexp."""
    m = np.max(a, axis=axis, keepdims=True)
    stable = a - m
    s = np.log(np.sum(np.exp(stable), axis=axis, keepdims=True)) + m
    if axis is not None:
        s = np.squeeze(s, axis=axis)
    return s


@dataclass
class HMMFitResult:
    """Fitted HMM outputs including decoded states, posteriors, transitions, and log-likelihood."""
    raw_states: np.ndarray
    state_probs: np.ndarray
    transition_matrix: np.ndarray
    log_likelihood: float


class GaussianHMM:
    """
    Gaussian HMM using EM (Baum-Welch).

    Supports both diagonal and full covariance matrices.  Full covariance
    captures the return-volatility correlation structure that distinguishes
    genuine regime transitions from noise.
    """

    def __init__(
        self,
        n_states: int = 4,
        max_iter: int = 60,
        min_covar: float = 1e-4,
        stickiness: float = 0.92,
        random_state: int = 42,
        min_duration: int = 3,
        prior_weight: float = 0.3,
        covariance_type: str = "full",
    ):
        """Initialize GaussianHMM."""
        self.n_states = n_states
        self.max_iter = max_iter
        self.min_covar = min_covar
        self.stickiness = float(np.clip(stickiness, 0.0, 0.999))
        self.random_state = random_state
        self.min_duration = max(1, int(min_duration))
        self.prior_weight = float(np.clip(prior_weight, 0.0, 1.0))
        self.covariance_type = covariance_type if covariance_type in ("full", "diag") else "full"

        self.pi_: Optional[np.ndarray] = None
        self.trans_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.vars_: Optional[np.ndarray] = None          # diag mode: (K, d)
        self.covars_: Optional[np.ndarray] = None         # full mode: (K, d, d)
        self.log_likelihood_: float = float("-inf")

    def _ensure_positive_definite(self, cov: np.ndarray) -> np.ndarray:
        """Verify positive-definiteness via Cholesky; if it fails, add
        progressively larger diagonal regularization (up to 5 attempts,
        multiplying by 10 each time)."""
        d = cov.shape[0]
        reg = self.min_covar
        for _ in range(5):
            try:
                np.linalg.cholesky(cov)
                return cov
            except np.linalg.LinAlgError:
                cov = cov + reg * np.eye(d)
                reg *= 10.0
        # Final fallback: force diagonal from the current matrix
        cov = np.diag(np.maximum(np.diag(cov), self.min_covar))
        return cov

    def _init_params(self, X: np.ndarray):
        """Internal helper for init params."""
        n, d = X.shape
        rng = np.random.RandomState(self.random_state)

        # k-means++ initialization: select centroids considering all dimensions.
        centroids = np.empty((self.n_states, d))
        # Pick the first centroid uniformly at random.
        centroids[0] = X[rng.randint(0, n)]
        for c in range(1, self.n_states):
            # Squared distances from each point to the nearest existing centroid.
            dists = np.min(
                ((X[:, None, :] - centroids[None, :c, :]) ** 2).sum(axis=2),
                axis=1,
            )
            # Avoid zero-sum edge case.
            dists = np.maximum(dists, 1e-12)
            probs = dists / dists.sum()
            centroids[c] = X[rng.choice(n, p=probs)]

        # One pass nearest-centroid assignment.
        dist = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assign = np.argmin(dist, axis=1)

        means = np.zeros((self.n_states, d))
        vars_ = np.zeros((self.n_states, d))
        covars = np.zeros((self.n_states, d, d))
        for k in range(self.n_states):
            mask = assign == k
            if mask.sum() < 3:
                row = X[rng.randint(0, n)]
                means[k] = row
                vars_[k] = np.maximum(np.var(X, axis=0), self.min_covar)
                covars[k] = np.diag(vars_[k])
            else:
                means[k] = X[mask].mean(axis=0)
                vars_[k] = np.maximum(X[mask].var(axis=0), self.min_covar)
                if self.covariance_type == "full":
                    cov = np.cov(X[mask], rowvar=False) + self.min_covar * np.eye(d)
                    covars[k] = self._ensure_positive_definite(cov)
                else:
                    covars[k] = np.diag(vars_[k])

        pi = np.full(self.n_states, 1.0 / self.n_states)
        off = (1.0 - self.stickiness) / max(1, self.n_states - 1)
        trans = np.full((self.n_states, self.n_states), off)
        np.fill_diagonal(trans, self.stickiness)
        trans /= trans.sum(axis=1, keepdims=True)

        self.pi_ = pi
        self.trans_ = trans
        self.means_ = means
        self.vars_ = vars_
        self.covars_ = covars

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """Internal helper for log emission."""
        T, d = X.shape
        K = self.n_states
        out = np.zeros((T, K))

        if self.covariance_type == "full" and self.covars_ is not None:
            for k in range(K):
                mu = self.means_[k]
                cov = self.covars_[k]
                # Regularize to ensure positive definiteness
                cov = cov + self.min_covar * np.eye(d)
                try:
                    L = np.linalg.cholesky(cov)
                    log_det = 2.0 * np.sum(np.log(np.diag(L)))
                    diff = X - mu  # (T, d)
                    solved = np.linalg.solve(L, diff.T).T  # (T, d)
                    maha = np.sum(solved ** 2, axis=1)  # (T,)
                    out[:, k] = -0.5 * (d * np.log(2 * np.pi) + log_det + maha)
                except np.linalg.LinAlgError:
                    # Fallback to diagonal if Cholesky fails
                    var = np.maximum(np.diag(cov), self.min_covar)
                    term = -0.5 * (np.log(2 * np.pi * var) + ((X - mu) ** 2) / var)
                    out[:, k] = term.sum(axis=1)
        else:
            for k in range(K):
                mu = self.means_[k]
                var = np.maximum(self.vars_[k], self.min_covar)
                term = -0.5 * (np.log(2 * np.pi * var) + ((X - mu) ** 2) / var)
                out[:, k] = term.sum(axis=1)

        return out

    def _forward_backward(self, log_emit: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Internal helper for forward backward."""
        T, K = log_emit.shape
        log_pi = np.log(np.maximum(self.pi_, 1e-12))
        log_A = np.log(np.maximum(self.trans_, 1e-12))

        # Forward pass
        alpha = np.zeros((T, K))
        alpha[0] = log_pi + log_emit[0]
        for t in range(1, T):
            alpha[t] = log_emit[t] + _logsumexp(alpha[t - 1][:, None] + log_A, axis=0)

        loglik = float(_logsumexp(alpha[-1], axis=0))

        # Backward pass
        beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            beta[t] = _logsumexp(log_A + log_emit[t + 1][None, :] + beta[t + 1][None, :], axis=1)

        # Posteriors
        gamma = np.exp(alpha + beta - loglik)
        gamma = np.maximum(gamma, 1e-12)
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi_sum = np.zeros((K, K))
        for t in range(T - 1):
            log_xi_t = (
                alpha[t][:, None]
                + log_A
                + log_emit[t + 1][None, :]
                + beta[t + 1][None, :]
                - loglik
            )
            xi_t = np.exp(log_xi_t)
            s = xi_t.sum()
            if s > 0:
                xi_t /= s
                xi_sum += xi_t

        return gamma, xi_sum, loglik

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        """Return the most likely state sequence via the Viterbi algorithm.

        Works entirely in log-space for numerical stability.

        Parameters
        ----------
        X : np.ndarray, shape (T, d)
            Observation matrix.

        Returns
        -------
        np.ndarray of int, shape (T,)
            Most likely hidden-state sequence.
        """
        if self.pi_ is None:
            raise RuntimeError("Model is not fitted")

        log_emit = self._log_emission(X)          # (T, K)
        T, K = log_emit.shape
        log_pi = np.log(np.maximum(self.pi_, 1e-12))
        log_A = np.log(np.maximum(self.trans_, 1e-12))

        # --- forward (delta) pass ---
        delta = np.zeros((T, K))                   # best log-prob ending in state k
        psi = np.zeros((T, K), dtype=int)           # back-pointers

        delta[0] = log_pi + log_emit[0]
        for t in range(1, T):
            # candidate[j, k] = delta[t-1, j] + log_A[j, k]
            candidate = delta[t - 1][:, None] + log_A   # (K, K)
            psi[t] = np.argmax(candidate, axis=0)        # best prev state for each k
            delta[t] = candidate[psi[t], np.arange(K)] + log_emit[t]

        # --- backtracking pass ---
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path

    def _smooth_duration(
        self,
        states: np.ndarray,
        probs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Merge very short runs into neighboring states (HSMM-like smoothing).

        Updates both states and probabilities to maintain consistency:
        for bars where smoothing changes the regime label, the probability
        of the replacement state is boosted and the row is renormalized.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Smoothed states and updated probabilities.
        """
        if len(states) == 0 or self.min_duration <= 1:
            return states, probs
        s = states.copy()
        p = probs.copy()
        n = len(s)

        i = 0
        while i < n:
            j = i + 1
            while j < n and s[j] == s[i]:
                j += 1
            run_len = j - i
            if run_len < self.min_duration:
                left_state = s[i - 1] if i > 0 else None
                right_state = s[j] if j < n else None
                if left_state is None and right_state is None:
                    i = j
                    continue

                if left_state is None:
                    repl = right_state
                elif right_state is None:
                    repl = left_state
                else:
                    # choose side with higher posterior support on this run
                    left_score = p[i:j, left_state].mean()
                    right_score = p[i:j, right_state].mean()
                    repl = left_state if left_score >= right_score else right_state

                s[i:j] = repl
                # Update probabilities for smoothed bars to be consistent
                for row_idx in range(i, j):
                    original_max = p[row_idx].max()
                    p[row_idx, :] *= 0.5
                    p[row_idx, repl] = max(original_max, 0.6)
                    row_sum = p[row_idx].sum()
                    if row_sum > 0:
                        p[row_idx] /= row_sum
            i = j
        return s, p

    def fit(self, X: np.ndarray) -> HMMFitResult:
        """Fit the transformer to the provided data."""
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if len(X) < max(20, self.n_states * 8):
            raise ValueError("Insufficient samples for HMM fit")

        self._init_params(X)

        prev_ll = float("-inf")
        for _ in range(self.max_iter):
            log_emit = self._log_emission(X)
            gamma, xi_sum, ll = self._forward_backward(log_emit)

            # M-step
            self.pi_ = gamma[0]
            trans = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-12)
            # Sticky prior shrinkage
            off = (1.0 - self.stickiness) / max(1, self.n_states - 1)
            sticky = np.full_like(trans, off)
            np.fill_diagonal(sticky, self.stickiness)
            trans = (1.0 - self.prior_weight) * trans + self.prior_weight * sticky
            trans /= np.maximum(trans.sum(axis=1, keepdims=True), 1e-12)
            self.trans_ = trans

            d = X.shape[1]
            for k in range(self.n_states):
                w = gamma[:, k]
                wsum = np.maximum(w.sum(), 1e-12)
                mu = (w[:, None] * X).sum(axis=0) / wsum
                diff = X - mu
                var = (w[:, None] * diff ** 2).sum(axis=0) / wsum
                self.means_[k] = mu
                self.vars_[k] = np.maximum(var, self.min_covar)

                if self.covariance_type == "full":
                    # Weighted outer product for full covariance
                    cov = (diff * w[:, None]).T @ diff / wsum
                    # Regularize: add min_covar to diagonal
                    cov += self.min_covar * np.eye(d)
                    # Ensure positive-definiteness via Cholesky verification
                    self.covars_[k] = self._ensure_positive_definite(cov)

            if abs(ll - prev_ll) < 1e-4:
                prev_ll = ll
                break
            prev_ll = ll

        self.log_likelihood_ = prev_ll
        probs = self.predict_proba(X)
        raw_states = self.viterbi(X)
        raw_states, probs = self._smooth_duration(raw_states, probs)
        return HMMFitResult(
            raw_states=raw_states,
            state_probs=probs,
            transition_matrix=self.trans_.copy(),
            log_likelihood=float(self.log_likelihood_),
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions from the provided inputs."""
        if self.pi_ is None:
            raise RuntimeError("Model is not fitted")
        log_emit = self._log_emission(X)
        gamma, _, _ = self._forward_backward(log_emit)
        return gamma


def select_hmm_states_bic(
    X: np.ndarray,
    min_states: int = 2,
    max_states: int = 6,
    **hmm_kwargs,
) -> Tuple[int, Dict[int, float]]:
    """Select the optimal number of HMM states using the Bayesian Information Criterion.

    Fits a :class:`GaussianHMM` for each candidate number of states in
    ``[min_states, max_states]`` and computes the BIC score:

        BIC = -2 * log_likelihood + n_params * log(n_samples)

    The candidate with the **lowest** BIC is selected.

    Parameters
    ----------
    X : np.ndarray, shape (T, d)
        Observation matrix (already standardised).
    min_states : int
        Smallest number of hidden states to try.
    max_states : int
        Largest number of hidden states to try.
    **hmm_kwargs
        Extra keyword arguments forwarded to :class:`GaussianHMM` (e.g.
        ``max_iter``, ``stickiness``, ``covariance_type``).

    Returns
    -------
    best_n_states : int
        Number of states with the lowest BIC.
    bic_scores : dict[int, float]
        Mapping from each candidate ``n_states`` to its BIC score.  Candidates
        whose fit failed are omitted.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")

    n_samples, d = X.shape
    log_n = np.log(n_samples)

    covariance_type = hmm_kwargs.get("covariance_type", "full")

    bic_scores: Dict[int, float] = {}

    for k in range(min_states, max_states + 1):
        try:
            model = GaussianHMM(n_states=k, **hmm_kwargs)
            fit_result = model.fit(X)
            ll = fit_result.log_likelihood

            # Count free parameters for a Gaussian HMM with K states and d features.
            # Transition matrix: K*(K-1) free params (each row sums to 1).
            # Initial distribution: (K-1) free params.
            # Means: K*d.
            # Covariances: depends on covariance_type.
            n_transition = k * (k - 1)
            n_initial = k - 1
            n_means = k * d
            if covariance_type == "full":
                n_cov = k * d * (d + 1) // 2
            else:  # diag
                n_cov = k * d

            n_params = n_transition + n_initial + n_means + n_cov
            bic = -2.0 * ll + n_params * log_n
            bic_scores[k] = bic
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            # Fit failed for this candidate — skip it.
            continue

    if not bic_scores:
        # All candidates failed; fall back to min_states.
        return min_states, bic_scores

    best_n_states = min(bic_scores, key=bic_scores.get)  # type: ignore[arg-type]
    return best_n_states, bic_scores


def build_hmm_observation_matrix(
    features: pd.DataFrame,
    backtest_safe: bool = False,
) -> pd.DataFrame:
    """Build an expanded observation matrix for HMM regime inference.

    Constructs up to a 15-dimensional observation vector from available
    features.  Each feature is selected for its ability to distinguish
    market regimes while maintaining causal validity (no look-ahead bias).

    Feature Composition (up to 15 features)
    ----------------------------------------
    **Core features (4, always available from OHLCV):**

    ====  ================  ========  ==========  ========================================
    Idx   Feature           Window    Causality   Regime signal
    ====  ================  ========  ==========  ========================================
     0    ret_1d            1 bar     CAUSAL      Directional bias; positive/negative skew
     1    vol_20d           20 bars   CAUSAL      Low vol = calm, high vol = stress
     2    natr              14 bars   CAUSAL      Tight range = trend, wide range = noise
     3    trend (SMASlope)  50 bars   CAUSAL      Positive = uptrend, negative = downtrend
    ====  ================  ========  ==========  ========================================

    **Extended features (up to 7, graceful fallback if unavailable):**

    ====  =====================  ========  ==========  ========================================
    Idx   Feature                Window    Causality   Regime signal
    ====  =====================  ========  ==========  ========================================
     4    credit_spread_proxy    252 bars  CAUSAL      GARCH/short-vol ratio; stress indicator
     5    market_breadth         20 bars   CAUSAL      Fraction of recent returns > 0
     6    vix_rank               252 bars  CAUSAL      Percentile rank of realized volatility
     7    volume_regime          60 bars   CAUSAL      Z-score of volume vs trailing average
     8    momentum_20d           20 bars   CAUSAL      20-day price momentum
     9    mean_reversion         100 bars  CAUSAL      0.5 - Hurst exponent (positive = MR)
    10    cross_correlation      20 bars   CAUSAL      Lagged return autocorrelation proxy
    ====  =====================  ========  ==========  ========================================

    **Regime interpretation by feature cluster:**

    - **Trending Bull (regime 0):** Positive ret, low vol, positive SMA slope,
      high breadth, low credit spread proxy, positive momentum
    - **Trending Bear (regime 1):** Negative ret, moderate vol, negative SMA slope,
      low breadth, rising credit spread proxy, negative momentum
    - **Mean Reverting (regime 2):** Near-zero ret, low vol, flat SMA slope,
      mixed breadth, low autocorrelation, high mean_reversion signal
    - **High Volatility (regime 3):** Extreme ret, high vol, high NATR,
      elevated VIX rank, volume spikes, high credit spread proxy

    Parameters
    ----------
    features : pd.DataFrame
        Computed features from the feature pipeline.  Must contain at minimum
        ``return_1d``, ``return_vol_20d``, ``NATR_14``, ``SMASlope_50``.
        Extended features are included when their source columns are present.
    backtest_safe : bool
        If True, use expanding-window standardization so that regime labels
        at time ``t`` use only data up to time ``t``. Default False preserves
        the full-series standardization for live inference.

    Returns
    -------
    pd.DataFrame
        Standardized observation matrix (mean 0, std 1 per column).
        Shape ``(n_bars, n_features)`` where ``n_features`` is 4–11
        depending on feature availability.

    Notes
    -----
    - All features are backward-looking (CAUSAL) — no future data used.
    - Missing values are forward-filled, then backward-filled, then zeroed.
    - Inf values are replaced with NaN before filling.
    - Each column is z-scored (mean 0, std 1) for numerical stability.
    - Zero-variance columns are replaced with constant 0.0.
    """
    obs = pd.DataFrame(index=features.index)

    # ── Core features (always available) ──
    obs["ret_1d"] = features.get("return_1d", pd.Series(0.0, index=features.index)).fillna(0.0)
    obs["vol_20d"] = features.get("return_vol_20d", pd.Series(0.01, index=features.index)).fillna(0.01)
    obs["natr"] = features.get("NATR_14", pd.Series(10.0, index=features.index)).fillna(10.0)
    obs["trend"] = features.get("SMASlope_50", pd.Series(0.0, index=features.index)).fillna(0.0)

    # ── Extended features (graceful fallback if unavailable) ──

    # Credit spread proxy: vol premium (high vol / low vol ratio)
    if "GARCH_252" in features.columns and "return_vol_20d" in features.columns:
        garch = features["GARCH_252"].fillna(features["return_vol_20d"].fillna(0.01))
        short_vol = features["return_vol_20d"].fillna(0.01)
        obs["credit_spread_proxy"] = (garch / short_vol.clip(lower=1e-6) - 1.0).fillna(0.0)

    # Market breadth proxy: fraction of recent returns that are positive
    if "return_1d" in features.columns:
        ret = features["return_1d"].fillna(0.0)
        obs["market_breadth"] = ret.rolling(20, min_periods=5).apply(
            lambda x: (x > 0).mean(), raw=True
        ).fillna(0.5)

    # VIX rank: percentile rank of realized vol over 252 days
    if "return_vol_20d" in features.columns:
        vol = features["return_vol_20d"].fillna(features["return_vol_20d"].median())
        obs["vix_rank"] = vol.rolling(252, min_periods=60).apply(
            lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]), raw=False
        ).fillna(0.5)

    # Volume regime: z-score of volume relative to trailing average
    if "Volume" in features.columns:
        vol_series = pd.to_numeric(features["Volume"], errors="coerce").fillna(0)
        vol_mean = vol_series.rolling(60, min_periods=20).mean()
        vol_std = vol_series.rolling(60, min_periods=20).std()
        obs["volume_regime"] = ((vol_series - vol_mean) / vol_std.clip(lower=1e-6)).fillna(0.0)

    # Momentum 20d
    if "return_20d" in features.columns:
        obs["momentum_20d"] = features["return_20d"].fillna(0.0)
    elif "Close" in features.columns:
        close = pd.to_numeric(features["Close"], errors="coerce")
        obs["momentum_20d"] = close.pct_change(20).fillna(0.0)

    # Mean reversion signal: inverse of Hurst exponent
    if "Hurst_100" in features.columns:
        obs["mean_reversion"] = (0.5 - features["Hurst_100"].fillna(0.5))

    # Cross-correlation proxy: lagged return autocorrelation
    if "AutoCorr_20_1" in features.columns:
        obs["cross_correlation"] = features["AutoCorr_20_1"].fillna(0.0)

    # ── Structural features (SPEC_10 T4 — graceful fallback) ──
    # These features are computed by the feature pipeline when available.
    # They expand the observation matrix from 11 to 14-15 features.
    from ..config import REGIME_EXPANDED_FEATURES_ENABLED
    if REGIME_EXPANDED_FEATURES_ENABLED:
        # Spectral entropy: high = noise-like (flat spectrum), low = periodic
        if "SpectralEntropy_252" in features.columns:
            obs["spectral_entropy"] = features["SpectralEntropy_252"].fillna(0.5)

        # SSA trend strength: high = strong trend, low = noisy/oscillatory
        if "SSATrendStr_60" in features.columns:
            obs["ssa_trend_strength"] = features["SSATrendStr_60"].fillna(0.3)

        # BOCPD changepoint confidence: computed inline if not pre-computed
        if "bocpd_changepoint_prob" in features.columns:
            obs["bocpd_changepoint"] = features["bocpd_changepoint_prob"].fillna(0.0)
        elif "return_1d" in features.columns:
            # Compute BOCPD inline with a lightweight pass
            try:
                from .bocpd import BOCPDDetector
                bocpd = BOCPDDetector(hazard_lambda=1.0 / 60, max_runlength=200)
                ret_vals = features["return_1d"].fillna(0.0).values
                if len(ret_vals) >= 10:
                    batch_result = bocpd.batch_update(ret_vals)
                    # Use rolling max of changepoint probs over 5 bars
                    cp_probs = pd.Series(
                        batch_result.changepoint_probs, index=features.index,
                    )
                    obs["bocpd_changepoint"] = cp_probs.rolling(
                        5, min_periods=1,
                    ).max().fillna(0.0)
            except Exception:
                pass  # Graceful fallback: omit BOCPD feature

        # Jump intensity: if available from feature pipeline
        if "JumpIntensity_20" in features.columns:
            obs["jump_intensity"] = features["JumpIntensity_20"].fillna(0.0)

    obs = obs.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    # Standardize columns for numerical stability.
    if backtest_safe:
        # Expanding-window standardization: regime labels at time t use only
        # data up to time t, preventing future data leakage.
        for c in obs.columns:
            expanding_mean = obs[c].expanding(min_periods=20).mean()
            expanding_std = obs[c].expanding(min_periods=20).std()
            expanding_std = expanding_std.replace(0.0, 1e-12)
            obs.loc[:, c] = (obs[c] - expanding_mean) / expanding_std
            # First 19 rows will be NaN — fill with 0
            obs[c] = obs[c].fillna(0.0)
    else:
        # Full-series standardization (fine for live inference)
        for c in obs.columns:
            s = obs[c]
            std = float(s.std())
            if std > 1e-12:
                obs.loc[:, c] = (s - s.mean()) / std
            else:
                obs.loc[:, c] = 0.0

    # Belt-and-suspenders: ensure no NaN or inf survived standardization
    # (e.g. zero-variance columns, single-element series where std() returns NaN)
    obs = obs.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return obs


def map_raw_states_to_regimes(raw_states: np.ndarray, features: pd.DataFrame) -> Dict[int, int]:
    """
    Map unlabeled HMM states -> semantic regimes used by the system.
    """
    stats = []
    for s in sorted(set(raw_states)):
        mask = raw_states == s
        if mask.sum() == 0:
            continue
        ret = features.get("return_1d", pd.Series(0.0, index=features.index))[mask].mean()
        vol = features.get("NATR_14", pd.Series(10.0, index=features.index))[mask].mean()
        trend = features.get("SMASlope_50", pd.Series(0.0, index=features.index))[mask].mean()
        hurst = features.get("Hurst_100", pd.Series(0.5, index=features.index))[mask].mean()
        stats.append((int(s), float(ret), float(vol), float(trend), float(hurst)))

    if not stats:
        return {0: 2}

    states = [x[0] for x in stats]
    vols = np.array([x[2] for x in stats])
    bull_score = np.array([x[1] + x[3] + (x[4] - 0.5) for x in stats])
    mr_score = np.array([0.5 - x[4] for x in stats])

    mapping: Dict[int, int] = {}

    # High volatility gets explicit priority.
    hv_state = states[int(np.argmax(vols))]
    mapping[hv_state] = 3

    remaining = [s for s in states if s != hv_state]
    if not remaining:
        return mapping

    rem_idx = [states.index(s) for s in remaining]
    bull_state = remaining[int(np.argmax(bull_score[rem_idx]))]
    bear_state = remaining[int(np.argmin(bull_score[rem_idx]))]
    mapping[bull_state] = 0
    mapping[bear_state] = 1

    for s in remaining:
        if s not in mapping:
            mapping[s] = 2

    # If all remaining scores are strongly mean-reverting, map weakest trend to MR.
    if len(remaining) >= 1:
        mr_candidate = remaining[int(np.argmax(mr_score[rem_idx]))]
        if mr_score[states.index(mr_candidate)] > 0.05:
            mapping[mr_candidate] = 2

    return mapping
