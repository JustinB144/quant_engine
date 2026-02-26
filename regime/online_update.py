"""Online Regime Updating via Forward Algorithm (SPEC_10 T5).

Provides incremental HMM state updates using the forward algorithm,
avoiding full model refit for each new observation.  Full refit occurs
periodically (default: monthly) while daily updates use the cached HMM
parameters and previous state probability vector.

The forward step is O(K^2 * d) per observation vs O(T * K^2 * d) for a
full refit, giving approximately T-fold speedup for daily updates.

Usage::

    # After initial full fit
    updater = OnlineRegimeUpdater(hmm_model)

    # Daily incremental update
    new_state_prob = updater.forward_step(observation, prev_state_prob)
    regime = np.argmax(new_state_prob)

    # Monthly full refit
    if updater.should_refit(last_refit_date):
        hmm_model.fit(all_data)
        updater = OnlineRegimeUpdater(hmm_model)
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class OnlineRegimeUpdater:
    """Incremental HMM regime updates using the forward algorithm.

    Caches HMM parameters (transition matrix, emission parameters) and
    per-security state probability vectors for fast daily updates.

    Parameters
    ----------
    hmm_model : GaussianHMM
        A fitted GaussianHMM instance with initialized parameters.
    refit_interval_days : int
        Number of days between full refits (default 30).
    anomaly_threshold : float
        Log-likelihood threshold below which an observation is flagged
        as anomalous.  Anomalous observations trigger a warning but
        do not skip the update.
    """

    def __init__(
        self,
        hmm_model: "GaussianHMM",
        refit_interval_days: Optional[int] = None,
        anomaly_threshold: float = -50.0,
    ):
        from ..config import REGIME_ONLINE_REFIT_DAYS

        if hmm_model.pi_ is None or hmm_model.trans_ is None:
            raise RuntimeError(
                "HMM model must be fitted before creating OnlineRegimeUpdater"
            )

        self.n_states = hmm_model.n_states
        self.transition_matrix = hmm_model.trans_.copy()
        self.log_transition = np.log(np.maximum(self.transition_matrix, 1e-12))

        # Store emission parameters for computing log-likelihoods
        self._hmm = hmm_model
        self.refit_interval_days = (
            refit_interval_days if refit_interval_days is not None
            else REGIME_ONLINE_REFIT_DAYS
        )
        self.anomaly_threshold = anomaly_threshold

        # Per-security state probability cache: {security_id: np.ndarray}
        self._state_cache: Dict[str, np.ndarray] = {}

    def forward_step(
        self,
        observation: np.ndarray,
        prev_state_prob: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Single forward algorithm step.

        Given a new observation and the previous state probability vector,
        compute the updated state probability.

        Parameters
        ----------
        observation : np.ndarray, shape (d,)
            New observation vector (same dimensionality as training data).
        prev_state_prob : np.ndarray, shape (K,)
            Previous state probability vector (sums to 1).

        Returns
        -------
        updated_state_prob : np.ndarray, shape (K,)
            Updated state probability vector.
        log_likelihood : float
            Log-likelihood of the observation under the model.
        """
        K = self.n_states
        obs = np.asarray(observation, dtype=float).reshape(1, -1)

        # Compute emission log-probabilities for each state
        log_emit = self._hmm._log_emission(obs).ravel()  # (K,)

        # Forward step: alpha_t(k) = emit(k) * sum_j [alpha_{t-1}(j) * A(j,k)]
        # Work in log space for numerical stability
        log_prev = np.log(np.maximum(prev_state_prob, 1e-12))

        # For each state k: log(sum_j exp(log_prev[j] + log_A[j,k]))
        log_alpha = np.zeros(K)
        for k in range(K):
            log_terms = log_prev + self.log_transition[:, k]
            max_term = np.max(log_terms)
            log_alpha[k] = log_emit[k] + max_term + np.log(
                np.sum(np.exp(log_terms - max_term))
            )

        # Normalize to get state probabilities
        max_log_alpha = np.max(log_alpha)
        alpha = np.exp(log_alpha - max_log_alpha)
        log_likelihood = max_log_alpha + np.log(np.sum(alpha))
        alpha_sum = alpha.sum()
        if alpha_sum > 0:
            alpha /= alpha_sum
        else:
            alpha = np.full(K, 1.0 / K)

        if log_likelihood < self.anomaly_threshold:
            logger.warning(
                "Anomalous observation: log-likelihood=%.2f (threshold=%.2f)",
                log_likelihood, self.anomaly_threshold,
            )

        return alpha, float(log_likelihood)

    def update_regime_for_security(
        self,
        security_id: str,
        observation: np.ndarray,
    ) -> Tuple[int, np.ndarray]:
        """Update regime for a single security using cached state.

        Parameters
        ----------
        security_id : str
            Unique identifier for the security.
        observation : np.ndarray, shape (d,)
            New observation vector.

        Returns
        -------
        regime : int
            Most likely regime (argmax of state probability).
        state_prob : np.ndarray, shape (K,)
            Updated state probability vector.
        """
        # Get previous state probability from cache
        if security_id in self._state_cache:
            prev_prob = self._state_cache[security_id]
        else:
            # First observation: uniform prior
            prev_prob = np.full(self.n_states, 1.0 / self.n_states)

        # Forward step
        state_prob, _ = self.forward_step(observation, prev_prob)

        # Update cache
        self._state_cache[security_id] = state_prob

        return int(np.argmax(state_prob)), state_prob

    def update_batch(
        self,
        security_observations: Dict[str, np.ndarray],
    ) -> Dict[str, Tuple[int, np.ndarray]]:
        """Update regimes for multiple securities in batch.

        Parameters
        ----------
        security_observations : dict
            ``{security_id: observation_vector}``.

        Returns
        -------
        dict
            ``{security_id: (regime, state_prob)}``.
        """
        results = {}
        for sec_id, obs in security_observations.items():
            try:
                regime, prob = self.update_regime_for_security(sec_id, obs)
                results[sec_id] = (regime, prob)
            except Exception as e:
                logger.warning(
                    "Online update failed for %s: %s", sec_id, e,
                )
        return results

    def should_refit(self, last_refit_date: date) -> bool:
        """Check if a full refit is needed.

        Parameters
        ----------
        last_refit_date : date
            Date of the last full HMM refit.

        Returns
        -------
        bool
            True if more than ``refit_interval_days`` have elapsed.
        """
        days_since = (date.today() - last_refit_date).days
        return days_since >= self.refit_interval_days

    def reset_security_cache(self, security_id: Optional[str] = None) -> None:
        """Reset cached state probabilities.

        Parameters
        ----------
        security_id : str or None
            If provided, reset only this security.  If None, reset all.
        """
        if security_id is not None:
            self._state_cache.pop(security_id, None)
        else:
            self._state_cache.clear()

    @property
    def cached_securities(self) -> int:
        """Number of securities with cached state probabilities."""
        return len(self._state_cache)

    def get_state_probabilities(self, security_id: str) -> Optional[np.ndarray]:
        """Get cached state probability vector for a security.

        Returns None if the security has no cached state.
        """
        return self._state_cache.get(security_id)
