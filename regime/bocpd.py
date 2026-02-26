"""
Bayesian Online Change-Point Detection (BOCPD) with Gaussian likelihood.

Implements the Adams & MacKay (2007) algorithm for real-time detection of
regime transitions.  Maintains a run-length posterior P(r_t | x_{1:t}) where
r_t is the number of observations since the last changepoint.

At each new observation the posterior is updated in three steps:
  1. Evaluate predictive likelihood under each run-length hypothesis
  2. Grow each run-length by 1 (no changepoint) or reset to 0 (changepoint)
  3. Normalize the posterior

The underlying observation model uses a Normal-Inverse-Gamma conjugate prior,
yielding a Student-t predictive distribution that naturally widens with
uncertainty early in a segment.

Computational complexity: O(R) per observation where R = max_runlength.
For R = 200 this is ~10 μs per bar — well within the 10 ms/bar budget
for a 1000-stock universe.

Reference
---------
Ryan P. Adams & David J.C. MacKay (2007).  "Bayesian Online Changepoint
Detection."  arXiv:0710.3742.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.special import gammaln

logger = logging.getLogger(__name__)


@dataclass
class BOCPDResult:
    """Output of BOCPD at a single timestep.

    Attributes
    ----------
    run_length_posterior : np.ndarray
        P(r_t | x_{1:t}), shape ``(max_runlength,)``.
    changepoint_prob : float
        P(changepoint at t) = posterior mass at r_t = 0.
    predicted_mean : float
        Posterior-weighted predictive mean.
    predicted_std : float
        Posterior-weighted predictive standard deviation.
    most_likely_runlength : int
        arg max_r P(r_t = r | x_{1:t}).
    """

    run_length_posterior: np.ndarray
    changepoint_prob: float
    predicted_mean: float
    predicted_std: float
    most_likely_runlength: int


@dataclass
class BOCPDBatchResult:
    """Output of BOCPD batch processing over a full time series.

    Attributes
    ----------
    changepoint_probs : np.ndarray
        P(changepoint at t) for each timestep, shape ``(T,)``.
    run_lengths : np.ndarray
        Most likely run-length at each timestep, shape ``(T,)``.
    predicted_means : np.ndarray
        Posterior-weighted predictive mean at each timestep, shape ``(T,)``.
    predicted_stds : np.ndarray
        Posterior-weighted predictive std at each timestep, shape ``(T,)``.
    """

    changepoint_probs: np.ndarray
    run_lengths: np.ndarray
    predicted_means: np.ndarray
    predicted_stds: np.ndarray


class BOCPDDetector:
    """Bayesian Online Change-Point Detection with Gaussian likelihood.

    Maintains run-length posterior P(r_t | x_{1:t}) where r_t is the number
    of observations since the last changepoint.

    The algorithm is fully vectorized over run-lengths for performance.
    Sufficient statistics (mean, precision, shape, rate) are stored as
    arrays rather than per-run-length dicts.

    Parameters
    ----------
    hazard_lambda : float
        Constant hazard rate — expected number of changepoints per bar.
        Default ``1/60`` means one expected change per 60 bars.
    hazard_func : str
        Hazard function type: ``"constant"`` or ``"geometric"``.
    max_runlength : int
        Maximum run-length to track.  Older hypotheses are pruned.
    mu_0 : float
        Prior mean for the observation model.
    kappa_0 : float
        Prior precision scaling (number of pseudo-observations for the mean).
    alpha_0 : float
        Prior shape for the inverse-gamma variance prior.
    beta_0 : float
        Prior rate for the inverse-gamma variance prior.
    """

    def __init__(
        self,
        hazard_lambda: float = 1.0 / 60,
        hazard_func: str = "constant",
        max_runlength: int = 200,
        mu_0: float = 0.0,
        kappa_0: float = 1.0,
        alpha_0: float = 0.5,
        beta_0: float = 0.5,
    ):
        if hazard_lambda <= 0 or hazard_lambda >= 1:
            raise ValueError(
                f"hazard_lambda must be in (0, 1), got {hazard_lambda}"
            )
        if max_runlength < 2:
            raise ValueError(
                f"max_runlength must be >= 2, got {max_runlength}"
            )
        if hazard_func not in ("constant", "geometric"):
            raise ValueError(
                f"hazard_func must be 'constant' or 'geometric', "
                f"got '{hazard_func}'"
            )

        self.hazard_lambda = hazard_lambda
        self.hazard_func = hazard_func
        self.max_runlength = max_runlength

        # Normal-Inverse-Gamma prior hyperparameters
        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        # Sufficient statistics arrays — one entry per possible run-length.
        # These are updated in-place for performance.
        self._mu = np.full(max_runlength, mu_0)
        self._kappa = np.full(max_runlength, kappa_0)
        self._alpha = np.full(max_runlength, alpha_0)
        self._beta = np.full(max_runlength, beta_0)

        # Run-length posterior — initialized to all mass at r=0.
        self._posterior: Optional[np.ndarray] = None
        self._t = 0  # Number of observations processed

    def reset(self) -> None:
        """Reset detector to initial state."""
        R = self.max_runlength
        self._mu = np.full(R, self.mu_0)
        self._kappa = np.full(R, self.kappa_0)
        self._alpha = np.full(R, self.alpha_0)
        self._beta = np.full(R, self.beta_0)
        self._posterior = None
        self._t = 0

    def _hazard(self, r: np.ndarray) -> np.ndarray:
        """Compute hazard rate H(r) = P(changepoint | run-length = r).

        Parameters
        ----------
        r : np.ndarray
            Run-length values.

        Returns
        -------
        np.ndarray
            Hazard rate for each run-length.
        """
        if self.hazard_func == "constant":
            return np.full_like(r, self.hazard_lambda, dtype=float)
        else:
            # Geometric: hazard increases slowly with run-length.
            # H(r) = min(hazard_lambda * (1 + r / 500), 0.5)
            # Capped at 0.5 so the growth probability never goes negative.
            return np.minimum(
                self.hazard_lambda * (1.0 + r.astype(float) / 500.0),
                0.5,
            )

    def _log_student_t(
        self,
        x: float,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """Log Student-t predictive density under NIG posterior.

        Parameters
        ----------
        x : float
            Observation.
        mu, kappa, alpha, beta : np.ndarray
            NIG sufficient statistics arrays.

        Returns
        -------
        np.ndarray
            Log predictive probability for each element.
        """
        nu = 2.0 * alpha
        pred_var = (beta / alpha) * (1.0 + 1.0 / kappa)
        pred_var = np.maximum(pred_var, 1e-12)
        z = (x - mu) ** 2 / (nu * pred_var)
        return (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(np.pi * nu * pred_var)
            - (nu + 1.0) / 2.0 * np.log1p(z)
        )

    def _log_predictive_prob(self, x: float) -> np.ndarray:
        """Compute log-predictive probability under each run-length hypothesis.

        Returns log P(x_t | r_{t-1}, x_{1:t-1}) for each run-length.
        """
        return self._log_student_t(
            x, self._mu, self._kappa, self._alpha, self._beta
        )

    def _log_prior_predictive(self, x: float) -> float:
        """Compute log-predictive probability under the prior model (r=0).

        This is the likelihood of the new observation assuming a changepoint
        just occurred (i.e., starting a fresh segment with prior parameters).
        """
        return float(self._log_student_t(
            x,
            np.array([self.mu_0]),
            np.array([self.kappa_0]),
            np.array([self.alpha_0]),
            np.array([self.beta_0]),
        )[0])

    def _update_suffstats(self, x: float) -> None:
        """Update Normal-Inverse-Gamma sufficient statistics with new observation.

        Shifts all stats by one position (grow run-lengths) and resets
        position 0 to the prior (new segment).

        Parameters
        ----------
        x : float
            New observation.
        """
        R = self.max_runlength

        # Shift: grow run-lengths by 1 (position k becomes k+1).
        # First compute updated stats before shifting.
        old_mu = self._mu.copy()
        old_kappa = self._kappa.copy()
        old_alpha = self._alpha.copy()
        old_beta = self._beta.copy()

        # Bayesian update for each run-length
        kappa_new = old_kappa + 1.0
        mu_new = (old_kappa * old_mu + x) / kappa_new
        alpha_new = old_alpha + 0.5
        beta_new = (
            old_beta
            + 0.5 * old_kappa * (x - old_mu) ** 2 / kappa_new
        )

        # Shift forward: run-length k's updated stats go to position k+1.
        self._mu[1:] = mu_new[: R - 1]
        self._kappa[1:] = kappa_new[: R - 1]
        self._alpha[1:] = alpha_new[: R - 1]
        self._beta[1:] = beta_new[: R - 1]

        # Reset position 0 to prior (new segment starting at this observation).
        self._mu[0] = self.mu_0
        self._kappa[0] = self.kappa_0
        self._alpha[0] = self.alpha_0
        self._beta[0] = self.beta_0

    def update(self, x: float) -> BOCPDResult:
        """Update run-length posterior with a new observation.

        Parameters
        ----------
        x : float
            New observation (scalar).

        Returns
        -------
        BOCPDResult
            Updated posterior, changepoint probability, and predictions.
        """
        R = self.max_runlength

        if not np.isfinite(x):
            # Skip non-finite observations — return previous state or default.
            if self._posterior is not None:
                ml_r = int(np.argmax(self._posterior))
                return BOCPDResult(
                    run_length_posterior=self._posterior.copy(),
                    changepoint_prob=float(self._posterior[0]),
                    predicted_mean=float(self._mu[ml_r]),
                    predicted_std=float(
                        np.sqrt(self._beta[ml_r] / max(self._alpha[ml_r], 0.5))
                    ),
                    most_likely_runlength=ml_r,
                )
            else:
                posterior = np.zeros(R)
                posterior[0] = 1.0
                self._posterior = posterior
                return BOCPDResult(
                    run_length_posterior=posterior.copy(),
                    changepoint_prob=1.0,
                    predicted_mean=self.mu_0,
                    predicted_std=1.0,
                    most_likely_runlength=0,
                )

        if self._posterior is None:
            # First observation: all mass at r=0.
            self._posterior = np.zeros(R)
            self._posterior[0] = 1.0
            self._t = 0

        self._t += 1

        # Step 1: Evaluate predictive likelihood under each run-length.
        log_pred = self._log_predictive_prob(x)

        # Also evaluate the prior model's predictive likelihood for the
        # changepoint term.  Per Adams & MacKay (2007), when a changepoint
        # occurs, the new segment starts fresh with prior parameters, so the
        # observation likelihood under the changepoint hypothesis is
        # P(x_t | prior), NOT P(x_t | r_{t-1}).
        log_prior_pred = self._log_prior_predictive(x)

        # Numerical stability: shift by the maximum log-likelihood before exp.
        log_max = max(np.max(log_pred), log_prior_pred)
        pred_probs = np.exp(log_pred - log_max)
        prior_pred_prob = np.exp(log_prior_pred - log_max)

        # Step 2: Compute growth and changepoint probabilities.
        r_indices = np.arange(R)
        H = self._hazard(r_indices)

        # Growth mass: P(r_t = r+1, x_{1:t}) = P(x_t | r) * (1-H(r)) * P(r | x_{1:t-1})
        growth_mass = pred_probs * (1.0 - H) * self._posterior

        # Changepoint mass: P(r_t = 0, x_{1:t}) = P(x_t | prior) * sum_r [H(r) * P(r | x_{1:t-1})]
        # The key insight: the observation likelihood under the changepoint
        # hypothesis uses the PRIOR model, not each run-length's model.
        cp_mass = prior_pred_prob * np.sum(H * self._posterior)

        # Build new posterior.
        new_posterior = np.zeros(R)
        new_posterior[0] = cp_mass  # Changepoint: reset to r=0.
        new_posterior[1:] = growth_mass[: R - 1]  # Growth: r -> r+1.

        # Step 3: Normalize.
        total = np.sum(new_posterior)
        if total > 0:
            new_posterior /= total
        else:
            # Degenerate case: reset to uniform.
            new_posterior = np.zeros(R)
            new_posterior[0] = 1.0

        # Step 4: Update sufficient statistics.
        self._update_suffstats(x)

        # Store posterior.
        self._posterior = new_posterior

        # Compute posterior-weighted predictions.
        predicted_mean = float(np.sum(new_posterior * self._mu))
        pred_var = (self._beta / np.maximum(self._alpha, 0.5)) * (
            1.0 + 1.0 / np.maximum(self._kappa, 1e-12)
        )
        predicted_std = float(np.sqrt(np.sum(new_posterior * pred_var)))

        # Changepoint probability = mass at r_t = 0.
        changepoint_prob = float(new_posterior[0])

        # Most likely run-length.
        most_likely_r = int(np.argmax(new_posterior))

        return BOCPDResult(
            run_length_posterior=new_posterior.copy(),
            changepoint_prob=changepoint_prob,
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
            most_likely_runlength=most_likely_r,
        )

    def batch_update(self, x_series: np.ndarray) -> BOCPDBatchResult:
        """Process an entire time series and return per-timestep results.

        Resets the detector state before processing.

        Parameters
        ----------
        x_series : np.ndarray, shape ``(T,)``
            Observation time series.

        Returns
        -------
        BOCPDBatchResult
            Changepoint probabilities, run-lengths, predictions for each bar.
        """
        x = np.asarray(x_series, dtype=float).ravel()
        T = len(x)

        if T == 0:
            return BOCPDBatchResult(
                changepoint_probs=np.array([]),
                run_lengths=np.array([]),
                predicted_means=np.array([]),
                predicted_stds=np.array([]),
            )

        self.reset()

        changepoint_probs = np.zeros(T)
        run_lengths = np.zeros(T, dtype=int)
        predicted_means = np.zeros(T)
        predicted_stds = np.zeros(T)

        for t in range(T):
            result = self.update(x[t])
            changepoint_probs[t] = result.changepoint_prob
            run_lengths[t] = result.most_likely_runlength
            predicted_means[t] = result.predicted_mean
            predicted_stds[t] = result.predicted_std

        return BOCPDBatchResult(
            changepoint_probs=changepoint_probs,
            run_lengths=run_lengths,
            predicted_means=predicted_means,
            predicted_stds=predicted_stds,
        )
