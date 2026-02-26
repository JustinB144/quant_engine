"""Health score confidence interval computation — Spec 09.

Provides bootstrap and parametric (t-distribution / normal) methods
for computing 95% confidence intervals on health check scores.

Each health check can report:
    score ± CI  (N = sample_count)

This module also supports CI propagation through weighted averages
(for domain and overall scores).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceResult:
    """Confidence interval for a health score."""

    mean: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    method: str  # "bootstrap", "t", "normal", "binomial", "insufficient"

    @property
    def ci_width(self) -> float:
        return self.ci_upper - self.ci_lower

    @property
    def is_low_confidence(self) -> bool:
        """Flag if sample count is too low for reliable CI."""
        return self.n_samples < 20

    def to_dict(self) -> dict:
        return {
            "mean": round(self.mean, 2),
            "ci_lower": round(self.ci_lower, 2),
            "ci_upper": round(self.ci_upper, 2),
            "ci_width": round(self.ci_width, 2),
            "n_samples": self.n_samples,
            "method": self.method,
            "low_confidence": self.is_low_confidence,
        }


class HealthConfidenceCalculator:
    """Computes confidence intervals for health check scores.

    Methods:
        - Bootstrap (resampling): robust for any distribution, used for small N.
        - Normal approximation (z): used for large N (>= 30).
        - t-distribution: used for moderate N (5–30).
        - Binomial: used for binary (pass/fail) proportions.
    """

    def __init__(
        self,
        ci_level: float = 0.95,
        n_bootstrap: int = 1000,
        bootstrap_threshold: int = 30,
        min_samples: int = 5,
    ):
        self.ci_level = ci_level
        self.n_bootstrap = n_bootstrap
        self.bootstrap_threshold = bootstrap_threshold
        self.min_samples = min_samples

    def compute_ci(
        self,
        samples: Optional[np.ndarray] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        n: Optional[int] = None,
        method: str = "auto",
    ) -> ConfidenceResult:
        """Compute confidence interval using the best available method.

        Parameters
        ----------
        samples : array-like, optional
            Raw sample values. If provided, mean/std/n are computed from them.
        mean, std, n : float/int, optional
            Summary statistics (used when raw samples are unavailable).
        method : str
            "auto" selects based on sample size; or "bootstrap", "t", "normal".

        Returns
        -------
        ConfidenceResult with mean, ci_lower, ci_upper, n_samples, method.
        """
        if samples is not None:
            samples = np.asarray(samples, dtype=float)
            samples = samples[np.isfinite(samples)]
            n_obs = len(samples)
            if n_obs < self.min_samples:
                m = float(np.mean(samples)) if n_obs > 0 else 0.0
                return ConfidenceResult(
                    mean=m, ci_lower=0.0, ci_upper=100.0,
                    n_samples=n_obs, method="insufficient",
                )

            if method == "auto":
                if n_obs < self.bootstrap_threshold:
                    return self.compute_ci_bootstrap(samples)
                else:
                    return self.compute_ci_normal(
                        float(np.mean(samples)),
                        float(np.std(samples, ddof=1)),
                        n_obs,
                    )
            elif method == "bootstrap":
                return self.compute_ci_bootstrap(samples)
            elif method == "t":
                return self.compute_ci_t(
                    float(np.mean(samples)),
                    float(np.std(samples, ddof=1)),
                    n_obs,
                )
            else:
                return self.compute_ci_normal(
                    float(np.mean(samples)),
                    float(np.std(samples, ddof=1)),
                    n_obs,
                )
        elif mean is not None and n is not None:
            s = std if std is not None else 0.0
            if n < self.min_samples:
                return ConfidenceResult(
                    mean=mean, ci_lower=0.0, ci_upper=100.0,
                    n_samples=n, method="insufficient",
                )
            if method == "auto":
                if n < self.bootstrap_threshold:
                    return self.compute_ci_t(mean, s, n)
                else:
                    return self.compute_ci_normal(mean, s, n)
            elif method == "t":
                return self.compute_ci_t(mean, s, n)
            else:
                return self.compute_ci_normal(mean, s, n)
        else:
            return ConfidenceResult(
                mean=0.0, ci_lower=0.0, ci_upper=100.0,
                n_samples=0, method="insufficient",
            )

    def compute_ci_bootstrap(
        self,
        samples: np.ndarray,
    ) -> ConfidenceResult:
        """Bootstrap confidence interval via resampling.

        Resamples with replacement ``n_bootstrap`` times, computes the
        mean of each resample, and returns percentile-based CI.
        """
        n_obs = len(samples)
        if n_obs == 0:
            return ConfidenceResult(
                mean=0.0, ci_lower=0.0, ci_upper=100.0,
                n_samples=0, method="insufficient",
            )

        rng = np.random.default_rng(seed=42)
        boot_means = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            resample = rng.choice(samples, size=n_obs, replace=True)
            boot_means[i] = resample.mean()

        alpha = 1.0 - self.ci_level
        ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        sample_mean = float(np.mean(samples))

        return ConfidenceResult(
            mean=sample_mean,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_samples=n_obs,
            method="bootstrap",
        )

    def compute_ci_normal(
        self,
        mean: float,
        std: float,
        n: int,
    ) -> ConfidenceResult:
        """Normal (z) approximation CI.  Suitable for N >= 30."""
        if n <= 0 or std < 0:
            return ConfidenceResult(
                mean=mean, ci_lower=mean, ci_upper=mean,
                n_samples=n, method="normal",
            )

        alpha = 1.0 - self.ci_level
        z = sp_stats.norm.ppf(1 - alpha / 2)
        se = std / math.sqrt(n)
        ci_lower = mean - z * se
        ci_upper = mean + z * se

        return ConfidenceResult(
            mean=mean, ci_lower=ci_lower, ci_upper=ci_upper,
            n_samples=n, method="normal",
        )

    def compute_ci_t(
        self,
        mean: float,
        std: float,
        n: int,
    ) -> ConfidenceResult:
        """Student's t-distribution CI.  Suitable for small N (5–30)."""
        if n <= 1 or std < 0:
            return ConfidenceResult(
                mean=mean, ci_lower=mean, ci_upper=mean,
                n_samples=n, method="t",
            )

        alpha = 1.0 - self.ci_level
        t_crit = sp_stats.t.ppf(1 - alpha / 2, df=n - 1)
        se = std / math.sqrt(n)
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se

        return ConfidenceResult(
            mean=mean, ci_lower=ci_lower, ci_upper=ci_upper,
            n_samples=n, method="t",
        )

    def compute_ci_binomial(
        self,
        successes: int,
        total: int,
    ) -> ConfidenceResult:
        """Wilson score interval for binary (pass/fail) proportions."""
        if total <= 0:
            return ConfidenceResult(
                mean=0.0, ci_lower=0.0, ci_upper=1.0,
                n_samples=0, method="insufficient",
            )

        p = successes / total
        alpha = 1.0 - self.ci_level
        z = sp_stats.norm.ppf(1 - alpha / 2)
        z2 = z * z

        # Wilson score interval
        denom = 1 + z2 / total
        centre = (p + z2 / (2 * total)) / denom
        spread = z * math.sqrt((p * (1 - p) + z2 / (4 * total)) / total) / denom

        ci_lower = max(0.0, centre - spread)
        ci_upper = min(1.0, centre + spread)

        return ConfidenceResult(
            mean=p, ci_lower=ci_lower, ci_upper=ci_upper,
            n_samples=total, method="binomial",
        )

    @staticmethod
    def propagate_weighted_ci(
        scores: Sequence[float],
        ci_widths: Sequence[float],
        weights: Sequence[float],
    ) -> Tuple[float, float]:
        """Propagate CI through a weighted average.

        For weighted mean S = sum(w_i * s_i), the CI half-width is:
            delta = sqrt(sum((w_i * half_width_i)^2))

        Parameters
        ----------
        scores : sequence of floats
            Individual check/domain scores.
        ci_widths : sequence of floats
            CI half-widths (ci_upper - ci_lower) / 2 for each score.
        weights : sequence of floats
            Weights for each score (should sum to 1).

        Returns
        -------
        (ci_lower, ci_upper) for the weighted average.
        """
        scores_arr = np.asarray(scores, dtype=float)
        widths_arr = np.asarray(ci_widths, dtype=float)
        weights_arr = np.asarray(weights, dtype=float)

        if weights_arr.sum() == 0:
            return (0.0, 0.0)

        # Normalize weights
        weights_arr = weights_arr / weights_arr.sum()
        weighted_mean = float(np.dot(weights_arr, scores_arr))

        # Propagate half-widths through weighted sum
        half_widths = widths_arr / 2.0
        combined_half_width = float(
            math.sqrt(np.sum((weights_arr * half_widths) ** 2))
        )

        return (
            weighted_mean - combined_half_width,
            weighted_mean + combined_half_width,
        )
