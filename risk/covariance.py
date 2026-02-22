"""
Covariance estimation utilities for portfolio risk controls.
"""
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class CovarianceEstimate:
    """Covariance estimation output bundle with metadata about the fit method and sample count."""
    covariance: pd.DataFrame
    n_observations: int
    method: str


class CovarianceEstimator:
    """
    Estimate a robust covariance matrix for asset returns.
    """

    def __init__(
        self,
        method: str = "ledoit_wolf",
        shrinkage: float = 0.15,
        annualization: float = 252.0,
    ):
        """Initialize CovarianceEstimator."""
        self.method = method
        self.shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
        self.annualization = float(max(1.0, annualization))

    def estimate(self, returns: pd.DataFrame) -> CovarianceEstimate:
        """estimate."""
        clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        if clean.shape[0] < 5 or clean.shape[1] == 0:
            cov = pd.DataFrame(
                np.eye(clean.shape[1], dtype=float) * 1e-8,
                index=clean.columns,
                columns=clean.columns,
            )
            return CovarianceEstimate(covariance=cov, n_observations=int(clean.shape[0]), method="fallback")

        # Forward-fill then back-fill partial NaN; drop rows still missing.
        # Avoids biasing covariance toward zero from naive fillna(0).
        clean = clean.ffill().bfill().dropna(how="any")
        if clean.shape[0] < min(30, max(5, clean.shape[1] + 1)):
            cov = pd.DataFrame(
                np.eye(returns.shape[1], dtype=float) * 1e-8,
                index=returns.columns,
                columns=returns.columns,
            )
            return CovarianceEstimate(covariance=cov, n_observations=int(clean.shape[0]), method="fallback")
        cov_values = self._estimate_values(clean.values)
        cov = pd.DataFrame(cov_values, index=clean.columns, columns=clean.columns)
        return CovarianceEstimate(covariance=cov, n_observations=int(clean.shape[0]), method=self.method)

    def portfolio_volatility(
        self,
        weights: Dict[str, float],
        covariance: pd.DataFrame,
    ) -> float:
        """portfolio volatility."""
        if not weights:
            return 0.0

        cols = [c for c in covariance.columns if c in weights]
        if not cols:
            return 0.0

        w = np.array([float(weights[c]) for c in cols], dtype=float)
        cov_vals = covariance.loc[cols, cols].values.astype(float)
        var = float(w @ cov_vals @ w.T)
        var = max(var, 0.0)
        return float(np.sqrt(var * self.annualization))

    def _estimate_values(self, values: np.ndarray) -> np.ndarray:
        """Internal helper for estimate values."""
        if values.ndim != 2:
            raise ValueError("returns array must be 2D")

        n_assets = int(values.shape[1])
        if n_assets == 1:
            series = values[:, 0].astype(float)
            ddof = 1 if len(series) > 1 else 0
            var = float(np.var(series, ddof=ddof))
            return np.array([[max(var, 1e-10)]], dtype=float)

        if self.method == "ledoit_wolf":
            try:
                from sklearn.covariance import LedoitWolf

                lw = LedoitWolf()
                lw.fit(values)
                cov = lw.covariance_
            except (ImportError, ValueError, RuntimeError):
                cov = np.cov(values, rowvar=False)
        else:
            cov = np.cov(values, rowvar=False)

        cov = np.asarray(cov, dtype=float)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=float)
        elif cov.ndim == 1:
            cov = np.diag(cov)

        # Additional diagonal shrinkage for numerical stability.
        diag = np.diag(np.diag(cov))
        cov = (1.0 - self.shrinkage) * cov + self.shrinkage * diag

        # Enforce PSD with small eigenvalue flooring.
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-10)
        cov_psd = (eigvecs * eigvals) @ eigvecs.T
        return cov_psd


# ---------------------------------------------------------------------------
# Regime-conditional covariance utilities
# ---------------------------------------------------------------------------

def compute_regime_covariance(
    returns: pd.DataFrame,
    regimes: pd.Series,
    min_obs: int = 30,
    shrinkage: float = 0.1,
) -> Dict[int, pd.DataFrame]:
    """Compute separate covariance matrices for each market regime.

    For each unique regime value in *regimes*, the corresponding rows of
    *returns* are selected and a sample covariance matrix is computed.
    Ledoit-Wolf-style diagonal shrinkage is applied:

        ``cov = (1 - shrinkage) * sample_cov + shrinkage * diag(sample_cov)``

    If a regime has fewer than *min_obs* observations, the full-sample
    covariance (computed across all regimes) is used as a fallback.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns with a DatetimeIndex (or similar) and one column per
        asset.
    regimes : pd.Series
        Integer regime labels aligned to *returns*.  Only the intersection
        of both indices is used.
    min_obs : int, default 30
        Minimum number of observations required to estimate a
        regime-specific covariance.
    shrinkage : float, default 0.1
        Shrinkage intensity toward the diagonal (0 = no shrinkage, 1 = pure
        diagonal).

    Returns
    -------
    Dict[int, pd.DataFrame]
        Mapping from regime label to its covariance matrix (assets x assets).
    """
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))

    # Align returns and regimes on their shared index.
    common_idx = returns.index.intersection(regimes.index)
    returns = returns.loc[common_idx]
    regimes = regimes.loc[common_idx]

    # Clean returns: replace infinities, forward/back-fill, drop remaining NaN.
    clean = returns.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna(how="any")
    regimes = regimes.reindex(clean.index)

    assets = clean.columns

    # ------------------------------------------------------------------
    # Full-sample covariance as a fallback for thin regimes.
    # ------------------------------------------------------------------
    full_sample_cov = clean.cov()
    full_diag = pd.DataFrame(
        np.diag(np.diag(full_sample_cov.values)),
        index=assets,
        columns=assets,
    )
    full_cov_shrunk = (1.0 - shrinkage) * full_sample_cov + shrinkage * full_diag

    # ------------------------------------------------------------------
    # Per-regime covariance
    # ------------------------------------------------------------------
    regime_covs: Dict[int, pd.DataFrame] = {}

    for regime_label in sorted(regimes.unique()):
        mask = regimes == regime_label
        regime_returns = clean.loc[mask]

        if len(regime_returns) < min_obs:
            regime_covs[int(regime_label)] = full_cov_shrunk.copy()
            continue

        sample_cov = regime_returns.cov()
        diag_cov = pd.DataFrame(
            np.diag(np.diag(sample_cov.values)),
            index=assets,
            columns=assets,
        )
        regime_covs[int(regime_label)] = (
            (1.0 - shrinkage) * sample_cov + shrinkage * diag_cov
        )

    return regime_covs


def get_regime_covariance(
    regime_covs: Dict[int, pd.DataFrame],
    current_regime: int,
) -> pd.DataFrame:
    """Return the covariance matrix for *current_regime*.

    If *current_regime* is not present in *regime_covs*, the function
    falls back to regime 0.  If regime 0 is also missing it returns
    whichever regime covariance is available (lowest key).

    Parameters
    ----------
    regime_covs : Dict[int, pd.DataFrame]
        Mapping produced by :func:`compute_regime_covariance`.
    current_regime : int
        The active regime label.

    Returns
    -------
    pd.DataFrame
        The covariance matrix for the requested (or fallback) regime.

    Raises
    ------
    ValueError
        If *regime_covs* is empty.
    """
    if not regime_covs:
        raise ValueError("regime_covs is empty — no covariance matrices available")

    if current_regime in regime_covs:
        return regime_covs[current_regime]

    # Fallback: regime 0, then lowest available key.
    if 0 in regime_covs:
        return regime_covs[0]

    fallback_key = min(regime_covs.keys())
    return regime_covs[fallback_key]
