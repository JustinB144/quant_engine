"""
Covariance estimation utilities for portfolio risk controls.

Supports three estimation methods:

- ``ledoit_wolf`` (default): sklearn's data-driven Ledoit-Wolf shrinkage.
  The optimal shrinkage intensity is determined automatically from the data
  (no manual shrinkage parameter needed).  Falls back to manual diagonal
  shrinkage if sklearn is unavailable.

- ``ewma``: Exponentially weighted moving average covariance.  Weights
  recent observations more heavily via a configurable half-life, making
  the estimator more responsive to regime changes.

- ``sample``: Plain sample covariance with optional manual diagonal
  shrinkage for regularisation.
"""
from dataclasses import dataclass
from typing import Dict
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CovarianceEstimate:
    """Covariance estimation output bundle with metadata about the fit method and sample count."""
    covariance: pd.DataFrame
    n_observations: int
    method: str
    shrinkage_intensity: float = 0.0


class CovarianceEstimator:
    """
    Estimate a robust covariance matrix for asset returns.

    Parameters
    ----------
    method : str
        Estimation method: ``"ledoit_wolf"``, ``"ewma"``, or ``"sample"``.
    shrinkage : float
        Manual shrinkage intensity (0-1).  Used as fallback for Ledoit-Wolf
        if sklearn fails, and as the diagonal shrinkage for ``"sample"``
        method.  Ignored when Ledoit-Wolf succeeds (data-driven shrinkage
        is used instead).
    annualization : float
        Factor to annualise the covariance (default 252 for daily data).
    half_life : int
        Half-life in observations for EWMA covariance (default 60).
        Only used when ``method="ewma"``.
    """

    def __init__(
        self,
        method: str = "ledoit_wolf",
        shrinkage: float = 0.15,
        annualization: float = 252.0,
        half_life: int = 60,
    ):
        """Initialize CovarianceEstimator."""
        self.method = method
        self.shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
        self.annualization = float(max(1.0, annualization))
        self.half_life = max(1, int(half_life))

    def estimate(self, returns: pd.DataFrame) -> CovarianceEstimate:
        """Estimate a robust covariance matrix from asset returns.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily (or periodic) asset returns.  Columns are assets, rows are
            time observations.

        Returns
        -------
        CovarianceEstimate
            Contains the estimated covariance matrix, observation count,
            method used, and the shrinkage intensity applied.
        """
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
        cov_values, shrinkage_used = self._estimate_values(clean)
        cov = pd.DataFrame(cov_values, index=clean.columns, columns=clean.columns)
        return CovarianceEstimate(
            covariance=cov,
            n_observations=int(clean.shape[0]),
            method=self.method,
            shrinkage_intensity=shrinkage_used,
        )

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

    def _estimate_values(self, clean: pd.DataFrame) -> tuple:
        """Internal helper — estimate covariance values and return shrinkage used.

        Parameters
        ----------
        clean : pd.DataFrame
            Cleaned returns (no NaN/inf, ffill/bfill already applied).

        Returns
        -------
        tuple[np.ndarray, float]
            (covariance_matrix, shrinkage_intensity)
        """
        values = clean.values
        if values.ndim != 2:
            raise ValueError("returns array must be 2D")

        n_assets = int(values.shape[1])
        shrinkage_used = 0.0

        if n_assets == 1:
            series = values[:, 0].astype(float)
            ddof = 1 if len(series) > 1 else 0
            var = float(np.var(series, ddof=ddof))
            return np.array([[max(var, 1e-10)]], dtype=float), 0.0

        if self.method == "ledoit_wolf":
            try:
                from sklearn.covariance import LedoitWolf

                lw = LedoitWolf()
                lw.fit(values)
                cov = lw.covariance_
                shrinkage_used = float(lw.shrinkage_)
                logger.debug("Ledoit-Wolf data-driven shrinkage: %.4f", shrinkage_used)
            except (ImportError, ValueError, RuntimeError):
                # Fallback: manual diagonal shrinkage on sample covariance
                logger.warning("Ledoit-Wolf failed, falling back to manual shrinkage=%.2f", self.shrinkage)
                cov = np.cov(values, rowvar=False)
                shrinkage_used = self.shrinkage
                diag = np.diag(np.diag(cov))
                cov = (1.0 - shrinkage_used) * cov + shrinkage_used * diag

        elif self.method == "ewma":
            cov, shrinkage_used = self._estimate_ewma(clean)

        else:
            # Sample covariance with manual diagonal shrinkage
            cov = np.cov(values, rowvar=False)
            shrinkage_used = self.shrinkage
            diag = np.diag(np.diag(cov))
            cov = (1.0 - shrinkage_used) * cov + shrinkage_used * diag

        cov = np.asarray(cov, dtype=float)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=float)
        elif cov.ndim == 1:
            cov = np.diag(cov)

        # Enforce PSD with small eigenvalue flooring.
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-10)
        cov_psd = (eigvecs * eigvals) @ eigvecs.T
        return cov_psd, shrinkage_used

    def _estimate_ewma(self, clean: pd.DataFrame) -> tuple:
        """Exponentially weighted covariance estimation.

        Uses pandas EWM to weight recent observations more heavily.
        During regime changes the EWMA estimator reacts faster than
        equal-weighted sample covariance.

        Parameters
        ----------
        clean : pd.DataFrame
            Cleaned returns DataFrame.

        Returns
        -------
        tuple[np.ndarray, float]
            (covariance_matrix, 0.0) — EWMA has no shrinkage intensity.
        """
        n_obs, n_assets = clean.shape

        # pandas ewm().cov() returns a MultiIndex DataFrame (date x asset, asset).
        # We extract the final cross-section (last date's covariance block).
        ewma_cov = clean.ewm(halflife=self.half_life).cov()
        last_date = clean.index[-1]
        cov = ewma_cov.loc[last_date].values.astype(float)

        logger.debug("EWMA covariance with half_life=%d on %d observations", self.half_life, n_obs)
        return cov, 0.0


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
