"""
Eigenvalue Spectrum Indicators — portfolio-level systemic risk analysis.

Analyzes the eigenvalue spectrum of rolling correlation matrices to detect
transitions from diversified (many independent components) to concentrated
(few dominant components) regimes. Concentration predicts volatility spikes
and systemic stress.

These features require multi-asset return data and are computed at the
universe level in FeaturePipeline.compute_universe().

All features are CAUSAL — they use only past and current data.
"""

import numpy as np
from typing import Dict, Optional


class EigenvalueAnalyzer:
    """Eigenvalue spectrum analysis of rolling correlation matrices.

    Produces four features (all cross-sectional, broadcast to each asset):
        - Eigenvalue concentration (HHI): systemic risk measure
        - Effective rank: number of independent components
        - Average correlation stress: mean pairwise correlation
        - Spectral condition number: ill-conditioning of correlation matrix

    Parameters
    ----------
    window : int
        Rolling correlation window in trading days.
    min_assets : int
        Minimum number of assets required to compute features.
    regularization : float
        Tikhonov regularization added to diagonal of correlation matrix
        for numerical stability (prevents singular matrices).
    """

    def __init__(
        self,
        window: int = 60,
        min_assets: int = 5,
        regularization: float = 0.01,
    ):
        if window < 10:
            raise ValueError(f"window must be >= 10, got {window}")
        self.window = window
        self.min_assets = min_assets
        self.regularization = regularization

    def _compute_correlation_matrix(
        self,
        returns_segment: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Compute regularized correlation matrix with numerical safeguards.

        Parameters
        ----------
        returns_segment : np.ndarray of shape (window, n_assets)
            Return matrix for one rolling window.

        Returns
        -------
        corr : np.ndarray of shape (n_assets, n_assets) or None
            Regularized correlation matrix, or None if degenerate.
        """
        n_assets = returns_segment.shape[1]

        # Check for constant columns (zero-variance assets)
        stds = np.std(returns_segment, axis=0, ddof=1)
        valid_mask = stds > 1e-12
        if np.sum(valid_mask) < self.min_assets:
            return None

        # Compute correlation on valid assets
        valid_rets = returns_segment[:, valid_mask]
        try:
            corr = np.corrcoef(valid_rets.T)
        except (FloatingPointError, ValueError):
            return None

        # Handle NaN in correlation (can arise from constant columns)
        if np.any(np.isnan(corr)):
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)

        # Tikhonov regularization for numerical stability
        corr = (1.0 - self.regularization) * corr + self.regularization * np.eye(
            corr.shape[0]
        )

        return corr

    def _eigenvalues_safe(self, corr: np.ndarray) -> np.ndarray:
        """Compute eigenvalues with fallback for degenerate matrices.

        Returns sorted non-negative eigenvalues.
        """
        try:
            eigvals = np.linalg.eigvalsh(corr)
        except np.linalg.LinAlgError:
            return np.ones(corr.shape[0]) / corr.shape[0]

        # Ensure non-negative (numerical roundoff can produce small negatives)
        return np.maximum(eigvals, 0.0)

    def compute_eigenvalue_concentration(
        self,
        returns_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Herfindahl-Hirschman Index (HHI) of eigenvalue spectrum.

        HHI = sum(λ_i^2) / (sum(λ_i))^2, where λ_i are eigenvalues.
        Ranges from 1/N (perfectly diversified) to 1 (single factor).

        High HHI (> 0.5): concentrated / systemic risk.
        Low HHI (< 0.3): diversified regime.

        Parameters
        ----------
        returns_dict : Dict[str, np.ndarray]
            Mapping from ticker to return array (all same length).

        Returns
        -------
        concentration : np.ndarray
            HHI values, indexed by bar position.
        """
        tickers = sorted(returns_dict.keys())
        if len(tickers) < self.min_assets:
            n_bars = len(next(iter(returns_dict.values()))) if returns_dict else 0
            return np.full(n_bars, np.nan)

        all_rets = np.column_stack([returns_dict[t] for t in tickers])
        n_bars = all_rets.shape[0]
        concentration = np.full(n_bars, np.nan)

        for i in range(self.window, n_bars):
            seg = all_rets[i - self.window: i]
            corr = self._compute_correlation_matrix(seg)
            if corr is None:
                continue

            eigvals = self._eigenvalues_safe(corr)
            total = np.sum(eigvals)
            if total > 1e-15:
                eigvals_norm = eigvals / total
                concentration[i] = np.sum(eigvals_norm ** 2)

        return concentration

    def compute_effective_rank(
        self,
        returns_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Effective rank: exp(entropy of normalized eigenvalues).

        Measures the effective number of independent components in the
        correlation structure. Low rank indicates few driving factors.

        Parameters
        ----------
        returns_dict : Dict[str, np.ndarray]
            Mapping from ticker to return array.

        Returns
        -------
        eff_rank : np.ndarray
            Effective rank values.
        """
        tickers = sorted(returns_dict.keys())
        if len(tickers) < self.min_assets:
            n_bars = len(next(iter(returns_dict.values()))) if returns_dict else 0
            return np.full(n_bars, np.nan)

        all_rets = np.column_stack([returns_dict[t] for t in tickers])
        n_bars = all_rets.shape[0]
        eff_rank = np.full(n_bars, np.nan)

        for i in range(self.window, n_bars):
            seg = all_rets[i - self.window: i]
            corr = self._compute_correlation_matrix(seg)
            if corr is None:
                continue

            eigvals = self._eigenvalues_safe(corr)
            total = np.sum(eigvals)
            if total > 1e-15:
                eigvals_norm = eigvals / total
                # Shannon entropy of eigenvalue distribution
                safe = np.where(eigvals_norm > 1e-30, eigvals_norm, 1e-30)
                entropy = -np.sum(safe * np.log(safe))
                eff_rank[i] = np.exp(entropy)

        return eff_rank

    def compute_avg_correlation_stress(
        self,
        returns_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Average pairwise correlation (systemic stress indicator).

        High avg correlation (> 0.6): assets moving together (stress).
        Low avg correlation (< 0.2): diversified regime.

        Parameters
        ----------
        returns_dict : Dict[str, np.ndarray]
            Mapping from ticker to return array.

        Returns
        -------
        avg_corr : np.ndarray
            Average upper-triangle pairwise correlations.
        """
        tickers = sorted(returns_dict.keys())
        if len(tickers) < self.min_assets:
            n_bars = len(next(iter(returns_dict.values()))) if returns_dict else 0
            return np.full(n_bars, np.nan)

        all_rets = np.column_stack([returns_dict[t] for t in tickers])
        n_bars = all_rets.shape[0]
        avg_corr = np.full(n_bars, np.nan)

        for i in range(self.window, n_bars):
            seg = all_rets[i - self.window: i]

            # Use raw corrcoef (not regularized) for stress measurement
            stds = np.std(seg, axis=0, ddof=1)
            valid_mask = stds > 1e-12
            if np.sum(valid_mask) < self.min_assets:
                continue

            try:
                corr = np.corrcoef(seg[:, valid_mask].T)
            except (FloatingPointError, ValueError):
                continue

            if np.any(np.isnan(corr)):
                corr = np.nan_to_num(corr, nan=0.0)

            # Upper triangle only (exclude diagonal)
            triu_indices = np.triu_indices_from(corr, k=1)
            if len(triu_indices[0]) > 0:
                avg_corr[i] = np.mean(corr[triu_indices])

        return avg_corr

    def compute_spectral_condition_number(
        self,
        returns_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Condition number: ratio of largest to smallest eigenvalue.

        High condition number indicates ill-conditioned correlation matrix
        (multicollinearity, numerical instability). Useful for detecting
        when portfolio optimization becomes unreliable.

        Parameters
        ----------
        returns_dict : Dict[str, np.ndarray]
            Mapping from ticker to return array.

        Returns
        -------
        cond_num : np.ndarray
            Log condition numbers (log-scaled for stability).
        """
        tickers = sorted(returns_dict.keys())
        if len(tickers) < self.min_assets:
            n_bars = len(next(iter(returns_dict.values()))) if returns_dict else 0
            return np.full(n_bars, np.nan)

        all_rets = np.column_stack([returns_dict[t] for t in tickers])
        n_bars = all_rets.shape[0]
        cond_num = np.full(n_bars, np.nan)

        for i in range(self.window, n_bars):
            seg = all_rets[i - self.window: i]
            corr = self._compute_correlation_matrix(seg)
            if corr is None:
                continue

            eigvals = self._eigenvalues_safe(corr)
            min_eig = np.min(eigvals)
            max_eig = np.max(eigvals)

            # Log condition number for numerical stability
            if min_eig > 1e-15:
                cond_num[i] = np.log(max_eig / min_eig)
            else:
                cond_num[i] = np.log(max_eig / 1e-10)

        return cond_num

    def compute_all(
        self,
        returns_dict: Dict[str, np.ndarray],
    ) -> dict:
        """Compute all eigenvalue features in a single pass.

        More efficient than calling individual methods since correlation
        matrix and eigenvalues are computed once per window.

        Parameters
        ----------
        returns_dict : Dict[str, np.ndarray]
            Mapping from ticker to return array.

        Returns
        -------
        dict
            Keys: 'eigenvalue_concentration', 'effective_rank',
                  'avg_correlation_stress', 'condition_number'
        """
        tickers = sorted(returns_dict.keys())
        if len(tickers) < self.min_assets:
            n_bars = len(next(iter(returns_dict.values()))) if returns_dict else 0
            empty = np.full(n_bars, np.nan)
            return {
                "eigenvalue_concentration": empty.copy(),
                "effective_rank": empty.copy(),
                "avg_correlation_stress": empty.copy(),
                "condition_number": empty.copy(),
            }

        all_rets = np.column_stack([returns_dict[t] for t in tickers])
        n_bars = all_rets.shape[0]

        concentration = np.full(n_bars, np.nan)
        eff_rank = np.full(n_bars, np.nan)
        avg_corr = np.full(n_bars, np.nan)
        cond_num = np.full(n_bars, np.nan)

        for i in range(self.window, n_bars):
            seg = all_rets[i - self.window: i]

            # Check valid assets
            stds = np.std(seg, axis=0, ddof=1)
            valid_mask = stds > 1e-12
            if np.sum(valid_mask) < self.min_assets:
                continue

            valid_seg = seg[:, valid_mask]

            # Raw correlation for avg_corr
            try:
                raw_corr = np.corrcoef(valid_seg.T)
            except (FloatingPointError, ValueError):
                continue

            if np.any(np.isnan(raw_corr)):
                raw_corr = np.nan_to_num(raw_corr, nan=0.0)
                np.fill_diagonal(raw_corr, 1.0)

            # Average pairwise correlation
            triu_idx = np.triu_indices_from(raw_corr, k=1)
            if len(triu_idx[0]) > 0:
                avg_corr[i] = np.mean(raw_corr[triu_idx])

            # Regularized correlation for eigenvalue analysis
            n_valid = np.sum(valid_mask)
            reg_corr = (1.0 - self.regularization) * raw_corr + \
                self.regularization * np.eye(n_valid)

            try:
                eigvals = np.linalg.eigvalsh(reg_corr)
            except np.linalg.LinAlgError:
                continue
            eigvals = np.maximum(eigvals, 0.0)

            total = np.sum(eigvals)
            if total < 1e-15:
                continue

            # HHI concentration
            eigvals_norm = eigvals / total
            concentration[i] = np.sum(eigvals_norm ** 2)

            # Effective rank
            safe = np.where(eigvals_norm > 1e-30, eigvals_norm, 1e-30)
            entropy = -np.sum(safe * np.log(safe))
            eff_rank[i] = np.exp(entropy)

            # Log condition number
            min_eig = np.min(eigvals)
            max_eig = np.max(eigvals)
            if min_eig > 1e-15:
                cond_num[i] = np.log(max_eig / min_eig)
            else:
                cond_num[i] = np.log(max_eig / 1e-10)

        return {
            "eigenvalue_concentration": concentration,
            "effective_rank": eff_rank,
            "avg_correlation_stress": avg_corr,
            "condition_number": cond_num,
        }
