"""
Correlation Regime Detection (NEW 11).

Detects regime changes in the correlation structure of the asset universe.
This captures a different dimension of market behaviour from the HMM-based
regime detector, which focuses on return and volatility regimes.

During market stress, pairwise correlations among equities tend to spike
(the "correlation breakdown" effect), meaning diversification fails exactly
when it is needed most.  This module flags those episodes so that
downstream models and the portfolio optimizer can react accordingly.

Components:
    CorrelationRegimeDetector:
        - compute_rolling_correlation: rolling average pairwise correlation
        - detect_correlation_spike: flags high-correlation regimes
        - get_correlation_features: produces a DataFrame ready for the
          feature pipeline (avg_pairwise_corr, corr_regime, corr_z_score)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


class CorrelationRegimeDetector:
    """Detect regime changes in pairwise correlation structure.

    Parameters
    ----------
    window : int
        Rolling window (trading days) for computing pairwise correlations.
        Default is 63 (one calendar quarter).
    z_score_lookback : int
        Lookback (trading days) for computing the z-score of the rolling
        average correlation.  Default is 252 (one year).
    threshold : float
        Average pairwise correlation above which a "correlation spike" regime
        is flagged.  Default is 0.70.
    """

    def __init__(
        self,
        window: int = 63,
        z_score_lookback: int = 252,
        threshold: float = 0.70,
    ) -> None:
        """Initialize CorrelationRegimeDetector."""
        self.window = window
        self.z_score_lookback = z_score_lookback
        self.threshold = threshold
        self._avg_corr: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_rolling_correlation(
        self,
        returns_dict: Dict[str, pd.Series],
        window: Optional[int] = None,
    ) -> pd.Series:
        """Compute rolling average pairwise correlation across the universe.

        For each rolling window, the full pairwise correlation matrix of
        asset returns is computed and the *average off-diagonal* element is
        returned.  This gives a single time-series summarising how
        "correlated" the universe is at each point in time.

        Parameters
        ----------
        returns_dict : dict[str, pd.Series]
            Mapping of asset identifier to daily return series.
        window : int, optional
            Override the instance-level rolling window.

        Returns
        -------
        pd.Series
            Rolling average pairwise correlation with a DatetimeIndex.
        """
        win = window if window is not None else self.window

        # Align all return series into a single DataFrame.
        ret_df = pd.DataFrame(returns_dict).sort_index().dropna(how="all")

        if ret_df.shape[1] < 2:
            # Need at least 2 assets for pairwise correlation.
            self._avg_corr = pd.Series(np.nan, index=ret_df.index, name="avg_pairwise_corr")
            return self._avg_corr

        n_assets = ret_df.shape[1]
        n_pairs = n_assets * (n_assets - 1) / 2

        # Rolling average pairwise correlation.
        # For each rolling window, compute the correlation matrix and take
        # the mean of the upper-triangle (off-diagonal) entries.
        avg_corr_values = pd.Series(np.nan, index=ret_df.index, name="avg_pairwise_corr")

        min_periods = max(3, win // 2)

        for end_idx in range(win, len(ret_df) + 1):
            start_idx = end_idx - win
            block = ret_df.iloc[start_idx:end_idx]

            # Drop assets with too few valid observations in this window.
            valid_cols = block.columns[block.notna().sum() >= min_periods]
            if len(valid_cols) < 2:
                continue

            corr_matrix = block[valid_cols].corr().values
            # Extract upper-triangle (excluding diagonal).
            n = len(valid_cols)
            upper_tri = corr_matrix[np.triu_indices(n, k=1)]
            avg_corr_values.iloc[end_idx - 1] = float(np.nanmean(upper_tri))

        self._avg_corr = avg_corr_values
        return self._avg_corr

    def detect_correlation_spike(
        self,
        threshold: Optional[float] = None,
    ) -> pd.Series:
        """Flag dates where average correlation exceeds the threshold.

        Must call :meth:`compute_rolling_correlation` first.

        Parameters
        ----------
        threshold : float, optional
            Override the instance-level threshold.

        Returns
        -------
        pd.Series
            Binary series: 1 if in a correlation spike regime, 0 otherwise.

        Raises
        ------
        RuntimeError
            If ``compute_rolling_correlation`` has not been called yet.
        """
        if self._avg_corr is None:
            raise RuntimeError(
                "Call compute_rolling_correlation() before detect_correlation_spike()."
            )

        thr = threshold if threshold is not None else self.threshold
        return (self._avg_corr >= thr).astype(int).rename("corr_regime")

    def get_correlation_features(
        self,
        returns_dict: Dict[str, pd.Series],
        window: Optional[int] = None,
        threshold: Optional[float] = None,
        z_score_lookback: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return a DataFrame with all correlation regime features.

        This is the main entry-point for the feature pipeline.

        Columns:
            avg_pairwise_corr : float
                Rolling average pairwise correlation.
            corr_regime : int
                1 if in a correlation spike, 0 otherwise.
            corr_z_score : float
                Z-score of the current average correlation relative to a
                longer-term rolling window (default 252 days).

        Parameters
        ----------
        returns_dict : dict[str, pd.Series]
            Mapping of asset identifier to daily return series.
        window : int, optional
            Rolling window for pairwise correlations (overrides instance).
        threshold : float, optional
            Spike threshold (overrides instance).
        z_score_lookback : int, optional
            Lookback for z-score computation (overrides instance).

        Returns
        -------
        pd.DataFrame
            DatetimeIndex with columns [avg_pairwise_corr, corr_regime,
            corr_z_score].
        """
        z_lb = z_score_lookback if z_score_lookback is not None else self.z_score_lookback

        # Step 1: compute rolling average pairwise correlation.
        avg_corr = self.compute_rolling_correlation(returns_dict, window=window)

        # Step 2: detect spike regime.
        regime = self.detect_correlation_spike(threshold=threshold)

        # Step 3: z-score relative to a longer lookback.
        rolling_mean = avg_corr.rolling(z_lb, min_periods=max(20, z_lb // 4)).mean()
        rolling_std = avg_corr.rolling(z_lb, min_periods=max(20, z_lb // 4)).std()
        z_score = ((avg_corr - rolling_mean) / rolling_std.replace(0, np.nan)).rename(
            "corr_z_score"
        )

        features = pd.DataFrame(
            {
                "avg_pairwise_corr": avg_corr,
                "corr_regime": regime,
                "corr_z_score": z_score,
            }
        )
        return features
