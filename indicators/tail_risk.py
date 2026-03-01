"""
Tail Risk & Jump Detection Indicators — extreme event analysis.

Detects price jumps (discontinuities), computes expected shortfall,
vol-of-vol, and downside/upside asymmetry. These features capture
tail risk dynamics that standard volatility measures miss.

All features are CAUSAL — they use only past and current data.
"""

import numpy as np
import pandas as pd


class TailRiskAnalyzer:
    """Detect jumps, extreme value statistics, and tail risk.

    Produces five features:
        - Jump intensity: fraction of bars with |z-score| > threshold
        - Expected shortfall: average of worst alpha% returns (CVaR)
        - Vol-of-vol: standard deviation of rolling volatilities
        - Semi-relative modulus: downside/upside semi-variance ratio
        - Extreme return percentage: fraction of bars with |return| > threshold

    Parameters
    ----------
    window : int
        Rolling lookback window in bars.
    jump_threshold : float
        Z-score threshold for flagging jumps (default 2.5 sigma).
    """

    def __init__(
        self,
        window: int = 20,
        jump_threshold: float = 2.5,
    ):
        if window < 5:
            raise ValueError(f"window must be >= 5, got {window}")
        if jump_threshold <= 0:
            raise ValueError(f"jump_threshold must be > 0, got {jump_threshold}")
        self.window = window
        self.jump_threshold = jump_threshold

    def compute_jump_intensity(self, returns: np.ndarray) -> np.ndarray:
        """Fraction of bars with returns exceeding jump_threshold sigma.

        High jump intensity indicates frequent large price moves (earnings
        shocks, gap events). Low intensity indicates stable, continuous
        price evolution.

        Parameters
        ----------
        returns : np.ndarray
            Log returns (or simple returns).

        Returns
        -------
        jump_intensity : np.ndarray
            Values in [0, 1], same length as returns.
        """
        n = len(returns)
        jump_intensity = np.full(n, np.nan)

        for i in range(self.window, n):
            seg = returns[i - self.window: i]
            mu = np.mean(seg)
            sigma = np.std(seg, ddof=1)

            if sigma > 1e-15:
                standardized = np.abs((seg - mu) / sigma)
                jump_intensity[i] = np.mean(standardized > self.jump_threshold)
            else:
                jump_intensity[i] = 0.0

        return jump_intensity

    def compute_expected_shortfall(
        self,
        returns: np.ndarray,
        alpha: float = 0.05,
    ) -> np.ndarray:
        """Expected Shortfall (CVaR) — average of worst alpha% returns.

        More sensitive to tail events than VaR. Measures the expected loss
        conditional on being in the worst alpha fraction of outcomes.

        Parameters
        ----------
        returns : np.ndarray
            Log returns (or simple returns).
        alpha : float
            Tail fraction (default 0.05 = worst 5%).

        Returns
        -------
        es : np.ndarray
            Expected shortfall (negative values = losses).
        """
        n = len(returns)
        es = np.full(n, np.nan)

        for i in range(self.window, n):
            seg = returns[i - self.window: i]
            k = max(1, int(np.floor(alpha * len(seg))))
            if k >= len(seg):
                k = len(seg) - 1
            sorted_returns = np.sort(seg)  # ascending
            es[i] = float(np.mean(sorted_returns[:k]))

        return es

    def compute_vol_of_vol(self, returns: np.ndarray) -> np.ndarray:
        """Volatility of volatility — captures vol clustering and bursty regimes.

        Computed as the standard deviation of sub-window rolling volatilities
        within the main window.

        High vol-of-vol: erratic volatility (jump risk, regime instability).
        Low vol-of-vol: steady volatility (stable regime).

        Parameters
        ----------
        returns : np.ndarray
            Log returns (or simple returns).

        Returns
        -------
        vov : np.ndarray
            Vol-of-vol values.
        """
        n = len(returns)
        vov = np.full(n, np.nan)

        # Inner window for sub-volatilities (quarter of main window, min 5)
        inner_window = max(5, self.window // 4)

        # Pre-compute rolling sub-volatilities
        rolling_vols = np.full(n, np.nan)
        for i in range(inner_window, n):
            rolling_vols[i] = np.std(returns[i - inner_window: i], ddof=1)

        # Outer rolling std of sub-volatilities
        for i in range(self.window, n):
            vols_seg = rolling_vols[i - self.window: i]
            valid_vols = vols_seg[~np.isnan(vols_seg)]
            if len(valid_vols) >= 3:
                vov[i] = np.std(valid_vols, ddof=1)

        return vov

    def compute_semi_relative_modulus(self, returns: np.ndarray) -> np.ndarray:
        """Downside/upside semi-variance ratio (asymmetry measure).

        Values > 1 indicate more downside risk than upside potential.
        Values < 1 indicate upside-skewed returns.
        Values near 1 indicate symmetric return distribution.

        Parameters
        ----------
        returns : np.ndarray
            Log returns (or simple returns).

        Returns
        -------
        srm : np.ndarray
            Semi-relative modulus (non-negative).
        """
        n = len(returns)
        srm = np.full(n, np.nan)

        for i in range(self.window, n):
            seg = returns[i - self.window: i]

            downside = seg[seg < 0]
            upside = seg[seg > 0]

            down_var = np.var(downside, ddof=1) if len(downside) > 1 else 0.0
            up_var = np.var(upside, ddof=1) if len(upside) > 1 else 0.0

            if up_var > 1e-15:
                srm[i] = np.sqrt(down_var / up_var)
            else:
                # No upside variance — extreme downside dominance
                srm[i] = np.sqrt(down_var) / 1e-10 if down_var > 0 else 0.0

        return srm

    def compute_extreme_return_pct(
        self,
        returns: np.ndarray,
        threshold: float = 0.02,
    ) -> np.ndarray:
        """Fraction of bars with |return| exceeding an absolute threshold.

        Unlike jump_intensity (which uses z-score relative to local vol),
        this uses a fixed threshold. Captures absolute extremes regardless
        of prevailing volatility.

        Parameters
        ----------
        returns : np.ndarray
            Log returns (or simple returns).
        threshold : float
            Absolute return threshold (default 0.02 = 2%).

        Returns
        -------
        extreme_pct : np.ndarray
            Values in [0, 1].
        """
        n = len(returns)
        extreme_pct = np.full(n, np.nan)

        for i in range(self.window, n):
            seg = returns[i - self.window: i]
            extreme_pct[i] = np.mean(np.abs(seg) > threshold)

        return extreme_pct

    def compute_all(self, returns: np.ndarray) -> dict:
        """Compute all tail risk features.

        Parameters
        ----------
        returns : np.ndarray
            Log returns (or simple returns).

        Returns
        -------
        dict
            Keys: 'jump_intensity', 'expected_shortfall', 'vol_of_vol',
                  'semi_relative_modulus', 'extreme_return_pct'
        """
        return {
            "jump_intensity": self.compute_jump_intensity(returns),
            "expected_shortfall": self.compute_expected_shortfall(returns),
            "vol_of_vol": self.compute_vol_of_vol(returns),
            "semi_relative_modulus": self.compute_semi_relative_modulus(returns),
            "extreme_return_pct": self.compute_extreme_return_pct(returns),
        }
