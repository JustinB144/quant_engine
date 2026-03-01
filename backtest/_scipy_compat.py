"""
Shared scipy import fallback for validation modules.

Centralises the optional scipy dependency so that backtest/validation.py
and backtest/advanced_validation.py don't each duplicate the full fallback
class hierarchy.
"""
from math import erf

import numpy as np
import pandas as pd

try:
    from scipy import stats as sp_stats
    from scipy.optimize import minimize as sp_minimize
except ImportError:  # pragma: no cover - optional dependency fallback
    sp_minimize = None  # type: ignore[assignment]

    class _NormFallback:
        @staticmethod
        def cdf(x):
            """cdf."""
            arr = np.asarray(x, dtype=float)
            vals = 0.5 * (1.0 + np.vectorize(erf)(arr / np.sqrt(2.0)))
            if np.isscalar(x):
                return float(vals)
            return vals

    class _StatsFallback:
        norm = _NormFallback()

        @staticmethod
        def spearmanr(a, b):
            """spearmanr."""
            x = np.asarray(a, dtype=float)
            y = np.asarray(b, dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 2:
                return np.nan, np.nan

            rx = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=float)
            ry = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)
            rx = rx - rx.mean()
            ry = ry - ry.mean()
            denom = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
            if denom <= 1e-12:
                return np.nan, np.nan
            return float(np.sum(rx * ry) / denom), np.nan

        @staticmethod
        def ttest_1samp(a, popmean):
            """ttest 1samp."""
            x = np.asarray(a, dtype=float)
            x = x[np.isfinite(x)]
            n = len(x)
            if n < 2:
                return np.nan, np.nan

            delta = float(np.mean(x) - popmean)
            std = float(np.std(x, ddof=1))
            if std <= 1e-12:
                if abs(delta) <= 1e-12:
                    return 0.0, 1.0
                return (np.inf if delta > 0 else -np.inf), 0.0

            tstat = delta / (std / np.sqrt(n))
            # Normal approximation for optional SciPy fallback path.
            p_two_sided = 2.0 * (1.0 - _NormFallback.cdf(abs(tstat)))
            return float(tstat), float(p_two_sided)

        @staticmethod
        def skew(a):
            """skew."""
            x = np.asarray(a, dtype=float)
            x = x[np.isfinite(x)]
            if len(x) < 3:
                return 0.0
            mu = float(np.mean(x))
            sigma = float(np.std(x, ddof=0))
            if sigma <= 1e-12:
                return 0.0
            z = (x - mu) / sigma
            return float(np.mean(z**3))

        @staticmethod
        def kurtosis(a):
            """kurtosis."""
            x = np.asarray(a, dtype=float)
            x = x[np.isfinite(x)]
            if len(x) < 4:
                return 0.0
            mu = float(np.mean(x))
            sigma = float(np.std(x, ddof=0))
            if sigma <= 1e-12:
                return 0.0
            z = (x - mu) / sigma
            return float(np.mean(z**4) - 3.0)

    sp_stats = _StatsFallback()  # type: ignore[assignment]


def require_scipy() -> None:
    """Raise ImportError if scipy is not available."""
    if sp_stats is None or isinstance(sp_stats, type) and not hasattr(sp_stats, '__module__'):
        raise ImportError("scipy is required for statistical validation")
