"""
Factor Exposure Manager — compute and enforce regime-conditioned factor bounds.

Computes portfolio-level factor exposures (beta, size, value, momentum,
volatility) and checks them against regime-conditioned bounds from
``config_data/universe.yaml``.  Factors with ``null`` bounds are monitored
but not constrained.

Integrates with ``PortfolioRiskManager`` to provide factor exposure
checks as part of the pre-trade risk workflow.

Spec 07, Task 4.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .universe_config import UniverseConfig

logger = logging.getLogger(__name__)


class FactorExposureManager:
    """Compute and check portfolio factor exposures against regime bounds.

    Parameters
    ----------
    universe_config : UniverseConfig, optional
        Configuration source for factor bounds.  If None, uses defaults.
    lookback : int
        Number of trading days for factor computation (default 252).
    beta_lookback : int
        Number of trading days for beta estimation (default 60).
    """

    # Factors that are always computed
    FACTOR_NAMES = ("beta", "size", "value", "momentum", "volatility")

    def __init__(
        self,
        universe_config: Optional[UniverseConfig] = None,
        lookback: int = 252,
        beta_lookback: int = 60,
    ):
        self._config = universe_config
        self._lookback = lookback
        self._beta_lookback = beta_lookback

    def compute_exposures(
        self,
        weights: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """Compute portfolio-level factor exposures.

        Parameters
        ----------
        weights : dict
            {ticker: portfolio_weight} of current holdings.
        price_data : dict
            {ticker: OHLCV DataFrame} with at least 'Close' and 'Volume'.
        benchmark_data : pd.DataFrame, optional
            Benchmark OHLCV (e.g. SPY) for beta computation.

        Returns
        -------
        Dict[str, float]
            Factor exposures: {factor_name: weighted_exposure}.
        """
        if not weights:
            return {}

        exposures: Dict[str, float] = {}
        position_data: List[Dict] = []

        for ticker, weight in weights.items():
            if ticker not in price_data or abs(weight) < 1e-8:
                continue
            ohlcv = price_data[ticker]
            if len(ohlcv) < 30:
                continue

            close = ohlcv["Close"].iloc[-self._lookback:]
            volume = ohlcv.get("Volume", pd.Series(0, index=close.index))
            volume = volume.iloc[-self._lookback:]
            returns = close.pct_change().dropna()
            if len(returns) < 20:
                continue

            # Size proxy: log average daily dollar volume
            dollar_vol = (close * volume).mean()
            log_dollar_vol = float(np.log1p(max(dollar_vol, 1.0)))

            # Realized volatility (annualized, 20-day)
            recent_ret = returns.iloc[-20:]
            realized_vol = float(recent_ret.std() * np.sqrt(252)) if len(recent_ret) > 5 else 0.20

            # Momentum: 12m - 1m return (or shorter approximation)
            if len(close) >= 252:
                mom_12m = float(close.iloc[-1] / close.iloc[-252] - 1)
                mom_1m = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else 0.0
                momentum = mom_12m - mom_1m
            elif len(close) >= 63:
                momentum = float(close.iloc[-1] / close.iloc[-63] - 1)
            else:
                momentum = 0.0

            # Value proxy: inverse of price-to-trailing-return ratio
            # Higher value = cheaper stock (lower P/R ratio)
            if len(close) >= 252:
                trailing_return = float(close.iloc[-1] / close.iloc[-252] - 1)
                value_proxy = -trailing_return  # Negative return = cheap = high value
            else:
                value_proxy = 0.0

            # Beta vs benchmark
            beta = 1.0  # Default
            if benchmark_data is not None and len(benchmark_data) > 30:
                beta = self._compute_single_beta(
                    returns, benchmark_data, self._beta_lookback,
                )

            position_data.append({
                "ticker": ticker,
                "weight": weight,
                "beta": beta,
                "log_dollar_vol": log_dollar_vol,
                "realized_vol": realized_vol,
                "momentum": momentum,
                "value_proxy": value_proxy,
            })

        if not position_data:
            return exposures

        signed_weights = np.array([p["weight"] for p in position_data])
        abs_weights = np.abs(signed_weights)
        total_abs_weight = abs_weights.sum()
        if total_abs_weight < 1e-8:
            return exposures

        norm_weights = abs_weights / total_abs_weight

        # Beta: signed weight contribution (shorts contribute negative beta)
        betas = np.array([p["beta"] for p in position_data])
        exposures["beta"] = float(np.sum(signed_weights * betas) / total_abs_weight)

        # Size: z-score of log dollar volume (relative to portfolio universe)
        log_dvols = np.array([p["log_dollar_vol"] for p in position_data])
        if len(log_dvols) > 1 and log_dvols.std() > 1e-6:
            z_scores = (log_dvols - log_dvols.mean()) / log_dvols.std()
            exposures["size"] = float(np.average(z_scores, weights=norm_weights))
        else:
            exposures["size"] = 0.0

        # Value: z-score of value proxy
        vals = np.array([p["value_proxy"] for p in position_data])
        if len(vals) > 1 and vals.std() > 1e-6:
            z_scores = (vals - vals.mean()) / vals.std()
            exposures["value"] = float(np.average(z_scores, weights=norm_weights))
        else:
            exposures["value"] = 0.0

        # Momentum: z-score of momentum
        moms = np.array([p["momentum"] for p in position_data])
        if len(moms) > 1 and moms.std() > 1e-6:
            z_scores = (moms - moms.mean()) / moms.std()
            exposures["momentum"] = float(np.average(z_scores, weights=norm_weights))
        else:
            exposures["momentum"] = 0.0

        # Volatility: weighted average realized vol (ratio to portfolio mean)
        vols = np.array([p["realized_vol"] for p in position_data])
        vol_mean = vols.mean()
        if vol_mean > 1e-6:
            exposures["volatility"] = float(np.average(vols / vol_mean, weights=norm_weights))
        else:
            exposures["volatility"] = 1.0

        return exposures

    @staticmethod
    def is_stress_regime(regime: int) -> bool:
        """Return True if regime is a stress state (2 or 3)."""
        return regime in (2, 3)

    def check_factor_bounds(
        self,
        exposures: Dict[str, float],
        regime: int = 0,
    ) -> Tuple[bool, Dict[str, str]]:
        """Check factor exposures against regime-conditioned bounds.

        Parameters
        ----------
        exposures : dict
            Factor exposures from :meth:`compute_exposures`.
        regime : int
            Current regime label (0-3).

        Returns
        -------
        tuple[bool, dict]
            ``(all_passed, violations)`` where violations maps factor name
            to a human-readable description of the violation.
        """
        is_stress = self.is_stress_regime(regime)

        violations: Dict[str, str] = {}

        for factor in self.FACTOR_NAMES:
            val = exposures.get(factor)
            if val is None:
                continue

            bounds = self._get_bounds(factor, is_stress)
            if bounds is None:
                continue  # Monitored only, not constrained

            lo, hi = bounds
            if val < lo:
                violations[factor] = (
                    f"{factor}={val:.3f} below lower bound {lo:.3f} "
                    f"({'stress' if is_stress else 'normal'} regime)"
                )
            elif val > hi:
                violations[factor] = (
                    f"{factor}={val:.3f} above upper bound {hi:.3f} "
                    f"({'stress' if is_stress else 'normal'} regime)"
                )

        return len(violations) == 0, violations

    def _get_bounds(
        self, factor: str, is_stress: bool,
    ) -> Optional[Tuple[float, float]]:
        """Get factor bounds from config, falling back to defaults."""
        if self._config is not None:
            return self._config.get_factor_bounds(factor, is_stress)

        # Default bounds when no config available
        defaults = {
            "beta": {"normal": (0.8, 1.2), "stress": (0.9, 1.1)},
            "volatility": {"normal": (0.8, 1.2), "stress": (0.5, 1.0)},
        }
        regime_key = "stress" if is_stress else "normal"
        return defaults.get(factor, {}).get(regime_key)

    @staticmethod
    def _compute_single_beta(
        asset_returns: pd.Series,
        benchmark_data: pd.DataFrame,
        lookback: int,
    ) -> float:
        """Compute beta of a single asset vs benchmark.

        Both covariance and variance are computed over the same overlapping
        date range to avoid mismatched denominators when assets have data gaps.
        """
        bench_returns = benchmark_data["Close"].pct_change().iloc[-lookback:]

        # Align asset and benchmark to common dates
        common = asset_returns.dropna().index.intersection(bench_returns.dropna().index)
        if len(common) < 20:
            return 1.0

        asset_aligned = asset_returns.loc[common]
        bench_aligned = bench_returns.loc[common]

        bench_var = bench_aligned.var()
        if bench_var < 1e-14:
            return 1.0

        cov = asset_aligned.cov(bench_aligned)
        return float(cov / bench_var)
