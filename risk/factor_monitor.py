"""
Factor Exposure Monitoring — track portfolio factor tilts and alert on violations.

Monitors unintended factor exposures across market beta, size, momentum,
volatility, and liquidity dimensions.  Uses simple factor proxies computable
from daily price data (no external factor model required).

Integration points:
    - API: /api/risk/factor-exposures endpoint
    - Autopilot: check factor limits before promotion
    - Health dashboard: factor tilt warnings

Factor proxies (all computable from price data):
    - Market beta:      rolling covariance with SPY / SPY variance
    - Size tilt:        log(avg daily dollar volume) z-score vs universe
    - Momentum tilt:    12m-1m return z-score vs universe
    - Volatility tilt:  20d realized vol z-score vs universe
    - Liquidity tilt:   avg dollar volume z-score vs universe
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FactorExposure:
    """Measured factor exposure for the portfolio."""
    factor: str
    value: float
    limit_low: float
    limit_high: float
    in_bounds: bool
    description: str = ""


@dataclass
class FactorExposureReport:
    """Complete factor exposure report for a portfolio."""
    exposures: Dict[str, FactorExposure]
    all_passed: bool
    violations: List[str]
    timestamp: str = ""


# Default exposure limits
DEFAULT_EXPOSURE_LIMITS = {
    "market_beta": (0.5, 1.5),       # Should be near 1.0 for equity long-only
    "size_zscore": (-1.5, 1.5),      # Avoid extreme size tilts
    "momentum_zscore": (-2.0, 2.0),  # Avoid excessive momentum chasing
    "volatility_zscore": (-1.5, 1.5),# Avoid extreme vol concentration
    "liquidity_zscore": (-1.5, 1.5), # Avoid extreme liquidity mismatch
}


class FactorExposureMonitor:
    """
    Track portfolio factor exposures and alert on unintended tilts.

    Uses simple factor proxies computable from price data.  No external
    factor model database required.

    Usage:
        monitor = FactorExposureMonitor()
        report = monitor.compute_report(
            positions={"AAPL": 0.10, "MSFT": 0.08, ...},
            price_data={"AAPL": ohlcv_df, "MSFT": ohlcv_df, ...},
            benchmark_returns=spy_returns_series,
        )
        if not report.all_passed:
            for violation in report.violations:
                alert(violation)
    """

    def __init__(
        self,
        limits: Optional[Dict[str, Tuple[float, float]]] = None,
        lookback_days: int = 252,
        beta_lookback: int = 60,
    ):
        """
        Args:
            limits: {factor_name: (low, high)} exposure limits.
            lookback_days: Days of history for factor proxy computation.
            beta_lookback: Days for rolling beta estimation.
        """
        self.limits = limits or DEFAULT_EXPOSURE_LIMITS.copy()
        self.lookback_days = lookback_days
        self.beta_lookback = beta_lookback

    def compute_exposures(
        self,
        positions: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
        benchmark_returns: Optional[pd.Series] = None,
        universe_stats: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """Compute portfolio factor exposures from positions and price data.

        Args:
            positions: {ticker: portfolio_weight} of current holdings.
            price_data: {ticker: OHLCV DataFrame} with at least 'Close'
                and 'Volume' columns.
            benchmark_returns: Optional benchmark return series for beta
                computation.
            universe_stats: Optional universe-level statistics for proper
                z-scoring. Dict of {factor_name: {"mean": float, "std": float}}.
                If provided, z-scores use universe-level mean/std instead of
                within-portfolio statistics (which are less meaningful).

        Returns:
            Dict of {factor_name: exposure_value}.
        """
        if not positions:
            return {}

        exposures: Dict[str, float] = {}

        # Compute per-position metrics
        position_metrics: List[Dict] = []
        for ticker, weight in positions.items():
            if ticker not in price_data or abs(weight) < 1e-8:
                continue
            ohlcv = price_data[ticker]
            if len(ohlcv) < 30:
                continue

            close = ohlcv["Close"].iloc[-self.lookback_days:]
            volume = ohlcv.get("Volume", pd.Series(0, index=close.index))
            volume = volume.iloc[-self.lookback_days:]

            returns = close.pct_change().dropna()
            if len(returns) < 20:
                continue

            # Daily dollar volume proxy (close * volume)
            dollar_vol = (close * volume).mean()

            # Realized volatility (annualized, 20-day)
            recent_ret = returns.iloc[-20:]
            realized_vol = float(recent_ret.std() * np.sqrt(252)) if len(recent_ret) > 5 else 0.20

            # Momentum: 12m - 1m return (if enough data)
            if len(close) >= 252:
                mom_12m = float(close.iloc[-1] / close.iloc[-252] - 1)
                mom_1m = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else 0.0
                momentum = mom_12m - mom_1m
            elif len(close) >= 63:
                momentum = float(close.iloc[-1] / close.iloc[-63] - 1)
            else:
                momentum = 0.0

            position_metrics.append({
                "ticker": ticker,
                "weight": weight,
                "returns": returns,
                "dollar_volume": dollar_vol,
                "realized_vol": realized_vol,
                "momentum": momentum,
            })

        if not position_metrics:
            return exposures

        # ── Market Beta ──
        if benchmark_returns is not None and len(benchmark_returns) > 20:
            portfolio_returns = self._compute_portfolio_returns(
                position_metrics, benchmark_returns.index,
            )
            if len(portfolio_returns) > 20:
                # Align
                common = portfolio_returns.index.intersection(benchmark_returns.index)
                if len(common) > 20:
                    port_r = portfolio_returns.loc[common]
                    bench_r = benchmark_returns.loc[common]
                    cov = port_r.cov(bench_r)
                    var = bench_r.var()
                    exposures["market_beta"] = float(cov / var) if var > 1e-12 else 1.0

        # ── Size tilt (dollar volume proxy) ──
        dollar_vols = np.array([m["dollar_volume"] for m in position_metrics])
        weights = np.array([abs(m["weight"]) for m in position_metrics])
        if len(dollar_vols) > 1 and dollar_vols.std() > 1e-6:
            log_dvols = np.log1p(np.maximum(dollar_vols, 1.0))
            if universe_stats is not None and "size_zscore" in universe_stats:
                u_mean = universe_stats["size_zscore"]["mean"]
                u_std = universe_stats["size_zscore"]["std"]
            else:
                # Fallback: within-portfolio z-scores (less meaningful)
                u_mean = log_dvols.mean()
                u_std = log_dvols.std()
            if u_std > 1e-6:
                zscores = (log_dvols - u_mean) / u_std
                exposures["size_zscore"] = float(np.average(zscores, weights=weights))

        # ── Momentum tilt ──
        momentums = np.array([m["momentum"] for m in position_metrics])
        if len(momentums) > 1 and momentums.std() > 1e-6:
            if universe_stats is not None and "momentum_zscore" in universe_stats:
                mom_mean = universe_stats["momentum_zscore"]["mean"]
                mom_std = universe_stats["momentum_zscore"]["std"]
            else:
                mom_mean = momentums.mean()
                mom_std = momentums.std()
            if mom_std > 1e-6:
                mom_z = (momentums - mom_mean) / mom_std
                exposures["momentum_zscore"] = float(np.average(mom_z, weights=weights))

        # ── Volatility tilt ──
        vols = np.array([m["realized_vol"] for m in position_metrics])
        if len(vols) > 1 and vols.std() > 1e-6:
            if universe_stats is not None and "volatility_zscore" in universe_stats:
                vol_mean = universe_stats["volatility_zscore"]["mean"]
                vol_std = universe_stats["volatility_zscore"]["std"]
            else:
                vol_mean = vols.mean()
                vol_std = vols.std()
            if vol_std > 1e-6:
                vol_z = (vols - vol_mean) / vol_std
                exposures["volatility_zscore"] = float(np.average(vol_z, weights=weights))

        # ── Liquidity tilt ──
        if len(dollar_vols) > 1 and dollar_vols.std() > 1e-6:
            if universe_stats is not None and "liquidity_zscore" in universe_stats:
                liq_mean = universe_stats["liquidity_zscore"]["mean"]
                liq_std = universe_stats["liquidity_zscore"]["std"]
            else:
                liq_mean = dollar_vols.mean()
                liq_std = dollar_vols.std()
            if liq_std > 1e-6:
                liq_z = (dollar_vols - liq_mean) / liq_std
                exposures["liquidity_zscore"] = float(np.average(liq_z, weights=weights))

        return exposures

    def check_limits(
        self,
        exposures: Dict[str, float],
        limits: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> FactorExposureReport:
        """Check if any exposure exceeds configured limits.

        Args:
            exposures: {factor_name: value} from compute_exposures.
            limits: Optional override limits.

        Returns:
            FactorExposureReport with pass/fail status and violations.
        """
        limits = limits or self.limits
        factor_exposures: Dict[str, FactorExposure] = {}
        violations: List[str] = []

        for factor, (lo, hi) in limits.items():
            val = exposures.get(factor)
            if val is None:
                factor_exposures[factor] = FactorExposure(
                    factor=factor,
                    value=0.0,
                    limit_low=lo,
                    limit_high=hi,
                    in_bounds=True,
                    description="Not computed (insufficient data)",
                )
                continue

            in_bounds = lo <= val <= hi
            factor_exposures[factor] = FactorExposure(
                factor=factor,
                value=val,
                limit_low=lo,
                limit_high=hi,
                in_bounds=in_bounds,
            )
            if not in_bounds:
                msg = f"{factor}={val:.2f} outside [{lo}, {hi}]"
                violations.append(msg)
                logger.warning("Factor constraint violation: %s", msg)

        return FactorExposureReport(
            exposures=factor_exposures,
            all_passed=len(violations) == 0,
            violations=violations,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def compute_report(
        self,
        positions: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
        benchmark_returns: Optional[pd.Series] = None,
    ) -> FactorExposureReport:
        """One-call method: compute exposures and check limits.

        Args:
            positions: {ticker: weight}.
            price_data: {ticker: OHLCV DataFrame}.
            benchmark_returns: Optional benchmark returns.

        Returns:
            Complete FactorExposureReport.
        """
        exposures = self.compute_exposures(positions, price_data, benchmark_returns)
        return self.check_limits(exposures)

    def _compute_portfolio_returns(
        self,
        position_metrics: List[Dict],
        target_index: pd.DatetimeIndex,
    ) -> pd.Series:
        """Compute weight-adjusted portfolio returns."""
        total_weight = sum(abs(m["weight"]) for m in position_metrics)
        if total_weight < 1e-8:
            return pd.Series(dtype=float)

        portfolio_returns = pd.Series(0.0, index=target_index, dtype=float)
        for m in position_metrics:
            r = m["returns"].reindex(target_index).fillna(0.0)
            w = m["weight"] / total_weight
            portfolio_returns += r * w

        return portfolio_returns.dropna()
