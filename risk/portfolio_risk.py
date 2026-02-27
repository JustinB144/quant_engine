"""
Portfolio Risk Manager — enforces sector, correlation, and exposure limits.

Prevents concentration risk across multiple dimensions:
    - Sector/industry exposure caps (regime-conditioned via Spec 07)
    - Pairwise correlation limits (regime-conditional covariance)
    - Beta exposure limits
    - Gross/net exposure limits
    - Single-name concentration
    - Factor exposure bounds (beta, volatility, size, value, momentum)
    - Continuous sizing backoff when constraints approach binding
    - Smooth constraint transitions across regime changes

Spec 07: Regime-Conditioned Portfolio Risk Manager
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import MAX_PORTFOLIO_VOL, CORRELATION_STRESS_THRESHOLDS
from .covariance import CovarianceEstimator, compute_regime_covariance, get_regime_covariance
from .factor_exposures import FactorExposureManager
from .universe_config import UniverseConfig, ConfigError

logger = logging.getLogger(__name__)


# Legacy sector map — used as fallback when UniverseConfig is unavailable.
_LEGACY_SECTOR_MAP = {
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "META": "tech",
    "NVDA": "tech", "AMD": "tech", "INTC": "tech", "CRM": "tech",
    "ADBE": "tech", "ORCL": "tech",
    "DDOG": "tech", "NET": "tech", "CRWD": "tech", "ZS": "tech",
    "SNOW": "tech", "MDB": "tech", "PANW": "tech", "FTNT": "tech",
    "JNJ": "healthcare", "PFE": "healthcare", "UNH": "healthcare",
    "ABBV": "healthcare", "MRK": "healthcare", "LLY": "healthcare",
    "TMO": "healthcare", "ABT": "healthcare",
    "AMZN": "consumer", "TSLA": "consumer", "HD": "consumer",
    "NKE": "consumer", "SBUX": "consumer", "MCD": "consumer",
    "TGT": "consumer", "COST": "consumer",
    "JPM": "financial", "BAC": "financial", "GS": "financial",
    "MS": "financial", "BLK": "financial", "V": "financial", "MA": "financial",
    "CAT": "industrial", "DE": "industrial", "GE": "industrial",
    "HON": "industrial", "BA": "industrial", "LMT": "industrial",
    "CAVA": "consumer", "BROS": "consumer", "TOST": "tech",
    "CHWY": "consumer", "ETSY": "consumer", "POOL": "consumer",
}


@dataclass
class RiskCheck:
    """Result of a portfolio risk check."""
    passed: bool
    violations: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    constraint_utilization: Dict[str, float] = field(default_factory=dict)
    recommended_weights: Optional[np.ndarray] = None


class ConstraintMultiplier:
    """Compute regime-conditioned constraint multipliers.

    In normal regimes (0=trending_bull, 1=trending_bear), multipliers are 1.0.
    In stress regimes (2=mean_reverting, 3=high_volatility), multipliers tighten
    constraints by the factors specified in ``universe.yaml``.

    Smooth transitions use exponential smoothing across regime changes to
    avoid abrupt position liquidations (Spec 07, Task 7).
    """

    def __init__(self, universe_config: Optional[UniverseConfig] = None):
        if universe_config is not None:
            self._normal_mults = universe_config.get_stress_multiplier_set(is_stress=False)
            self._stress_mults = universe_config.get_stress_multiplier_set(is_stress=True)
        else:
            self._normal_mults = {
                "sector_cap": 1.0, "correlation_limit": 1.0,
                "gross_exposure": 1.0, "turnover": 1.0,
            }
            self._stress_mults = {
                "sector_cap": 0.6, "correlation_limit": 0.7,
                "gross_exposure": 0.8, "turnover": 0.5,
            }
        # Initialize smoothed multipliers to normal
        self._smoothed: Dict[str, float] = dict(self._normal_mults)
        self._prev_regime_is_stress: Optional[bool] = None

    @staticmethod
    def is_stress_regime(regime: int) -> bool:
        """Return True if the integer regime represents a stress state.

        Regime mapping (from ``config.REGIME_NAMES``):
            0 = trending_bull  (normal)
            1 = trending_bear  (normal — directional, not structural stress)
            2 = mean_reverting (stress — structural break/rotation)
            3 = high_volatility (stress — crisis/tail)
        """
        return regime in (2, 3)

    def get_multipliers(self, regime: int) -> Dict[str, float]:
        """Return raw (unsmoothed) constraint multipliers for *regime*."""
        if self.is_stress_regime(regime):
            return dict(self._stress_mults)
        return dict(self._normal_mults)

    def get_multipliers_smoothed(
        self, regime: int, alpha: float = 0.3,
    ) -> Dict[str, float]:
        """Return exponentially smoothed constraint multipliers.

        Smoothing prevents step discontinuities when regime changes:
        ``smoothed = alpha * current + (1 - alpha) * previous``

        Parameters
        ----------
        regime : int
            Current regime label (0-3).
        alpha : float
            Smoothing factor.  Default 0.3 gives ~2-day half-life.
            Can be overridden via ``CONSTRAINT_MULTIPLIER_SMOOTHING_ALPHA`` env var.

        Returns
        -------
        Dict[str, float]
            Smoothed multipliers for each constraint dimension.
        """
        import os
        alpha = float(os.environ.get("CONSTRAINT_MULTIPLIER_SMOOTHING_ALPHA", alpha))
        alpha = max(0.0, min(1.0, alpha))

        target = self.get_multipliers(regime)
        is_stress = self.is_stress_regime(regime)

        # On first call, initialize directly (no smoothing)
        if self._prev_regime_is_stress is None:
            self._smoothed = dict(target)
            self._prev_regime_is_stress = is_stress
            return dict(self._smoothed)

        # Apply exponential smoothing
        for key in target:
            prev = self._smoothed.get(key, target[key])
            self._smoothed[key] = alpha * target[key] + (1.0 - alpha) * prev

        self._prev_regime_is_stress = is_stress
        return dict(self._smoothed)

    def reset(self) -> None:
        """Reset smoothed state (e.g., at start of a new backtest)."""
        self._smoothed = dict(self._normal_mults)
        self._prev_regime_is_stress = None


class PortfolioRiskManager:
    """
    Enforces portfolio-level risk constraints with regime conditioning.

    Checks run before each trade entry to ensure the portfolio
    stays within predefined risk bounds.  Constraint thresholds adapt
    to the detected market regime via ``ConstraintMultiplier``.

    Spec 07: Regime-Conditioned Portfolio Risk Manager.
    """

    def __init__(
        self,
        max_sector_pct: float = 0.40,
        max_corr_between: float = 0.85,
        max_gross_exposure: float = 1.0,
        max_single_name_pct: float = 0.10,
        max_beta_exposure: float = 1.5,
        max_portfolio_vol: float = MAX_PORTFOLIO_VOL,
        correlation_lookback: int = 60,
        covariance_method: str = "ledoit_wolf",
        sector_map: Optional[Dict[str, str]] = None,
        universe_config: Optional[UniverseConfig] = None,
    ):
        """Initialize PortfolioRiskManager.

        Parameters
        ----------
        universe_config : UniverseConfig, optional
            Centralized universe metadata.  If None, falls back to the
            legacy hardcoded sector map and default constraints.
        """
        # Base constraint thresholds (may be overridden by universe_config)
        self.max_sector_pct = max_sector_pct
        self.max_corr = max_corr_between
        self.max_gross = max_gross_exposure
        self.max_single = max_single_name_pct
        self.max_beta = max_beta_exposure
        self.max_portfolio_vol = max_portfolio_vol
        self.corr_lookback = correlation_lookback

        # Universe config (centralized metadata from YAML)
        self.universe_config = universe_config
        if universe_config is None:
            try:
                self.universe_config = UniverseConfig()
            except ConfigError:
                logger.info(
                    "UniverseConfig not available — using legacy sector map and default constraints."
                )

        # Load base constraints from universe config if available
        if self.universe_config is not None:
            cb = self.universe_config.constraint_base
            self.max_sector_pct = cb.get("sector_cap", max_sector_pct)
            self.max_corr = cb.get("correlation_limit", max_corr_between)
            self.max_gross = cb.get("gross_exposure", max_gross_exposure)
            self.max_single = cb.get("single_name_cap", max_single_name_pct)

        # Sector lookup: universe_config takes precedence over legacy map
        if sector_map is not None:
            self.sector_map = {str(k).upper(): str(v) for k, v in sector_map.items()}
        elif self.universe_config is not None:
            self.sector_map = {}  # Will use universe_config.get_sector()
        else:
            self.sector_map = {str(k).upper(): str(v) for k, v in _LEGACY_SECTOR_MAP.items()}

        self.covariance_estimator = CovarianceEstimator(method=covariance_method)

        # Regime-conditioned constraint multiplier
        self.multiplier = ConstraintMultiplier(self.universe_config)

        # Factor exposure manager (Spec 07 T4)
        self.factor_exposure_manager = FactorExposureManager(
            universe_config=self.universe_config,
        )

        # Cache for regime-conditional covariance matrices
        self._regime_cov_cache: Dict[int, pd.DataFrame] = {}

    @staticmethod
    def _infer_ticker_from_price_df(df: Optional[pd.DataFrame]) -> Optional[str]:
        """Internal helper for infer ticker from price df."""
        if df is None or len(df) == 0:
            return None
        attrs_ticker = str(df.attrs.get("ticker", "")).upper().strip()
        if attrs_ticker:
            return attrs_ticker
        if "ticker" in df.columns:
            s = df["ticker"].dropna()
            if len(s) > 0:
                return str(s.iloc[-1]).upper().strip()
        return None

    def _resolve_sector(self, asset_id: str, price_data: Dict[str, pd.DataFrame]) -> str:
        """
        Resolve sector for a PERMNO-first key, falling back to ticker metadata.

        Checks ``universe_config`` first (centralized YAML), then the legacy
        ``sector_map`` dict, then infers from price DataFrame attrs.
        """
        key = str(asset_id).upper().strip()

        # Prefer UniverseConfig (centralized YAML metadata)
        if self.universe_config is not None:
            sector = self.universe_config.get_sector(key)
            if sector != "other":
                return sector

        # Legacy sector_map fallback
        if key in self.sector_map:
            return self.sector_map[key]

        # Infer ticker from price DataFrame metadata
        df = price_data.get(str(asset_id))
        mapped_ticker = self._infer_ticker_from_price_df(df)
        if mapped_ticker:
            if self.universe_config is not None:
                sector = self.universe_config.get_sector(mapped_ticker)
                if sector != "other":
                    return sector
            if mapped_ticker in self.sector_map:
                return self.sector_map[mapped_ticker]

        return "other"

    def check_new_position(
        self,
        ticker: str,
        position_size: float,
        current_positions: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
        benchmark_data: Optional[pd.DataFrame] = None,
        regime: Optional[int] = None,
        regime_labels: Optional[pd.Series] = None,
    ) -> RiskCheck:
        """
        Check if adding a new position violates any risk constraints.

        When *regime* is provided, constraint thresholds are adjusted by
        regime-conditioned multipliers (tighter in stress regimes).

        Args:
            ticker: proposed new position
            position_size: proposed size as fraction of capital
            current_positions: {ticker: size_pct} of existing positions
            price_data: OHLCV data for all tickers
            benchmark_data: SPY OHLCV for beta calculation
            regime: current regime label (0-3) for constraint conditioning
            regime_labels: per-bar regime labels for regime-conditional covariance

        Returns:
            RiskCheck with pass/fail, violation details, constraint utilization,
            and recommended (backoff-adjusted) weights.
        """
        violations = []
        metrics = {}
        constraint_util = {}
        proposed_positions = {**current_positions, ticker: position_size}

        # ── Compute effective constraints (regime-conditioned) ──
        eff_sector_cap = self.max_sector_pct
        eff_corr_limit = self.max_corr
        eff_gross = self.max_gross
        eff_single = self.max_single

        if regime is not None:
            mults = self.multiplier.get_multipliers_smoothed(regime)
            eff_sector_cap = self.max_sector_pct * mults.get("sector_cap", 1.0)
            eff_corr_limit = self.max_corr * mults.get("correlation_limit", 1.0)
            eff_gross = self.max_gross * mults.get("gross_exposure", 1.0)
            metrics["regime"] = regime
            metrics["constraint_multipliers"] = mults

        # ── SPEC-P03: Dynamic correlation-based constraint tightening ──
        # Compute average pairwise correlation across all proposed positions
        # and apply additional tightening when correlation is elevated.
        # This fires faster than regime detection, catching correlation
        # spikes before the regime detector reclassifies.
        avg_corr = self._compute_avg_pairwise_correlation(
            proposed_positions, price_data,
        )
        metrics["avg_pairwise_corr"] = avg_corr

        corr_stress_mult = self._get_correlation_stress_multiplier(avg_corr)
        if corr_stress_mult < 1.0:
            eff_sector_cap *= corr_stress_mult
            eff_single *= corr_stress_mult
            eff_gross *= corr_stress_mult
            metrics["corr_stress_multiplier"] = corr_stress_mult
            logger.info(
                "Correlation stress tightening: avg_corr=%.3f, multiplier=%.2f",
                avg_corr, corr_stress_mult,
            )

        # ── Single-name check ──
        if position_size > eff_single:
            violations.append(
                f"Position size {position_size:.1%} > max {eff_single:.1%}"
            )
        if eff_single > 0:
            constraint_util["single_name"] = position_size / eff_single

        # ── Gross exposure check ──
        gross = sum(proposed_positions.values())
        metrics["gross_exposure"] = gross
        if eff_gross > 0:
            constraint_util["gross_exposure"] = gross / eff_gross
        if gross > eff_gross:
            violations.append(
                f"Gross exposure {gross:.1%} > max {eff_gross:.1%}"
            )

        # ── Sector concentration check ──
        sector = self._resolve_sector(ticker, price_data)
        sector_exposure = position_size
        for t, s in current_positions.items():
            if self._resolve_sector(t, price_data) == sector:
                sector_exposure += s
        metrics["sector_exposure"] = {sector: sector_exposure}
        if eff_sector_cap > 0:
            constraint_util["sector_cap"] = sector_exposure / eff_sector_cap
        if sector_exposure > eff_sector_cap:
            violations.append(
                f"Sector '{sector}' exposure {sector_exposure:.1%} > max {eff_sector_cap:.1%}"
            )

        # ── Correlation check (regime-conditional covariance) ──
        if ticker in price_data and current_positions:
            max_corr_found = self._check_correlations(
                ticker, list(current_positions.keys()), price_data,
                regime=regime, regime_labels=regime_labels,
            )
            metrics["max_pairwise_corr"] = max_corr_found
            if eff_corr_limit > 0:
                constraint_util["correlation"] = max_corr_found / eff_corr_limit
            if max_corr_found > eff_corr_limit:
                violations.append(
                    f"Max pairwise correlation {max_corr_found:.2f} > {eff_corr_limit:.2f}"
                )

        # ── Beta exposure check ──
        if benchmark_data is not None:
            portfolio_beta = self._estimate_portfolio_beta(
                proposed_positions,
                price_data, benchmark_data,
            )
            metrics["portfolio_beta"] = portfolio_beta
            if abs(portfolio_beta) > self.max_beta:
                violations.append(
                    f"Portfolio beta {portfolio_beta:.2f} > max {self.max_beta:.2f}"
                )

        # ── Portfolio volatility check ──
        portfolio_vol = self._estimate_portfolio_vol(
            positions=proposed_positions,
            price_data=price_data,
        )
        metrics["portfolio_vol"] = portfolio_vol
        if portfolio_vol > self.max_portfolio_vol:
            violations.append(
                f"Portfolio vol {portfolio_vol:.1%} > max {self.max_portfolio_vol:.1%}"
            )

        # ── Factor exposure check (Spec 07 T4) ──
        factor_result = self.check_factor_exposures(
            positions=proposed_positions,
            price_data=price_data,
            benchmark_data=benchmark_data,
            regime=regime,
        )
        metrics["factor_exposures"] = factor_result["exposures"]
        if factor_result["violations"]:
            for factor, msg in factor_result["violations"].items():
                violations.append(f"Factor exposure: {msg}")

        # ── Compute sizing backoff recommendation ──
        recommended_weights = None
        if constraint_util:
            max_util = max(constraint_util.values())
            if max_util > 0.70:
                backoff_factor = self._compute_backoff_factor(max_util)
                recommended_weights = np.array([
                    v * backoff_factor for v in proposed_positions.values()
                ])
                metrics["backoff_factor"] = backoff_factor
                metrics["max_constraint_utilization"] = max_util

        return RiskCheck(
            passed=len(violations) == 0,
            violations=violations,
            metrics=metrics,
            constraint_utilization=constraint_util,
            recommended_weights=recommended_weights,
        )

    def compute_constraint_utilization(
        self,
        positions: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
        regime: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute utilization ratios for all constraints.

        Utilization = current_value / limit.  Values > 1.0 indicate violations.

        Parameters
        ----------
        positions : dict
            {ticker: weight} of current portfolio.
        price_data : dict
            {ticker: OHLCV DataFrame}.
        regime : int, optional
            Current regime for conditioned limits.

        Returns
        -------
        Dict[str, float]
            Utilization ratios keyed by constraint name.
        """
        util = {}

        # Effective limits
        eff_sector_cap = self.max_sector_pct
        eff_corr_limit = self.max_corr
        eff_gross = self.max_gross

        if regime is not None:
            mults = self.multiplier.get_multipliers_smoothed(regime)
            eff_sector_cap *= mults.get("sector_cap", 1.0)
            eff_corr_limit *= mults.get("correlation_limit", 1.0)
            eff_gross *= mults.get("gross_exposure", 1.0)

        # SPEC-P03: correlation-based constraint tightening
        avg_corr = self._compute_avg_pairwise_correlation(positions, price_data)
        corr_stress_mult = self._get_correlation_stress_multiplier(avg_corr)
        if corr_stress_mult < 1.0:
            eff_sector_cap *= corr_stress_mult
            eff_gross *= corr_stress_mult

        # Gross
        gross = sum(positions.values())
        if eff_gross > 0:
            util["gross_exposure"] = gross / eff_gross

        # Single name
        eff_single = self.max_single * corr_stress_mult
        if positions and eff_single > 0:
            util["single_name"] = max(positions.values()) / eff_single

        # Sector
        sector_weights: Dict[str, float] = {}
        for t, w in positions.items():
            sector = self._resolve_sector(t, price_data)
            sector_weights[sector] = sector_weights.get(sector, 0.0) + w
        if sector_weights and eff_sector_cap > 0:
            util["sector_cap"] = max(sector_weights.values()) / eff_sector_cap

        return util

    def check_factor_exposures(
        self,
        positions: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
        benchmark_data: Optional[pd.DataFrame] = None,
        regime: Optional[int] = None,
    ) -> Dict:
        """Compute factor exposures and check against regime-conditioned bounds.

        Parameters
        ----------
        positions : dict
            {ticker: weight} of portfolio positions.
        price_data : dict
            {ticker: OHLCV DataFrame}.
        benchmark_data : pd.DataFrame, optional
            Benchmark OHLCV (e.g. SPY) for beta computation.
        regime : int, optional
            Current regime label (0-3).  If None, defaults to 0 (normal).

        Returns
        -------
        dict
            ``{"exposures": Dict[str, float], "passed": bool,
            "violations": Dict[str, str]}``
        """
        exposures = self.factor_exposure_manager.compute_exposures(
            weights=positions,
            price_data=price_data,
            benchmark_data=benchmark_data,
        )

        effective_regime = regime if regime is not None else 0
        passed, violations = self.factor_exposure_manager.check_factor_bounds(
            exposures, regime=effective_regime,
        )

        return {
            "exposures": exposures,
            "passed": passed,
            "violations": violations,
        }

    def _compute_backoff_factor(self, max_utilization: float) -> float:
        """Compute continuous backoff factor from constraint utilization.

        Uses the backoff policy from universe config: when utilization
        crosses threshold levels, position sizes are scaled down by
        the corresponding factor.

        Parameters
        ----------
        max_utilization : float
            Maximum constraint utilization ratio across all dimensions.

        Returns
        -------
        float
            Scaling factor in (0, 1] to apply to position sizes.
        """
        if self.universe_config is not None:
            policy = self.universe_config.backoff_policy
        else:
            policy = {
                "thresholds": [0.70, 0.80, 0.90, 0.95],
                "backoff_factors": [0.9, 0.7, 0.5, 0.25],
            }

        thresholds = policy.get("thresholds", [0.70, 0.80, 0.90, 0.95])
        factors = policy.get("backoff_factors", [0.9, 0.7, 0.5, 0.25])

        backoff = 1.0
        for threshold, factor in zip(thresholds, factors):
            if max_utilization >= threshold:
                backoff = factor
            else:
                break

        return backoff

    def _check_correlations(
        self,
        new_ticker: str,
        existing_tickers: List[str],
        price_data: Dict[str, pd.DataFrame],
        regime: Optional[int] = None,
        regime_labels: Optional[pd.Series] = None,
    ) -> float:
        """Find max correlation between new ticker and existing positions.

        When *regime* and *regime_labels* are provided, uses
        ``compute_regime_covariance()`` to estimate the correlation structure
        for the current regime.  Falls back to pairwise sample correlation
        when regime-conditional estimation is unavailable.
        """
        if new_ticker not in price_data:
            return 0.0

        # Attempt regime-conditional covariance
        if regime is not None and regime_labels is not None:
            try:
                return self._check_correlations_regime(
                    new_ticker, existing_tickers, price_data,
                    regime, regime_labels,
                )
            except (ValueError, KeyError):
                pass  # Fall through to pairwise method

        # Pairwise sample correlation fallback
        new_returns = price_data[new_ticker]["Close"].pct_change().iloc[-self.corr_lookback:]
        max_corr = 0.0

        for ticker in existing_tickers:
            if ticker not in price_data:
                continue
            existing_returns = price_data[ticker]["Close"].pct_change().iloc[-self.corr_lookback:]
            common = new_returns.index.intersection(existing_returns.index)
            if len(common) < 20:
                continue
            corr = new_returns.loc[common].corr(existing_returns.loc[common])
            if not np.isnan(corr):
                max_corr = max(max_corr, abs(corr))

        return float(max_corr)

    def _check_correlations_regime(
        self,
        new_ticker: str,
        existing_tickers: List[str],
        price_data: Dict[str, pd.DataFrame],
        regime: int,
        regime_labels: pd.Series,
    ) -> float:
        """Compute max correlation using regime-conditional covariance.

        Caches per-regime covariance matrices to avoid repeated computation.
        Falls back to global covariance when the regime has insufficient
        observations (< 30).
        """
        # Build returns matrix for all relevant tickers
        all_tickers = [new_ticker] + [t for t in existing_tickers if t in price_data]
        if len(all_tickers) < 2:
            return 0.0

        returns_dict = {}
        for ticker in all_tickers:
            if ticker not in price_data:
                continue
            series = price_data[ticker]["Close"].pct_change().iloc[-self.corr_lookback:]
            series = series.replace([np.inf, -np.inf], np.nan)
            returns_dict[ticker] = series

        returns_df = pd.DataFrame(returns_dict).dropna(how="all")
        if returns_df.shape[0] < 20 or returns_df.shape[1] < 2:
            return 0.0

        # Compute or retrieve cached regime covariance
        cache_key = regime
        if cache_key not in self._regime_cov_cache:
            # Align regime_labels with returns
            common_idx = returns_df.index.intersection(regime_labels.index)
            if len(common_idx) < 20:
                raise ValueError("Insufficient regime labels for correlation check")
            aligned_returns = returns_df.loc[common_idx].dropna(how="any")
            aligned_regimes = regime_labels.loc[aligned_returns.index]

            regime_covs = compute_regime_covariance(
                aligned_returns, aligned_regimes,
            )
            # Cache all computed regime covariances
            self._regime_cov_cache.update(regime_covs)

        cov_matrix = get_regime_covariance(self._regime_cov_cache, regime)

        # Extract correlation matrix from covariance
        available = [t for t in [new_ticker] + existing_tickers
                     if t in cov_matrix.columns]
        if new_ticker not in available or len(available) < 2:
            raise ValueError("New ticker not in covariance matrix")

        cov_sub = cov_matrix.loc[available, available]
        vols = np.sqrt(np.maximum(np.diag(cov_sub.values), 1e-14))
        vol_outer = np.outer(vols, vols)
        corr_matrix = cov_sub.values / vol_outer

        # Find max abs correlation between new_ticker and existing
        new_idx = available.index(new_ticker)
        max_corr = 0.0
        for i, ticker in enumerate(available):
            if ticker == new_ticker:
                continue
            c = corr_matrix[new_idx, i]
            if np.isfinite(c):
                max_corr = max(max_corr, abs(c))

        return float(max_corr)

    def _compute_avg_pairwise_correlation(
        self,
        positions: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
    ) -> float:
        """Compute average absolute pairwise correlation across all positions.

        Uses the most recent ``corr_lookback`` bars of close-to-close returns.
        Positions without price data are silently skipped.

        Parameters
        ----------
        positions : dict
            {ticker: weight} of current portfolio positions.
        price_data : dict
            {ticker: OHLCV DataFrame}.

        Returns
        -------
        float
            Average absolute value of upper-triangle pairwise correlations,
            or 0.0 when fewer than 2 tickers have sufficient data.
        """
        tickers = [t for t in positions if t in price_data]
        if len(tickers) < 2:
            return 0.0

        returns = {}
        for ticker in tickers:
            series = price_data[ticker]["Close"].pct_change().iloc[-self.corr_lookback:]
            series = series.replace([np.inf, -np.inf], np.nan)
            returns[ticker] = series

        ret_df = pd.DataFrame(returns).dropna(how="all")
        if ret_df.shape[0] < 20 or ret_df.shape[1] < 2:
            return 0.0

        corr_matrix = ret_df.corr().values
        n = corr_matrix.shape[0]
        upper_tri = np.abs(corr_matrix[np.triu_indices(n, k=1)])
        # Filter out NaN values that can occur with insufficient overlap
        upper_tri = upper_tri[np.isfinite(upper_tri)]
        if len(upper_tri) == 0:
            return 0.0

        return float(np.mean(upper_tri))

    @staticmethod
    def _get_correlation_stress_multiplier(avg_corr: float) -> float:
        """Map average pairwise correlation to a constraint stress multiplier.

        Iterates through ``CORRELATION_STRESS_THRESHOLDS`` (sorted ascending)
        and returns the tightest multiplier whose threshold is exceeded.

        Parameters
        ----------
        avg_corr : float
            Average absolute pairwise correlation of the portfolio.

        Returns
        -------
        float
            Multiplier in (0, 1].  1.0 means no tightening.
        """
        multiplier = 1.0
        for threshold in sorted(CORRELATION_STRESS_THRESHOLDS.keys()):
            if avg_corr > threshold:
                multiplier = CORRELATION_STRESS_THRESHOLDS[threshold]
        return multiplier

    def invalidate_regime_cov_cache(self) -> None:
        """Clear the cached regime-conditional covariance matrices."""
        self._regime_cov_cache.clear()

    def _estimate_portfolio_beta(
        self,
        positions: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame,
    ) -> float:
        """Estimate portfolio beta vs benchmark."""
        bench_returns = benchmark_data["Close"].pct_change().iloc[-self.corr_lookback:]
        bench_var = bench_returns.var()
        if bench_var == 0:
            return 0.0

        weighted_beta = 0.0
        total_weight = 0.0

        for ticker, weight in positions.items():
            if ticker not in price_data:
                continue
            stock_returns = price_data[ticker]["Close"].pct_change().iloc[-self.corr_lookback:]
            common = stock_returns.index.intersection(bench_returns.index)
            if len(common) < 20:
                continue
            cov = stock_returns.loc[common].cov(bench_returns.loc[common])
            beta = cov / bench_var
            weighted_beta += weight * beta
            total_weight += weight

        return float(weighted_beta) if total_weight > 0 else 0.0

    def _estimate_portfolio_vol(
        self,
        positions: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
    ) -> float:
        """
        Estimate annualized portfolio volatility from covariance matrix.
        """
        if not positions:
            return 0.0

        tickers = [t for t in positions if t in price_data]
        if not tickers:
            return 0.0

        returns = {}
        for ticker in tickers:
            series = price_data[ticker]["Close"].pct_change().iloc[-self.corr_lookback:]
            series = series.replace([np.inf, -np.inf], np.nan)
            returns[ticker] = series

        ret_df = pd.DataFrame(returns).dropna(how="all")
        if ret_df.shape[0] < 5:
            return 0.0

        estimate = self.covariance_estimator.estimate(ret_df)
        weights = {t: float(positions[t]) for t in estimate.covariance.columns if t in positions}
        return self.covariance_estimator.portfolio_volatility(weights, estimate.covariance)

    def portfolio_summary(
        self,
        positions: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
    ) -> Dict:
        """Generate a portfolio risk summary."""
        summary = {
            "n_positions": len(positions),
            "gross_exposure": sum(positions.values()),
            "sector_breakdown": {},
            "largest_position": max(positions.values()) if positions else 0,
        }

        # Sector breakdown
        for ticker, size in positions.items():
            sector = self._resolve_sector(ticker, price_data)
            summary["sector_breakdown"][sector] = (
                summary["sector_breakdown"].get(sector, 0) + size
            )

        # Correlation matrix of current positions
        if len(positions) > 1:
            tickers = [t for t in positions if t in price_data]
            if len(tickers) > 1:
                returns = pd.DataFrame({
                    t: price_data[t]["Close"].pct_change().iloc[-self.corr_lookback:]
                    for t in tickers
                })
                corr_matrix = returns.corr()
                # Average pairwise correlation
                n = len(tickers)
                if n > 1:
                    upper_tri = corr_matrix.values[np.triu_indices(n, k=1)]
                    avg_abs_corr = float(np.mean(np.abs(upper_tri)))
                    summary["avg_pairwise_corr"] = float(np.mean(upper_tri))
                    summary["avg_abs_pairwise_corr"] = avg_abs_corr
                    summary["max_pairwise_corr"] = float(np.max(upper_tri))
                    summary["corr_stress_multiplier"] = self._get_correlation_stress_multiplier(avg_abs_corr)

        return summary
