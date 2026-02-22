"""
Portfolio Risk Manager — enforces sector, correlation, and exposure limits.

Prevents concentration risk across multiple dimensions:
    - Sector/industry exposure caps
    - Pairwise correlation limits
    - Beta exposure limits
    - Gross/net exposure limits
    - Single-name concentration
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import MAX_PORTFOLIO_VOL
from .covariance import CovarianceEstimator


# Default sector mappings for common tickers
SECTOR_MAP = {
    # Tech
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "META": "tech",
    "NVDA": "tech", "AMD": "tech", "INTC": "tech", "CRM": "tech",
    "ADBE": "tech", "ORCL": "tech",
    # Mid-cap tech
    "DDOG": "tech", "NET": "tech", "CRWD": "tech", "ZS": "tech",
    "SNOW": "tech", "MDB": "tech", "PANW": "tech", "FTNT": "tech",
    # Healthcare
    "JNJ": "healthcare", "PFE": "healthcare", "UNH": "healthcare",
    "ABBV": "healthcare", "MRK": "healthcare", "LLY": "healthcare",
    "TMO": "healthcare", "ABT": "healthcare",
    # Consumer
    "AMZN": "consumer", "TSLA": "consumer", "HD": "consumer",
    "NKE": "consumer", "SBUX": "consumer", "MCD": "consumer",
    "TGT": "consumer", "COST": "consumer",
    # Financial
    "JPM": "financial", "BAC": "financial", "GS": "financial",
    "MS": "financial", "BLK": "financial", "V": "financial", "MA": "financial",
    # Industrial
    "CAT": "industrial", "DE": "industrial", "GE": "industrial",
    "HON": "industrial", "BA": "industrial", "LMT": "industrial",
    # Small/mid volatile
    "CAVA": "consumer", "BROS": "consumer", "TOST": "tech",
    "CHWY": "consumer", "ETSY": "consumer", "POOL": "consumer",
}


@dataclass
class RiskCheck:
    """Result of a portfolio risk check."""
    passed: bool
    violations: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)


class PortfolioRiskManager:
    """
    Enforces portfolio-level risk constraints.

    Checks run before each trade entry to ensure the portfolio
    stays within predefined risk bounds.
    """

    def __init__(
        self,
        max_sector_pct: float = 0.40,         # 40% max per sector
        max_corr_between: float = 0.85,        # reject if corr > 85% with existing
        max_gross_exposure: float = 1.0,        # 100% of capital (no leverage)
        max_single_name_pct: float = 0.10,     # 10% max single name
        max_beta_exposure: float = 1.5,         # net beta limit
        max_portfolio_vol: float = MAX_PORTFOLIO_VOL,  # annualized
        correlation_lookback: int = 60,         # days for correlation calc
        covariance_method: str = "ledoit_wolf",
        sector_map: Optional[Dict[str, str]] = None,
    ):
        """Initialize PortfolioRiskManager."""
        self.max_sector_pct = max_sector_pct
        self.max_corr = max_corr_between
        self.max_gross = max_gross_exposure
        self.max_single = max_single_name_pct
        self.max_beta = max_beta_exposure
        self.max_portfolio_vol = max_portfolio_vol
        self.corr_lookback = correlation_lookback
        raw_sector_map = sector_map or SECTOR_MAP
        self.sector_map = {str(k).upper(): str(v) for k, v in raw_sector_map.items()}
        self.covariance_estimator = CovarianceEstimator(method=covariance_method)

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
        """
        key = str(asset_id).upper().strip()
        if key in self.sector_map:
            return self.sector_map[key]
        df = price_data.get(str(asset_id))
        mapped_ticker = self._infer_ticker_from_price_df(df)
        if mapped_ticker and mapped_ticker in self.sector_map:
            return self.sector_map[mapped_ticker]
        return "other"

    def check_new_position(
        self,
        ticker: str,
        position_size: float,
        current_positions: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> RiskCheck:
        """
        Check if adding a new position violates any risk constraints.

        Args:
            ticker: proposed new position
            position_size: proposed size as fraction of capital
            current_positions: {ticker: size_pct} of existing positions
            price_data: OHLCV data for all tickers
            benchmark_data: SPY OHLCV for beta calculation

        Returns:
            RiskCheck with pass/fail and violation details
        """
        violations = []
        metrics = {}
        proposed_positions = {**current_positions, ticker: position_size}

        # ── Single-name check ──
        if position_size > self.max_single:
            violations.append(
                f"Position size {position_size:.1%} > max {self.max_single:.1%}"
            )

        # ── Gross exposure check ──
        gross = sum(proposed_positions.values())
        metrics["gross_exposure"] = gross
        if gross > self.max_gross:
            violations.append(
                f"Gross exposure {gross:.1%} > max {self.max_gross:.1%}"
            )

        # ── Sector concentration check ──
        sector = self._resolve_sector(ticker, price_data)
        sector_exposure = position_size
        for t, s in current_positions.items():
            if self._resolve_sector(t, price_data) == sector:
                sector_exposure += s
        metrics["sector_exposure"] = {sector: sector_exposure}
        if sector_exposure > self.max_sector_pct:
            violations.append(
                f"Sector '{sector}' exposure {sector_exposure:.1%} > max {self.max_sector_pct:.1%}"
            )

        # ── Correlation check ──
        if ticker in price_data and current_positions:
            max_corr_found = self._check_correlations(
                ticker, list(current_positions.keys()), price_data
            )
            metrics["max_pairwise_corr"] = max_corr_found
            if max_corr_found > self.max_corr:
                violations.append(
                    f"Max pairwise correlation {max_corr_found:.2f} > {self.max_corr:.2f}"
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

        return RiskCheck(
            passed=len(violations) == 0,
            violations=violations,
            metrics=metrics,
        )

    def _check_correlations(
        self,
        new_ticker: str,
        existing_tickers: List[str],
        price_data: Dict[str, pd.DataFrame],
    ) -> float:
        """Find max correlation between new ticker and existing positions."""
        if new_ticker not in price_data:
            return 0.0

        new_returns = price_data[new_ticker]["Close"].pct_change().iloc[-self.corr_lookback:]
        max_corr = 0.0

        for ticker in existing_tickers:
            if ticker not in price_data:
                continue
            existing_returns = price_data[ticker]["Close"].pct_change().iloc[-self.corr_lookback:]
            # Align indices
            common = new_returns.index.intersection(existing_returns.index)
            if len(common) < 20:
                continue
            corr = new_returns.loc[common].corr(existing_returns.loc[common])
            if not np.isnan(corr):
                max_corr = max(max_corr, abs(corr))

        return float(max_corr)

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
                    summary["avg_pairwise_corr"] = float(np.mean(upper_tri))
                    summary["max_pairwise_corr"] = float(np.max(upper_tri))

        return summary
