"""
Risk Metrics — VaR, CVaR, tail risk, MAE/MFE, and advanced risk analytics.

Provides comprehensive risk measurement for both individual trades
and portfolio-level analysis.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RiskReport:
    """Comprehensive risk metrics report."""
    # Value at Risk — Historical
    var_95: float           # 5th percentile loss (historical)
    var_99: float           # 1st percentile loss (historical)
    cvar_95: float          # Expected shortfall at 95% (historical)
    cvar_99: float          # Expected shortfall at 99% (historical)

    # Value at Risk — Parametric (normal distribution assumption)
    var_95_parametric: float = 0.0
    var_99_parametric: float = 0.0

    # Value at Risk — Cornish-Fisher (skewness/kurtosis adjusted)
    var_95_cornish_fisher: float = 0.0
    var_99_cornish_fisher: float = 0.0

    # Tail risk
    tail_ratio: float = 0.0       # abs(95th pctile / 5th pctile) — skew of returns
    max_loss: float = 0.0         # Worst single return
    max_gain: float = 0.0         # Best single return

    # Drawdown analytics
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # bars
    ulcer_index: float = 0.0        # Ulcer Index (Martin 1987)

    # Trade-level
    mae_mean: float = 0.0        # Mean Maximum Adverse Excursion
    mfe_mean: float = 0.0        # Mean Maximum Favorable Excursion
    edge_ratio: float = 0.0      # MFE / MAE — how much edge vs pain

    # Distribution
    skewness: float = 0.0
    kurtosis: float = 0.0        # Excess kurtosis

    # Calmar ratio (annualized return / max drawdown)
    calmar_ratio: float = 0.0

    details: Dict = field(default_factory=dict)


class RiskMetrics:
    """
    Computes comprehensive risk metrics from trade returns and equity curves.
    """

    def __init__(self, annual_trading_days: int = 252):
        """Initialize RiskMetrics."""
        self.trading_days = annual_trading_days

    def compute_full_report(
        self,
        trade_returns: np.ndarray,
        equity_curve: Optional[pd.Series] = None,
        trade_details: Optional[List[Dict]] = None,
        holding_days: int = 10,
    ) -> RiskReport:
        """
        Compute all risk metrics.

        Args:
            trade_returns: array of per-trade net returns
            equity_curve: optional daily equity curve for drawdown analysis
            trade_details: optional list of trade dicts with intra-trade data
            holding_days: average holding period

        Returns:
            RiskReport with all metrics
        """
        if len(trade_returns) == 0:
            return self._empty_report()

        returns = np.array(trade_returns, dtype=float)

        # ── Value at Risk (Historical simulation) ──
        var_95 = float(np.percentile(returns, 5))
        var_99 = float(np.percentile(returns, 1))

        # ── CVaR / Expected Shortfall ──
        cvar_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else var_95
        cvar_99 = float(returns[returns <= var_99].mean()) if (returns <= var_99).any() else var_99

        # ── Parametric VaR (normal distribution assumption) ──
        from scipy.stats import norm

        mu, sigma = float(returns.mean()), float(returns.std(ddof=1))
        var_95_param = mu + sigma * norm.ppf(0.05) if sigma > 1e-12 else var_95
        var_99_param = mu + sigma * norm.ppf(0.01) if sigma > 1e-12 else var_99

        # ── Cornish-Fisher VaR (adjusts for non-normality) ──
        var_95_cf, var_99_cf = self._cornish_fisher_var(returns, mu, sigma)

        # ── Tail ratio ──
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        tail_ratio = abs(p95 / p5) if abs(p5) > 1e-12 else 0.0

        # ── Drawdown analytics ──
        if equity_curve is not None and len(equity_curve) > 1:
            dd_metrics = self._drawdown_analytics(equity_curve)
        else:
            # Build equity from trade returns
            cum_eq = np.cumprod(1 + returns)
            dd_metrics = self._drawdown_analytics_array(cum_eq)

        # ── MAE / MFE ──
        if trade_details:
            mae_mean, mfe_mean, edge_ratio = self._compute_mae_mfe(trade_details)
        else:
            # Approximate from returns
            mae_mean = float(returns[returns < 0].mean()) if (returns < 0).any() else 0
            mfe_mean = float(returns[returns > 0].mean()) if (returns > 0).any() else 0
            edge_ratio = abs(mfe_mean / mae_mean) if mae_mean != 0 else 0

        # ── Distribution ──
        skewness = float(pd.Series(returns).skew())
        kurtosis = float(pd.Series(returns).kurt())  # Excess kurtosis

        # ── Calmar ratio ──
        ann_return = float(returns.mean() * (self.trading_days / holding_days))
        calmar = abs(ann_return / dd_metrics["max_drawdown"]) if dd_metrics["max_drawdown"] != 0 else 0

        return RiskReport(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            var_95_parametric=float(var_95_param),
            var_99_parametric=float(var_99_param),
            var_95_cornish_fisher=float(var_95_cf),
            var_99_cornish_fisher=float(var_99_cf),
            tail_ratio=float(tail_ratio),
            max_loss=float(returns.min()),
            max_gain=float(returns.max()),
            max_drawdown=dd_metrics["max_drawdown"],
            avg_drawdown=dd_metrics["avg_drawdown"],
            max_drawdown_duration=dd_metrics["max_dd_duration"],
            ulcer_index=dd_metrics["ulcer_index"],
            mae_mean=float(mae_mean),
            mfe_mean=float(mfe_mean),
            edge_ratio=float(edge_ratio),
            skewness=skewness,
            kurtosis=kurtosis,
            calmar_ratio=float(calmar),
            details={
                "n_trades": len(returns),
                "annualized_return": ann_return,
                "holding_days": holding_days,
            },
        )

    @staticmethod
    def _cornish_fisher_var(
        returns: np.ndarray, mu: float, sigma: float,
    ) -> tuple:
        """Cornish-Fisher VaR at 95% and 99% — adjusts for skewness and kurtosis.

        The Cornish-Fisher expansion modifies the standard normal quantile
        to account for non-normality in the return distribution:

            z_cf = z + (z^2 - 1)/6 * S + (z^3 - 3z)/24 * K - (2z^3 - 5z)/36 * S^2

        where S = skewness, K = excess kurtosis, z = normal quantile.

        Returns
        -------
        tuple[float, float]
            (var_95_cf, var_99_cf)
        """
        from scipy.stats import norm

        if sigma < 1e-12:
            return mu, mu

        s = float(pd.Series(returns).skew()) if len(returns) > 2 else 0.0
        k = float(pd.Series(returns).kurt()) if len(returns) > 3 else 0.0

        cf_vars = []
        for alpha in (0.05, 0.01):
            z = norm.ppf(alpha)
            z_cf = (
                z
                + (z**2 - 1) / 6.0 * s
                + (z**3 - 3 * z) / 24.0 * k
                - (2 * z**3 - 5 * z) / 36.0 * s**2
            )
            cf_vars.append(mu + sigma * z_cf)

        return cf_vars[0], cf_vars[1]

    def _drawdown_analytics(self, equity: pd.Series) -> Dict:
        """Compute drawdown metrics from a daily equity curve."""
        eq = equity.values
        return self._drawdown_analytics_array(eq)

    def _drawdown_analytics_array(self, equity: np.ndarray) -> Dict:
        """Compute drawdown metrics from an equity array."""
        if len(equity) < 2:
            return {
                "max_drawdown": 0, "avg_drawdown": 0,
                "max_dd_duration": 0, "ulcer_index": 0,
            }

        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max

        max_dd = float(drawdowns.min())

        # Average drawdown (only periods in drawdown)
        in_dd = drawdowns[drawdowns < 0]
        avg_dd = float(in_dd.mean()) if len(in_dd) > 0 else 0

        # Max drawdown duration (bars)
        max_duration = 0
        current_duration = 0
        for d in drawdowns:
            if d < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        # Ulcer Index (Martin 1987): RMS of drawdowns
        # Measures both depth and duration of drawdowns
        ulcer = float(np.sqrt(np.mean(drawdowns**2)))

        return {
            "max_drawdown": max_dd,
            "avg_drawdown": avg_dd,
            "max_dd_duration": max_duration,
            "ulcer_index": ulcer,
        }

    def _compute_mae_mfe(self, trade_details: List[Dict]) -> tuple:
        """
        Compute Maximum Adverse/Favorable Excursion from intra-trade data.

        MAE = worst unrealized loss during the trade
        MFE = best unrealized gain during the trade
        Edge ratio = MFE / MAE (how much edge vs drawdown per trade)
        """
        maes = []
        mfes = []

        for trade in trade_details:
            intra_returns = trade.get("intra_trade_returns", [])
            if not intra_returns:
                continue

            cum_returns = np.cumprod(1 + np.array(intra_returns)) - 1
            maes.append(float(cum_returns.min()))
            mfes.append(float(cum_returns.max()))

        if not maes:
            return 0, 0, 0

        mae_mean = float(np.mean(maes))
        mfe_mean = float(np.mean(mfes))
        edge = abs(mfe_mean / mae_mean) if mae_mean != 0 else 0

        return mae_mean, mfe_mean, edge

    def _empty_report(self) -> RiskReport:
        """Internal helper for empty report."""
        return RiskReport(
            var_95=0, var_99=0, cvar_95=0, cvar_99=0,
            tail_ratio=0, max_loss=0, max_gain=0,
            max_drawdown=0, avg_drawdown=0, max_drawdown_duration=0,
            ulcer_index=0, mae_mean=0, mfe_mean=0, edge_ratio=0,
            skewness=0, kurtosis=0, calmar_ratio=0,
        )

    def print_report(self, report: RiskReport):
        """Pretty-print a risk report."""
        print(f"\n{'='*50}")
        print(f"RISK METRICS REPORT")
        print(f"{'='*50}")
        print(f"  Value at Risk (Historical):")
        print(f"    VaR 95%: {report.var_95:.4f}")
        print(f"    VaR 99%: {report.var_99:.4f}")
        print(f"    CVaR 95%: {report.cvar_95:.4f}")
        print(f"    CVaR 99%: {report.cvar_99:.4f}")
        print(f"  Value at Risk (Parametric):")
        print(f"    VaR 95%: {report.var_95_parametric:.4f}")
        print(f"    VaR 99%: {report.var_99_parametric:.4f}")
        print(f"  Value at Risk (Cornish-Fisher):")
        print(f"    VaR 95%: {report.var_95_cornish_fisher:.4f}")
        print(f"    VaR 99%: {report.var_99_cornish_fisher:.4f}")
        print(f"  Tail Risk:")
        print(f"    Tail ratio: {report.tail_ratio:.2f}")
        print(f"    Max loss: {report.max_loss:.4f}")
        print(f"    Max gain: {report.max_gain:.4f}")
        print(f"    Skewness: {report.skewness:.2f}")
        print(f"    Excess kurtosis: {report.kurtosis:.2f}")
        print(f"  Drawdown:")
        print(f"    Max drawdown: {report.max_drawdown:.1%}")
        print(f"    Avg drawdown: {report.avg_drawdown:.1%}")
        print(f"    Max DD duration: {report.max_drawdown_duration} bars")
        print(f"    Ulcer Index: {report.ulcer_index:.4f}")
        print(f"  Trade Quality:")
        print(f"    MAE (mean): {report.mae_mean:.4f}")
        print(f"    MFE (mean): {report.mfe_mean:.4f}")
        print(f"    Edge ratio: {report.edge_ratio:.2f}")
        print(f"  Performance:")
        print(f"    Calmar ratio: {report.calmar_ratio:.2f}")
