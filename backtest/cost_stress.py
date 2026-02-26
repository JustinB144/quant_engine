"""
Cost stress testing — Truth Layer T5.

Tests strategy robustness to transaction cost assumptions by running
the backtest at multiple cost levels and computing breakeven cost.

Usage:
    tester = CostStressTester(base_cost_bps=20.0)
    result = tester.run_sweep(daily_returns, trade_costs_per_trade)
    print(tester.report(result))
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CostStressPoint:
    """Result for a single cost multiplier level."""
    multiplier: float
    effective_cost_bps: float
    total_return: float
    sharpe_ratio: float
    n_trades: int


@dataclass
class CostStressResult:
    """Aggregate result of cost stress sweep."""
    base_cost_bps: float
    multipliers: List[float]
    points: List[CostStressPoint] = field(default_factory=list)
    breakeven_multiplier: float = float("inf")
    breakeven_cost_bps: float = float("inf")

    def to_dict(self) -> Dict:
        """Serialize to a dictionary for JSON output."""
        return {
            "base_cost_bps": self.base_cost_bps,
            "multipliers": self.multipliers,
            "breakeven_multiplier": self.breakeven_multiplier,
            "breakeven_cost_bps": self.breakeven_cost_bps,
            "points": [
                {
                    "multiplier": p.multiplier,
                    "effective_cost_bps": p.effective_cost_bps,
                    "total_return": p.total_return,
                    "sharpe_ratio": p.sharpe_ratio,
                    "n_trades": p.n_trades,
                }
                for p in self.points
            ],
        }


class CostStressTester:
    """Tests strategy robustness to transaction cost assumptions.

    Rather than re-running the full backtester at each cost level (which
    would be 4-5x slower), this class works on already-computed trade-level
    returns and adjusts the cost component analytically.

    Parameters
    ----------
    base_cost_bps : float
        The base transaction cost in basis points (e.g., 20.0).
    multipliers : list[float] or None
        Cost multiplier levels to test.  Default: ``[0.5, 1.0, 2.0, 5.0]``.
    """

    def __init__(
        self,
        base_cost_bps: float,
        multipliers: Optional[List[float]] = None,
    ):
        self.base_cost_bps = base_cost_bps
        self.multipliers = multipliers or [0.5, 1.0, 2.0, 5.0]

    def run_sweep(
        self,
        gross_returns: np.ndarray,
        cost_per_trade_bps: float,
        n_trades: int,
    ) -> CostStressResult:
        """Run cost stress sweep on trade-level gross returns.

        This method analytically adjusts returns for each cost multiplier
        level rather than re-running the full backtest.

        Parameters
        ----------
        gross_returns : np.ndarray
            Array of per-trade gross returns (before transaction costs).
        cost_per_trade_bps : float
            Base cost per trade in basis points.
        n_trades : int
            Total number of trades.

        Returns
        -------
        CostStressResult
        """
        gross = np.asarray(gross_returns, dtype=float)
        gross = gross[np.isfinite(gross)]

        if len(gross) == 0:
            return CostStressResult(
                base_cost_bps=self.base_cost_bps,
                multipliers=self.multipliers,
            )

        points: List[CostStressPoint] = []

        for mult in sorted(self.multipliers):
            effective_cost = cost_per_trade_bps * mult
            # Adjust returns: subtract incremental cost
            # (cost_per_trade_bps is round-trip, applied per trade)
            net_returns = gross - (effective_cost / 10_000)

            total_ret = float(np.prod(1 + net_returns) - 1)

            mean_ret = float(np.mean(net_returns))
            std_ret = float(np.std(net_returns, ddof=1)) if len(net_returns) > 1 else 1e-10
            sharpe = (mean_ret / max(std_ret, 1e-10)) * np.sqrt(252)

            points.append(CostStressPoint(
                multiplier=mult,
                effective_cost_bps=round(effective_cost, 2),
                total_return=round(total_ret, 6),
                sharpe_ratio=round(float(sharpe), 4),
                n_trades=n_trades,
            ))

        # Estimate breakeven cost multiplier (where Sharpe crosses zero)
        breakeven_mult = float("inf")
        breakeven_bps = float("inf")

        if len(points) >= 2:
            sharpes = [p.sharpe_ratio for p in points]
            mults = [p.multiplier for p in points]

            # Find first zero-crossing
            for i in range(len(sharpes) - 1):
                if sharpes[i] > 0 and sharpes[i + 1] <= 0:
                    # Linear interpolation
                    denom = sharpes[i] - sharpes[i + 1]
                    if abs(denom) > 1e-10:
                        frac = sharpes[i] / denom
                        breakeven_mult = mults[i] + frac * (mults[i + 1] - mults[i])
                        breakeven_bps = self.base_cost_bps * breakeven_mult
                    break

            # All positive Sharpe = very robust
            if all(s > 0 for s in sharpes):
                breakeven_mult = float("inf")
                breakeven_bps = float("inf")
            # All negative Sharpe = not even viable at lowest cost
            elif all(s <= 0 for s in sharpes):
                breakeven_mult = 0.0
                breakeven_bps = 0.0

        result = CostStressResult(
            base_cost_bps=self.base_cost_bps,
            multipliers=list(self.multipliers),
            points=points,
            breakeven_multiplier=round(breakeven_mult, 4) if np.isfinite(breakeven_mult) else float("inf"),
            breakeven_cost_bps=round(breakeven_bps, 2) if np.isfinite(breakeven_bps) else float("inf"),
        )

        logger.info(
            "Cost stress sweep: base=%.1f bps, breakeven=%.1f bps (%.2fx)",
            self.base_cost_bps,
            breakeven_bps if np.isfinite(breakeven_bps) else -1,
            breakeven_mult if np.isfinite(breakeven_mult) else -1,
        )

        return result

    def report(self, result: CostStressResult) -> str:
        """Generate human-readable cost stress report.

        Parameters
        ----------
        result : CostStressResult
            Output from ``run_sweep()``.

        Returns
        -------
        str
            Formatted report string.
        """
        lines = [
            "Cost Stress Test Report",
            "=" * 55,
            f"Base cost: {result.base_cost_bps:.1f} bps",
            "",
            f"{'Multiplier':<12} {'Cost (bps)':<12} {'Sharpe':<10} {'Return':<12} {'Trades':<8}",
            "-" * 55,
        ]

        for p in result.points:
            ret_pct = f"{p.total_return * 100:.2f}%"
            lines.append(
                f"{p.multiplier:<12.1f} {p.effective_cost_bps:<12.1f} "
                f"{p.sharpe_ratio:<10.3f} {ret_pct:<12} {p.n_trades:<8}"
            )

        lines.append("-" * 55)

        if np.isfinite(result.breakeven_cost_bps):
            lines.append(
                f"Breakeven cost: {result.breakeven_cost_bps:.1f} bps "
                f"({result.breakeven_multiplier:.2f}x base)"
            )
        else:
            lines.append("Breakeven cost: >inf (strategy profitable at all tested levels)")

        return "\n".join(lines)
