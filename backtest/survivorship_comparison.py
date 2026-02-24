"""
Survivorship Bias Comparison — quantify the impact of survivorship bias on backtests.

Runs identical backtest configurations on both survivorship-free and
survivor-only universes, then computes the bias in every key metric.

Typical findings:
    - Sharpe inflated 20-40% by survivorship bias
    - Win rate inflated 5-15%
    - Max drawdown understated 10-30%

This tool helps validate that the system's survivorship-bias controls
(data/survivorship.py) are working correctly and quantifies residual bias.

Integration points:
    - Autopilot: compare strategies on both universes before promotion
    - Validation: include survivorship comparison in advanced validation
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class UniverseMetrics:
    """Backtest metrics for a single universe type."""
    universe_type: str  # "full" or "survivors_only"
    sharpe: float
    win_rate: float
    n_trades: int
    max_drawdown: float
    total_return: float
    avg_return: float
    profit_factor: float
    avg_holding_days: float


@dataclass
class SurvivorshipComparisonResult:
    """Complete comparison between full and survivor-only universes."""
    full_universe: UniverseMetrics
    survivors_only: UniverseMetrics
    bias: Dict[str, float]
    summary: str


def _extract_metrics(result, universe_type: str) -> UniverseMetrics:
    """Extract key metrics from a BacktestResult."""
    return UniverseMetrics(
        universe_type=universe_type,
        sharpe=getattr(result, "sharpe_ratio", 0.0),
        win_rate=getattr(result, "win_rate", 0.0),
        n_trades=getattr(result, "total_trades", 0),
        max_drawdown=getattr(result, "max_drawdown", 0.0),
        total_return=getattr(result, "total_return", 0.0),
        avg_return=getattr(result, "avg_return", 0.0),
        profit_factor=getattr(result, "profit_factor", 0.0),
        avg_holding_days=getattr(result, "avg_holding_days", 0.0),
    )


def compare_survivorship_impact(
    result_full,
    result_survivors,
) -> SurvivorshipComparisonResult:
    """Compare two backtest results: full universe vs survivors-only.

    Both results should come from the same strategy with the same
    parameters, differing only in the universe of securities.

    Args:
        result_full: BacktestResult from a survivorship-free universe
            (includes delisted securities).
        result_survivors: BacktestResult from a survivors-only universe
            (only tickers that survived to end of period).

    Returns:
        SurvivorshipComparisonResult with per-metric bias breakdown.
    """
    full = _extract_metrics(result_full, "full")
    surv = _extract_metrics(result_survivors, "survivors_only")

    # Compute bias metrics
    sharpe_bias = surv.sharpe - full.sharpe
    win_rate_bias = surv.win_rate - full.win_rate
    drawdown_bias = surv.max_drawdown - full.max_drawdown  # Less negative = understated
    return_bias = surv.total_return - full.total_return

    # Percentage bias where denominator is meaningful
    full_sharpe_abs = max(0.01, abs(full.sharpe))
    sharpe_bias_pct = (sharpe_bias / full_sharpe_abs) * 100

    bias = {
        "sharpe_bias": sharpe_bias,
        "sharpe_bias_pct": sharpe_bias_pct,
        "win_rate_bias": win_rate_bias,
        "drawdown_bias": drawdown_bias,
        "return_bias": return_bias,
        "trade_count_diff": surv.n_trades - full.n_trades,
    }

    summary_lines = [
        "Survivorship Bias Analysis:",
        f"  Full universe:     Sharpe={full.sharpe:.2f}, WR={full.win_rate:.1%}, "
        f"Trades={full.n_trades}, MaxDD={full.max_drawdown:.1%}",
        f"  Survivors only:    Sharpe={surv.sharpe:.2f}, WR={surv.win_rate:.1%}, "
        f"Trades={surv.n_trades}, MaxDD={surv.max_drawdown:.1%}",
        f"  Bias:",
        f"    Sharpe inflation:  {sharpe_bias:+.2f} ({sharpe_bias_pct:+.1f}%)",
        f"    Win rate inflation: {win_rate_bias:+.1%}",
        f"    Drawdown understatement: {drawdown_bias:+.1%}",
        f"    Return inflation: {return_bias:+.2%}",
    ]

    return SurvivorshipComparisonResult(
        full_universe=full,
        survivors_only=surv,
        bias=bias,
        summary="\n".join(summary_lines),
    )


def quick_survivorship_check(
    predictions: pd.DataFrame,
    price_data_full: Dict[str, pd.DataFrame],
    price_data_survivors: Dict[str, pd.DataFrame],
) -> Dict:
    """Quick survivorship bias check without running a full backtest.

    Compares prediction coverage and data completeness between the two
    universes as a proxy for survivorship bias magnitude.

    Args:
        predictions: Prediction DataFrame with ticker/permno column.
        price_data_full: {ticker: OHLCV} including delisted securities.
        price_data_survivors: {ticker: OHLCV} only surviving securities.

    Returns:
        Dict with universe coverage comparison.
    """
    full_tickers = set(price_data_full.keys())
    surv_tickers = set(price_data_survivors.keys())
    dropped = full_tickers - surv_tickers

    # Count data points
    full_bars = sum(len(df) for df in price_data_full.values())
    surv_bars = sum(len(df) for df in price_data_survivors.values())

    return {
        "full_universe_tickers": len(full_tickers),
        "survivors_only_tickers": len(surv_tickers),
        "dropped_tickers": len(dropped),
        "dropped_ticker_names": sorted(dropped)[:20],  # Cap at 20 for display
        "full_universe_bars": full_bars,
        "survivors_only_bars": surv_bars,
        "bar_coverage_ratio": surv_bars / max(1, full_bars),
        "estimated_bias_risk": (
            "HIGH" if len(dropped) > 10
            else "MEDIUM" if len(dropped) > 3
            else "LOW"
        ),
    }
