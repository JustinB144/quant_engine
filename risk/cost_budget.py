"""
Transaction Cost Budget Optimization — minimize implementation cost for rebalances.

Optimizes the total cost of implementing a set of portfolio changes, not just
individual trades.  When the full rebalance exceeds the cost budget, trades
are prioritized by their importance-to-cost ratio and executed greedily until
the budget is exhausted.

This reduces turnover drag by deferring low-value, high-cost trades to
future rebalance cycles.

Cost model:
    - Half-spread cost: 0.5 * spread_bps (pay half the bid-ask spread)
    - Market impact: impact_coeff * sqrt(participation_rate)
    - Participation rate: abs(trade_size) / daily_dollar_volume

Integration points:
    - Backtester: apply cost budget to daily rebalance decisions
    - PaperTrader: constrain rebalance to cost budget
    - Autopilot: cost-aware strategy evaluation
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RebalanceResult:
    """Result of a cost-budgeted rebalance optimization."""
    final_weights: pd.Series
    executed_trades: Dict[str, float]
    deferred_trades: Dict[str, float]
    total_cost_bps: float
    budget_bps: float
    is_partial: bool
    n_executed: int
    n_deferred: int
    cost_savings_bps: float  # Cost saved by deferring trades
    tracking_error_vs_target: float  # L2 distance from target weights


def estimate_trade_cost_bps(
    trade_size_weight: float,
    daily_volume: float,
    spread_bps: float = 3.0,
    impact_coeff_bps: float = 25.0,
    portfolio_value_usd: float = 1_000_000.0,
) -> float:
    """Estimate the implementation cost of a single trade in basis points.

    Cost = half_spread + market_impact
    Market impact uses the square-root model: impact = coeff * sqrt(participation)

    Args:
        trade_size_weight: Absolute trade size as fraction of portfolio.
        daily_volume: Average daily dollar volume of the security.
        spread_bps: Bid-ask spread in basis points.
        impact_coeff_bps: Market impact coefficient.
        portfolio_value_usd: Portfolio value in USD for correct participation rate.

    Returns:
        Estimated cost in basis points of portfolio value.
    """
    if daily_volume <= 0:
        return 100.0  # Penalize illiquid trades heavily

    # Participation rate: trade_dollar_value / daily_dollar_volume
    trade_dollar_value = abs(trade_size_weight) * portfolio_value_usd
    participation = trade_dollar_value / max(daily_volume, 1e-12)
    participation = min(participation, 1.0)  # Cap at 100%

    half_spread = 0.5 * spread_bps
    impact = impact_coeff_bps * np.sqrt(participation)

    return (half_spread + impact) * abs(trade_size_weight)


def optimize_rebalance_cost(
    current_weights: pd.Series,
    target_weights: pd.Series,
    daily_volumes: pd.Series,
    urgency: float = 0.5,
    total_budget_bps: float = 50.0,
    spread_bps: float = 3.0,
    impact_coeff_bps: float = 25.0,
    min_trade_weight: float = 1e-4,
    portfolio_value_usd: float = 1_000_000.0,
) -> RebalanceResult:
    """Optimize portfolio rebalance within a transaction cost budget.

    Given current and target portfolios, find the optimal subset of trades
    that minimizes tracking error vs the target while staying within the
    transaction cost budget.

    When the full rebalance is within budget, all trades are executed.
    When over-budget, trades are prioritized by importance/cost ratio
    (larger weight changes are more important for tracking error).

    Args:
        current_weights: Current portfolio weights (sums to ~1.0).
        target_weights: Desired portfolio weights.
        daily_volumes: Average daily dollar volume per ticker.
        urgency: Trade urgency (0=patient, 1=urgent).  Higher urgency
            allows more aggressive execution but higher impact costs.
        total_budget_bps: Maximum total implementation cost in basis points.
        spread_bps: Bid-ask spread assumption.
        impact_coeff_bps: Market impact coefficient.
        min_trade_weight: Minimum trade size to execute (filter noise).
        portfolio_value_usd: Portfolio value in USD for correct participation rate.

    Returns:
        RebalanceResult with executed/deferred trades and cost analysis.
    """
    # Align all series to the same index (union of current and target)
    all_tickers = sorted(set(current_weights.index) | set(target_weights.index))
    current = current_weights.reindex(all_tickers).fillna(0.0)
    target = target_weights.reindex(all_tickers).fillna(0.0)
    volumes = daily_volumes.reindex(all_tickers).fillna(1e6)  # Default 1M volume

    # Compute required trades
    trades = target - current
    trades = trades[trades.abs() > min_trade_weight]

    if trades.empty:
        return RebalanceResult(
            final_weights=current,
            executed_trades={},
            deferred_trades={},
            total_cost_bps=0.0,
            budget_bps=total_budget_bps,
            is_partial=False,
            n_executed=0,
            n_deferred=0,
            cost_savings_bps=0.0,
            tracking_error_vs_target=0.0,
        )

    # Estimate cost per trade
    costs: Dict[str, float] = {}
    for ticker, trade_size in trades.items():
        vol = max(volumes.get(ticker, 1e6), 1.0)
        # Scale impact by urgency (higher urgency = more participation)
        urgency_scaled_impact = impact_coeff_bps * (0.5 + 0.5 * urgency)
        costs[ticker] = estimate_trade_cost_bps(
            trade_size_weight=abs(trade_size),
            daily_volume=vol,
            spread_bps=spread_bps,
            impact_coeff_bps=urgency_scaled_impact,
            portfolio_value_usd=portfolio_value_usd,
        )

    total_cost = sum(costs.values())

    if total_cost <= total_budget_bps:
        # Full rebalance is within budget
        final_weights = target.copy()
        return RebalanceResult(
            final_weights=final_weights,
            executed_trades=dict(trades),
            deferred_trades={},
            total_cost_bps=total_cost,
            budget_bps=total_budget_bps,
            is_partial=False,
            n_executed=len(trades),
            n_deferred=0,
            cost_savings_bps=0.0,
            tracking_error_vs_target=0.0,
        )

    # Over budget: prioritize trades by |weight change| / cost ratio
    logger.warning(
        "Rebalance cost %.1f bps exceeds budget %.1f bps — partial rebalance",
        total_cost, total_budget_bps,
    )
    # Larger weight changes matter more for tracking error
    priority = pd.Series({
        ticker: abs(trades[ticker]) / max(costs[ticker], 1e-10)
        for ticker in trades.index
    })
    priority = priority.sort_values(ascending=False)

    # Greedy: execute trades in priority order until budget exhausted
    remaining_budget = total_budget_bps
    executed_trades: Dict[str, float] = {}
    deferred_trades: Dict[str, float] = {}

    for ticker in priority.index:
        trade_cost = costs[ticker]
        if trade_cost <= remaining_budget:
            executed_trades[ticker] = float(trades[ticker])
            remaining_budget -= trade_cost
        else:
            deferred_trades[ticker] = float(trades[ticker])

    # Build partial rebalance weights
    final_weights = current.copy()
    for ticker, change in executed_trades.items():
        final_weights[ticker] = final_weights.get(ticker, 0.0) + change

    # Renormalize to sum to 1.0 (preserve cash-like residual)
    weight_sum = final_weights.sum()
    if weight_sum > 1e-8:
        final_weights = final_weights / weight_sum

    # Tracking error: L2 distance from target
    common = sorted(set(final_weights.index) & set(target.index))
    if common:
        tracking_error = float(np.sqrt(
            ((final_weights.reindex(common).fillna(0) -
              target.reindex(common).fillna(0)) ** 2).sum()
        ))
    else:
        tracking_error = 0.0

    cost_spent = total_budget_bps - remaining_budget
    cost_saved = total_cost - cost_spent

    return RebalanceResult(
        final_weights=final_weights,
        executed_trades=executed_trades,
        deferred_trades=deferred_trades,
        total_cost_bps=cost_spent,
        budget_bps=total_budget_bps,
        is_partial=True,
        n_executed=len(executed_trades),
        n_deferred=len(deferred_trades),
        cost_savings_bps=cost_saved,
        tracking_error_vs_target=tracking_error,
    )
