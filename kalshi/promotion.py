"""
Event-strategy promotion helpers for Kalshi walk-forward outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import logging

import numpy as np
import pandas as pd

from ..autopilot.promotion_gate import PromotionDecision, PromotionGate
from ..autopilot.strategy_discovery import StrategyCandidate
from ..backtest.engine import BacktestResult
from .walkforward import EventWalkForwardResult, evaluate_event_contract_metrics

logger = logging.getLogger(__name__)

_EPS = 1e-12


@dataclass
class EventPromotionConfig:
    """Strategy metadata used to evaluate an event strategy through the shared promotion gate."""
    strategy_id: str = "kalshi_event_default"
    horizon: int = 1
    entry_threshold: float = 0.0
    confidence_threshold: float = 0.0
    use_risk_management: bool = False
    max_positions: int = 1
    position_size_pct: float = 1.0


def _to_backtest_result(
    event_returns: Iterable[float],
    horizon_days: int = 1,
) -> BacktestResult:
    """Internal helper for to backtest result."""
    ret = np.asarray(list(event_returns), dtype=float)
    ret = ret[np.isfinite(ret)]

    n = int(ret.size)
    if n == 0:
        idx = pd.date_range("2000-01-01", periods=1, freq="D")
        zero = pd.Series([0.0], index=idx, dtype=float)
        return BacktestResult(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_return=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            profit_factor=0.0,
            avg_holding_days=float(max(1, int(horizon_days))),
            trades_per_year=0.0,
            returns_series=zero,
            equity_curve=pd.Series([1.0], index=idx, dtype=float),
            daily_equity=pd.Series([1.0], index=idx, dtype=float),
            trades=[],
        )

    wins = ret[ret > 0.0]
    losses = ret[ret < 0.0]
    win_rate = float((ret > 0.0).mean())

    avg_ret = float(np.nanmean(ret))
    avg_win = float(np.nanmean(wins)) if wins.size > 0 else 0.0
    avg_loss = float(np.nanmean(losses)) if losses.size > 0 else 0.0

    total_return = float(np.prod(1.0 + ret) - 1.0)
    years = max(float(n * max(1, int(horizon_days)) / 252.0), _EPS)
    annualized_return = float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1.0 else -1.0

    std = float(np.nanstd(ret))
    sharpe = float(np.nanmean(ret) / std * np.sqrt(252.0 / max(1, int(horizon_days)))) if std > _EPS else 0.0

    downside = ret[ret < 0.0]
    downside_std = float(np.nanstd(downside)) if downside.size > 0 else 0.0
    sortino = float(np.nanmean(ret) / downside_std * np.sqrt(252.0 / max(1, int(horizon_days)))) if downside_std > _EPS else sharpe

    gross_profit = float(np.sum(wins)) if wins.size > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losses))) if losses.size > 0 else 0.0
    if gross_loss <= _EPS and gross_profit > 0:
        profit_factor = 99.0
    elif gross_loss <= _EPS:
        profit_factor = 0.0
    else:
        profit_factor = float(gross_profit / gross_loss)

    idx = pd.date_range("2000-01-01", periods=n, freq="D")
    returns_series = pd.Series(ret, index=idx, dtype=float)
    equity = (1.0 + returns_series).cumprod()
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    trades_per_year = float(n / years) if years > _EPS else float(n)

    return BacktestResult(
        total_trades=n,
        winning_trades=int((ret > 0.0).sum()),
        losing_trades=int((ret < 0.0).sum()),
        win_rate=win_rate,
        avg_return=avg_ret,
        avg_win=avg_win,
        avg_loss=avg_loss,
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        profit_factor=profit_factor,
        avg_holding_days=float(max(1, int(horizon_days))),
        trades_per_year=trades_per_year,
        returns_series=returns_series,
        equity_curve=equity,
        daily_equity=equity,
        trades=[],
    )


def _compute_event_regime_performance(
    walkforward_result: EventWalkForwardResult,
) -> Dict[str, Dict[str, float]]:
    """Compute regime-conditioned returns from walk-forward event types and returns."""
    regime_returns: Dict[str, list] = {}
    for etype, ret in zip(walkforward_result.event_types, walkforward_result.event_returns):
        if np.isfinite(ret):
            regime_returns.setdefault(etype, []).append(ret)

    regime_perf: Dict[str, Dict[str, float]] = {}
    for regime, rets in regime_returns.items():
        arr = np.array(rets, dtype=float)
        if len(arr) >= 3:
            cum = np.cumprod(1 + arr)
            running_max = np.maximum.accumulate(cum)
            drawdowns = cum / running_max - 1
            regime_perf[regime] = {
                "mean_return": float(np.mean(arr)),
                "sharpe": float(np.mean(arr) / max(np.std(arr), 1e-8)),
                "max_drawdown": float(np.min(drawdowns)),
                "n_trades": len(arr),
            }
            logger.debug(
                "Event regime %s: n=%d, sharpe=%.3f, max_dd=%.4f",
                regime, len(arr), regime_perf[regime]["sharpe"],
                regime_perf[regime]["max_drawdown"],
            )
    return regime_perf


def evaluate_event_promotion(
    walkforward_result: EventWalkForwardResult,
    config: Optional[EventPromotionConfig] = None,
    gate: Optional[PromotionGate] = None,
    extra_contract_metrics: Optional[Dict[str, object]] = None,
) -> PromotionDecision:
    """
    Evaluate Kalshi event strategy promotion from walk-forward outputs.

    This automatically includes event contract metrics and routes through
    PromotionGate event-mode checks.
    """
    cfg = config or EventPromotionConfig()
    gate_obj = gate or PromotionGate()

    candidate = StrategyCandidate(
        strategy_id=str(cfg.strategy_id),
        horizon=int(cfg.horizon),
        entry_threshold=float(cfg.entry_threshold),
        confidence_threshold=float(cfg.confidence_threshold),
        use_risk_management=bool(cfg.use_risk_management),
        max_positions=int(cfg.max_positions),
        position_size_pct=float(cfg.position_size_pct),
    )

    backtest_like = _to_backtest_result(
        event_returns=walkforward_result.event_returns,
        horizon_days=max(1, int(cfg.horizon)),
    )

    regime_perf = _compute_event_regime_performance(walkforward_result)
    backtest_like.regime_performance = regime_perf
    if regime_perf:
        logger.info(
            "Event regime performance computed for %d event types: %s",
            len(regime_perf), list(regime_perf.keys()),
        )

    event_metrics = evaluate_event_contract_metrics(walkforward_result)

    contract_metrics: Dict[str, object] = dict(extra_contract_metrics or {})
    contract_metrics.setdefault("wf_oos_corr", float(walkforward_result.wf_oos_corr))
    contract_metrics.setdefault(
        "wf_positive_fold_fraction",
        float(walkforward_result.wf_positive_fold_fraction),
    )
    contract_metrics.setdefault("wf_is_oos_gap", float(walkforward_result.wf_is_oos_gap))
    contract_metrics.setdefault(
        "regime_positive_fraction",
        float(event_metrics.get("event_regime_stability", np.nan)),
    )
    contract_metrics.setdefault("n_trials", float(walkforward_result.n_trials_total))

    return gate_obj.evaluate_event_strategy(
        candidate=candidate,
        result=backtest_like,
        event_metrics=event_metrics,
        contract_metrics=contract_metrics,
    )

