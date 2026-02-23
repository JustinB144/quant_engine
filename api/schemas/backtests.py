"""Backtest schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class BacktestSummary(BaseModel):
    """Backtest run summary."""

    available: bool = False
    horizon: Optional[int] = None
    total_trades: Optional[int] = None
    win_rate: Optional[float] = None
    avg_return: Optional[float] = None
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    max_drawdown: Optional[float] = None
    profit_factor: Optional[float] = None
    annualized_return: Optional[float] = None
    trades_per_year: Optional[float] = None
    regime_breakdown: Optional[Dict[str, Any]] = None


class TradeRecord(BaseModel):
    """Single trade from backtest."""

    permno: Optional[str] = None
    entry_date: Optional[str] = None
    exit_date: Optional[str] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    predicted_return: Optional[float] = None
    actual_return: Optional[float] = None
    net_return: Optional[float] = None
    regime: Optional[str] = None
    confidence: Optional[float] = None
    holding_days: Optional[int] = None
    exit_reason: Optional[str] = None


class EquityCurvePoint(BaseModel):
    """Single point on an equity curve."""

    date: str
    value: float


class RegimeBreakdown(BaseModel):
    """Per-regime performance stats."""

    regime: str
    trade_count: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
