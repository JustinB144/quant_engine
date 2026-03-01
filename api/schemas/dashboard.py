"""Dashboard-related schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DashboardKPIs(BaseModel):
    """Key performance indicators for the dashboard summary."""

    model_config = ConfigDict(extra="allow")

    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    total_return: Optional[float] = None
    annualized_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    total_trades: Optional[int] = None
    profit_factor: Optional[float] = None
    trades_per_year: Optional[float] = None
    regime_breakdown: Optional[Dict[str, Any]] = None


class RegimeInfo(BaseModel):
    """Current regime detection results."""

    current_label: str = "Unavailable"
    as_of: str = "---"
    current_probs: Dict[str, float] = Field(default_factory=dict)
    transition_matrix: Optional[List[List[float]]] = None
    prob_history: Optional[List[Dict[str, Any]]] = None


class EquityPoint(BaseModel):
    """Single point on an equity curve."""

    date: str
    value: float
