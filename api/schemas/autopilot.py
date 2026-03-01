"""Autopilot schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CycleReport(BaseModel):
    """Latest autopilot cycle results."""

    available: bool = False
    # Remaining fields are dynamic from the cycle report JSON


class StrategyInfo(BaseModel):
    """Active strategy from registry."""

    strategy_id: str
    promoted_at: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    metrics: Dict[str, Any] = Field(default_factory=dict)
    status: str = "active"


class PaperState(BaseModel):
    """Paper trading state."""

    available: bool = False
