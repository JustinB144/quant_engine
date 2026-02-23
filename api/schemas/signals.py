"""Signal schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class SignalRow(BaseModel):
    """Single prediction signal."""

    permno: Optional[str] = None
    ticker: Optional[str] = None
    predicted_return: Optional[float] = None
    confidence: Optional[float] = None
    regime: Optional[str] = None
    blend_alpha: Optional[float] = None


class SignalsSummary(BaseModel):
    """Signals overview for a horizon."""

    available: bool = False
    horizon: int = 10
    signals: List[Dict[str, Any]] = []
    total: int = 0
