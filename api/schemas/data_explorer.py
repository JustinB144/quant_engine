"""Data explorer schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UniverseInfo(BaseModel):
    """Universe configuration summary."""

    full_size: int
    quick_size: int
    full_tickers: List[str]
    quick_tickers: List[str]
    benchmark: str


class OHLCVBar(BaseModel):
    """Single OHLCV bar."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class TickerDetail(BaseModel):
    """Detailed ticker response with OHLCV bars."""

    ticker: str
    permno: Optional[str] = None
    found: bool = False
    bars: List[OHLCVBar] = Field(default_factory=list)
    total_bars: Optional[int] = None
