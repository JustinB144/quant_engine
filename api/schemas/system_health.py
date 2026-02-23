"""System health schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class QuickStatus(BaseModel):
    """Lightweight health check response."""

    status: str = "healthy"
    checks: Dict[str, str] = {}
    timestamp: Optional[str] = None


class AlertEvent(BaseModel):
    """A single health check item."""

    name: str
    status: str
    detail: str = ""
    value: str = ""
    recommendation: str = ""


class SystemHealthDetail(BaseModel):
    """Full system health assessment."""

    overall_score: float = 0.0
    overall_status: str = "---"
    generated_at: Optional[str] = None
    data_integrity_score: float = 0.0
    promotion_score: float = 0.0
    wf_score: float = 0.0
    execution_score: float = 0.0
    complexity_score: float = 0.0
    survivorship_checks: List[AlertEvent] = []
    data_quality_checks: List[AlertEvent] = []
    promotion_checks: List[AlertEvent] = []
    wf_checks: List[AlertEvent] = []
    execution_checks: List[AlertEvent] = []
    complexity_checks: List[AlertEvent] = []
    strengths: List[AlertEvent] = []
    promotion_funnel: Dict[str, int] = {}
    feature_inventory: Dict[str, int] = {}
    knob_inventory: List[Dict[str, str]] = []
