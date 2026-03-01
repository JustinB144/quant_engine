"""System health schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class QuickStatus(BaseModel):
    """Lightweight health check response."""

    model_config = ConfigDict(extra="allow")

    status: str = "healthy"
    checks: Dict[str, str] = Field(default_factory=dict)
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
    survivorship_checks: List[AlertEvent] = Field(default_factory=list)
    data_quality_checks: List[AlertEvent] = Field(default_factory=list)
    promotion_checks: List[AlertEvent] = Field(default_factory=list)
    wf_checks: List[AlertEvent] = Field(default_factory=list)
    execution_checks: List[AlertEvent] = Field(default_factory=list)
    complexity_checks: List[AlertEvent] = Field(default_factory=list)
    strengths: List[AlertEvent] = Field(default_factory=list)
    promotion_funnel: Dict[str, int] = Field(default_factory=dict)
    feature_inventory: Dict[str, int] = Field(default_factory=dict)
    knob_inventory: List[Dict[str, str]] = Field(default_factory=list)
