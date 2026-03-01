"""Core types for health assessment.

Shared dataclass and helpers used by all health modules.
Extracted from health_service.py (SPEC_AUDIT_FIX_04 T5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

_SEVERITY_WEIGHTS = {"critical": 3.0, "standard": 1.0, "informational": 0.5}


@dataclass
class HealthCheckResult:
    """Structured result from a single health check."""

    name: str
    domain: str
    score: float  # 0–100
    status: str  # PASS, WARN, FAIL, UNAVAILABLE
    explanation: str = ""
    methodology: str = ""
    data_available: bool = True
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    severity: str = "standard"  # "critical", "standard", "informational"
    raw_value: Optional[float] = None  # The actual measured value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "domain": self.domain, "score": self.score,
            "status": self.status, "explanation": self.explanation,
            "methodology": self.methodology, "data_available": self.data_available,
            "raw_metrics": self.raw_metrics, "thresholds": self.thresholds,
            "severity": self.severity, "raw_value": self.raw_value,
        }


def _unavailable(name: str, domain: str, reason: str) -> HealthCheckResult:
    """Return a standard UNAVAILABLE result (score=0, excluded from averages)."""
    return HealthCheckResult(
        name=name, domain=domain, score=0.0, status="UNAVAILABLE",
        explanation=reason, methodology="N/A — data not available",
        data_available=False,
    )
