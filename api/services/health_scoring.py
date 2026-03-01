"""Health scoring logic — domain score aggregation and severity classification.

Extracted from HealthService (SPEC_AUDIT_FIX_04 T5) for independent testability.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .health_service import HealthCheckResult

logger = logging.getLogger(__name__)

_SEVERITY_WEIGHTS = {"critical": 3.0, "standard": 1.0, "informational": 0.5}


def domain_score(checks: List[HealthCheckResult]) -> Optional[float]:
    """Compute severity-weighted mean score, excluding UNAVAILABLE checks.

    Severity weights: critical=3.0, standard=1.0, informational=0.5.
    Returns None if all checks are UNAVAILABLE.
    """
    available = [c for c in checks if c.status != "UNAVAILABLE"]
    if not available:
        return None
    weighted_sum = sum(
        c.score * _SEVERITY_WEIGHTS.get(c.severity, 1.0) for c in available
    )
    weight_total = sum(
        _SEVERITY_WEIGHTS.get(c.severity, 1.0) for c in available
    )
    return float(weighted_sum / weight_total) if weight_total > 0 else 0.0


def domain_status(checks: List[HealthCheckResult]) -> str:
    """Determine domain-level status."""
    total = len(checks)
    unavailable = sum(1 for c in checks if c.status == "UNAVAILABLE")
    if total > 0 and unavailable / total > 0.5:
        return "UNAVAILABLE"
    available = [c for c in checks if c.status != "UNAVAILABLE"]
    if not available:
        return "UNAVAILABLE"
    avg = float(np.mean([c.score for c in available]))
    if avg >= 75:
        return "PASS"
    if avg >= 50:
        return "WARN"
    return "FAIL"


def compute_overall_score(
    domains: Dict[str, Dict[str, Any]],
) -> tuple[Optional[float], str]:
    """Compute weighted overall health score from domain results.

    Returns
    -------
    (overall_score, overall_status)
    """
    available_domains = {
        k: v for k, v in domains.items() if v["score"] is not None
    }
    if available_domains:
        weight_sum = sum(v["weight"] for v in available_domains.values())
        overall = (
            sum(v["weight"] * v["score"] for v in available_domains.values())
            / weight_sum
            if weight_sum > 0
            else None
        )
    else:
        overall = None

    if overall is None:
        overall_status = "UNAVAILABLE"
    elif overall >= 75:
        overall_status = "PASS"
    elif overall >= 50:
        overall_status = "WARN"
    else:
        overall_status = "FAIL"

    return overall, overall_status
