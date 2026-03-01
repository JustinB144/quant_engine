"""Health scoring logic — domain score aggregation and severity classification.

Extracted from HealthService (SPEC_AUDIT_FIX_04 T5) for independent testability.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .health_types import HealthCheckResult

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


def compute_confidence_intervals(
    domains: Dict[str, Dict[str, Any]],
    overall_score: Optional[float],
) -> tuple[Dict[str, Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Compute per-domain and overall confidence intervals.

    Returns (domain_ci, overall_ci).
    """
    from .health_confidence import HealthConfidenceCalculator

    ci_calc = HealthConfidenceCalculator()
    domain_ci: Dict[str, Dict[str, Any]] = {}

    for dname, dinfo in domains.items():
        avail = [c for c in dinfo["checks"] if c.get("status") != "UNAVAILABLE"]
        scores = [c.get("score", 0.0) for c in avail]
        if scores:
            ci_result = ci_calc.compute_ci(samples=np.array(scores))
            domain_ci[dname] = ci_result.to_dict()
        else:
            domain_ci[dname] = {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                                "n_samples": 0, "method": "insufficient",
                                "low_confidence": True, "ci_width": 0.0}

    overall_ci = None
    available_domains = {k: v for k, v in domains.items() if v["score"] is not None}
    if overall_score is not None and available_domains:
        d_scores, d_widths, d_weights = [], [], []
        for dname, dinfo in available_domains.items():
            d_scores.append(dinfo["score"])
            ci = domain_ci.get(dname, {})
            d_widths.append(ci.get("ci_width", 0.0))
            d_weights.append(dinfo["weight"])
        ci_lower, ci_upper = HealthConfidenceCalculator.propagate_weighted_ci(
            d_scores, d_widths, d_weights)
        overall_ci = {
            "ci_lower": round(ci_lower, 1),
            "ci_upper": round(ci_upper, 1),
            "ci_width": round(ci_upper - ci_lower, 1),
        }

    return domain_ci, overall_ci
