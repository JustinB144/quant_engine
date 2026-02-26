"""
Diagnostics API router — system self-diagnosis and root-cause analysis.

Endpoints:
    GET /api/diagnostics  — run self-diagnostic analysis
"""
from __future__ import annotations

import logging
from typing import Dict

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])


@router.get("")
async def get_diagnostics() -> Dict:
    """Run system self-diagnostic analysis.

    Correlates health metrics with recent performance to identify
    root causes of any degradation.  Returns findings with severity,
    evidence, and recommended actions.
    """
    try:
        from ..services.diagnostics import SystemDiagnostics

        diag = SystemDiagnostics()

        # In a fully wired system, these would come from the health service
        # and paper trader state.  For now, return the diagnostic schema.
        report = diag.diagnose_performance(
            equity_curve=None,
            health_history={},
            regime_history={},
            trade_history={},
        )

        return {
            "status": report.status,
            "primary_cause": report.primary_cause,
            "diagnostics": [
                {
                    "cause": d.cause,
                    "severity": d.severity,
                    "evidence": d.evidence,
                    "recommendation": d.recommendation,
                    "metric_value": d.metric_value,
                    "metric_threshold": d.metric_threshold,
                }
                for d in report.diagnostics
            ],
            "recent_return": report.recent_return,
            "recent_sharpe": report.recent_sharpe,
            "timestamp": report.timestamp,
        }
    except Exception as e:
        logger.warning("Diagnostics computation failed: %s", e)
        return {
            "status": "error",
            "message": str(e),
            "diagnostics": [],
        }
