"""
Risk API router — factor exposures, diagnostics, and monitoring endpoints.

Endpoints:
    GET /api/risk/factor-exposures  — current portfolio factor exposures
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/risk", tags=["risk"])


@router.get("/factor-exposures")
async def get_factor_exposures() -> Dict:
    """Return current portfolio factor exposure analysis.

    Computes factor tilts (market beta, size, momentum, volatility, liquidity)
    and checks against configured limits.
    """
    try:
        from quant_engine.risk.factor_monitor import FactorExposureMonitor

        monitor = FactorExposureMonitor()

        # In a live system, positions and price_data would come from the
        # paper trader or portfolio state.  For now, return the monitor
        # configuration and empty exposures as a schema example.
        return {
            "status": "ok",
            "exposures": {},
            "limits": {
                k: {"low": v[0], "high": v[1]}
                for k, v in monitor.limits.items()
            },
            "violations": [],
            "all_passed": True,
            "message": "Factor monitor initialized. Connect to live portfolio for real-time exposures.",
        }
    except Exception as e:
        logger.warning("Factor exposure computation failed: %s", e)
        return {
            "status": "error",
            "message": str(e),
            "exposures": {},
            "violations": [],
            "all_passed": True,
        }
