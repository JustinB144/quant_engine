"""
Risk API router — factor exposures, diagnostics, and monitoring endpoints.

Endpoints:
    GET /api/risk/factor-exposures  — current portfolio factor exposures
"""
from __future__ import annotations

import logging

from fastapi import APIRouter

from ..schemas.envelope import ApiResponse, ResponseMeta

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/risk", tags=["risk"])


@router.get("/factor-exposures")
async def get_factor_exposures() -> ApiResponse:
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
        data = {
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

        # No live data connected — return placeholder with ok=False
        return ApiResponse(
            ok=False,
            error="No live data connected. Showing schema only.",
            data=data,
            meta=ResponseMeta(data_mode="placeholder"),
        )
    except Exception as e:
        logger.warning("Factor exposure computation failed: %s", e)
        return ApiResponse.fail(f"Factor exposure computation failed: {e}")
