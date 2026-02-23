"""IV Surface computation endpoints."""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter

from ..schemas.envelope import ApiResponse

router = APIRouter(prefix="/api/iv-surface", tags=["iv-surface"])


def _compute_arb_free_svi() -> dict:
    """Build arb-free SVI surface from synthetic market data."""
    import numpy as np
    from models.iv.models import ArbitrageFreeSVIBuilder, generate_synthetic_market_surface

    market = generate_synthetic_market_surface(S=100, r=0.05, q=0.01)
    builder = ArbitrageFreeSVIBuilder()
    result = builder.build_surface(
        spot=market["S"],
        strikes=market["strikes"],
        expiries=market["expiries"],
        market_iv_grid=market["iv_grid"],
        r=market["r"],
        q=market["q"],
    )

    # Convert numpy arrays to JSON-serializable lists
    moneyness = market["moneyness"].tolist()
    expiries = result["expiries"].tolist()
    raw_iv = (result["raw_iv_grid"] * 100).tolist()  # as percentage
    adj_iv = (result["adj_iv_grid"] * 100).tolist()   # as percentage
    market_iv = (market["iv_grid"] * 100).tolist()     # as percentage

    # Per-slice fit quality
    objectives = [float(o) if np.isfinite(o) else None for o in result["objectives"]]

    # Compute max calendar arbitrage violation before/after
    raw_tv = result["raw_total_variance"]
    adj_tv = result["adj_total_variance"]
    raw_cal_violation = 0.0
    adj_cal_violation = 0.0
    for i in range(1, raw_tv.shape[0]):
        diff_raw = raw_tv[i] - raw_tv[i - 1]
        diff_adj = adj_tv[i] - adj_tv[i - 1]
        raw_cal_violation = max(raw_cal_violation, float(-np.min(diff_raw)))
        adj_cal_violation = max(adj_cal_violation, float(-np.min(diff_adj)))

    return {
        "moneyness": moneyness,
        "expiries": expiries,
        "raw_iv": raw_iv,
        "adj_iv": adj_iv,
        "market_iv": market_iv,
        "objectives": objectives,
        "n_expiries": len(expiries),
        "n_strikes": len(moneyness),
        "raw_calendar_violation": raw_cal_violation,
        "adj_calendar_violation": adj_cal_violation,
    }


@router.get("/arb-free-svi")
async def arb_free_svi_surface() -> ApiResponse:
    """Compute arbitrage-free SVI surface from synthetic market data."""
    t0 = time.monotonic()
    data = await asyncio.to_thread(_compute_arb_free_svi)
    elapsed = (time.monotonic() - t0) * 1000
    return ApiResponse.success(data, elapsed_ms=elapsed)
