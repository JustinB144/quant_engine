"""Dashboard endpoints — KPIs and regime overview."""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends

from ..cache.manager import CacheManager
from ..deps.providers import get_cache
from ..schemas.envelope import ApiResponse
from ..services.backtest_service import BacktestService
from ..services.regime_service import RegimeService

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary")
async def dashboard_summary(cache: CacheManager = Depends(get_cache)) -> ApiResponse:
    cached = cache.get("dashboard:summary")
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    svc = BacktestService()
    data = await asyncio.to_thread(svc.get_latest_results, 10)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set("dashboard:summary", data)
    return ApiResponse.success(data, elapsed_ms=elapsed)


@router.get("/regime")
async def dashboard_regime(cache: CacheManager = Depends(get_cache)) -> ApiResponse:
    cached = cache.get("regime:current")
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    svc = RegimeService()
    data = await asyncio.to_thread(svc.detect_current_regime)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set("regime:current", data)
    return ApiResponse.success(data, elapsed_ms=elapsed)
