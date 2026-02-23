"""System health endpoints."""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends

from ..cache.manager import CacheManager
from ..deps.providers import get_cache
from ..schemas.envelope import ApiResponse
from ..services.health_service import HealthService

router = APIRouter(tags=["health"])


@router.get("/api/health")
async def quick_health(cache: CacheManager = Depends(get_cache)) -> ApiResponse:
    cached = cache.get("health:quick")
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    svc = HealthService()
    data = await asyncio.to_thread(svc.get_quick_status)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set("health:quick", data)
    return ApiResponse.success(data, elapsed_ms=elapsed)


@router.get("/api/health/detailed")
async def detailed_health(cache: CacheManager = Depends(get_cache)) -> ApiResponse:
    cached = cache.get("health:detailed")
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    svc = HealthService()
    data = await asyncio.to_thread(svc.get_detailed_health)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set("health:detailed", data)
    return ApiResponse.success(data, elapsed_ms=elapsed)
