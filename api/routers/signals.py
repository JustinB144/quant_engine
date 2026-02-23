"""Signal / prediction endpoints."""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends

from ..cache.manager import CacheManager
from ..deps.providers import get_cache
from ..schemas.envelope import ApiResponse
from ..services.results_service import ResultsService

router = APIRouter(prefix="/api/signals", tags=["signals"])


@router.get("/latest")
async def latest_signals(
    horizon: int = 10,
    cache: CacheManager = Depends(get_cache),
) -> ApiResponse:
    cache_key = f"signals:latest:{horizon}"
    cached = cache.get(cache_key)
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    svc = ResultsService()
    data = await asyncio.to_thread(svc.get_latest_predictions, horizon)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set(cache_key, data)
    return ApiResponse.success(data, elapsed_ms=elapsed)
