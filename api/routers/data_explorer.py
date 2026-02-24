"""Data explorer endpoints — universe + per-ticker OHLCV."""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends

from ..cache.manager import CacheManager
from ..deps.providers import get_cache
from ..errors import DataNotFoundError
from ..schemas.envelope import ApiResponse
from ..services.data_service import DataService

router = APIRouter(prefix="/api/data", tags=["data"])


@router.get("/universe")
async def get_universe(cache: CacheManager = Depends(get_cache)) -> ApiResponse:
    cached = cache.get("data:universe")
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    svc = DataService()
    data = await asyncio.to_thread(svc.get_universe_info)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set("data:universe", data, ttl=300)
    return ApiResponse.success(data, elapsed_ms=elapsed)


@router.get("/status")
async def get_data_status(cache: CacheManager = Depends(get_cache)) -> ApiResponse:
    """Per-ticker cache health: source, freshness, bar counts, timeframes.

    Cached for 60 seconds to avoid repeated filesystem scans.
    """
    cached = cache.get("data:status")
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    svc = DataService()
    data = await asyncio.to_thread(svc.get_cache_status)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set("data:status", data, ttl=60)
    return ApiResponse.success(data, elapsed_ms=elapsed)


@router.get("/ticker/{ticker}")
async def get_ticker(ticker: str, years: int = 5) -> ApiResponse:
    t0 = time.monotonic()
    svc = DataService()
    data = await asyncio.to_thread(svc.load_single_ticker, ticker.upper(), years)
    elapsed = (time.monotonic() - t0) * 1000
    if not data.get("found"):
        raise DataNotFoundError(f"Ticker {ticker.upper()} not found in data sources")
    return ApiResponse.success(data, elapsed_ms=elapsed)
