"""Backtest result + compute endpoints."""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends

from ..cache.invalidation import invalidate_on_backtest
from ..cache.manager import CacheManager
from ..deps.auth import require_auth
from ..deps.providers import get_cache, get_job_runner, get_job_store
from ..jobs.runner import JobRunner
from ..jobs.store import JobStore
from ..schemas.compute import BacktestRequest
from ..schemas.envelope import ApiResponse
from ..services.backtest_service import BacktestService

router = APIRouter(prefix="/api/backtests", tags=["backtests"])


def _extract_backtest_meta(data: dict) -> dict:
    """Extract transparency fields from backtest data for ResponseMeta."""
    meta_fields: dict = {}
    if "sizing_method" in data:
        meta_fields["sizing_method"] = data["sizing_method"]
    if "walk_forward_mode" in data:
        meta_fields["walk_forward_mode"] = data["walk_forward_mode"]
    if "model_version" in data:
        meta_fields["model_version"] = data["model_version"]
    return meta_fields


@router.get("/latest")
async def latest_backtest(
    horizon: int = 10,
    cache: CacheManager = Depends(get_cache),
) -> ApiResponse:
    cache_key = f"dashboard:backtest:{horizon}"
    cached = cache.get(cache_key)
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    svc = BacktestService()
    data = await asyncio.to_thread(svc.get_latest_results, horizon)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set(cache_key, data)
    meta_fields = _extract_backtest_meta(data)
    return ApiResponse.success(data, elapsed_ms=elapsed, **meta_fields)


@router.get("/latest/trades")
async def latest_trades(horizon: int = 10, limit: int = 200, offset: int = 0) -> ApiResponse:
    t0 = time.monotonic()
    svc = BacktestService()
    data = await asyncio.to_thread(svc.get_latest_trades, horizon, limit, offset)
    elapsed = (time.monotonic() - t0) * 1000
    return ApiResponse.success(data, elapsed_ms=elapsed)


@router.get("/latest/equity-curve")
async def equity_curve(
    horizon: int = 10,
    cache: CacheManager = Depends(get_cache),
) -> ApiResponse:
    cache_key = f"dashboard:equity:{horizon}"
    cached = cache.get(cache_key)
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    svc = BacktestService()
    data = await asyncio.to_thread(svc.get_equity_curve, horizon)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set(cache_key, data)
    return ApiResponse.success(data, elapsed_ms=elapsed)


@router.post("/run", dependencies=[Depends(require_auth)])
async def run_backtest(
    req: BacktestRequest,
    store: JobStore = Depends(get_job_store),
    runner: JobRunner = Depends(get_job_runner),
    cache: CacheManager = Depends(get_cache),
) -> ApiResponse:
    from ..jobs.backtest_job import execute_backtest_job

    rec = await store.create_job("backtest", req.model_dump())
    await runner.submit(rec.job_id, execute_backtest_job, req.model_dump())
    invalidate_on_backtest(cache)
    return ApiResponse.success({"job_id": rec.job_id, "job_type": "backtest", "status": "queued"})
