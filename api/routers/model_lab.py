"""Model lab endpoints — versions, health, feature importance, train, predict."""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends

from ..cache.invalidation import invalidate_on_train
from ..cache.manager import CacheManager
from ..deps.providers import get_cache, get_job_runner, get_job_store
from ..jobs.runner import JobRunner
from ..jobs.store import JobStore
from ..schemas.compute import PredictRequest, TrainRequest
from ..schemas.envelope import ApiResponse
from ..services.model_service import ModelService

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/versions")
async def list_versions() -> ApiResponse:
    t0 = time.monotonic()
    svc = ModelService()
    data = await asyncio.to_thread(svc.list_versions)
    elapsed = (time.monotonic() - t0) * 1000
    return ApiResponse.success(data, elapsed_ms=elapsed)


@router.get("/health")
async def model_health(cache: CacheManager = Depends(get_cache)) -> ApiResponse:
    cached = cache.get("model_health:latest")
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    svc = ModelService()
    data = await asyncio.to_thread(svc.get_model_health)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set("model_health:latest", data)
    return ApiResponse.success(data, elapsed_ms=elapsed)


@router.get("/features/importance")
async def feature_importance(cache: CacheManager = Depends(get_cache)) -> ApiResponse:
    cached = cache.get("feature_importance:latest")
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    svc = ModelService()
    data = await asyncio.to_thread(svc.get_feature_importance)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set("feature_importance:latest", data)
    return ApiResponse.success(data, elapsed_ms=elapsed)


@router.post("/train")
async def train_model(
    req: TrainRequest,
    store: JobStore = Depends(get_job_store),
    runner: JobRunner = Depends(get_job_runner),
    cache: CacheManager = Depends(get_cache),
) -> ApiResponse:
    from ..jobs.train_job import execute_train_job

    rec = await store.create_job("train", req.model_dump())
    await runner.submit(rec.job_id, execute_train_job, req.model_dump())
    invalidate_on_train(cache)
    return ApiResponse.success({"job_id": rec.job_id, "job_type": "train", "status": "queued"})


@router.post("/predict")
async def predict_model(
    req: PredictRequest,
    store: JobStore = Depends(get_job_store),
    runner: JobRunner = Depends(get_job_runner),
) -> ApiResponse:
    from ..jobs.predict_job import execute_predict_job

    rec = await store.create_job("predict", req.model_dump())
    await runner.submit(rec.job_id, execute_predict_job, req.model_dump())
    return ApiResponse.success({"job_id": rec.job_id, "job_type": "predict", "status": "queued"})
