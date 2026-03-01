"""System health endpoints."""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

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


@router.get("/api/health/history")
async def health_history() -> ApiResponse:
    """Return recent health score snapshots with rolling averages and trend."""
    t0 = time.monotonic()
    svc = HealthService()
    data = await asyncio.to_thread(svc.get_health_history_with_trends, limit=90)
    elapsed = (time.monotonic() - t0) * 1000
    return ApiResponse.success(data, elapsed_ms=elapsed)


@router.get("/api/v1/system/model-age")
async def model_age() -> ApiResponse:
    """Return age of the currently deployed model in days and the version ID."""
    t0 = time.monotonic()
    try:
        from quant_engine.config import MODEL_DIR
        registry_path = Path(MODEL_DIR) / "registry.json"
        if not registry_path.exists():
            data = {"age_days": None, "version_id": None, "status": "no_registry"}
        else:
            with open(registry_path, "r") as f:
                reg = json.load(f)
            versions = reg.get("versions", []) if isinstance(reg, dict) else []
            if not versions:
                data = {"age_days": None, "version_id": None, "status": "no_versions"}
            else:
                latest = versions[-1]
                version_id = latest.get("version_id", "unknown")
                train_date_str = latest.get("training_date", None)
                if train_date_str:
                    train_date = datetime.fromisoformat(train_date_str)
                    if train_date.tzinfo is None:
                        train_date = train_date.replace(tzinfo=timezone.utc)
                    age_days = (datetime.now(timezone.utc) - train_date).days
                else:
                    age_days = None
                data = {
                    "age_days": age_days,
                    "version_id": version_id,
                    "status": "stale" if age_days and age_days > 30 else "fresh",
                }
    except (OSError, json.JSONDecodeError, ValueError, ImportError) as e:
        data = {"age_days": None, "version_id": None, "status": "error", "detail": str(e)}
    elapsed = (time.monotonic() - t0) * 1000
    return ApiResponse.success(data, elapsed_ms=elapsed)


@router.get("/api/v1/system/data-mode")
async def data_mode() -> ApiResponse:
    """Return current data source mode (wrds, cache, demo) and any active fallbacks."""
    t0 = time.monotonic()
    try:
        from quant_engine.config import WRDS_ENABLED, DATA_CACHE_DIR
        from quant_engine.data.loader import get_data_provenance

        provenance = get_data_provenance()
        n_fallbacks = sum(1 for v in provenance.values() if not v.get("trusted", True))

        if WRDS_ENABLED:
            mode = "wrds"
        else:
            mode = "cache"

        cache_dir = Path(DATA_CACHE_DIR)
        n_cached = len(list(cache_dir.glob("*.parquet"))) if cache_dir.exists() else 0

        data = {
            "mode": mode,
            "wrds_enabled": WRDS_ENABLED,
            "n_cached_files": n_cached,
            "n_fallbacks": n_fallbacks,
            "fallback_tickers": list(provenance.keys())[:10],
            "status": "degraded" if n_fallbacks > 0 else "clean",
        }
    except (OSError, ValueError, ImportError) as e:
        data = {"mode": "unknown", "status": "error", "detail": str(e)}
    elapsed = (time.monotonic() - t0) * 1000
    return ApiResponse.success(data, elapsed_ms=elapsed)
