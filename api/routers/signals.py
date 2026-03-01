"""Signal / prediction endpoints."""
from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, Depends

from ..cache.manager import CacheManager
from ..deps.providers import get_cache
from ..schemas.envelope import ApiResponse
from ..services.results_service import ResultsService

router = APIRouter(prefix="/api/signals", tags=["signals"])


def _get_signal_meta_fields() -> dict:
    """Collect transparency metadata for signal responses."""
    meta_fields: dict = {}
    try:
        from quant_engine.config import REGIME_TRADE_POLICY
        # Report True if *any* regime has suppression enabled (enabled=False)
        meta_fields["regime_suppressed"] = any(
            not p["enabled"] for p in REGIME_TRADE_POLICY.values()
        )
        meta_fields["regime_trade_policy"] = json.dumps(
            REGIME_TRADE_POLICY, default=str
        )
    except (ImportError, AttributeError):
        pass
    try:
        from quant_engine.models.versioning import ModelRegistry
        registry = ModelRegistry()
        latest = registry.get_latest()
        if latest:
            meta_fields["model_version"] = latest.version_id
    except (ImportError, OSError):
        pass
    return meta_fields


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
    meta_fields = await asyncio.to_thread(_get_signal_meta_fields)
    return ApiResponse.success(data, elapsed_ms=elapsed, **meta_fields)
