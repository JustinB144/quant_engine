"""Runtime config management endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Body, Depends
from fastapi.responses import JSONResponse

from ..cache.invalidation import invalidate_on_config_change
from ..cache.manager import CacheManager
from ..config import RuntimeConfig
from ..deps.providers import get_cache, get_runtime_config
from ..schemas.envelope import ApiResponse

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("")
async def get_config(rc: RuntimeConfig = Depends(get_runtime_config)) -> ApiResponse:
    return ApiResponse.success(rc.get_adjustable())


@router.patch("")
async def patch_config(
    updates: dict = Body(...),
    rc: RuntimeConfig = Depends(get_runtime_config),
    cache: CacheManager = Depends(get_cache),
) -> ApiResponse:
    try:
        new_state = rc.patch(updates)
    except (KeyError, ValueError) as exc:
        resp = ApiResponse.fail(str(exc))
        return JSONResponse(status_code=422, content=resp.model_dump())
    invalidate_on_config_change(cache)
    return ApiResponse.success(new_state)
