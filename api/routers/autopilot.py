"""Autopilot endpoints — cycle reports, strategies, paper state, run-cycle."""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends

from ..deps.auth import require_auth
from ..deps.providers import get_job_runner, get_job_store
from ..jobs.runner import JobRunner
from ..jobs.store import JobStore
from ..schemas.compute import AutopilotRequest
from ..schemas.envelope import ApiResponse
from ..services.autopilot_service import AutopilotService

router = APIRouter(prefix="/api/autopilot", tags=["autopilot"])


def _get_autopilot_meta() -> dict:
    """Collect transparency metadata for autopilot responses."""
    meta_fields: dict = {}
    try:
        from quant_engine.config import WF_MAX_TRAIN_DATES
        meta_fields["walk_forward_mode"] = "full" if WF_MAX_TRAIN_DATES > 0 else "single_split"
    except (ImportError, AttributeError):
        meta_fields["walk_forward_mode"] = "single_split"

    # Detect predictor type: check if trained model exists
    try:
        from quant_engine.models.versioning import ModelRegistry
        registry = ModelRegistry()
        latest = registry.get_latest()
        if latest:
            meta_fields["predictor_type"] = "ensemble"
            meta_fields["model_version"] = latest.version_id
        else:
            meta_fields["predictor_type"] = "heuristic"
    except (ImportError, OSError):
        meta_fields["predictor_type"] = "heuristic"

    return meta_fields


@router.get("/latest-cycle")
async def latest_cycle() -> ApiResponse:
    t0 = time.monotonic()
    svc = AutopilotService()
    data = await asyncio.to_thread(svc.get_latest_cycle)
    elapsed = (time.monotonic() - t0) * 1000
    meta_fields = await asyncio.to_thread(_get_autopilot_meta)
    return ApiResponse.success(data, elapsed_ms=elapsed, **meta_fields)


@router.get("/strategies")
async def strategies() -> ApiResponse:
    t0 = time.monotonic()
    svc = AutopilotService()
    data = await asyncio.to_thread(svc.get_strategy_registry)
    elapsed = (time.monotonic() - t0) * 1000
    return ApiResponse.success(data, elapsed_ms=elapsed)


@router.get("/paper-state")
async def paper_state() -> ApiResponse:
    t0 = time.monotonic()
    svc = AutopilotService()
    data = await asyncio.to_thread(svc.get_paper_state)
    elapsed = (time.monotonic() - t0) * 1000
    meta_fields = await asyncio.to_thread(_get_autopilot_meta)
    return ApiResponse.success(data, elapsed_ms=elapsed, **meta_fields)


@router.post("/run-cycle", dependencies=[Depends(require_auth)])
async def run_cycle(
    req: AutopilotRequest,
    store: JobStore = Depends(get_job_store),
    runner: JobRunner = Depends(get_job_runner),
) -> ApiResponse:
    from ..jobs.autopilot_job import execute_autopilot_job

    rec = await store.create_job("autopilot", req.model_dump())
    await runner.submit(rec.job_id, execute_autopilot_job, req.model_dump())
    return ApiResponse.success({"job_id": rec.job_id, "job_type": "autopilot", "status": "queued"})
