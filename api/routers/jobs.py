"""Job management endpoints."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from ..deps.auth import require_auth
from ..deps.providers import get_job_runner, get_job_store
from ..errors import JobNotFoundError
from ..jobs.models import JobRecord
from ..jobs.store import JobStore
from ..jobs.runner import JobRunner
from ..schemas.envelope import ApiResponse

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _not_found(job_id: str) -> None:
    """Raise JobNotFoundError to be handled by the global error handler."""
    raise JobNotFoundError(f"Job '{job_id}' not found")


@router.get("")
async def list_jobs(
    limit: int = 50,
    store: JobStore = Depends(get_job_store),
) -> ApiResponse:
    jobs = await store.list_jobs(limit=limit)
    return ApiResponse.success([j.model_dump() for j in jobs])


@router.get("/{job_id}")
async def get_job(
    job_id: str,
    store: JobStore = Depends(get_job_store),
) -> ApiResponse:
    rec = await store.get_job(job_id)
    if rec is None:
        _not_found(job_id)
    return ApiResponse.success(rec.model_dump())


@router.get("/{job_id}/events")
async def job_events(
    job_id: str,
    store: JobStore = Depends(get_job_store),
    runner: JobRunner = Depends(get_job_runner),
):
    rec = await store.get_job(job_id)
    if rec is None:
        _not_found(job_id)

    async def _generate():
        async for event in runner.subscribe_events(job_id):
            yield {"event": event.get("event", "message"), "data": json.dumps(event)}

    return EventSourceResponse(_generate())


@router.post("/{job_id}/cancel", dependencies=[Depends(require_auth)])
async def cancel_job(
    job_id: str,
    store: JobStore = Depends(get_job_store),
    runner: JobRunner = Depends(get_job_runner),
) -> ApiResponse:
    rec = await store.get_job(job_id)
    if rec is None:
        _not_found(job_id)
    cancelled = await runner.cancel(job_id)
    return ApiResponse.success({"cancelled": cancelled})
