"""Tests for the job system: store, runner, lifecycle."""
import asyncio

import pytest

from quant_engine.api.jobs.models import JobRecord, JobStatus
from quant_engine.api.jobs.store import JobStore
from quant_engine.api.jobs.runner import JobRunner


@pytest.fixture
async def store():
    s = JobStore(":memory:")
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
async def runner(store):
    return JobRunner(store)


@pytest.mark.asyncio
async def test_create_and_get_job(store):
    rec = await store.create_job("test_type", {"key": "val"})
    assert rec.job_type == "test_type"
    assert rec.status == JobStatus.queued

    fetched = await store.get_job(rec.job_id)
    assert fetched is not None
    assert fetched.job_id == rec.job_id
    assert fetched.params == {"key": "val"}


@pytest.mark.asyncio
async def test_list_jobs(store):
    await store.create_job("a")
    await store.create_job("b")
    jobs = await store.list_jobs()
    assert len(jobs) == 2


@pytest.mark.asyncio
async def test_update_status(store):
    rec = await store.create_job("test")
    await store.update_status(rec.job_id, JobStatus.running, started_at="2026-01-01T00:00:00")
    fetched = await store.get_job(rec.job_id)
    assert fetched.status == JobStatus.running
    assert fetched.started_at == "2026-01-01T00:00:00"


@pytest.mark.asyncio
async def test_update_progress(store):
    rec = await store.create_job("test")
    await store.update_progress(rec.job_id, 0.5, "halfway")
    fetched = await store.get_job(rec.job_id)
    assert fetched.progress == 0.5
    assert fetched.progress_message == "halfway"


@pytest.mark.asyncio
async def test_cancel_queued_job(store):
    rec = await store.create_job("test")
    result = await store.cancel_job(rec.job_id)
    assert result is True
    fetched = await store.get_job(rec.job_id)
    assert fetched.status == JobStatus.cancelled


@pytest.mark.asyncio
async def test_cancel_completed_job_fails(store):
    rec = await store.create_job("test")
    await store.update_status(rec.job_id, JobStatus.succeeded)
    result = await store.cancel_job(rec.job_id)
    assert result is False


@pytest.mark.asyncio
async def test_get_nonexistent_job(store):
    result = await store.get_job("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_job_runner_succeeds(store, runner):
    def my_job(progress_callback=None, cancel_event=None):
        if progress_callback:
            progress_callback(0.5, "half")
        return {"answer": 42}

    rec = await store.create_job("test")
    await runner.submit(rec.job_id, my_job)
    # Wait for completion
    await asyncio.sleep(0.3)
    fetched = await store.get_job(rec.job_id)
    assert fetched.status == JobStatus.succeeded
    assert fetched.result == {"answer": 42}


@pytest.mark.asyncio
async def test_job_runner_failure(store, runner):
    def failing_job(progress_callback=None, cancel_event=None):
        raise ValueError("boom")

    rec = await store.create_job("test")
    await runner.submit(rec.job_id, failing_job)
    await asyncio.sleep(0.3)
    fetched = await store.get_job(rec.job_id)
    assert fetched.status == JobStatus.failed
    assert "boom" in fetched.error


@pytest.mark.asyncio
async def test_sse_events(store, runner):
    def slow_job(progress_callback=None, cancel_event=None):
        import time
        time.sleep(0.1)
        if progress_callback:
            progress_callback(0.5, "half")
        time.sleep(0.1)
        return {"done": True}

    rec = await store.create_job("test")
    await runner.submit(rec.job_id, slow_job)

    events = []
    async for event in runner.subscribe_events(rec.job_id):
        events.append(event)
        if event.get("event") == "done":
            break

    event_types = [e.get("event") for e in events]
    assert "status" in event_types  # initial status
    assert "done" in event_types
