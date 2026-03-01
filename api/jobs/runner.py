"""Async job runner with concurrency control and SSE event streaming."""
from __future__ import annotations

import asyncio
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from .models import JobStatus
from .store import JobStore

logger = logging.getLogger(__name__)


class JobRunner:
    """Executes job functions in background threads with bounded concurrency."""

    def __init__(self, store: JobStore, max_concurrent: int = 2) -> None:
        self._store = store
        self._sem = asyncio.Semaphore(max_concurrent)
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._event_subscribers: Dict[str, list] = {}

    # ── Submit & Run ─────────────────────────────────────────────────

    async def submit(
        self,
        job_id: str,
        fn: Callable[..., Any],
        *args: Any,
        on_success: Optional[Callable[[], None]] = None,
        **kwargs: Any,
    ) -> None:
        """Schedule *fn* to run in a thread for *job_id*.

        Parameters
        ----------
        on_success : optional callback invoked after the job succeeds.
        """
        task = asyncio.create_task(self._run(job_id, fn, *args, on_success=on_success, **kwargs))
        self._active_tasks[job_id] = task

    async def _run(
        self,
        job_id: str,
        fn: Callable,
        *args: Any,
        on_success: Optional[Callable[[], None]] = None,
        **kwargs: Any,
    ) -> None:
        async with self._sem:
            now = datetime.now(timezone.utc).isoformat()
            await self._store.update_status(job_id, JobStatus.running, started_at=now)
            await self._emit(job_id, {"event": "started", "job_id": job_id})

            try:
                # Build a progress callback the sync job function can call
                loop = asyncio.get_running_loop()

                def progress_callback(pct: float, msg: str = ""):
                    asyncio.run_coroutine_threadsafe(
                        self._on_progress(job_id, pct, msg), loop
                    )

                result = await asyncio.to_thread(fn, *args, progress_callback=progress_callback, **kwargs)

                now = datetime.now(timezone.utc).isoformat()
                await self._store.update_status(
                    job_id, JobStatus.succeeded, completed_at=now, result=result or {}
                )
                await self._store.update_progress(job_id, 1.0, "Done")
                if on_success is not None:
                    on_success()
                await self._emit(job_id, {"event": "completed", "job_id": job_id})
            except asyncio.CancelledError:
                now = datetime.now(timezone.utc).isoformat()
                await self._store.update_status(job_id, JobStatus.cancelled, completed_at=now)
                await self._emit(job_id, {"event": "cancelled", "job_id": job_id})
            except Exception as exc:
                now = datetime.now(timezone.utc).isoformat()
                tb = traceback.format_exc()
                logger.error("Job %s failed: %s\n%s", job_id, exc, tb)
                await self._store.update_status(
                    job_id, JobStatus.failed, completed_at=now, error=str(exc)
                )
                await self._emit(job_id, {"event": "failed", "job_id": job_id, "error": str(exc)})
            finally:
                self._active_tasks.pop(job_id, None)
                await self._emit(job_id, {"event": "done", "job_id": job_id})

    async def _on_progress(self, job_id: str, pct: float, msg: str) -> None:
        await self._store.update_progress(job_id, pct, msg)
        await self._emit(job_id, {"event": "progress", "job_id": job_id, "progress": pct, "message": msg})

    # ── Cancel ───────────────────────────────────────────────────────

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running job."""
        task = self._active_tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            return True
        return await self._store.cancel_job(job_id)

    # ── SSE Event Streaming ──────────────────────────────────────────

    async def subscribe_events(self, job_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Yield SSE events for a job until it is done."""
        queue: asyncio.Queue = asyncio.Queue()
        self._event_subscribers.setdefault(job_id, []).append(queue)
        try:
            # Send current state as first event
            rec = await self._store.get_job(job_id)
            if rec:
                yield {"event": "status", "data": rec.model_dump()}

            while True:
                event = await queue.get()
                yield event
                if event.get("event") in ("done", "cancelled"):
                    break
        finally:
            subs = self._event_subscribers.get(job_id, [])
            if queue in subs:
                subs.remove(queue)

    async def _emit(self, job_id: str, event: Dict[str, Any]) -> None:
        for q in self._event_subscribers.get(job_id, []):
            await q.put(event)
