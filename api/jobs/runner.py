"""Async job runner with concurrency control and SSE event streaming."""
from __future__ import annotations

import asyncio
import logging
import threading
import traceback
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from .models import JobStatus
from .store import JobStore

logger = logging.getLogger(__name__)


class JobCancelled(Exception):
    """Raised by job functions when cooperative cancellation is detected."""


class JobQueueFullError(Exception):
    """Raised when the job queue is at capacity."""


class JobRunner:
    """Executes job functions in background threads with bounded concurrency."""

    def __init__(self, store: JobStore, max_concurrent: int = 2, max_queued: int = 20) -> None:
        self._store = store
        self._sem = asyncio.Semaphore(max_concurrent)
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._cancel_events: Dict[str, threading.Event] = {}
        self._event_subscribers: Dict[str, list] = {}
        self._max_queued = max_queued

    # ── Submit & Run ─────────────────────────────────────────────────

    @property
    def pending_count(self) -> int:
        """Number of jobs queued or running."""
        return len(self._active_tasks)

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

        Raises
        ------
        JobQueueFullError
            If the number of pending jobs exceeds ``max_queued``.
        """
        if len(self._active_tasks) >= self._max_queued:
            raise JobQueueFullError(
                f"Job queue full. {self._max_queued} jobs pending. Try again later."
            )
        cancel_event = threading.Event()
        self._cancel_events[job_id] = cancel_event
        task = asyncio.create_task(
            self._run(job_id, fn, *args, cancel_event=cancel_event, on_success=on_success, **kwargs)
        )
        self._active_tasks[job_id] = task

    async def _run(
        self,
        job_id: str,
        fn: Callable,
        *args: Any,
        cancel_event: Optional[threading.Event] = None,
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

                def progress_callback(pct: float, msg: str = "") -> None:
                    asyncio.run_coroutine_threadsafe(
                        self._on_progress(job_id, pct, msg), loop
                    )

                result = await asyncio.to_thread(
                    fn, *args,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                    **kwargs,
                )

                now = datetime.now(timezone.utc).isoformat()
                await self._store.update_status(
                    job_id, JobStatus.succeeded, completed_at=now, result=result or {}
                )
                await self._store.update_progress(job_id, 1.0, "Done")
                if on_success is not None:
                    on_success()
                await self._emit(job_id, {"event": "completed", "job_id": job_id})
            except (asyncio.CancelledError, JobCancelled):
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
                self._cancel_events.pop(job_id, None)
                await self._emit(job_id, {"event": "done", "job_id": job_id})

    async def _on_progress(self, job_id: str, pct: float, msg: str) -> None:
        await self._store.update_progress(job_id, pct, msg)
        await self._emit(job_id, {"event": "progress", "job_id": job_id, "progress": pct, "message": msg})

    # ── Cancel ───────────────────────────────────────────────────────

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running job.

        Sets the cooperative cancellation event so the thread-side function
        can detect cancellation, and also cancels the asyncio wrapper task.
        """
        cancel_event = self._cancel_events.get(job_id)
        if cancel_event is not None:
            cancel_event.set()
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
