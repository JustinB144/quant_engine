"""SQLite-backed persistence for job records."""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

import aiosqlite

from .models import JobRecord, JobStatus


class JobStore:
    """Async SQLite store for job lifecycle tracking."""

    def __init__(self, db_path: str = "api_jobs.db") -> None:
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Create the jobs table if it doesn't exist."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                progress REAL DEFAULT 0.0,
                progress_message TEXT DEFAULT '',
                params TEXT DEFAULT '{}',
                result TEXT,
                error TEXT
            )
        """)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    # ── CRUD ─────────────────────────────────────────────────────────

    async def create_job(self, job_type: str, params: Dict[str, Any] | None = None) -> JobRecord:
        """Insert a new queued job and return its record."""
        if self._db is None:
            await self.initialize()
        rec = JobRecord(
            job_id=uuid.uuid4().hex[:12],
            job_type=job_type,
            params=params or {},
        )
        await self._db.execute(
            "INSERT INTO jobs (job_id, job_type, status, created_at, params) VALUES (?,?,?,?,?)",
            (rec.job_id, rec.job_type, rec.status.value, rec.created_at, json.dumps(rec.params)),
        )
        await self._db.commit()
        return rec

    async def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Fetch a single job by ID."""
        if self._db is None:
            await self.initialize()
        async with self._db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return self._row_to_record(row, cur.description)

    async def list_jobs(self, limit: int = 50) -> List[JobRecord]:
        """List jobs ordered by creation time (newest first)."""
        if self._db is None:
            await self.initialize()
        async with self._db.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
            desc = cur.description
        return [self._row_to_record(r, desc) for r in rows]

    async def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        started_at: str | None = None,
        completed_at: str | None = None,
        result: Dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Update job status and optional completion fields."""
        sets = ["status = ?"]
        vals: list = [status.value]
        if started_at is not None:
            sets.append("started_at = ?")
            vals.append(started_at)
        if completed_at is not None:
            sets.append("completed_at = ?")
            vals.append(completed_at)
        if result is not None:
            sets.append("result = ?")
            vals.append(json.dumps(result))
        if error is not None:
            sets.append("error = ?")
            vals.append(error)
        vals.append(job_id)
        if self._db is None:
            await self.initialize()
        await self._db.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?", vals)
        await self._db.commit()

    async def update_progress(self, job_id: str, progress: float, message: str = "") -> None:
        """Update job progress (0.0 – 1.0) and optional message."""
        if self._db is None:
            await self.initialize()
        await self._db.execute(
            "UPDATE jobs SET progress = ?, progress_message = ? WHERE job_id = ?",
            (progress, message, job_id),
        )
        await self._db.commit()

    async def cancel_job(self, job_id: str) -> bool:
        """Mark a job as cancelled if it is still queued or running."""
        if self._db is None:
            await self.initialize()
        async with self._db.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            return False
        if row[0] not in (JobStatus.queued.value, JobStatus.running.value):
            return False
        await self.update_status(job_id, JobStatus.cancelled)
        return True

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _row_to_record(row, description) -> JobRecord:
        cols = [d[0] for d in description]
        d = dict(zip(cols, row))
        d["params"] = json.loads(d.get("params") or "{}")
        d["result"] = json.loads(d["result"]) if d.get("result") else None
        return JobRecord(**d)
