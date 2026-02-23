"""Job data models."""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class JobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class JobRecord(BaseModel):
    """Persistent representation of a compute job."""

    job_id: str
    job_type: str
    status: JobStatus = JobStatus.queued
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: float = 0.0
    progress_message: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
