"""Log retrieval endpoint."""
from __future__ import annotations

import logging
from collections import deque

from fastapi import APIRouter

from ..schemas.envelope import ApiResponse

router = APIRouter(prefix="/api/logs", tags=["logs"])

# In-memory ring buffer for recent log records.
_LOG_BUFFER: deque = deque(maxlen=500)


class _BufferHandler(logging.Handler):
    """Captures log records into the ring buffer."""

    def emit(self, record: logging.LogRecord) -> None:
        _LOG_BUFFER.append({
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        })


# Attach the handler to the root quant_engine logger.
_handler = _BufferHandler()
_handler.setLevel(logging.INFO)
logging.getLogger("quant_engine").addHandler(_handler)


@router.get("")
async def get_logs(last_n: int = 100) -> ApiResponse:
    entries = list(_LOG_BUFFER)[-last_n:]
    return ApiResponse.success(entries)
