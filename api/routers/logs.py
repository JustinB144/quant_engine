"""Log retrieval endpoint."""
from __future__ import annotations

import logging
import time
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


_handler = _BufferHandler()
_handler.setLevel(logging.INFO)


def setup_log_buffer() -> None:
    """Attach the buffer handler to the ``quant_engine`` logger.

    Safe to call multiple times — a duplicate-attachment guard prevents
    adding the handler more than once.  Call this from the app lifespan.
    """
    qe_logger = logging.getLogger("quant_engine")
    if not any(isinstance(h, _BufferHandler) for h in qe_logger.handlers):
        qe_logger.addHandler(_handler)


def teardown_log_buffer() -> None:
    """Detach the buffer handler.  Useful for test cleanup."""
    qe_logger = logging.getLogger("quant_engine")
    qe_logger.removeHandler(_handler)
    _LOG_BUFFER.clear()


@router.get("")
async def get_logs(last_n: int = 100) -> ApiResponse:
    t0 = time.monotonic()
    entries = list(_LOG_BUFFER)[-last_n:]
    elapsed = (time.monotonic() - t0) * 1000
    return ApiResponse.success(entries, elapsed_ms=elapsed)
