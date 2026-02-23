"""Custom exceptions and FastAPI error handler registration."""
from __future__ import annotations

import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .schemas.envelope import ApiResponse

logger = logging.getLogger(__name__)


# ── Custom Exceptions ────────────────────────────────────────────────


class DataNotFoundError(Exception):
    """Requested data or ticker not available."""


class TrainingFailedError(Exception):
    """Model training could not complete."""


class JobNotFoundError(Exception):
    """Requested job ID does not exist."""


class ConfigValidationError(Exception):
    """Runtime config patch contains invalid keys or values."""


class ServiceUnavailableError(Exception):
    """An engine dependency is not ready (e.g. WRDS offline)."""


# ── Error → HTTP mapping ────────────────────────────────────────────

_EXCEPTION_STATUS = {
    DataNotFoundError: 404,
    JobNotFoundError: 404,
    ConfigValidationError: 422,
    TrainingFailedError: 500,
    ServiceUnavailableError: 503,
}


def _make_handler(status_code: int):
    """Create a handler that wraps an exception in ApiResponse."""

    async def _handler(request: Request, exc: Exception) -> JSONResponse:
        resp = ApiResponse.fail(str(exc))
        return JSONResponse(status_code=status_code, content=resp.model_dump())

    return _handler


def register_error_handlers(app: FastAPI) -> None:
    """Register custom exception handlers on the FastAPI app."""
    for exc_cls, status in _EXCEPTION_STATUS.items():
        app.add_exception_handler(exc_cls, _make_handler(status))

    # Catch-all for unexpected errors
    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled error: %s\n%s", exc, traceback.format_exc())
        resp = ApiResponse.fail("Internal server error")
        return JSONResponse(status_code=500, content=resp.model_dump())
