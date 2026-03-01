"""Authentication dependency for mutation endpoints."""
from __future__ import annotations

import logging
import os

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

# Auth configuration — read from environment
API_AUTH_ENABLED: bool = os.environ.get("QUANT_ENGINE_API_AUTH_ENABLED", "true").lower() in (
    "true", "1", "yes",
)
API_AUTH_TOKEN: str = os.environ.get("QUANT_ENGINE_API_TOKEN", "")


async def require_auth(request: Request) -> None:
    """FastAPI dependency that enforces bearer-token / API-key authentication.

    Reads the token from ``Authorization: Bearer <token>`` or the
    ``X-API-Key`` header.  Returns immediately when auth is disabled
    (local dev mode).

    Raises
    ------
    HTTPException(401)
        If the token is missing, empty, or does not match.
    """
    if not API_AUTH_ENABLED:
        return

    if not API_AUTH_TOKEN:
        logger.warning(
            "API_AUTH_ENABLED is True but QUANT_ENGINE_API_TOKEN is not set. "
            "All mutation requests will be rejected."
        )
        raise HTTPException(status_code=401, detail="Server auth token not configured")

    # Try Authorization header first, then X-API-Key
    token: str | None = None

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()

    if not token:
        token = request.headers.get("X-API-Key", "").strip() or None

    if not token:
        raise HTTPException(status_code=401, detail="Missing authentication token")

    if token != API_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
