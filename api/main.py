"""FastAPI application factory and server entry point."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import ApiSettings
from .deps.providers import get_cache, get_job_store, get_settings
from .errors import register_error_handlers

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    settings: ApiSettings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.info("Starting quant_engine API on %s:%s", settings.host, settings.port)

    # Initialise async resources
    store = get_job_store()
    await store.initialize()

    yield

    # Cleanup
    logger.info("Shutting down quant_engine API")


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    """Build and return the FastAPI application."""
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="Quant Engine API",
        version="0.1.0",
        lifespan=_lifespan,
    )

    # CORS
    origins = [o.strip() for o in settings.cors_origins.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Error handlers
    register_error_handlers(app)

    # Routers — imported lazily so missing engine deps don't block startup
    from .routers import all_routers

    for router in all_routers():
        app.include_router(router)

    return app


def run_server() -> None:
    """CLI entry point: ``python -m quant_engine.api.main``."""
    import uvicorn

    settings = get_settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    run_server()
