"""FastAPI application factory and server entry point."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import ApiSettings
from .deps.providers import get_cache, get_job_store, get_settings
from .errors import register_error_handlers

logger = logging.getLogger(__name__)

# Background retrain monitor interval (seconds)
_RETRAIN_CHECK_INTERVAL = 300  # 5 minutes


async def _retrain_monitor_loop() -> None:
    """Background task that periodically checks retrain triggers.

    If a retrain is overdue, invalidates stale caches so the dashboard
    KPIs reflect the current state.  Runs every 5 minutes.
    """
    from .cache.invalidation import invalidate_on_train

    while True:
        await asyncio.sleep(_RETRAIN_CHECK_INTERVAL)
        try:
            from .services.backtest_service import BacktestService
            staleness = await asyncio.to_thread(BacktestService()._compute_model_staleness)
            if staleness.get("overdue"):
                logger.warning(
                    "Model retrain overdue (%s days). Invalidating dashboard cache.",
                    staleness.get("days"),
                )
                cache = get_cache()
                invalidate_on_train(cache)
        except Exception:  # noqa: BLE001
            logger.debug("Retrain monitor check failed", exc_info=True)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    settings: ApiSettings = get_settings()

    # Configure structured log format using engine config
    try:
        from quant_engine.config import LOG_LEVEL, LOG_FORMAT
        effective_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    except (ImportError, AttributeError):
        effective_level = getattr(logging, settings.log_level.upper(), logging.INFO)
        LOG_FORMAT = "structured"

    if LOG_FORMAT == "json":
        fmt = '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
    else:
        fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

    logging.basicConfig(
        level=effective_level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger.info("Starting quant_engine API on %s:%s", settings.host, settings.port)

    # Run config validation on startup
    try:
        from quant_engine.config import validate_config
        issues = validate_config()
        for issue in issues:
            level = issue.get("level", "WARNING")
            msg = issue.get("message", "")
            if level == "ERROR":
                logger.error("Config validation: %s", msg)
            else:
                logger.warning("Config validation: %s", msg)
        if not issues:
            logger.info("Config validation: all checks passed")
    except Exception as e:
        logger.warning("Config validation could not run: %s", e)

    # Initialise async resources
    store = get_job_store()
    await store.initialize()

    # Start background retrain monitor
    monitor_task = asyncio.create_task(_retrain_monitor_loop())

    yield

    # Cleanup
    monitor_task.cancel()
    logger.info("Shutting down quant_engine API")


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    """Build and return the FastAPI application."""
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="Quant Engine API",
        description="Continuous Feature ML Trading System — REST API for backtesting, signals, model training, and autopilot.",
        version="2.0.0",
        lifespan=_lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
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
