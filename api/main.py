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

# Background health scan interval (seconds) — T2 of SPEC_AUDIT_FIX_04
_HEALTH_SCAN_INTERVAL = 900  # 15 minutes

# Shared HealthService instance for background scans
_health_service_instance = None


async def _health_scan_loop() -> None:
    """Background task that periodically runs heavy cache scans.

    Runs _check_data_anomalies and _check_market_microstructure every
    15 minutes so the health endpoint can serve cached results instantly.
    """
    global _health_service_instance
    from .services.health_service import HealthService

    _health_service_instance = HealthService()

    # Run initial scan immediately
    try:
        await asyncio.to_thread(_health_service_instance.run_background_scans)
    except Exception:
        logger.debug("Initial health scan failed", exc_info=True)

    while True:
        await asyncio.sleep(_HEALTH_SCAN_INTERVAL)
        try:
            await asyncio.to_thread(_health_service_instance.run_background_scans)
        except Exception:  # noqa: BLE001
            logger.debug("Background health scan failed", exc_info=True)


def get_health_service_instance():
    """Return the shared HealthService with cached background scan results."""
    global _health_service_instance
    if _health_service_instance is None:
        from .services.health_service import HealthService
        _health_service_instance = HealthService()
    return _health_service_instance


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

    # Clear stale caches from previous runs
    try:
        from .cache.invalidation import invalidate_on_data_refresh
        invalidate_on_data_refresh(get_cache())
        logger.info("Cleared stale data caches from previous run")
    except Exception:
        logger.debug("Startup cache invalidation skipped", exc_info=True)

    # Attach log buffer handler
    from .routers.logs import setup_log_buffer, teardown_log_buffer
    setup_log_buffer()

    # Initialise async resources
    store = get_job_store()
    await store.initialize()

    # Start background retrain monitor
    monitor_task = asyncio.create_task(_retrain_monitor_loop())

    # Start background health scan (SPEC_AUDIT_FIX_04 T2)
    health_scan_task = asyncio.create_task(_health_scan_loop())

    yield

    # Cleanup
    monitor_task.cancel()
    health_scan_task.cancel()
    teardown_log_buffer()
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
    allow_creds = "*" not in origins
    if not allow_creds:
        logger.warning(
            "CORS_ORIGINS contains '*'. Credentials will NOT be allowed. "
            "Set explicit origins (e.g. 'http://localhost:5173') for "
            "credentialed cross-origin requests."
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_creds,
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
