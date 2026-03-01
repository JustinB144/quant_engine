"""Combined API + frontend static serving entry point.

Usage:
    # Development (API only, frontend uses Vite dev server):
    python run_server.py

    # Production (API + static frontend from build):
    python run_server.py --static

    # Custom host/port:
    python run_server.py --host 0.0.0.0 --port 9000 --static
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure quant_engine is importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quant Engine API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--static", action="store_true", help="Serve frontend/dist as static files")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import uvicorn

    from quant_engine.api.config import ApiSettings
    from quant_engine.api.main import create_app

    settings = ApiSettings(host=args.host, port=args.port, log_level=args.log_level)
    app = create_app(settings)

    if args.static:
        if not FRONTEND_DIST.exists():
            logger.error(
                "Frontend build not found at %s. Run 'cd frontend && npm run build' first.",
                FRONTEND_DIST,
            )
            sys.exit(1)

        from fastapi.staticfiles import StaticFiles

        # Serve static assets (JS/CSS/images) from dist/assets/
        app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="static-assets")

        # SPA fallback: serve index.html for all non-API routes
        from fastapi import HTTPException
        from fastapi.responses import FileResponse

        @app.get("/{full_path:path}")
        async def spa_fallback(full_path: str):
            # Unmatched API routes and OpenAPI docs should return a clean 404,
            # not the SPA fallback page (which would produce a confusing 200/500).
            if full_path.startswith("api/") or full_path in (
                "docs",
                "redoc",
                "openapi.json",
            ):
                raise HTTPException(
                    status_code=404, detail=f"Not found: /{full_path}"
                )

            # Serve a specific static file from dist/ if it exists (e.g. favicon.ico)
            file_path = (FRONTEND_DIST / full_path).resolve()
            dist_root = FRONTEND_DIST.resolve()
            if file_path.is_relative_to(dist_root) and file_path.is_file():
                return FileResponse(str(file_path))

            # SPA fallback: all other routes get index.html so React Router
            # can handle client-side routing (including its own 404 page).
            return FileResponse(str(FRONTEND_DIST / "index.html"))

        logger.info("Serving frontend static files from %s", FRONTEND_DIST)

    logger.info("Starting Quant Engine API on %s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
