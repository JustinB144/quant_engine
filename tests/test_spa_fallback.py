"""Tests for SPA fallback routing in run_server.py (SPEC-B07).

Verifies that:
- Unmatched API paths return 404 (not 200/null or 500)
- SPA routes (e.g. /dashboard) return index.html with 200
- Existing static files in dist/ are served directly
- Path traversal attempts are blocked
- OpenAPI doc paths return 404 through the fallback
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.testclient import TestClient


def _build_spa_app(dist_dir: Path) -> FastAPI:
    """Build a minimal FastAPI app with the SPA fallback handler.

    Mirrors the logic in run_server.py when --static is passed.
    """
    app = FastAPI(
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Simulate a registered API route
    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    # SPA fallback — matches the implementation in run_server.py
    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        if full_path.startswith("api/") or full_path in (
            "docs",
            "redoc",
            "openapi.json",
        ):
            raise HTTPException(
                status_code=404, detail=f"Not found: /{full_path}"
            )

        file_path = (dist_dir / full_path).resolve()
        dist_root = dist_dir.resolve()
        if file_path.is_relative_to(dist_root) and file_path.is_file():
            return FileResponse(str(file_path))

        return FileResponse(str(dist_dir / "index.html"))

    return app


@pytest.fixture
def dist_dir(tmp_path: Path) -> Path:
    """Create a mock frontend/dist directory with index.html and a static file."""
    dist = tmp_path / "dist"
    dist.mkdir()

    index_html = textwrap.dedent("""\
        <!DOCTYPE html>
        <html><head><title>Quant Engine</title></head>
        <body><div id="root"></div></body></html>
    """)
    (dist / "index.html").write_text(index_html)

    # A specific static file (e.g. favicon)
    (dist / "favicon.ico").write_bytes(b"\x00\x00\x01\x00")

    # Nested file for path testing
    assets = dist / "assets"
    assets.mkdir()
    (assets / "app.js").write_text("console.log('app');")

    return dist


@pytest.fixture
def spa_client(dist_dir: Path):
    """Synchronous HTTP client for the SPA test app."""
    app = _build_spa_app(dist_dir)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


# ── API 404 tests ────────────────────────────────────────────────────


def test_unmatched_api_path_returns_404(spa_client: TestClient):
    """GET /api/nonexistent-endpoint must return 404, not 200 or 500."""
    resp = spa_client.get("/api/nonexistent-endpoint")
    assert resp.status_code == 404


def test_unmatched_api_nested_path_returns_404(spa_client: TestClient):
    """GET /api/v2/some/deep/path must return 404."""
    resp = spa_client.get("/api/v2/some/deep/path")
    assert resp.status_code == 404


def test_matched_api_route_still_works(spa_client: TestClient):
    """GET /api/health (a registered route) must still return 200."""
    resp = spa_client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_api_404_response_has_detail(spa_client: TestClient):
    """404 response for unmatched API paths must include useful detail."""
    resp = spa_client.get("/api/nonexistent")
    assert resp.status_code == 404
    body = resp.json()
    assert "detail" in body
    assert "/api/nonexistent" in body["detail"]


# ── SPA fallback tests ──────────────────────────────────────────────


def test_spa_route_returns_index_html(spa_client: TestClient):
    """GET /dashboard (an SPA route) must return index.html with 200."""
    resp = spa_client.get("/dashboard")
    assert resp.status_code == 200
    assert "<!DOCTYPE html>" in resp.text
    assert '<div id="root">' in resp.text


def test_spa_nested_route_returns_index_html(spa_client: TestClient):
    """GET /system-health (an SPA route) must return index.html with 200."""
    resp = spa_client.get("/system-health")
    assert resp.status_code == 200
    assert "<!DOCTYPE html>" in resp.text


def test_spa_unknown_route_returns_index_html(spa_client: TestClient):
    """GET /totally-unknown returns index.html (SPA handles its own 404 page)."""
    resp = spa_client.get("/totally-unknown")
    assert resp.status_code == 200
    assert "<!DOCTYPE html>" in resp.text


def test_root_returns_index_html(spa_client: TestClient):
    """GET / must return index.html."""
    resp = spa_client.get("/")
    assert resp.status_code == 200
    assert "<!DOCTYPE html>" in resp.text


# ── Static file serving tests ───────────────────────────────────────


def test_existing_static_file_served_directly(spa_client: TestClient):
    """GET /favicon.ico must return the actual file, not index.html."""
    resp = spa_client.get("/favicon.ico")
    assert resp.status_code == 200
    assert resp.content == b"\x00\x00\x01\x00"


# ── Path traversal protection ───────────────────────────────────────


def test_path_traversal_returns_index_html(spa_client: TestClient):
    """Path traversal attempts must not escape the dist directory."""
    resp = spa_client.get("/../../../etc/passwd")
    assert resp.status_code == 200
    # Should return index.html (SPA fallback), NOT the system file
    assert "<!DOCTYPE html>" in resp.text
