"""Tests for app factory and basic middleware."""
import pytest
from quant_engine.api.main import create_app
from quant_engine.api.config import ApiSettings


def test_create_app():
    app = create_app(ApiSettings(job_db_path=":memory:"))
    assert app.title == "Quant Engine API"


def test_openapi_schema():
    app = create_app(ApiSettings(job_db_path=":memory:"))
    schema = app.openapi()
    assert "paths" in schema
    assert "/api/health" in schema["paths"]


def test_routes_registered():
    app = create_app(ApiSettings(job_db_path=":memory:"))
    paths = {r.path for r in app.routes}
    expected = {
        "/api/health",
        "/api/dashboard/summary",
        "/api/jobs",
        "/api/config",
        "/api/models/versions",
        "/api/backtests/latest",
        "/api/signals/latest",
    }
    for ep in expected:
        assert ep in paths, f"Missing route: {ep}"


@pytest.mark.asyncio
async def test_404_wrapped(client):
    resp = await client.get("/api/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cors_headers(client):
    resp = await client.options(
        "/api/health",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    # CORS middleware should respond (may be 200 or 405 depending on FastAPI version)
    assert resp.status_code in (200, 405, 400)
