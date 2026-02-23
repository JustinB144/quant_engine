"""Integration tests — full app startup, envelope consistency, config patch."""
import pytest


@pytest.mark.asyncio
async def test_all_gets_return_envelope(client):
    """Every GET endpoint should return an ApiResponse envelope."""
    get_endpoints = [
        "/api/health",
        "/api/dashboard/summary",
        "/api/data/universe",
        "/api/models/versions",
        "/api/signals/latest",
        "/api/backtests/latest",
        "/api/backtests/latest/trades",
        "/api/backtests/latest/equity-curve",
        "/api/autopilot/latest-cycle",
        "/api/autopilot/strategies",
        "/api/autopilot/paper-state",
        "/api/config",
        "/api/logs",
        "/api/jobs",
    ]
    for ep in get_endpoints:
        resp = await client.get(ep)
        assert resp.status_code == 200, f"{ep} returned {resp.status_code}"
        body = resp.json()
        assert "ok" in body, f"{ep} missing 'ok'"
        assert "data" in body, f"{ep} missing 'data'"
        assert "meta" in body, f"{ep} missing 'meta'"


@pytest.mark.asyncio
async def test_config_patch_and_read(client):
    """PATCH /api/config should update values and be reflected in GET."""
    # Read current config
    resp = await client.get("/api/config")
    original = resp.json()["data"]
    original_threshold = original["ENTRY_THRESHOLD"]

    # Patch
    new_val = 0.01
    resp = await client.patch("/api/config", json={"ENTRY_THRESHOLD": new_val})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["data"]["ENTRY_THRESHOLD"] == new_val

    # Read again
    resp = await client.get("/api/config")
    assert resp.json()["data"]["ENTRY_THRESHOLD"] == new_val

    # Restore
    await client.patch("/api/config", json={"ENTRY_THRESHOLD": original_threshold})


@pytest.mark.asyncio
async def test_config_patch_invalid_key(client):
    """PATCH with invalid key should return 422."""
    resp = await client.patch("/api/config", json={"INVALID_KEY": 42})
    assert resp.status_code == 422
    body = resp.json()
    assert body["ok"] is False


@pytest.mark.asyncio
async def test_meta_has_generated_at(client):
    """Response meta should always include generated_at timestamp."""
    resp = await client.get("/api/health")
    meta = resp.json()["meta"]
    assert "generated_at" in meta
    assert meta["generated_at"] is not None
