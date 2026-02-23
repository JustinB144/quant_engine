"""Tests for all GET endpoints — verify ApiResponse envelope."""
import pytest


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "data" in body
    assert "meta" in body


@pytest.mark.asyncio
async def test_dashboard_summary(client):
    resp = await client.get("/api/dashboard/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True


@pytest.mark.asyncio
async def test_data_universe(client):
    resp = await client.get("/api/data/universe")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "full_size" in body["data"]


@pytest.mark.asyncio
async def test_models_versions(client):
    resp = await client.get("/api/models/versions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True


@pytest.mark.asyncio
async def test_signals_latest(client):
    resp = await client.get("/api/signals/latest?horizon=10")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True


@pytest.mark.asyncio
async def test_backtests_latest(client):
    resp = await client.get("/api/backtests/latest?horizon=10")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True


@pytest.mark.asyncio
async def test_backtests_trades(client):
    resp = await client.get("/api/backtests/latest/trades?horizon=10")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True


@pytest.mark.asyncio
async def test_backtests_equity_curve(client):
    resp = await client.get("/api/backtests/latest/equity-curve?horizon=10")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True


@pytest.mark.asyncio
async def test_autopilot_latest_cycle(client):
    resp = await client.get("/api/autopilot/latest-cycle")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True


@pytest.mark.asyncio
async def test_autopilot_strategies(client):
    resp = await client.get("/api/autopilot/strategies")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True


@pytest.mark.asyncio
async def test_autopilot_paper_state(client):
    resp = await client.get("/api/autopilot/paper-state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True


@pytest.mark.asyncio
async def test_config_get(client):
    resp = await client.get("/api/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "ENTRY_THRESHOLD" in body["data"]


@pytest.mark.asyncio
async def test_logs(client):
    resp = await client.get("/api/logs?last_n=10")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True


@pytest.mark.asyncio
async def test_jobs_list_empty(client):
    resp = await client.get("/api/jobs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
