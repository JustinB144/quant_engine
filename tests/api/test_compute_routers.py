"""Tests for POST compute endpoints — verify job creation."""
import pytest


@pytest.mark.asyncio
async def test_train_creates_job(client):
    resp = await client.post("/api/models/train", json={"horizons": [10]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "job_id" in body["data"]
    assert body["data"]["job_type"] == "train"
    assert body["data"]["status"] == "queued"


@pytest.mark.asyncio
async def test_predict_creates_job(client):
    resp = await client.post("/api/models/predict", json={"horizon": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "job_id" in body["data"]
    assert body["data"]["job_type"] == "predict"


@pytest.mark.asyncio
async def test_backtest_creates_job(client):
    resp = await client.post("/api/backtests/run", json={"horizon": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "job_id" in body["data"]
    assert body["data"]["job_type"] == "backtest"


@pytest.mark.asyncio
async def test_autopilot_creates_job(client):
    resp = await client.post("/api/autopilot/run-cycle", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "job_id" in body["data"]
    assert body["data"]["job_type"] == "autopilot"


@pytest.mark.asyncio
async def test_job_status_queryable(client):
    # Create a job
    resp = await client.post("/api/models/train", json={"horizons": [10]})
    job_id = resp.json()["data"]["job_id"]

    # Query its status
    resp = await client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["data"]["job_id"] == job_id


@pytest.mark.asyncio
async def test_nonexistent_job_404(client):
    resp = await client.get("/api/jobs/nonexistent123")
    assert resp.status_code == 404
    body = resp.json()
    assert body["ok"] is False
