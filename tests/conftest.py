"""Shared test fixtures for the quant_engine test suite."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ── Data fixtures ────────────────────────────────────────────────────


@pytest.fixture
def synthetic_ohlcv_data():
    """10 synthetic stock series, each with 500 daily bars."""
    rng = np.random.default_rng(42)
    tickers = [f"TEST{i}" for i in range(10)]
    data = {}
    for i, ticker in enumerate(tickers):
        permno = 10000 + i
        n = 500
        dates = pd.bdate_range("2022-01-03", periods=n)
        close = 100.0 + np.cumsum(rng.normal(0.0005, 0.02, n))
        close = np.maximum(close, 1.0)
        high = close * (1 + rng.uniform(0, 0.02, n))
        low = close * (1 - rng.uniform(0, 0.02, n))
        opn = close * (1 + rng.normal(0, 0.005, n))
        vol = rng.integers(100_000, 10_000_000, n).astype(float)
        df = pd.DataFrame(
            {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
            index=dates,
        )
        df.attrs["ticker"] = ticker
        df.attrs["permno"] = permno
        data[permno] = df
    return data


@pytest.fixture
def synthetic_trades_csv(tmp_path):
    """Generate a synthetic backtest trades CSV and return its path."""
    rng = np.random.default_rng(99)
    n = 200
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    rows = []
    for i in range(n):
        entry_date = pd.Timestamp("2023-01-03") + pd.Timedelta(days=i * 2)
        exit_date = entry_date + pd.Timedelta(days=rng.integers(1, 15))
        ret = float(rng.normal(0.002, 0.03))
        rows.append({
            "ticker": rng.choice(tickers),
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "entry_price": round(float(100 + rng.normal(0, 20)), 2),
            "exit_price": round(float(100 + rng.normal(0, 20)), 2),
            "predicted_return": round(ret + rng.normal(0, 0.01), 6),
            "actual_return": round(ret, 6),
            "net_return": round(ret, 6),
            "regime": rng.choice(["Trending Bull", "Trending Bear", "Mean Reverting", "High Volatility"]),
            "confidence": round(float(rng.uniform(0.4, 0.9)), 4),
            "holding_days": int(rng.integers(1, 15)),
            "position_size": round(float(rng.uniform(0.02, 0.10)), 4),
            "exit_reason": rng.choice(["horizon", "stop_loss", "target"]),
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "backtest_10d_trades.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def synthetic_model_meta(tmp_model_dir):
    """Write a synthetic model metadata JSON and return the model dir."""
    meta = {
        "version_id": "abc12345",
        "training_date": "2024-01-15",
        "horizon": 10,
        "n_samples": 5000,
        "n_features": 45,
        "oos_spearman": 0.08,
        "cv_gap": 0.02,
        "holdout_r2": 0.04,
        "holdout_spearman": 0.07,
        "global_feature_importance": {
            f"feat_{i}": float(np.random.default_rng(i).uniform(0, 0.1))
            for i in range(30)
        },
        "regime_models": {
            "0": {"name": "Trending Bull", "feature_importance": {f"feat_{i}": float(np.random.default_rng(i + 100).uniform(0, 0.1)) for i in range(20)}},
            "1": {"name": "Trending Bear", "feature_importance": {f"feat_{i}": float(np.random.default_rng(i + 200).uniform(0, 0.1)) for i in range(20)}},
            "2": {"name": "Mean Reverting", "feature_importance": {f"feat_{i}": float(np.random.default_rng(i + 300).uniform(0, 0.1)) for i in range(20)}},
            "3": {"name": "High Volatility", "feature_importance": {f"feat_{i}": float(np.random.default_rng(i + 400).uniform(0, 0.1)) for i in range(20)}},
        },
    }
    meta_path = tmp_model_dir / "model_20240115_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return tmp_model_dir


@pytest.fixture
def tmp_results_dir(tmp_path):
    """Temporary results directory."""
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Temporary model directory."""
    d = tmp_path / "trained_models"
    d.mkdir()
    return d


@pytest.fixture
def tmp_data_cache_dir(tmp_path):
    """Temporary data cache directory."""
    d = tmp_path / "data_cache"
    d.mkdir()
    return d


# ── API fixtures ─────────────────────────────────────────────────────


@pytest.fixture
async def app(tmp_path):
    """Create a test FastAPI app with a fresh per-test job store."""
    import api.deps.providers as _prov
    from api.config import ApiSettings
    from api.jobs.store import JobStore
    from api.jobs.runner import JobRunner
    from api.main import create_app

    db_path = str(tmp_path / "test_jobs.db")
    settings = ApiSettings(job_db_path=db_path)

    # Create fresh singletons for this test
    store = JobStore(db_path)
    await store.initialize()
    runner = JobRunner(store)

    # Inject into the provider module
    _prov._job_store = store
    _prov._job_runner = runner

    application = create_app(settings)
    yield application

    # Cleanup
    await store.close()
    _prov._job_store = None
    _prov._job_runner = None


@pytest.fixture
async def client(app):
    """Async HTTP client bound to the test app."""
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac
