"""Benchmark comparison endpoints."""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends

from ..cache.manager import CacheManager
from ..deps.providers import get_cache
from ..schemas.envelope import ApiResponse

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


def _compute_comparison() -> dict:
    """Build benchmark vs. strategy comparison (sync)."""
    from pathlib import Path

    import pandas as pd

    from quant_engine.config import DATA_CACHE_DIR, RESULTS_DIR
    from api.services.data_helpers import (
        build_portfolio_returns,
        compute_risk_metrics,
        load_benchmark_returns,
        load_trades,
    )

    trades_path = RESULTS_DIR / "backtest_10d_trades.csv"
    trades = load_trades(trades_path)
    strategy_returns = build_portfolio_returns(trades)
    ref_index = strategy_returns.index if len(strategy_returns) else pd.DatetimeIndex([])
    benchmark_returns = load_benchmark_returns(Path(DATA_CACHE_DIR), ref_index)

    strat_metrics = compute_risk_metrics(strategy_returns)
    bench_metrics = compute_risk_metrics(benchmark_returns)

    return {
        "strategy": strat_metrics,
        "benchmark": bench_metrics,
        "strategy_points": len(strategy_returns),
        "benchmark_points": len(benchmark_returns),
    }


def _compute_equity_curves() -> dict:
    """Build cumulative equity curves for strategy and benchmark (sync)."""
    from pathlib import Path

    import pandas as pd

    from quant_engine.config import DATA_CACHE_DIR, RESULTS_DIR
    from api.services.data_helpers import (
        build_equity_curves,
        build_portfolio_returns,
        load_benchmark_returns,
        load_trades,
    )

    trades_path = RESULTS_DIR / "backtest_10d_trades.csv"
    trades = load_trades(trades_path)
    strategy_returns = build_portfolio_returns(trades)
    ref_index = strategy_returns.index if len(strategy_returns) else pd.DatetimeIndex([])
    benchmark_returns = load_benchmark_returns(Path(DATA_CACHE_DIR), ref_index)

    return build_equity_curves(strategy_returns, benchmark_returns)


def _compute_rolling_metrics() -> dict:
    """Build rolling 60D correlation, alpha, beta, relative strength (sync)."""
    from pathlib import Path

    import pandas as pd

    from quant_engine.config import DATA_CACHE_DIR, RESULTS_DIR
    from api.services.data_helpers import (
        build_portfolio_returns,
        compute_rolling_metrics,
        load_benchmark_returns,
        load_trades,
    )

    trades_path = RESULTS_DIR / "backtest_10d_trades.csv"
    trades = load_trades(trades_path)
    strategy_returns = build_portfolio_returns(trades)
    ref_index = strategy_returns.index if len(strategy_returns) else pd.DatetimeIndex([])
    benchmark_returns = load_benchmark_returns(Path(DATA_CACHE_DIR), ref_index)

    return compute_rolling_metrics(strategy_returns, benchmark_returns)


@router.get("/comparison")
async def benchmark_comparison(cache: CacheManager = Depends(get_cache)) -> ApiResponse:
    cached = cache.get("benchmark:comparison")
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    data = await asyncio.to_thread(_compute_comparison)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set("benchmark:comparison", data)
    return ApiResponse.success(data, elapsed_ms=elapsed)


@router.get("/equity-curves")
async def benchmark_equity_curves(cache: CacheManager = Depends(get_cache)) -> ApiResponse:
    """Strategy + SPY cumulative return time series for equity comparison chart."""
    cached = cache.get("benchmark:equity-curves")
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    data = await asyncio.to_thread(_compute_equity_curves)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set("benchmark:equity-curves", data)
    return ApiResponse.success(data, elapsed_ms=elapsed)


@router.get("/rolling-metrics")
async def benchmark_rolling_metrics(cache: CacheManager = Depends(get_cache)) -> ApiResponse:
    """Rolling 60D correlation, alpha, beta, relative strength time series."""
    cached = cache.get("benchmark:rolling-metrics")
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    data = await asyncio.to_thread(_compute_rolling_metrics)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set("benchmark:rolling-metrics", data)
    return ApiResponse.success(data, elapsed_ms=elapsed)
