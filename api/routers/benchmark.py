"""Benchmark comparison endpoint."""
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
