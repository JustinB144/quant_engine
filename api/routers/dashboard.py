"""Dashboard endpoints — KPIs, regime overview, time series analytics."""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends

from ..cache.manager import CacheManager
from ..deps.providers import get_cache
from ..schemas.envelope import ApiResponse
from ..services.backtest_service import BacktestService
from ..services.regime_service import RegimeService

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary")
async def dashboard_summary(cache: CacheManager = Depends(get_cache)) -> ApiResponse:
    cached = cache.get("dashboard:summary")
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    svc = BacktestService()
    data = await asyncio.to_thread(svc.get_latest_results, 10)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set("dashboard:summary", data)
    # Extract transparency meta from enriched data
    meta_fields: dict = {}
    if "model_version" in data:
        meta_fields["model_version"] = data["model_version"]
    if "sizing_method" in data:
        meta_fields["sizing_method"] = data["sizing_method"]
    if "walk_forward_mode" in data:
        meta_fields["walk_forward_mode"] = data["walk_forward_mode"]
    return ApiResponse.success(data, elapsed_ms=elapsed, **meta_fields)


@router.get("/regime")
async def dashboard_regime(cache: CacheManager = Depends(get_cache)) -> ApiResponse:
    cached = cache.get("regime:current")
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    svc = RegimeService()
    data = await asyncio.to_thread(svc.detect_current_regime)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set("regime:current", data)
    return ApiResponse.success(data, elapsed_ms=elapsed)


def _compute_returns_distribution() -> dict:
    """Build return histogram data with VaR/CVaR lines (sync)."""
    from quant_engine.config import RESULTS_DIR
    from api.services.data_helpers import (
        build_portfolio_returns,
        compute_returns_distribution,
        load_trades,
    )

    trades_path = RESULTS_DIR / "backtest_10d_trades.csv"
    trades = load_trades(trades_path)
    returns = build_portfolio_returns(trades)
    return compute_returns_distribution(returns)


def _compute_rolling_risk() -> dict:
    """Build rolling vol, Sharpe, drawdown time series (sync)."""
    from quant_engine.config import RESULTS_DIR
    from api.services.data_helpers import (
        build_portfolio_returns,
        compute_rolling_risk,
        load_trades,
    )

    trades_path = RESULTS_DIR / "backtest_10d_trades.csv"
    trades = load_trades(trades_path)
    returns = build_portfolio_returns(trades)
    return compute_rolling_risk(returns)


def _compute_equity_with_benchmark() -> dict:
    """Build equity curve with benchmark overlay (sync)."""
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


def _compute_attribution() -> dict:
    """Build factor attribution analysis (sync)."""
    from pathlib import Path

    from quant_engine.config import DATA_CACHE_DIR, RESULTS_DIR
    from api.services.data_helpers import (
        build_portfolio_returns,
        compute_attribution,
        load_trades,
    )

    trades_path = RESULTS_DIR / "backtest_10d_trades.csv"
    trades = load_trades(trades_path)
    strategy_returns = build_portfolio_returns(trades)
    return compute_attribution(strategy_returns, Path(DATA_CACHE_DIR))


@router.get("/returns-distribution")
async def returns_distribution(cache: CacheManager = Depends(get_cache)) -> ApiResponse:
    """Daily return histogram data with VaR/CVaR lines."""
    cached = cache.get("dashboard:returns-distribution")
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    data = await asyncio.to_thread(_compute_returns_distribution)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set("dashboard:returns-distribution", data)
    return ApiResponse.success(data, elapsed_ms=elapsed)


@router.get("/rolling-risk")
async def rolling_risk(cache: CacheManager = Depends(get_cache)) -> ApiResponse:
    """Rolling volatility, Sharpe, and drawdown time series."""
    cached = cache.get("dashboard:rolling-risk")
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    data = await asyncio.to_thread(_compute_rolling_risk)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set("dashboard:rolling-risk", data)
    return ApiResponse.success(data, elapsed_ms=elapsed)


@router.get("/equity")
async def equity_with_benchmark(cache: CacheManager = Depends(get_cache)) -> ApiResponse:
    """Equity curve with benchmark overlay time series."""
    cached = cache.get("dashboard:equity-benchmark")
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    data = await asyncio.to_thread(_compute_equity_with_benchmark)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set("dashboard:equity-benchmark", data)
    return ApiResponse.success(data, elapsed_ms=elapsed)


@router.get("/attribution")
async def attribution(cache: CacheManager = Depends(get_cache)) -> ApiResponse:
    """Factor attribution: tech-minus-def and momentum-spread decomposition."""
    cached = cache.get("dashboard:attribution")
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    data = await asyncio.to_thread(_compute_attribution)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set("dashboard:attribution", data)
    return ApiResponse.success(data, elapsed_ms=elapsed)
