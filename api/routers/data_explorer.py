"""Data explorer endpoints — universe + per-ticker OHLCV + bars + indicators."""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, Depends, Query
from pydantic import BaseModel

from ..cache.manager import CacheManager
from ..deps.auth import require_auth
from ..deps.providers import get_cache
from ..errors import DataNotFoundError
from ..schemas.envelope import ApiResponse
from ..services.data_service import DataService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data", tags=["data"])


class BatchIndicatorRequest(BaseModel):
    """Request body for batch indicator computation."""

    timeframe: str = "1d"
    indicators: List[str] = ["rsi_14", "macd", "bollinger_20"]

# ── Timeframe mapping ──────────────────────────────────────────────────

_TIMEFRAME_PATTERNS = {
    "5min": ["5min", "5m"],
    "15min": ["15min", "15m"],
    "30min": ["30min", "30m"],
    "1hr": ["1hr", "1h", "60min"],
    "1d": ["1d", "daily"],
}


def _find_cached_parquet(cache_dir: Path, ticker: str, timeframe: str) -> Optional[Path]:
    """Find a cached parquet matching the ticker and timeframe."""
    patterns = _TIMEFRAME_PATTERNS.get(timeframe, [timeframe])
    ticker_upper = ticker.upper()

    for pat in patterns:
        # Exact match: TICKER_TIMEFRAME.parquet
        exact = cache_dir / f"{ticker_upper}_{pat}.parquet"
        if exact.exists():
            return exact
        # PERMNO-style: try common PERMNO patterns
        for p in cache_dir.glob(f"*_{pat}.parquet"):
            # Read meta to check ticker
            meta = p.with_suffix("").with_suffix(f".{pat}.meta.json")
            if not meta.exists():
                meta = p.parent / f"{p.stem}.meta.json"
            # Simple name match
            if p.stem.split("_")[0].upper() == ticker_upper:
                return p
        # Range-based pattern: TICKER_TIMEFRAME_START_END.parquet
        matches = list(cache_dir.glob(f"{ticker_upper}_{pat}_*.parquet"))
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)
        # Also try: TICKER_daily_START_END.parquet for 1d
        if pat == "daily":
            matches = list(cache_dir.glob(f"{ticker_upper}_daily_*.parquet"))
            if matches:
                return max(matches, key=lambda p: p.stat().st_mtime)

    return None


def _available_timeframes(cache_dir: Path, ticker: str) -> List[str]:
    """Discover which timeframes have cached data for a ticker."""
    available = []
    for tf in _TIMEFRAME_PATTERNS:
        if _find_cached_parquet(cache_dir, ticker, tf) is not None:
            available.append(tf)
    return available


def _load_bars(cache_dir: Path, ticker: str, timeframe: str, max_bars: int) -> Dict[str, Any]:
    """Load OHLCV bars from cached parquet."""
    path = _find_cached_parquet(cache_dir, ticker.upper(), timeframe)
    if path is None:
        return {"found": False}

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Ensure required columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            # Try lowercase
            lc = col.lower()
            if lc in df.columns:
                df[col] = df[lc]
            else:
                df[col] = 0.0

    df = df.tail(max_bars)

    bars = []
    for ts, row in df.iterrows():
        bars.append({
            "time": ts.isoformat(),
            "open": round(float(row.get("Open", 0)), 4),
            "high": round(float(row.get("High", 0)), 4),
            "low": round(float(row.get("Low", 0)), 4),
            "close": round(float(row.get("Close", 0)), 4),
            "volume": round(float(row.get("Volume", 0)), 0),
        })

    tf_available = _available_timeframes(cache_dir, ticker)

    return {
        "found": True,
        "ticker": ticker.upper(),
        "timeframe": timeframe,
        "bars": bars,
        "total_bars": len(bars),
        "available_timeframes": tf_available,
    }


# ── Indicator computation ──────────────────────────────────────────────

# Indicator registry: name -> (type, compute_fn)
# type is "overlay" (drawn on price) or "panel" (separate pane)

def _compute_sma(df: pd.DataFrame, period: int) -> Dict[str, Any]:
    values = df["Close"].rolling(period).mean()
    return {
        "type": "overlay",
        "values": [
            {"time": ts.isoformat(), "value": round(float(v), 4)}
            for ts, v in values.dropna().items()
        ],
    }


def _compute_ema(df: pd.DataFrame, period: int) -> Dict[str, Any]:
    values = df["Close"].ewm(span=period, adjust=False).mean()
    return {
        "type": "overlay",
        "values": [
            {"time": ts.isoformat(), "value": round(float(v), 4)}
            for ts, v in values.dropna().items()
        ],
    }


def _compute_bollinger(df: pd.DataFrame, period: int) -> Dict[str, Any]:
    middle = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std()
    upper = middle + 2 * std
    lower = middle - 2 * std
    valid = middle.dropna().index
    return {
        "type": "overlay",
        "upper": [{"time": ts.isoformat(), "value": round(float(upper[ts]), 4)} for ts in valid],
        "middle": [{"time": ts.isoformat(), "value": round(float(middle[ts]), 4)} for ts in valid],
        "lower": [{"time": ts.isoformat(), "value": round(float(lower[ts]), 4)} for ts in valid],
    }


def _compute_rsi(df: pd.DataFrame, period: int) -> Dict[str, Any]:
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return {
        "type": "panel",
        "values": [
            {"time": ts.isoformat(), "value": round(float(v), 2)}
            for ts, v in rsi.dropna().items()
        ],
        "thresholds": {"overbought": 70, "oversold": 30},
    }


def _compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, Any]:
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    valid = macd_line.dropna().index
    return {
        "type": "panel",
        "macd": [{"time": ts.isoformat(), "value": round(float(macd_line[ts]), 4)} for ts in valid],
        "signal": [{"time": ts.isoformat(), "value": round(float(signal_line[ts]), 4)} for ts in valid],
        "histogram": [{"time": ts.isoformat(), "value": round(float(histogram[ts]), 4)} for ts in valid],
    }


def _compute_atr(df: pd.DataFrame, period: int) -> Dict[str, Any]:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return {
        "type": "panel",
        "values": [
            {"time": ts.isoformat(), "value": round(float(v), 4)}
            for ts, v in atr.dropna().items()
        ],
    }


def _compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, Any]:
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    k = 100.0 * (df["Close"] - low_min) / (high_max - low_min).replace(0, 1e-10)
    d = k.rolling(d_period).mean()
    valid = k.dropna().index
    return {
        "type": "panel",
        "k": [{"time": ts.isoformat(), "value": round(float(k[ts]), 2)} for ts in valid],
        "d": [{"time": ts.isoformat(), "value": round(float(d.get(ts, 0)), 2)} for ts in valid if pd.notna(d.get(ts))],
        "thresholds": {"overbought": 80, "oversold": 20},
    }


def _compute_obv(df: pd.DataFrame) -> Dict[str, Any]:
    direction = np.where(df["Close"] > df["Close"].shift(1), 1, np.where(df["Close"] < df["Close"].shift(1), -1, 0))
    obv = (df["Volume"] * direction).cumsum()
    return {
        "type": "panel",
        "values": [
            {"time": ts.isoformat(), "value": round(float(v), 0)}
            for ts, v in obv.dropna().items()
        ],
    }


def _compute_vwap(df: pd.DataFrame) -> Dict[str, Any]:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_vol = df["Volume"].cumsum()
    cum_tp_vol = (typical * df["Volume"]).cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, 1e-10)
    return {
        "type": "overlay",
        "values": [
            {"time": ts.isoformat(), "value": round(float(v), 4)}
            for ts, v in vwap.dropna().items()
        ],
    }


def _compute_adx(df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100.0 * plus_dm.rolling(period).mean() / atr.replace(0, 1e-10)
    minus_di = 100.0 * minus_dm.rolling(period).mean() / atr.replace(0, 1e-10)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    adx = dx.rolling(period).mean()

    return {
        "type": "panel",
        "values": [
            {"time": ts.isoformat(), "value": round(float(v), 2)}
            for ts, v in adx.dropna().items()
        ],
        "thresholds": {"strong_trend": 25},
    }


# Indicator dispatch table
_INDICATOR_DISPATCH = {
    "sma": lambda df, p: _compute_sma(df, p),
    "ema": lambda df, p: _compute_ema(df, p),
    "bollinger": lambda df, p: _compute_bollinger(df, p),
    "rsi": lambda df, p: _compute_rsi(df, p),
    "macd": lambda df, _: _compute_macd(df),
    "atr": lambda df, p: _compute_atr(df, p),
    "stochastic": lambda df, _: _compute_stochastic(df),
    "obv": lambda df, _: _compute_obv(df),
    "vwap": lambda df, _: _compute_vwap(df),
    "adx": lambda df, p: _compute_adx(df, p),
}

_AVAILABLE_INDICATORS = sorted(_INDICATOR_DISPATCH.keys())


def _parse_indicator_spec(spec: str):
    """Parse 'rsi_14' into ('rsi', 14) or 'macd' into ('macd', 0)."""
    parts = spec.lower().split("_", 1)
    name = parts[0]
    period = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 14
    return name, period


def _compute_indicators_for_ticker(
    cache_dir: Path, ticker: str, timeframe: str, indicator_specs: List[str],
) -> Dict[str, Any]:
    """Load OHLCV and compute requested indicators."""
    path = _find_cached_parquet(cache_dir, ticker.upper(), timeframe)
    if path is None:
        return {"found": False}

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Ensure required columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            lc = col.lower()
            if lc in df.columns:
                df[col] = df[lc]
            else:
                df[col] = 0.0

    # Convert to float
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    results: Dict[str, Any] = {}
    for spec in indicator_specs:
        name, period = _parse_indicator_spec(spec)
        fn = _INDICATOR_DISPATCH.get(name)
        if fn is None:
            results[spec] = {"error": f"Unknown indicator: {name}"}
            continue
        try:
            results[spec] = fn(df, period)
        except Exception as e:
            logger.warning("Indicator %s failed for %s: %s", spec, ticker, e)
            results[spec] = {"error": str(e)}

    return {
        "found": True,
        "ticker": ticker.upper(),
        "timeframe": timeframe,
        "indicators": results,
        "available_indicators": _AVAILABLE_INDICATORS,
    }


# ── Existing endpoints ─────────────────────────────────────────────────


@router.get("/universe")
async def get_universe(cache: CacheManager = Depends(get_cache)) -> ApiResponse:
    cached = cache.get("data:universe")
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    svc = DataService()
    data = await asyncio.to_thread(svc.get_universe_info)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set("data:universe", data, ttl=300)
    return ApiResponse.success(data, elapsed_ms=elapsed)


@router.get("/status")
async def get_data_status(cache: CacheManager = Depends(get_cache)) -> ApiResponse:
    """Per-ticker cache health: source, freshness, bar counts, timeframes.

    Cached for 60 seconds to avoid repeated filesystem scans.
    """
    cached = cache.get("data:status")
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    svc = DataService()
    data = await asyncio.to_thread(svc.get_cache_status)
    elapsed = (time.monotonic() - t0) * 1000
    cache.set("data:status", data, ttl=60)
    return ApiResponse.success(data, elapsed_ms=elapsed)


@router.get("/ticker/{ticker}")
async def get_ticker(ticker: str, years: int = 5) -> ApiResponse:
    t0 = time.monotonic()
    svc = DataService()
    data = await asyncio.to_thread(svc.load_single_ticker, ticker.upper(), years)
    elapsed = (time.monotonic() - t0) * 1000
    if not data.get("found"):
        raise DataNotFoundError(f"Ticker {ticker.upper()} not found in data sources")
    return ApiResponse.success(data, elapsed_ms=elapsed)


# ── New: Bars with timeframe support ───────────────────────────────────


@router.get("/ticker/{ticker}/bars")
async def get_ticker_bars(
    ticker: str,
    timeframe: str = Query("1d", description="Timeframe: 5min, 15min, 30min, 1hr, 1d"),
    bars: int = Query(500, ge=1, le=5000, description="Number of bars to return"),
    cache: CacheManager = Depends(get_cache),
) -> ApiResponse:
    """Return OHLCV bars for any cached timeframe."""
    cache_key = f"bars:{ticker.upper()}:{timeframe}:{bars}"
    cached = cache.get(cache_key)
    if cached is not None:
        return ApiResponse.from_cached(cached)

    t0 = time.monotonic()

    from quant_engine.config import DATA_CACHE_DIR

    data = await asyncio.to_thread(
        _load_bars, Path(DATA_CACHE_DIR), ticker, timeframe, bars
    )
    elapsed = (time.monotonic() - t0) * 1000

    if not data.get("found"):
        raise DataNotFoundError(
            f"No {timeframe} data cached for {ticker.upper()}"
        )

    cache.set(cache_key, data, ttl=60)
    return ApiResponse.success(data, elapsed_ms=elapsed)


# ── New: Indicator computation ─────────────────────────────────────────


@router.get("/ticker/{ticker}/indicators")
async def get_ticker_indicators(
    ticker: str,
    timeframe: str = Query("1d", description="Timeframe for OHLCV data"),
    indicators: str = Query(
        "rsi_14", description="Comma-separated indicator specs (e.g. rsi_14,macd,bollinger_20)"
    ),
    cache: CacheManager = Depends(get_cache),
) -> ApiResponse:
    """Compute technical indicators on cached OHLCV data."""
    specs = [s.strip() for s in indicators.split(",") if s.strip()]
    cache_key = f"indicators:{ticker.upper()}:{timeframe}:{','.join(sorted(specs))}"
    cached = cache.get(cache_key)
    if cached is not None:
        return ApiResponse.from_cached(cached)

    t0 = time.monotonic()

    from quant_engine.config import DATA_CACHE_DIR

    data = await asyncio.to_thread(
        _compute_indicators_for_ticker,
        Path(DATA_CACHE_DIR), ticker, timeframe, specs,
    )
    elapsed = (time.monotonic() - t0) * 1000

    if not data.get("found"):
        raise DataNotFoundError(
            f"No {timeframe} data cached for {ticker.upper()}"
        )

    cache.set(cache_key, data, ttl=300)
    return ApiResponse.success(data, elapsed_ms=elapsed)


@router.post("/ticker/{ticker}/indicators/batch", dependencies=[Depends(require_auth)])
async def batch_indicators(
    ticker: str,
    body: BatchIndicatorRequest = Body(),
    cache: CacheManager = Depends(get_cache),
) -> ApiResponse:
    """Compute multiple indicators in one pass over the OHLCV data."""
    timeframe = body.timeframe
    indicators = body.indicators
    cache_key = f"indicators_batch:{ticker.upper()}:{timeframe}:{','.join(sorted(indicators))}"
    cached = cache.get(cache_key)
    if cached is not None:
        return ApiResponse.from_cached(cached)

    t0 = time.monotonic()

    from quant_engine.config import DATA_CACHE_DIR

    data = await asyncio.to_thread(
        _compute_indicators_for_ticker,
        Path(DATA_CACHE_DIR), ticker, timeframe, indicators,
    )
    elapsed = (time.monotonic() - t0) * 1000

    if not data.get("found"):
        raise DataNotFoundError(
            f"No {timeframe} data cached for {ticker.upper()}"
        )

    cache.set(cache_key, data, ttl=300)
    return ApiResponse.success(data, elapsed_ms=elapsed)
