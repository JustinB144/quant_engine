"""Wraps data.loader for API consumption."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DataService:
    """Thin wrapper around ``data.loader`` — all methods are synchronous."""

    def load_universe(
        self,
        tickers: Optional[List[str]] = None,
        years: int = 5,
        full: bool = False,
    ) -> Dict[str, Any]:
        """Load OHLCV universe and return summary dict."""
        from quant_engine.config import UNIVERSE_FULL, UNIVERSE_QUICK
        from quant_engine.data.loader import load_universe

        if tickers is None:
            tickers = UNIVERSE_FULL if full else UNIVERSE_QUICK
        data = load_universe(tickers, years=years, verbose=False)
        summaries = []
        for permno, df in data.items():
            ticker = str(df.attrs.get("ticker", ""))
            summaries.append({
                "permno": str(permno),
                "ticker": ticker,
                "bars": len(df),
                "start": str(df.index.min().date()) if len(df) else None,
                "end": str(df.index.max().date()) if len(df) else None,
            })
        return {"count": len(data), "tickers": summaries}

    def load_single_ticker(self, ticker: str, years: int = 5) -> Dict[str, Any]:
        """Load OHLCV for one ticker and return bars + metadata."""
        from quant_engine.data.loader import load_universe

        data = load_universe([ticker], years=years, verbose=False)
        if not data:
            return {"ticker": ticker, "found": False, "bars": []}
        permno, df = next(iter(data.items()))
        # Downsample if > 2500 points
        if len(df) > 2500:
            step = max(1, len(df) // 2500)
            df = df.iloc[::step]
        bars = []
        for ts, row in df.iterrows():
            bars.append({
                "date": str(ts.date()),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
            })
        return {
            "ticker": ticker,
            "permno": str(permno),
            "found": True,
            "bars": bars,
            "total_bars": len(data[permno]),
        }

    def get_cached_tickers(self) -> List[Dict[str, str]]:
        """List tickers present in the local cache."""
        from quant_engine.config import DATA_CACHE_DIR

        cache_dir = Path(DATA_CACHE_DIR)
        results = []
        if cache_dir.exists():
            for p in sorted(cache_dir.glob("*_1d.parquet")):
                name = p.stem.replace("_1d", "")
                results.append({"name": name, "path": str(p)})
        return results

    def get_universe_info(self) -> Dict[str, Any]:
        """Return configured universe definitions."""
        from quant_engine.config import UNIVERSE_FULL, UNIVERSE_QUICK, BENCHMARK

        return {
            "full_size": len(UNIVERSE_FULL),
            "quick_size": len(UNIVERSE_QUICK),
            "full_tickers": UNIVERSE_FULL,
            "quick_tickers": UNIVERSE_QUICK,
            "benchmark": BENCHMARK,
        }

    def get_cache_status(self) -> Dict[str, Any]:
        """Scan DATA_CACHE_DIR and return per-ticker cache health.

        For each ticker with daily data, reports: source, last bar date,
        total bars, available timeframes, freshness, and days stale.

        Returns
        -------
        dict
            ``{"tickers": [...], "summary": {...}}`` with per-ticker
            status entries and an aggregate summary.
        """
        from quant_engine.config import DATA_CACHE_DIR

        cache_dir = Path(DATA_CACHE_DIR)
        if not cache_dir.exists():
            return {
                "tickers": [],
                "summary": {
                    "total_cached": 0,
                    "fresh": 0,
                    "stale": 0,
                    "very_stale": 0,
                    "cache_dir": str(cache_dir),
                    "cache_exists": False,
                },
            }

        today = date.today()

        # Collect all available timeframes per ticker stem
        # Files are named like: AAPL_1d.parquet, AAPL_15min_2021-02-23_2026-02-20.parquet
        all_parquets = list(cache_dir.glob("*.parquet"))
        ticker_timeframes: Dict[str, List[str]] = {}
        for p in all_parquets:
            stem = p.stem
            # Extract ticker and timeframe from filename patterns
            # Pattern 1: TICKER_1d (daily, permno-based or ticker-based)
            # Pattern 2: TICKER_TIMEFRAME_START_END
            # Pattern 3: TICKER_daily_START_END
            parts = stem.split("_")
            if len(parts) >= 2:
                ticker_key = parts[0]
                timeframe = parts[1]
                # Normalize timeframe names
                if timeframe in ("1d", "daily"):
                    timeframe = "1d"
                ticker_timeframes.setdefault(ticker_key, set()).add(timeframe)

        # Scan daily meta.json files for primary info
        meta_files = sorted(cache_dir.glob("*_1d.meta.json"))
        tickers_status: List[Dict[str, Any]] = []
        counts = {"fresh": 0, "stale": 0, "very_stale": 0}

        for meta_path in meta_files:
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to read %s: %s", meta_path, exc)
                continue

            ticker = meta.get("ticker", meta_path.stem.replace("_1d.meta", ""))
            permno = meta.get("permno", "")
            source = meta.get("source", "unknown")
            end_date_str = meta.get("end_date", "")
            n_bars = meta.get("n_bars", 0)

            # Compute freshness
            days_stale = None
            freshness = "UNKNOWN"
            if end_date_str:
                try:
                    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d").date()
                    days_stale = (today - end_dt).days
                    if days_stale < 3:
                        freshness = "FRESH"
                        counts["fresh"] += 1
                    elif days_stale <= 14:
                        freshness = "STALE"
                        counts["stale"] += 1
                    else:
                        freshness = "VERY_STALE"
                        counts["very_stale"] += 1
                except ValueError:
                    counts["very_stale"] += 1

            # Gather available timeframes for this ticker
            ticker_key = meta_path.stem.replace("_1d.meta", "")
            timeframes = sorted(ticker_timeframes.get(ticker_key, {"1d"}))

            tickers_status.append({
                "ticker": ticker,
                "permno": str(permno),
                "source": source,
                "last_bar_date": end_date_str,
                "start_date": meta.get("start_date", ""),
                "total_bars": n_bars,
                "timeframes_available": timeframes,
                "freshness": freshness,
                "days_stale": days_stale,
            })

        return {
            "tickers": tickers_status,
            "summary": {
                "total_cached": len(tickers_status),
                "fresh": counts["fresh"],
                "stale": counts["stale"],
                "very_stale": counts["very_stale"],
                "cache_dir": str(cache_dir),
                "cache_exists": True,
            },
        }
