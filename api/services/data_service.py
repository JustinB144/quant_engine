"""Wraps data.loader for API consumption."""
from __future__ import annotations

import logging
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
