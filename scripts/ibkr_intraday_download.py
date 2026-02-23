#!/usr/bin/env python3
"""
IBKR Intraday Data Downloader for quant_engine cache.

Downloads intraday OHLCV data (1m, 5m, 15m, 30m, 1h, 4h) from Interactive
Brokers using chunked historical requests.  Saves in the canonical
``{TICKER}_{timeword}_{start}_{end}.parquet`` format with ``.meta.json``
sidecars.

IBKR limits per-request lookback, so this script automatically breaks the
target duration into chunks and concatenates the results.

Requires TWS or IB Gateway running on localhost:7497 (paper) or 7496 (live).

Usage:
    python3 -u scripts/ibkr_intraday_download.py                         # all tf, all missing
    python3 -u scripts/ibkr_intraday_download.py --timeframes 1m 5m      # specific timeframes
    python3 -u scripts/ibkr_intraday_download.py --tickers AAPL MSFT     # specific tickers
    python3 -u scripts/ibkr_intraday_download.py --years 3               # 3 years of history
    python3 -u scripts/ibkr_intraday_download.py --dry-run               # survey only
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# ── Resolve paths ────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_QE_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_QE_ROOT))

# ib_insync / eventkit requires an event loop on import (Python 3.12+)
import asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Import config directly from project root (avoid nested quant_engine/ ambiguity)
import importlib.util as _ilu

_cfg_spec = _ilu.spec_from_file_location("_cfg", _QE_ROOT / "config.py")
_cfg = _ilu.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)
DATA_CACHE_DIR = _cfg.DATA_CACHE_DIR
UNIVERSE_FULL = _cfg.UNIVERSE_FULL

_lc_spec = _ilu.spec_from_file_location("_lc", _QE_ROOT / "data" / "local_cache.py",
                                         submodule_search_locations=[])
_lc = _ilu.module_from_spec(_lc_spec)
# Inject config vars that local_cache needs via its module globals
_lc.DATA_CACHE_DIR = DATA_CACHE_DIR
_lc.FRAMEWORK_DIR = _cfg.FRAMEWORK_DIR
try:
    _lc_spec.loader.exec_module(_lc)
    _normalize_ohlcv_columns = _lc._normalize_ohlcv_columns
    _write_cache_meta = _lc._write_cache_meta
except Exception:
    # Fallback: define minimal versions inline
    def _normalize_ohlcv_columns(df):
        col_map = {"open": "Open", "high": "High", "low": "Low",
                    "close": "Close", "volume": "Volume", "adj close": "Adj Close"}
        df.columns = [col_map.get(c.lower().strip(), c) for c in df.columns]
        return df

    def _write_cache_meta(data_path, ticker, df, source="ibkr", meta=None):
        import json
        meta_path = data_path.with_suffix(".meta.json")
        info = {
            "ticker": ticker, "source": source,
            "start": str(df.index.min().date()) if len(df) > 0 else "",
            "end": str(df.index.max().date()) if len(df) > 0 else "",
            "n_bars": len(df),
        }
        if meta:
            info.update(meta)
        with open(meta_path, "w") as f:
            json.dump(info, f, indent=2)

REQUIRED_OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# ── IBKR chunk configuration ────────────────────────────────────────────────
# Max calendar days per reqHistoricalData call for each bar size
IBKR_CHUNK_DAYS = {
    "1m":   1,
    "5m":   5,
    "15m":  10,
    "30m":  20,
    "1h":   29,
    "4h":  115,
}

# IBKR barSizeSetting strings
IBKR_BAR_SIZES = {
    "1m": "1 min", "5m": "5 mins", "15m": "15 mins",
    "30m": "30 mins", "1h": "1 hour", "4h": "4 hours",
}

# Filename timeword mapping (matches local_cache.py conventions)
TF_SUFFIX = {
    "1m": "1min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "1hour", "4h": "4hour",
}

# Default years of history per timeframe
DEFAULT_HISTORY_YEARS = {
    "4h":  5,
    "1h":  5,
    "30m": 5,
    "15m": 5,
    "5m":  5,
    "1m":  5,
}

MAX_CONSEC_FAIL = 5  # Skip ticker after N consecutive chunk failures


# ── Cache Survey ─────────────────────────────────────────────────────────────

def survey_intraday(
    cache_dir: Path,
    tickers: List[str],
    timeframes: List[str],
) -> Dict[str, Dict[str, Optional[dict]]]:
    """
    Survey intraday cache for given tickers/timeframes.

    Returns:
        {timeframe: {ticker: info_or_None}}
        info = {"path": Path, "start": str, "end": str, "bars": int}
        None means missing.
    """
    result: Dict[str, Dict[str, Optional[dict]]] = {}

    for tf in timeframes:
        tf_word = TF_SUFFIX[tf]
        tf_data: Dict[str, Optional[dict]] = {}

        # Index existing files
        existing: Dict[str, dict] = {}
        for f in cache_dir.glob(f"*_{tf_word}_*.parquet"):
            parts = f.stem.split(f"_{tf_word}_")
            if len(parts) != 2:
                continue
            ticker = parts[0]
            dates = parts[1].split("_")
            if len(dates) == 2:
                existing[ticker] = {
                    "path": f,
                    "start": dates[0],
                    "end": dates[1],
                    "bars": 0,
                }

        for t in tickers:
            tf_data[t] = existing.get(t)

        result[tf] = tf_data

    return result


# ── IBKR Chunked Download ───────────────────────────────────────────────────

def download_intraday_chunked(
    ib,
    ticker: str,
    timeframe: str,
    target_days: int,
) -> Optional[pd.DataFrame]:
    """
    Download intraday data for a single ticker using IBKR chunked requests.

    Args:
        ib: Connected ib_insync.IB instance
        ticker: Stock symbol
        timeframe: Canonical timeframe ("1m", "5m", "15m", etc.)
        target_days: Total calendar days of history to fetch

    Returns:
        Combined DataFrame or None
    """
    from ib_insync import Stock, util as ib_util

    bar_size = IBKR_BAR_SIZES.get(timeframe)
    chunk_days = IBKR_CHUNK_DAYS.get(timeframe, 10)
    if not bar_size:
        return None

    try:
        contract = Stock(ticker, "SMART", "USD")
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            print(f"      Could not qualify {ticker}")
            return None
    except Exception as e:
        print(f"      Contract error {ticker}: {e}")
        return None

    all_chunks: List[pd.DataFrame] = []
    remaining_days = target_days
    current_end = datetime.now()
    consec_failures = 0
    total_failed = 0

    while remaining_days > 0:
        days_this = min(chunk_days, remaining_days)

        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=current_end,
                durationStr=f"{days_this} D",
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
                timeout=15,
            )
            if bars:
                chunk_df = ib_util.df(bars)
                chunk_df = chunk_df.rename(columns={
                    "date": "Date", "open": "Open", "high": "High",
                    "low": "Low", "close": "Close", "volume": "Volume",
                })
                if "Date" in chunk_df.columns:
                    chunk_df.loc[:, "Date"] = pd.to_datetime(chunk_df["Date"])
                    chunk_df = chunk_df.set_index("Date")
                cols = [c for c in REQUIRED_OHLCV if c in chunk_df.columns]
                chunk_df = chunk_df[cols]
                all_chunks.append(chunk_df)
                consec_failures = 0
            else:
                total_failed += 1
                consec_failures += 1
        except Exception:
            total_failed += 1
            consec_failures += 1

        if consec_failures >= MAX_CONSEC_FAIL:
            break

        current_end = current_end - timedelta(days=days_this + 1)
        remaining_days -= days_this
        time.sleep(0.1)  # IBKR pacing

    if not all_chunks:
        return None

    combined = pd.concat(all_chunks).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.dropna(subset=[c for c in REQUIRED_OHLCV if c in combined.columns])
    return combined if len(combined) > 0 else None


def save_intraday(
    ticker: str,
    timeframe: str,
    df: pd.DataFrame,
    cache_dir: Path,
    existing_path: Optional[Path] = None,
) -> Path:
    """
    Save intraday data to cache. If existing data exists, merge (keep both,
    deduplicate by timestamp).
    """
    tf_word = TF_SUFFIX[timeframe]

    # Merge with existing if available
    if existing_path is not None and existing_path.exists():
        try:
            old = pd.read_parquet(existing_path)
            old = _normalize_ohlcv_columns(old)
            if not isinstance(old.index, pd.DatetimeIndex):
                old.index = pd.DatetimeIndex(pd.to_datetime(old.index, errors="coerce"))
            cols = [c for c in REQUIRED_OHLCV if c in old.columns and c in df.columns]
            merged = pd.concat([old[cols], df[cols]]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            merged = merged.dropna(subset=cols)
            df = merged
        except (OSError, ValueError):
            pass

    # Determine filename
    start_str = df.index.min().strftime("%Y-%m-%d")
    end_str = df.index.max().strftime("%Y-%m-%d")
    filename = f"{ticker.upper()}_{tf_word}_{start_str}_{end_str}.parquet"
    out_path = cache_dir / filename

    df.to_parquet(out_path)

    _write_cache_meta(
        data_path=out_path,
        ticker=ticker,
        df=df,
        source="ibkr",
        meta={"timeframe": timeframe},
    )

    # Remove old file if name changed
    if existing_path is not None and existing_path.exists() and existing_path != out_path:
        try:
            existing_path.unlink()
            old_meta = existing_path.with_suffix(".meta.json")
            if old_meta.exists():
                old_meta.unlink()
        except OSError:
            pass

    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IBKR intraday data downloader for quant_engine cache"
    )
    parser.add_argument(
        "--timeframes", nargs="+", default=None,
        help="Timeframes to download (default: 1m 5m 15m 30m 1h 4h)",
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Override: only process these tickers (default: UNIVERSE_FULL)",
    )
    parser.add_argument(
        "--years", type=float, default=5.0,
        help="Years of history to download (default: 5)",
    )
    parser.add_argument(
        "--missing-only", action="store_true",
        help="Only download tickers with no existing data for the timeframe",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Survey cache only, don't download",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="IBKR host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=7497,
        help="IBKR port (default: 7497 paper, 7496 live)",
    )
    parser.add_argument(
        "--client-id", type=int, default=21,
        help="IBKR client ID (default: 21)",
    )
    args = parser.parse_args()

    all_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h"]
    timeframes = args.timeframes or all_timeframes
    tickers = [t.upper() for t in (args.tickers or UNIVERSE_FULL)]

    print("=" * 70)
    print("  IBKR INTRADAY DOWNLOADER — quant_engine cache")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    cache_dir = DATA_CACHE_DIR
    print(f"\n  Cache dir:   {cache_dir}")
    print(f"  Tickers:     {len(tickers)}")
    print(f"  Timeframes:  {', '.join(timeframes)}")
    print(f"  History:     {args.years}y")

    # Survey
    survey = survey_intraday(cache_dir, tickers, timeframes)

    print(f"\n  {'Timeframe':<10} {'Present':<10} {'Missing':<10} {'Chunks/ticker':<15} {'Est. time'}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*15} {'─'*15}")

    for tf in timeframes:
        tf_data = survey[tf]
        present = sum(1 for v in tf_data.values() if v is not None)
        missing = sum(1 for v in tf_data.values() if v is None)
        target_days = int(args.years * 365)
        chunk_days = IBKR_CHUNK_DAYS.get(tf, 10)
        chunks_per = target_days // max(chunk_days, 1)
        tickers_to_dl = missing if args.missing_only else len(tickers)
        est_seconds = chunks_per * tickers_to_dl * 0.12
        est_min = est_seconds / 60
        print(f"  {tf:<10} {present:<10} {missing:<10} {chunks_per:<15} ~{est_min:.0f} min ({tickers_to_dl} tickers)")

    if args.dry_run:
        print("\n  [DRY RUN] No downloads performed.")
        return

    # Connect to IBKR
    from ib_insync import IB
    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=15)
    except Exception as e:
        print(f"\n  FATAL: Cannot connect to IBKR — {e}")
        print(f"  Make sure TWS or IB Gateway is running on {args.host}:{args.port}.")
        sys.exit(1)

    print(f"\n  IBKR connected: {ib.isConnected()}")

    total_success = 0
    total_failed = 0
    total_skipped = 0
    start_time = time.time()

    for tf in timeframes:
        tf_word = TF_SUFFIX[tf]
        target_days = int(args.years * 365)
        chunk_days = IBKR_CHUNK_DAYS.get(tf, 10)
        chunks_per = target_days // max(chunk_days, 1)
        tf_survey = survey[tf]

        # Determine which tickers to download
        if args.missing_only:
            work = [(t, tf_survey[t]) for t in tickers if tf_survey[t] is None]
        else:
            work = [(t, tf_survey[t]) for t in tickers]

        if not work:
            print(f"\n  ── {tf} ── nothing to download")
            continue

        print(f"\n  ── {tf} ({len(work)} tickers, ~{chunks_per} chunks each) ──")

        tf_success = 0
        tf_failed = 0

        for i, (ticker, existing_info) in enumerate(work):
            existing_path = existing_info["path"] if existing_info else None
            label = f"  [{i+1}/{len(work)}]"

            # Check if IBKR still connected
            if not ib.isConnected():
                print(f"{label} IBKR disconnected — reconnecting...")
                try:
                    ib.disconnect()
                    time.sleep(2)
                    ib.connect(args.host, args.port, clientId=args.client_id, timeout=15)
                    print(f"{label} Reconnected")
                except Exception as e:
                    print(f"{label} Reconnect failed: {e} — stopping {tf}")
                    break

            print(f"{label} {ticker}...", end="", flush=True)

            df = download_intraday_chunked(ib, ticker, tf, target_days)

            if df is None or len(df) == 0:
                print(f" NO DATA")
                tf_failed += 1
                continue

            try:
                out_path = save_intraday(
                    ticker, tf, df, cache_dir,
                    existing_path=existing_path,
                )
                saved = pd.read_parquet(out_path)
                span = (saved.index.max() - saved.index.min()).days / 365.25
                print(f" {len(saved):,} bars ({saved.index.min().date()} -> {saved.index.max().date()}) [{span:.1f}y]")
                tf_success += 1
            except Exception as e:
                print(f" SAVE ERROR: {e}")
                tf_failed += 1

        total_success += tf_success
        total_failed += tf_failed
        print(f"  {tf}: {tf_success} OK, {tf_failed} failed")

    # Disconnect
    ib.disconnect()
    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Success: {total_success}")
    print(f"  Failed:  {total_failed}")
    print(f"  Elapsed: {elapsed / 60:.1f} minutes")
    print(f"  Cache:   {cache_dir}")


if __name__ == "__main__":
    main()
