#!/usr/bin/env python3
"""
IBKR Daily Gap-Fill Downloader for quant_engine cache.

Downloads daily OHLCV data from Interactive Brokers to:
  1. Gap-fill 152 stale tickers (WRDS data ends 2024-12-31)
  2. Download full history for 28 missing UNIVERSE_FULL tickers

Merges with existing WRDS parquet data (WRDS takes precedence on overlapping
dates).  Saves in the canonical ``{TICKER}_daily_{start}_{end}.parquet``
format with ``.meta.json`` sidecars.

Requires TWS or IB Gateway running on localhost:7497 (paper) or 7496 (live).

Usage:
    python3 -u scripts/ibkr_daily_gapfill.py                # all stale + missing
    python3 -u scripts/ibkr_daily_gapfill.py --tickers AAPL MSFT
    python3 -u scripts/ibkr_daily_gapfill.py --missing-only  # only 28 missing
    python3 -u scripts/ibkr_daily_gapfill.py --stale-only    # only 152 stale
    python3 -u scripts/ibkr_daily_gapfill.py --dry-run       # survey only
"""

import argparse
import json
import os
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
_FRAMEWORK_DIR = _QE_ROOT.parent
sys.path.insert(0, str(_FRAMEWORK_DIR))
sys.path.insert(0, str(_QE_ROOT))

# ib_insync / eventkit requires an event loop on import (Python 3.12+)
import asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from quant_engine.config import UNIVERSE_FULL
from quant_engine.data.local_cache import (
    _normalize_ohlcv_columns,
    _write_cache_meta,
    load_ohlcv_with_meta,
)

# Use the root-level config for correct DATA_CACHE_DIR path
DATA_CACHE_DIR = _QE_ROOT / "data" / "cache"

REQUIRED_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


# ── Cache Survey ─────────────────────────────────────────────────────────────

def survey_cache(cache_dir: Path) -> Tuple[Dict[str, dict], List[str], List[str]]:
    """
    Survey the cache to find stale and missing tickers.

    Returns:
        (existing, stale_tickers, missing_tickers)
        existing: {ticker: {"path": Path, "start": str, "end": str, "bars": int}}
        stale_tickers: tickers with end_date <= 2024-12-31
        missing_tickers: UNIVERSE_FULL tickers not in cache at all
    """
    existing: Dict[str, dict] = {}

    for f in sorted(cache_dir.glob("*_daily_*.parquet")):
        stem = f.stem
        parts = stem.split("_daily_")
        if len(parts) != 2:
            continue
        ticker = parts[0]
        dates = parts[1].split("_")
        if len(dates) != 2:
            continue
        existing[ticker] = {
            "path": f,
            "start": dates[0],
            "end": dates[1],
            "bars": 0,
        }
        # Also check meta for bar count
        meta_path = f.with_suffix(".meta.json")
        if not meta_path.exists():
            meta_path = cache_dir / f"{ticker}_1d.meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r") as mf:
                    meta = json.load(mf)
                existing[ticker]["bars"] = meta.get("n_bars", 0)
            except (OSError, json.JSONDecodeError):
                pass

    # Also check _1d.parquet files
    for f in sorted(cache_dir.glob("*_1d.parquet")):
        ticker = f.stem.replace("_1d", "")
        if ticker[0].isdigit():
            continue  # Skip PERMNO-keyed files
        if ticker not in existing:
            existing[ticker] = {"path": f, "start": "", "end": "", "bars": 0}

    stale = [t for t, info in existing.items() if info["end"] <= "2024-12-31"]
    cached_tickers = set(existing.keys())
    missing = [t for t in UNIVERSE_FULL if t not in cached_tickers]

    return existing, stale, missing


# ── IBKR Download ────────────────────────────────────────────────────────────

def download_daily_ibkr(
    ib,
    ticker: str,
    duration: str = "2 Y",
) -> Optional[pd.DataFrame]:
    """
    Download daily OHLCV data for a single ticker from IBKR.

    Args:
        ib: Connected ib_insync.IB instance
        ticker: Stock ticker symbol
        duration: IBKR duration string (e.g. "2 Y", "20 Y")

    Returns:
        DataFrame with DatetimeIndex and OHLCV columns, or None
    """
    from ib_insync import Stock, util as ib_util

    try:
        contract = Stock(ticker, "SMART", "USD")
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            print(f"    Could not qualify contract for {ticker}")
            return None

        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        time.sleep(0.05)  # Respect IBKR pacing

        if not bars:
            return None

        df = ib_util.df(bars)
        df = df.rename(columns={
            "date": "Date", "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume",
        })
        if "Date" in df.columns:
            df.loc[:, "Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")

        # Ensure canonical OHLCV columns
        available = [c for c in REQUIRED_OHLCV if c in df.columns]
        if len(available) < 5:
            return None
        df = df[available].sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.dropna(subset=available)
        return df if len(df) > 0 else None

    except Exception as e:
        print(f"    IBKR error for {ticker}: {e}")
        return None


def merge_and_save(
    ticker: str,
    existing_path: Optional[Path],
    ibkr_df: pd.DataFrame,
    cache_dir: Path,
) -> Path:
    """
    Merge IBKR data with existing WRDS data and save.

    WRDS data takes precedence on overlapping dates.
    Saves as {TICKER}_daily_{start}_{end}.parquet with meta.json.

    Returns:
        Path to saved file
    """
    # Load existing data if available
    wrds_df = None
    if existing_path is not None and existing_path.exists():
        try:
            wrds_df = pd.read_parquet(existing_path)
            wrds_df = _normalize_ohlcv_columns(wrds_df)
            if not isinstance(wrds_df.index, pd.DatetimeIndex):
                wrds_df.index = pd.DatetimeIndex(
                    pd.to_datetime(wrds_df.index, errors="coerce")
                )
            wrds_df = wrds_df.sort_index()
        except (OSError, ValueError) as e:
            print(f"    Could not read existing file for {ticker}: {e}")
            wrds_df = None

    # Merge: WRDS takes precedence on overlapping dates
    if wrds_df is not None and len(wrds_df) > 0:
        # Keep only IBKR rows that are AFTER the last WRDS date
        wrds_last = wrds_df.index.max()
        ibkr_new = ibkr_df[ibkr_df.index > wrds_last]

        if len(ibkr_new) > 0:
            # Ensure same columns
            common_cols = [c for c in REQUIRED_OHLCV if c in wrds_df.columns and c in ibkr_new.columns]
            merged = pd.concat([wrds_df[common_cols], ibkr_new[common_cols]])
        else:
            merged = wrds_df[[c for c in REQUIRED_OHLCV if c in wrds_df.columns]]
    else:
        merged = ibkr_df[[c for c in REQUIRED_OHLCV if c in ibkr_df.columns]]

    merged = merged.sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.dropna(subset=[c for c in REQUIRED_OHLCV if c in merged.columns])

    if len(merged) == 0:
        raise ValueError(f"No data after merge for {ticker}")

    # Determine filename
    start_str = merged.index.min().strftime("%Y-%m-%d")
    end_str = merged.index.max().strftime("%Y-%m-%d")
    filename = f"{ticker.upper()}_daily_{start_str}_{end_str}.parquet"
    out_path = cache_dir / filename

    # Save parquet
    merged.to_parquet(out_path)

    # Write meta.json
    source = "wrds+ibkr" if wrds_df is not None and len(wrds_df) > 0 else "ibkr"
    _write_cache_meta(
        data_path=out_path,
        ticker=ticker,
        df=merged,
        source=source,
        meta={"timeframe": "daily"},
    )

    # Remove old file if we created a new one with different name
    if existing_path is not None and existing_path.exists() and existing_path != out_path:
        try:
            existing_path.unlink()
            # Also remove old meta.json
            old_meta = existing_path.with_suffix(".meta.json")
            if old_meta.exists():
                old_meta.unlink()
        except OSError:
            pass

    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IBKR daily gap-fill for quant_engine cache"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Override: only process these tickers",
    )
    parser.add_argument(
        "--stale-only", action="store_true",
        help="Only gap-fill stale tickers (skip missing)",
    )
    parser.add_argument(
        "--missing-only", action="store_true",
        help="Only download missing UNIVERSE_FULL tickers",
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
        "--client-id", type=int, default=20,
        help="IBKR client ID (default: 20)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  IBKR DAILY GAP-FILL — quant_engine cache")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    cache_dir = DATA_CACHE_DIR
    print(f"\n  Cache dir: {cache_dir}")
    print(f"  UNIVERSE_FULL: {len(UNIVERSE_FULL)} tickers")

    # Survey
    existing, stale_tickers, missing_tickers = survey_cache(cache_dir)
    print(f"\n  Existing daily files: {len(existing)}")
    print(f"  Stale (end <= 2024-12-31): {len(stale_tickers)}")
    print(f"  Missing from UNIVERSE_FULL: {len(missing_tickers)}")

    if missing_tickers:
        print(f"  Missing: {', '.join(missing_tickers)}")

    if args.dry_run:
        print("\n  [DRY RUN] No downloads performed.")
        return

    # Build work list
    work: List[Tuple[str, str, Optional[Path]]] = []

    if args.tickers:
        # Override: only specified tickers
        for t in args.tickers:
            t = t.upper()
            if t in existing:
                work.append((t, "stale", existing[t]["path"]))
            else:
                work.append((t, "missing", None))
    else:
        if not args.missing_only:
            for t in stale_tickers:
                work.append((t, "stale", existing[t]["path"]))
        if not args.stale_only:
            for t in missing_tickers:
                work.append((t, "missing", None))

    if not work:
        print("\n  Nothing to download.")
        return

    print(f"\n  Work items: {len(work)} tickers")

    # Connect to IBKR
    from ib_insync import IB
    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=15)
    except Exception as e:
        print(f"\n  FATAL: Cannot connect to IBKR — {e}")
        print(f"  Make sure TWS or IB Gateway is running on {args.host}:{args.port}.")
        sys.exit(1)

    print(f"  IBKR connected: {ib.isConnected()}\n")

    # Process
    success = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    for i, (ticker, category, existing_path) in enumerate(work):
        label = f"[{i+1}/{len(work)}]"

        # Determine duration
        if category == "stale":
            # Gap-fill: need ~14 months (2025-01-01 to present)
            duration = "2 Y"
        else:
            # Missing: get max history
            duration = "20 Y"

        print(f"  {label} {ticker} ({category}, {duration})...", end="", flush=True)

        ibkr_df = download_daily_ibkr(ib, ticker, duration=duration)

        if ibkr_df is None or len(ibkr_df) == 0:
            print(f" NO DATA")
            failed += 1
            continue

        try:
            out_path = merge_and_save(ticker, existing_path, ibkr_df, cache_dir)
            merged = pd.read_parquet(out_path)
            span = (merged.index.max() - merged.index.min()).days / 365.25
            print(
                f" {len(merged):,} bars "
                f"({merged.index.min().date()} -> {merged.index.max().date()}) "
                f"[{span:.1f}y]"
            )
            success += 1
        except Exception as e:
            print(f" ERROR: {e}")
            failed += 1

    # Disconnect
    ib.disconnect()
    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Elapsed: {elapsed / 60:.1f} minutes")
    print(f"  Cache:   {cache_dir}")


if __name__ == "__main__":
    main()
