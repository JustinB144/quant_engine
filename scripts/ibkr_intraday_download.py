#!/usr/bin/env python3
"""
IBKR 1m-Only Intraday Pipeline: Download 1m → Quality Gate → Resample → Save.

Downloads 1-minute OHLCV data from Interactive Brokers, runs the 13-point
quality gate, then resamples to 5m/15m/30m/1h/4h.  IBKR is the sole data
source (ground truth) — no cross-source validation needed.

Saves in the canonical ``{TICKER}_{timeword}_{start}_{end}.parquet`` format
with ``.meta.json`` and ``.quality.json`` sidecars.

IBKR limits per-request lookback for 1m data to 1 calendar day, so this script
automatically breaks the target duration into day-sized chunks and concatenates.

Requires TWS or IB Gateway running on localhost:7497 (paper) or 7496 (live).

Usage:
    python3 -u scripts/ibkr_intraday_download.py                         # all tickers, incremental 1m → resample
    python3 -u scripts/ibkr_intraday_download.py --tickers AAPL MSFT     # specific tickers
    python3 -u scripts/ibkr_intraday_download.py --workers 3             # 3 parallel IBKR connections
    python3 -u scripts/ibkr_intraday_download.py --years 10              # 10 years of history
    python3 -u scripts/ibkr_intraday_download.py --no-incremental        # force full re-download
    python3 -u scripts/ibkr_intraday_download.py --resample-to 5m 1h     # only resample to specific TFs
    python3 -u scripts/ibkr_intraday_download.py --missing-only          # only download missing tickers
    python3 -u scripts/ibkr_intraday_download.py --dry-run               # survey only
"""

import argparse
import datetime as _dt
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Resolve paths ────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_QE_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_QE_ROOT))

# ib_insync / eventkit requires an event loop on import (Python 3.12+)
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Import config directly from project root (avoid nested quant_engine/ ambiguity)
import importlib.util as _ilu

_cfg_spec = _ilu.spec_from_file_location("_cfg", _QE_ROOT / "config.py")
_cfg = _ilu.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)
DATA_CACHE_DIR = _cfg.DATA_CACHE_DIR
UNIVERSE_FULL = _cfg.UNIVERSE_FULL
UNIVERSE_INTRADAY = getattr(_cfg, "UNIVERSE_INTRADAY", UNIVERSE_FULL)

_lc_spec = _ilu.spec_from_file_location("_lc", _QE_ROOT / "data" / "local_cache.py",
                                         submodule_search_locations=[])
_lc = _ilu.module_from_spec(_lc_spec)
_lc.DATA_CACHE_DIR = DATA_CACHE_DIR
_lc.FRAMEWORK_DIR = _cfg.FRAMEWORK_DIR
try:
    _lc_spec.loader.exec_module(_lc)
    _normalize_ohlcv_columns = _lc._normalize_ohlcv_columns
    _write_cache_meta = _lc._write_cache_meta
except Exception:
    def _normalize_ohlcv_columns(df):
        col_map = {"open": "Open", "high": "High", "low": "Low",
                    "close": "Close", "volume": "Volume", "adj close": "Adj Close"}
        df.columns = [col_map.get(c.lower().strip(), c) for c in df.columns]
        return df

    def _write_cache_meta(data_path, ticker, df, source="ibkr", meta=None):
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

# ── Quality gate import ──────────────────────────────────────────────────────
validate_intraday_bars = None
write_quality_report = None
quarantine_ticker = None
_QUALITY_AVAILABLE = False

try:
    _iq_path = _QE_ROOT / "data" / "intraday_quality.py"
    if _iq_path.exists():
        _iq_spec = _ilu.spec_from_file_location("intraday_quality", _iq_path)
        _iq_mod = _ilu.module_from_spec(_iq_spec)
        _iq_spec.loader.exec_module(_iq_mod)
        validate_intraday_bars = _iq_mod.validate_intraday_bars
        write_quality_report = _iq_mod.write_quality_report
        quarantine_ticker = _iq_mod.quarantine_ticker
        _QUALITY_AVAILABLE = True
except Exception as _qe:
    print(f"  WARNING: Quality gate not available: {_qe}")

logger = logging.getLogger(__name__)

REQUIRED_OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# ── IBKR configuration ──────────────────────────────────────────────────────
IBKR_CHUNK_DAYS_1M = 1           # Max 1 calendar day per 1m request (IBKR hard limit)
IBKR_BAR_SIZE_1M = "1 min"
IBKR_TIMEOUT = 120               # Seconds per request (1m historical can be slow)
MAX_CONSEC_FAIL = 5              # Skip ticker after N consecutive chunk failures
MAX_RETRIES_PER_CHUNK = 2        # Retry empty returns with backoff
DEFAULT_HISTORY_YEARS = 10       # 10 years of 1m history
DEFAULT_PACE = 1.0               # 1.0s between requests (safe: ~40 req/min vs 60 limit)
DEFAULT_WORKERS = 1              # Parallel IBKR connections (each gets own client ID)

RESAMPLE_TARGETS = ["5m", "15m", "30m", "1h", "4h"]

# Filename timeword mapping (matches local_cache.py conventions)
TF_SUFFIX = {
    "1m": "1min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "1hour", "4h": "4hour",
}


# ── Resample ─────────────────────────────────────────────────────────────────

def resample_1m_to_tf(
    df_1m: pd.DataFrame,
    target_tf: str,
) -> Optional[pd.DataFrame]:
    """
    Resample validated 1-minute bars to a higher timeframe.

    Bar alignment matches existing IBKR data conventions:
    - 5m/15m/30m: Standard resample, naturally aligns to market times
    - 1h: Resample at 1h, rename 09:00→09:30 (7 bars/day)
    - 4h: UTC-aligned 4h bins matching IBKR convention
           EST (Nov-Mar): 3 bars at 09:30, 11:00, 15:00
           EDT (Mar-Nov): 2 bars at 09:30, 12:00

    Args:
        df_1m: Validated 1-minute DataFrame with RTH data only.
               Index must be tz-naive Eastern Time DatetimeIndex.
        target_tf: Target timeframe ('5m', '15m', '30m', '1h', '4h')

    Returns:
        Resampled DataFrame, or None if input is empty.
    """
    if df_1m is None or df_1m.empty:
        return None

    ohlcv_agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }

    freq_map = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h"}

    if target_tf in freq_map:
        freq = freq_map[target_tf]
        resampled = df_1m.resample(freq).agg(ohlcv_agg).dropna(subset=["Open"])

        if target_tf == "1h":
            # Standard 1h resample bins start at :00, so the first RTH bin
            # is 09:00 (containing bars 09:30-09:59).  Rename to 09:30 to
            # match existing IBKR convention.
            new_index = []
            for ts in resampled.index:
                if ts.hour == 9 and ts.minute == 0:
                    new_index.append(ts.replace(minute=30))
                else:
                    new_index.append(ts)
            resampled.index = pd.DatetimeIndex(new_index)

        return resampled

    elif target_tf == "4h":
        # UTC-aligned 4h bins matching IBKR's 4-hour bar convention.
        all_bars = []
        rth_start = _dt.time(9, 30)

        for date, day_df in df_1m.groupby(df_1m.index.date):
            if day_df.empty:
                continue

            # Determine DST status for this date
            ts_aware = pd.Timestamp(date).tz_localize("America/New_York")
            is_dst = bool(ts_aware.dst())

            if is_dst:
                # EDT (UTC-4): 4h UTC bins hit 08:00 and 12:00 local
                bin_edges = [
                    (rth_start, _dt.time(12, 0)),
                    (_dt.time(12, 0), _dt.time(16, 0)),
                ]
                bin_labels = [_dt.time(9, 30), _dt.time(12, 0)]
            else:
                # EST (UTC-5): 4h UTC bins hit 07:00, 11:00, 15:00 local
                bin_edges = [
                    (rth_start, _dt.time(11, 0)),
                    (_dt.time(11, 0), _dt.time(15, 0)),
                    (_dt.time(15, 0), _dt.time(16, 0)),
                ]
                bin_labels = [_dt.time(9, 30), _dt.time(11, 0), _dt.time(15, 0)]

            for (start_t, end_t), label_t in zip(bin_edges, bin_labels):
                mask = (day_df.index.time >= start_t) & (day_df.index.time < end_t)
                bin_df = day_df[mask]
                if len(bin_df) > 0:
                    bar = pd.Series({
                        "Open": bin_df["Open"].iloc[0],
                        "High": bin_df["High"].max(),
                        "Low": bin_df["Low"].min(),
                        "Close": bin_df["Close"].iloc[-1],
                        "Volume": bin_df["Volume"].sum(),
                    })
                    ts = pd.Timestamp(_dt.datetime.combine(date, label_t))
                    all_bars.append((ts, bar))

        if not all_bars:
            return None

        result = pd.DataFrame(
            [bar for _, bar in all_bars],
            index=pd.DatetimeIndex([ts for ts, _ in all_bars]),
        )
        return result

    else:
        raise ValueError(f"Unsupported target timeframe for resampling: {target_tf}")


# ── Cache Survey ─────────────────────────────────────────────────────────────

def survey_cache(
    cache_dir: Path,
    tickers: List[str],
) -> Dict[str, Dict[str, Optional[dict]]]:
    """
    Survey intraday cache for all timeframes.

    Returns:
        {timeframe: {ticker: info_or_None}}
        info = {"path": Path, "start": str, "end": str, "end_date": date}
    """
    all_tfs = ["1m"] + RESAMPLE_TARGETS
    result: Dict[str, Dict[str, Optional[dict]]] = {}

    for tf in all_tfs:
        tf_word = TF_SUFFIX[tf]
        tf_data: Dict[str, Optional[dict]] = {}

        existing: Dict[str, dict] = {}
        for f in cache_dir.glob(f"*_{tf_word}_*.parquet"):
            parts = f.stem.split(f"_{tf_word}_")
            if len(parts) != 2:
                continue
            ticker = parts[0]
            dates = parts[1].split("_")
            if len(dates) == 2:
                try:
                    end_date = datetime.strptime(dates[1], "%Y-%m-%d").date()
                except ValueError:
                    end_date = None
                existing[ticker] = {
                    "path": f,
                    "start": dates[0],
                    "end": dates[1],
                    "end_date": end_date,
                }

        for t in tickers:
            tf_data[t] = existing.get(t)

        result[tf] = tf_data

    return result


# ── Quality Gate ─────────────────────────────────────────────────────────────

def run_quality_gate(
    df: pd.DataFrame,
    ticker: str,
    timeframe: str,
    source: str = "ibkr",
) -> Tuple[pd.DataFrame, dict]:
    """
    Run the 13-point quality gate on intraday data.

    Returns:
        (cleaned_df, qc_summary) where qc_summary has keys:
        pass, quality_score, quarantine, quarantine_reason,
        rejected_bars, flagged_bars, issues, report
    """
    if not _QUALITY_AVAILABLE:
        return df, {
            "pass": True, "quality_score": 1.0,
            "quarantine": False, "quarantine_reason": "",
            "rejected_bars": 0, "flagged_bars": 0,
            "issues": ["quality gate not available"], "report": None,
        }

    cleaned_df, report = validate_intraday_bars(df, ticker, timeframe, source)

    issues = []
    for check in report.checks:
        if not check.passed or check.flagged_count > 0:
            issues.append(f"{check.check_name}: {check.message}")

    return cleaned_df, {
        "pass": not report.quarantine,
        "quality_score": report.quality_score,
        "quarantine": report.quarantine,
        "quarantine_reason": report.quarantine_reason,
        "rejected_bars": report.total_rejected,
        "flagged_bars": report.total_flagged,
        "issues": issues,
        "report": report,
    }


# ── IBKR Chunked 1m Download ────────────────────────────────────────────────

def download_1m_chunked(
    ib,
    ticker: str,
    target_days: int,
    pace: float = 2.0,
) -> Optional[pd.DataFrame]:
    """
    Download 1-minute data for a single ticker using IBKR chunked requests.

    Handles:
    - Empty BarDataList (truthy but len==0) from Error 162
    - Retry with backoff on empty returns
    - Automatic reconnection on disconnect
    - tz normalization (strip timezone to tz-naive Eastern Time)

    Args:
        ib: Connected ib_insync.IB instance
        ticker: Stock symbol
        target_days: Total calendar days of history to fetch
        pace: Seconds between chunk requests (IBKR pacing)

    Returns:
        Combined DataFrame (tz-naive) or None
    """
    from ib_insync import Stock, util as ib_util

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
    total_chunks = 0
    total_failed = 0

    while remaining_days > 0:
        days_this = min(IBKR_CHUNK_DAYS_1M, remaining_days)
        success = False

        for attempt in range(1 + MAX_RETRIES_PER_CHUNK):
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=current_end,
                    durationStr=f"{days_this} D",
                    barSizeSetting=IBKR_BAR_SIZE_1M,
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                    timeout=IBKR_TIMEOUT,
                )

                # Fix: BarDataList is truthy even when empty (len==0) on Error 162
                if bars is not None and len(bars) > 0:
                    chunk_df = ib_util.df(bars)
                    chunk_df = chunk_df.rename(columns={
                        "date": "Date", "open": "Open", "high": "High",
                        "low": "Low", "close": "Close", "volume": "Volume",
                    })
                    if "Date" in chunk_df.columns:
                        chunk_df["Date"] = pd.to_datetime(chunk_df["Date"])
                        chunk_df = chunk_df.set_index("Date")

                    # Strip timezone if present (normalize to tz-naive Eastern)
                    if chunk_df.index.tz is not None:
                        chunk_df.index = chunk_df.index.tz_localize(None)

                    cols = [c for c in REQUIRED_OHLCV if c in chunk_df.columns]
                    chunk_df = chunk_df[cols]
                    all_chunks.append(chunk_df)
                    consec_failures = 0
                    success = True
                    break
                else:
                    # Empty return — retry with backoff
                    if attempt < MAX_RETRIES_PER_CHUNK:
                        backoff = 5 * (attempt + 1)
                        time.sleep(backoff)
                        continue

            except Exception:
                if attempt < MAX_RETRIES_PER_CHUNK:
                    time.sleep(5 * (attempt + 1))
                    continue

        total_chunks += 1
        if not success:
            total_failed += 1
            consec_failures += 1

        if consec_failures >= MAX_CONSEC_FAIL:
            print(f" ({MAX_CONSEC_FAIL} consecutive failures, stopping)")
            break

        current_end = current_end - timedelta(days=days_this + 1)
        remaining_days -= days_this
        time.sleep(pace)

    if not all_chunks:
        return None

    combined = pd.concat(all_chunks).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.dropna(subset=[c for c in REQUIRED_OHLCV if c in combined.columns])
    return combined if len(combined) > 0 else None


# ── Save ─────────────────────────────────────────────────────────────────────

def save_intraday(
    ticker: str,
    timeframe: str,
    df: pd.DataFrame,
    cache_dir: Path,
    existing_path: Optional[Path] = None,
    source: str = "ibkr",
) -> Path:
    """
    Save intraday data to cache. Merges with existing data if present.
    Handles tz-naive/tz-aware normalization before merge.
    """
    tf_word = TF_SUFFIX[timeframe]

    # Ensure df is tz-naive
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)

    # Merge with existing if available
    if existing_path is not None and existing_path.exists():
        try:
            old = pd.read_parquet(existing_path)
            old = _normalize_ohlcv_columns(old)
            if not isinstance(old.index, pd.DatetimeIndex):
                old.index = pd.DatetimeIndex(pd.to_datetime(old.index, errors="coerce"))
            # Normalize old data to tz-naive too
            if hasattr(old.index, 'tz') and old.index.tz is not None:
                old.index = old.index.tz_localize(None)
            cols = [c for c in REQUIRED_OHLCV if c in old.columns and c in df.columns]
            missing = set(REQUIRED_OHLCV) - set(cols)
            if missing:
                logger.warning(
                    "%s: Merge would drop required OHLCV columns %s — skipping merge, using new data only",
                    ticker, missing,
                )
                # Fallback: use new data only (which should have all columns)
                merged = df[REQUIRED_OHLCV] if all(c in df.columns for c in REQUIRED_OHLCV) else df
            else:
                merged = pd.concat([old[cols], df[cols]]).sort_index()
                merged = merged[~merged.index.duplicated(keep="last")]
                merged = merged.dropna(subset=cols)
            assert all(c in merged.columns for c in REQUIRED_OHLCV), \
                f"Post-merge OHLCV assertion failed: missing {set(REQUIRED_OHLCV) - set(merged.columns)}"
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
        source=source,
        meta={"timeframe": timeframe},
    )

    # Remove old file if name changed
    if existing_path is not None and existing_path.exists() and existing_path != out_path:
        try:
            existing_path.unlink()
            old_meta = existing_path.with_suffix(".meta.json")
            if old_meta.exists():
                old_meta.unlink()
            old_quality = existing_path.with_suffix(".quality.json")
            if old_quality.exists():
                old_quality.unlink()
        except OSError:
            pass

    return out_path


def write_quality_sidecar(
    out_path: Path,
    ticker: str,
    timeframe: str,
    source: str,
    df: pd.DataFrame,
    qc: dict,
) -> None:
    """Write .quality.json sidecar for a saved file."""
    span = (df.index.max() - df.index.min()).days / 365.25
    sidecar_data = {
        "ticker": ticker,
        "timeframe": timeframe,
        "source": source,
        "timestamp": datetime.now().isoformat(),
        "bars_saved": len(df),
        "date_range": {
            "start": str(df.index.min().date()),
            "end": str(df.index.max().date()),
            "span_years": round(span, 2),
        },
        "quality_check": {
            "passed": qc["pass"],
            "quality_score": qc["quality_score"],
            "quarantine": qc.get("quarantine", False),
            "quarantine_reason": qc.get("quarantine_reason", ""),
            "rejected_bars": qc.get("rejected_bars", 0),
            "flagged_bars": qc.get("flagged_bars", 0),
            "issues": qc.get("issues", []),
        },
    }
    quality_path = out_path.with_suffix(".quality.json")
    try:
        with open(quality_path, "w") as f:
            json.dump(sidecar_data, f, indent=2, default=str)
    except Exception:
        pass

    # Also write detailed report if available
    if _QUALITY_AVAILABLE and qc.get("report") is not None:
        try:
            write_quality_report(out_path.with_suffix(".report.json"), qc["report"])
        except Exception:
            pass


# ── Main Pipeline ────────────────────────────────────────────────────────────

def _compute_incremental_days(
    existing_info: Optional[dict],
    target_days: int,
) -> Tuple[int, Optional[str]]:
    """
    Compute how many days to download for incremental mode.

    Returns:
        (days_to_download, reason_str)
        reason_str is None for full download, or a description like "incremental from 2026-02-11"
    """
    if existing_info is None or existing_info.get("end_date") is None:
        return target_days, None

    end_date = existing_info["end_date"]
    today = datetime.now().date()
    gap_days = (today - end_date).days

    if gap_days <= 1:
        # Already up to date (today or yesterday)
        return 0, f"up-to-date (ends {end_date})"

    # Add a small overlap (3 days) to catch any bars we might have missed
    return min(gap_days + 3, target_days), f"incremental from {end_date} ({gap_days}d gap)"


# Thread-safe print lock for parallel workers
_print_lock = threading.Lock()

def _tprint(*args, **kwargs):
    """Thread-safe print."""
    with _print_lock:
        print(*args, **kwargs)


def process_ticker(
    ib,
    ticker: str,
    cache_dir: Path,
    target_days: int,
    resample_targets: List[str],
    survey: Dict[str, Dict[str, Optional[dict]]],
    pace: float,
    index: int,
    total: int,
    incremental: bool = True,
) -> dict:
    """
    Full pipeline for one ticker:
    1. Download 1m from IBKR (incremental if data exists)
    2. Run 13-point quality gate
    3. Resample to all target timeframes
    4. Save everything with sidecars

    Returns:
        {"success": bool, "bars": int, "resampled": list, "skipped": bool}
    """
    label = f"  [{index}/{total}]"
    existing_1m = survey.get("1m", {}).get(ticker)
    existing_path = existing_1m["path"] if existing_1m else None

    # Incremental: only download the gap if data already exists
    if incremental:
        dl_days, reason = _compute_incremental_days(existing_1m, target_days)
        if dl_days == 0:
            _tprint(f"{label} {ticker} — {reason}, skipping download → resample only")
            # Still resample from existing data
            try:
                saved_1m = pd.read_parquet(existing_path)
                if hasattr(saved_1m.index, 'tz') and saved_1m.index.tz is not None:
                    saved_1m.index = saved_1m.index.tz_localize(None)
            except Exception as e:
                _tprint(f"{label} {ticker} — cannot read existing: {e}")
                return {"success": False, "bars": 0, "resampled": [], "skipped": True}

            resampled_tfs = _resample_and_save(
                ticker, saved_1m, resample_targets, survey, cache_dir,
                qc_score=1.0, label=label,
            )
            return {"success": True, "bars": len(saved_1m), "resampled": resampled_tfs, "skipped": True}
    else:
        dl_days = target_days
        reason = None

    dl_label = f"({reason})" if reason else f"({dl_days}d)"
    _tprint(f"{label} {ticker} — downloading 1m {dl_label}...", end="", flush=True)

    # Step 1: Download 1m
    df = download_1m_chunked(ib, ticker, dl_days, pace=pace)

    if df is None or len(df) == 0:
        _tprint(f" NO DATA")
        return {"success": False, "bars": 0, "resampled": [], "skipped": False}

    _tprint(f" {len(df):,} raw bars", end="", flush=True)

    # Step 2: Quality gate on new data
    cleaned_df, qc = run_quality_gate(df, ticker, "1m", source="ibkr")

    if qc["quarantine"]:
        _tprint(f" QUARANTINED: {qc['quarantine_reason']}")
        if quarantine_ticker is not None:
            quarantine_ticker(ticker, "1m", cache_dir, qc["quarantine_reason"])
        return {"success": False, "bars": 0, "resampled": [], "skipped": False}

    _tprint(f" → {len(cleaned_df):,} clean (q={qc['quality_score']:.3f})", end="", flush=True)

    # Step 3: Save 1m (merges with existing automatically)
    try:
        out_path = save_intraday(
            ticker, "1m", cleaned_df, cache_dir,
            existing_path=existing_path,
            source="ibkr",
        )
        saved_1m = pd.read_parquet(out_path)
        span = (saved_1m.index.max() - saved_1m.index.min()).days / 365.25
        write_quality_sidecar(out_path, ticker, "1m", "ibkr", saved_1m, qc)
        _tprint(f" → saved {len(saved_1m):,} [{span:.1f}y]")
    except Exception as e:
        _tprint(f" SAVE ERROR: {e}")
        return {"success": False, "bars": 0, "resampled": [], "skipped": False}

    # Step 4: Resample to higher timeframes
    resampled_tfs = _resample_and_save(
        ticker, saved_1m, resample_targets, survey, cache_dir,
        qc_score=qc["quality_score"], label=label,
    )

    return {"success": True, "bars": len(saved_1m), "resampled": resampled_tfs, "skipped": False}


def _resample_and_save(
    ticker: str,
    saved_1m: pd.DataFrame,
    resample_targets: List[str],
    survey: Dict[str, Dict[str, Optional[dict]]],
    cache_dir: Path,
    qc_score: float,
    label: str,
) -> List[str]:
    """Resample 1m data to all target timeframes and save."""
    resampled_tfs = []
    for tf in resample_targets:
        try:
            resampled = resample_1m_to_tf(saved_1m, tf)
            if resampled is None or len(resampled) == 0:
                _tprint(f"{label}    {tf}: no data after resample")
                continue

            existing_tf = survey.get(tf, {}).get(ticker)
            existing_tf_path = existing_tf["path"] if existing_tf else None

            tf_out = save_intraday(
                ticker, tf, resampled, cache_dir,
                existing_path=existing_tf_path,
                source="ibkr:resampled_from_1m",
            )
            saved_tf = pd.read_parquet(tf_out)

            resample_qc = {
                "pass": True, "quality_score": qc_score,
                "quarantine": False, "quarantine_reason": "",
                "rejected_bars": 0, "flagged_bars": 0,
                "issues": [f"resampled from validated 1m (q={qc_score:.3f})"],
                "report": None,
            }
            write_quality_sidecar(tf_out, ticker, tf, "ibkr:resampled_from_1m", saved_tf, resample_qc)

            _tprint(f"{label}    {tf}: {len(saved_tf):,} bars")
            resampled_tfs.append(tf)
        except Exception as e:
            _tprint(f"{label}    {tf}: ERROR {e}")

    return resampled_tfs


# ── Main ─────────────────────────────────────────────────────────────────────

def _worker_loop(
    worker_id: int,
    work_queue: List[Tuple[int, str, Optional[dict]]],
    host: str,
    port: int,
    base_client_id: int,
    cache_dir: Path,
    target_days: int,
    resample_targets: List[str],
    survey: Dict[str, Dict[str, Optional[dict]]],
    pace: float,
    total: int,
    incremental: bool,
    results: List[dict],
) -> None:
    """Worker thread: connects to IBKR with its own client ID and processes tickers."""
    # Each thread needs its own event loop for ib_insync/eventkit
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    from ib_insync import IB

    client_id = base_client_id + worker_id
    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=30)
    except Exception as e:
        _tprint(f"  [Worker {worker_id}] FATAL: Cannot connect (clientId={client_id}): {e}")
        return

    _tprint(f"  [Worker {worker_id}] Connected (clientId={client_id})")

    for idx, ticker, existing_info in work_queue:
        # Reconnect if needed
        if not ib.isConnected():
            _tprint(f"  [Worker {worker_id}] Reconnecting...")
            try:
                ib.disconnect()
                time.sleep(2)
                ib.connect(host, port, clientId=client_id, timeout=30)
            except Exception as e:
                _tprint(f"  [Worker {worker_id}] Reconnect failed: {e} — stopping")
                break

        result = process_ticker(
            ib, ticker, cache_dir, target_days, resample_targets,
            survey, pace, idx, total, incremental=incremental,
        )
        results.append(result)

    try:
        ib.disconnect()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="IBKR 1m-only intraday pipeline: download 1m → quality gate → resample"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Override: only process these tickers (default: UNIVERSE_INTRADAY)",
    )
    parser.add_argument(
        "--years", type=float, default=DEFAULT_HISTORY_YEARS,
        help=f"Years of 1m history to download (default: {DEFAULT_HISTORY_YEARS})",
    )
    parser.add_argument(
        "--resample-to", nargs="+", default=None,
        help=f"Timeframes to resample to (default: {' '.join(RESAMPLE_TARGETS)})",
    )
    parser.add_argument(
        "--missing-only", action="store_true",
        help="Only download tickers with no existing 1m data",
    )
    parser.add_argument(
        "--no-incremental", action="store_true",
        help="Force full re-download even if data already exists",
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
        help="IBKR base client ID (default: 21). Workers use ID, ID+1, ID+2, ...",
    )
    parser.add_argument(
        "--pace", type=float, default=DEFAULT_PACE,
        help=f"Seconds between IBKR chunk requests (default: {DEFAULT_PACE}). "
             "IBKR allows ~60 requests per 10 min; 1.0s is safe for a single connection.",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Parallel IBKR connections (default: {DEFAULT_WORKERS}). "
             "Each worker uses a separate client ID. 3-4 workers is a good max.",
    )
    args = parser.parse_args()

    tickers = [t.upper() for t in (args.tickers or UNIVERSE_INTRADAY)]
    resample_targets = args.resample_to or RESAMPLE_TARGETS
    target_days = int(args.years * 365)
    incremental = not args.no_incremental

    # Validate resample targets
    for tf in resample_targets:
        if tf not in TF_SUFFIX:
            print(f"  ERROR: Unknown timeframe '{tf}'. Valid: {', '.join(RESAMPLE_TARGETS)}")
            sys.exit(1)

    print("=" * 70)
    print("  IBKR 1m PIPELINE — download → quality gate → resample")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    cache_dir = DATA_CACHE_DIR
    chunks_full = target_days // max(IBKR_CHUNK_DAYS_1M, 1)

    print(f"\n  Cache dir:     {cache_dir}")
    print(f"  Tickers:       {len(tickers)}")
    print(f"  History:       {args.years:.0f} years ({target_days} days)")
    print(f"  Incremental:   {'YES' if incremental else 'NO (full re-download)'}")
    print(f"  Resample to:   {', '.join(resample_targets)}")
    print(f"  Quality gate:  {'YES' if _QUALITY_AVAILABLE else 'NO (module not found)'}")
    print(f"  Pacing:        {args.pace:.1f}s between chunks")
    print(f"  Workers:       {args.workers}")

    # Survey
    survey = survey_cache(cache_dir, tickers)

    # Print survey summary
    print(f"\n  {'Timeframe':<10} {'Present':<10} {'Missing':<10}")
    print(f"  {'─'*10} {'─'*10} {'─'*10}")
    for tf in ["1m"] + resample_targets:
        tf_data = survey.get(tf, {})
        present = sum(1 for v in tf_data.values() if v is not None)
        missing = sum(1 for v in tf_data.values() if v is None)
        print(f"  {tf:<10} {present:<10} {missing:<10}")

    # Determine work list
    if args.missing_only:
        work = [(t, survey.get("1m", {}).get(t)) for t in tickers
                if survey.get("1m", {}).get(t) is None]
    else:
        work = [(t, survey.get("1m", {}).get(t)) for t in tickers]

    # Estimate time (accounting for incremental)
    total_chunks = 0
    for ticker, info in work:
        if incremental:
            dl_days, _ = _compute_incremental_days(info, target_days)
        else:
            dl_days = target_days
        total_chunks += dl_days // max(IBKR_CHUNK_DAYS_1M, 1)

    eff_workers = min(args.workers, len(work)) if work else 1
    est_seconds = total_chunks * (args.pace + 0.5) / eff_workers
    est_hours = est_seconds / 3600
    avg_chunks = total_chunks / len(work) if work else 0

    print(f"\n  Work:          {len(work)} tickers, ~{total_chunks:,} total chunks")
    print(f"  Avg chunks:    ~{avg_chunks:.0f}/ticker (incremental)" if incremental else f"  Chunks/ticker: ~{chunks_full}")
    print(f"  Estimated:     ~{est_hours:.1f} hours ({est_seconds / 60:.0f} min) with {eff_workers} worker(s)")

    if args.dry_run:
        print("\n  [DRY RUN] No downloads performed.")
        return

    if not work:
        print("\n  Nothing to download.")
        return

    # Single-worker mode (simple, no threading)
    if args.workers <= 1:
        from ib_insync import IB
        ib = IB()
        try:
            ib.connect(args.host, args.port, clientId=args.client_id, timeout=30)
        except Exception as e:
            print(f"\n  FATAL: Cannot connect to IBKR — {e}")
            print(f"  Make sure TWS or IB Gateway is running on {args.host}:{args.port}.")
            sys.exit(1)

        print(f"\n  IBKR connected: {ib.isConnected()}")
        print(f"  Processing {len(work)} tickers...\n")

        total_success = 0
        total_failed = 0
        total_bars = 0
        start_time = time.time()

        for i, (ticker, existing_info) in enumerate(work):
            if not ib.isConnected():
                print(f"  IBKR disconnected — reconnecting...")
                try:
                    ib.disconnect()
                    time.sleep(2)
                    ib.connect(args.host, args.port, clientId=args.client_id, timeout=30)
                    print(f"  Reconnected")
                except Exception as e:
                    print(f"  Reconnect failed: {e} — stopping")
                    break

            result = process_ticker(
                ib, ticker, cache_dir, target_days, resample_targets,
                survey, args.pace, i + 1, len(work), incremental=incremental,
            )

            if result["success"]:
                total_success += 1
                total_bars += result["bars"]
            else:
                total_failed += 1

        ib.disconnect()

    else:
        # Multi-worker mode: split work across threads, each with own IBKR connection
        print(f"\n  Launching {args.workers} parallel workers...\n")

        # Round-robin split work into per-worker queues
        worker_queues: List[List[Tuple[int, str, Optional[dict]]]] = [[] for _ in range(args.workers)]
        for i, (ticker, info) in enumerate(work):
            worker_queues[i % args.workers].append((i + 1, ticker, info))

        all_results: List[dict] = []
        start_time = time.time()
        threads = []

        for wid in range(args.workers):
            t = threading.Thread(
                target=_worker_loop,
                args=(
                    wid, worker_queues[wid],
                    args.host, args.port, args.client_id,
                    cache_dir, target_days, resample_targets,
                    survey, args.pace, len(work), incremental, all_results,
                ),
                daemon=True,
            )
            threads.append(t)
            t.start()
            time.sleep(1)  # Stagger connections

        for t in threads:
            t.join()

        total_success = sum(1 for r in all_results if r["success"])
        total_failed = sum(1 for r in all_results if not r["success"])
        total_bars = sum(r["bars"] for r in all_results)

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Success:    {total_success}")
    print(f"  Failed:     {total_failed}")
    print(f"  Total bars: {total_bars:,}")
    print(f"  Elapsed:    {elapsed / 60:.1f} min ({elapsed / 3600:.1f} hours)")
    print(f"  Cache:      {cache_dir}")


if __name__ == "__main__":
    main()
