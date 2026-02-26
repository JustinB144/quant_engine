#!/usr/bin/env python3
"""
Hybrid Intraday Data Downloader: Alpaca (primary) + IBKR (validation/gap-fill).

Downloads intraday OHLCV data using Alpaca Markets free API as the primary
source (200 req/min, 10+ years of 1-minute history), then optionally validates
against IBKR data and fills gaps where Alpaca is missing bars.

Saves in the canonical ``{TICKER}_{timeword}_{start}_{end}.parquet`` format
with ``.meta.json`` sidecars, fully compatible with local_cache.py.

Setup:
    pip install alpaca-py --break-system-packages
    # Set env vars or use --api-key / --api-secret flags:
    export ALPACA_API_KEY="your_key"
    export ALPACA_API_SECRET="your_secret"
    # Free account (even unfunded) works for historical data.

Usage:
    python3 -u scripts/alpaca_intraday_download.py                          # all tf, all missing
    python3 -u scripts/alpaca_intraday_download.py --timeframes 1m 5m       # specific timeframes
    python3 -u scripts/alpaca_intraday_download.py --tickers AAPL MSFT      # specific tickers
    python3 -u scripts/alpaca_intraday_download.py --years 10               # 10 years of history
    python3 -u scripts/alpaca_intraday_download.py --validate-ibkr          # cross-check with IBKR
    python3 -u scripts/alpaca_intraday_download.py --dry-run                # survey only
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
sys.path.insert(0, str(_QE_ROOT))

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

    def _write_cache_meta(data_path, ticker, df, source="alpaca", meta=None):
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

# ── Import quality modules ───────────────────────────────────────────────────
# These provide the real 13-point quality gate and IBKR cross-validation.
# We use importlib to avoid issues with relative imports when running as a script.
_QUALITY_MODULES_AVAILABLE = False
validate_intraday_bars = None
write_quality_report = None
quarantine_ticker = None
CrossSourceValidator = None
CrossValidationReport = None

try:
    _iq_path = _QE_ROOT / "data" / "intraday_quality.py"
    _csv_path = _QE_ROOT / "data" / "cross_source_validator.py"

    if _iq_path.exists() and _csv_path.exists():
        _iq_spec = _ilu.spec_from_file_location("intraday_quality", _iq_path)
        _iq_mod = _ilu.module_from_spec(_iq_spec)
        _iq_spec.loader.exec_module(_iq_mod)
        validate_intraday_bars = _iq_mod.validate_intraday_bars
        write_quality_report = _iq_mod.write_quality_report
        quarantine_ticker = _iq_mod.quarantine_ticker

        _csv_spec = _ilu.spec_from_file_location("cross_source_validator", _csv_path)
        _csv_mod = _ilu.module_from_spec(_csv_spec)
        _csv_spec.loader.exec_module(_csv_mod)
        CrossSourceValidator = _csv_mod.CrossSourceValidator
        CrossValidationReport = _csv_mod.CrossValidationReport

        _QUALITY_MODULES_AVAILABLE = True
except Exception as _qe:
    _QUALITY_MODULES_AVAILABLE = False
    print(f"  WARNING: Quality modules not importable ({_qe}). "
          f"Falling back to basic checks.")

# ── Timeframe configuration ──────────────────────────────────────────────────
# Alpaca TimeFrame strings
ALPACA_TIMEFRAMES = {
    "1m":  ("1", "Min"),
    "5m":  ("5", "Min"),
    "15m": ("15", "Min"),
    "30m": ("30", "Min"),
    "1h":  ("1", "Hour"),
    "4h":  ("4", "Hour"),
}

# Filename timeword mapping (matches local_cache.py / ibkr downloader)
TF_SUFFIX = {
    "1m": "1min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "1hour", "4h": "4hour",
}

# Default years of history per timeframe
DEFAULT_HISTORY_YEARS = {
    "4h":  10,
    "1h":  10,
    "30m": 10,
    "15m": 10,
    "5m":  10,
    "1m":  10,
}

# Alpaca returns max ~10,000 bars per request. Chunk by calendar days to stay under.
ALPACA_CHUNK_DAYS = {
    "1m":    3,    # ~390 bars/day * 3 = ~1170
    "5m":   15,    # ~78 bars/day * 15 = ~1170
    "15m":  45,    # ~26 bars/day * 45 = ~1170
    "30m":  90,    # ~13 bars/day * 90 = ~1170
    "1h":  180,    # ~7 bars/day * 180 = ~1260
    "4h":  365,    # ~2 bars/day * 365 = ~730
}

MAX_CONSEC_FAIL = 5


# ── Cache Survey ─────────────────────────────────────────────────────────────

def survey_intraday(
    cache_dir: Path,
    tickers: List[str],
    timeframes: List[str],
) -> Dict[str, Dict[str, Optional[dict]]]:
    """Survey intraday cache for given tickers/timeframes."""
    result: Dict[str, Dict[str, Optional[dict]]] = {}

    for tf in timeframes:
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


# ── Alpaca Chunked Download ──────────────────────────────────────────────────

def _build_alpaca_client(api_key: str, api_secret: str):
    """Build Alpaca StockHistoricalDataClient."""
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(api_key, api_secret)


def download_alpaca_chunked(
    client,
    ticker: str,
    timeframe: str,
    target_days: int,
    pace: float = 0.35,
) -> Optional[pd.DataFrame]:
    """
    Download intraday data for a single ticker using Alpaca chunked requests.

    Args:
        client: Alpaca StockHistoricalDataClient instance
        ticker: Stock symbol
        timeframe: Canonical timeframe ("1m", "5m", "15m", etc.)
        target_days: Total calendar days of history to fetch
        pace: Seconds to sleep between requests (0.35s = ~170 req/min, under 200 limit)

    Returns:
        Combined DataFrame or None
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment, DataFeed

    tf_amount, tf_unit_str = ALPACA_TIMEFRAMES.get(timeframe, ("1", "Min"))
    unit_map = {"Min": TimeFrameUnit.Minute, "Hour": TimeFrameUnit.Hour}
    tf_obj = TimeFrame(int(tf_amount), unit_map[tf_unit_str])

    chunk_days = ALPACA_CHUNK_DAYS.get(timeframe, 10)

    all_chunks: List[pd.DataFrame] = []
    remaining_days = target_days
    current_end = datetime.now(timezone.utc)
    consec_failures = 0

    while remaining_days > 0:
        days_this = min(chunk_days, remaining_days)
        chunk_start = current_end - timedelta(days=days_this)

        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=[ticker],
                timeframe=tf_obj,
                start=chunk_start,
                end=current_end,
                adjustment=Adjustment.ALL,  # Split + dividend adjusted
                feed=DataFeed.SIP,          # Full consolidated feed
            )
            bars = client.get_stock_bars(request_params)

            if bars and bars.data and ticker in bars.data:
                bar_list = bars.data[ticker]
                if bar_list:
                    records = []
                    for b in bar_list:
                        records.append({
                            "Date": b.timestamp,
                            "Open": float(b.open),
                            "High": float(b.high),
                            "Low": float(b.low),
                            "Close": float(b.close),
                            "Volume": int(b.volume),
                        })
                    chunk_df = pd.DataFrame(records)
                    chunk_df["Date"] = pd.to_datetime(chunk_df["Date"], utc=True)
                    chunk_df = chunk_df.set_index("Date")
                    chunk_df.index = chunk_df.index.tz_convert("America/New_York").tz_localize(None)
                    all_chunks.append(chunk_df)
                    consec_failures = 0
                else:
                    consec_failures += 1
            else:
                consec_failures += 1

        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str or "limit" in err_str:
                # Rate limited — back off and retry
                print(f" [RATE LIMIT — sleeping 30s]", end="", flush=True)
                time.sleep(30)
                continue  # Retry same chunk
            consec_failures += 1

        if consec_failures >= MAX_CONSEC_FAIL:
            break

        current_end = chunk_start - timedelta(days=1)
        remaining_days -= days_this
        time.sleep(pace)

    if not all_chunks:
        return None

    combined = pd.concat(all_chunks).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.dropna(subset=[c for c in REQUIRED_OHLCV if c in combined.columns])

    # Filter to Regular Trading Hours only (09:30–16:00 ET)
    # Alpaca returns extended hours bars by default; we strip them here
    # to avoid the RTH quality check quarantining everything.
    if len(combined) > 0:
        t = combined.index.time
        import datetime as _dt
        rth_start = _dt.time(9, 30)
        rth_end = _dt.time(16, 0)
        rth_mask = (t >= rth_start) & (t < rth_end)
        pre_filter = len(combined)
        combined = combined[rth_mask]
        dropped = pre_filter - len(combined)
        if dropped > 0:
            print(f" [filtered {dropped} ext-hrs bars]", end="", flush=True)

    return combined if len(combined) > 0 else None


# ── IBKR Cross-Source Validation ─────────────────────────────────────────────

def validate_with_ibkr(
    ib,
    ticker: str,
    timeframe: str,
    primary_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """
    Cross-validate primary source data against IBKR ground truth.

    Uses the full CrossSourceValidator with stratified date sampling across
    the entire date range, 0.15% close tolerance, and automatic bar replacement.

    Returns:
        (corrected_df, result_dict)
    """
    if _QUALITY_MODULES_AVAILABLE:
        try:
            validator = CrossSourceValidator(
                ib=ib,
                close_tolerance_pct=getattr(_cfg, 'INTRADAY_CLOSE_TOLERANCE_PCT', 0.15),
                open_tolerance_pct=getattr(_cfg, 'INTRADAY_OPEN_TOLERANCE_PCT', 0.20),
                highlow_tolerance_pct=getattr(_cfg, 'INTRADAY_HIGHLOW_TOLERANCE_PCT', 0.25),
                volume_tolerance_pct=getattr(_cfg, 'INTRADAY_VOLUME_TOLERANCE_PCT', 5.0),
                sample_windows=getattr(_cfg, 'INTRADAY_VALIDATION_SAMPLE_WINDOWS', 10),
                days_per_window=getattr(_cfg, 'INTRADAY_VALIDATION_DAYS_PER_WINDOW', 2),
                max_mismatch_rate=getattr(_cfg, 'INTRADAY_MAX_MISMATCH_RATE_PCT', 5.0) / 100.0,
                ibkr_pace=2.0,
            )
            corrected_df, report = validator.validate_ticker(primary_df, ticker, timeframe)
            return corrected_df, {
                "valid": report.passed,
                "checked_bars": report.overlapping_bars,
                "price_mismatches": report.price_mismatches,
                "bars_replaced": report.bars_replaced,
                "bars_inserted": report.bars_inserted,
                "mismatch_rate": report.mismatch_rate,
                "missing_in_primary": report.missing_in_primary,
                "split_mismatches": report.split_mismatches,
                "report": report,
            }
        except Exception as e:
            print(f" [IBKR VALIDATION ERROR: {e}]", end="", flush=True)
            return primary_df, {
                "valid": False, "checked_bars": 0, "price_mismatches": 0,
                "bars_replaced": 0, "bars_inserted": 0, "mismatch_rate": 0.0,
                "missing_in_primary": 0, "split_mismatches": 0, "report": None,
                "error": str(e),
            }
    else:
        # Fallback: no validation possible without quality modules
        return primary_df, {
            "valid": False, "checked_bars": 0, "price_mismatches": 0,
            "bars_replaced": 0, "bars_inserted": 0, "mismatch_rate": 0.0,
            "missing_in_primary": 0, "split_mismatches": 0, "report": None,
            "error": "Quality modules not available",
        }


# ── Save ─────────────────────────────────────────────────────────────────────

def save_intraday(
    ticker: str,
    timeframe: str,
    df: pd.DataFrame,
    cache_dir: Path,
    existing_path: Optional[Path] = None,
    source: str = "alpaca",
) -> Path:
    """Save intraday data to cache, merging with existing if present."""
    tf_word = TF_SUFFIX[timeframe]

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
        except OSError:
            pass

    return out_path


# ── Quality Check ────────────────────────────────────────────────────────────

def quality_check(df: pd.DataFrame, ticker: str, timeframe: str = "1h", source: str = "alpaca"):
    """
    Run quality checks on downloaded data.

    Uses the full 13-point quality gate from data/intraday_quality.py when
    available, falling back to basic checks if the module can't be imported.

    Returns:
        (cleaned_df, report_dict) where:
        - cleaned_df: DataFrame with hard-rejected bars removed
        - report_dict: {"pass": bool, "issues": [...], "quality_score": float,
                        "quarantine": bool, "report": full_report_or_None}
    """
    if _QUALITY_MODULES_AVAILABLE:
        # Use the real 13-point quality gate
        cleaned_df, report = validate_intraday_bars(df, ticker, timeframe, source)
        issues = []
        for check in report.checks:
            if not check.passed:
                issues.append(f"{check.check_name}: {check.message}")
        return cleaned_df, {
            "pass": not report.quarantine and report.total_rejected == 0,
            "issues": issues,
            "quality_score": report.quality_score,
            "quarantine": report.quarantine,
            "quarantine_reason": report.quarantine_reason,
            "rejected_bars": report.total_rejected,
            "flagged_bars": report.total_flagged,
            "report": report,
        }
    else:
        # Fallback: basic checks (no bar removal)
        issues = []
        if "Volume" in df.columns:
            zero_vol_pct = (df["Volume"] == 0).sum() / len(df) * 100
            if zero_vol_pct > 25:
                issues.append(f"High zero-volume bars: {zero_vol_pct:.1f}%")
        if "Close" in df.columns and len(df) > 1:
            max_ret = df["Close"].pct_change().dropna().abs().max()
            if max_ret > 0.50:
                issues.append(f"Extreme return detected: {max_ret:.1%}")
        if df.index.duplicated().sum() > 0:
            issues.append(f"{df.index.duplicated().sum()} duplicate timestamps")
        if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
            bad_hl = (df["High"] < df["Low"]).sum()
            if bad_hl > 0:
                issues.append(f"{bad_hl} bars with High < Low")
        return df, {
            "pass": len(issues) == 0,
            "issues": issues,
            "quality_score": 1.0 if len(issues) == 0 else 0.5,
            "quarantine": False,
            "quarantine_reason": "",
            "rejected_bars": 0,
            "flagged_bars": 0,
            "report": None,
        }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid intraday downloader: Alpaca (primary) + IBKR (validation/gap-fill)"
    )
    parser.add_argument(
        "--timeframes", nargs="+", default=None,
        help="Timeframes to download (default: 1m 5m 15m 30m 1h 4h)",
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Override: only process these tickers (default: UNIVERSE_INTRADAY)",
    )
    parser.add_argument(
        "--years", type=float, default=10.0,
        help="Years of history to fetch (default: 10)",
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
        "--pace", type=float, default=0.35,
        help="Seconds between Alpaca requests (default: 0.35 = ~170 req/min, "
             "under 200/min free tier limit)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Alpaca API key (default: ALPACA_API_KEY env var)",
    )
    parser.add_argument(
        "--api-secret", default=None,
        help="Alpaca API secret (default: ALPACA_API_SECRET env var)",
    )
    parser.add_argument(
        "--validate-ibkr", action="store_true",
        help="Cross-validate Alpaca data against IBKR (requires TWS/Gateway)",
    )
    parser.add_argument(
        "--ibkr-host", default="127.0.0.1",
        help="IBKR host for validation/gap-fill (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--ibkr-port", type=int, default=7497,
        help="IBKR port for validation/gap-fill (default: 7497)",
    )
    parser.add_argument(
        "--ibkr-client-id", type=int, default=22,
        help="IBKR client ID for validation/gap-fill (default: 22, distinct from main downloader)",
    )
    args = parser.parse_args()

    # Resolve API credentials
    api_key = args.api_key or os.environ.get("ALPACA_API_KEY", "")
    api_secret = args.api_secret or os.environ.get("ALPACA_API_SECRET", "")

    all_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h"]
    timeframes = args.timeframes or all_timeframes
    tickers = [t.upper() for t in (args.tickers or UNIVERSE_INTRADAY)]

    # Allow --dry-run without API keys (survey only)
    if not api_key or not api_secret:
        if not args.dry_run:
            print("\n  ERROR: Alpaca API credentials required.")
            print("  Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables,")
            print("  or use --api-key and --api-secret flags.")
            print("  Sign up free at: https://app.alpaca.markets/signup")
            sys.exit(1)

    print("=" * 70)
    print("  HYBRID INTRADAY DOWNLOADER — Alpaca + IBKR")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    cache_dir = DATA_CACHE_DIR
    print(f"\n  Cache dir:    {cache_dir}")
    print(f"  Tickers:      {len(tickers)}")
    print(f"  Timeframes:   {', '.join(timeframes)}")
    print(f"  History:      {args.years:.0f} years")
    print(f"  Pacing:       {args.pace:.2f}s between requests ({60 / max(args.pace, 0.01):.0f} req/min)")
    print(f"  Primary:      Alpaca Markets (free tier)")
    if args.validate_ibkr:
        print(f"  Validation:   IBKR @ {args.ibkr_host}:{args.ibkr_port}")

    # Survey
    survey = survey_intraday(cache_dir, tickers, timeframes)

    print(f"\n  {'Timeframe':<10} {'Present':<10} {'Missing':<10} {'Chunks/ticker':<15} {'Est. time'}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*15} {'─'*15}")

    for tf in timeframes:
        tf_data = survey[tf]
        present = sum(1 for v in tf_data.values() if v is not None)
        missing = sum(1 for v in tf_data.values() if v is None)
        target_days = int(args.years * 365)
        chunk_days = ALPACA_CHUNK_DAYS.get(tf, 10)
        chunks_per = target_days // max(chunk_days, 1)
        tickers_to_dl = missing if args.missing_only else len(tickers)
        est_seconds = chunks_per * tickers_to_dl * (args.pace + 0.3)
        est_min = est_seconds / 60
        print(f"  {tf:<10} {present:<10} {missing:<10} {chunks_per:<15} ~{est_min:.0f} min ({tickers_to_dl} tickers)")

    if args.dry_run:
        print("\n  [DRY RUN] No downloads performed.")
        return

    # Build Alpaca client
    client = _build_alpaca_client(api_key, api_secret)
    print(f"\n  Alpaca client initialized")

    # Connect IBKR if needed
    ib = None
    if args.validate_ibkr:
        try:
            from ib_insync import IB
            ib = IB()
            ib.connect(args.ibkr_host, args.ibkr_port, clientId=args.ibkr_client_id, timeout=15)
            print(f"  IBKR connected: {ib.isConnected()}")
        except Exception as e:
            print(f"  IBKR connection failed: {e} — skipping validation/gap-fill")
            ib = None

    total_success = 0
    total_failed = 0
    total_quality_issues = 0
    total_ibkr_fills = 0
    total_ibkr_discrepancies = 0
    start_time = time.time()

    for tf in timeframes:
        tf_word = TF_SUFFIX[tf]
        target_days = int(args.years * 365)
        tf_survey = survey[tf]

        if args.missing_only:
            work = [(t, tf_survey[t]) for t in tickers if tf_survey[t] is None]
        else:
            work = [(t, tf_survey[t]) for t in tickers]

        if not work:
            print(f"\n  ── {tf} ── nothing to download")
            continue

        chunks_per = target_days // max(ALPACA_CHUNK_DAYS.get(tf, 10), 1)
        print(f"\n  ── {tf} ({len(work)} tickers, ~{chunks_per} chunks each) ──")

        tf_success = 0
        tf_failed = 0

        for i, (ticker, existing_info) in enumerate(work):
            existing_path = existing_info["path"] if existing_info else None
            label = f"  [{i+1}/{len(work)}]"

            print(f"{label} {ticker}...", end="", flush=True)

            # ── Step 1: Download from Alpaca ──
            df = download_alpaca_chunked(client, ticker, tf, target_days, pace=args.pace)

            if df is None or len(df) == 0:
                print(f" NO DATA")
                tf_failed += 1
                continue

            source = "alpaca"
            qc_report_dict = None
            val_report_dict = None

            # ── Step 2: Quality check (13-point gate) ──
            df, qc = quality_check(df, ticker, tf, source)
            qc_report_dict = qc
            if not qc["pass"]:
                total_quality_issues += 1
                issue_summary = ', '.join(qc['issues'][:3])
                if len(qc['issues']) > 3:
                    issue_summary += f" (+{len(qc['issues'])-3} more)"
                print(f" [QC {qc['quality_score']:.2f}: {issue_summary}]", end="", flush=True)

            # Check quarantine BEFORE saving
            if qc.get("quarantine"):
                print(f" [QUARANTINED: {qc.get('quarantine_reason', 'quality gate')}]", end="", flush=True)
                if _QUALITY_MODULES_AVAILABLE:
                    quarantine_dir = getattr(_cfg, 'INTRADAY_QUARANTINE_DIR',
                                            DATA_CACHE_DIR / "quarantine")
                    quarantine_ticker(ticker, tf, quarantine_dir,
                                     qc.get("quarantine_reason", "quality gate"))
                # Still save the data (it may be partially usable) but flag it
                source = "alpaca:quarantined"

            # ── Step 3: IBKR cross-validation (if enabled) ──
            if args.validate_ibkr and ib and ib.isConnected():
                df, val = validate_with_ibkr(ib, ticker, tf, df)
                val_report_dict = val
                if val["bars_replaced"] > 0 or val["bars_inserted"] > 0:
                    total_ibkr_fills += val["bars_replaced"] + val["bars_inserted"]
                    source = "alpaca+ibkr"
                if not val["valid"]:
                    total_ibkr_discrepancies += 1
                    print(f" [IBKR: {val['price_mismatches']} mismatches/{val['checked_bars']} checked, "
                          f"rate={val['mismatch_rate']:.1%}, "
                          f"replaced={val['bars_replaced']}, inserted={val['bars_inserted']}]",
                          end="", flush=True)
                elif val["checked_bars"] > 0:
                    print(f" [IBKR OK: {val['checked_bars']} bars, "
                          f"replaced={val['bars_replaced']}, inserted={val['bars_inserted']}]",
                          end="", flush=True)

            # ── Step 4: Save ──
            try:
                out_path = save_intraday(
                    ticker, tf, df, cache_dir,
                    existing_path=existing_path,
                    source=source,
                )
                saved = pd.read_parquet(out_path)
                span = (saved.index.max() - saved.index.min()).days / 365.25
                print(f" {len(saved):,} bars ({saved.index.min().date()} -> {saved.index.max().date()}) [{span:.1f}y]")
                tf_success += 1

                # ── Step 5: Write quality report JSON sidecar ──
                quality_sidecar = out_path.with_suffix(".quality.json")
                sidecar_data = {
                    "ticker": ticker,
                    "timeframe": tf,
                    "source": source,
                    "timestamp": datetime.now().isoformat(),
                    "bars_saved": len(saved),
                    "date_range": {
                        "start": str(saved.index.min().date()),
                        "end": str(saved.index.max().date()),
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
                if val_report_dict:
                    sidecar_data["ibkr_validation"] = {
                        "valid": val_report_dict["valid"],
                        "checked_bars": val_report_dict["checked_bars"],
                        "price_mismatches": val_report_dict["price_mismatches"],
                        "bars_replaced": val_report_dict["bars_replaced"],
                        "bars_inserted": val_report_dict["bars_inserted"],
                        "mismatch_rate": val_report_dict["mismatch_rate"],
                        "missing_in_primary": val_report_dict.get("missing_in_primary", 0),
                        "split_mismatches": val_report_dict.get("split_mismatches", 0),
                        "error": val_report_dict.get("error"),
                    }
                # Also write the full report using the quality module if available
                if _QUALITY_MODULES_AVAILABLE and qc.get("report") is not None:
                    try:
                        write_quality_report(out_path, qc["report"])
                    except Exception as qr_err:
                        print(f" [QR WRITE WARN: {qr_err}]", end="", flush=True)
                # Write lightweight sidecar regardless
                try:
                    with open(quality_sidecar, "w") as qf:
                        # Remove non-serializable report object before writing
                        sidecar_write = {k: v for k, v in sidecar_data.items()}
                        json.dump(sidecar_write, qf, indent=2, default=str)
                except Exception as qs_err:
                    print(f" [SIDECAR WARN: {qs_err}]", end="", flush=True)

            except Exception as e:
                print(f" SAVE ERROR: {e}")
                tf_failed += 1

        total_success += tf_success
        total_failed += tf_failed
        print(f"  {tf}: {tf_success} OK, {tf_failed} failed")

    # Disconnect IBKR if connected
    if ib and ib.isConnected():
        ib.disconnect()

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Success:          {total_success}")
    print(f"  Failed:           {total_failed}")
    print(f"  Quality issues:   {total_quality_issues}")
    if args.validate_ibkr:
        print(f"  IBKR corrections: {total_ibkr_fills} bars replaced/inserted")
        print(f"  IBKR mismatches:  {total_ibkr_discrepancies} tickers flagged")
    print(f"  Elapsed:          {elapsed / 60:.1f} minutes")
    print(f"  Cache:            {cache_dir}")


if __name__ == "__main__":
    main()
