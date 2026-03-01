#!/usr/bin/env python3
"""
NYSE TAQ Daily Product Intraday OHLCV Download

Downloads historical intraday OHLCV data from WRDS NYSE TAQ Daily Product
for the full UNIVERSE_INTRADAY (128 tickers) across all 6 configured
timeframes, from 2003-09-10 to present.

Core optimization: download raw tick data ONCE per trading day for ALL
tickers in a single SQL query, then resample to all 6 timeframes locally.
This reduces WRDS round-trips from ~730,000 to ~5,700.

Usage:
    python3 run_wrds_taq_intraday_download.py
    python3 run_wrds_taq_intraday_download.py --dry-run
    python3 run_wrds_taq_intraday_download.py --tickers AAPL,MSFT --start 2020-01-01
"""

import argparse
import gc
import json
import logging
import os
import signal
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Path setup ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from quant_engine.config import (
    DATA_CACHE_DIR,
    INTRADAY_CACHE_SOURCE,
    INTRADAY_MAX_REJECTED_BAR_PCT,
    INTRADAY_MIN_BARS,
    INTRADAY_TIMEFRAMES,
    MARKET_CLOSE,
    MARKET_OPEN,
    TAQ_START_DATE,
    UNIVERSE_INTRADAY,
)
from quant_engine.data.local_cache import (
    REQUIRED_OHLCV,
    _IBKR_TIMEFRAME_MAP,
    _write_cache_meta,
    load_intraday_ohlcv,
)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

# ── Constants ─────────────────────────────────────────────────────────────
CHECKPOINT_PATH = DATA_CACHE_DIR / "taq_download_checkpoint.yaml"
DEFAULT_BATCH_DAYS = 30
DEFAULT_DELAY = 0.05
RETRY_DELAYS = [1.0, 2.0, 4.0]  # Exponential backoff for WRDS failures

# Timeframe minutes for resampling from 1m bars
TIMEFRAME_MINUTES: Dict[str, int] = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240,
}


# ── Schema Discovery ─────────────────────────────────────────────────────

def _map_columns(columns: List[str]) -> Dict[str, str]:
    """Map discovered TAQ column names to canonical keys.

    TAQ columns vary by era. This function maps them to a standard
    set of keys: 'sym', 'time', 'price', 'size', 'corr'.

    Parameters
    ----------
    columns : list of str
        Column names from a TAQ table probe.

    Returns
    -------
    dict
        Mapping from canonical key to actual column name.
    """
    col_map: Dict[str, str] = {}

    # Symbol column
    for c in columns:
        cl = c.lower()
        if cl in ('sym_root', 'symbol', 'sym'):
            col_map['sym'] = c
            break
    if 'sym' not in col_map:
        raise RuntimeError(f"No symbol column found in {columns}")

    # Time column
    for c in columns:
        cl = c.lower()
        if cl in ('time_m', 'date_m', 'time', 'timestamp', 'datetime'):
            col_map['time'] = c
            break
    if 'time' not in col_map:
        raise RuntimeError(f"No time column found in {columns}")

    # Price column
    for c in columns:
        cl = c.lower()
        if cl in ('price', 'prc', 'trade_price'):
            col_map['price'] = c
            break
    if 'price' not in col_map:
        raise RuntimeError(f"No price column found in {columns}")

    # Size column
    for c in columns:
        cl = c.lower()
        if cl in ('size', 'vol', 'trade_size', 'volume'):
            col_map['size'] = c
            break
    if 'size' not in col_map:
        raise RuntimeError(f"No size column found in {columns}")

    # Trade correction column
    for c in columns:
        cl = c.lower()
        if cl in ('tr_corr', 'corr', 'correction', 'tr_corrn'):
            col_map['corr'] = c
            break
    if 'corr' not in col_map:
        raise RuntimeError(f"No correction column found in {columns}")

    return col_map


def discover_taq_schema(wrds_provider: "WRDSProvider") -> Tuple[str, str, Dict[str, str]]:
    """Probe WRDS to discover exact TAQ Daily Product table schema.

    TAQ Daily Product tables follow patterns like taqm_ct.ct_YYYYMMDD.
    This function probes multiple schemas on known trading days to find
    the accessible one.

    Parameters
    ----------
    wrds_provider : WRDSProvider
        Active WRDS connection.

    Returns
    -------
    tuple of (schema, prefix, col_map)
        Discovered schema name, table prefix, and column mapping.

    Raises
    ------
    RuntimeError
        If no accessible TAQ tables are found.
    """
    test_dates = ['20100104', '20050104', '20150105']
    candidates = [
        ('taqm_ct', 'ct'),
        ('taq', 'ct'),
        ('taqm', 'ct'),
    ]

    for schema, prefix in candidates:
        for test_date in test_dates:
            table = f'{schema}.{prefix}_{test_date}'
            result = wrds_provider._query_silent(
                f'SELECT * FROM {table} LIMIT 1'
            )
            if not result.empty:
                columns = list(result.columns)
                col_map = _map_columns(columns)
                logger.info(
                    "Discovered TAQ schema: %s.%s (columns: %s)",
                    schema, prefix, col_map,
                )
                return schema, prefix, col_map

    raise RuntimeError(
        "No accessible TAQ Daily Product tables found. "
        "Check WRDS subscription and credentials."
    )


# ── Resampling ────────────────────────────────────────────────────────────

def resample_from_1m(bars_1m: pd.DataFrame, target_minutes: int) -> pd.DataFrame:
    """Resample 1-minute OHLCV bars to a higher timeframe.

    Parameters
    ----------
    bars_1m : pd.DataFrame
        1-minute OHLCV bars with DatetimeIndex.
    target_minutes : int
        Target bar size in minutes (e.g. 5, 15, 30, 60, 240).

    Returns
    -------
    pd.DataFrame
        Resampled OHLCV bars.
    """
    rule = f'{target_minutes}min'
    ohlcv = pd.DataFrame()
    ohlcv['Open'] = bars_1m['Open'].resample(rule).first()
    ohlcv['High'] = bars_1m['High'].resample(rule).max()
    ohlcv['Low'] = bars_1m['Low'].resample(rule).min()
    ohlcv['Close'] = bars_1m['Close'].resample(rule).last()
    ohlcv['Volume'] = bars_1m['Volume'].resample(rule).sum()
    return ohlcv.dropna(subset=['Open', 'Close'])


def aggregate_to_1m(
    ticks: pd.DataFrame,
    price_col: str,
    size_col: str,
) -> pd.DataFrame:
    """Aggregate raw ticks to 1-minute OHLCV bars.

    Matches the existing get_taqmsec_ohlcv() aggregation pattern.

    Parameters
    ----------
    ticks : pd.DataFrame
        Raw tick data with DatetimeIndex.
    price_col : str
        Name of the price column.
    size_col : str
        Name of the size/volume column.

    Returns
    -------
    pd.DataFrame
        1-minute OHLCV bars filtered to regular trading hours.
    """
    ticks[price_col] = pd.to_numeric(ticks[price_col], errors='coerce')
    ticks[size_col] = pd.to_numeric(ticks[size_col], errors='coerce')
    ticks = ticks.dropna(subset=[price_col, size_col])

    if ticks.empty:
        return pd.DataFrame(columns=REQUIRED_OHLCV)

    ohlcv_1m = ticks[price_col].resample('1min').agg(['first', 'max', 'min', 'last'])
    ohlcv_1m.columns = ['Open', 'High', 'Low', 'Close']
    ohlcv_1m['Volume'] = ticks[size_col].resample('1min').sum()
    ohlcv_1m = ohlcv_1m.dropna(subset=['Open', 'Close'])
    ohlcv_1m = ohlcv_1m.between_time(MARKET_OPEN, MARKET_CLOSE)
    ohlcv_1m.index.name = 'Date'
    return ohlcv_1m


# ── Tick Parsing ──────────────────────────────────────────────────────────

def _parse_ticks_for_day(
    day_df: pd.DataFrame,
    date_str: str,
    time_col: str,
    price_col: str,
    size_col: str,
) -> pd.DataFrame:
    """Parse raw tick DataFrame into time-indexed format.

    Parameters
    ----------
    day_df : pd.DataFrame
        Raw tick data from WRDS query.
    date_str : str
        Date string in YYYY-MM-DD format.
    time_col : str
        Name of the time column.
    price_col : str
        Name of the price column.
    size_col : str
        Name of the size column.

    Returns
    -------
    pd.DataFrame
        Tick data with DatetimeIndex and 'price', 'size' columns.
    """
    if day_df.empty:
        return pd.DataFrame(columns=['price', 'size'])

    day_df = day_df.copy()
    day_df['datetime'] = pd.to_datetime(
        date_str + ' ' + day_df[time_col].astype(str),
        format='mixed',
        errors='coerce',
    )
    day_df = day_df.dropna(subset=['datetime'])
    day_df.index = pd.DatetimeIndex(day_df.pop('datetime'))
    day_df = day_df.rename(columns={price_col: 'price', size_col: 'size'})
    return day_df[['price', 'size']]


# ── Checkpoint ────────────────────────────────────────────────────────────

def load_checkpoint() -> Optional[Dict]:
    """Load download checkpoint from YAML file.

    Returns
    -------
    dict or None
        Checkpoint data, or None if no checkpoint exists.
    """
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        with open(CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
            if yaml is not None:
                return yaml.safe_load(f)
            else:
                return json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("Could not load checkpoint: %s", e)
        return None


def save_checkpoint(
    last_completed_date: str,
    tickers_in_scope: int,
    schema_used: str,
    started_at: str,
    days_processed: int,
) -> None:
    """Save download checkpoint atomically.

    Parameters
    ----------
    last_completed_date : str
        Last fully processed trading day (YYYY-MM-DD).
    tickers_in_scope : int
        Number of tickers being downloaded.
    schema_used : str
        TAQ schema identifier.
    started_at : str
        ISO timestamp when download started.
    days_processed : int
        Total trading days processed so far.
    """
    checkpoint = {
        'last_completed_date': last_completed_date,
        'tickers_in_scope': tickers_in_scope,
        'schema_used': schema_used,
        'started_at': started_at,
        'days_processed': days_processed,
    }

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(CHECKPOINT_PATH.parent), suffix='.tmp',
    )
    try:
        os.close(fd)
        with open(tmp, 'w', encoding='utf-8') as f:
            if yaml is not None:
                yaml.dump(checkpoint, f, default_flow_style=False)
            else:
                json.dump(checkpoint, f, indent=2)
        os.replace(tmp, str(CHECKPOINT_PATH))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ── File I/O ──────────────────────────────────────────────────────────────

def _parquet_path(
    ticker: str,
    timeframe: str,
    start_date: str,
    end_date: str,
) -> Path:
    """Build the parquet file path for a ticker/timeframe.

    Parameters
    ----------
    ticker : str
        Stock symbol.
    timeframe : str
        Canonical timeframe code (e.g. '5m').
    start_date : str
        Start date in YYYYMMDD format.
    end_date : str
        End date in YYYYMMDD format.

    Returns
    -------
    Path
        Full path to the parquet file.
    """
    ibkr_tf = _IBKR_TIMEFRAME_MAP.get(timeframe, timeframe)
    filename = f"{ticker.upper()}_{ibkr_tf}_{start_date}_{end_date}.parquet"
    return DATA_CACHE_DIR / filename


def _write_meta_sidecar(
    parquet_path: Path,
    ticker: str,
    timeframe: str,
    df: pd.DataFrame,
) -> None:
    """Write metadata sidecar JSON file.

    Parameters
    ----------
    parquet_path : Path
        Path to the associated parquet file.
    ticker : str
        Stock symbol.
    timeframe : str
        Canonical timeframe code.
    df : pd.DataFrame
        The OHLCV DataFrame being saved.
    """
    meta = {
        "timeframe": timeframe,
        "columns": list(df.columns),
    }
    _write_cache_meta(parquet_path, ticker, df, source="wrds_taq", meta=meta)


def flush_ticker_data(
    ticker: str,
    timeframe: str,
    df: pd.DataFrame,
    start_date_str: str,
    end_date_str: str,
) -> None:
    """Write accumulated OHLCV data to parquet with atomic writes.

    Parameters
    ----------
    ticker : str
        Stock symbol.
    timeframe : str
        Canonical timeframe code.
    df : pd.DataFrame
        Accumulated OHLCV bars.
    start_date_str : str
        Overall start date in YYYYMMDD format.
    end_date_str : str
        Overall end date in YYYYMMDD format.
    """
    if df.empty:
        return

    # Ensure correct column order, dtypes, and sorting
    df = df[REQUIRED_OHLCV].copy()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(np.float64)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')]
    df = df.dropna(subset=REQUIRED_OHLCV)
    df.index.name = 'Date'

    if df.empty:
        return

    target = _parquet_path(ticker, timeframe, start_date_str, end_date_str)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: tempfile + os.replace
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix='.tmp')
    try:
        os.close(fd)
        df.to_parquet(tmp)
        os.replace(tmp, str(target))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

    # Write metadata sidecar
    _write_meta_sidecar(target, ticker, timeframe, df)
    logger.info(
        "Saved %s %s: %d bars (%s to %s)",
        ticker, timeframe, len(df),
        df.index[0].strftime('%Y-%m-%d'),
        df.index[-1].strftime('%Y-%m-%d'),
    )


# ── Trading Calendar ──────────────────────────────────────────────────────

def _get_trading_days(start_dt: datetime, end_dt: datetime) -> pd.DatetimeIndex:
    """Get NYSE trading days between start and end dates.

    Parameters
    ----------
    start_dt : datetime
        Start date.
    end_dt : datetime
        End date.

    Returns
    -------
    pd.DatetimeIndex
        NYSE trading days in the date range.
    """
    s_str = start_dt.strftime('%Y-%m-%d')
    e_str = end_dt.strftime('%Y-%m-%d')

    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=s_str, end_date=e_str)
        return schedule.index
    except (ImportError, OSError, ValueError, RuntimeError):
        logger.info("NYSE calendar unavailable, falling back to business days")
        return pd.bdate_range(start=s_str, end=e_str)


# ── Main Download Logic ──────────────────────────────────────────────────

def _query_day_with_retry(
    wrds_provider: "WRDSProvider",
    trade_date: str,
    tickers: List[str],
    schema: str,
    prefix: str,
    col_map: Dict[str, str],
) -> Optional[pd.DataFrame]:
    """Query a single day with exponential backoff on failure.

    Parameters
    ----------
    wrds_provider : WRDSProvider
        Active WRDS connection.
    trade_date : str
        Date in YYYYMMDD format.
    tickers : list of str
        Ticker symbols to fetch.
    schema : str
        TAQ schema name.
    prefix : str
        Table prefix.
    col_map : dict
        Column name mapping.

    Returns
    -------
    pd.DataFrame or None
        Raw tick data, or None if all retries failed.
    """
    for attempt, delay in enumerate(RETRY_DELAYS):
        try:
            df = wrds_provider.get_taq_daily_ticks_batch(
                trade_date, tickers, schema, prefix, col_map,
            )
            return df
        except (OSError, ValueError, RuntimeError) as e:
            err_str = str(e).lower()
            if 'insufficientprivilege' in err_str:
                logger.warning(
                    "InsufficientPrivilege for %s — skipping", trade_date,
                )
                return None
            logger.warning(
                "WRDS query failed for %s (attempt %d/%d): %s",
                trade_date, attempt + 1, len(RETRY_DELAYS), e,
            )
            if attempt < len(RETRY_DELAYS) - 1:
                time.sleep(delay)

    logger.warning("All retries exhausted for %s — skipping", trade_date)
    return None


def run_download(
    tickers: List[str],
    timeframes: List[str],
    start_date: str,
    end_date: str,
    batch_days: int = DEFAULT_BATCH_DAYS,
    delay: float = DEFAULT_DELAY,
    force: bool = False,
    dry_run: bool = False,
    skip_1m: bool = False,
) -> None:
    """Execute the full TAQ intraday download.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to download.
    timeframes : list of str
        Timeframe codes to produce.
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).
    batch_days : int
        Days to accumulate before flushing to disk.
    delay : float
        Seconds between WRDS queries.
    force : bool
        Ignore checkpoint and re-download all.
    dry_run : bool
        Show plan without downloading.
    skip_1m : bool
        Skip saving 1-minute files.
    """
    try:
        from quant_engine.data.wrds_provider import WRDSProvider
    except ImportError as e:
        logger.error("Cannot import WRDSProvider: %s", e)
        sys.exit(1)

    provider = WRDSProvider()
    if not provider.available():
        logger.error(
            "WRDS connection not available. "
            "Check ~/.pgpass for WRDS credentials."
        )
        sys.exit(1)

    logger.info("WRDS connection established.")

    # Discover TAQ schema
    logger.info("Discovering TAQ Daily Product schema...")
    schema, prefix, col_map = discover_taq_schema(provider)
    schema_label = f"{schema}.{prefix}"
    logger.info("Using schema: %s", schema_label)

    # Parse dates
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    # Load checkpoint (unless --force)
    days_processed = 0
    started_at = datetime.now().isoformat()

    if not force:
        checkpoint = load_checkpoint()
        if checkpoint:
            last_date = checkpoint.get('last_completed_date')
            if last_date:
                resume_dt = datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)
                if resume_dt <= end_dt:
                    logger.info(
                        "Resuming from checkpoint: %s (after %s)",
                        resume_dt.strftime('%Y-%m-%d'), last_date,
                    )
                    start_dt = resume_dt
                    days_processed = checkpoint.get('days_processed', 0)
                    started_at = checkpoint.get('started_at', started_at)
                else:
                    logger.info("Checkpoint indicates download is complete.")
                    return

    # Get trading days
    trading_days = _get_trading_days(start_dt, end_dt)
    if len(trading_days) == 0:
        logger.info("No trading days in range %s to %s", start_date, end_date)
        return

    # Determine which timeframes to save
    save_timeframes = [tf for tf in timeframes if not (skip_1m and tf == '1m')]

    # Date strings for filenames
    overall_start_str = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
    overall_end_str = end_dt.strftime('%Y%m%d')

    logger.info(
        "Download plan: %d tickers, %d trading days, %d timeframes, "
        "batch_days=%d",
        len(tickers), len(trading_days), len(save_timeframes), batch_days,
    )

    if dry_run:
        logger.info("DRY RUN — no data will be downloaded.")
        logger.info("  Tickers: %s", ', '.join(tickers[:10]))
        if len(tickers) > 10:
            logger.info("  ... and %d more", len(tickers) - 10)
        logger.info("  Date range: %s to %s", start_date, end_date)
        logger.info("  Trading days: %d", len(trading_days))
        logger.info("  Timeframes: %s", ', '.join(save_timeframes))
        logger.info("  Schema: %s", schema_label)
        logger.info("  Estimated WRDS queries: %d", len(trading_days))
        return

    # Accumulators: ticker -> list of daily 1m DataFrames
    accumulators: Dict[str, List[pd.DataFrame]] = {
        t: [] for t in tickers
    }

    # Track accumulated data for final write
    all_accumulated: Dict[str, pd.DataFrame] = {
        t: pd.DataFrame(columns=REQUIRED_OHLCV) for t in tickers
    }

    # Signal handler for clean shutdown
    interrupted = False

    def _handle_interrupt(signum: int, frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        logger.warning("Interrupt received — saving checkpoint and exiting...")

    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_interrupt)

    try:
        for batch_start in range(0, len(trading_days), batch_days):
            if interrupted:
                break

            batch = trading_days[batch_start:batch_start + batch_days]
            logger.info(
                "Processing batch: %s to %s (%d days, %d/%d total)",
                batch[0].strftime('%Y-%m-%d'),
                batch[-1].strftime('%Y-%m-%d'),
                len(batch),
                batch_start + len(batch),
                len(trading_days),
            )

            for day in batch:
                if interrupted:
                    break

                date_str = day.strftime('%Y-%m-%d')
                date_yyyymmdd = day.strftime('%Y%m%d')

                # Single WRDS query for all tickers
                all_ticks = _query_day_with_retry(
                    provider, date_yyyymmdd, tickers,
                    schema, prefix, col_map,
                )

                if all_ticks is None or all_ticks.empty:
                    days_processed += 1
                    continue

                sym_col = col_map['sym']
                time_col = col_map['time']
                price_col = col_map['price']
                size_col = col_map['size']

                # Group by symbol and aggregate to 1m bars
                for ticker_sym, ticker_ticks in all_ticks.groupby(sym_col):
                    ticker_sym = str(ticker_sym).upper().strip()
                    if ticker_sym not in accumulators:
                        continue

                    parsed = _parse_ticks_for_day(
                        ticker_ticks, date_str, time_col, price_col, size_col,
                    )
                    if parsed.empty:
                        continue

                    bars_1m = aggregate_to_1m(parsed, 'price', 'size')
                    if not bars_1m.empty:
                        accumulators[ticker_sym].append(bars_1m)

                days_processed += 1

                if delay > 0:
                    time.sleep(delay)

            # Flush accumulators to accumulated data every batch
            for ticker in tickers:
                if accumulators[ticker]:
                    batch_df = pd.concat(accumulators[ticker], axis=0)
                    if all_accumulated[ticker].empty:
                        all_accumulated[ticker] = batch_df
                    else:
                        all_accumulated[ticker] = pd.concat(
                            [all_accumulated[ticker], batch_df], axis=0,
                        )
                    accumulators[ticker] = []

            # Save checkpoint
            last_day = batch[-1].strftime('%Y-%m-%d')
            save_checkpoint(
                last_completed_date=last_day,
                tickers_in_scope=len(tickers),
                schema_used=schema_label,
                started_at=started_at,
                days_processed=days_processed,
            )
            logger.info("Checkpoint saved: %s (%d days)", last_day, days_processed)

            # Memory management
            gc.collect()

        # Final flush: write all accumulated data to parquet
        logger.info("Writing final parquet files...")
        for ticker in tickers:
            if all_accumulated[ticker].empty:
                logger.debug("No data for %s — skipping", ticker)
                continue

            bars_1m = all_accumulated[ticker].copy()
            bars_1m = bars_1m.sort_index()
            bars_1m = bars_1m[~bars_1m.index.duplicated(keep='last')]

            for tf in save_timeframes:
                tf_minutes = TIMEFRAME_MINUTES[tf]
                if tf_minutes == 1:
                    tf_data = bars_1m
                else:
                    tf_data = resample_from_1m(bars_1m, tf_minutes)

                if not tf_data.empty:
                    flush_ticker_data(
                        ticker, tf, tf_data,
                        overall_start_str, overall_end_str,
                    )

        logger.info(
            "Download complete: %d days processed for %d tickers.",
            days_processed, len(tickers),
        )

    except OSError as e:
        err_str = str(e).lower()
        if 'no space' in err_str or 'disk full' in err_str:
            logger.critical("Disk full — saving checkpoint and exiting.")
        else:
            logger.error("OS error during download: %s", e)
        # Save checkpoint on any OS error
        if trading_days is not None and len(trading_days) > 0:
            last_idx = min(
                batch_start + batch_days - 1, len(trading_days) - 1,
            )
            save_checkpoint(
                last_completed_date=trading_days[last_idx].strftime('%Y-%m-%d'),
                tickers_in_scope=len(tickers),
                schema_used=schema_label,
                started_at=started_at,
                days_processed=days_processed,
            )
        raise
    finally:
        signal.signal(signal.SIGINT, original_sigint)


# ── Verification ──────────────────────────────────────────────────────────

def run_verify(tickers: List[str], timeframes: List[str]) -> None:
    """Verify existing cache files using the 13-point quality validation.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to verify.
    timeframes : list of str
        Timeframe codes to verify.
    """
    try:
        from quant_engine.data.intraday_quality import (
            quarantine_ticker,
            validate_intraday_bars,
            write_quality_report,
        )
    except ImportError as e:
        logger.error("Cannot import intraday_quality: %s", e)
        return

    total_passed = 0
    total_quarantined = 0

    for ticker in tickers:
        for timeframe in timeframes:
            df = load_intraday_ohlcv(ticker, timeframe)
            if df is None or df.empty:
                logger.debug("No data for %s %s — skipping verification", ticker, timeframe)
                continue

            cleaned, report = validate_intraday_bars(
                df, ticker, timeframe, source='wrds_taq',
            )

            # Write quality report sidecar
            ibkr_tf = _IBKR_TIMEFRAME_MAP.get(timeframe, timeframe)
            # Find the actual parquet file
            pattern = f"{ticker.upper()}_{ibkr_tf}_*.parquet"
            matches = sorted(DATA_CACHE_DIR.glob(pattern))
            if matches:
                quality_path = matches[-1].with_suffix('.quality.json')
                write_quality_report(quality_path, report)

            if report.quarantine:
                logger.warning(
                    "QUARANTINE %s %s: %s",
                    ticker, timeframe, report.quarantine_reason,
                )
                quarantine_ticker(
                    ticker, timeframe,
                    cache_dir=DATA_CACHE_DIR,
                    reason=report.quarantine_reason,
                    source_path=matches[-1] if matches else None,
                )
                total_quarantined += 1
            else:
                total_passed += 1

    logger.info(
        "Verification complete: %d passed, %d quarantined",
        total_passed, total_quarantined,
    )


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for TAQ intraday download."""
    parser = argparse.ArgumentParser(
        description="Download NYSE TAQ Daily Product intraday OHLCV data from WRDS",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show plan without downloading",
    )
    parser.add_argument(
        "--tickers", type=str, default=None,
        help="Comma-separated ticker list (default: full UNIVERSE_INTRADAY)",
    )
    parser.add_argument(
        "--timeframes", type=str, default=None,
        help="Comma-separated timeframes (default: all 6). "
             "Note: 1m is always downloaded internally for resampling.",
    )
    parser.add_argument(
        "--start", type=str, default=TAQ_START_DATE,
        help=f"Start date YYYY-MM-DD (default: {TAQ_START_DATE})",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore checkpoint, re-download all",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Verify existing cache files without downloading",
    )
    parser.add_argument(
        "--batch-days", type=int, default=DEFAULT_BATCH_DAYS,
        help=f"Days to accumulate before flushing (default: {DEFAULT_BATCH_DAYS})",
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"Seconds between WRDS queries (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--skip-1m", action="store_true",
        help="Skip saving 1m files (saves ~75%% disk space)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Resolve tickers
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
    else:
        tickers = list(UNIVERSE_INTRADAY)

    # Resolve timeframes
    if args.timeframes:
        timeframes = [t.strip() for t in args.timeframes.split(',')]
    else:
        timeframes = list(INTRADAY_TIMEFRAMES)

    # Resolve end date
    end_date = args.end or datetime.now().strftime('%Y-%m-%d')

    logger.info("=" * 60)
    logger.info("NYSE TAQ DAILY PRODUCT INTRADAY DOWNLOAD")
    logger.info("=" * 60)

    if args.verify_only:
        run_verify(tickers, timeframes)
        return

    run_download(
        tickers=tickers,
        timeframes=timeframes,
        start_date=args.start,
        end_date=end_date,
        batch_days=args.batch_days,
        delay=args.delay,
        force=args.force,
        dry_run=args.dry_run,
        skip_1m=args.skip_1m,
    )


if __name__ == '__main__':
    main()
