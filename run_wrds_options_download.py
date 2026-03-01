#!/usr/bin/env python3
"""Download and cache OptionMetrics options data from WRDS.

SPEC 3: OptionMetrics Options Data Download

Downloads three data products:
  1. Link table (PERMNO -> SecID mapping)
  2. IV Surface (daily implied volatility features)
  3. Volume & Open Interest (daily aggregates)

Data is stored in ``data/cache/options/`` with atomic writes and
metadata sidecars.  Supports checkpoint/resume for long-running
downloads.

Usage::

    python3 run_wrds_options_download.py
    python3 run_wrds_options_download.py --dry-run
    python3 run_wrds_options_download.py --tickers AAPL,MSFT --iv-only
    python3 run_wrds_options_download.py --force --start 2000-01-01
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from quant_engine.config import (
    DATA_CACHE_DIR,
    UNIVERSE_FULL,
    UNIVERSE_INTRADAY,
    REQUIRE_PERMNO,
)
from quant_engine.data.wrds_provider import WRDSProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPTIONS_CACHE_DIR = DATA_CACHE_DIR / "options"
LINK_TABLE_PATH = DATA_CACHE_DIR / "optionmetrics_link_table.parquet"
LINK_TABLE_META_PATH = DATA_CACHE_DIR / "optionmetrics_link_table.meta.json"
META_SOURCE = "wrds_optionmetrics"
DEFAULT_START = "2000-01-01"
DEFAULT_END = datetime.now().strftime("%Y-%m-%d")
INTER_QUERY_DELAY = 0.1  # seconds between WRDS queries


# ---------------------------------------------------------------------------
# Atomic write helpers (matching local_cache.py patterns)
# ---------------------------------------------------------------------------

def _atomic_write_parquet(target: Path, df: pd.DataFrame) -> None:
    """Write a DataFrame to parquet atomically."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        os.close(fd)
        df.to_parquet(tmp)
        os.replace(tmp, str(target))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_meta(target: Path, payload: dict) -> None:
    """Write a metadata JSON sidecar atomically."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        os.close(fd)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)
        os.replace(tmp, str(target))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_pair(parquet_path: Path, meta_path: Path,
                df: pd.DataFrame, meta_payload: dict) -> None:
    """Write parquet + meta.json as an atomic pair."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(parquet_path.parent)) as tmp_dir:
        tmp_parquet = Path(tmp_dir) / parquet_path.name
        tmp_meta = Path(tmp_dir) / meta_path.name
        df.to_parquet(tmp_parquet)
        tmp_meta.write_text(
            json.dumps(meta_payload, default=str, indent=2, sort_keys=True)
        )
        os.replace(str(tmp_parquet), str(parquet_path))
        os.replace(str(tmp_meta), str(meta_path))


# ---------------------------------------------------------------------------
# Meta payload builder
# ---------------------------------------------------------------------------

def _build_meta(
    permno: str,
    product: str,
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> dict:
    """Build metadata sidecar payload for an options cache file."""
    return {
        "permno": str(permno),
        "product": product,
        "source": META_SOURCE,
        "saved_at": datetime.utcnow().isoformat(),
        "start": start_date,
        "end": end_date,
        "rows": int(len(df)),
        "columns": list(df.columns),
    }


# ---------------------------------------------------------------------------
# 1. Link table download
# ---------------------------------------------------------------------------

def _build_full_universe() -> List[str]:
    """Return the union of UNIVERSE_FULL and UNIVERSE_INTRADAY."""
    return sorted(set(UNIVERSE_FULL) | set(UNIVERSE_INTRADAY))


def download_link_table(
    wrds: WRDSProvider,
    tickers: List[str],
    start_date: str,
    end_date: str,
    force: bool = False,
) -> pd.DataFrame:
    """Download the OptionMetrics PERMNO -> SecID link table.

    Parameters
    ----------
    wrds : WRDSProvider
        Active WRDS provider instance.
    tickers : list of str
        Ticker symbols to resolve.
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).
    force : bool
        If True, re-download even if cached.

    Returns
    -------
    pd.DataFrame
        Link table with columns: permno, secid, link_start, link_end.
    """
    if not force and LINK_TABLE_PATH.exists():
        logger.info("Link table already cached at %s — loading", LINK_TABLE_PATH)
        return pd.read_parquet(LINK_TABLE_PATH)

    logger.info("Resolving PERMNOs for %d tickers ...", len(tickers))
    permnos: List[str] = []
    for ticker in tickers:
        permno = wrds.resolve_permno(ticker)
        if permno is not None:
            permnos.append(permno)
        else:
            logger.debug("Could not resolve PERMNO for %s — skipping", ticker)
        time.sleep(0.02)  # gentle rate limit

    permnos = sorted(set(permnos))
    if not permnos:
        logger.error("No PERMNOs resolved — cannot build link table")
        return pd.DataFrame(columns=["permno", "secid", "link_start", "link_end"])

    logger.info("Resolved %d unique PERMNOs, querying OptionMetrics link table ...", len(permnos))
    link = wrds.get_optionmetrics_link(
        permnos=permnos,
        start_date=start_date,
        end_date=end_date,
    )
    if link.empty:
        logger.warning("OptionMetrics link table returned empty for %d PERMNOs", len(permnos))
        return link

    logger.info("Link table: %d rows covering %d PERMNOs",
                len(link), link["permno"].nunique())

    # Save atomically
    meta = {
        "source": META_SOURCE,
        "saved_at": datetime.utcnow().isoformat(),
        "rows": int(len(link)),
        "permnos_queried": len(permnos),
        "columns": list(link.columns),
        "start": start_date,
        "end": end_date,
    }
    _write_pair(LINK_TABLE_PATH, LINK_TABLE_META_PATH, link, meta)
    logger.info("Saved link table to %s", LINK_TABLE_PATH)
    return link


# ---------------------------------------------------------------------------
# 2. IV Surface download (yearly batches)
# ---------------------------------------------------------------------------

def download_iv_surface(
    wrds: WRDSProvider,
    link: pd.DataFrame,
    start_date: str,
    end_date: str,
    force: bool = False,
    target_permnos: Optional[Set[str]] = None,
) -> int:
    """Download IV surface data for all PERMNOs in the link table.

    Queries in yearly batches per PERMNO group to avoid WRDS memory limits.
    Uses the existing ``get_option_surface_features()`` method which
    applies ``_nearest_iv()`` for ATM/delta selection.

    Parameters
    ----------
    wrds : WRDSProvider
        Active WRDS provider instance.
    link : pd.DataFrame
        Link table with permno, secid, link_start, link_end.
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).
    force : bool
        If True, re-download even if cached.
    target_permnos : set of str, optional
        If provided, only download for these PERMNOs.

    Returns
    -------
    int
        Number of PERMNOs successfully processed.
    """
    if link.empty:
        logger.warning("Empty link table — skipping IV surface download")
        return 0

    unique_permnos = sorted(link["permno"].unique())
    if target_permnos:
        unique_permnos = [p for p in unique_permnos if str(p) in target_permnos]

    logger.info("Downloading IV surface for %d PERMNOs (%s to %s) ...",
                len(unique_permnos), start_date, end_date)

    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    success_count = 0

    for i, permno in enumerate(unique_permnos, 1):
        permno_str = str(permno)
        out_path = OPTIONS_CACHE_DIR / f"{permno_str}_iv_surface.parquet"
        meta_path = OPTIONS_CACHE_DIR / f"{permno_str}_iv_surface.meta.json"

        if not force and out_path.exists():
            logger.debug("IV surface cached for PERMNO %s — skipping", permno_str)
            success_count += 1
            continue

        logger.info("[%d/%d] PERMNO %s: fetching IV surface ...",
                    i, len(unique_permnos), permno_str)

        yearly_dfs: List[pd.DataFrame] = []
        for year in range(start_year, end_year + 1):
            yr_start = f"{year}-01-01" if year > start_year else start_date
            yr_end = f"{year}-12-31" if year < end_year else end_date

            try:
                chunk = wrds.get_option_surface_features(
                    permnos=[permno_str],
                    start_date=yr_start,
                    end_date=yr_end,
                )
                if not chunk.empty:
                    yearly_dfs.append(chunk)
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("IV surface query failed for PERMNO %s year %d: %s",
                               permno_str, year, e)

            time.sleep(INTER_QUERY_DELAY)

        if not yearly_dfs:
            logger.debug("No IV surface data for PERMNO %s", permno_str)
            continue

        combined = pd.concat(yearly_dfs)
        # Drop duplicates and sort
        if isinstance(combined.index, pd.MultiIndex):
            combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

        meta = _build_meta(permno_str, "iv_surface", combined, start_date, end_date)
        _write_pair(out_path, meta_path, combined, meta)
        success_count += 1
        logger.info("[%d/%d] PERMNO %s: saved %d IV surface rows",
                    i, len(unique_permnos), permno_str, len(combined))

    logger.info("IV surface download complete: %d/%d PERMNOs",
                success_count, len(unique_permnos))
    return success_count


# ---------------------------------------------------------------------------
# 3. Volume & OI download (yearly batches)
# ---------------------------------------------------------------------------

def download_options_volume(
    wrds: WRDSProvider,
    link: pd.DataFrame,
    start_date: str,
    end_date: str,
    force: bool = False,
    target_permnos: Optional[Set[str]] = None,
) -> int:
    """Download options volume and open interest for all PERMNOs.

    Queries in yearly batches per PERMNO to avoid WRDS memory limits.
    Uses the existing ``query_options_volume()`` method.

    Parameters
    ----------
    wrds : WRDSProvider
        Active WRDS provider instance.
    link : pd.DataFrame
        Link table with permno, secid, link_start, link_end.
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).
    force : bool
        If True, re-download even if cached.
    target_permnos : set of str, optional
        If provided, only download for these PERMNOs.

    Returns
    -------
    int
        Number of PERMNOs successfully processed.
    """
    if link.empty:
        logger.warning("Empty link table — skipping volume/OI download")
        return 0

    unique_permnos = sorted(link["permno"].unique())
    if target_permnos:
        unique_permnos = [p for p in unique_permnos if str(p) in target_permnos]

    logger.info("Downloading volume/OI for %d PERMNOs (%s to %s) ...",
                len(unique_permnos), start_date, end_date)

    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    success_count = 0

    for i, permno in enumerate(unique_permnos, 1):
        permno_str = str(permno)
        out_path = OPTIONS_CACHE_DIR / f"{permno_str}_options_volume.parquet"
        meta_path = OPTIONS_CACHE_DIR / f"{permno_str}_options_volume.meta.json"

        if not force and out_path.exists():
            logger.debug("Volume/OI cached for PERMNO %s — skipping", permno_str)
            success_count += 1
            continue

        logger.info("[%d/%d] PERMNO %s: fetching volume/OI ...",
                    i, len(unique_permnos), permno_str)

        yearly_dfs: List[pd.DataFrame] = []
        for year in range(start_year, end_year + 1):
            yr_start = f"{year}-01-01" if year > start_year else start_date
            yr_end = f"{year}-12-31" if year < end_year else end_date

            try:
                chunk = wrds.query_options_volume(
                    permno=permno_str,
                    start_date=yr_start,
                    end_date=yr_end,
                )
                if chunk is not None and not chunk.empty:
                    yearly_dfs.append(chunk)
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("Volume/OI query failed for PERMNO %s year %d: %s",
                               permno_str, year, e)

            time.sleep(INTER_QUERY_DELAY)

        if not yearly_dfs:
            logger.debug("No volume/OI data for PERMNO %s", permno_str)
            continue

        combined = pd.concat(yearly_dfs, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        combined = combined.drop_duplicates(subset=["date"], keep="last")
        combined = combined.sort_values("date").reset_index(drop=True)

        meta = _build_meta(permno_str, "options_volume", combined, start_date, end_date)
        _write_pair(out_path, meta_path, combined, meta)
        success_count += 1
        logger.info("[%d/%d] PERMNO %s: saved %d volume/OI rows",
                    i, len(unique_permnos), permno_str, len(combined))

    logger.info("Volume/OI download complete: %d/%d PERMNOs",
                success_count, len(unique_permnos))
    return success_count


# ---------------------------------------------------------------------------
# Ticker -> PERMNO resolution helper
# ---------------------------------------------------------------------------

def _resolve_target_permnos(
    wrds: WRDSProvider,
    tickers: List[str],
) -> Set[str]:
    """Resolve a list of tickers to their PERMNOs."""
    permnos: Set[str] = set()
    for ticker in tickers:
        permno = wrds.resolve_permno(ticker)
        if permno is not None:
            permnos.add(permno)
        else:
            logger.warning("Could not resolve PERMNO for ticker %s", ticker)
    return permnos


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download OptionMetrics options data from WRDS (SPEC 3).",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without downloading")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated tickers (default: full union universe)")
    parser.add_argument("--start", type=str, default=DEFAULT_START,
                        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})")
    parser.add_argument("--end", type=str, default=DEFAULT_END,
                        help=f"End date YYYY-MM-DD (default: {DEFAULT_END})")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cached data, re-download everything")
    parser.add_argument("--link-table-only", action="store_true",
                        help="Download only the link table, skip IV and volume")
    parser.add_argument("--iv-only", action="store_true",
                        help="Download link table + IV surface only")
    parser.add_argument("--volume-only", action="store_true",
                        help="Download link table + volume/OI only")
    return parser.parse_args()


def main() -> None:
    """Entry point for OptionMetrics data download."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _parse_args()

    # Determine ticker universe
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = _build_full_universe()

    logger.info("=" * 60)
    logger.info("OptionMetrics Download — SPEC 3")
    logger.info("=" * 60)
    logger.info("Tickers:   %d", len(tickers))
    logger.info("Date range: %s to %s", args.start, args.end)
    logger.info("Force:      %s", args.force)
    logger.info("Dry run:    %s", args.dry_run)

    do_iv = not args.volume_only
    do_volume = not args.iv_only

    if args.link_table_only:
        do_iv = False
        do_volume = False

    logger.info("Products:   link_table=%s, iv_surface=%s, volume_oi=%s",
                True, do_iv, do_volume)

    if args.dry_run:
        logger.info("DRY RUN — no data will be downloaded")
        logger.info("Would create: %s", OPTIONS_CACHE_DIR)
        logger.info("Would create: %s", LINK_TABLE_PATH)
        if do_iv:
            logger.info("Would download IV surface for up to %d PERMNOs", len(tickers))
        if do_volume:
            logger.info("Would download volume/OI for up to %d PERMNOs", len(tickers))
        return

    # Ensure output directory exists
    OPTIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize WRDS
    wrds = WRDSProvider()
    if not wrds.available():
        logger.error("WRDS not available — cannot proceed. "
                     "Set WRDS_USERNAME env var and ensure ~/.pgpass is configured.")
        sys.exit(1)

    # Resolve target PERMNOs if specific tickers were provided
    target_permnos: Optional[Set[str]] = None
    if args.tickers:
        target_permnos = _resolve_target_permnos(wrds, tickers)
        if not target_permnos:
            logger.error("No PERMNOs resolved for specified tickers — exiting")
            sys.exit(1)
        logger.info("Resolved %d target PERMNOs from %d tickers",
                    len(target_permnos), len(tickers))

    t0 = time.time()

    # Step 1: Link table
    link = download_link_table(
        wrds=wrds,
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        force=args.force,
    )
    if link.empty:
        logger.error("Link table is empty — cannot download IV or volume data")
        sys.exit(1)

    # Step 2: IV Surface
    iv_count = 0
    if do_iv:
        iv_count = download_iv_surface(
            wrds=wrds,
            link=link,
            start_date=args.start,
            end_date=args.end,
            force=args.force,
            target_permnos=target_permnos,
        )

    # Step 3: Volume & OI
    vol_count = 0
    if do_volume:
        vol_count = download_options_volume(
            wrds=wrds,
            link=link,
            start_date=args.start,
            end_date=args.end,
            force=args.force,
            target_permnos=target_permnos,
        )

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("OptionMetrics download complete in %.1f minutes", elapsed / 60)
    logger.info("Link table:  %d rows", len(link))
    logger.info("IV surface:  %d PERMNOs processed", iv_count)
    logger.info("Volume/OI:   %d PERMNOs processed", vol_count)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
