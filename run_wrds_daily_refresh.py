#!/usr/bin/env python3
"""
Re-download all daily OHLCV data from WRDS CRSP to replace old cache files
that have O=H=L=C (all identical prices).

The WRDS provider code has already been fixed to query openprc/askhi/bidlo
from CRSP dsf — this script re-downloads the data and replaces the cache.

Usage:
    python3 run_wrds_daily_refresh.py --dry-run       # Preview without downloading
    python3 run_wrds_daily_refresh.py --skip-cleanup   # Download but keep old files
    python3 run_wrds_daily_refresh.py --verify-only    # Only verify existing files
    python3 run_wrds_daily_refresh.py                  # Full run: download + cleanup
    python3 run_wrds_daily_refresh.py --tickers AAPL,MSFT,NVDA  # Specific tickers
    python3 run_wrds_daily_refresh.py --gics           # Refresh GICS sector mapping
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_engine.config import DATA_CACHE_DIR, UNIVERSE_FULL, BENCHMARK
from quant_engine.data.local_cache import list_cached_tickers, load_ohlcv_with_meta, save_ohlcv


def _build_ticker_list(tickers_arg, skip_terminal=True):
    """Build the full ticker list from cached + UNIVERSE_FULL + BENCHMARK.

    If *skip_terminal* is True, exclude tickers whose cache metadata has
    ``is_terminal=True`` (delisted stocks with complete, immutable data).
    """
    if tickers_arg:
        return [t.strip().upper() for t in tickers_arg.split(",") if t.strip()]
    cached = set(list_cached_tickers())
    universe = set(UNIVERSE_FULL)
    benchmark = {BENCHMARK}
    combined = sorted(cached | universe | benchmark)

    if not skip_terminal:
        return combined

    active = []
    skipped_terminal = []
    for ticker in combined:
        _, meta, _ = load_ohlcv_with_meta(ticker)
        if meta and meta.get("is_terminal") is True:
            skipped_terminal.append(ticker)
            continue
        active.append(ticker)
    if skipped_terminal:
        print(f"  Skipping {len(skipped_terminal)} terminal/delisted tickers "
              f"(immutable cache): {', '.join(skipped_terminal[:10])}"
              f"{'...' if len(skipped_terminal) > 10 else ''}")
    return active


def _verify_file(path):
    """Verify OHLCV quality for a single parquet file. Returns dict of results."""
    try:
        df = pd.read_parquet(path)
    except (OSError, ValueError) as e:
        return {"status": "read_error", "error": str(e)}

    n = len(df)
    if n == 0:
        return {"status": "empty"}

    result = {"status": "ok", "n_bars": n}

    # Check O != C ratio
    if "Open" in df.columns and "Close" in df.columns:
        o_ne_c = (df["Open"] != df["Close"]).sum()
        result["o_ne_c_ratio"] = float(o_ne_c / n)
        result["o_ne_c_count"] = int(o_ne_c)
    else:
        result["o_ne_c_ratio"] = 0.0

    # Check High >= Low
    if "High" in df.columns and "Low" in df.columns:
        h_ge_l = (df["High"] >= df["Low"]).all()
        result["high_ge_low"] = bool(h_ge_l)
    else:
        result["high_ge_low"] = None

    # Check High >= max(Open, Close)
    if all(c in df.columns for c in ["High", "Open", "Close"]):
        oc_max = df[["Open", "Close"]].max(axis=1)
        h_ge_oc = (df["High"] >= oc_max - 1e-6).all()
        result["high_ge_max_oc"] = bool(h_ge_oc)
    else:
        result["high_ge_max_oc"] = None

    # Check Low <= min(Open, Close)
    if all(c in df.columns for c in ["Low", "Open", "Close"]):
        oc_min = df[["Open", "Close"]].min(axis=1)
        l_le_oc = (df["Low"] <= oc_min + 1e-6).all()
        result["low_le_min_oc"] = bool(l_le_oc)
    else:
        result["low_le_min_oc"] = None

    # Date range
    if hasattr(df.index, "min"):
        result["start"] = str(pd.to_datetime(df.index.min()).date())
        result["end"] = str(pd.to_datetime(df.index.max()).date())

    return result


def _verify_all(cache_dir):
    """Run verification on all _1d.parquet files in cache."""
    cache = Path(cache_dir)
    files = sorted(cache.glob("*_1d.parquet"))
    if not files:
        print("  No _1d.parquet files found in cache.")
        return

    real_ohlcv = 0
    still_ohlc_same = 0
    quality_issues = 0
    total = len(files)

    for f in files:
        ticker = f.stem.replace("_1d", "")
        result = _verify_file(f)
        if result["status"] != "ok":
            print(f"  {ticker}: {result['status']} — {result.get('error', '')}")
            quality_issues += 1
            continue

        ratio = result.get("o_ne_c_ratio", 0)
        bars = result.get("n_bars", 0)
        h_ge_l = result.get("high_ge_low", True)
        h_ge_oc = result.get("high_ge_max_oc", True)
        l_le_oc = result.get("low_le_min_oc", True)

        issues = []
        if h_ge_l is False:
            issues.append("High<Low")
        if h_ge_oc is False:
            issues.append("High<max(O,C)")
        if l_le_oc is False:
            issues.append("Low>min(O,C)")

        if ratio < 0.10:
            still_ohlc_same += 1
            label = "O=H=L=C"
        else:
            real_ohlcv += 1
            label = "real OHLCV"

        issue_str = f" ISSUES: {', '.join(issues)}" if issues else ""
        if issues:
            quality_issues += 1
        print(f"  {ticker}: {bars} bars, O!=C on {ratio:.1%} of rows ({label})"
              f" [{result.get('start', '?')} → {result.get('end', '?')}]{issue_str}")

    print(f"\n  Verification Summary:")
    print(f"    Total tickers: {total}")
    print(f"    Real OHLCV (O!=C >10%): {real_ohlcv}")
    print(f"    Still O=H=L=C: {still_ohlc_same}")
    print(f"    Quality issues: {quality_issues}")


def _cleanup_old_daily(cache_dir, downloaded_tickers):
    """Remove old {TICKER}_daily_{dates}.parquet and .meta.json files."""
    cache = Path(cache_dir)
    removed = 0
    for ticker in downloaded_tickers:
        for old_file in sorted(cache.glob(f"{ticker.upper()}_daily_*.parquet")):
            print(f"  Removing: {old_file.name}")
            old_file.unlink()
            removed += 1
            # Remove corresponding meta.json
            meta = old_file.with_suffix("").with_suffix(".meta.json")
            if not meta.exists():
                # Try alternate naming: {stem}.meta.json
                meta = old_file.parent / f"{old_file.stem}.meta.json"
            if meta.exists():
                print(f"  Removing: {meta.name}")
                meta.unlink()
                removed += 1
    return removed


GICS_SECTOR_NAMES = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer Discretionary",
    "30": "Consumer Staples",
    "35": "Health Care",
    "40": "Financials",
    "45": "Information Technology",
    "50": "Communication Services",
    "55": "Utilities",
    "60": "Real Estate",
}


def refresh_gics_sectors(provider, tickers: List[str]) -> Dict[str, str]:
    """Query Compustat for GICS sector mapping.

    Queries ``comp.security`` joined with ``comp.company`` to retrieve
    the GICS sector code for each ticker, then maps the 2-digit code
    to a human-readable sector name.

    Parameters
    ----------
    provider : WRDSProvider
        An initialised WRDS provider with an active connection.
    tickers : list[str]
        Tickers to look up.

    Returns
    -------
    dict[str, str]
        Mapping of ticker -> GICS sector name.
    """
    if not tickers:
        return {}

    # Process in batches to avoid SQL query size limits
    batch_size = 200
    all_results: Dict[str, str] = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        placeholders = ",".join(f"'{t}'" for t in batch)
        query = f"""
            SELECT DISTINCT a.tic, b.gsector
            FROM comp.security a
            JOIN comp.company b ON a.gvkey = b.gvkey
            WHERE a.tic IN ({placeholders})
            AND b.gsector IS NOT NULL
        """
        df = provider._query(query)
        if df.empty:
            continue
        for _, row in df.iterrows():
            ticker = str(row["tic"]).strip().upper()
            sector_code = str(int(row["gsector"])) if pd.notna(row["gsector"]) else None
            if sector_code and sector_code in GICS_SECTOR_NAMES:
                all_results[ticker] = GICS_SECTOR_NAMES[sector_code]
            elif sector_code:
                all_results[ticker] = "Unknown"

    return all_results


def _write_gics_to_universe_yaml(sector_map: Dict[str, str], yaml_path: Path) -> None:
    """Merge refreshed GICS sectors into universe.yaml.

    Reads the existing YAML, replaces the ``sectors`` key with the new
    mapping (grouped by sector name), and writes back. Preserves all
    other keys (liquidity_tiers, borrowability, etc.).

    Parameters
    ----------
    sector_map : dict[str, str]
        Mapping of ticker -> sector name from WRDS.
    yaml_path : Path
        Path to ``config_data/universe.yaml``.
    """
    # Load existing YAML to preserve non-sector keys
    if yaml_path.exists():
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    # Group tickers by sector
    sectors: Dict[str, List[str]] = {}
    for ticker, sector_name in sorted(sector_map.items()):
        # Normalize sector name to lowercase for YAML key consistency
        key = sector_name.lower().replace(" ", "_")
        sectors.setdefault(key, []).append(ticker)

    # Sort tickers within each sector
    for key in sectors:
        sectors[key] = sorted(sectors[key])

    data["sectors"] = sectors

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def main():
    """Run the WRDS daily refresh workflow and emit a summary of refreshed datasets and outputs."""
    parser = argparse.ArgumentParser(
        description="Re-download daily OHLCV from WRDS CRSP to fix O=H=L=C cache files",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be downloaded without downloading")
    parser.add_argument("--skip-cleanup", action="store_true",
                        help="Download new files but keep old _daily_ files")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated ticker list (default: all ~183)")
    parser.add_argument("--years", type=int, default=20,
                        help="Lookback years (default: 20)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Tickers per WRDS query (default: 50)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only run verification on existing _1d.parquet files")
    parser.add_argument("--gics", action="store_true",
                        help="Refresh GICS sector mapping from WRDS Compustat and write to config_data/universe.yaml")
    parser.add_argument("--include-terminal", action="store_true",
                        help="Include delisted/terminal tickers in refresh (normally skipped as immutable)")
    parser.add_argument("--backfill-terminal", action="store_true",
                        help="Scan cache and mark delisted stocks as terminal (one-time migration)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"WRDS DAILY DATA RE-DOWNLOAD")
    print(f"{'='*60}")

    # Handle --backfill-terminal before building the ticker list
    if args.backfill_terminal:
        print(f"\n── Backfill Terminal Metadata ──")
        from quant_engine.data.local_cache import backfill_terminal_metadata
        result = backfill_terminal_metadata(DATA_CACHE_DIR)
        print(f"  Scanned: {result['scanned']}")
        print(f"  Newly marked terminal: {result['updated']}")
        print(f"  Already terminal: {result['already_terminal']}")
        print(f"  Active (no delist): {result['active']}")
        print(f"  Errors: {result['errors']}")
        return

    tickers = _build_ticker_list(args.tickers, skip_terminal=not args.include_terminal)
    print(f"  Tickers: {len(tickers)}")
    print(f"  Cache dir: {DATA_CACHE_DIR}")

    if args.verify_only:
        print(f"\n── Verification Only ──")
        _verify_all(DATA_CACHE_DIR)
        return

    if args.gics:
        print(f"\n── GICS Sector Refresh ──")
        try:
            from quant_engine.data.wrds_provider import WRDSProvider
        except ImportError as e:
            print(f"  ERROR: Cannot import WRDSProvider: {e}")
            sys.exit(1)

        provider = WRDSProvider()
        if not provider.available():
            print("  ERROR: WRDS connection not available.")
            print("  Check ~/.pgpass for WRDS credentials.")
            sys.exit(1)

        print(f"  Querying Compustat for GICS sector codes...")
        sector_map = refresh_gics_sectors(provider, tickers)
        print(f"  Found GICS sectors for {len(sector_map)} / {len(tickers)} tickers")

        if sector_map:
            yaml_path = Path(__file__).parent / "config_data" / "universe.yaml"
            _write_gics_to_universe_yaml(sector_map, yaml_path)
            print(f"  Updated {yaml_path}")

            # Print sector summary
            sector_counts: Dict[str, int] = {}
            for sector_name in sector_map.values():
                sector_counts[sector_name] = sector_counts.get(sector_name, 0) + 1
            print(f"\n  Sector distribution:")
            for sector_name, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
                print(f"    {sector_name}: {count} tickers")
        else:
            print("  WARNING: No GICS data returned from Compustat.")
        return

    if args.dry_run:
        print(f"\n── Dry Run ──")
        print(f"  Would download {len(tickers)} tickers from WRDS CRSP")
        print(f"  Date range: {datetime.now().year - args.years}-01-01 → today")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Output: {{TICKER}}_1d.parquet + .meta.json in {DATA_CACHE_DIR}")
        print(f"\n  Tickers: {', '.join(tickers[:20])}{'...' if len(tickers) > 20 else ''}")
        if not args.skip_cleanup:
            # Count old files that would be removed
            old_count = sum(1 for _ in DATA_CACHE_DIR.glob("*_daily_*.parquet"))
            print(f"\n  Old _daily_ files that would be removed: {old_count}")
        return

    # Connect to WRDS
    print(f"\n── Connecting to WRDS ──")
    try:
        from quant_engine.data.wrds_provider import WRDSProvider
    except ImportError as e:
        print(f"  ERROR: Cannot import WRDSProvider: {e}")
        sys.exit(1)

    provider = WRDSProvider()
    if not provider.available():
        print("  ERROR: WRDS connection not available.")
        print("  Check ~/.pgpass for WRDS credentials.")
        sys.exit(1)
    print("  WRDS connection established.")

    # Download in batches
    start_date = f"{datetime.now().year - args.years}-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    print(f"\n── Downloading ({start_date} → {end_date}) ──")

    downloaded = []
    failed = []
    not_found = []
    batch_size = args.batch_size
    t0 = time.time()

    for batch_start in range(0, len(tickers), batch_size):
        batch = tickers[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        print(f"\n  Batch {batch_num}/{total_batches}: {len(batch)} tickers "
              f"({batch[0]}...{batch[-1]})")

        retry_batch_size = max(1, batch_size // 2)
        try:
            result = provider.get_crsp_prices_with_delistings(
                batch, start_date=start_date, end_date=end_date,
            )
        except (OSError, ValueError, RuntimeError) as e:
            print(f"    Batch failed ({type(e).__name__}: {e}), retrying with half batch size...")
            # Retry with smaller batches
            result = {}
            for sub_start in range(0, len(batch), retry_batch_size):
                sub_batch = batch[sub_start:sub_start + retry_batch_size]
                try:
                    sub_result = provider.get_crsp_prices_with_delistings(
                        sub_batch, start_date=start_date, end_date=end_date,
                    )
                    result.update(sub_result)
                except (OSError, ValueError, RuntimeError) as e2:
                    print(f"    Sub-batch also failed ({type(e2).__name__}: {e2}), skipping.")
                    for t in sub_batch:
                        failed.append((t, str(e2)))

        # Map permno results back to tickers
        # Build ticker -> best permno result mapping
        ticker_results = {}
        for permno_key, df in result.items():
            if len(df) == 0:
                continue
            if "ticker" in df.columns:
                ticker_val = str(df["ticker"].iloc[-1]).strip().upper()
            else:
                ticker_val = None

            if ticker_val and ticker_val in [t.upper() for t in batch]:
                # If multiple permnos map to same ticker, keep the longest
                if ticker_val not in ticker_results or len(df) > len(ticker_results[ticker_val][1]):
                    ticker_results[ticker_val] = (permno_key, df)

        # Track which tickers in this batch were found
        found_tickers = set(ticker_results.keys())
        for t in batch:
            if t.upper() not in found_tickers:
                not_found.append(t)

        # Save each ticker's data
        for ticker_val, (permno_key, df) in ticker_results.items():
            o_ne_c = (df["Open"] != df["Close"]).sum() if "Open" in df.columns and "Close" in df.columns else 0
            pct = 100.0 * o_ne_c / len(df) if len(df) > 0 else 0
            start = str(pd.to_datetime(df.index.min()).date()) if len(df) > 0 else "?"
            end = str(pd.to_datetime(df.index.max()).date()) if len(df) > 0 else "?"

            try:
                save_ohlcv(
                    ticker_val, df,
                    source="wrds_delisting",
                    meta={"permno": permno_key, "ticker": ticker_val,
                          "years_requested": args.years},
                )
                downloaded.append(ticker_val)
                print(f"    {ticker_val} (permno={permno_key}): {len(df)} bars, "
                      f"{start} → {end}, O!=C on {pct:.1f}% of rows")
            except ValueError as e:
                print(f"    {ticker_val}: save_ohlcv failed — {e}")
                failed.append((ticker_val, str(e)))
            except (OSError, RuntimeError) as e:
                print(f"    {ticker_val}: unexpected error — {type(e).__name__}: {e}")
                failed.append((ticker_val, str(e)))

    elapsed = time.time() - t0

    # Verify
    print(f"\n── Verification ──")
    _verify_all(DATA_CACHE_DIR)

    # Cleanup
    removed = 0
    if not args.skip_cleanup and downloaded:
        print(f"\n── Cleanup old _daily_ files ──")
        removed = _cleanup_old_daily(DATA_CACHE_DIR, downloaded)
        print(f"  Removed {removed} old files")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total tickers attempted: {len(tickers)}")
    print(f"  Successfully downloaded: {len(downloaded)}")
    print(f"  Not found in CRSP: {len(not_found)}")
    if not_found:
        print(f"    {', '.join(not_found)}")
    print(f"  Failed: {len(failed)}")
    if failed:
        for t, err in failed:
            print(f"    {t}: {err}")
    print(f"  Old daily files removed: {removed}")
    print(f"  New _1d.parquet files created: {len(downloaded)}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"\n  Features unlocked: ~15 (ATR, NATR, Parkinson Vol, GK Vol, YZ Vol,")
    print(f"    Stochastic, Williams %R, CCI, Candlestick Body, Gap %, etc.)")


if __name__ == "__main__":
    main()
