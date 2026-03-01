#!/usr/bin/env python3
"""
Re-download all daily OHLCV data from WRDS CRSP to replace old cache files
that have O=H=L=C (all identical prices).

Usage:
    python3 run_wrds_daily_refresh.py --dry-run       # Preview without downloading
    python3 run_wrds_daily_refresh.py --skip-cleanup   # Download but keep old files
    python3 run_wrds_daily_refresh.py --verify-only    # Only verify existing files
    python3 run_wrds_daily_refresh.py                  # Full run: download + cleanup
    python3 run_wrds_daily_refresh.py --tickers AAPL,MSFT,NVDA  # Specific tickers
    python3 run_wrds_daily_refresh.py --gics           # Refresh GICS sector mapping
    python3 run_wrds_daily_refresh.py --delisted        # Download all delisted stocks
"""
import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_engine.config import DATA_CACHE_DIR, UNIVERSE_FULL, BENCHMARK, SURVIVORSHIP_DB
from quant_engine.data.local_cache import list_cached_tickers, load_ohlcv_with_meta, save_ohlcv
from quant_engine.data.survivorship import DelistingHandler, DelistingEvent, DelistingReason


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _crsp_dlstcd_to_reason(dlstcd) -> DelistingReason:
    """Map CRSP delisting code to DelistingReason enum."""
    try:
        code = int(dlstcd)
    except (ValueError, TypeError):
        return DelistingReason.UNKNOWN
    if 200 <= code < 300:
        if code in (233, 241, 242, 243, 244):
            return DelistingReason.MERGER_CASH
        return DelistingReason.MERGER_STOCK
    if 300 <= code < 400:
        return DelistingReason.EXCHANGE_DELISTING
    if 400 <= code < 500:
        if 450 <= code < 460:
            return DelistingReason.GOING_PRIVATE
        return DelistingReason.BANKRUPTCY
    if 500 <= code < 600:
        if 570 <= code < 580:
            return DelistingReason.SPINOFF
        return DelistingReason.VOLUNTARY
    return DelistingReason.UNKNOWN


def _delisted_cache_symbol(ticker: str, permno) -> str:
    """Build a collision-safe cache symbol for delisted securities."""
    permno_str = str(int(permno))
    ticker_u = str(ticker).strip().upper() if ticker else ""
    if ticker_u:
        return f"{ticker_u}__PERMNO{permno_str}"
    return f"PERMNO{permno_str}"


def _sanitize_permno_in_clause(values) -> str:
    """Build a numeric SQL IN list for PERMNO values."""
    clean = []
    for v in values:
        s = str(v).strip()
        if s.isdigit():
            clean.append(str(int(s)))
    return ",".join(sorted(set(clean)))


# ─────────────────────────────────────────────────────────────────────────────
# Ticker list for normal refresh
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Verification helpers
# ─────────────────────────────────────────────────────────────────────────────

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

    if "Open" in df.columns and "Close" in df.columns:
        o_ne_c = (df["Open"] != df["Close"]).sum()
        result["o_ne_c_ratio"] = float(o_ne_c / n)
        result["o_ne_c_count"] = int(o_ne_c)
    else:
        result["o_ne_c_ratio"] = 0.0

    if "High" in df.columns and "Low" in df.columns:
        result["high_ge_low"] = bool((df["High"] >= df["Low"]).all())
    else:
        result["high_ge_low"] = None

    if all(c in df.columns for c in ["High", "Open", "Close"]):
        oc_max = df[["Open", "Close"]].max(axis=1)
        result["high_ge_max_oc"] = bool((df["High"] >= oc_max - 1e-6).all())
    else:
        result["high_ge_max_oc"] = None

    if all(c in df.columns for c in ["Low", "Open", "Close"]):
        oc_min = df[["Open", "Close"]].min(axis=1)
        result["low_le_min_oc"] = bool((df["Low"] <= oc_min + 1e-6).all())
    else:
        result["low_le_min_oc"] = None

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
            meta = old_file.with_suffix("").with_suffix(".meta.json")
            if not meta.exists():
                meta = old_file.parent / f"{old_file.stem}.meta.json"
            if meta.exists():
                print(f"  Removing: {meta.name}")
                meta.unlink()
                removed += 1
    return removed


# ─────────────────────────────────────────────────────────────────────────────
# GICS sector refresh
# ─────────────────────────────────────────────────────────────────────────────

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
    """Query Compustat for GICS sector mapping."""
    if not tickers:
        return {}

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
    """Merge refreshed GICS sectors into universe.yaml."""
    if yaml_path.exists():
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    sectors: Dict[str, List[str]] = {}
    for ticker, sector_name in sorted(sector_map.items()):
        key = sector_name.lower().replace(" ", "_")
        sectors.setdefault(key, []).append(ticker)

    for key in sectors:
        sectors[key] = sorted(sectors[key])

    data["sectors"] = sectors

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ─────────────────────────────────────────────────────────────────────────────
# Delisted stock discovery and download
# ─────────────────────────────────────────────────────────────────────────────

def _discover_delisted_stocks(provider, start_date: str, end_date: str) -> pd.DataFrame:
    """Query WRDS for all delisted common stocks on major exchanges.

    Returns a DataFrame with columns: permno, ticker, dlstdt, dlret, dlstcd, comnam.

    Queries crsp.dsedelist directly, filtered to:
      - Common shares (shrcd 10, 11)
      - Major exchanges (NYSE=1, AMEX=2, NASDAQ=3)
      - Meaningful delisting codes (>= 200; codes 100-199 are still active)
    """
    sql = f"""
        SELECT DISTINCT
            d.permno,
            n.ticker,
            d.dlstdt,
            d.dlret,
            d.dlstcd,
            n.comnam
        FROM crsp.dsedelist AS d
        JOIN crsp.msenames AS n
          ON d.permno = n.permno
         AND n.namedt <= d.dlstdt
         AND (n.nameendt >= d.dlstdt OR n.nameendt IS NULL)
         AND n.shrcd IN (10, 11)
         AND n.exchcd IN (1, 2, 3)
        WHERE d.dlstdt >= '{start_date}'::date
          AND d.dlstdt <= '{end_date}'::date
          AND d.dlstcd >= 200
        ORDER BY d.dlstdt
    """
    return provider._query(sql)


def _filter_terminal_delists(
    provider,
    delist_df: pd.DataFrame,
    grace_days: int = 5,
) -> pd.DataFrame:
    """Keep only delistings where the PERMNO truly stops trading after dlstdt.

    Rejects cases where a stock has a delisting event but continues trading
    afterward (e.g., exchange transfers coded as delistings).
    """
    if delist_df.empty:
        return delist_df

    permnos = _sanitize_permno_in_clause(delist_df["permno"].tolist())
    if not permnos:
        return delist_df.iloc[0:0].copy()

    # Query last trade date for each PERMNO in batches
    batch_size = 800
    permno_list = permnos.split(",")
    frames = []
    for i in range(0, len(permno_list), batch_size):
        batch = ",".join(permno_list[i:i + batch_size])
        sql = f"""
            SELECT a.permno, MAX(a.date) AS last_trade_date
            FROM crsp.dsf AS a
            WHERE a.permno IN ({batch})
            GROUP BY a.permno
        """
        f = provider._query(sql)
        if not f.empty:
            frames.append(f)

    if not frames:
        return delist_df

    last_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["permno"], keep="last")
    out = delist_df.copy()
    out["dlstdt"] = pd.to_datetime(out["dlstdt"], errors="coerce")
    last_df["last_trade_date"] = pd.to_datetime(last_df["last_trade_date"], errors="coerce")
    out = out.merge(last_df[["permno", "last_trade_date"]], on="permno", how="left")

    cutoff = out["dlstdt"] + pd.to_timedelta(int(grace_days), unit="D")
    terminal_mask = out["last_trade_date"].isna() | (out["last_trade_date"] <= cutoff)
    filtered = out.loc[terminal_mask].copy()

    n_rejected = len(out) - len(filtered)
    if n_rejected > 0:
        print(f"  Filtered out {n_rejected} non-terminal delistings "
              f"(stock kept trading after dlstdt + {grace_days}d)")

    return filtered.drop(columns=["last_trade_date"], errors="ignore")


def _run_delisted_download(args):
    """Discover and download all historically delisted stocks from WRDS CRSP."""
    start_date = f"{datetime.now().year - args.years}-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

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

    # Discover delisted stocks
    print(f"\n── Discovering delisted stocks ({start_date} → {end_date}) ──")
    print(f"  Querying crsp.dsedelist for all delisted common stocks "
          f"on NYSE/AMEX/NASDAQ...")
    delist_df = _discover_delisted_stocks(provider, start_date, end_date)

    if delist_df.empty:
        print("  No delisted stocks found in WRDS.")
        return

    print(f"  Raw delisting events found: {len(delist_df)}")

    # Filter to truly terminal delistings (stock stops trading)
    print(f"  Validating terminal status (checking last trade dates)...")
    delist_df = _filter_terminal_delists(
        provider, delist_df, grace_days=args.terminal_grace_days,
    )
    if delist_df.empty:
        print("  No terminal delisted stocks found after validation.")
        return

    # De-duplicate: one row per permno (keep the latest delisting event)
    delist_df['dlstdt'] = pd.to_datetime(delist_df['dlstdt'])
    delist_df = delist_df.sort_values('dlstdt').drop_duplicates(
        subset=['permno'], keep='last',
    )

    # Filter out stocks we already have as terminal in cache
    already_terminal = []
    to_download = []
    for _, row in delist_df.iterrows():
        ticker_val = (str(row['ticker']).strip().upper()
                      if pd.notna(row.get('ticker')) else None)
        cache_symbol = _delisted_cache_symbol(ticker_val, row["permno"])
        _, meta, _ = load_ohlcv_with_meta(cache_symbol)
        if meta and (meta.get("is_terminal") is True
                     or meta.get("is_delisted_download") is True):
            already_terminal.append(cache_symbol)
            continue
        to_download.append(row)

    to_download_df = pd.DataFrame(to_download)
    print(f"  Terminal delisted stocks: {len(delist_df)}")
    print(f"  Already in cache: {len(already_terminal)}")
    print(f"  To download: {len(to_download_df)}")

    if to_download_df.empty:
        print("  Nothing to download — all delisted stocks already cached.")
        return

    # Show what we found (first 50)
    show_limit = 50
    print(f"\n  Delisted stocks to download (showing first {show_limit}):")
    for i, (_, row) in enumerate(to_download_df.iterrows()):
        if i >= show_limit:
            print(f"    ... and {len(to_download_df) - show_limit} more")
            break
        ticker_val = str(row['ticker']).strip() if pd.notna(row.get('ticker')) else '???'
        comnam = str(row['comnam']).strip() if pd.notna(row.get('comnam')) else ''
        dlstdt = row['dlstdt']
        dlstcd = int(row['dlstcd']) if pd.notna(row.get('dlstcd')) else '?'
        dlret = float(row['dlret']) if pd.notna(row.get('dlret')) else 0.0
        reason = _crsp_dlstcd_to_reason(dlstcd)
        print(f"    {ticker_val:<8s} permno={int(row['permno']):>6d}  "
              f"delisted={str(dlstdt.date()):>10s}  dlret={dlret:+.4f}  "
              f"reason={reason.value:<20s} {comnam}")

    if args.dry_run:
        print(f"\n  DRY RUN — would download {len(to_download_df)} delisted stocks.")
        return

    # Download prices by PERMNO (tickers get recycled, permnos are permanent)
    delist_handler = DelistingHandler(db_path=str(SURVIVORSHIP_DB))
    batch_size = args.batch_size
    t0 = time.time()

    downloaded = []
    failed = []
    not_found = []
    delisted_recorded = []

    # Build permno -> row mapping
    permno_to_row = {}
    for _, row in to_download_df.iterrows():
        permno_to_row[str(int(row['permno']))] = row

    permno_list = list(permno_to_row.keys())

    print(f"\n── Downloading {len(permno_list)} delisted stocks by PERMNO ──")

    for batch_start in range(0, len(permno_list), batch_size):
        batch = permno_list[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(permno_list) + batch_size - 1) // batch_size
        print(f"\n  Batch {batch_num}/{total_batches}: {len(batch)} permnos")

        try:
            result = provider.get_crsp_prices_with_delistings(
                batch, start_date=start_date, end_date=end_date,
            )
        except (OSError, ValueError, RuntimeError) as e:
            print(f"    Batch failed ({type(e).__name__}: {e}), retrying individually...")
            result = {}
            for p in batch:
                try:
                    sub_result = provider.get_crsp_prices_with_delistings(
                        [p], start_date=start_date, end_date=end_date,
                    )
                    result.update(sub_result)
                except (OSError, ValueError, RuntimeError) as e2:
                    row_info = permno_to_row.get(p, {})
                    ticker_val = (str(row_info.get('ticker', p)).strip()
                                  if hasattr(row_info, 'get') else p)
                    print(f"    permno {p} ({ticker_val}): failed — {e2}")
                    failed.append((ticker_val, str(e2)))

        # Process results
        for permno_key, df in result.items():
            if len(df) == 0:
                continue

            # Get ticker from the discovery query (authoritative)
            row_info = permno_to_row.get(permno_key)
            if row_info is not None:
                ticker_val = (str(row_info['ticker']).strip().upper()
                              if pd.notna(row_info.get('ticker')) else None)
            else:
                ticker_val = None

            # Fallback: get ticker from the price data itself
            if not ticker_val and "ticker" in df.columns:
                ticker_val = (str(df["ticker"].dropna().iloc[-1]).strip().upper()
                              if df["ticker"].notna().any() else None)

            cache_symbol = _delisted_cache_symbol(ticker_val, permno_key)

            o_ne_c = ((df["Open"] != df["Close"]).sum()
                       if "Open" in df.columns and "Close" in df.columns else 0)
            pct = 100.0 * o_ne_c / len(df) if len(df) > 0 else 0
            start = str(pd.to_datetime(df.index.min()).date()) if len(df) > 0 else "?"
            end = str(pd.to_datetime(df.index.max()).date()) if len(df) > 0 else "?"

            try:
                save_ohlcv(
                    cache_symbol, df,
                    source="wrds_delisting",
                    meta={"permno": permno_key, "ticker": ticker_val,
                          "cache_symbol": cache_symbol,
                          "years_requested": args.years,
                          "is_delisted_download": True,
                          "is_terminal": True},
                )
                downloaded.append(cache_symbol)
                ticker_print = ticker_val or "N/A"
                print(f"    {cache_symbol}: {len(df)} bars, "
                      f"{start} → {end}, O!=C {pct:.0f}%")

                # Record delisting event
                if row_info is not None:
                    dlret_val = (float(row_info['dlret'])
                                 if pd.notna(row_info.get('dlret')) else 0.0)
                    dlstcd_val = (int(row_info['dlstcd'])
                                  if pd.notna(row_info.get('dlstcd')) else None)
                    last_price = (float(df["Close"].iloc[-1])
                                  if "Close" in df.columns and len(df) > 0 else 0.0)
                    delist_date = (row_info['dlstdt'].date()
                                   if hasattr(row_info['dlstdt'], 'date')
                                   else row_info['dlstdt'])
                    reason = _crsp_dlstcd_to_reason(dlstcd_val)
                    meta_dict = {"permno": permno_key}
                    if dlstcd_val is not None:
                        meta_dict["dlstcd"] = dlstcd_val
                    if ticker_val:
                        meta_dict["ticker"] = ticker_val
                    event = DelistingEvent(
                        symbol=cache_symbol,
                        delisting_date=delist_date,
                        reason=reason,
                        last_price=last_price,
                        delisting_return=dlret_val,
                        metadata=meta_dict,
                    )
                    delist_handler.record_delisting(event)
                    delist_handler.preserve_price_history(cache_symbol, df)
                    delisted_recorded.append(cache_symbol)
                    print(f"      ↳ DELISTED {delist_date} dlret={dlret_val:+.4f} "
                          f"reason={reason.value} dlstcd={dlstcd_val}")

            except (ValueError, OSError, RuntimeError) as e:
                fail_key = ticker_val or cache_symbol
                print(f"    {fail_key}: save failed — {e}")
                failed.append((fail_key, str(e)))

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*60}")
    print(f"DELISTED DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"  Delisted stocks discovered: {len(delist_df)}")
    print(f"  Already terminal in cache: {len(already_terminal)}")
    print(f"  Download attempted: {len(to_download_df)}")
    print(f"  Successfully downloaded: {len(downloaded)}")
    if downloaded:
        print(f"    {', '.join(downloaded[:20])}{'...' if len(downloaded) > 20 else ''}")
    print(f"  Not found / no data: {len(not_found)}")
    if not_found:
        print(f"    {', '.join(str(x) for x in not_found)}")
    print(f"  Failed: {len(failed)}")
    if failed:
        for t, err in failed[:10]:
            print(f"    {t}: {err}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")
    print(f"  Delisting events recorded: {len(delisted_recorded)}")
    print(f"  Elapsed: {elapsed:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Main: normal refresh workflow
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Run the WRDS daily refresh workflow."""
    parser = argparse.ArgumentParser(
        description="Re-download daily OHLCV from WRDS CRSP",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be downloaded")
    parser.add_argument("--skip-cleanup", action="store_true",
                        help="Download new files but keep old _daily_ files")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated ticker list (default: all)")
    parser.add_argument("--years", type=int, default=20,
                        help="Lookback years (default: 20)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Tickers per WRDS query (default: 50)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing _1d.parquet files")
    parser.add_argument("--gics", action="store_true",
                        help="Refresh GICS sector mapping from WRDS Compustat")
    parser.add_argument("--include-terminal", action="store_true",
                        help="Include delisted/terminal tickers in refresh")
    parser.add_argument("--backfill-terminal", action="store_true",
                        help="Scan cache and mark delisted stocks as terminal")
    parser.add_argument("--delisted", action="store_true",
                        help="Discover and download ALL delisted stocks from WRDS CRSP")
    parser.add_argument("--terminal-grace-days", type=int, default=5,
                        help="Grace days for terminal filter (default: 5)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"WRDS DAILY DATA RE-DOWNLOAD")
    print(f"{'='*60}")

    # Handle --backfill-terminal
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

    # Handle --delisted
    if args.delisted:
        _run_delisted_download(args)
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
            sys.exit(1)

        print(f"  Querying Compustat for GICS sector codes...")
        sector_map = refresh_gics_sectors(provider, tickers)
        print(f"  Found GICS sectors for {len(sector_map)} / {len(tickers)} tickers")

        if sector_map:
            yaml_path = Path(__file__).parent / "config_data" / "universe.yaml"
            _write_gics_to_universe_yaml(sector_map, yaml_path)
            print(f"  Updated {yaml_path}")

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

    # Initialize DelistingHandler
    delist_handler = DelistingHandler(db_path=str(SURVIVORSHIP_DB))

    # Download in batches
    start_date = f"{datetime.now().year - args.years}-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    print(f"\n── Downloading ({start_date} → {end_date}) ──")

    downloaded = []
    failed = []
    not_found = []
    delisted_recorded = []
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
            print(f"    Batch failed ({type(e).__name__}: {e}), retrying...")
            result = {}
            for sub_start in range(0, len(batch), retry_batch_size):
                sub_batch = batch[sub_start:sub_start + retry_batch_size]
                try:
                    sub_result = provider.get_crsp_prices_with_delistings(
                        sub_batch, start_date=start_date, end_date=end_date,
                    )
                    result.update(sub_result)
                except (OSError, ValueError, RuntimeError) as e2:
                    for t in sub_batch:
                        failed.append((t, str(e2)))

        # Map permno results back to tickers
        ticker_results = {}
        for permno_key, df in result.items():
            if len(df) == 0:
                continue
            if "ticker" in df.columns:
                ticker_val = str(df["ticker"].iloc[-1]).strip().upper()
            else:
                ticker_val = None

            if ticker_val and ticker_val in [t.upper() for t in batch]:
                if (ticker_val not in ticker_results
                        or len(df) > len(ticker_results[ticker_val][1])):
                    ticker_results[ticker_val] = (permno_key, df)

        found_tickers = set(ticker_results.keys())
        for t in batch:
            if t.upper() not in found_tickers:
                not_found.append(t)

        # Save each ticker's data
        for ticker_val, (permno_key, df) in ticker_results.items():
            o_ne_c = ((df["Open"] != df["Close"]).sum()
                       if "Open" in df.columns and "Close" in df.columns else 0)
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

                # Persist delisting event
                if "delist_event" in df.columns:
                    delist_col = pd.to_numeric(
                        df["delist_event"], errors="coerce",
                    ).fillna(0)
                    if int(delist_col.max()) == 1:
                        delist_rows = df[delist_col == 1]
                        delist_idx = delist_rows.index.max()
                        delist_row = df.loc[delist_idx]
                        dlret_val = float(delist_row.get("dlret", 0) or 0)
                        dlstcd_val = delist_row.get("dlstcd", None)
                        if pd.isna(dlstcd_val):
                            dlstcd_val = None
                        last_price = float(delist_row.get("Close", 0) or 0)
                        delist_date = pd.to_datetime(delist_idx).date()
                        reason = _crsp_dlstcd_to_reason(dlstcd_val)
                        meta_dict = {"permno": permno_key}
                        if dlstcd_val is not None:
                            meta_dict["dlstcd"] = int(dlstcd_val)
                        event = DelistingEvent(
                            symbol=ticker_val,
                            delisting_date=delist_date,
                            reason=reason,
                            last_price=last_price,
                            delisting_return=dlret_val,
                            metadata=meta_dict,
                        )
                        delist_handler.record_delisting(event)
                        delist_handler.preserve_price_history(ticker_val, df)
                        delisted_recorded.append(ticker_val)
                        print(f"      ↳ DELISTED on {delist_date} "
                              f"(dlret={dlret_val:.4f}, reason={reason.value}, "
                              f"dlstcd={dlstcd_val})")
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
    print(f"  Delisting events recorded: {len(delisted_recorded)}")
    if delisted_recorded:
        print(f"    {', '.join(delisted_recorded)}")
    print(f"  Old daily files removed: {removed}")
    print(f"  New _1d.parquet files created: {len(downloaded)}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"\n  Features unlocked: ~15 (ATR, NATR, Parkinson Vol, GK Vol, YZ Vol,")
    print(f"    Stochastic, Williams %R, CCI, Candlestick Body, Gap %, etc.)")

    # ── Write reproducibility manifest ──
    try:
        from quant_engine.reproducibility import build_run_manifest, write_run_manifest
        from quant_engine.config import RESULTS_DIR as _RESULTS_DIR
        manifest = build_run_manifest(
            run_type="wrds_daily_refresh",
            config_snapshot=vars(args),
            script_name="run_wrds_daily_refresh",
            extra={
                "tickers_refreshed": len(downloaded),
                "refresh_date": datetime.now().isoformat(),
            },
        )
        write_run_manifest(manifest, output_dir=_RESULTS_DIR)
    except Exception as e:
        logger.warning("Could not write reproducibility manifest: %s", e)


if __name__ == "__main__":
    main()
