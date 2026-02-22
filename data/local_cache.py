"""
Local data cache for daily OHLCV data.

Primary storage is parquet when available, with a CSV fallback path so the
engine can run in minimal environments without pyarrow/fastparquet.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

from ..config import DATA_CACHE_DIR, FRAMEWORK_DIR

REQUIRED_OHLCV = ["Open", "High", "Low", "Close", "Volume"]
OPTIONAL_CACHE_COLUMNS = ["Return", "total_ret", "dlret", "delist_event", "permno", "ticker"]
DATE_COLUMNS = ["Date", "date", "Datetime", "datetime", "Timestamp", "timestamp"]
FALLBACK_SOURCE_DIRS = [
    FRAMEWORK_DIR / "data_cache",
    FRAMEWORK_DIR / "automated_portfolio_system" / "data_cache",
]


def _ensure_cache_dir() -> Path:
    """Create cache directory if it doesn't exist."""
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_CACHE_DIR


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV column names to quant_engine's canonical schema."""
    column_map = {}
    used = set()
    for col in df.columns:
        col_str = str(col)
        col_l = col_str.lower()
        target = None
        if "open" in col_l:
            target = "Open"
        elif "high" in col_l:
            target = "High"
        elif "low" in col_l:
            target = "Low"
        elif "close" in col_l and "adj" not in col_l:
            target = "Close"
        elif "volume" in col_l:
            target = "Volume"
        elif col_l in {"return", "ret"}:
            target = "Return"
        elif col_l in {"total_ret", "totalreturn", "total_return"}:
            target = "total_ret"
        elif col_l == "dlret":
            target = "dlret"
        elif col_l in {"delist_event", "is_delist_event"}:
            target = "delist_event"
        elif col_l == "permno":
            target = "permno"
        elif col_l in {"ticker", "tic"}:
            target = "ticker"
        if target and target not in used:
            column_map[col] = target
            used.add(target)
    out = df.rename(columns=column_map)
    return out


def _to_daily_ohlcv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Convert any candidate frame into validated daily OHLCV."""
    if df is None or len(df) == 0:
        return None
    out = _normalize_ohlcv_columns(df)
    if not set(REQUIRED_OHLCV).issubset(out.columns):
        return None
    extra_cols = [c for c in OPTIONAL_CACHE_COLUMNS if c in out.columns]
    out = out[REQUIRED_OHLCV + [c for c in extra_cols if c not in REQUIRED_OHLCV]].copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce"))
    out = out.sort_index()
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")]
    if len(out) > 2:
        step = pd.Series(out.index).diff().median()
        if pd.notna(step) and step < pd.Timedelta(hours=12):
            return None
    out = out.dropna(subset=REQUIRED_OHLCV)
    return out if len(out) > 0 else None


def _read_csv_ohlcv(path: Path) -> Optional[pd.DataFrame]:
    """Internal helper to read csv ohlcv from storage."""
    try:
        raw = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as e:
        logger.debug("Could not read CSV %s: %s", path.name, e)
        return None
    date_col = next((c for c in DATE_COLUMNS if c in raw.columns), None)
    if date_col is None:
        if len(raw.columns) == 0:
            return None
        date_col = raw.columns[0]
    raw.loc[:, date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col])
    raw.index = pd.DatetimeIndex(raw.pop(date_col))
    return _to_daily_ohlcv(raw)


def _candidate_csv_paths(cache_root: Path, ticker: str) -> List[Path]:
    """Internal helper for candidate csv paths."""
    t = ticker.upper()
    candidates = [cache_root / f"{t}_1d.csv", cache_root / f"{t}.csv"]
    candidates.extend(sorted(cache_root.glob(f"{t}_daily_*.csv")))
    candidates.extend(sorted(cache_root.glob(f"{t}_20*.csv")))
    # Keep order stable while removing duplicates.
    unique = []
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


def _cache_meta_path(data_path: Path, ticker: str) -> Path:
    """Return the metadata sidecar path for a cache data file.

    Canonical daily files ({TICKER}_1d.*) and IBKR-style daily files
    ({TICKER}_daily_*.*) both map to the single canonical sidecar
    ``{TICKER}_1d.meta.json`` so that lookup by ticker always finds
    the same metadata regardless of the underlying filename variant.

    Intraday files get file-adjacent metadata to avoid collisions when
    multiple timeframes coexist for the same ticker.
    """
    canonical = data_path.parent / f"{ticker.upper()}_1d.meta.json"
    # Intraday files (4hour, 1hour, 30min, etc.) — file-adjacent meta
    for tf_word in ("_4HOUR_", "_1HOUR_", "_30MIN_", "_15MIN_", "_5MIN_", "_1MIN_"):
        if tf_word in data_path.name.upper():
            return data_path.with_suffix(".meta.json")
    # All daily variants (canonical _1d.* and IBKR _daily_*) share the
    # single canonical sidecar so metadata is always discoverable.
    return canonical


def _read_cache_meta(data_path: Path, ticker: str) -> Dict[str, object]:
    """Internal helper to read cache meta from storage."""
    candidates = [
        _cache_meta_path(data_path, ticker),
        data_path.with_suffix(".meta.json"),
    ]
    for meta_path in candidates:
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                return raw
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
    return {}


def _write_cache_meta(
    data_path: Path,
    ticker: str,
    df: pd.DataFrame,
    source: str,
    meta: Optional[Dict[str, object]] = None,
):
    """Internal helper to write cache meta to storage."""
    payload: Dict[str, object] = {
        "ticker": ticker.upper(),
        "source": str(source).lower().strip() or "unknown",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_date": str(pd.to_datetime(df.index.min()).date()) if len(df) > 0 else None,
        "end_date": str(pd.to_datetime(df.index.max()).date()) if len(df) > 0 else None,
        "n_bars": int(len(df)),
        "format": data_path.suffix.lstrip(".").lower(),
    }
    if meta:
        payload.update(meta)
    meta_path = _cache_meta_path(data_path, ticker)
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except OSError as e:
        logger.debug("Could not write cache meta for %s: %s", ticker, e)


def save_ohlcv(
    ticker: str,
    df: pd.DataFrame,
    cache_dir: Optional[Path] = None,
    source: str = "unknown",
    meta: Optional[Dict[str, object]] = None,
) -> Path:
    """
    Save OHLCV DataFrame to local cache.

    Args:
        ticker: Stock ticker symbol
        df: DataFrame with OHLCV columns and DatetimeIndex
        cache_dir: Override cache directory (default: DATA_CACHE_DIR)

    Returns:
        Path to saved file
    """
    d = Path(cache_dir) if cache_dir else _ensure_cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    daily = _to_daily_ohlcv(df)
    if daily is None:
        raise ValueError(f"Invalid OHLCV data for {ticker}")
    parquet_path = d / f"{ticker.upper()}_1d.parquet"
    try:
        daily.to_parquet(parquet_path)
        _write_cache_meta(parquet_path, ticker=ticker, df=daily, source=source, meta=meta)
        return parquet_path
    except (ImportError, OSError) as e:
        logger.debug("Parquet write failed for %s, falling back to CSV: %s", ticker, e)
        csv_path = d / f"{ticker.upper()}_1d.csv"
        daily.to_csv(csv_path, index=True)
        _write_cache_meta(csv_path, ticker=ticker, df=daily, source=source, meta=meta)
        return csv_path


def load_ohlcv_with_meta(
    ticker: str,
    cache_dir: Optional[Path] = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, object], Optional[Path]]:
    """
    Load OHLCV and sidecar metadata from cache roots.

    Returns (dataframe, metadata, source_path).
    """
    d = Path(cache_dir) if cache_dir else DATA_CACHE_DIR
    roots = [d] + [p for p in FALLBACK_SOURCE_DIRS if p != d]
    target = str(ticker).upper().strip()
    for root in roots:
        parquet_path = root / f"{ticker.upper()}_1d.parquet"
        if parquet_path.exists():
            try:
                raw = pd.read_parquet(parquet_path)
                normalized = _to_daily_ohlcv(raw)
                if normalized is not None:
                    meta = _read_cache_meta(parquet_path, ticker)
                    return normalized, meta, parquet_path
            except (OSError, ValueError, ImportError) as e:
                logger.debug("Could not read parquet %s: %s", parquet_path.name, e)
        # Glob for IBKR-style daily parquet: {TICKER}_daily_{start}_{end}.parquet
        for parquet_daily in sorted(root.glob(f"{ticker.upper()}_daily_*.parquet")):
            try:
                raw = pd.read_parquet(parquet_daily)
                normalized = _to_daily_ohlcv(raw)
                if normalized is not None:
                    meta = _read_cache_meta(parquet_daily, ticker)
                    return normalized, meta, parquet_daily
            except (OSError, ValueError, ImportError) as e:
                logger.debug("Could not read parquet %s: %s", parquet_daily.name, e)
        for csv_path in _candidate_csv_paths(root, ticker):
            if not csv_path.exists():
                continue
            normalized = _read_csv_ohlcv(csv_path)
            if normalized is not None:
                meta = _read_cache_meta(csv_path, ticker)
                return normalized, meta, csv_path

        # Alias lookup: if files are keyed by PERMNO, resolve by metadata ticker/permno.
        for meta_path in root.glob("*_1d.meta.json"):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(meta, dict):
                continue
            meta_ticker = str(meta.get("ticker", "")).upper().strip()
            meta_permno = str(meta.get("permno", "")).upper().strip()
            if target not in {meta_ticker, meta_permno}:
                continue

            stem = meta_path.stem.replace(".meta", "")
            parquet_alias = root / f"{stem}.parquet"
            csv_alias = root / f"{stem}.csv"
            candidate_path = parquet_alias if parquet_alias.exists() else csv_alias
            if not candidate_path.exists():
                continue
            if candidate_path.suffix.lower() == ".parquet":
                try:
                    raw = pd.read_parquet(candidate_path)
                    normalized = _to_daily_ohlcv(raw)
                except (OSError, ValueError, ImportError):
                    normalized = None
            else:
                normalized = _read_csv_ohlcv(candidate_path)
            if normalized is not None:
                return normalized, meta, candidate_path
    return None, {}, None


def load_ohlcv(ticker: str, cache_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """
    Load OHLCV DataFrame from local cache.

    Returns None if not cached.
    """
    df, _, _ = load_ohlcv_with_meta(ticker=ticker, cache_dir=cache_dir)
    return df


_IBKR_TIMEFRAME_MAP = {
    "4h": "4hour", "1h": "1hour", "30m": "30min",
    "15m": "15min", "5m": "5min", "1m": "1min",
}


def load_intraday_ohlcv(
    ticker: str,
    timeframe: str,
    cache_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """
    Load intraday OHLCV data from cache.

    Handles IBKR naming: {TICKER}_{timeword}_{start}_{end}.parquet
    Does NOT apply _to_daily_ohlcv() (which rejects sub-daily data).

    Args:
        ticker: Stock symbol (e.g. "AAPL")
        timeframe: Canonical code ("4h", "1h", "30m", "15m", "5m", "1m")
        cache_dir: Override cache directory

    Returns:
        DataFrame with DatetimeIndex and OHLCV columns, or None
    """
    d = Path(cache_dir) if cache_dir else DATA_CACHE_DIR
    t = ticker.upper()
    ibkr_tf = _IBKR_TIMEFRAME_MAP.get(timeframe, timeframe)

    # Try parquet first (faster, preserves dtypes)
    for pattern in [f"{t}_{ibkr_tf}_*.parquet", f"{t}_{timeframe}_*.parquet",
                    f"{t}_{timeframe}.parquet"]:
        matches = sorted(d.glob(pattern))
        for path in matches:
            try:
                df = pd.read_parquet(path)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.DatetimeIndex(pd.to_datetime(df.index, errors="coerce"))
                df = _normalize_ohlcv_columns(df)
                if set(REQUIRED_OHLCV).issubset(df.columns):
                    df = df[REQUIRED_OHLCV].copy()
                    df = df.sort_index()
                    df = df[~df.index.isna()]
                    df = df[~df.index.duplicated(keep="last")]
                    df = df.dropna(subset=REQUIRED_OHLCV)
                    if len(df) > 0:
                        return df
            except (OSError, ValueError, ImportError) as e:
                logger.debug("Could not read intraday parquet %s: %s", path.name, e)
                continue

    # CSV fallback
    for pattern in [f"{t}_{ibkr_tf}_*.csv", f"{t}_{timeframe}_*.csv",
                    f"{t}_{timeframe}.csv"]:
        matches = sorted(d.glob(pattern))
        for path in matches:
            try:
                df = pd.read_csv(path, parse_dates=True, index_col=0)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.DatetimeIndex(pd.to_datetime(df.index, errors="coerce"))
                df = _normalize_ohlcv_columns(df)
                if set(REQUIRED_OHLCV).issubset(df.columns):
                    df = df[REQUIRED_OHLCV].copy()
                    df = df.sort_index()
                    df = df[~df.index.isna()]
                    df = df[~df.index.duplicated(keep="last")]
                    df = df.dropna(subset=REQUIRED_OHLCV)
                    if len(df) > 0:
                        return df
            except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
                logger.debug("Could not read intraday CSV %s: %s", path.name, e)
                continue
    return None


def list_intraday_timeframes(
    ticker: str,
    cache_dir: Optional[Path] = None,
) -> List[str]:
    """Return list of available intraday timeframes for a ticker in the cache."""
    d = Path(cache_dir) if cache_dir else DATA_CACHE_DIR
    t = ticker.upper()
    found = []
    for canonical, ibkr_name in _IBKR_TIMEFRAME_MAP.items():
        if list(d.glob(f"{t}_{ibkr_name}_*.*")):
            found.append(canonical)
    return found


def list_cached_tickers(cache_dir: Optional[Path] = None) -> List[str]:
    """List all tickers available in cache roots."""
    d = Path(cache_dir) if cache_dir else DATA_CACHE_DIR
    roots = [d] + [p for p in FALLBACK_SOURCE_DIRS if p != d]
    tickers = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.glob("*_1d.parquet"):
            tickers.add(p.stem.replace("_1d", "").upper())
        for p in root.glob("*_1d.csv"):
            tickers.add(p.stem.replace("_1d", "").upper())
        for p in root.glob("*_daily_*.parquet"):
            tickers.add(p.stem.split("_daily_")[0].upper())
        for p in root.glob("*_daily_*.csv"):
            tickers.add(p.stem.split("_daily_")[0].upper())
    return sorted(tickers)


def _daily_cache_files(root: Path) -> List[Path]:
    """Return de-duplicated daily-cache candidate files for one root."""
    candidates: List[Path] = []
    candidates.extend(sorted(root.glob("*_1d.parquet")))
    candidates.extend(sorted(root.glob("*_daily_*.parquet")))
    candidates.extend(sorted(root.glob("*_1d.csv")))
    candidates.extend(sorted(root.glob("*_daily_*.csv")))
    unique: List[Path] = []
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


_INTRADAY_TIMEWORD_TO_CANONICAL = {
    "4HOUR": "4h", "1HOUR": "1h", "30MIN": "30m",
    "15MIN": "15m", "5MIN": "5m", "1MIN": "1m",
}


def _ticker_from_cache_path(path: Path) -> str:
    """Internal helper for ticker from cache path."""
    stem = path.stem.upper()
    if "_DAILY_" in stem:
        return stem.split("_DAILY_")[0]
    for tf_word in _INTRADAY_TIMEWORD_TO_CANONICAL:
        if f"_{tf_word}_" in stem:
            return stem.split(f"_{tf_word}_")[0]
    if stem.endswith("_1D"):
        return stem[:-3]
    if "_" not in stem:
        return stem
    return stem.split("_")[0]


def _timeframe_from_cache_path(path: Path) -> str:
    """Determine the canonical timeframe from a cache file path."""
    stem = path.stem.upper()
    if "_DAILY_" in stem or stem.endswith("_1D"):
        return "daily"
    for tf_word, canonical in _INTRADAY_TIMEWORD_TO_CANONICAL.items():
        if f"_{tf_word}_" in stem:
            return canonical
    return "daily"


def _all_cache_files(root: Path) -> List[Path]:
    """Return de-duplicated daily + intraday cache candidate files for one root."""
    candidates: List[Path] = []
    # Daily patterns (parquet preferred)
    candidates.extend(sorted(root.glob("*_1d.parquet")))
    candidates.extend(sorted(root.glob("*_daily_*.parquet")))
    candidates.extend(sorted(root.glob("*_1d.csv")))
    candidates.extend(sorted(root.glob("*_daily_*.csv")))
    # Intraday patterns
    for tf_word in ("4hour", "1hour", "30min", "15min", "5min", "1min"):
        candidates.extend(sorted(root.glob(f"*_{tf_word}_*.parquet")))
        candidates.extend(sorted(root.glob(f"*_{tf_word}_*.csv")))
    unique: List[Path] = []
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


def rehydrate_cache_metadata(
    cache_roots: Optional[List[Path]] = None,
    source_by_root: Optional[Mapping[str, str]] = None,
    default_source: str = "unknown",
    only_missing: bool = True,
    overwrite_source: bool = False,
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Backfill metadata sidecars for existing cache files without rewriting price data.

    Args:
        cache_roots: Roots to scan. Defaults to local cache + fallback roots.
        source_by_root: Optional mapping {"/path/to/root": "wrds|ibkr|..."}.
        default_source: Source label when a root has no explicit mapping.
        only_missing: If True, skip files that already have metadata.
        overwrite_source: If True, replace existing metadata source with mapped/default source.
        dry_run: If True, return counts without writing metadata files.

    Returns:
        Summary counters.
    """
    roots = [Path(p) for p in (cache_roots or [DATA_CACHE_DIR, *FALLBACK_SOURCE_DIRS])]
    root_source = {
        str(DATA_CACHE_DIR.resolve()): "wrds",
        str((FRAMEWORK_DIR / "automated_portfolio_system" / "data_cache").resolve()): "ibkr",
        str((FRAMEWORK_DIR / "data_cache").resolve()): "ibkr",
    }
    if source_by_root:
        for k, v in source_by_root.items():
            root_source[str(Path(k).expanduser().resolve())] = str(v).lower().strip()

    default_source_norm = str(default_source).lower().strip() or "unknown"
    summary = {
        "roots_seen": 0,
        "files_scanned": 0,
        "written": 0,
        "skipped_existing": 0,
        "skipped_unreadable": 0,
        "skipped_missing_root": 0,
    }

    reserved = {
        "ticker",
        "source",
        "saved_at_utc",
        "start_date",
        "end_date",
        "n_bars",
        "format",
    }

    for root in roots:
        if not root.exists():
            summary["skipped_missing_root"] += 1
            continue
        summary["roots_seen"] += 1
        root_key = str(root.expanduser().resolve())
        root_source_label = root_source.get(root_key, default_source_norm)

        for path in _all_cache_files(root):
            summary["files_scanned"] += 1
            ticker = _ticker_from_cache_path(path)
            if not ticker:
                summary["skipped_unreadable"] += 1
                continue

            meta_path = _cache_meta_path(path, ticker)
            if only_missing and meta_path.exists():
                summary["skipped_existing"] += 1
                continue

            timeframe = _timeframe_from_cache_path(path)
            is_intraday = timeframe != "daily"

            if path.suffix.lower() == ".parquet":
                try:
                    raw = pd.read_parquet(path)
                except (OSError, ValueError, ImportError) as e:
                    logger.debug("Could not read parquet %s: %s", path.name, e)
                    summary["skipped_unreadable"] += 1
                    continue
                if is_intraday:
                    # Skip _to_daily_ohlcv for intraday (it rejects sub-daily)
                    normalized = _normalize_ohlcv_columns(raw)
                    if not isinstance(normalized.index, pd.DatetimeIndex):
                        normalized.index = pd.DatetimeIndex(
                            pd.to_datetime(normalized.index, errors="coerce")
                        )
                    if not set(REQUIRED_OHLCV).issubset(normalized.columns):
                        normalized = None
                    elif len(normalized) == 0:
                        normalized = None
                else:
                    normalized = _to_daily_ohlcv(raw)
            else:
                if is_intraday:
                    try:
                        raw = pd.read_csv(path, parse_dates=True, index_col=0)
                    except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
                        logger.debug("Could not read CSV %s: %s", path.name, e)
                        summary["skipped_unreadable"] += 1
                        continue
                    normalized = _normalize_ohlcv_columns(raw)
                    if not isinstance(normalized.index, pd.DatetimeIndex):
                        normalized.index = pd.DatetimeIndex(
                            pd.to_datetime(normalized.index, errors="coerce")
                        )
                    if not set(REQUIRED_OHLCV).issubset(normalized.columns):
                        normalized = None
                    elif len(normalized) == 0:
                        normalized = None
                else:
                    normalized = _read_csv_ohlcv(path)

            if normalized is None:
                summary["skipped_unreadable"] += 1
                continue

            existing = _read_cache_meta(path, ticker)
            existing_source = str(existing.get("source", "")).lower().strip()
            # Intraday files are sourced from IBKR; daily from WRDS
            effective_root_source = "ibkr" if is_intraday else root_source_label
            source = (
                effective_root_source
                if overwrite_source or not existing_source
                else existing_source
            )

            meta_extras = {k: v for k, v in existing.items() if k not in reserved}
            meta_extras["rehydrated"] = True
            meta_extras["timeframe"] = timeframe

            if not dry_run:
                _write_cache_meta(
                    data_path=path,
                    ticker=ticker,
                    df=normalized,
                    source=source,
                    meta=meta_extras,
                )
            summary["written"] += 1

    return summary


def load_ibkr_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Scan a directory of IBKR-downloaded files (CSV or parquet).

    Expects files named like {TICKER}.csv, {TICKER}.parquet, or {TICKER}_1d.parquet.
    Returns {ticker: DataFrame} dict.
    """
    data = {}
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return data

    for path in sorted(data_dir.iterdir()):
        if path.suffix not in (".csv", ".parquet"):
            continue

        ticker = path.stem.upper().replace("_1D", "").replace("_1d", "")

        try:
            if path.suffix == ".csv":
                df = pd.read_csv(path, parse_dates=True, index_col=0)
            else:
                df = pd.read_parquet(path)

            # Ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.DatetimeIndex(pd.to_datetime(df.index))

            # Normalize column names
            col_map = {}
            for c in df.columns:
                cl = c.lower()
                if "open" in cl and "open" not in [v.lower() for v in col_map.values()]:
                    col_map[c] = "Open"
                elif "high" in cl and "high" not in [v.lower() for v in col_map.values()]:
                    col_map[c] = "High"
                elif "low" in cl and "low" not in [v.lower() for v in col_map.values()]:
                    col_map[c] = "Low"
                elif "close" in cl and "adj" not in cl and "close" not in [v.lower() for v in col_map.values()]:
                    col_map[c] = "Close"
                elif "volume" in cl and "volume" not in [v.lower() for v in col_map.values()]:
                    col_map[c] = "Volume"
            if col_map:
                df = df.rename(columns=col_map)

            if set(REQUIRED_OHLCV).issubset(df.columns):
                data[ticker] = df[REQUIRED_OHLCV].dropna()

        except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as e:
            logger.debug("Could not load IBKR file %s: %s", path.name, e)
            continue

    return data


def cache_universe(
    data: Dict[str, pd.DataFrame],
    cache_dir: Optional[Path] = None,
    source: str = "ibkr",
):
    """Save all tickers in a data dict to the local cache."""
    for ticker, df in data.items():
        save_ohlcv(ticker, df, cache_dir=cache_dir, source=source)
