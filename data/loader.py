"""
Data loader — self-contained data loading with multiple sources.

Priority order:
    1. Trusted local cache (WRDS/IBKR provenance + validation)
    2. WRDS CRSP (authoritative source)
    3. Validated local cache fallback

No external sys.path dependencies.
"""
import logging
import re
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import pandas_market_calendars as mcal
    _NYSE_CAL = mcal.get_calendar("NYSE")
except ImportError:
    mcal = None
    _NYSE_CAL = None

logger = logging.getLogger(__name__)

# Module-level tracker recording all data source fallbacks for provenance auditing.
_fallback_tracker: Dict[str, Dict[str, object]] = {}

# Module-level tracker recording per-ticker skip reasons from load_universe().
# Cleared at the start of each load_universe() call.
_skip_tracker: Dict[str, str] = {}

from ..config import (
    CACHE_MAX_STALENESS_DAYS,
    CACHE_TRUSTED_SOURCES,
    CACHE_WRDS_SPAN_ADVANTAGE_DAYS,
    DATA_QUALITY_ENABLED,
    LOOKBACK_YEARS,
    MIN_BARS,
    OPTIONMETRICS_ENABLED,
    REQUIRE_PERMNO,
    TRUTH_LAYER_FAIL_ON_CORRUPT,
    WRDS_ENABLED,
)
from .local_cache import (
    list_cached_tickers,
    load_ohlcv_with_meta as cache_load_with_meta,
    save_ohlcv as cache_save,
)
from .provider_registry import get_provider
from .quality import assess_ohlcv_quality

REQUIRED_OHLCV = ["Open", "High", "Low", "Close", "Volume"]
OPTIONAL_PANEL_COLUMNS = ["Return", "total_ret", "dlret", "delist_event", "permno", "ticker"]
TRUSTED_CACHE_SOURCES = {str(s).lower().strip() for s in CACHE_TRUSTED_SOURCES}

# Ticker validation — consistent with wrds_provider._TICKER_RE.
# Prevents path traversal and rejects empty / overly long inputs.
_TICKER_RE = re.compile(r'^[A-Z0-9.\-/^]{1,12}$')


def _permno_from_meta(meta: Optional[Dict[str, object]]) -> Optional[str]:
    """Internal helper for permno from meta."""
    if not meta:
        return None
    value = meta.get("permno", None)
    if value is None:
        return None
    try:
        return str(int(str(value).strip()))
    except (ValueError, TypeError):
        return None


def _ticker_from_meta(meta: Optional[Dict[str, object]]) -> Optional[str]:
    """Internal helper for ticker from meta."""
    if not meta:
        return None
    val = str(meta.get("ticker", "")).upper().strip()
    return val if val else None


def _attach_id_attrs(
    df: pd.DataFrame,
    permno: Optional[str],
    ticker: Optional[str],
) -> pd.DataFrame:
    """Internal helper for attach id attrs."""
    if df is None:
        return df
    out = df.copy()
    if permno is not None:
        out.attrs["permno"] = str(permno)
    if ticker is not None:
        out.attrs["ticker"] = str(ticker)
    return out


def _cache_source(meta: Optional[Dict[str, object]]) -> str:
    """Internal helper for cache source."""
    raw = (meta or {}).get("source", "unknown")
    return str(raw).lower().strip() or "unknown"


def _get_last_trading_day(ref_date: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """Return the last completed trading day on or before *ref_date*.

    Uses NYSE calendar when ``pandas_market_calendars`` is installed,
    otherwise falls back to ``pd.bdate_range`` (ignores holidays).

    Parameters
    ----------
    ref_date : pd.Timestamp or None
        Reference date.  Defaults to now.

    Returns
    -------
    pd.Timestamp
        Last trading day on or before *ref_date*.
    """
    if ref_date is None:
        ref_date = pd.Timestamp.now().normalize()
    else:
        ref_date = pd.Timestamp(ref_date).normalize()

    if _NYSE_CAL is not None:
        # Look back up to 10 days to find a trading day
        start = ref_date - pd.Timedelta(days=10)
        schedule = _NYSE_CAL.schedule(start_date=start, end_date=ref_date)
        if len(schedule) > 0:
            return pd.Timestamp(schedule.index[-1]).normalize()

    # Fallback: use business day range (start/end form handles weekends
    # correctly — the periods=1 form returns empty when end is non-business).
    bdays = pd.bdate_range(start=ref_date - pd.Timedelta(days=10), end=ref_date)
    if len(bdays) > 0:
        return bdays[-1].normalize()

    return ref_date


def _trading_days_between(
    start: pd.Timestamp, end: pd.Timestamp,
) -> int:
    """Count trading days between *start* and *end* (inclusive of end).

    Uses NYSE calendar when available, otherwise falls back to
    ``pd.bdate_range``.

    Parameters
    ----------
    start : pd.Timestamp
        Start date.
    end : pd.Timestamp
        End date.

    Returns
    -------
    int
        Number of trading days between start and end.
    """
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()

    if start >= end:
        return 0

    if _NYSE_CAL is not None:
        schedule = _NYSE_CAL.schedule(start_date=start, end_date=end)
        return len(schedule)

    return len(pd.bdate_range(start=start, end=end))


def _cache_is_usable(
    cached: Optional[pd.DataFrame],
    meta: Optional[Dict[str, object]],
    years: int,
    require_recent: bool,
    require_trusted: bool,
) -> bool:
    """Check if cached OHLCV data is fresh enough to use.

    Uses the NYSE trading calendar to count elapsed trading days for
    staleness checks, avoiding false positives on weekends and holidays.

    Falls back to ``pd.bdate_range`` when ``pandas_market_calendars``
    is unavailable.

    Terminal cache entries (delisted stocks whose history is immutable)
    bypass the staleness check entirely — a stock that delisted will
    never produce new bars.
    """
    if cached is None or len(cached) < MIN_BARS:
        return False

    if DATA_QUALITY_ENABLED:
        quality = assess_ohlcv_quality(cached)
        if not quality.passed:
            return False

    source = _cache_source(meta)
    if require_trusted and source not in TRUSTED_CACHE_SOURCES:
        return False

    idx = pd.to_datetime(cached.index)
    if len(idx) == 0:
        return False

    required_start = pd.Timestamp(date.today() - timedelta(days=int(years * 365.25)))
    # Allow a modest buffer for partial history windows/market holidays.
    if idx.min() > required_start + pd.Timedelta(days=90):
        return False

    # Terminal entries (delisted stocks) skip the staleness check entirely.
    # A stock that delisted is historically immutable — its data is complete.
    if meta and meta.get("is_terminal") is True:
        logger.debug("Terminal cache hit — skipping staleness check (meta)")
        return True

    # Also detect terminal state from the data itself if metadata is missing
    if "delist_event" in cached.columns:
        delist_col = pd.to_numeric(cached["delist_event"], errors="coerce").fillna(0)
        if int(delist_col.max()) == 1:
            logger.debug("Terminal cache hit — skipping staleness check (delist_event)")
            return True

    if require_recent:
        last_trading_day = _get_last_trading_day()
        cache_end = pd.Timestamp(idx.max().date())
        trading_days_elapsed = _trading_days_between(cache_end, last_trading_day)
        if trading_days_elapsed > int(CACHE_MAX_STALENESS_DAYS):
            return False

    return True


def _cached_universe_subset(candidates: List[str]) -> List[str]:
    """Prefer locally cached symbols to keep offline runs deterministic."""
    named_cached = {str(t).upper() for t in list_cached_tickers()}
    subset: List[str] = []
    for ticker in candidates:
        t = str(ticker).upper()
        local_df, local_meta, _ = cache_load_with_meta(t)
        if _cache_is_usable(
            cached=local_df,
            meta=local_meta,
            years=LOOKBACK_YEARS,
            require_recent=False,
            require_trusted=False,
        ):
            if REQUIRE_PERMNO and _permno_from_meta(local_meta) is None:
                continue
            subset.append(t)
    return subset


def _normalize_ohlcv(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return a sorted, deterministic OHLCV frame or None if invalid."""
    if df is None or len(df) == 0:
        return None
    if not set(REQUIRED_OHLCV).issubset(df.columns):
        return None
    extras = [c for c in OPTIONAL_PANEL_COLUMNS if c in df.columns]
    cols = REQUIRED_OHLCV + [c for c in extras if c not in REQUIRED_OHLCV]
    out = df[cols].copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out.dropna(subset=REQUIRED_OHLCV)
    return out if len(out) > 0 else None


def _harmonize_return_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize return columns so backtests can consume total-return streams.
    """
    if df is None or len(df) == 0:
        return df

    out = df.copy(deep=True)
    if "Return" not in out.columns and "ret" in out.columns:
        out.loc[:, "Return"] = pd.to_numeric(out["ret"], errors="coerce")
    if "Return" not in out.columns and "Close" in out.columns:
        out.loc[:, "Return"] = pd.to_numeric(out["Close"], errors="coerce").pct_change()
    if "Return" in out.columns:
        out.loc[:, "Return"] = pd.to_numeric(out["Return"], errors="coerce")
    if "total_ret" not in out.columns:
        if "Return" in out.columns:
            out.loc[:, "total_ret"] = out["Return"]
        else:
            out.loc[:, "total_ret"] = np.nan
    else:
        out.loc[:, "total_ret"] = pd.to_numeric(out["total_ret"], errors="coerce")
    if "dlret" in out.columns:
        out.loc[:, "dlret"] = pd.to_numeric(out["dlret"], errors="coerce")
    else:
        out.loc[:, "dlret"] = np.nan
    if "delist_event" not in out.columns:
        out.loc[:, "delist_event"] = np.where(out["dlret"].notna(), 1, 0)
    else:
        out.loc[:, "delist_event"] = (
            pd.to_numeric(out["delist_event"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    return out


def _merge_option_surface_from_prefetch(
    df: Optional[pd.DataFrame],
    permno: Optional[str],
    option_surface: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """
    Merge pre-fetched OptionMetrics surface rows into a single PERMNO panel.
    """
    if df is None or option_surface is None or len(option_surface) == 0 or permno is None:
        return df
    try:
        key = str(int(str(permno).strip()))
    except (ValueError, TypeError):
        return df

    if isinstance(option_surface.index, pd.MultiIndex):
        level0 = option_surface.index.get_level_values(0).astype(str)
        if key not in set(level0):
            return df
        opt = option_surface.xs(key, level=0, drop_level=True).copy()
    else:
        opt = option_surface.copy()

    if len(opt) == 0:
        return df
    opt.index = pd.to_datetime(opt.index)
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    return out.join(opt, how="left")


def load_ohlcv(
    ticker: str,
    years: int = 15,
    use_cache: bool = True,
    use_wrds: bool = WRDS_ENABLED,
) -> Optional[pd.DataFrame]:
    """
    Load daily OHLCV data for a single ticker.

    Returns DataFrame with columns: Open, High, Low, Close, Volume
    and a DatetimeIndex, or None if fetch fails.
    """
    requested_symbol = str(ticker).upper().strip()
    if not _TICKER_RE.match(requested_symbol):
        logger.warning("Invalid ticker format rejected: %r", ticker)
        return None
    cached = None
    cache_meta: Dict[str, object] = {}
    if use_cache:
        cached, cache_meta, _ = cache_load_with_meta(requested_symbol)
        cached = _normalize_ohlcv(cached)
        cached = _harmonize_return_columns(cached) if cached is not None else None
        if cached is not None:
            cached = _attach_id_attrs(
                cached,
                permno=_permno_from_meta(cache_meta),
                ticker=_ticker_from_meta(cache_meta) or requested_symbol,
            )

    cached_permno = _permno_from_meta(cache_meta)

    trusted_cache_ready = _cache_is_usable(
        cached=cached,
        meta=cache_meta,
        years=years,
        require_recent=True,
        require_trusted=True,
    )
    fallback_cache_ready = _cache_is_usable(
        cached=cached,
        meta=cache_meta,
        years=years,
        require_recent=True,
        require_trusted=False,
    )

    # Prefer trusted cache provenance to avoid unnecessary WRDS round-trips.
    if trusted_cache_ready and ((not REQUIRE_PERMNO) or (cached_permno is not None)):
        return cached

    wrds_provider = None

    # ── 1. WRDS CRSP (if enabled) ──
    if use_wrds and WRDS_ENABLED:
        try:
            wrds_provider = get_provider("wrds")
            if wrds_provider.available():
                end = date.today()
                start = end - timedelta(days=int(years * 365.25))
                wrds_map = wrds_provider.get_crsp_prices(
                    tickers=[requested_symbol],
                    start_date=str(start),
                    end_date=str(end),
                )
                if wrds_map:
                    permno_key = max(wrds_map.keys(), key=lambda k: len(wrds_map[k]))
                    wrds_df = wrds_map[permno_key]
                    if OPTIONMETRICS_ENABLED:
                        try:
                            opt_surface = wrds_provider.get_option_surface_features(
                                permnos=[str(permno_key)],
                                start_date=str(start),
                                end_date=str(end),
                            )
                            merged = _merge_option_surface_from_prefetch(
                                df=wrds_df,
                                permno=str(permno_key),
                                option_surface=opt_surface,
                            )
                            if merged is not None:
                                wrds_df = merged
                        except (OSError, ValueError, RuntimeError):
                            pass
                    result = _normalize_ohlcv(wrds_df)
                    result = _harmonize_return_columns(result) if result is not None else None

                    # Quality-gate WRDS data before caching
                    if result is not None and DATA_QUALITY_ENABLED:
                        quality = assess_ohlcv_quality(result)
                        if not quality.passed:
                            logger.warning(
                                "WRDS data for %s failed quality check: %s — not caching",
                                requested_symbol,
                                "; ".join(quality.warnings[:3]),
                            )
                            # Don't cache bad data; fall through to fallback paths
                            result = None
                else:
                    permno_key = None
                    result = None
                if result is not None and permno_key is not None:
                    try:
                        permno_key = str(int(str(permno_key).strip()))
                    except (ValueError, TypeError):
                        if REQUIRE_PERMNO:
                            permno_key = None
                            result = None
                    if result is None or permno_key is None:
                        pass
                    else:
                        wrds_ticker = (
                            str(wrds_df["ticker"].iloc[-1]).upper().strip()
                            if "ticker" in wrds_df.columns and len(wrds_df) > 0
                            else requested_symbol
                        )
                        result = _attach_id_attrs(
                            result,
                            permno=str(permno_key),
                            ticker=wrds_ticker,
                        )
                        # If cache has materially longer validated history, keep the cache path.
                        if fallback_cache_ready and cached is not None:
                            wrds_span = (result.index.max() - result.index.min()).days
                            cached_span = (cached.index.max() - cached.index.min()).days
                            if cached_span > wrds_span + int(CACHE_WRDS_SPAN_ADVANTAGE_DAYS):
                                return cached
                        if use_cache:
                            cache_save(
                                str(permno_key),
                                result,
                                source="wrds",
                                meta={
                                    "years_requested": int(years),
                                    "permno": str(permno_key),
                                    "ticker": wrds_ticker,
                                },
                            )
                        return result
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning(
                "WRDS fetch failed: %s | ticker=%s | years=%d | wrds_enabled=%s",
                e, requested_symbol, years, WRDS_ENABLED,
            )

    # ── 2. Local cache fallback (IBKR/local history) ──
    if fallback_cache_ready and cached is not None and ((not REQUIRE_PERMNO) or (cached_permno is not None)):
        source = _cache_source(cache_meta)
        logger.warning(
            "Data fallback: %s loaded from non-trusted cache (source=%s). "
            "WRDS data was unavailable or insufficient. Results may be affected by survivorship bias.",
            requested_symbol,
            source,
        )
        permno_key = cached_permno if cached_permno is not None else requested_symbol
        _fallback_tracker[requested_symbol] = {
            "source": source,
            "reason": "wrds_unavailable" if use_wrds else "wrds_disabled",
            "bars": len(cached),
            "trusted": False,
            "permno": permno_key,  # Cross-reference to data dict key
        }
        return cached

    return None


def get_data_provenance() -> Dict[str, Dict[str, object]]:
    """Return a summary of data source provenance and any fallbacks that occurred.

    Returns dict mapping ticker -> {"source": str, "reason": str, "trusted": bool,
    "permno": str, ...}.  The ``permno`` key cross-references the data dict key
    returned by ``load_universe()``.
    Useful for auditing whether the system is running on institutional or demo data.

    Note: The tracker is reset at the start of each ``load_universe()`` call to
    prevent stale entries from accumulating across multiple invocations.
    """
    return dict(_fallback_tracker)


def get_skip_reasons() -> Dict[str, str]:
    """Return per-ticker skip reasons from the most recent load_universe() call.

    Returns dict mapping ticker -> reason string (e.g. "permno unresolved",
    "insufficient data (42 bars, need 500)").
    """
    return dict(_skip_tracker)


def load_universe(
    tickers: List[str],
    years: int = 15,
    verbose: bool = True,
    use_cache: bool = True,
    use_wrds: bool = WRDS_ENABLED,
) -> Dict[str, pd.DataFrame]:
    """Load OHLCV data for multiple symbols. Returns {permno: DataFrame}."""
    global _skip_tracker, _fallback_tracker
    _skip_tracker = {}
    _fallback_tracker = {}  # Reset per load cycle to avoid stale entries

    # Validate ticker strings before processing
    valid_tickers = [t for t in tickers if _TICKER_RE.match(str(t).upper().strip())]
    if len(valid_tickers) < len(tickers):
        logger.warning("Rejected %d invalid ticker strings", len(tickers) - len(valid_tickers))

    data: Dict[str, pd.DataFrame] = {}
    skipped: Dict[str, str] = {}
    for i, symbol in enumerate(valid_tickers):
        if verbose:
            print(f"  Loading {symbol} ({i+1}/{len(valid_tickers)})...", end="", flush=True)
        df = load_ohlcv(symbol, years=years, use_cache=use_cache, use_wrds=use_wrds)
        if df is not None and len(df) >= MIN_BARS:  # Config-driven threshold
            if DATA_QUALITY_ENABLED:
                quality = assess_ohlcv_quality(df)
                if not quality.passed:
                    reason = f"quality: {', '.join(quality.warnings)}"
                    skipped[symbol] = reason
                    if verbose:
                        print(f" SKIPPED ({reason})")
                    continue
            permno = df.attrs.get("permno", None)
            if permno is None and REQUIRE_PERMNO:
                skipped[symbol] = "permno unresolved"
                if verbose:
                    print(" SKIPPED (permno unresolved)")
                continue
            key = str(permno) if permno is not None else str(symbol).upper().strip()
            if key in data:
                # Prefer longer history when multiple symbols map to same PERMNO.
                if len(df) > len(data[key]):
                    data[key] = df
            else:
                data[key] = df
            if verbose:
                ticker_lbl = df.attrs.get("ticker", str(symbol).upper().strip())
                print(f" {len(df)} bars (permno={key}, ticker={ticker_lbl})")
        else:
            bars = len(df) if df is not None else 0
            reason = f"insufficient data ({bars} bars, need {MIN_BARS})" if df is not None else "load_ohlcv returned None"
            skipped[symbol] = reason
            if verbose:
                print(f" SKIPPED ({reason})")

    # Always log skip summary so diagnostics are available even with verbose=False
    if skipped:
        logger.warning(
            "load_universe: %d/%d tickers skipped — %s",
            len(skipped), len(valid_tickers),
            "; ".join(f"{sym}: {r}" for sym, r in skipped.items()),
        )
    # Persist skip reasons for downstream consumers (e.g. orchestrator error messages)
    _skip_tracker.update(skipped)

    # Truth Layer: data integrity preflight (blocks corrupt data from pipeline)
    if TRUTH_LAYER_FAIL_ON_CORRUPT and data:
        from ..validation.data_integrity import DataIntegrityValidator
        validator = DataIntegrityValidator(fail_fast=True)
        integrity_result = validator.validate_universe(data)
        if not integrity_result.passed:
            raise RuntimeError(
                f"Data integrity preflight failed: "
                f"{integrity_result.n_stocks_failed} corrupted tickers — "
                f"{', '.join(integrity_result.failed_tickers[:10])}"
            )

    return data


def _tag_survivorship_safe(
    data: Dict[str, pd.DataFrame],
    safe: bool,
) -> Dict[str, pd.DataFrame]:
    """Tag all DataFrames in *data* with ``survivorship_safe`` attribute."""
    for df in data.values():
        df.attrs["survivorship_safe"] = safe
    return data


def load_survivorship_universe(
    as_of_date: Optional[str] = None,
    years: int = 15,
    verbose: bool = True,
    strict: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Load a survivorship-bias-free universe using WRDS CRSP.

    Gets the S&P 500 membership as of `as_of_date` (not today's constituents),
    then loads OHLCV data for all members including those that have since delisted.

    Args:
        as_of_date: Historical date string (YYYY-MM-DD). Defaults to `years` ago.
        years: Years of data to load.
        verbose: Print progress.
        strict: If True, raise RuntimeError instead of falling back to static universe.

    Returns:
        {ticker: DataFrame} — includes delisted stocks with their full price history.
        Each DataFrame has ``attrs["survivorship_safe"]`` set to True (WRDS PIT path)
        or False (static fallback).
    """
    try:
        from .survivorship import hydrate_universe_history_from_snapshots
        from ..config import (
            SURVIVORSHIP_DB,
            SURVIVORSHIP_UNIVERSE_NAME,
            SURVIVORSHIP_SNAPSHOT_FREQ,
            UNIVERSE_FULL,
        )

        if not WRDS_ENABLED:
            logger.warning(
                "WRDS disabled — survivorship-safe universe unavailable. "
                "Falling back to static universe (SURVIVORSHIP BIAS RISK)."
            )
            if strict:
                raise RuntimeError("Survivorship-safe loading requires WRDS but WRDS is disabled")
            cached_subset = _cached_universe_subset(UNIVERSE_FULL)
            if verbose:
                if cached_subset:
                    print(
                        "  WRDS disabled — falling back to cached static universe "
                        f"({len(cached_subset)} tickers)",
                    )
                else:
                    print("  WRDS disabled — falling back to static universe")
            fallback = cached_subset if cached_subset else UNIVERSE_FULL
            result = load_universe(fallback, years=years, verbose=verbose)
            return _tag_survivorship_safe(result, safe=False)

        provider = get_provider("wrds")
        if not provider.available():
            logger.warning(
                "WRDS not available — survivorship-safe universe unavailable. "
                "Falling back to static universe (SURVIVORSHIP BIAS RISK)."
            )
            if strict:
                raise RuntimeError("Survivorship-safe loading requires WRDS but WRDS is not available")
            cached_subset = _cached_universe_subset(UNIVERSE_FULL)
            if verbose:
                if cached_subset:
                    print(
                        "  WRDS not available — falling back to cached static universe "
                        f"({len(cached_subset)} tickers)",
                    )
                else:
                    print("  WRDS not available — falling back to static universe")
            fallback = cached_subset if cached_subset else UNIVERSE_FULL
            result = load_universe(fallback, years=years, verbose=verbose)
            return _tag_survivorship_safe(result, safe=False)

        if as_of_date is None:
            as_of = date.today() - timedelta(days=int(years * 365.25))
            as_of_date = str(as_of)
        end_date = str(date.today())

        if verbose:
            print(
                f"  Loading PIT {SURVIVORSHIP_UNIVERSE_NAME} history "
                f"{as_of_date} -> {end_date} ({SURVIVORSHIP_SNAPSHOT_FREQ})...",
            )

        history = provider.get_sp500_history(
            start_date=as_of_date,
            end_date=end_date,
            freq="quarterly" if str(SURVIVORSHIP_SNAPSHOT_FREQ).lower().startswith("q") else "annual",
        )
        history_ids: List[str] = []
        if history is not None and len(history) > 0:
            hydrate_universe_history_from_snapshots(
                snapshots=history,
                universe_name=SURVIVORSHIP_UNIVERSE_NAME,
                db_path=str(SURVIVORSHIP_DB),
                verbose=verbose,
            )
            if "permno" in history.columns and history["permno"].notna().any():
                history_ids = sorted(
                    {str(int(x)) for x in history["permno"].dropna()},
                )
            else:
                history_ids = sorted(
                    set(history["ticker"].dropna().astype(str).str.strip().str.upper()),
                )

        # Fallback to single-date snapshot if history is unavailable.
        if not history_ids:
            members = provider.get_sp500_universe(as_of_date=as_of_date, include_permno=True)
            history_ids = sorted({str(int(p)) for p, _ in members}) if members else []

        if not history_ids:
            logger.warning(
                "WRDS returned empty PIT universe — survivorship-safe universe unavailable. "
                "Falling back to static universe (SURVIVORSHIP BIAS RISK)."
            )
            if strict:
                raise RuntimeError("Survivorship-safe loading failed: WRDS returned empty PIT universe")
            cached_subset = _cached_universe_subset(UNIVERSE_FULL)
            if verbose:
                if cached_subset:
                    print(
                        "  WRDS returned empty PIT universe — falling back to cached static "
                        f"({len(cached_subset)} tickers)",
                    )
                else:
                    print("  WRDS returned empty PIT universe — falling back to static")
            fallback = cached_subset if cached_subset else UNIVERSE_FULL
            result = load_universe(fallback, years=years, verbose=verbose)
            return _tag_survivorship_safe(result, safe=False)

        if verbose:
            print(f"  PIT universe members to load: {len(history_ids)}")

        # Use delisting-aware path so terminal events are preserved.
        result = load_with_delistings(
            tickers=history_ids,
            years=years,
            verbose=verbose,
        )
        return _tag_survivorship_safe(result, safe=True)

    except (OSError, ValueError, RuntimeError) as e:
        logger.warning(
            "Survivorship universe load failed — falling back to static universe "
            "(SURVIVORSHIP BIAS RISK): %s | years=%d",
            e, years,
        )
        if strict:
            raise RuntimeError(
                f"Survivorship-safe loading failed: {e}"
            ) from e
        from ..config import UNIVERSE_FULL
        cached_subset = _cached_universe_subset(UNIVERSE_FULL)
        fallback = cached_subset if cached_subset else UNIVERSE_FULL
        result = load_universe(fallback, years=years, verbose=verbose)
        return _tag_survivorship_safe(result, safe=False)


def load_with_delistings(
    tickers: List[str],
    years: int = 15,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load OHLCV data including delisting returns from CRSP.

    For stocks that delisted during the period, appends the delisting return
    as a final row so backtests correctly account for the terminal event.
    """
    data: Dict[str, pd.DataFrame] = {}
    remaining: List[str] = []

    # Reuse trusted delisting-aware cache first.
    for symbol in tickers:
        cached_df = None
        cached_meta: Dict[str, object] = {}
        cached_df, cached_meta, _ = cache_load_with_meta(symbol)
        cached_df = _normalize_ohlcv(cached_df)
        cached_df = _harmonize_return_columns(cached_df) if cached_df is not None else None
        if _cache_is_usable(
            cached=cached_df,
            meta=cached_meta,
            years=years,
            require_recent=False,  # delisted names may have old terminal bars
            require_trusted=True,
        ):
            permno = _permno_from_meta(cached_meta) or str(symbol).strip()
            if REQUIRE_PERMNO and _permno_from_meta(cached_meta) is None:
                remaining.append(symbol)
                continue
            cached_df = _attach_id_attrs(
                cached_df,
                permno=str(permno),
                ticker=_ticker_from_meta(cached_meta),
            )
            data[str(permno)] = cached_df
        else:
            remaining.append(symbol)

    if not remaining:
        if verbose:
            print(f"  Loaded {len(data)} tickers from trusted delisting cache")
        return data

    if WRDS_ENABLED:
        try:
            provider = get_provider("wrds")
            if provider.available():
                end = date.today()
                start = end - timedelta(days=int(years * 365.25))
                wrds_data = provider.get_crsp_prices_with_delistings(
                    tickers=remaining,
                    start_date=str(start),
                    end_date=str(end),
                )
                option_surface = None
                if OPTIONMETRICS_ENABLED and wrds_data:
                    try:
                        option_surface = provider.get_option_surface_features(
                            permnos=list(wrds_data.keys()),
                            start_date=str(start),
                            end_date=str(end),
                        )
                    except (OSError, ValueError, RuntimeError):
                        option_surface = None
                for symbol in remaining:
                    permno_key = str(symbol).strip()
                    df = wrds_data.get(permno_key)
                    if df is None:
                        # When input symbols were tickers, provider returns PERMNO keys.
                        for k, v in wrds_data.items():
                            if "ticker" in v.columns and len(v) > 0:
                                if str(v["ticker"].iloc[-1]).upper().strip() == str(symbol).upper().strip():
                                    permno_key = str(k)
                                    df = v
                                    break
                    if df is None or len(df) == 0:
                        continue
                    if "permno" in df.columns and df["permno"].notna().any():
                        try:
                            permno_key = str(int(pd.to_numeric(df["permno"], errors="coerce").dropna().iloc[-1]))
                        except (ValueError, TypeError):
                            pass
                    merged = _merge_option_surface_from_prefetch(
                        df=df,
                        permno=permno_key,
                        option_surface=option_surface,
                    )
                    if merged is not None:
                        df = merged
                    normalized = _normalize_ohlcv(df)
                    normalized = _harmonize_return_columns(normalized) if normalized is not None else None
                    if normalized is None or len(normalized) < 500:
                        continue
                    if REQUIRE_PERMNO:
                        try:
                            permno_key = str(int(str(permno_key).strip()))
                        except (ValueError, TypeError):
                            continue
                    ticker_lbl = (
                        str(df["ticker"].iloc[-1]).upper().strip()
                        if "ticker" in df.columns and len(df) > 0
                        else None
                    )
                    normalized = _attach_id_attrs(
                        normalized,
                        permno=permno_key,
                        ticker=ticker_lbl,
                    )
                    data[permno_key] = normalized
                    cache_save(
                        permno_key,
                        normalized,
                        source="wrds_delisting",
                        meta={
                            "years_requested": int(years),
                            "permno": permno_key,
                            "ticker": ticker_lbl,
                        },
                    )

                if data:
                    if verbose:
                        print(f"  Loaded {len(data)} tickers with WRDS/cached delisting-aware data")
                    return data
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning(
                "WRDS delisting fetch failed: %s | remaining_tickers=%d | years=%d",
                e, len(remaining), years,
            )

    # Fallback to standard universe load if WRDS delisting path fails.
    fallback = load_universe(
        remaining,
        years=years,
        verbose=verbose,
        use_cache=True,
        use_wrds=WRDS_ENABLED,
    )
    data.update(fallback)

    # Truth Layer: validate delisting-augmented data before returning
    if TRUTH_LAYER_FAIL_ON_CORRUPT and data:
        from ..validation.data_integrity import DataIntegrityValidator
        validator = DataIntegrityValidator(fail_fast=False)  # Don't fail on first bad ticker
        integrity_result = validator.validate_universe(data)
        if not integrity_result.passed:
            # Remove corrupt tickers rather than failing entirely —
            # delisted companies may legitimately have unusual price patterns
            for bad_ticker in integrity_result.failed_tickers:
                logger.warning("Removing corrupt delisted ticker %s from universe", bad_ticker)
                data.pop(bad_ticker, None)
            if not data:
                raise RuntimeError(
                    f"All {integrity_result.n_stocks_failed} tickers failed integrity check"
                )

    return data


def warn_if_survivorship_biased(
    data: Dict[str, pd.DataFrame],
    context: str = "pipeline",
) -> bool:
    """Check and log a warning if the data is not survivorship-safe.

    Returns True if the data is survivorship-safe, False otherwise.
    """
    sample_df = next(iter(data.values()), None)
    if sample_df is not None and not sample_df.attrs.get("survivorship_safe", True):
        logger.warning(
            "Running %s on survivorship-BIASED universe — results may overstate performance",
            context,
        )
        return False
    return True
