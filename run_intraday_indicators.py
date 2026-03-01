#!/usr/bin/env python3
"""
Intraday Indicators Computation & Caching (SPEC 2)

Computes intraday microstructure indicators from cached TAQ OHLCV data
(SPEC 1 output) and stores them in the feature store.  This replaces the
live WRDS TAQmsec queries that features/intraday.py currently makes per
ticker per day.

Produces 14 indicators per ticker per trading day:
  Existing 6: intraday_vol_ratio, vwap_deviation, amihud_illiquidity,
              kyle_lambda, realized_vol_5m, microstructure_noise
  New 8:      intraday_range, close_location_value, intraday_momentum,
              volume_clock_skew, bid_ask_spread_proxy, parkinson_vol,
              garman_klass_vol, volume_weighted_return

Usage:
    python3 run_intraday_indicators.py
    python3 run_intraday_indicators.py --tickers AAPL,MSFT
    python3 run_intraday_indicators.py --start 2020-01-01 --end 2025-12-31
    python3 run_intraday_indicators.py --force
    python3 run_intraday_indicators.py --verify
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Path setup ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from quant_engine.config import (
    DATA_CACHE_DIR,
    INTRADAY_MIN_BARS,
    MARKET_CLOSE,
    MARKET_OPEN,
    REQUIRE_PERMNO,
    TAQ_START_DATE,
    UNIVERSE_INTRADAY,
)
from quant_engine.data.feature_store import FeatureStore
from quant_engine.data.local_cache import REQUIRED_OHLCV, load_intraday_ohlcv
from quant_engine.data.wrds_provider import WRDSProvider

# ── Constants ─────────────────────────────────────────────────────────────
FEATURE_VERSION = "intraday_v1"

# All 14 indicator column names (ordered)
INDICATOR_COLUMNS: List[str] = [
    # Existing 6
    "intraday_vol_ratio",
    "vwap_deviation",
    "amihud_illiquidity",
    "kyle_lambda",
    "realized_vol_5m",
    "microstructure_noise",
    # New 8
    "intraday_range",
    "close_location_value",
    "intraday_momentum",
    "volume_clock_skew",
    "bid_ask_spread_proxy",
    "parkinson_vol",
    "garman_klass_vol",
    "volume_weighted_return",
]


# ── Indicator computation ────────────────────────────────────────────────


def _resample_to_5m(bars_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute bars to 5-minute bars within a single day.

    Parameters
    ----------
    bars_1m : pd.DataFrame
        1-minute OHLCV DataFrame with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        5-minute OHLCV DataFrame.
    """
    ohlcv = pd.DataFrame()
    ohlcv["Open"] = bars_1m["Open"].resample("5min").first()
    ohlcv["High"] = bars_1m["High"].resample("5min").max()
    ohlcv["Low"] = bars_1m["Low"].resample("5min").min()
    ohlcv["Close"] = bars_1m["Close"].resample("5min").last()
    ohlcv["Volume"] = bars_1m["Volume"].resample("5min").sum()
    return ohlcv.dropna(subset=["Open", "Close"])


def _compute_day_indicators(day_bars_1m: pd.DataFrame) -> Dict[str, float]:
    """Compute all 14 indicators for a single trading day from 1m bars.

    Parameters
    ----------
    day_bars_1m : pd.DataFrame
        1-minute OHLCV bars for a single trading day.

    Returns
    -------
    dict[str, float]
        Indicator name to value mapping.
    """
    result: Dict[str, float] = {col: float("nan") for col in INDICATOR_COLUMNS}

    if day_bars_1m is None or len(day_bars_1m) < 10:
        return result

    close = pd.to_numeric(day_bars_1m["Close"], errors="coerce")
    volume = pd.to_numeric(day_bars_1m["Volume"], errors="coerce").fillna(0)
    high = pd.to_numeric(day_bars_1m["High"], errors="coerce")
    low = pd.to_numeric(day_bars_1m["Low"], errors="coerce")
    open_ = pd.to_numeric(day_bars_1m["Open"], errors="coerce")

    # ── 1. intraday_vol_ratio ────────────────────────────────────────
    try:
        first_hour = day_bars_1m.between_time(MARKET_OPEN, "10:30")
        last_hour = day_bars_1m.between_time("15:00", MARKET_CLOSE)
        vol_first = pd.to_numeric(first_hour["Volume"], errors="coerce").sum()
        vol_last = pd.to_numeric(last_hour["Volume"], errors="coerce").sum()
        result["intraday_vol_ratio"] = (
            float(vol_first / vol_last) if vol_last > 0 else float("nan")
        )
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        pass

    # ── 2. vwap_deviation ────────────────────────────────────────────
    try:
        typical_price = (high + low + close) / 3.0
        dollar_vol = typical_price * volume
        total_dollar = dollar_vol.sum()
        total_vol = volume.sum()
        if total_vol > 0:
            vwap = total_dollar / total_vol
            last_close = close.iloc[-1]
            result["vwap_deviation"] = (
                float((last_close - vwap) / vwap) if vwap > 0 else float("nan")
            )
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        pass

    # ── 3. amihud_illiquidity ────────────────────────────────────────
    try:
        ret_1m = close.pct_change().replace([np.inf, -np.inf], np.nan)
        dollar_volume = close * volume
        valid = (dollar_volume > 0) & ret_1m.notna()
        if valid.sum() > 5:
            amihud = (ret_1m[valid].abs() / dollar_volume[valid]).mean() * 1e6
            result["amihud_illiquidity"] = float(amihud)
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        pass

    # ── 4. kyle_lambda ───────────────────────────────────────────────
    try:
        ret_1m = close.pct_change().replace([np.inf, -np.inf], np.nan)
        signed_vol = np.sign(ret_1m) * np.sqrt(volume)
        mask = ret_1m.notna() & signed_vol.notna() & np.isfinite(signed_vol)
        if mask.sum() > 10:
            y = ret_1m[mask].values
            x = signed_vol[mask].values
            cov_xy = np.cov(y, x, ddof=1)
            if cov_xy.shape == (2, 2) and cov_xy[1, 1] > 1e-15:
                result["kyle_lambda"] = float(cov_xy[0, 1] / cov_xy[1, 1])
    except (KeyError, ValueError, TypeError):
        pass

    # ── 5. realized_vol_5m ───────────────────────────────────────────
    try:
        bars_5m_close = close.resample("5min").last().dropna()
        if len(bars_5m_close) > 5:
            ret_5m = np.log(bars_5m_close / bars_5m_close.shift(1)).dropna()
            rv_5m = ret_5m.std() * np.sqrt(78.0 * 252.0)
            result["realized_vol_5m"] = float(rv_5m)
    except (KeyError, ValueError, TypeError):
        pass

    # ── 6. microstructure_noise ──────────────────────────────────────
    try:
        log_ret_1m = np.log(close / close.shift(1)).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        bars_5m_close = close.resample("5min").last().dropna()
        log_ret_5m = np.log(bars_5m_close / bars_5m_close.shift(1)).dropna()

        if len(log_ret_1m) > 10 and len(log_ret_5m) > 5:
            var_1m = log_ret_1m.var()
            var_5m = log_ret_5m.var()
            if var_5m > 1e-15:
                result["microstructure_noise"] = float(
                    (var_1m / var_5m) - (1.0 / 5.0)
                )
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        pass

    # ── 7. intraday_range ────────────────────────────────────────────
    try:
        day_high = high.max()
        day_low = low.min()
        day_open = open_.iloc[0]
        if day_open > 0 and np.isfinite(day_high) and np.isfinite(day_low):
            result["intraday_range"] = float((day_high - day_low) / day_open)
    except (KeyError, ValueError, TypeError, IndexError):
        pass

    # ── 8. close_location_value ──────────────────────────────────────
    try:
        day_close = close.iloc[-1]
        day_high = high.max()
        day_low = low.min()
        hl_range = day_high - day_low
        if hl_range > 0:
            result["close_location_value"] = float(
                (day_close - day_low) / hl_range
            )
    except (KeyError, ValueError, TypeError, IndexError):
        pass

    # ── 9. intraday_momentum ─────────────────────────────────────────
    try:
        day_open = open_.iloc[0]
        day_close = close.iloc[-1]
        # Get noon price (closest bar to 12:00)
        noon_bars = day_bars_1m.between_time("11:59", "12:01")
        if len(noon_bars) > 0:
            noon_close = pd.to_numeric(noon_bars["Close"], errors="coerce").iloc[-1]
        else:
            # Fallback: closest bar before 12:00
            morning = day_bars_1m.between_time(MARKET_OPEN, "12:00")
            if len(morning) > 0:
                noon_close = pd.to_numeric(morning["Close"], errors="coerce").iloc[-1]
            else:
                noon_close = float("nan")
        if day_open > 0 and noon_close > 0 and np.isfinite(noon_close):
            am_ret = (noon_close - day_open) / day_open
            pm_ret = (day_close - noon_close) / noon_close
            result["intraday_momentum"] = float(am_ret - pm_ret)
    except (KeyError, ValueError, TypeError, IndexError):
        pass

    # ── 10. volume_clock_skew ────────────────────────────────────────
    try:
        cum_vol = volume.cumsum()
        total_vol = cum_vol.iloc[-1]
        if total_vol > 0:
            half_vol = total_vol / 2.0
            # Find time index where cumulative volume crosses 50%
            crossed = cum_vol >= half_vol
            if crossed.any():
                time_50pct = cum_vol[crossed].index[0]
                # Midpoint of trading day: 12:45 ET
                day_date = day_bars_1m.index[0].date()
                midpoint = pd.Timestamp(
                    year=day_date.year, month=day_date.month, day=day_date.day,
                    hour=12, minute=45,
                )
                # Skew in minutes
                delta_minutes = (time_50pct - midpoint).total_seconds() / 60.0
                # Normalize by half-day length (195 minutes from 9:30 to 12:45)
                result["volume_clock_skew"] = float(delta_minutes / 195.0)
    except (KeyError, ValueError, TypeError, IndexError):
        pass

    # ── 11. bid_ask_spread_proxy ─────────────────────────────────────
    try:
        mid = (high + low) / 2.0
        spread_proxy = 2.0 * (close - mid).abs() / close
        valid_spread = spread_proxy.replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_spread) > 5:
            result["bid_ask_spread_proxy"] = float(valid_spread.mean())
    except (KeyError, ValueError, TypeError):
        pass

    # ── 12. parkinson_vol (from 5m bars) ─────────────────────────────
    try:
        bars_5m = _resample_to_5m(day_bars_1m)
        if len(bars_5m) > 5:
            h5 = pd.to_numeric(bars_5m["High"], errors="coerce")
            l5 = pd.to_numeric(bars_5m["Low"], errors="coerce")
            valid_hl = (h5 > 0) & (l5 > 0)
            if valid_hl.sum() > 5:
                log_hl = np.log(h5[valid_hl] / l5[valid_hl])
                n = valid_hl.sum()
                park_var = (log_hl ** 2).sum() / (4.0 * n * np.log(2.0))
                result["parkinson_vol"] = float(np.sqrt(park_var) * np.sqrt(252.0))
    except (KeyError, ValueError, TypeError):
        pass

    # ── 13. garman_klass_vol (from 5m bars) ──────────────────────────
    try:
        bars_5m = _resample_to_5m(day_bars_1m)
        if len(bars_5m) > 5:
            o5 = pd.to_numeric(bars_5m["Open"], errors="coerce")
            h5 = pd.to_numeric(bars_5m["High"], errors="coerce")
            l5 = pd.to_numeric(bars_5m["Low"], errors="coerce")
            c5 = pd.to_numeric(bars_5m["Close"], errors="coerce")
            valid = (o5 > 0) & (h5 > 0) & (l5 > 0) & (c5 > 0)
            if valid.sum() > 5:
                log_hl = np.log(h5[valid] / l5[valid])
                log_co = np.log(c5[valid] / o5[valid])
                n = valid.sum()
                gk_var = (
                    0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
                ).mean()
                if gk_var > 0:
                    result["garman_klass_vol"] = float(
                        np.sqrt(gk_var) * np.sqrt(252.0)
                    )
    except (KeyError, ValueError, TypeError):
        pass

    # ── 14. volume_weighted_return ───────────────────────────────────
    try:
        ret_1m = close.pct_change().replace([np.inf, -np.inf], np.nan)
        valid = ret_1m.notna() & (volume > 0)
        if valid.sum() > 5:
            vwr = (ret_1m[valid] * volume[valid]).sum() / volume[valid].sum()
            result["volume_weighted_return"] = float(vwr)
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        pass

    return result


def compute_all_indicators(
    bars_1m: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Compute all 14 indicators for a ticker from its full 1m bar history.

    Groups by trading day and computes indicators vectorized across
    all days.

    Parameters
    ----------
    bars_1m : pd.DataFrame
        Full 1-minute OHLCV history with DatetimeIndex.
    start_date : str, optional
        Filter to dates on or after this date (YYYY-MM-DD).
    end_date : str, optional
        Filter to dates on or before this date (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        One row per trading day, 14 float64 indicator columns.
        Index is DatetimeIndex (daily dates).
    """
    if bars_1m is None or bars_1m.empty:
        return pd.DataFrame(columns=INDICATOR_COLUMNS)

    # Apply date filters
    if start_date:
        bars_1m = bars_1m[bars_1m.index >= start_date]
    if end_date:
        bars_1m = bars_1m[bars_1m.index <= end_date]

    if bars_1m.empty:
        return pd.DataFrame(columns=INDICATOR_COLUMNS)

    # Group by trading day
    daily_groups = bars_1m.groupby(bars_1m.index.date)

    rows: List[Dict[str, float]] = []
    dates: List[pd.Timestamp] = []

    for day_date, day_bars in daily_groups:
        if len(day_bars) < 10:
            continue
        indicators = _compute_day_indicators(day_bars)
        rows.append(indicators)
        dates.append(pd.Timestamp(day_date))

    if not rows:
        return pd.DataFrame(columns=INDICATOR_COLUMNS)

    result = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="Date"))
    result = result[INDICATOR_COLUMNS]  # Enforce column order
    result = result.astype(np.float64)
    result = result.sort_index()
    return result


# ── PERMNO Resolution ────────────────────────────────────────────────────


def _resolve_permno(
    ticker: str,
    wrds_provider: Optional[WRDSProvider],
) -> Optional[str]:
    """Resolve a ticker to its CRSP PERMNO.

    Parameters
    ----------
    ticker : str
        Stock symbol.
    wrds_provider : WRDSProvider or None
        WRDSProvider instance for CRSP lookup.

    Returns
    -------
    str or None
        PERMNO string, or None if resolution fails.
    """
    if wrds_provider is None or not wrds_provider.available():
        logger.warning("WRDSProvider unavailable — using ticker '%s' as key", ticker)
        return ticker

    try:
        permno = wrds_provider.resolve_permno(ticker)
        if permno is not None:
            return permno
        logger.warning(
            "Could not resolve PERMNO for %s — using ticker as key", ticker,
        )
        return ticker
    except (OSError, ValueError, RuntimeError) as e:
        logger.warning(
            "PERMNO resolution failed for %s: %s — using ticker as key", ticker, e,
        )
        return ticker


# ── Core pipeline ────────────────────────────────────────────────────────


def run_compute(
    tickers: List[str],
    start_date: str,
    end_date: str,
    force: bool = False,
) -> Dict[str, int]:
    """Compute and cache intraday indicators for all tickers.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to process.
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).
    force : bool
        If True, recompute even if cached.

    Returns
    -------
    dict[str, int]
        Summary counters.
    """
    store = FeatureStore(store_dir=DATA_CACHE_DIR / "feature_store")
    computed_at = datetime.now().strftime("%Y-%m-%d")

    # Initialize WRDS for PERMNO resolution
    wrds_provider: Optional[WRDSProvider] = None
    if REQUIRE_PERMNO:
        try:
            wrds_provider = WRDSProvider()
            if not wrds_provider.available():
                logger.warning("WRDSProvider not available — using tickers as keys")
                wrds_provider = None
        except (OSError, RuntimeError) as e:
            logger.warning("WRDSProvider init failed: %s — using tickers as keys", e)

    summary = {
        "processed": 0,
        "skipped_no_data": 0,
        "skipped_cached": 0,
        "errors": 0,
    }

    total = len(tickers)
    for idx, ticker in enumerate(tickers, 1):
        logger.info("[%d/%d] Processing %s...", idx, total, ticker)

        # Resolve PERMNO
        permno = _resolve_permno(ticker, wrds_provider)
        if permno is None:
            summary["skipped_no_data"] += 1
            continue

        # Check existing cache (skip if not --force)
        if not force:
            existing = store.load_features(
                permno=permno, feature_version=FEATURE_VERSION,
            )
            if existing is not None and len(existing) > 0:
                logger.info(
                    "  %s: cached (%d rows), skipping (use --force to recompute)",
                    ticker, len(existing),
                )
                summary["skipped_cached"] += 1
                continue

        # Load 1m bars from cache
        bars_1m = load_intraday_ohlcv(ticker, "1m")
        if bars_1m is None or len(bars_1m) < INTRADAY_MIN_BARS:
            logger.warning(
                "  %s: insufficient 1m data (%d bars, need %d)",
                ticker,
                len(bars_1m) if bars_1m is not None else 0,
                INTRADAY_MIN_BARS,
            )
            summary["skipped_no_data"] += 1
            continue

        try:
            indicators_df = compute_all_indicators(
                bars_1m, start_date=start_date, end_date=end_date,
            )

            if indicators_df.empty:
                logger.warning("  %s: no trading days produced indicators", ticker)
                summary["skipped_no_data"] += 1
                continue

            # Save to feature store
            store.save_features(
                permno=permno,
                features=indicators_df,
                computed_at=computed_at,
                feature_version=FEATURE_VERSION,
            )

            logger.info(
                "  %s (PERMNO %s): saved %d days of indicators",
                ticker, permno, len(indicators_df),
            )
            summary["processed"] += 1

        except (OSError, ValueError, RuntimeError) as e:
            logger.error("  %s: computation failed: %s", ticker, e)
            summary["errors"] += 1

    return summary


# ── Verification ─────────────────────────────────────────────────────────


def run_verify(
    tickers: List[str],
    start_date: str,
    end_date: str,
) -> None:
    """Compare cached indicators vs live WRDS TAQmsec for 2022+ dates.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to verify.
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).
    """
    from quant_engine.features.intraday import compute_intraday_features

    store = FeatureStore(store_dir=DATA_CACHE_DIR / "feature_store")

    try:
        wrds_provider = WRDSProvider()
        if not wrds_provider.available():
            logger.error("WRDSProvider not available — cannot verify")
            return
    except (OSError, RuntimeError) as e:
        logger.error("WRDSProvider init failed: %s", e)
        return

    # Only verify 2022+ dates (TAQmsec coverage)
    verify_start = max(start_date, "2022-01-04")
    if verify_start > end_date:
        logger.info("No dates in range for verification (TAQmsec starts 2022)")
        return

    existing_features = [
        "intraday_vol_ratio", "vwap_deviation", "amihud_illiquidity",
        "kyle_lambda", "realized_vol_5m", "microstructure_noise",
    ]

    total_compared = 0
    total_mismatches = 0

    for ticker in tickers[:5]:  # Verify a subset to avoid long runtime
        permno = _resolve_permno(ticker, wrds_provider)
        if permno is None:
            continue

        cached = store.load_features(
            permno=permno, feature_version=FEATURE_VERSION,
        )
        if cached is None:
            logger.warning("  %s: no cached indicators to verify", ticker)
            continue

        # Pick 5 sample dates from the cached data in the verify range
        cached_range = cached[
            (cached.index >= verify_start) & (cached.index <= end_date)
        ]
        if len(cached_range) < 5:
            continue

        sample_dates = cached_range.index[
            np.linspace(0, len(cached_range) - 1, 5, dtype=int)
        ]

        for date in sample_dates:
            date_str = date.strftime("%Y-%m-%d")
            live = compute_intraday_features(ticker, date_str, wrds_provider)

            for feat in existing_features:
                cached_val = cached.loc[date, feat] if date in cached.index else float("nan")
                live_val = live.get(feat, float("nan"))

                if np.isnan(cached_val) and np.isnan(live_val):
                    continue

                total_compared += 1
                if not np.isnan(cached_val) and not np.isnan(live_val):
                    rel_diff = abs(cached_val - live_val) / (abs(live_val) + 1e-12)
                    if rel_diff > 0.01:  # 1% tolerance
                        total_mismatches += 1
                        logger.warning(
                            "  MISMATCH %s %s %s: cached=%.6f live=%.6f (diff=%.2f%%)",
                            ticker, date_str, feat, cached_val, live_val, rel_diff * 100,
                        )

    logger.info(
        "Verification complete: %d comparisons, %d mismatches (%.1f%%)",
        total_compared, total_mismatches,
        (total_mismatches / total_compared * 100) if total_compared > 0 else 0.0,
    )


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for intraday indicators computation."""
    parser = argparse.ArgumentParser(
        description="Compute intraday microstructure indicators from cached TAQ OHLCV data",
    )
    parser.add_argument(
        "--tickers", type=str, default=None,
        help="Comma-separated ticker list (default: full UNIVERSE_INTRADAY)",
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
        help="Recompute even if cached",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Compare vs live WRDS for 2022+ dates",
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
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        tickers = list(UNIVERSE_INTRADAY)

    # Resolve end date
    end_date = args.end or datetime.now().strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info("INTRADAY INDICATORS COMPUTATION & CACHING")
    logger.info("=" * 60)
    logger.info("Tickers: %d", len(tickers))
    logger.info("Date range: %s to %s", args.start, end_date)
    logger.info("Feature version: %s", FEATURE_VERSION)
    logger.info("Force recompute: %s", args.force)

    if args.verify:
        run_verify(tickers, args.start, end_date)
        return

    summary = run_compute(
        tickers=tickers,
        start_date=args.start,
        end_date=end_date,
        force=args.force,
    )

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for key, val in summary.items():
        logger.info("  %s: %d", key, val)


if __name__ == "__main__":
    main()
