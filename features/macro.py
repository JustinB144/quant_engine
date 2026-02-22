"""
FRED macro indicator features for quant_engine.

Provides ``MacroFeatureProvider`` which pulls key macro series from FRED
(Federal Reserve Economic Data) and transforms them into model-ready features.

Supported series:
    VIXCLS      - CBOE VIX (daily)
    T10Y2Y      - 10Y-2Y Treasury term spread (daily)
    BAMLC0A0CM  - BofA US Corporate Master Option-Adjusted Spread (daily)
    ICSA        - Initial Jobless Claims (weekly, forward-filled to daily)
    UMCSENT     - Univ of Michigan Consumer Sentiment (monthly, forward-filled)

External dependencies are wrapped in try/except with pure requests fallback.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS = 1e-12

# FRED series configuration: series_id -> (column_name, frequency_hint)
_FRED_SERIES: Dict[str, tuple] = {
    "VIXCLS": ("macro_vix", "daily"),
    "T10Y2Y": ("macro_term_spread", "daily"),
    "BAMLC0A0CM": ("macro_credit_spread", "daily"),
    "ICSA": ("macro_initial_claims", "weekly"),
    "UMCSENT": ("macro_consumer_sentiment", "monthly"),
}

_FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"


def _cache_key(series_id: str, start: str, end: str) -> str:
    """Generate a deterministic cache filename."""
    raw = f"{series_id}_{start}_{end}"
    h = hashlib.md5(raw.encode()).hexdigest()[:12]
    return f"fred_{series_id}_{h}.parquet"


class MacroFeatureProvider:
    """FRED API integration for macro indicator features.

    Parameters
    ----------
    api_key : str, optional
        FRED API key.  If not supplied, the provider checks the
        ``FRED_API_KEY`` environment variable.
    cache_dir : Path, optional
        Directory for caching downloaded data.  Defaults to
        ``~/.quant_engine/fred_cache``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """Initialize MacroFeatureProvider."""
        import os

        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        if cache_dir is None:
            self.cache_dir = Path.home() / ".quant_engine" / "fred_cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_series_fredapi(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """Fetch a FRED series using the ``fredapi`` library."""
        from fredapi import Fred  # type: ignore

        fred = Fred(api_key=self.api_key)
        data = fred.get_series(
            series_id,
            observation_start=start_date,
            observation_end=end_date,
        )
        data.index = pd.to_datetime(data.index)
        data.name = series_id
        return data

    def _fetch_series_requests(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """Fetch a FRED series using raw ``requests`` (fallback)."""
        import requests  # type: ignore

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
            "sort_order": "asc",
        }
        resp = requests.get(_FRED_API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        observations = payload.get("observations", [])
        dates = []
        values = []
        for obs in observations:
            d = obs.get("date")
            v = obs.get("value", ".")
            if d and v != ".":
                try:
                    dates.append(pd.Timestamp(d))
                    values.append(float(v))
                except (ValueError, TypeError):
                    continue

        if not dates:
            return pd.Series(dtype=float, name=series_id)

        s = pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id)
        return s

    def _fetch_series(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """Fetch a single FRED series with library fallback and caching."""
        # Check cache first
        cache_file = self.cache_dir / _cache_key(series_id, start_date, end_date)
        if cache_file.exists():
            try:
                cached = pd.read_parquet(cache_file)
                if series_id in cached.columns:
                    logger.debug("Cache hit for %s", series_id)
                    return cached[series_id].dropna()
            except (OSError, ValueError):
                pass  # Corrupted cache; re-fetch

        # Try fredapi first, then raw requests
        data: Optional[pd.Series] = None
        try:
            data = self._fetch_series_fredapi(series_id, start_date, end_date)
            logger.debug("Fetched %s via fredapi (%d observations)", series_id, len(data))
        except (ImportError, OSError, ValueError) as exc_fredapi:
            logger.debug("fredapi failed for %s: %s; trying requests", series_id, exc_fredapi)
            try:
                data = self._fetch_series_requests(series_id, start_date, end_date)
                logger.debug(
                    "Fetched %s via requests (%d observations)", series_id, len(data)
                )
            except (ImportError, OSError, ValueError) as exc_req:
                logger.warning(
                    "Could not fetch %s from FRED: fredapi=%s, requests=%s",
                    series_id,
                    exc_fredapi,
                    exc_req,
                )
                return pd.Series(dtype=float, name=series_id)

        if data is None or data.empty:
            return pd.Series(dtype=float, name=series_id)

        # Cache the result
        try:
            cache_df = pd.DataFrame({series_id: data})
            cache_df.to_parquet(cache_file)
        except (OSError, ValueError) as exc_cache:
            logger.debug("Could not cache %s: %s", series_id, exc_cache)

        return data

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def get_macro_features(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Download FRED macro series and compute level + momentum features.

        Parameters
        ----------
        start_date : str
            Start date in YYYY-MM-DD format.
        end_date : str
            End date in YYYY-MM-DD format.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex with columns like ``macro_vix``, ``macro_vix_mom``,
            ``macro_term_spread``, ``macro_term_spread_mom``, etc.
            Lower-frequency series are forward-filled to daily.
        """
        # Build a business-day date range
        date_range = pd.bdate_range(start=start_date, end=end_date)
        result = pd.DataFrame(index=date_range)
        result.index.name = "date"

        for series_id, (col_name, freq_hint) in _FRED_SERIES.items():
            raw = self._fetch_series(series_id, start_date, end_date)

            if raw.empty:
                result[col_name] = np.nan
                result[f"{col_name}_mom"] = np.nan
                continue

            # Convert to numeric and reindex to business days
            raw = pd.to_numeric(raw, errors="coerce")
            raw.index = pd.to_datetime(raw.index)

            # Reindex to business days and forward-fill (weekly/monthly data)
            aligned = raw.reindex(date_range).ffill(limit=5)

            # Level
            result[col_name] = aligned

            # 20-day momentum (change over 20 business days)
            result[f"{col_name}_mom"] = aligned.diff(20)

        return result.replace([np.inf, -np.inf], np.nan)
