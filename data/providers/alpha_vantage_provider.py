"""
Alpha Vantage intraday data provider.

Fetches intraday OHLCV history using the ``month=YYYY-MM`` parameter for
deep historical coverage (20+ years).  Requires a paid API key ($49.99/mo
or higher) for practical rate limits (75 req/min).

Free tier (25 req/day) is insufficient for 128 tickers.

Data characteristics:
    - SIP-aggregated (same quality tier as Alpaca)
    - Column names: ``1. open``, ``2. high``, ``3. low``, ``4. close``, ``5. volume``
    - Timestamps in US/Eastern — normalized to tz-naive ET (matching IBKR convention)
    - ``extended_hours=false`` to match IBKR RTH data

Rate limiting:
    - Free tier: 25 requests/day
    - $49.99 tier: 75 requests/minute
    - $99.99 tier: 150 requests/minute
    - Default pace of 0.85s/request stays safely under 75 req/min
"""

import logging
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Canonical OHLCV columns expected by the quant engine
_REQUIRED_OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# Alpha Vantage column name mapping
_AV_COLUMN_MAP = {
    "1. open": "Open",
    "2. high": "High",
    "3. low": "Low",
    "4. close": "Close",
    "5. volume": "Volume",
}

# Map canonical timeframe codes to AV interval strings
_AV_INTERVALS = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "60min",
}


class AlphaVantageProvider:
    """
    Alpha Vantage intraday data provider with month-by-month historical fetch.

    Supports 1m, 5m, 15m, 30m, and 1h timeframes.  AV does not support 4h bars
    natively; use 1h and resample if needed.

    Args:
        api_key: Alpha Vantage API key (required for all tiers).
        pace: Seconds to wait between requests.  0.85s yields ~70 req/min,
              safely under the 75 req/min paid-tier limit.
        max_retries: Maximum retries per request on rate-limit (HTTP 429) errors.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(
        self,
        api_key: str,
        pace: float = 0.85,
        max_retries: int = 4,
    ):
        if not api_key:
            raise ValueError("Alpha Vantage API key is required")
        self.api_key = api_key
        self.pace = pace
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "quant_engine/1.0"})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_month(
        self,
        ticker: str,
        timeframe: str,
        year: int,
        month: int,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch one month of intraday data from Alpha Vantage.

        Args:
            ticker: Stock ticker symbol (e.g. ``"AAPL"``).
            timeframe: Canonical timeframe code (``"1m"``, ``"5m"``, etc.).
            year: Calendar year.
            month: Calendar month (1-12).

        Returns:
            DataFrame with DatetimeIndex (tz-naive ET) and OHLCV columns,
            or ``None`` if no data is available for the requested period.
        """
        interval = _AV_INTERVALS.get(timeframe)
        if interval is None:
            logger.error(
                "Unsupported timeframe %r for Alpha Vantage. "
                "Supported: %s",
                timeframe,
                ", ".join(sorted(_AV_INTERVALS.keys())),
            )
            return None

        month_str = f"{year:04d}-{month:02d}"

        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": ticker.upper(),
            "interval": interval,
            "month": month_str,
            "outputsize": "full",
            "extended_hours": "false",
            "datatype": "json",
            "apikey": self.api_key,
        }

        data = self._request_with_retry(params, ticker, month_str)
        if data is None:
            return None

        # AV nests intraday data under a key like "Time Series (5min)"
        ts_key = next(
            (k for k in data if k.startswith("Time Series")),
            None,
        )
        if ts_key is None:
            # Check for error messages
            if "Error Message" in data:
                logger.warning(
                    "AV error for %s %s: %s",
                    ticker,
                    month_str,
                    data["Error Message"],
                )
            elif "Note" in data:
                logger.warning(
                    "AV rate limit note for %s %s: %s",
                    ticker,
                    month_str,
                    data["Note"],
                )
            elif "Information" in data:
                logger.warning(
                    "AV info for %s %s: %s",
                    ticker,
                    month_str,
                    data["Information"],
                )
            return None

        ts_data = data[ts_key]
        if not ts_data:
            logger.debug("No bars in AV response for %s %s", ticker, month_str)
            return None

        df = pd.DataFrame.from_dict(ts_data, orient="index")
        df = df.rename(columns=_AV_COLUMN_MAP)

        # Parse index as datetime
        df.index = pd.to_datetime(df.index)

        # AV returns ET timestamps — ensure tz-naive ET
        if df.index.tz is not None:
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)

        # Convert to numeric
        for col in _REQUIRED_OHLCV:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Keep only OHLCV columns
        available = [c for c in _REQUIRED_OHLCV if c in df.columns]
        if len(available) < len(_REQUIRED_OHLCV):
            logger.warning(
                "AV response for %s %s missing columns: %s",
                ticker,
                month_str,
                set(_REQUIRED_OHLCV) - set(available),
            )
            return None

        df = df[_REQUIRED_OHLCV].copy()
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df = df.dropna(subset=_REQUIRED_OHLCV)

        # Filter to RTH only (09:30-16:00 ET)
        if len(df) > 0:
            import datetime as _dt

            rth_start = _dt.time(9, 30)
            rth_end = _dt.time(16, 0)
            times = df.index.time
            rth_mask = (times >= rth_start) & (times < rth_end)
            df = df[rth_mask]

        if len(df) == 0:
            return None

        logger.debug(
            "Fetched %d bars for %s %s (%s)",
            len(df),
            ticker,
            month_str,
            timeframe,
        )
        return df

    def fetch_range(
        self,
        ticker: str,
        timeframe: str,
        start_year: int,
        end_year: int,
        start_month: int = 1,
        end_month: int = 12,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch a date range by iterating months.  Concatenates and deduplicates.

        Args:
            ticker: Stock ticker symbol.
            timeframe: Canonical timeframe code.
            start_year: First year to fetch (inclusive).
            end_year: Last year to fetch (inclusive).
            start_month: First month in start_year (default 1).
            end_month: Last month in end_year (default 12).

        Returns:
            Combined DataFrame or ``None`` if no data was fetched.
        """
        all_chunks = []
        total_requests = 0
        empty_months = 0

        for year in range(start_year, end_year + 1):
            m_start = start_month if year == start_year else 1
            m_end = end_month if year == end_year else 12

            for month in range(m_start, m_end + 1):
                chunk = self.fetch_month(ticker, timeframe, year, month)
                total_requests += 1

                if chunk is not None and len(chunk) > 0:
                    all_chunks.append(chunk)
                    empty_months = 0
                else:
                    empty_months += 1

                # If we've seen many empty months, we may have hit the
                # boundary of available data — stop early
                if empty_months >= 6:
                    logger.info(
                        "AV %s: 6 consecutive empty months at %04d-%02d, "
                        "stopping early",
                        ticker,
                        year,
                        month,
                    )
                    break

                time.sleep(self.pace)

            if empty_months >= 6:
                break

        if not all_chunks:
            logger.warning(
                "AV returned no data for %s %s (%d-%d)",
                ticker,
                timeframe,
                start_year,
                end_year,
            )
            return None

        combined = pd.concat(all_chunks).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = combined.dropna(subset=_REQUIRED_OHLCV)

        logger.info(
            "AV fetch complete: %s %s — %d bars from %d requests "
            "(%s to %s)",
            ticker,
            timeframe,
            len(combined),
            total_requests,
            combined.index.min().date(),
            combined.index.max().date(),
        )

        return combined if len(combined) > 0 else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request_with_retry(
        self,
        params: dict,
        ticker: str,
        context: str,
    ) -> Optional[dict]:
        """
        Issue an HTTP GET with exponential backoff on rate-limit errors.

        Backoff schedule: 10s, 30s, 60s, 120s.

        Returns:
            Parsed JSON dict, or ``None`` on unrecoverable failure.
        """
        backoff_schedule = [10, 30, 60, 120]

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.get(
                    self.BASE_URL,
                    params=params,
                    timeout=30,
                )

                if resp.status_code == 429:
                    wait = backoff_schedule[min(attempt, len(backoff_schedule) - 1)]
                    logger.warning(
                        "AV rate limit for %s %s (attempt %d/%d), "
                        "backing off %ds",
                        ticker,
                        context,
                        attempt + 1,
                        self.max_retries + 1,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()

                data = resp.json()

                # AV sometimes returns a "Note" for rate limits without 429
                if "Note" in data and "call frequency" in data.get("Note", "").lower():
                    wait = backoff_schedule[min(attempt, len(backoff_schedule) - 1)]
                    logger.warning(
                        "AV soft rate limit for %s %s: %s — backing off %ds",
                        ticker,
                        context,
                        data["Note"][:100],
                        wait,
                    )
                    time.sleep(wait)
                    continue

                return data

            except requests.exceptions.Timeout:
                logger.warning(
                    "AV timeout for %s %s (attempt %d/%d)",
                    ticker,
                    context,
                    attempt + 1,
                    self.max_retries + 1,
                )
            except requests.exceptions.ConnectionError as exc:
                logger.warning(
                    "AV connection error for %s %s: %s (attempt %d/%d)",
                    ticker,
                    context,
                    exc,
                    attempt + 1,
                    self.max_retries + 1,
                )
            except requests.exceptions.HTTPError as exc:
                logger.error(
                    "AV HTTP error for %s %s: %s",
                    ticker,
                    context,
                    exc,
                )
                return None
            except ValueError as exc:
                logger.error(
                    "AV JSON parse error for %s %s: %s",
                    ticker,
                    context,
                    exc,
                )
                return None

            if attempt < self.max_retries:
                wait = backoff_schedule[min(attempt, len(backoff_schedule) - 1)]
                time.sleep(wait)

        logger.error(
            "AV request failed after %d attempts for %s %s",
            self.max_retries + 1,
            ticker,
            context,
        )
        return None
