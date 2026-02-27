"""
Alpaca Markets intraday data provider.

Downloads intraday OHLCV data using the Alpaca Markets free API.

Data characteristics:
    - Free tier: 200 req/min, 10+ year depth
    - SIP consolidated feed (full market data)
    - Split + dividend adjusted
    - Timestamps returned in UTC — converted to tz-naive ET
    - RTH only (extended hours filtered out)

Rate limiting:
    - Free tier: 200 requests/minute
    - Default pace of 0.35s/request yields ~170 req/min (safe margin)
    - Exponential backoff on HTTP 429: 10s, 30s, 60s, 120s
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical OHLCV columns expected by the quant engine
_REQUIRED_OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# Alpaca TimeFrame configuration
_ALPACA_TIMEFRAMES = {
    "1m": ("1", "Min"),
    "5m": ("5", "Min"),
    "15m": ("15", "Min"),
    "30m": ("30", "Min"),
    "1h": ("1", "Hour"),
    "4h": ("4", "Hour"),
}

# Max calendar days per Alpaca request to stay under the ~10,000 bar limit.
# Calendar days include weekends/holidays (no bars), so effective trading days ≈ 70%.
_CHUNK_DAYS = {
    "1m": 14,      # ~10 trading days * 390 bars = ~3,900 bars
    "5m": 45,      # ~32 trading days * 78 bars = ~2,500 bars
    "15m": 120,    # ~85 trading days * 26 bars = ~2,200 bars
    "30m": 180,    # ~127 trading days * 13 bars = ~1,650 bars
    "1h": 365,     # ~252 trading days * 7 bars = ~1,764 bars
    "4h": 730,     # ~504 trading days * 2 bars = ~1,008 bars
}

# Maximum consecutive empty/failed chunks before giving up on a ticker
_MAX_CONSEC_FAIL = 5


class AlpacaProvider:
    """
    Alpaca Markets intraday data provider with chunked downloading.

    Handles rate limiting with exponential backoff, timezone conversion,
    and RTH filtering.

    Args:
        api_key: Alpaca API key.
        api_secret: Alpaca API secret.
        pace: Seconds between requests (0.35 = ~170 req/min, under 200 limit).
        max_retries: Maximum retries per chunk on rate-limit errors.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        pace: float = 0.35,
        max_retries: int = 4,
    ):
        if not api_key or not api_secret:
            raise ValueError(
                "Alpaca API key and secret are required. "
                "Sign up free at https://app.alpaca.markets/signup"
            )
        self.api_key = api_key
        self.api_secret = api_secret
        self.pace = pace
        self.max_retries = max_retries
        self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_range(
        self,
        ticker: str,
        timeframe: str,
        target_days: int,
    ) -> Optional[pd.DataFrame]:
        """
        Download intraday data with chunking, rate limit recovery, and tz handling.

        Args:
            ticker: Stock ticker symbol (e.g. ``"AAPL"``).
            timeframe: Canonical timeframe (``"1m"``, ``"5m"``, etc.).
            target_days: Total calendar days of history to fetch.

        Returns:
            Combined DataFrame with DatetimeIndex (tz-naive ET) and OHLCV columns,
            or ``None`` if no data could be fetched.
        """
        client = self._get_client()
        tf_obj = self._build_timeframe(timeframe)
        if tf_obj is None:
            return None

        chunk_days = _CHUNK_DAYS.get(timeframe, 10)

        all_chunks: List[pd.DataFrame] = []
        remaining_days = target_days
        current_end = datetime.now(timezone.utc)
        consec_failures = 0

        while remaining_days > 0:
            days_this = min(chunk_days, remaining_days)
            chunk_start = current_end - timedelta(days=days_this)

            chunk_df = self._fetch_chunk(
                client,
                ticker,
                tf_obj,
                chunk_start,
                current_end,
            )

            if chunk_df is not None and len(chunk_df) > 0:
                all_chunks.append(chunk_df)
                consec_failures = 0
            else:
                consec_failures += 1

            if consec_failures >= _MAX_CONSEC_FAIL:
                logger.warning(
                    "Alpaca %s %s: %d consecutive failures, stopping",
                    ticker,
                    timeframe,
                    consec_failures,
                )
                break

            current_end = chunk_start - timedelta(days=1)
            remaining_days -= days_this
            time.sleep(self.pace)

        if not all_chunks:
            logger.warning(
                "Alpaca returned no data for %s %s (%d days requested)",
                ticker,
                timeframe,
                target_days,
            )
            return None

        combined = pd.concat(all_chunks).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = combined.dropna(
            subset=[c for c in _REQUIRED_OHLCV if c in combined.columns]
        )

        # Filter to RTH only (09:30-16:00 ET)
        combined = self._filter_rth(combined)

        if combined is None or len(combined) == 0:
            return None

        logger.info(
            "Alpaca fetch complete: %s %s — %d bars (%s to %s)",
            ticker,
            timeframe,
            len(combined),
            combined.index.min().date(),
            combined.index.max().date(),
        )

        return combined

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        """Lazily build and cache the Alpaca client."""
        if self._client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            self._client = StockHistoricalDataClient(
                self.api_key,
                self.api_secret,
            )
        return self._client

    def _build_timeframe(self, timeframe: str):
        """Convert canonical timeframe to Alpaca TimeFrame object."""
        config = _ALPACA_TIMEFRAMES.get(timeframe)
        if config is None:
            logger.error(
                "Unsupported timeframe %r for Alpaca. Supported: %s",
                timeframe,
                ", ".join(sorted(_ALPACA_TIMEFRAMES.keys())),
            )
            return None

        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        amount_str, unit_str = config
        unit_map = {"Min": TimeFrameUnit.Minute, "Hour": TimeFrameUnit.Hour}
        return TimeFrame(int(amount_str), unit_map[unit_str])

    def _fetch_chunk(
        self,
        client,
        ticker: str,
        tf_obj,
        start: datetime,
        end: datetime,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch a single chunk with exponential backoff on rate limits.

        Returns:
            DataFrame chunk or ``None`` on failure.
        """
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest

        for attempt in range(self.max_retries + 1):
            try:
                request_params = StockBarsRequest(
                    symbol_or_symbols=[ticker],
                    timeframe=tf_obj,
                    start=start,
                    end=end,
                    adjustment=Adjustment.ALL,
                    feed=DataFeed.SIP,
                )
                bars = client.get_stock_bars(request_params)

                if bars and bars.data and ticker in bars.data:
                    bar_list = bars.data[ticker]
                    if bar_list:
                        return self._bars_to_dataframe(bar_list)

                # Empty response — not a failure, just no data for this range
                return None

            except Exception as exc:
                err_str = str(exc).lower()
                is_rate_limit = (
                    "rate" in err_str
                    or "429" in err_str
                    or "limit" in err_str
                    or "too many" in err_str
                )

                if is_rate_limit and attempt < self.max_retries:
                    wait = self._handle_rate_limit(attempt)
                    logger.warning(
                        "Alpaca rate limit for %s (attempt %d/%d), "
                        "backing off %.0fs",
                        ticker,
                        attempt + 1,
                        self.max_retries + 1,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                logger.error(
                    "Alpaca fetch error for %s chunk %s-%s: %s "
                    "(attempt %d/%d)",
                    ticker,
                    start.strftime("%Y-%m-%d"),
                    end.strftime("%Y-%m-%d"),
                    exc,
                    attempt + 1,
                    self.max_retries + 1,
                )
                return None

        return None

    def _handle_rate_limit(self, retry_count: int) -> float:
        """
        Calculate exponential backoff delay.

        Schedule: 10s, 30s, 60s, 120s.  Max 4 retries.

        Args:
            retry_count: Zero-based retry attempt number.

        Returns:
            Number of seconds to wait.
        """
        backoff_schedule = [10.0, 30.0, 60.0, 120.0]
        return backoff_schedule[min(retry_count, len(backoff_schedule) - 1)]

    def _bars_to_dataframe(self, bar_list) -> pd.DataFrame:
        """
        Convert Alpaca bar objects to a canonical DataFrame.

        Timestamps are converted from UTC to tz-naive ET.
        """
        records = []
        for bar in bar_list:
            records.append({
                "Date": bar.timestamp,
                "Open": float(bar.open),
                "High": float(bar.high),
                "Low": float(bar.low),
                "Close": float(bar.close),
                "Volume": int(bar.volume),
            })

        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df.set_index("Date")

        # Alpaca returns UTC — convert to tz-naive Eastern Time
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)

        return df

    @staticmethod
    def _filter_rth(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Filter to Regular Trading Hours only (09:30-16:00 ET).

        Alpaca returns extended-hours bars by default; strip them to match
        IBKR RTH convention and avoid quality gate quarantine.
        """
        if df is None or len(df) == 0:
            return df

        import datetime as _dt

        rth_start = _dt.time(9, 30)
        rth_end = _dt.time(16, 0)
        times = df.index.time
        rth_mask = (times >= rth_start) & (times < rth_end)

        pre_filter = len(df)
        df = df[rth_mask]
        dropped = pre_filter - len(df)

        if dropped > 0:
            logger.debug(
                "Filtered %d extended-hours bars (%d remaining)",
                dropped,
                len(df),
            )

        return df if len(df) > 0 else None
