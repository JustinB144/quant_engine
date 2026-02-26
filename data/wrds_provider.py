"""
wrds_provider.py
================
WRDS (Wharton Research Data Services) data provider.

Gives the auto-discovery pipeline access to:
  • CRSP daily prices — survivorship-bias-free, all stocks ever listed
  • CRSP dsp500list  — exact S&P 500 membership on any historical date
  • Compustat fundq  — quarterly fundamentals with rdq (earnings announcement date)
  • I/B/E/S         — earnings estimates, actuals, surprises
  • TFN s34         — 13F institutional ownership filings

Credentials are stored in ~/.pgpass (created automatically on first login).
No password prompts after initial setup.

Usage:
    from research.data_providers.wrds_provider import WRDSProvider

    provider = WRDSProvider()
    if provider.available():
        # Survivorship-bias-free S&P 500 universe as of Jan 1, 2005
        tickers = provider.get_sp500_universe(as_of_date='2005-01-01')

        # 20 years of daily prices for the full universe
        prices = provider.get_crsp_prices(permnos, start='2005-01-01', end='2025-01-01')

        # Quarterly earnings fundamentals
        fundamentals = provider.get_fundamentals(tickers, start='2005-01-01')

        # Earnings surprise signals
        surprises = provider.get_earnings_surprises(tickers, start='2005-01-01')
"""

import logging
import os
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional: NYSE trading calendar for accurate holiday skipping in TAQmsec queries.
# Without this, pd.bdate_range (Mon-Fri) is used — still works but sends ~30
# extra no-op queries per year (holidays that have no TAQmsec table).
try:
    import pandas_market_calendars as _mcal
    _NYSE_CALENDAR = _mcal.get_calendar('NYSE')
    _NYSE_CALENDAR_AVAILABLE = True
except ImportError:
    _NYSE_CALENDAR = None
    _NYSE_CALENDAR_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Connection singleton (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────

_wrds_connection = None
_wrds_lock = threading.Lock()
_WRDS_USERNAME = os.environ.get('WRDS_USERNAME')
if not _WRDS_USERNAME:
    logger.debug(
        'WRDS_USERNAME env var not set; WRDS will be unavailable. '
        'Set it via: export WRDS_USERNAME=<your_username>'
    )

import re as _re
_TICKER_RE = _re.compile(r'^[A-Z0-9.\-/^]{1,12}$')
_PERMNO_RE = _re.compile(r'^\d{1,10}$')

def _sanitize_ticker_list(tickers: list) -> str:
    """Build a SQL-safe IN-clause string from ticker symbols.

    Validates each ticker against a strict alphanumeric+punctuation regex
    to prevent SQL injection when tickers are interpolated into queries.
    """
    clean = []
    for t in tickers:
        t_upper = str(t).upper().strip()
        if _TICKER_RE.match(t_upper):
            clean.append(t_upper)
    if not clean:
        return "'__NONE__'"
    return "'" + "','".join(clean) + "'"


def _sanitize_permno_list(permnos: list) -> str:
    """Build a SQL-safe IN-clause string from PERMNO values."""
    clean = []
    for p in permnos:
        p_str = str(p).strip()
        if _PERMNO_RE.match(p_str):
            clean.append(str(int(p_str)))
    if not clean:
        return "-1"
    return ",".join(clean)
_WRDS_HOST = 'wrds-pgdata.wharton.upenn.edu'
_WRDS_PORT = '9737'
_WRDS_DB = 'wrds'


def _read_pgpass_password() -> str | None:
    """
    Read the WRDS password from ~/.pgpass so the wrds library doesn't
    prompt interactively. SQLAlchemy doesn't honour .pgpass natively
    (that's a libpq feature), so we parse it ourselves.

    .pgpass format: hostname:port:database:username:password
    """
    import os
    pgpass_path = os.path.expanduser('~/.pgpass')
    if not os.path.exists(pgpass_path):
        return None
    try:
        with open(pgpass_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(':')
                if len(parts) < 5:
                    continue
                host, port, db, user = parts[0], parts[1], parts[2], parts[3]
                pw = ':'.join(parts[4:])  # password may contain colons
                # Match against WRDS credentials (support wildcard '*')
                if ((host == '*' or host == _WRDS_HOST)
                        and (port == '*' or port == _WRDS_PORT)
                        and (db == '*' or db == _WRDS_DB)
                        and (user == '*' or user == _WRDS_USERNAME)):
                    return pw
    except (OSError, PermissionError, UnicodeDecodeError) as e:
        logger.debug("Could not read .pgpass: %s", e)
    return None


def _get_connection():
    """Get or create a cached WRDS connection. Returns None if unavailable.

    Thread-safe: uses a module-level lock to prevent concurrent initialization.
    """
    global _wrds_connection

    if not _WRDS_USERNAME:
        return None

    # Fast path: connection already exists and is alive
    if _wrds_connection is not None:
        try:
            _wrds_connection.raw_sql('SELECT 1')
            return _wrds_connection
        except (OSError, ValueError, RuntimeError):
            _wrds_connection = None

    with _wrds_lock:
        # Double-check after acquiring lock (another thread may have initialized)
        if _wrds_connection is not None:
            try:
                _wrds_connection.raw_sql('SELECT 1')
                return _wrds_connection
            except (OSError, ValueError, RuntimeError):
                _wrds_connection = None

        try:
            import sqlalchemy as sa
            import urllib.parse

            pgpass_pw = _read_pgpass_password()
            if pgpass_pw:
                pw_enc = urllib.parse.quote(pgpass_pw)
                uri = (f"postgresql://{_WRDS_USERNAME}:{pw_enc}"
                       f"@{_WRDS_HOST}:{_WRDS_PORT}/{_WRDS_DB}")
                try:
                    engine = sa.create_engine(
                        uri, isolation_level="AUTOCOMMIT",
                        connect_args={"sslmode": "require", "connect_timeout": 15},
                    )
                    test_conn = engine.connect()
                    test_conn.execute(sa.text("SELECT 1"))
                    test_conn.close()

                    import wrds
                    _wrds_connection = wrds.Connection(autoconnect=False)
                    _wrds_connection.engine = engine
                    _wrds_connection.connection = engine.connect()
                    _wrds_connection._username = _WRDS_USERNAME
                    try:
                        _wrds_connection.load_library_list()
                    except (AttributeError, RuntimeError):
                        pass  # library list is optional
                    return _wrds_connection
                except (sa.exc.OperationalError, sa.exc.DatabaseError, ConnectionError) as e:
                    logger.warning("WRDS direct connection failed: %s", e)

            import sys
            if sys.stdin.isatty():
                import wrds
                _wrds_connection = wrds.Connection(wrds_username=_WRDS_USERNAME)
                return _wrds_connection
            else:
                logger.info("WRDS: No .pgpass password and not interactive — skipping.")
                return None
        except (ImportError, ConnectionError, OSError) as e:
            logger.warning("WRDS connection failed: %s", e)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Main Provider Class
# ─────────────────────────────────────────────────────────────────────────────

class WRDSProvider:
    """
    WRDS data provider for the auto-discovery pipeline.

    All methods return empty DataFrames/lists gracefully if WRDS is
    unavailable, so the pipeline continues using IBKR/local cache fallbacks.
    """

    def __init__(self):
        """Initialize WRDSProvider."""
        self._db = _get_connection()
        if self._db is None:
            logger.info('WRDS not available — using IBKR/local cache fallbacks')

    def available(self) -> bool:
        """Return whether the resource is available in the current runtime."""
        return self._db is not None

    def _query(self, sql: str) -> pd.DataFrame:
        """Run a SQL query and return a DataFrame. Returns empty df on error."""
        if self._db is None:
            return pd.DataFrame()
        try:
            return self._db.raw_sql(sql)
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning('WRDS query error: %s', e)
            return pd.DataFrame()

    def _query_silent(self, sql: str) -> pd.DataFrame:
        """Run a SQL query but suppress probing errors (used for optional tables)."""
        if self._db is None:
            return pd.DataFrame()
        try:
            return self._db.raw_sql(sql)
        except (OSError, ValueError, RuntimeError):
            return pd.DataFrame()

    # ─────────────────────────────────────────────────────────────────────────
    # S&P 500 Universe — survivorship-bias-free
    # ─────────────────────────────────────────────────────────────────────────

    def get_sp500_universe(
        self,
        as_of_date: str = None,
        include_permno: bool = False,
    ) -> List[str]:
        """
        Return the list of tickers that were IN the S&P 500 on a specific date.

        This is the core survivorship-bias fix: instead of using today's 500
        survivors, we get whoever was actually in the index on the backtest
        start date — including companies that later went bankrupt, were acquired,
        or delisted.

        Args:
            as_of_date:     'YYYY-MM-DD'. Default: today.
            include_permno: If True, return list of (permno, ticker) tuples.

        Returns:
            List of ticker strings (or list of (permno, ticker) tuples).

        Example:
            # Who was in the S&P 500 at the start of the 2008 financial crisis?
            universe = provider.get_sp500_universe('2008-01-01')
            # → ~500 tickers including Lehman Brothers, Bear Stearns, etc.
        """
        date = as_of_date or datetime.now().strftime('%Y-%m-%d')

        # Strategy 1: crsp_a_stock.dsp500list (requires crsp_a_indexes schema)
        # Strategy 2: comp.idxcst_his — Compustat index constituent history (fallback)
        # Strategy 3: crsp.dsf broad universe — just grab actively traded stocks (last resort)

        # ── Try CRSP dsp500list first ──
        sql_crsp = f"""
            SELECT DISTINCT a.permno, b.ticker
            FROM crsp.dsp500list AS a
            JOIN crsp.msenames AS b
              ON a.permno = b.permno
             AND b.namedt  <= '{date}'::date
             AND (b.nameendt >= '{date}'::date OR b.nameendt IS NULL)
            WHERE a.start  <= '{date}'::date
              AND (a.ending >= '{date}'::date OR a.ending IS NULL)
            ORDER BY b.ticker
        """
        df = self._query(sql_crsp)

        if df.empty:
            # ── Fallback: Compustat index constituent history ──
            # comp.idxcst_his tracks S&P 500 membership (gvkeyx = '000003')
            # Join through comp.fundq (not comp.company) to get tickers —
            # comp.fundq.tic is confirmed to work on this WRDS subscription.
            sql_comp = f"""
                SELECT DISTINCT b.tic AS ticker
                FROM comp.idxcst_his AS a
                JOIN (SELECT DISTINCT gvkey, tic FROM comp.fundq
                      WHERE tic IS NOT NULL) AS b
                  ON a.gvkey = b.gvkey
                WHERE a.gvkeyx = '000003'
                  AND a."from" <= '{date}'::date
                  AND (a.thru >= '{date}'::date OR a.thru IS NULL)
                ORDER BY b.tic
            """
            df2 = self._query(sql_comp)
            if not df2.empty:
                tickers_out = df2['ticker'].dropna().str.strip().tolist()
                return [t for t in tickers_out if t]

            # ── Last resort: actively traded large-caps from CRSP ──
            sql_broad = f"""
                SELECT DISTINCT b.ticker
                FROM crsp.dsf AS a
                JOIN crsp.msenames AS b
                  ON a.permno = b.permno
                 AND b.namedt  <= '{date}'::date
                 AND (b.nameendt >= '{date}'::date OR b.nameendt IS NULL)
                WHERE a.date BETWEEN ('{date}'::date - INTERVAL '30 days') AND '{date}'::date
                  AND a.prc IS NOT NULL
                  AND ABS(a.prc) * a.vol > 5000000
                  AND b.shrcd IN (10, 11)
                  AND b.exchcd IN (1, 2, 3)
                ORDER BY b.ticker
                LIMIT 500
            """
            df3 = self._query(sql_broad)
            if not df3.empty:
                tickers_out = df3['ticker'].dropna().str.strip().tolist()
                return [t for t in tickers_out if t]

            return []

        if include_permno:
            return list(zip(df['permno'].tolist(), df['ticker'].tolist()))

        tickers_out = df['ticker'].dropna().str.strip().tolist()
        return [t for t in tickers_out if t]

    def get_sp500_history(
        self,
        start_date: str = '2000-01-01',
        end_date: str = None,
        freq: str = 'annual',
    ) -> pd.DataFrame:
        """
        Return the full history of S&P 500 constituents over a date range.

        Useful for building a rolling universe: at each rebalance date, use
        whoever was in the index then.

        Args:
            start_date: 'YYYY-MM-DD'
            end_date:   'YYYY-MM-DD' (default: today)
            freq:       'annual' or 'quarterly' — how often to snapshot

        Returns:
            DataFrame with columns: date, permno, ticker
            Each row = one stock that was in the index on that snapshot date.
        """
        end = end_date or datetime.now().strftime('%Y-%m-%d')

        # Generate snapshot dates
        dates = pd.date_range(start=start_date, end=end,
                              freq='YS' if freq == 'annual' else 'QS')

        all_frames = []
        for snap_date in dates:
            d = snap_date.strftime('%Y-%m-%d')
            members = self.get_sp500_universe(as_of_date=d, include_permno=True)
            if members:
                frame = pd.DataFrame(members, columns=['permno', 'ticker'])
                frame['date'] = d
                all_frames.append(frame)

        if not all_frames:
            return pd.DataFrame(columns=['date', 'permno', 'ticker'])

        return pd.concat(all_frames, ignore_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # CRSP Price History — survivorship-bias-free daily prices
    # ─────────────────────────────────────────────────────────────────────────

    def resolve_permno(
        self,
        ticker: str,
        as_of_date: str = None,
    ) -> Optional[str]:
        """
        Resolve a ticker to the active CRSP PERMNO on a date.
        """
        t = str(ticker).upper().strip()
        if not _TICKER_RE.match(t):
            return None
        ref_date = as_of_date or datetime.now().strftime('%Y-%m-%d')
        sql = f"""
            SELECT b.permno
            FROM crsp.msenames AS b
            WHERE b.ticker = '{t}'
              AND b.namedt  <= '{ref_date}'::date
              AND (b.nameendt >= '{ref_date}'::date OR b.nameendt IS NULL)
            ORDER BY b.nameendt NULLS FIRST, b.namedt DESC
            LIMIT 1
        """
        df = self._query(sql)
        if df.empty:
            return None
        try:
            return str(int(df["permno"].iloc[0]))
        except (ValueError, KeyError, TypeError):
            return None

    def get_crsp_prices(
        self,
        tickers: List[str],
        start_date: str = '2000-01-01',
        end_date: str = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch daily OHLCV-equivalent data from CRSP.

        CRSP covers every stock ever listed on NYSE/AMEX/NASDAQ since 1926,
        including those that went bankrupt, were acquired, or delisted.
        The `ret` column includes delisting returns so you never understate
        a strategy's losses from holding a stock that went to zero.

        Args:
            tickers:    List of ticker symbols OR PERMNO identifiers
            start_date: 'YYYY-MM-DD'
            end_date:   'YYYY-MM-DD' (default: today)

        Returns:
            Dict mapping PERMNO(string) → DataFrame(Date, Open, High, Low, Close,
                                                    Volume, Return, total_ret,
                                                    dlret, delist_event, permno, ticker)
            Note: Uses openprc/askhi/bidlo when available from CRSP, falls
            back to prc (close) when not.  CRSP prices are negative when
            they represent bid/ask midpoints — .abs() is applied.
        """
        end = end_date or datetime.now().strftime('%Y-%m-%d')
        if not tickers:
            return {}

        all_permno = all(_PERMNO_RE.match(str(t).strip()) for t in tickers)
        if all_permno:
            permno_list = _sanitize_permno_list(tickers)
            where_clause = f"a.permno IN ({permno_list})"
        else:
            ticker_list = _sanitize_ticker_list(tickers)
            where_clause = f"b.ticker IN ({ticker_list})"

        sql = f"""
            SELECT a.permno, a.date, a.prc, a.openprc, a.askhi, a.bidlo,
                   a.ret, a.vol, a.shrout,
                   b.ticker, b.comnam
            FROM crsp.dsf AS a
            JOIN crsp.msenames AS b
              ON a.permno = b.permno
             AND b.namedt  <= a.date
             AND (b.nameendt >= a.date OR b.nameendt IS NULL)
            WHERE {where_clause}
              AND a.date >= '{start_date}'::date
              AND a.date <= '{end}'::date
            ORDER BY a.permno, a.date
        """
        df = self._query(sql)

        if df.empty:
            return {}

        # CRSP price is negative when it's a bid-ask midpoint (no trade) — take abs.
        # Use .loc[:, col] assignments and .copy() for pandas 3.0 CoW compatibility.
        df = df.copy()
        df.loc[:, 'prc'] = df['prc'].abs()
        df.loc[:, 'openprc'] = pd.to_numeric(df['openprc'], errors='coerce').abs()
        df.loc[:, 'askhi'] = pd.to_numeric(df['askhi'], errors='coerce').abs()
        df.loc[:, 'bidlo'] = pd.to_numeric(df['bidlo'], errors='coerce').abs()
        df.loc[:, 'date'] = pd.to_datetime(df['date'])
        df.loc[:, 'ret'] = pd.to_numeric(df['ret'], errors='coerce')
        df.loc[:, 'vol'] = pd.to_numeric(df['vol'], errors='coerce')

        result: Dict[str, pd.DataFrame] = {}
        for permno, group in df.groupby('permno'):
            g = group.copy()
            g.index = pd.DatetimeIndex(g.pop('date'))
            g = g.sort_index()
            permno_key = str(int(permno))
            ticker_val = (
                g["ticker"].dropna().astype(str).iloc[-1]
                if "ticker" in g.columns and g["ticker"].notna().any()
                else ""
            )
            ohlcv = pd.DataFrame({
                'Open':   g['openprc'].fillna(g['prc']),
                'High':   g['askhi'].fillna(g['prc']),
                'Low':    g['bidlo'].fillna(g['prc']),
                'Close':  g['prc'],
                'Volume': g['vol'].fillna(0),
                'Return': g['ret'],    # daily return including dividends
                'total_ret': g['ret'],  # default = ret; delisting path enriches this
                'dlret': np.nan,
                'delist_event': 0,
                'permno': permno_key,
                'ticker': ticker_val,
            })
            ohlcv.index.name = 'Date'
            result[permno_key] = ohlcv

        return result

    def get_crsp_prices_with_delistings(
        self,
        tickers: List[str],
        start_date: str = '2000-01-01',
        end_date: str = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Same as get_crsp_prices but integrates delisting returns into
        a dedicated `total_ret` stream, without inserting synthetic OHLC bars.

        This is critical for unbiased backtesting: if a strategy holds a
        stock that later gets delisted (e.g. bankruptcy), the backtest must
        include the terminal loss via returns, not by fabricating 0-price bars.
        """
        prices = self.get_crsp_prices(tickers, start_date, end_date)

        end = end_date or datetime.now().strftime('%Y-%m-%d')
        all_permno = all(_PERMNO_RE.match(str(t).strip()) for t in tickers)
        if all_permno:
            permno_list = _sanitize_permno_list(tickers)
            where_clause = f"a.permno IN ({permno_list})"
        else:
            ticker_list = _sanitize_ticker_list(tickers)
            where_clause = f"b.ticker IN ({ticker_list})"

        # Fetch delisting info
        sql = f"""
            SELECT a.permno, a.dlstdt, a.dlret, a.dlstcd,
                   b.ticker
            FROM crsp.dsedelist AS a
            JOIN crsp.msenames AS b
              ON a.permno = b.permno
             AND b.namedt  <= a.dlstdt
             AND (b.nameendt >= a.dlstdt OR b.nameendt IS NULL)
            WHERE {where_clause}
              AND a.dlstdt >= '{start_date}'::date
              AND a.dlstdt <= '{end}'::date
              AND a.dlret IS NOT NULL
        """
        delist_df = self._query(sql)

        if not delist_df.empty:
            delist_df['dlstdt'] = pd.to_datetime(delist_df['dlstdt'])
            delist_df['dlret'] = pd.to_numeric(delist_df['dlret'], errors='coerce')
            for _, row in delist_df.iterrows():
                permno_key = str(int(row['permno']))
                if permno_key in prices:
                    df = prices[permno_key]
                    if len(df) == 0:
                        continue
                    dlret = row['dlret']
                    if pd.isna(dlret):
                        continue
                    event_dt = pd.Timestamp(row['dlstdt'])
                    if event_dt in df.index:
                        eff_dt = event_dt
                    else:
                        prior = df.index[df.index <= event_dt]
                        if len(prior) == 0:
                            continue
                        eff_dt = pd.Timestamp(prior.max())

                    base_ret = pd.to_numeric(pd.Series([df.at[eff_dt, 'Return']]), errors='coerce').iloc[0]
                    if pd.isna(base_ret):
                        base_ret = 0.0

                    total_ret = (1.0 + float(base_ret)) * (1.0 + float(dlret)) - 1.0
                    df.at[eff_dt, 'dlret'] = float(dlret)
                    df.at[eff_dt, 'total_ret'] = float(total_ret)
                    df.at[eff_dt, 'delist_event'] = 1
                    if pd.notna(row.get('ticker', None)):
                        df.at[eff_dt, 'ticker'] = str(row['ticker']).strip()
                    prices[permno_key] = df.sort_index()

        return prices

    # ─────────────────────────────────────────────────────────────────────────
    # OptionMetrics (WRDS) — PERMNO-linked options surface factors
    # ─────────────────────────────────────────────────────────────────────────

    def get_optionmetrics_link(
        self,
        permnos: List[str],
        start_date: str = '2000-01-01',
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Resolve WRDS OptionMetrics secid link rows for PERMNOs.

        Tries common WRDS table variants; returns empty DataFrame if unavailable.
        """
        if not permnos:
            return pd.DataFrame(columns=["permno", "secid", "link_start", "link_end"])
        end = end_date or datetime.now().strftime('%Y-%m-%d')
        permno_list = _sanitize_permno_list(permnos)

        candidates = [
            ("optionm.crsp_link", "permno", "secid", "sdate", "edate"),
            ("optionm.om_crsp_link", "permno", "secid", "sdate", "edate"),
            ("optionm.security_link", "permno", "secid", "start_date", "end_date"),
        ]
        for table, perm_col, secid_col, start_col, end_col in candidates:
            sql = f"""
                SELECT {perm_col} AS permno,
                       {secid_col} AS secid,
                       {start_col} AS link_start,
                       {end_col} AS link_end
                FROM {table}
                WHERE {perm_col} IN ({permno_list})
                  AND {start_col} <= '{end}'::date
                  AND ({end_col} >= '{start_date}'::date OR {end_col} IS NULL)
            """
            df = self._query_silent(sql)
            if df.empty:
                continue
            df["permno"] = pd.to_numeric(df["permno"], errors="coerce")
            df["secid"] = pd.to_numeric(df["secid"], errors="coerce")
            df["link_start"] = pd.to_datetime(df["link_start"], errors="coerce")
            df["link_end"] = pd.to_datetime(df["link_end"], errors="coerce")
            df = df.dropna(subset=["permno", "secid", "link_start"])
            if df.empty:
                continue
            df["permno"] = df["permno"].astype(int).astype(str)
            df["secid"] = df["secid"].astype(int)
            return df[["permno", "secid", "link_start", "link_end"]].drop_duplicates()

        return pd.DataFrame(columns=["permno", "secid", "link_start", "link_end"])

    @staticmethod
    def _nearest_iv(
        group: pd.DataFrame,
        target_dte: int,
        cp_flag: Optional[str] = None,
        delta_target: Optional[float] = None,
    ) -> float:
        """Internal helper for nearest iv."""
        sub = group
        if cp_flag is not None:
            sub = sub[sub["cp_flag"] == str(cp_flag).upper()]
        if sub.empty:
            return np.nan

        score = (sub["dte"] - float(target_dte)).abs()
        if delta_target is not None and sub["delta"].notna().any():
            score = score + (sub["delta"].abs() - float(delta_target)).abs()

        idx = score.replace([np.inf, -np.inf], np.nan).dropna().index
        if len(idx) == 0:
            return np.nan
        best = sub.loc[idx].loc[score.loc[idx].idxmin()]
        return float(best["iv"]) if pd.notna(best["iv"]) else np.nan

    def get_option_surface_features(
        self,
        permnos: List[str],
        start_date: str = '2000-01-01',
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Build minimal daily option-surface features keyed by (permno, date).

        Output columns:
            iv_atm_30, iv_atm_60, iv_atm_90, iv_put_25d, iv_call_25d
        """
        empty = pd.DataFrame(
            columns=["iv_atm_30", "iv_atm_60", "iv_atm_90", "iv_put_25d", "iv_call_25d"],
        )
        if not permnos:
            return empty
        end = end_date or datetime.now().strftime('%Y-%m-%d')

        link = self.get_optionmetrics_link(
            permnos=permnos,
            start_date=start_date,
            end_date=end,
        )
        if link.empty:
            return empty

        secids = sorted({int(x) for x in pd.to_numeric(link["secid"], errors="coerce").dropna().astype(int).tolist()})
        if not secids:
            return empty
        secid_list = ",".join(str(s) for s in secids)

        query_candidates = [
            """
            SELECT date, secid, exdate, cp_flag,
                   impl_volatility AS iv,
                   delta
            FROM optionm.opprcd
            WHERE secid IN ({secid_list})
              AND date >= '{start_date}'::date
              AND date <= '{end}'::date
              AND exdate IS NOT NULL
              AND impl_volatility IS NOT NULL
            """,
            """
            SELECT date, secid, exdate, cp_flag,
                   impl_volatility AS iv,
                   NULL::double precision AS delta
            FROM optionm.opprcd
            WHERE secid IN ({secid_list})
              AND date >= '{start_date}'::date
              AND date <= '{end}'::date
              AND exdate IS NOT NULL
              AND impl_volatility IS NOT NULL
            """,
            """
            SELECT date, secid, exdate, cp_flag,
                   impl_vol AS iv,
                   delta
            FROM optionm.opprcd
            WHERE secid IN ({secid_list})
              AND date >= '{start_date}'::date
              AND date <= '{end}'::date
              AND exdate IS NOT NULL
              AND impl_vol IS NOT NULL
            """,
        ]

        opt = pd.DataFrame()
        for q in query_candidates:
            sql = q.format(secid_list=secid_list, start_date=start_date, end=end)
            opt = self._query_silent(sql)
            if not opt.empty:
                break
        if opt.empty:
            return empty

        opt["date"] = pd.to_datetime(opt["date"], errors="coerce")
        opt["exdate"] = pd.to_datetime(opt["exdate"], errors="coerce")
        opt["secid"] = pd.to_numeric(opt["secid"], errors="coerce")
        opt["iv"] = pd.to_numeric(opt["iv"], errors="coerce")
        opt["delta"] = pd.to_numeric(opt["delta"], errors="coerce")
        opt["cp_flag"] = opt["cp_flag"].astype(str).str.upper().str[:1]
        opt = opt.dropna(subset=["date", "exdate", "secid", "iv"])
        if opt.empty:
            return empty

        opt["dte"] = (opt["exdate"] - opt["date"]).dt.days
        opt = opt[(opt["dte"] >= 5) & (opt["dte"] <= 120) & (opt["iv"] > 0)]
        if opt.empty:
            return empty

        # Map secid/date -> permno via link date ranges (G2: validity gate).
        # When multiple secids overlap for the same PERMNO on a given date,
        # pick deterministically: longest overlap first, then most recent start.
        link = link.copy()
        link["link_end"] = pd.to_datetime(link["link_end"], errors="coerce")
        link["_span_days"] = (
            link["link_end"].fillna(pd.Timestamp.now()) - link["link_start"]
        ).dt.days
        mapped_rows = []
        for secid, g in opt.groupby("secid"):
            lk = link[link["secid"] == int(secid)]
            if lk.empty:
                continue
            lk = lk.sort_values(["_span_days", "link_start"], ascending=[False, False])
            for row in g.itertuples(index=False):
                m = lk[
                    (lk["link_start"] <= row.date)
                    & ((lk["link_end"].isna()) | (lk["link_end"] >= row.date))
                ]
                if m.empty:
                    continue
                permno = str(m.iloc[0]["permno"])
                mapped_rows.append(
                    {
                        "permno": permno,
                        "date": pd.Timestamp(row.date),
                        "cp_flag": row.cp_flag,
                        "iv": float(row.iv),
                        "delta": float(row.delta) if pd.notna(row.delta) else np.nan,
                        "dte": int(row.dte),
                    },
                )

        if not mapped_rows:
            return empty

        mapped = pd.DataFrame(mapped_rows)
        feature_rows = []
        for (permno, dt), g in mapped.groupby(["permno", "date"]):
            row = {
                "permno": str(permno),
                "date": pd.Timestamp(dt),
                "iv_atm_30": self._nearest_iv(g, target_dte=30),
                "iv_atm_60": self._nearest_iv(g, target_dte=60),
                "iv_atm_90": self._nearest_iv(g, target_dte=90),
                "iv_put_25d": self._nearest_iv(g, target_dte=30, cp_flag="P", delta_target=0.25),
                "iv_call_25d": self._nearest_iv(g, target_dte=30, cp_flag="C", delta_target=0.25),
            }
            feature_rows.append(row)

        if not feature_rows:
            return empty

        out = pd.DataFrame(feature_rows)
        out = out.set_index(["permno", "date"]).sort_index()
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # Compustat Fundamentals — quarterly, point-in-time via rdq
    # ─────────────────────────────────────────────────────────────────────────

    def get_fundamentals(
        self,
        tickers: List[str],
        start_date: str = '2000-01-01',
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Fetch quarterly fundamentals from Compustat with point-in-time dates.

        The `rdq` column is the earnings announcement date — the date the
        information became publicly available. Using rdq instead of datadate
        (the fiscal period end) prevents look-ahead bias: we only use data
        that was actually available to a trader on each signal date.

        Returns:
            DataFrame with MultiIndex (rdq, tic) and columns:
                datadate, fyearq, fqtr,
                revtq    — quarterly revenue
                ebitdaq  — EBITDA (if available; else computed as ni+da)
                niq      — net income
                epspxq   — EPS (excl. extraordinary items)
                atq      — total assets
                ceqq     — common equity
                dlttq    — long-term debt
                prccq    — stock price at fiscal quarter end
                pe_ratio — trailing P/E (prccq / epspxq annualised)
                debt_equity — dlttq / ceqq
        """
        end = end_date or datetime.now().strftime('%Y-%m-%d')
        if not tickers:
            return pd.DataFrame()

        ticker_list = _sanitize_ticker_list(tickers)

        sql = f"""
            SELECT gvkey, tic, conm, datadate, rdq, fyearq, fqtr,
                   revtq, niq, epspxq, atq, ceqq, dlttq, prccq,
                   oibdpq, saleq, cogsq, xsgaq, dpq
            FROM comp.fundq
            WHERE tic IN ({ticker_list})
              AND rdq  >= '{start_date}'::date
              AND rdq  <= '{end}'::date
              AND indfmt  = 'INDL'
              AND datafmt = 'STD'
              AND popsrc  = 'D'
              AND consol  = 'C'
              AND rdq IS NOT NULL
            ORDER BY tic, rdq
        """
        df = self._query(sql)

        if df.empty:
            return pd.DataFrame()

        df['rdq']      = pd.to_datetime(df['rdq'])
        df['datadate'] = pd.to_datetime(df['datadate'])

        # Derived signals
        # P/E ratio: price / (4 * quarterly EPS) — forward annualised
        df['pe_ratio'] = np.where(
            (df['epspxq'] > 0) & df['prccq'].notna(),
            df['prccq'] / (df['epspxq'] * 4),
            np.nan,
        )
        # Debt/equity
        df['debt_equity'] = np.where(
            df['ceqq'] > 0,
            df['dlttq'] / df['ceqq'],
            np.nan,
        )
        # EBITDA (use reported oibdpq; fall back to niq + dpq)
        df['ebitdaq'] = df['oibdpq'].fillna(df['niq'] + df['dpq'].fillna(0))

        # Revenue growth QoQ (within ticker)
        df = df.sort_values(['tic', 'rdq'])
        df['rev_growth_qoq'] = df.groupby('tic')['revtq'].pct_change()

        return df.set_index(['rdq', 'tic']).sort_index()

    # ─────────────────────────────────────────────────────────────────────────
    # I/B/E/S Earnings Surprises
    # ─────────────────────────────────────────────────────────────────────────

    def get_earnings_surprises(
        self,
        tickers: List[str],
        start_date: str = '2000-01-01',
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Fetch quarterly earnings announcements and EPS surprises from I/B/E/S.

        Earnings surprise (actual EPS vs consensus estimate) is one of the
        most robust documented alpha signals — post-earnings announcement
        drift (PEAD) persists for weeks after the announcement.

        Returns:
            DataFrame indexed by (anndats_act, ticker) with columns:
                fpedats      — fiscal period end date
                actual       — actual EPS reported
                meanest      — consensus mean EPS estimate
                numest       — number of analyst estimates
                stdev        — estimate dispersion (std dev)
                surprise_pct — (actual - meanest) / abs(meanest) * 100
                beat         — True if surprise_pct > 0
        """
        end = end_date or datetime.now().strftime('%Y-%m-%d')
        if not tickers:
            return pd.DataFrame()

        ticker_list = _sanitize_ticker_list(tickers)

        sql = f"""
            SELECT ticker, fpedats, statpers, anndats_act,
                   actual, meanest, medest, stdev, numest, highest, lowest
            FROM ibes.statsum_epsus
            WHERE ticker IN ({ticker_list})
              AND fpi      = '1'
              AND fiscalp  = 'QTR'
              AND actual   IS NOT NULL
              AND anndats_act >= '{start_date}'::date
              AND anndats_act <= '{end}'::date
            ORDER BY ticker, anndats_act
        """
        df = self._query(sql)

        if df.empty:
            return pd.DataFrame()

        df['anndats_act'] = pd.to_datetime(df['anndats_act'])
        df['fpedats']     = pd.to_datetime(df['fpedats'])

        # Earnings surprise: how much did actual beat/miss the consensus?
        df['surprise_pct'] = np.where(
            df['meanest'].abs() > 1e-6,
            (df['actual'] - df['meanest']) / df['meanest'].abs() * 100,
            np.nan,
        )
        df['beat'] = df['surprise_pct'] > 0

        # Estimate dispersion (uncertainty before announcement)
        df['dispersion'] = np.where(
            df['meanest'].abs() > 1e-6,
            df['stdev'] / df['meanest'].abs(),
            np.nan,
        )

        return df.set_index(['anndats_act', 'ticker']).sort_index()

    # ─────────────────────────────────────────────────────────────────────────
    # Institutional Ownership (13F filings via TFN s34)
    # ─────────────────────────────────────────────────────────────────────────

    def get_institutional_ownership(
        self,
        tickers: List[str],
        start_date: str = '2000-01-01',
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Fetch quarterly institutional ownership from 13F filings (TFN s34).

        Institutional ownership changes are a useful signal:
        - Rising institutional ownership → "smart money" accumulation
        - Falling ownership → distribution phase
        - Very low ownership → potential undervalued/underfollowed stock

        Returns:
            DataFrame indexed by (fdate, ticker) with columns:
                total_shares_held — aggregate shares held by all institutions
                num_institutions  — number of reporting institutions
                pct_institutional — shares held / shares outstanding (approx)
        """
        end = end_date or datetime.now().strftime('%Y-%m-%d')
        if not tickers:
            return pd.DataFrame()

        ticker_list = _sanitize_ticker_list(tickers)

        sql = f"""
            SELECT fdate, ticker,
                   SUM(shares)          AS total_shares_held,
                   COUNT(DISTINCT mgrno) AS num_institutions,
                   AVG(prc)              AS avg_price
            FROM tfn.s34
            WHERE ticker IN ({ticker_list})
              AND fdate >= '{start_date}'::date
              AND fdate <= '{end}'::date
              AND shares > 0
            GROUP BY fdate, ticker
            ORDER BY ticker, fdate
        """
        df = self._query(sql)

        if df.empty:
            return pd.DataFrame()

        df['fdate'] = pd.to_datetime(df['fdate'])
        df['total_shares_held'] = pd.to_numeric(df['total_shares_held'], errors='coerce')
        df['num_institutions']  = pd.to_numeric(df['num_institutions'], errors='coerce')

        return df.set_index(['fdate', 'ticker']).sort_index()

    # ─────────────────────────────────────────────────────────────────────────
    # TAQmsec Intraday OHLCV — tick data aggregated to minute bars
    # ─────────────────────────────────────────────────────────────────────────

    # Earliest date TAQmsec tables are accessible under this WRDS subscription.
    # Confirmed by probing: 2022-01-04 is the first date with real trade data.
    # Years before 2022 return UndefinedTable or InsufficientPrivilege.
    # Years 2025+ are not yet available (publication lag on academic WRDS accounts).
    TAQMSEC_FLOOR_DATE = '2022-01-04'

    def get_taqmsec_ohlcv(
        self,
        ticker: str,
        timeframe: str = '1m',
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Build OHLCV bars from WRDS TAQmsec tick data for a single ticker.

        TAQmsec (`taqmsec.ctm_YYYYMMDD`) stores millisecond-resolution trade
        prints. We aggregate them into 1m, 5m, 15m, 30m, or 45m OHLCV bars.

        Accessible range on this WRDS subscription: 2022-01-04 → ~2024-12-31.
          - Pre-2022: UndefinedTable / InsufficientPrivilege (no data).
          - 2025+:    InsufficientPrivilege (publication lag on academic accounts).

        No artificial time cap is applied — the full accessible window is used.
        Missing tables (holidays, weekends, permission errors) are silently
        skipped via the except-continue pattern inside the per-day loop.

        Args:
            ticker:     Stock symbol (e.g., 'AAPL')
            timeframe:  '1m', '5m', '15m', '30m', or '45m'
            start_date: 'YYYY-MM-DD' — clamped to TAQMSEC_FLOOR_DATE if earlier
            end_date:   'YYYY-MM-DD' (default: today)

        Returns:
            DataFrame with DatetimeIndex and columns: Open, High, Low, Close, Volume
            Empty DataFrame if TAQmsec is not accessible or no data found.
        """
        if self._db is None:
            return pd.DataFrame()

        end_dt   = datetime.strptime(end_date,   '%Y-%m-%d') if end_date   else datetime.now()
        start_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.now()

        # Clamp start to the earliest accessible TAQmsec date on this subscription
        floor_dt = datetime.strptime(self.TAQMSEC_FLOOR_DATE, '%Y-%m-%d')
        if start_dt < floor_dt:
            start_dt = floor_dt

        if start_dt > end_dt:
            return pd.DataFrame()

        # Map timeframe → minute frequency for aggregation
        freq_minutes = {
            '1m':  1,
            '5m':  5,
            '15m': 15,
            '30m': 30,
            '45m': 45,
        }
        bar_minutes = freq_minutes.get(timeframe, 1)
        resample_rule = f'{bar_minutes}min'

        # TAQmsec tables are partitioned by date: taqmsec.ctm_YYYYMMDD
        # Build the list of actual NYSE trading days (no weekends, no holidays).
        # This avoids ~30 wasted round-trips per year for holiday dates that
        # have no TAQmsec table.  Falls back to Mon-Fri bdate_range if the
        # NYSE calendar library is unavailable.
        s_str = start_dt.strftime('%Y-%m-%d')
        e_str = end_dt.strftime('%Y-%m-%d')
        if _NYSE_CALENDAR_AVAILABLE:
            try:
                schedule = _NYSE_CALENDAR.schedule(start_date=s_str, end_date=e_str)
                trading_dates = schedule.index  # DatetimeIndex of NYSE open days
            except (OSError, ValueError, RuntimeError):
                trading_dates = pd.bdate_range(start=s_str, end=e_str)
        else:
            trading_dates = pd.bdate_range(start=s_str, end=e_str)

        if len(trading_dates) == 0:
            return pd.DataFrame()

        # No artificial cap — query everything in the accessible window.
        # The floor date (TAQMSEC_FLOOR_DATE) already limits the range to
        # what this subscription can actually return.

        sym = ticker.upper().strip()
        if not _TICKER_RE.match(sym):
            return pd.DataFrame()
        all_ticks: List[pd.DataFrame] = []

        for trade_date in trading_dates:
            table = f"taqmsec.ctm_{trade_date.strftime('%Y%m%d')}"
            date_str = trade_date.strftime('%Y-%m-%d')
            sql = f"""
                SELECT time_m, price, size
                FROM {table}
                WHERE sym_root = '{sym}'
                  AND tr_corr = '00'
                  AND price > 0
                  AND size  > 0
            """
            try:
                day_df = self._query(sql)
                if not day_df.empty:
                    # time_m is stored as 'HH:MM:SS.ffffff' (full time string).
                    # Combine with the date string to form a full datetime.
                    day_df['datetime'] = pd.to_datetime(
                        date_str + ' ' + day_df['time_m'].astype(str),
                        format='mixed',
                        errors='coerce',
                    )
                    day_df = day_df.dropna(subset=['datetime'])
                    day_df.index = pd.DatetimeIndex(day_df.pop('datetime'))
                    all_ticks.append(day_df[['price', 'size']])
            except (OSError, ValueError, RuntimeError):
                continue   # table may not exist for this date

        if not all_ticks:
            return pd.DataFrame()

        ticks = pd.concat(all_ticks).sort_index()
        ticks['price'] = pd.to_numeric(ticks['price'], errors='coerce')
        ticks['size']  = pd.to_numeric(ticks['size'],  errors='coerce')
        ticks = ticks.dropna()

        # Aggregate ticks → OHLCV bars
        ohlcv = ticks['price'].resample(resample_rule).agg(['first', 'max', 'min', 'last'])
        ohlcv.columns = ['Open', 'High', 'Low', 'Close']
        ohlcv['Volume'] = ticks['size'].resample(resample_rule).sum()
        ohlcv = ohlcv.dropna(subset=['Open', 'Close'])

        # Filter to regular trading hours (9:30–16:00 ET)
        ohlcv = ohlcv.between_time('09:30', '16:00')

        ohlcv.index.name = 'Date'
        return ohlcv

    # ─────────────────────────────────────────────────────────────────────────
    # OptionMetrics Options Volume — put/call volume and open interest
    # ─────────────────────────────────────────────────────────────────────────

    def query_options_volume(
        self,
        permno: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """
        Query OptionMetrics ``optionm.opprcd`` for daily aggregate options
        volume and open interest keyed by (date, cp_flag).

        Returns a DataFrame with columns:
            date, put_volume, call_volume, put_oi, call_oi

        Each row is one trading date. Returns ``None`` if WRDS is unavailable
        or no data is found.
        """
        if self._db is None:
            return None

        try:
            end = end_date or datetime.now().strftime('%Y-%m-%d')

            # Resolve PERMNO -> OptionMetrics secid via link table
            link = self.get_optionmetrics_link(
                permnos=[str(permno)],
                start_date=start_date,
                end_date=end,
            )
            if link.empty:
                return None

            secids = sorted({
                int(x) for x in
                pd.to_numeric(link["secid"], errors="coerce")
                .dropna().astype(int).tolist()
            })
            if not secids:
                return None
            secid_list = ",".join(str(s) for s in secids)

            # Query aggregate volume and open interest by cp_flag per day
            query_candidates = [
                f"""
                SELECT date, cp_flag,
                       SUM(volume)        AS total_volume,
                       SUM(open_interest) AS total_oi
                FROM optionm.opprcd
                WHERE secid IN ({secid_list})
                  AND date >= '{start_date}'::date
                  AND date <= '{end}'::date
                  AND volume IS NOT NULL
                GROUP BY date, cp_flag
                ORDER BY date, cp_flag
                """,
            ]

            raw = pd.DataFrame()
            for q in query_candidates:
                raw = self._query_silent(q)
                if not raw.empty:
                    break
            if raw.empty:
                return None

            raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
            raw["cp_flag"] = raw["cp_flag"].astype(str).str.upper().str[:1]
            raw["total_volume"] = pd.to_numeric(raw["total_volume"], errors="coerce").fillna(0)
            raw["total_oi"] = pd.to_numeric(raw["total_oi"], errors="coerce").fillna(0)

            # Pivot into put/call columns per date
            put_raw = raw[raw["cp_flag"] == "P"].copy()
            put_raw.index = pd.DatetimeIndex(put_raw.pop("date"), name="date")
            puts = put_raw[["total_volume", "total_oi"]].rename(
                columns={"total_volume": "put_volume", "total_oi": "put_oi"}
            )
            call_raw = raw[raw["cp_flag"] == "C"].copy()
            call_raw.index = pd.DatetimeIndex(call_raw.pop("date"), name="date")
            calls = call_raw[["total_volume", "total_oi"]]
            calls = calls.rename(columns={"total_volume": "call_volume", "total_oi": "call_oi"})

            out = puts.join(calls, how="outer").fillna(0).sort_index()
            out.index.name = "date"
            return out.reset_index()

        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("WRDS query_options_volume error: %s", e)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Compustat Short Interest
    # ─────────────────────────────────────────────────────────────────────────

    def query_short_interest(
        self,
        permno: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """
        Query Compustat ``comp.sec_shortint`` (or equivalent table) for
        short interest data.

        Returns a DataFrame with columns:
            settlement_date, short_interest, avg_daily_volume

        Returns ``None`` if WRDS is unavailable or no data is found.
        """
        if self._db is None:
            return None

        try:
            end = end_date or datetime.now().strftime('%Y-%m-%d')
            permno_str = str(int(str(permno).strip()))

            # First resolve permno -> gvkey via CRSP/Compustat merge link
            link_candidates = [
                f"""
                SELECT DISTINCT a.gvkey
                FROM crsp.ccmxpf_lnkhist AS a
                WHERE a.lpermno = {permno_str}
                  AND a.linktype IN ('LU','LC','LD','LN','LS','LX')
                  AND a.linkprim IN ('P','C')
                  AND a.linkdt <= '{end}'::date
                  AND (a.linkenddt >= '{start_date}'::date OR a.linkenddt IS NULL)
                """,
                f"""
                SELECT DISTINCT a.gvkey
                FROM crsp.ccmxpf_linktable AS a
                WHERE a.lpermno = {permno_str}
                  AND a.linkdt <= '{end}'::date
                  AND (a.linkenddt >= '{start_date}'::date OR a.linkenddt IS NULL)
                """,
            ]

            gvkeys = []
            for lsql in link_candidates:
                ldf = self._query_silent(lsql)
                if not ldf.empty:
                    gvkeys = [str(g) for g in ldf["gvkey"].dropna().tolist()]
                    break

            if not gvkeys:
                # Fallback: try ticker-based lookup via comp.fundq
                ticker = self._permno_to_ticker(permno, start_date)
                if ticker is None:
                    return None
                t_upper = str(ticker).upper().strip()
                if not _TICKER_RE.match(t_upper):
                    return None
                gvkey_sql = f"""
                    SELECT DISTINCT gvkey FROM comp.fundq
                    WHERE tic = '{t_upper}' AND gvkey IS NOT NULL
                    LIMIT 5
                """
                gdf = self._query_silent(gvkey_sql)
                if gdf.empty:
                    return None
                gvkeys = [str(g) for g in gdf["gvkey"].dropna().tolist()]

            gvkey_list = ",".join(f"'{g}'" for g in gvkeys)

            # Try multiple table/schema variants
            table_candidates = [
                (
                    "comp.sec_shortint",
                    f"""
                    SELECT datadate  AS settlement_date,
                           shortint  AS short_interest,
                           shortintadj AS short_interest_adj,
                           splitadjdate
                    FROM comp.sec_shortint
                    WHERE gvkey IN ({gvkey_list})
                      AND datadate >= '{start_date}'::date
                      AND datadate <= '{end}'::date
                    ORDER BY datadate
                    """,
                ),
                (
                    "comp.sec_shortinterest",
                    f"""
                    SELECT datadate  AS settlement_date,
                           shortint  AS short_interest,
                           NULL::double precision AS short_interest_adj,
                           NULL::date AS splitadjdate
                    FROM comp.sec_shortinterest
                    WHERE gvkey IN ({gvkey_list})
                      AND datadate >= '{start_date}'::date
                      AND datadate <= '{end}'::date
                    ORDER BY datadate
                    """,
                ),
            ]

            raw = pd.DataFrame()
            for _tbl_name, q in table_candidates:
                raw = self._query_silent(q)
                if not raw.empty:
                    break

            if raw.empty:
                return None

            raw["settlement_date"] = pd.to_datetime(raw["settlement_date"], errors="coerce")
            raw["short_interest"] = pd.to_numeric(raw["short_interest"], errors="coerce")
            raw = raw.dropna(subset=["settlement_date", "short_interest"])

            if raw.empty:
                return None

            # Compute avg_daily_volume from CRSP for days_to_cover
            vol_sql = f"""
                SELECT date, vol AS volume
                FROM crsp.dsf
                WHERE permno = {permno_str}
                  AND date >= '{start_date}'::date
                  AND date <= '{end}'::date
                  AND vol IS NOT NULL
                ORDER BY date
            """
            vol_df = self._query_silent(vol_sql)
            if not vol_df.empty:
                vol_df["date"] = pd.to_datetime(vol_df["date"], errors="coerce")
                vol_df["volume"] = pd.to_numeric(vol_df["volume"], errors="coerce")
                vol_df.index = pd.DatetimeIndex(vol_df.pop("date"))
                vol_df = vol_df.sort_index()
                vol_df["avg_daily_volume"] = vol_df["volume"].rolling(20, min_periods=5).mean()

                # Merge avg_daily_volume onto short interest dates via asof join
                raw = raw.sort_values("settlement_date")
                raw.index = pd.DatetimeIndex(raw.pop("settlement_date"), name="settlement_date")
                raw = raw.join(
                    vol_df[["avg_daily_volume"]],
                    how="left",
                )
                # For dates without exact match, use nearest prior value
                raw["avg_daily_volume"] = raw["avg_daily_volume"].ffill()
                raw = raw.reset_index()
            else:
                raw["avg_daily_volume"] = np.nan

            return raw[["settlement_date", "short_interest", "avg_daily_volume"]]

        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("WRDS query_short_interest error: %s", e)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Insider Transactions (TFN / Thomson Reuters)
    # ─────────────────────────────────────────────────────────────────────────

    def query_insider_transactions(
        self,
        permno: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """
        Query insider transaction data from TFN tables for Form 4 filings.

        Tries ``tfn.table1`` (insider trades) with CRSP PERMNO linkage.

        Returns a DataFrame with columns:
            filing_date, insider_name, shares, price, transaction_type,
            buy_sell

        Returns ``None`` if WRDS is unavailable or no data is found.
        """
        if self._db is None:
            return None

        try:
            end = end_date or datetime.now().strftime('%Y-%m-%d')

            # Resolve PERMNO to ticker for insider data matching
            ticker = self._permno_to_ticker(permno, start_date)
            if ticker is None:
                return None
            t_upper = str(ticker).upper().strip()
            if not _TICKER_RE.match(t_upper):
                return None

            # Try multiple insider transaction table variants
            table_candidates = [
                # TFN insider filing data (Thomson Financial Network)
                f"""
                SELECT trandate      AS filing_date,
                       personname    AS insider_name,
                       shares        AS shares,
                       tprice        AS price,
                       trancode      AS transaction_type,
                       acqdisp       AS buy_sell
                FROM tfn.table1
                WHERE ticker = '{t_upper}'
                  AND trandate >= '{start_date}'::date
                  AND trandate <= '{end}'::date
                  AND shares IS NOT NULL
                  AND shares > 0
                ORDER BY trandate
                """,
                # Alternative: tfn.insidertrade
                f"""
                SELECT trandate      AS filing_date,
                       personname    AS insider_name,
                       shares        AS shares,
                       tprice        AS price,
                       trancode      AS transaction_type,
                       acqdisp       AS buy_sell
                FROM tfn.insidertrade
                WHERE ticker = '{t_upper}'
                  AND trandate >= '{start_date}'::date
                  AND trandate <= '{end}'::date
                  AND shares IS NOT NULL
                  AND shares > 0
                ORDER BY trandate
                """,
                # Alternative: tfn.v1 (older schema)
                f"""
                SELECT trandate      AS filing_date,
                       personname    AS insider_name,
                       shares        AS shares,
                       tprice        AS price,
                       trancode      AS transaction_type,
                       acqdisp       AS buy_sell
                FROM tfn.v1
                WHERE ticker = '{t_upper}'
                  AND trandate >= '{start_date}'::date
                  AND trandate <= '{end}'::date
                  AND shares IS NOT NULL
                  AND shares > 0
                ORDER BY trandate
                """,
            ]

            raw = pd.DataFrame()
            for q in table_candidates:
                raw = self._query_silent(q)
                if not raw.empty:
                    break

            if raw.empty:
                return None

            raw["filing_date"] = pd.to_datetime(raw["filing_date"], errors="coerce")
            raw["shares"] = pd.to_numeric(raw["shares"], errors="coerce")
            raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
            raw = raw.dropna(subset=["filing_date"])

            if raw.empty:
                return None

            # Classify buy/sell: acqdisp='A' means acquisition (buy),
            # 'D' means disposition (sell)
            raw["buy_sell"] = raw["buy_sell"].astype(str).str.upper().str[:1]

            return raw

        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("WRDS query_insider_transactions error: %s", e)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: PERMNO → Ticker resolution
    # ─────────────────────────────────────────────────────────────────────────

    def _permno_to_ticker(
        self, permno: str, as_of_date: str = None
    ) -> Optional[str]:
        """Resolve a PERMNO to its most recent ticker symbol."""
        if self._db is None:
            return None
        try:
            permno_str = str(int(str(permno).strip()))
            ref_date = as_of_date or datetime.now().strftime('%Y-%m-%d')
            sql = f"""
                SELECT ticker
                FROM crsp.msenames
                WHERE permno = {permno_str}
                  AND namedt <= '{ref_date}'::date
                  AND (nameendt >= '{ref_date}'::date OR nameendt IS NULL)
                ORDER BY nameendt NULLS FIRST, namedt DESC
                LIMIT 1
            """
            df = self._query_silent(sql)
            if df.empty:
                return None
            t = str(df["ticker"].iloc[0]).strip()
            return t if t else None
        except (OSError, ValueError, RuntimeError):
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience: get full OHLCV-compatible data for the pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def get_ohlcv(
        self,
        ticker: str,
        start_date: str = '2000-01-01',
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Single-ticker OHLCV fetch from CRSP (drop-in for DataLayer.get_ohlcv).
        Returns DataFrame with columns: Open, High, Low, Close, Volume.
        """
        result = self.get_crsp_prices([ticker], start_date, end_date)
        if not result:
            return pd.DataFrame()
        # A single requested symbol should map to at most one active PERMNO in practice.
        permno_key = sorted(result.keys())[0]
        out = result[permno_key][['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        out.attrs["permno"] = permno_key
        if "ticker" in result[permno_key].columns and len(result[permno_key]) > 0:
            out.attrs["ticker"] = str(result[permno_key]["ticker"].iloc[-1])
        else:
            out.attrs["ticker"] = str(ticker).upper()
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Singleton factory
# ─────────────────────────────────────────────────────────────────────────────

_default_provider: Optional['WRDSProvider'] = None


def get_wrds_provider() -> 'WRDSProvider':
    """Get or create the default WRDSProvider singleton."""
    global _default_provider
    if _default_provider is None:
        _default_provider = WRDSProvider()
    return _default_provider


def wrds_available() -> bool:
    """Quick check: is WRDS accessible?"""
    try:
        return get_wrds_provider().available()
    except (OSError, ValueError, RuntimeError):
        return False
