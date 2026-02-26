"""
Alternative data framework — WRDS-backed implementation.

Provides a unified interface for ingesting non-price data sources:
  - Earnings surprises (WRDS I/B/E/S)
  - Short interest (Compustat sec_shortint via WRDS)
  - Options flow (OptionMetrics via WRDS)
  - Insider transactions (TFN via WRDS)
  - Institutional ownership changes (TFN s34 via WRDS)

Each data method first attempts to pull from the WRDS provider. If WRDS is
unavailable the method returns ``None`` so the pipeline degrades gracefully.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy-loaded WRDS provider singleton; avoids import-time DB connections.
_wrds_provider = None


def _get_wrds():
    """Return the cached WRDSProvider singleton, or None."""
    global _wrds_provider
    if _wrds_provider is not None:
        return _wrds_provider
    try:
        from data.wrds_provider import get_wrds_provider
        _wrds_provider = get_wrds_provider()
        return _wrds_provider
    except ImportError:
        pass
    # Also try the fully-qualified import path used in some entry points
    try:
        from quant_engine.data.wrds_provider import get_wrds_provider
        _wrds_provider = get_wrds_provider()
        return _wrds_provider
    except ImportError:
        return None


class AlternativeDataProvider:
    """WRDS-backed alternative data provider.

    Supported sources:
    - Earnings surprise (I/B/E/S via WRDS)
    - Short interest (Compustat sec_shortint via WRDS)
    - Options flow (OptionMetrics opprcd via WRDS)
    - Insider transactions (TFN via WRDS)
    - Institutional ownership (TFN s34 via WRDS)
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialise the provider.

        Parameters
        ----------
        cache_dir : Path, optional
            Directory used for caching downloaded alternative data.  If
            ``None``, caching is disabled.
        """
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._wrds = _get_wrds()

    # ── helpers ──────────────────────────────────────────────────────────

    def _resolve_permno(self, ticker: str, as_of: str = None) -> Optional[str]:
        """Resolve *ticker* to a CRSP PERMNO via the WRDS provider."""
        if self._wrds is None or not self._wrds.available():
            return None
        return self._wrds.resolve_permno(ticker, as_of_date=as_of)

    # ── 2.1  Earnings Surprise (I/B/E/S) ────────────────────────────────

    def get_earnings_surprise(
        self,
        ticker: str,
        lookback_days: int = 90,
        as_of_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """Return earnings surprise data for *ticker* from WRDS I/B/E/S.

        Output columns:
            report_date, eps_estimate, eps_actual, surprise_pct,
            surprise_zscore, beat_streak, revision_momentum

        Uses ``rdq`` (report date of quarterly earnings) for point-in-time
        semantics so the pipeline never looks ahead.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        lookback_days : int
            Number of calendar days to look back for earnings reports.
            Set to a large value (e.g. 3650) for longer history.
        as_of_date : datetime, optional
            Point-in-time cutoff.  When running inside a backtest this
            should be set to the current bar date so that no future
            earnings announcements leak into the feature set.  Defaults
            to ``datetime.now()`` for live / interactive use.

        Returns
        -------
        pd.DataFrame or None
        """
        if self._wrds is None or not self._wrds.available():
            logger.debug("get_earnings_surprise(%s): WRDS unavailable", ticker)
            return None

        try:
            reference_dt = as_of_date if as_of_date is not None else datetime.now()
            end_date = reference_dt.strftime('%Y-%m-%d')
            start_date = (reference_dt - timedelta(days=max(lookback_days, 365 * 5))).strftime('%Y-%m-%d')

            raw = self._wrds.get_earnings_surprises(
                tickers=[ticker.upper().strip()],
                start_date=start_date,
                end_date=end_date,
            )
            if raw is None or raw.empty:
                logger.debug("get_earnings_surprise(%s): no IBES data", ticker)
                return None

            # The IBES result is indexed by (anndats_act, ticker).
            # Flatten to a simple DataFrame.
            df = raw.reset_index()

            # Map to the expected column schema
            out = pd.DataFrame()
            out["report_date"] = df["anndats_act"]
            out["eps_estimate"] = pd.to_numeric(df["meanest"], errors="coerce")
            out["eps_actual"] = pd.to_numeric(df["actual"], errors="coerce")
            out["surprise_pct"] = pd.to_numeric(df["surprise_pct"], errors="coerce")
            out["dispersion"] = pd.to_numeric(df.get("dispersion", df.get("stdev", pd.Series(dtype=float))), errors="coerce")
            out["numest"] = pd.to_numeric(df.get("numest", pd.Series(dtype=float)), errors="coerce")

            # -- Derived feature 1: surprise_zscore = surprise_pct / dispersion
            disp = pd.to_numeric(df.get("stdev", pd.Series(dtype=float)), errors="coerce")
            out["surprise_zscore"] = np.where(
                disp.abs() > 1e-8,
                out["surprise_pct"] / disp,
                np.nan,
            )

            # -- Derived feature 2: beat_streak (consecutive beats)
            beat = (out["surprise_pct"] > 0).astype(int)
            streaks = []
            current_streak = 0
            for b in beat:
                if b == 1:
                    current_streak += 1
                else:
                    current_streak = 0
                streaks.append(current_streak)
            out["beat_streak"] = streaks

            # -- Derived feature 3: revision_momentum
            # Change in meanest over trailing 90 days (approx. one quarter)
            out = out.sort_values("report_date").reset_index(drop=True)
            out["revision_momentum"] = out["eps_estimate"].diff()

            # Apply lookback filter
            cutoff = pd.Timestamp(reference_dt - timedelta(days=lookback_days))
            out = out[out["report_date"] >= cutoff].reset_index(drop=True)

            if out.empty:
                return None

            return out

        except (ValueError, KeyError, TypeError) as e:
            logger.warning("get_earnings_surprise(%s) failed: %s", ticker, e)
            return None

    # ── 2.2  Options Flow (OptionMetrics) ───────────────────────────────

    def get_options_flow(
        self,
        ticker: str,
        as_of_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """Return options flow data for *ticker* from WRDS OptionMetrics.

        Output columns:
            date, put_volume, call_volume, put_call_ratio,
            total_open_interest, unusual_volume_flag

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        as_of_date : datetime, optional
            Point-in-time cutoff for backtest safety.  Defaults to
            ``datetime.now()``.

        Returns
        -------
        pd.DataFrame or None
        """
        if self._wrds is None or not self._wrds.available():
            logger.debug("get_options_flow(%s): WRDS unavailable", ticker)
            return None

        try:
            permno = self._resolve_permno(ticker)
            if permno is None:
                logger.debug("get_options_flow(%s): cannot resolve PERMNO", ticker)
                return None

            reference_dt = as_of_date if as_of_date is not None else datetime.now()
            end_date = reference_dt.strftime('%Y-%m-%d')
            start_date = (reference_dt - timedelta(days=365 * 3)).strftime('%Y-%m-%d')

            raw = self._wrds.query_options_volume(
                permno=permno,
                start_date=start_date,
                end_date=end_date,
            )
            if raw is None or raw.empty:
                logger.debug("get_options_flow(%s): no OptionMetrics data", ticker)
                return None

            df = raw.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            for col in ("put_volume", "call_volume", "put_oi", "call_oi"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            # Derived features
            df["put_call_ratio"] = np.where(
                df["call_volume"] > 0,
                df["put_volume"] / df["call_volume"],
                np.nan,
            )
            df["total_open_interest"] = df.get("put_oi", 0) + df.get("call_oi", 0)
            df["total_volume"] = df["put_volume"] + df["call_volume"]

            # Unusual volume flag: total volume > 2x its 20-day rolling average
            df = df.sort_values("date").reset_index(drop=True)
            avg_vol_20 = df["total_volume"].rolling(20, min_periods=5).mean()
            df["unusual_volume_flag"] = (df["total_volume"] > 2.0 * avg_vol_20).astype(int)

            out_cols = [
                "date", "put_volume", "call_volume", "put_call_ratio",
                "total_open_interest", "unusual_volume_flag",
            ]
            return df[[c for c in out_cols if c in df.columns]]

        except (ValueError, KeyError, TypeError) as e:
            logger.warning("get_options_flow(%s) failed: %s", ticker, e)
            return None

    # ── 2.3  Short Interest (Compustat) ─────────────────────────────────

    def get_short_interest(
        self,
        ticker: str,
        as_of_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """Return short interest data for *ticker* from Compustat via WRDS.

        Output columns:
            settlement_date, short_interest, avg_daily_volume,
            short_interest_ratio, short_interest_change, days_to_cover,
            squeeze_risk

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        as_of_date : datetime, optional
            Point-in-time cutoff for backtest safety.  Defaults to
            ``datetime.now()``.

        Returns
        -------
        pd.DataFrame or None
        """
        if self._wrds is None or not self._wrds.available():
            logger.debug("get_short_interest(%s): WRDS unavailable", ticker)
            return None

        try:
            permno = self._resolve_permno(ticker)
            if permno is None:
                logger.debug("get_short_interest(%s): cannot resolve PERMNO", ticker)
                return None

            reference_dt = as_of_date if as_of_date is not None else datetime.now()
            end_date = reference_dt.strftime('%Y-%m-%d')
            start_date = (reference_dt - timedelta(days=365 * 3)).strftime('%Y-%m-%d')

            raw = self._wrds.query_short_interest(
                permno=permno,
                start_date=start_date,
                end_date=end_date,
            )
            if raw is None or raw.empty:
                logger.debug("get_short_interest(%s): no short interest data", ticker)
                return None

            df = raw.copy()
            df["settlement_date"] = pd.to_datetime(df["settlement_date"], errors="coerce")
            df["short_interest"] = pd.to_numeric(df["short_interest"], errors="coerce")
            df["avg_daily_volume"] = pd.to_numeric(df["avg_daily_volume"], errors="coerce")
            df = df.sort_values("settlement_date").reset_index(drop=True)

            # -- short_interest_ratio = short_interest / avg_daily_volume
            df["short_interest_ratio"] = np.where(
                df["avg_daily_volume"] > 0,
                df["short_interest"] / df["avg_daily_volume"],
                np.nan,
            )

            # -- short_interest_change = MoM delta (approx 1-period diff since
            #    short interest is typically reported semi-monthly)
            df["short_interest_change"] = df["short_interest"].diff()

            # -- days_to_cover = short_interest / avg_daily_volume
            df["days_to_cover"] = df["short_interest_ratio"]  # same calculation

            # -- squeeze_risk: z-score of short_interest_ratio relative to
            #    its trailing history. High z → crowded short, higher squeeze risk.
            rolling_mean = df["short_interest_ratio"].rolling(12, min_periods=3).mean()
            rolling_std = df["short_interest_ratio"].rolling(12, min_periods=3).std()
            df["squeeze_risk"] = np.where(
                rolling_std > 1e-8,
                (df["short_interest_ratio"] - rolling_mean) / rolling_std,
                0.0,
            )

            out_cols = [
                "settlement_date", "short_interest", "avg_daily_volume",
                "short_interest_ratio", "short_interest_change",
                "days_to_cover", "squeeze_risk",
            ]
            return df[[c for c in out_cols if c in df.columns]]

        except (ValueError, KeyError, TypeError) as e:
            logger.warning("get_short_interest(%s) failed: %s", ticker, e)
            return None

    # ── 2.4  Insider Transactions (TFN) ─────────────────────────────────

    def get_insider_transactions(
        self,
        ticker: str,
        as_of_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """Return insider transaction data for *ticker* from TFN via WRDS.

        Output columns:
            filing_date, insider_name, shares, price, transaction_type,
            buy_sell, value, net_insider_buying, insider_buy_count,
            cluster_buy_signal

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        as_of_date : datetime, optional
            Point-in-time cutoff for backtest safety.  Defaults to
            ``datetime.now()``.

        Returns
        -------
        pd.DataFrame or None
        """
        if self._wrds is None or not self._wrds.available():
            logger.debug("get_insider_transactions(%s): WRDS unavailable", ticker)
            return None

        try:
            permno = self._resolve_permno(ticker)
            if permno is None:
                logger.debug("get_insider_transactions(%s): cannot resolve PERMNO", ticker)
                return None

            reference_dt = as_of_date if as_of_date is not None else datetime.now()
            end_date = reference_dt.strftime('%Y-%m-%d')
            start_date = (reference_dt - timedelta(days=365 * 3)).strftime('%Y-%m-%d')

            raw = self._wrds.query_insider_transactions(
                permno=permno,
                start_date=start_date,
                end_date=end_date,
            )
            if raw is None or raw.empty:
                logger.debug("get_insider_transactions(%s): no insider data", ticker)
                return None

            df = raw.copy()
            df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
            df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0)
            df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)

            # Compute dollar value
            df["value"] = df["shares"] * df["price"]

            # buy_sell: 'A' = acquisition (buy), 'D' = disposition (sell)
            df["is_buy"] = df["buy_sell"].astype(str).str.upper().str[:1] == "A"

            # -- net_insider_buying: daily net dollar value (buy - sell)
            df["signed_value"] = np.where(df["is_buy"], df["value"], -df["value"])
            daily_net = df.groupby("filing_date")["signed_value"].sum()
            df = df.merge(
                daily_net.rename("net_insider_buying"),
                left_on="filing_date",
                right_index=True,
                how="left",
            )

            # -- insider_buy_count: rolling 30-day count of buy transactions
            df = df.sort_values("filing_date").reset_index(drop=True)
            buy_dates = df.loc[df["is_buy"], "filing_date"]
            if not buy_dates.empty:
                buy_counts = buy_dates.groupby(buy_dates).count()
                buy_counts_full = buy_counts.reindex(
                    pd.date_range(buy_counts.index.min(), buy_counts.index.max())
                ).fillna(0)
                rolling_buy_count = buy_counts_full.rolling(30, min_periods=1).sum()
                df = df.merge(
                    rolling_buy_count.rename("insider_buy_count").reset_index().rename(
                        columns={"index": "filing_date"}
                    ),
                    on="filing_date",
                    how="left",
                )
            else:
                df["insider_buy_count"] = 0

            # -- cluster_buy_signal: 1 if >= 3 distinct insiders bought
            #    within a 30-day window
            if "insider_name" in df.columns:
                df_buy = df[df["is_buy"]].copy()
                if not df_buy.empty:
                    df_buy["filing_date_dt"] = pd.to_datetime(df_buy["filing_date"])
                    cluster_signals = []
                    for _, row in df.iterrows():
                        if pd.isna(row["filing_date"]):
                            cluster_signals.append(0)
                            continue
                        d = pd.Timestamp(row["filing_date"])
                        window_start = d - timedelta(days=30)
                        window_buys = df_buy[
                            (df_buy["filing_date_dt"] >= window_start)
                            & (df_buy["filing_date_dt"] <= d)
                        ]
                        n_distinct = window_buys["insider_name"].nunique()
                        cluster_signals.append(1 if n_distinct >= 3 else 0)
                    df["cluster_buy_signal"] = cluster_signals
                else:
                    df["cluster_buy_signal"] = 0
            else:
                df["cluster_buy_signal"] = 0

            # Clean up temp columns
            df = df.drop(columns=["is_buy", "signed_value", "filing_date_dt"],
                         errors="ignore")

            return df

        except (ValueError, KeyError, TypeError) as e:
            logger.warning("get_insider_transactions(%s) failed: %s", ticker, e)
            return None

    # ── 2.5  Institutional Ownership Changes (13F / TFN s34) ────────────

    def get_institutional_ownership(
        self,
        ticker: str,
        as_of_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """Return institutional ownership data with QoQ change features.

        Output columns:
            fdate, total_shares_held, num_institutions,
            inst_ownership_change, num_institutions_change,
            new_positions, closed_positions

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        as_of_date : datetime, optional
            Point-in-time cutoff for backtest safety.  Defaults to
            ``datetime.now()``.

        Returns
        -------
        pd.DataFrame or None
        """
        if self._wrds is None or not self._wrds.available():
            logger.debug("get_institutional_ownership(%s): WRDS unavailable", ticker)
            return None

        try:
            reference_dt = as_of_date if as_of_date is not None else datetime.now()
            end_date = reference_dt.strftime('%Y-%m-%d')
            start_date = (reference_dt - timedelta(days=365 * 5)).strftime('%Y-%m-%d')

            raw = self._wrds.get_institutional_ownership(
                tickers=[ticker.upper().strip()],
                start_date=start_date,
                end_date=end_date,
            )
            if raw is None or raw.empty:
                logger.debug("get_institutional_ownership(%s): no data", ticker)
                return None

            # Flatten multi-index (fdate, ticker) -> simple DataFrame
            df = raw.reset_index()
            df["fdate"] = pd.to_datetime(df["fdate"], errors="coerce")
            df["total_shares_held"] = pd.to_numeric(df["total_shares_held"], errors="coerce")
            df["num_institutions"] = pd.to_numeric(df["num_institutions"], errors="coerce")
            df = df.sort_values("fdate").reset_index(drop=True)

            # -- inst_ownership_change: QoQ delta in total_shares_held
            df["inst_ownership_change"] = df["total_shares_held"].diff()

            # -- num_institutions_change: QoQ delta in num_institutions
            df["num_institutions_change"] = df["num_institutions"].diff()

            # For new_positions and closed_positions, we need per-manager data.
            # Approximate from aggregate: if num_institutions increased, the
            # delta is a lower bound on new positions. If it decreased, the
            # magnitude is a lower bound on closed positions.
            inst_delta = df["num_institutions_change"].fillna(0)
            df["new_positions"] = inst_delta.clip(lower=0).astype(int)
            df["closed_positions"] = (-inst_delta).clip(lower=0).astype(int)

            out_cols = [
                "fdate", "total_shares_held", "num_institutions",
                "inst_ownership_change", "num_institutions_change",
                "new_positions", "closed_positions",
            ]
            return df[[c for c in out_cols if c in df.columns]]

        except (ValueError, KeyError, TypeError) as e:
            logger.warning("get_institutional_ownership(%s) failed: %s", ticker, e)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Feature aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compute_alternative_features(
    ticker: str,
    provider: Optional[AlternativeDataProvider] = None,
    cache_dir: Optional[Path] = None,
    as_of_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Gather all available alternative data and return as a feature DataFrame.

    Tries each alternative data source via *provider* (or a freshly created
    :class:`AlternativeDataProvider` if none is supplied).  Any source that
    returns data is summarised into numeric feature columns and merged into
    a single DataFrame indexed by date.  Sources that return ``None`` are
    silently skipped.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    provider : AlternativeDataProvider, optional
        An existing provider instance.  If ``None``, one is created with
        *cache_dir*.
    cache_dir : Path, optional
        Passed to :class:`AlternativeDataProvider` when *provider* is
        ``None``.
    as_of_date : datetime, optional
        Point-in-time cutoff.  During backtests this should be set to the
        current bar date to prevent future data from leaking into
        features.  Defaults to ``datetime.now()`` for live use.

    Returns
    -------
    pd.DataFrame
        A (possibly empty) DataFrame with a DatetimeIndex and one column
        per alternative feature.  Column names are prefixed with ``alt_``.
    """
    if provider is None:
        provider = AlternativeDataProvider(cache_dir=cache_dir)

    frames: list[pd.DataFrame] = []

    # --- Earnings surprise -----------------------------------------------
    earnings = provider.get_earnings_surprise(ticker, as_of_date=as_of_date)
    if earnings is not None and not earnings.empty:
        if "report_date" in earnings.columns and "surprise_pct" in earnings.columns:
            keep = ["report_date", "surprise_pct"]
            for extra in ("surprise_zscore", "beat_streak", "revision_momentum"):
                if extra in earnings.columns:
                    keep.append(extra)
            ef = earnings[keep].copy()
            ef["report_date"] = pd.to_datetime(ef["report_date"])
            ef.index = pd.DatetimeIndex(ef.pop("report_date"), name="report_date")
            rename_map = {c: f"alt_{c}" for c in ef.columns}
            ef = ef.rename(columns=rename_map)
            frames.append(ef)

    # --- Short interest --------------------------------------------------
    si = provider.get_short_interest(ticker, as_of_date=as_of_date)
    if si is not None and not si.empty:
        date_col = "settlement_date" if "settlement_date" in si.columns else si.columns[0]
        keep_cols = [
            c for c in (
                "short_interest_ratio", "short_interest_change",
                "days_to_cover", "squeeze_risk",
            ) if c in si.columns
        ]
        if keep_cols:
            sf = si[[date_col] + keep_cols].copy()
            sf[date_col] = pd.to_datetime(sf[date_col])
            sf.index = pd.DatetimeIndex(sf.pop(date_col), name=date_col)
            sf = sf.rename(columns={c: f"alt_{c}" for c in keep_cols})
            frames.append(sf)

    # --- Options flow ----------------------------------------------------
    options = provider.get_options_flow(ticker, as_of_date=as_of_date)
    if options is not None and not options.empty:
        date_col = "date" if "date" in options.columns else options.columns[0]
        keep_cols = [
            c for c in (
                "put_call_ratio", "total_open_interest", "unusual_volume_flag",
            ) if c in options.columns
        ]
        if keep_cols:
            of = options[[date_col] + keep_cols].copy()
            of[date_col] = pd.to_datetime(of[date_col])
            of.index = pd.DatetimeIndex(of.pop(date_col), name=date_col)
            of = of.rename(columns={c: f"alt_{c}" for c in keep_cols})
            frames.append(of)

    # --- Insider transactions --------------------------------------------
    insider = provider.get_insider_transactions(ticker, as_of_date=as_of_date)
    if insider is not None and not insider.empty:
        if "filing_date" in insider.columns:
            keep_cols = [
                c for c in (
                    "net_insider_buying", "insider_buy_count", "cluster_buy_signal",
                ) if c in insider.columns
            ]
            if keep_cols:
                inf = insider[["filing_date"] + keep_cols].copy()
                inf["filing_date"] = pd.to_datetime(inf["filing_date"])
                # Aggregate to daily (multiple filings on the same day)
                daily = inf.groupby("filing_date")[keep_cols].sum()
                # For cluster_buy_signal, take max (any-day signal)
                if "cluster_buy_signal" in keep_cols:
                    daily["cluster_buy_signal"] = (
                        inf.groupby("filing_date")["cluster_buy_signal"].max()
                    )
                daily = daily.rename(columns={c: f"alt_{c}" for c in daily.columns})
                frames.append(daily)

    # --- Institutional ownership changes ---------------------------------
    inst = provider.get_institutional_ownership(ticker, as_of_date=as_of_date)
    if inst is not None and not inst.empty:
        date_col = "fdate" if "fdate" in inst.columns else inst.columns[0]
        keep_cols = [
            c for c in (
                "inst_ownership_change", "num_institutions_change",
                "new_positions", "closed_positions",
            ) if c in inst.columns
        ]
        if keep_cols:
            iof = inst[[date_col] + keep_cols].copy()
            iof[date_col] = pd.to_datetime(iof[date_col])
            iof.index = pd.DatetimeIndex(iof.pop(date_col), name=date_col)
            iof = iof.rename(columns={c: f"alt_{c}" for c in keep_cols})
            frames.append(iof)

    if not frames:
        return pd.DataFrame()

    result = frames[0]
    for f in frames[1:]:
        result = result.join(f, how="outer")

    result.index.name = "date"
    result = result.sort_index()
    return result
