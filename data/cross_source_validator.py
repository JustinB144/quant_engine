"""
Cross-source validation system comparing Alpaca/Alpha Vantage against IBKR.

IBKR is always ground truth. If sources disagree, IBKR wins.
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from ib_insync import IB, Stock, util as ib_util

logger = logging.getLogger(__name__)


@dataclass
class CrossValidationReport:
    """Report from cross-source validation."""
    ticker: str
    timeframe: str
    sample_windows: int              # How many date windows sampled
    sample_days: int                 # Total days sampled
    overlapping_bars: int            # Bars compared
    price_mismatches: int            # Bars where close > 0.15% off
    open_mismatches: int             # Bars where open > 0.20% off
    highlow_mismatches: int          # Bars where H or L > 0.25% off
    volume_anomalies: int            # Volume > 50% different AND > 100 shares diff
    missing_in_primary: int          # Bars in IBKR but not in primary source
    phantom_in_primary: int          # Bars in primary but not IBKR (extended hours leak?)
    split_mismatches: int            # Suspected unadjusted split bars
    bars_replaced: int               # Bars swapped to IBKR data
    bars_inserted: int               # IBKR bars added to fill gaps
    passed: bool                     # True if mismatch rate < 5%
    mismatch_rate: float             # price_mismatches / max(overlapping_bars, 1)
    details: List[dict] = field(default_factory=list)  # Per-bar mismatch details for audit trail


class CrossSourceValidator:
    """Validates primary source (Alpaca/AV) against IBKR ground truth."""

    # Known stock split ratios (forward and reverse)
    KNOWN_SPLIT_RATIOS = [
        2.0, 3.0, 4.0, 5.0, 10.0, 1.5,  # 3:2
        20.0,  # Forward splits
        0.5, 0.333, 0.25, 0.2, 0.1, 0.667,  # 2:3
        0.05  # Reverse splits
    ]

    IBKR_BAR_SIZES = {
        "1m": "1 min",
        "5m": "5 mins",
        "15m": "15 mins",
        "30m": "30 mins",
        "1h": "1 hour",
        "4h": "4 hours",
    }

    def __init__(
        self,
        ib: IB,
        close_tolerance_pct: float = 0.15,
        open_tolerance_pct: float = 0.20,
        highlow_tolerance_pct: float = 0.25,
        volume_tolerance_pct: float = 5.0,
        volume_min_abs_diff: int = 100,
        sample_windows: int = 10,
        days_per_window: int = 2,
        max_mismatch_rate: float = 0.05,
        ibkr_pace: float = 2.0,
    ):
        """
        Initialize cross-source validator.

        Args:
            ib: Connected ib_insync.IB instance
            close_tolerance_pct: Price mismatch threshold for close (%)
            open_tolerance_pct: Price mismatch threshold for open (%)
            highlow_tolerance_pct: Price mismatch threshold for high/low (%)
            volume_tolerance_pct: Volume mismatch threshold (%)
            volume_min_abs_diff: Minimum absolute volume difference to report
            sample_windows: Number of date windows to sample
            days_per_window: Trading days to sample per window
            max_mismatch_rate: Threshold for validation failure
            ibkr_pace: Seconds to wait between IBKR requests
        """
        self.ib = ib
        self.close_tolerance_pct = close_tolerance_pct
        self.open_tolerance_pct = open_tolerance_pct
        self.highlow_tolerance_pct = highlow_tolerance_pct
        self.volume_tolerance_pct = volume_tolerance_pct
        self.volume_min_abs_diff = volume_min_abs_diff
        self.sample_windows = sample_windows
        self.days_per_window = days_per_window
        self.max_mismatch_rate = max_mismatch_rate
        self.ibkr_pace = ibkr_pace

    def validate_ticker(
        self,
        primary_df: pd.DataFrame,
        ticker: str,
        timeframe: str,
    ) -> Tuple[pd.DataFrame, CrossValidationReport]:
        """
        Main validation entry point.

        Compares primary source (Alpaca/AV) against IBKR ground truth.
        Returns corrected DataFrame and detailed report.

        Args:
            primary_df: DataFrame from Alpaca/Alpha Vantage
            ticker: Stock ticker
            timeframe: Timeframe (1m, 5m, 15m, 30m, 1h, 4h)

        Returns:
            (corrected_df, report)
        """
        logger.info(f"Starting cross-source validation for {ticker} {timeframe}")

        # Initialize report
        report = CrossValidationReport(
            ticker=ticker,
            timeframe=timeframe,
            sample_windows=self.sample_windows,
            sample_days=0,
            overlapping_bars=0,
            price_mismatches=0,
            open_mismatches=0,
            highlow_mismatches=0,
            volume_anomalies=0,
            missing_in_primary=0,
            phantom_in_primary=0,
            split_mismatches=0,
            bars_replaced=0,
            bars_inserted=0,
            passed=False,
            mismatch_rate=0.0,
        )

        # Handle empty primary data
        if primary_df.empty:
            logger.warning(f"Primary data empty for {ticker}")
            report.details.append({"error": "Primary data empty"})
            return primary_df, report

        # Ensure datetime index
        if not isinstance(primary_df.index, pd.DatetimeIndex):
            primary_df.index = pd.to_datetime(primary_df.index)

        # Select sample dates from primary data
        sample_dates = self._select_sample_dates(primary_df)
        report.sample_days = len(sample_dates)

        if not sample_dates:
            logger.warning(f"No sample dates selected for {ticker}")
            report.details.append({"error": "No sample dates selected"})
            return primary_df, report

        logger.info(f"Sampling {len(sample_dates)} dates for {ticker}")

        # Fetch IBKR data for sample dates
        ibkr_df = self._fetch_ibkr_for_dates(ticker, timeframe, sample_dates)

        if ibkr_df is None or ibkr_df.empty:
            logger.warning(f"IBKR returned no data for {ticker}")
            report.passed = False
            report.overlapping_bars = 0
            report.details.append({
                "error": "IBKR returned no data - validation not performed"
            })
            return primary_df, report

        # Compare bars
        comparison_result = self._compare_bars(primary_df, ibkr_df, ticker, timeframe)

        # Update report with comparison results
        report.overlapping_bars = comparison_result["overlapping_bars"]
        report.price_mismatches = comparison_result["price_mismatches"]
        report.open_mismatches = comparison_result["open_mismatches"]
        report.highlow_mismatches = comparison_result["highlow_mismatches"]
        report.volume_anomalies = comparison_result["volume_anomalies"]
        report.missing_in_primary = comparison_result["missing_in_primary"]
        report.phantom_in_primary = comparison_result["phantom_in_primary"]
        report.split_mismatches = comparison_result["split_mismatches"]
        report.details = comparison_result["details"]

        # Calculate mismatch rate
        report.mismatch_rate = (
            report.price_mismatches / max(report.overlapping_bars, 1)
        )

        # Determine pass/fail
        report.passed = report.mismatch_rate < self.max_mismatch_rate

        logger.info(
            f"Validation report: {report.overlapping_bars} bars compared, "
            f"{report.price_mismatches} mismatches, "
            f"rate={report.mismatch_rate:.2%}, passed={report.passed}"
        )

        # If we have mismatches, fix them
        if report.price_mismatches > 0 or report.missing_in_primary > 0:
            primary_df_corrected, bars_replaced, bars_inserted = self._replace_bad_bars(
                primary_df, ibkr_df, comparison_result["mismatches"]
            )
            report.bars_replaced = bars_replaced
            report.bars_inserted = bars_inserted
        else:
            primary_df_corrected = primary_df.copy()

        logger.info(
            f"Validation complete: {report.bars_replaced} replaced, "
            f"{report.bars_inserted} inserted"
        )

        return primary_df_corrected, report

    def _select_sample_dates(self, primary_df: pd.DataFrame) -> List[pd.Timestamp]:
        """
        Select stratified sample of trading dates.

        Divides date range into sample_windows equal windows,
        selects days_per_window random trading days from each window.

        Args:
            primary_df: Input DataFrame with datetime index

        Returns:
            List of sampled dates
        """
        if primary_df.empty:
            return []

        dates = primary_df.index.unique()
        if len(dates) < self.sample_windows:
            return list(dates)

        # Divide into windows
        window_size = len(dates) // self.sample_windows
        sample_dates = []

        for i in range(self.sample_windows):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size if i < self.sample_windows - 1 else len(dates)

            window_dates = dates[start_idx:end_idx].tolist()
            num_to_sample = min(self.days_per_window, len(window_dates))

            sampled = random.sample(window_dates, num_to_sample)
            sample_dates.extend(sampled)

        return sorted(sample_dates)

    def _fetch_ibkr_for_dates(
        self,
        ticker: str,
        timeframe: str,
        dates: List[pd.Timestamp],
    ) -> Optional[pd.DataFrame]:
        """
        Fetch IBKR data for specified dates.

        Groups dates into contiguous ranges for efficient downloading.

        Args:
            ticker: Stock ticker
            timeframe: Timeframe (1m, 5m, 15m, 30m, 1h, 4h)
            dates: List of dates to fetch

        Returns:
            DataFrame with IBKR data, or None if fetch fails
        """
        if not dates:
            return None

        logger.info(f"Fetching IBKR data for {ticker} {timeframe} ({len(dates)} dates)")

        try:
            # Convert timeframe to IBKR format
            if timeframe not in self.IBKR_BAR_SIZES:
                logger.warning(f"Unsupported timeframe: {timeframe}")
                return None

            ibkr_barsize = self.IBKR_BAR_SIZES[timeframe]

            # Group dates into contiguous ranges
            dates_sorted = sorted(dates)
            ranges = self._group_dates_into_ranges(dates_sorted)

            all_bars = []

            for start_date, end_date in ranges:
                logger.debug(f"Downloading IBKR {ticker} {start_date} to {end_date}")

                try:
                    contract = Stock(ticker, "SMART", "USD")
                    bars = self.ib.reqHistoricalData(
                        contract,
                        endDateTime=end_date.strftime("%Y%m%d %H:%M:%S"),
                        durationStr=f"{(end_date - start_date).days + 1} D",
                        barSizeSetting=ibkr_barsize,
                        whatToShow="TRADES",
                        useRTH=True,
                        formatDate=1,
                        keepUpToDate=False,
                    )

                    if bars:
                        df = ib_util.df(bars)
                        # Normalize IBKR columns to canonical title-case
                        df = df.rename(columns={
                            "date": "Date", "open": "Open", "high": "High",
                            "low": "Low", "close": "Close", "volume": "Volume",
                        })
                        if "Date" in df.columns:
                            df["Date"] = pd.to_datetime(df["Date"])
                            df = df.set_index("Date")
                        all_bars.append(df)

                    # Pace requests
                    import time
                    time.sleep(self.ibkr_pace)

                except Exception as e:
                    logger.error(f"IBKR download failed for {ticker}: {e}")
                    return None

            if not all_bars:
                logger.warning(f"No IBKR data fetched for {ticker}")
                return None

            # Combine all bars
            ibkr_df = pd.concat(all_bars, ignore_index=False)

            # Remove duplicates and sort
            ibkr_df = ibkr_df[~ibkr_df.index.duplicated(keep='first')]
            ibkr_df = ibkr_df.sort_index()

            # Ensure datetime index
            if not isinstance(ibkr_df.index, pd.DatetimeIndex):
                ibkr_df.index = pd.to_datetime(ibkr_df.index)

            # Filter to only requested dates (to handle boundary effects)
            # index.date returns a numpy array, so use np.isin instead of .isin()
            requested_dates = set(d.date() if hasattr(d, 'date') else d for d in dates)
            ibkr_dates = ibkr_df.index.date  # numpy array of date objects
            ibkr_df = ibkr_df[np.isin(ibkr_dates, list(requested_dates))]

            logger.info(f"IBKR fetch complete: {len(ibkr_df)} bars")
            return ibkr_df

        except Exception as e:
            logger.error(f"IBKR fetch error: {e}")
            return None

    def _group_dates_into_ranges(
        self,
        dates: List[pd.Timestamp],
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Group dates into contiguous ranges.

        Args:
            dates: Sorted list of dates

        Returns:
            List of (start_date, end_date) tuples
        """
        if not dates:
            return []

        ranges = []
        start = dates[0]
        prev = dates[0]

        for date in dates[1:]:
            # Check if there's a gap > 1 day
            if (date - prev).days > 1:
                ranges.append((start, prev))
                start = date
            prev = date

        ranges.append((start, prev))
        return ranges

    def _compare_bars(
        self,
        primary_df: pd.DataFrame,
        ibkr_df: pd.DataFrame,
        ticker: str,
        timeframe: str,
    ) -> Dict:
        """
        Core comparison logic.

        Compares O, H, L, C, V for overlapping timestamps.
        Records every mismatch with details.

        Args:
            primary_df: Primary source DataFrame
            ibkr_df: IBKR DataFrame
            ticker: Stock ticker
            timeframe: Timeframe

        Returns:
            Dictionary with comparison results
        """
        result = {
            "overlapping_bars": 0,
            "price_mismatches": 0,
            "open_mismatches": 0,
            "highlow_mismatches": 0,
            "volume_anomalies": 0,
            "missing_in_primary": 0,
            "phantom_in_primary": 0,
            "split_mismatches": 0,
            "details": [],
            "mismatches": [],
        }

        # Ensure datetime indices
        if not isinstance(primary_df.index, pd.DatetimeIndex):
            primary_df.index = pd.to_datetime(primary_df.index)
        if not isinstance(ibkr_df.index, pd.DatetimeIndex):
            ibkr_df.index = pd.to_datetime(ibkr_df.index)

        # Find overlapping timestamps
        overlapping_idx = primary_df.index.intersection(ibkr_df.index)
        result["overlapping_bars"] = len(overlapping_idx)

        logger.info(f"Comparing {len(overlapping_idx)} overlapping bars")

        if len(overlapping_idx) == 0:
            result["details"].append({"error": "No overlapping bars"})
            return result

        # Compare each overlapping bar
        for ts in overlapping_idx:
            prim_row = primary_df.loc[ts]
            ibkr_row = ibkr_df.loc[ts]

            # Handle multi-index case (multiple bars at same timestamp)
            if isinstance(prim_row, pd.DataFrame):
                prim_row = prim_row.iloc[0]
            if isinstance(ibkr_row, pd.DataFrame):
                ibkr_row = ibkr_row.iloc[0]

            bar_detail = {"timestamp": ts}
            is_mismatch = False

            # Compare close
            if self._prices_differ(prim_row.get("Close"), ibkr_row.get("Close"),
                                    self.close_tolerance_pct):
                result["price_mismatches"] += 1
                bar_detail["close_mismatch"] = {
                    "primary": prim_row.get("Close"),
                    "ibkr": ibkr_row.get("Close"),
                    "pct_diff": self._pct_diff(prim_row.get("Close"),
                                               ibkr_row.get("Close")),
                }
                is_mismatch = True

            # Compare open
            if self._prices_differ(prim_row.get("Open"), ibkr_row.get("Open"),
                                    self.open_tolerance_pct):
                result["open_mismatches"] += 1
                bar_detail["open_mismatch"] = {
                    "primary": prim_row.get("Open"),
                    "ibkr": ibkr_row.get("Open"),
                    "pct_diff": self._pct_diff(prim_row.get("Open"),
                                               ibkr_row.get("Open")),
                }
                is_mismatch = True

            # Compare high
            if self._prices_differ(prim_row.get("High"), ibkr_row.get("High"),
                                    self.highlow_tolerance_pct):
                result["highlow_mismatches"] += 1
                bar_detail["high_mismatch"] = {
                    "primary": prim_row.get("High"),
                    "ibkr": ibkr_row.get("High"),
                    "pct_diff": self._pct_diff(prim_row.get("High"),
                                               ibkr_row.get("High")),
                }
                is_mismatch = True

            # Compare low
            if self._prices_differ(prim_row.get("Low"), ibkr_row.get("Low"),
                                    self.highlow_tolerance_pct):
                result["highlow_mismatches"] += 1
                bar_detail["low_mismatch"] = {
                    "primary": prim_row.get("Low"),
                    "ibkr": ibkr_row.get("Low"),
                    "pct_diff": self._pct_diff(prim_row.get("Low"),
                                               ibkr_row.get("Low")),
                }
                is_mismatch = True

            # Compare volume
            if self._volumes_differ(prim_row.get("Volume"), ibkr_row.get("Volume"),
                                     self.volume_tolerance_pct,
                                     self.volume_min_abs_diff):
                result["volume_anomalies"] += 1
                bar_detail["volume_mismatch"] = {
                    "primary": int(prim_row.get("volume", 0)),
                    "ibkr": int(ibkr_row.get("volume", 0)),
                    "pct_diff": self._pct_diff(prim_row.get("Volume"),
                                               ibkr_row.get("Volume")),
                }
                is_mismatch = True

            # Detect split mismatches
            if self._is_suspected_split(prim_row.get("Close"), ibkr_row.get("Close")):
                result["split_mismatches"] += 1
                bar_detail["split_suspected"] = True
                is_mismatch = False  # Don't count as regular mismatch

            if is_mismatch or "split_suspected" in bar_detail:
                result["details"].append(bar_detail)
                result["mismatches"].append({
                    "timestamp": ts,
                    "type": "PRICE_MISMATCH" if is_mismatch else "SPLIT_MISMATCH",
                    "detail": bar_detail,
                })

        # Find missing in primary
        missing_in_primary_idx = ibkr_df.index.difference(primary_df.index)
        result["missing_in_primary"] = len(missing_in_primary_idx)
        for ts in missing_in_primary_idx:
            result["mismatches"].append({
                "timestamp": ts,
                "type": "MISSING_IN_PRIMARY",
                "detail": {"timestamp": ts, "reason": "Bar in IBKR but not in primary"},
            })

        # Find phantom in primary
        phantom_in_primary_idx = primary_df.index.difference(ibkr_df.index)
        result["phantom_in_primary"] = len(phantom_in_primary_idx)

        logger.info(
            f"Comparison complete: {result['price_mismatches']} price mismatches, "
            f"{result['missing_in_primary']} missing in primary, "
            f"{result['phantom_in_primary']} phantom in primary"
        )

        return result

    def _replace_bad_bars(
        self,
        primary_df: pd.DataFrame,
        ibkr_df: pd.DataFrame,
        mismatches: List[Dict],
    ) -> Tuple[pd.DataFrame, int, int]:
        """
        Replace bad bars with IBKR data.

        - PRICE_MISMATCH: replace with IBKR data
        - MISSING_IN_PRIMARY: insert IBKR bar
        - SPLIT_MISMATCH: mark for quarantine (skip)

        Args:
            primary_df: Primary source DataFrame
            ibkr_df: IBKR DataFrame
            mismatches: List of mismatch records

        Returns:
            (corrected_df, bars_replaced, bars_inserted)
        """
        corrected_df = primary_df.copy()
        bars_replaced = 0
        bars_inserted = 0

        for mismatch in mismatches:
            ts = mismatch["timestamp"]
            mismatch_type = mismatch["type"]

            if mismatch_type == "PRICE_MISMATCH":
                # Replace bar with IBKR data
                if ts in ibkr_df.index:
                    ibkr_row = ibkr_df.loc[ts]
                    if isinstance(ibkr_row, pd.DataFrame):
                        ibkr_row = ibkr_row.iloc[0]

                    for col in ["Open", "High", "Low", "Close", "Volume"]:
                        if col in corrected_df.columns and col in ibkr_row.index:
                            corrected_df.loc[ts, col] = ibkr_row[col]

                    bars_replaced += 1
                    logger.debug(f"Replaced bar at {ts}")

            elif mismatch_type == "MISSING_IN_PRIMARY":
                # Insert IBKR bar
                if ts in ibkr_df.index:
                    ibkr_row = ibkr_df.loc[ts]
                    if isinstance(ibkr_row, pd.DataFrame):
                        ibkr_row = ibkr_row.iloc[0]

                    # Create new row matching the DataFrame's column dtypes
                    new_row = pd.Series(
                        {col: ibkr_row[col]
                         for col in ["Open", "High", "Low", "Close", "Volume"]
                         if col in corrected_df.columns and col in ibkr_row.index},
                        dtype=float,
                    )

                    # Insert at correct position
                    corrected_df = pd.concat([
                        corrected_df[corrected_df.index < ts],
                        pd.DataFrame([new_row], index=[ts]),
                        corrected_df[corrected_df.index > ts],
                    ])

                    bars_inserted += 1
                    logger.debug(f"Inserted bar at {ts}")

            elif mismatch_type == "SPLIT_MISMATCH":
                # Mark for quarantine, don't try to fix
                logger.warning(f"Split mismatch detected at {ts}, quarantining")

        return corrected_df, bars_replaced, bars_inserted

    def _prices_differ(self, price1: float, price2: float, tolerance_pct: float) -> bool:
        """Check if two prices differ beyond tolerance."""
        if pd.isna(price1) or pd.isna(price2):
            return False
        if price1 == 0 or price2 == 0:
            return price1 != price2
        pct_diff = abs((price1 - price2) / price2 * 100)
        return pct_diff > tolerance_pct

    def _volumes_differ(
        self,
        vol1: float,
        vol2: float,
        tolerance_pct: float,
        min_abs_diff: int,
    ) -> bool:
        """Check if volumes differ beyond tolerance."""
        if pd.isna(vol1) or pd.isna(vol2):
            return False
        vol1 = float(vol1)
        vol2 = float(vol2)

        abs_diff = abs(vol1 - vol2)
        if abs_diff < min_abs_diff:
            return False

        if vol2 == 0:
            return vol1 != vol2

        pct_diff = (abs_diff / vol2) * 100
        return pct_diff > tolerance_pct

    def _pct_diff(self, val1: float, val2: float) -> float:
        """Calculate percentage difference between two values."""
        if pd.isna(val1) or pd.isna(val2):
            return 0.0
        if val2 == 0:
            return 0.0 if val1 == 0 else 100.0
        return abs((val1 - val2) / val2) * 100

    def _is_suspected_split(self, price1: float, price2: float) -> bool:
        """Detect if prices suggest a stock split."""
        if pd.isna(price1) or pd.isna(price2):
            return False
        if price1 == 0 or price2 == 0:
            return False

        ratio = price1 / price2

        for known_ratio in self.KNOWN_SPLIT_RATIOS:
            pct_diff = abs((ratio - known_ratio) / known_ratio)
            if pct_diff < 0.01:  # Within 1%
                return True

        return False
