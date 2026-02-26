"""
Tests for SPEC-B02: Fix look-ahead bias in alternative data.

Verifies that every method in AlternativeDataProvider respects the
``as_of_date`` parameter so that backtest queries never leak future data.
Uses a mock WRDS backend to return controlled date-stamped records.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from quant_engine.data.alternative import AlternativeDataProvider, compute_alternative_features


# ---------------------------------------------------------------------------
# Mock WRDS helpers
# ---------------------------------------------------------------------------

def _make_earnings_df() -> pd.DataFrame:
    """Simulate WRDS I/B/E/S output spanning 2022-2025."""
    dates = pd.to_datetime([
        "2022-04-15", "2022-07-20", "2022-10-18", "2023-01-25",
        "2023-04-20", "2023-07-18", "2023-10-17", "2024-01-24",
        "2024-04-19", "2024-07-22", "2024-10-21", "2025-01-23",
    ])
    n = len(dates)
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "anndats_act": dates,
        "ticker": "AAPL",
        "fpedats": dates - timedelta(days=30),
        "actual": rng.normal(1.50, 0.3, n),
        "meanest": rng.normal(1.45, 0.25, n),
        "stdev": rng.uniform(0.05, 0.2, n),
        "numest": rng.integers(10, 40, n),
    })
    df["surprise_pct"] = np.where(
        df["meanest"].abs() > 1e-6,
        (df["actual"] - df["meanest"]) / df["meanest"].abs() * 100,
        np.nan,
    )
    df["dispersion"] = np.where(
        df["meanest"].abs() > 1e-6,
        df["stdev"] / df["meanest"].abs(),
        np.nan,
    )
    return df.set_index(["anndats_act", "ticker"]).sort_index()


def _make_options_df() -> pd.DataFrame:
    """Simulate OptionMetrics data spanning 2022-2025."""
    dates = pd.bdate_range("2022-01-03", "2025-06-30")
    rng = np.random.default_rng(7)
    n = len(dates)
    return pd.DataFrame({
        "date": dates,
        "put_volume": rng.integers(1000, 50000, n),
        "call_volume": rng.integers(1000, 50000, n),
        "put_oi": rng.integers(10000, 500000, n),
        "call_oi": rng.integers(10000, 500000, n),
    })


def _make_short_interest_df() -> pd.DataFrame:
    """Simulate Compustat short interest spanning 2022-2025."""
    # Bimonthly dates (1st and 15th)
    dates = pd.date_range("2022-01-15", "2025-06-15", freq="SMS")
    rng = np.random.default_rng(11)
    n = len(dates)
    return pd.DataFrame({
        "settlement_date": dates,
        "short_interest": rng.integers(500_000, 5_000_000, n).astype(float),
        "avg_daily_volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    })


def _make_insider_df() -> pd.DataFrame:
    """Simulate TFN insider transactions spanning 2022-2025."""
    rng = np.random.default_rng(13)
    dates = pd.date_range("2022-01-10", "2025-06-10", freq="30D")
    n = len(dates)
    return pd.DataFrame({
        "filing_date": dates,
        "insider_name": [f"Insider_{i % 5}" for i in range(n)],
        "shares": rng.integers(100, 10000, n),
        "price": rng.uniform(100, 200, n).round(2),
        "buy_sell": rng.choice(["A", "D"], n),
        "transaction_type": "P",
    })


def _make_institutional_df() -> pd.DataFrame:
    """Simulate TFN s34 institutional ownership spanning 2022-2025."""
    dates = pd.date_range("2022-03-31", "2025-06-30", freq="QE")
    rng = np.random.default_rng(17)
    n = len(dates)
    df = pd.DataFrame({
        "fdate": dates,
        "ticker": "AAPL",
        "total_shares_held": rng.integers(1_000_000, 50_000_000, n).astype(float),
        "num_institutions": rng.integers(100, 500, n).astype(float),
    })
    return df.set_index(["fdate", "ticker"]).sort_index()


def _build_mock_wrds(
    earnings_df=None,
    options_df=None,
    short_interest_df=None,
    insider_df=None,
    institutional_df=None,
):
    """Build a MagicMock that mimics the WRDS provider interface."""
    mock = MagicMock()
    mock.available.return_value = True
    mock.resolve_permno.return_value = "12345"

    def _filter_earnings(tickers, start_date, end_date):
        if earnings_df is None:
            return pd.DataFrame()
        df = earnings_df.reset_index()
        df["anndats_act"] = pd.to_datetime(df["anndats_act"])
        mask = (
            (df["anndats_act"] >= pd.Timestamp(start_date))
            & (df["anndats_act"] <= pd.Timestamp(end_date))
        )
        result = df[mask].set_index(["anndats_act", "ticker"]).sort_index()
        return result

    def _filter_options(permno, start_date, end_date):
        if options_df is None:
            return pd.DataFrame()
        df = options_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        mask = (
            (df["date"] >= pd.Timestamp(start_date))
            & (df["date"] <= pd.Timestamp(end_date))
        )
        return df[mask].reset_index(drop=True)

    def _filter_short_interest(permno, start_date, end_date):
        if short_interest_df is None:
            return pd.DataFrame()
        df = short_interest_df.copy()
        df["settlement_date"] = pd.to_datetime(df["settlement_date"])
        mask = (
            (df["settlement_date"] >= pd.Timestamp(start_date))
            & (df["settlement_date"] <= pd.Timestamp(end_date))
        )
        return df[mask].reset_index(drop=True)

    def _filter_insider(permno, start_date, end_date):
        if insider_df is None:
            return pd.DataFrame()
        df = insider_df.copy()
        df["filing_date"] = pd.to_datetime(df["filing_date"])
        mask = (
            (df["filing_date"] >= pd.Timestamp(start_date))
            & (df["filing_date"] <= pd.Timestamp(end_date))
        )
        return df[mask].reset_index(drop=True)

    def _filter_institutional(tickers, start_date, end_date):
        if institutional_df is None:
            return pd.DataFrame()
        df = institutional_df.reset_index()
        df["fdate"] = pd.to_datetime(df["fdate"])
        mask = (
            (df["fdate"] >= pd.Timestamp(start_date))
            & (df["fdate"] <= pd.Timestamp(end_date))
        )
        result = df[mask].set_index(["fdate", "ticker"]).sort_index()
        return result

    mock.get_earnings_surprises.side_effect = _filter_earnings
    mock.query_options_volume.side_effect = _filter_options
    mock.query_short_interest.side_effect = _filter_short_interest
    mock.query_insider_transactions.side_effect = _filter_insider
    mock.get_institutional_ownership.side_effect = _filter_institutional
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEarningsSurpriseLookahead(unittest.TestCase):
    """Verify get_earnings_surprise() respects as_of_date."""

    def setUp(self):
        self.earnings_df = _make_earnings_df()
        self.mock_wrds = _build_mock_wrds(earnings_df=self.earnings_df)

    def _make_provider(self) -> AlternativeDataProvider:
        provider = AlternativeDataProvider.__new__(AlternativeDataProvider)
        provider.cache_dir = None
        provider._wrds = self.mock_wrds
        return provider

    def test_as_of_date_excludes_future_earnings(self):
        """Earnings after as_of_date must not appear in the result."""
        provider = self._make_provider()
        cutoff = datetime(2024, 1, 1)
        result = provider.get_earnings_surprise(
            "AAPL", lookback_days=3650, as_of_date=cutoff,
        )
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        max_date = pd.Timestamp(result["report_date"].max())
        self.assertLessEqual(
            max_date, pd.Timestamp(cutoff),
            f"Found earnings dated {max_date} after as_of_date {cutoff}",
        )

    def test_as_of_date_none_uses_current_time(self):
        """When as_of_date is None, behaviour should be equivalent to now()."""
        provider = self._make_provider()
        result = provider.get_earnings_surprise("AAPL", lookback_days=3650)
        # Should return all data since mock data goes up to 2025-01-23
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)

    def test_as_of_date_early_returns_no_data(self):
        """as_of_date before any data should return None."""
        provider = self._make_provider()
        result = provider.get_earnings_surprise(
            "AAPL", lookback_days=90, as_of_date=datetime(2020, 1, 1),
        )
        self.assertIsNone(result)

    def test_lookback_filter_uses_as_of_date(self):
        """The lookback filter should be relative to as_of_date, not now()."""
        provider = self._make_provider()
        # Ask for 90 days of lookback from 2023-08-01
        # That window is ~2023-05-03 to 2023-08-01
        cutoff = datetime(2023, 8, 1)
        result = provider.get_earnings_surprise(
            "AAPL", lookback_days=90, as_of_date=cutoff,
        )
        if result is not None and not result.empty:
            min_date = pd.Timestamp(result["report_date"].min())
            expected_start = pd.Timestamp(cutoff - timedelta(days=90))
            self.assertGreaterEqual(min_date, expected_start)
            max_date = pd.Timestamp(result["report_date"].max())
            self.assertLessEqual(max_date, pd.Timestamp(cutoff))

    def test_wrds_called_with_correct_dates(self):
        """The WRDS query must use as_of_date as the end bound."""
        provider = self._make_provider()
        cutoff = datetime(2023, 6, 15)
        provider.get_earnings_surprise(
            "AAPL", lookback_days=365, as_of_date=cutoff,
        )
        call_args = self.mock_wrds.get_earnings_surprises.call_args
        self.assertEqual(call_args.kwargs["end_date"], "2023-06-15")


class TestOptionsFlowLookahead(unittest.TestCase):
    """Verify get_options_flow() respects as_of_date."""

    def setUp(self):
        self.options_df = _make_options_df()
        self.mock_wrds = _build_mock_wrds(options_df=self.options_df)

    def _make_provider(self) -> AlternativeDataProvider:
        provider = AlternativeDataProvider.__new__(AlternativeDataProvider)
        provider.cache_dir = None
        provider._wrds = self.mock_wrds
        return provider

    def test_as_of_date_excludes_future_options(self):
        provider = self._make_provider()
        cutoff = datetime(2024, 1, 1)
        result = provider.get_options_flow("AAPL", as_of_date=cutoff)
        self.assertIsNotNone(result)
        max_date = pd.Timestamp(result["date"].max())
        self.assertLessEqual(max_date, pd.Timestamp(cutoff))

    def test_wrds_called_with_correct_dates(self):
        provider = self._make_provider()
        cutoff = datetime(2023, 6, 15)
        provider.get_options_flow("AAPL", as_of_date=cutoff)
        call_args = self.mock_wrds.query_options_volume.call_args
        self.assertEqual(call_args.kwargs["end_date"], "2023-06-15")


class TestShortInterestLookahead(unittest.TestCase):
    """Verify get_short_interest() respects as_of_date."""

    def setUp(self):
        self.si_df = _make_short_interest_df()
        self.mock_wrds = _build_mock_wrds(short_interest_df=self.si_df)

    def _make_provider(self) -> AlternativeDataProvider:
        provider = AlternativeDataProvider.__new__(AlternativeDataProvider)
        provider.cache_dir = None
        provider._wrds = self.mock_wrds
        return provider

    def test_as_of_date_excludes_future_short_interest(self):
        provider = self._make_provider()
        cutoff = datetime(2024, 1, 1)
        result = provider.get_short_interest("AAPL", as_of_date=cutoff)
        self.assertIsNotNone(result)
        max_date = pd.Timestamp(result["settlement_date"].max())
        self.assertLessEqual(max_date, pd.Timestamp(cutoff))

    def test_wrds_called_with_correct_dates(self):
        provider = self._make_provider()
        cutoff = datetime(2023, 6, 15)
        provider.get_short_interest("AAPL", as_of_date=cutoff)
        call_args = self.mock_wrds.query_short_interest.call_args
        self.assertEqual(call_args.kwargs["end_date"], "2023-06-15")


class TestInsiderTransactionsLookahead(unittest.TestCase):
    """Verify get_insider_transactions() respects as_of_date."""

    def setUp(self):
        self.insider_df = _make_insider_df()
        self.mock_wrds = _build_mock_wrds(insider_df=self.insider_df)

    def _make_provider(self) -> AlternativeDataProvider:
        provider = AlternativeDataProvider.__new__(AlternativeDataProvider)
        provider.cache_dir = None
        provider._wrds = self.mock_wrds
        return provider

    def test_as_of_date_excludes_future_insider(self):
        provider = self._make_provider()
        cutoff = datetime(2024, 1, 1)
        result = provider.get_insider_transactions("AAPL", as_of_date=cutoff)
        self.assertIsNotNone(result)
        max_date = pd.Timestamp(result["filing_date"].max())
        self.assertLessEqual(max_date, pd.Timestamp(cutoff))

    def test_wrds_called_with_correct_dates(self):
        provider = self._make_provider()
        cutoff = datetime(2023, 6, 15)
        provider.get_insider_transactions("AAPL", as_of_date=cutoff)
        call_args = self.mock_wrds.query_insider_transactions.call_args
        self.assertEqual(call_args.kwargs["end_date"], "2023-06-15")


class TestInstitutionalOwnershipLookahead(unittest.TestCase):
    """Verify get_institutional_ownership() respects as_of_date."""

    def setUp(self):
        self.inst_df = _make_institutional_df()
        self.mock_wrds = _build_mock_wrds(institutional_df=self.inst_df)

    def _make_provider(self) -> AlternativeDataProvider:
        provider = AlternativeDataProvider.__new__(AlternativeDataProvider)
        provider.cache_dir = None
        provider._wrds = self.mock_wrds
        return provider

    def test_as_of_date_excludes_future_ownership(self):
        provider = self._make_provider()
        cutoff = datetime(2024, 1, 1)
        result = provider.get_institutional_ownership("AAPL", as_of_date=cutoff)
        self.assertIsNotNone(result)
        max_date = pd.Timestamp(result["fdate"].max())
        self.assertLessEqual(max_date, pd.Timestamp(cutoff))

    def test_wrds_called_with_correct_dates(self):
        provider = self._make_provider()
        cutoff = datetime(2023, 6, 15)
        provider.get_institutional_ownership("AAPL", as_of_date=cutoff)
        call_args = self.mock_wrds.get_institutional_ownership.call_args
        self.assertEqual(call_args.kwargs["end_date"], "2023-06-15")


class TestComputeAlternativeFeaturesLookahead(unittest.TestCase):
    """Verify compute_alternative_features() threads as_of_date correctly."""

    def setUp(self):
        self.earnings_df = _make_earnings_df()
        self.inst_df = _make_institutional_df()
        self.mock_wrds = _build_mock_wrds(
            earnings_df=self.earnings_df,
            institutional_df=self.inst_df,
        )

    def _make_provider(self) -> AlternativeDataProvider:
        provider = AlternativeDataProvider.__new__(AlternativeDataProvider)
        provider.cache_dir = None
        provider._wrds = self.mock_wrds
        return provider

    def test_as_of_date_threaded_to_sub_calls(self):
        """compute_alternative_features passes as_of_date to all data methods."""
        provider = self._make_provider()
        cutoff = datetime(2024, 1, 1)
        result = compute_alternative_features(
            "AAPL", provider=provider, as_of_date=cutoff,
        )
        # Earnings query must have been bounded by as_of_date
        earnings_call = self.mock_wrds.get_earnings_surprises.call_args
        self.assertEqual(earnings_call.kwargs["end_date"], "2024-01-01")

        # Institutional ownership query must also be bounded
        inst_call = self.mock_wrds.get_institutional_ownership.call_args
        self.assertEqual(inst_call.kwargs["end_date"], "2024-01-01")

        # All dates in the result must be <= as_of_date
        if not result.empty:
            max_idx = result.index.max()
            self.assertLessEqual(
                max_idx, pd.Timestamp(cutoff),
                f"Feature index {max_idx} exceeds as_of_date {cutoff}",
            )

    def test_as_of_date_none_works(self):
        """Default behaviour (as_of_date=None) does not raise."""
        provider = self._make_provider()
        result = compute_alternative_features("AAPL", provider=provider)
        # Should succeed — may or may not have data depending on mock range
        self.assertIsInstance(result, pd.DataFrame)


class TestNoRemainingDatetimeNow(unittest.TestCase):
    """Structural test: verify no direct datetime.now() usage outside the
    guarded ``as_of_date if as_of_date is not None else datetime.now()``
    pattern remains in alternative.py."""

    def test_no_unguarded_datetime_now(self):
        """Every datetime.now() in alternative.py must be in a reference_dt assignment."""
        import ast
        import quant_engine.data.alternative as mod

        source_file = mod.__file__
        with open(source_file) as f:
            source = f.read()

        # Use AST to find all string literals (docstrings) and their line ranges
        tree = ast.parse(source)
        docstring_lines: set[int] = set()
        for node in ast.walk(tree):
            # Docstrings appear as Expr nodes containing a Constant string
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                for ln in range(node.lineno, node.end_lineno + 1):
                    docstring_lines.add(ln)

        lines = source.split("\n")
        violations = []
        for i, line in enumerate(lines, 1):
            if i in docstring_lines:
                continue
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "datetime.now()" in stripped:
                # The only allowed pattern is: reference_dt = as_of_date if ... else datetime.now()
                if "reference_dt" not in stripped:
                    violations.append((i, stripped))

        self.assertEqual(
            len(violations), 0,
            f"Found unguarded datetime.now() calls:\n"
            + "\n".join(f"  L{n}: {l}" for n, l in violations),
        )


if __name__ == "__main__":
    unittest.main()
