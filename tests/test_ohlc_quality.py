"""
Tests for SPEC-D01: OHLC relationship validation.

Covers:
    - check_ohlc_relationships() standalone function
    - Integration with assess_ohlcv_quality()
    - Integration with generate_quality_report()
    - Integration with DataIntegrityValidator
    - Edge cases: empty DataFrame, single bar, missing columns
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Helpers ───────────────────────────────────────────────────────────


def _make_ohlcv(
    n: int = 500,
    seed: int = 42,
    zero_vol_frac: float = 0.0,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with valid OHLC relationships."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    close = 100.0 + np.cumsum(rng.normal(0.0005, 0.015, n))
    close = np.maximum(close, 1.0)
    # Ensure valid OHLC: High >= max(O,C), Low <= min(O,C), High >= Low
    opn = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(close, opn) * (1 + rng.uniform(0.001, 0.015, n))
    low = np.minimum(close, opn) * (1 - rng.uniform(0.001, 0.015, n))
    vol = rng.integers(500_000, 5_000_000, n).astype(float)

    if zero_vol_frac > 0:
        n_zero = int(n * zero_vol_frac)
        zero_idx = rng.choice(n, size=n_zero, replace=False)
        vol[zero_idx] = 0.0

    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


def _inject_high_lt_close(df: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
    """Set High < Close at specified integer positions."""
    df = df.copy()
    for i in indices:
        # Make High significantly below Close
        df.iloc[i, df.columns.get_loc("High")] = df.iloc[i]["Close"] * 0.95
    return df


def _inject_low_gt_open(df: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
    """Set Low > Open at specified integer positions."""
    df = df.copy()
    for i in indices:
        df.iloc[i, df.columns.get_loc("Low")] = df.iloc[i]["Open"] * 1.05
    return df


def _inject_high_lt_low(df: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
    """Set High < Low at specified integer positions."""
    df = df.copy()
    for i in indices:
        original_high = df.iloc[i]["High"]
        original_low = df.iloc[i]["Low"]
        df.iloc[i, df.columns.get_loc("High")] = original_low * 0.95
        df.iloc[i, df.columns.get_loc("Low")] = original_high * 1.05
    return df


# ═══════════════════════════════════════════════════════════════════════
# check_ohlc_relationships() — Standalone function tests
# ═══════════════════════════════════════════════════════════════════════


class TestCheckOhlcRelationships:
    """Tests for the standalone check_ohlc_relationships function."""

    def test_clean_data_returns_no_violations(self):
        """Valid OHLC data should return an empty violations list."""
        from quant_engine.data.quality import check_ohlc_relationships

        df = _make_ohlcv(n=200)
        violations = check_ohlc_relationships(df)
        assert violations == []

    def test_high_lt_close_detected(self):
        """Bars where High < Close should be flagged."""
        from quant_engine.data.quality import check_ohlc_relationships

        df = _make_ohlcv(n=100)
        df = _inject_high_lt_close(df, [10, 20, 30])
        violations = check_ohlc_relationships(df)

        assert len(violations) >= 1
        high_violation = [v for v in violations if "High < max(Open, Close)" in v]
        assert len(high_violation) == 1
        assert "3 bars" in high_violation[0]

    def test_low_gt_open_detected(self):
        """Bars where Low > Open should be flagged."""
        from quant_engine.data.quality import check_ohlc_relationships

        df = _make_ohlcv(n=100)
        df = _inject_low_gt_open(df, [5, 15])
        violations = check_ohlc_relationships(df)

        assert len(violations) >= 1
        low_violation = [v for v in violations if "Low > min(Open, Close)" in v]
        assert len(low_violation) == 1
        assert "2 bars" in low_violation[0]

    def test_high_lt_low_detected(self):
        """Bars where High < Low should be flagged (spec verification case)."""
        from quant_engine.data.quality import check_ohlc_relationships

        df = _make_ohlcv(n=100)
        df = _inject_high_lt_low(df, [50])
        violations = check_ohlc_relationships(df)

        assert len(violations) >= 1
        hl_violation = [v for v in violations if "High < Low" in v]
        assert len(hl_violation) == 1
        assert "1 bars" in hl_violation[0]

    def test_multiple_violation_types(self):
        """Multiple violation types should all be reported."""
        from quant_engine.data.quality import check_ohlc_relationships

        df = _make_ohlcv(n=100)
        df = _inject_high_lt_close(df, [10])
        df = _inject_low_gt_open(df, [20])
        df = _inject_high_lt_low(df, [30])
        violations = check_ohlc_relationships(df)

        # Should have at least 3 violation messages
        assert len(violations) >= 3

    def test_empty_dataframe_returns_no_violations(self):
        """Empty DataFrame should return no violations."""
        from quant_engine.data.quality import check_ohlc_relationships

        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        violations = check_ohlc_relationships(df)
        assert violations == []

    def test_none_returns_no_violations(self):
        """None input should return no violations."""
        from quant_engine.data.quality import check_ohlc_relationships

        violations = check_ohlc_relationships(None)
        assert violations == []

    def test_missing_columns_returns_no_violations(self):
        """DataFrame missing OHLC columns should return no violations."""
        from quant_engine.data.quality import check_ohlc_relationships

        df = pd.DataFrame({"Close": [100, 101], "Volume": [1000, 2000]})
        violations = check_ohlc_relationships(df)
        assert violations == []

    def test_single_bar_clean(self):
        """Single valid bar should pass."""
        from quant_engine.data.quality import check_ohlc_relationships

        df = pd.DataFrame(
            {"Open": [100.0], "High": [105.0], "Low": [98.0], "Close": [103.0], "Volume": [1000.0]},
            index=pd.bdate_range("2022-01-03", periods=1),
        )
        violations = check_ohlc_relationships(df)
        assert violations == []

    def test_single_bar_violated(self):
        """Single bar with High < Low should be flagged."""
        from quant_engine.data.quality import check_ohlc_relationships

        df = pd.DataFrame(
            {"Open": [100.0], "High": [95.0], "Low": [105.0], "Close": [103.0], "Volume": [1000.0]},
            index=pd.bdate_range("2022-01-03", periods=1),
        )
        violations = check_ohlc_relationships(df)
        assert len(violations) >= 1

    def test_exact_equality_passes(self):
        """O=H=L=C (flat bar) should be valid — no strict inequality required."""
        from quant_engine.data.quality import check_ohlc_relationships

        df = pd.DataFrame(
            {"Open": [100.0], "High": [100.0], "Low": [100.0], "Close": [100.0], "Volume": [1000.0]},
            index=pd.bdate_range("2022-01-03", periods=1),
        )
        violations = check_ohlc_relationships(df)
        assert violations == []


# ═══════════════════════════════════════════════════════════════════════
# _ohlc_violation_counts() — Internal helper tests
# ═══════════════════════════════════════════════════════════════════════


class TestOhlcViolationCounts:
    """Tests for the internal _ohlc_violation_counts helper."""

    def test_clean_data_all_zeros(self):
        """Valid data should return all zero counts."""
        from quant_engine.data.quality import _ohlc_violation_counts

        df = _make_ohlcv(n=100)
        counts = _ohlc_violation_counts(df)
        assert counts == {"bad_high": 0, "bad_low": 0, "bad_hl": 0}

    def test_counts_match_injected_violations(self):
        """Counts should exactly match the number of injected violations."""
        from quant_engine.data.quality import _ohlc_violation_counts

        df = _make_ohlcv(n=100)
        df = _inject_high_lt_close(df, [10, 20, 30])
        counts = _ohlc_violation_counts(df)
        assert counts["bad_high"] == 3

    def test_empty_returns_zeros(self):
        """Empty DataFrame should return all zeros."""
        from quant_engine.data.quality import _ohlc_violation_counts

        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        counts = _ohlc_violation_counts(df)
        assert counts == {"bad_high": 0, "bad_low": 0, "bad_hl": 0}


# ═══════════════════════════════════════════════════════════════════════
# assess_ohlcv_quality() — Integration tests
# ═══════════════════════════════════════════════════════════════════════


class TestAssessQualityOhlcIntegration:
    """Tests for OHLC validation integrated into assess_ohlcv_quality."""

    def test_clean_data_passes_with_ohlc_metrics(self):
        """Clean data should pass and include zero OHLC violation metrics."""
        from quant_engine.data.quality import assess_ohlcv_quality

        df = _make_ohlcv(n=200)
        report = assess_ohlcv_quality(df)

        assert report.passed is True
        assert report.metrics["ohlc_violation_count"] == 0.0
        assert report.metrics["ohlc_high_violations"] == 0.0
        assert report.metrics["ohlc_low_violations"] == 0.0
        assert report.metrics["ohlc_hl_violations"] == 0.0

    def test_ohlc_violations_cause_failure(self):
        """OHLC violations should cause the quality check to fail."""
        from quant_engine.data.quality import assess_ohlcv_quality

        df = _make_ohlcv(n=200)
        df = _inject_high_lt_low(df, [10, 20, 30])
        report = assess_ohlcv_quality(df)

        assert report.passed is False
        assert report.metrics["ohlc_violation_count"] > 0
        ohlc_warnings = [w for w in report.warnings if "ohlc_violations" in w]
        assert len(ohlc_warnings) == 1

    def test_ohlc_violation_metrics_accurate(self):
        """OHLC violation metric counts should match injected violations."""
        from quant_engine.data.quality import assess_ohlcv_quality

        df = _make_ohlcv(n=200)
        df = _inject_high_lt_close(df, [10, 20])
        df = _inject_low_gt_open(df, [50])
        report = assess_ohlcv_quality(df)

        assert report.metrics["ohlc_high_violations"] == 2.0
        assert report.metrics["ohlc_low_violations"] == 1.0

    def test_fail_on_error_raises_on_ohlc_violations(self):
        """fail_on_error=True should raise ValueError when OHLC violations exist."""
        from quant_engine.data.quality import assess_ohlcv_quality

        df = _make_ohlcv(n=200)
        df = _inject_high_lt_low(df, [10])

        with pytest.raises(ValueError, match="quality check failed"):
            assess_ohlcv_quality(df, fail_on_error=True)

    def test_ohlc_warning_message_contains_details(self):
        """OHLC warning message should describe the specific violations."""
        from quant_engine.data.quality import assess_ohlcv_quality

        df = _make_ohlcv(n=200)
        df = _inject_high_lt_low(df, [10])
        report = assess_ohlcv_quality(df)

        ohlc_warnings = [w for w in report.warnings if "ohlc_violations" in w]
        assert len(ohlc_warnings) == 1
        assert "High < Low" in ohlc_warnings[0]


# ═══════════════════════════════════════════════════════════════════════
# generate_quality_report() — Dashboard integration tests
# ═══════════════════════════════════════════════════════════════════════


class TestQualityReportOhlcIntegration:
    """Tests for OHLC validation integrated into generate_quality_report."""

    def test_clean_universe_has_zero_ohlc_violations(self):
        """Clean universe should report zero OHLC violations for all tickers."""
        from quant_engine.data.quality import generate_quality_report

        universe = {
            f"STOCK{i}": _make_ohlcv(n=200, seed=42 + i) for i in range(3)
        }
        report = generate_quality_report(universe)

        assert "ohlc_violation_count" in report.columns
        assert (report["ohlc_violation_count"] == 0).all()

    def test_corrupted_ticker_has_nonzero_ohlc_violations(self):
        """Ticker with OHLC violations should have nonzero count in report."""
        from quant_engine.data.quality import generate_quality_report

        clean = _make_ohlcv(n=200, seed=42)
        corrupted = _make_ohlcv(n=200, seed=43)
        corrupted = _inject_high_lt_low(corrupted, [10, 20, 30, 40, 50])

        universe = {"CLEAN": clean, "BAD": corrupted}
        report = generate_quality_report(universe)

        assert report.loc["CLEAN", "ohlc_violation_count"] == 0
        assert report.loc["BAD", "ohlc_violation_count"] > 0

    def test_ohlc_violations_lower_quality_score(self):
        """Tickers with OHLC violations should have a lower quality score."""
        from quant_engine.data.quality import generate_quality_report

        clean = _make_ohlcv(n=200, seed=42)
        corrupted = _make_ohlcv(n=200, seed=43)
        corrupted = _inject_high_lt_low(corrupted, list(range(0, 100, 2)))

        universe = {"CLEAN": clean, "BAD": corrupted}
        report = generate_quality_report(universe)

        assert report.loc["CLEAN", "quality_score"] > report.loc["BAD", "quality_score"]

    def test_empty_ticker_has_zero_ohlc_violations(self):
        """Empty DataFrame ticker should have zero OHLC violations."""
        from quant_engine.data.quality import generate_quality_report

        universe = {"EMPTY": pd.DataFrame()}
        report = generate_quality_report(universe)

        assert report.loc["EMPTY", "ohlc_violation_count"] == 0


# ═══════════════════════════════════════════════════════════════════════
# DataIntegrityValidator — Propagation tests
# ═══════════════════════════════════════════════════════════════════════


class TestDataIntegrityOhlcPropagation:
    """Tests that OHLC violations propagate through DataIntegrityValidator."""

    def test_validator_blocks_ohlc_corrupt_data(self):
        """DataIntegrityValidator should block data with OHLC violations."""
        from quant_engine.validation.data_integrity import DataIntegrityValidator

        corrupted = _make_ohlcv(n=200, seed=42)
        corrupted = _inject_high_lt_low(corrupted, [10])

        validator = DataIntegrityValidator(fail_fast=True)
        with pytest.raises(RuntimeError, match="integrity check failed"):
            validator.validate_universe({"BAD": corrupted})

    def test_validator_passes_clean_data(self):
        """DataIntegrityValidator should pass clean OHLC data."""
        from quant_engine.validation.data_integrity import DataIntegrityValidator

        clean = _make_ohlcv(n=200, seed=42)
        validator = DataIntegrityValidator(fail_fast=True)
        result = validator.validate_universe({"CLEAN": clean})

        assert result.passed is True
        assert result.n_stocks_failed == 0

    def test_validator_no_fail_fast_reports_ohlc_failures(self):
        """Without fail_fast, validator should report OHLC failures in result."""
        from quant_engine.validation.data_integrity import DataIntegrityValidator

        corrupted = _make_ohlcv(n=200, seed=42)
        corrupted = _inject_high_lt_low(corrupted, [10])

        validator = DataIntegrityValidator(fail_fast=False)
        result = validator.validate_universe({"BAD": corrupted})

        assert result.passed is False
        assert "BAD" in result.failed_tickers
        assert any("ohlc" in r.lower() for reasons in result.failed_reasons.values() for r in reasons)
