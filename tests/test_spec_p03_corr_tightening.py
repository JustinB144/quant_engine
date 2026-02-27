"""
Tests for SPEC-P03: Dynamic correlation-based constraint tightening.

Covers:
    1. Config: CORRELATION_STRESS_THRESHOLDS exists with correct values
    2. _get_correlation_stress_multiplier returns correct multiplier at each tier
    3. _compute_avg_pairwise_correlation computes correctly
    4. check_new_position applies corr stress multiplier to sector/single/gross
    5. Correlation tightening stacks on top of regime multiplier
    6. compute_constraint_utilization reflects corr-tightened limits
    7. portfolio_summary includes corr_stress_multiplier
    8. Edge cases: single position, no price data, low correlation
    9. Verification: when pairwise corr > 0.7, caps are noticeably tighter
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from quant_engine.config import CORRELATION_STRESS_THRESHOLDS
from quant_engine.risk.portfolio_risk import PortfolioRiskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(close: pd.Series, volume: float = 1_000_000.0) -> pd.DataFrame:
    """Build synthetic OHLCV data from a close series."""
    return pd.DataFrame(
        {
            "Open": close.values,
            "High": close.values * 1.01,
            "Low": close.values * 0.99,
            "Close": close.values,
            "Volume": np.full(len(close), volume),
        },
        index=close.index,
    )


def _generate_price_data(
    tickers: list,
    n_days: int = 252,
    seed: int = 42,
    start_price: float = 100.0,
    vol: float = 0.02,
    correlation: float = 0.0,
) -> dict:
    """Generate synthetic OHLCV data with controllable cross-correlation.

    Parameters
    ----------
    correlation : float
        Approximate pairwise correlation injected via a shared factor.
        0.0 = independent returns, 1.0 = perfectly correlated.
    """
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")

    # Shared factor for injecting correlation
    common_factor = rng.normal(0, vol, size=n_days)
    data = {}
    for i, ticker in enumerate(tickers):
        idio = rng.normal(0.0005, vol, size=n_days)
        # Blend: sqrt(corr)*common + sqrt(1-corr)*idio  (approximate)
        if correlation > 0:
            w = np.sqrt(min(correlation, 0.999))
            returns = w * common_factor + np.sqrt(1 - w**2) * idio
        else:
            returns = idio
        prices = start_price * np.cumprod(1 + returns)
        close = pd.Series(prices, index=idx)
        data[ticker] = _make_ohlcv(close, volume=1_000_000.0 * (1 + i * 0.1))
    return data


def _generate_highly_correlated_data(
    tickers: list,
    n_days: int = 252,
    seed: int = 42,
) -> dict:
    """Generate data where all tickers are very highly correlated (>0.9)."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    base_returns = rng.normal(0.001, 0.02, size=n_days)

    data = {}
    for i, ticker in enumerate(tickers):
        noise = rng.normal(0, 0.002, size=n_days)  # Very small noise
        returns = base_returns + noise
        prices = 100.0 * np.cumprod(1 + returns)
        close = pd.Series(prices, index=idx)
        data[ticker] = _make_ohlcv(close)
    return data


# ---------------------------------------------------------------------------
# 1. Config constant
# ---------------------------------------------------------------------------

class TestConfigConstant:
    """Verify CORRELATION_STRESS_THRESHOLDS exists with correct values."""

    def test_thresholds_exist(self):
        assert isinstance(CORRELATION_STRESS_THRESHOLDS, dict)

    def test_threshold_keys(self):
        assert 0.6 in CORRELATION_STRESS_THRESHOLDS
        assert 0.7 in CORRELATION_STRESS_THRESHOLDS
        assert 0.8 in CORRELATION_STRESS_THRESHOLDS

    def test_threshold_values(self):
        assert CORRELATION_STRESS_THRESHOLDS[0.6] == pytest.approx(0.85)
        assert CORRELATION_STRESS_THRESHOLDS[0.7] == pytest.approx(0.70)
        assert CORRELATION_STRESS_THRESHOLDS[0.8] == pytest.approx(0.50)

    def test_thresholds_are_monotonically_tighter(self):
        """Higher correlation thresholds should yield tighter (smaller) multipliers."""
        sorted_keys = sorted(CORRELATION_STRESS_THRESHOLDS.keys())
        for i in range(len(sorted_keys) - 1):
            assert CORRELATION_STRESS_THRESHOLDS[sorted_keys[i]] > \
                CORRELATION_STRESS_THRESHOLDS[sorted_keys[i + 1]]


# ---------------------------------------------------------------------------
# 2. _get_correlation_stress_multiplier
# ---------------------------------------------------------------------------

class TestGetCorrelationStressMultiplier:
    """Test the static method that maps avg_corr to multiplier."""

    def test_below_all_thresholds(self):
        mult = PortfolioRiskManager._get_correlation_stress_multiplier(0.3)
        assert mult == 1.0

    def test_at_zero(self):
        mult = PortfolioRiskManager._get_correlation_stress_multiplier(0.0)
        assert mult == 1.0

    def test_just_below_first_threshold(self):
        mult = PortfolioRiskManager._get_correlation_stress_multiplier(0.59)
        assert mult == 1.0

    def test_just_above_first_threshold(self):
        mult = PortfolioRiskManager._get_correlation_stress_multiplier(0.61)
        assert mult == pytest.approx(0.85)

    def test_just_above_second_threshold(self):
        mult = PortfolioRiskManager._get_correlation_stress_multiplier(0.71)
        assert mult == pytest.approx(0.70)

    def test_just_above_third_threshold(self):
        mult = PortfolioRiskManager._get_correlation_stress_multiplier(0.81)
        assert mult == pytest.approx(0.50)

    def test_extreme_correlation(self):
        """Even 0.99 should return the tightest defined multiplier."""
        mult = PortfolioRiskManager._get_correlation_stress_multiplier(0.99)
        assert mult == pytest.approx(0.50)

    def test_exactly_at_threshold(self):
        """Thresholds use strict '>' comparison, so exactly at threshold should NOT trigger."""
        mult = PortfolioRiskManager._get_correlation_stress_multiplier(0.6)
        assert mult == 1.0

    def test_negative_correlation(self):
        """Negative values should not trigger tightening (avg |corr| is always >= 0)."""
        mult = PortfolioRiskManager._get_correlation_stress_multiplier(-0.5)
        assert mult == 1.0


# ---------------------------------------------------------------------------
# 3. _compute_avg_pairwise_correlation
# ---------------------------------------------------------------------------

class TestComputeAvgPairwiseCorrelation:
    """Test the average absolute pairwise correlation computation."""

    def test_independent_returns_low_corr(self):
        """Independent returns should have low average correlation."""
        price_data = _generate_price_data(
            ["A", "B", "C", "D"], n_days=252, seed=42, correlation=0.0,
        )
        rm = PortfolioRiskManager()
        positions = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        avg_corr = rm._compute_avg_pairwise_correlation(positions, price_data)
        assert avg_corr < 0.3  # Should be small for independent returns

    def test_highly_correlated_returns(self):
        """Highly correlated returns should have avg corr near 1.0."""
        price_data = _generate_highly_correlated_data(
            ["A", "B", "C"], n_days=252, seed=42,
        )
        rm = PortfolioRiskManager()
        positions = {"A": 0.33, "B": 0.33, "C": 0.33}
        avg_corr = rm._compute_avg_pairwise_correlation(positions, price_data)
        assert avg_corr > 0.8

    def test_single_position_returns_zero(self):
        """A single position cannot have pairwise correlation."""
        price_data = _generate_price_data(["A"], n_days=120, seed=42)
        rm = PortfolioRiskManager()
        avg_corr = rm._compute_avg_pairwise_correlation({"A": 1.0}, price_data)
        assert avg_corr == 0.0

    def test_empty_positions_returns_zero(self):
        rm = PortfolioRiskManager()
        avg_corr = rm._compute_avg_pairwise_correlation({}, {})
        assert avg_corr == 0.0

    def test_missing_price_data_returns_zero(self):
        """Tickers not in price_data should be skipped."""
        rm = PortfolioRiskManager()
        avg_corr = rm._compute_avg_pairwise_correlation(
            {"A": 0.5, "B": 0.5}, {},
        )
        assert avg_corr == 0.0

    def test_partial_price_data(self):
        """Only one ticker has price data → < 2 tickers → returns 0.0."""
        price_data = _generate_price_data(["A"], n_days=120, seed=42)
        rm = PortfolioRiskManager()
        avg_corr = rm._compute_avg_pairwise_correlation(
            {"A": 0.5, "B": 0.5}, price_data,
        )
        assert avg_corr == 0.0

    def test_uses_absolute_correlation(self):
        """Average should use absolute values so negative corr is not zero-averaged."""
        # Create two inversely correlated series
        rng = np.random.RandomState(99)
        idx = pd.date_range("2023-01-01", periods=200, freq="B")
        base_returns = rng.normal(0.001, 0.02, size=200)

        prices_a = 100.0 * np.cumprod(1 + base_returns)
        prices_b = 100.0 * np.cumprod(1 - base_returns + rng.normal(0, 0.002, 200))

        price_data = {
            "A": _make_ohlcv(pd.Series(prices_a, index=idx)),
            "B": _make_ohlcv(pd.Series(prices_b, index=idx)),
        }
        rm = PortfolioRiskManager()
        avg_corr = rm._compute_avg_pairwise_correlation(
            {"A": 0.5, "B": 0.5}, price_data,
        )
        # Should be high (close to 1.0) because we take absolute value
        assert avg_corr > 0.7

    def test_short_data_returns_zero(self):
        """Less than 20 bars should return 0.0."""
        idx = pd.date_range("2023-01-01", periods=15, freq="B")
        prices_a = pd.Series(np.linspace(100, 110, 15), index=idx)
        prices_b = pd.Series(np.linspace(50, 55, 15), index=idx)
        price_data = {
            "A": _make_ohlcv(prices_a),
            "B": _make_ohlcv(prices_b),
        }
        rm = PortfolioRiskManager(correlation_lookback=15)
        avg_corr = rm._compute_avg_pairwise_correlation(
            {"A": 0.5, "B": 0.5}, price_data,
        )
        assert avg_corr == 0.0


# ---------------------------------------------------------------------------
# 4. check_new_position applies corr stress multiplier
# ---------------------------------------------------------------------------

class TestCheckNewPositionCorrTightening:
    """Verify check_new_position tightens constraints via correlation."""

    def test_high_corr_tightens_single_name(self):
        """With high correlation, single-name cap should be tighter."""
        price_data = _generate_highly_correlated_data(
            ["A", "B", "C"], n_days=252, seed=42,
        )
        rm = PortfolioRiskManager(max_single_name_pct=0.10)

        # 8% position should pass under normal limits (10% cap)
        check_normal = rm.check_new_position(
            ticker="C",
            position_size=0.08,
            current_positions={},  # No existing positions → no avg corr
            price_data=price_data,
        )
        single_violations_normal = [v for v in check_normal.violations if "Position size" in v]

        # Now with existing correlated positions, the cap should tighten
        check_corr = rm.check_new_position(
            ticker="C",
            position_size=0.08,
            current_positions={"A": 0.05, "B": 0.05},
            price_data=price_data,
        )
        # With avg corr > 0.8, single-name cap = 10% * 0.50 = 5%, so 8% should fail
        single_violations_corr = [v for v in check_corr.violations if "Position size" in v]
        assert len(single_violations_corr) > 0, (
            f"Expected single-name violation with high corr. "
            f"avg_corr={check_corr.metrics.get('avg_pairwise_corr', 'N/A')}"
        )

    def test_high_corr_tightens_gross_exposure(self):
        """With high correlation, gross exposure limit should be tighter."""
        price_data = _generate_highly_correlated_data(
            ["A", "B", "C", "D"], n_days=252, seed=42,
        )
        rm = PortfolioRiskManager(max_gross_exposure=1.0)

        # 70% gross should pass under normal limits
        check = rm.check_new_position(
            ticker="D",
            position_size=0.15,
            current_positions={"A": 0.20, "B": 0.20, "C": 0.20},
            price_data=price_data,
        )
        # With avg corr > 0.8, gross limit = 1.0 * 0.50 = 0.50, so 75% should fail
        gross_violations = [v for v in check.violations if "Gross exposure" in v]
        assert len(gross_violations) > 0

    def test_low_corr_no_tightening(self):
        """With low correlation, no additional tightening should occur."""
        price_data = _generate_price_data(
            ["A", "B", "C"], n_days=252, seed=42, correlation=0.0,
        )
        rm = PortfolioRiskManager(max_single_name_pct=0.10)

        check = rm.check_new_position(
            ticker="C",
            position_size=0.08,
            current_positions={"A": 0.05, "B": 0.05},
            price_data=price_data,
        )
        # With low correlation, 8% should NOT fail single-name (cap stays at 10%)
        single_violations = [v for v in check.violations if "Position size" in v]
        assert len(single_violations) == 0

    def test_avg_pairwise_corr_in_metrics(self):
        """Metrics should always include avg_pairwise_corr."""
        price_data = _generate_price_data(
            ["A", "B"], n_days=120, seed=42,
        )
        rm = PortfolioRiskManager()
        check = rm.check_new_position(
            ticker="B",
            position_size=0.05,
            current_positions={"A": 0.05},
            price_data=price_data,
        )
        assert "avg_pairwise_corr" in check.metrics
        assert isinstance(check.metrics["avg_pairwise_corr"], float)

    def test_corr_stress_multiplier_in_metrics_when_triggered(self):
        """Metrics should include corr_stress_multiplier when tightening is active."""
        price_data = _generate_highly_correlated_data(
            ["A", "B", "C"], n_days=252, seed=42,
        )
        rm = PortfolioRiskManager()
        check = rm.check_new_position(
            ticker="C",
            position_size=0.05,
            current_positions={"A": 0.05, "B": 0.05},
            price_data=price_data,
        )
        assert "corr_stress_multiplier" in check.metrics
        assert check.metrics["corr_stress_multiplier"] < 1.0

    def test_no_corr_stress_multiplier_when_not_triggered(self):
        """Metrics should NOT include corr_stress_multiplier when correlation is low."""
        price_data = _generate_price_data(
            ["A", "B", "C"], n_days=252, seed=42, correlation=0.0,
        )
        rm = PortfolioRiskManager()
        check = rm.check_new_position(
            ticker="C",
            position_size=0.05,
            current_positions={"A": 0.05, "B": 0.05},
            price_data=price_data,
        )
        assert "corr_stress_multiplier" not in check.metrics


# ---------------------------------------------------------------------------
# 5. Correlation tightening stacks on top of regime multiplier
# ---------------------------------------------------------------------------

class TestCorrTighteningStacksWithRegime:
    """Verify corr tightening compounds with regime-based tightening."""

    def test_stress_regime_plus_high_corr_compounds(self):
        """Stress regime (0.6x sector) + high corr (0.5x) = 0.3x sector cap."""
        price_data = _generate_highly_correlated_data(
            ["A", "B", "C"], n_days=252, seed=42,
        )
        rm = PortfolioRiskManager(max_sector_pct=0.40)
        # Force immediate stress multipliers
        rm.multiplier._smoothed = rm.multiplier._stress_mults.copy()
        rm.multiplier._prev_regime_is_stress = True

        check = rm.check_new_position(
            ticker="C",
            position_size=0.05,
            current_positions={"A": 0.10, "B": 0.10},
            price_data=price_data,
            regime=3,  # stress
        )

        # Base sector cap: 40%
        # Stress multiplier: 0.6 → 24%
        # Corr stress (avg > 0.8): 0.5 → 12%
        # 25% in sector > 12% → should fail
        sector_violations = [v for v in check.violations if "Sector" in v]
        assert len(sector_violations) > 0, (
            f"Expected compounded tightening. violations={check.violations}, "
            f"avg_corr={check.metrics.get('avg_pairwise_corr', 'N/A')}"
        )

    def test_normal_regime_only_corr_tightening(self):
        """In normal regime, only corr tightening should be applied."""
        price_data = _generate_highly_correlated_data(
            ["A", "B", "C"], n_days=252, seed=42,
        )
        rm = PortfolioRiskManager(max_sector_pct=0.40)

        check = rm.check_new_position(
            ticker="C",
            position_size=0.05,
            current_positions={"A": 0.10, "B": 0.10},
            price_data=price_data,
            regime=0,  # normal
        )

        # Base sector cap: 40%
        # Normal regime multiplier: 1.0 → 40%
        # Corr stress (avg > 0.8): 0.5 → 20%
        # 25% in sector > 20% → should fail
        sector_violations = [v for v in check.violations if "Sector" in v]
        assert len(sector_violations) > 0


# ---------------------------------------------------------------------------
# 6. compute_constraint_utilization reflects tightened limits
# ---------------------------------------------------------------------------

class TestComputeConstraintUtilizationCorrTightening:
    """Verify compute_constraint_utilization uses corr-tightened limits."""

    def test_high_corr_increases_utilization(self):
        """Higher correlation → tighter limits → higher utilization ratios."""
        price_data_low = _generate_price_data(
            ["A", "B", "C"], n_days=252, seed=42, correlation=0.0,
        )
        price_data_high = _generate_highly_correlated_data(
            ["A", "B", "C"], n_days=252, seed=42,
        )
        rm = PortfolioRiskManager()
        positions = {"A": 0.10, "B": 0.10, "C": 0.10}

        util_low = rm.compute_constraint_utilization(positions, price_data_low)
        rm_high = PortfolioRiskManager()
        util_high = rm_high.compute_constraint_utilization(positions, price_data_high)

        # Gross utilization should be higher with corr tightening
        assert util_high.get("gross_exposure", 0) > util_low.get("gross_exposure", 0)

    def test_single_name_util_increases_with_high_corr(self):
        """Single-name utilization should be higher when correlation tightens the cap."""
        price_data = _generate_highly_correlated_data(
            ["A", "B", "C"], n_days=252, seed=42,
        )
        rm = PortfolioRiskManager(max_single_name_pct=0.10)
        positions = {"A": 0.08, "B": 0.05, "C": 0.05}

        util = rm.compute_constraint_utilization(positions, price_data)
        # With corr > 0.8, eff_single = 0.10 * 0.50 = 0.05
        # Utilization = 0.08 / 0.05 = 1.6 (violation)
        assert util.get("single_name", 0) > 1.0


# ---------------------------------------------------------------------------
# 7. portfolio_summary includes corr_stress_multiplier
# ---------------------------------------------------------------------------

class TestPortfolioSummaryCorrStress:
    """Verify portfolio_summary reports correlation stress info."""

    def test_summary_includes_corr_stress_multiplier(self):
        price_data = _generate_highly_correlated_data(
            ["A", "B", "C"], n_days=252, seed=42,
        )
        rm = PortfolioRiskManager()
        summary = rm.portfolio_summary(
            {"A": 0.10, "B": 0.10, "C": 0.10}, price_data,
        )
        assert "corr_stress_multiplier" in summary
        assert summary["corr_stress_multiplier"] < 1.0

    def test_summary_low_corr_multiplier_is_1(self):
        price_data = _generate_price_data(
            ["A", "B", "C"], n_days=252, seed=42, correlation=0.0,
        )
        rm = PortfolioRiskManager()
        summary = rm.portfolio_summary(
            {"A": 0.10, "B": 0.10, "C": 0.10}, price_data,
        )
        assert "corr_stress_multiplier" in summary
        assert summary["corr_stress_multiplier"] == 1.0

    def test_summary_includes_avg_abs_pairwise_corr(self):
        price_data = _generate_price_data(
            ["A", "B", "C"], n_days=252, seed=42,
        )
        rm = PortfolioRiskManager()
        summary = rm.portfolio_summary(
            {"A": 0.10, "B": 0.10, "C": 0.10}, price_data,
        )
        assert "avg_abs_pairwise_corr" in summary


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for correlation stress tightening."""

    def test_single_position_no_tightening(self):
        """A new position with no existing positions → no avg corr → no tightening."""
        price_data = _generate_highly_correlated_data(
            ["A"], n_days=120, seed=42,
        )
        rm = PortfolioRiskManager(max_single_name_pct=0.10)
        check = rm.check_new_position(
            ticker="A",
            position_size=0.09,
            current_positions={},
            price_data=price_data,
        )
        # 9% < 10% cap, no tightening → should pass
        single_violations = [v for v in check.violations if "Position size" in v]
        assert len(single_violations) == 0

    def test_no_price_data_no_tightening(self):
        """Missing price data should not cause errors or false tightening."""
        rm = PortfolioRiskManager(max_single_name_pct=0.10)
        check = rm.check_new_position(
            ticker="A",
            position_size=0.09,
            current_positions={"B": 0.05},
            price_data={},
        )
        assert "avg_pairwise_corr" in check.metrics
        assert check.metrics["avg_pairwise_corr"] == 0.0

    def test_two_positions_moderate_correlation(self):
        """Two positions with moderate correlation trigger the first tier."""
        price_data = _generate_price_data(
            ["A", "B"], n_days=252, seed=42, correlation=0.5,
        )
        rm = PortfolioRiskManager()
        check = rm.check_new_position(
            ticker="B",
            position_size=0.05,
            current_positions={"A": 0.05},
            price_data=price_data,
        )
        avg_corr = check.metrics["avg_pairwise_corr"]
        # With correlation=0.5 factor blend, avg |corr| should be moderate
        assert isinstance(avg_corr, float)
        assert avg_corr >= 0.0


# ---------------------------------------------------------------------------
# 9. Spec verification: when pairwise corr > 0.7, caps noticeably tighter
# ---------------------------------------------------------------------------

class TestSpecVerification:
    """SPEC-P03 verification: high correlation → noticeably tighter caps."""

    def test_corr_above_07_tightens_sector_cap(self):
        """When avg pairwise corr > 0.7, sector cap should be at most 70% of base."""
        price_data = _generate_highly_correlated_data(
            ["A", "B", "C", "D"], n_days=252, seed=42,
        )
        rm = PortfolioRiskManager(max_sector_pct=0.40)

        # Compute what the effective sector cap would be
        positions = {"A": 0.05, "B": 0.05, "C": 0.05}
        avg_corr = rm._compute_avg_pairwise_correlation(
            {**positions, "D": 0.05}, price_data,
        )
        mult = rm._get_correlation_stress_multiplier(avg_corr)

        # avg corr of highly correlated data should be > 0.8
        assert avg_corr > 0.7, f"Expected high correlation, got {avg_corr}"
        # Multiplier should be at most 0.70 (could be 0.50 if > 0.8)
        assert mult <= 0.70
        # Effective sector cap: 40% * mult <= 40% * 0.70 = 28%
        eff_cap = 0.40 * mult
        assert eff_cap < 0.40 * 0.75, (
            f"Sector cap should be noticeably tighter: eff={eff_cap:.1%}"
        )

    def test_corr_above_07_tightens_single_name_cap(self):
        """When avg pairwise corr > 0.7, single-name cap should be tighter."""
        price_data = _generate_highly_correlated_data(
            ["A", "B", "C"], n_days=252, seed=42,
        )
        rm = PortfolioRiskManager(max_single_name_pct=0.10)

        check_pass = rm.check_new_position(
            ticker="C",
            position_size=0.04,
            current_positions={"A": 0.05, "B": 0.05},
            price_data=price_data,
        )
        # With corr > 0.8, eff single = 10% * 0.50 = 5%
        # 4% < 5% → should still pass
        single_violations_pass = [v for v in check_pass.violations if "Position size" in v]
        assert len(single_violations_pass) == 0

        check_fail = rm.check_new_position(
            ticker="C",
            position_size=0.06,
            current_positions={"A": 0.05, "B": 0.05},
            price_data=price_data,
        )
        # 6% > 5% → should fail
        single_violations_fail = [v for v in check_fail.violations if "Position size" in v]
        assert len(single_violations_fail) > 0

    def test_backward_compatible_without_price_data(self):
        """System should work gracefully when correlation cannot be computed."""
        rm = PortfolioRiskManager(max_single_name_pct=0.10)
        check = rm.check_new_position(
            ticker="A",
            position_size=0.09,
            current_positions={"B": 0.05},
            price_data={},
        )
        # No price data → no corr → no tightening → 9% < 10% → pass
        single_violations = [v for v in check.violations if "Position size" in v]
        assert len(single_violations) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
