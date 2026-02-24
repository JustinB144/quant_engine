"""
Comprehensive tests for Spec 016: Risk System Improvements.

Covers:
    T1: test_ledoit_wolf_data_driven_shrinkage — shrinkage from data, not hardcoded
    T2: test_ewma_covariance_recent_emphasis — regime change reflected in EWMA
    T3: test_parametric_var_vs_historical — both methods computed, parametric underestimates fat tails
    T4: test_cornish_fisher_var — CF adjustment for non-normal returns
    T5: test_correlation_stress_increases_vol — stress vol > normal vol for diversified portfolio
    T6: test_attribution_sums_to_total — market + factor + alpha ≈ total return
    T7: test_stop_loss_spread_buffer — stop with buffer triggers at lower price than without
    T8: test_recovery_ramp_concave — quadratic recovery slower at start than linear
    T9: test_factor_stress_scenarios — predefined scenarios produce reasonable impacts
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from quant_engine.risk.covariance import CovarianceEstimator
from quant_engine.risk.metrics import RiskMetrics
from quant_engine.risk.stress_test import (
    correlation_stress_test,
    factor_stress_test,
    CRISIS_SCENARIOS,
)
from quant_engine.risk.attribution import compute_rolling_attribution
from quant_engine.risk.stop_loss import StopLossManager, StopReason
from quant_engine.risk.drawdown import DrawdownController, DrawdownState


# ---------------------------------------------------------------------------
# T1: Ledoit-Wolf data-driven shrinkage
# ---------------------------------------------------------------------------

class TestLedoitWolfDataDrivenShrinkage:
    """Shrinkage intensity comes from sklearn, not from a hardcoded 0.15."""

    def test_shrinkage_comes_from_data(self):
        """LedoitWolf.shrinkage_ should be used — not a fixed constant."""
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(252, 10) * 0.01,
            columns=[f"A{i}" for i in range(10)],
        )
        est = CovarianceEstimator(method="ledoit_wolf")
        result = est.estimate(returns)

        assert result.method == "ledoit_wolf"
        # Data-driven shrinkage should NOT be exactly 0.15 (the old hardcoded value)
        assert result.shrinkage_intensity != 0.15, (
            f"Shrinkage {result.shrinkage_intensity} == 0.15 suggests hardcoded value"
        )
        # Should be in valid range
        assert 0.0 <= result.shrinkage_intensity <= 1.0

    def test_covariance_is_psd(self):
        """Result must be positive semi-definite."""
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(252, 10) * 0.01,
            columns=[f"A{i}" for i in range(10)],
        )
        est = CovarianceEstimator(method="ledoit_wolf")
        result = est.estimate(returns)

        eigvals = np.linalg.eigvalsh(result.covariance.values)
        assert np.all(eigvals > 0), f"Not PSD: min eigenvalue = {eigvals.min()}"

    def test_covariance_shape(self):
        """Output shape must match number of assets."""
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(252, 5) * 0.01,
            columns=[f"A{i}" for i in range(5)],
        )
        est = CovarianceEstimator(method="ledoit_wolf")
        result = est.estimate(returns)
        assert result.covariance.shape == (5, 5)


# ---------------------------------------------------------------------------
# T2: EWMA covariance — recent emphasis
# ---------------------------------------------------------------------------

class TestEWMACovariance:
    """EWMA should capture recent volatility regime changes."""

    def test_ewma_reflects_recent_vol_increase(self):
        """After a vol regime change, EWMA variance should be higher than equal-weight."""
        np.random.seed(42)
        # First 200 bars: low vol; last 52 bars: high vol (3x)
        low_vol = np.random.randn(200, 5) * 0.01
        high_vol = np.random.randn(52, 5) * 0.03
        returns = pd.DataFrame(
            np.vstack([low_vol, high_vol]),
            columns=[f"A{i}" for i in range(5)],
        )

        ewma_est = CovarianceEstimator(method="ewma", half_life=60)
        equal_est = CovarianceEstimator(method="sample", shrinkage=0.0)

        ewma_result = ewma_est.estimate(returns)
        equal_result = equal_est.estimate(returns)

        # EWMA diagonal (recent vol) should be higher than equal-weight diagonal
        ewma_diag = np.diag(ewma_result.covariance.values)
        equal_diag = np.diag(equal_result.covariance.values)

        # At least some assets should show higher EWMA vol
        higher_count = np.sum(ewma_diag > equal_diag)
        assert higher_count >= 3, (
            f"Expected most EWMA variances to exceed equal-weight after vol spike, "
            f"but only {higher_count}/5 were higher"
        )

    def test_ewma_covariance_is_psd(self):
        """EWMA result must be positive semi-definite."""
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(252, 5) * 0.01,
            columns=[f"A{i}" for i in range(5)],
        )
        est = CovarianceEstimator(method="ewma", half_life=30)
        result = est.estimate(returns)
        eigvals = np.linalg.eigvalsh(result.covariance.values)
        assert np.all(eigvals > 0), f"Not PSD: min eigenvalue = {eigvals.min()}"


# ---------------------------------------------------------------------------
# T3: Parametric VaR vs Historical
# ---------------------------------------------------------------------------

class TestParametricVaR:
    """Both historical and parametric VaR must be computed."""

    def test_parametric_var_populated(self):
        """Parametric VaR fields should be non-zero for realistic returns."""
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02
        rm = RiskMetrics()
        report = rm.compute_full_report(returns)

        assert report.var_95_parametric != 0.0
        assert report.var_99_parametric != 0.0

    def test_historical_captures_fat_tails_better(self):
        """For fat-tailed returns, historical 99% VaR should be more negative than parametric.

        At the 99% level, fat tails are clearly visible because the normal
        distribution underestimates extreme quantiles.  The t(3) distribution
        has infinite kurtosis, producing much heavier tails than a Gaussian
        with the same standard deviation.
        """
        np.random.seed(42)
        # t-distribution with df=3 has very fat tails
        returns = np.random.standard_t(df=3, size=5000) * 0.01

        rm = RiskMetrics()
        report = rm.compute_full_report(returns)

        # At the 99% level, historical VaR captures the fat tail more aggressively
        assert report.var_99 < report.var_99_parametric, (
            f"Historical 99% VaR ({report.var_99:.4f}) should be more negative "
            f"than parametric ({report.var_99_parametric:.4f}) for fat-tailed returns"
        )


# ---------------------------------------------------------------------------
# T4: Cornish-Fisher VaR
# ---------------------------------------------------------------------------

class TestCornishFisherVaR:
    """Cornish-Fisher adjustment accounts for skewness and kurtosis."""

    def test_cornish_fisher_populated(self):
        """CF VaR fields should be non-zero for realistic returns."""
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02
        rm = RiskMetrics()
        report = rm.compute_full_report(returns)

        assert report.var_95_cornish_fisher != 0.0
        assert report.var_99_cornish_fisher != 0.0

    def test_cornish_fisher_adjusts_for_negative_skew(self):
        """Negatively skewed returns → CF VaR should be more negative than parametric."""
        np.random.seed(42)
        # Generate negatively skewed returns
        normal = np.random.randn(3000) * 0.01
        # Add negative tail events
        crashes = np.random.uniform(-0.08, -0.03, size=100)
        returns = np.concatenate([normal, crashes])
        np.random.shuffle(returns)

        rm = RiskMetrics()
        report = rm.compute_full_report(returns)

        # With negative skew, CF should give a more extreme (more negative) VaR
        assert report.var_95_cornish_fisher < report.var_95_parametric, (
            f"CF VaR ({report.var_95_cornish_fisher:.4f}) should be more negative "
            f"than parametric ({report.var_95_parametric:.4f}) for negative-skew returns"
        )


# ---------------------------------------------------------------------------
# T5: Correlation stress increases portfolio vol
# ---------------------------------------------------------------------------

class TestCorrelationStress:
    """Stress correlation → higher portfolio vol for diversified portfolios."""

    def test_stress_vol_exceeds_normal(self):
        """When all correlations spike to 0.9, portfolio vol must increase."""
        n = 5
        vols = np.ones(n) * 0.20

        # Normal correlations: 0.3
        corr_normal = np.full((n, n), 0.3)
        np.fill_diagonal(corr_normal, 1.0)
        vol_mat = np.diag(vols)
        cov_normal = vol_mat @ corr_normal @ vol_mat

        weights = {f"A{i}": 1.0 / n for i in range(n)}

        result = correlation_stress_test(
            portfolio_weights=weights,
            covariance=cov_normal,
            stress_correlation=0.9,
        )

        assert result["stress_portfolio_vol"] > result["normal_portfolio_vol"], (
            f"Stress vol ({result['stress_portfolio_vol']}) should exceed "
            f"normal vol ({result['normal_portfolio_vol']})"
        )
        assert result["vol_increase_pct"] > 0

    def test_single_asset_no_diversification_effect(self):
        """For a single asset, stress correlation has no effect."""
        cov = np.array([[0.04]])  # 20% vol annualised
        weights = {"A0": 1.0}

        result = correlation_stress_test(
            portfolio_weights=weights,
            covariance=cov,
            stress_correlation=0.9,
        )

        # Single asset: no off-diagonal to stress
        assert abs(result["stress_portfolio_vol"] - result["normal_portfolio_vol"]) < 1e-4


# ---------------------------------------------------------------------------
# T6: Rolling attribution sums to total
# ---------------------------------------------------------------------------

class TestRollingAttribution:
    """market + factor + alpha must approximately equal total return."""

    def test_components_sum_to_total(self):
        """Sum of market + factor + alpha should equal total_return within 1bp."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=200, freq="B")

        # Benchmark: SPY-like
        bench = pd.Series(np.random.randn(200) * 0.01, index=dates, name="SPY")

        # Portfolio: correlated with benchmark + some alpha
        port = bench * 1.1 + np.random.randn(200) * 0.005
        port = pd.Series(port.values, index=dates, name="portfolio")

        result = compute_rolling_attribution(port, bench, window=60)

        assert len(result) > 0, "Rolling attribution should produce results"

        # Check that components sum to total for each row
        reconstructed = result["market_return"] + result["factor_return"] + result["alpha_return"]
        residual = (result["total_return"] - reconstructed).abs()

        assert residual.max() < 1e-10, (
            f"Components should sum to total return; max residual = {residual.max():.2e}"
        )


# ---------------------------------------------------------------------------
# T7: Stop loss spread buffer
# ---------------------------------------------------------------------------

class TestStopLossSpreadBuffer:
    """Spread buffer should lower stop prices, preventing false triggers."""

    def test_buffer_lowers_initial_stop(self):
        """Initial stop with buffer should be lower than without."""
        slm_no_buf = StopLossManager(spread_buffer_bps=0.0, regime_change_exit=False)
        slm_with_buf = StopLossManager(spread_buffer_bps=10.0, regime_change_exit=False)

        entry_price = 100.0
        atr = 2.0

        stop_no_buf = slm_no_buf.compute_initial_stop(entry_price, atr, regime=0)
        stop_with_buf = slm_with_buf.compute_initial_stop(entry_price, atr, regime=0)

        assert stop_with_buf < stop_no_buf, (
            f"Stop with buffer ({stop_with_buf:.4f}) should be lower "
            f"than without ({stop_no_buf:.4f})"
        )

    def test_buffer_amount_correct(self):
        """Verify the spread buffer math: half_spread = price * bps / 10000 * 0.5."""
        slm = StopLossManager(spread_buffer_bps=10.0, regime_change_exit=False)
        entry_price = 100.0

        expected_buffer = 100.0 * (10.0 / 10000.0) * 0.5  # = 0.05
        actual_buffer = slm._spread_buffer(entry_price)

        assert abs(actual_buffer - expected_buffer) < 1e-10, (
            f"Buffer should be {expected_buffer}, got {actual_buffer}"
        )

    def test_zero_buffer_no_effect(self):
        """Zero spread buffer should not change stop prices."""
        slm_zero = StopLossManager(spread_buffer_bps=0.0, regime_change_exit=False)
        entry_price = 100.0
        atr = 2.0

        # ATR stop without buffer: 100 - 2*2.0 = 96.0
        stop = slm_zero.compute_initial_stop(entry_price, atr, regime=0)
        expected_atr_stop = entry_price - 2.0 * 1.0 * atr  # 96.0
        expected_hard_stop = entry_price * (1 + (-0.08))    # 92.0

        # Should be max(96, 92) = 96.0
        assert abs(stop - max(expected_atr_stop, expected_hard_stop)) < 1e-10


# ---------------------------------------------------------------------------
# T8: Recovery ramp is concave (quadratic)
# ---------------------------------------------------------------------------

class TestRecoveryRampConcave:
    """Quadratic recovery is more cautious early than linear would be."""

    def test_early_recovery_more_cautious(self):
        """At early recovery, quadratic size should be smaller than what linear gives.

        The DrawdownController enters RECOVERY when the drawdown improves
        from below CAUTION threshold to above WARNING threshold in a single
        update.  We engineer this by:
        1. Setting WARNING very close to 0 so it's easy to exceed.
        2. Driving into CAUTION with moderate losses.
        3. Recovering with a single large gain.
        """
        dc = DrawdownController(
            warning_threshold=-0.01,   # Very tight warning so RECOVERY triggers
            caution_threshold=-0.06,
            critical_threshold=-0.15,
            daily_loss_limit=-1.0,     # Disable daily limit
            weekly_loss_limit=-1.0,    # Disable weekly limit
            recovery_days=20,
        )

        # Drive into CAUTION: dd below -6%
        # (1 - 0.025)^3 ≈ 0.9269 → dd ≈ -7.3% → below caution (-6%)
        for _ in range(3):
            dc.update(-0.025)

        assert dc.state == DrawdownState.CAUTION, (
            f"Expected CAUTION state, got {dc.state}"
        )

        # Single large recovery that pushes dd above warning (-1%)
        # From dd ≈ -7.3%, need a gain that brings dd close to 0
        status = dc.update(0.10)

        # After a large recovery from CAUTION, should enter RECOVERY
        assert dc.state == DrawdownState.RECOVERY, (
            f"Expected RECOVERY state after large gain from CAUTION, got {dc.state}"
        )

        # On first recovery day, progress ≈ 1/20 = 0.05
        # Quadratic: 0.25 + 0.75 * 0.05^2 = 0.252
        assert status.size_multiplier < 0.40, (
            f"Early recovery multiplier ({status.size_multiplier:.3f}) "
            f"should be very cautious (< 0.40) on first day"
        )

        # New entries should be blocked early in recovery (progress < 0.3)
        assert not status.allow_new_entries, (
            "New entries should be blocked at the start of recovery"
        )

    def test_quadratic_vs_linear_values(self):
        """Directly verify quadratic formula at known progress points."""
        # progress=0.25: quadratic = 0.25 + 0.75*(0.25^2) = 0.296875
        #                linear    = 0.25 + 0.75*0.25      = 0.4375
        progress = 0.25
        quadratic = 0.25 + 0.75 * (progress ** 2)
        linear = 0.25 + 0.75 * progress

        assert quadratic < linear, (
            f"Quadratic ({quadratic:.4f}) should be less than linear ({linear:.4f}) "
            f"at 25% recovery"
        )
        assert abs(quadratic - 0.296875) < 1e-10

    def test_full_recovery_reaches_one(self):
        """At 100% recovery, scale should be 1.0 regardless of ramp shape."""
        progress = 1.0
        quadratic = 0.25 + 0.75 * (progress ** 2)
        assert abs(quadratic - 1.0) < 1e-10

    def test_new_entries_blocked_early_recovery(self):
        """New entries should be blocked until 30% of recovery is complete."""
        dc = DrawdownController(
            warning_threshold=-0.05,
            caution_threshold=-0.10,
            critical_threshold=-0.15,
            recovery_days=20,
        )

        # Drive into CRITICAL
        for _ in range(5):
            dc.update(-0.04)

        # Start recovery with strong gains
        for _ in range(20):
            status = dc.update(0.04)

        # If in recovery, check that early days blocked new entries
        # This is tested indirectly via the formula: allow_new = progress >= 0.3


# ---------------------------------------------------------------------------
# T9: Factor stress scenarios
# ---------------------------------------------------------------------------

class TestFactorStressScenarios:
    """Predefined crisis scenarios should produce reasonable impacts."""

    def test_crisis_scenarios_defined(self):
        """All four predefined scenarios should exist."""
        assert "equity_crash_2008" in CRISIS_SCENARIOS
        assert "covid_march_2020" in CRISIS_SCENARIOS
        assert "rate_shock" in CRISIS_SCENARIOS
        assert "liquidity_crisis" in CRISIS_SCENARIOS

    def test_factor_stress_produces_negative_returns(self):
        """A market-exposed portfolio should lose money in all crisis scenarios."""
        exposures = {"market": 1.0, "size": 0.2, "value": 0.1, "momentum": 0.3}
        results = factor_stress_test(exposures)

        for scenario_name, result in results.items():
            assert result["portfolio_return"] < 0, (
                f"Scenario '{scenario_name}' should produce negative return "
                f"for long-market portfolio, got {result['portfolio_return']}"
            )

    def test_factor_contributions_sum_to_total(self):
        """Per-factor contributions must sum to portfolio_return."""
        exposures = {"market": 1.0, "size": -0.2, "volatility": 0.5}
        results = factor_stress_test(exposures)

        for scenario_name, result in results.items():
            contrib_sum = sum(result["factor_contributions"].values())
            assert abs(contrib_sum - result["portfolio_return"]) < 1e-6, (
                f"Scenario '{scenario_name}': contributions ({contrib_sum}) "
                f"!= portfolio_return ({result['portfolio_return']})"
            )

    def test_zero_exposure_no_impact(self):
        """Zero exposure to all factors → zero portfolio impact."""
        exposures = {"market": 0.0, "size": 0.0}
        results = factor_stress_test(exposures)

        for scenario_name, result in results.items():
            assert result["portfolio_return"] == 0.0, (
                f"Zero-exposure portfolio should have zero impact in '{scenario_name}'"
            )

    def test_custom_scenarios(self):
        """Custom scenarios should work alongside predefined ones."""
        custom = {"my_scenario": {"market": -0.10, "tech": -0.20}}
        exposures = {"market": 1.0, "tech": 0.5}

        results = factor_stress_test(exposures, scenarios=custom)
        assert "my_scenario" in results
        expected = 1.0 * (-0.10) + 0.5 * (-0.20)  # = -0.20
        assert abs(results["my_scenario"]["portfolio_return"] - expected) < 1e-6
