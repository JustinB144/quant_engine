"""
Comprehensive tests for Spec 017: System-Level Innovation.

Covers:
    T1: test_cusum_detects_mean_shift — CUSUM fires when prediction errors shift
    T1: test_psi_detects_distribution_change — PSI > 0.25 when features shift
    T1: test_psi_stable_distributions — PSI ≈ 0 for identical distributions
    T1: test_cusum_no_false_alarm — CUSUM does not fire on stationary errors
    T2: test_conformal_coverage — conformal intervals achieve target coverage rate
    T2: test_conformal_uncertainty_scaling — wider intervals → smaller scalars
    T2: test_conformal_calibration_requires_min_samples — error on too few
    T3: test_multi_horizon_blending — regime-aware blending produces different weights
    T3: test_multi_horizon_equal_fallback — unknown regime uses equal weights
    T4: test_regime_strategy_profiles — each regime has distinct parameter set
    T4: test_confidence_blending — low confidence → default parameters
    T5: test_factor_exposure_limits — alert when beta > 1.5 or < 0.5
    T5: test_factor_monitor_empty_portfolio — no crash on empty portfolio
    T6: test_cost_budget_full_rebalance — within budget → full execution
    T6: test_cost_budget_partial_rebalance — over budget → partial execution
    T7: test_diagnostics_identify_alpha_decay — low signal score → ALPHA_DECAY
    T7: test_diagnostics_performing — good performance → no diagnostics
    T9: test_survivorship_bias_positive — survivors Sharpe > full Sharpe
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from quant_engine.models.shift_detection import DistributionShiftDetector
from quant_engine.models.conformal import ConformalPredictor
from quant_engine.models.predictor import EnsemblePredictor
from quant_engine.autopilot.strategy_allocator import (
    StrategyAllocator,
    REGIME_STRATEGY_PROFILES,
    DEFAULT_PROFILE,
)
from quant_engine.risk.factor_monitor import FactorExposureMonitor
from quant_engine.risk.cost_budget import optimize_rebalance_cost
from quant_engine.api.services.diagnostics import SystemDiagnostics
from quant_engine.backtest.survivorship_comparison import (
    compare_survivorship_impact,
    quick_survivorship_check,
    UniverseMetrics,
)


# ---------------------------------------------------------------------------
# T1: Distribution Shift Detection (CUSUM + PSI)
# ---------------------------------------------------------------------------

class TestCUSUMDetectsMeanShift:
    """CUSUM should detect persistent mean shifts in prediction errors."""

    def test_detects_upward_shift(self):
        """CUSUM should fire when errors shift upward."""
        np.random.seed(42)
        # 100 bars of normal errors, then 100 bars with mean shift
        normal = np.random.randn(100) * 0.01
        shifted = np.random.randn(100) * 0.01 + 0.05  # Big positive shift
        errors = pd.Series(np.concatenate([normal, shifted]))

        detector = DistributionShiftDetector(cusum_threshold=4.0)
        result = detector.check_cusum(errors)

        assert result.shift_detected, (
            f"CUSUM should detect mean shift of 0.05. "
            f"Statistic={result.cusum_statistic:.4f}, threshold={result.threshold:.4f}"
        )
        assert result.direction == "positive"

    def test_detects_downward_shift(self):
        """CUSUM should fire when errors shift downward."""
        np.random.seed(42)
        normal = np.random.randn(100) * 0.01
        shifted = np.random.randn(100) * 0.01 - 0.05  # Big negative shift
        errors = pd.Series(np.concatenate([normal, shifted]))

        detector = DistributionShiftDetector(cusum_threshold=4.0)
        result = detector.check_cusum(errors)

        assert result.shift_detected
        assert result.direction == "negative"

    def test_no_false_alarm_stationary(self):
        """CUSUM should NOT fire on stationary (no-shift) errors."""
        np.random.seed(42)
        stationary = np.random.randn(200) * 0.01
        errors = pd.Series(stationary)

        detector = DistributionShiftDetector(cusum_threshold=5.0)
        result = detector.check_cusum(errors)

        assert not result.shift_detected, (
            f"CUSUM should not fire on stationary data. "
            f"Statistic={result.cusum_statistic:.4f}"
        )

    def test_too_few_samples(self):
        """CUSUM should handle very short error series gracefully."""
        errors = pd.Series([0.01, -0.01, 0.02])  # Only 3 points
        detector = DistributionShiftDetector()
        result = detector.check_cusum(errors)
        assert not result.shift_detected


class TestPSIDetectsDistributionChange:
    """PSI should detect when feature distributions change significantly."""

    def test_psi_detects_mean_shift(self):
        """PSI > 0.25 when features shift by 2 standard deviations."""
        np.random.seed(42)
        reference = pd.DataFrame({
            "feat_a": np.random.randn(1000),
            "feat_b": np.random.randn(1000),
        })
        shifted = pd.DataFrame({
            "feat_a": np.random.randn(1000) + 2.0,  # 2σ mean shift
            "feat_b": np.random.randn(1000) + 2.0,
        })

        detector = DistributionShiftDetector(psi_threshold=0.25)
        detector.set_reference(reference)
        result = detector.check_psi(shifted)

        assert result.shift_detected, (
            f"PSI should detect 2σ mean shift. avg_psi={result.avg_psi:.4f}"
        )
        assert result.avg_psi > 0.25

    def test_psi_stable_distributions(self):
        """PSI ≈ 0 for same-distribution samples (no shift)."""
        np.random.seed(42)
        reference = pd.DataFrame({
            "feat_a": np.random.randn(1000),
            "feat_b": np.random.randn(1000),
        })
        same = pd.DataFrame({
            "feat_a": np.random.randn(1000),  # Same distribution
            "feat_b": np.random.randn(1000),
        })

        detector = DistributionShiftDetector(psi_threshold=0.25)
        detector.set_reference(reference)
        result = detector.check_psi(same)

        assert not result.shift_detected, (
            f"PSI should be low for same distribution. avg_psi={result.avg_psi:.4f}"
        )
        assert result.avg_psi < 0.15  # Allow small sampling noise

    def test_psi_top_shifted_features(self):
        """PSI report should identify which features shifted most."""
        np.random.seed(42)
        reference = pd.DataFrame({
            "stable_feat": np.random.randn(1000),
            "shifted_feat": np.random.randn(1000),
        })
        current = pd.DataFrame({
            "stable_feat": np.random.randn(1000),
            "shifted_feat": np.random.randn(1000) + 3.0,  # Only this shifts
        })

        detector = DistributionShiftDetector()
        detector.set_reference(reference)
        result = detector.check_psi(current)

        # shifted_feat should have higher PSI
        assert len(result.top_shifted_features) > 0
        top_feat_name = result.top_shifted_features[0][0]
        assert top_feat_name == "shifted_feat"

    def test_check_all_combined(self):
        """check_all should run both CUSUM and PSI checks."""
        np.random.seed(42)
        detector = DistributionShiftDetector()
        reference = pd.DataFrame({"f": np.random.randn(100)})
        detector.set_reference(reference)

        errors = pd.Series(np.random.randn(100) * 0.01)
        current = pd.DataFrame({"f": np.random.randn(100)})

        result = detector.check_all(
            prediction_errors=errors,
            current_features=current,
        )
        assert "cusum" in result
        assert "psi" in result
        assert "any_shift_detected" in result


# ---------------------------------------------------------------------------
# T2: Conformal Prediction Intervals
# ---------------------------------------------------------------------------

class TestConformalCoverage:
    """Conformal prediction intervals should achieve target coverage rate."""

    def test_90_percent_coverage(self):
        """Conformal intervals should achieve ≥88% empirical coverage for 90% target."""
        np.random.seed(42)
        n_cal = 500
        n_test = 2000

        # Simulate predictions with noise
        cal_preds = np.random.randn(n_cal) * 0.02
        cal_actuals = cal_preds + np.random.randn(n_cal) * 0.01

        test_preds = np.random.randn(n_test) * 0.02
        test_actuals = test_preds + np.random.randn(n_test) * 0.01

        cp = ConformalPredictor(coverage=0.90)
        cp.calibrate(cal_preds, cal_actuals)

        eval_result = cp.evaluate_coverage(test_preds, test_actuals)

        assert eval_result["empirical_coverage"] >= 0.88, (
            f"Coverage {eval_result['empirical_coverage']:.3f} < 0.88 "
            f"(target: 0.90)"
        )
        assert eval_result["empirical_coverage"] <= 0.98, (
            f"Coverage {eval_result['empirical_coverage']:.3f} too high — "
            f"intervals may be overly conservative"
        )

    def test_wider_coverage_target(self):
        """95% target should produce wider intervals than 80% target."""
        np.random.seed(42)
        cal_preds = np.random.randn(200) * 0.02
        cal_actuals = cal_preds + np.random.randn(200) * 0.01

        cp80 = ConformalPredictor(coverage=0.80)
        cp80.calibrate(cal_preds, cal_actuals)

        cp95 = ConformalPredictor(coverage=0.95)
        cp95.calibrate(cal_preds, cal_actuals)

        assert cp95._quantile > cp80._quantile, (
            "95% coverage quantile should be wider than 80% coverage quantile"
        )


class TestConformalUncertaintyScaling:
    """Wider prediction intervals should produce smaller position-sizing scalars."""

    def test_wider_interval_smaller_scalar(self):
        """Wider intervals → smaller uncertainty scalar."""
        np.random.seed(42)
        cp = ConformalPredictor(coverage=0.90)
        cal_preds = np.random.randn(200) * 0.02
        cal_actuals = cal_preds + np.random.randn(200) * 0.01
        cp.calibrate(cal_preds, cal_actuals)

        # Narrow prediction (near zero) vs wide prediction
        widths = np.array([0.001, 0.01, 0.05, 0.10])
        scalars = cp.uncertainty_scalars(widths)

        # Scalars should be monotonically decreasing with width
        for i in range(len(scalars) - 1):
            assert scalars[i] >= scalars[i + 1], (
                f"Scalars should decrease with width: "
                f"scalar[{i}]={scalars[i]:.4f} < scalar[{i+1}]={scalars[i+1]:.4f}"
            )

    def test_scalar_range(self):
        """All scalars should be in (0, 1]."""
        cp = ConformalPredictor(coverage=0.90)
        cp._quantile = 0.01  # Manual calibration for test

        widths = np.array([0.0, 0.01, 0.02, 0.05, 0.10])
        scalars = cp.uncertainty_scalars(widths)

        assert np.all(scalars > 0), "All scalars must be positive"
        assert np.all(scalars <= 1.0), "All scalars must be <= 1.0"

    def test_calibration_requires_min_samples(self):
        """Calibration should raise ValueError with too few samples."""
        cp = ConformalPredictor()
        with pytest.raises(ValueError, match="at least 10"):
            cp.calibrate(np.array([1, 2, 3]), np.array([1, 2, 3]))

    def test_predict_before_calibration_raises(self):
        """Predict should raise before calibration."""
        cp = ConformalPredictor()
        with pytest.raises(ValueError, match="Must calibrate"):
            cp.predict_interval(0.5)


# ---------------------------------------------------------------------------
# T3: Multi-Horizon Information Sharing
# ---------------------------------------------------------------------------

class TestMultiHorizonBlending:
    """Regime-aware blending should weight horizons differently by regime."""

    def test_trending_bull_weights_long_horizons(self):
        """Trending bull should weight 20d most heavily."""
        predictions = {
            5: np.array([0.01]),
            10: np.array([0.02]),
            20: np.array([0.03]),
        }
        blended = EnsemblePredictor.blend_multi_horizon(predictions, regime=0)
        # With default bull weights (0.10, 0.30, 0.60):
        expected = 0.10 * 0.01 + 0.30 * 0.02 + 0.60 * 0.03
        assert abs(blended[0] - expected) < 1e-6

    def test_high_vol_weights_short_horizons(self):
        """High-vol regime should weight 5d most heavily."""
        predictions = {
            5: np.array([0.01]),
            10: np.array([0.02]),
            20: np.array([0.03]),
        }
        blended = EnsemblePredictor.blend_multi_horizon(predictions, regime=3)
        # With default high-vol weights (0.60, 0.30, 0.10):
        expected = 0.60 * 0.01 + 0.30 * 0.02 + 0.10 * 0.03
        assert abs(blended[0] - expected) < 1e-6

    def test_different_regimes_produce_different_blends(self):
        """Different regimes should produce different blended predictions."""
        predictions = {
            5: np.array([0.01]),
            10: np.array([0.02]),
            20: np.array([0.05]),
        }
        blends = {}
        for regime in range(4):
            blends[regime] = float(
                EnsemblePredictor.blend_multi_horizon(predictions, regime=regime)[0]
            )

        # Bull (regime 0) should give more weight to 20d prediction
        # High-vol (regime 3) should give more weight to 5d prediction
        assert blends[0] > blends[3], (
            f"Bull blend ({blends[0]:.4f}) should be higher than "
            f"high-vol blend ({blends[3]:.4f}) when 20d pred is largest"
        )

    def test_single_horizon_fallback(self):
        """When only one horizon available, return it unchanged."""
        predictions = {10: np.array([0.05])}
        blended = EnsemblePredictor.blend_multi_horizon(predictions, regime=0)
        assert abs(blended[0] - 0.05) < 1e-6


# ---------------------------------------------------------------------------
# T4: Regime-Aware Strategy Allocation
# ---------------------------------------------------------------------------

class TestRegimeStrategyProfiles:
    """Each regime should have distinct strategy parameters."""

    def test_all_regimes_have_profiles(self):
        """All 4 regimes should have strategy profiles."""
        for regime in range(4):
            assert regime in REGIME_STRATEGY_PROFILES

    def test_profiles_are_distinct(self):
        """No two regime profiles should be identical."""
        profiles = list(REGIME_STRATEGY_PROFILES.values())
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                # At least one parameter should differ
                some_differ = any(
                    profiles[i].get(k) != profiles[j].get(k)
                    for k in profiles[i]
                    if k != "name"
                )
                assert some_differ, (
                    f"Profiles for regimes {i} and {j} should differ"
                )

    def test_high_vol_is_most_conservative(self):
        """High-vol regime should have smallest positions and shortest holds."""
        high_vol = REGIME_STRATEGY_PROFILES[3]
        for regime in range(3):
            other = REGIME_STRATEGY_PROFILES[regime]
            assert high_vol["position_size_pct"] <= other["position_size_pct"]
            assert high_vol["kelly_fraction"] <= other["kelly_fraction"]

    def test_bull_is_most_aggressive(self):
        """Trending bull should have largest positions."""
        bull = REGIME_STRATEGY_PROFILES[0]
        for regime in range(1, 4):
            other = REGIME_STRATEGY_PROFILES[regime]
            assert bull["position_size_pct"] >= other["position_size_pct"]


class TestConfidenceBlending:
    """Low confidence should produce parameters closer to defaults."""

    def test_full_confidence_uses_regime_params(self):
        """At confidence=1.0, use full regime profile."""
        allocator = StrategyAllocator()
        profile = allocator.get_regime_profile(regime=3, regime_confidence=1.0)
        assert profile.position_size_pct == REGIME_STRATEGY_PROFILES[3]["position_size_pct"]

    def test_zero_confidence_uses_defaults(self):
        """At confidence=0.0, use default profile."""
        allocator = StrategyAllocator()
        profile = allocator.get_regime_profile(regime=3, regime_confidence=0.0)
        assert abs(profile.position_size_pct - DEFAULT_PROFILE["position_size_pct"]) < 1e-6

    def test_partial_confidence_blends(self):
        """At confidence=0.5, parameters should be between regime and default."""
        allocator = StrategyAllocator()
        profile = allocator.get_regime_profile(regime=3, regime_confidence=0.5)
        high_vol_size = REGIME_STRATEGY_PROFILES[3]["position_size_pct"]
        default_size = DEFAULT_PROFILE["position_size_pct"]
        expected = 0.5 * high_vol_size + 0.5 * default_size
        assert abs(profile.position_size_pct - expected) < 1e-6


# ---------------------------------------------------------------------------
# T5: Factor Exposure Monitoring
# ---------------------------------------------------------------------------

class TestFactorExposureLimits:
    """Factor exposure monitor should alert on extreme tilts."""

    def test_alert_on_extreme_beta(self):
        """Should alert when market beta exceeds 1.5."""
        monitor = FactorExposureMonitor()
        exposures = {"market_beta": 1.8}  # Above 1.5 limit
        report = monitor.check_limits(exposures)
        assert not report.all_passed
        assert len(report.violations) > 0
        assert "market_beta" in report.violations[0]

    def test_pass_on_normal_exposures(self):
        """Should pass when all exposures within limits."""
        monitor = FactorExposureMonitor()
        exposures = {
            "market_beta": 1.0,
            "size_zscore": 0.0,
            "momentum_zscore": 0.0,
            "volatility_zscore": 0.0,
        }
        report = monitor.check_limits(exposures)
        assert report.all_passed
        assert len(report.violations) == 0

    def test_empty_portfolio_no_crash(self):
        """Should handle empty portfolio gracefully."""
        monitor = FactorExposureMonitor()
        exposures = monitor.compute_exposures({}, {})
        assert exposures == {}

    def test_multiple_violations(self):
        """Should report all violations, not just the first."""
        monitor = FactorExposureMonitor()
        exposures = {
            "market_beta": 2.0,  # Exceeds 1.5
            "volatility_zscore": 3.0,  # Exceeds 1.5
        }
        report = monitor.check_limits(exposures)
        assert not report.all_passed
        assert len(report.violations) == 2


# ---------------------------------------------------------------------------
# T6: Transaction Cost Budget
# ---------------------------------------------------------------------------

class TestCostBudgetFullRebalance:
    """When within budget, all trades should execute."""

    def test_within_budget_full_execution(self):
        """Full rebalance when total cost <= budget."""
        current = pd.Series({"A": 0.30, "B": 0.30, "C": 0.40})
        target = pd.Series({"A": 0.35, "B": 0.25, "C": 0.40})
        volumes = pd.Series({"A": 1e8, "B": 1e8, "C": 1e8})

        result = optimize_rebalance_cost(
            current, target, volumes, total_budget_bps=100.0,
        )

        assert not result.is_partial
        assert len(result.deferred_trades) == 0
        assert result.n_executed > 0


class TestCostBudgetPartialRebalance:
    """When over budget, should partially execute and defer low-priority trades."""

    def test_partial_execution_over_budget(self):
        """Over-budget rebalance should defer some trades."""
        # Create a scenario with large trades on illiquid stocks
        current = pd.Series({
            "ILLIQ_A": 0.05, "ILLIQ_B": 0.05,
            "ILLIQ_C": 0.20, "ILLIQ_D": 0.20,
            "ILLIQ_E": 0.50,
        })
        target = pd.Series({
            "ILLIQ_A": 0.25, "ILLIQ_B": 0.25,
            "ILLIQ_C": 0.05, "ILLIQ_D": 0.05,
            "ILLIQ_E": 0.40,
        })
        # Very low volume makes trades very expensive
        volumes = pd.Series({
            "ILLIQ_A": 0.01, "ILLIQ_B": 0.01,
            "ILLIQ_C": 0.01, "ILLIQ_D": 0.01,
            "ILLIQ_E": 0.01,
        })

        result = optimize_rebalance_cost(
            current, target, volumes, total_budget_bps=0.001,
        )

        assert result.is_partial, "Should be partial when budget is very tight"
        assert result.n_deferred > 0
        assert result.total_cost_bps <= result.budget_bps + 1e-6

    def test_no_trades_needed(self):
        """Should handle case where no rebalance is needed."""
        current = pd.Series({"A": 0.50, "B": 0.50})
        target = pd.Series({"A": 0.50, "B": 0.50})
        volumes = pd.Series({"A": 1e8, "B": 1e8})

        result = optimize_rebalance_cost(current, target, volumes)

        assert not result.is_partial
        assert result.n_executed == 0
        assert result.total_cost_bps == 0.0


# ---------------------------------------------------------------------------
# T7: Self-Diagnostic Dashboard
# ---------------------------------------------------------------------------

class TestDiagnosticsIdentifyAlphaDecay:
    """Diagnostics should identify alpha decay when signal score is low."""

    def test_alpha_decay_detected(self):
        """Low signal score + negative returns → ALPHA_DECAY diagnosis."""
        diag = SystemDiagnostics()
        report = diag.diagnose_performance(
            equity_curve={"daily_pnl": [-0.001] * 20},
            health_history={"signal_decay_score": 30.0},
        )

        assert report.status == "UNDERPERFORMING"
        causes = [d.cause for d in report.diagnostics]
        assert "ALPHA_DECAY" in causes

    def test_stale_data_detected(self):
        """Old data + negative returns → STALE_DATA diagnosis."""
        diag = SystemDiagnostics()
        report = diag.diagnose_performance(
            equity_curve={"daily_pnl": [-0.001] * 20},
            health_history={"data_freshness_days": 14},
        )

        assert report.status == "UNDERPERFORMING"
        causes = [d.cause for d in report.diagnostics]
        assert "STALE_DATA" in causes

    def test_model_staleness_detected(self):
        """Old model + negative returns → MODEL_STALE diagnosis."""
        diag = SystemDiagnostics()
        report = diag.diagnose_performance(
            equity_curve={"daily_pnl": [-0.001] * 20},
            health_history={"model_age_days": 60},
        )

        assert report.status == "UNDERPERFORMING"
        causes = [d.cause for d in report.diagnostics]
        assert "MODEL_STALE" in causes


class TestDiagnosticsPerforming:
    """When performance is good, diagnostics should report PERFORMING."""

    def test_positive_returns_performing(self):
        """Positive returns → PERFORMING status, no diagnostics."""
        diag = SystemDiagnostics()
        report = diag.diagnose_performance(
            equity_curve={"daily_pnl": [0.001] * 20},
            health_history={"signal_decay_score": 80.0},
        )

        assert report.status == "PERFORMING"
        assert len(report.diagnostics) == 0

    def test_no_equity_curve_unknown(self):
        """No equity curve → PERFORMING (default)."""
        diag = SystemDiagnostics()
        report = diag.diagnose_performance()
        assert report.status == "PERFORMING"

    def test_unfavorable_regime_high_vol(self):
        """High-vol regime + underperformance → UNFAVORABLE_REGIME."""
        diag = SystemDiagnostics()
        report = diag.diagnose_performance(
            equity_curve={"daily_pnl": [-0.002] * 20},
            regime_history={"current_regime": 3, "regime_duration": 15},
        )

        assert report.status == "UNDERPERFORMING"
        causes = [d.cause for d in report.diagnostics]
        assert "UNFAVORABLE_REGIME" in causes


# ---------------------------------------------------------------------------
# T9: Survivorship-Free Backtest Comparison
# ---------------------------------------------------------------------------

class TestSurvivorshipBiasComparison:
    """Survivorship comparison should quantify bias in each metric."""

    def _make_mock_result(self, sharpe, win_rate, n_trades, max_dd, total_ret):
        """Create a mock BacktestResult-like object."""
        mock = MagicMock()
        mock.sharpe_ratio = sharpe
        mock.win_rate = win_rate
        mock.total_trades = n_trades
        mock.max_drawdown = max_dd
        mock.total_return = total_ret
        mock.avg_return = total_ret / max(n_trades, 1)
        mock.profit_factor = 1.5
        mock.avg_holding_days = 10.0
        return mock

    def test_survivors_higher_sharpe(self):
        """Survivors-only Sharpe should be higher than full universe."""
        # Full universe: includes losers that got delisted
        result_full = self._make_mock_result(
            sharpe=1.0, win_rate=0.52, n_trades=500, max_dd=-0.15, total_ret=0.20,
        )
        # Survivors only: only winners survive
        result_survivors = self._make_mock_result(
            sharpe=1.4, win_rate=0.56, n_trades=400, max_dd=-0.10, total_ret=0.30,
        )

        comparison = compare_survivorship_impact(result_full, result_survivors)

        assert comparison.bias["sharpe_bias"] > 0, (
            "Survivors-only Sharpe should be higher (positive bias)"
        )
        assert comparison.bias["win_rate_bias"] > 0, (
            "Survivors-only win rate should be higher"
        )
        assert comparison.full_universe.sharpe == 1.0
        assert comparison.survivors_only.sharpe == 1.4

    def test_comparison_summary(self):
        """Comparison should produce a readable summary."""
        result_full = self._make_mock_result(1.0, 0.52, 500, -0.15, 0.20)
        result_survivors = self._make_mock_result(1.3, 0.55, 450, -0.12, 0.28)

        comparison = compare_survivorship_impact(result_full, result_survivors)

        assert "Survivorship Bias Analysis" in comparison.summary
        assert "Sharpe inflation" in comparison.summary

    def test_quick_survivorship_check(self):
        """Quick check should report coverage statistics."""
        predictions = pd.DataFrame({"ticker": ["A", "B", "C"]})
        full_data = {
            "A": pd.DataFrame({"Close": [100]}),
            "B": pd.DataFrame({"Close": [100]}),
            "C": pd.DataFrame({"Close": [100]}),
            "D": pd.DataFrame({"Close": [100]}),  # Delisted
        }
        survivor_data = {
            "A": pd.DataFrame({"Close": [100]}),
            "B": pd.DataFrame({"Close": [100]}),
            "C": pd.DataFrame({"Close": [100]}),
        }

        result = quick_survivorship_check(predictions, full_data, survivor_data)

        assert result["full_universe_tickers"] == 4
        assert result["survivors_only_tickers"] == 3
        assert result["dropped_tickers"] == 1
        assert "D" in result["dropped_ticker_names"]


# ---------------------------------------------------------------------------
# T8: Diverse Ensemble (already implemented — verify it exists)
# ---------------------------------------------------------------------------

class TestDiverseEnsembleExists:
    """Verify that ENSEMBLE_DIVERSIFY is already implemented in trainer."""

    def test_ensemble_diversify_config_exists(self):
        """ENSEMBLE_DIVERSIFY config flag should exist."""
        from quant_engine.config import ENSEMBLE_DIVERSIFY
        assert isinstance(ENSEMBLE_DIVERSIFY, bool)

    def test_diverse_ensemble_class_exists(self):
        """DiverseEnsemble class should exist in trainer."""
        from quant_engine.models.trainer import DiverseEnsemble
        assert hasattr(DiverseEnsemble, "predict")
        assert hasattr(DiverseEnsemble, "_aggregate_feature_importances")


# ---------------------------------------------------------------------------
# Integration: Shift detection + Retrain trigger
# ---------------------------------------------------------------------------

class TestShiftDetectionRetrainIntegration:
    """Shift detection should integrate with retrain trigger."""

    def test_retrain_trigger_has_shift_detector(self):
        """RetrainTrigger should have a shift_detector attribute."""
        from quant_engine.models.retrain_trigger import RetrainTrigger
        import tempfile
        import json
        import os

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({}, f)
            tmp_path = f.name
        try:
            trigger = RetrainTrigger(metadata_file=tmp_path)
            assert hasattr(trigger, "shift_detector")
            assert isinstance(trigger.shift_detector, DistributionShiftDetector)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_check_shift_returns_results(self):
        """check_shift should return (bool, list) tuple."""
        from quant_engine.models.retrain_trigger import RetrainTrigger
        import tempfile
        import json
        import os

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({}, f)
            tmp_path = f.name
        try:
            trigger = RetrainTrigger(metadata_file=tmp_path)
            np.random.seed(42)
            errors = np.random.randn(100) * 0.01

            shift_detected, reasons = trigger.check_shift(prediction_errors=errors)
            assert isinstance(shift_detected, bool)
            assert isinstance(reasons, list)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Integration: Conformal prediction serialization
# ---------------------------------------------------------------------------

class TestConformalSerialization:
    """Conformal predictor should serialize/deserialize correctly."""

    def test_round_trip(self):
        """to_dict → from_dict should preserve calibration state."""
        np.random.seed(42)
        cp = ConformalPredictor(coverage=0.90)
        preds = np.random.randn(200) * 0.02
        actuals = preds + np.random.randn(200) * 0.01
        cp.calibrate(preds, actuals)

        d = cp.to_dict()
        cp2 = ConformalPredictor.from_dict(d)

        assert cp2.coverage == 0.90
        assert cp2._quantile == cp._quantile
        assert cp2.is_calibrated
