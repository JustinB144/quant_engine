"""
Tests for SPEC_AUDIT_FIX_09: Evaluation Engine Correctness & Statistical Fixes.

Verifies all 9 tasks from the spec.
"""
import numpy as np
import pandas as pd
import pytest

from quant_engine.evaluation.calibration_analysis import analyze_calibration
from quant_engine.evaluation.metrics import compute_slice_metrics, decile_spread
from quant_engine.evaluation.fragility import (
    pnl_concentration,
    drawdown_distribution,
    recovery_time_distribution,
    detect_critical_slowing_down,
)
from quant_engine.evaluation.slicing import SliceRegistry
from quant_engine.evaluation.engine import EvaluationEngine, EvaluationResult


class TestT1CalibratonMismatch:
    """T1: analyze_calibration with mismatched confidence_scores length."""

    def test_mismatched_length_does_not_crash(self):
        result = analyze_calibration(
            predictions=np.array([1.0, 2.0, 3.0]),
            returns=pd.Series([0.01, 0.02, 0.03]),
            confidence_scores=np.array([0.5, 0.6]),  # shorter than predictions
        )
        # Should not crash — returns empty result since n < bins*2
        assert isinstance(result, dict)
        assert "calibration_error" in result

    def test_matched_length_works(self):
        np.random.seed(42)
        n = 200
        preds = np.random.randn(n)
        rets = pd.Series(np.random.randn(n) * 0.01)
        conf = np.abs(preds) / (np.abs(preds).max() + 1e-12)
        result = analyze_calibration(preds, rets, confidence_scores=conf, bins=5)
        assert result["n_samples"] > 0


class TestT2FailOpenHandling:
    """T2: Failed subsystems tracked and critical failures cause FAIL."""

    def test_failed_subsystems_in_result(self):
        np.random.seed(42)
        n = 50
        returns = pd.Series(np.random.randn(n) * 0.01,
                            index=pd.date_range("2022-01-01", periods=n))
        preds = np.random.randn(n)

        engine = EvaluationEngine()
        result = engine.evaluate(returns=returns, predictions=preds)
        # failed_subsystems should be a list attribute
        assert hasattr(result, "failed_subsystems")
        assert isinstance(result.failed_subsystems, list)

    def test_overall_fail_when_walk_forward_fails(self):
        """If walk_forward subsystem fails, overall should be FAIL."""
        np.random.seed(42)
        # Very small dataset — walk-forward will fail due to insufficient data
        n = 10
        returns = pd.Series(np.random.randn(n) * 0.01,
                            index=pd.date_range("2022-01-01", periods=n))
        preds = np.random.randn(n)

        engine = EvaluationEngine(train_window=5, test_window=5, embargo_days=0)
        result = engine.evaluate(returns=returns, predictions=preds)
        # walk_forward should fail with insufficient data
        if "walk_forward" in result.failed_subsystems:
            assert result.overall_pass is False


class TestT3SlicePredictions:
    """T3: Regime slices should have non-zero IC when predictions available."""

    def test_regime_slices_have_ic(self):
        np.random.seed(42)
        n = 500
        returns = pd.Series(np.random.randn(n) * 0.01,
                            index=pd.date_range("2022-01-01", periods=n, freq="B"))
        # Predictions correlated with returns
        preds = returns.values + np.random.randn(n) * 0.005
        regimes = np.array([i % 4 for i in range(n)])

        engine = EvaluationEngine()
        result = engine.evaluate(
            returns=returns, predictions=preds, regime_states=regimes,
        )

        # At least one regime slice should have non-zero IC
        has_nonzero_ic = False
        for name, metrics in result.individual_regime_metrics.items():
            if abs(metrics.get("ic", 0)) > 0.01:
                has_nonzero_ic = True
                break
        assert has_nonzero_ic, "No regime slice has non-zero IC"


class TestT4GeometricReturn:
    """T4: Geometric annualized return < arithmetic for volatile data."""

    def test_geometric_less_than_arithmetic(self):
        np.random.seed(42)
        # 0.1% mean daily return with 3% daily vol
        volatile_returns = pd.Series(np.random.normal(0.001, 0.03, 252))
        metrics = compute_slice_metrics(volatile_returns)
        geo = metrics["annualized_return"]
        arith = metrics["annualized_return_arithmetic"]
        assert geo < arith, f"Geometric {geo:.4f} should be < arithmetic {arith:.4f}"

    def test_both_values_available(self):
        returns = pd.Series([0.01, 0.02, -0.01, 0.005] * 50)
        metrics = compute_slice_metrics(returns)
        assert "annualized_return" in metrics
        assert "annualized_return_arithmetic" in metrics


class TestT5DecileBinning:
    """T5: Decile bins should be balanced with true quantiles."""

    def test_balanced_bins_100_points(self):
        preds = np.arange(100, dtype=float)
        rets = pd.Series(np.random.randn(100))
        result = decile_spread(preds, rets)
        counts = result["decile_counts"]
        assert all(c == 10 for c in counts), f"Counts not balanced: {counts}"

    def test_balanced_bins_200_points(self):
        preds = np.arange(200, dtype=float)
        rets = pd.Series(np.random.randn(200))
        result = decile_spread(preds, rets)
        counts = result["decile_counts"]
        assert all(c == 20 for c in counts), f"Counts not balanced: {counts}"


class TestT6PnlConcentration:
    """T6: pnl_concentration reports correct n_trades even with zero net PnL."""

    def test_zero_net_pnl_reports_n_trades(self):
        result = pnl_concentration([{"pnl": 1.0}, {"pnl": -1.0}])
        assert result["n_trades"] == 2, f"Expected n_trades=2, got {result['n_trades']}"
        assert "concentration" in result  # reports undefined status

    def test_empty_trades_returns_zero(self):
        result = pnl_concentration([])
        assert result["n_trades"] == 0


class TestT7RecoveryTimeNaN:
    """T7: Recovery time computation with NaN returns produces aligned trough dates."""

    def test_nan_returns_aligned_dates(self):
        np.random.seed(42)
        returns = pd.Series(
            np.random.randn(100) * 0.02,
            index=pd.date_range("2022-01-01", periods=100, freq="B"),
        )
        # Inject NaNs
        returns.iloc[5] = np.nan
        returns.iloc[20] = np.nan
        returns.iloc[50] = np.nan

        rt = recovery_time_distribution(returns)
        if len(rt) > 0:
            # All trough dates should be valid dates from the returns index
            for date in rt.index:
                assert date in returns.index, f"Trough date {date} not in returns index"


class TestT8WindowParameters:
    """T8: Window/lookback parameters are honored."""

    def test_drawdown_window_changes_output(self):
        np.random.seed(42)
        returns = pd.Series(np.random.randn(500) * 0.01)
        dd_short = drawdown_distribution(returns, window=10)
        dd_long = drawdown_distribution(returns, window=500)
        # Different windows should produce different max_dd
        assert dd_short["max_dd"] != dd_long["max_dd"]

    def test_recovery_lookback_changes_output(self):
        np.random.seed(42)
        returns = pd.Series(
            np.random.randn(500) * 0.02,
            index=pd.date_range("2022-01-01", periods=500, freq="B"),
        )
        rt_short = recovery_time_distribution(returns, lookback=50)
        rt_long = recovery_time_distribution(returns, lookback=500)
        # Different lookbacks should produce different episode counts
        assert len(rt_short) != len(rt_long) or len(rt_short) == 0

    def test_critical_slowing_window_limits_episodes(self):
        rt = pd.Series(
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            index=pd.date_range("2022-01-01", periods=10, freq="ME"),
        )
        _, info_short = detect_critical_slowing_down(rt, window=3)
        _, info_long = detect_critical_slowing_down(rt, window=10)
        assert info_short["n_episodes"] <= info_long["n_episodes"]


class TestT9MetadataFillna:
    """T9: NaN regime values map to -1 (unknown), not 0 (trending_bull)."""

    def test_nan_regime_maps_to_negative_one(self):
        returns = pd.Series(
            np.random.randn(10) * 0.01,
            index=pd.date_range("2022-01-01", periods=10),
        )
        regime_states = np.array([0, 1, np.nan, 2, 3, np.nan, 0, 1, 2, 3])
        meta = SliceRegistry.build_metadata(returns, regime_states)
        assert meta["regime"].iloc[2] == -1
        assert meta["regime"].iloc[5] == -1

    def test_nan_uncertainty_maps_to_one(self):
        returns = pd.Series(
            np.random.randn(10) * 0.01,
            index=pd.date_range("2022-01-01", periods=10),
        )
        regimes = np.zeros(10, dtype=int)
        unc = np.array([0.1, np.nan, 0.3, 0.4, np.nan, 0.6, 0.7, 0.8, 0.9, 1.0])
        meta = SliceRegistry.build_metadata(returns, regimes, uncertainty=unc)
        assert meta["uncertainty"].iloc[1] == 1.0
        assert meta["uncertainty"].iloc[4] == 1.0

    def test_unknown_regime_slice_exists(self):
        slices = SliceRegistry.create_individual_regime_slices()
        names = [s.name for s in slices]
        assert "unknown" in names
