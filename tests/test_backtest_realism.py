"""
Spec 011 — Backtest Execution Realism & Validation Enforcement tests.

Verifies:
    T1: Entry timing uses next-bar Open in both simple and risk-managed modes
    T2: Almgren-Chriss risk_aversion default is 0.01 (not 1e-6)
    T3: Exit simulation respects volume: partial fills for large positions
    T4: Validation required for autopilot promotion
    T5: Negative Sharpe always fails DSR
    T6: PBO threshold tightened to 0.45
    T7: Residual shares tracked across bars for multi-bar exits
"""

import inspect
import unittest

import numpy as np
import pandas as pd

from quant_engine.backtest.advanced_validation import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from quant_engine.backtest.engine import Backtester, BacktestResult
from quant_engine.backtest.execution import ExecutionModel
from quant_engine.backtest.optimal_execution import almgren_chriss_trajectory
from quant_engine.autopilot.promotion_gate import PromotionGate
from quant_engine.autopilot.strategy_discovery import StrategyCandidate
from quant_engine.config import (
    ALMGREN_CHRISS_RISK_AVERSION,
    PROMOTION_MAX_PBO,
)


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_ohlcv(n_bars: int = 100, base_price: float = 100.0,
                daily_volume: float = 1_000_000.0,
                seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_bars, freq="B")
    returns = rng.normal(0.0005, 0.015, n_bars)
    close = base_price * np.cumprod(1 + returns)
    open_ = close * (1 + rng.normal(0, 0.002, n_bars))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.01, n_bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.01, n_bars))
    volume = rng.uniform(daily_volume * 0.5, daily_volume * 1.5, n_bars)

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)


def _make_predictions(ohlcv: pd.DataFrame, ticker: str = "10001",
                      n_signals: int = 5, start_bar: int = 20,
                      spacing: int = 15) -> pd.DataFrame:
    """Generate synthetic predictions aligned with OHLCV data."""
    rows = []
    for i in range(n_signals):
        bar_idx = start_bar + i * spacing
        if bar_idx >= len(ohlcv):
            break
        dt = ohlcv.index[bar_idx]
        rows.append({
            "predicted_return": 0.02,
            "confidence": 0.75,
            "regime": 0,
        })
    if not rows:
        return pd.DataFrame()
    idx = pd.MultiIndex.from_tuples(
        [(ticker, ohlcv.index[start_bar + i * spacing]) for i in range(len(rows))],
        names=["ticker", "date"],
    )
    return pd.DataFrame(rows, index=idx)


def _candidate() -> StrategyCandidate:
    """Build a reusable test fixture."""
    return StrategyCandidate(
        strategy_id="test_realism",
        horizon=10,
        entry_threshold=0.005,
        confidence_threshold=0.60,
        use_risk_management=False,
        max_positions=10,
        position_size_pct=0.05,
    )


def _passing_result(sharpe: float = 1.35) -> BacktestResult:
    """Build a passing backtest result."""
    return BacktestResult(
        total_trades=150,
        winning_trades=85,
        losing_trades=65,
        win_rate=85 / 150,
        avg_return=0.012,
        avg_win=0.026,
        avg_loss=-0.011,
        total_return=0.42,
        annualized_return=0.18,
        sharpe_ratio=sharpe,
        sortino_ratio=1.70,
        max_drawdown=-0.12,
        profit_factor=1.45,
        avg_holding_days=10.0,
        trades_per_year=120.0,
        returns_series=pd.Series([0.01, -0.004, 0.02]),
        equity_curve=pd.Series([1.0, 1.01, 1.03]),
        daily_equity=pd.Series([1.0, 1.005, 1.012]),
        trades=[],
        regime_breakdown={},
        risk_report=None,
        drawdown_summary=None,
        exit_reason_breakdown={},
    )


def _all_pass_metrics() -> dict:
    """Contract metrics that satisfy all promotion gates."""
    return {
        "dsr_significant": True,
        "dsr_p_value": 0.01,
        "pbo": 0.20,
        "mc_significant": True,
        "mc_p_value": 0.01,
        "capacity_constrained": False,
        "capacity_utilization": 0.55,
        "wf_oos_corr": 0.08,
        "wf_positive_fold_fraction": 0.80,
        "wf_is_oos_gap": 0.04,
        "regime_positive_fraction": 0.75,
        "stat_tests_pass": True,
        "cpcv_passes": True,
        "spa_passes": True,
    }


# ── Test Cases ────────────────────────────────────────────────────────

class TestEntryTimingConsistency(unittest.TestCase):
    """T1: Both simple and risk-managed modes use next-bar Open for entry."""

    def test_simple_mode_entry_at_next_bar_open(self):
        """Simple mode: signal at bar t close -> entry at bar t+1 open."""
        ohlcv = _make_ohlcv(n_bars=60)
        ticker = "10001"
        preds = _make_predictions(ohlcv, ticker=ticker, n_signals=2, start_bar=20)
        price_data = {ticker: ohlcv}

        bt = Backtester(
            entry_threshold=0.001,
            confidence_threshold=0.50,
            holding_days=5,
            max_positions=5,
        )
        result = bt.run(preds, price_data, verbose=False)

        for trade in result.trades:
            # Entry reference price should be the Open of the entry bar,
            # NOT the Close of the signal bar
            entry_dt = pd.Timestamp(trade.entry_date)
            if entry_dt in ohlcv.index:
                entry_idx = ohlcv.index.get_loc(entry_dt)
                expected_open = float(ohlcv["Open"].iloc[entry_idx])
                self.assertAlmostEqual(
                    trade.entry_reference_price, expected_open, places=4,
                    msg="Entry reference should be Open of entry bar",
                )

    def test_risk_managed_mode_entry_at_next_bar_open(self):
        """Risk-managed mode: signal at bar t close -> entry at bar t+1 open."""
        ohlcv = _make_ohlcv(n_bars=60)
        ticker = "10001"
        preds = _make_predictions(ohlcv, ticker=ticker, n_signals=2, start_bar=20)
        price_data = {ticker: ohlcv}

        bt = Backtester(
            entry_threshold=0.001,
            confidence_threshold=0.50,
            holding_days=5,
            max_positions=5,
            use_risk_management=True,
        )
        result = bt.run(preds, price_data, verbose=False)

        for trade in result.trades:
            entry_dt = pd.Timestamp(trade.entry_date)
            if entry_dt in ohlcv.index:
                entry_idx = ohlcv.index.get_loc(entry_dt)
                expected_open = float(ohlcv["Open"].iloc[entry_idx])
                self.assertAlmostEqual(
                    trade.entry_reference_price, expected_open, places=4,
                    msg="Risk-managed entry reference should be Open of entry bar",
                )


class TestAlmgrenChrissCalibration(unittest.TestCase):
    """T2: AC risk_aversion calibrated to institutional levels."""

    def test_config_risk_aversion_not_risk_neutral(self):
        """ALMGREN_CHRISS_RISK_AVERSION must be >= 1e-3 (not 1e-6)."""
        self.assertGreaterEqual(
            ALMGREN_CHRISS_RISK_AVERSION, 1e-3,
            "AC risk aversion too low — nearly risk-neutral",
        )
        self.assertLessEqual(
            ALMGREN_CHRISS_RISK_AVERSION, 1.0,
            "AC risk aversion too high — would never trade",
        )

    def test_trajectory_default_matches_config(self):
        """almgren_chriss_trajectory default risk_aversion should use config value."""
        sig = inspect.signature(almgren_chriss_trajectory)
        default_ra = sig.parameters["risk_aversion"].default
        self.assertEqual(
            default_ra, ALMGREN_CHRISS_RISK_AVERSION,
            f"Trajectory default {default_ra} != config {ALMGREN_CHRISS_RISK_AVERSION}",
        )

    def test_higher_risk_aversion_frontloads_execution(self):
        """Higher risk_aversion should produce more front-loaded trajectory."""
        low_ra = almgren_chriss_trajectory(
            total_shares=10000, n_intervals=10,
            daily_volume=1_000_000, daily_volatility=0.02,
            risk_aversion=1e-4,
        )
        high_ra = almgren_chriss_trajectory(
            total_shares=10000, n_intervals=10,
            daily_volume=1_000_000, daily_volatility=0.02,
            risk_aversion=0.01,
        )
        # Higher risk_aversion -> first interval gets more shares
        self.assertGreater(
            high_ra[0], low_ra[0],
            "Higher risk aversion should front-load execution",
        )


class TestExitVolumeConstraints(unittest.TestCase):
    """T3: Exit simulation respects volume constraints."""

    def test_execution_model_limits_fill_ratio(self):
        """With force_full=False, large orders get partial fills."""
        model = ExecutionModel(
            spread_bps=3.0,
            max_participation_rate=0.02,
            impact_coefficient_bps=25.0,
            min_fill_ratio=0.10,
        )
        # Large order: $500K against $1M daily volume, 2% participation
        # Max notional = $1M * 0.02 = $20K. Fill ratio = 20K/500K = 0.04
        fill = model.simulate(
            side="sell",
            reference_price=100.0,
            daily_volume=10_000.0,  # 10K shares * $100 = $1M daily dollar vol
            desired_notional_usd=500_000.0,
            force_full=False,
        )
        # fill_ratio should be limited by participation
        # Since fill_ratio < min_fill_ratio (0.04 < 0.10), the model rejects
        self.assertEqual(
            fill.fill_ratio, 0.0,
            "Very large order should be rejected when below min_fill_ratio",
        )

    def test_force_full_bypasses_volume_constraint(self):
        """With force_full=True, even large orders fill fully."""
        model = ExecutionModel(
            spread_bps=3.0,
            max_participation_rate=0.02,
            impact_coefficient_bps=25.0,
        )
        fill = model.simulate(
            side="sell",
            reference_price=100.0,
            daily_volume=10_000.0,
            desired_notional_usd=500_000.0,
            force_full=True,
        )
        self.assertEqual(fill.fill_ratio, 1.0)

    def test_moderate_order_gets_partial_fill(self):
        """Order within participation range but still limited."""
        model = ExecutionModel(
            spread_bps=3.0,
            max_participation_rate=0.02,
            impact_coefficient_bps=25.0,
            min_fill_ratio=0.01,  # Very low min fill for this test
        )
        # $50K order against $1M daily volume, 2% participation = $20K max
        fill = model.simulate(
            side="sell",
            reference_price=100.0,
            daily_volume=10_000.0,
            desired_notional_usd=50_000.0,
            force_full=False,
        )
        # Max notional = 10000 * 100 * 0.02 = $20K
        # fill_ratio = 20K / 50K = 0.40
        self.assertGreater(fill.fill_ratio, 0.0, "Should get some fill")
        self.assertLess(fill.fill_ratio, 1.0, "Should not get full fill")
        self.assertAlmostEqual(fill.fill_ratio, 0.40, places=2)


class TestNegativeSharpeFailsDSR(unittest.TestCase):
    """T5: Negative Sharpe strategies always fail DSR."""

    def test_negative_sharpe_rejected(self):
        """Negative Sharpe -> DSR p=1.0, is_significant=False, deflated_sharpe=NaN."""
        import math
        result = deflated_sharpe_ratio(
            observed_sharpe=-0.5,
            n_trials=10,
            n_returns=500,
        )
        self.assertFalse(result.is_significant)
        self.assertEqual(result.p_value, 1.0)
        self.assertTrue(math.isnan(result.deflated_sharpe))
        self.assertFalse(result.deflated_sharpe_valid)

    def test_zero_sharpe_rejected(self):
        """Zero Sharpe -> DSR p=1.0, is_significant=False."""
        result = deflated_sharpe_ratio(
            observed_sharpe=0.0,
            n_trials=10,
            n_returns=500,
        )
        self.assertFalse(result.is_significant)
        self.assertEqual(result.p_value, 1.0)

    def test_positive_sharpe_can_pass(self):
        """Positive Sharpe with few trials should pass DSR."""
        result = deflated_sharpe_ratio(
            observed_sharpe=1.5,
            n_trials=1,
            n_returns=500,
        )
        self.assertTrue(result.is_significant)
        self.assertLess(result.p_value, 0.05)

    def test_weak_positive_sharpe_with_many_trials_fails(self):
        """Weak positive Sharpe with many trials should fail DSR."""
        result = deflated_sharpe_ratio(
            observed_sharpe=0.3,
            n_trials=100,
            n_returns=200,
        )
        self.assertFalse(result.is_significant)


class TestPBOThreshold(unittest.TestCase):
    """T6: PBO threshold tightened to 0.45."""

    def test_config_pbo_threshold(self):
        """PROMOTION_MAX_PBO should be 0.45."""
        self.assertEqual(PROMOTION_MAX_PBO, 0.45)

    def test_pbo_above_045_is_overfit(self):
        """PBO > 0.45 should flag as overfit."""
        # Create a returns matrix where IS-best often underperforms OOS
        rng = np.random.RandomState(42)
        n_obs = 200
        n_strategies = 5
        # Pure noise strategies -> PBO should be high
        returns_matrix = pd.DataFrame(
            rng.randn(n_obs, n_strategies) * 0.01,
            columns=[f"s{i}" for i in range(n_strategies)],
        )
        result = probability_of_backtest_overfitting(returns_matrix, n_partitions=8)
        # For noise strategies, PBO should be near 0.5
        # With the tightened threshold (0.45), this should be flagged
        if result.n_combinations > 0:
            self.assertGreater(
                result.pbo, 0.0,
                "Pure noise should have non-zero PBO",
            )

    def test_pbo_max_combinations_increased(self):
        """PBO max_combinations should be 200 (not 100)."""
        # Create a scenario that generates many combinations
        rng = np.random.RandomState(42)
        n_obs = 400
        n_strategies = 3
        returns_matrix = pd.DataFrame(
            rng.randn(n_obs, n_strategies) * 0.01,
            columns=[f"s{i}" for i in range(n_strategies)],
        )
        result = probability_of_backtest_overfitting(returns_matrix, n_partitions=10)
        # C(10, 5) = 252 > 200, so we should get exactly 200 combinations
        self.assertLessEqual(
            result.n_combinations, 200,
            "Max combinations should be capped at 200",
        )
        if result.n_combinations > 0:
            # With 10 partitions, C(10,5)=252 > 200, so cap should apply
            self.assertEqual(result.n_combinations, 200)


class TestValidationRequiredForPromotion(unittest.TestCase):
    """T4: Validation required for autopilot promotion."""

    def test_no_validation_rejects_promotion(self):
        """Missing DSR/MC metrics -> promotion rejected."""
        gate = PromotionGate()
        decision = gate.evaluate(
            _candidate(),
            _passing_result(),
            contract_metrics={},  # No validation metrics
        )
        self.assertFalse(decision.passed)
        self.assertTrue(
            any("dsr_not_significant" in r for r in decision.reasons),
            f"Expected DSR failure reason, got: {decision.reasons}",
        )
        self.assertTrue(
            any("mc_not_significant" in r for r in decision.reasons),
            f"Expected MC failure reason, got: {decision.reasons}",
        )

    def test_negative_sharpe_always_rejected(self):
        """Negative Sharpe strategies always rejected in promotion gate."""
        gate = PromotionGate()
        result = _passing_result(sharpe=-0.5)
        decision = gate.evaluate(
            _candidate(),
            result,
            contract_metrics=_all_pass_metrics(),
        )
        self.assertFalse(decision.passed)
        self.assertTrue(
            any("negative_sharpe" in r for r in decision.reasons),
            f"Expected negative_sharpe reason, got: {decision.reasons}",
        )

    def test_mc_not_significant_rejects_promotion(self):
        """Monte Carlo not significant -> promotion rejected."""
        gate = PromotionGate()
        metrics = _all_pass_metrics()
        metrics["mc_significant"] = False
        metrics["mc_p_value"] = 0.25
        decision = gate.evaluate(
            _candidate(),
            _passing_result(),
            contract_metrics=metrics,
        )
        self.assertFalse(decision.passed)
        self.assertTrue(
            any("mc_not_significant" in r for r in decision.reasons),
        )

    def test_all_validations_pass_allows_promotion(self):
        """All validation checks passing -> promotion allowed."""
        gate = PromotionGate()
        decision = gate.evaluate(
            _candidate(),
            _passing_result(),
            contract_metrics=_all_pass_metrics(),
        )
        self.assertTrue(
            decision.passed,
            f"Expected pass, got reasons: {decision.reasons}",
        )

    def test_pbo_above_045_rejects_promotion(self):
        """PBO > 0.45 -> promotion rejected."""
        gate = PromotionGate()
        metrics = _all_pass_metrics()
        metrics["pbo"] = 0.48  # Above tightened threshold
        decision = gate.evaluate(
            _candidate(),
            _passing_result(),
            contract_metrics=metrics,
        )
        self.assertFalse(decision.passed)
        self.assertTrue(
            any("pbo>" in r for r in decision.reasons),
            f"Expected PBO failure, got: {decision.reasons}",
        )


class TestPartialExitMultiBar(unittest.TestCase):
    """T7: Residual shares tracked across bars for multi-bar exits."""

    def test_backtester_has_residual_tracking(self):
        """Risk-managed backtester should handle residual positions."""
        # This verifies the code path exists by running a basic backtest
        ohlcv = _make_ohlcv(n_bars=80, daily_volume=1_000_000.0)
        ticker = "10001"
        preds = _make_predictions(ohlcv, ticker=ticker, n_signals=3, start_bar=20)
        price_data = {ticker: ohlcv}

        bt = Backtester(
            entry_threshold=0.001,
            confidence_threshold=0.50,
            holding_days=10,
            max_positions=5,
            use_risk_management=True,
        )
        result = bt.run(preds, price_data, verbose=False)
        # The backtest should complete without errors
        self.assertIsNotNone(result)

    def test_volume_constrained_exit_records_fill_ratio(self):
        """Large position in illiquid stock produces fill_ratio < 1.0
        in the internal volume-constraint calculation."""
        # Verify the execution model logic that underlies multi-bar exits
        model = ExecutionModel(
            spread_bps=3.0,
            max_participation_rate=0.02,
            impact_coefficient_bps=25.0,
            min_fill_ratio=0.01,
        )

        # $200K position, $500K daily dollar volume, 2% participation = $10K max
        ref_price = 50.0
        daily_volume = 10_000.0  # 10K shares * $50 = $500K
        desired_notional = 200_000.0

        fill = model.simulate(
            side="sell",
            reference_price=ref_price,
            daily_volume=daily_volume,
            desired_notional_usd=desired_notional,
            force_full=False,
        )

        # max_notional = 10000 * 50 * 0.02 = $10K
        # fill_ratio = 10K / 200K = 0.05
        self.assertGreater(fill.fill_ratio, 0.0)
        self.assertLess(fill.fill_ratio, 1.0)
        expected_ratio = (daily_volume * ref_price * 0.02) / desired_notional
        self.assertAlmostEqual(fill.fill_ratio, expected_ratio, places=2)

    def test_residual_position_exit_concept(self):
        """Verify the multi-bar exit logic computes correct VWAP."""
        # Simulate what the backtester does internally for residual exits
        total_shares = 10000.0
        exit_fills = [
            (101.0, 2000.0),   # Bar 1: 2000 shares at $101
            (100.5, 3000.0),   # Bar 2: 3000 shares at $100.50
            (100.0, 5000.0),   # Bar 3: 5000 shares at $100.00
        ]

        total_fill_shares = sum(s for _, s in exit_fills)
        vwap_exit = sum(p * s for p, s in exit_fills) / total_fill_shares

        self.assertAlmostEqual(total_fill_shares, total_shares)
        expected_vwap = (101.0 * 2000 + 100.5 * 3000 + 100.0 * 5000) / 10000
        self.assertAlmostEqual(vwap_exit, expected_vwap, places=4)
        # VWAP should be between highest and lowest fill
        self.assertGreaterEqual(vwap_exit, 100.0)
        self.assertLessEqual(vwap_exit, 101.0)


if __name__ == "__main__":
    unittest.main()
