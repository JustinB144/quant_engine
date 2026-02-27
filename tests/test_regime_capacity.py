"""
SPEC-V03: Regime-dependent capacity analysis tests.

Verifies:
    T1: capacity_analysis returns stress fields when stress_regimes provided
    T2: Stress capacity is lower than overall for illiquid stress trades
    T3: Stress capacity is None when no stress trades exist
    T4: Stress capacity is None when fewer than min_stress_trades
    T5: Promotion gate rejects strategies with low stress capacity
    T6: Promotion gate rejects strategies with low stress capacity ratio
    T7: Promotion gate passes when stress capacity is adequate
    T8: Stress capacity metrics flow through contract_metrics
    T9: run_advanced_validation passes stress_regimes to capacity
    T10: capacity_analysis handles dict-style trades for stress filtering
"""

import unittest
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from quant_engine.backtest.advanced_validation import (
    CapacityResult,
    capacity_analysis,
    run_advanced_validation,
)
from quant_engine.autopilot.promotion_gate import PromotionGate
from quant_engine.autopilot.strategy_discovery import StrategyCandidate
from quant_engine.backtest.engine import BacktestResult
from quant_engine.config import (
    PROMOTION_MIN_STRESS_CAPACITY_USD,
    PROMOTION_MIN_STRESS_CAPACITY_RATIO,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@dataclass
class _MockTrade:
    """Lightweight trade fixture with configurable regime."""
    ticker: str
    entry_date: str
    exit_date: str
    regime: int
    position_size: float = 0.05
    entry_price: float = 100.0
    exit_price: float = 102.0
    predicted_return: float = 0.02
    actual_return: float = 0.02
    net_return: float = 0.019
    confidence: float = 0.75
    holding_days: int = 10


def _make_ohlcv(
    n_bars: int = 200,
    base_price: float = 100.0,
    daily_volume: float = 500_000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_bars, freq="B")
    returns = rng.normal(0.0005, 0.015, n_bars)
    close = base_price * np.cumprod(1 + returns)
    open_ = close * (1 + rng.normal(0, 0.002, n_bars))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.01, n_bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.01, n_bars))
    volume = rng.uniform(daily_volume * 0.5, daily_volume * 1.5, n_bars)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates)


def _make_trades(
    n_calm: int = 40,
    n_stress: int = 20,
    ticker: str = "10001",
    stress_regimes: List[int] = None,
) -> List[_MockTrade]:
    """Create a mix of calm and stress regime trades."""
    if stress_regimes is None:
        stress_regimes = [2, 3]
    dates = pd.bdate_range("2022-06-01", periods=n_calm + n_stress, freq="B")
    trades = []
    for i in range(n_calm):
        trades.append(_MockTrade(
            ticker=ticker,
            entry_date=str(dates[i].date()),
            exit_date=str((dates[i] + pd.Timedelta(days=10)).date()),
            regime=0,
        ))
    for i in range(n_stress):
        regime = stress_regimes[i % len(stress_regimes)]
        trades.append(_MockTrade(
            ticker=ticker,
            entry_date=str(dates[n_calm + i].date()),
            exit_date=str((dates[n_calm + i] + pd.Timedelta(days=10)).date()),
            regime=regime,
        ))
    return trades


def _candidate() -> StrategyCandidate:
    """Build a reusable test fixture object."""
    return StrategyCandidate(
        strategy_id="h10_e50_c60_r0_m10",
        horizon=10,
        entry_threshold=0.005,
        confidence_threshold=0.60,
        use_risk_management=False,
        max_positions=10,
        position_size_pct=0.05,
    )


def _result_with_regime_perf(regime_performance: dict = None) -> BacktestResult:
    """Build a BacktestResult with custom regime_performance data."""
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
        sharpe_ratio=1.50,
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
        regime_performance=regime_performance or {},
        risk_report=None,
        drawdown_summary=None,
        exit_reason_breakdown={},
    )


def _passing_contract_metrics() -> dict:
    """Build contract metrics that satisfy all advanced gates."""
    return {
        "dsr_significant": True,
        "dsr_p_value": 0.01,
        "pbo": 0.30,
        "mc_significant": True,
        "mc_p_value": 0.01,
        "capacity_constrained": False,
        "capacity_utilization": 0.55,
        "estimated_capacity_usd": 5_000_000,
        "wf_oos_corr": 0.08,
        "wf_positive_fold_fraction": 0.80,
        "wf_is_oos_gap": 0.04,
        "regime_positive_fraction": 0.75,
        "stat_tests_pass": True,
        "cpcv_passes": True,
        "spa_passes": True,
    }


# ── capacity_analysis unit tests ──────────────────────────────────────


class TestCapacityAnalysisStressRegimes(unittest.TestCase):
    """T1-T4: capacity_analysis stress regime functionality."""

    def test_t1_stress_fields_populated_when_stress_regimes_provided(self):
        """capacity_analysis should populate stress fields when stress trades exist."""
        trades = _make_trades(n_calm=30, n_stress=15)
        price_data = {"10001": _make_ohlcv()}
        result = capacity_analysis(
            trades=trades,
            price_data=price_data,
            stress_regimes=[2, 3],
        )
        self.assertIsNotNone(result.stress_capacity_usd)
        self.assertIsNotNone(result.stress_market_impact_bps)
        self.assertIsNotNone(result.stress_participation_rate)
        self.assertIsNotNone(result.stress_trades_per_day)
        self.assertIsNotNone(result.stress_avg_daily_volume)
        self.assertGreater(result.stress_capacity_usd, 0)

    def test_t2_stress_capacity_lower_with_concentrated_stress_trades(self):
        """Stress capacity should be lower when stress trades are more concentrated.

        During stress, trade frequency may be higher (multiple trades on the
        same day) while volume drops, producing lower capacity.
        """
        dates = pd.bdate_range("2022-06-01", periods=60, freq="B")
        trades = []
        # Calm trades: 1 per day, spread over 40 days
        for i in range(40):
            trades.append(_MockTrade(
                ticker="10001",
                entry_date=str(dates[i].date()),
                exit_date=str((dates[i] + pd.Timedelta(days=10)).date()),
                regime=0,
            ))
        # Stress trades: 3 per day on just 5 days (15 total trades)
        for i in range(5):
            for _ in range(3):
                trades.append(_MockTrade(
                    ticker="10001",
                    entry_date=str(dates[40 + i].date()),
                    exit_date=str((dates[40 + i] + pd.Timedelta(days=10)).date()),
                    regime=3,
                ))
        price_data = {"10001": _make_ohlcv()}
        result = capacity_analysis(
            trades=trades, price_data=price_data, stress_regimes=[3],
        )
        self.assertIsNotNone(result.stress_capacity_usd)
        self.assertGreater(result.estimated_capacity_usd, 0)
        self.assertGreater(result.stress_capacity_usd, 0)
        # Stress trades have 3 trades/day vs ~1.2 overall => higher
        # participation => lower capacity
        self.assertGreater(
            result.stress_trades_per_day, result.trades_per_day,
            "Stress trade frequency should be higher (concentrated trades)",
        )
        self.assertLess(
            result.stress_capacity_usd, result.estimated_capacity_usd,
            "Stress capacity should be lower when trades are concentrated",
        )

    def test_t3_stress_fields_none_when_no_stress_trades(self):
        """Stress fields should be None when no trades are in stress regimes."""
        # All trades in regime 0 (calm)
        trades = _make_trades(n_calm=40, n_stress=0)
        price_data = {"10001": _make_ohlcv()}
        result = capacity_analysis(
            trades=trades,
            price_data=price_data,
            stress_regimes=[2, 3],
        )
        self.assertIsNone(result.stress_capacity_usd)
        self.assertIsNone(result.stress_market_impact_bps)

    def test_t4_stress_fields_none_when_fewer_than_min_trades(self):
        """Stress fields should be None when stress trades < min_stress_trades."""
        trades = _make_trades(n_calm=40, n_stress=5)
        price_data = {"10001": _make_ohlcv()}
        result = capacity_analysis(
            trades=trades,
            price_data=price_data,
            stress_regimes=[2, 3],
            min_stress_trades=10,
        )
        self.assertIsNone(result.stress_capacity_usd)

    def test_stress_fields_none_when_stress_regimes_is_none(self):
        """No stress analysis when stress_regimes is None (backward compat)."""
        trades = _make_trades(n_calm=30, n_stress=20)
        price_data = {"10001": _make_ohlcv()}
        result = capacity_analysis(
            trades=trades,
            price_data=price_data,
            stress_regimes=None,
        )
        self.assertIsNone(result.stress_capacity_usd)

    def test_empty_trades_returns_zero_capacity(self):
        """Empty trade list should return zero capacity without errors."""
        result = capacity_analysis(
            trades=[], price_data={}, stress_regimes=[2, 3],
        )
        self.assertEqual(result.estimated_capacity_usd, 0)
        self.assertIsNone(result.stress_capacity_usd)

    def test_stress_capacity_with_custom_regimes(self):
        """Custom stress regimes should be respected."""
        trades = _make_trades(n_calm=30, n_stress=15, stress_regimes=[1])
        price_data = {"10001": _make_ohlcv()}

        # Using regime 1 as stress: should find stress trades
        result_custom = capacity_analysis(
            trades=trades, price_data=price_data, stress_regimes=[1],
        )
        self.assertIsNotNone(result_custom.stress_capacity_usd)

        # Using regime 2 as stress: should NOT find stress trades
        result_default = capacity_analysis(
            trades=trades, price_data=price_data, stress_regimes=[2],
        )
        self.assertIsNone(result_default.stress_capacity_usd)


class TestCapacityAnalysisDictTrades(unittest.TestCase):
    """T10: capacity_analysis handles dict-style trades for stress filtering."""

    def test_dict_trades_with_regime_key(self):
        """Dict-style trades with 'regime' key should be filtered correctly."""
        dates = pd.bdate_range("2022-06-01", periods=30, freq="B")
        trades = []
        for i in range(20):
            trades.append({
                "ticker": "10001",
                "entry_date": str(dates[i].date()),
                "regime": 0,
                "position_size": 0.05,
            })
        for i in range(20, 30):
            trades.append({
                "ticker": "10001",
                "entry_date": str(dates[i].date()),
                "regime": 3,
                "position_size": 0.05,
            })
        price_data = {"10001": _make_ohlcv()}
        result = capacity_analysis(
            trades=trades, price_data=price_data, stress_regimes=[3],
        )
        self.assertIsNotNone(result.stress_capacity_usd)
        self.assertGreater(result.stress_capacity_usd, 0)


# ── Promotion gate stress capacity tests ──────────────────────────────


class TestPromotionGateStressCapacity(unittest.TestCase):
    """T5-T8: Promotion gate stress capacity enforcement."""

    def test_t5_fails_when_stress_capacity_below_minimum(self):
        """Strategy with stress capacity below threshold should fail."""
        gate = PromotionGate()
        metrics = _passing_contract_metrics()
        metrics["stress_capacity_usd"] = 100_000  # Below 500K default
        decision = gate.evaluate(
            _candidate(), _result_with_regime_perf(), contract_metrics=metrics,
        )
        self.assertFalse(decision.passed)
        self.assertTrue(
            any("stress_capacity(" in r for r in decision.reasons),
            f"Expected stress_capacity failure, got: {decision.reasons}",
        )

    def test_t6_fails_when_stress_capacity_ratio_below_minimum(self):
        """Strategy with low stress/overall capacity ratio should fail."""
        gate = PromotionGate()
        metrics = _passing_contract_metrics()
        metrics["estimated_capacity_usd"] = 10_000_000
        metrics["stress_capacity_usd"] = 1_000_000  # 10% ratio, below 20% default
        decision = gate.evaluate(
            _candidate(), _result_with_regime_perf(), contract_metrics=metrics,
        )
        self.assertFalse(decision.passed)
        self.assertTrue(
            any("stress_capacity_ratio(" in r for r in decision.reasons),
            f"Expected stress_capacity_ratio failure, got: {decision.reasons}",
        )

    def test_t7_passes_when_stress_capacity_adequate(self):
        """Strategy with sufficient stress capacity should pass."""
        gate = PromotionGate()
        metrics = _passing_contract_metrics()
        metrics["estimated_capacity_usd"] = 5_000_000
        metrics["stress_capacity_usd"] = 2_000_000  # 40% ratio, above 20%
        decision = gate.evaluate(
            _candidate(), _result_with_regime_perf(), contract_metrics=metrics,
        )
        stress_reasons = [r for r in decision.reasons if "stress_capacity" in r]
        self.assertEqual(
            len(stress_reasons), 0,
            f"Unexpected stress capacity failures: {stress_reasons}",
        )

    def test_t8_stress_metrics_stored_in_decision(self):
        """Stress capacity metrics should be recorded in decision.metrics."""
        gate = PromotionGate()
        metrics = _passing_contract_metrics()
        metrics["estimated_capacity_usd"] = 5_000_000
        metrics["stress_capacity_usd"] = 2_000_000
        decision = gate.evaluate(
            _candidate(), _result_with_regime_perf(), contract_metrics=metrics,
        )
        self.assertIn("stress_capacity_usd", decision.metrics)
        self.assertIn("stress_capacity_ratio", decision.metrics)
        self.assertAlmostEqual(
            decision.metrics["stress_capacity_ratio"], 0.40, places=2,
        )

    def test_no_stress_capacity_no_failure(self):
        """When stress_capacity_usd is not in metrics, gate should not trigger."""
        gate = PromotionGate()
        metrics = _passing_contract_metrics()
        # No stress_capacity_usd key
        decision = gate.evaluate(
            _candidate(), _result_with_regime_perf(), contract_metrics=metrics,
        )
        stress_reasons = [r for r in decision.reasons if "stress_capacity" in r]
        self.assertEqual(len(stress_reasons), 0)

    def test_custom_stress_capacity_thresholds(self):
        """Custom min_stress_capacity_usd / ratio should be respected."""
        gate = PromotionGate(
            min_stress_capacity_usd=1_000_000,
            min_stress_capacity_ratio=0.50,
        )
        metrics = _passing_contract_metrics()
        metrics["estimated_capacity_usd"] = 5_000_000
        metrics["stress_capacity_usd"] = 2_000_000  # Passes USD, fails ratio
        decision = gate.evaluate(
            _candidate(), _result_with_regime_perf(), contract_metrics=metrics,
        )
        stress_reasons = [r for r in decision.reasons if "stress_capacity" in r]
        # Should fail on ratio (2M/5M = 40% < 50%) but pass on USD (2M > 1M)
        self.assertTrue(
            any("stress_capacity_ratio(" in r for r in stress_reasons),
        )
        self.assertFalse(
            any("stress_capacity(" in r and "ratio" not in r for r in stress_reasons),
        )

    def test_boundary_stress_capacity_exactly_at_limit(self):
        """Stress capacity exactly at threshold should NOT trigger failure."""
        gate = PromotionGate()
        metrics = _passing_contract_metrics()
        metrics["estimated_capacity_usd"] = 2_500_000
        metrics["stress_capacity_usd"] = PROMOTION_MIN_STRESS_CAPACITY_USD  # Exactly 500K
        decision = gate.evaluate(
            _candidate(), _result_with_regime_perf(), contract_metrics=metrics,
        )
        # Stress capacity at exactly the threshold passes (not strictly below)
        usd_reasons = [
            r for r in decision.reasons
            if "stress_capacity(" in r and "ratio" not in r
        ]
        self.assertEqual(len(usd_reasons), 0,
                         "Stress capacity exactly at limit should not fail")

    def test_boundary_stress_ratio_exactly_at_limit(self):
        """Stress ratio exactly at threshold should NOT trigger failure."""
        gate = PromotionGate()
        metrics = _passing_contract_metrics()
        metrics["estimated_capacity_usd"] = 5_000_000
        # 20% ratio = exactly at PROMOTION_MIN_STRESS_CAPACITY_RATIO (0.20)
        metrics["stress_capacity_usd"] = 1_000_000
        decision = gate.evaluate(
            _candidate(), _result_with_regime_perf(), contract_metrics=metrics,
        )
        ratio_reasons = [r for r in decision.reasons if "stress_capacity_ratio(" in r]
        self.assertEqual(len(ratio_reasons), 0,
                         "Stress ratio exactly at limit should not fail")


# ── run_advanced_validation integration ───────────────────────────────


class TestRunAdvancedValidationStress(unittest.TestCase):
    """T9: run_advanced_validation forwards stress_regimes."""

    def test_t9_stress_regimes_forwarded(self):
        """run_advanced_validation should pass stress_regimes to capacity_analysis."""
        trades = _make_trades(n_calm=30, n_stress=20)
        price_data = {"10001": _make_ohlcv()}
        trade_returns = np.array([t.net_return for t in trades])

        report = run_advanced_validation(
            trade_returns=trade_returns,
            trades=trades,
            price_data=price_data,
            stress_regimes=[2, 3],
            verbose=False,
        )
        cap = report.capacity
        self.assertIsNotNone(cap)
        self.assertIsNotNone(cap.stress_capacity_usd,
                             "stress_capacity_usd should be populated")
        self.assertGreater(cap.stress_capacity_usd, 0)

    def test_no_stress_regimes_backward_compat(self):
        """Without stress_regimes, run_advanced_validation should work as before."""
        trades = _make_trades(n_calm=30, n_stress=20)
        price_data = {"10001": _make_ohlcv()}
        trade_returns = np.array([t.net_return for t in trades])

        report = run_advanced_validation(
            trade_returns=trade_returns,
            trades=trades,
            price_data=price_data,
            verbose=False,
        )
        cap = report.capacity
        self.assertIsNotNone(cap)
        self.assertIsNone(cap.stress_capacity_usd,
                          "stress fields should be None when not requested")


# ── Config defaults ───────────────────────────────────────────────────


class TestConfigDefaults(unittest.TestCase):
    """Verify SPEC-V03 config constants exist and are reasonable."""

    def test_min_stress_capacity_usd_is_positive(self):
        """PROMOTION_MIN_STRESS_CAPACITY_USD should be > 0."""
        self.assertGreater(PROMOTION_MIN_STRESS_CAPACITY_USD, 0)

    def test_min_stress_capacity_ratio_between_zero_and_one(self):
        """PROMOTION_MIN_STRESS_CAPACITY_RATIO should be in (0, 1)."""
        self.assertGreater(PROMOTION_MIN_STRESS_CAPACITY_RATIO, 0)
        self.assertLess(PROMOTION_MIN_STRESS_CAPACITY_RATIO, 1)


if __name__ == "__main__":
    unittest.main()
