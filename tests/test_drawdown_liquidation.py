"""
Test module for drawdown liquidation behavior and regressions.
"""

import unittest
from types import SimpleNamespace

import pandas as pd

from quant_engine.backtest.engine import Backtester
from quant_engine.risk.drawdown import DrawdownState, DrawdownStatus
from quant_engine.risk.stop_loss import StopReason, StopResult


class _FakePositionSizer:
    """Test double used to isolate behavior in this test module."""
    def size_position(self, **kwargs):
        return SimpleNamespace(final_size=0.05)


class _FakeDrawdownController:
    """Test double used to isolate behavior in this test module."""
    def __init__(self):
        self.calls = 0

    def update(self, daily_pnl):
        self.calls += 1
        if self.calls == 1:
            return DrawdownStatus(
                state=DrawdownState.NORMAL,
                current_drawdown=0.0,
                peak_equity=1.0,
                current_equity=1.0,
                size_multiplier=1.0,
                allow_new_entries=True,
                force_liquidate=False,
                days_in_state=0,
                daily_pnl=float(daily_pnl),
                weekly_pnl=float(daily_pnl),
                messages=[],
            )
        return DrawdownStatus(
            state=DrawdownState.CRITICAL,
            current_drawdown=-0.2,
            peak_equity=1.0,
            current_equity=0.8,
            size_multiplier=0.0,
            allow_new_entries=False,
            force_liquidate=True,
            days_in_state=0,
            daily_pnl=float(daily_pnl),
            weekly_pnl=float(daily_pnl),
            messages=[],
        )

    def get_summary(self):
        return {}


class _FakeStopLossManager:
    """Test double used to isolate behavior in this test module."""
    def evaluate(self, **kwargs):
        return StopResult(
            should_exit=False,
            reason=StopReason.NONE,
            stop_price=0.0,
            trailing_high=0.0,
            unrealized_pnl=0.0,
            bars_held=0,
            details={},
        )


class _FakePortfolioRisk:
    """Test double used to isolate behavior in this test module."""
    def check_new_position(self, **kwargs):
        return SimpleNamespace(passed=True)


class _FakeRiskMetrics:
    """Test double used to isolate behavior in this test module."""
    def compute_full_report(self, *args, **kwargs):
        return SimpleNamespace(
            var_95=0.0,
            cvar_95=0.0,
            tail_ratio=0.0,
            ulcer_index=0.0,
            calmar_ratio=0.0,
        )


class DrawdownLiquidationTests(unittest.TestCase):
    """Test cases covering drawdown liquidation behavior and system invariants."""
    def test_critical_drawdown_forces_liquidation(self):
        bt = Backtester(
            use_risk_management=True,
            holding_days=5,
            max_participation_rate=1.0,
            min_fill_ratio=0.0,
        )

        def _fake_init():
            bt._position_sizer = _FakePositionSizer()
            bt._drawdown_ctrl = _FakeDrawdownController()
            bt._stop_loss_mgr = _FakeStopLossManager()
            bt._portfolio_risk = _FakePortfolioRisk()
            bt._risk_metrics = _FakeRiskMetrics()

        bt._init_risk_components = _fake_init

        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        preds = pd.DataFrame(
            {
                "predicted_return": [0.03, 0.03, 0.03, 0.03],
                "confidence": [0.95, 0.95, 0.95, 0.95],
                "regime": [0, 0, 0, 0],
            },
            index=pd.MultiIndex.from_product([["10001"], dates], names=["permno", "date"]),
        )

        ohlcv = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0],
                "High": [101.0, 102.0, 103.0, 104.0],
                "Low": [99.0, 100.0, 101.0, 102.0],
                "Close": [100.5, 101.5, 102.5, 103.5],
                "Volume": [1000, 1000, 1000, 1000],
            },
            index=dates,
        )

        result = bt.run(predictions=preds, price_data={"10001": ohlcv}, verbose=False)
        reasons = [t.exit_reason for t in result.trades]
        self.assertIn("circuit_breaker", reasons)


if __name__ == "__main__":
    unittest.main()
