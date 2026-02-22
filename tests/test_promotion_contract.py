"""
Test module for promotion contract behavior and regressions.
"""

import unittest

import pandas as pd

from quant_engine.autopilot.promotion_gate import PromotionGate
from quant_engine.autopilot.strategy_discovery import StrategyCandidate
from quant_engine.backtest.engine import BacktestResult


def _candidate() -> StrategyCandidate:
    """Build a reusable test fixture object for promotion-contract assertions."""
    return StrategyCandidate(
        strategy_id="h10_e50_c60_r0_m10",
        horizon=10,
        entry_threshold=0.005,
        confidence_threshold=0.60,
        use_risk_management=False,
        max_positions=10,
        position_size_pct=0.05,
    )


def _result() -> BacktestResult:
    """Build a reusable test fixture object for promotion-contract assertions."""
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
        sharpe_ratio=1.35,
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


class PromotionContractTests(unittest.TestCase):
    """Test cases covering promotion contract behavior and system invariants."""
    def test_contract_fails_when_advanced_requirements_fail(self):
        gate = PromotionGate()
        decision = gate.evaluate(
            _candidate(),
            _result(),
            contract_metrics={
                "dsr_significant": False,
                "dsr_p_value": 0.22,
                "pbo": 0.62,
                "capacity_constrained": True,
                "capacity_utilization": 1.4,
                "wf_oos_corr": -0.01,
                "wf_positive_fold_fraction": 0.40,
                "wf_is_oos_gap": 0.35,
                "regime_positive_fraction": 0.25,
            },
        )
        self.assertFalse(decision.passed)
        self.assertTrue(any("dsr_not_significant" in r for r in decision.reasons))
        self.assertTrue(any("pbo>" in r for r in decision.reasons))
        self.assertTrue(any("wf_oos_corr<" in r for r in decision.reasons))

    def test_contract_passes_when_all_checks_pass(self):
        gate = PromotionGate()
        decision = gate.evaluate(
            _candidate(),
            _result(),
            contract_metrics={
                "dsr_significant": True,
                "dsr_p_value": 0.01,
                "pbo": 0.30,
                "capacity_constrained": False,
                "capacity_utilization": 0.55,
                "wf_oos_corr": 0.08,
                "wf_positive_fold_fraction": 0.80,
                "wf_is_oos_gap": 0.04,
                "regime_positive_fraction": 0.75,
            },
        )
        self.assertTrue(decision.passed)
        self.assertEqual(len(decision.reasons), 0)


if __name__ == "__main__":
    unittest.main()
