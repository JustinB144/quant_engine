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
            },
        )
        self.assertTrue(decision.passed)
        self.assertEqual(len(decision.reasons), 0)


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
        "wf_oos_corr": 0.08,
        "wf_positive_fold_fraction": 0.80,
        "wf_is_oos_gap": 0.04,
        "regime_positive_fraction": 0.75,
        "stat_tests_pass": True,
        "cpcv_passes": True,
        "spa_passes": True,
    }


def _result_with_regime_perf(regime_performance: dict) -> BacktestResult:
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
        regime_performance=regime_performance,
        risk_report=None,
        drawdown_summary=None,
        exit_reason_breakdown={},
    )


class StressRegimeGateTests(unittest.TestCase):
    """SPEC-V02: Worst-case bucket promotion gates for stress regimes."""

    def test_fails_when_stress_regime_drawdown_exceeds_limit(self):
        """Strategy with overall Sharpe 1.5 but -25% drawdown in regime 3 should fail."""
        regime_perf = {
            0: {"n_trades": 60, "sharpe": 1.8, "max_drawdown": -0.08,
                "win_rate": 0.62, "cumulative_return": 0.30,
                "avg_return": 0.005, "avg_win": 0.02, "avg_loss": -0.01,
                "sortino": 2.0, "profit_factor": 2.1, "avg_holding_days": 8.0},
            3: {"n_trades": 30, "sharpe": -0.20, "max_drawdown": -0.25,
                "win_rate": 0.40, "cumulative_return": -0.08,
                "avg_return": -0.003, "avg_win": 0.015, "avg_loss": -0.012,
                "sortino": -0.3, "profit_factor": 0.8, "avg_holding_days": 12.0},
        }
        gate = PromotionGate()
        result = _result_with_regime_perf(regime_perf)
        decision = gate.evaluate(
            _candidate(), result, contract_metrics=_passing_contract_metrics(),
        )
        self.assertFalse(decision.passed)
        self.assertTrue(
            any("stress_regime_3_drawdown" in r for r in decision.reasons),
            f"Expected stress_regime_3_drawdown failure, got: {decision.reasons}",
        )

    def test_fails_when_stress_regime_sharpe_too_negative(self):
        """Strategy should fail if stress regime Sharpe is below -0.50 threshold."""
        regime_perf = {
            0: {"n_trades": 80, "sharpe": 2.0, "max_drawdown": -0.06,
                "win_rate": 0.65, "cumulative_return": 0.35,
                "avg_return": 0.005, "avg_win": 0.02, "avg_loss": -0.01,
                "sortino": 2.2, "profit_factor": 2.3, "avg_holding_days": 7.0},
            2: {"n_trades": 20, "sharpe": -0.80, "max_drawdown": -0.10,
                "win_rate": 0.35, "cumulative_return": -0.05,
                "avg_return": -0.003, "avg_win": 0.01, "avg_loss": -0.008,
                "sortino": -0.9, "profit_factor": 0.6, "avg_holding_days": 15.0},
        }
        gate = PromotionGate()
        result = _result_with_regime_perf(regime_perf)
        decision = gate.evaluate(
            _candidate(), result, contract_metrics=_passing_contract_metrics(),
        )
        self.assertFalse(decision.passed)
        self.assertTrue(
            any("stress_regime_2_sharpe" in r for r in decision.reasons),
            f"Expected stress_regime_2_sharpe failure, got: {decision.reasons}",
        )

    def test_fails_on_both_drawdown_and_sharpe_violations(self):
        """Strategy with both violations on regime 3 should report both reasons."""
        regime_perf = {
            3: {"n_trades": 25, "sharpe": -0.75, "max_drawdown": -0.22,
                "win_rate": 0.30, "cumulative_return": -0.12,
                "avg_return": -0.005, "avg_win": 0.01, "avg_loss": -0.012,
                "sortino": -0.9, "profit_factor": 0.5, "avg_holding_days": 14.0},
        }
        gate = PromotionGate()
        result = _result_with_regime_perf(regime_perf)
        decision = gate.evaluate(
            _candidate(), result, contract_metrics=_passing_contract_metrics(),
        )
        self.assertFalse(decision.passed)
        self.assertTrue(
            any("stress_regime_3_drawdown" in r for r in decision.reasons),
        )
        self.assertTrue(
            any("stress_regime_3_sharpe" in r for r in decision.reasons),
        )

    def test_passes_when_stress_metrics_within_limits(self):
        """Strategy with acceptable stress regime performance should pass."""
        regime_perf = {
            0: {"n_trades": 60, "sharpe": 1.8, "max_drawdown": -0.08,
                "win_rate": 0.62, "cumulative_return": 0.30,
                "avg_return": 0.005, "avg_win": 0.02, "avg_loss": -0.01,
                "sortino": 2.0, "profit_factor": 2.1, "avg_holding_days": 8.0},
            2: {"n_trades": 20, "sharpe": -0.30, "max_drawdown": -0.10,
                "win_rate": 0.42, "cumulative_return": -0.03,
                "avg_return": -0.002, "avg_win": 0.012, "avg_loss": -0.009,
                "sortino": -0.35, "profit_factor": 0.85, "avg_holding_days": 11.0},
            3: {"n_trades": 15, "sharpe": -0.10, "max_drawdown": -0.12,
                "win_rate": 0.45, "cumulative_return": -0.02,
                "avg_return": -0.001, "avg_win": 0.01, "avg_loss": -0.008,
                "sortino": -0.12, "profit_factor": 0.90, "avg_holding_days": 13.0},
        }
        gate = PromotionGate()
        result = _result_with_regime_perf(regime_perf)
        decision = gate.evaluate(
            _candidate(), result, contract_metrics=_passing_contract_metrics(),
        )
        # Should pass — no stress regime violations
        stress_reasons = [r for r in decision.reasons if "stress_regime" in r]
        self.assertEqual(len(stress_reasons), 0,
                         f"Unexpected stress failures: {stress_reasons}")

    def test_no_failure_when_regime_performance_empty(self):
        """Stress gates should be inactive when no regime data is available."""
        gate = PromotionGate()
        result = _result_with_regime_perf({})
        decision = gate.evaluate(
            _candidate(), result, contract_metrics=_passing_contract_metrics(),
        )
        stress_reasons = [r for r in decision.reasons if "stress_regime" in r]
        self.assertEqual(len(stress_reasons), 0)

    def test_skips_stress_regimes_with_few_trades(self):
        """Regimes with fewer than 5 trades should not trigger gates."""
        regime_perf = {
            3: {"n_trades": 3, "sharpe": -2.0, "max_drawdown": -0.40,
                "win_rate": 0.20, "cumulative_return": -0.15,
                "avg_return": -0.05, "avg_win": 0.005, "avg_loss": -0.06,
                "sortino": -2.5, "profit_factor": 0.2, "avg_holding_days": 20.0},
        }
        gate = PromotionGate()
        result = _result_with_regime_perf(regime_perf)
        decision = gate.evaluate(
            _candidate(), result, contract_metrics=_passing_contract_metrics(),
        )
        stress_reasons = [r for r in decision.reasons if "stress_regime" in r]
        self.assertEqual(len(stress_reasons), 0,
                         "Should skip stress gates for regimes with <5 trades")

    def test_stress_metrics_stored_in_diagnostics(self):
        """Stress regime metrics should be recorded in decision.metrics for diagnostics."""
        regime_perf = {
            2: {"n_trades": 10, "sharpe": 0.20, "max_drawdown": -0.05,
                "win_rate": 0.50, "cumulative_return": 0.02,
                "avg_return": 0.002, "avg_win": 0.01, "avg_loss": -0.006,
                "sortino": 0.25, "profit_factor": 1.2, "avg_holding_days": 9.0},
            3: {"n_trades": 8, "sharpe": -0.10, "max_drawdown": -0.08,
                "win_rate": 0.45, "cumulative_return": -0.01,
                "avg_return": -0.001, "avg_win": 0.008, "avg_loss": -0.005,
                "sortino": -0.12, "profit_factor": 0.95, "avg_holding_days": 10.0},
        }
        gate = PromotionGate()
        result = _result_with_regime_perf(regime_perf)
        decision = gate.evaluate(
            _candidate(), result, contract_metrics=_passing_contract_metrics(),
        )
        # Diagnostic metrics should be present
        self.assertIn("stress_regime_2_drawdown", decision.metrics)
        self.assertIn("stress_regime_2_sharpe", decision.metrics)
        self.assertIn("stress_regime_2_n_trades", decision.metrics)
        self.assertIn("stress_regime_3_drawdown", decision.metrics)
        self.assertIn("stress_regime_3_sharpe", decision.metrics)
        self.assertIn("stress_regime_3_n_trades", decision.metrics)
        self.assertEqual(decision.metrics["stress_regime_2_n_trades"], 10)
        self.assertAlmostEqual(decision.metrics["stress_regime_3_sharpe"], -0.10)

    def test_stress_resilience_bonus_in_score(self):
        """Strategies resilient in stress should get a score bonus."""
        regime_perf_good = {
            2: {"n_trades": 15, "sharpe": 0.50, "max_drawdown": -0.06,
                "win_rate": 0.55, "cumulative_return": 0.04,
                "avg_return": 0.003, "avg_win": 0.012, "avg_loss": -0.007,
                "sortino": 0.6, "profit_factor": 1.4, "avg_holding_days": 9.0},
            3: {"n_trades": 10, "sharpe": 0.30, "max_drawdown": -0.07,
                "win_rate": 0.50, "cumulative_return": 0.02,
                "avg_return": 0.002, "avg_win": 0.01, "avg_loss": -0.006,
                "sortino": 0.35, "profit_factor": 1.2, "avg_holding_days": 10.0},
        }
        regime_perf_bad = {
            2: {"n_trades": 15, "sharpe": -0.40, "max_drawdown": -0.12,
                "win_rate": 0.38, "cumulative_return": -0.04,
                "avg_return": -0.003, "avg_win": 0.008, "avg_loss": -0.008,
                "sortino": -0.5, "profit_factor": 0.7, "avg_holding_days": 12.0},
            3: {"n_trades": 10, "sharpe": -0.45, "max_drawdown": -0.14,
                "win_rate": 0.35, "cumulative_return": -0.05,
                "avg_return": -0.005, "avg_win": 0.007, "avg_loss": -0.009,
                "sortino": -0.5, "profit_factor": 0.6, "avg_holding_days": 14.0},
        }
        gate = PromotionGate()
        cm = _passing_contract_metrics()
        result_good = _result_with_regime_perf(regime_perf_good)
        result_bad = _result_with_regime_perf(regime_perf_bad)
        decision_good = gate.evaluate(_candidate(), result_good, contract_metrics=cm)
        decision_bad = gate.evaluate(_candidate(), result_bad, contract_metrics=cm)
        self.assertGreater(decision_good.score, decision_bad.score,
                           "Stress-resilient strategy should score higher")
        self.assertGreater(decision_good.metrics.get("stress_resilience_bonus", 0), 0)
        self.assertLess(decision_bad.metrics.get("stress_resilience_bonus", 0), 0)

    def test_custom_stress_regimes_parameter(self):
        """Gate should use custom stress_regimes when provided."""
        # Only regime 1 is a "stress" regime in this custom config
        regime_perf = {
            1: {"n_trades": 20, "sharpe": -0.80, "max_drawdown": -0.30,
                "win_rate": 0.30, "cumulative_return": -0.10,
                "avg_return": -0.005, "avg_win": 0.01, "avg_loss": -0.012,
                "sortino": -1.0, "profit_factor": 0.5, "avg_holding_days": 15.0},
            2: {"n_trades": 20, "sharpe": 0.50, "max_drawdown": -0.05,
                "win_rate": 0.55, "cumulative_return": 0.05,
                "avg_return": 0.003, "avg_win": 0.012, "avg_loss": -0.007,
                "sortino": 0.6, "profit_factor": 1.4, "avg_holding_days": 9.0},
        }
        gate = PromotionGate(stress_regimes=[1])
        result = _result_with_regime_perf(regime_perf)
        decision = gate.evaluate(
            _candidate(), result, contract_metrics=_passing_contract_metrics(),
        )
        # Should fail on regime 1 (custom stress regime)
        self.assertTrue(
            any("stress_regime_1_drawdown" in r for r in decision.reasons),
        )
        # Should NOT fail on regime 2 (not in custom stress list)
        self.assertFalse(
            any("stress_regime_2" in r for r in decision.reasons),
        )

    def test_boundary_drawdown_exactly_at_limit(self):
        """Drawdown exactly at -15% (the limit) should NOT trigger failure."""
        regime_perf = {
            3: {"n_trades": 10, "sharpe": 0.0, "max_drawdown": -0.15,
                "win_rate": 0.45, "cumulative_return": 0.0,
                "avg_return": 0.0, "avg_win": 0.008, "avg_loss": -0.008,
                "sortino": 0.0, "profit_factor": 1.0, "avg_holding_days": 10.0},
        }
        gate = PromotionGate()
        result = _result_with_regime_perf(regime_perf)
        decision = gate.evaluate(
            _candidate(), result, contract_metrics=_passing_contract_metrics(),
        )
        stress_dd_reasons = [r for r in decision.reasons if "stress_regime_3_drawdown" in r]
        self.assertEqual(len(stress_dd_reasons), 0,
                         "Drawdown exactly at limit should not fail")

    def test_boundary_sharpe_exactly_at_limit(self):
        """Sharpe exactly at -0.50 (the limit) should NOT trigger failure."""
        regime_perf = {
            2: {"n_trades": 10, "sharpe": -0.50, "max_drawdown": -0.10,
                "win_rate": 0.40, "cumulative_return": -0.03,
                "avg_return": -0.003, "avg_win": 0.008, "avg_loss": -0.008,
                "sortino": -0.55, "profit_factor": 0.8, "avg_holding_days": 12.0},
        }
        gate = PromotionGate()
        result = _result_with_regime_perf(regime_perf)
        decision = gate.evaluate(
            _candidate(), result, contract_metrics=_passing_contract_metrics(),
        )
        stress_sharpe_reasons = [r for r in decision.reasons if "stress_regime_2_sharpe" in r]
        self.assertEqual(len(stress_sharpe_reasons), 0,
                         "Sharpe exactly at limit should not fail")


if __name__ == "__main__":
    unittest.main()
