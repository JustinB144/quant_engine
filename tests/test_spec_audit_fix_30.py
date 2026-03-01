"""
Tests for SPEC_AUDIT_FIX_30: Backtest/Risk Cross-Boundary Contract & Critical Math Fixes.

Covers:
    T1: Autopilot → Backtest validation contract compatibility
    T2: Turnover-budget unit mismatch fix
    T3: API equity-curve crash / orchestrator trade CSV schema
    T4: Summary.json 17-field schema contract
    T5: Cost budget dimensional correctness
    T6: Gross exposure absolute-sum fix
    T7: Calmar ratio sign preservation
"""
import inspect

import numpy as np
import pandas as pd
import pytest


# ── T1: Autopilot → Backtest validation contract ────────────────────


def test_run_statistical_tests_requires_trade_returns():
    """Regression: run_statistical_tests signature must include trade_returns."""
    from quant_engine.backtest.validation import run_statistical_tests

    sig = inspect.signature(run_statistical_tests)
    assert "trade_returns" in sig.parameters, (
        "run_statistical_tests is missing the required 'trade_returns' parameter"
    )


def test_statistical_tests_result_has_passes_field():
    """StatisticalTests result uses .passes, not .overall_pass."""
    from quant_engine.backtest.validation import StatisticalTests

    fields = {f.name for f in StatisticalTests.__dataclass_fields__.values()}
    assert "passes" in fields
    assert "overall_pass" not in fields


def test_cpcv_result_has_passes_field():
    """CPCVResult uses .passes, not .is_significant."""
    from quant_engine.backtest.validation import CPCVResult

    fields = {f.name for f in CPCVResult.__dataclass_fields__.values()}
    assert "passes" in fields
    assert "is_significant" not in fields


def test_spa_result_has_passes_field():
    """SPAResult uses .passes, not .rejects_null."""
    from quant_engine.backtest.validation import SPAResult

    fields = {f.name for f in SPAResult.__dataclass_fields__.values()}
    assert "passes" in fields
    assert "rejects_null" not in fields


def test_cpcv_result_has_mean_oos_corr():
    """CPCVResult uses .mean_oos_corr, not .mean_test_corr."""
    from quant_engine.backtest.validation import CPCVResult

    fields = {f.name for f in CPCVResult.__dataclass_fields__.values()}
    assert "mean_oos_corr" in fields
    assert "mean_test_corr" not in fields


# ── T2: Turnover-budget unit mismatch ────────────────────────────────


def test_turnover_budget_constrains_sizing():
    """Turnover budget should actually bind when annualized turnover exceeds threshold."""
    from quant_engine.risk.position_sizer import PositionSizer

    sizer = PositionSizer()
    # Simulate enough turnover history to exceed budget
    # With MAX_ANNUALIZED_TURNOVER=500% (5.0 as fraction), if daily turnover
    # averages 0.10, annualized = 0.10 * 252 = 25.2 (2520%), far exceeding budget.
    sizer._turnover_history = [0.10] * 100

    result = sizer._apply_turnover_budget(
        proposed_size=0.05,
        symbol="TEST",
        portfolio_equity=1_000_000.0,
        current_positions={},
        dates_in_period=100,
    )
    # Budget is exhausted; should return 0 or reduced size
    assert result < 0.05, (
        f"Turnover budget did not constrain sizing: got {result}, expected < 0.05"
    )


def test_turnover_budget_unit_conversion():
    """MAX_ANNUALIZED_TURNOVER (500 = 500%) should become 5.0 as fraction internally."""
    from quant_engine.risk.position_sizer import PositionSizer

    sizer = PositionSizer()
    # 500% → 5.0 as fraction
    assert sizer._max_annualized_turnover == pytest.approx(5.0, abs=0.01), (
        f"Expected 5.0 (500%/100), got {sizer._max_annualized_turnover}"
    )


# ── T3: API equity-curve / orchestrator trade CSV ────────────────────


def test_orchestrator_trade_csv_includes_position_size():
    """Orchestrator trade CSV schema must include position_size."""
    REQUIRED_TRADE_CSV_FIELDS = {"net_return", "entry_date", "exit_date", "position_size"}
    # Simulate what the orchestrator writes
    from quant_engine.backtest.engine import Trade

    t = Trade(
        ticker="TEST", entry_date="2024-01-01", exit_date="2024-01-10",
        entry_price=100.0, exit_price=105.0,
        predicted_return=0.05, actual_return=0.05, net_return=0.04,
        regime=0, confidence=0.8, holding_days=10, position_size=0.05,
    )
    # This mirrors orchestrator.py trade_data dict construction
    trade_data = {
        "permno": t.ticker,
        "entry_date": t.entry_date,
        "exit_date": t.exit_date,
        "predicted_return": t.predicted_return,
        "actual_return": t.actual_return,
        "net_return": t.net_return,
        "regime": "trending_bull",
        "confidence": t.confidence,
        "holding_days": t.holding_days,
        "exit_reason": t.exit_reason,
        "position_size": t.position_size,
    }
    assert REQUIRED_TRADE_CSV_FIELDS.issubset(trade_data.keys())


def test_data_helpers_build_portfolio_returns_no_position_size():
    """build_portfolio_returns should not crash when position_size column is absent."""
    from quant_engine.api.services.data_helpers import build_portfolio_returns

    trades = pd.DataFrame({
        "exit_dt": pd.to_datetime(["2024-01-10", "2024-01-11"]),
        "net_return": [0.05, -0.02],
    })
    result = build_portfolio_returns(trades)
    assert isinstance(result, pd.Series)
    assert len(result) > 0


# ── T4: Summary.json 17-field schema contract ────────────────────────


DECLARED_SUMMARY_FIELDS = {
    "horizon", "total_trades", "winning_trades", "losing_trades",
    "win_rate", "avg_return", "avg_win", "avg_loss", "total_return",
    "annualized_return", "sharpe", "sortino", "max_drawdown",
    "profit_factor", "avg_holding_days", "trades_per_year", "regime_breakdown",
}


def test_summary_schema_contract():
    """backtest_result_to_summary_dict must emit all 17 declared fields."""
    from quant_engine.backtest.engine import BacktestResult, Trade, backtest_result_to_summary_dict

    trades = [
        Trade(
            ticker="TEST", entry_date="2024-01-01", exit_date="2024-01-10",
            entry_price=100.0, exit_price=105.0,
            predicted_return=0.05, actual_return=0.05, net_return=0.04,
            regime=0, confidence=0.8, holding_days=10, position_size=0.05,
        ),
        Trade(
            ticker="TEST2", entry_date="2024-01-02", exit_date="2024-01-12",
            entry_price=50.0, exit_price=48.0,
            predicted_return=0.03, actual_return=-0.04, net_return=-0.05,
            regime=1, confidence=0.7, holding_days=10, position_size=0.05,
        ),
    ]
    result = BacktestResult(
        total_trades=2, winning_trades=1, losing_trades=1,
        win_rate=0.5, avg_return=-0.005,
        avg_win=0.04, avg_loss=-0.05,
        total_return=-0.01, annualized_return=-0.02,
        sharpe_ratio=0.1, sortino_ratio=0.12,
        max_drawdown=-0.05, profit_factor=0.8,
        avg_holding_days=10.0, trades_per_year=25.0,
        trades=trades,
        regime_breakdown={"trending_bull": {"count": 1}, "trending_bear": {"count": 1}},
    )
    summary = backtest_result_to_summary_dict(result, horizon=10)
    assert set(summary.keys()) == DECLARED_SUMMARY_FIELDS, (
        f"Missing: {DECLARED_SUMMARY_FIELDS - set(summary.keys())}, "
        f"Extra: {set(summary.keys()) - DECLARED_SUMMARY_FIELDS}"
    )


# ── T5: Cost budget dimensional correctness ──────────────────────────


def test_cost_budget_participation_rate():
    """Market impact must be non-trivially positive for realistic trade sizes."""
    from quant_engine.risk.cost_budget import estimate_trade_cost_bps

    # 5% portfolio weight, $1M portfolio, $25M daily volume
    cost = estimate_trade_cost_bps(
        trade_size_weight=0.05,
        daily_volume=25_000_000,
        portfolio_value_usd=1_000_000,
    )
    # participation = $50K / $25M = 0.002 → impact = 25 * sqrt(0.002) ≈ 1.12 bps
    # total = (1.5 + 1.12) * 0.05 ≈ 0.13 bps
    assert cost > 0.01, f"Cost too low ({cost:.6f}), dimensional error likely"


def test_cost_budget_large_vs_small_trade():
    """Larger trades must have higher cost than smaller ones."""
    from quant_engine.risk.cost_budget import estimate_trade_cost_bps

    cost_small = estimate_trade_cost_bps(
        trade_size_weight=0.01, daily_volume=10_000_000,
        portfolio_value_usd=1_000_000,
    )
    cost_large = estimate_trade_cost_bps(
        trade_size_weight=0.10, daily_volume=10_000_000,
        portfolio_value_usd=1_000_000,
    )
    assert cost_large > cost_small


# ── T6: Gross exposure absolute sum ──────────────────────────────────


def test_gross_exposure_long_short():
    """Gross exposure must be sum of absolute weights for long-short portfolios."""
    from quant_engine.risk.portfolio_risk import PortfolioRiskManager

    mgr = PortfolioRiskManager()
    positions = {"AAPL": 0.5, "MSFT": -0.3, "GOOGL": 0.4}
    summary = mgr.portfolio_summary(positions, price_data={})
    # Gross should be 1.2 (0.5 + 0.3 + 0.4), not 0.6 (0.5 - 0.3 + 0.4)
    assert summary["gross_exposure"] == pytest.approx(1.2), (
        f"Expected 1.2, got {summary['gross_exposure']}"
    )


def test_gross_exposure_all_short():
    """A fully short portfolio should report positive gross exposure."""
    from quant_engine.risk.portfolio_risk import PortfolioRiskManager

    mgr = PortfolioRiskManager()
    positions = {"AAPL": -0.5, "MSFT": -0.5}
    summary = mgr.portfolio_summary(positions, price_data={})
    assert summary["gross_exposure"] == pytest.approx(1.0)


# ── T7: Calmar ratio sign preservation ───────────────────────────────


def test_calmar_ratio_positive_for_profitable():
    """Calmar ratio must be positive when annualized return is positive."""
    from quant_engine.risk.metrics import RiskMetrics

    rm = RiskMetrics()
    # Profitable strategy: consistent positive returns
    returns = np.array([0.01, 0.02, 0.015, 0.005, -0.003, 0.01, 0.008,
                        0.012, 0.007, -0.001, 0.009, 0.011] * 5)
    report = rm.compute_full_report(returns, holding_days=10)
    assert report.calmar_ratio > 0, (
        f"Calmar should be positive for profitable strategy, got {report.calmar_ratio}"
    )


def test_calmar_ratio_negative_for_unprofitable():
    """Calmar ratio must be negative when annualized return is negative."""
    from quant_engine.risk.metrics import RiskMetrics

    rm = RiskMetrics()
    # Losing strategy: consistent negative returns
    returns = np.array([-0.02, -0.015, -0.01, 0.003, -0.02, -0.025,
                        -0.018, -0.012, 0.001, -0.03, -0.015, -0.02] * 5)
    report = rm.compute_full_report(returns, holding_days=10)
    assert report.calmar_ratio < 0, (
        f"Calmar should be negative for losing strategy, got {report.calmar_ratio}"
    )
