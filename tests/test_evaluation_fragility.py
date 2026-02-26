"""Tests for evaluation/fragility.py — PnL concentration, drawdown, recovery, critical slowing."""

import numpy as np
import pandas as pd
import pytest

from quant_engine.evaluation.fragility import (
    pnl_concentration,
    drawdown_distribution,
    recovery_time_distribution,
    detect_critical_slowing_down,
    consecutive_loss_frequency,
)


class TestPnlConcentration:
    """Test PnL concentration analysis."""

    def test_single_trade_dominates(self):
        """One huge trade should show 100% concentration."""
        trades = [
            {"pnl": 100.0},
            {"pnl": 1.0},
            {"pnl": 1.0},
            {"pnl": 1.0},
            {"pnl": 1.0},
            {"pnl": 1.0},
        ]
        result = pnl_concentration(trades, top_n_list=[5])
        assert result["top_5_pct"] > 0.95
        assert result["n_trades"] == 6

    def test_evenly_distributed_pnl(self):
        """Equal PnL trades: top 5 of 100 = ~5%."""
        trades = [{"pnl": 1.0} for _ in range(100)]
        result = pnl_concentration(trades, top_n_list=[5, 10, 20])
        assert abs(result["top_5_pct"] - 0.05) < 0.01
        assert abs(result["top_10_pct"] - 0.10) < 0.01
        assert abs(result["top_20_pct"] - 0.20) < 0.01
        assert result["fragile"] is False

    def test_fragile_flag(self):
        """High concentration should flag as fragile."""
        trades = [{"pnl": 100.0}] * 5 + [{"pnl": 0.1}] * 95
        result = pnl_concentration(trades, top_n_list=[5, 10, 20])
        assert result["fragile"] is True

    def test_empty_trades(self):
        result = pnl_concentration([], top_n_list=[5])
        assert result["n_trades"] == 0
        assert result["fragile"] is False

    def test_herfindahl_index(self):
        """HHI should be near 1/N for equal trades."""
        n = 100
        trades = [{"pnl": 1.0} for _ in range(n)]
        result = pnl_concentration(trades)
        assert abs(result["herfindahl_index"] - 1.0 / n) < 0.001


class TestDrawdownDistribution:
    """Test drawdown distribution analysis."""

    def test_basic_drawdown_stats(self):
        rng = np.random.RandomState(42)
        returns = pd.Series(rng.normal(0.0005, 0.02, 500))
        result = drawdown_distribution(returns)

        assert result["max_dd"] < 0  # Drawdown is negative
        assert result["n_episodes"] > 0
        assert 0 <= result["pct_time_underwater"] <= 1.0

    def test_no_drawdown_flat_returns(self):
        returns = pd.Series(np.ones(100) * 0.001)
        result = drawdown_distribution(returns)
        assert result["max_dd"] == 0.0
        assert result["n_episodes"] == 0

    def test_single_large_drawdown(self):
        """One big drop followed by recovery."""
        returns = pd.Series(
            [0.01] * 50 + [-0.05] * 5 + [0.01] * 45,
        )
        result = drawdown_distribution(returns)
        assert result["max_dd"] < -0.10
        assert result["n_episodes"] >= 1


class TestRecoveryTimeDistribution:
    """Test recovery time computation."""

    def test_recovery_times_are_positive(self):
        rng = np.random.RandomState(42)
        returns = pd.Series(
            rng.normal(0.0003, 0.02, 500),
            index=pd.bdate_range("2022-01-03", periods=500),
        )
        rt = recovery_time_distribution(returns)
        if len(rt) > 0:
            assert (rt > 0).all()

    def test_empty_returns(self):
        rt = recovery_time_distribution(pd.Series(dtype=float))
        assert len(rt) == 0

    def test_no_drawdown_no_recovery(self):
        returns = pd.Series(
            np.ones(100) * 0.001,
            index=pd.bdate_range("2022-01-03", periods=100),
        )
        rt = recovery_time_distribution(returns)
        assert len(rt) == 0


class TestCriticalSlowingDown:
    """Test critical slowing down detection."""

    def test_increasing_recovery_time_detected(self):
        """Recovery times trending upward should trigger detection."""
        recovery_times = pd.Series(
            [5, 8, 12, 15, 20, 25, 30, 40, 50, 60],
            index=pd.bdate_range("2022-01-03", periods=10),
        )
        critical, info = detect_critical_slowing_down(recovery_times)
        assert info["slope"] > 0
        assert info["recent_trend"] == "increasing"

    def test_stable_recovery_time_no_detection(self):
        """Flat recovery times should not trigger."""
        recovery_times = pd.Series(
            [10, 11, 9, 10, 12, 10, 11, 9, 10, 11],
            index=pd.bdate_range("2022-01-03", periods=10),
        )
        critical, info = detect_critical_slowing_down(recovery_times)
        assert info["recent_trend"] == "stable"

    def test_insufficient_data(self):
        recovery_times = pd.Series([5.0], index=pd.bdate_range("2022-01-03", periods=1))
        critical, info = detect_critical_slowing_down(recovery_times)
        assert critical is False
        assert info["recent_trend"] == "insufficient_data"


class TestConsecutiveLossFrequency:
    """Test consecutive loss streak analysis."""

    def test_loss_streaks_detected(self):
        """Mix of wins and losses should find streaks."""
        rng = np.random.RandomState(42)
        returns = pd.Series(rng.normal(-0.001, 0.02, 200))
        result = consecutive_loss_frequency(returns)
        assert result["max_streak"] > 0
        assert result["mean_streak"] > 0

    def test_all_positive_no_streaks(self):
        returns = pd.Series(np.ones(100) * 0.01)
        result = consecutive_loss_frequency(returns)
        assert result["max_streak"] == 0
