"""
Comprehensive tests for Spec 009: Kelly Position Sizing Overhaul.

Covers:
    T1: Kelly formula returns 0 for negative edge (not min_position)
    T2: Convex drawdown governor (exponential, not linear)
    T3: Per-regime Bayesian win-rate tracking
    T4: Regime stats updated from trade DataFrame
    T5: Confidence scalar capped at [0.5, 1.0]
    T6: Portfolio-level correlation penalty
    T7: DrawdownController wired into PaperTrader
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from quant_engine.risk.position_sizer import PositionSizer, PositionSize
from quant_engine.risk.drawdown import DrawdownController, DrawdownState


# ---------------------------------------------------------------------------
# T1: Kelly formula — negative edge → 0
# ---------------------------------------------------------------------------

class TestKellyNegativeEdge:
    """T1: Kelly returns 0.0 for negative-edge signals, not min_position."""

    def test_kelly_negative_edge_returns_zero(self):
        """When p*b < q, the Kelly fraction must be exactly 0.0."""
        ps = PositionSizer()
        # win_rate=0.3, avg_win=0.02, avg_loss=-0.03 → edge < 0
        size = ps._kelly(0.3, 0.02, -0.03)
        assert size == 0.0, f"Negative edge should give 0.0, got {size}"

    def test_kelly_positive_edge_returns_positive(self):
        """Positive-edge signals must produce a positive Kelly fraction."""
        ps = PositionSizer()
        size = ps._kelly(0.55, 0.03, -0.02)
        assert size > 0.0, f"Positive edge should give >0, got {size}"

    def test_kelly_invalid_inputs_return_zero(self):
        """Invalid inputs (win_rate=0, =1, avg_loss>=0) all return 0."""
        ps = PositionSizer()
        assert ps._kelly(0.0, 0.02, -0.02) == 0.0, "win_rate=0 should be 0"
        assert ps._kelly(1.0, 0.02, -0.02) == 0.0, "win_rate=1 should be 0"
        assert ps._kelly(0.5, 0.02, 0.0) == 0.0, "avg_loss>=0 should be 0"
        assert ps._kelly(0.5, 0.02, 0.001) == 0.0, "avg_loss>0 should be 0"

    def test_kelly_small_sample_penalty(self):
        """Low n_trades should reduce Kelly via sample penalty."""
        ps = PositionSizer(max_position_pct=1.0)  # avoid cap
        ps._current_n_trades = 10
        penalized = ps._kelly(0.55, 0.03, -0.02)
        ps._current_n_trades = 100
        full = ps._kelly(0.55, 0.03, -0.02)
        assert penalized < full, (
            f"Small sample ({penalized}) should be less than full ({full})"
        )
        # Penalty ratio should be n_trades / 50
        assert penalized == pytest.approx(full * (10 / 50), rel=1e-6)

    def test_kelly_n_trades_passed_through_size_position(self):
        """size_position() forwards n_trades to _kelly via _current_n_trades."""
        ps = PositionSizer(max_position_pct=1.0)
        result_low = ps.size_position(
            ticker="TEST", win_rate=0.55, avg_win=0.03, avg_loss=-0.02,
            realized_vol=0.25, atr=2.0, price=100.0, n_trades=10,
        )
        result_high = ps.size_position(
            ticker="TEST", win_rate=0.55, avg_win=0.03, avg_loss=-0.02,
            realized_vol=0.25, atr=2.0, price=100.0, n_trades=100,
        )
        assert result_low.raw_kelly < result_high.raw_kelly


# ---------------------------------------------------------------------------
# T2: Convex drawdown governor
# ---------------------------------------------------------------------------

class TestDrawdownGovernor:
    """T2: Exponential drawdown governor is more lenient early."""

    def test_convex_at_50pct_drawdown(self):
        """At 50% of max_dd, sizing should be > 50% (unlike linear = 50%)."""
        ps = PositionSizer()
        frac = ps._apply_drawdown_governor(0.5, -0.10, -0.20)
        assert frac > 0.25, f"At 50% DD, frac should be >0.25, got {frac}"

    def test_aggressive_at_90pct_drawdown(self):
        """At 90% of max_dd, sizing should be < 10%."""
        ps = PositionSizer()
        frac = ps._apply_drawdown_governor(0.5, -0.18, -0.20)
        assert frac < 0.10, f"At 90% DD, frac should be <0.10, got {frac}"

    def test_no_drawdown_full_sizing(self):
        """Zero drawdown returns the full kelly_fraction."""
        ps = PositionSizer()
        frac = ps._apply_drawdown_governor(0.5, 0.0, -0.20)
        assert frac == 0.5

    def test_beyond_max_dd_returns_zero(self):
        """When DD exceeds max, sizing is exactly 0."""
        ps = PositionSizer()
        frac = ps._apply_drawdown_governor(0.5, -0.25, -0.20)
        assert frac == 0.0

    def test_positive_drawdown_returns_full(self):
        """Positive drawdown (equity above HWM) returns full sizing."""
        ps = PositionSizer()
        frac = ps._apply_drawdown_governor(0.5, 0.05, -0.20)
        assert frac == 0.5

    def test_convex_curve_is_smooth(self):
        """Governor is monotonically decreasing as DD deepens."""
        ps = PositionSizer()
        prev = 1.0
        for pct in range(0, 100, 5):
            dd = -(pct / 100.0) * 0.20
            frac = ps._apply_drawdown_governor(1.0, dd, -0.20)
            assert frac <= prev + 1e-9, (
                f"Governor should be monotonic: at {pct}% got {frac}, prev was {prev}"
            )
            prev = frac


# ---------------------------------------------------------------------------
# T3: Per-regime Bayesian
# ---------------------------------------------------------------------------

class TestPerRegimeBayesian:
    """T3: Bayesian win-rate tracked per-regime with separate priors."""

    def _make_trades(self):
        """Create trade data where bull (0) wins more than HV (3)."""
        return pd.DataFrame({
            "net_return": (
                [0.02, 0.03, -0.01, 0.01, -0.02]       # regime 0 (3W 2L)
                + [-0.03, -0.01, -0.02, -0.01, -0.02]   # regime 3 (0W 5L)
                + [0.01, 0.02, 0.03, 0.01, 0.02,
                   0.01, 0.02, 0.01, 0.02, 0.01]         # regime 0 (10W 0L)
            ),
            "regime": [0]*5 + [3]*5 + [0]*10,
        })

    def test_bull_kelly_greater_than_hv(self):
        """Bull regime with more wins should have higher Kelly than HV."""
        ps = PositionSizer(max_position_pct=1.0)  # avoid cap
        ps.update_kelly_bayesian(self._make_trades())
        bull = ps.get_bayesian_kelly(0.02, -0.02, regime=0)
        hv = ps.get_bayesian_kelly(0.02, -0.02, regime=3)
        assert bull > hv, f"Bull ({bull}) should be > HV ({hv})"

    def test_global_fallback_when_few_regime_trades(self):
        """Regime with <10 trades falls back to global posterior."""
        ps = PositionSizer(max_position_pct=1.0)
        ps.update_kelly_bayesian(self._make_trades())
        # Regime 2 has 0 trades → falls back to global
        global_kelly = ps.get_bayesian_kelly(0.02, -0.02, regime=2)
        no_regime_kelly = ps.get_bayesian_kelly(0.02, -0.02, regime=None)
        assert global_kelly == no_regime_kelly

    def test_per_regime_counters_populated(self):
        """After update, per-regime counters should reflect trade data."""
        ps = PositionSizer()
        ps.update_kelly_bayesian(self._make_trades())
        # Bull: 13W, 2L
        assert ps._bayesian_regime[0]["wins"] == 13
        assert ps._bayesian_regime[0]["losses"] == 2
        # HV: 0W, 5L
        assert ps._bayesian_regime[3]["wins"] == 0
        assert ps._bayesian_regime[3]["losses"] == 5
        # Regime 2 untouched
        assert ps._bayesian_regime[2]["wins"] == 0


# ---------------------------------------------------------------------------
# T4: Regime stats from trade history
# ---------------------------------------------------------------------------

class TestRegimeStatsUpdate:
    """T4: Regime stats updated from trade DataFrame with regime column.

    SPEC-P01: MIN_REGIME_TRADES_FOR_STATS = 30 — tests use 40 trades
    per regime to exceed the threshold.
    """

    def test_update_from_trade_df(self):
        """Stats for trending_bull should reflect trade data."""
        ps = PositionSizer()
        # 40 trades: 32 wins + 8 losses → win_rate = 0.80
        returns = [0.03, 0.02, -0.01, 0.04, 0.01,
                   0.02, -0.02, 0.03, 0.01, 0.02] * 4
        trades = pd.DataFrame({
            "net_return": returns,
            "regime": [0] * 40,
        })
        ps.update_regime_stats(trades, persist=False)
        stats = ps.regime_stats["trending_bull"]
        assert stats["n_trades"] == 40
        assert stats["win_rate"] == pytest.approx(0.8, abs=0.01)

    def test_too_few_trades_keeps_defaults(self):
        """Regimes with < MIN_REGIME_TRADES_FOR_STATS trades keep defaults."""
        ps = PositionSizer()
        default_wr = ps.regime_stats["high_volatility"]["win_rate"]
        # Only 10 trades — below the 30-trade threshold
        trades = pd.DataFrame({
            "net_return": [0.01, -0.01, 0.02] * 3 + [0.01],
            "regime": [3] * 10,
        })
        ps.update_regime_stats(trades, persist=False)
        assert ps.regime_stats["high_volatility"]["win_rate"] == default_wr

    def test_multiple_regimes_updated(self):
        """Multiple regimes in one call are all updated."""
        ps = PositionSizer()
        trades = pd.DataFrame({
            "net_return": [0.02] * 35 + [-0.01] * 35,
            "regime": [0] * 35 + [1] * 35,
        })
        ps.update_regime_stats(trades, persist=False)
        assert ps.regime_stats["trending_bull"]["n_trades"] == 35
        assert ps.regime_stats["trending_bear"]["n_trades"] == 35
        assert ps.regime_stats["trending_bull"]["win_rate"] == 1.0
        assert ps.regime_stats["trending_bear"]["win_rate"] == 0.0


# ---------------------------------------------------------------------------
# T5: Confidence scalar capped
# ---------------------------------------------------------------------------

class TestConfidenceScalar:
    """T5: Confidence scalar range is [0.5, 1.0], never 1.5."""

    def test_max_confidence_scalar_is_one(self):
        """At confidence=1.0, scalar should be exactly 1.0."""
        ps = PositionSizer()
        result = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0, confidence=1.0,
        )
        assert result.sizing_details["confidence_scalar"] == 1.0

    def test_zero_confidence_scalar_is_half(self):
        """At confidence=0.0, scalar should be exactly 0.5."""
        ps = PositionSizer()
        result = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0, confidence=0.0,
        )
        assert result.sizing_details["confidence_scalar"] == 0.5

    def test_confidence_never_amplifies(self):
        """No confidence value should produce a scalar > 1.0."""
        ps = PositionSizer()
        for conf in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            result = ps.size_position(
                "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
                confidence=conf,
            )
            assert result.sizing_details["confidence_scalar"] <= 1.0, (
                f"Scalar {result.sizing_details['confidence_scalar']} > 1.0 at conf={conf}"
            )


# ---------------------------------------------------------------------------
# T6: Portfolio-level correlation penalty
# ---------------------------------------------------------------------------

class TestPortfolioCorrelationPenalty:
    """T6: High-correlation positions get reduced allocation."""

    @pytest.fixture
    def returns_data(self):
        """Generate return series with known correlation structure."""
        np.random.seed(42)
        n = 100
        base = np.random.randn(n) * 0.01
        idx = pd.date_range("2025-01-01", periods=n)
        return {
            "EXISTING": pd.Series(base, index=idx),
            "HIGH_CORR": pd.Series(base + np.random.randn(n) * 0.002, index=idx),
            "UNCORR": pd.Series(np.random.randn(n) * 0.01, index=idx),
        }

    def test_high_corr_smaller_than_uncorr(self, returns_data):
        """Highly correlated position should be smaller than uncorrelated."""
        ps = PositionSizer()
        existing = {"EXISTING": 0.10}
        size_high = ps.size_portfolio_aware(
            "HIGH_CORR", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            existing_positions=existing, returns_data=returns_data,
        )
        size_uncorr = ps.size_portfolio_aware(
            "UNCORR", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            existing_positions=existing, returns_data=returns_data,
        )
        assert size_high < size_uncorr

    def test_no_positions_returns_base(self, returns_data):
        """Empty portfolio should return the base (unpenalized) size."""
        ps = PositionSizer()
        size_aware = ps.size_portfolio_aware(
            "HIGH_CORR", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            existing_positions={}, returns_data=returns_data,
        )
        base = ps.size_position(
            "HIGH_CORR", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
        ).final_size
        assert size_aware == base

    def test_negative_corr_no_penalty(self, returns_data):
        """Negatively correlated positions should not be penalized."""
        ps = PositionSizer()
        # Create negatively correlated series
        returns_data["NEG_CORR"] = -returns_data["EXISTING"] + np.random.randn(100) * 0.001
        existing = {"EXISTING": 0.10}
        size_neg = ps.size_portfolio_aware(
            "NEG_CORR", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            existing_positions=existing, returns_data=returns_data,
        )
        base = ps.size_position(
            "NEG_CORR", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
        ).final_size
        # Negatively correlated → no penalty, should get full or near-full size
        assert size_neg >= base * 0.99


# ---------------------------------------------------------------------------
# T7: DrawdownController integration
# ---------------------------------------------------------------------------

class TestDrawdownControllerIntegration:
    """T7: DrawdownController blocks entries and forces liquidation."""

    def test_normal_state_allows_entries(self):
        """In normal state, entries are allowed with full sizing."""
        dc = DrawdownController()
        status = dc.update(0.01)  # Positive day
        assert status.state == DrawdownState.NORMAL
        assert status.allow_new_entries is True
        assert status.size_multiplier == 1.0
        assert status.force_liquidate is False

    def test_large_loss_triggers_caution(self):
        """A large daily loss should trigger CAUTION and block entries."""
        dc = DrawdownController()
        status = dc.update(-0.05)  # -5% daily loss
        assert status.state == DrawdownState.CAUTION
        assert status.allow_new_entries is False
        assert status.size_multiplier < 1.0

    def test_critical_drawdown_forces_liquidation(self):
        """Deep drawdown should trigger CRITICAL and force liquidation."""
        # Relax daily/weekly limits so the cumulative drawdown tier fires.
        dc = DrawdownController(
            initial_equity=1.0,
            daily_loss_limit=-0.50,   # relaxed
            weekly_loss_limit=-0.50,  # relaxed
        )
        for _ in range(10):
            status = dc.update(-0.02)  # 2% loss per day, cumulative ~-18%
        assert status.force_liquidate is True
        assert status.state == DrawdownState.CRITICAL

    def test_paper_trader_has_drawdown_controller(self):
        """PaperTrader should initialize with a DrawdownController."""
        from quant_engine.autopilot.paper_trader import PaperTrader
        pt = PaperTrader()
        assert hasattr(pt, "_dd_controller")
        assert isinstance(pt._dd_controller, DrawdownController)

    def test_recovery_allows_gradual_reentry(self):
        """After crisis subsides, recovery uses concave ramp with cautious early entry.

        Spec 016 T7: New entries are blocked until 30% of recovery period
        elapses, then gradually re-enabled with quadratic size ramp.
        """
        dc = DrawdownController(recovery_days=5)
        # Enter caution
        dc.update(-0.05)
        # Recover
        status = dc.update(0.03)
        if status.state == DrawdownState.RECOVERY:
            # Early recovery: entries blocked (progress < 0.3)
            assert status.allow_new_entries is False
            assert status.size_multiplier < 1.0
            # After 30%+ of recovery_days (2+ of 5), entries should be allowed
            for _ in range(2):
                status = dc.update(0.001)
            if status.state == DrawdownState.RECOVERY:
                assert status.allow_new_entries is True


# ---------------------------------------------------------------------------
# Integration: size_position with all fixes
# ---------------------------------------------------------------------------

class TestSizePositionIntegration:
    """End-to-end tests for the updated size_position method."""

    def test_negative_edge_signal_still_gets_sized_via_blend(self):
        """A negative-edge signal contributes 0 Kelly but still gets vol + ATR."""
        ps = PositionSizer()
        result = ps.size_position(
            "TEST", win_rate=0.3, avg_win=0.02, avg_loss=-0.03,
            realized_vol=0.25, atr=2.0, price=100.0,
        )
        # Kelly is 0 but vol_scaled and atr_based contribute
        assert result.raw_kelly == 0.0
        assert result.vol_scaled > 0
        assert result.atr_based > 0
        assert result.final_size > 0

    def test_drawdown_reduces_kelly_in_blend(self):
        """Current drawdown should reduce the Kelly component of the blend."""
        ps = PositionSizer()
        result_normal = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            current_drawdown=0.0,
        )
        result_dd = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            current_drawdown=-0.15,
        )
        assert result_dd.half_kelly < result_normal.half_kelly
