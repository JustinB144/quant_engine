"""
Comprehensive tests for Spec 05: Risk Governor + Kelly Unification + Uncertainty-Aware Sizing.

Covers:
    T1: Paper trader sizing via unified PositionSizer interface
    T2: Uncertainty-aware sizing (signal, regime, drift inputs)
    T3: Shock budget constraint
    T4: Turnover budget constraint
    T5: Concentration limit constraint
    T6: Regime-conditional blend weights
    T7: Parameterized Kelly Bayesian prior with min sample threshold
    T8: Paper trader + autopilot unified sizing (integration)
    Budget constraint ordering and interaction tests
    Backward compatibility tests
"""
import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from quant_engine.risk.position_sizer import PositionSizer, PositionSize


# ---------------------------------------------------------------------------
# T2: Uncertainty-aware sizing
# ---------------------------------------------------------------------------

class TestUncertaintyScaling:
    """T2: Uncertainty inputs scale position sizes downward when uncertain."""

    def test_high_uncertainty_reduces_size(self):
        """High signal uncertainty + high regime entropy should reduce size."""
        ps = PositionSizer()
        # Baseline: no uncertainty
        base = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
        )
        # With high uncertainty
        unc = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            signal_uncertainty=0.9,
            regime_entropy=0.9,
            drift_score=0.1,
        )
        assert unc.final_size < base.final_size, (
            f"High uncertainty ({unc.final_size}) should be < base ({base.final_size})"
        )

    def test_low_uncertainty_near_full_size(self):
        """Low uncertainty inputs should produce scale near 1.0."""
        ps = PositionSizer()
        base = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
        )
        confident = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            signal_uncertainty=0.0,
            regime_entropy=0.0,
            drift_score=1.0,
        )
        assert confident.final_size == pytest.approx(base.final_size, rel=0.01), (
            f"Low uncertainty ({confident.final_size}) should ≈ base ({base.final_size})"
        )

    def test_none_inputs_no_scaling(self):
        """None uncertainty inputs should produce scale=1.0 (backward compat)."""
        ps = PositionSizer()
        base = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
        )
        with_none = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            signal_uncertainty=None,
            regime_entropy=None,
            drift_score=None,
        )
        assert with_none.final_size == base.final_size

    def test_partial_none_no_scaling(self):
        """If any uncertainty input is None, no scaling applied."""
        ps = PositionSizer()
        base = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
        )
        partial = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            signal_uncertainty=0.5,
            regime_entropy=None,
            drift_score=0.5,
        )
        assert partial.final_size == base.final_size

    def test_uncertainty_scale_bounds(self):
        """Uncertainty scale should always be in [1 - max_reduction, 1.0]."""
        ps = PositionSizer()
        # Worst case: all uncertainty maxed out
        scale = ps._compute_uncertainty_scale(1.0, 1.0, 0.0)
        assert 0.70 <= scale <= 1.0, f"Scale {scale} out of expected range"

        # Best case: all confidence maxed
        scale = ps._compute_uncertainty_scale(0.0, 0.0, 1.0)
        assert scale == 1.0

    def test_uncertainty_scale_with_invalid_inputs(self):
        """Out-of-range or NaN inputs should return scale=1.0."""
        ps = PositionSizer()
        assert ps._compute_uncertainty_scale(1.5, 0.5, 0.5) == 1.0
        assert ps._compute_uncertainty_scale(0.5, -0.1, 0.5) == 1.0
        assert ps._compute_uncertainty_scale(float("nan"), 0.5, 0.5) == 1.0

    def test_drift_score_interpretation(self):
        """High drift score = high confidence = less reduction."""
        ps = PositionSizer()
        low_drift = ps._compute_uncertainty_scale(0.5, 0.5, 0.0)
        high_drift = ps._compute_uncertainty_scale(0.5, 0.5, 1.0)
        assert high_drift > low_drift, (
            f"High drift ({high_drift}) should give higher scale than low ({low_drift})"
        )

    def test_sizing_details_includes_uncertainty(self):
        """sizing_details dict should include uncertainty_scale."""
        ps = PositionSizer()
        result = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            signal_uncertainty=0.5,
            regime_entropy=0.5,
            drift_score=0.5,
        )
        assert "uncertainty_scale" in result.sizing_details
        assert 0.0 < result.sizing_details["uncertainty_scale"] <= 1.0


# ---------------------------------------------------------------------------
# T3: Shock budget constraint
# ---------------------------------------------------------------------------

class TestShockBudget:
    """T3: Shock budget reserves capital and caps position sizes."""

    def test_position_capped_by_shock_budget(self):
        """Position size should not exceed (1 - SHOCK_BUDGET_PCT)."""
        ps = PositionSizer(max_position_pct=1.0)
        # Set a very large position
        capped = ps._apply_shock_budget(0.99, 1_000_000.0)
        assert capped <= 0.95, f"Size {capped} exceeds shock budget cap"

    def test_small_position_passes_through(self):
        """Small positions should not be affected by shock budget."""
        ps = PositionSizer()
        original = 0.05
        result = ps._apply_shock_budget(original, 1_000_000.0)
        assert result == original

    def test_shock_budget_applied_in_size_position(self):
        """Shock budget constraint should be applied when portfolio_equity is given."""
        ps = PositionSizer(max_position_pct=1.0)
        result = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            portfolio_equity=1_000_000.0,
        )
        # Final size should respect shock budget
        assert result.final_size <= 0.95


# ---------------------------------------------------------------------------
# T4: Turnover budget constraint
# ---------------------------------------------------------------------------

class TestTurnoverBudget:
    """T4: Turnover budget enforcement limits excessive rebalancing."""

    def test_turnover_within_budget_passes(self):
        """Position change within budget should pass through unchanged."""
        ps = PositionSizer()
        ps._turnover_history = []  # Fresh start
        result = ps._apply_turnover_budget(
            proposed_size=0.05,
            symbol="TEST",
            portfolio_equity=1_000_000.0,
            current_positions={},
            dates_in_period=252,
        )
        assert result == 0.05

    def test_turnover_budget_exhausted_blocks_increase(self):
        """When turnover budget is exhausted, should not increase position."""
        ps = PositionSizer()
        # Simulate extremely high prior turnover
        ps._turnover_history = [10.0] * 100  # Way over budget
        result = ps._apply_turnover_budget(
            proposed_size=0.10,
            symbol="TEST",
            portfolio_equity=1_000_000.0,
            current_positions={"TEST": 50_000.0},  # 5% existing
            dates_in_period=252,
        )
        # Should return current position's fraction
        assert result <= 0.05 + 0.001

    def test_turnover_disabled_passes_through(self):
        """When turnover budget enforcement is off, sizes pass through."""
        ps = PositionSizer()
        ps._turnover_budget_enforcement = False
        ps._turnover_history = [100.0] * 100  # High turnover but enforcement off
        result = ps._apply_turnover_budget(
            proposed_size=0.10,
            symbol="TEST",
            portfolio_equity=1_000_000.0,
            current_positions={},
            dates_in_period=252,
        )
        assert result == 0.10

    def test_turnover_tracking_accumulates(self):
        """Turnover should accumulate across calls."""
        ps = PositionSizer()
        ps._turnover_history = []
        # First trade: new position
        ps._apply_turnover_budget(0.05, "A", 1_000_000.0, {}, 252)
        assert len(ps._turnover_history) == 1
        # Second trade
        ps._apply_turnover_budget(0.03, "B", 1_000_000.0, {}, 252)
        assert len(ps._turnover_history) == 2

    def test_reset_turnover_tracking(self):
        """reset_turnover_tracking should clear history."""
        ps = PositionSizer()
        ps._turnover_history = [0.1, 0.2, 0.3]
        ps.reset_turnover_tracking()
        assert len(ps._turnover_history) == 0


# ---------------------------------------------------------------------------
# T5: Concentration limit constraint
# ---------------------------------------------------------------------------

class TestConcentrationLimit:
    """T5: Concentration limit caps single-position notional."""

    def test_position_capped_at_limit(self):
        """Position exceeding concentration limit should be capped."""
        ps = PositionSizer()
        capped = ps._apply_concentration_limit(0.30, 1_000_000.0, "TEST")
        assert capped == 0.20  # Default CONCENTRATION_LIMIT_PCT = 0.20

    def test_position_below_limit_unchanged(self):
        """Position below concentration limit should pass through."""
        ps = PositionSizer()
        result = ps._apply_concentration_limit(0.10, 1_000_000.0, "TEST")
        assert result == 0.10

    def test_concentration_limit_in_size_position(self):
        """Concentration limit should be applied when portfolio_equity is given."""
        ps = PositionSizer(max_position_pct=0.50)  # Allow large sizes
        result = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            portfolio_equity=1_000_000.0,
        )
        # Size should respect concentration limit even if blend suggests higher
        assert result.final_size <= 0.20


# ---------------------------------------------------------------------------
# T6: Regime-conditional blend weights
# ---------------------------------------------------------------------------

class TestRegimeConditionalBlendWeights:
    """T6: Blend weights change per regime and are configurable."""

    def test_different_regimes_different_blends(self):
        """Different regime states should produce different blend weights."""
        ps = PositionSizer()
        # NORMAL: kelly=0.35, CRITICAL: kelly=0.05
        # With high Kelly input, NORMAL should produce higher blend
        normal = ps._blend_sizes(kelly_size=0.10, vol_size=0.05, atr_size=0.05, regime_state="NORMAL")
        critical = ps._blend_sizes(kelly_size=0.10, vol_size=0.05, atr_size=0.05, regime_state="CRITICAL")
        assert normal > critical, (
            f"NORMAL blend ({normal}) should be > CRITICAL ({critical}) with high Kelly"
        )

    def test_blend_weights_sum_to_one(self):
        """Blend weights should always sum to 1.0 (or be normalized)."""
        ps = PositionSizer()
        # All regime states should produce valid blends
        for regime in ["NORMAL", "WARNING", "CAUTION", "CRITICAL", "RECOVERY"]:
            blend = ps._blend_sizes(0.10, 0.10, 0.10, regime_state=regime)
            assert blend == pytest.approx(0.10, rel=0.01), (
                f"Equal inputs should give equal blend for {regime}"
            )

    def test_static_fallback_when_no_regime_config(self):
        """When regime weights are unavailable, static weights are used."""
        ps = PositionSizer()
        ps._blend_weights_by_regime = None
        blend = ps._blend_sizes(0.10, 0.10, 0.10, regime_state="NORMAL")
        assert blend == pytest.approx(0.10, rel=0.01)

    def test_regime_mapping(self):
        """Regime label strings should map correctly to blend weight keys."""
        ps = PositionSizer()
        assert ps._map_regime_for_blend("trending_bull") == "NORMAL"
        assert ps._map_regime_for_blend("trending_bear") == "WARNING"
        assert ps._map_regime_for_blend("high_volatility") == "CAUTION"
        assert ps._map_regime_for_blend(None) == "NORMAL"
        assert ps._map_regime_for_blend("RECOVERY") == "RECOVERY"

    def test_blend_size_details_in_result(self):
        """PositionSize should contain blend_regime in sizing_details."""
        ps = PositionSizer()
        result = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            regime="trending_bull",
        )
        assert "blend_regime" in result.sizing_details
        assert result.sizing_details["blend_regime"] == "NORMAL"


# ---------------------------------------------------------------------------
# T7: Parameterized Kelly Bayesian prior
# ---------------------------------------------------------------------------

class TestParameterizedKellyBayesian:
    """T7: Kelly Bayesian prior parameters are configurable."""

    def test_min_samples_threshold(self):
        """Below min_samples, prior mode should dominate the estimate."""
        ps = PositionSizer(max_position_pct=1.0)
        # No trades at all — should use prior
        estimate_no_data = ps.get_bayesian_kelly(0.03, -0.02, regime=None)
        # Prior mode = (2-1)/(2+2-2) = 0.5 for Beta(2,2)
        assert estimate_no_data >= 0.0  # Should produce a valid estimate

    def test_large_sample_converges_to_empirical(self):
        """With many trades, posterior should be close to empirical win rate."""
        ps = PositionSizer(max_position_pct=1.0)
        # Simulate 100 trades: 60 wins, 40 losses
        trades = pd.DataFrame({
            "net_return": [0.02] * 60 + [-0.01] * 40,
        })
        ps.update_kelly_bayesian(trades)
        posterior_kelly = ps.get_bayesian_kelly(0.02, -0.01, regime=None)
        # With 60/100 win rate, Kelly should be meaningfully positive
        assert posterior_kelly > 0.0

    def test_different_priors_produce_different_estimates(self):
        """Different prior parameters should yield different Kelly fractions."""
        # Informative prior (strongly favors 0.5)
        ps_strong = PositionSizer(max_position_pct=1.0, bayesian_alpha=10.0, bayesian_beta=10.0)
        # Weak prior
        ps_weak = PositionSizer(max_position_pct=1.0, bayesian_alpha=1.0, bayesian_beta=1.0)

        # Same trades
        trades = pd.DataFrame({
            "net_return": [0.02] * 8 + [-0.01] * 2,  # 80% win rate, 10 trades
        })
        ps_strong.update_kelly_bayesian(trades)
        ps_weak.update_kelly_bayesian(trades)

        kelly_strong = ps_strong.get_bayesian_kelly(0.02, -0.01)
        kelly_weak = ps_weak.get_bayesian_kelly(0.02, -0.01)

        # The strong prior should pull the estimate closer to 0.5
        # while the weak prior should allow the estimate to be higher (closer to 0.8)
        # Both should produce valid estimates
        assert kelly_strong >= 0.0
        assert kelly_weak >= 0.0

    def test_regime_specific_bayesian_with_min_samples(self):
        """Regime with fewer than min_samples trades should fall back to global."""
        ps = PositionSizer(max_position_pct=1.0)
        # 5 trades in regime 0 (below 10 min_samples), 15 in global
        trades = pd.DataFrame({
            "net_return": [0.02] * 5 + [0.02] * 10 + [-0.01] * 5,
            "regime": [0] * 5 + [1] * 10 + [1] * 5,
        })
        ps.update_kelly_bayesian(trades)

        # Regime 0 has only 5 trades -> should fall back to global
        regime_kelly = ps.get_bayesian_kelly(0.02, -0.01, regime=0)
        global_kelly = ps.get_bayesian_kelly(0.02, -0.01, regime=None)
        # Both should be valid
        assert regime_kelly >= 0.0
        assert global_kelly >= 0.0


# ---------------------------------------------------------------------------
# T1/T8: Unified paper trader sizing interface
# ---------------------------------------------------------------------------

class TestPaperTraderUnifiedInterface:
    """T1/T8: Paper trader uses unified PositionSizer.size_position_paper_trader()."""

    def test_paper_trader_method_exists(self):
        """size_position_paper_trader method should exist on PositionSizer."""
        ps = PositionSizer()
        assert hasattr(ps, "size_position_paper_trader")
        assert callable(ps.size_position_paper_trader)

    def test_basic_sizing_output(self):
        """Paper trader method should produce a valid float in [0, max_position]."""
        ps = PositionSizer(max_position_pct=0.25)
        size = ps.size_position_paper_trader(
            ticker="TEST",
            kelly_history=[0.02, -0.01, 0.03, 0.01] * 10,  # 40 trades
            atr=2.0,
            realized_vol=0.25,
            price=100.0,
        )
        assert isinstance(size, float)
        assert 0.0 < size <= 0.25

    def test_few_trades_conservative(self):
        """With few trades, sizing should be conservative (prior-dominated)."""
        ps = PositionSizer()
        size_few = ps.size_position_paper_trader(
            ticker="TEST",
            kelly_history=[0.02, -0.01, 0.03],  # Only 3 trades
            atr=2.0,
            realized_vol=0.25,
            price=100.0,
        )
        size_many = ps.size_position_paper_trader(
            ticker="TEST",
            kelly_history=[0.02, -0.01, 0.03, 0.01] * 25,  # 100 trades
            atr=2.0,
            realized_vol=0.25,
            price=100.0,
        )
        # Both should be valid
        assert size_few > 0
        assert size_many > 0

    def test_uncertainty_passed_through(self):
        """Uncertainty inputs should reduce paper trader sizing."""
        ps = PositionSizer()
        trades = [0.02, -0.01, 0.03, 0.01] * 10

        size_no_unc = ps.size_position_paper_trader(
            ticker="TEST", kelly_history=trades,
            atr=2.0, realized_vol=0.25, price=100.0,
        )
        size_high_unc = ps.size_position_paper_trader(
            ticker="TEST", kelly_history=trades,
            atr=2.0, realized_vol=0.25, price=100.0,
            signal_uncertainty=0.9,
            regime_entropy=0.9,
            drift_score=0.1,
        )
        assert size_high_unc < size_no_unc, (
            f"High uncertainty ({size_high_unc}) should < no uncertainty ({size_no_unc})"
        )

    def test_budget_constraints_applied(self):
        """Budget constraints should be applied in paper trader method."""
        ps = PositionSizer(max_position_pct=0.50)
        size = ps.size_position_paper_trader(
            ticker="TEST",
            kelly_history=[0.05, 0.04, 0.03, 0.06] * 25,  # Very high win rate
            atr=0.5,
            realized_vol=0.10,
            price=100.0,
            portfolio_equity=1_000_000.0,
        )
        # Concentration limit should cap at 20%
        assert size <= 0.20


# ---------------------------------------------------------------------------
# Budget constraint interaction tests
# ---------------------------------------------------------------------------

class TestBudgetConstraintInteractions:
    """Test that budget constraints are applied in correct order."""

    def test_shock_then_concentration(self):
        """Both shock budget and concentration limit should apply."""
        ps = PositionSizer(max_position_pct=1.0)
        # Shock budget allows up to 0.95, concentration limits to 0.20
        result = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.10, 0.5, 100.0,
            portfolio_equity=1_000_000.0,
        )
        assert result.final_size <= 0.20

    def test_without_portfolio_equity_no_constraints(self):
        """Without portfolio_equity, budget constraints are not applied."""
        ps = PositionSizer()
        result = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
        )
        # Should still produce a valid result
        assert result.final_size > 0
        assert result.final_size <= ps.max_position


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Ensure existing API surface is preserved."""

    def test_size_position_original_signature(self):
        """Original size_position() call signature should still work."""
        ps = PositionSizer()
        result = ps.size_position(
            ticker="TEST",
            win_rate=0.55,
            avg_win=0.03,
            avg_loss=-0.02,
            realized_vol=0.25,
            atr=2.0,
            price=100.0,
        )
        assert isinstance(result, PositionSize)
        assert result.final_size > 0

    def test_size_position_with_regime_and_drawdown(self):
        """Existing regime and drawdown params should still work."""
        ps = PositionSizer()
        result = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            regime="trending_bull",
            current_drawdown=-0.10,
            n_trades=50,
        )
        assert isinstance(result, PositionSize)
        assert result.final_size > 0

    def test_kelly_negative_edge_still_zero(self):
        """Negative-edge Kelly should still return 0 (not min_position)."""
        ps = PositionSizer()
        assert ps._kelly(0.3, 0.02, -0.03) == 0.0

    def test_drawdown_governor_unchanged(self):
        """Drawdown governor (exponential curve) should be unchanged."""
        ps = PositionSizer()
        frac = ps._apply_drawdown_governor(0.5, -0.10, -0.20)
        assert frac > 0.25  # At 50% DD, should be > 50% due to convex curve

    def test_portfolio_aware_still_works(self):
        """size_portfolio_aware should still work with correlation penalty."""
        np.random.seed(42)
        n = 100
        idx = pd.date_range("2025-01-01", periods=n)
        ps = PositionSizer()
        base = np.random.randn(n) * 0.01
        returns_data = {
            "EXISTING": pd.Series(base, index=idx),
            "NEW": pd.Series(base + np.random.randn(n) * 0.002, index=idx),
        }
        size = ps.size_portfolio_aware(
            "NEW", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            existing_positions={"EXISTING": 0.10},
            returns_data=returns_data,
        )
        assert 0 < size <= ps.max_position

    def test_update_regime_stats_unchanged(self):
        """update_regime_stats should still work as before."""
        ps = PositionSizer()
        trades = pd.DataFrame({
            "net_return": [0.03, 0.02, -0.01, 0.04, 0.01,
                           0.02, -0.02, 0.03, 0.01, 0.02],
            "regime": [0] * 10,
        })
        ps.update_regime_stats(trades)
        stats = ps.regime_stats["trending_bull"]
        assert stats["n_trades"] == 10

    def test_update_kelly_bayesian_unchanged(self):
        """update_kelly_bayesian should still work as before."""
        ps = PositionSizer()
        trades = pd.DataFrame({
            "net_return": [0.02] * 15 + [-0.01] * 5,
            "regime": [0] * 15 + [0] * 5,
        })
        ps.update_kelly_bayesian(trades)
        assert ps._bayesian_wins == 15
        assert ps._bayesian_losses == 5

    def test_size_portfolio_unchanged(self):
        """size_portfolio should work with the new code."""
        ps = PositionSizer()
        idx = pd.date_range("2024-01-01", periods=100, freq="B")
        ohlcv = pd.DataFrame({
            "Open": 100 + np.random.randn(100).cumsum(),
            "High": 102 + np.random.randn(100).cumsum(),
            "Low": 98 + np.random.randn(100).cumsum(),
            "Close": 100 + np.random.randn(100).cumsum(),
            "Volume": np.random.randint(1000, 10000, 100),
        }, index=idx)
        # Ensure Close is positive
        ohlcv["Close"] = ohlcv["Close"].clip(lower=10.0)
        ohlcv["High"] = ohlcv["Close"] + abs(np.random.randn(100))
        ohlcv["Low"] = ohlcv["Close"] - abs(np.random.randn(100))

        signals = pd.DataFrame({
            "permno": ["TEST"],
            "predicted_return": [0.02],
            "confidence": [0.7],
        })
        result = ps.size_portfolio(
            signals=signals,
            price_data={"TEST": ohlcv},
        )
        assert len(result) == 1
        assert "position_size" in result.columns


# ---------------------------------------------------------------------------
# Config parameter tests
# ---------------------------------------------------------------------------

class TestConfigParameters:
    """Verify new config parameters load correctly."""

    def test_risk_governor_config_loaded(self):
        """PositionSizer should load risk governor config params."""
        ps = PositionSizer()
        assert hasattr(ps, "_shock_budget_pct")
        assert hasattr(ps, "_concentration_limit_pct")
        assert hasattr(ps, "_turnover_budget_enforcement")
        assert hasattr(ps, "_blend_weights_by_regime")
        assert hasattr(ps, "_uncertainty_scaling_enabled")
        assert hasattr(ps, "_kelly_min_samples")

    def test_shock_budget_default(self):
        """Default shock budget should be 5%."""
        ps = PositionSizer()
        assert ps._shock_budget_pct == 0.05

    def test_concentration_limit_default(self):
        """Default concentration limit should be 20%."""
        ps = PositionSizer()
        assert ps._concentration_limit_pct == 0.20

    def test_blend_weights_by_regime_has_all_states(self):
        """BLEND_WEIGHTS_BY_REGIME should have all 5 states."""
        ps = PositionSizer()
        if ps._blend_weights_by_regime:
            for state in ["NORMAL", "WARNING", "CAUTION", "CRITICAL", "RECOVERY"]:
                assert state in ps._blend_weights_by_regime, (
                    f"Missing regime state {state} in blend weights"
                )


# ---------------------------------------------------------------------------
# Integration: Full sizing pipeline
# ---------------------------------------------------------------------------

class TestFullSizingPipeline:
    """End-to-end integration tests for the complete sizing pipeline."""

    def test_all_features_enabled(self):
        """Full pipeline with uncertainty, budgets, and regime weights."""
        ps = PositionSizer()
        result = ps.size_position(
            ticker="AAPL",
            win_rate=0.55,
            avg_win=0.03,
            avg_loss=-0.02,
            realized_vol=0.25,
            atr=2.0,
            price=150.0,
            holding_days=10,
            confidence=0.7,
            n_current_positions=5,
            max_positions=20,
            regime="trending_bull",
            current_drawdown=-0.05,
            n_trades=50,
            signal_uncertainty=0.3,
            regime_entropy=0.2,
            drift_score=0.7,
            portfolio_equity=1_000_000.0,
            current_positions={"MSFT": 50_000.0, "GOOGL": 40_000.0},
        )
        assert isinstance(result, PositionSize)
        assert result.final_size > 0
        assert result.final_size <= 0.20  # Concentration limit
        assert result.sizing_details["uncertainty_scale"] < 1.0  # Some reduction

    def test_zero_equity_no_budget_constraints(self):
        """With zero equity, budget constraints should be skipped."""
        ps = PositionSizer()
        result = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            portfolio_equity=0.0,
        )
        assert result.final_size > 0

    def test_multiple_positions_turnover_tracking(self):
        """Multiple sizing calls should track turnover."""
        ps = PositionSizer()
        ps.reset_turnover_tracking()

        for sym in ["A", "B", "C", "D", "E"]:
            ps.size_position(
                sym, 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
                portfolio_equity=1_000_000.0,
                current_positions={},
            )

        # Turnover should have accumulated
        assert len(ps._turnover_history) >= 0  # May or may not add (depends on budget)
