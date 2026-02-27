"""Tests for SPEC-P02: Wire uncertainty into Bayesian Kelly sizing.

Verifies:
  T1: regime_uncertainty parameter reduces Kelly-derived position sizes
  T2: High uncertainty (entropy ≈ 1.0) yields ~80-85% of normal Kelly
      per REGIME_UNCERTAINTY_SIZING_MAP defaults
  T3: Zero regime_uncertainty leaves sizes unchanged (backward compat)
  T4: get_bayesian_kelly() also applies uncertainty scaling
  T5: Backtest engine risk-managed mode threads regime_uncertainty
  T6: Paper trader interface forwards regime_uncertainty
  T7: sizing_details includes regime_uncertainty_mult
  T8: UncertaintyGate multiplier is consistent between sizer and gate
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from quant_engine.risk.position_sizer import PositionSizer, PositionSize
from quant_engine.regime.uncertainty_gate import UncertaintyGate


# ---------------------------------------------------------------------------
# T1: regime_uncertainty reduces Kelly-derived sizes
# ---------------------------------------------------------------------------

class TestRegimeUncertaintyReducesKelly:
    """T1: When regime_uncertainty > 0, Kelly components are scaled down."""

    def test_high_uncertainty_reduces_final_size(self):
        """High regime_uncertainty should produce smaller final_size."""
        ps = PositionSizer()
        base = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            regime_uncertainty=0.0,
        )
        uncertain = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            regime_uncertainty=0.9,
        )
        assert uncertain.final_size < base.final_size, (
            f"High uncertainty ({uncertain.final_size:.4f}) should be "
            f"< baseline ({base.final_size:.4f})"
        )

    def test_half_kelly_reduced_by_uncertainty(self):
        """half_kelly should be directly reduced by the uncertainty multiplier."""
        ps = PositionSizer()
        base = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            regime_uncertainty=0.0,
        )
        uncertain = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            regime_uncertainty=0.9,
        )
        assert uncertain.half_kelly < base.half_kelly, (
            f"Uncertain half_kelly ({uncertain.half_kelly:.4f}) should be "
            f"< base ({base.half_kelly:.4f})"
        )

    def test_raw_kelly_reduced_by_uncertainty(self):
        """raw_kelly should also be reduced by the uncertainty multiplier."""
        ps = PositionSizer()
        base = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            regime_uncertainty=0.0,
        )
        uncertain = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            regime_uncertainty=0.9,
        )
        assert uncertain.raw_kelly < base.raw_kelly, (
            f"Uncertain raw_kelly ({uncertain.raw_kelly:.4f}) should be "
            f"< base ({base.raw_kelly:.4f})"
        )

    def test_vol_and_atr_unaffected_by_regime_uncertainty(self):
        """Vol-scaled and ATR-based components should NOT be affected."""
        ps = PositionSizer()
        base = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            regime_uncertainty=0.0,
        )
        uncertain = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            regime_uncertainty=0.9,
        )
        assert uncertain.vol_scaled == base.vol_scaled
        assert uncertain.atr_based == base.atr_based

    def test_monotonic_reduction_with_increasing_uncertainty(self):
        """Higher regime_uncertainty should produce equal or smaller sizes."""
        ps = PositionSizer()
        prev_size = float("inf")
        for unc in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
            result = ps.size_position(
                "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
                regime_uncertainty=unc,
            )
            assert result.half_kelly <= prev_size + 1e-9, (
                f"At uncertainty={unc}, half_kelly={result.half_kelly:.4f} "
                f"> prev={prev_size:.4f}"
            )
            prev_size = result.half_kelly


# ---------------------------------------------------------------------------
# T2: High uncertainty yields ~80-85% of normal (per sizing map)
# ---------------------------------------------------------------------------

class TestUncertaintyMapValues:
    """T2: Verify 80-85% reduction range per REGIME_UNCERTAINTY_SIZING_MAP."""

    def test_max_uncertainty_yields_85pct(self):
        """At entropy=1.0, the sizing map gives multiplier=0.85."""
        gate = UncertaintyGate()
        mult = gate.compute_size_multiplier(1.0)
        assert mult == pytest.approx(0.85, abs=0.01), (
            f"At max entropy, multiplier should be ~0.85, got {mult}"
        )

    def test_mid_uncertainty_yields_95pct(self):
        """At entropy=0.5, the sizing map gives multiplier=0.95."""
        gate = UncertaintyGate()
        mult = gate.compute_size_multiplier(0.5)
        assert mult == pytest.approx(0.95, abs=0.01), (
            f"At mid entropy, multiplier should be ~0.95, got {mult}"
        )

    def test_kelly_at_max_uncertainty_is_85pct_of_base(self):
        """Kelly component at max uncertainty should be ~85% of baseline."""
        ps = PositionSizer()
        base = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            regime_uncertainty=0.0,
        )
        max_unc = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            regime_uncertainty=1.0,
        )
        if base.half_kelly > 0:
            ratio = max_unc.half_kelly / base.half_kelly
            assert 0.80 <= ratio <= 0.90, (
                f"Expected Kelly ratio ~0.85 at max uncertainty, got {ratio:.3f}"
            )

    def test_floor_multiplier_respected(self):
        """Multiplier should never go below REGIME_UNCERTAINTY_MIN_MULTIPLIER."""
        from quant_engine.config import REGIME_UNCERTAINTY_MIN_MULTIPLIER
        gate = UncertaintyGate()
        # Even with extreme entropy, multiplier should be >= floor
        for unc in [0.9, 0.95, 1.0, 1.5]:
            mult = gate.compute_size_multiplier(unc)
            assert mult >= REGIME_UNCERTAINTY_MIN_MULTIPLIER, (
                f"Multiplier {mult} below floor {REGIME_UNCERTAINTY_MIN_MULTIPLIER} "
                f"at uncertainty={unc}"
            )


# ---------------------------------------------------------------------------
# T3: Zero regime_uncertainty leaves sizes unchanged (backward compat)
# ---------------------------------------------------------------------------

class TestZeroUncertaintyBackwardCompat:
    """T3: regime_uncertainty=0 produces identical results to no parameter."""

    def test_default_zero_matches_no_param(self):
        """Calling without regime_uncertainty should match regime_uncertainty=0."""
        ps = PositionSizer()
        base = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
        )
        explicit_zero = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            regime_uncertainty=0.0,
        )
        assert base.final_size == explicit_zero.final_size
        assert base.half_kelly == explicit_zero.half_kelly
        assert base.raw_kelly == explicit_zero.raw_kelly

    def test_multiplier_is_one_at_zero_uncertainty(self):
        """Sizing details should show regime_uncertainty_mult=1.0 at zero."""
        ps = PositionSizer()
        result = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            regime_uncertainty=0.0,
        )
        assert result.sizing_details["regime_uncertainty_mult"] == 1.0

    def test_existing_api_unchanged(self):
        """Existing callers without regime_uncertainty should work identically."""
        ps = PositionSizer()
        result = ps.size_position(
            ticker="TEST",
            win_rate=0.55,
            avg_win=0.03,
            avg_loss=-0.02,
            realized_vol=0.25,
            atr=2.0,
            price=100.0,
            holding_days=10,
            confidence=0.7,
            n_current_positions=3,
            max_positions=20,
            regime="trending_bull",
            current_drawdown=-0.05,
        )
        assert isinstance(result, PositionSize)
        assert result.final_size > 0


# ---------------------------------------------------------------------------
# T4: get_bayesian_kelly() applies uncertainty scaling
# ---------------------------------------------------------------------------

class TestBayesianKellyUncertainty:
    """T4: get_bayesian_kelly() receives and applies regime_uncertainty."""

    def _make_trades(self):
        """Create 20 trades: 12 wins, 8 losses → 60% win rate."""
        return pd.DataFrame({
            "net_return": [0.02] * 12 + [-0.01] * 8,
            "regime": [0] * 20,
        })

    def test_bayesian_kelly_with_zero_uncertainty(self):
        """Zero uncertainty should not change the Bayesian Kelly result."""
        ps = PositionSizer(max_position_pct=1.0)
        ps.update_kelly_bayesian(self._make_trades())
        base = ps.get_bayesian_kelly(0.02, -0.01, regime=None)
        with_zero = ps.get_bayesian_kelly(
            0.02, -0.01, regime=None, regime_uncertainty=0.0,
        )
        assert base == with_zero

    def test_bayesian_kelly_reduced_by_uncertainty(self):
        """High uncertainty should reduce the Bayesian Kelly fraction."""
        ps = PositionSizer(max_position_pct=1.0)
        ps.update_kelly_bayesian(self._make_trades())
        base = ps.get_bayesian_kelly(0.02, -0.01, regime=None)
        uncertain = ps.get_bayesian_kelly(
            0.02, -0.01, regime=None, regime_uncertainty=0.9,
        )
        assert uncertain < base, (
            f"Uncertain Kelly ({uncertain:.4f}) should be < base ({base:.4f})"
        )

    def test_regime_specific_bayesian_with_uncertainty(self):
        """Regime-specific Bayesian Kelly should also be scaled by uncertainty."""
        ps = PositionSizer(max_position_pct=1.0)
        # 15 trades in regime 0 (above min_samples=10)
        trades = pd.DataFrame({
            "net_return": [0.02] * 10 + [-0.01] * 5,
            "regime": [0] * 15,
        })
        ps.update_kelly_bayesian(trades)
        base = ps.get_bayesian_kelly(0.02, -0.01, regime=0)
        uncertain = ps.get_bayesian_kelly(
            0.02, -0.01, regime=0, regime_uncertainty=0.8,
        )
        assert uncertain < base, (
            f"Regime-specific uncertain ({uncertain:.4f}) should be < base ({base:.4f})"
        )

    def test_insufficient_data_with_uncertainty(self):
        """Even with insufficient data (prior mode), uncertainty should scale."""
        ps = PositionSizer(max_position_pct=1.0)
        # No trades — uses prior
        base = ps.get_bayesian_kelly(0.02, -0.01, regime=None)
        uncertain = ps.get_bayesian_kelly(
            0.02, -0.01, regime=None, regime_uncertainty=0.9,
        )
        if base > 0:
            assert uncertain < base
        else:
            assert uncertain == 0.0


# ---------------------------------------------------------------------------
# T5: Backtest engine threads regime_uncertainty (integration)
# ---------------------------------------------------------------------------

class TestBacktestEngineThreading:
    """T5: Verify backtest engine passes regime_uncertainty to position sizer."""

    def test_position_sizer_has_uncertainty_gate(self):
        """PositionSizer should have _uncertainty_gate attribute."""
        ps = PositionSizer()
        assert hasattr(ps, "_uncertainty_gate")
        assert isinstance(ps._uncertainty_gate, UncertaintyGate)

    def test_backtester_init_risk_components(self):
        """Backtester should initialize _position_sizer with uncertainty gate."""
        from quant_engine.backtest.engine import Backtester
        bt = Backtester(use_risk_management=True)
        bt._init_risk_components()
        assert hasattr(bt._position_sizer, "_uncertainty_gate")
        assert isinstance(bt._position_sizer._uncertainty_gate, UncertaintyGate)


# ---------------------------------------------------------------------------
# T6: Paper trader interface forwards regime_uncertainty
# ---------------------------------------------------------------------------

class TestPaperTraderForwarding:
    """T6: size_position_paper_trader() accepts and forwards regime_uncertainty."""

    def test_paper_trader_accepts_regime_uncertainty(self):
        """Paper trader sizing should accept regime_uncertainty kwarg."""
        ps = PositionSizer()
        size = ps.size_position_paper_trader(
            ticker="TEST",
            kelly_history=[0.02, -0.01, 0.03, 0.01] * 10,
            atr=2.0,
            realized_vol=0.25,
            price=100.0,
            regime_uncertainty=0.8,
        )
        assert isinstance(size, float)
        assert size > 0

    def test_paper_trader_uncertainty_reduces_size(self):
        """High regime_uncertainty should reduce paper trader position size."""
        ps = PositionSizer()
        history = [0.02, -0.01, 0.03, 0.01] * 10

        size_no_unc = ps.size_position_paper_trader(
            ticker="TEST", kelly_history=history,
            atr=2.0, realized_vol=0.25, price=100.0,
            regime_uncertainty=0.0,
        )
        size_high_unc = ps.size_position_paper_trader(
            ticker="TEST", kelly_history=history,
            atr=2.0, realized_vol=0.25, price=100.0,
            regime_uncertainty=0.9,
        )
        assert size_high_unc < size_no_unc, (
            f"High uncertainty ({size_high_unc:.4f}) should < "
            f"no uncertainty ({size_no_unc:.4f})"
        )


# ---------------------------------------------------------------------------
# T7: sizing_details includes regime_uncertainty fields
# ---------------------------------------------------------------------------

class TestSizingDetailsFields:
    """T7: sizing_details dict has regime_uncertainty and regime_uncertainty_mult."""

    def test_sizing_details_has_uncertainty_fields(self):
        """sizing_details should include regime_uncertainty and mult."""
        ps = PositionSizer()
        result = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            regime_uncertainty=0.7,
        )
        assert "regime_uncertainty" in result.sizing_details
        assert "regime_uncertainty_mult" in result.sizing_details
        assert result.sizing_details["regime_uncertainty"] == 0.7
        assert 0.80 <= result.sizing_details["regime_uncertainty_mult"] <= 1.0

    def test_details_mult_matches_gate(self):
        """The recorded multiplier should match UncertaintyGate output."""
        ps = PositionSizer()
        gate = UncertaintyGate()
        for unc in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
            result = ps.size_position(
                "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
                regime_uncertainty=unc,
            )
            expected = gate.compute_size_multiplier(unc) if unc > 0 else 1.0
            assert result.sizing_details["regime_uncertainty_mult"] == pytest.approx(
                expected, rel=1e-6,
            ), f"Mismatch at uncertainty={unc}"


# ---------------------------------------------------------------------------
# T8: Consistency between sizer and standalone gate
# ---------------------------------------------------------------------------

class TestSizerGateConsistency:
    """T8: Multiplier from PositionSizer matches standalone UncertaintyGate."""

    def test_kelly_ratio_matches_gate_multiplier(self):
        """The ratio of uncertain/base Kelly should match the gate multiplier."""
        ps = PositionSizer()
        gate = UncertaintyGate()

        for unc in [0.3, 0.5, 0.7, 0.9, 1.0]:
            base = ps.size_position(
                "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
                regime_uncertainty=0.0,
            )
            uncertain = ps.size_position(
                "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
                regime_uncertainty=unc,
            )
            if base.half_kelly > 0:
                ratio = uncertain.half_kelly / base.half_kelly
                expected_mult = gate.compute_size_multiplier(unc)
                assert ratio == pytest.approx(expected_mult, rel=1e-6), (
                    f"At uncertainty={unc}: kelly ratio={ratio:.4f} != "
                    f"gate mult={expected_mult:.4f}"
                )


# ---------------------------------------------------------------------------
# Integration: Full pipeline with SPEC-P02 + existing features
# ---------------------------------------------------------------------------

class TestP02Integration:
    """End-to-end tests combining SPEC-P02 with other sizing features."""

    def test_uncertainty_with_regime_stats(self):
        """Regime uncertainty should work alongside regime-conditional stats."""
        ps = PositionSizer()
        result = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            regime="trending_bull",
            regime_uncertainty=0.8,
        )
        assert isinstance(result, PositionSize)
        assert result.final_size > 0
        assert result.sizing_details["regime_uncertainty_mult"] < 1.0

    def test_uncertainty_with_drawdown(self):
        """Uncertainty + drawdown should compound to reduce Kelly further."""
        ps = PositionSizer()
        dd_only = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            current_drawdown=-0.10,
            regime_uncertainty=0.0,
        )
        both = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            current_drawdown=-0.10,
            regime_uncertainty=0.9,
        )
        assert both.half_kelly <= dd_only.half_kelly

    def test_uncertainty_with_signal_uncertainty_stacks(self):
        """SPEC-P02 regime_uncertainty and Spec-05 signal uncertainty stack."""
        ps = PositionSizer()
        regime_only = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            regime_uncertainty=0.8,
        )
        both = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            regime_uncertainty=0.8,
            signal_uncertainty=0.8,
            regime_entropy=0.8,
            drift_score=0.2,
        )
        assert both.final_size <= regime_only.final_size

    def test_uncertainty_with_budget_constraints(self):
        """Budget constraints should apply on top of uncertainty-scaled Kelly."""
        ps = PositionSizer(max_position_pct=0.50)
        result = ps.size_position(
            "TEST", 0.55, 0.03, -0.02, 0.25, 2.0, 100.0,
            regime_uncertainty=0.9,
            portfolio_equity=1_000_000.0,
        )
        # Concentration limit still applies
        assert result.final_size <= 0.20

    def test_negative_edge_with_uncertainty(self):
        """Negative-edge Kelly stays at 0 regardless of uncertainty."""
        ps = PositionSizer()
        result = ps.size_position(
            "TEST", 0.3, 0.02, -0.03, 0.25, 2.0, 100.0,
            regime_uncertainty=0.9,
        )
        assert result.raw_kelly == 0.0
        assert result.half_kelly == 0.0
        # Still gets vol + ATR contribution
        assert result.final_size > 0
