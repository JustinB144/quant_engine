"""
Tests for SPEC-W01: Wire structural state into execution simulator.

Verifies that:
  - compute_shock_vectors produces valid ShockVectors for every bar
  - Shock vectors contain BOCPD, jump, drift, and stress data
  - The backtester's execution model receives structural state params
  - Structural state conditioning increases costs during high-uncertainty periods
  - The no-trade gate blocks entries during extreme stress
  - Graceful degradation when structural data is unavailable
  - End-to-end integration through the Backtester.run() path
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from quant_engine.regime.shock_vector import (
    ShockVector,
    ShockVectorValidator,
    compute_shock_vectors,
)
from quant_engine.backtest.execution import ExecutionModel, ExecutionFill


# ── Fixtures ────────────────────────────────────────────────────────────


def _make_ohlcv(n: int = 200, seed: int = 42, ticker: str = "TEST") -> pd.DataFrame:
    """Create synthetic OHLCV data with realistic price dynamics."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-03", periods=n)
    close = 100.0 + np.cumsum(rng.normal(0.0005, 0.015, n))
    close = np.maximum(close, 1.0)
    high = close * (1 + rng.uniform(0.001, 0.025, n))
    low = close * (1 - rng.uniform(0.001, 0.025, n))
    opn = close * (1 + rng.normal(0, 0.005, n))
    vol = rng.integers(500_000, 5_000_000, n).astype(float)
    df = pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )
    df.attrs["ticker"] = ticker
    df.attrs["permno"] = ticker
    return df


def _make_ohlcv_with_shock(
    n: int = 200, shock_idx: int = 150, seed: int = 42, ticker: str = "TEST",
) -> pd.DataFrame:
    """Create OHLCV data with a large price shock at shock_idx."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-03", periods=n)
    returns = rng.normal(0.0005, 0.015, n)
    # Inject a large shock
    returns[shock_idx] = -0.08  # 8% drop
    close = 100.0 + np.cumsum(returns)
    close = np.maximum(close, 1.0)
    high = close * (1 + rng.uniform(0.001, 0.025, n))
    low = close * (1 - rng.uniform(0.001, 0.025, n))
    opn = close * (1 + rng.normal(0, 0.005, n))
    vol = rng.integers(500_000, 5_000_000, n).astype(float)
    # Volume spike near the shock
    vol[shock_idx - 2 : shock_idx + 3] *= 3
    df = pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )
    df.attrs["ticker"] = ticker
    df.attrs["permno"] = ticker
    return df


# ── compute_shock_vectors Unit Tests ────────────────────────────────────


class TestComputeShockVectors:
    """Tests for the compute_shock_vectors batch function."""

    def test_basic_output_structure(self):
        """Should return one ShockVector per bar, keyed by date index."""
        ohlcv = _make_ohlcv(n=100)
        result = compute_shock_vectors(ohlcv=ohlcv, ticker="TEST")

        assert isinstance(result, dict)
        assert len(result) == 100
        # Keys should match OHLCV index
        for dt in ohlcv.index:
            assert dt in result
            assert isinstance(result[dt], ShockVector)

    def test_all_vectors_are_valid(self):
        """Every ShockVector produced should pass validation."""
        ohlcv = _make_ohlcv(n=150)
        result = compute_shock_vectors(ohlcv=ohlcv, ticker="VALID")

        for dt, sv in result.items():
            is_valid, errors = ShockVectorValidator.validate(sv)
            assert is_valid, f"Invalid ShockVector at {dt}: {errors}"

    def test_ticker_propagated(self):
        """Ticker identifier should be set on all vectors."""
        ohlcv = _make_ohlcv(n=50)
        result = compute_shock_vectors(ohlcv=ohlcv, ticker="AAPL")

        for sv in result.values():
            assert sv.ticker == "AAPL"

    def test_bocpd_changepoint_probs_populated(self):
        """BOCPD changepoint probs should be non-zero for at least some bars."""
        ohlcv = _make_ohlcv(n=200)
        result = compute_shock_vectors(ohlcv=ohlcv, ticker="TEST")

        cp_probs = [sv.bocpd_changepoint_prob for sv in result.values()]
        # Not all should be zero (BOCPD should detect some changepoints)
        assert any(p > 0 for p in cp_probs), "BOCPD should detect some changepoints"

    def test_jump_detection_flags_shock(self):
        """A large price shock should trigger jump detection."""
        ohlcv = _make_ohlcv_with_shock(n=200, shock_idx=150)
        result = compute_shock_vectors(ohlcv=ohlcv, ticker="TEST")

        # The shock bar (idx 150) should have jump_detected=True
        shock_date = ohlcv.index[150]
        sv = result[shock_date]
        assert sv.jump_detected is True
        assert sv.jump_magnitude < 0  # Negative jump

    def test_drift_score_in_structural_features(self):
        """drift_score should be present in structural_features."""
        ohlcv = _make_ohlcv(n=100)
        result = compute_shock_vectors(ohlcv=ohlcv, ticker="TEST")

        for sv in result.values():
            assert "drift_score" in sv.structural_features
            assert 0.0 <= sv.structural_features["drift_score"] <= 1.0

    def test_systemic_stress_in_structural_features(self):
        """systemic_stress should be present in structural_features."""
        ohlcv = _make_ohlcv(n=100)
        result = compute_shock_vectors(ohlcv=ohlcv, ticker="TEST")

        for sv in result.values():
            assert "systemic_stress" in sv.structural_features
            assert 0.0 <= sv.structural_features["systemic_stress"] <= 1.0

    def test_regime_series_used(self):
        """When regime_series is provided, it should be reflected in vectors."""
        ohlcv = _make_ohlcv(n=100)
        regime = pd.Series(
            np.array([0] * 40 + [1] * 30 + [3] * 30),
            index=ohlcv.index,
            dtype=int,
        )
        result = compute_shock_vectors(
            ohlcv=ohlcv, regime_series=regime, ticker="TEST",
        )

        # First 40 bars should have regime 0
        for dt in ohlcv.index[:40]:
            assert result[dt].hmm_regime == 0
        # Bars 40-69 should have regime 1
        for dt in ohlcv.index[40:70]:
            assert result[dt].hmm_regime == 1
        # Bars 70-99 should have regime 3
        for dt in ohlcv.index[70:]:
            assert result[dt].hmm_regime == 3

    def test_confidence_series_used(self):
        """When confidence_series is provided, uncertainty = 1 - confidence."""
        ohlcv = _make_ohlcv(n=50)
        confidence = pd.Series(0.8, index=ohlcv.index)
        result = compute_shock_vectors(
            ohlcv=ohlcv, confidence_series=confidence, ticker="TEST",
        )

        for sv in result.values():
            assert abs(sv.hmm_confidence - 0.8) < 1e-6
            assert abs(sv.hmm_uncertainty - 0.2) < 1e-6

    def test_empty_ohlcv_returns_empty(self):
        """Empty OHLCV should return empty dict."""
        ohlcv = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        result = compute_shock_vectors(ohlcv=ohlcv, ticker="EMPTY")
        assert result == {}

    def test_missing_close_column_returns_empty(self):
        """Missing 'Close' column should return empty dict gracefully."""
        ohlcv = pd.DataFrame(
            {"Open": [100], "High": [101], "Low": [99], "Volume": [1e6]},
            index=pd.bdate_range("2023-01-03", periods=1),
        )
        result = compute_shock_vectors(ohlcv=ohlcv, ticker="NOCL")
        assert result == {}

    def test_none_ohlcv_returns_empty(self):
        """None ohlcv should return empty dict."""
        result = compute_shock_vectors(ohlcv=None, ticker="NONE")
        assert result == {}

    def test_short_series_no_crash(self):
        """Very short series (< vol_lookback) should not crash."""
        ohlcv = _make_ohlcv(n=5)
        result = compute_shock_vectors(ohlcv=ohlcv, ticker="SHORT")
        assert len(result) == 5
        for sv in result.values():
            is_valid, _ = ShockVectorValidator.validate(sv)
            assert is_valid

    def test_no_regime_defaults_to_zero(self):
        """Without regime_series, all bars default to regime 0."""
        ohlcv = _make_ohlcv(n=30)
        result = compute_shock_vectors(ohlcv=ohlcv, ticker="TEST")
        for sv in result.values():
            assert sv.hmm_regime == 0

    def test_no_confidence_defaults_to_half(self):
        """Without confidence_series, confidence defaults to 0.5."""
        ohlcv = _make_ohlcv(n=30)
        result = compute_shock_vectors(ohlcv=ohlcv, ticker="TEST")
        for sv in result.values():
            assert sv.hmm_confidence == 0.5
            assert sv.hmm_uncertainty == 0.5


# ── Execution Model Structural State Tests ──────────────────────────────


class TestExecutionModelStructuralState:
    """Verify ExecutionModel properly uses structural state parameters."""

    def _make_model(self) -> ExecutionModel:
        return ExecutionModel(
            spread_bps=3.0,
            impact_coefficient_bps=25.0,
            max_participation_rate=0.05,
            dynamic_costs=True,
            structural_stress_enabled=True,
        )

    def test_structural_params_increase_costs(self):
        """Costs should be higher with structural stress than without."""
        model = self._make_model()

        # Baseline: no structural state
        baseline = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=2_000_000.0,
            desired_notional_usd=100_000.0,
            force_full=True,
            realized_vol=0.20,
            overnight_gap=0.005,
            intraday_range=0.02,
        )

        # Stressed: high changepoint probability + high uncertainty
        stressed = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=2_000_000.0,
            desired_notional_usd=100_000.0,
            force_full=True,
            realized_vol=0.20,
            overnight_gap=0.005,
            intraday_range=0.02,
            break_probability=0.8,
            structure_uncertainty=0.9,
            systemic_stress=0.85,
        )

        assert stressed.spread_bps > baseline.spread_bps
        assert stressed.impact_bps > baseline.impact_bps
        assert stressed.structural_mult > 1.0

    def test_drift_score_reduces_costs(self):
        """High drift score (strong trend) should reduce costs."""
        model = self._make_model()

        no_drift = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=2_000_000.0,
            desired_notional_usd=100_000.0,
            force_full=True,
            realized_vol=0.20,
            overnight_gap=0.005,
            intraday_range=0.02,
            drift_score=0.0,
        )

        high_drift = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=2_000_000.0,
            desired_notional_usd=100_000.0,
            force_full=True,
            realized_vol=0.20,
            overnight_gap=0.005,
            intraday_range=0.02,
            drift_score=0.9,
        )

        # Drift reduces the composite multiplier
        assert high_drift.structural_mult < no_drift.structural_mult

    def test_no_trade_gate_blocks_entry(self):
        """Extreme systemic stress should block entry orders."""
        model = self._make_model()

        fill = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=2_000_000.0,
            desired_notional_usd=100_000.0,
            urgency_type="entry",
            systemic_stress=0.98,  # Above the 0.95 threshold
        )

        assert fill.fill_ratio == 0.0
        assert fill.no_trade_blocked is True

    def test_no_trade_gate_allows_exit(self):
        """Extreme stress should NOT block exit orders (exits are forced)."""
        model = self._make_model()

        fill = model.simulate(
            side="sell",
            reference_price=100.0,
            daily_volume=2_000_000.0,
            desired_notional_usd=100_000.0,
            urgency_type="exit",
            systemic_stress=0.98,
        )

        # Exit should still execute (no_trade gate only blocks entries)
        assert fill.fill_ratio > 0.0

    def test_structural_disabled_ignores_params(self):
        """When structural stress is disabled, params should be ignored."""
        model = ExecutionModel(
            spread_bps=3.0,
            impact_coefficient_bps=25.0,
            dynamic_costs=True,
            structural_stress_enabled=False,
        )

        fill_with = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=2_000_000.0,
            desired_notional_usd=100_000.0,
            force_full=True,
            realized_vol=0.20,
            break_probability=0.9,
            structure_uncertainty=0.9,
            systemic_stress=0.9,
        )

        fill_without = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=2_000_000.0,
            desired_notional_usd=100_000.0,
            force_full=True,
            realized_vol=0.20,
        )

        assert fill_with.structural_mult == 1.0
        assert abs(fill_with.spread_bps - fill_without.spread_bps) < 1e-6

    def test_none_structural_params_are_safe(self):
        """All-None structural params should produce structural_mult = 1.0."""
        model = self._make_model()

        fill = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=2_000_000.0,
            desired_notional_usd=100_000.0,
            force_full=True,
            realized_vol=0.20,
            break_probability=None,
            structure_uncertainty=None,
            drift_score=None,
            systemic_stress=None,
        )

        assert fill.structural_mult == 1.0

    def test_cost_details_contain_structural_breakdown(self):
        """Cost details should contain the structural multiplier breakdown."""
        model = self._make_model()

        fill = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=2_000_000.0,
            desired_notional_usd=100_000.0,
            force_full=True,
            realized_vol=0.20,
            break_probability=0.3,
            structure_uncertainty=0.6,
            drift_score=0.2,
            systemic_stress=0.4,
        )

        details = fill.cost_details
        assert "break_prob_mult" in details
        assert "uncertainty_mult" in details
        assert "drift_mult" in details
        assert "stress_mult" in details
        assert "composite_structural_mult" in details
        assert details["composite_structural_mult"] > 1.0


# ── Backtester Integration Tests ────────────────────────────────────────


class TestBacktesterStructuralWiring:
    """Test that the Backtester properly wires structural state to execution."""

    def _make_backtest_data(self, n: int = 100):
        """Build minimal predictions + price_data for a backtest."""
        permno = "10001"
        ohlcv = _make_ohlcv(n=n, ticker=permno)
        ohlcv.attrs["ticker"] = permno
        ohlcv.attrs["permno"] = permno

        # Build MultiIndex predictions
        dates = ohlcv.index
        pred_idx = pd.MultiIndex.from_arrays(
            [[permno] * n, dates],
            names=["permno", "date"],
        )

        rng = np.random.default_rng(42)
        preds = pd.DataFrame(
            {
                "predicted_return": rng.normal(0.005, 0.01, n),
                "confidence": rng.uniform(0.5, 0.9, n),
                "regime": np.zeros(n, dtype=int),
            },
            index=pred_idx,
        )

        # Ensure some signals pass the threshold
        preds.loc[preds["predicted_return"] > 0.003, "predicted_return"] = 0.02

        price_data = {permno: ohlcv}
        return preds, price_data

    @patch("quant_engine.validation.preconditions.enforce_preconditions")
    def test_shock_vectors_precomputed_in_run(self, mock_precond):
        """Backtester.run() should pre-compute shock vectors."""
        from quant_engine.backtest.engine import Backtester

        preds, price_data = self._make_backtest_data(n=100)
        bt = Backtester(
            entry_threshold=0.01,
            confidence_threshold=0.4,
            holding_days=5,
            max_positions=3,
        )
        bt.run(preds, price_data, verbose=False)

        # Shock vectors should have been populated
        assert len(bt._shock_vectors) > 0
        for key, sv in bt._shock_vectors.items():
            assert isinstance(key, tuple)
            assert len(key) == 2  # (ticker, date)
            assert isinstance(sv, ShockVector)

    @patch("quant_engine.validation.preconditions.enforce_preconditions")
    def test_structural_state_passed_to_simulate(self, mock_precond):
        """The execution model's simulate() should receive structural params."""
        from quant_engine.backtest.engine import Backtester

        preds, price_data = self._make_backtest_data(n=100)
        bt = Backtester(
            entry_threshold=0.01,
            confidence_threshold=0.4,
            holding_days=5,
            max_positions=3,
        )

        # Wrap simulate to capture calls
        original_simulate = bt.execution_model.simulate
        calls = []

        def tracking_simulate(**kwargs):
            calls.append(kwargs)
            return original_simulate(**kwargs)

        bt.execution_model.simulate = tracking_simulate
        bt.run(preds, price_data, verbose=False)

        # At least one simulate() call should have structural params
        if calls:
            has_structural = any(
                c.get("break_probability") is not None
                or c.get("structure_uncertainty") is not None
                for c in calls
            )
            assert has_structural, (
                "No simulate() calls received structural state params. "
                "SPEC-W01 wiring is not active."
            )

    @patch("quant_engine.validation.preconditions.enforce_preconditions")
    def test_structural_disabled_skips_computation(self, mock_precond):
        """When EXEC_STRUCTURAL_STRESS_ENABLED=False, skip shock vector computation."""
        from quant_engine.backtest.engine import Backtester

        preds, price_data = self._make_backtest_data(n=100)

        with patch("quant_engine.backtest.engine.EXEC_STRUCTURAL_STRESS_ENABLED", False):
            bt = Backtester(
                entry_threshold=0.01,
                confidence_threshold=0.4,
                holding_days=5,
                max_positions=3,
            )
            bt.run(preds, price_data, verbose=False)

        # Shock vectors dict should be empty
        assert len(bt._shock_vectors) == 0

    @patch("quant_engine.validation.preconditions.enforce_preconditions")
    def test_graceful_on_shock_computation_failure(self, mock_precond):
        """If shock vector computation fails for a ticker, backtest should continue."""
        from quant_engine.backtest.engine import Backtester

        preds, price_data = self._make_backtest_data(n=100)
        bt = Backtester(
            entry_threshold=0.01,
            confidence_threshold=0.4,
            holding_days=5,
            max_positions=3,
        )

        # Patch compute_shock_vectors to raise
        with patch(
            "quant_engine.backtest.engine.compute_shock_vectors",
            side_effect=RuntimeError("BOCPD exploded"),
        ):
            # Should not raise — graceful degradation
            result = bt.run(preds, price_data, verbose=False)
            assert result is not None


# ── Comparative Cost Tests ──────────────────────────────────────────────


class TestStructuralStateCostImpact:
    """Verify that structural state conditioning produces meaningful cost differences."""

    def test_high_uncertainty_period_higher_costs(self):
        """Periods with high uncertainty should have higher execution costs."""
        model = ExecutionModel(
            spread_bps=3.0,
            impact_coefficient_bps=25.0,
            dynamic_costs=True,
            structural_stress_enabled=True,
        )

        base_params = dict(
            side="buy",
            reference_price=100.0,
            daily_volume=2_000_000.0,
            desired_notional_usd=100_000.0,
            force_full=True,
            realized_vol=0.20,
            overnight_gap=0.005,
            intraday_range=0.02,
        )

        # Calm market: low uncertainty, low changepoint prob
        calm = model.simulate(
            **base_params,
            break_probability=0.02,
            structure_uncertainty=0.1,
            drift_score=0.5,
            systemic_stress=0.1,
        )

        # Transition period: high uncertainty, moderate changepoint
        transition = model.simulate(
            **base_params,
            break_probability=0.30,
            structure_uncertainty=0.8,
            drift_score=0.1,
            systemic_stress=0.6,
        )

        # Crisis: extreme stress
        crisis = model.simulate(
            **base_params,
            break_probability=0.70,
            structure_uncertainty=0.95,
            drift_score=0.0,
            systemic_stress=0.90,
        )

        # Costs should increase: calm < transition < crisis
        calm_total = calm.cost_details.get("total_bps", 0)
        transition_total = transition.cost_details.get("total_bps", 0)
        crisis_total = crisis.cost_details.get("total_bps", 0)

        assert calm_total < transition_total, (
            f"Calm costs ({calm_total:.2f}) should be < transition ({transition_total:.2f})"
        )
        assert transition_total < crisis_total, (
            f"Transition costs ({transition_total:.2f}) should be < crisis ({crisis_total:.2f})"
        )

    def test_structural_mult_bounded(self):
        """Composite structural multiplier should be clipped to [0.70, 3.0]."""
        model = ExecutionModel(
            spread_bps=3.0,
            impact_coefficient_bps=25.0,
            dynamic_costs=True,
            structural_stress_enabled=True,
        )

        # Extreme values
        fill = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=2_000_000.0,
            desired_notional_usd=100_000.0,
            force_full=True,
            realized_vol=0.20,
            break_probability=1.0,
            structure_uncertainty=1.0,
            drift_score=0.0,
            systemic_stress=1.0,
        )

        assert fill.structural_mult <= 3.0
        assert fill.structural_mult >= 0.70


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
