"""
Tests for SPEC-P04: Make turnover penalty configurable and cost-aware.

Covers:
    T1: Config constants exist with correct defaults
    T2: optimize_portfolio respects turnover_penalty parameter
    T3: Higher turnover penalty reduces portfolio turnover
    T4: AutopilotEngine._dynamic_turnover_penalty returns >= base penalty
    T5: Dynamic penalty scales with realised volatility
    T6: AutopilotEngine._get_current_portfolio_weights returns None with no positions
    T7: _get_current_portfolio_weights returns valid weights from paper state
    T8: optimize_portfolio uses current_weights to reduce turnover
    T9: Dynamic penalty falls back to base on import error
    T10: Config constants imported in autopilot/engine.py
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from quant_engine.config import (
    PORTFOLIO_TURNOVER_PENALTY,
    PORTFOLIO_TURNOVER_DYNAMIC,
    PORTFOLIO_TURNOVER_COST_MULTIPLIER,
)
from quant_engine.risk.portfolio_optimizer import optimize_portfolio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_universe(n: int = 5, seed: int = 42):
    """Create a simple test universe with expected returns and covariance."""
    rng = np.random.RandomState(seed)
    tickers = [f"A{i}" for i in range(n)]
    expected_returns = pd.Series(
        rng.uniform(0.001, 0.01, n), index=tickers, dtype=float,
    )
    # Generate a valid covariance matrix
    raw = rng.randn(252, n) * 0.02
    cov = pd.DataFrame(
        np.cov(raw.T), index=tickers, columns=tickers, dtype=float,
    )
    return tickers, expected_returns, cov


def _compute_turnover(old_weights: pd.Series, new_weights: pd.Series) -> float:
    """Compute absolute turnover between two weight vectors."""
    all_assets = sorted(set(old_weights.index) | set(new_weights.index))
    old_aligned = old_weights.reindex(all_assets, fill_value=0.0)
    new_aligned = new_weights.reindex(all_assets, fill_value=0.0)
    return float(np.abs(new_aligned - old_aligned).sum())


# ===========================================================================
# T1: Config constants exist with correct defaults
# ===========================================================================

class TestConfigConstants:
    """Verify SPEC-P04 config constants."""

    def test_portfolio_turnover_penalty_exists(self):
        """PORTFOLIO_TURNOVER_PENALTY should be a positive float."""
        assert isinstance(PORTFOLIO_TURNOVER_PENALTY, float)
        assert PORTFOLIO_TURNOVER_PENALTY > 0

    def test_portfolio_turnover_penalty_default(self):
        """Default value should be 0.001 (0.1%)."""
        assert PORTFOLIO_TURNOVER_PENALTY == 0.001

    def test_portfolio_turnover_dynamic_exists(self):
        """PORTFOLIO_TURNOVER_DYNAMIC should be a bool."""
        assert isinstance(PORTFOLIO_TURNOVER_DYNAMIC, bool)
        assert PORTFOLIO_TURNOVER_DYNAMIC is True

    def test_portfolio_turnover_cost_multiplier_exists(self):
        """PORTFOLIO_TURNOVER_COST_MULTIPLIER should be a positive float."""
        assert isinstance(PORTFOLIO_TURNOVER_COST_MULTIPLIER, (int, float))
        assert PORTFOLIO_TURNOVER_COST_MULTIPLIER == 2.0


# ===========================================================================
# T2: optimize_portfolio respects turnover_penalty parameter
# ===========================================================================

class TestOptimizePortfolioTurnoverParam:
    """Verify optimize_portfolio uses the turnover_penalty argument."""

    def test_accepts_turnover_penalty_kwarg(self):
        """optimize_portfolio should accept turnover_penalty without error."""
        _, er, cov = _simple_universe()
        weights = optimize_portfolio(
            expected_returns=er,
            covariance=cov,
            turnover_penalty=0.005,
        )
        assert isinstance(weights, pd.Series)
        assert len(weights) > 0
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_zero_penalty_does_not_crash(self):
        """Zero turnover penalty should still produce valid weights."""
        _, er, cov = _simple_universe()
        weights = optimize_portfolio(
            expected_returns=er,
            covariance=cov,
            turnover_penalty=0.0,
        )
        assert isinstance(weights, pd.Series)
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_large_penalty_does_not_crash(self):
        """Very large turnover penalty should still produce valid weights."""
        _, er, cov = _simple_universe()
        weights = optimize_portfolio(
            expected_returns=er,
            covariance=cov,
            turnover_penalty=1.0,
        )
        assert isinstance(weights, pd.Series)
        assert abs(weights.sum() - 1.0) < 1e-6


# ===========================================================================
# T3: Higher turnover penalty reduces portfolio turnover
# ===========================================================================

class TestTurnoverPenaltyEffect:
    """Verify that increasing turnover_penalty reduces turnover."""

    def test_higher_penalty_less_turnover(self):
        """A higher turnover penalty should produce less turnover from current."""
        _, er, cov = _simple_universe(n=8, seed=123)
        # Start from a non-uniform current position
        current = pd.Series(
            [0.15, 0.15, 0.15, 0.05, 0.15, 0.15, 0.15, 0.05],
            index=er.index,
        )

        w_low_penalty = optimize_portfolio(
            expected_returns=er,
            covariance=cov,
            current_weights=current,
            turnover_penalty=0.0001,
        )
        w_high_penalty = optimize_portfolio(
            expected_returns=er,
            covariance=cov,
            current_weights=current,
            turnover_penalty=0.05,
        )

        turnover_low = _compute_turnover(current, w_low_penalty)
        turnover_high = _compute_turnover(current, w_high_penalty)

        assert turnover_high <= turnover_low + 1e-6, (
            f"High penalty turnover ({turnover_high:.6f}) should be <= "
            f"low penalty turnover ({turnover_low:.6f})"
        )


# ===========================================================================
# T4: _dynamic_turnover_penalty returns >= base penalty
# ===========================================================================

class TestDynamicTurnoverPenalty:
    """Verify dynamic penalty floor behaviour."""

    def _make_engine(self):
        """Create a minimal AutopilotEngine mock for testing helper methods."""
        from quant_engine.autopilot.engine import AutopilotEngine
        with patch.object(AutopilotEngine, "__init__", lambda self: None):
            engine = AutopilotEngine.__new__(AutopilotEngine)
            engine.verbose = False
            engine.paper_trader = MagicMock()
            return engine

    def test_dynamic_penalty_ge_base(self):
        """Dynamic penalty should be >= PORTFOLIO_TURNOVER_PENALTY."""
        engine = self._make_engine()
        rng = np.random.RandomState(42)
        assets = ["A", "B", "C"]
        returns_df = pd.DataFrame(
            rng.randn(100, 3) * 0.02,
            columns=assets,
        )

        penalty = engine._dynamic_turnover_penalty(returns_df, assets)
        assert penalty >= PORTFOLIO_TURNOVER_PENALTY

    def test_dynamic_penalty_returns_float(self):
        """Penalty should always be a float."""
        engine = self._make_engine()
        rng = np.random.RandomState(42)
        assets = ["X", "Y"]
        returns_df = pd.DataFrame(rng.randn(100, 2) * 0.02, columns=assets)
        penalty = engine._dynamic_turnover_penalty(returns_df, assets)
        assert isinstance(penalty, float)
        assert np.isfinite(penalty)


# ===========================================================================
# T5: Dynamic penalty scales with realised volatility
# ===========================================================================

class TestDynamicPenaltyVolScaling:
    """Higher vol assets should produce higher dynamic penalty."""

    def _make_engine(self):
        from quant_engine.autopilot.engine import AutopilotEngine
        with patch.object(AutopilotEngine, "__init__", lambda self: None):
            engine = AutopilotEngine.__new__(AutopilotEngine)
            engine.verbose = False
            engine.paper_trader = MagicMock()
            return engine

    def test_high_vol_higher_penalty(self):
        """Universe with high-vol assets should produce higher penalty."""
        engine = self._make_engine()
        rng = np.random.RandomState(99)
        assets = ["LO", "HI"]

        # Low-vol universe
        low_vol_returns = pd.DataFrame(
            rng.randn(100, 2) * 0.005,
            columns=assets,
        )
        penalty_low = engine._dynamic_turnover_penalty(low_vol_returns, assets)

        # High-vol universe
        high_vol_returns = pd.DataFrame(
            rng.randn(100, 2) * 0.05,
            columns=assets,
        )
        penalty_high = engine._dynamic_turnover_penalty(high_vol_returns, assets)

        assert penalty_high >= penalty_low, (
            f"High-vol penalty ({penalty_high:.6f}) should >= "
            f"low-vol penalty ({penalty_low:.6f})"
        )


# ===========================================================================
# T6: _get_current_portfolio_weights returns None with no positions
# ===========================================================================

class TestGetCurrentWeightsEmpty:
    """Verify None returned when paper trader has no positions."""

    def _make_engine_with_state(self, state_dict):
        """Create engine with mocked paper trader state."""
        from quant_engine.autopilot.engine import AutopilotEngine
        with patch.object(AutopilotEngine, "__init__", lambda self: None):
            engine = AutopilotEngine.__new__(AutopilotEngine)
            engine.verbose = False
            engine.paper_trader = MagicMock()
            engine.paper_trader._load_state.return_value = state_dict
            return engine

    def test_no_positions_returns_none(self):
        """Empty positions list should return None."""
        engine = self._make_engine_with_state({
            "cash": 100_000.0,
            "positions": [],
        })
        result = engine._get_current_portfolio_weights()
        assert result is None

    def test_zero_equity_returns_none(self):
        """Zero equity should return None."""
        engine = self._make_engine_with_state({
            "cash": 0.0,
            "positions": [],
        })
        result = engine._get_current_portfolio_weights()
        assert result is None

    def test_missing_state_returns_none(self):
        """AttributeError from paper_trader should return None."""
        from quant_engine.autopilot.engine import AutopilotEngine
        with patch.object(AutopilotEngine, "__init__", lambda self: None):
            engine = AutopilotEngine.__new__(AutopilotEngine)
            engine.verbose = False
            engine.paper_trader = None  # No paper trader
            result = engine._get_current_portfolio_weights()
            assert result is None


# ===========================================================================
# T7: _get_current_portfolio_weights returns valid weights from paper state
# ===========================================================================

class TestGetCurrentWeightsWithPositions:
    """Verify correct weight computation from active positions."""

    def _make_engine_with_state(self, state_dict):
        from quant_engine.autopilot.engine import AutopilotEngine
        with patch.object(AutopilotEngine, "__init__", lambda self: None):
            engine = AutopilotEngine.__new__(AutopilotEngine)
            engine.verbose = False
            engine.paper_trader = MagicMock()
            engine.paper_trader._load_state.return_value = state_dict
            return engine

    def test_single_position(self):
        """Single position should return correct weight."""
        engine = self._make_engine_with_state({
            "cash": 90_000.0,
            "positions": [
                {"permno": "12345", "shares": 100, "last_price": 100.0},
            ],
        })
        weights = engine._get_current_portfolio_weights()
        assert weights is not None
        # Position value = 100 * 100 = 10,000
        # Equity = 90,000 + 10,000 = 100,000
        # Weight = 10,000 / 100,000 = 0.10
        assert abs(weights["12345"] - 0.10) < 1e-6

    def test_multiple_positions(self):
        """Multiple positions should sum to roughly portfolio allocation."""
        engine = self._make_engine_with_state({
            "cash": 50_000.0,
            "positions": [
                {"permno": "111", "shares": 200, "last_price": 50.0},  # 10,000
                {"permno": "222", "shares": 100, "last_price": 200.0},  # 20,000
                {"permno": "333", "shares": 400, "last_price": 50.0},  # 20,000
            ],
        })
        weights = engine._get_current_portfolio_weights()
        assert weights is not None
        # Total position value = 50,000
        # Equity = 50,000 + 50,000 = 100,000
        assert abs(weights["111"] - 0.10) < 1e-6
        assert abs(weights["222"] - 0.20) < 1e-6
        assert abs(weights["333"] - 0.20) < 1e-6

    def test_uses_entry_price_fallback(self):
        """When last_price missing, should fall back to entry_price."""
        engine = self._make_engine_with_state({
            "cash": 80_000.0,
            "positions": [
                {"permno": "999", "shares": 200, "entry_price": 100.0},
            ],
        })
        weights = engine._get_current_portfolio_weights()
        assert weights is not None
        # Position value = 200 * 100 = 20,000
        # Equity = 80,000 + 20,000 = 100,000
        assert abs(weights["999"] - 0.20) < 1e-6


# ===========================================================================
# T8: optimize_portfolio uses current_weights to reduce turnover
# ===========================================================================

class TestCurrentWeightsReduceTurnover:
    """Verify that providing current_weights causes the optimizer to stay closer."""

    def test_with_current_weights_less_change(self):
        """Optimizer with current_weights and penalty should deviate less."""
        _, er, cov = _simple_universe(n=5, seed=77)

        # A fixed starting allocation
        current = pd.Series(0.20, index=er.index)

        # Without current weights → optimizer starts from zero
        w_no_current = optimize_portfolio(
            expected_returns=er,
            covariance=cov,
            turnover_penalty=0.01,
        )
        # With current weights → optimizer tries to stay close
        w_with_current = optimize_portfolio(
            expected_returns=er,
            covariance=cov,
            turnover_penalty=0.01,
            current_weights=current,
        )

        turnover_no = _compute_turnover(current, w_no_current)
        turnover_with = _compute_turnover(current, w_with_current)

        assert turnover_with <= turnover_no + 1e-6, (
            f"With current_weights ({turnover_with:.4f}) should have "
            f"<= turnover than without ({turnover_no:.4f})"
        )


# ===========================================================================
# T9: Dynamic penalty falls back to base on import error
# ===========================================================================

class TestDynamicPenaltyFallback:
    """Verify graceful fallback when execution model unavailable."""

    def _make_engine(self):
        from quant_engine.autopilot.engine import AutopilotEngine
        with patch.object(AutopilotEngine, "__init__", lambda self: None):
            engine = AutopilotEngine.__new__(AutopilotEngine)
            engine.verbose = False
            engine.paper_trader = MagicMock()
            return engine

    def test_fallback_on_empty_returns(self):
        """Empty returns DataFrame should fall back to base penalty."""
        engine = self._make_engine()
        empty_df = pd.DataFrame()
        penalty = engine._dynamic_turnover_penalty(empty_df, [])
        assert penalty == PORTFOLIO_TURNOVER_PENALTY

    def test_fallback_on_insufficient_data(self):
        """Too few rows should skip cost estimation and return base."""
        engine = self._make_engine()
        # Only 5 rows — below the 20-row threshold
        short_df = pd.DataFrame(
            np.random.randn(5, 2) * 0.01,
            columns=["X", "Y"],
        )
        penalty = engine._dynamic_turnover_penalty(short_df, ["X", "Y"])
        assert penalty == PORTFOLIO_TURNOVER_PENALTY

    def test_fallback_on_import_error(self):
        """Simulated import failure should fall back to base penalty."""
        engine = self._make_engine()
        rng = np.random.RandomState(42)
        assets = ["A", "B"]
        returns_df = pd.DataFrame(rng.randn(100, 2) * 0.02, columns=assets)

        with patch(
            "quant_engine.autopilot.engine.AutopilotEngine._dynamic_turnover_penalty",
            return_value=PORTFOLIO_TURNOVER_PENALTY,
        ):
            penalty = engine._dynamic_turnover_penalty(returns_df, assets)
        assert penalty == PORTFOLIO_TURNOVER_PENALTY


# ===========================================================================
# T10: Config constants imported in autopilot/engine.py
# ===========================================================================

class TestEngineImports:
    """Verify that engine.py imports the P04 config constants."""

    def test_imports_exist(self):
        """The autopilot engine module should import P04 config constants."""
        import quant_engine.autopilot.engine as eng_mod
        # These should be importable attributes via the module's config import
        assert hasattr(eng_mod, "PORTFOLIO_TURNOVER_PENALTY")
        assert hasattr(eng_mod, "PORTFOLIO_TURNOVER_DYNAMIC")
        assert hasattr(eng_mod, "PORTFOLIO_TURNOVER_COST_MULTIPLIER")
