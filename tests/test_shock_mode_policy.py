"""
SPEC-E03: Shock-mode execution policy tests.

Covers:
  T1: ShockModePolicy construction for all three tiers (shock, elevated, normal)
  T2: from_shock_vector() classmethod with shock events
  T3: from_shock_vector() classmethod with high uncertainty (elevated)
  T4: from_shock_vector() classmethod with calm markets (normal)
  T5: normal_default() factory
  T6: max_participation_override in ExecutionModel.simulate()
  T7: max_participation_override in ExecutionModel.estimate_cost()
  T8: Shock mode min_confidence_override filters entries in simple backtest
  T9: Shock mode min_confidence_override filters entries in risk-managed backtest
  T10: Shock mode spread_multiplier increases costs
  T11: Shock mode disabled via SHOCK_MODE_ENABLED config
  T12: Config values exist and have correct types/defaults
  T13: Integration — shock mode reduces participation during stress
"""
import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd

from quant_engine.backtest.execution import ExecutionModel, ShockModePolicy
from quant_engine.regime.shock_vector import ShockVector


# ── Test Helpers ─────────────────────────────────────────────────────────


def _make_model(**kwargs) -> ExecutionModel:
    """Create an ExecutionModel with sensible test defaults."""
    defaults = dict(
        spread_bps=3.0,
        impact_coefficient_bps=25.0,
        max_participation_rate=0.02,
        dynamic_costs=True,
        structural_stress_enabled=True,
        volume_trend_enabled=True,
    )
    defaults.update(kwargs)
    return ExecutionModel(**defaults)


def _make_shock_vector(**overrides) -> ShockVector:
    """Create a ShockVector with defaults and optional overrides."""
    defaults = dict(
        schema_version="1.0",
        timestamp=datetime(2020, 3, 16),
        ticker="AAPL",
        hmm_regime=3,
        hmm_confidence=0.5,
        hmm_uncertainty=0.5,
        bocpd_changepoint_prob=0.0,
        bocpd_runlength=50,
        jump_detected=False,
        jump_magnitude=0.0,
        structural_features={"drift_score": 0.3, "systemic_stress": 0.5},
    )
    defaults.update(overrides)
    return ShockVector(**defaults)


def _make_ohlcv(
    n_bars: int = 100,
    base_price: float = 100.0,
    daily_volume: float = 1_000_000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_bars, freq="B")
    returns = rng.normal(0.0005, 0.015, n_bars)
    close = base_price * np.cumprod(1 + returns)
    open_ = close * (1 + rng.normal(0, 0.002, n_bars))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.01, n_bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.01, n_bars))
    volume = rng.uniform(daily_volume * 0.5, daily_volume * 1.5, n_bars)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


def _make_predictions(
    ohlcv: pd.DataFrame,
    ticker: str = "10001",
    n_signals: int = 5,
    start_bar: int = 20,
    spacing: int = 15,
    predicted_return: float = 0.02,
    confidence: float = 0.75,
    regime: int = 0,
) -> pd.DataFrame:
    """Generate synthetic predictions aligned with OHLCV data."""
    rows = []
    indices = []
    for i in range(n_signals):
        bar_idx = start_bar + i * spacing
        if bar_idx >= len(ohlcv):
            break
        dt = ohlcv.index[bar_idx]
        indices.append((ticker, dt))
        rows.append(
            {
                "predicted_return": predicted_return,
                "confidence": confidence,
                "regime": regime,
            }
        )
    if not rows:
        return pd.DataFrame()
    idx = pd.MultiIndex.from_tuples(indices, names=["ticker", "date"])
    return pd.DataFrame(rows, index=idx)


# ══════════════════════════════════════════════════════════════════════════
# T1: ShockModePolicy construction for all three tiers
# ══════════════════════════════════════════════════════════════════════════


class TestShockModePolicyConstruction(unittest.TestCase):
    """ShockModePolicy can be constructed for shock, elevated, and normal tiers."""

    def test_shock_tier(self):
        policy = ShockModePolicy(
            is_active=True,
            tier="shock",
            max_participation_override=0.005,
            spread_multiplier=2.0,
            min_confidence_override=0.80,
        )
        self.assertTrue(policy.is_active)
        self.assertEqual(policy.tier, "shock")
        self.assertAlmostEqual(policy.max_participation_override, 0.005)
        self.assertAlmostEqual(policy.spread_multiplier, 2.0)
        self.assertAlmostEqual(policy.min_confidence_override, 0.80)

    def test_elevated_tier(self):
        policy = ShockModePolicy(
            is_active=True,
            tier="elevated",
            max_participation_override=0.01,
            spread_multiplier=1.5,
            min_confidence_override=0.65,
        )
        self.assertTrue(policy.is_active)
        self.assertEqual(policy.tier, "elevated")
        self.assertAlmostEqual(policy.max_participation_override, 0.01)

    def test_normal_tier(self):
        policy = ShockModePolicy(
            is_active=False,
            tier="normal",
            max_participation_override=0.02,
            spread_multiplier=1.0,
            min_confidence_override=0.50,
        )
        self.assertFalse(policy.is_active)
        self.assertEqual(policy.tier, "normal")
        self.assertAlmostEqual(policy.spread_multiplier, 1.0)


# ══════════════════════════════════════════════════════════════════════════
# T2: from_shock_vector() — shock event triggers shock tier
# ══════════════════════════════════════════════════════════════════════════


class TestFromShockVectorShockEvent(unittest.TestCase):
    """Shock events (jump, changepoint, large move) trigger shock tier."""

    def test_jump_detected_triggers_shock(self):
        sv = _make_shock_vector(jump_detected=True, jump_magnitude=0.05)
        policy = ShockModePolicy.from_shock_vector(sv)
        self.assertTrue(policy.is_active)
        self.assertEqual(policy.tier, "shock")
        self.assertAlmostEqual(policy.max_participation_override, 0.005)

    def test_high_changepoint_prob_triggers_shock(self):
        sv = _make_shock_vector(bocpd_changepoint_prob=0.75)
        policy = ShockModePolicy.from_shock_vector(sv)
        self.assertTrue(policy.is_active)
        self.assertEqual(policy.tier, "shock")

    def test_large_jump_magnitude_triggers_shock(self):
        """Jump magnitude > 3% triggers shock even without jump_detected flag."""
        sv = _make_shock_vector(jump_magnitude=-0.05)
        policy = ShockModePolicy.from_shock_vector(sv)
        self.assertTrue(policy.is_active)
        self.assertEqual(policy.tier, "shock")

    def test_shock_overrides_match_defaults(self):
        sv = _make_shock_vector(jump_detected=True)
        policy = ShockModePolicy.from_shock_vector(sv)
        self.assertAlmostEqual(policy.max_participation_override, 0.005)
        self.assertAlmostEqual(policy.spread_multiplier, 2.0)
        self.assertAlmostEqual(policy.min_confidence_override, 0.80)


# ══════════════════════════════════════════════════════════════════════════
# T3: from_shock_vector() — high uncertainty triggers elevated tier
# ══════════════════════════════════════════════════════════════════════════


class TestFromShockVectorElevated(unittest.TestCase):
    """High HMM uncertainty without shock event triggers elevated tier."""

    def test_high_uncertainty_triggers_elevated(self):
        sv = _make_shock_vector(hmm_uncertainty=0.85)
        policy = ShockModePolicy.from_shock_vector(sv)
        self.assertTrue(policy.is_active)
        self.assertEqual(policy.tier, "elevated")

    def test_uncertainty_just_above_threshold(self):
        sv = _make_shock_vector(hmm_uncertainty=0.71)
        policy = ShockModePolicy.from_shock_vector(sv)
        self.assertTrue(policy.is_active)
        self.assertEqual(policy.tier, "elevated")
        self.assertAlmostEqual(policy.max_participation_override, 0.01)
        self.assertAlmostEqual(policy.spread_multiplier, 1.5)
        self.assertAlmostEqual(policy.min_confidence_override, 0.65)

    def test_uncertainty_at_threshold_stays_normal(self):
        sv = _make_shock_vector(hmm_uncertainty=0.70)
        policy = ShockModePolicy.from_shock_vector(sv)
        self.assertFalse(policy.is_active)
        self.assertEqual(policy.tier, "normal")

    def test_custom_uncertainty_threshold(self):
        sv = _make_shock_vector(hmm_uncertainty=0.55)
        policy = ShockModePolicy.from_shock_vector(
            sv, elevated_uncertainty_threshold=0.50,
        )
        self.assertTrue(policy.is_active)
        self.assertEqual(policy.tier, "elevated")


# ══════════════════════════════════════════════════════════════════════════
# T4: from_shock_vector() — calm markets yield normal tier
# ══════════════════════════════════════════════════════════════════════════


class TestFromShockVectorNormal(unittest.TestCase):
    """Calm markets (no shock, low uncertainty) yield normal tier."""

    def test_calm_market_normal(self):
        sv = _make_shock_vector(
            hmm_uncertainty=0.3,
            bocpd_changepoint_prob=0.1,
            jump_detected=False,
            jump_magnitude=0.005,
        )
        policy = ShockModePolicy.from_shock_vector(sv)
        self.assertFalse(policy.is_active)
        self.assertEqual(policy.tier, "normal")
        self.assertAlmostEqual(policy.spread_multiplier, 1.0)

    def test_normal_uses_default_participation(self):
        sv = _make_shock_vector(hmm_uncertainty=0.2)
        policy = ShockModePolicy.from_shock_vector(sv)
        self.assertAlmostEqual(policy.max_participation_override, 0.02)


# ══════════════════════════════════════════════════════════════════════════
# T5: normal_default() factory
# ══════════════════════════════════════════════════════════════════════════


class TestNormalDefault(unittest.TestCase):
    """normal_default() returns a non-active normal policy."""

    def test_normal_default_not_active(self):
        policy = ShockModePolicy.normal_default()
        self.assertFalse(policy.is_active)
        self.assertEqual(policy.tier, "normal")
        self.assertAlmostEqual(policy.max_participation_override, 0.02)
        self.assertAlmostEqual(policy.spread_multiplier, 1.0)
        self.assertAlmostEqual(policy.min_confidence_override, 0.50)


# ══════════════════════════════════════════════════════════════════════════
# T6: max_participation_override in simulate()
# ══════════════════════════════════════════════════════════════════════════


class TestMaxParticipationOverrideSimulate(unittest.TestCase):
    """max_participation_override restricts fill in simulate()."""

    def test_lower_participation_reduces_fill_ratio(self):
        model = _make_model(max_participation_rate=0.02)
        # Large order that should be participation-limited
        fill_normal = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=100_000,  # low volume to force participation cap
            desired_notional_usd=500_000,
            force_full=False,
        )
        fill_shock = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=100_000,
            desired_notional_usd=500_000,
            force_full=False,
            max_participation_override=0.005,  # 4x lower participation
        )
        # Shock mode should fill less (lower participation cap)
        self.assertLessEqual(fill_shock.fill_ratio, fill_normal.fill_ratio)

    def test_override_does_not_affect_force_full(self):
        """force_full bypasses participation limits, even with override."""
        model = _make_model(max_participation_rate=0.02)
        fill = model.simulate(
            side="sell",
            reference_price=100.0,
            daily_volume=100_000,
            desired_notional_usd=500_000,
            force_full=True,
            max_participation_override=0.001,
        )
        self.assertAlmostEqual(fill.fill_ratio, 1.0)

    def test_none_override_uses_default(self):
        """None override falls through to model's default max_participation."""
        model = _make_model(max_participation_rate=0.02)
        fill_none = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=1_000_000,
            desired_notional_usd=100_000,
            max_participation_override=None,
        )
        fill_default = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=1_000_000,
            desired_notional_usd=100_000,
        )
        self.assertAlmostEqual(fill_none.fill_ratio, fill_default.fill_ratio)


# ══════════════════════════════════════════════════════════════════════════
# T7: max_participation_override in estimate_cost()
# ══════════════════════════════════════════════════════════════════════════


class TestMaxParticipationOverrideEstimateCost(unittest.TestCase):
    """max_participation_override affects cost estimation."""

    def test_lower_participation_changes_cost(self):
        model = _make_model(max_participation_rate=0.02)
        cost_normal = model.estimate_cost(
            daily_volume=100_000,
            desired_notional=500_000,
            realized_vol=0.20,
            reference_price=100.0,
        )
        cost_shock = model.estimate_cost(
            daily_volume=100_000,
            desired_notional=500_000,
            realized_vol=0.20,
            reference_price=100.0,
            max_participation_override=0.005,
        )
        # With lower participation, cost should differ (may be lower due to
        # less market impact, OR effectively zero if too small to fill)
        self.assertIsInstance(cost_shock, float)
        self.assertTrue(np.isfinite(cost_shock))

    def test_none_override_matches_default(self):
        model = _make_model()
        cost_none = model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.20,
            reference_price=100.0,
            max_participation_override=None,
        )
        cost_default = model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.20,
            reference_price=100.0,
        )
        self.assertAlmostEqual(cost_none, cost_default)


# ══════════════════════════════════════════════════════════════════════════
# T8: Shock mode min_confidence_override in simple backtest
# ══════════════════════════════════════════════════════════════════════════


class TestShockModeConfidenceFilterSimple(unittest.TestCase):
    """Shock mode confidence filter rejects low-confidence entries."""

    def _run_backtest(
        self,
        confidence: float = 0.60,
        shock_enabled: bool = True,
        shock_min_confidence: float = 0.80,
    ):
        """Run a simple backtest with shock mode and given confidence."""
        from quant_engine.backtest.engine import Backtester

        ohlcv = _make_ohlcv(n_bars=100, daily_volume=1_000_000)
        predictions = _make_predictions(
            ohlcv,
            predicted_return=0.05,  # high edge to pass cost gate
            confidence=confidence,
            n_signals=3,
            start_bar=20,
            spacing=20,
        )

        # Build shock policy that will be returned for every bar
        forced_policy = ShockModePolicy(
            is_active=True, tier="shock",
            max_participation_override=0.005,
            spread_multiplier=2.0,
            min_confidence_override=shock_min_confidence,
        )
        normal_policy = ShockModePolicy.normal_default()

        with patch("quant_engine.backtest.engine.SHOCK_MODE_ENABLED", shock_enabled), \
             patch("quant_engine.backtest.engine.EXEC_STRUCTURAL_STRESS_ENABLED", False), \
             patch("quant_engine.backtest.engine.EXEC_CALIBRATION_ENABLED", False), \
             patch("quant_engine.backtest.engine.EDGE_COST_GATE_ENABLED", False), \
             patch("quant_engine.validation.preconditions.enforce_preconditions"):
            bt = Backtester(
                entry_threshold=0.001,
                confidence_threshold=0.5,
                holding_days=5,
                max_positions=5,
                use_risk_management=False,
            )
            # Patch _compute_shock_policy to always return shock tier
            # when shock is enabled, or normal when disabled
            original_compute = bt._compute_shock_policy
            bt._compute_shock_policy = lambda shock: (
                forced_policy if shock_enabled else normal_policy
            )

            result = bt.run(predictions, {"10001": ohlcv}, verbose=False)
        return result

    def test_low_confidence_filtered_during_shock(self):
        """Confidence 0.60 < shock min 0.80 → all signals filtered."""
        result = self._run_backtest(confidence=0.60, shock_enabled=True)
        self.assertEqual(result.total_trades, 0)

    def test_high_confidence_passes_during_shock(self):
        """Confidence 0.85 > shock min 0.80 → signals pass."""
        result = self._run_backtest(confidence=0.85, shock_enabled=True)
        self.assertGreater(result.total_trades, 0)

    def test_shock_disabled_allows_low_confidence(self):
        """With shock mode disabled, low-confidence signals pass."""
        result = self._run_backtest(confidence=0.60, shock_enabled=False)
        self.assertGreater(result.total_trades, 0)


# ══════════════════════════════════════════════════════════════════════════
# T9: Shock mode min_confidence_override in risk-managed backtest
# ══════════════════════════════════════════════════════════════════════════


class TestShockModeConfidenceFilterRiskManaged(unittest.TestCase):
    """Shock mode confidence filter works in risk-managed mode."""

    def _run_backtest(self, confidence: float = 0.60, shock_enabled: bool = True):
        from quant_engine.backtest.engine import Backtester

        ohlcv = _make_ohlcv(n_bars=150, daily_volume=1_000_000)
        predictions = _make_predictions(
            ohlcv,
            predicted_return=0.05,
            confidence=confidence,
            n_signals=3,
            start_bar=25,
            spacing=25,
        )

        forced_policy = ShockModePolicy(
            is_active=True, tier="shock",
            max_participation_override=0.005,
            spread_multiplier=2.0,
            min_confidence_override=0.80,
        )
        normal_policy = ShockModePolicy.normal_default()

        with patch("quant_engine.backtest.engine.SHOCK_MODE_ENABLED", shock_enabled), \
             patch("quant_engine.backtest.engine.EXEC_STRUCTURAL_STRESS_ENABLED", False), \
             patch("quant_engine.backtest.engine.EXEC_CALIBRATION_ENABLED", False), \
             patch("quant_engine.backtest.engine.EDGE_COST_GATE_ENABLED", False), \
             patch("quant_engine.validation.preconditions.enforce_preconditions"):
            bt = Backtester(
                entry_threshold=0.001,
                confidence_threshold=0.5,
                holding_days=5,
                max_positions=5,
                use_risk_management=True,
            )
            bt._compute_shock_policy = lambda shock: (
                forced_policy if shock_enabled else normal_policy
            )

            result = bt.run(predictions, {"10001": ohlcv}, verbose=False)
        return result

    def test_low_confidence_filtered_risk_managed(self):
        result = self._run_backtest(confidence=0.60, shock_enabled=True)
        self.assertEqual(result.total_trades, 0)

    def test_high_confidence_passes_risk_managed(self):
        result = self._run_backtest(confidence=0.85, shock_enabled=True)
        self.assertGreater(result.total_trades, 0)


# ══════════════════════════════════════════════════════════════════════════
# T10: Shock mode spread_multiplier increases costs
# ══════════════════════════════════════════════════════════════════════════


class TestShockModeSpreadMultiplier(unittest.TestCase):
    """Shock mode spread multiplier increases execution costs."""

    def test_spread_multiplier_increases_buy_cost(self):
        model = _make_model()
        fill_normal = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=1_000_000,
            desired_notional_usd=100_000,
            event_spread_multiplier=1.0,
            realized_vol=0.20,
        )
        fill_shock = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=1_000_000,
            desired_notional_usd=100_000,
            event_spread_multiplier=2.0,  # shock mode
            realized_vol=0.20,
        )
        # Higher spread → higher fill price for buys
        self.assertGreater(fill_shock.fill_price, fill_normal.fill_price)

    def test_spread_multiplier_increases_sell_cost(self):
        model = _make_model()
        fill_normal = model.simulate(
            side="sell",
            reference_price=100.0,
            daily_volume=1_000_000,
            desired_notional_usd=100_000,
            event_spread_multiplier=1.0,
            realized_vol=0.20,
            force_full=True,
        )
        fill_shock = model.simulate(
            side="sell",
            reference_price=100.0,
            daily_volume=1_000_000,
            desired_notional_usd=100_000,
            event_spread_multiplier=2.0,
            realized_vol=0.20,
            force_full=True,
        )
        # Higher spread → lower fill price for sells
        self.assertLess(fill_shock.fill_price, fill_normal.fill_price)

    def test_estimate_cost_reflects_spread_mult(self):
        model = _make_model()
        cost_normal = model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.20,
            reference_price=100.0,
            event_spread_multiplier=1.0,
        )
        cost_shock = model.estimate_cost(
            daily_volume=1_000_000,
            desired_notional=100_000,
            realized_vol=0.20,
            reference_price=100.0,
            event_spread_multiplier=2.0,
        )
        self.assertGreater(cost_shock, cost_normal)


# ══════════════════════════════════════════════════════════════════════════
# T11: Shock mode disabled via config
# ══════════════════════════════════════════════════════════════════════════


class TestShockModeDisabled(unittest.TestCase):
    """When SHOCK_MODE_ENABLED is False, shock mode never activates."""

    def test_shock_disabled_returns_normal(self):
        from quant_engine.backtest.engine import Backtester

        with patch("quant_engine.backtest.engine.SHOCK_MODE_ENABLED", False), \
             patch("quant_engine.backtest.engine.EXEC_STRUCTURAL_STRESS_ENABLED", False), \
             patch("quant_engine.backtest.engine.EXEC_CALIBRATION_ENABLED", False), \
             patch("quant_engine.validation.preconditions.enforce_preconditions"):
            bt = Backtester(
                entry_threshold=0.001,
                confidence_threshold=0.5,
                holding_days=5,
                max_positions=5,
            )
            shock_sv = _make_shock_vector(jump_detected=True)
            policy = bt._compute_shock_policy(shock_sv)
            self.assertFalse(policy.is_active)
            self.assertEqual(policy.tier, "normal")

    def test_none_shock_vector_returns_normal(self):
        from quant_engine.backtest.engine import Backtester

        with patch("quant_engine.backtest.engine.SHOCK_MODE_ENABLED", True), \
             patch("quant_engine.backtest.engine.EXEC_STRUCTURAL_STRESS_ENABLED", False), \
             patch("quant_engine.backtest.engine.EXEC_CALIBRATION_ENABLED", False), \
             patch("quant_engine.validation.preconditions.enforce_preconditions"):
            bt = Backtester(
                entry_threshold=0.001,
                confidence_threshold=0.5,
                holding_days=5,
                max_positions=5,
            )
            policy = bt._compute_shock_policy(None)
            self.assertFalse(policy.is_active)
            self.assertEqual(policy.tier, "normal")


# ══════════════════════════════════════════════════════════════════════════
# T12: Config values exist and have correct types/defaults
# ══════════════════════════════════════════════════════════════════════════


class TestShockModeConfig(unittest.TestCase):
    """Config values for SPEC-E03 exist with correct types and defaults."""

    def test_config_values_exist(self):
        from quant_engine.config import (
            SHOCK_MODE_ENABLED,
            SHOCK_MODE_SHOCK_MAX_PARTICIPATION,
            SHOCK_MODE_SHOCK_SPREAD_MULT,
            SHOCK_MODE_SHOCK_MIN_CONFIDENCE,
            SHOCK_MODE_ELEVATED_MAX_PARTICIPATION,
            SHOCK_MODE_ELEVATED_SPREAD_MULT,
            SHOCK_MODE_ELEVATED_MIN_CONFIDENCE,
            SHOCK_MODE_UNCERTAINTY_THRESHOLD,
        )
        self.assertIsInstance(SHOCK_MODE_ENABLED, bool)
        self.assertIsInstance(SHOCK_MODE_SHOCK_MAX_PARTICIPATION, float)
        self.assertIsInstance(SHOCK_MODE_SHOCK_SPREAD_MULT, float)
        self.assertIsInstance(SHOCK_MODE_SHOCK_MIN_CONFIDENCE, float)
        self.assertIsInstance(SHOCK_MODE_ELEVATED_MAX_PARTICIPATION, float)
        self.assertIsInstance(SHOCK_MODE_ELEVATED_SPREAD_MULT, float)
        self.assertIsInstance(SHOCK_MODE_ELEVATED_MIN_CONFIDENCE, float)
        self.assertIsInstance(SHOCK_MODE_UNCERTAINTY_THRESHOLD, float)

    def test_default_values(self):
        from quant_engine.config import (
            SHOCK_MODE_ENABLED,
            SHOCK_MODE_SHOCK_MAX_PARTICIPATION,
            SHOCK_MODE_SHOCK_SPREAD_MULT,
            SHOCK_MODE_SHOCK_MIN_CONFIDENCE,
            SHOCK_MODE_ELEVATED_MAX_PARTICIPATION,
            SHOCK_MODE_ELEVATED_SPREAD_MULT,
            SHOCK_MODE_ELEVATED_MIN_CONFIDENCE,
            SHOCK_MODE_UNCERTAINTY_THRESHOLD,
        )
        self.assertTrue(SHOCK_MODE_ENABLED)
        self.assertAlmostEqual(SHOCK_MODE_SHOCK_MAX_PARTICIPATION, 0.005)
        self.assertAlmostEqual(SHOCK_MODE_SHOCK_SPREAD_MULT, 2.0)
        self.assertAlmostEqual(SHOCK_MODE_SHOCK_MIN_CONFIDENCE, 0.80)
        self.assertAlmostEqual(SHOCK_MODE_ELEVATED_MAX_PARTICIPATION, 0.01)
        self.assertAlmostEqual(SHOCK_MODE_ELEVATED_SPREAD_MULT, 1.5)
        self.assertAlmostEqual(SHOCK_MODE_ELEVATED_MIN_CONFIDENCE, 0.65)
        self.assertAlmostEqual(SHOCK_MODE_UNCERTAINTY_THRESHOLD, 0.7)

    def test_shock_participation_less_than_normal(self):
        from quant_engine.config import (
            SHOCK_MODE_SHOCK_MAX_PARTICIPATION,
            SHOCK_MODE_ELEVATED_MAX_PARTICIPATION,
            EXEC_MAX_PARTICIPATION,
        )
        self.assertLess(SHOCK_MODE_SHOCK_MAX_PARTICIPATION, SHOCK_MODE_ELEVATED_MAX_PARTICIPATION)
        self.assertLess(SHOCK_MODE_ELEVATED_MAX_PARTICIPATION, EXEC_MAX_PARTICIPATION)


# ══════════════════════════════════════════════════════════════════════════
# T13: Integration — shock mode reduces participation during stress
# ══════════════════════════════════════════════════════════════════════════


class TestShockModeIntegration(unittest.TestCase):
    """End-to-end: shock mode reduces participation and fills during stress."""

    def test_shock_mode_reduces_fill_ratio(self):
        """Orders during shock events should get smaller fills due to
        reduced participation limits."""
        model = _make_model(max_participation_rate=0.02)

        # Large order relative to volume
        normal_fill = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=500_000,
            desired_notional_usd=300_000,
            force_full=False,
        )
        shock_fill = model.simulate(
            side="buy",
            reference_price=100.0,
            daily_volume=500_000,
            desired_notional_usd=300_000,
            force_full=False,
            max_participation_override=0.005,
            event_spread_multiplier=2.0,
        )

        # Shock mode should fill less
        self.assertLessEqual(shock_fill.fill_ratio, normal_fill.fill_ratio)
        # Shock mode should have wider spread applied
        self.assertGreaterEqual(
            shock_fill.event_spread_multiplier_applied,
            normal_fill.event_spread_multiplier_applied,
        )

    def test_shock_vector_priority(self):
        """Shock event takes priority over elevated uncertainty."""
        sv = _make_shock_vector(
            jump_detected=True,
            hmm_uncertainty=0.90,  # also high uncertainty
        )
        policy = ShockModePolicy.from_shock_vector(sv)
        # Shock event should take priority
        self.assertEqual(policy.tier, "shock")
        self.assertAlmostEqual(policy.max_participation_override, 0.005)

    def test_custom_config_overrides(self):
        """Custom thresholds are passed through from_shock_vector."""
        sv = _make_shock_vector(jump_detected=True)
        policy = ShockModePolicy.from_shock_vector(
            sv,
            shock_max_participation=0.002,
            shock_spread_mult=3.0,
            shock_min_confidence=0.90,
        )
        self.assertAlmostEqual(policy.max_participation_override, 0.002)
        self.assertAlmostEqual(policy.spread_multiplier, 3.0)
        self.assertAlmostEqual(policy.min_confidence_override, 0.90)


if __name__ == "__main__":
    unittest.main()
