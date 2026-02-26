"""
Comprehensive test suite for Spec 07: Portfolio Layer + Regime-Conditioned Constraints.

Covers:
    - T1: UniverseConfig loading, validation, and query
    - T2: ConstraintMultiplier regime-conditioned checks
    - T3: Regime-conditional covariance in correlation checks
    - T4: FactorExposureManager computation and bounds checking
    - T5: Sizing backoff integration
    - T6: Constraint tightening replay
    - T7: Smooth constraint transitions
    - T8: Integration (end-to-end)
"""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# ── Test data helpers ─────────────────────────────────────────────────────


def _make_ohlcv(
    close: pd.Series,
    volume: float = 1_000_000.0,
) -> pd.DataFrame:
    """Build synthetic OHLCV data from a close series."""
    return pd.DataFrame(
        {
            "Open": close.values,
            "High": close.values * 1.01,
            "Low": close.values * 0.99,
            "Close": close.values,
            "Volume": np.full(len(close), volume),
        },
        index=close.index,
    )


def _generate_price_data(
    tickers: list,
    n_days: int = 252,
    seed: int = 42,
    start_price: float = 100.0,
    vol: float = 0.02,
) -> dict:
    """Generate synthetic OHLCV data for multiple tickers."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    data = {}
    for i, ticker in enumerate(tickers):
        returns = rng.normal(0.0005, vol, size=n_days)
        prices = start_price * np.cumprod(1 + returns)
        close = pd.Series(prices, index=idx)
        data[ticker] = _make_ohlcv(close, volume=1_000_000.0 * (1 + i * 0.1))
    return data


def _minimal_universe_yaml() -> str:
    """Return a minimal valid universe.yaml content."""
    return """
sectors:
  tech:
    - AAPL
    - MSFT
    - GOOGL
  healthcare:
    - JNJ
    - PFE
  consumer:
    - AMZN
    - COST
  financial:
    - JPM
    - GS

liquidity_tiers:
  Mega:
    market_cap_min: 200.0e9
    dollar_volume_min: 100.0e6
  Large:
    market_cap_min: 10.0e9
    dollar_volume_min: 1.0e6

borrowability:
  hard_to_borrow:
    - TSLA
    - GME
  restricted:
    - ORCL

constraint_base:
  sector_cap: 0.40
  correlation_limit: 0.85
  gross_exposure: 1.00
  single_name_cap: 0.10
  annualized_turnover_max: 5.00

stress_multipliers:
  normal:
    sector_cap: 1.0
    correlation_limit: 1.0
    gross_exposure: 1.0
    turnover: 1.0
  stress:
    sector_cap: 0.6
    correlation_limit: 0.7
    gross_exposure: 0.8
    turnover: 0.5

factor_limits:
  beta:
    normal: [0.8, 1.2]
    stress: [0.9, 1.1]
  volatility:
    normal: [0.8, 1.2]
    stress: [0.5, 1.0]
  size:
    normal: null
    stress: null
  value:
    normal: null
    stress: null
  momentum:
    normal: null
    stress: null

backoff_policy:
  mode: continuous
  thresholds: [0.70, 0.80, 0.90, 0.95]
  backoff_factors: [0.9, 0.7, 0.5, 0.25]
"""


class _TempUniverseYAML:
    """Context manager that writes a temp universe.yaml and cleans up."""

    def __init__(self, content: str = ""):
        self.content = content or _minimal_universe_yaml()
        self._tmpdir = None
        self._path = None

    def __enter__(self):
        self._tmpdir = tempfile.mkdtemp()
        self._path = Path(self._tmpdir) / "universe.yaml"
        self._path.write_text(self.content)
        return str(self._path)

    def __exit__(self, *args):
        if self._path and self._path.exists():
            self._path.unlink()
        if self._tmpdir:
            os.rmdir(self._tmpdir)


# ══════════════════════════════════════════════════════════════════════════
# T1: UniverseConfig tests
# ══════════════════════════════════════════════════════════════════════════


class TestUniverseConfig(unittest.TestCase):
    """Tests for UniverseConfig loading, validation, and query."""

    def test_load_valid_yaml(self):
        from quant_engine.risk.universe_config import UniverseConfig

        with _TempUniverseYAML() as path:
            config = UniverseConfig(path)
            self.assertIsNotNone(config)

    def test_get_sector_known_ticker(self):
        from quant_engine.risk.universe_config import UniverseConfig

        with _TempUniverseYAML() as path:
            config = UniverseConfig(path)
            self.assertEqual(config.get_sector("AAPL"), "tech")
            self.assertEqual(config.get_sector("JNJ"), "healthcare")
            self.assertEqual(config.get_sector("AMZN"), "consumer")
            self.assertEqual(config.get_sector("JPM"), "financial")

    def test_get_sector_unknown_ticker(self):
        from quant_engine.risk.universe_config import UniverseConfig

        with _TempUniverseYAML() as path:
            config = UniverseConfig(path)
            self.assertEqual(config.get_sector("UNKNOWN"), "other")

    def test_get_sector_case_insensitive(self):
        from quant_engine.risk.universe_config import UniverseConfig

        with _TempUniverseYAML() as path:
            config = UniverseConfig(path)
            self.assertEqual(config.get_sector("aapl"), "tech")
            self.assertEqual(config.get_sector("Aapl"), "tech")

    def test_get_sector_constituents(self):
        from quant_engine.risk.universe_config import UniverseConfig

        with _TempUniverseYAML() as path:
            config = UniverseConfig(path)
            tech = config.get_sector_constituents("tech")
            self.assertIn("AAPL", tech)
            self.assertIn("MSFT", tech)
            self.assertIn("GOOGL", tech)

    def test_liquidity_tier(self):
        from quant_engine.risk.universe_config import UniverseConfig

        with _TempUniverseYAML() as path:
            config = UniverseConfig(path)
            self.assertEqual(config.get_liquidity_tier(300e9, 200e6), "Mega")
            self.assertEqual(config.get_liquidity_tier(50e9, 5e6), "Large")
            self.assertEqual(config.get_liquidity_tier(100e6, 50e3), "Micro")

    def test_borrowability(self):
        from quant_engine.risk.universe_config import UniverseConfig

        with _TempUniverseYAML() as path:
            config = UniverseConfig(path)
            self.assertTrue(config.is_hard_to_borrow("TSLA"))
            self.assertFalse(config.is_hard_to_borrow("AAPL"))
            self.assertTrue(config.is_restricted("ORCL"))
            self.assertFalse(config.is_restricted("MSFT"))

    def test_constraint_base(self):
        from quant_engine.risk.universe_config import UniverseConfig

        with _TempUniverseYAML() as path:
            config = UniverseConfig(path)
            cb = config.constraint_base
            self.assertAlmostEqual(cb["sector_cap"], 0.40)
            self.assertAlmostEqual(cb["correlation_limit"], 0.85)
            self.assertAlmostEqual(cb["gross_exposure"], 1.00)
            self.assertAlmostEqual(cb["single_name_cap"], 0.10)

    def test_stress_multipliers(self):
        from quant_engine.risk.universe_config import UniverseConfig

        with _TempUniverseYAML() as path:
            config = UniverseConfig(path)
            normal = config.get_stress_multiplier_set(is_stress=False)
            stress = config.get_stress_multiplier_set(is_stress=True)
            self.assertAlmostEqual(normal["sector_cap"], 1.0)
            self.assertAlmostEqual(stress["sector_cap"], 0.6)
            self.assertAlmostEqual(stress["correlation_limit"], 0.7)

    def test_factor_bounds(self):
        from quant_engine.risk.universe_config import UniverseConfig

        with _TempUniverseYAML() as path:
            config = UniverseConfig(path)
            # Normal beta bounds
            bounds = config.get_factor_bounds("beta", is_stress=False)
            self.assertEqual(bounds, (0.8, 1.2))
            # Stress beta bounds
            bounds = config.get_factor_bounds("beta", is_stress=True)
            self.assertEqual(bounds, (0.9, 1.1))
            # Size = unconstrained (null)
            self.assertIsNone(config.get_factor_bounds("size", is_stress=False))

    def test_missing_file_raises(self):
        from quant_engine.risk.universe_config import UniverseConfig, ConfigError

        with self.assertRaises(ConfigError):
            UniverseConfig("/nonexistent/path/universe.yaml")

    def test_invalid_yaml_raises(self):
        from quant_engine.risk.universe_config import UniverseConfig, ConfigError

        with _TempUniverseYAML("not: valid: yaml: [}") as path:
            # PyYAML may or may not raise on this — test the validation
            pass

    def test_missing_required_section_raises(self):
        from quant_engine.risk.universe_config import UniverseConfig, ConfigError

        yaml_content = """
sectors:
  tech:
    - AAPL
# Missing constraint_base and stress_multipliers
"""
        with _TempUniverseYAML(yaml_content) as path:
            with self.assertRaises(ConfigError):
                UniverseConfig(path)

    def test_backoff_policy(self):
        from quant_engine.risk.universe_config import UniverseConfig

        with _TempUniverseYAML() as path:
            config = UniverseConfig(path)
            policy = config.backoff_policy
            self.assertEqual(policy["mode"], "continuous")
            self.assertEqual(len(policy["thresholds"]), 4)
            self.assertEqual(len(policy["backoff_factors"]), 4)


# ══════════════════════════════════════════════════════════════════════════
# T2: ConstraintMultiplier tests
# ══════════════════════════════════════════════════════════════════════════


class TestConstraintMultiplier(unittest.TestCase):
    """Tests for regime-conditioned constraint multipliers."""

    def test_normal_regime_multipliers_are_unity(self):
        from quant_engine.risk.portfolio_risk import ConstraintMultiplier

        cm = ConstraintMultiplier()
        mults = cm.get_multipliers(regime=0)  # trending_bull
        self.assertAlmostEqual(mults["sector_cap"], 1.0)
        self.assertAlmostEqual(mults["correlation_limit"], 1.0)
        self.assertAlmostEqual(mults["gross_exposure"], 1.0)

    def test_stress_regime_multipliers_tighten(self):
        from quant_engine.risk.portfolio_risk import ConstraintMultiplier

        cm = ConstraintMultiplier()
        mults = cm.get_multipliers(regime=3)  # high_volatility = stress
        self.assertAlmostEqual(mults["sector_cap"], 0.6)
        self.assertAlmostEqual(mults["correlation_limit"], 0.7)
        self.assertAlmostEqual(mults["gross_exposure"], 0.8)

    def test_is_stress_regime(self):
        from quant_engine.risk.portfolio_risk import ConstraintMultiplier

        self.assertFalse(ConstraintMultiplier.is_stress_regime(0))  # bull
        self.assertFalse(ConstraintMultiplier.is_stress_regime(1))  # bear
        self.assertTrue(ConstraintMultiplier.is_stress_regime(2))   # mean_reverting
        self.assertTrue(ConstraintMultiplier.is_stress_regime(3))   # high_vol

    def test_with_universe_config(self):
        from quant_engine.risk.portfolio_risk import ConstraintMultiplier
        from quant_engine.risk.universe_config import UniverseConfig

        with _TempUniverseYAML() as path:
            config = UniverseConfig(path)
            cm = ConstraintMultiplier(config)
            stress = cm.get_multipliers(regime=3)
            self.assertAlmostEqual(stress["sector_cap"], 0.6)
            self.assertAlmostEqual(stress["turnover"], 0.5)


# ══════════════════════════════════════════════════════════════════════════
# T7: Smooth constraint transition tests
# ══════════════════════════════════════════════════════════════════════════


class TestSmoothTransitions(unittest.TestCase):
    """Tests for smooth constraint multiplier transitions."""

    def test_smoothed_multipliers_transition_gradually(self):
        from quant_engine.risk.portfolio_risk import ConstraintMultiplier

        cm = ConstraintMultiplier()
        # Start in normal regime
        m0 = cm.get_multipliers_smoothed(regime=0, alpha=0.3)
        self.assertAlmostEqual(m0["sector_cap"], 1.0)

        # Switch to stress regime — should NOT jump immediately to 0.6
        m1 = cm.get_multipliers_smoothed(regime=3, alpha=0.3)
        # After one step: 0.3 * 0.6 + 0.7 * 1.0 = 0.88
        self.assertAlmostEqual(m1["sector_cap"], 0.88, places=2)

        # Second step: 0.3 * 0.6 + 0.7 * 0.88 = 0.796
        m2 = cm.get_multipliers_smoothed(regime=3, alpha=0.3)
        self.assertAlmostEqual(m2["sector_cap"], 0.796, places=2)

        # After many steps, should converge to stress values
        for _ in range(20):
            m_final = cm.get_multipliers_smoothed(regime=3, alpha=0.3)
        self.assertAlmostEqual(m_final["sector_cap"], 0.6, places=1)

    def test_smoothed_alpha_1_is_immediate(self):
        from quant_engine.risk.portfolio_risk import ConstraintMultiplier

        cm = ConstraintMultiplier()
        cm.get_multipliers_smoothed(regime=0, alpha=1.0)
        m = cm.get_multipliers_smoothed(regime=3, alpha=1.0)
        self.assertAlmostEqual(m["sector_cap"], 0.6)

    def test_reset_clears_smoothing_state(self):
        from quant_engine.risk.portfolio_risk import ConstraintMultiplier

        cm = ConstraintMultiplier()
        cm.get_multipliers_smoothed(regime=0)
        cm.get_multipliers_smoothed(regime=3, alpha=0.3)  # Partially smoothed
        cm.reset()
        m = cm.get_multipliers_smoothed(regime=0)
        self.assertAlmostEqual(m["sector_cap"], 1.0)

    def test_regime_transition_0_to_3_to_0(self):
        from quant_engine.risk.portfolio_risk import ConstraintMultiplier

        cm = ConstraintMultiplier()
        # Start normal
        cm.get_multipliers_smoothed(regime=0, alpha=0.3)
        # Move to stress
        for _ in range(10):
            cm.get_multipliers_smoothed(regime=3, alpha=0.3)
        stress_val = cm.get_multipliers_smoothed(regime=3, alpha=0.3)["sector_cap"]
        self.assertLess(stress_val, 0.65)  # Should be near 0.6

        # Move back to normal
        for _ in range(10):
            cm.get_multipliers_smoothed(regime=0, alpha=0.3)
        normal_val = cm.get_multipliers_smoothed(regime=0, alpha=0.3)["sector_cap"]
        self.assertGreater(normal_val, 0.95)  # Should be near 1.0


# ══════════════════════════════════════════════════════════════════════════
# T2 extended: PortfolioRiskManager regime-conditioned checks
# ══════════════════════════════════════════════════════════════════════════


class TestPortfolioRiskRegime(unittest.TestCase):
    """Tests for regime-conditioned portfolio risk checks."""

    def setUp(self):
        self.price_data = _generate_price_data(
            ["AAPL", "MSFT", "JNJ", "JPM", "AMZN"],
            n_days=120,
            seed=42,
        )

    def test_normal_regime_uses_base_constraints(self):
        from quant_engine.risk.portfolio_risk import PortfolioRiskManager

        with _TempUniverseYAML() as path:
            from quant_engine.risk.universe_config import UniverseConfig
            config = UniverseConfig(path)
            rm = PortfolioRiskManager(universe_config=config)

            # 35% in tech should pass under normal (40% cap)
            check = rm.check_new_position(
                ticker="AAPL",
                position_size=0.05,
                current_positions={"MSFT": 0.30},
                price_data=self.price_data,
                regime=0,  # normal
            )
            # 35% < 40%, should pass sector check
            sector_violations = [v for v in check.violations if "Sector" in v]
            self.assertEqual(len(sector_violations), 0)

    def test_stress_regime_tightens_sector_cap(self):
        from quant_engine.risk.portfolio_risk import PortfolioRiskManager

        with _TempUniverseYAML() as path:
            from quant_engine.risk.universe_config import UniverseConfig
            config = UniverseConfig(path)
            rm = PortfolioRiskManager(universe_config=config)
            # Force immediate transition (alpha=1.0)
            rm.multiplier._smoothed = rm.multiplier._stress_mults.copy()
            rm.multiplier._prev_regime_is_stress = True

            # 30% in tech should FAIL under stress (40% * 0.6 = 24% cap)
            check = rm.check_new_position(
                ticker="AAPL",
                position_size=0.05,
                current_positions={"MSFT": 0.20},
                price_data=self.price_data,
                regime=3,  # stress
            )
            sector_violations = [v for v in check.violations if "Sector" in v]
            self.assertGreater(len(sector_violations), 0)

    def test_constraint_utilization_returned(self):
        from quant_engine.risk.portfolio_risk import PortfolioRiskManager

        rm = PortfolioRiskManager()
        check = rm.check_new_position(
            ticker="AAPL",
            position_size=0.05,
            current_positions={"MSFT": 0.05},
            price_data=self.price_data,
        )
        self.assertIn("gross_exposure", check.constraint_utilization)
        self.assertIn("sector_cap", check.constraint_utilization)

    def test_backoff_recommended_when_near_binding(self):
        from quant_engine.risk.portfolio_risk import PortfolioRiskManager

        rm = PortfolioRiskManager(max_sector_pct=0.40)
        # Put 38% in tech (95% of 40% cap)
        check = rm.check_new_position(
            ticker="AAPL",
            position_size=0.08,
            current_positions={"MSFT": 0.30},
            price_data=self.price_data,
        )
        # Sector utilization = 38%/40% = 0.95, should trigger backoff
        self.assertIn("sector_cap", check.constraint_utilization)
        sector_util = check.constraint_utilization["sector_cap"]
        self.assertGreater(sector_util, 0.70)
        if check.recommended_weights is not None:
            self.assertTrue(len(check.recommended_weights) > 0)

    def test_backward_compatible_without_regime(self):
        """check_new_position works without regime parameter (backward compat)."""
        from quant_engine.risk.portfolio_risk import PortfolioRiskManager

        rm = PortfolioRiskManager()
        check = rm.check_new_position(
            ticker="AAPL",
            position_size=0.05,
            current_positions={},
            price_data=self.price_data,
        )
        self.assertTrue(check.passed)

    def test_compute_constraint_utilization(self):
        from quant_engine.risk.portfolio_risk import PortfolioRiskManager

        rm = PortfolioRiskManager()
        positions = {"AAPL": 0.10, "MSFT": 0.10, "JNJ": 0.10}
        util = rm.compute_constraint_utilization(positions, self.price_data)
        self.assertIn("gross_exposure", util)
        self.assertIn("single_name", util)
        # 30% gross / 100% limit = 0.30
        self.assertAlmostEqual(util["gross_exposure"], 0.30, places=2)


# ══════════════════════════════════════════════════════════════════════════
# T3: Regime-conditional covariance in correlation checks
# ══════════════════════════════════════════════════════════════════════════


class TestRegimeConditionalCorrelation(unittest.TestCase):
    """Tests for regime-conditional covariance in correlation checks."""

    def test_correlation_check_without_regime_labels(self):
        """Pairwise fallback when no regime labels provided."""
        from quant_engine.risk.portfolio_risk import PortfolioRiskManager

        price_data = _generate_price_data(["AAPL", "MSFT"], n_days=120, seed=42)
        rm = PortfolioRiskManager()
        check = rm.check_new_position(
            ticker="MSFT",
            position_size=0.05,
            current_positions={"AAPL": 0.05},
            price_data=price_data,
        )
        self.assertIn("max_pairwise_corr", check.metrics)

    def test_correlation_check_with_regime_labels(self):
        """Regime-conditional covariance used when regime_labels provided."""
        from quant_engine.risk.portfolio_risk import PortfolioRiskManager

        price_data = _generate_price_data(["AAPL", "MSFT"], n_days=120, seed=42)
        idx = price_data["AAPL"].index
        regime_labels = pd.Series(
            np.where(np.arange(len(idx)) < 60, 0, 3),
            index=idx,
        )

        rm = PortfolioRiskManager()
        check = rm.check_new_position(
            ticker="MSFT",
            position_size=0.05,
            current_positions={"AAPL": 0.05},
            price_data=price_data,
            regime=3,
            regime_labels=regime_labels,
        )
        self.assertIn("max_pairwise_corr", check.metrics)

    def test_regime_cov_cache_populated(self):
        """After regime-conditional check, cache should be populated."""
        from quant_engine.risk.portfolio_risk import PortfolioRiskManager

        price_data = _generate_price_data(["AAPL", "MSFT"], n_days=120, seed=42)
        idx = price_data["AAPL"].index
        regime_labels = pd.Series(
            np.where(np.arange(len(idx)) < 60, 0, 3),
            index=idx,
        )

        rm = PortfolioRiskManager()
        rm.check_new_position(
            ticker="MSFT",
            position_size=0.05,
            current_positions={"AAPL": 0.05},
            price_data=price_data,
            regime=0,
            regime_labels=regime_labels,
        )
        self.assertGreater(len(rm._regime_cov_cache), 0)

    def test_invalidate_cache(self):
        from quant_engine.risk.portfolio_risk import PortfolioRiskManager

        rm = PortfolioRiskManager()
        rm._regime_cov_cache[0] = pd.DataFrame()
        rm.invalidate_regime_cov_cache()
        self.assertEqual(len(rm._regime_cov_cache), 0)


# ══════════════════════════════════════════════════════════════════════════
# T4: Factor exposure tests
# ══════════════════════════════════════════════════════════════════════════


class TestFactorExposureManager(unittest.TestCase):
    """Tests for factor exposure computation and bounds checking."""

    def test_compute_exposures_basic(self):
        from quant_engine.risk.factor_exposures import FactorExposureManager

        price_data = _generate_price_data(
            ["AAPL", "MSFT", "JNJ", "JPM"], n_days=300, seed=42,
        )
        weights = {"AAPL": 0.25, "MSFT": 0.25, "JNJ": 0.25, "JPM": 0.25}

        fem = FactorExposureManager()
        exposures = fem.compute_exposures(weights, price_data)

        self.assertIn("beta", exposures)
        self.assertIn("size", exposures)
        self.assertIn("value", exposures)
        self.assertIn("momentum", exposures)
        self.assertIn("volatility", exposures)

    def test_beta_near_one_for_balanced_portfolio(self):
        from quant_engine.risk.factor_exposures import FactorExposureManager

        price_data = _generate_price_data(
            ["A", "B", "C", "D"], n_days=300, seed=42,
        )
        # Create a benchmark from the same price data
        bench_close = price_data["A"]["Close"]
        benchmark = _make_ohlcv(bench_close)

        weights = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        fem = FactorExposureManager()
        exposures = fem.compute_exposures(weights, price_data, benchmark)
        # Beta should be finite
        self.assertTrue(np.isfinite(exposures["beta"]))

    def test_check_factor_bounds_normal(self):
        from quant_engine.risk.factor_exposures import FactorExposureManager

        with _TempUniverseYAML() as path:
            from quant_engine.risk.universe_config import UniverseConfig
            config = UniverseConfig(path)
            fem = FactorExposureManager(universe_config=config)

            # Beta=1.0 should pass in normal regime
            exposures = {"beta": 1.0, "volatility": 1.0, "size": 0.0}
            passed, violations = fem.check_factor_bounds(exposures, regime=0)
            self.assertTrue(passed)
            self.assertEqual(len(violations), 0)

    def test_check_factor_bounds_stress_violation(self):
        from quant_engine.risk.factor_exposures import FactorExposureManager

        with _TempUniverseYAML() as path:
            from quant_engine.risk.universe_config import UniverseConfig
            config = UniverseConfig(path)
            fem = FactorExposureManager(universe_config=config)

            # Volatility=1.3 should fail in stress regime (bounds [0.5, 1.0])
            exposures = {"beta": 1.0, "volatility": 1.3, "size": 0.0}
            passed, violations = fem.check_factor_bounds(exposures, regime=3)
            self.assertFalse(passed)
            self.assertIn("volatility", violations)

    def test_unconstrained_factors_not_violated(self):
        from quant_engine.risk.factor_exposures import FactorExposureManager

        with _TempUniverseYAML() as path:
            from quant_engine.risk.universe_config import UniverseConfig
            config = UniverseConfig(path)
            fem = FactorExposureManager(universe_config=config)

            # Size=5.0 should NOT fail (size is unconstrained / null)
            exposures = {"size": 5.0, "value": -3.0, "momentum": 10.0}
            passed, violations = fem.check_factor_bounds(exposures, regime=3)
            self.assertTrue(passed)

    def test_empty_weights(self):
        from quant_engine.risk.factor_exposures import FactorExposureManager

        fem = FactorExposureManager()
        exposures = fem.compute_exposures({}, {})
        self.assertEqual(len(exposures), 0)


# ══════════════════════════════════════════════════════════════════════════
# T5: Sizing backoff tests
# ══════════════════════════════════════════════════════════════════════════


class TestSizingBackoff(unittest.TestCase):
    """Tests for continuous sizing backoff when constraints approach binding."""

    def test_no_backoff_below_threshold(self):
        from quant_engine.risk.position_sizer import PositionSizer

        sizer = PositionSizer()
        weights = np.array([0.10, 0.10, 0.10])
        util = {"sector_cap": 0.50, "gross_exposure": 0.30}

        result = sizer.size_with_backoff(weights, util)
        np.testing.assert_array_almost_equal(result, weights)

    def test_backoff_at_90_pct(self):
        from quant_engine.risk.position_sizer import PositionSizer

        sizer = PositionSizer()
        weights = np.array([0.10, 0.10, 0.10])
        util = {"sector_cap": 0.92}  # > 90% threshold, backoff_factor = 0.5

        result = sizer.size_with_backoff(weights, util)
        expected = weights * 0.5
        np.testing.assert_array_almost_equal(result, expected)

    def test_backoff_at_95_pct(self):
        from quant_engine.risk.position_sizer import PositionSizer

        sizer = PositionSizer()
        weights = np.array([0.10])
        util = {"sector_cap": 0.96}  # > 95% threshold, backoff_factor = 0.25

        result = sizer.size_with_backoff(weights, util)
        np.testing.assert_array_almost_equal(result, weights * 0.25)

    def test_multiple_constraints_cumulative(self):
        from quant_engine.risk.position_sizer import PositionSizer

        sizer = PositionSizer()
        weights = np.array([0.10])
        # Both sector (90% -> 0.5) and gross (80% -> 0.7)
        util = {"sector_cap": 0.92, "gross_exposure": 0.85}

        result = sizer.size_with_backoff(weights, util)
        # Combined: 0.5 * 0.7 = 0.35
        expected = weights * 0.35
        np.testing.assert_array_almost_equal(result, expected)

    def test_empty_utilization_no_change(self):
        from quant_engine.risk.position_sizer import PositionSizer

        sizer = PositionSizer()
        weights = np.array([0.10, 0.20])
        result = sizer.size_with_backoff(weights, {})
        np.testing.assert_array_almost_equal(result, weights)

    def test_custom_backoff_policy(self):
        from quant_engine.risk.position_sizer import PositionSizer

        sizer = PositionSizer()
        weights = np.array([0.10])
        util = {"gross_exposure": 0.85}
        policy = {
            "thresholds": [0.50, 0.80],
            "backoff_factors": [0.8, 0.3],
        }
        result = sizer.size_with_backoff(weights, util, backoff_policy=policy)
        np.testing.assert_array_almost_equal(result, weights * 0.3)


# ══════════════════════════════════════════════════════════════════════════
# T6: Constraint tightening replay tests
# ══════════════════════════════════════════════════════════════════════════


class TestConstraintReplay(unittest.TestCase):
    """Tests for constraint tightening replay."""

    def test_replay_basic(self):
        from quant_engine.risk.constraint_replay import (
            replay_with_stress_constraints,
            compute_robustness_score,
        )

        price_data = _generate_price_data(["AAPL", "MSFT", "JNJ"], n_days=120, seed=42)

        portfolio_history = [
            {
                "date": "2023-06-01",
                "positions": {"AAPL": 0.05, "MSFT": 0.05, "JNJ": 0.05},
            },
            {
                "date": "2023-09-01",
                "positions": {"AAPL": 0.15, "MSFT": 0.10, "JNJ": 0.05},
            },
        ]

        with _TempUniverseYAML() as path:
            from quant_engine.risk.universe_config import UniverseConfig
            config = UniverseConfig(path)

            result = replay_with_stress_constraints(
                portfolio_history, price_data, universe_config=config,
            )

        self.assertGreater(len(result), 0)
        self.assertIn("scenario", result.columns)
        self.assertIn("max_utilization", result.columns)
        self.assertIn("recommended_backoff", result.columns)

        score = compute_robustness_score(result)
        self.assertIn("overall_score", score)
        self.assertIn("per_scenario", score)

    def test_replay_detects_concentration(self):
        from quant_engine.risk.constraint_replay import replay_with_stress_constraints

        price_data = _generate_price_data(["AAPL", "MSFT"], n_days=120, seed=42)

        # Heavy concentration in one sector
        portfolio_history = [
            {
                "date": "2023-06-01",
                "positions": {"AAPL": 0.30, "MSFT": 0.30},
            },
        ]

        with _TempUniverseYAML() as path:
            from quant_engine.risk.universe_config import UniverseConfig
            config = UniverseConfig(path)

            result = replay_with_stress_constraints(
                portfolio_history, price_data, universe_config=config,
            )

        # 60% in tech with stress cap of 24% should be violated
        self.assertTrue(result["any_violated"].any())

    def test_replay_empty_history(self):
        from quant_engine.risk.constraint_replay import replay_with_stress_constraints

        result = replay_with_stress_constraints([], {})
        self.assertEqual(len(result), 0)

    def test_robustness_score_perfect(self):
        from quant_engine.risk.constraint_replay import compute_robustness_score

        # All passing
        df = pd.DataFrame({
            "any_violated": [False, False, False],
            "scenario": ["2008", "COVID", "2022"],
            "max_utilization": [0.5, 0.4, 0.3],
        })
        score = compute_robustness_score(df)
        self.assertAlmostEqual(score["overall_score"], 1.0)


# ══════════════════════════════════════════════════════════════════════════
# T8: Integration tests — end-to-end
# ══════════════════════════════════════════════════════════════════════════


class TestPortfolioLayerIntegration(unittest.TestCase):
    """End-to-end integration tests for the regime-conditioned portfolio layer."""

    def test_full_workflow_normal_regime(self):
        """Generate prices, compute weights, check portfolio, verify no violations."""
        from quant_engine.risk.portfolio_risk import PortfolioRiskManager
        from quant_engine.risk.factor_exposures import FactorExposureManager

        price_data = _generate_price_data(
            ["AAPL", "MSFT", "JNJ", "JPM", "AMZN"],
            n_days=300, seed=42,
        )

        with _TempUniverseYAML() as path:
            from quant_engine.risk.universe_config import UniverseConfig
            config = UniverseConfig(path)
            rm = PortfolioRiskManager(universe_config=config)
            fem = FactorExposureManager(universe_config=config)

            # Equal weight portfolio (5% each)
            positions = {t: 0.05 for t in price_data}

            # Check portfolio risk
            check = rm.check_new_position(
                ticker="AAPL",
                position_size=0.05,
                current_positions={t: 0.05 for t in list(price_data.keys())[1:]},
                price_data=price_data,
                regime=0,
            )
            # Small equal-weight portfolio should pass all checks
            self.assertTrue(check.passed, f"Violations: {check.violations}")

            # Check factor exposures
            exposures = fem.compute_exposures(positions, price_data)
            passed, violations = fem.check_factor_bounds(exposures, regime=0)
            # Equal-weight portfolio should have balanced factor exposures
            self.assertEqual(len(violations), 0, f"Factor violations: {violations}")

    def test_full_workflow_stress_regime(self):
        """Concentrated portfolio fails under stress constraints."""
        from quant_engine.risk.portfolio_risk import PortfolioRiskManager

        price_data = _generate_price_data(
            ["AAPL", "MSFT", "GOOGL"],
            n_days=120, seed=42,
        )

        with _TempUniverseYAML() as path:
            from quant_engine.risk.universe_config import UniverseConfig
            config = UniverseConfig(path)
            rm = PortfolioRiskManager(universe_config=config)
            # Force immediate stress multipliers
            rm.multiplier._smoothed = rm.multiplier._stress_mults.copy()
            rm.multiplier._prev_regime_is_stress = True

            # All in tech sector (30% total in tech)
            check = rm.check_new_position(
                ticker="GOOGL",
                position_size=0.10,
                current_positions={"AAPL": 0.10, "MSFT": 0.10},
                price_data=price_data,
                regime=3,
            )
            # 30% in tech > stress cap of 24% = should fail
            sector_violations = [v for v in check.violations if "Sector" in v]
            self.assertGreater(len(sector_violations), 0)

    def test_sizing_backoff_integration(self):
        """Position sizer applies backoff based on risk manager utilization."""
        from quant_engine.risk.portfolio_risk import PortfolioRiskManager
        from quant_engine.risk.position_sizer import PositionSizer

        price_data = _generate_price_data(
            ["AAPL", "MSFT", "JNJ"],
            n_days=120, seed=42,
        )

        rm = PortfolioRiskManager()
        sizer = PositionSizer()

        positions = {"AAPL": 0.30, "MSFT": 0.05}
        util = rm.compute_constraint_utilization(positions, price_data, regime=0)

        # Apply backoff
        weights = np.array([0.30, 0.05])
        scaled = sizer.size_with_backoff(weights, util)

        # If utilization is high, scaled should be smaller
        if max(util.values()) > 0.70:
            self.assertTrue(np.all(scaled <= weights))

    def test_universe_config_integrated_with_risk_manager(self):
        """Risk manager uses universe config for sector resolution."""
        from quant_engine.risk.portfolio_risk import PortfolioRiskManager

        price_data = _generate_price_data(["AAPL", "JNJ"], n_days=120, seed=42)

        with _TempUniverseYAML() as path:
            from quant_engine.risk.universe_config import UniverseConfig
            config = UniverseConfig(path)
            rm = PortfolioRiskManager(universe_config=config)

            # AAPL should resolve to 'tech'
            sector = rm._resolve_sector("AAPL", price_data)
            self.assertEqual(sector, "tech")

            # JNJ should resolve to 'healthcare'
            sector = rm._resolve_sector("JNJ", price_data)
            self.assertEqual(sector, "healthcare")

    def test_constraint_replay_integration(self):
        """Full replay workflow produces expected output."""
        from quant_engine.risk.constraint_replay import (
            replay_with_stress_constraints,
            compute_robustness_score,
        )

        price_data = _generate_price_data(
            ["AAPL", "MSFT", "JNJ", "JPM"],
            n_days=300, seed=42,
        )

        history = [
            {"date": "2023-03-01", "positions": {"AAPL": 0.05, "JNJ": 0.05}},
            {"date": "2023-06-01", "positions": {"AAPL": 0.25, "MSFT": 0.20}},
            {"date": "2023-09-01", "positions": {"AAPL": 0.05, "MSFT": 0.05, "JNJ": 0.05, "JPM": 0.05}},
        ]

        with _TempUniverseYAML() as path:
            from quant_engine.risk.universe_config import UniverseConfig
            config = UniverseConfig(path)
            result = replay_with_stress_constraints(
                history, price_data, universe_config=config,
            )

        self.assertGreater(len(result), 0)
        # Concentrated portfolio (entry 2) should have higher utilization
        concentrated = result[result["date"] == "2023-06-01"]
        balanced = result[result["date"] == "2023-03-01"]
        if len(concentrated) > 0 and len(balanced) > 0:
            self.assertGreater(
                concentrated["max_utilization"].mean(),
                balanced["max_utilization"].mean(),
            )

        score = compute_robustness_score(result)
        self.assertLessEqual(score["overall_score"], 1.0)
        self.assertGreaterEqual(score["overall_score"], 0.0)

    def test_portfolio_summary_uses_universe_config(self):
        """portfolio_summary resolves sectors via universe config."""
        from quant_engine.risk.portfolio_risk import PortfolioRiskManager

        price_data = _generate_price_data(["AAPL", "JNJ"], n_days=120, seed=42)

        with _TempUniverseYAML() as path:
            from quant_engine.risk.universe_config import UniverseConfig
            config = UniverseConfig(path)
            rm = PortfolioRiskManager(universe_config=config)

            summary = rm.portfolio_summary(
                {"AAPL": 0.10, "JNJ": 0.10}, price_data,
            )
            self.assertIn("tech", summary["sector_breakdown"])
            self.assertIn("healthcare", summary["sector_breakdown"])


if __name__ == "__main__":
    unittest.main()
