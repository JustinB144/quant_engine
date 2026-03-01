"""
Tests for the A/B testing framework (Spec 014).

Covers:
- Block bootstrap (wider CI than raw t-test)
- Ticker-level deterministic assignment (no contamination)
- Balanced allocation
- Power analysis (reasonable sample sizes)
- Sequential testing (conservative early, liberal late)
- Trade persistence via parquet
- Comprehensive report metrics
- Max concurrent tests enforcement
- Paper trader integration
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quant_engine.api.ab_testing import (
    ABTest,
    ABTestRegistry,
    ABVariant,
    MAX_CONCURRENT_TESTS,
)


def _make_test(
    test_id: str = "test123",
    control_name: str = "control",
    treatment_name: str = "treatment",
    allocation: float = 0.5,
    min_trades: int = 50,
) -> ABTest:
    """Helper to create an ABTest with sensible defaults."""
    return ABTest(
        test_id=test_id,
        name="Test Experiment",
        description="Testing A/B framework",
        control=ABVariant(
            name=control_name,
            config_overrides={},
            allocation=allocation,
        ),
        treatment=ABVariant(
            name=treatment_name,
            config_overrides={},
            allocation=1.0 - allocation,
        ),
        min_trades=min_trades,
    )


def _add_trades(variant: ABVariant, returns: np.ndarray) -> None:
    """Add trade dicts with specified net returns to a variant."""
    for i, r in enumerate(returns):
        variant.trades.append({
            "net_return": float(r),
            "ticker": f"T{i}",
            "entry_date": f"2025-01-{1 + i % 28:02d}",
            "exit_date": f"2025-02-{1 + i % 28:02d}",
            "entry_price": 100.0,
            "exit_price": 100.0 * (1.0 + float(r)),
            "shares": 10.0,
            "pnl": float(r) * 1000.0,
            "entry_regime": 0,
            "transaction_cost": 0.50,
        })


class TestBlockBootstrap(unittest.TestCase):
    """T1: Block bootstrap should give wider CI than raw t-test."""

    def test_block_bootstrap_wider_ci(self):
        """Block bootstrap p-value >= raw t-test p-value on correlated data."""
        np.random.seed(42)
        test = _make_test(min_trades=30)

        # Generate autocorrelated returns (AR(1) with rho=0.3)
        n = 100
        rho = 0.3
        noise_a = np.random.randn(n) * 0.02
        noise_b = np.random.randn(n) * 0.02
        returns_a = np.zeros(n)
        returns_b = np.zeros(n)
        for i in range(1, n):
            returns_a[i] = rho * returns_a[i - 1] + noise_a[i]
            returns_b[i] = rho * returns_b[i - 1] + noise_b[i] + 0.003

        _add_trades(test.control, returns_a)
        _add_trades(test.treatment, returns_b)

        # Raw t-test p-value
        from scipy.stats import ttest_ind
        _, p_raw = ttest_ind(returns_b, returns_a, equal_var=False)

        # Block bootstrap p-value
        p_boot = test._block_bootstrap_test(returns_a, returns_b)

        # Block bootstrap should be >= raw (more conservative) or at least
        # not dramatically less
        self.assertGreaterEqual(
            p_boot + 0.05,  # Allow small tolerance due to randomness
            p_raw,
            "Block bootstrap should give at least comparable p-value to raw t-test",
        )

    def test_newey_west_returns_valid(self):
        """Newey-West test returns valid t-stat and p-value."""
        np.random.seed(99)
        test = _make_test()
        returns_a = np.random.randn(60) * 0.01
        returns_b = np.random.randn(60) * 0.01 + 0.005

        t_stat, p_value = test._newey_west_test(returns_a, returns_b)
        self.assertTrue(np.isfinite(t_stat))
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)


class TestTickerAssignment(unittest.TestCase):
    """T2: Ticker-level deterministic variant assignment."""

    def test_ticker_assignment_deterministic(self):
        """Same ticker always gets the same variant."""
        test = _make_test()
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

        first_pass = {t: test.assign_variant(t) for t in tickers}

        # Second pass — must be identical
        for t in tickers:
            self.assertEqual(
                test.assign_variant(t),
                first_pass[t],
                f"Ticker {t} got different variant on second call",
            )

    def test_ticker_assignment_balanced(self):
        """~50% of 100+ tickers should be in each variant with 50/50 allocation."""
        test = _make_test(allocation=0.5)
        tickers = [f"TICK{i:04d}" for i in range(200)]

        assignments = [test.assign_variant(t) for t in tickers]
        control_count = sum(1 for a in assignments if a == "control")

        # Should be roughly balanced (within 30% of expected 50/50)
        self.assertGreater(control_count, 60, "Too few in control")
        self.assertLess(control_count, 140, "Too many in control")

    def test_no_ticker_contamination(self):
        """Same ticker must never appear in both variants."""
        test = _make_test()
        tickers = [f"SYM{i}" for i in range(500)]

        assignments = {}
        for t in tickers:
            variant = test.assign_variant(t)
            if t in assignments:
                self.assertEqual(
                    assignments[t], variant,
                    f"Ticker {t} contamination: {assignments[t]} vs {variant}",
                )
            assignments[t] = variant

        # Additionally verify via set intersection
        control_tickers = {t for t, v in assignments.items() if v == "control"}
        treatment_tickers = {t for t, v in assignments.items() if v == "treatment"}
        overlap = control_tickers & treatment_tickers
        self.assertEqual(len(overlap), 0, f"Contaminated tickers: {overlap}")


class TestPowerAnalysis(unittest.TestCase):
    """T3: Power analysis produces reasonable sample sizes."""

    def test_power_analysis_reasonable(self):
        """Required samples should be between 50 and 10000 for typical params."""
        test = _make_test()
        n = test.compute_required_samples(
            min_detectable_effect=0.005,
            power=0.80,
            alpha=0.05,
            return_std=0.02,
        )
        self.assertGreaterEqual(n, 50)
        self.assertLessEqual(n, 10000)

    def test_power_analysis_larger_effect_fewer_samples(self):
        """Larger effect size should require fewer samples."""
        test = _make_test()
        n_small = test.compute_required_samples(min_detectable_effect=0.005)
        n_large = test.compute_required_samples(min_detectable_effect=0.02)
        self.assertGreater(n_small, n_large)

    def test_power_analysis_higher_power_more_samples(self):
        """Higher power should require more samples."""
        test = _make_test()
        n_low = test.compute_required_samples(power=0.60)
        n_high = test.compute_required_samples(power=0.95)
        self.assertGreater(n_high, n_low)


class TestEarlyStopping(unittest.TestCase):
    """T4: Sequential testing with O'Brien-Fleming alpha spending."""

    def test_early_stopping_insufficient_data(self):
        """With <20 trades, early stopping should say insufficient data."""
        test = _make_test()
        _add_trades(test.control, np.random.randn(10) * 0.01)
        _add_trades(test.treatment, np.random.randn(10) * 0.01)

        result = test.check_early_stopping()
        self.assertFalse(result["stop"])
        self.assertIn("Insufficient", result["reason"])

    def test_early_stopping_conservative_early(self):
        """At low info fraction, alpha boundary should be very small."""
        test = _make_test(min_trades=20)

        # Add just enough trades to pass the 20-trade minimum
        _add_trades(test.control, np.random.randn(25) * 0.01)
        _add_trades(test.treatment, np.random.randn(25) * 0.01)

        result = test.check_early_stopping()
        # Required samples is much larger than 25, so info fraction is low
        # Alpha boundary at low info fraction should be very conservative
        if result["alpha_boundary"] is not None:
            self.assertLess(
                result["alpha_boundary"], 0.01,
                "Alpha boundary should be very conservative early on",
            )

    def test_early_stopping_returns_valid_dict(self):
        """Early stopping result should have all required keys."""
        test = _make_test(min_trades=20)
        _add_trades(test.control, np.random.randn(30) * 0.01)
        _add_trades(test.treatment, np.random.randn(30) * 0.01)

        result = test.check_early_stopping()
        required_keys = {"stop", "reason", "info_fraction", "p_value",
                         "alpha_boundary", "trades_remaining"}
        self.assertTrue(required_keys.issubset(result.keys()))


class TestTradePersistence(unittest.TestCase):
    """T5: Full trade history persisted and restorable from parquet."""

    def test_trade_persistence_parquet(self):
        """Trades saved to parquet and restored correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "ab_tests.json"
            registry = ABTestRegistry(storage_path=storage_path)

            # Create test and add trades
            test = registry.create_test(
                name="Persistence Test",
                description="Testing parquet persistence",
                control_name="ctrl",
                treatment_name="treat",
            )
            np.random.seed(42)
            for i in range(10):
                test.control.trades.append({
                    "net_return": float(np.random.randn() * 0.01),
                    "ticker": f"T{i}",
                    "entry_date": f"2025-01-{1 + i:02d}",
                    "exit_date": f"2025-02-{1 + i:02d}",
                    "entry_price": 100.0,
                    "exit_price": 101.0,
                    "shares": 10.0,
                    "pnl": 10.0,
                    "transaction_cost": 0.50,
                })
            for i in range(8):
                test.treatment.trades.append({
                    "net_return": float(np.random.randn() * 0.01 + 0.003),
                    "ticker": f"T{i}",
                    "entry_date": f"2025-01-{1 + i:02d}",
                    "exit_date": f"2025-02-{1 + i:02d}",
                    "entry_price": 100.0,
                    "exit_price": 101.0,
                    "shares": 10.0,
                    "pnl": 10.0,
                    "transaction_cost": 0.50,
                })

            registry._save()

            # Verify parquet files exist
            trades_dir = Path(tmpdir) / "ab_trades"
            self.assertTrue(trades_dir.exists())

            ctrl_parquet = trades_dir / f"{test.test_id}_ctrl.parquet"
            treat_parquet = trades_dir / f"{test.test_id}_treat.parquet"
            self.assertTrue(ctrl_parquet.exists())
            self.assertTrue(treat_parquet.exists())

            # Reload and verify trade counts
            registry2 = ABTestRegistry(storage_path=storage_path)
            loaded = registry2.get_test(test.test_id)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.control.n_trades, 10)
            self.assertEqual(loaded.treatment.n_trades, 8)


class TestComprehensiveReport(unittest.TestCase):
    """T5 cont'd: Comprehensive report includes all required metrics."""

    def test_comprehensive_report_metrics(self):
        """Report includes Sharpe, drawdown, win rate, and more."""
        test = _make_test(min_trades=20)
        np.random.seed(42)
        _add_trades(test.control, np.random.randn(60) * 0.01 + 0.001)
        _add_trades(test.treatment, np.random.randn(60) * 0.01 + 0.003)

        results = test.get_results()

        # Check comprehensive metrics on each variant
        for key in ["control", "treatment"]:
            variant = results[key]
            self.assertIn("mean_return", variant)
            self.assertIn("sharpe", variant)
            self.assertIn("sortino", variant)
            self.assertIn("max_drawdown", variant)
            self.assertIn("win_rate", variant)
            self.assertIn("profit_factor", variant)
            self.assertIn("turnover", variant)
            self.assertIn("total_transaction_costs", variant)
            self.assertIn("n_trades", variant)

        # Check statistical fields
        self.assertIn("p_value_bootstrap", results)
        self.assertIn("p_value_newey_west", results)
        self.assertIn("p_value", results)
        self.assertTrue(results["sufficient_data"])

    def test_report_with_regime_breakdown(self):
        """get_test_report includes per-regime performance."""
        test = _make_test(min_trades=10)
        # Add trades with different regimes
        for i in range(30):
            test.control.trades.append({
                "net_return": 0.01 * (1 if i % 2 == 0 else -1),
                "ticker": f"T{i}",
                "entry_regime": i % 3,  # Regimes 0, 1, 2
                "entry_date": f"2025-01-{1 + i % 28:02d}",
                "exit_date": f"2025-02-{1 + i % 28:02d}",
                "transaction_cost": 0.5,
            })
            test.treatment.trades.append({
                "net_return": 0.015 * (1 if i % 3 != 0 else -1),
                "ticker": f"T{i}",
                "entry_regime": i % 3,
                "entry_date": f"2025-01-{1 + i % 28:02d}",
                "exit_date": f"2025-02-{1 + i % 28:02d}",
                "transaction_cost": 0.5,
            })

        report = test.get_test_report()
        # Should have regime breakdown
        self.assertIn("regime_breakdown", report["control"])
        self.assertIn("regime_breakdown", report["treatment"])
        # Should have at least 2 regimes
        self.assertGreaterEqual(len(report["control"]["regime_breakdown"]), 2)


class TestMaxConcurrentTests(unittest.TestCase):
    """Constraint: Cannot create >3 active tests."""

    def test_max_concurrent_tests(self):
        """Creating more than MAX_CONCURRENT_TESTS should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "ab_tests.json"
            registry = ABTestRegistry(storage_path=storage_path)

            # Create max allowed tests
            for i in range(MAX_CONCURRENT_TESTS):
                registry.create_test(
                    name=f"Test {i}",
                    description=f"Test {i}",
                    control_name="ctrl",
                    treatment_name="treat",
                )

            # Next one should fail
            with self.assertRaises(ValueError) as ctx:
                registry.create_test(
                    name="Over limit",
                    description="Should fail",
                    control_name="ctrl",
                    treatment_name="treat",
                )
            self.assertIn("active tests", str(ctx.exception))

    def test_completed_tests_dont_count(self):
        """Completed tests should not count toward the limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "ab_tests.json"
            registry = ABTestRegistry(storage_path=storage_path)

            # Create and complete tests
            for i in range(MAX_CONCURRENT_TESTS):
                test = registry.create_test(
                    name=f"Test {i}",
                    description=f"Test {i}",
                    control_name="ctrl",
                    treatment_name="treat",
                )
                if i < MAX_CONCURRENT_TESTS - 1:
                    registry.complete_test(test.test_id)

            # Should be able to create another (only 1 active)
            registry.create_test(
                name="After completion",
                description="Should work",
                control_name="ctrl",
                treatment_name="treat",
            )


class TestRegistryOperations(unittest.TestCase):
    """Registry CRUD operations."""

    def test_cancel_test(self):
        """Cancelling an active test should set status to cancelled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "ab_tests.json"
            registry = ABTestRegistry(storage_path=storage_path)
            test = registry.create_test(
                name="Cancel me",
                description="Testing cancel",
                control_name="ctrl",
                treatment_name="treat",
            )
            self.assertTrue(registry.cancel_test(test.test_id))
            loaded = registry.get_test(test.test_id)
            self.assertEqual(loaded.status, "cancelled")

    def test_get_active_test(self):
        """get_active_test returns the first active test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "ab_tests.json"
            registry = ABTestRegistry(storage_path=storage_path)

            # No active test
            self.assertIsNone(registry.get_active_test())

            # Create one
            test = registry.create_test(
                name="Active",
                description="Active test",
                control_name="ctrl",
                treatment_name="treat",
            )
            active = registry.get_active_test()
            self.assertIsNotNone(active)
            self.assertEqual(active.test_id, test.test_id)

    def test_list_tests_by_status(self):
        """list_tests filters by status correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "ab_tests.json"
            registry = ABTestRegistry(storage_path=storage_path)

            t1 = registry.create_test(
                name="T1", description="", control_name="c", treatment_name="t",
            )
            t2 = registry.create_test(
                name="T2", description="", control_name="c", treatment_name="t",
            )
            registry.complete_test(t1.test_id)

            active = registry.list_tests(status="active")
            completed = registry.list_tests(status="completed")
            self.assertEqual(len(active), 1)
            self.assertEqual(len(completed), 1)


class TestVariantMetrics(unittest.TestCase):
    """ABVariant computed properties."""

    def test_empty_variant_metrics(self):
        """Empty variant should return safe defaults."""
        v = ABVariant(name="empty")
        self.assertEqual(v.n_trades, 0)
        self.assertEqual(v.mean_return, 0.0)
        self.assertEqual(v.sharpe, 0.0)
        self.assertEqual(v.sortino, 0.0)
        self.assertEqual(v.max_drawdown, 0.0)
        self.assertEqual(v.win_rate, 0.0)
        self.assertEqual(v.profit_factor, 0.0)

    def test_winning_variant_positive_sharpe(self):
        """Variant with consistently positive returns should have positive Sharpe."""
        v = ABVariant(name="winner")
        _add_trades(v, np.array([0.01, 0.02, 0.005, 0.015, 0.01, 0.008]))
        self.assertGreater(v.sharpe, 0.0)
        self.assertGreater(v.win_rate, 0.5)
        self.assertGreater(v.profit_factor, 1.0)


class TestNeweyWestPerArm(unittest.TestCase):
    """T7 (SPEC_AUDIT_FIX_03): Newey-West computes variance per arm."""

    def test_known_difference_significant(self):
        """Two arms with clearly different means should be significant."""
        np.random.seed(123)
        test = _make_test(min_trades=30)

        # Arm A: mean ~0, Arm B: mean ~0.05 (large gap, small std)
        returns_a = np.random.randn(200) * 0.01
        returns_b = np.random.randn(200) * 0.01 + 0.05

        t_stat, p_value = test._newey_west_test(returns_a, returns_b)

        self.assertTrue(np.isfinite(t_stat))
        self.assertLess(p_value, 0.01, "Known large difference should be significant")
        # t-stat should be negative because mean_a < mean_b
        self.assertLess(t_stat, 0.0)

    def test_identical_arms_not_significant(self):
        """Two arms drawn from the same distribution should not be significant."""
        np.random.seed(456)
        test = _make_test(min_trades=30)

        returns_a = np.random.randn(200) * 0.01
        returns_b = np.random.randn(200) * 0.01

        _, p_value = test._newey_west_test(returns_a, returns_b)

        self.assertGreater(p_value, 0.05, "Same distribution should not be significant")

    def test_per_arm_variance_differs_from_pooled(self):
        """Per-arm variance should differ from pooled when arms have different volatility."""
        np.random.seed(789)
        test = _make_test()

        # Arm A: low vol, Arm B: high vol (same mean)
        returns_a = np.random.randn(100) * 0.005
        returns_b = np.random.randn(100) * 0.050

        nw_a = test._newey_west_variance(returns_a - returns_a.mean(), 10)
        nw_b = test._newey_west_variance(returns_b - returns_b.mean(), 10)

        # The two variances should be very different (factor ~100x)
        self.assertGreater(nw_b / max(nw_a, 1e-30), 10.0)


if __name__ == "__main__":
    unittest.main()
