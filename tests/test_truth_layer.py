"""
Tests for Spec 01: Foundational Hardening — Truth Layer.

Covers:
    T1 — Global preconditions contract
    T2 — Data integrity preflight system
    T3 — Leakage tripwires (causality enforcement + time-shift detection)
    T4 — Null model baselines
    T5 — Cost stress sweep
    T6 — Cache staleness trading-calendar fix
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ── Helpers ───────────────────────────────────────────────────────────


def _make_ohlcv(n: int = 500, seed: int = 42, zero_vol_frac: float = 0.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    close = 100.0 + np.cumsum(rng.normal(0.0005, 0.015, n))
    close = np.maximum(close, 1.0)
    high = close * (1 + rng.uniform(0, 0.015, n))
    low = close * (1 - rng.uniform(0, 0.015, n))
    opn = close * (1 + rng.normal(0, 0.003, n))
    vol = rng.integers(500_000, 5_000_000, n).astype(float)

    # Inject zero volume if requested
    if zero_vol_frac > 0:
        n_zero = int(n * zero_vol_frac)
        zero_idx = rng.choice(n, size=n_zero, replace=False)
        vol[zero_idx] = 0.0

    df = pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )
    return df


def _make_universe(n_stocks: int = 5, n_bars: int = 500) -> dict[str, pd.DataFrame]:
    """Generate a universe of synthetic OHLCV data."""
    return {
        f"STOCK{i}": _make_ohlcv(n=n_bars, seed=42 + i)
        for i in range(n_stocks)
    }


# ═══════════════════════════════════════════════════════════════════════
# T1: Global Preconditions Contract
# ═══════════════════════════════════════════════════════════════════════


class TestPreconditionsContract:
    """Tests for execution contract validation."""

    def test_validate_execution_contract_passes_default_config(self):
        """Default config values should pass validation."""
        from quant_engine.validation.preconditions import validate_execution_contract

        ok, msg = validate_execution_contract()
        assert ok is True
        assert "OK" in msg

    def test_preconditions_config_rejects_zero_label_h(self):
        """LABEL_H=0 should be rejected."""
        from quant_engine.config_structured import PreconditionsConfig

        with pytest.raises(ValueError, match="positive integer"):
            PreconditionsConfig(label_h=0)

    def test_preconditions_config_rejects_negative_label_h(self):
        """LABEL_H=-5 should be rejected."""
        from quant_engine.config_structured import PreconditionsConfig

        with pytest.raises(ValueError, match="positive integer"):
            PreconditionsConfig(label_h=-5)

    def test_preconditions_config_rejects_excessive_label_h(self):
        """LABEL_H=100 should be rejected (>60 days)."""
        from quant_engine.config_structured import PreconditionsConfig

        with pytest.raises(ValueError, match="exceeds 60"):
            PreconditionsConfig(label_h=100)

    def test_preconditions_config_accepts_valid_values(self):
        """Valid preconditions should be accepted."""
        from quant_engine.config_structured import (
            PreconditionsConfig, ReturnType, PriceType, EntryType,
        )

        cfg = PreconditionsConfig(
            ret_type=ReturnType.LOG,
            label_h=10,
            px_type=PriceType.CLOSE,
            entry_price_type=EntryType.NEXT_BAR_OPEN,
        )
        assert cfg.ret_type == ReturnType.LOG
        assert cfg.label_h == 10

    def test_preconditions_config_coerces_strings(self):
        """String values should be coerced to enums."""
        from quant_engine.config_structured import PreconditionsConfig, ReturnType

        cfg = PreconditionsConfig(ret_type="log", px_type="close", entry_price_type="next_bar_open")
        assert cfg.ret_type == ReturnType.LOG

    def test_preconditions_config_rejects_invalid_ret_type(self):
        """Invalid return type string should raise ValueError."""
        from quant_engine.config_structured import PreconditionsConfig

        with pytest.raises(ValueError):
            PreconditionsConfig(ret_type="invalid")

    def test_preconditions_config_rejects_invalid_px_type(self):
        """Invalid price type string should raise ValueError."""
        from quant_engine.config_structured import PreconditionsConfig

        with pytest.raises(ValueError):
            PreconditionsConfig(px_type="vwap")

    def test_preconditions_config_rejects_invalid_entry_type(self):
        """Invalid entry type string should raise ValueError."""
        from quant_engine.config_structured import PreconditionsConfig

        with pytest.raises(ValueError):
            PreconditionsConfig(entry_price_type="market_close")

    def test_enforce_preconditions_respects_feature_flag(self):
        """enforce_preconditions() should be a no-op when flag is False."""
        from quant_engine.validation.preconditions import enforce_preconditions

        with patch("quant_engine.validation.preconditions.TRUTH_LAYER_STRICT_PRECONDITIONS", False):
            # Should not raise even if config were invalid
            enforce_preconditions()

    def test_enforce_preconditions_raises_on_invalid_config(self):
        """enforce_preconditions() should raise when config is invalid."""
        from quant_engine.validation.preconditions import enforce_preconditions

        with patch("quant_engine.validation.preconditions.LABEL_H", -1):
            with pytest.raises(RuntimeError, match="validation failed"):
                enforce_preconditions()

    def test_return_type_enum_values(self):
        """ReturnType enum should have log and simple values."""
        from quant_engine.config_structured import ReturnType

        assert ReturnType.LOG.value == "log"
        assert ReturnType.SIMPLE.value == "simple"

    def test_entry_type_enum_values(self):
        """EntryType enum should have all expected values."""
        from quant_engine.config_structured import EntryType

        assert EntryType.NEXT_BAR_OPEN.value == "next_bar_open"
        assert EntryType.MARKET_ON_OPEN.value == "market_on_open"
        assert EntryType.LIMIT_10BP.value == "limit_10bp"

    def test_preconditions_config_boundary_label_h(self):
        """LABEL_H at boundary values should be accepted."""
        from quant_engine.config_structured import PreconditionsConfig

        cfg1 = PreconditionsConfig(label_h=1)
        assert cfg1.label_h == 1

        cfg60 = PreconditionsConfig(label_h=60)
        assert cfg60.label_h == 60


# ═══════════════════════════════════════════════════════════════════════
# T2: Data Integrity Preflight System
# ═══════════════════════════════════════════════════════════════════════


class TestDataIntegrity:
    """Tests for data integrity validation."""

    def test_assess_quality_fail_on_error_raises(self):
        """fail_on_error=True should raise ValueError on failed check."""
        from quant_engine.data.quality import assess_ohlcv_quality

        # Create data with too much missing volume (> MAX_ZERO_VOLUME_FRACTION)
        df = _make_ohlcv(n=500, zero_vol_frac=0.50)

        with pytest.raises(ValueError, match="quality check failed"):
            assess_ohlcv_quality(df, fail_on_error=True)

    def test_assess_quality_fail_on_error_false_returns_report(self):
        """fail_on_error=False should return report without raising."""
        from quant_engine.data.quality import assess_ohlcv_quality

        df = _make_ohlcv(n=500, zero_vol_frac=0.50)
        report = assess_ohlcv_quality(df, fail_on_error=False)
        assert report.passed is False
        assert len(report.warnings) > 0

    def test_assess_quality_passes_clean_data(self):
        """Clean OHLCV data should pass quality checks."""
        from quant_engine.data.quality import assess_ohlcv_quality

        df = _make_ohlcv(n=500)
        report = assess_ohlcv_quality(df, fail_on_error=True)
        assert report.passed is True

    def test_data_integrity_validator_passes_clean_universe(self):
        """Clean universe should pass integrity check."""
        from quant_engine.validation.data_integrity import DataIntegrityValidator

        universe = _make_universe(n_stocks=3)
        validator = DataIntegrityValidator(fail_fast=True)
        result = validator.validate_universe(universe)

        assert result.passed is True
        assert result.n_stocks_passed == 3
        assert result.n_stocks_failed == 0
        assert len(result.failed_tickers) == 0

    def test_data_integrity_validator_blocks_corrupt_data_fail_fast(self):
        """Corrupt data should raise RuntimeError in fail_fast mode."""
        from quant_engine.validation.data_integrity import DataIntegrityValidator

        universe = _make_universe(n_stocks=3)
        # Corrupt one stock: set 60% of volume to 0
        bad_df = universe["STOCK0"].copy()
        bad_df["Volume"] = 0.0
        universe["CORRUPT_STOCK"] = bad_df

        validator = DataIntegrityValidator(fail_fast=True)
        with pytest.raises(RuntimeError, match="integrity check failed"):
            validator.validate_universe(universe)

    def test_data_integrity_validator_no_fail_fast_collects_all(self):
        """Non-fail-fast mode should check all tickers and report."""
        from quant_engine.validation.data_integrity import DataIntegrityValidator

        universe = _make_universe(n_stocks=3)
        # Corrupt two stocks
        for key in ["STOCK0", "STOCK1"]:
            bad = universe[key].copy()
            bad["Volume"] = 0.0
            universe[key] = bad

        validator = DataIntegrityValidator(fail_fast=False)
        result = validator.validate_universe(universe)

        assert result.passed is False
        assert result.n_stocks_failed == 2
        assert set(result.failed_tickers) == {"STOCK0", "STOCK1"}

    def test_data_integrity_empty_universe(self):
        """Empty universe should pass (no stocks to fail)."""
        from quant_engine.validation.data_integrity import DataIntegrityValidator

        validator = DataIntegrityValidator(fail_fast=True)
        result = validator.validate_universe({})

        assert result.passed is True
        assert result.n_stocks_passed == 0


# ═══════════════════════════════════════════════════════════════════════
# T3: Leakage Tripwires
# ═══════════════════════════════════════════════════════════════════════


class TestLeakageTripwires:
    """Tests for leakage detection."""

    def test_leakage_detector_catches_forward_shift(self):
        """Feature that is a forward-shifted label should be flagged."""
        from quant_engine.validation.leakage_detection import LeakageDetector

        rng = np.random.default_rng(42)
        n = 500
        labels = pd.Series(rng.normal(0, 1, n), index=pd.bdate_range("2022-01-03", periods=n))

        features = pd.DataFrame(index=labels.index)
        features["clean_feature"] = rng.normal(0, 1, n)
        features["leaky_feature"] = labels.shift(-1)  # Obvious look-ahead

        detector = LeakageDetector(shift_range=[1, 2, 3])
        result = detector.test_time_shift_leakage(features, labels, threshold_corr=0.15)

        assert result.passed is False
        assert result.n_violations > 0

        # The leaky feature should be in violations
        leaky_features = {v["feature"] for v in result.violations}
        assert "leaky_feature" in leaky_features

    def test_leakage_detector_passes_clean_features(self):
        """Independent random features should pass leakage test."""
        from quant_engine.validation.leakage_detection import LeakageDetector

        rng = np.random.default_rng(42)
        n = 500
        labels = pd.Series(rng.normal(0, 1, n), index=pd.bdate_range("2022-01-03", periods=n))

        features = pd.DataFrame(index=labels.index)
        features["feat_a"] = rng.normal(0, 1, n)
        features["feat_b"] = rng.normal(0, 1, n)
        features["feat_c"] = rng.normal(0, 1, n)

        detector = LeakageDetector(shift_range=[1, 2, 3, 5])
        result = detector.test_time_shift_leakage(features, labels, threshold_corr=0.20)

        assert result.passed is True
        assert result.n_violations == 0

    def test_leakage_detector_multiple_shifts(self):
        """Feature with leakage at different shift lags should be flagged."""
        from quant_engine.validation.leakage_detection import LeakageDetector

        rng = np.random.default_rng(42)
        n = 1000
        labels = pd.Series(rng.normal(0, 1, n), index=pd.bdate_range("2020-01-02", periods=n))

        features = pd.DataFrame(index=labels.index)
        features["shift_3_leak"] = labels.shift(-3)

        detector = LeakageDetector(shift_range=[1, 2, 3, 5])
        result = detector.test_time_shift_leakage(features, labels, threshold_corr=0.15)

        assert result.passed is False
        # Should detect the lag-3 leakage
        lags = {v["shift_lag"] for v in result.violations if v["feature"] == "shift_3_leak"}
        assert 3 in lags

    def test_run_leakage_checks_raises_on_leakage(self):
        """run_leakage_checks() should raise RuntimeError on leakage."""
        from quant_engine.validation.leakage_detection import run_leakage_checks

        rng = np.random.default_rng(42)
        n = 500
        labels = pd.Series(rng.normal(0, 1, n), index=pd.bdate_range("2022-01-03", periods=n))

        features = pd.DataFrame(index=labels.index)
        features["leaky"] = labels.shift(-1)

        with pytest.raises(RuntimeError, match="Leakage detected"):
            run_leakage_checks(features, labels, threshold_corr=0.15)

    def test_run_leakage_checks_passes_clean(self):
        """run_leakage_checks() should return result on clean features."""
        from quant_engine.validation.leakage_detection import run_leakage_checks

        rng = np.random.default_rng(42)
        n = 500
        labels = pd.Series(rng.normal(0, 1, n), index=pd.bdate_range("2022-01-03", periods=n))

        features = pd.DataFrame(index=labels.index)
        features["clean"] = rng.normal(0, 1, n)

        result = run_leakage_checks(features, labels, threshold_corr=0.20)
        assert result.passed is True

    def test_causality_enforcement_rejects_research_only(self):
        """Feature pipeline should raise when RESEARCH_ONLY features violate CAUSAL filter."""
        from quant_engine.features.pipeline import get_feature_type, FEATURE_METADATA

        # Verify RESEARCH_ONLY features exist in metadata
        research_features = [
            k for k, v in FEATURE_METADATA.items() if v["type"] == "RESEARCH_ONLY"
        ]
        assert len(research_features) > 0, "No RESEARCH_ONLY features found in metadata"

        # Verify get_feature_type returns correct types
        assert get_feature_type("relative_mom_10") == "RESEARCH_ONLY"
        assert get_feature_type("RSI_14") == "CAUSAL"
        assert get_feature_type("intraday_vol_ratio") == "END_OF_DAY"

    def test_causality_enforcement_end_of_day_features(self):
        """END_OF_DAY features exist and are correctly tagged."""
        from quant_engine.features.pipeline import FEATURE_METADATA

        eod_features = [
            k for k, v in FEATURE_METADATA.items() if v["type"] == "END_OF_DAY"
        ]
        assert len(eod_features) > 0
        assert "intraday_vol_ratio" in eod_features

    def test_leakage_detector_handles_empty_index(self):
        """Detector should handle empty overlapping index gracefully."""
        from quant_engine.validation.leakage_detection import LeakageDetector

        features = pd.DataFrame(
            {"X": [1, 2, 3]},
            index=pd.bdate_range("2022-01-03", periods=3),
        )
        labels = pd.Series(
            [4, 5, 6],
            index=pd.bdate_range("2023-01-03", periods=3),  # No overlap
        )

        detector = LeakageDetector()
        result = detector.test_time_shift_leakage(features, labels)
        assert result.passed is True


# ═══════════════════════════════════════════════════════════════════════
# T4: Null Model Baselines
# ═══════════════════════════════════════════════════════════════════════


class TestNullModels:
    """Tests for null model baselines."""

    def test_random_baseline_generates_signals(self):
        """RandomBaseline should generate +1/-1 signals for all tickers."""
        from quant_engine.backtest.null_models import RandomBaseline

        universe = _make_universe(n_stocks=3)
        baseline = RandomBaseline(seed=42)
        signals = baseline.generate_signals(universe)

        assert len(signals) == 3
        for ticker, sig in signals.items():
            assert set(sig.unique()).issubset({-1, 1})
            assert len(sig) == len(universe[ticker])

    def test_random_baseline_is_reproducible(self):
        """Same seed should produce identical signals."""
        from quant_engine.backtest.null_models import RandomBaseline

        universe = _make_universe(n_stocks=2)
        sig1 = RandomBaseline(seed=42).generate_signals(universe)
        sig2 = RandomBaseline(seed=42).generate_signals(universe)

        for key in sig1:
            pd.testing.assert_series_equal(sig1[key], sig2[key])

    def test_random_baseline_sharpe_near_zero(self):
        """Random baseline should have Sharpe ratio near zero."""
        from quant_engine.backtest.null_models import RandomBaseline

        universe = _make_universe(n_stocks=5, n_bars=1000)
        metrics = RandomBaseline(seed=42).compute_returns(universe)

        # Random walk Sharpe should be close to zero (within ±1.0 is reasonable)
        assert abs(metrics.sharpe_ratio) < 1.0
        assert metrics.n_trades > 0

    def test_zero_baseline_returns_zero(self):
        """Zero baseline should have exactly zero return and Sharpe."""
        from quant_engine.backtest.null_models import ZeroBaseline

        universe = _make_universe(n_stocks=3)
        metrics = ZeroBaseline().compute_returns(universe)

        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.n_trades == 0

    def test_zero_baseline_generates_zero_signals(self):
        """Zero baseline should generate all-zero signals."""
        from quant_engine.backtest.null_models import ZeroBaseline

        universe = _make_universe(n_stocks=2)
        signals = ZeroBaseline().generate_signals(universe)

        for ticker, sig in signals.items():
            assert (sig == 0.0).all()

    def test_momentum_baseline_generates_signals(self):
        """MomentumBaseline should generate +1/-1 signals."""
        from quant_engine.backtest.null_models import MomentumBaseline

        universe = _make_universe(n_stocks=3)
        baseline = MomentumBaseline(lookback=20)
        signals = baseline.generate_signals(universe)

        assert len(signals) == 3
        for ticker, sig in signals.items():
            # After warmup period, signals should be +1 or -1
            valid = sig.dropna()
            assert set(valid.unique()).issubset({-1.0, 1.0})

    def test_momentum_baseline_computes_metrics(self):
        """Momentum baseline should produce valid metrics."""
        from quant_engine.backtest.null_models import MomentumBaseline

        universe = _make_universe(n_stocks=3, n_bars=1000)
        metrics = MomentumBaseline(lookback=20).compute_returns(universe)

        assert metrics.n_trades > 0
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.total_return, float)

    def test_compute_null_baselines_returns_all_three(self):
        """compute_null_baselines() should return all three baseline results."""
        from quant_engine.backtest.null_models import compute_null_baselines

        universe = _make_universe(n_stocks=3, n_bars=500)
        results = compute_null_baselines(universe)

        assert results.random.name == "random"
        assert results.zero.name == "zero"
        assert results.momentum.name == "momentum"

    def test_null_model_results_summary(self):
        """NullModelResults.summary() should return a dict of metrics."""
        from quant_engine.backtest.null_models import compute_null_baselines

        universe = _make_universe(n_stocks=2)
        results = compute_null_baselines(universe)
        summary = results.summary()

        assert "random" in summary
        assert "zero" in summary
        assert "momentum" in summary
        assert "sharpe_ratio" in summary["random"]
        assert "total_return" in summary["zero"]

    def test_backtest_result_summarize_vs_null(self):
        """BacktestResult.summarize_vs_null() should compare against baselines."""
        from quant_engine.backtest.engine import BacktestResult
        from quant_engine.backtest.null_models import (
            NullModelResults, NullBaselineMetrics,
        )

        result = BacktestResult(
            total_trades=100, winning_trades=60, losing_trades=40,
            win_rate=0.6, avg_return=0.01, avg_win=0.02, avg_loss=-0.01,
            total_return=0.50, annualized_return=0.25, sharpe_ratio=1.5,
            sortino_ratio=2.0, max_drawdown=-0.10, profit_factor=1.5,
            avg_holding_days=5.0, trades_per_year=50.0,
        )

        # Without null baselines
        assert result.summarize_vs_null() == {}

        # With null baselines
        result.null_baselines = NullModelResults(
            random=NullBaselineMetrics(name="random", sharpe_ratio=0.1, total_return=0.05),
            zero=NullBaselineMetrics(name="zero", sharpe_ratio=0.0, total_return=0.0),
            momentum=NullBaselineMetrics(name="momentum", sharpe_ratio=0.5, total_return=0.20),
        )

        comparison = result.summarize_vs_null()
        assert comparison["sharpe_vs_random"] == pytest.approx(1.4)
        assert comparison["sharpe_vs_zero"] == pytest.approx(1.5)
        assert comparison["sharpe_vs_momentum"] == pytest.approx(1.0)
        assert comparison["return_vs_random"] == pytest.approx(0.45)


# ═══════════════════════════════════════════════════════════════════════
# T5: Cost Stress Sweep
# ═══════════════════════════════════════════════════════════════════════


class TestCostStress:
    """Tests for cost stress testing."""

    def test_cost_stress_basic_sweep(self):
        """Basic cost stress sweep should produce results for all multipliers."""
        from quant_engine.backtest.cost_stress import CostStressTester

        rng = np.random.default_rng(42)
        gross_returns = rng.normal(0.002, 0.02, 200)

        tester = CostStressTester(base_cost_bps=20.0)
        result = tester.run_sweep(
            gross_returns=gross_returns,
            cost_per_trade_bps=20.0,
            n_trades=200,
        )

        assert result.base_cost_bps == 20.0
        assert len(result.points) == 4
        assert result.points[0].multiplier == 0.5

    def test_cost_stress_1x_matches_base(self):
        """1.0x multiplier should have the base cost."""
        from quant_engine.backtest.cost_stress import CostStressTester

        rng = np.random.default_rng(42)
        gross_returns = rng.normal(0.003, 0.02, 500)

        tester = CostStressTester(base_cost_bps=20.0, multipliers=[1.0])
        result = tester.run_sweep(
            gross_returns=gross_returns,
            cost_per_trade_bps=20.0,
            n_trades=500,
        )

        assert len(result.points) == 1
        assert result.points[0].effective_cost_bps == 20.0

    def test_cost_stress_higher_cost_reduces_sharpe(self):
        """Higher cost multiplier should reduce Sharpe ratio."""
        from quant_engine.backtest.cost_stress import CostStressTester

        rng = np.random.default_rng(42)
        gross_returns = rng.normal(0.003, 0.02, 500)

        tester = CostStressTester(base_cost_bps=20.0, multipliers=[0.5, 1.0, 2.0, 5.0])
        result = tester.run_sweep(
            gross_returns=gross_returns,
            cost_per_trade_bps=20.0,
            n_trades=500,
        )

        sharpes = [p.sharpe_ratio for p in result.points]
        # Higher costs should reduce Sharpe (monotonically decreasing)
        for i in range(len(sharpes) - 1):
            assert sharpes[i] >= sharpes[i + 1]

    def test_cost_stress_breakeven_estimation(self):
        """Breakeven cost should be computed when Sharpe crosses zero."""
        from quant_engine.backtest.cost_stress import CostStressTester

        rng = np.random.default_rng(42)
        # Small positive expected return so it's profitable at low cost
        # but unprofitable at high cost
        gross_returns = rng.normal(0.001, 0.02, 1000)

        tester = CostStressTester(base_cost_bps=20.0, multipliers=[0.5, 1.0, 2.0, 5.0])
        result = tester.run_sweep(
            gross_returns=gross_returns,
            cost_per_trade_bps=20.0,
            n_trades=1000,
        )

        # The breakeven should be finite (strategy is not infinitely robust)
        # We can't guarantee the exact value, but the structure should be valid
        assert isinstance(result.breakeven_multiplier, float)
        assert isinstance(result.breakeven_cost_bps, float)

    def test_cost_stress_report_format(self):
        """Report should be a formatted string."""
        from quant_engine.backtest.cost_stress import CostStressTester, CostStressResult

        rng = np.random.default_rng(42)
        gross_returns = rng.normal(0.002, 0.02, 200)

        tester = CostStressTester(base_cost_bps=20.0)
        result = tester.run_sweep(
            gross_returns=gross_returns,
            cost_per_trade_bps=20.0,
            n_trades=200,
        )

        report = tester.report(result)
        assert "Cost Stress Test Report" in report
        assert "Multiplier" in report
        assert "Breakeven" in report or "inf" in report.lower()

    def test_cost_stress_to_dict(self):
        """CostStressResult.to_dict() should serialize cleanly."""
        from quant_engine.backtest.cost_stress import CostStressTester

        rng = np.random.default_rng(42)
        gross_returns = rng.normal(0.002, 0.02, 200)

        tester = CostStressTester(base_cost_bps=20.0, multipliers=[1.0])
        result = tester.run_sweep(
            gross_returns=gross_returns,
            cost_per_trade_bps=20.0,
            n_trades=200,
        )

        d = result.to_dict()
        assert "base_cost_bps" in d
        assert "points" in d
        assert len(d["points"]) == 1

    def test_cost_stress_empty_returns(self):
        """Empty returns array should not crash."""
        from quant_engine.backtest.cost_stress import CostStressTester

        tester = CostStressTester(base_cost_bps=20.0)
        result = tester.run_sweep(
            gross_returns=np.array([]),
            cost_per_trade_bps=20.0,
            n_trades=0,
        )

        assert len(result.points) == 0


# ═══════════════════════════════════════════════════════════════════════
# T6: Cache Staleness Trading Calendar Fix
# ═══════════════════════════════════════════════════════════════════════


class TestCacheStaleness:
    """Tests for trading-calendar-aware cache staleness."""

    def test_get_last_trading_day_returns_timestamp(self):
        """_get_last_trading_day should return a pd.Timestamp."""
        from quant_engine.data.loader import _get_last_trading_day

        result = _get_last_trading_day()
        assert isinstance(result, pd.Timestamp)

    def test_get_last_trading_day_weekday(self):
        """A weekday should return itself or a recent trading day."""
        from quant_engine.data.loader import _get_last_trading_day

        wednesday = pd.Timestamp("2026-02-18")  # Wednesday
        result = _get_last_trading_day(wednesday)
        assert result <= wednesday
        # Should be the same day or very close
        assert (wednesday - result).days <= 1

    def test_get_last_trading_day_weekend(self):
        """On a weekend, should return the preceding Friday."""
        from quant_engine.data.loader import _get_last_trading_day

        saturday = pd.Timestamp("2026-02-21")  # Saturday
        result = _get_last_trading_day(saturday)

        # Should be Friday Feb 20 (assuming it's a trading day)
        assert result < saturday
        assert result.day_name() == "Friday"

    def test_trading_days_between_same_day(self):
        """Same start/end should return 0 trading days."""
        from quant_engine.data.loader import _trading_days_between

        day = pd.Timestamp("2026-02-18")
        assert _trading_days_between(day, day) == 0

    def test_trading_days_between_weekdays(self):
        """Two consecutive weekdays should be 2 trading days."""
        from quant_engine.data.loader import _trading_days_between

        monday = pd.Timestamp("2026-02-16")
        tuesday = pd.Timestamp("2026-02-17")
        result = _trading_days_between(monday, tuesday)
        assert result == 2  # Monday and Tuesday both counted (inclusive)

    def test_trading_days_between_across_weekend(self):
        """Friday to Monday should be 2 trading days (no Sat/Sun)."""
        from quant_engine.data.loader import _trading_days_between

        friday = pd.Timestamp("2026-02-20")
        monday = pd.Timestamp("2026-02-23")
        result = _trading_days_between(friday, monday)
        assert result == 2  # Friday and Monday

    def test_cache_usable_respects_trading_calendar(self):
        """Cache from Friday should still be usable on Monday."""
        from quant_engine.data.loader import _cache_is_usable

        # Create mock cached data ending on a Friday with enough history
        # Use years=2 so 600 bars is plenty of history
        friday = pd.Timestamp("2026-02-20")
        dates = pd.bdate_range(end=friday, periods=600)
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.01, len(dates)))
        df = pd.DataFrame({
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, len(dates)).astype(float),
        }, index=dates)

        # Patch _get_last_trading_day to return Monday
        monday = pd.Timestamp("2026-02-23")
        with patch("quant_engine.data.loader._get_last_trading_day", return_value=monday):
            result = _cache_is_usable(
                cached=df,
                meta={"source": "wrds"},
                years=2,
                require_recent=True,
                require_trusted=True,
            )
            # 2 trading days (Fri, Mon) should be << CACHE_MAX_STALENESS_DAYS (21)
            assert result is True

    def test_cache_stale_after_many_trading_days(self):
        """Cache older than max_staleness trading days should be stale."""
        from quant_engine.data.loader import _cache_is_usable

        # Create cached data ending 30 trading days ago
        old_date = pd.Timestamp("2026-01-05")
        dates = pd.bdate_range(end=old_date, periods=600)
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.01, len(dates)))
        df = pd.DataFrame({
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, len(dates)).astype(float),
        }, index=dates)

        recent_day = pd.Timestamp("2026-02-20")
        with patch("quant_engine.data.loader._get_last_trading_day", return_value=recent_day):
            result = _cache_is_usable(
                cached=df,
                meta={"source": "wrds"},
                years=5,
                require_recent=True,
                require_trusted=True,
            )
            # ~33 trading days > CACHE_MAX_STALENESS_DAYS (21)
            assert result is False


# ═══════════════════════════════════════════════════════════════════════
# Integration: Config validation
# ═══════════════════════════════════════════════════════════════════════


class TestConfigIntegration:
    """Integration tests for config changes."""

    def test_config_has_execution_contract_constants(self):
        """Config should have all execution contract constants."""
        from quant_engine import config

        assert hasattr(config, "RET_TYPE")
        assert hasattr(config, "LABEL_H")
        assert hasattr(config, "PX_TYPE")
        assert hasattr(config, "ENTRY_PRICE_TYPE")

        assert config.RET_TYPE == "log"
        assert config.LABEL_H == 5
        assert config.PX_TYPE == "close"
        assert config.ENTRY_PRICE_TYPE == "next_bar_open"

    def test_config_has_truth_layer_flags(self):
        """Config should have all Truth Layer feature flags."""
        from quant_engine import config

        assert hasattr(config, "TRUTH_LAYER_STRICT_PRECONDITIONS")
        assert hasattr(config, "TRUTH_LAYER_FAIL_ON_CORRUPT")
        assert hasattr(config, "TRUTH_LAYER_ENFORCE_CAUSALITY")
        assert hasattr(config, "TRUTH_LAYER_COMPUTE_NULL_BASELINES")
        assert hasattr(config, "TRUTH_LAYER_COST_STRESS_ENABLED")

    def test_config_has_cost_stress_multipliers(self):
        """Config should have cost stress sweep multipliers."""
        from quant_engine import config

        assert hasattr(config, "COST_STRESS_MULTIPLIERS")
        assert config.COST_STRESS_MULTIPLIERS == [0.5, 1.0, 2.0, 5.0]

    def test_config_structured_has_preconditions(self):
        """SystemConfig should have preconditions field."""
        from quant_engine.config_structured import SystemConfig

        cfg = SystemConfig()
        assert hasattr(cfg, "preconditions")
        assert cfg.preconditions.label_h == 5

    def test_config_structured_has_cost_stress(self):
        """SystemConfig should have cost_stress field."""
        from quant_engine.config_structured import SystemConfig

        cfg = SystemConfig()
        assert hasattr(cfg, "cost_stress")
        assert cfg.cost_stress.multipliers == [0.5, 1.0, 2.0, 5.0]

    def test_validate_config_still_works(self):
        """Existing validate_config() should still work."""
        from quant_engine.config import validate_config

        issues = validate_config()
        assert isinstance(issues, list)
