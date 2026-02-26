"""Tests for Health System Spec 09 enhancements.

Verifies:
  - Information Ratio check computes rolling IR and scores correctly
  - Quantified survivorship bias reports PnL loss % (not binary)
  - Confidence intervals (bootstrap + normal + t + binomial)
  - Health history with rolling averages and trend detection
  - Health-to-risk feedback gate scales position sizes
  - Alert system detects degradation and domain failures
  - Updated domain weights (execution 20%, governance 10%)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from quant_engine.api.services.health_service import (
    HealthCheckResult,
    HealthService,
    _unavailable,
)
from quant_engine.api.services.health_confidence import (
    HealthConfidenceCalculator,
    ConfidenceResult,
)
from quant_engine.api.services.health_alerts import (
    Alert,
    HealthAlertManager,
)
from quant_engine.api.services.health_risk_feedback import (
    HealthRiskGate,
)


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def svc():
    return HealthService()


@pytest.fixture
def trades_csv(tmp_path):
    """Create a realistic backtest trades CSV with correlated predictions."""
    np.random.seed(42)
    n = 300
    predicted = np.random.normal(0.005, 0.02, n)
    actual = predicted * 0.3 + np.random.normal(0, 0.015, n)
    df = pd.DataFrame({
        "ticker": np.random.choice(["AAPL", "MSFT", "GOOGL", "NVDA", "META"], n),
        "entry_date": pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d"),
        "exit_date": pd.date_range("2024-01-15", periods=n, freq="B").strftime("%Y-%m-%d"),
        "entry_price": np.random.uniform(100, 200, n),
        "exit_price": np.random.uniform(100, 200, n),
        "predicted_return": predicted,
        "actual_return": actual,
        "net_return": actual - 0.001,
        "regime": np.random.choice(["trending_bull", "mean_reverting"], n),
        "confidence": np.random.uniform(0.5, 0.9, n),
        "holding_days": np.random.randint(3, 15, n),
        "position_size": np.full(n, 0.05),
        "exit_reason": np.random.choice(["trailing_stop", "time_stop"], n),
    })
    path = tmp_path / "results" / "backtest_10d_trades.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return tmp_path


@pytest.fixture
def cache_with_spy(tmp_path):
    """Create SPY benchmark cache data."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(123)
    dates = pd.date_range("2023-01-01", periods=500, freq="B")
    close = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, len(dates)))
    df = pd.DataFrame({"Close": close, "Volume": np.random.uniform(1e7, 5e7, len(dates))},
                       index=dates)
    df.to_parquet(cache_dir / "SPY_1d.parquet")
    return cache_dir


@pytest.fixture
def model_registry(tmp_path):
    """Create a model registry with multiple versions."""
    reg = {
        "latest": "v3",
        "versions": [
            {"version_id": "v1", "cv_gap": 0.06, "holdout_spearman": 0.07},
            {"version_id": "v2", "cv_gap": 0.04, "holdout_spearman": 0.09},
            {"version_id": "v3", "cv_gap": 0.03, "holdout_spearman": 0.11},
        ],
    }
    path = tmp_path / "trained_models" / "registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(reg, f)
    return tmp_path


# ═══════════════════════════════════════════════════════════════════════
# T1: Information Ratio
# ═══════════════════════════════════════════════════════════════════════

class TestInformationRatio:
    def test_ir_computed_on_valid_trades(self, svc, trades_csv):
        """IR check should produce a valid score when trades are available."""
        with patch("quant_engine.config.RESULTS_DIR", trades_csv / "results"), \
             patch("quant_engine.config.DATA_CACHE_DIR", trades_csv / "nonexistent"), \
             patch("quant_engine.config.BENCHMARK", "SPY"):
            result = svc._check_information_ratio()

        assert result.name == "information_ratio"
        assert result.domain == "signal_quality"
        assert result.data_available is True
        assert result.status in ("PASS", "WARN", "FAIL")
        assert 0.0 <= result.score <= 100.0
        assert "current_ir" in result.raw_metrics
        assert "n_trades" in result.raw_metrics

    def test_ir_unavailable_no_trades(self, svc, tmp_path):
        """IR check should be UNAVAILABLE when no trades exist."""
        with patch("quant_engine.config.RESULTS_DIR", tmp_path / "results"):
            result = svc._check_information_ratio()
        assert result.status == "UNAVAILABLE"
        assert result.score == 0.0

    def test_ir_unavailable_too_few_trades(self, svc, tmp_path):
        """IR check should be UNAVAILABLE with < 20 trades."""
        n = 10
        df = pd.DataFrame({
            "ticker": ["AAPL"] * n,
            "actual_return": np.random.normal(0.005, 0.02, n),
            "predicted_return": np.random.normal(0.005, 0.02, n),
            "net_return": np.random.normal(0.005, 0.02, n),
            "position_size": [0.05] * n,
        })
        path = tmp_path / "results" / "backtest_10d_trades.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

        with patch("quant_engine.config.RESULTS_DIR", tmp_path / "results"):
            result = svc._check_information_ratio()
        assert result.status == "UNAVAILABLE"

    def test_ir_high_alpha_scores_well(self, svc, tmp_path):
        """Consistently positive excess returns should score high."""
        np.random.seed(99)
        n = 200
        actual = np.random.normal(0.01, 0.005, n)  # Strong positive returns
        df = pd.DataFrame({
            "ticker": ["AAPL"] * n,
            "actual_return": actual,
            "predicted_return": actual * 0.9,
            "net_return": actual - 0.001,
            "position_size": [0.05] * n,
        })
        path = tmp_path / "results" / "backtest_10d_trades.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

        with patch("quant_engine.config.RESULTS_DIR", tmp_path / "results"), \
             patch("quant_engine.config.DATA_CACHE_DIR", tmp_path / "nonexistent"), \
             patch("quant_engine.config.BENCHMARK", "SPY"):
            result = svc._check_information_ratio()

        assert result.status == "PASS"
        assert result.score >= 70.0
        assert result.raw_metrics["current_ir"] > 0.5

    def test_ir_methodology_present(self, svc, trades_csv):
        """IR check should have non-empty methodology."""
        with patch("quant_engine.config.RESULTS_DIR", trades_csv / "results"), \
             patch("quant_engine.config.DATA_CACHE_DIR", trades_csv / "nonexistent"), \
             patch("quant_engine.config.BENCHMARK", "SPY"):
            result = svc._check_information_ratio()
        assert result.methodology
        assert "IR" in result.methodology


# ═══════════════════════════════════════════════════════════════════════
# T2: Quantified Survivorship Bias
# ═══════════════════════════════════════════════════════════════════════

class TestQuantifiedSurvivorshipBias:
    def test_quantified_pnl_loss(self, svc, tmp_path):
        """When trade data is available, survivorship should quantify PnL impact."""
        # Create cache with a delisted ticker
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Recent data
        df_recent = pd.DataFrame({
            "Close": np.random.uniform(100, 200, 100),
            "Volume": np.random.uniform(1e6, 1e7, 100),
        }, index=pd.date_range("2025-10-01", periods=100))
        df_recent.to_parquet(cache_dir / "AAPL_1d.parquet")

        # Old data (delisted proxy)
        df_old = pd.DataFrame({
            "Close": np.random.uniform(50, 100, 100),
            "Volume": np.random.uniform(1e5, 1e6, 100),
        }, index=pd.date_range("2020-01-01", periods=100))
        df_old.to_parquet(cache_dir / "DELISTED_1d.parquet")

        # Create trades CSV with some DELISTED trades
        n = 100
        tickers = ["AAPL"] * 90 + ["DELISTED"] * 10
        df_trades = pd.DataFrame({
            "ticker": tickers,
            "net_return": np.random.normal(0.002, 0.02, n),
            "position_size": [0.05] * n,
            "predicted_return": np.random.normal(0.005, 0.02, n),
            "actual_return": np.random.normal(0.003, 0.03, n),
        })
        path = tmp_path / "results" / "backtest_10d_trades.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df_trades.to_csv(path, index=False)

        with patch("quant_engine.config.DATA_CACHE_DIR", cache_dir), \
             patch("quant_engine.config.RESULTS_DIR", tmp_path / "results"):
            result = svc._check_survivorship_bias()

        assert result.data_available is True
        assert "pnl_lost_pct" in result.raw_metrics
        assert result.raw_metrics["pnl_quantified"] is True

    def test_no_trades_falls_back_to_binary(self, svc, tmp_path):
        """Without trade data, survivorship uses binary detection."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            "Close": np.random.uniform(100, 200, 100),
            "Volume": np.random.uniform(1e6, 1e7, 100),
            "total_ret": np.random.normal(0.001, 0.02, 100),
        }, index=pd.date_range("2020-01-01", periods=100))
        df.to_parquet(cache_dir / "DELIST_1d.parquet")

        with patch("quant_engine.config.DATA_CACHE_DIR", cache_dir), \
             patch("quant_engine.config.RESULTS_DIR", tmp_path / "no_results"):
            result = svc._check_survivorship_bias()

        # Should still produce a result using binary logic
        assert result.data_available is True
        assert result.status in ("PASS", "WARN", "FAIL")


# ═══════════════════════════════════════════════════════════════════════
# T3: Confidence Intervals
# ═══════════════════════════════════════════════════════════════════════

class TestHealthConfidenceIntervals:
    def test_bootstrap_ci(self):
        """Bootstrap CI should contain the true mean."""
        np.random.seed(42)
        samples = np.random.normal(70, 10, 50)
        calc = HealthConfidenceCalculator()
        result = calc.compute_ci_bootstrap(samples)

        assert result.method == "bootstrap"
        assert result.ci_lower < result.mean < result.ci_upper
        assert result.n_samples == 50

    def test_normal_ci(self):
        """Normal CI should be symmetric around the mean."""
        calc = HealthConfidenceCalculator()
        result = calc.compute_ci_normal(mean=75.0, std=10.0, n=100)

        assert result.method == "normal"
        assert result.ci_lower < 75.0 < result.ci_upper
        # Check symmetry
        lower_dist = 75.0 - result.ci_lower
        upper_dist = result.ci_upper - 75.0
        assert abs(lower_dist - upper_dist) < 0.01

    def test_t_ci(self):
        """t-distribution CI should be wider than normal for small N."""
        calc = HealthConfidenceCalculator()
        t_result = calc.compute_ci_t(mean=75.0, std=10.0, n=10)
        z_result = calc.compute_ci_normal(mean=75.0, std=10.0, n=10)

        assert t_result.method == "t"
        # t-distribution has fatter tails → wider CI
        assert t_result.ci_width >= z_result.ci_width

    def test_binomial_ci(self):
        """Wilson score interval for binary proportions."""
        calc = HealthConfidenceCalculator()
        result = calc.compute_ci_binomial(successes=60, total=100)

        assert result.method == "binomial"
        assert 0.5 < result.mean < 0.7
        assert result.ci_lower > 0.0
        assert result.ci_upper < 1.0
        # True proportion (0.6) should be inside CI
        assert result.ci_lower < 0.6 < result.ci_upper

    def test_auto_selects_bootstrap_for_small_n(self):
        """Auto method should use bootstrap for N < 30."""
        np.random.seed(42)
        samples = np.random.normal(70, 10, 15)
        calc = HealthConfidenceCalculator(bootstrap_threshold=30)
        result = calc.compute_ci(samples=samples)
        assert result.method == "bootstrap"

    def test_auto_selects_normal_for_large_n(self):
        """Auto method should use normal for N >= 30."""
        np.random.seed(42)
        samples = np.random.normal(70, 10, 100)
        calc = HealthConfidenceCalculator(bootstrap_threshold=30)
        result = calc.compute_ci(samples=samples)
        assert result.method == "normal"

    def test_insufficient_samples(self):
        """Very small N should return 'insufficient' method."""
        calc = HealthConfidenceCalculator(min_samples=5)
        result = calc.compute_ci(samples=np.array([1.0, 2.0]))
        assert result.method == "insufficient"
        assert result.is_low_confidence is True

    def test_ci_width_decreases_with_n(self):
        """CI should get tighter as sample size increases."""
        calc = HealthConfidenceCalculator()
        ci_10 = calc.compute_ci_normal(mean=75.0, std=10.0, n=10)
        ci_100 = calc.compute_ci_normal(mean=75.0, std=10.0, n=100)
        ci_1000 = calc.compute_ci_normal(mean=75.0, std=10.0, n=1000)

        assert ci_10.ci_width > ci_100.ci_width > ci_1000.ci_width

    def test_propagate_weighted_ci(self):
        """CI propagation through weighted average."""
        scores = [80.0, 70.0, 60.0]
        widths = [10.0, 8.0, 12.0]
        weights = [0.4, 0.3, 0.3]

        ci_lower, ci_upper = HealthConfidenceCalculator.propagate_weighted_ci(
            scores, widths, weights,
        )

        # Weighted mean = 0.4*80 + 0.3*70 + 0.3*60 = 71.0
        expected_mean = 71.0
        assert ci_lower < expected_mean < ci_upper
        assert ci_lower > 0

    def test_to_dict_format(self):
        """ConfidenceResult.to_dict should include all expected fields."""
        result = ConfidenceResult(
            mean=75.0, ci_lower=70.0, ci_upper=80.0,
            n_samples=50, method="normal",
        )
        d = result.to_dict()
        assert "mean" in d
        assert "ci_lower" in d
        assert "ci_upper" in d
        assert "ci_width" in d
        assert "n_samples" in d
        assert "method" in d
        assert "low_confidence" in d
        assert d["ci_width"] == 10.0


# ═══════════════════════════════════════════════════════════════════════
# T4: Health History and Trending
# ═══════════════════════════════════════════════════════════════════════

class TestHealthHistoryTrending:
    def test_rolling_average_7d(self, svc):
        """7-day rolling average should smooth noise."""
        scores = [50.0, 60.0, 55.0, 65.0, 70.0, 68.0, 72.0, 75.0, 80.0, 78.0]
        result = HealthService._compute_rolling_average(scores, window=7)
        assert len(result) == len(scores)
        # Rolling average should be smoother
        std_raw = float(np.std(scores))
        std_rolling = float(np.std(result))
        assert std_rolling <= std_raw

    def test_rolling_average_short_series(self, svc):
        """Rolling average should handle series shorter than window."""
        scores = [50.0, 60.0, 70.0]
        result = HealthService._compute_rolling_average(scores, window=7)
        assert len(result) == 3

    def test_trend_detection_improving(self, svc):
        """Consistently increasing scores should be 'improving'."""
        scores = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0]
        trend, slope = HealthService._detect_trend(scores)
        assert trend == "improving"
        assert slope > 0.5

    def test_trend_detection_degrading(self, svc):
        """Consistently decreasing scores should be 'degrading'."""
        scores = [95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 50.0]
        trend, slope = HealthService._detect_trend(scores)
        assert trend == "degrading"
        assert slope < -0.5

    def test_trend_detection_stable(self, svc):
        """Flat scores should be 'stable'."""
        scores = [70.0, 70.2, 69.8, 70.1, 69.9, 70.0, 70.3, 69.7, 70.0, 70.1]
        trend, slope = HealthService._detect_trend(scores)
        assert trend == "stable"
        assert abs(slope) <= 0.5

    def test_trend_detection_too_few(self, svc):
        """< 3 scores should return 'unknown'."""
        trend, slope = HealthService._detect_trend([70.0])
        assert trend == "unknown"

    def test_history_with_trends_structure(self, svc, tmp_path):
        """get_health_history_with_trends returns correct structure."""
        db_path = tmp_path / "health_history.db"

        with patch.object(HealthService, "_get_health_db_path", return_value=db_path):
            # Save some snapshots
            for i in range(10):
                svc.save_health_snapshot({
                    "overall_score": 60.0 + i * 3,
                    "domains": {"test": {"score": 60.0 + i * 3, "status": "PASS"}},
                })

            result = svc.get_health_history_with_trends(limit=90)

        assert "snapshots" in result
        assert "rolling_7d" in result
        assert "rolling_30d" in result
        assert "trend" in result
        assert "trend_slope" in result
        assert len(result["snapshots"]) == 10
        assert result["trend"] == "improving"  # scores are monotonically increasing

    def test_save_and_retrieve(self, svc, tmp_path):
        """Save and retrieve health snapshots."""
        db_path = tmp_path / "health_history.db"

        with patch.object(HealthService, "_get_health_db_path", return_value=db_path):
            svc.save_health_snapshot({
                "overall_score": 82.5,
                "domains": {
                    "data_integrity": {"score": 85.0, "status": "PASS"},
                    "signal_quality": {"score": 80.0, "status": "PASS"},
                },
            })

            history = svc.get_health_history(limit=10)

        assert len(history) >= 1
        latest = history[-1]
        assert latest["overall_score"] == 82.5
        assert "domain_scores" in latest


# ═══════════════════════════════════════════════════════════════════════
# T5: Health to Risk Feedback
# ═══════════════════════════════════════════════════════════════════════

class TestHealthRiskGate:
    def test_full_health_full_size(self):
        """Health=100 should produce multiplier=1.0."""
        gate = HealthRiskGate()
        mult = gate.compute_size_multiplier(100.0)
        assert mult == pytest.approx(1.0, abs=0.01)

    def test_moderate_health_reduces_size(self):
        """Health=60 should produce multiplier=0.8."""
        gate = HealthRiskGate()
        mult = gate.compute_size_multiplier(60.0)
        assert mult == pytest.approx(0.80, abs=0.05)

    def test_low_health_severe_reduction(self):
        """Health=40 should produce multiplier=0.5."""
        gate = HealthRiskGate()
        mult = gate.compute_size_multiplier(40.0)
        assert mult == pytest.approx(0.50, abs=0.05)

    def test_zero_health_near_halt(self):
        """Health=0 should produce multiplier≈0.05 (near-halt)."""
        gate = HealthRiskGate()
        mult = gate.compute_size_multiplier(0.0)
        assert mult == pytest.approx(0.05, abs=0.01)

    def test_apply_health_gate_scales_position(self):
        """apply_health_gate should multiply position size by multiplier."""
        gate = HealthRiskGate()
        original = 0.05
        adjusted = gate.apply_health_gate(original, health_score=60.0)
        expected = original * 0.8
        assert adjusted == pytest.approx(expected, abs=0.005)

    def test_apply_health_gate_weights_array(self):
        """apply_health_gate_weights should scale entire weight vector."""
        gate = HealthRiskGate()
        weights = np.array([0.05, 0.04, 0.03])
        adjusted = gate.apply_health_gate_weights(weights, health_score=60.0)
        expected_mult = gate.compute_size_multiplier(60.0)
        np.testing.assert_allclose(adjusted, weights * expected_mult, atol=0.001)

    def test_should_halt_trading(self):
        """Trading should halt when health is below threshold."""
        gate = HealthRiskGate(halt_threshold=0.05)
        assert gate.should_halt_trading(health_score=3.0) is True
        assert gate.should_halt_trading(health_score=10.0) is False
        assert gate.should_halt_trading(health_score=50.0) is False

    def test_disabled_gate_identity(self):
        """Disabled gate should return identity multiplier."""
        gate = HealthRiskGate(enabled=False)
        assert gate.compute_size_multiplier(10.0) == 1.0
        assert gate.apply_health_gate(0.05, 10.0) == 0.05
        assert gate.should_halt_trading(1.0) is False

    def test_smoothing(self):
        """Smoothed health should not jump immediately."""
        gate = HealthRiskGate(smoothing_alpha=0.5)
        gate.update_health(80.0)
        assert gate._smoothed_health == 80.0

        gate.update_health(40.0)
        # With alpha=0.5: smoothed = 0.5*80 + 0.5*40 = 60
        assert gate._smoothed_health == pytest.approx(60.0, abs=0.1)

    def test_interpolation_between_breakpoints(self):
        """Intermediate health values should interpolate linearly."""
        gate = HealthRiskGate()
        # Health=70 is between 60 (mult=0.80) and 80 (mult=0.95)
        mult_70 = gate.compute_size_multiplier(70.0)
        # Linear interpolation: 0.80 + (0.70-0.60)/(0.80-0.60) * (0.95-0.80)
        # = 0.80 + 0.5 * 0.15 = 0.875
        assert mult_70 == pytest.approx(0.875, abs=0.01)

    def test_get_status(self):
        """get_status should return diagnostic dict."""
        gate = HealthRiskGate()
        gate.update_health(75.0)
        status = gate.get_status()
        assert "enabled" in status
        assert "smoothed_health" in status
        assert "current_multiplier" in status
        assert status["enabled"] is True


# ═══════════════════════════════════════════════════════════════════════
# T6: Alert System
# ═══════════════════════════════════════════════════════════════════════

class TestHealthAlerts:
    def test_degradation_alert_triggered(self):
        """Alert when health drops > 10% day-over-day."""
        mgr = HealthAlertManager(degradation_threshold=0.10)
        alert = mgr.check_health_degradation(
            health_today=65.0, health_yesterday=80.0,
        )
        assert alert is not None
        assert alert.alert_type == "CRITICAL"
        assert "degrad" in alert.message.lower()

    def test_degradation_alert_not_triggered(self):
        """No alert when health drops < 10%."""
        mgr = HealthAlertManager(degradation_threshold=0.10)
        alert = mgr.check_health_degradation(
            health_today=75.0, health_yesterday=80.0,
        )
        assert alert is None

    def test_domain_failure_alert(self):
        """Alert when domain score < critical threshold."""
        mgr = HealthAlertManager(domain_critical_threshold=50.0)
        alerts = mgr.check_domain_failures({
            "data_integrity": 80.0,
            "signal_quality": 30.0,  # below 50
            "execution_quality": 45.0,  # below 50
        })
        assert len(alerts) == 2
        domains = {a.domain for a in alerts}
        assert "signal_quality" in domains
        assert "execution_quality" in domains

    def test_no_domain_failure_above_threshold(self):
        """No alerts when all domains are healthy."""
        mgr = HealthAlertManager(domain_critical_threshold=50.0)
        alerts = mgr.check_domain_failures({
            "data_integrity": 80.0,
            "signal_quality": 75.0,
            "execution_quality": 60.0,
        })
        assert len(alerts) == 0

    def test_deduplication(self):
        """Duplicate alerts should be suppressed."""
        mgr = HealthAlertManager(dedup_window_seconds=3600)

        alert1 = Alert(alert_type="CRITICAL", message="Test alert", domain="test")
        alert2 = Alert(alert_type="CRITICAL", message="Test alert", domain="test")

        sent1 = mgr.process_alerts([alert1])
        assert len(sent1) == 1

        sent2 = mgr.process_alerts([alert2])
        assert len(sent2) == 0  # suppressed

    def test_alert_to_dict(self):
        """Alert.to_dict should include all fields."""
        alert = Alert(
            alert_type="STANDARD",
            message="Domain failing",
            domain="signal_quality",
            health_score=45.0,
        )
        d = alert.to_dict()
        assert d["alert_type"] == "STANDARD"
        assert d["domain"] == "signal_quality"
        assert d["health_score"] == 45.0
        assert "timestamp" in d

    def test_low_confidence_alert(self):
        """Alert for checks with too few samples."""
        mgr = HealthAlertManager()
        alert = mgr.check_low_confidence("test_check", n_samples=5, threshold=20)
        assert alert is not None
        assert alert.alert_type == "INFORMATIONAL"

        alert2 = mgr.check_low_confidence("test_check", n_samples=50, threshold=20)
        assert alert2 is None

    def test_null_health_no_degradation_alert(self):
        """No alert when yesterday's health is None."""
        mgr = HealthAlertManager()
        alert = mgr.check_health_degradation(
            health_today=65.0, health_yesterday=None,
        )
        assert alert is None


# ═══════════════════════════════════════════════════════════════════════
# Domain Weights (Spec 09)
# ═══════════════════════════════════════════════════════════════════════

class TestUpdatedDomainWeights:
    def test_execution_weight_increased(self, svc, trades_csv):
        """Execution quality should have weight 0.20 (up from 0.15)."""
        with patch("quant_engine.config.RESULTS_DIR", trades_csv / "results"), \
             patch("quant_engine.config.MODEL_DIR", trades_csv / "models"), \
             patch("quant_engine.config.DATA_CACHE_DIR", trades_csv / "cache"):
            result = svc.compute_comprehensive_health()

        assert result["domains"]["execution_quality"]["weight"] == 0.20

    def test_governance_weight_decreased(self, svc, trades_csv):
        """Model governance should have weight 0.10 (down from 0.15)."""
        with patch("quant_engine.config.RESULTS_DIR", trades_csv / "results"), \
             patch("quant_engine.config.MODEL_DIR", trades_csv / "models"), \
             patch("quant_engine.config.DATA_CACHE_DIR", trades_csv / "cache"):
            result = svc.compute_comprehensive_health()

        assert result["domains"]["model_governance"]["weight"] == 0.10

    def test_weights_sum_to_one(self, svc, trades_csv):
        """All domain weights must sum to 1.0."""
        with patch("quant_engine.config.RESULTS_DIR", trades_csv / "results"), \
             patch("quant_engine.config.MODEL_DIR", trades_csv / "models"), \
             patch("quant_engine.config.DATA_CACHE_DIR", trades_csv / "cache"):
            result = svc.compute_comprehensive_health()

        total = sum(d["weight"] for d in result["domains"].values())
        assert total == pytest.approx(1.0)

    def test_ir_check_included_in_signal_quality(self, svc, trades_csv):
        """Information Ratio should appear in signal_quality checks."""
        with patch("quant_engine.config.RESULTS_DIR", trades_csv / "results"), \
             patch("quant_engine.config.MODEL_DIR", trades_csv / "models"), \
             patch("quant_engine.config.DATA_CACHE_DIR", trades_csv / "cache"):
            result = svc.compute_comprehensive_health()

        signal_checks = result["domains"]["signal_quality"]["checks"]
        check_names = [c["name"] for c in signal_checks]
        assert "information_ratio" in check_names

    def test_comprehensive_health_has_ci(self, svc, trades_csv):
        """Comprehensive health should include confidence intervals."""
        with patch("quant_engine.config.RESULTS_DIR", trades_csv / "results"), \
             patch("quant_engine.config.MODEL_DIR", trades_csv / "models"), \
             patch("quant_engine.config.DATA_CACHE_DIR", trades_csv / "cache"):
            result = svc.compute_comprehensive_health()

        # Overall CI
        if result["overall_score"] is not None:
            assert "overall_ci" in result

        # Domain CIs
        for dname, dinfo in result["domains"].items():
            assert "ci" in dinfo

    def test_comprehensive_health_has_alerts(self, svc, trades_csv):
        """Comprehensive health should include alerts list."""
        with patch("quant_engine.config.RESULTS_DIR", trades_csv / "results"), \
             patch("quant_engine.config.MODEL_DIR", trades_csv / "models"), \
             patch("quant_engine.config.DATA_CACHE_DIR", trades_csv / "cache"):
            result = svc.compute_comprehensive_health()

        assert "alerts" in result
        assert isinstance(result["alerts"], list)

    def test_updated_methodology_string(self, svc, trades_csv):
        """Methodology should reflect updated weights."""
        with patch("quant_engine.config.RESULTS_DIR", trades_csv / "results"), \
             patch("quant_engine.config.MODEL_DIR", trades_csv / "models"), \
             patch("quant_engine.config.DATA_CACHE_DIR", trades_csv / "cache"):
            result = svc.compute_comprehensive_health()

        methodology = result["overall_methodology"]
        assert "Execution Quality (20%)" in methodology
        assert "Model Governance (10%)" in methodology
