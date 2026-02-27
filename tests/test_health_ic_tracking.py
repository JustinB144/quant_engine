"""Tests for SPEC-H01: IC tracking in health system.

Verifies:
  - IC history save and retrieve via SQLite
  - Rolling IC mean and trend computation
  - WARNING threshold at IC < 0.01
  - CRITICAL threshold at IC < 0
  - Fallback to latest_cycle.json when no DB history
  - Edge cases: no data, single cycle, all negative
  - IC tracking check wired into signal_quality domain
  - Autopilot engine saves IC to health tracking DB
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pytest

from quant_engine.api.services.health_service import (
    HealthCheckResult,
    HealthService,
    _unavailable,
)


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def svc():
    return HealthService()


@pytest.fixture
def ic_db(tmp_path):
    """Provide a temporary IC tracking database path."""
    return tmp_path / "ic_tracking.db"


@pytest.fixture
def trades_csv(tmp_path):
    """Create a realistic backtest trades CSV for comprehensive health tests."""
    import pandas as pd
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


# ═══════════════════════════════════════════════════════════════════════
# T1: IC History Storage
# ═══════════════════════════════════════════════════════════════════════

class TestICHistoryStorage:
    def test_save_and_retrieve(self, svc, ic_db):
        """IC snapshots should round-trip through SQLite."""
        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            svc.save_ic_snapshot(
                ic_mean=0.025,
                ic_ir=1.5,
                n_candidates=36,
                n_passed=3,
                best_strategy_id="h10_e40_c50_r1_m10",
            )

            history = svc.get_ic_history(limit=10)

        assert len(history) == 1
        entry = history[0]
        assert entry["ic_mean"] == pytest.approx(0.025)
        assert entry["ic_ir"] == pytest.approx(1.5)
        assert entry["n_candidates"] == 36
        assert entry["n_passed"] == 3
        assert entry["best_strategy_id"] == "h10_e40_c50_r1_m10"
        assert "timestamp" in entry

    def test_multiple_saves_ordered_chronologically(self, svc, ic_db):
        """Multiple snapshots should be returned oldest-first."""
        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            svc.save_ic_snapshot(ic_mean=0.01)
            svc.save_ic_snapshot(ic_mean=0.02)
            svc.save_ic_snapshot(ic_mean=0.03)

            history = svc.get_ic_history(limit=10)

        assert len(history) == 3
        assert history[0]["ic_mean"] == pytest.approx(0.01)
        assert history[1]["ic_mean"] == pytest.approx(0.02)
        assert history[2]["ic_mean"] == pytest.approx(0.03)

    def test_limit_parameter(self, svc, ic_db):
        """get_ic_history should respect the limit parameter."""
        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            for i in range(10):
                svc.save_ic_snapshot(ic_mean=0.01 * i)

            history = svc.get_ic_history(limit=3)

        assert len(history) == 3
        # Should be the 3 most recent, ordered oldest-first
        assert history[0]["ic_mean"] == pytest.approx(0.07)
        assert history[1]["ic_mean"] == pytest.approx(0.08)
        assert history[2]["ic_mean"] == pytest.approx(0.09)

    def test_pruning_old_entries(self, svc, ic_db):
        """Old entries beyond MAX_IC_SNAPSHOTS should be pruned."""
        original_max = HealthService._MAX_IC_SNAPSHOTS
        try:
            HealthService._MAX_IC_SNAPSHOTS = 5  # Small limit for testing
            with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
                for i in range(10):
                    svc.save_ic_snapshot(ic_mean=0.01 * i)

                history = svc.get_ic_history(limit=100)
            assert len(history) == 5
        finally:
            HealthService._MAX_IC_SNAPSHOTS = original_max

    def test_empty_database(self, svc, ic_db):
        """Empty database should return empty list."""
        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            history = svc.get_ic_history(limit=10)
        assert history == []

    def test_save_with_none_optional_fields(self, svc, ic_db):
        """Optional fields should handle None gracefully."""
        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            svc.save_ic_snapshot(ic_mean=0.015)
            history = svc.get_ic_history(limit=10)

        assert len(history) == 1
        assert history[0]["ic_mean"] == pytest.approx(0.015)
        assert history[0]["ic_ir"] is None
        assert history[0]["n_candidates"] == 0
        assert history[0]["best_strategy_id"] == ""


# ═══════════════════════════════════════════════════════════════════════
# T2: IC Tracking Health Check
# ═══════════════════════════════════════════════════════════════════════

class TestICTrackingHealthCheck:
    def test_strong_ic_passes(self, svc, ic_db):
        """IC mean >= 0.03 should score 90 with PASS."""
        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            for _ in range(5):
                svc.save_ic_snapshot(ic_mean=0.04)

            result = svc._check_ic_tracking()

        assert result.name == "ic_tracking"
        assert result.domain == "signal_quality"
        assert result.status == "PASS"
        assert result.score == 90.0
        assert result.raw_metrics["rolling_ic_mean"] >= 0.03
        assert "Strong IC" in result.explanation

    def test_acceptable_ic_passes(self, svc, ic_db):
        """IC mean between 0.01 and 0.03 should score 70 with PASS."""
        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            for _ in range(5):
                svc.save_ic_snapshot(ic_mean=0.02)

            result = svc._check_ic_tracking()

        assert result.status == "PASS"
        assert result.score == 70.0
        assert "Acceptable IC" in result.explanation

    def test_weak_ic_warns(self, svc, ic_db):
        """IC mean between 0 and 0.01 should score 40 with WARN."""
        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            for _ in range(5):
                svc.save_ic_snapshot(ic_mean=0.005)

            result = svc._check_ic_tracking()

        assert result.status == "WARN"
        assert result.score == 40.0
        assert "Weak IC" in result.explanation

    def test_negative_ic_fails(self, svc, ic_db):
        """IC mean < 0 should score 15 with FAIL (CRITICAL)."""
        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            for _ in range(5):
                svc.save_ic_snapshot(ic_mean=-0.02)

            result = svc._check_ic_tracking()

        assert result.status == "FAIL"
        assert result.score == 15.0
        assert "CRITICAL" in result.explanation
        assert "anti-predictive" in result.explanation

    def test_no_data_unavailable(self, svc, ic_db, tmp_path):
        """No IC data should return UNAVAILABLE."""
        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db), \
             patch("quant_engine.config.AUTOPILOT_CYCLE_REPORT",
                   tmp_path / "nonexistent.json"):
            result = svc._check_ic_tracking()

        assert result.status == "UNAVAILABLE"
        assert result.score == 0.0

    def test_single_cycle(self, svc, ic_db):
        """Single cycle should still produce a valid result."""
        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            svc.save_ic_snapshot(ic_mean=0.05)
            result = svc._check_ic_tracking()

        assert result.status == "PASS"
        assert result.score == 90.0
        assert result.raw_metrics["n_cycles"] == 1
        assert result.raw_metrics["trend"] == "unknown"

    def test_trend_detection_improving(self, svc, ic_db):
        """Improving IC series should be labeled 'improving'."""
        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            for i in range(5):
                svc.save_ic_snapshot(ic_mean=0.01 + i * 0.01)

            result = svc._check_ic_tracking()

        assert result.raw_metrics["trend"] == "improving"
        assert result.raw_metrics["trend_slope"] > 0.001

    def test_trend_detection_degrading(self, svc, ic_db):
        """Declining IC series should be labeled 'degrading'."""
        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            for i in range(5):
                svc.save_ic_snapshot(ic_mean=0.05 - i * 0.01)

            result = svc._check_ic_tracking()

        assert result.raw_metrics["trend"] == "degrading"
        assert result.raw_metrics["trend_slope"] < -0.001

    def test_trend_detection_stable(self, svc, ic_db):
        """Stable IC series should be labeled 'stable'."""
        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            for _ in range(5):
                svc.save_ic_snapshot(ic_mean=0.02)

            result = svc._check_ic_tracking()

        assert result.raw_metrics["trend"] == "stable"

    def test_raw_metrics_complete(self, svc, ic_db):
        """Raw metrics should include all expected fields."""
        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            svc.save_ic_snapshot(ic_mean=0.025)
            svc.save_ic_snapshot(ic_mean=0.030)
            svc.save_ic_snapshot(ic_mean=0.035)
            result = svc._check_ic_tracking()

        raw = result.raw_metrics
        assert "rolling_ic_mean" in raw
        assert "ic_std" in raw
        assert "n_cycles" in raw
        assert "trend" in raw
        assert "trend_slope" in raw
        assert "latest_ic" in raw
        assert "ic_values" in raw
        assert raw["n_cycles"] == 3

    def test_methodology_present(self, svc, ic_db):
        """IC tracking check should have a non-empty methodology."""
        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            svc.save_ic_snapshot(ic_mean=0.02)
            result = svc._check_ic_tracking()

        assert result.methodology
        assert "IC" in result.methodology
        assert "Spearman" in result.methodology

    def test_thresholds_present(self, svc, ic_db):
        """IC tracking check should include threshold values."""
        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            svc.save_ic_snapshot(ic_mean=0.02)
            result = svc._check_ic_tracking()

        assert "ic_strong" in result.thresholds
        assert "ic_warn" in result.thresholds
        assert "ic_critical" in result.thresholds

    def test_raw_value_set(self, svc, ic_db):
        """raw_value should be set to rolling IC mean."""
        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            svc.save_ic_snapshot(ic_mean=0.025)
            result = svc._check_ic_tracking()

        assert result.raw_value == pytest.approx(0.025)

    def test_boundary_ic_at_zero(self, svc, ic_db):
        """IC exactly at 0 should be WARN (not FAIL)."""
        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            svc.save_ic_snapshot(ic_mean=0.0)
            result = svc._check_ic_tracking()

        assert result.status == "WARN"
        assert result.score == 40.0

    def test_boundary_ic_at_warn_threshold(self, svc, ic_db):
        """IC exactly at 0.01 should be PASS (not WARN)."""
        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            svc.save_ic_snapshot(ic_mean=0.01)
            result = svc._check_ic_tracking()

        assert result.status == "PASS"
        assert result.score == 70.0


# ═══════════════════════════════════════════════════════════════════════
# T3: Fallback to Cycle Report
# ═══════════════════════════════════════════════════════════════════════

class TestICFallbackToCycleReport:
    def test_fallback_extracts_ic_from_report(self, svc, ic_db, tmp_path):
        """When no DB history, should fall back to latest_cycle.json."""
        report = {
            "top_decisions": [
                {"metrics": {"ic_mean": 0.015, "ic_ir": 1.2}},
                {"metrics": {"ic_mean": 0.025, "ic_ir": 1.8}},
                {"metrics": {"sharpe": 0.5}},  # no ic_mean
            ]
        }
        report_path = tmp_path / "autopilot" / "latest_cycle.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f)

        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db), \
             patch("quant_engine.config.AUTOPILOT_CYCLE_REPORT", report_path):
            result = svc._check_ic_tracking()

        assert result.status == "PASS"
        assert result.raw_metrics["rolling_ic_mean"] == pytest.approx(0.025)
        assert result.raw_metrics["n_cycles"] == 1

    def test_fallback_no_ic_in_report(self, svc, ic_db, tmp_path):
        """Report with no IC values should return UNAVAILABLE."""
        report = {
            "top_decisions": [
                {"metrics": {"sharpe": 0.5}},
            ]
        }
        report_path = tmp_path / "autopilot" / "latest_cycle.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f)

        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db), \
             patch("quant_engine.config.AUTOPILOT_CYCLE_REPORT", report_path):
            result = svc._check_ic_tracking()

        assert result.status == "UNAVAILABLE"

    def test_extract_ic_from_cycle_report_static(self, tmp_path):
        """_extract_ic_from_cycle_report should return best IC."""
        report = {
            "top_decisions": [
                {"metrics": {"ic_mean": 0.01}},
                {"metrics": {"ic_mean": 0.05}},
                {"metrics": {"ic_mean": 0.03}},
            ]
        }
        report_path = tmp_path / "latest_cycle.json"
        with open(report_path, "w") as f:
            json.dump(report, f)

        with patch("quant_engine.config.AUTOPILOT_CYCLE_REPORT", report_path):
            result = HealthService._extract_ic_from_cycle_report()

        assert result == pytest.approx(0.05)

    def test_extract_ic_missing_report(self, tmp_path):
        """Missing report file should return None."""
        with patch("quant_engine.config.AUTOPILOT_CYCLE_REPORT",
                   tmp_path / "nonexistent.json"):
            result = HealthService._extract_ic_from_cycle_report()
        assert result is None


# ═══════════════════════════════════════════════════════════════════════
# T4: Integration with Signal Quality Domain
# ═══════════════════════════════════════════════════════════════════════

class TestICTrackingDomainIntegration:
    def test_ic_tracking_in_signal_quality_domain(self, svc, trades_csv, ic_db):
        """IC tracking check should appear in signal_quality domain."""
        with patch("quant_engine.config.RESULTS_DIR", trades_csv / "results"), \
             patch("quant_engine.config.MODEL_DIR", trades_csv / "models"), \
             patch("quant_engine.config.DATA_CACHE_DIR", trades_csv / "cache"), \
             patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            result = svc.compute_comprehensive_health()

        signal_checks = result["domains"]["signal_quality"]["checks"]
        check_names = [c["name"] for c in signal_checks]
        assert "ic_tracking" in check_names

    def test_signal_quality_has_5_checks(self, svc, trades_csv, ic_db):
        """Signal quality should now have 5 checks (4 original + IC tracking)."""
        with patch("quant_engine.config.RESULTS_DIR", trades_csv / "results"), \
             patch("quant_engine.config.MODEL_DIR", trades_csv / "models"), \
             patch("quant_engine.config.DATA_CACHE_DIR", trades_csv / "cache"), \
             patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            result = svc.compute_comprehensive_health()

        signal_checks = result["domains"]["signal_quality"]["checks"]
        assert len(signal_checks) == 5


# ═══════════════════════════════════════════════════════════════════════
# T5: Autopilot Engine IC Saving
# ═══════════════════════════════════════════════════════════════════════

class TestAutopilotICSaving:
    def test_save_ic_called_with_valid_data(self, ic_db):
        """_save_ic_to_health_tracking should save best IC from decisions."""
        from quant_engine.autopilot.engine import AutopilotEngine

        @dataclass
        class FakeCandidate:
            strategy_id: str = "test_strat"
            def to_dict(self):
                return {"strategy_id": self.strategy_id}

        @dataclass
        class FakeDecision:
            candidate: FakeCandidate = field(default_factory=FakeCandidate)
            passed: bool = False
            score: float = 50.0
            reasons: list = field(default_factory=list)
            metrics: Dict = field(default_factory=dict)

        decisions = [
            FakeDecision(metrics={"ic_mean": 0.02, "ic_ir": 1.2}),
            FakeDecision(metrics={"ic_mean": 0.05, "ic_ir": 2.1},
                        candidate=FakeCandidate(strategy_id="best_strat")),
            FakeDecision(metrics={"sharpe": 0.5}),  # no IC
        ]

        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            engine = AutopilotEngine.__new__(AutopilotEngine)
            engine.logger = MagicMock()
            engine._log = MagicMock()
            engine._save_ic_to_health_tracking(decisions, ["c1", "c2", "c3"])

            svc = HealthService()
            history = svc.get_ic_history(limit=10)

        assert len(history) == 1
        assert history[0]["ic_mean"] == pytest.approx(0.05)
        assert history[0]["ic_ir"] == pytest.approx(2.1)
        assert history[0]["n_candidates"] == 3

    def test_save_ic_no_ic_values(self, ic_db):
        """No IC values should not save anything."""
        from quant_engine.autopilot.engine import AutopilotEngine

        @dataclass
        class FakeDecision:
            passed: bool = False
            metrics: Dict = field(default_factory=dict)

        decisions = [
            FakeDecision(metrics={"sharpe": 0.5}),
        ]

        with patch.object(HealthService, "_get_ic_db_path", return_value=ic_db):
            engine = AutopilotEngine.__new__(AutopilotEngine)
            engine.logger = MagicMock()
            engine._log = MagicMock()
            engine._save_ic_to_health_tracking(decisions, ["c1"])

            svc = HealthService()
            history = svc.get_ic_history(limit=10)

        assert len(history) == 0

    def test_save_ic_handles_errors_gracefully(self, ic_db):
        """Errors during IC saving should be logged, not raised."""
        from quant_engine.autopilot.engine import AutopilotEngine

        engine = AutopilotEngine.__new__(AutopilotEngine)
        engine.logger = MagicMock()
        engine._log = MagicMock()

        # Pass invalid decisions (should not raise)
        engine._save_ic_to_health_tracking(None, [])  # type: ignore


# ═══════════════════════════════════════════════════════════════════════
# T6: Config Constants
# ═══════════════════════════════════════════════════════════════════════

class TestICTrackingConfig:
    def test_config_constants_exist(self):
        """IC tracking config constants should be importable."""
        from quant_engine.config import (
            IC_TRACKING_LOOKBACK,
            IC_TRACKING_WARN_THRESHOLD,
            IC_TRACKING_CRITICAL_THRESHOLD,
        )

        assert IC_TRACKING_LOOKBACK == 20
        assert IC_TRACKING_WARN_THRESHOLD == 0.01
        assert IC_TRACKING_CRITICAL_THRESHOLD == 0.0

    def test_warn_threshold_less_than_strong(self):
        """WARN threshold should be less than strong IC threshold."""
        from quant_engine.config import IC_TRACKING_WARN_THRESHOLD
        assert IC_TRACKING_WARN_THRESHOLD < 0.03  # strong threshold

    def test_critical_below_warn(self):
        """CRITICAL threshold should be below WARN threshold."""
        from quant_engine.config import (
            IC_TRACKING_WARN_THRESHOLD,
            IC_TRACKING_CRITICAL_THRESHOLD,
        )
        assert IC_TRACKING_CRITICAL_THRESHOLD < IC_TRACKING_WARN_THRESHOLD
