"""Tests for SPEC-H03: Execution quality monitoring in health system.

Verifies:
  - Execution quality fill save and retrieve via SQLite
  - Rolling cost surprise computation (actual_cost_bps - predicted_cost_bps)
  - WARNING threshold at mean surprise > 2 bps
  - CRITICAL threshold at mean surprise > 5 bps
  - Trend detection (worsening, improving, stable)
  - Edge cases: no data, single fill, all conservative
  - Cost surprise check wired into execution_quality domain
  - Paper trader saves fills to execution quality DB
  - Config constants accessible
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

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
def eq_db(tmp_path):
    """Provide a temporary execution quality tracking database path."""
    return tmp_path / "exec_quality.db"


def _save_fills(svc, eq_db, fills):
    """Helper to save multiple fill records under a patched DB path."""
    with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
        for fill in fills:
            svc.save_execution_quality_fill(**fill)


# ═══════════════════════════════════════════════════════════════════════
# T1: Execution Quality Fill Storage
# ═══════════════════════════════════════════════════════════════════════

class TestExecQualityStorage:
    def test_save_and_retrieve(self, svc, eq_db):
        """Fill records should round-trip through SQLite."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            svc.save_execution_quality_fill(
                symbol="AAPL",
                side="buy",
                predicted_cost_bps=5.0,
                actual_cost_bps=7.0,
                fill_ratio=0.95,
                participation_rate=0.01,
                regime=0,
            )

            history = svc.get_execution_quality_history(limit=10)

        assert len(history) == 1
        entry = history[0]
        assert entry["symbol"] == "AAPL"
        assert entry["side"] == "buy"
        assert entry["predicted_cost_bps"] == pytest.approx(5.0)
        assert entry["actual_cost_bps"] == pytest.approx(7.0)
        assert entry["cost_surprise_bps"] == pytest.approx(2.0)  # 7 - 5
        assert entry["fill_ratio"] == pytest.approx(0.95)
        assert entry["participation_rate"] == pytest.approx(0.01)
        assert entry["regime"] == 0
        assert "timestamp" in entry

    def test_cost_surprise_computation(self, svc, eq_db):
        """Cost surprise should be actual - predicted (positive = underestimates)."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            # Model overestimates (conservative) → negative surprise
            svc.save_execution_quality_fill(
                symbol="MSFT", side="buy",
                predicted_cost_bps=10.0, actual_cost_bps=6.0,
            )
            # Model underestimates → positive surprise
            svc.save_execution_quality_fill(
                symbol="GOOGL", side="sell",
                predicted_cost_bps=5.0, actual_cost_bps=9.0,
            )

            history = svc.get_execution_quality_history(limit=10)

        assert history[0]["cost_surprise_bps"] == pytest.approx(-4.0)  # 6 - 10
        assert history[1]["cost_surprise_bps"] == pytest.approx(4.0)   # 9 - 5

    def test_multiple_saves_ordered_chronologically(self, svc, eq_db):
        """Multiple fill records should be returned oldest-first."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            svc.save_execution_quality_fill(
                symbol="A", side="buy",
                predicted_cost_bps=5.0, actual_cost_bps=3.0,
            )
            svc.save_execution_quality_fill(
                symbol="B", side="sell",
                predicted_cost_bps=5.0, actual_cost_bps=5.0,
            )
            svc.save_execution_quality_fill(
                symbol="C", side="buy",
                predicted_cost_bps=5.0, actual_cost_bps=8.0,
            )

            history = svc.get_execution_quality_history(limit=10)

        assert len(history) == 3
        assert history[0]["symbol"] == "A"
        assert history[1]["symbol"] == "B"
        assert history[2]["symbol"] == "C"

    def test_limit_parameter(self, svc, eq_db):
        """get_execution_quality_history should respect the limit parameter."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            for i in range(10):
                svc.save_execution_quality_fill(
                    symbol=f"T{i}", side="buy",
                    predicted_cost_bps=5.0, actual_cost_bps=5.0 + i,
                )

            history = svc.get_execution_quality_history(limit=3)

        assert len(history) == 3
        # Should be the 3 most recent, ordered oldest-first
        assert history[0]["symbol"] == "T7"
        assert history[1]["symbol"] == "T8"
        assert history[2]["symbol"] == "T9"

    def test_pruning_old_entries(self, svc, eq_db):
        """Old entries beyond MAX_EXEC_QUALITY_RECORDS should be pruned."""
        original_max = HealthService._MAX_EXEC_QUALITY_RECORDS
        try:
            HealthService._MAX_EXEC_QUALITY_RECORDS = 5  # Small limit for testing
            with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
                for i in range(10):
                    svc.save_execution_quality_fill(
                        symbol=f"T{i}", side="buy",
                        predicted_cost_bps=5.0, actual_cost_bps=6.0,
                    )

                history = svc.get_execution_quality_history(limit=100)
            assert len(history) == 5
        finally:
            HealthService._MAX_EXEC_QUALITY_RECORDS = original_max

    def test_empty_database(self, svc, eq_db):
        """Empty database should return empty list."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            history = svc.get_execution_quality_history(limit=10)
        assert history == []

    def test_save_with_none_optional_fields(self, svc, eq_db):
        """Optional fields should handle None gracefully."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            svc.save_execution_quality_fill(
                symbol="NVDA", side="sell",
                predicted_cost_bps=8.0, actual_cost_bps=9.0,
            )
            history = svc.get_execution_quality_history(limit=10)

        assert len(history) == 1
        assert history[0]["fill_ratio"] is None
        assert history[0]["participation_rate"] is None
        assert history[0]["regime"] is None
        assert history[0]["cost_surprise_bps"] == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════
# T2: Cost Surprise Health Check
# ═══════════════════════════════════════════════════════════════════════

class TestCostSurpriseHealthCheck:
    def test_conservative_model_passes(self, svc, eq_db):
        """Mean cost surprise <= 0 should score 90 with PASS."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            for _ in range(5):
                svc.save_execution_quality_fill(
                    symbol="AAPL", side="buy",
                    predicted_cost_bps=10.0, actual_cost_bps=7.0,  # surprise = -3
                )

            result = svc._check_cost_surprise()

        assert result.name == "cost_surprise"
        assert result.domain == "execution_quality"
        assert result.status == "PASS"
        assert result.score == 90.0
        assert result.raw_metrics["mean_surprise_bps"] < 0
        assert "conservative" in result.explanation.lower()

    def test_slight_underestimation_passes(self, svc, eq_db):
        """Mean cost surprise between 0 and 2 bps should score 70 with PASS."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            for _ in range(5):
                svc.save_execution_quality_fill(
                    symbol="MSFT", side="buy",
                    predicted_cost_bps=5.0, actual_cost_bps=6.0,  # surprise = 1
                )

            result = svc._check_cost_surprise()

        assert result.status == "PASS"
        assert result.score == 70.0
        assert "underestimation" in result.explanation.lower()

    def test_meaningful_underestimation_warns(self, svc, eq_db):
        """Mean cost surprise between 2 and 5 bps should score 40 with WARN."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            for _ in range(5):
                svc.save_execution_quality_fill(
                    symbol="GOOGL", side="buy",
                    predicted_cost_bps=5.0, actual_cost_bps=8.5,  # surprise = 3.5
                )

            result = svc._check_cost_surprise()

        assert result.status == "WARN"
        assert result.score == 40.0
        assert "underestimates" in result.explanation.lower()

    def test_severe_underestimation_fails(self, svc, eq_db):
        """Mean cost surprise > 5 bps should score 15 with FAIL (CRITICAL)."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            for _ in range(5):
                svc.save_execution_quality_fill(
                    symbol="NVDA", side="buy",
                    predicted_cost_bps=3.0, actual_cost_bps=10.0,  # surprise = 7
                )

            result = svc._check_cost_surprise()

        assert result.status == "FAIL"
        assert result.score == 15.0
        assert "CRITICAL" in result.explanation
        assert "Severe" in result.explanation

    def test_no_data_unavailable(self, svc, eq_db):
        """No execution quality data should return UNAVAILABLE."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            result = svc._check_cost_surprise()

        assert result.status == "UNAVAILABLE"
        assert result.score == 0.0
        assert "No execution quality data" in result.explanation

    def test_single_fill(self, svc, eq_db):
        """Single fill should still produce a valid result."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            svc.save_execution_quality_fill(
                symbol="META", side="sell",
                predicted_cost_bps=8.0, actual_cost_bps=6.0,  # surprise = -2
            )
            result = svc._check_cost_surprise()

        assert result.status == "PASS"
        assert result.score == 90.0
        assert result.raw_metrics["n_fills"] == 1
        assert result.raw_metrics["trend"] == "unknown"

    def test_exact_zero_surprise_passes(self, svc, eq_db):
        """Mean surprise of exactly 0 should score 90 (conservative threshold is <=)."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            for _ in range(5):
                svc.save_execution_quality_fill(
                    symbol="AMZN", side="buy",
                    predicted_cost_bps=5.0, actual_cost_bps=5.0,  # surprise = 0
                )

            result = svc._check_cost_surprise()

        assert result.status == "PASS"
        assert result.score == 90.0
        assert result.raw_metrics["mean_surprise_bps"] == pytest.approx(0.0)

    def test_boundary_at_warn_threshold(self, svc, eq_db):
        """Mean surprise exactly at warn threshold should score 70 (<=)."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            for _ in range(5):
                svc.save_execution_quality_fill(
                    symbol="TSLA", side="buy",
                    predicted_cost_bps=5.0, actual_cost_bps=7.0,  # surprise = 2.0
                )

            result = svc._check_cost_surprise()

        assert result.status == "PASS"
        assert result.score == 70.0

    def test_boundary_at_critical_threshold(self, svc, eq_db):
        """Mean surprise exactly at critical threshold should score 40 (<=)."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            for _ in range(5):
                svc.save_execution_quality_fill(
                    symbol="JPM", side="buy",
                    predicted_cost_bps=5.0, actual_cost_bps=10.0,  # surprise = 5.0
                )

            result = svc._check_cost_surprise()

        assert result.status == "WARN"
        assert result.score == 40.0


# ═══════════════════════════════════════════════════════════════════════
# T3: Trend Detection
# ═══════════════════════════════════════════════════════════════════════

class TestCostSurpriseTrend:
    def test_trend_worsening(self, svc, eq_db):
        """Increasing cost surprise should be labeled 'worsening'."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            for i in range(5):
                svc.save_execution_quality_fill(
                    symbol=f"T{i}", side="buy",
                    predicted_cost_bps=5.0,
                    actual_cost_bps=5.0 + i * 1.0,  # surprise: 0, 1, 2, 3, 4
                )

            result = svc._check_cost_surprise()

        assert result.raw_metrics["trend"] == "worsening"
        assert result.raw_metrics["trend_slope_bps_per_fill"] > 0.05

    def test_trend_improving(self, svc, eq_db):
        """Decreasing cost surprise should be labeled 'improving'."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            for i in range(5):
                svc.save_execution_quality_fill(
                    symbol=f"T{i}", side="buy",
                    predicted_cost_bps=5.0,
                    actual_cost_bps=9.0 - i * 1.0,  # surprise: 4, 3, 2, 1, 0
                )

            result = svc._check_cost_surprise()

        assert result.raw_metrics["trend"] == "improving"
        assert result.raw_metrics["trend_slope_bps_per_fill"] < -0.05

    def test_trend_stable(self, svc, eq_db):
        """Constant cost surprise should be labeled 'stable'."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            for i in range(5):
                svc.save_execution_quality_fill(
                    symbol=f"T{i}", side="buy",
                    predicted_cost_bps=5.0, actual_cost_bps=6.0,  # surprise = 1 constant
                )

            result = svc._check_cost_surprise()

        assert result.raw_metrics["trend"] == "stable"

    def test_trend_unknown_with_fewer_than_3(self, svc, eq_db):
        """With fewer than 3 fills, trend should be 'unknown'."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            svc.save_execution_quality_fill(
                symbol="A", side="buy",
                predicted_cost_bps=5.0, actual_cost_bps=3.0,
            )
            svc.save_execution_quality_fill(
                symbol="B", side="sell",
                predicted_cost_bps=5.0, actual_cost_bps=4.0,
            )

            result = svc._check_cost_surprise()

        assert result.raw_metrics["trend"] == "unknown"


# ═══════════════════════════════════════════════════════════════════════
# T4: Raw Metrics Completeness
# ═══════════════════════════════════════════════════════════════════════

class TestCostSurpriseRawMetrics:
    def test_raw_metrics_complete(self, svc, eq_db):
        """Raw metrics should include all expected fields."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            for i in range(5):
                svc.save_execution_quality_fill(
                    symbol=f"T{i}", side="buy",
                    predicted_cost_bps=5.0, actual_cost_bps=6.0,
                )
            result = svc._check_cost_surprise()

        raw = result.raw_metrics
        assert "mean_surprise_bps" in raw
        assert "median_surprise_bps" in raw
        assert "std_surprise_bps" in raw
        assert "pct_underestimated" in raw
        assert "n_fills" in raw
        assert "trend" in raw
        assert "trend_slope_bps_per_fill" in raw
        assert "latest_surprise_bps" in raw
        assert raw["n_fills"] == 5

    def test_methodology_present(self, svc, eq_db):
        """Cost surprise check should have a non-empty methodology."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            svc.save_execution_quality_fill(
                symbol="AAPL", side="buy",
                predicted_cost_bps=5.0, actual_cost_bps=6.0,
            )
            result = svc._check_cost_surprise()

        assert result.methodology
        assert "cost surprise" in result.methodology.lower()
        assert "actual_cost_bps" in result.methodology

    def test_thresholds_in_result(self, svc, eq_db):
        """Thresholds dict should contain expected keys."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            svc.save_execution_quality_fill(
                symbol="AAPL", side="buy",
                predicted_cost_bps=5.0, actual_cost_bps=6.0,
            )
            result = svc._check_cost_surprise()

        assert "warn_surprise_bps" in result.thresholds
        assert "critical_surprise_bps" in result.thresholds
        assert result.thresholds["warn_surprise_bps"] == pytest.approx(2.0)
        assert result.thresholds["critical_surprise_bps"] == pytest.approx(5.0)

    def test_pct_underestimated_correct(self, svc, eq_db):
        """pct_underestimated should reflect the fraction of positive surprises."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            # 3 underestimates (positive surprise), 2 overestimates (negative)
            svc.save_execution_quality_fill(
                symbol="A", side="buy",
                predicted_cost_bps=5.0, actual_cost_bps=8.0,  # +3
            )
            svc.save_execution_quality_fill(
                symbol="B", side="buy",
                predicted_cost_bps=5.0, actual_cost_bps=7.0,  # +2
            )
            svc.save_execution_quality_fill(
                symbol="C", side="buy",
                predicted_cost_bps=5.0, actual_cost_bps=6.0,  # +1
            )
            svc.save_execution_quality_fill(
                symbol="D", side="buy",
                predicted_cost_bps=5.0, actual_cost_bps=3.0,  # -2
            )
            svc.save_execution_quality_fill(
                symbol="E", side="buy",
                predicted_cost_bps=5.0, actual_cost_bps=2.0,  # -3
            )

            result = svc._check_cost_surprise()

        assert result.raw_metrics["pct_underestimated"] == pytest.approx(0.6)

    def test_raw_value_is_mean_surprise(self, svc, eq_db):
        """raw_value should equal the mean cost surprise."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            for _ in range(5):
                svc.save_execution_quality_fill(
                    symbol="AAPL", side="buy",
                    predicted_cost_bps=5.0, actual_cost_bps=7.0,  # surprise = 2.0
                )

            result = svc._check_cost_surprise()

        assert result.raw_value == pytest.approx(2.0)


# ═══════════════════════════════════════════════════════════════════════
# T5: Integration with Execution Quality Domain
# ═══════════════════════════════════════════════════════════════════════

class TestCostSurpriseDomainIntegration:
    def test_cost_surprise_in_exec_domain(self, svc, eq_db, tmp_path):
        """cost_surprise should appear in the execution_quality domain checks."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db), \
             patch.object(HealthService, "_get_ic_db_path", return_value=tmp_path / "ic.db"), \
             patch.object(HealthService, "_get_disagreement_db_path", return_value=tmp_path / "d.db"), \
             patch("quant_engine.config.RESULTS_DIR", str(tmp_path)), \
             patch("quant_engine.config.AUTOPILOT_CYCLE_REPORT", str(tmp_path / "nonexistent.json")), \
             patch("quant_engine.config.MODEL_DIR", str(tmp_path / "models")):
            runtime = svc.compute_comprehensive_health()

        exec_domain = runtime["domains"]["execution_quality"]
        check_names = [c["name"] for c in exec_domain["checks"]]
        assert "cost_surprise" in check_names

    def test_cost_surprise_severity_is_standard(self, svc, eq_db, tmp_path):
        """cost_surprise check should use standard severity in exec domain."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db), \
             patch.object(HealthService, "_get_ic_db_path", return_value=tmp_path / "ic.db"), \
             patch.object(HealthService, "_get_disagreement_db_path", return_value=tmp_path / "d.db"), \
             patch("quant_engine.config.RESULTS_DIR", str(tmp_path)), \
             patch("quant_engine.config.AUTOPILOT_CYCLE_REPORT", str(tmp_path / "nonexistent.json")), \
             patch("quant_engine.config.MODEL_DIR", str(tmp_path / "models")):
            runtime = svc.compute_comprehensive_health()

        exec_domain = runtime["domains"]["execution_quality"]
        cs_check = next(
            c for c in exec_domain["checks"] if c["name"] == "cost_surprise"
        )
        # In exec domain, all checks are set to severity="standard"
        assert cs_check["severity"] == "standard"


# ═══════════════════════════════════════════════════════════════════════
# T6: Paper Trader Integration
# ═══════════════════════════════════════════════════════════════════════

class TestPaperTraderExecQualityWiring:
    def test_record_fill_feedback_calls_exec_quality(self, tmp_path, eq_db):
        """_record_fill_feedback should call _record_execution_quality."""
        # Create a minimal fill object
        fill = MagicMock()
        fill.fill_ratio = 0.95
        fill.participation_rate = 0.015
        fill.fill_price = 101.0  # buy: slippage = (101 - 100) / 100 * 10000 = 100 bps
        fill.cost_details = {"total_bps": 8.5}

        with patch("quant_engine.autopilot.paper_trader.EXEC_CALIBRATION_FEEDBACK_ENABLED", False):
            import pandas as pd
            from quant_engine.autopilot.paper_trader import PaperTrader

            trader = PaperTrader(state_path=tmp_path / "state.json")

            with patch.object(trader, "_record_execution_quality") as mock_rec:
                trader._record_fill_feedback(
                    fill=fill,
                    reference_price=100.0,
                    side="buy",
                    symbol="AAPL",
                    daily_volume=1_000_000,
                    price=101.0,
                    regime=0,
                    as_of=pd.Timestamp("2026-01-15"),
                )

            mock_rec.assert_called_once()
            call_kwargs = mock_rec.call_args[1]
            assert call_kwargs["symbol"] == "AAPL"
            assert call_kwargs["side"] == "buy"
            assert call_kwargs["predicted_cost_bps"] == pytest.approx(8.5)
            assert call_kwargs["fill_ratio"] == pytest.approx(0.95)
            assert call_kwargs["participation_rate"] == pytest.approx(0.015)
            assert call_kwargs["regime"] == 0

    def test_record_execution_quality_saves_to_db(self, tmp_path, eq_db):
        """_record_execution_quality should save fill data to health service DB."""
        with patch("quant_engine.autopilot.paper_trader.EXEC_CALIBRATION_FEEDBACK_ENABLED", False):
            from quant_engine.autopilot.paper_trader import PaperTrader

            trader = PaperTrader(state_path=tmp_path / "state.json")

            with patch.object(
                HealthService, "_get_exec_quality_db_path", return_value=eq_db,
            ), patch(
                "quant_engine.tracking.execution_tracker._get_exec_quality_db_path",
                return_value=eq_db,
            ):
                trader._record_execution_quality(
                    symbol="MSFT",
                    side="sell",
                    predicted_cost_bps=6.0,
                    actual_cost_bps=8.0,
                    fill_ratio=0.92,
                    participation_rate=0.02,
                    regime=1,
                )

                svc = HealthService()
                history = svc.get_execution_quality_history(limit=10)

        assert len(history) == 1
        assert history[0]["symbol"] == "MSFT"
        assert history[0]["cost_surprise_bps"] == pytest.approx(2.0)

    def test_record_fill_feedback_skips_bad_fills(self, tmp_path):
        """_record_fill_feedback should skip fills with zero fill_ratio."""
        fill = MagicMock()
        fill.fill_ratio = 0.0  # Bad fill
        fill.participation_rate = 0.015
        fill.fill_price = 101.0
        fill.cost_details = {"total_bps": 8.5}

        with patch("quant_engine.autopilot.paper_trader.EXEC_CALIBRATION_FEEDBACK_ENABLED", False):
            import pandas as pd
            from quant_engine.autopilot.paper_trader import PaperTrader

            trader = PaperTrader(state_path=tmp_path / "state.json")

            with patch.object(trader, "_record_execution_quality") as mock_rec:
                trader._record_fill_feedback(
                    fill=fill,
                    reference_price=100.0,
                    side="buy",
                    symbol="AAPL",
                    daily_volume=1_000_000,
                    price=101.0,
                    regime=0,
                    as_of=pd.Timestamp("2026-01-15"),
                )

            mock_rec.assert_not_called()

    def test_record_fill_feedback_skips_zero_predicted_cost(self, tmp_path):
        """_record_fill_feedback should skip fills with zero predicted cost."""
        fill = MagicMock()
        fill.fill_ratio = 0.95
        fill.participation_rate = 0.015
        fill.fill_price = 101.0
        fill.cost_details = {"total_bps": 0.0}  # Zero predicted cost

        with patch("quant_engine.autopilot.paper_trader.EXEC_CALIBRATION_FEEDBACK_ENABLED", False):
            import pandas as pd
            from quant_engine.autopilot.paper_trader import PaperTrader

            trader = PaperTrader(state_path=tmp_path / "state.json")

            with patch.object(trader, "_record_execution_quality") as mock_rec:
                trader._record_fill_feedback(
                    fill=fill,
                    reference_price=100.0,
                    side="buy",
                    symbol="AAPL",
                    daily_volume=1_000_000,
                    price=101.0,
                    regime=0,
                    as_of=pd.Timestamp("2026-01-15"),
                )

            mock_rec.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════
# T7: Config Constants
# ═══════════════════════════════════════════════════════════════════════

class TestExecQualityConfig:
    def test_config_constants_exist(self):
        """EXEC_QUALITY_* constants should be importable from config."""
        from quant_engine.config import (
            EXEC_QUALITY_LOOKBACK,
            EXEC_QUALITY_WARN_SURPRISE_BPS,
            EXEC_QUALITY_CRITICAL_SURPRISE_BPS,
        )
        assert isinstance(EXEC_QUALITY_LOOKBACK, int)
        assert isinstance(EXEC_QUALITY_WARN_SURPRISE_BPS, float)
        assert isinstance(EXEC_QUALITY_CRITICAL_SURPRISE_BPS, float)

    def test_warn_less_than_critical(self):
        """Warning threshold should be less than critical threshold."""
        from quant_engine.config import (
            EXEC_QUALITY_WARN_SURPRISE_BPS,
            EXEC_QUALITY_CRITICAL_SURPRISE_BPS,
        )
        assert EXEC_QUALITY_WARN_SURPRISE_BPS < EXEC_QUALITY_CRITICAL_SURPRISE_BPS

    def test_lookback_positive(self):
        """Lookback should be a positive integer."""
        from quant_engine.config import EXEC_QUALITY_LOOKBACK
        assert EXEC_QUALITY_LOOKBACK > 0


# ═══════════════════════════════════════════════════════════════════════
# T8: Edge Cases and Mixed Scenarios
# ═══════════════════════════════════════════════════════════════════════

class TestCostSurpriseEdgeCases:
    def test_mixed_buy_sell_fills(self, svc, eq_db):
        """Mix of buy and sell fills should all be tracked."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            svc.save_execution_quality_fill(
                symbol="AAPL", side="buy",
                predicted_cost_bps=5.0, actual_cost_bps=4.0,
            )
            svc.save_execution_quality_fill(
                symbol="MSFT", side="sell",
                predicted_cost_bps=5.0, actual_cost_bps=3.0,
            )

            history = svc.get_execution_quality_history(limit=10)

        assert len(history) == 2
        assert history[0]["side"] == "buy"
        assert history[1]["side"] == "sell"

    def test_all_negative_surprises(self, svc, eq_db):
        """All negative surprises (conservative model) should score 90."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            for i in range(10):
                svc.save_execution_quality_fill(
                    symbol=f"T{i}", side="buy",
                    predicted_cost_bps=10.0, actual_cost_bps=5.0,
                )

            result = svc._check_cost_surprise()

        assert result.score == 90.0
        assert result.raw_metrics["pct_underestimated"] == pytest.approx(0.0)

    def test_large_surprise_values(self, svc, eq_db):
        """Very large surprise values should not cause numerical issues."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            svc.save_execution_quality_fill(
                symbol="PENNY", side="buy",
                predicted_cost_bps=2.0, actual_cost_bps=100.0,  # surprise = 98
            )

            result = svc._check_cost_surprise()

        assert result.status == "FAIL"
        assert result.score == 15.0
        assert np.isfinite(result.raw_metrics["mean_surprise_bps"])

    def test_multiple_regimes(self, svc, eq_db):
        """Fills across different regimes should all be tracked."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            for regime in [0, 1, 2, 3]:
                svc.save_execution_quality_fill(
                    symbol=f"R{regime}", side="buy",
                    predicted_cost_bps=5.0, actual_cost_bps=4.0,
                    regime=regime,
                )

            history = svc.get_execution_quality_history(limit=10)

        assert len(history) == 4
        regimes = [h["regime"] for h in history]
        assert set(regimes) == {0, 1, 2, 3}

    def test_db_table_creation_idempotent(self, svc, eq_db):
        """Calling _ensure_exec_quality_table multiple times should be safe."""
        with patch.object(HealthService, "_get_exec_quality_db_path", return_value=eq_db):
            HealthService._ensure_exec_quality_table()
            HealthService._ensure_exec_quality_table()
            svc.save_execution_quality_fill(
                symbol="TEST", side="buy",
                predicted_cost_bps=5.0, actual_cost_bps=5.0,
            )
            history = svc.get_execution_quality_history(limit=10)

        assert len(history) == 1
