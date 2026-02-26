"""Tests for the rewritten health check system (Spec 010).

Verifies:
  - UNAVAILABLE results return score=0 (not 50)
  - methodology field is always populated
  - domain scoring excludes UNAVAILABLE checks
  - signal decay uses rolling IC trend
  - survivorship checks actual data columns
  - execution quality handles missing TCA
  - overall score calculation with weighted domains
  - HealthCheckResult dataclass structure
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
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
def trades_csv(tmp_path):
    """Create a realistic backtest trades CSV."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "ticker": np.random.choice(["AAPL", "MSFT", "GOOGL", "NVDA", "META"], n),
        "entry_date": pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d"),
        "exit_date": pd.date_range("2024-01-15", periods=n, freq="B").strftime("%Y-%m-%d"),
        "entry_price": np.random.uniform(100, 200, n),
        "exit_price": np.random.uniform(100, 200, n),
        "predicted_return": np.random.normal(0.005, 0.02, n),
        "actual_return": np.random.normal(0.003, 0.03, n),
        "net_return": np.random.normal(0.002, 0.03, n),
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
def paper_state(tmp_path):
    """Create a paper state with some positions."""
    state = {
        "cash": 800000.0,
        "equity": 1000000.0,
        "realized_pnl": 5000.0,
        "positions": [
            {"ticker": "AAPL", "value": 50000.0},
            {"ticker": "MSFT", "value": 50000.0},
            {"ticker": "GOOGL", "value": 40000.0},
            {"ticker": "NVDA", "value": 30000.0},
            {"ticker": "META", "value": 30000.0},
        ],
        "trades": [],
        "last_update": "2026-02-20T00:00:00+00:00",
    }
    path = tmp_path / "results" / "autopilot" / "paper_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f)
    return tmp_path


@pytest.fixture
def model_registry(tmp_path):
    """Create a model registry with 2 versions."""
    reg = {
        "latest": "v2",
        "versions": [
            {"version_id": "v1", "cv_gap": 0.04, "holdout_spearman": 0.08},
            {"version_id": "v2", "cv_gap": 0.03, "holdout_spearman": 0.10},
        ],
    }
    path = tmp_path / "trained_models" / "registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(reg, f)
    return tmp_path


# ── Test 1: UNAVAILABLE returns score=0, not 50 ────────────────────────

class TestUnavailableScoring:
    def test_unavailable_helper_returns_zero(self):
        result = _unavailable("test_check", "test_domain", "No data")
        assert result.score == 0.0
        assert result.status == "UNAVAILABLE"
        assert result.data_available is False

    def test_unavailable_signal_decay_no_data(self, svc, tmp_path):
        """Signal decay should return score=0 when no trades CSV exists."""
        with patch("quant_engine.config.RESULTS_DIR", tmp_path / "results"):
            result = svc._check_signal_decay()
        assert result.status == "UNAVAILABLE"
        assert result.score == 0.0
        assert result.score != 50.0  # explicitly check it's not the old default

    def test_unavailable_tail_risk_no_data(self, svc, tmp_path):
        with patch("quant_engine.config.RESULTS_DIR", tmp_path / "results"):
            result = svc._check_tail_risk()
        assert result.status == "UNAVAILABLE"
        assert result.score == 0.0

    def test_unavailable_execution_no_data(self, svc, tmp_path):
        with patch("quant_engine.config.RESULTS_DIR", tmp_path / "results"):
            result = svc._check_execution_quality()
        assert result.status == "UNAVAILABLE"
        assert result.score == 0.0


# ── Test 2: methodology field always present ───────────────────────────

class TestMethodologyField:
    def test_unavailable_has_methodology(self):
        result = _unavailable("test", "domain", "reason")
        assert result.methodology  # non-empty

    def test_signal_decay_has_methodology(self, svc, trades_csv):
        with patch("quant_engine.config.RESULTS_DIR", trades_csv / "results"):
            result = svc._check_signal_decay()
        assert result.methodology
        assert "IC" in result.methodology or "rolling" in result.methodology.lower()

    def test_all_checks_have_methodology(self, svc, trades_csv, paper_state, model_registry):
        """Every check must populate the methodology field."""
        with patch("quant_engine.config.RESULTS_DIR", trades_csv / "results"), \
             patch("quant_engine.config.MODEL_DIR", model_registry / "trained_models"), \
             patch("quant_engine.config.DATA_CACHE_DIR", trades_csv / "cache"):
            checks = [
                svc._check_signal_decay(),
                svc._check_prediction_distribution(),
                svc._check_tail_risk(),
                svc._check_execution_quality(),
                svc._check_cv_gap_trend(),
            ]
        for check in checks:
            assert check.methodology, f"{check.name} missing methodology"


# ── Test 3: domain scoring excludes UNAVAILABLE ────────────────────────

class TestDomainScoring:
    def test_excludes_unavailable_from_average(self, svc):
        checks = [
            HealthCheckResult(name="a", domain="test", score=80.0, status="PASS"),
            _unavailable("b", "test", "no data"),
            HealthCheckResult(name="c", domain="test", score=60.0, status="WARN"),
        ]
        # Should average 80 and 60, not 80, 0, and 60
        score = svc._domain_score(checks)
        assert score == pytest.approx(70.0)

    def test_all_unavailable_returns_none(self, svc):
        checks = [
            _unavailable("a", "test", "no data"),
            _unavailable("b", "test", "no data"),
        ]
        score = svc._domain_score(checks)
        assert score is None

    def test_domain_status_unavailable_majority(self, svc):
        checks = [
            _unavailable("a", "test", "no data"),
            _unavailable("b", "test", "no data"),
            HealthCheckResult(name="c", domain="test", score=80.0, status="PASS"),
        ]
        status = svc._domain_status(checks)
        assert status == "UNAVAILABLE"

    def test_domain_status_pass(self, svc):
        checks = [
            HealthCheckResult(name="a", domain="test", score=80.0, status="PASS"),
            HealthCheckResult(name="b", domain="test", score=90.0, status="PASS"),
        ]
        status = svc._domain_status(checks)
        assert status == "PASS"


# ── Test 4: signal decay uses rolling IC, not autocorrelation ──────────

class TestSignalDecayIC:
    def test_declining_ic_detected(self, svc, tmp_path):
        """Create trades where IC declines over time → should WARN or FAIL."""
        np.random.seed(0)
        n = 400
        predicted = np.random.normal(0.005, 0.02, n)
        # First half: actual correlates with predicted (good IC)
        actual = np.zeros(n)
        actual[:200] = predicted[:200] * 0.5 + np.random.normal(0, 0.01, 200)
        # Second half: no correlation (IC decayed)
        actual[200:] = np.random.normal(0, 0.02, 200)

        df = pd.DataFrame({
            "ticker": "AAPL", "entry_date": "2024-01-01", "exit_date": "2024-01-10",
            "entry_price": 150.0, "exit_price": 155.0,
            "predicted_return": predicted, "actual_return": actual,
            "net_return": actual - 0.001, "regime": "trending_bull",
            "confidence": 0.7, "holding_days": 10, "position_size": 0.05,
            "exit_reason": "trailing_stop",
        })
        path = tmp_path / "results" / "backtest_10d_trades.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

        with patch("quant_engine.config.RESULTS_DIR", tmp_path / "results"):
            result = svc._check_signal_decay()

        assert result.name == "signal_decay"
        assert result.data_available is True
        # With declining IC, should not be PASS
        assert result.status in ("WARN", "FAIL")
        assert "ratio" in result.raw_metrics

    def test_stable_ic_passes(self, svc, tmp_path):
        """Create trades with stable IC → should PASS."""
        np.random.seed(42)
        n = 400
        predicted = np.random.normal(0.005, 0.02, n)
        actual = predicted * 0.3 + np.random.normal(0, 0.015, n)

        df = pd.DataFrame({
            "ticker": "AAPL", "entry_date": "2024-01-01", "exit_date": "2024-01-10",
            "entry_price": 150.0, "exit_price": 155.0,
            "predicted_return": predicted, "actual_return": actual,
            "net_return": actual - 0.001, "regime": "trending_bull",
            "confidence": 0.7, "holding_days": 10, "position_size": 0.05,
            "exit_reason": "trailing_stop",
        })
        path = tmp_path / "results" / "backtest_10d_trades.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

        with patch("quant_engine.config.RESULTS_DIR", tmp_path / "results"):
            result = svc._check_signal_decay()

        assert result.status == "PASS"
        assert result.score >= 80


# ── Test 5: survivorship checks data columns ──────────────────────────

class TestSurvivorshipCheck:
    def test_with_total_ret_column(self, svc, tmp_path):
        """Parquets with total_ret column should improve survivorship score."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create parquet WITH total_ret column — use recent dates so
        # AAPL is NOT flagged as delisted (must be within last year)
        df = pd.DataFrame({
            "Close": np.random.uniform(100, 200, 100),
            "Volume": np.random.uniform(1e6, 1e7, 100),
            "total_ret": np.random.normal(0.001, 0.02, 100),
        }, index=pd.date_range("2025-10-01", periods=100))
        df.to_parquet(cache_dir / "AAPL_1d.parquet")

        # Create parquet with old data (delisted proxy)
        df_old = pd.DataFrame({
            "Close": np.random.uniform(50, 100, 100),
            "Volume": np.random.uniform(1e5, 1e6, 100),
            "total_ret": np.random.normal(-0.01, 0.05, 100),
        }, index=pd.date_range("2020-01-01", periods=100))
        df_old.to_parquet(cache_dir / "DELIST_1d.parquet")

        # Patch RESULTS_DIR too so the enhanced survivorship check
        # doesn't load real trade data from outside the test fixture
        with patch("quant_engine.config.DATA_CACHE_DIR", cache_dir), \
             patch("quant_engine.config.RESULTS_DIR", tmp_path / "results"):
            result = svc._check_survivorship_bias()

        assert result.status == "PASS"
        assert result.score >= 85
        assert result.raw_metrics["has_total_ret"] > 0

    def test_without_total_ret_column(self, svc, tmp_path):
        """Parquets without total_ret and recent data should FAIL survivorship."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Use very recent dates so no ticker appears delisted
        df = pd.DataFrame({
            "Close": np.random.uniform(100, 200, 100),
            "Volume": np.random.uniform(1e6, 1e7, 100),
        }, index=pd.date_range("2025-10-01", periods=100))
        df.to_parquet(cache_dir / "AAPL_1d.parquet")

        with patch("quant_engine.config.DATA_CACHE_DIR", cache_dir):
            result = svc._check_survivorship_bias()

        assert result.status == "FAIL"
        assert result.score <= 35


# ── Test 6: execution quality handles missing TCA ──────────────────────

class TestExecutionQualityTCA:
    def test_without_tca_uses_correlation_proxy(self, svc, trades_csv):
        """Without TCA columns, should use pred-actual correlation proxy."""
        with patch("quant_engine.config.RESULTS_DIR", trades_csv / "results"):
            result = svc._check_execution_quality()

        assert result.data_available is True
        assert result.raw_metrics.get("data_source") == "prediction_proxy"

    def test_with_tca_columns(self, svc, tmp_path):
        """With TCA columns, should analyze fill ratio and impact."""
        n = 100
        df = pd.DataFrame({
            "ticker": "AAPL",
            "predicted_return": np.random.normal(0.005, 0.02, n),
            "actual_return": np.random.normal(0.003, 0.03, n),
            "net_return": np.random.normal(0.002, 0.03, n),
            "fill_ratio": np.random.uniform(0.90, 1.0, n),
            "entry_impact_bps": np.random.uniform(1, 5, n),
            "exit_impact_bps": np.random.uniform(1, 5, n),
            "position_size": 0.05,
            "entry_date": "2024-01-01", "exit_date": "2024-01-10",
            "entry_price": 150, "exit_price": 155,
            "regime": "trending_bull", "confidence": 0.7,
            "holding_days": 10, "exit_reason": "trailing_stop",
        })
        path = tmp_path / "results" / "backtest_10d_trades.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

        with patch("quant_engine.config.RESULTS_DIR", tmp_path / "results"):
            result = svc._check_execution_quality()

        assert result.raw_metrics.get("data_source") == "TCA"
        assert result.status == "PASS"


# ── Test 7: overall score calculation ──────────────────────────────────

class TestOverallScoreCalculation:
    def test_comprehensive_health_structure(self, svc, trades_csv, paper_state, model_registry):
        """Verify output structure of compute_comprehensive_health."""
        with patch("quant_engine.config.RESULTS_DIR", trades_csv / "results"), \
             patch("quant_engine.config.MODEL_DIR", model_registry / "trained_models"), \
             patch("quant_engine.config.DATA_CACHE_DIR", trades_csv / "nonexistent"):
            result = svc.compute_comprehensive_health()

        assert "overall_score" in result
        assert "overall_status" in result
        assert "checks_available" in result
        assert "checks_total" in result
        assert result["overall_status"] in ("PASS", "WARN", "FAIL")
        assert result["checks_total"] > 0
        assert isinstance(result["checks_available"], int)
        assert isinstance(result["overall_score"], float)

        # Check domain structure
        for domain_name in ["data_integrity", "signal_quality", "risk_management",
                            "execution_quality", "model_governance"]:
            assert domain_name in result["domains"]
            d = result["domains"][domain_name]
            assert "score" in d
            assert "weight" in d
            assert "status" in d
            assert "checks" in d

    def test_weights_sum_to_one(self, svc, trades_csv):
        with patch("quant_engine.config.RESULTS_DIR", trades_csv / "results"), \
             patch("quant_engine.config.MODEL_DIR", trades_csv / "models"), \
             patch("quant_engine.config.DATA_CACHE_DIR", trades_csv / "cache"):
            result = svc.compute_comprehensive_health()

        total_weight = sum(d["weight"] for d in result["domains"].values())
        assert total_weight == pytest.approx(1.0)


# ── Test 8: HealthCheckResult dataclass ────────────────────────────────

class TestHealthCheckResultDataclass:
    def test_to_dict(self):
        r = HealthCheckResult(
            name="test", domain="test_domain", score=85.0, status="PASS",
            explanation="All good", methodology="Check stuff",
            raw_metrics={"a": 1}, thresholds={"t": 0.5},
        )
        d = r.to_dict()
        assert d["name"] == "test"
        assert d["score"] == 85.0
        assert d["methodology"] == "Check stuff"
        assert d["data_available"] is True

    def test_default_values(self):
        r = HealthCheckResult(name="x", domain="d", score=0.0, status="FAIL")
        assert r.explanation == ""
        assert r.methodology == ""
        assert r.data_available is True
        assert r.raw_metrics == {}
        assert r.thresholds == {}
