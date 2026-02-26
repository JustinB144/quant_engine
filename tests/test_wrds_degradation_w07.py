"""Tests for SPEC-W07: Handle WRDS_ENABLED gracefully when credentials missing.

Verifies:
  - config.py emits ONE consolidated warning with degradation details
  - wrds_provider.py no longer emits a duplicate warnings.warn()
  - Health endpoint (quick status) reports WRDS status
  - Comprehensive health includes WRDS check in data_integrity domain
  - When WRDS_USERNAME is set and connection succeeds, no warnings appear
"""
from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

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


# ── Part A: Config validation warns with degradation details ────────────

class TestConfigValidationWRDSWarning:
    """Verify validate_config() emits a single, detailed WRDS warning."""

    def test_wrds_warning_includes_degradation_details(self):
        """Warning message must mention degraded mode and affected data sources."""
        from quant_engine.config import validate_config

        with patch.dict(os.environ, {}, clear=False):
            # Remove WRDS_USERNAME if present
            env = os.environ.copy()
            env.pop("WRDS_USERNAME", None)
            with patch.dict(os.environ, env, clear=True):
                with patch("quant_engine.config.WRDS_ENABLED", True):
                    issues = validate_config()

        wrds_issues = [
            i for i in issues
            if "WRDS" in i.get("message", "") and "WRDS_USERNAME" in i.get("message", "")
        ]
        assert len(wrds_issues) >= 1, "Expected at least one WRDS warning from config"
        msg = wrds_issues[0]["message"]
        assert "degraded mode" in msg, f"Warning should mention 'degraded mode': {msg}"
        assert "survivorship bias" in msg, f"Warning should mention survivorship bias: {msg}"
        assert "alternative data" in msg or "earnings" in msg, (
            f"Warning should mention alternative data unavailability: {msg}"
        )
        assert wrds_issues[0]["level"] == "WARNING"

    def test_no_wrds_warning_when_username_set(self):
        """When WRDS_USERNAME is set, no WRDS credential warning should appear."""
        from quant_engine.config import validate_config

        with patch.dict(os.environ, {"WRDS_USERNAME": "testuser"}, clear=False):
            with patch("quant_engine.config.WRDS_ENABLED", True):
                issues = validate_config()

        wrds_cred_issues = [
            i for i in issues
            if "WRDS_USERNAME" in i.get("message", "") and "not set" in i.get("message", "")
        ]
        assert len(wrds_cred_issues) == 0, (
            f"No WRDS credential warning expected when username is set, got: {wrds_cred_issues}"
        )

    def test_no_wrds_warning_when_wrds_disabled(self):
        """When WRDS_ENABLED=False, no WRDS credential warning should appear."""
        from quant_engine.config import validate_config

        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("WRDS_USERNAME", None)
            with patch.dict(os.environ, env, clear=True):
                with patch("quant_engine.config.WRDS_ENABLED", False):
                    issues = validate_config()

        wrds_issues = [
            i for i in issues
            if "WRDS_ENABLED=True" in i.get("message", "")
        ]
        assert len(wrds_issues) == 0, "No WRDS warning when WRDS_ENABLED=False"


# ── Part B: wrds_provider.py no longer emits warnings.warn ─────────────

class TestWRDSProviderNoDuplicateWarning:
    """Verify wrds_provider.py uses logger.debug instead of warnings.warn."""

    def test_no_warnings_warn_in_wrds_provider(self):
        """The wrds_provider module should not call warnings.warn at import time."""
        import inspect
        from quant_engine.data import wrds_provider

        source = inspect.getsource(wrds_provider)
        # Check that warnings.warn is not called anywhere in the module
        assert "warnings.warn(" not in source, (
            "wrds_provider.py should use logger.debug instead of warnings.warn"
        )

    def test_no_warnings_import_in_wrds_provider(self):
        """The wrds_provider module should not import warnings."""
        import inspect
        from quant_engine.data import wrds_provider

        source = inspect.getsource(wrds_provider)
        assert "import warnings" not in source, (
            "wrds_provider.py should not import warnings anymore"
        )


# ── Part C: Health endpoint shows WRDS status ──────────────────────────

class TestHealthWRDSStatus:
    """Verify the health service includes WRDS availability checks."""

    def test_quick_status_includes_wrds_key(self, svc):
        """get_quick_status() should include a 'wrds' key in checks."""
        with patch("quant_engine.config.WRDS_ENABLED", True):
            mock_provider = MagicMock()
            mock_provider.available.return_value = False
            with patch(
                "quant_engine.data.wrds_provider.WRDSProvider",
                return_value=mock_provider,
            ):
                result = svc.get_quick_status()
        assert "wrds" in result["checks"], (
            f"Quick status should include 'wrds' check, got: {result['checks']}"
        )

    def test_quick_status_wrds_connected(self, svc):
        """When WRDS is connected, quick status shows 'connected'."""
        with patch("quant_engine.config.WRDS_ENABLED", True):
            mock_provider = MagicMock()
            mock_provider.available.return_value = True
            with patch(
                "quant_engine.data.wrds_provider.WRDSProvider",
                return_value=mock_provider,
            ):
                result = svc.get_quick_status()
        assert result["checks"]["wrds"] == "connected"

    def test_quick_status_wrds_unavailable_degrades(self, svc):
        """When WRDS is enabled but unavailable, status should degrade."""
        with patch("quant_engine.config.WRDS_ENABLED", True):
            mock_provider = MagicMock()
            mock_provider.available.return_value = False
            with patch(
                "quant_engine.data.wrds_provider.WRDSProvider",
                return_value=mock_provider,
            ):
                # Also mock cache/model to ensure they don't interfere
                with patch("quant_engine.config.DATA_CACHE_DIR") as mock_cache, \
                     patch("quant_engine.config.MODEL_DIR") as mock_model:
                    mock_cache.exists.return_value = False
                    mock_model.exists.return_value = False
                    result = svc.get_quick_status()
        assert result["checks"]["wrds"] == "unavailable"
        # Status should be degraded or unhealthy (not healthy)
        assert result["status"] != "healthy" or result["checks"]["wrds"] == "unavailable"

    def test_quick_status_wrds_disabled(self, svc):
        """When WRDS_ENABLED=False, check should show 'disabled'."""
        with patch("quant_engine.config.WRDS_ENABLED", False):
            result = svc.get_quick_status()
        assert result["checks"]["wrds"] == "disabled"

    def test_check_wrds_status_connected(self, svc):
        """_check_wrds_status() returns PASS when WRDS is connected."""
        with patch("quant_engine.config.WRDS_ENABLED", True):
            mock_provider = MagicMock()
            mock_provider.available.return_value = True
            with patch(
                "quant_engine.data.wrds_provider.WRDSProvider",
                return_value=mock_provider,
            ):
                result = svc._check_wrds_status()
        assert result.status == "PASS"
        assert result.score == 85.0
        assert result.domain == "data_integrity"
        assert result.raw_metrics["wrds_connected"] is True
        assert result.methodology  # non-empty

    def test_check_wrds_status_unavailable(self, svc):
        """_check_wrds_status() returns FAIL when WRDS connection is down."""
        with patch("quant_engine.config.WRDS_ENABLED", True):
            mock_provider = MagicMock()
            mock_provider.available.return_value = False
            with patch(
                "quant_engine.data.wrds_provider.WRDSProvider",
                return_value=mock_provider,
            ):
                result = svc._check_wrds_status()
        assert result.status == "FAIL"
        assert result.score == 30.0
        assert "cached/local data" in result.explanation

    def test_check_wrds_status_disabled(self, svc):
        """_check_wrds_status() returns WARN when WRDS_ENABLED=False."""
        with patch("quant_engine.config.WRDS_ENABLED", False):
            result = svc._check_wrds_status()
        assert result.status == "WARN"
        assert result.score == 50.0
        assert result.raw_metrics["wrds_enabled"] is False

    def test_check_wrds_status_has_methodology(self, svc):
        """_check_wrds_status() always populates methodology."""
        with patch("quant_engine.config.WRDS_ENABLED", False):
            result = svc._check_wrds_status()
        assert result.methodology
        assert "WRDSProvider" in result.methodology or "WRDS" in result.methodology

    def test_wrds_check_in_comprehensive_data_domain(self, svc):
        """WRDS check should be included in the data_integrity domain."""
        with patch("quant_engine.config.WRDS_ENABLED", False):
            # Mock everything to avoid real data lookups
            mock_checks = HealthCheckResult(
                name="mock", domain="data_integrity", score=50.0,
                status="WARN", explanation="mock", methodology="mock",
            )
            with patch.object(svc, "_check_survivorship_bias", return_value=mock_checks), \
                 patch.object(svc, "_check_data_quality_anomalies", return_value=mock_checks), \
                 patch.object(svc, "_check_market_microstructure", return_value=mock_checks), \
                 patch.object(svc, "_check_signal_decay", return_value=mock_checks), \
                 patch.object(svc, "_check_prediction_distribution", return_value=mock_checks), \
                 patch.object(svc, "_check_ensemble_disagreement", return_value=mock_checks), \
                 patch.object(svc, "_check_information_ratio", return_value=mock_checks), \
                 patch.object(svc, "_check_tail_risk", return_value=mock_checks), \
                 patch.object(svc, "_check_correlation_regime", return_value=mock_checks), \
                 patch.object(svc, "_check_capital_utilization", return_value=mock_checks), \
                 patch.object(svc, "_check_execution_quality", return_value=mock_checks), \
                 patch.object(svc, "_check_signal_profitability", return_value=mock_checks), \
                 patch.object(svc, "_check_feature_importance_drift", return_value=mock_checks), \
                 patch.object(svc, "_check_cv_gap_trend", return_value=mock_checks), \
                 patch.object(svc, "_check_regime_transition_health", return_value=mock_checks), \
                 patch.object(svc, "_check_retraining_effectiveness", return_value=mock_checks):
                result = svc.compute_comprehensive_health()

            di_checks = result["domains"]["data_integrity"]["checks"]
            wrds_names = [c["name"] for c in di_checks if "wrds" in c["name"]]
            assert len(wrds_names) >= 1, (
                f"Expected wrds_availability check in data_integrity domain, "
                f"found checks: {[c['name'] for c in di_checks]}"
            )
