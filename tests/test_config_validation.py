"""Tests for SPEC-A02: Config validation completeness.

Verifies that validate_config() catches invalid configuration values
including ensemble weight sums, negative costs, threshold ranges,
label horizons, blend weights, Kelly fraction, drawdown ordering,
and other critical constraints.
"""

import pytest
from unittest.mock import patch


class TestEnsembleWeightValidation:
    """SPEC-A02: Ensemble weights must sum to 1.0."""

    def test_valid_ensemble_weights_no_error(self):
        """Default ensemble weights (0.5 + 0.3 + 0.2 = 1.0) should pass."""
        from quant_engine.config import validate_config

        issues = validate_config()
        ensemble_errors = [
            i for i in issues
            if "REGIME_ENSEMBLE_DEFAULT_WEIGHTS" in i["message"]
        ]
        assert not ensemble_errors, f"Unexpected ensemble weight errors: {ensemble_errors}"

    def test_ensemble_weights_sum_too_high(self):
        """Weights summing to 1.1 should produce an ERROR."""
        bad_weights = {"hmm": 0.5, "rule": 0.3, "jump": 0.3}
        with patch("quant_engine.config.REGIME_ENSEMBLE_DEFAULT_WEIGHTS", bad_weights):
            from quant_engine.config import validate_config

            issues = validate_config()
        ensemble_errors = [
            i for i in issues
            if "REGIME_ENSEMBLE_DEFAULT_WEIGHTS" in i["message"]
            and i["level"] == "ERROR"
        ]
        assert len(ensemble_errors) == 1
        assert "1.1" in ensemble_errors[0]["message"]

    def test_ensemble_weights_sum_too_low(self):
        """Weights summing to 0.8 should produce an ERROR."""
        bad_weights = {"hmm": 0.4, "rule": 0.2, "jump": 0.2}
        with patch("quant_engine.config.REGIME_ENSEMBLE_DEFAULT_WEIGHTS", bad_weights):
            from quant_engine.config import validate_config

            issues = validate_config()
        ensemble_errors = [
            i for i in issues
            if "REGIME_ENSEMBLE_DEFAULT_WEIGHTS" in i["message"]
            and i["level"] == "ERROR"
        ]
        assert len(ensemble_errors) == 1


class TestCostMultiplierValidation:
    """SPEC-A02: No negative cost multipliers."""

    def test_negative_transaction_cost(self):
        """Negative TRANSACTION_COST_BPS should produce an ERROR."""
        with patch("quant_engine.config.TRANSACTION_COST_BPS", -5):
            from quant_engine.config import validate_config

            issues = validate_config()
        cost_errors = [
            i for i in issues
            if "TRANSACTION_COST_BPS" in i["message"] and i["level"] == "ERROR"
        ]
        assert len(cost_errors) == 1

    def test_negative_spread(self):
        """Negative EXEC_SPREAD_BPS should produce an ERROR."""
        with patch("quant_engine.config.EXEC_SPREAD_BPS", -1.0):
            from quant_engine.config import validate_config

            issues = validate_config()
        spread_errors = [
            i for i in issues
            if "EXEC_SPREAD_BPS" in i["message"] and i["level"] == "ERROR"
        ]
        assert len(spread_errors) == 1

    def test_negative_impact_coeff(self):
        """Negative EXEC_IMPACT_COEFF_BPS should produce an ERROR."""
        with patch("quant_engine.config.EXEC_IMPACT_COEFF_BPS", -10.0):
            from quant_engine.config import validate_config

            issues = validate_config()
        impact_errors = [
            i for i in issues
            if "EXEC_IMPACT_COEFF_BPS" in i["message"] and i["level"] == "ERROR"
        ]
        assert len(impact_errors) == 1

    def test_zero_cost_is_valid(self):
        """Zero transaction costs are valid (e.g., simulating no-fee trading)."""
        with patch("quant_engine.config.TRANSACTION_COST_BPS", 0):
            from quant_engine.config import validate_config

            issues = validate_config()
        cost_errors = [
            i for i in issues
            if "TRANSACTION_COST_BPS" in i["message"] and i["level"] == "ERROR"
        ]
        assert not cost_errors


class TestEntropyThresholdValidation:
    """SPEC-A02: Regime threshold ranges."""

    def test_entropy_threshold_zero(self):
        """Entropy threshold of 0 should produce an ERROR (boundary exclusive)."""
        with patch("quant_engine.config.REGIME_UNCERTAINTY_ENTROPY_THRESHOLD", 0.0):
            from quant_engine.config import validate_config

            issues = validate_config()
        entropy_errors = [
            i for i in issues
            if "REGIME_UNCERTAINTY_ENTROPY_THRESHOLD" in i["message"]
            and i["level"] == "ERROR"
        ]
        assert len(entropy_errors) == 1

    def test_entropy_threshold_too_high(self):
        """Entropy threshold >= 2.0 should produce an ERROR."""
        with patch("quant_engine.config.REGIME_UNCERTAINTY_ENTROPY_THRESHOLD", 2.0):
            from quant_engine.config import validate_config

            issues = validate_config()
        entropy_errors = [
            i for i in issues
            if "REGIME_UNCERTAINTY_ENTROPY_THRESHOLD" in i["message"]
            and i["level"] == "ERROR"
        ]
        assert len(entropy_errors) == 1

    def test_entropy_threshold_negative(self):
        """Negative entropy threshold should produce an ERROR."""
        with patch("quant_engine.config.REGIME_UNCERTAINTY_ENTROPY_THRESHOLD", -0.5):
            from quant_engine.config import validate_config

            issues = validate_config()
        entropy_errors = [
            i for i in issues
            if "REGIME_UNCERTAINTY_ENTROPY_THRESHOLD" in i["message"]
            and i["level"] == "ERROR"
        ]
        assert len(entropy_errors) == 1

    def test_valid_entropy_threshold(self):
        """Default entropy threshold (0.50) should pass."""
        from quant_engine.config import validate_config

        issues = validate_config()
        entropy_errors = [
            i for i in issues
            if "REGIME_UNCERTAINTY_ENTROPY_THRESHOLD" in i["message"]
        ]
        assert not entropy_errors


class TestLabelHorizonValidation:
    """SPEC-A02: Label horizon must be positive."""

    def test_label_h_zero(self):
        """LABEL_H=0 should produce an ERROR."""
        with patch("quant_engine.config.LABEL_H", 0):
            from quant_engine.config import validate_config

            issues = validate_config()
        label_errors = [
            i for i in issues
            if "LABEL_H" in i["message"] and i["level"] == "ERROR"
        ]
        assert len(label_errors) == 1

    def test_label_h_negative(self):
        """LABEL_H=-1 should produce an ERROR."""
        with patch("quant_engine.config.LABEL_H", -1):
            from quant_engine.config import validate_config

            issues = validate_config()
        label_errors = [
            i for i in issues
            if "LABEL_H" in i["message"] and i["level"] == "ERROR"
        ]
        assert len(label_errors) == 1

    def test_label_h_valid(self):
        """LABEL_H=5 should pass."""
        from quant_engine.config import validate_config

        issues = validate_config()
        label_errors = [
            i for i in issues
            if "LABEL_H" in i["message"] and "invalid" in i["message"].lower()
        ]
        assert not label_errors


class TestForwardHorizonsValidation:
    """Forward horizons must all be positive integers."""

    def test_empty_forward_horizons(self):
        """Empty FORWARD_HORIZONS should produce an ERROR."""
        with patch("quant_engine.config.FORWARD_HORIZONS", []):
            from quant_engine.config import validate_config

            issues = validate_config()
        horizon_errors = [
            i for i in issues
            if "FORWARD_HORIZONS" in i["message"] and i["level"] == "ERROR"
        ]
        assert len(horizon_errors) == 1

    def test_negative_forward_horizon(self):
        """Negative value in FORWARD_HORIZONS should produce an ERROR."""
        with patch("quant_engine.config.FORWARD_HORIZONS", [5, -1, 20]):
            from quant_engine.config import validate_config

            issues = validate_config()
        horizon_errors = [
            i for i in issues
            if "FORWARD_HORIZONS" in i["message"] and i["level"] == "ERROR"
        ]
        assert len(horizon_errors) == 1


class TestBlendWeightsValidation:
    """Blend weights must sum to 1.0."""

    def test_static_blend_weights_invalid(self):
        """Static blend weights that don't sum to 1.0 should produce an ERROR."""
        bad_blend = {"kelly": 0.5, "vol_scaled": 0.4, "atr_based": 0.3}
        with patch("quant_engine.config.BLEND_WEIGHTS_STATIC", bad_blend):
            from quant_engine.config import validate_config

            issues = validate_config()
        blend_errors = [
            i for i in issues
            if "BLEND_WEIGHTS_STATIC" in i["message"] and i["level"] == "ERROR"
        ]
        assert len(blend_errors) == 1

    def test_regime_blend_weights_invalid(self):
        """Per-regime blend weights that don't sum to 1.0 should produce an ERROR."""
        bad_regime_blend = {
            "NORMAL": {"kelly": 0.5, "vol_scaled": 0.5, "atr_based": 0.5},
            "WARNING": {"kelly": 0.25, "vol_scaled": 0.45, "atr_based": 0.30},
        }
        with patch("quant_engine.config.BLEND_WEIGHTS_BY_REGIME", bad_regime_blend):
            from quant_engine.config import validate_config

            issues = validate_config()
        blend_errors = [
            i for i in issues
            if "BLEND_WEIGHTS_BY_REGIME" in i["message"] and i["level"] == "ERROR"
        ]
        # Only NORMAL should fail (1.5 != 1.0); WARNING sums to 1.0
        assert len(blend_errors) == 1
        assert "NORMAL" in blend_errors[0]["message"]

    def test_valid_blend_weights(self):
        """Default blend weights should pass."""
        from quant_engine.config import validate_config

        issues = validate_config()
        blend_errors = [
            i for i in issues
            if "BLEND_WEIGHTS" in i["message"] and i["level"] == "ERROR"
        ]
        assert not blend_errors


class TestKellyFractionValidation:
    """Kelly fraction must be in (0, 1]."""

    def test_kelly_fraction_zero(self):
        """KELLY_FRACTION=0 should produce an ERROR."""
        with patch("quant_engine.config.KELLY_FRACTION", 0.0):
            from quant_engine.config import validate_config

            issues = validate_config()
        kelly_errors = [
            i for i in issues
            if "KELLY_FRACTION" in i["message"] and i["level"] == "ERROR"
        ]
        assert len(kelly_errors) == 1

    def test_kelly_fraction_above_one(self):
        """KELLY_FRACTION=1.5 (super-Kelly) should produce an ERROR."""
        with patch("quant_engine.config.KELLY_FRACTION", 1.5):
            from quant_engine.config import validate_config

            issues = validate_config()
        kelly_errors = [
            i for i in issues
            if "KELLY_FRACTION" in i["message"] and i["level"] == "ERROR"
        ]
        assert len(kelly_errors) == 1

    def test_kelly_fraction_one(self):
        """KELLY_FRACTION=1.0 (full Kelly) should pass."""
        with patch("quant_engine.config.KELLY_FRACTION", 1.0):
            from quant_engine.config import validate_config

            issues = validate_config()
        kelly_errors = [
            i for i in issues
            if "KELLY_FRACTION" in i["message"] and i["level"] == "ERROR"
        ]
        assert not kelly_errors


class TestDrawdownThresholdValidation:
    """Drawdown thresholds must be negative and ordered WARNING > CAUTION > CRITICAL."""

    def test_positive_drawdown_threshold(self):
        """Positive drawdown threshold should produce an ERROR."""
        with patch("quant_engine.config.DRAWDOWN_WARNING_THRESHOLD", 0.05):
            from quant_engine.config import validate_config

            issues = validate_config()
        dd_errors = [
            i for i in issues
            if "Drawdown thresholds" in i["message"] and i["level"] == "ERROR"
        ]
        assert len(dd_errors) == 1

    def test_misordered_drawdown_thresholds(self):
        """WARNING < CAUTION should produce an ERROR (wrong order)."""
        with patch("quant_engine.config.DRAWDOWN_WARNING_THRESHOLD", -0.15), \
             patch("quant_engine.config.DRAWDOWN_CAUTION_THRESHOLD", -0.10), \
             patch("quant_engine.config.DRAWDOWN_CRITICAL_THRESHOLD", -0.05):
            from quant_engine.config import validate_config

            issues = validate_config()
        dd_errors = [
            i for i in issues
            if "Drawdown thresholds" in i["message"] and i["level"] == "ERROR"
        ]
        assert len(dd_errors) == 1
        assert "ordered" in dd_errors[0]["message"]


class TestPositionSizingValidation:
    """Position sizing parameters must be valid."""

    def test_position_size_pct_zero(self):
        """POSITION_SIZE_PCT=0 should produce an ERROR."""
        with patch("quant_engine.config.POSITION_SIZE_PCT", 0.0):
            from quant_engine.config import validate_config

            issues = validate_config()
        psp_errors = [
            i for i in issues
            if "POSITION_SIZE_PCT" in i["message"] and i["level"] == "ERROR"
        ]
        assert len(psp_errors) == 1

    def test_position_size_pct_over_one(self):
        """POSITION_SIZE_PCT=1.5 (>100% of capital) should produce an ERROR."""
        with patch("quant_engine.config.POSITION_SIZE_PCT", 1.5):
            from quant_engine.config import validate_config

            issues = validate_config()
        psp_errors = [
            i for i in issues
            if "POSITION_SIZE_PCT" in i["message"] and i["level"] == "ERROR"
        ]
        assert len(psp_errors) == 1

    def test_max_positions_zero(self):
        """MAX_POSITIONS=0 should produce an ERROR."""
        with patch("quant_engine.config.MAX_POSITIONS", 0):
            from quant_engine.config import validate_config

            issues = validate_config()
        mp_errors = [
            i for i in issues
            if "MAX_POSITIONS" in i["message"] and i["level"] == "ERROR"
        ]
        assert len(mp_errors) == 1


class TestConfidenceThresholdValidation:
    """Confidence threshold must be in [0, 1]."""

    def test_confidence_below_zero(self):
        """Negative confidence threshold should produce an ERROR."""
        with patch("quant_engine.config.CONFIDENCE_THRESHOLD", -0.1):
            from quant_engine.config import validate_config

            issues = validate_config()
        conf_errors = [
            i for i in issues
            if "CONFIDENCE_THRESHOLD" in i["message"] and i["level"] == "ERROR"
        ]
        assert len(conf_errors) == 1

    def test_confidence_above_one(self):
        """Confidence > 1.0 should produce an ERROR."""
        with patch("quant_engine.config.CONFIDENCE_THRESHOLD", 1.5):
            from quant_engine.config import validate_config

            issues = validate_config()
        conf_errors = [
            i for i in issues
            if "CONFIDENCE_THRESHOLD" in i["message"] and i["level"] == "ERROR"
        ]
        assert len(conf_errors) == 1


class TestPortfolioVolValidation:
    """MAX_PORTFOLIO_VOL must be positive."""

    def test_zero_portfolio_vol(self):
        """MAX_PORTFOLIO_VOL=0 should produce an ERROR."""
        with patch("quant_engine.config.MAX_PORTFOLIO_VOL", 0.0):
            from quant_engine.config import validate_config

            issues = validate_config()
        vol_errors = [
            i for i in issues
            if "MAX_PORTFOLIO_VOL" in i["message"] and i["level"] == "ERROR"
        ]
        assert len(vol_errors) == 1


class TestRegimeRiskMultiplierValidation:
    """Regime risk multipliers must be non-negative."""

    def test_negative_risk_multiplier(self):
        """Negative regime risk multiplier should produce an ERROR."""
        bad_mults = {0: 1.0, 1: -0.5, 2: 0.95, 3: 0.60}
        with patch("quant_engine.config.REGIME_RISK_MULTIPLIER", bad_mults):
            from quant_engine.config import validate_config

            issues = validate_config()
        mult_errors = [
            i for i in issues
            if "REGIME_RISK_MULTIPLIER" in i["message"] and i["level"] == "ERROR"
        ]
        assert len(mult_errors) == 1
        assert "[1]" in mult_errors[0]["message"]


class TestDefaultConfigValid:
    """The default config.py values must pass all validations without ERRORs."""

    def test_no_errors_in_default_config(self):
        """Default config should not produce any ERROR-level issues from the new checks."""
        from quant_engine.config import validate_config

        issues = validate_config()
        # Filter to only SPEC-A02 related errors (checks 7-20)
        a02_keywords = [
            "REGIME_ENSEMBLE_DEFAULT_WEIGHTS",
            "TRANSACTION_COST_BPS",
            "EXEC_SPREAD_BPS",
            "EXEC_IMPACT_COEFF_BPS",
            "REGIME_UNCERTAINTY_ENTROPY_THRESHOLD",
            "LABEL_H",
            "FORWARD_HORIZONS",
            "BLEND_WEIGHTS_STATIC",
            "BLEND_WEIGHTS_BY_REGIME",
            "KELLY_FRACTION",
            "Drawdown thresholds",
            "POSITION_SIZE_PCT",
            "MAX_POSITIONS",
            "CONFIDENCE_THRESHOLD",
            "MAX_PORTFOLIO_VOL",
            "REGIME_RISK_MULTIPLIER",
        ]
        a02_errors = [
            i for i in issues
            if i["level"] == "ERROR"
            and any(kw in i["message"] for kw in a02_keywords)
        ]
        assert not a02_errors, f"Default config produced SPEC-A02 errors: {a02_errors}"

    def test_validate_config_returns_list(self):
        """validate_config() must return a list."""
        from quant_engine.config import validate_config

        issues = validate_config()
        assert isinstance(issues, list)

    def test_validate_config_issues_have_required_keys(self):
        """Each issue must have 'level' and 'message' keys."""
        from quant_engine.config import validate_config

        issues = validate_config()
        for issue in issues:
            assert "level" in issue, f"Issue missing 'level': {issue}"
            assert "message" in issue, f"Issue missing 'message': {issue}"
            assert issue["level"] in ("WARNING", "ERROR"), (
                f"Invalid level '{issue['level']}' in issue: {issue}"
            )
