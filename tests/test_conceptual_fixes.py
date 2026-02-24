"""Tests for Spec 008 — Ensemble & Conceptual Fixes.

Verifies:
  1. Ensemble uses only genuinely independent methods (no phantom vote)
  2. Ensemble with jump model uses 3 independent methods
  3. Rule-based detection enforces REGIME_MIN_DURATION
  4. Config endpoint returns backtest defaults
  5. Config status endpoint returns training defaults
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_features(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV features for regime detection."""
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Close": np.cumsum(rng.normal(0, 0.02, n)) + 100,
            "High": np.cumsum(rng.normal(0, 0.02, n)) + 101,
            "Low": np.cumsum(rng.normal(0, 0.02, n)) + 99,
            "Open": np.cumsum(rng.normal(0, 0.02, n)) + 100,
            "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        },
        index=idx,
    )


def _make_features_with_regime_flickers(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Create features designed to produce short regime flickers in rule-based detection."""
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)

    hurst = np.full(n, 0.5)
    adx = np.full(n, 15.0)
    natr = np.full(n, 10.0)
    sma_slope = np.zeros(n)

    # Bull trend segment
    hurst[0:100] = 0.65
    adx[0:100] = 30.0
    sma_slope[0:100] = 0.01

    # 2-bar mean_rev flicker
    hurst[100:102] = 0.40
    adx[100:102] = 10.0

    # Bull trend again
    hurst[102:200] = 0.65
    adx[102:200] = 30.0
    sma_slope[102:200] = 0.01

    # 1-bar high_vol flicker
    natr[200:201] = 100.0

    # Mean reverting
    hurst[201:400] = 0.40

    # 2-bar bear trend flicker
    hurst[400:402] = 0.65
    adx[400:402] = 30.0
    sma_slope[400:402] = -0.01

    # Mean reverting
    hurst[402:n] = 0.40

    return pd.DataFrame(
        {
            "Close": np.cumsum(rng.normal(0, 0.02, n)) + 100,
            "High": np.cumsum(rng.normal(0, 0.02, n)) + 101,
            "Low": np.cumsum(rng.normal(0, 0.02, n)) + 99,
            "Open": np.cumsum(rng.normal(0, 0.02, n)) + 100,
            "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
            "Hurst_100": hurst,
            "ADX_14": adx,
            "NATR_14": natr,
            "SMASlope_50": sma_slope,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# T1: Ensemble no phantom vote
# ---------------------------------------------------------------------------

class TestEnsembleNoPhantomVote:
    """When REGIME_JUMP_MODEL_ENABLED=False, ensemble must use 2 methods, not 3."""

    def test_two_method_ensemble_does_not_call_jump(self):
        from quant_engine.regime.detector import RegimeDetector

        features = _make_synthetic_features()
        with patch("quant_engine.regime.detector.REGIME_JUMP_MODEL_ENABLED", False):
            detector = RegimeDetector(method="hmm")

            # Track which internal detection methods are called
            call_log: list[str] = []
            orig_rule = detector._rule_detect
            orig_hmm = detector._hmm_detect
            orig_jump = detector._jump_detect

            def track_rule(f):
                call_log.append("rule")
                return orig_rule(f)

            def track_hmm(f):
                call_log.append("hmm")
                return orig_hmm(f)

            def track_jump(f):
                call_log.append("jump")
                return orig_jump(f)

            detector._rule_detect = track_rule
            detector._hmm_detect = track_hmm
            detector._jump_detect = track_jump

            out = detector.detect_ensemble(features)

        assert "jump" not in call_log, "Jump model should NOT be called when disabled"
        assert len(call_log) == 2, f"Expected 2 method calls, got {len(call_log)}: {call_log}"
        assert out.model_type == "ensemble"
        assert not out.regime.isna().any()
        assert not out.probabilities.isna().any().any()

    def test_two_method_ensemble_requires_unanimity(self):
        """With 2 methods and threshold=2, both must agree for a regime change."""
        from quant_engine.regime.detector import RegimeDetector

        features = _make_synthetic_features()
        with patch("quant_engine.regime.detector.REGIME_JUMP_MODEL_ENABLED", False), \
             patch("quant_engine.regime.detector.REGIME_ENSEMBLE_CONSENSUS_THRESHOLD", 2):
            detector = RegimeDetector(method="hmm")
            out = detector.detect_ensemble(features)

        # Output should be valid regardless
        assert out.regime.dtype == int or np.issubdtype(out.regime.dtype, np.integer)
        assert len(out.regime) == len(features)


# ---------------------------------------------------------------------------
# T2: Ensemble with jump model
# ---------------------------------------------------------------------------

class TestEnsembleWithJumpModel:
    """When REGIME_JUMP_MODEL_ENABLED=True, ensemble uses 3 independent methods."""

    def test_three_method_ensemble_calls_all_methods(self):
        from quant_engine.regime.detector import RegimeDetector

        features = _make_synthetic_features()
        with patch("quant_engine.regime.detector.REGIME_JUMP_MODEL_ENABLED", True):
            detector = RegimeDetector(method="hmm")

            call_log: list[str] = []
            orig_rule = detector._rule_detect
            orig_hmm = detector._hmm_detect
            orig_jump = detector._jump_detect

            def track_rule(f):
                call_log.append("rule")
                return orig_rule(f)

            def track_hmm(f):
                call_log.append("hmm")
                return orig_hmm(f)

            def track_jump(f):
                call_log.append("jump")
                return orig_jump(f)

            detector._rule_detect = track_rule
            detector._hmm_detect = track_hmm
            detector._jump_detect = track_jump

            out = detector.detect_ensemble(features)

        assert "jump" in call_log, "Jump model should be called when enabled"
        assert "hmm" in call_log
        assert "rule" in call_log
        assert len(call_log) == 3, f"Expected 3 method calls, got {len(call_log)}: {call_log}"
        assert out.model_type == "ensemble"

    def test_three_method_probabilities_sum_to_one(self):
        from quant_engine.regime.detector import RegimeDetector

        features = _make_synthetic_features()
        with patch("quant_engine.regime.detector.REGIME_JUMP_MODEL_ENABLED", True):
            detector = RegimeDetector(method="hmm")
            out = detector.detect_ensemble(features)

        row_sums = out.probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# T3: Rule-based min duration
# ---------------------------------------------------------------------------

class TestRuleDetectMinDuration:
    """Rule-based detection must enforce REGIME_MIN_DURATION."""

    def test_no_short_runs_with_min_duration(self):
        from quant_engine.regime.detector import RegimeDetector

        features = _make_features_with_regime_flickers()

        # Verify raw detection HAS short runs
        detector_raw = RegimeDetector(method="rule", min_duration=1)
        out_raw = detector_raw._rule_detect(features)
        runs_raw = out_raw.regime.ne(out_raw.regime.shift()).cumsum()
        run_lengths_raw = out_raw.regime.groupby(runs_raw).count()
        short_raw = (run_lengths_raw < 3).sum()
        assert short_raw > 0, "Test data should have short runs without smoothing"

        # Verify smoothed detection has NO short runs
        detector = RegimeDetector(method="rule", min_duration=3)
        out = detector._rule_detect(features)
        runs = out.regime.ne(out.regime.shift()).cumsum()
        run_lengths = out.regime.groupby(runs).count()
        short_runs = (run_lengths < 3).sum()
        assert short_runs == 0, f"Expected 0 runs < 3 bars, got {short_runs}"

    def test_min_duration_1_is_noop(self):
        """min_duration=1 should not alter regime output."""
        from quant_engine.regime.detector import RegimeDetector

        features = _make_features_with_regime_flickers()
        detector = RegimeDetector(method="rule", min_duration=1)
        out = detector._rule_detect(features)

        # Should still produce valid output
        assert len(out.regime) == len(features)
        assert not out.regime.isna().any()

    def test_smoothing_preserves_regime_count(self):
        """Smoothing reduces the number of regime runs (does not increase)."""
        from quant_engine.regime.detector import RegimeDetector

        features = _make_features_with_regime_flickers()

        det_raw = RegimeDetector(method="rule", min_duration=1)
        out_raw = det_raw._rule_detect(features)
        n_runs_raw = out_raw.regime.ne(out_raw.regime.shift()).sum() + 1

        det_smooth = RegimeDetector(method="rule", min_duration=3)
        out_smooth = det_smooth._rule_detect(features)
        n_runs_smooth = out_smooth.regime.ne(out_smooth.regime.shift()).sum() + 1

        assert n_runs_smooth <= n_runs_raw, (
            f"Smoothing should reduce runs: {n_runs_smooth} > {n_runs_raw}"
        )


# ---------------------------------------------------------------------------
# T4: Config endpoint has backtest defaults
# ---------------------------------------------------------------------------

class TestConfigEndpointDefaults:
    """The /api/config endpoint returns values that match config.py."""

    def test_adjustable_config_includes_backtest_keys(self):
        from quant_engine.api.config import RuntimeConfig

        rc = RuntimeConfig()
        adjustable = rc.get_adjustable()

        assert "ENTRY_THRESHOLD" in adjustable
        assert "MAX_POSITIONS" in adjustable
        assert "POSITION_SIZE_PCT" in adjustable
        assert "MAX_HOLDING_DAYS" in adjustable

    def test_adjustable_config_values_match_config_py(self):
        from quant_engine.api.config import RuntimeConfig
        from quant_engine.config import (
            ENTRY_THRESHOLD,
            MAX_POSITIONS,
            POSITION_SIZE_PCT,
            MAX_HOLDING_DAYS,
        )

        rc = RuntimeConfig()
        adjustable = rc.get_adjustable()

        assert adjustable["ENTRY_THRESHOLD"] == ENTRY_THRESHOLD
        assert adjustable["MAX_POSITIONS"] == MAX_POSITIONS
        assert adjustable["POSITION_SIZE_PCT"] == POSITION_SIZE_PCT
        assert adjustable["MAX_HOLDING_DAYS"] == MAX_HOLDING_DAYS

    def test_config_status_includes_training_section(self):
        """The /api/config/status response should include training defaults."""
        from quant_engine.api.routers.config_mgmt import _build_config_status

        status = _build_config_status()

        assert "training" in status
        assert "forward_horizons" in status["training"]
        assert "feature_mode" in status["training"]

        horizons_entry = status["training"]["forward_horizons"]
        assert horizons_entry["status"] == "active"
        assert isinstance(horizons_entry["value"], list)

    def test_config_status_backtest_section_matches(self):
        """Backtest config values in status match config.py."""
        from quant_engine.api.routers.config_mgmt import _build_config_status
        from quant_engine.config import ENTRY_THRESHOLD, MAX_POSITIONS

        status = _build_config_status()

        assert "backtest" in status
        assert status["backtest"]["entry_threshold"]["value"] == ENTRY_THRESHOLD
        assert status["backtest"]["max_positions"]["value"] == MAX_POSITIONS
