"""End-to-end integration tests for the upgraded regime detection pipeline (SPEC_10 T7).

Verifies the full pipeline works together:
  - Confidence-weighted ensemble voting (T2)
  - Regime uncertainty gating (T3)
  - Expanded observation matrix (T4)
  - Online regime updating (T5)
  - Cross-sectional regime consensus (T6)
  - MIN_REGIME_SAMPLES reduction (T7)
  - Regime transitions across market phases
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_phase_features():
    """Generate synthetic features with clear multi-phase regime structure.

    Phases:
      - Phase 1 (0-200):  low-vol uptrend   (trending_bull, regime 0)
      - Phase 2 (200-350): high-vol crash    (high_volatility, regime 3)
      - Phase 3 (350-500): sideways chop     (mean_reverting, regime 2)
      - Phase 4 (500-600): moderate recovery (trending_bull, regime 0)
    """
    np.random.seed(42)
    n = 600
    idx = pd.date_range("2020-01-01", periods=n, freq="B")

    prices = np.zeros(n)
    prices[0] = 100.0
    for i in range(1, n):
        if i < 200:
            prices[i] = prices[i - 1] * (1 + 0.0005 + np.random.randn() * 0.008)
        elif i < 350:
            prices[i] = prices[i - 1] * (1 - 0.002 + np.random.randn() * 0.025)
        elif i < 500:
            prices[i] = prices[i - 1] * (1 + np.random.randn() * 0.005)
        else:
            prices[i] = prices[i - 1] * (1 + 0.0003 + np.random.randn() * 0.010)

    features = pd.DataFrame(
        {
            "Close": prices,
            "High": prices * (1 + np.abs(np.random.randn(n) * 0.005)),
            "Low": prices * (1 - np.abs(np.random.randn(n) * 0.005)),
            "Open": prices * (1 + np.random.randn(n) * 0.002),
            "Volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
        },
        index=idx,
    )
    features["return_1d"] = features["Close"].pct_change()
    features["return_vol_20d"] = features["return_1d"].rolling(20).std()
    features["NATR_14"] = (features["High"] - features["Low"]) / features["Close"]
    features["SMASlope_50"] = features["Close"].rolling(50).mean().pct_change(5)
    features["Hurst_100"] = 0.5
    features["ADX_14"] = 25.0
    features["return_20d"] = features["return_1d"].rolling(20).sum()
    features["GARCH_252"] = features["return_1d"].rolling(60).std()
    features["AutoCorr_20_1"] = features["return_1d"].rolling(20).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0.0,
        raw=True,
    )
    return features.iloc[60:].copy()  # trim warmup


@pytest.fixture
def multi_security_features(multi_phase_features):
    """Generate features for multiple securities with correlated regimes."""
    np.random.seed(123)
    base = multi_phase_features
    securities = {}
    n = len(base)

    for i in range(10):
        feat = base.copy()
        # Add small idiosyncratic noise but keep same regime structure
        noise = np.random.randn(n) * 0.001
        feat["return_1d"] = feat["return_1d"] + noise
        feat["return_vol_20d"] = feat["return_1d"].rolling(20).std()
        securities[f"SEC_{i}"] = feat

    return securities


# ---------------------------------------------------------------------------
# Test 1: Ensemble voting end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestEnsembleVotingEndToEnd:
    """Confidence-weighted ensemble produces valid output on multi-phase data."""

    def test_ensemble_produces_valid_output(self, multi_phase_features):
        from quant_engine.regime.detector import RegimeDetector, RegimeOutput

        detector = RegimeDetector(method="hmm")
        out = detector.detect_ensemble(multi_phase_features)

        assert isinstance(out, RegimeOutput)
        assert out.model_type == "ensemble"
        assert len(out.regime) == len(multi_phase_features)
        assert len(out.confidence) == len(multi_phase_features)

    def test_ensemble_regimes_canonical(self, multi_phase_features):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="hmm")
        out = detector.detect_ensemble(multi_phase_features)

        unique = set(int(x) for x in out.regime.unique())
        assert unique.issubset({0, 1, 2, 3}), f"Non-canonical regimes: {unique}"

    def test_ensemble_probabilities_valid(self, multi_phase_features):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="hmm")
        out = detector.detect_ensemble(multi_phase_features)

        assert not out.probabilities.isna().any().any(), "Ensemble probs have NaN"
        row_sums = out.probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.02)

    def test_ensemble_uncertainty_populated(self, multi_phase_features):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="hmm")
        out = detector.detect_ensemble(multi_phase_features)

        assert out.uncertainty is not None
        assert len(out.uncertainty) == len(multi_phase_features)
        assert (out.uncertainty >= 0.0).all()
        assert (out.uncertainty <= 1.0).all()

    def test_ensemble_confidence_range(self, multi_phase_features):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="hmm")
        out = detector.detect_ensemble(multi_phase_features)

        assert (out.confidence >= 0.0).all()
        assert (out.confidence <= 1.0).all()


# ---------------------------------------------------------------------------
# Test 2: Uncertainty gate integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestUncertaintyGateIntegration:
    """Uncertainty gate correctly modifies positions based on regime entropy."""

    def test_uncertainty_gate_reduces_sizes(self, multi_phase_features):
        from quant_engine.regime.detector import RegimeDetector
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        detector = RegimeDetector(method="hmm")
        out = detector.detect_ensemble(multi_phase_features)

        gate = UncertaintyGate()
        assert out.uncertainty is not None

        # Find a period with moderate uncertainty
        high_unc_mask = out.uncertainty > 0.3
        if high_unc_mask.any():
            sample_unc = float(out.uncertainty[high_unc_mask].iloc[0])
            multiplier = gate.compute_size_multiplier(sample_unc)
            assert 0.0 < multiplier <= 1.0

    def test_gate_series_produces_valid_output(self, multi_phase_features):
        from quant_engine.regime.detector import RegimeDetector
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        detector = RegimeDetector(method="hmm")
        out = detector.detect_ensemble(multi_phase_features)

        gate = UncertaintyGate()
        assert out.uncertainty is not None
        result = gate.gate_series(out.uncertainty)

        assert "multiplier" in result.columns
        assert "is_uncertain" in result.columns
        assert "assume_stress" in result.columns
        assert len(result) == len(multi_phase_features)
        assert (result["multiplier"] >= 0.0).all()
        assert (result["multiplier"] <= 1.0).all()

    def test_stress_assumption_on_high_uncertainty(self, multi_phase_features):
        """When uncertainty is very high, gate recommends stress regime."""
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        gate = UncertaintyGate()
        assert gate.should_assume_stress(0.95) is True
        assert gate.should_assume_stress(0.10) is False

    def test_weight_reduction_applied(self):
        """Verify that weight vectors are actually scaled down."""
        from quant_engine.regime.uncertainty_gate import UncertaintyGate

        gate = UncertaintyGate()
        weights = np.array([0.10, 0.20, 0.30, 0.15, 0.25])

        # At maximum uncertainty, weights should be reduced
        adjusted = gate.apply_uncertainty_gate(weights, uncertainty=1.0)
        assert np.all(adjusted <= weights)
        assert np.all(adjusted >= 0.0)

        # At zero uncertainty, weights should be unchanged
        unchanged = gate.apply_uncertainty_gate(weights, uncertainty=0.0)
        np.testing.assert_allclose(unchanged, weights)


# ---------------------------------------------------------------------------
# Test 3: Regime transitions
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRegimeTransitions:
    """Ensemble detects phase transitions in multi-phase data."""

    def test_detects_multiple_regimes(self, multi_phase_features):
        """Should detect at least 2 distinct regimes across the phases."""
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="hmm")
        out = detector.detect_ensemble(multi_phase_features)

        n_unique = out.regime.nunique()
        assert n_unique >= 2, (
            f"Only {n_unique} regime(s) detected across 4 market phases"
        )

    def test_regime_features_complete(self, multi_phase_features):
        """regime_features() output includes all expected columns."""
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="hmm")
        rf = detector.regime_features(multi_phase_features)

        required = {
            "regime", "regime_confidence", "regime_model_type",
            "regime_duration",
            "regime_prob_0", "regime_prob_1", "regime_prob_2", "regime_prob_3",
            "regime_0", "regime_1", "regime_2", "regime_3",
            "regime_transition_prob",
        }
        assert required.issubset(set(rf.columns)), (
            f"Missing: {required - set(rf.columns)}"
        )

    def test_regime_duration_computed(self, multi_phase_features):
        """regime_duration tracks time since last regime change."""
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="hmm")
        rf = detector.regime_features(multi_phase_features)

        assert rf["regime_duration"].min() >= 1
        assert not rf["regime_duration"].isna().any()


# ---------------------------------------------------------------------------
# Test 4: Cross-sectional consensus integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestConsensusIntegration:
    """Regime consensus computed across multiple securities."""

    def test_consensus_from_multi_security(self, multi_security_features):
        from quant_engine.regime.detector import RegimeDetector
        from quant_engine.regime.consensus import RegimeConsensus

        detector = RegimeDetector(method="hmm")
        # Detect regimes for each security at the last bar
        regimes = []
        for sec_id, feat in multi_security_features.items():
            out = detector.detect_full(feat)
            regimes.append(int(out.regime.iloc[-1]))

        rc = RegimeConsensus()
        result = rc.compute_consensus(regimes)

        assert "consensus" in result
        assert "regime_pcts" in result
        assert "consensus_regime" in result
        assert 0.0 <= result["consensus"] <= 1.0
        assert result["n_securities"] == len(multi_security_features)

    def test_consensus_series_from_batch(self, multi_security_features):
        """Compute consensus over time for a batch of securities."""
        from quant_engine.regime.detector import RegimeDetector
        from quant_engine.regime.consensus import RegimeConsensus

        detector = RegimeDetector(method="rule")

        # Build regime matrix (rows=dates, cols=securities) using rule-based
        # (fast, deterministic) for a 30-day window at the end
        sample_feat = next(iter(multi_security_features.values()))
        dates = sample_feat.index[-30:]
        regime_matrix = pd.DataFrame(index=dates)

        for sec_id, feat in multi_security_features.items():
            out = detector.detect_full(feat)
            regime_matrix[sec_id] = out.regime.reindex(dates)

        regime_matrix = regime_matrix.dropna()
        assert len(regime_matrix) > 0

        rc = RegimeConsensus()
        series = rc.compute_consensus_series(regime_matrix.astype(int))

        assert "consensus" in series.columns
        assert "consensus_regime" in series.columns
        assert len(series) == len(regime_matrix)

    def test_correlated_securities_have_high_consensus(self, multi_security_features):
        """Securities with same structure should show high consensus."""
        from quant_engine.regime.detector import RegimeDetector
        from quant_engine.regime.consensus import RegimeConsensus

        detector = RegimeDetector(method="rule")
        regimes = []
        for sec_id, feat in multi_security_features.items():
            out = detector.detect_full(feat)
            regimes.append(int(out.regime.iloc[-1]))

        rc = RegimeConsensus()
        result = rc.compute_consensus(regimes)

        # All securities have same structure, so consensus should be high
        assert result["consensus"] >= 0.60, (
            f"Consensus {result['consensus']:.2f} too low for correlated securities"
        )


# ---------------------------------------------------------------------------
# Test 5: Online update integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestOnlineUpdateIntegration:
    """Online updater works with the full pipeline."""

    def test_online_update_after_hmm_fit(self, multi_phase_features):
        from quant_engine.regime.hmm import GaussianHMM, build_hmm_observation_matrix
        from quant_engine.regime.online_update import OnlineRegimeUpdater

        obs_df = build_hmm_observation_matrix(multi_phase_features)
        X = obs_df.values.astype(float)

        model = GaussianHMM(n_states=4, max_iter=30, covariance_type="diag")
        model.fit(X)

        updater = OnlineRegimeUpdater(model)

        # Online update for multiple securities
        results = {}
        for i in range(5):
            regime, prob = updater.update_regime_for_security(f"SEC_{i}", X[i])
            results[f"SEC_{i}"] = (regime, prob)

        assert len(results) == 5
        for sec_id, (regime, prob) in results.items():
            assert 0 <= regime <= 3
            np.testing.assert_allclose(prob.sum(), 1.0, atol=1e-6)

    def test_online_batch_update(self, multi_phase_features):
        from quant_engine.regime.hmm import GaussianHMM, build_hmm_observation_matrix
        from quant_engine.regime.online_update import OnlineRegimeUpdater

        obs_df = build_hmm_observation_matrix(multi_phase_features)
        X = obs_df.values.astype(float)

        model = GaussianHMM(n_states=4, max_iter=30, covariance_type="diag")
        model.fit(X)

        updater = OnlineRegimeUpdater(model)
        securities = {f"SEC_{i}": X[i] for i in range(20)}
        results = updater.update_batch(securities)

        assert len(results) == 20
        assert updater.cached_securities == 20


# ---------------------------------------------------------------------------
# Test 6: Confidence calibration integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestConfidenceCalibrationIntegration:
    """Confidence calibrator works with actual detector outputs."""

    def test_calibrate_and_use_in_ensemble(self, multi_phase_features):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="hmm")

        # Use rule-based regimes as "ground truth" for calibration
        rule_out = detector._rule_detect(multi_phase_features)
        actual_regimes = rule_out.regime.values

        # Calibrate
        weights = detector.calibrate_confidence_weights(
            multi_phase_features, actual_regimes,
        )

        assert isinstance(weights, dict)
        assert len(weights) >= 2  # at least hmm and rule
        total = sum(weights.values())
        np.testing.assert_allclose(total, 1.0, atol=1e-6)

        # Now run ensemble with calibrated weights
        out = detector.detect_ensemble(multi_phase_features)
        assert out.model_type == "ensemble"
        assert len(out.regime) == len(multi_phase_features)

    def test_calibrator_ecm_populated(self, multi_phase_features):
        from quant_engine.regime.detector import RegimeDetector

        detector = RegimeDetector(method="hmm")
        rule_out = detector._rule_detect(multi_phase_features)
        actual_regimes = rule_out.regime.values

        detector.calibrate_confidence_weights(
            multi_phase_features, actual_regimes,
        )

        assert detector._confidence_calibrator is not None
        assert detector._confidence_calibrator.fitted

        # Component weights exist for expected components
        cw = detector._confidence_calibrator.component_weights
        assert "hmm" in cw
        assert "rule" in cw


# ---------------------------------------------------------------------------
# Test 7: MIN_REGIME_SAMPLES config
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMinRegimeSamplesConfig:
    """MIN_REGIME_SAMPLES and MIN_REGIME_DAYS config changes are correct."""

    def test_min_regime_samples_value(self):
        from quant_engine.config import MIN_REGIME_SAMPLES

        assert MIN_REGIME_SAMPLES == 50, (
            f"MIN_REGIME_SAMPLES={MIN_REGIME_SAMPLES}, expected 50 (SPEC_10 T7)"
        )

    def test_min_regime_days_value(self):
        from quant_engine.config import MIN_REGIME_DAYS

        assert MIN_REGIME_DAYS == 10, (
            f"MIN_REGIME_DAYS={MIN_REGIME_DAYS}, expected 10"
        )

    def test_short_regime_would_train(self):
        """A 50-sample, 15-day regime should pass the training threshold (SPEC_10 T7)."""
        from quant_engine.config import MIN_REGIME_SAMPLES, MIN_REGIME_DAYS

        n_samples = 50
        n_days = 15
        assert n_samples >= MIN_REGIME_SAMPLES
        assert n_days >= MIN_REGIME_DAYS

    def test_moderate_regime_would_train(self):
        """A 100-sample, 15-day regime should also pass (above threshold)."""
        from quant_engine.config import MIN_REGIME_SAMPLES, MIN_REGIME_DAYS

        n_samples = 100
        n_days = 15
        assert n_samples >= MIN_REGIME_SAMPLES
        assert n_days >= MIN_REGIME_DAYS

    def test_very_short_regime_blocked(self):
        """A 30-sample, 5-day regime should be blocked."""
        from quant_engine.config import MIN_REGIME_SAMPLES, MIN_REGIME_DAYS

        n_samples = 30
        n_days = 5
        passes = n_samples >= MIN_REGIME_SAMPLES and n_days >= MIN_REGIME_DAYS
        assert not passes

    def test_boundary_regime_blocked(self):
        """A 49-sample regime should be blocked (just below 50 threshold)."""
        from quant_engine.config import MIN_REGIME_SAMPLES, MIN_REGIME_DAYS

        n_samples = 49
        n_days = 15
        passes = n_samples >= MIN_REGIME_SAMPLES and n_days >= MIN_REGIME_DAYS
        assert not passes


# ---------------------------------------------------------------------------
# Test 8: Full pipeline end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFullPipelineEndToEnd:
    """The entire upgraded regime pipeline works together."""

    def test_detect_with_shock_context(self, multi_phase_features):
        """ShockVector generation works with ensemble detection."""
        from quant_engine.regime.detector import RegimeDetector
        from quant_engine.regime.shock_vector import ShockVector

        detector = RegimeDetector(method="hmm", enable_bocpd=True)
        sv = detector.detect_with_shock_context(
            multi_phase_features, ticker="TEST",
        )

        assert isinstance(sv, ShockVector)
        assert sv.ticker == "TEST"
        assert 0 <= sv.hmm_regime <= 3
        assert 0.0 <= sv.hmm_confidence <= 1.0
        assert 0.0 <= sv.hmm_uncertainty <= 1.0

    def test_batch_regime_detection(self, multi_security_features):
        """detect_regimes_batch works with multiple securities."""
        from quant_engine.regime.detector import detect_regimes_batch

        regime_data, regimes, regime_probs = detect_regimes_batch(
            multi_security_features,
        )

        assert len(regime_data) > 0
        assert len(regimes) > 0
        assert set(regimes.unique()).issubset({0, 1, 2, 3})

    def test_pipeline_ensemble_uncertainty_consensus(self, multi_security_features):
        """Full flow: ensemble → uncertainty → consensus."""
        from quant_engine.regime.detector import RegimeDetector
        from quant_engine.regime.uncertainty_gate import UncertaintyGate
        from quant_engine.regime.consensus import RegimeConsensus

        detector = RegimeDetector(method="hmm")
        gate = UncertaintyGate()
        rc = RegimeConsensus()

        last_regimes = []
        last_uncertainties = []

        for sec_id, feat in multi_security_features.items():
            out = detector.detect_ensemble(feat)

            # Verify ensemble output
            assert out.model_type == "ensemble"
            assert out.uncertainty is not None

            last_regime = int(out.regime.iloc[-1])
            last_unc = float(out.uncertainty.iloc[-1])

            last_regimes.append(last_regime)
            last_uncertainties.append(last_unc)

            # Apply uncertainty gate
            multiplier = gate.compute_size_multiplier(last_unc)
            assert 0.0 < multiplier <= 1.0

        # Compute consensus
        consensus_result = rc.compute_consensus(last_regimes)
        assert 0.0 <= consensus_result["consensus"] <= 1.0
        assert consensus_result["n_securities"] == len(multi_security_features)

        # Check early warning
        warning, reason = rc.early_warning(consensus_result["consensus"])
        assert isinstance(warning, bool)

    def test_regime_exports_complete(self):
        """All SPEC_10 modules are exported from the regime package."""
        from quant_engine.regime import (
            ConfidenceCalibrator,
            RegimeConsensus,
            OnlineRegimeUpdater,
            UncertaintyGate,
            RegimeDetector,
            RegimeOutput,
            GaussianHMM,
            StatisticalJumpModel,
            PyPIJumpModel,
            BOCPDDetector,
            ShockVector,
        )

        # Verify they are actual classes, not None
        assert ConfidenceCalibrator is not None
        assert RegimeConsensus is not None
        assert OnlineRegimeUpdater is not None
        assert UncertaintyGate is not None

    def test_config_constants_exist(self):
        """All SPEC_10 config constants are defined."""
        from quant_engine.config import (
            MIN_REGIME_SAMPLES,
            MIN_REGIME_DAYS,
            REGIME_ENSEMBLE_DEFAULT_WEIGHTS,
            REGIME_ENSEMBLE_DISAGREEMENT_THRESHOLD,
            REGIME_ENSEMBLE_UNCERTAIN_FALLBACK,
            REGIME_UNCERTAINTY_ENTROPY_THRESHOLD,
            REGIME_UNCERTAINTY_STRESS_THRESHOLD,
            REGIME_UNCERTAINTY_SIZING_MAP,
            REGIME_UNCERTAINTY_MIN_MULTIPLIER,
            REGIME_CONSENSUS_THRESHOLD,
            REGIME_CONSENSUS_EARLY_WARNING,
            REGIME_CONSENSUS_DIVERGENCE_WINDOW,
            REGIME_CONSENSUS_DIVERGENCE_SLOPE,
            REGIME_ONLINE_UPDATE_ENABLED,
            REGIME_ONLINE_REFIT_DAYS,
            REGIME_EXPANDED_FEATURES_ENABLED,
        )

        assert MIN_REGIME_SAMPLES == 50
        assert MIN_REGIME_DAYS == 10
        assert isinstance(REGIME_ENSEMBLE_DEFAULT_WEIGHTS, dict)
        assert isinstance(REGIME_UNCERTAINTY_SIZING_MAP, dict)
        assert REGIME_CONSENSUS_THRESHOLD == 0.80
        assert REGIME_ONLINE_REFIT_DAYS == 30
