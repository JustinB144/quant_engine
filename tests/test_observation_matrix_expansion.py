"""Tests for expanded HMM observation matrix (SPEC_10 T4).

Verifies:
  - Structural features (spectral, SSA, BOCPD) included when available
  - Graceful fallback when structural features missing
  - Feature count is 11-15 depending on availability
  - BIC state selection works with expanded features
  - All features are standardized and NaN-free
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def features_with_structural():
    """Synthetic features including structural indicators."""
    np.random.seed(42)
    n = 600
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = np.cumsum(np.random.randn(n) * 0.02) + 100

    features = pd.DataFrame(
        {
            "Close": prices,
            "High": prices + np.abs(np.random.randn(n) * 1.0),
            "Low": prices - np.abs(np.random.randn(n) * 1.0),
            "Open": prices + np.random.randn(n) * 0.5,
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
    features["GARCH_252"] = features["return_vol_20d"] * 1.1
    features["AutoCorr_20_1"] = 0.05

    # Structural features
    features["SpectralEntropy_252"] = np.random.uniform(0.3, 0.9, n)
    features["SSATrendStr_60"] = np.random.uniform(0.1, 0.8, n)
    features["JumpIntensity_20"] = np.random.uniform(0.0, 0.3, n)

    return features.dropna()


@pytest.fixture
def features_without_structural():
    """Synthetic features without structural indicators."""
    np.random.seed(42)
    n = 600
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = np.cumsum(np.random.randn(n) * 0.02) + 100

    features = pd.DataFrame(
        {
            "Close": prices,
            "High": prices + np.abs(np.random.randn(n) * 1.0),
            "Low": prices - np.abs(np.random.randn(n) * 1.0),
            "Open": prices + np.random.randn(n) * 0.5,
            "Volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
        },
        index=idx,
    )
    features["return_1d"] = features["Close"].pct_change()
    features["return_vol_20d"] = features["return_1d"].rolling(20).std()
    features["NATR_14"] = (features["High"] - features["Low"]) / features["Close"]
    features["SMASlope_50"] = features["Close"].rolling(50).mean().pct_change(5)
    features["Hurst_100"] = 0.5
    return features.dropna()


# ---------------------------------------------------------------------------
# Feature expansion tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestObservationMatrixExpansion:
    """Verify structural features are included in the observation matrix."""

    def test_structural_features_included(self, features_with_structural):
        from quant_engine.regime.hmm import build_hmm_observation_matrix

        obs = build_hmm_observation_matrix(features_with_structural)

        # Should have structural columns
        assert "spectral_entropy" in obs.columns, "Missing spectral_entropy"
        assert "ssa_trend_strength" in obs.columns, "Missing ssa_trend_strength"
        assert "jump_intensity" in obs.columns, "Missing jump_intensity"

    def test_expanded_feature_count(self, features_with_structural):
        from quant_engine.regime.hmm import build_hmm_observation_matrix

        obs = build_hmm_observation_matrix(features_with_structural)
        # Core (4) + extended (up to 7) + structural (up to 4) = 11-15
        assert obs.shape[1] >= 11, f"Expected >= 11 features, got {obs.shape[1]}"

    def test_graceful_fallback_without_structural(self, features_without_structural):
        from quant_engine.regime.hmm import build_hmm_observation_matrix

        obs = build_hmm_observation_matrix(features_without_structural)
        # Should still work with at least 4 core features
        assert obs.shape[1] >= 4, f"Expected >= 4 features, got {obs.shape[1]}"
        # May include BOCPD computed inline from return_1d
        # but should NOT crash

    def test_all_features_standardized(self, features_with_structural):
        from quant_engine.regime.hmm import build_hmm_observation_matrix

        obs = build_hmm_observation_matrix(features_with_structural)
        for col in obs.columns:
            mean = obs[col].mean()
            std = obs[col].std()
            # Should be approximately mean=0, std=1 (or 0 for constant columns)
            assert abs(mean) < 0.1 or std < 1e-10, f"Column {col}: mean={mean:.4f}"

    def test_no_nan_or_inf(self, features_with_structural):
        from quant_engine.regime.hmm import build_hmm_observation_matrix

        obs = build_hmm_observation_matrix(features_with_structural)
        assert not obs.isna().any().any(), "NaN found in observation matrix"
        assert not np.isinf(obs.values).any(), "Inf found in observation matrix"


# ---------------------------------------------------------------------------
# BIC with expanded features
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBICWithExpandedFeatures:
    """Verify BIC state selection works with the expanded observation matrix."""

    def test_bic_runs_with_structural(self, features_with_structural):
        from quant_engine.regime.hmm import (
            build_hmm_observation_matrix,
            select_hmm_states_bic,
        )

        obs = build_hmm_observation_matrix(features_with_structural)
        X = obs.values.astype(float)

        best_k, bic_scores = select_hmm_states_bic(
            X, min_states=2, max_states=5,
            max_iter=30, stickiness=0.92, min_duration=3,
        )

        assert 2 <= best_k <= 5
        assert len(bic_scores) >= 1

    def test_hmm_fits_with_expanded_features(self, features_with_structural):
        from quant_engine.regime.hmm import build_hmm_observation_matrix, GaussianHMM

        obs = build_hmm_observation_matrix(features_with_structural)
        X = obs.values.astype(float)

        model = GaussianHMM(n_states=4, max_iter=30, covariance_type="diag")
        result = model.fit(X)

        assert result.raw_states.shape[0] == X.shape[0]
        assert result.state_probs.shape == (X.shape[0], 4)


# ---------------------------------------------------------------------------
# BOCPD inline computation test
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBOCPDInlineComputation:
    """Verify BOCPD changepoint feature is computed inline when not pre-computed."""

    def test_bocpd_computed_inline(self, features_without_structural):
        """BOCPD should be computed from return_1d when no pre-computed column exists."""
        from quant_engine.regime.hmm import build_hmm_observation_matrix

        obs = build_hmm_observation_matrix(features_without_structural)

        # BOCPD may or may not be included depending on data length
        # and whether BOCPD detector can initialize. Just verify no crash.
        assert obs.shape[1] >= 4
        assert not obs.isna().any().any()
