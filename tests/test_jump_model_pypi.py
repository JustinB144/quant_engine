"""Unit tests for the PyPI JumpModel wrapper (Spec 001).

Verifies:
  - Basic fit produces correct output shapes and types
  - Regime labels are in valid range {0, ..., K-1}
  - Regime probabilities sum to ~1.0 per row
  - CV lambda selection stays within configured range
  - Fallback on short data (< MIN_OBSERVATIONS)
  - NaN handling in features
  - predict_online returns correct shapes
  - Higher lambda produces more persistent regimes (fewer transitions)
  - Continuous vs discrete mode behavior
  - Canonical regime mapping via map_raw_states_to_regimes
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def random_obs():
    """Generate random observation matrix for testing (500 x 4)."""
    np.random.seed(42)
    return np.random.randn(500, 4)


@pytest.fixture
def short_obs():
    """Generate short observation matrix (< MIN_OBSERVATIONS)."""
    np.random.seed(42)
    return np.random.randn(100, 4)


@pytest.fixture
def synthetic_features():
    """Generate a synthetic feature DataFrame with known structure."""
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
    features["SMASlope_50"] = features["Close"].rolling(20).mean().pct_change(5)
    features["Hurst_100"] = 0.5
    features["ADX_14"] = 25.0
    return features.dropna()


# ── T6.1: Basic fit ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestFitBasic:
    """Fit on random data and verify output shapes and types."""

    def test_fit_returns_jump_model_result(self, random_obs):
        from quant_engine.regime.jump_model_pypi import PyPIJumpModel
        from quant_engine.regime.jump_model_legacy import JumpModelResult

        model = PyPIJumpModel(n_regimes=4, jump_penalty=0.5)
        result = model.fit(random_obs)

        assert isinstance(result, JumpModelResult)

    def test_fit_output_shapes(self, random_obs):
        from quant_engine.regime.jump_model_pypi import PyPIJumpModel

        T, D = random_obs.shape
        K = 4
        model = PyPIJumpModel(n_regimes=K, jump_penalty=0.5)
        result = model.fit(random_obs)

        assert result.regime_sequence.shape == (T,)
        assert result.regime_probs.shape == (T, K)
        assert result.centroids.shape == (K, D)
        assert isinstance(result.jump_penalty, float)
        assert isinstance(result.converged, bool)

    def test_fit_output_dtypes(self, random_obs):
        from quant_engine.regime.jump_model_pypi import PyPIJumpModel

        model = PyPIJumpModel(n_regimes=4, jump_penalty=0.5)
        result = model.fit(random_obs)

        assert result.regime_sequence.dtype in (np.int32, np.int64, int)
        assert result.regime_probs.dtype == np.float64


# ── T6.2: Valid regime labels ─────────────────────────────────────────────


@pytest.mark.unit
class TestValidRegimes:
    """All regime labels must be in {0, ..., K-1}."""

    def test_fit_returns_valid_regimes(self, random_obs):
        from quant_engine.regime.jump_model_pypi import PyPIJumpModel

        K = 4
        model = PyPIJumpModel(n_regimes=K, jump_penalty=0.5)
        result = model.fit(random_obs)

        unique_labels = set(int(x) for x in result.regime_sequence)
        assert unique_labels.issubset(set(range(K))), (
            f"Labels {unique_labels} not subset of {set(range(K))}"
        )


# ── T6.3: Probs sum to one ───────────────────────────────────────────────


@pytest.mark.unit
class TestProbsSumToOne:
    """Each row of regime_probs must sum to ~1.0."""

    def test_fit_probs_sum_to_one(self, random_obs):
        from quant_engine.regime.jump_model_pypi import PyPIJumpModel

        model = PyPIJumpModel(n_regimes=4, jump_penalty=0.5)
        result = model.fit(random_obs)

        row_sums = result.regime_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_probs_non_negative(self, random_obs):
        from quant_engine.regime.jump_model_pypi import PyPIJumpModel

        model = PyPIJumpModel(n_regimes=4, jump_penalty=0.5)
        result = model.fit(random_obs)

        assert np.all(result.regime_probs >= 0.0), "Probabilities contain negative values"

    def test_probs_no_nan(self, random_obs):
        from quant_engine.regime.jump_model_pypi import PyPIJumpModel

        model = PyPIJumpModel(n_regimes=4, jump_penalty=0.5)
        result = model.fit(random_obs)

        assert not np.any(np.isnan(result.regime_probs)), "Probabilities contain NaN"


# ── T6.4: CV lambda selection ─────────────────────────────────────────────


@pytest.mark.unit
class TestCVLambdaSelection:
    """CV must select a lambda within the configured range."""

    def test_cv_lambda_within_range(self, random_obs):
        from quant_engine.regime.jump_model_pypi import PyPIJumpModel

        lam_range = (0.01, 0.10)
        model = PyPIJumpModel(
            n_regimes=4,
            jump_penalty=None,  # trigger CV
            lambda_range=lam_range,
            lambda_steps=10,
            cv_folds=3,
        )
        result = model.fit(random_obs)

        assert lam_range[0] <= result.jump_penalty <= lam_range[1], (
            f"Selected lambda {result.jump_penalty} outside range {lam_range}"
        )

    def test_cv_lambda_is_float(self, random_obs):
        from quant_engine.regime.jump_model_pypi import PyPIJumpModel

        model = PyPIJumpModel(n_regimes=4, jump_penalty=None, cv_folds=3, lambda_steps=5)
        result = model.fit(random_obs)

        assert isinstance(result.jump_penalty, float)


# ── T6.5: Fallback on short data ─────────────────────────────────────────


@pytest.mark.unit
class TestFallbackShortData:
    """<200 rows should raise ValueError cleanly."""

    def test_short_data_raises(self, short_obs):
        from quant_engine.regime.jump_model_pypi import PyPIJumpModel

        model = PyPIJumpModel(n_regimes=4, jump_penalty=0.5)

        with pytest.raises(ValueError, match="at least"):
            model.fit(short_obs)


# ── T6.6: NaN handling ───────────────────────────────────────────────────


@pytest.mark.unit
class TestNaNHandling:
    """Features with NaN should be handled without crash."""

    def test_nan_features_handled(self, random_obs):
        from quant_engine.regime.jump_model_pypi import PyPIJumpModel

        X = random_obs.copy()
        # Inject some NaN values
        X[10:20, 0] = np.nan
        X[50:55, 2] = np.nan

        model = PyPIJumpModel(n_regimes=4, jump_penalty=0.5)
        result = model.fit(X)

        assert not np.any(np.isnan(result.regime_probs)), "NaN in probabilities after NaN input"
        assert result.regime_sequence.shape == (X.shape[0],)

    def test_all_nan_column_raises(self):
        from quant_engine.regime.jump_model_pypi import PyPIJumpModel

        np.random.seed(42)
        X = np.random.randn(300, 4)
        X[:, 1] = np.nan  # Entire column is NaN

        model = PyPIJumpModel(n_regimes=4, jump_penalty=0.5)

        with pytest.raises(ValueError):
            model.fit(X)


# ── T6.7: predict_online shape ────────────────────────────────────────────


@pytest.mark.unit
class TestPredictOnline:
    """Single-step prediction returns correct shape."""

    def test_predict_online_shape(self, random_obs):
        from quant_engine.regime.jump_model_pypi import PyPIJumpModel

        model = PyPIJumpModel(n_regimes=4, jump_penalty=0.5)
        model.fit(random_obs)

        # Online prediction on a batch
        online_labels = model.predict_online(random_obs[:50])
        assert online_labels.shape == (50,)

    def test_predict_online_valid_labels(self, random_obs):
        from quant_engine.regime.jump_model_pypi import PyPIJumpModel

        K = 4
        model = PyPIJumpModel(n_regimes=K, jump_penalty=0.5)
        model.fit(random_obs)

        online_labels = model.predict_online(random_obs[:50])
        unique = set(int(x) for x in online_labels)
        assert unique.issubset(set(range(K)))

    def test_predict_online_not_fitted_raises(self, random_obs):
        from quant_engine.regime.jump_model_pypi import PyPIJumpModel

        model = PyPIJumpModel(n_regimes=4)

        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_online(random_obs[:10])

    def test_predict_proba_online_shape(self, random_obs):
        from quant_engine.regime.jump_model_pypi import PyPIJumpModel

        K = 4
        model = PyPIJumpModel(n_regimes=K, jump_penalty=0.5)
        model.fit(random_obs)

        online_probs = model.predict_proba_online(random_obs[:50])
        assert online_probs.shape == (50, K)
        row_sums = online_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


# ── T6.8: Regime persistence ─────────────────────────────────────────────


@pytest.mark.unit
class TestRegimePersistence:
    """Higher lambda should produce fewer regime transitions."""

    def test_higher_lambda_fewer_transitions(self, random_obs):
        from quant_engine.regime.jump_model_pypi import PyPIJumpModel

        # Low penalty: many transitions
        model_low = PyPIJumpModel(n_regimes=4, jump_penalty=0.01)
        result_low = model_low.fit(random_obs)
        transitions_low = np.sum(np.diff(result_low.regime_sequence) != 0)

        # High penalty: few transitions
        model_high = PyPIJumpModel(n_regimes=4, jump_penalty=2.0)
        result_high = model_high.fit(random_obs)
        transitions_high = np.sum(np.diff(result_high.regime_sequence) != 0)

        assert transitions_high <= transitions_low, (
            f"High penalty ({transitions_high} transitions) should have <= "
            f"low penalty ({transitions_low} transitions)"
        )


# ── T6.9: Continuous vs discrete ──────────────────────────────────────────


@pytest.mark.unit
class TestContinuousVsDiscrete:
    """Continuous mode produces soft probabilities, discrete produces hard."""

    def test_discrete_mode_hard_probs(self, random_obs):
        from quant_engine.regime.jump_model_pypi import PyPIJumpModel

        # Discrete mode with forced hard probs check (bypass softmax fallback)
        # We test via the raw model
        from jumpmodels.jump import JumpModel

        model = JumpModel(n_components=4, jump_penalty=0.5, cont=False, n_init=3)
        model.fit(random_obs)
        probs = np.asarray(model.predict_proba(random_obs))
        unique_vals = np.unique(probs)
        assert set(unique_vals).issubset({0.0, 1.0}), (
            f"Discrete mode should produce hard probs, got values: {unique_vals}"
        )

    def test_wrapper_always_returns_soft_probs(self, random_obs):
        """The wrapper should always produce soft probs regardless of mode."""
        from quant_engine.regime.jump_model_pypi import PyPIJumpModel

        # Discrete mode — wrapper computes softmax from centroids
        model = PyPIJumpModel(n_regimes=4, jump_penalty=0.5, use_continuous=False)
        result = model.fit(random_obs)

        # Check that not all probs are 0 or 1 (i.e., softmax was applied)
        has_soft = np.any((result.regime_probs > 0) & (result.regime_probs < 1))
        assert has_soft, "Wrapper should produce soft probs even in discrete mode"


# ── T6.10: Canonical mapping ─────────────────────────────────────────────


@pytest.mark.unit
class TestCanonicalMapping:
    """Raw states correctly map to 4 canonical regimes."""

    def test_canonical_mapping(self, synthetic_features):
        from quant_engine.regime.hmm import (
            build_hmm_observation_matrix,
            map_raw_states_to_regimes,
        )
        from quant_engine.regime.jump_model_pypi import PyPIJumpModel

        obs_df = build_hmm_observation_matrix(synthetic_features)
        X = obs_df.values.astype(float)

        model = PyPIJumpModel(n_regimes=4, jump_penalty=0.5)
        result = model.fit(X)

        mapping = map_raw_states_to_regimes(result.regime_sequence, synthetic_features)

        # Mapping must cover all raw states found in sequence
        raw_states = set(int(x) for x in result.regime_sequence)
        for s in raw_states:
            assert s in mapping, f"Raw state {s} not in mapping"

        # All mapped values must be in {0, 1, 2, 3}
        for s, r in mapping.items():
            assert r in {0, 1, 2, 3}, f"Mapped regime {r} not in canonical set"
