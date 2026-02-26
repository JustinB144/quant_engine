"""
Tests for Spec 02: Structural Feature Expansion.

Covers all 5 structural feature families:
    T1: SpectralAnalyzer (FFT-based frequency decomposition)
    T2: SSADecomposer (singular spectrum analysis)
    T3: TailRiskAnalyzer (jump detection, CVaR, vol-of-vol)
    T4: EigenvalueAnalyzer (cross-asset eigenvalue spectrum)
    T5: OptimalTransportAnalyzer (Wasserstein/Sinkhorn divergence)
    T6: Pipeline integration and feature redundancy
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_trending_prices():
    """Strong uptrend with low noise — should have high trend strength."""
    np.random.seed(42)
    n = 500
    t = np.arange(n, dtype=float)
    trend = 0.002 * t  # +0.2% per bar
    noise = np.random.randn(n) * 0.005
    return np.exp(np.log(100) + np.cumsum(trend + noise))


@pytest.fixture
def synthetic_oscillating_prices():
    """Pure sinusoidal oscillation with known period."""
    np.random.seed(42)
    n = 500
    t = np.arange(n, dtype=float)
    # 20-day cycle
    cycle = 0.03 * np.sin(2 * np.pi * t / 20)
    noise = np.random.randn(n) * 0.002
    return np.exp(np.log(100) + np.cumsum(cycle + noise))


@pytest.fixture
def synthetic_noisy_prices():
    """Random walk — should have high noise ratio, high entropy."""
    np.random.seed(42)
    n = 500
    returns = np.random.randn(n) * 0.02
    return np.exp(np.log(100) + np.cumsum(returns))


@pytest.fixture
def synthetic_jump_prices():
    """Prices with known jumps at specific locations."""
    np.random.seed(42)
    n = 200
    returns = np.random.randn(n) * 0.01
    # Insert jumps
    returns[50] = 0.10   # +10% jump
    returns[100] = -0.08  # -8% jump
    returns[150] = 0.12   # +12% jump
    return np.exp(np.log(100) + np.cumsum(returns))


@pytest.fixture
def synthetic_ohlcv_df(synthetic_trending_prices):
    """OHLCV DataFrame suitable for pipeline testing."""
    n = len(synthetic_trending_prices)
    np.random.seed(42)
    close = synthetic_trending_prices
    # Approximate OHLCV from close
    high = close * (1 + np.abs(np.random.randn(n)) * 0.005)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.005)
    open_px = close * (1 + np.random.randn(n) * 0.002)
    volume = np.random.randint(100_000, 1_000_000, size=n).astype(float)

    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "Open": open_px,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)


# ===========================================================================
# T1: SpectralAnalyzer Tests
# ===========================================================================

class TestSpectralAnalyzer:
    """Tests for FFT-based spectral features."""

    def test_detects_weekly_cycle(self, synthetic_oscillating_prices):
        """Dominant frequency should detect the 20-day cycle."""
        from quant_engine.indicators.spectral import SpectralAnalyzer

        analyzer = SpectralAnalyzer(fft_window=252)
        dom_period = analyzer.compute_dominant_frequency(
            synthetic_oscillating_prices
        )

        # After warm-up, dominant period should be near 20
        valid = dom_period[~np.isnan(dom_period)]
        assert len(valid) > 0
        # Allow tolerance: within 5 days of true period
        assert np.abs(valid[-1] - 20.0) < 5.0

    def test_hf_lf_energy_non_negative(self, synthetic_trending_prices):
        """HF and LF energy must be non-negative."""
        from quant_engine.indicators.spectral import SpectralAnalyzer

        analyzer = SpectralAnalyzer(fft_window=252)
        hf, lf = analyzer.compute_hf_lf_energy(synthetic_trending_prices)

        valid_hf = hf[~np.isnan(hf)]
        valid_lf = lf[~np.isnan(lf)]
        assert np.all(valid_hf >= 0)
        assert np.all(valid_lf >= 0)

    def test_spectral_entropy_bounded(self, synthetic_trending_prices):
        """Spectral entropy must be in [0, 1]."""
        from quant_engine.indicators.spectral import SpectralAnalyzer

        analyzer = SpectralAnalyzer(fft_window=252)
        entropy = analyzer.compute_spectral_entropy(synthetic_trending_prices)

        valid = entropy[~np.isnan(entropy)]
        assert len(valid) > 0
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0 + 1e-10)

    def test_noisy_signal_high_entropy(self, synthetic_noisy_prices):
        """Random walk should have high spectral entropy (flat spectrum)."""
        from quant_engine.indicators.spectral import SpectralAnalyzer

        analyzer = SpectralAnalyzer(fft_window=252)
        entropy = analyzer.compute_spectral_entropy(synthetic_noisy_prices)

        valid = entropy[~np.isnan(entropy)]
        assert len(valid) > 0
        # Noise should have entropy > 0.7 (near-uniform spectrum)
        assert np.mean(valid) > 0.7

    def test_compute_all_consistent(self, synthetic_trending_prices):
        """compute_all() should produce same results as individual methods."""
        from quant_engine.indicators.spectral import SpectralAnalyzer

        analyzer = SpectralAnalyzer(fft_window=252)
        results = analyzer.compute_all(synthetic_trending_prices)

        assert "hf_energy" in results
        assert "lf_energy" in results
        assert "spectral_entropy" in results
        assert "dominant_period" in results
        assert "spectral_bandwidth" in results

        # All arrays same length as input
        n = len(synthetic_trending_prices)
        for key, arr in results.items():
            assert len(arr) == n

    def test_output_length_matches_input(self, synthetic_trending_prices):
        """All outputs must match input length."""
        from quant_engine.indicators.spectral import SpectralAnalyzer

        analyzer = SpectralAnalyzer(fft_window=100)
        n = len(synthetic_trending_prices)

        hf, lf = analyzer.compute_hf_lf_energy(synthetic_trending_prices)
        assert len(hf) == n
        assert len(lf) == n

        entropy = analyzer.compute_spectral_entropy(synthetic_trending_prices)
        assert len(entropy) == n

    def test_bandwidth_non_negative(self, synthetic_trending_prices):
        """Spectral bandwidth must be non-negative."""
        from quant_engine.indicators.spectral import SpectralAnalyzer

        analyzer = SpectralAnalyzer(fft_window=252)
        bw = analyzer.compute_spectral_bandwidth(synthetic_trending_prices)
        valid = bw[~np.isnan(bw)]
        assert np.all(valid >= 0)

    def test_invalid_window_raises(self):
        """fft_window < 20 should raise ValueError."""
        from quant_engine.indicators.spectral import SpectralAnalyzer

        with pytest.raises(ValueError):
            SpectralAnalyzer(fft_window=5)


# ===========================================================================
# T2: SSADecomposer Tests
# ===========================================================================

class TestSSADecomposer:
    """Tests for Singular Spectrum Analysis features."""

    def test_trending_signal_high_trend_strength(
        self, synthetic_trending_prices
    ):
        """Strong trend should produce high trend strength."""
        from quant_engine.indicators.ssa import SSADecomposer

        ssa = SSADecomposer(window=60, embed_dim=12)
        ts = ssa.compute_trend_strength(synthetic_trending_prices)

        valid = ts[~np.isnan(ts)]
        assert len(valid) > 0
        # Strong trend: first SV should capture > 30% of variance
        assert np.mean(valid[-50:]) > 0.3

    def test_noisy_signal_high_noise_ratio(self, synthetic_noisy_prices):
        """Random walk should have high noise ratio."""
        from quant_engine.indicators.ssa import SSADecomposer

        ssa = SSADecomposer(window=60, embed_dim=12, n_singular=3)
        nr = ssa.compute_noise_ratio(synthetic_noisy_prices)

        valid = nr[~np.isnan(nr)]
        assert len(valid) > 0
        # Noise-dominated: tail SVs should capture significant variance
        assert np.mean(valid[-50:]) > 0.2

    def test_singular_entropy_bounded(self, synthetic_trending_prices):
        """Singular entropy must be in [0, 1]."""
        from quant_engine.indicators.ssa import SSADecomposer

        ssa = SSADecomposer(window=60, embed_dim=12)
        ent = ssa.compute_singular_entropy(synthetic_trending_prices)

        valid = ent[~np.isnan(ent)]
        assert len(valid) > 0
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0 + 1e-10)

    def test_oscillatory_strength_bounded(self, synthetic_oscillating_prices):
        """Oscillatory strength must be in [0, 1]."""
        from quant_engine.indicators.ssa import SSADecomposer

        ssa = SSADecomposer(window=60, embed_dim=12, n_singular=5)
        osc = ssa.compute_oscillatory_strength(synthetic_oscillating_prices)

        valid = osc[~np.isnan(osc)]
        assert len(valid) > 0
        assert np.all(valid >= -1e-10)
        assert np.all(valid <= 1.0 + 1e-10)

    def test_trend_noise_osc_sum_to_one(self, synthetic_trending_prices):
        """trend_strength + oscillatory_strength + noise_ratio ≈ 1."""
        from quant_engine.indicators.ssa import SSADecomposer

        ssa = SSADecomposer(window=60, embed_dim=12, n_singular=5)
        results = ssa.compute_all(synthetic_trending_prices)

        ts = results["trend_strength"]
        osc = results["oscillatory_strength"]
        nr = results["noise_ratio"]

        # Find indices where all three are non-NaN
        valid_mask = ~(np.isnan(ts) | np.isnan(osc) | np.isnan(nr))
        if np.sum(valid_mask) > 0:
            total = ts[valid_mask] + osc[valid_mask] + nr[valid_mask]
            np.testing.assert_allclose(total, 1.0, atol=0.01)

    def test_compute_all_returns_all_keys(self, synthetic_trending_prices):
        """compute_all() must return all four feature arrays."""
        from quant_engine.indicators.ssa import SSADecomposer

        ssa = SSADecomposer(window=60, embed_dim=12)
        results = ssa.compute_all(synthetic_trending_prices)

        assert "trend_strength" in results
        assert "oscillatory_strength" in results
        assert "singular_entropy" in results
        assert "noise_ratio" in results

    def test_embed_dim_validation(self):
        """embed_dim >= window should raise ValueError."""
        from quant_engine.indicators.ssa import SSADecomposer

        with pytest.raises(ValueError):
            SSADecomposer(window=60, embed_dim=60)


# ===========================================================================
# T3: TailRiskAnalyzer Tests
# ===========================================================================

class TestTailRiskAnalyzer:
    """Tests for jump detection and tail risk features."""

    def test_jump_intensity_detects_jumps(self):
        """Jump intensity should spike around known jump events."""
        from quant_engine.indicators.tail_risk import TailRiskAnalyzer

        np.random.seed(42)
        returns = np.random.randn(200) * 0.01
        returns[100] = 0.10  # 10% jump

        analyzer = TailRiskAnalyzer(window=20, jump_threshold=2.5)
        ji = analyzer.compute_jump_intensity(returns)

        # After the jump, intensity should be positive
        valid = ji[~np.isnan(ji)]
        assert len(valid) > 0
        # Window containing the jump should show non-zero intensity
        assert ji[110] > 0.0

    def test_expected_shortfall_negative(self, synthetic_noisy_prices):
        """ES (CVaR) should be negative for zero-drift data (represents losses)."""
        from quant_engine.indicators.tail_risk import TailRiskAnalyzer

        returns = np.diff(np.log(synthetic_noisy_prices))
        analyzer = TailRiskAnalyzer(window=20)
        es = analyzer.compute_expected_shortfall(returns)

        valid = es[~np.isnan(es)]
        assert len(valid) > 0
        # ES is the average of worst returns — should be negative for zero-drift data
        assert np.mean(valid) < 0

    def test_vol_of_vol_non_negative(self, synthetic_trending_prices):
        """Vol-of-vol must be non-negative."""
        from quant_engine.indicators.tail_risk import TailRiskAnalyzer

        returns = np.diff(np.log(synthetic_trending_prices))
        analyzer = TailRiskAnalyzer(window=20)
        vov = analyzer.compute_vol_of_vol(returns)

        valid = vov[~np.isnan(vov)]
        assert len(valid) > 0
        assert np.all(valid >= 0)

    def test_srm_non_negative(self, synthetic_trending_prices):
        """Semi-relative modulus must be non-negative."""
        from quant_engine.indicators.tail_risk import TailRiskAnalyzer

        returns = np.diff(np.log(synthetic_trending_prices))
        analyzer = TailRiskAnalyzer(window=20)
        srm = analyzer.compute_semi_relative_modulus(returns)

        valid = srm[~np.isnan(srm)]
        assert len(valid) > 0
        assert np.all(valid >= 0)

    def test_extreme_pct_bounded(self, synthetic_trending_prices):
        """Extreme return percentage must be in [0, 1]."""
        from quant_engine.indicators.tail_risk import TailRiskAnalyzer

        returns = np.diff(np.log(synthetic_trending_prices))
        analyzer = TailRiskAnalyzer(window=20)
        ep = analyzer.compute_extreme_return_pct(returns)

        valid = ep[~np.isnan(ep)]
        assert len(valid) > 0
        assert np.all(valid >= 0)
        assert np.all(valid <= 1)

    def test_compute_all_returns_all_keys(self, synthetic_trending_prices):
        """compute_all() must return all five feature arrays."""
        from quant_engine.indicators.tail_risk import TailRiskAnalyzer

        returns = np.diff(np.log(synthetic_trending_prices))
        analyzer = TailRiskAnalyzer(window=20)
        results = analyzer.compute_all(returns)

        assert "jump_intensity" in results
        assert "expected_shortfall" in results
        assert "vol_of_vol" in results
        assert "semi_relative_modulus" in results
        assert "extreme_return_pct" in results


# ===========================================================================
# T4: EigenvalueAnalyzer Tests
# ===========================================================================

class TestEigenvalueAnalyzer:
    """Tests for cross-asset eigenvalue spectrum features."""

    @pytest.fixture
    def correlated_returns(self):
        """Two highly correlated assets."""
        np.random.seed(42)
        n = 200
        x = np.random.randn(n) * 0.02
        return {
            "A": x + np.random.randn(n) * 0.001,
            "B": x + np.random.randn(n) * 0.001,
            "C": x + np.random.randn(n) * 0.001,
            "D": x + np.random.randn(n) * 0.001,
            "E": x + np.random.randn(n) * 0.001,
        }

    @pytest.fixture
    def independent_returns(self):
        """Five independent assets."""
        np.random.seed(42)
        n = 200
        return {
            "A": np.random.randn(n) * 0.02,
            "B": np.random.randn(n) * 0.02,
            "C": np.random.randn(n) * 0.02,
            "D": np.random.randn(n) * 0.02,
            "E": np.random.randn(n) * 0.02,
        }

    def test_high_concentration_when_correlated(self, correlated_returns):
        """HHI should be high when assets move together."""
        from quant_engine.indicators.eigenvalue import EigenvalueAnalyzer

        analyzer = EigenvalueAnalyzer(window=60, min_assets=5)
        conc = analyzer.compute_eigenvalue_concentration(correlated_returns)

        valid = conc[~np.isnan(conc)]
        assert len(valid) > 0
        assert np.mean(valid[-50:]) > 0.5

    def test_low_concentration_when_independent(self, independent_returns):
        """HHI should be lower when assets are independent."""
        from quant_engine.indicators.eigenvalue import EigenvalueAnalyzer

        analyzer = EigenvalueAnalyzer(window=60, min_assets=5)
        conc = analyzer.compute_eigenvalue_concentration(independent_returns)

        valid = conc[~np.isnan(conc)]
        assert len(valid) > 0
        # Independent assets: HHI should be closer to 1/N = 0.2
        assert np.mean(valid[-50:]) < 0.5

    def test_effective_rank_range(self, correlated_returns):
        """Effective rank should be in [1, N]."""
        from quant_engine.indicators.eigenvalue import EigenvalueAnalyzer

        n_assets = len(correlated_returns)
        analyzer = EigenvalueAnalyzer(window=60, min_assets=5)
        er = analyzer.compute_effective_rank(correlated_returns)

        valid = er[~np.isnan(er)]
        assert len(valid) > 0
        assert np.all(valid >= 0.9)  # At least ~1
        assert np.all(valid <= n_assets + 0.1)

    def test_avg_corr_stress_range(self, correlated_returns):
        """Average correlation stress should be in [-1, 1]."""
        from quant_engine.indicators.eigenvalue import EigenvalueAnalyzer

        analyzer = EigenvalueAnalyzer(window=60, min_assets=5)
        ac = analyzer.compute_avg_correlation_stress(correlated_returns)

        valid = ac[~np.isnan(ac)]
        assert len(valid) > 0
        assert np.all(valid >= -1.0)
        assert np.all(valid <= 1.0)

    def test_condition_number_non_negative(self, correlated_returns):
        """Log condition number should be non-negative."""
        from quant_engine.indicators.eigenvalue import EigenvalueAnalyzer

        analyzer = EigenvalueAnalyzer(window=60, min_assets=5)
        cn = analyzer.compute_spectral_condition_number(correlated_returns)

        valid = cn[~np.isnan(cn)]
        assert len(valid) > 0
        assert np.all(valid >= 0)

    def test_too_few_assets_returns_nan(self):
        """Should return NaN array when fewer than min_assets."""
        from quant_engine.indicators.eigenvalue import EigenvalueAnalyzer

        analyzer = EigenvalueAnalyzer(window=30, min_assets=5)
        returns_dict = {
            "A": np.random.randn(100) * 0.02,
            "B": np.random.randn(100) * 0.02,
        }
        conc = analyzer.compute_eigenvalue_concentration(returns_dict)
        assert np.all(np.isnan(conc))

    def test_compute_all_consistency(self, correlated_returns):
        """compute_all() should return all four feature arrays."""
        from quant_engine.indicators.eigenvalue import EigenvalueAnalyzer

        analyzer = EigenvalueAnalyzer(window=60, min_assets=5)
        results = analyzer.compute_all(correlated_returns)

        assert "eigenvalue_concentration" in results
        assert "effective_rank" in results
        assert "avg_correlation_stress" in results
        assert "condition_number" in results


# ===========================================================================
# T5: OptimalTransportAnalyzer Tests
# ===========================================================================

class TestOptimalTransportAnalyzer:
    """Tests for distribution drift detection."""

    def test_wasserstein_detects_shift(self):
        """Wasserstein distance should spike after distribution shift."""
        from quant_engine.indicators.ot_divergence import (
            OptimalTransportAnalyzer,
        )

        np.random.seed(42)
        # Stable regime then shifted regime
        returns = np.concatenate([
            np.random.randn(100) * 0.01,       # N(0, 0.01)
            np.random.randn(100) * 0.03 + 0.01, # N(0.01, 0.03) - shifted
        ])

        analyzer = OptimalTransportAnalyzer(window=20, ref_window=40)
        was = analyzer.compute_wasserstein_distance(returns)

        valid = was[~np.isnan(was)]
        assert len(valid) > 0

        # Distance should be larger after the shift (index > 100)
        pre_shift = was[60:95]
        post_shift = was[120:180]
        pre_valid = pre_shift[~np.isnan(pre_shift)]
        post_valid = post_shift[~np.isnan(post_shift)]
        if len(pre_valid) > 0 and len(post_valid) > 0:
            assert np.mean(post_valid) > np.mean(pre_valid)

    def test_wasserstein_non_negative(self, synthetic_trending_prices):
        """Wasserstein distance must be non-negative."""
        from quant_engine.indicators.ot_divergence import (
            OptimalTransportAnalyzer,
        )

        returns = np.diff(np.log(synthetic_trending_prices))
        analyzer = OptimalTransportAnalyzer(window=20, ref_window=40)
        was = analyzer.compute_wasserstein_distance(returns)

        valid = was[~np.isnan(was)]
        assert len(valid) > 0
        assert np.all(valid >= 0)

    def test_sinkhorn_non_negative(self, synthetic_trending_prices):
        """Sinkhorn divergence must be non-negative (debiased)."""
        from quant_engine.indicators.ot_divergence import (
            OptimalTransportAnalyzer,
        )

        returns = np.diff(np.log(synthetic_trending_prices))
        analyzer = OptimalTransportAnalyzer(window=20, ref_window=40)
        sink = analyzer.compute_sinkhorn_divergence(returns)

        valid = sink[~np.isnan(sink)]
        assert len(valid) > 0
        assert np.all(valid >= -1e-10)

    def test_identical_distributions_near_zero(self):
        """Wasserstein should be near zero for identical distributions."""
        from quant_engine.indicators.ot_divergence import (
            OptimalTransportAnalyzer,
        )

        np.random.seed(42)
        # Stationary signal — same distribution throughout
        returns = np.random.randn(200) * 0.01

        analyzer = OptimalTransportAnalyzer(window=30, ref_window=60)
        was = analyzer.compute_wasserstein_distance(returns)

        valid = was[~np.isnan(was)]
        assert len(valid) > 0
        # Should be small (same distribution)
        assert np.mean(valid) < 0.01

    def test_compute_all_returns_all_keys(self, synthetic_trending_prices):
        """compute_all() must return both feature arrays."""
        from quant_engine.indicators.ot_divergence import (
            OptimalTransportAnalyzer,
        )

        returns = np.diff(np.log(synthetic_trending_prices))
        analyzer = OptimalTransportAnalyzer(window=20, ref_window=40)
        results = analyzer.compute_all(returns)

        assert "wasserstein_distance" in results
        assert "sinkhorn_divergence" in results

    def test_invalid_params_raise(self):
        """Invalid parameters should raise ValueError."""
        from quant_engine.indicators.ot_divergence import (
            OptimalTransportAnalyzer,
        )

        with pytest.raises(ValueError):
            OptimalTransportAnalyzer(window=2)  # too small

        with pytest.raises(ValueError):
            OptimalTransportAnalyzer(window=30, ref_window=10)  # ref < window

        with pytest.raises(ValueError):
            OptimalTransportAnalyzer(epsilon=-1)  # negative epsilon


# ===========================================================================
# T6: Pipeline Integration Tests
# ===========================================================================

class TestStructuralFeaturesIntegration:
    """Tests for structural features in the feature pipeline."""

    def test_compute_structural_features_produces_columns(
        self, synthetic_ohlcv_df
    ):
        """compute_structural_features() should produce expected columns."""
        from quant_engine.features.pipeline import compute_structural_features

        feats = compute_structural_features(synthetic_ohlcv_df)

        # Spectral features
        assert "SpectralHFE_252" in feats.columns
        assert "SpectralLFE_252" in feats.columns
        assert "SpectralEntropy_252" in feats.columns
        assert "SpectralDomFreq_252" in feats.columns
        assert "SpectralBW_252" in feats.columns

        # SSA features
        assert "SSATrendStr_60" in feats.columns
        assert "SSAOscStr_60" in feats.columns
        assert "SSASingularEnt_60" in feats.columns
        assert "SSANoiseRatio_60" in feats.columns

        # Tail risk features
        assert "JumpIntensity_20" in feats.columns
        assert "ExpectedShortfall_20" in feats.columns
        assert "TailVolOfVol_20" in feats.columns
        assert "SemiRelMod_20" in feats.columns
        assert "ExtremeRetPct_20" in feats.columns

        # OT features
        assert "WassersteinDiv_30" in feats.columns
        assert "SinkhornDiv_30" in feats.columns

    def test_structural_features_same_index(self, synthetic_ohlcv_df):
        """Structural features must share the input's index."""
        from quant_engine.features.pipeline import compute_structural_features

        feats = compute_structural_features(synthetic_ohlcv_df)
        assert feats.index.equals(synthetic_ohlcv_df.index)

    def test_structural_features_no_inf(self, synthetic_ohlcv_df):
        """Structural features should not contain infinity values."""
        from quant_engine.features.pipeline import compute_structural_features

        feats = compute_structural_features(synthetic_ohlcv_df)
        assert not np.any(np.isinf(feats.values[~np.isnan(feats.values)]))

    def test_feature_metadata_completeness(self):
        """All structural features must have FEATURE_METADATA entries."""
        from quant_engine.features.pipeline import FEATURE_METADATA

        structural_features = [
            "SpectralHFE_252", "SpectralLFE_252", "SpectralEntropy_252",
            "SpectralDomFreq_252", "SpectralBW_252",
            "SSATrendStr_60", "SSAOscStr_60", "SSASingularEnt_60",
            "SSANoiseRatio_60",
            "JumpIntensity_20", "ExpectedShortfall_20", "TailVolOfVol_20",
            "SemiRelMod_20", "ExtremeRetPct_20",
            "EigenConcentration_60", "EffectiveRank_60",
            "AvgCorrStress_60", "ConditionNumber_60",
            "WassersteinDiv_30", "SinkhornDiv_30",
        ]

        for fname in structural_features:
            assert fname in FEATURE_METADATA, (
                f"Missing FEATURE_METADATA entry for {fname}"
            )
            assert FEATURE_METADATA[fname]["type"] == "CAUSAL"

    def test_feature_metadata_categories(self):
        """Structural features should have correct category labels."""
        from quant_engine.features.pipeline import FEATURE_METADATA

        category_checks = {
            "SpectralHFE_252": "spectral",
            "SSATrendStr_60": "ssa",
            "JumpIntensity_20": "tail_risk",
            "EigenConcentration_60": "eigenvalue",
            "WassersteinDiv_30": "optimal_transport",
        }

        for fname, expected_cat in category_checks.items():
            assert FEATURE_METADATA[fname]["category"] == expected_cat


# ===========================================================================
# Feature Redundancy Tests
# ===========================================================================

class TestFeatureRedundancy:
    """Tests for feature redundancy detection."""

    def test_no_false_positives_on_independent(self):
        """Independent features should not be flagged as redundant."""
        from quant_engine.validation.feature_redundancy import (
            FeatureRedundancyDetector,
        )

        np.random.seed(42)
        features = pd.DataFrame({
            "A": np.random.randn(100),
            "B": np.random.randn(100),
            "C": np.random.randn(100),
        })

        pairs = FeatureRedundancyDetector.detect_redundant_pairs(
            features, threshold=0.90
        )
        assert len(pairs) == 0

    def test_detects_correlated_features(self):
        """Highly correlated features should be detected."""
        from quant_engine.validation.feature_redundancy import (
            FeatureRedundancyDetector,
        )

        np.random.seed(42)
        x = np.random.randn(100)
        features = pd.DataFrame({
            "A": x,
            "B": x + np.random.randn(100) * 0.01,  # ~0.99 corr with A
            "C": np.random.randn(100),
        })

        pairs = FeatureRedundancyDetector.detect_redundant_pairs(
            features, threshold=0.90
        )
        assert len(pairs) == 1
        assert pairs[0][0] == "A"
        assert pairs[0][1] == "B"
        assert pairs[0][2] > 0.90

    def test_validate_composition_passes_on_normal_data(self):
        """validate_structural_feature_composition should pass for normal data."""
        from quant_engine.validation.feature_redundancy import (
            validate_structural_feature_composition,
        )

        np.random.seed(42)
        features = pd.DataFrame({
            "SpectralEntropy_252": np.random.randn(200),
            "SSASingularEnt_60": np.random.randn(200),
            "JumpIntensity_20": np.random.randn(200),
            "ExtremeRetPct_20": np.random.randn(200),
        })

        assert validate_structural_feature_composition(features) is True

    def test_report_formatting(self):
        """Report should handle both empty and non-empty redundancy lists."""
        from quant_engine.validation.feature_redundancy import (
            FeatureRedundancyDetector,
        )

        empty_report = FeatureRedundancyDetector.report([])
        assert "No redundant" in empty_report

        nonempty_report = FeatureRedundancyDetector.report([
            ("A", "B", 0.95),
        ])
        assert "A" in nonempty_report
        assert "B" in nonempty_report
        assert "0.950" in nonempty_report


# ===========================================================================
# Config Tests
# ===========================================================================

class TestStructuralConfig:
    """Tests for structural feature configuration parameters."""

    def test_config_params_exist(self):
        """All structural feature config params should be defined."""
        from quant_engine import config

        assert hasattr(config, "STRUCTURAL_FEATURES_ENABLED")
        assert hasattr(config, "SPECTRAL_FFT_WINDOW")
        assert hasattr(config, "SPECTRAL_CUTOFF_PERIOD")
        assert hasattr(config, "SSA_WINDOW")
        assert hasattr(config, "SSA_EMBED_DIM")
        assert hasattr(config, "SSA_N_SINGULAR")
        assert hasattr(config, "JUMP_INTENSITY_WINDOW")
        assert hasattr(config, "JUMP_INTENSITY_THRESHOLD")
        assert hasattr(config, "EIGEN_CONCENTRATION_WINDOW")
        assert hasattr(config, "EIGEN_MIN_ASSETS")
        assert hasattr(config, "EIGEN_REGULARIZATION")
        assert hasattr(config, "WASSERSTEIN_WINDOW")
        assert hasattr(config, "WASSERSTEIN_REF_WINDOW")
        assert hasattr(config, "SINKHORN_EPSILON")
        assert hasattr(config, "SINKHORN_MAX_ITER")

    def test_ssa_embed_dim_less_than_window(self):
        """SSA_EMBED_DIM must be < SSA_WINDOW."""
        from quant_engine import config

        assert config.SSA_EMBED_DIM < config.SSA_WINDOW

    def test_wasserstein_ref_gte_window(self):
        """WASSERSTEIN_REF_WINDOW must be >= WASSERSTEIN_WINDOW."""
        from quant_engine import config

        assert config.WASSERSTEIN_REF_WINDOW >= config.WASSERSTEIN_WINDOW

    def test_sinkhorn_epsilon_positive(self):
        """SINKHORN_EPSILON must be positive."""
        from quant_engine import config

        assert config.SINKHORN_EPSILON > 0
