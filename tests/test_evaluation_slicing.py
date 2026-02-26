"""Tests for evaluation/slicing.py — PerformanceSlice and SliceRegistry."""

import numpy as np
import pandas as pd
import pytest

from quant_engine.evaluation.slicing import PerformanceSlice, SliceRegistry


@pytest.fixture
def sample_returns():
    """500-day return series with known properties."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2022-01-03", periods=500)
    returns = pd.Series(rng.normal(0.0005, 0.015, 500), index=dates)
    return returns


@pytest.fixture
def sample_regimes():
    """Regime labels: 250 normal (0/1), 100 mean-rev (2), 150 high-vol (3)."""
    regimes = np.array(
        [0] * 100 + [1] * 150 + [2] * 100 + [3] * 150, dtype=int
    )
    return regimes


class TestPerformanceSlice:
    """Test PerformanceSlice.apply() on various conditions."""

    def test_apply_returns_correct_subset(self, sample_returns):
        mask_fn = lambda meta: meta["regime"].values == 0
        slc = PerformanceSlice(name="Bull", condition=mask_fn)

        meta = pd.DataFrame({"regime": np.zeros(500, dtype=int)}, index=sample_returns.index)
        filtered, info = slc.apply(sample_returns, meta)

        assert info["name"] == "Bull"
        assert info["n_samples"] == 500
        assert info["low_confidence"] is False

    def test_apply_empty_slice(self, sample_returns):
        mask_fn = lambda meta: np.zeros(len(meta), dtype=bool)
        slc = PerformanceSlice(name="Empty", condition=mask_fn)

        meta = pd.DataFrame({"regime": np.zeros(500, dtype=int)}, index=sample_returns.index)
        filtered, info = slc.apply(sample_returns, meta)

        assert info["n_samples"] == 0
        assert info["low_confidence"] is True
        assert len(filtered) == 0

    def test_apply_small_slice_flags_low_confidence(self, sample_returns):
        """Slice with < min_samples should flag low confidence."""
        mask = np.zeros(500, dtype=bool)
        mask[:10] = True
        mask_fn = lambda meta: mask

        slc = PerformanceSlice(name="Tiny", condition=mask_fn, min_samples=20)
        meta = pd.DataFrame({"regime": np.zeros(500, dtype=int)}, index=sample_returns.index)
        filtered, info = slc.apply(sample_returns, meta)

        assert info["n_samples"] == 10
        assert info["low_confidence"] is True


class TestSliceRegistry:
    """Test SliceRegistry.create_regime_slices() and build_metadata()."""

    def test_create_regime_slices_returns_five(self, sample_regimes):
        slices = SliceRegistry.create_regime_slices(sample_regimes)
        assert len(slices) == 5
        names = [s.name for s in slices]
        assert "Normal" in names
        assert "High Vol" in names
        assert "Crash" in names
        assert "Recovery" in names
        assert "Trendless" in names

    def test_normal_slice_selects_regimes_0_and_1(self, sample_returns, sample_regimes):
        slices = SliceRegistry.create_regime_slices(sample_regimes)
        normal_slice = [s for s in slices if s.name == "Normal"][0]

        meta = SliceRegistry.build_metadata(sample_returns, sample_regimes)
        filtered, info = normal_slice.apply(sample_returns, meta)

        assert info["n_samples"] == 250  # 100 regime-0 + 150 regime-1

    def test_high_vol_slice_selects_regime_3(self, sample_returns, sample_regimes):
        slices = SliceRegistry.create_regime_slices(sample_regimes)
        hv_slice = [s for s in slices if s.name == "High Vol"][0]

        meta = SliceRegistry.build_metadata(sample_returns, sample_regimes)
        filtered, info = hv_slice.apply(sample_returns, meta)

        assert info["n_samples"] == 150  # 150 regime-3

    def test_build_metadata_has_required_columns(self, sample_returns, sample_regimes):
        meta = SliceRegistry.build_metadata(sample_returns, sample_regimes)

        required_cols = [
            "regime", "cumulative_return", "drawdown",
            "trailing_return_20d", "volatility", "volatility_median",
        ]
        for col in required_cols:
            assert col in meta.columns, f"Missing column: {col}"

        assert len(meta) == len(sample_returns)

    def test_create_individual_regime_slices(self, sample_returns, sample_regimes):
        slices = SliceRegistry.create_individual_regime_slices()
        assert len(slices) == 4  # One per regime code

        meta = SliceRegistry.build_metadata(sample_returns, sample_regimes)
        total = 0
        for slc in slices:
            filtered, info = slc.apply(sample_returns, meta)
            total += info["n_samples"]

        # All regimes together should cover all samples
        assert total == 500

    def test_slices_handle_volatility_input(self, sample_returns, sample_regimes):
        vol = pd.Series(
            np.random.RandomState(7).uniform(0.1, 0.3, 500),
            index=sample_returns.index,
        )
        meta = SliceRegistry.build_metadata(sample_returns, sample_regimes, volatility=vol)
        assert "volatility" in meta.columns
        assert not meta["volatility"].isna().any()
