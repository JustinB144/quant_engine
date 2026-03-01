"""Tests for SPEC_AUDIT_FIX_11: Indicators Mathematical Correctness & Causality."""

import numpy as np
import pandas as pd
import pytest

from indicators.indicators import (
    Aroon,
    PivotHigh,
    PivotLow,
    RegimePersistence,
    create_indicator,
    get_all_indicators,
    _sanitize_output,
)
from indicators.tail_risk import TailRiskAnalyzer


def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a simple OHLCV DataFrame for testing."""
    rng = np.random.RandomState(seed)
    close = 100 + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.rand(n) * 2
    low = close - rng.rand(n) * 2
    open_ = close + rng.randn(n) * 0.3
    volume = rng.randint(1000, 10000, size=n).astype(float)
    return pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    })


# =============================================================================
# T1: Aroon Oscillator Inversion
# =============================================================================

class TestAroonFix:
    """Verify Aroon Up=100 when most recent bar is the period high."""

    def test_aroon_up_100_when_latest_is_high(self):
        """If the most recent bar is the highest, Aroon Up should be 100."""
        n = 30
        period = 25
        # Create data where the last bar is clearly the highest
        prices = np.arange(1.0, n + 1)  # monotonically increasing
        df = pd.DataFrame({
            'Open': prices,
            'High': prices,  # Last bar = highest
            'Low': prices - 0.5,
            'Close': prices,
            'Volume': np.ones(n) * 1000,
        })
        aroon = Aroon(period=period)
        result = aroon.calculate(df)
        # Aroon Oscillator = AroonUp - AroonDown
        # For monotonically increasing data: AroonUp = 100, AroonDown is low
        # So oscillator should be high positive
        last_value = result.iloc[-1]
        # AroonUp should be 100 (most recent is highest) and
        # AroonDown should be low (most recent is NOT the lowest)
        assert last_value > 50, f"Aroon should be strongly positive, got {last_value}"

    def test_aroon_down_100_when_latest_is_low(self):
        """If the most recent bar is the lowest, Aroon Down should be 100."""
        n = 30
        period = 25
        # Monotonically decreasing prices
        prices = np.arange(n, 0.0, -1)
        df = pd.DataFrame({
            'Open': prices,
            'High': prices + 0.5,
            'Low': prices,  # Last bar = lowest
            'Close': prices,
            'Volume': np.ones(n) * 1000,
        })
        aroon = Aroon(period=period)
        result = aroon.calculate(df)
        last_value = result.iloc[-1]
        # AroonDown=100 (newest is lowest), AroonUp is low
        # Oscillator = Up - Down, should be strongly negative
        assert last_value < -50, f"Aroon should be strongly negative, got {last_value}"

    def test_aroon_values_bounded(self):
        """Aroon oscillator should be in [-100, 100]."""
        df = _make_ohlcv(100)
        aroon = Aroon(period=25)
        result = aroon.calculate(df)
        valid = result.dropna()
        assert valid.min() >= -100, f"Aroon min {valid.min()} below -100"
        assert valid.max() <= 100, f"Aroon max {valid.max()} above 100"


# =============================================================================
# T2: PivotHigh/PivotLow Look-Ahead Bias
# =============================================================================

class TestPivotCausality:
    """PivotHigh/PivotLow must use trailing window only (no center=True)."""

    def test_pivot_high_no_future_data(self):
        """PivotHigh at bar i should depend only on bars <= i."""
        df = _make_ohlcv(50)
        pivot = PivotHigh(left_bars=5, right_bars=5)
        full_result = pivot.calculate(df)

        # Truncate at bar 30 and compute; result at bar 30 should match
        df_truncated = df.iloc[:31].copy()
        trunc_result = pivot.calculate(df_truncated)

        # Value at bar 30 should be the same in both cases (causal)
        assert full_result.iloc[30] == trunc_result.iloc[30], (
            "PivotHigh at bar 30 differs when future data is added — look-ahead bias"
        )

    def test_pivot_low_no_future_data(self):
        """PivotLow at bar i should depend only on bars <= i."""
        df = _make_ohlcv(50)
        pivot = PivotLow(left_bars=5, right_bars=5)
        full_result = pivot.calculate(df)

        df_truncated = df.iloc[:31].copy()
        trunc_result = pivot.calculate(df_truncated)

        assert full_result.iloc[30] == trunc_result.iloc[30], (
            "PivotLow at bar 30 differs when future data is added — look-ahead bias"
        )

    def test_pivot_high_uses_trailing_window(self):
        """Verify no center=True in rolling call (structural check)."""
        import inspect
        src = inspect.getsource(PivotHigh.calculate)
        assert "center=True" not in src, "PivotHigh still uses center=True"

    def test_pivot_low_uses_trailing_window(self):
        """Verify no center=True in rolling call (structural check)."""
        import inspect
        src = inspect.getsource(PivotLow.calculate)
        assert "center=True" not in src, "PivotLow still uses center=True"


# =============================================================================
# T3: Expected Shortfall Off-By-One
# =============================================================================

class TestExpectedShortfall:
    """ES at alpha should average worst floor(alpha*n) returns."""

    def test_es_5pct_20_values(self):
        """ES at 5% for 20 values should be the mean of worst 1 value = -10."""
        # Pad with a leading zero so window=20 produces its first result at index 20
        returns = np.array([
            0,  # padding so the 20-element segment starts at index 1
            -10, -5, -3, -1, 0, 1, 2, 3, 5, 10,
            12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        ], dtype=float)
        analyzer = TailRiskAnalyzer(window=20)
        es = analyzer.compute_expected_shortfall(returns, alpha=0.05)
        # At index 20, seg = returns[0:20], worst floor(0.05*20)=1 value = -10
        assert es[20] == pytest.approx(-10.0, abs=1e-10), (
            f"ES at 5% for 20 values should be -10.0, got {es[20]}"
        )

    def test_es_10pct_30_values(self):
        """ES at 10% for 30 values: floor(0.1*30)=3, average worst 3."""
        returns = np.zeros(50)
        # Place known values in the last 30
        seg = np.arange(30, dtype=float) - 15  # -15 to 14
        returns[20:50] = seg
        analyzer = TailRiskAnalyzer(window=30)
        es = analyzer.compute_expected_shortfall(returns, alpha=0.10)
        # worst 3 of [-15, -14, ..., 14] = [-15, -14, -13], mean = -14
        assert es[50 - 1] == pytest.approx(-14.0, abs=1e-10)


# =============================================================================
# T4: RegimePersistence NaN Treatment
# =============================================================================

class TestRegimePersistenceNaN:
    """NaN regime states should not be counted as regime 0."""

    def test_nan_sma_produces_nan_persistence(self):
        """During SMA warmup, persistence should be NaN, not a count."""
        df = _make_ohlcv(30)
        rp = RegimePersistence(period=20)
        result = rp.calculate(df)
        # First period-1 values should be NaN (SMA warmup)
        assert result.iloc[:19].isna().all(), (
            "Persistence during SMA warmup should be NaN"
        )

    def test_nan_regime_not_counted_as_zero(self):
        """Injected NaN in SMA should produce NaN persistence, not false match."""
        df = _make_ohlcv(50)
        # Force some Close values to NaN so SMA becomes NaN
        df.loc[df.index[25:28], 'Close'] = np.nan
        rp = RegimePersistence(period=20)
        result = rp.calculate(df)
        # Bars where SMA is NaN should have NaN persistence
        sma = df['Close'].rolling(window=20).mean()
        nan_sma_idx = sma[sma.isna()].index
        for idx in nan_sma_idx:
            pos = df.index.get_loc(idx)
            if pos >= 19:
                assert pd.isna(result.iloc[pos]), (
                    f"Persistence at bar {pos} should be NaN when SMA is NaN"
                )


# =============================================================================
# T5: Indicator registry in create_indicator()
# (INDICATOR_ALIASES removed per SPEC_AUDIT_FIX_24 T3 — all aliases are now
#  in get_all_indicators() registry directly)
# =============================================================================

class TestIndicatorAliases:
    """create_indicator() should resolve registry entries."""

    def test_registry_resolution(self):
        """All registry entries should create valid indicators."""
        for name, cls in get_all_indicators().items():
            ind = create_indicator(name)
            assert isinstance(ind, cls), (
                f"Registry entry '{name}' did not create {cls.__name__}"
            )

    def test_unknown_indicator_raises(self):
        """Unknown names should still raise ValueError."""
        with pytest.raises(ValueError, match="Unknown indicator"):
            create_indicator("NonExistentIndicator_XYZ")

    def test_registry_takes_precedence(self):
        """Main registry keys should work directly."""
        ind = create_indicator("RSI")
        assert ind.name == "RSI_14"


# =============================================================================
# T6: Inf/NaN Sanitization
# =============================================================================

class TestInfSanitization:
    """Indicator output should never contain inf/-inf."""

    def test_sanitize_replaces_inf(self):
        """_sanitize_output replaces inf with NaN."""
        s = pd.Series([1.0, np.inf, -np.inf, 3.0, np.nan])
        result = _sanitize_output(s)
        assert not np.any(np.isinf(result.values)), "inf values should be replaced"
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])
        assert result.iloc[0] == 1.0
        assert result.iloc[3] == 3.0

    def test_sanitize_preserves_clean_data(self):
        """_sanitize_output should not modify clean data."""
        s = pd.Series([1.0, 2.0, 3.0, np.nan])
        result = _sanitize_output(s)
        pd.testing.assert_series_equal(result, s)

    def test_compute_method_sanitizes(self):
        """The compute() method should sanitize inf values."""
        from indicators.indicators import BollingerBandWidth
        df = _make_ohlcv(50)
        ind = BollingerBandWidth(period=20)
        result = ind.compute(df)
        assert not np.any(np.isinf(result.dropna().values)), (
            "compute() should produce no inf values"
        )

    def test_indicators_no_inf_in_output(self):
        """Spot-check several indicators for inf in output."""
        from indicators.indicators import (
            Stochastic, WilliamsR, ATRChannel, ValueAreaPosition, CCI,
        )
        df = _make_ohlcv(200)
        for cls in [Stochastic, WilliamsR, ATRChannel, CCI]:
            ind = cls()
            result = ind.compute(df)
            inf_count = np.isinf(result.values).sum()
            assert inf_count == 0, (
                f"{ind.name} has {inf_count} inf values after compute()"
            )
