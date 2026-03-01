"""Tests for SPEC_AUDIT_FIX_22: Indicator Numerical Stability.

Verifies:
- SRM cap in TailRiskAnalyzer
- Flat-bar input produces NaN (not inf) for guarded indicators
- Zero-price input produces finite values (not -inf) for log-based indicators
"""

import numpy as np
import pandas as pd
import pytest

from indicators.tail_risk import TailRiskAnalyzer, _SRM_CAP
from indicators.indicators import (
    Stochastic, WilliamsR, CandleBody, AccumulationDistribution,
    VolumeRatio, RVOL, VWAP, MFI, ADX, VolatilityRegime,
    NetVolumeTrend, CCI, PriceVsSMA, SMASlope, GapPercent,
    DistanceFromHigh, DistanceFromLow, ATRTrailingStop, ATRChannel,
    RiskPerATR, PriceVsVWAP, VWAPBands, PriceVsPOC, ValueAreaPosition,
    ParkinsonVolatility, GarmanKlassVolatility, YangZhangVolatility,
    HurstExponent, DFA, NATR, BollingerBandWidth, RSI, ROC,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flat_df(n: int = 200, price: float = 100.0) -> pd.DataFrame:
    """DataFrame where OHLC are identical (flat bars) with zero volume."""
    return pd.DataFrame({
        'Open': np.full(n, price),
        'High': np.full(n, price),
        'Low': np.full(n, price),
        'Close': np.full(n, price),
        'Volume': np.zeros(n),
    })


def _make_normal_df(n: int = 200) -> pd.DataFrame:
    """DataFrame with realistic random walk prices and volume."""
    rng = np.random.RandomState(42)
    close = 100 * np.exp(np.cumsum(rng.randn(n) * 0.01))
    high = close * (1 + rng.rand(n) * 0.02)
    low = close * (1 - rng.rand(n) * 0.02)
    open_ = close * (1 + (rng.rand(n) - 0.5) * 0.01)
    volume = rng.randint(1000, 100000, n).astype(float)
    return pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    })


def _make_zero_price_df(n: int = 200) -> pd.DataFrame:
    """DataFrame containing zero prices (edge case for log guards)."""
    df = _make_normal_df(n)
    # Set a few bars to have zero prices
    df.loc[50, ['Open', 'High', 'Low', 'Close']] = 0.0
    df.loc[51, 'Low'] = 0.0
    return df


# ---------------------------------------------------------------------------
# T1: SRM Cap tests
# ---------------------------------------------------------------------------

class TestSRMCap:
    """Verify SRM never exceeds _SRM_CAP."""

    def test_semi_relative_modulus_capped(self):
        """Verify SRM doesn't exceed cap even during extreme drawdown."""
        analyzer = TailRiskAnalyzer(window=20)
        # All negative returns (zero upside variance)
        returns = np.full(100, -0.01)
        srm = analyzer.compute_semi_relative_modulus(returns)
        assert np.all(srm[20:] <= _SRM_CAP), f"SRM exceeded cap: max={np.nanmax(srm)}"

    def test_srm_cap_value(self):
        """Verify _SRM_CAP is 10.0 as specified."""
        assert _SRM_CAP == 10.0

    def test_srm_normal_branch_capped(self):
        """Verify SRM is capped even when up_var is tiny but nonzero."""
        analyzer = TailRiskAnalyzer(window=20)
        # Mix: mostly negative, a few tiny positives to have nonzero up_var
        returns = np.full(100, -0.01)
        returns[::20] = 1e-10  # Tiny positive
        srm = analyzer.compute_semi_relative_modulus(returns)
        assert np.all(np.isnan(srm) | (srm <= _SRM_CAP)), \
            f"SRM exceeded cap in normal branch: max={np.nanmax(srm)}"

    def test_srm_zero_returns(self):
        """Verify SRM is 0 when all returns are zero."""
        analyzer = TailRiskAnalyzer(window=20)
        returns = np.zeros(100)
        srm = analyzer.compute_semi_relative_modulus(returns)
        assert np.all(np.isnan(srm) | (srm == 0.0))


# ---------------------------------------------------------------------------
# T2: Division guard tests — flat bars
# ---------------------------------------------------------------------------

class TestFlatBarGuards:
    """Verify flat-bar input produces NaN (not inf) for guarded indicators."""

    @pytest.fixture
    def flat_df(self):
        return _make_flat_df()

    def _assert_no_inf(self, series: pd.Series, name: str):
        """Assert no inf values in series."""
        values = series.dropna().values
        if len(values) > 0:
            assert not np.any(np.isinf(values)), \
                f"{name} produced inf values: {values[np.isinf(values)]}"

    def test_stochastic_flat(self, flat_df):
        result = Stochastic(14).compute(flat_df)
        self._assert_no_inf(result, "Stochastic")

    def test_williams_r_flat(self, flat_df):
        result = WilliamsR(14).compute(flat_df)
        self._assert_no_inf(result, "WilliamsR")

    def test_candle_body_flat(self, flat_df):
        result = CandleBody().compute(flat_df)
        self._assert_no_inf(result, "CandleBody")

    def test_accumulation_distribution_flat(self, flat_df):
        result = AccumulationDistribution(10).compute(flat_df)
        self._assert_no_inf(result, "AccumulationDistribution")

    def test_volume_ratio_flat(self, flat_df):
        result = VolumeRatio(20).compute(flat_df)
        self._assert_no_inf(result, "VolumeRatio")

    def test_rvol_flat(self, flat_df):
        result = RVOL(20).compute(flat_df)
        self._assert_no_inf(result, "RVOL")

    def test_vwap_flat(self, flat_df):
        result = VWAP(20).compute(flat_df)
        self._assert_no_inf(result, "VWAP")

    def test_mfi_flat(self, flat_df):
        result = MFI(14).compute(flat_df)
        self._assert_no_inf(result, "MFI")

    def test_adx_flat(self, flat_df):
        result = ADX(14).compute(flat_df)
        self._assert_no_inf(result, "ADX")

    def test_volatility_regime_flat(self, flat_df):
        result = VolatilityRegime(20, 100).compute(flat_df)
        self._assert_no_inf(result, "VolatilityRegime")

    def test_net_volume_trend_flat(self, flat_df):
        result = NetVolumeTrend(14).compute(flat_df)
        self._assert_no_inf(result, "NetVolumeTrend")

    def test_cci_flat(self, flat_df):
        result = CCI(20).compute(flat_df)
        self._assert_no_inf(result, "CCI")

    def test_price_vs_sma_flat(self, flat_df):
        result = PriceVsSMA(50).compute(flat_df)
        self._assert_no_inf(result, "PriceVsSMA")

    def test_sma_slope_flat(self, flat_df):
        result = SMASlope(20, 5).compute(flat_df)
        self._assert_no_inf(result, "SMASlope")

    def test_gap_percent_flat(self, flat_df):
        result = GapPercent().compute(flat_df)
        self._assert_no_inf(result, "GapPercent")

    def test_distance_from_high_flat(self, flat_df):
        result = DistanceFromHigh(52).compute(flat_df)
        self._assert_no_inf(result, "DistanceFromHigh")

    def test_distance_from_low_flat(self, flat_df):
        result = DistanceFromLow(52).compute(flat_df)
        self._assert_no_inf(result, "DistanceFromLow")

    def test_atr_trailing_stop_flat(self, flat_df):
        result = ATRTrailingStop(14, 2.0).compute(flat_df)
        self._assert_no_inf(result, "ATRTrailingStop")

    def test_atr_channel_flat(self, flat_df):
        result = ATRChannel(14, 2.0).compute(flat_df)
        self._assert_no_inf(result, "ATRChannel")

    def test_risk_per_atr_flat(self, flat_df):
        result = RiskPerATR(14, 5).compute(flat_df)
        self._assert_no_inf(result, "RiskPerATR")

    def test_price_vs_vwap_flat(self, flat_df):
        result = PriceVsVWAP(20).compute(flat_df)
        self._assert_no_inf(result, "PriceVsVWAP")

    def test_vwap_bands_flat(self, flat_df):
        result = VWAPBands(20, 2.0).compute(flat_df)
        self._assert_no_inf(result, "VWAPBands")

    def test_natr_flat(self, flat_df):
        result = NATR(14).compute(flat_df)
        self._assert_no_inf(result, "NATR")

    def test_bb_width_flat(self, flat_df):
        result = BollingerBandWidth(20).compute(flat_df)
        self._assert_no_inf(result, "BollingerBandWidth")

    def test_rsi_flat(self, flat_df):
        result = RSI(14).compute(flat_df)
        self._assert_no_inf(result, "RSI")

    def test_roc_flat(self, flat_df):
        result = ROC(10).compute(flat_df)
        self._assert_no_inf(result, "ROC")


# ---------------------------------------------------------------------------
# T3: Log positivity guard tests
# ---------------------------------------------------------------------------

class TestLogPositivityGuards:
    """Verify zero-price input produces finite values (not -inf) for log-based indicators."""

    @pytest.fixture
    def zero_price_df(self):
        return _make_zero_price_df()

    def _assert_no_neg_inf(self, series: pd.Series, name: str):
        """Assert no -inf values in series."""
        values = series.dropna().values
        if len(values) > 0:
            assert not np.any(np.isneginf(values)), \
                f"{name} produced -inf values"

    def test_parkinson_volatility_zero_price(self, zero_price_df):
        result = ParkinsonVolatility(20).compute(zero_price_df)
        self._assert_no_neg_inf(result, "ParkinsonVolatility")

    def test_garman_klass_volatility_zero_price(self, zero_price_df):
        result = GarmanKlassVolatility(20).compute(zero_price_df)
        self._assert_no_neg_inf(result, "GarmanKlassVolatility")

    def test_yang_zhang_volatility_zero_price(self, zero_price_df):
        result = YangZhangVolatility(20).compute(zero_price_df)
        self._assert_no_neg_inf(result, "YangZhangVolatility")

    def test_hurst_exponent_zero_price(self, zero_price_df):
        result = HurstExponent(100).compute(zero_price_df)
        self._assert_no_neg_inf(result, "HurstExponent")

    def test_dfa_zero_price(self, zero_price_df):
        result = DFA(100).compute(zero_price_df)
        self._assert_no_neg_inf(result, "DFA")


# ---------------------------------------------------------------------------
# Smoke test: normal data produces no inf
# ---------------------------------------------------------------------------

class TestNormalDataNoInf:
    """Verify that normal data also doesn't produce inf after guards."""

    @pytest.fixture
    def normal_df(self):
        return _make_normal_df(500)

    def test_all_guarded_indicators_normal(self, normal_df):
        """All guarded indicators should produce finite output on normal data."""
        indicators = [
            Stochastic(14), WilliamsR(14), CandleBody(), ADX(14),
            VolumeRatio(20), RVOL(20), VWAP(20), MFI(14),
            CCI(20), PriceVsSMA(50), ATRChannel(14), RiskPerATR(14, 5),
            ParkinsonVolatility(20), GarmanKlassVolatility(20),
            YangZhangVolatility(20), NATR(14), BollingerBandWidth(20),
            RSI(14), ROC(10),
        ]
        for ind in indicators:
            result = ind.compute(normal_df)
            values = result.dropna().values
            assert not np.any(np.isinf(values)), \
                f"{ind.name} produced inf on normal data"
