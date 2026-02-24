"""
Feature Pipeline — computes model features from OHLCV data.

Includes:
1) classical technical indicators,
2) raw OHLCV transforms,
3) interaction features,
4) research-derived factors (single-asset and cross-asset).

Feature causality types:
    CAUSAL        — uses only past and current data; safe for production
    END_OF_DAY    — uses full-day data; safe for daily predictions only
    RESEARCH_ONLY — may contain cross-sectional or other non-causal info;
                    for offline analysis only
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..indicators import (
    # Volatility - Basic
    ATR, NATR, BollingerBandWidth, HistoricalVolatility,
    BBWidthPercentile, NATRPercentile, VolatilitySqueeze,
    # Momentum
    RSI, MACD, MACDSignal, MACDHistogram, ROC,
    Stochastic, StochasticD, WilliamsR, CCI,
    # Trend
    SMA, EMA, PriceVsSMA, SMASlope, ADX, Aroon,
    EMAAlignment, TrendStrength, PriceVsEMAStack,
    MarketRegime, VolatilityRegime,
    # Volume
    VolumeRatio, OBV, OBVSlope, MFI, RVOL,
    NetVolumeTrend, VolumeForce, AccumulationDistribution,
    # Price Action
    HigherHighs, LowerLows, CandleBody, CandleDirection, GapPercent,
    # Support/Resistance
    DistanceFromHigh, DistanceFromLow, PricePercentile,
    # Pivot/Breakout
    PivotHigh, PivotLow, NBarHighBreak, NBarLowBreak, RangeBreakout,
    # ATR Risk
    ATRTrailingStop, ATRChannel, RiskPerATR,
    # VWAP
    VWAP, PriceVsVWAP, VWAPBands,
    # Value Area
    ValueAreaHigh, ValueAreaLow, POC,
    PriceVsPOC, ValueAreaPosition, AboveValueArea, BelowValueArea,
    # Advanced Volatility
    ParkinsonVolatility, GarmanKlassVolatility, YangZhangVolatility,
    VolatilityCone, VolOfVol, GARCHVolatility, VolTermStructure,
    # Statistical
    HurstExponent, MeanReversionHalfLife, ZScore, VarianceRatio,
    Autocorrelation, KalmanTrend,
    # Information Theory
    ShannonEntropy, ApproximateEntropy,
    # Microstructure
    AmihudIlliquidity, KyleLambda, RollSpread,
    # Fractal
    FractalDimension, DFA, DominantCycle,
    # Distribution
    ReturnSkewness, ReturnKurtosis,
    # Regime Detection
    CUSUMDetector, RegimePersistence,
)
from .research_factors import (
    ResearchFactorConfig,
    compute_cross_asset_research_factors,
    compute_single_asset_research_factors,
)
from .options_factors import compute_option_surface_factors
from .wave_flow import compute_wave_flow_decomposition


# ---------------------------------------------------------------------------
# Feature Metadata Registry
# ---------------------------------------------------------------------------
# Every feature produced by the pipeline should have an entry here.
# Types:
#   CAUSAL        — backward-looking only; safe for live prediction
#   END_OF_DAY    — requires full-day data; safe for daily-close prediction
#   RESEARCH_ONLY — cross-sectional or non-causal; offline analysis only
#
# Features not listed here are assumed CAUSAL (indicators are backward-looking
# by construction).  Only features that deviate need explicit annotation.
# ---------------------------------------------------------------------------

FEATURE_METADATA: Dict[str, Dict[str, str]] = {
    # ── Technical indicators (all CAUSAL — rolling lookback) ──────────
    "ATR_14": {"type": "CAUSAL", "category": "volatility"},
    "NATR_14": {"type": "CAUSAL", "category": "volatility"},
    "BBWidth_20": {"type": "CAUSAL", "category": "volatility"},
    "HV_20": {"type": "CAUSAL", "category": "volatility"},
    "BBWPct_20_100": {"type": "CAUSAL", "category": "volatility"},
    "NATRPct_14_100": {"type": "CAUSAL", "category": "volatility"},
    "Squeeze_20": {"type": "CAUSAL", "category": "volatility"},
    "ParkVol_20": {"type": "CAUSAL", "category": "volatility"},
    "GKVol_20": {"type": "CAUSAL", "category": "volatility"},
    "YZVol_20": {"type": "CAUSAL", "category": "volatility"},
    "VolCone_20_252": {"type": "CAUSAL", "category": "volatility"},
    "VoV_20_60": {"type": "CAUSAL", "category": "volatility"},
    "GARCH_252": {"type": "CAUSAL", "category": "volatility"},
    "VolTS_10_60": {"type": "CAUSAL", "category": "volatility"},
    "RSI_14": {"type": "CAUSAL", "category": "momentum"},
    "RSI_5": {"type": "CAUSAL", "category": "momentum"},
    "MACD_12_26": {"type": "CAUSAL", "category": "momentum"},
    "MACDSignal_12_26_9": {"type": "CAUSAL", "category": "momentum"},
    "MACDHist_12_26_9": {"type": "CAUSAL", "category": "momentum"},
    "ROC_10": {"type": "CAUSAL", "category": "momentum"},
    "ROC_20": {"type": "CAUSAL", "category": "momentum"},
    "ROC_50": {"type": "CAUSAL", "category": "momentum"},
    "Stoch_14": {"type": "CAUSAL", "category": "momentum"},
    "StochD_14_3": {"type": "CAUSAL", "category": "momentum"},
    "WillR_14": {"type": "CAUSAL", "category": "momentum"},
    "CCI_20": {"type": "CAUSAL", "category": "momentum"},
    "SMA_20": {"type": "CAUSAL", "category": "trend"},
    "SMA_50": {"type": "CAUSAL", "category": "trend"},
    "SMA_200": {"type": "CAUSAL", "category": "trend"},
    "EMA_8": {"type": "CAUSAL", "category": "trend"},
    "EMA_21": {"type": "CAUSAL", "category": "trend"},
    "EMA_50": {"type": "CAUSAL", "category": "trend"},
    "PriceVsSMA_20": {"type": "CAUSAL", "category": "trend"},
    "PriceVsSMA_50": {"type": "CAUSAL", "category": "trend"},
    "PriceVsSMA_200": {"type": "CAUSAL", "category": "trend"},
    "SMASlope_20_5": {"type": "CAUSAL", "category": "trend"},
    "SMASlope_50_5": {"type": "CAUSAL", "category": "trend"},
    "ADX_14": {"type": "CAUSAL", "category": "trend"},
    "Aroon_14": {"type": "CAUSAL", "category": "trend"},
    "EMAAlign_8_21_50": {"type": "CAUSAL", "category": "trend"},
    "TrendStr_14": {"type": "CAUSAL", "category": "trend"},
    "PriceVsEMAs": {"type": "CAUSAL", "category": "trend"},
    "Regime_50": {"type": "CAUSAL", "category": "trend"},
    "VolRegime_20": {"type": "CAUSAL", "category": "trend"},
    "VolRatio_20": {"type": "CAUSAL", "category": "volume"},
    "OBV": {"type": "CAUSAL", "category": "volume"},
    "OBVSlope_14": {"type": "CAUSAL", "category": "volume"},
    "MFI_14": {"type": "CAUSAL", "category": "volume"},
    "RVOL_20": {"type": "CAUSAL", "category": "volume"},
    "NetVol_14": {"type": "CAUSAL", "category": "volume"},
    "VForce_13": {"type": "CAUSAL", "category": "volume"},
    "ADSlope_10": {"type": "CAUSAL", "category": "volume"},
    "HH_5": {"type": "CAUSAL", "category": "price_action"},
    "LL_5": {"type": "CAUSAL", "category": "price_action"},
    "CandleBody": {"type": "CAUSAL", "category": "price_action"},
    "CandleDir_5": {"type": "CAUSAL", "category": "price_action"},
    "Gap": {"type": "CAUSAL", "category": "price_action"},
    "DistHigh_252": {"type": "CAUSAL", "category": "support_resistance"},
    "DistLow_252": {"type": "CAUSAL", "category": "support_resistance"},
    "PricePct_252": {"type": "CAUSAL", "category": "support_resistance"},
    "PivotHi_5_5": {"type": "CAUSAL", "category": "breakout"},
    "PivotLo_5_5": {"type": "CAUSAL", "category": "breakout"},
    "HiBreak_5": {"type": "CAUSAL", "category": "breakout"},
    "LoBreak_5": {"type": "CAUSAL", "category": "breakout"},
    "RangeBO_20": {"type": "CAUSAL", "category": "breakout"},
    "ATRStop_14_2.0": {"type": "CAUSAL", "category": "atr_risk"},
    "ATRChan_14": {"type": "CAUSAL", "category": "atr_risk"},
    "RiskATR_14_20": {"type": "CAUSAL", "category": "atr_risk"},
    "VWAP_20": {"type": "CAUSAL", "category": "vwap"},
    "PriceVsVWAP_20": {"type": "CAUSAL", "category": "vwap"},
    "VWAPBand_20": {"type": "CAUSAL", "category": "vwap"},
    "VAH_20": {"type": "CAUSAL", "category": "value_area"},
    "VAL_20": {"type": "CAUSAL", "category": "value_area"},
    "POC_20": {"type": "CAUSAL", "category": "value_area"},
    "PriceVsPOC_20": {"type": "CAUSAL", "category": "value_area"},
    "VAPos_20": {"type": "CAUSAL", "category": "value_area"},
    "AboveVA_20": {"type": "CAUSAL", "category": "value_area"},
    "BelowVA_20": {"type": "CAUSAL", "category": "value_area"},
    "Hurst_100": {"type": "CAUSAL", "category": "statistical"},
    "HalfLife_60": {"type": "CAUSAL", "category": "statistical"},
    "ZScore_20": {"type": "CAUSAL", "category": "statistical"},
    "ZScore_50": {"type": "CAUSAL", "category": "statistical"},
    "VarRatio_100_5": {"type": "CAUSAL", "category": "statistical"},
    "AutoCorr_20_1": {"type": "CAUSAL", "category": "statistical"},
    "Kalman_0p01_1p0": {"type": "CAUSAL", "category": "statistical"},
    "Entropy_20": {"type": "CAUSAL", "category": "information_theory"},
    "ApEn_50": {"type": "CAUSAL", "category": "information_theory"},
    "Amihud_20": {"type": "CAUSAL", "category": "microstructure"},
    "Kyle_20": {"type": "CAUSAL", "category": "microstructure"},
    "RollSpd_20": {"type": "CAUSAL", "category": "microstructure"},
    "FracDim_100": {"type": "CAUSAL", "category": "fractal"},
    "DFA_100": {"type": "CAUSAL", "category": "fractal"},
    "DomCycle_100": {"type": "CAUSAL", "category": "fractal"},
    "Skew_60": {"type": "CAUSAL", "category": "distribution"},
    "Kurt_60": {"type": "CAUSAL", "category": "distribution"},
    "CUSUM_50": {"type": "CAUSAL", "category": "regime"},
    "RegPersist_20": {"type": "CAUSAL", "category": "regime"},

    # ── Raw OHLCV features (all CAUSAL) ──────────────────────────────
    "return_1d": {"type": "CAUSAL", "category": "returns"},
    "return_2d": {"type": "CAUSAL", "category": "returns"},
    "return_3d": {"type": "CAUSAL", "category": "returns"},
    "return_5d": {"type": "CAUSAL", "category": "returns"},
    "return_10d": {"type": "CAUSAL", "category": "returns"},
    "return_20d": {"type": "CAUSAL", "category": "returns"},
    "log_return_1d": {"type": "CAUSAL", "category": "returns"},
    "intraday_range": {"type": "CAUSAL", "category": "price_action"},
    "overnight_gap": {"type": "CAUSAL", "category": "price_action"},
    "log_volume": {"type": "CAUSAL", "category": "volume"},
    "dollar_volume": {"type": "CAUSAL", "category": "volume"},
    "volume_change": {"type": "CAUSAL", "category": "volume"},
    "return_vol_5d": {"type": "CAUSAL", "category": "volatility"},
    "return_vol_20d": {"type": "CAUSAL", "category": "volatility"},
    "return_vol_60d": {"type": "CAUSAL", "category": "volatility"},
    "price_in_range_20d": {"type": "CAUSAL", "category": "price_action"},
    "price_in_range_60d": {"type": "CAUSAL", "category": "price_action"},

    # ── HAR volatility features (all CAUSAL) ─────────────────────────
    "RV_daily": {"type": "CAUSAL", "category": "volatility"},
    "RV_weekly": {"type": "CAUSAL", "category": "volatility"},
    "RV_monthly": {"type": "CAUSAL", "category": "volatility"},
    "HAR_composite": {"type": "CAUSAL", "category": "volatility"},
    "HAR_ratio_dw": {"type": "CAUSAL", "category": "volatility"},
    "HAR_ratio_wm": {"type": "CAUSAL", "category": "volatility"},

    # ── Multi-scale features (all CAUSAL) ────────────────────────────
    "RSI_10": {"type": "CAUSAL", "category": "momentum"},
    "RSI_20": {"type": "CAUSAL", "category": "momentum"},
    "RSI_50": {"type": "CAUSAL", "category": "momentum"},
    "MOM_5": {"type": "CAUSAL", "category": "momentum"},
    "MOM_10": {"type": "CAUSAL", "category": "momentum"},
    "MOM_20": {"type": "CAUSAL", "category": "momentum"},
    "MOM_60": {"type": "CAUSAL", "category": "momentum"},
    "VOL_5": {"type": "CAUSAL", "category": "volatility"},
    "VOL_10": {"type": "CAUSAL", "category": "volatility"},
    "VOL_20": {"type": "CAUSAL", "category": "volatility"},
    "VOL_60": {"type": "CAUSAL", "category": "volatility"},

    # ── Research factors: TSMom (CAUSAL — backward-looking) ──────────
    "TSMom_lag21": {"type": "CAUSAL", "category": "momentum"},
    "TSMom_lag63": {"type": "CAUSAL", "category": "momentum"},
    "TSMom_lag126": {"type": "CAUSAL", "category": "momentum"},
    "TSMom_lag252": {"type": "CAUSAL", "category": "momentum"},
    "TSMom_12m1m": {"type": "CAUSAL", "category": "momentum"},
    "TSMom_Ensemble": {"type": "CAUSAL", "category": "momentum"},
    "TSMom_SignAgreement": {"type": "CAUSAL", "category": "momentum"},

    # ── Research factors: vol-scaled momentum (CAUSAL) ───────────────
    "vsmom_5": {"type": "CAUSAL", "category": "momentum"},
    "vsmom_10": {"type": "CAUSAL", "category": "momentum"},
    "vsmom_20": {"type": "CAUSAL", "category": "momentum"},
    "vsmom_60": {"type": "CAUSAL", "category": "momentum"},
    "vsmom_120": {"type": "CAUSAL", "category": "momentum"},
    "vsmom_252": {"type": "CAUSAL", "category": "momentum"},
    "reversal_horizon": {"type": "CAUSAL", "category": "momentum"},
    "momentum_persistence": {"type": "CAUSAL", "category": "momentum"},

    # ── Relative momentum — RESEARCH_ONLY (needs cross-sectional adjustment) ─
    "relative_mom_10": {
        "type": "RESEARCH_ONLY", "category": "momentum",
        "reason": "Cross-sectional adjustment needed at prediction time",
    },
    "relative_mom_20": {
        "type": "RESEARCH_ONLY", "category": "momentum",
        "reason": "Cross-sectional adjustment needed at prediction time",
    },
    "relative_mom_60": {
        "type": "RESEARCH_ONLY", "category": "momentum",
        "reason": "Cross-sectional adjustment needed at prediction time",
    },

    # ── Research factors: OFI (CAUSAL — rolling) ─────────────────────
    "OFI_20": {"type": "CAUSAL", "category": "order_flow"},
    "OFI_Z_20": {"type": "CAUSAL", "category": "order_flow"},
    "OFI_DepthAdj_20": {"type": "CAUSAL", "category": "order_flow"},
    "ImpactLambda_20": {"type": "CAUSAL", "category": "order_flow"},
    "ImpactLinear_20": {"type": "CAUSAL", "category": "order_flow"},
    "ImpactSqrt_20": {"type": "CAUSAL", "category": "order_flow"},
    "ofi_normalized": {"type": "CAUSAL", "category": "order_flow"},
    "ofi_persistence": {"type": "CAUSAL", "category": "order_flow"},
    "ofi_momentum": {"type": "CAUSAL", "category": "order_flow"},

    # ── Research factors: Markov queue (CAUSAL) ──────────────────────
    "QueueImbalance_Proxy": {"type": "CAUSAL", "category": "microstructure"},
    "LOB_State": {"type": "CAUSAL", "category": "microstructure"},
    "LOB_PUp_State_63": {"type": "CAUSAL", "category": "microstructure"},
    "LOB_PDown_State_63": {"type": "CAUSAL", "category": "microstructure"},
    "LOB_TransitionEntropy_63": {"type": "CAUSAL", "category": "microstructure"},
    "LOB_StateDuration": {"type": "CAUSAL", "category": "microstructure"},

    # ── Research factors: signature path (CAUSAL) ────────────────────
    "SigL1_Return_20": {"type": "CAUSAL", "category": "path_signature"},
    "SigL1_Volume_20": {"type": "CAUSAL", "category": "path_signature"},
    "SigL2_XX_20": {"type": "CAUSAL", "category": "path_signature"},
    "SigL2_XY_20": {"type": "CAUSAL", "category": "path_signature"},
    "SigL2_YY_20": {"type": "CAUSAL", "category": "path_signature"},
    "SigLevyArea_20": {"type": "CAUSAL", "category": "path_signature"},

    # ── Research factors: vol surface (CAUSAL) ───────────────────────
    "VolSurf_Level": {"type": "CAUSAL", "category": "volatility"},
    "VolSurf_Slope": {"type": "CAUSAL", "category": "volatility"},
    "VolSurf_Curvature": {"type": "CAUSAL", "category": "volatility"},
    "VolSurf_PC1": {"type": "CAUSAL", "category": "volatility"},
    "VolSurf_PC2": {"type": "CAUSAL", "category": "volatility"},
    "VolSurf_PC3": {"type": "CAUSAL", "category": "volatility"},

    # ── Wave-flow decomposition (CAUSAL) ─────────────────────────────
    "flow_component": {"type": "CAUSAL", "category": "regime"},
    "wave_component": {"type": "CAUSAL", "category": "regime"},
    "dominant_frequency": {"type": "CAUSAL", "category": "regime"},
    "spectral_entropy": {"type": "CAUSAL", "category": "regime"},
    "wave_amplitude": {"type": "CAUSAL", "category": "regime"},
    "flow_strength": {"type": "CAUSAL", "category": "regime"},
    "wave_flow_ratio": {"type": "CAUSAL", "category": "regime"},
    "wave_regime_indicator": {"type": "CAUSAL", "category": "regime"},

    # ── Option surface factors (CAUSAL — rolling lookback) ───────────
    "iv_atm_30": {"type": "CAUSAL", "category": "options"},
    "iv_atm_60": {"type": "CAUSAL", "category": "options"},
    "iv_atm_90": {"type": "CAUSAL", "category": "options"},
    "term_slope_30_90": {"type": "CAUSAL", "category": "options"},
    "skew_25d": {"type": "CAUSAL", "category": "options"},
    "curvature": {"type": "CAUSAL", "category": "options"},
    "vrp_10": {"type": "CAUSAL", "category": "options"},
    "vrp_30": {"type": "CAUSAL", "category": "options"},
    "vrp_60": {"type": "CAUSAL", "category": "options"},
    "iv_rank_1y": {"type": "CAUSAL", "category": "options"},
    "skew_rank_1y": {"type": "CAUSAL", "category": "options"},
    "vrp_rank_1y": {"type": "CAUSAL", "category": "options"},

    # ── Cross-sectional z-scored momentum (CAUSAL within universe) ───
    "cs_zmom_10": {"type": "CAUSAL", "category": "cross_sectional"},
    "cs_zmom_20": {"type": "CAUSAL", "category": "cross_sectional"},
    "cs_zmom_60": {"type": "CAUSAL", "category": "cross_sectional"},

    # ── Cross-asset network factors (CAUSAL — lag-shifted) ───────────
    "NetMom_Spillover": {"type": "CAUSAL", "category": "cross_asset"},
    "NetMom_LeadCentrality": {"type": "CAUSAL", "category": "cross_asset"},
    "NetMom_RecvCentrality": {"type": "CAUSAL", "category": "cross_asset"},
    "NetMom_GraphDensity": {"type": "CAUSAL", "category": "cross_asset"},
    "VolSpillover_In": {"type": "CAUSAL", "category": "cross_asset"},
    "VolSpillover_Out": {"type": "CAUSAL", "category": "cross_asset"},
    "VolSpillover_Net": {"type": "CAUSAL", "category": "cross_asset"},

    # ── Path signature features (CAUSAL) ─────────────────────────────
    "sig_5d_0": {"type": "CAUSAL", "category": "path_signature"},
    "sig_5d_1": {"type": "CAUSAL", "category": "path_signature"},
    "sig_5d_2": {"type": "CAUSAL", "category": "path_signature"},
    "sig_5d_3": {"type": "CAUSAL", "category": "path_signature"},
    "sig_5d_4": {"type": "CAUSAL", "category": "path_signature"},
    "sig_5d_5": {"type": "CAUSAL", "category": "path_signature"},
    "sig_20d_0": {"type": "CAUSAL", "category": "path_signature"},
    "sig_20d_1": {"type": "CAUSAL", "category": "path_signature"},
    "sig_20d_2": {"type": "CAUSAL", "category": "path_signature"},
    "sig_20d_3": {"type": "CAUSAL", "category": "path_signature"},
    "sig_20d_4": {"type": "CAUSAL", "category": "path_signature"},
    "sig_20d_5": {"type": "CAUSAL", "category": "path_signature"},
    "sig_60d_0": {"type": "CAUSAL", "category": "path_signature"},
    "sig_60d_1": {"type": "CAUSAL", "category": "path_signature"},
    "sig_60d_2": {"type": "CAUSAL", "category": "path_signature"},
    "sig_60d_3": {"type": "CAUSAL", "category": "path_signature"},
    "sig_60d_4": {"type": "CAUSAL", "category": "path_signature"},
    "sig_60d_5": {"type": "CAUSAL", "category": "path_signature"},

    # ── DTW lead-lag features (CAUSAL) ───────────────────────────────
    "dtw_leader_score": {"type": "CAUSAL", "category": "cross_asset"},
    "dtw_follower_score": {"type": "CAUSAL", "category": "cross_asset"},
    "dtw_avg_lag": {"type": "CAUSAL", "category": "cross_asset"},

    # ── Intraday features — END_OF_DAY (require full-day data) ───────
    "intraday_vol_ratio": {"type": "END_OF_DAY", "category": "microstructure"},
    "vwap_deviation": {
        "type": "END_OF_DAY", "category": "microstructure",
        "reason": "Uses full-day VWAP which is unavailable intraday",
    },
    "amihud_illiquidity": {"type": "END_OF_DAY", "category": "microstructure"},
    "kyle_lambda": {"type": "END_OF_DAY", "category": "microstructure"},
    "realized_vol_5m": {"type": "END_OF_DAY", "category": "microstructure"},
    "microstructure_noise": {"type": "END_OF_DAY", "category": "microstructure"},

    # ── Rolling VWAP features (CAUSAL — rolling window) ────────────
    "rolling_vwap_20": {"type": "CAUSAL", "category": "vwap"},
    "rolling_vwap_deviation_20": {"type": "CAUSAL", "category": "vwap"},

    # ── LOB features — END_OF_DAY ────────────────────────────────────
    "trade_arrival_rate": {"type": "END_OF_DAY", "category": "microstructure"},
    "duration_between_trades_mean": {"type": "END_OF_DAY", "category": "microstructure"},
    "duration_between_trades_std": {"type": "END_OF_DAY", "category": "microstructure"},
    "quote_update_intensity": {"type": "END_OF_DAY", "category": "microstructure"},
    "price_impact_asymmetry": {"type": "END_OF_DAY", "category": "microstructure"},
    "queue_imbalance": {"type": "END_OF_DAY", "category": "microstructure"},
    "fill_probability_proxy": {"type": "END_OF_DAY", "category": "microstructure"},
}


def get_feature_type(feature_name: str) -> str:
    """Return the causality type for a feature.

    Returns ``'CAUSAL'`` for any feature not explicitly registered, since
    indicator-based features are backward-looking by construction.
    """
    entry = FEATURE_METADATA.get(feature_name)
    if entry is not None:
        return entry["type"]
    # Interaction features (prefixed X_) inherit CAUSAL from their inputs
    if feature_name.startswith("X_"):
        return "CAUSAL"
    # Default: assume causal (indicators are backward-looking)
    return "CAUSAL"


def _filter_causal_features(features: "pd.DataFrame") -> "pd.DataFrame":
    """Keep only features with type CAUSAL or END_OF_DAY (drop RESEARCH_ONLY)."""
    keep = [c for c in features.columns if get_feature_type(c) != "RESEARCH_ONLY"]
    return features[keep]


def _build_indicator_set() -> list:
    """Instantiate all indicators with default parameters."""
    return [
        # ── Volatility ─────────────────────────────
        ATR(14), NATR(14),
        BollingerBandWidth(20), HistoricalVolatility(20),
        BBWidthPercentile(20, 100), NATRPercentile(14, 100),
        VolatilitySqueeze(20),
        ParkinsonVolatility(20), GarmanKlassVolatility(20),
        YangZhangVolatility(20),
        VolatilityCone(20, 252), VolOfVol(20, 60),
        GARCHVolatility(252), VolTermStructure(10, 60),

        # ── Momentum ───────────────────────────────
        RSI(14), RSI(5),
        MACD(12, 26), MACDSignal(12, 26, 9), MACDHistogram(12, 26, 9),
        ROC(10), ROC(20), ROC(50),
        Stochastic(14), StochasticD(14),
        WilliamsR(14), CCI(20),

        # ── Trend ──────────────────────────────────
        SMA(20), SMA(50), SMA(200),
        EMA(8), EMA(21), EMA(50),
        PriceVsSMA(20), PriceVsSMA(50), PriceVsSMA(200),
        SMASlope(20), SMASlope(50),
        ADX(14), Aroon(14),
        EMAAlignment(), TrendStrength(14),
        PriceVsEMAStack(),
        MarketRegime(50), VolatilityRegime(20),

        # ── Volume ─────────────────────────────────
        VolumeRatio(20), OBV(), OBVSlope(14), MFI(14),
        RVOL(20), NetVolumeTrend(14),
        VolumeForce(13), AccumulationDistribution(10),

        # ── Price Action ───────────────────────────
        HigherHighs(5), LowerLows(5),
        CandleBody(), CandleDirection(), GapPercent(),
        DistanceFromHigh(252), DistanceFromLow(252),
        PricePercentile(252),

        # ── Pivot / Breakout ───────────────────────
        PivotHigh(5, 5), PivotLow(5, 5),
        NBarHighBreak(5), NBarLowBreak(5),
        RangeBreakout(20),

        # ── ATR Risk ───────────────────────────────
        ATRTrailingStop(14, 2.0), ATRChannel(14),
        RiskPerATR(14, 20),

        # ── VWAP ───────────────────────────────────
        VWAP(20), PriceVsVWAP(20), VWAPBands(20),

        # ── Value Area ─────────────────────────────
        ValueAreaHigh(20), ValueAreaLow(20), POC(20),
        PriceVsPOC(20), ValueAreaPosition(20),
        AboveValueArea(20), BelowValueArea(20),

        # ── Statistical ────────────────────────────
        HurstExponent(100), MeanReversionHalfLife(60),
        ZScore(20), ZScore(50),
        VarianceRatio(100, 5), Autocorrelation(20, 1),
        KalmanTrend(0.01, 1.0),

        # ── Information Theory ─────────────────────
        ShannonEntropy(20), ApproximateEntropy(50),

        # ── Microstructure ─────────────────────────
        AmihudIlliquidity(20), KyleLambda(20), RollSpread(20),

        # ── Fractal / Spectral ─────────────────────
        FractalDimension(100), DFA(100), DominantCycle(100),

        # ── Distribution ───────────────────────────
        ReturnSkewness(60), ReturnKurtosis(60),

        # ── Regime Detection ───────────────────────
        CUSUMDetector(50), RegimePersistence(20),
    ]


def _build_minimal_indicator_set() -> list:
    """Lean indicator set for the 'minimal' feature mode.

    Designed to reduce complexity and overfitting risk by keeping only
    the highest-value, lowest-redundancy features across key categories:
    trend, momentum, volatility, volume, mean-reversion, and
    microstructure.  ~20 indicators vs ~80+ in the full set.
    """
    return [
        # Volatility (3) — one realized, one range-based, one term-structure
        NATR(14),
        GarmanKlassVolatility(20),
        VolTermStructure(10, 60),

        # Momentum (4) — RSI, MACD, ROC at two horizons
        RSI(14),
        MACD(12, 26),
        ROC(10),
        ROC(20),

        # Trend (3) — price vs SMA, ADX, slope
        PriceVsSMA(50),
        ADX(14),
        SMASlope(50),

        # Volume (2) — relative volume, Amihud illiquidity
        RVOL(20),
        AmihudIlliquidity(20),

        # Mean-reversion (3) — z-score, Hurst, autocorrelation
        ZScore(20),
        HurstExponent(100),
        Autocorrelation(20, 1),

        # Distribution (2) — skewness and kurtosis of returns
        ReturnSkewness(60),
        ReturnKurtosis(60),

        # Price position (1) — distance from high
        DistanceFromHigh(252),
    ]


# Module-level singletons — built once
_INDICATORS = None
_MINIMAL_INDICATORS = None


def _get_indicators(minimal: bool = False):
    """Return the indicator set (cached at module level)."""
    if minimal:
        global _MINIMAL_INDICATORS
        if _MINIMAL_INDICATORS is None:
            _MINIMAL_INDICATORS = _build_minimal_indicator_set()
        return _MINIMAL_INDICATORS
    global _INDICATORS
    if _INDICATORS is None:
        _INDICATORS = _build_indicator_set()
    return _INDICATORS


def compute_indicator_features(
    df: pd.DataFrame,
    verbose: bool = False,
    minimal: bool = False,
) -> pd.DataFrame:
    """
    Compute indicator-based features as continuous columns.

    Args:
        df: OHLCV DataFrame with DatetimeIndex
        verbose: print progress
        minimal: if True, use the lean ~20-indicator set

    Returns:
        DataFrame with one column per indicator (same index as input)
    """
    indicators = _get_indicators(minimal=minimal)
    features = pd.DataFrame(index=df.index)

    for ind in indicators:
        try:
            result = ind.calculate(df)
            features[ind.name] = result
        except (ValueError, RuntimeError) as e:
            if verbose:
                print(f"  Warning: {ind.name} failed: {e}")
            features[ind.name] = np.nan

    return features


def compute_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute raw OHLCV-derived features (returns, volume, gaps, etc.)."""
    features = pd.DataFrame(index=df.index)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Returns at multiple horizons
    for n in [1, 2, 3, 5, 10, 20]:
        features[f"return_{n}d"] = close.pct_change(n)

    # Log returns (for stationarity)
    features["log_return_1d"] = np.log(close / close.shift(1))

    # Intraday range
    features["intraday_range"] = (high - low) / close
    features["overnight_gap"] = df["Open"] / close.shift(1) - 1

    # Volume features
    features["log_volume"] = np.log1p(volume)
    features["dollar_volume"] = np.log1p(close * volume)
    features["volume_change"] = volume.pct_change()

    # Rolling return statistics
    ret = close.pct_change()
    features["return_vol_5d"] = ret.rolling(5).std()
    features["return_vol_20d"] = ret.rolling(20).std()
    features["return_vol_60d"] = ret.rolling(60).std()

    # Price position within recent range
    features["price_in_range_20d"] = (close - low.rolling(20).min()) / (
        high.rolling(20).max() - low.rolling(20).min() + 1e-10
    )
    features["price_in_range_60d"] = (close - low.rolling(60).min()) / (
        high.rolling(60).max() - low.rolling(60).min() + 1e-10
    )

    return features


def compute_har_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute HAR (Heterogeneous Autoregressive) realized volatility features.

    The HAR model decomposes volatility into daily, weekly, and monthly
    components, capturing the well-documented heterogeneous behaviour of
    market participants operating at different time horizons.

    Features:
        RV_daily   — 5-day rolling realized volatility (daily returns)
        RV_weekly  — 5-day rolling window realized vol (same as daily here,
                     but kept separate for HAR nomenclature clarity)
        RV_monthly — 22-day rolling window realized volatility
        HAR_composite   — 0.5 * RV_daily + 0.3 * RV_weekly + 0.2 * RV_monthly
        HAR_ratio_dw    — RV_daily / RV_weekly  (short vs medium-term vol)
        HAR_ratio_wm    — RV_weekly / RV_monthly (medium vs long-term vol)
    """
    close = df["Close"]
    returns = close.pct_change()

    # Realized volatility at three HAR horizons
    rv_daily = returns.rolling(5).std()
    rv_weekly = returns.rolling(21).std()
    rv_monthly = returns.rolling(22).std()

    # Composite HAR measure (standard Corsi 2009 weighting)
    har_composite = 0.5 * rv_daily + 0.3 * rv_weekly + 0.2 * rv_monthly

    # Volatility term-structure ratios
    har_ratio_dw = rv_daily / rv_weekly.replace(0, np.nan)
    har_ratio_wm = rv_weekly / rv_monthly.replace(0, np.nan)

    features = pd.DataFrame({
        "RV_daily": rv_daily,
        "RV_weekly": rv_weekly,
        "RV_monthly": rv_monthly,
        "HAR_composite": har_composite,
        "HAR_ratio_dw": har_ratio_dw,
        "HAR_ratio_wm": har_ratio_wm,
    }, index=df.index)

    return features


def compute_multiscale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum, RSI, and volatility features at multiple time scales.

    These multi-scale windows are aligned with the 10-day prediction horizon
    and capture short-, medium-, and long-term dynamics.

    Features:
        RSI_5, RSI_10, RSI_20, RSI_50  — Relative Strength Index at 4 scales
        MOM_5, MOM_10, MOM_20, MOM_60  — Price momentum (close / close.shift(n) - 1)
        VOL_5, VOL_10, VOL_20, VOL_60  — Rolling return volatility (std)
    """
    features = pd.DataFrame(index=df.index)
    close = df["Close"]
    returns = close.pct_change()

    # ── RSI at multiple windows ──────────────────────────────────
    for window in [5, 10, 20, 50]:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window, min_periods=window).mean()
        avg_loss = loss.rolling(window, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        features[f"RSI_{window}"] = 100.0 - (100.0 / (1.0 + rs))

    # ── Momentum at multiple horizons ────────────────────────────
    for n in [5, 10, 20, 60]:
        features[f"MOM_{n}"] = close / close.shift(n) - 1

    # ── Rolling volatility at multiple scales ────────────────────
    for n in [5, 10, 20, 60]:
        features[f"VOL_{n}"] = returns.rolling(n).std()

    return features


def compute_interaction_features(
    features: pd.DataFrame,
    pairs: Optional[list] = None,
) -> pd.DataFrame:
    """
    Generate interaction features from pairs of continuous indicators.

    Operations:
        multiply: A * B (captures regime-conditional effects)
        ratio: A / B (relative magnitude)
        center_50: A - 50 (for 0-100 oscillators)
    """
    from ..config import INTERACTION_PAIRS

    if pairs is None:
        pairs = INTERACTION_PAIRS

    interactions = pd.DataFrame(index=features.index)

    for entry in pairs:
        feat_a = entry[0]
        feat_b = entry[1]
        op = entry[2]

        if feat_a not in features.columns:
            continue

        if op == "multiply" and feat_b in features.columns:
            col_name = f"X_{feat_a}_x_{feat_b}"
            a = features[feat_a]
            b = features[feat_b]
            # Standardize before multiplying to keep scale manageable
            # Let NaN propagate naturally; min_periods handles warm-up
            a_std = (a - a.rolling(252, min_periods=60).mean()) / (
                a.rolling(252, min_periods=60).std() + 1e-10
            )
            b_std = (b - b.rolling(252, min_periods=60).mean()) / (
                b.rolling(252, min_periods=60).std() + 1e-10
            )
            interactions[col_name] = a_std * b_std

        elif op == "ratio" and feat_b in features.columns:
            col_name = f"X_{feat_a}_div_{feat_b}"
            b = features[feat_b].replace(0, np.nan)
            interactions[col_name] = features[feat_a] / b

        elif op == "center_50":
            col_name = f"X_{feat_a}_centered"
            interactions[col_name] = features[feat_a] - 50

    return interactions


def compute_targets(
    df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    benchmark_close: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compute forward return targets for supervised learning.

    When ``benchmark_close`` is provided the targets are *excess* returns
    (stock return minus benchmark return over the same horizon).  This
    removes market beta from the prediction target so the model learns
    stock-specific alpha rather than market direction.
    """
    from ..config import FORWARD_HORIZONS

    if horizons is None:
        horizons = FORWARD_HORIZONS

    targets = pd.DataFrame(index=df.index)
    close = df["Close"]
    ret_stream: Optional[pd.Series] = None
    if "total_ret" in df.columns:
        ret_stream = pd.to_numeric(df["total_ret"], errors="coerce")
    elif "Return" in df.columns:
        ret_stream = pd.to_numeric(df["Return"], errors="coerce")

    # Pre-compute benchmark forward returns if provided.
    bench_fwd: Dict[int, pd.Series] = {}
    if benchmark_close is not None:
        bench_aligned = benchmark_close.reindex(df.index).ffill()
        for h in horizons:
            bench_fwd[h] = bench_aligned.shift(-h) / bench_aligned - 1

    for h in horizons:
        close_based = close.shift(-h) / close - 1
        if ret_stream is None:
            raw_target = close_based
        else:
            gross = (1.0 + ret_stream).shift(-1)
            ret_based = gross.rolling(window=h, min_periods=h).apply(np.prod, raw=True).shift(-(h - 1)) - 1.0
            raw_target = ret_based.where(ret_based.notna(), close_based)

        if h in bench_fwd:
            targets[f"target_{h}d"] = raw_target - bench_fwd[h]
        else:
            targets[f"target_{h}d"] = raw_target

    return targets


def _winsorize_expanding(df: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    """
    Winsorize features using expanding-window quantiles (no look-ahead).

    For each column, clips values to [1st, 99th] percentile computed from
    all data up to and including the current row.
    """
    # Deduplicate columns before winsorizing to avoid pandas errors
    # when setting values on DataFrames with non-unique column names.
    deduped = df.loc[:, ~df.columns.duplicated()]
    result = deduped.copy()
    for col in deduped.columns:
        series = deduped[col].dropna()
        if len(series) < 100:
            continue
        # Use expanding window for causal quantiles
        lo = series.expanding(min_periods=100).quantile(lower_q)
        hi = series.expanding(min_periods=100).quantile(upper_q)
        # Reindex to original index and clip
        lo = lo.reindex(deduped.index).ffill()
        hi = hi.reindex(deduped.index).ffill()
        result[col] = deduped[col].clip(lower=lo, upper=hi)
    return result


class FeaturePipeline:
    """
    End-to-end feature computation pipeline.

    Usage:
        pipeline = FeaturePipeline()
        features, targets = pipeline.compute(ohlcv_df)
    """

    def __init__(
        self,
        feature_mode: str = "full",
        include_interactions: bool = True,
        include_research_factors: bool = True,
        include_cross_asset_factors: bool = True,
        include_options_factors: bool = True,
        research_config: Optional[ResearchFactorConfig] = None,
        verbose: bool = False,
        production_mode: bool = False,
    ):
        """Initialize FeaturePipeline.

        Args:
            production_mode: When True, automatically filter out features
                tagged as RESEARCH_ONLY in FEATURE_METADATA.  Defaults to
                False for backward compatibility.
        """
        mode = str(feature_mode).lower()
        self.feature_mode = mode
        self.production_mode = production_mode
        self.minimal_indicators = False
        if mode == "minimal":
            # Lean profile (~20 indicators, no extras) — minimises
            # overfitting risk and forces signal to come from a small,
            # interpretable feature set.
            self.minimal_indicators = True
            self.include_interactions = False
            self.include_research_factors = False
            self.include_cross_asset_factors = False
            self.include_options_factors = False
        elif mode == "core":
            # Reduced-complexity profile for robust baseline research.
            self.include_interactions = False
            self.include_research_factors = False
            self.include_cross_asset_factors = False
            self.include_options_factors = False
        else:
            self.include_interactions = include_interactions
            self.include_research_factors = include_research_factors
            self.include_cross_asset_factors = include_cross_asset_factors
            self.include_options_factors = include_options_factors
        self.research_config = research_config or ResearchFactorConfig()
        self.verbose = verbose

    def compute(
        self,
        df: pd.DataFrame,
        compute_targets_flag: bool = True,
        benchmark_close: Optional[pd.Series] = None,
    ) -> tuple:
        """
        Compute all features and targets from OHLCV data.

        Args:
            df: OHLCV DataFrame for a single ticker.
            compute_targets_flag: Whether to compute targets.
            benchmark_close: SPY (or other benchmark) close prices for excess
                return targets.  Ignored when compute_targets_flag is False.

        Returns:
            (features_df, targets_df) — both aligned to same index
            targets_df is None if compute_targets_flag=False
        """
        use_minimal = getattr(self, "minimal_indicators", False)
        if self.verbose:
            n_ind = len(_get_indicators(minimal=use_minimal))
            label = "minimal" if use_minimal else "full"
            print(f"  Computing {n_ind} indicator features ({label})...")
        indicator_feats = compute_indicator_features(
            df, verbose=self.verbose, minimal=use_minimal,
        )

        if self.verbose:
            print(f"  Computing raw OHLCV features...")
        raw_feats = compute_raw_features(df)

        # Combine
        features = pd.concat([indicator_feats, raw_feats], axis=1)

        # HAR volatility and multi-scale features (research-grade)
        if self.include_research_factors:
            if self.verbose:
                print("  Computing HAR volatility features...")
            har_feats = compute_har_volatility_features(df)
            features = pd.concat([features, har_feats], axis=1)

            if self.verbose:
                print("  Computing multi-scale features...")
            multiscale_feats = compute_multiscale_features(df)
            features = pd.concat([features, multiscale_feats], axis=1)

        # Option surface / VRP factors (if OptionMetrics-enriched columns exist).
        if self.include_options_factors:
            option_feats = compute_option_surface_factors(df)
            if not option_feats.empty:
                features = pd.concat([features, option_feats], axis=1)

        # Research-derived single-asset features
        if self.include_research_factors:
            if self.verbose:
                print("  Computing research factors (single-asset)...")
            research_feats = compute_single_asset_research_factors(
                df=df,
                config=self.research_config,
            )
            features = pd.concat([features, research_feats], axis=1)

            if self.verbose:
                print("  Computing wave-flow decomposition...")
            wave_flow_feats = compute_wave_flow_decomposition(df)
            features = pd.concat([features, wave_flow_feats], axis=1)

        # Replace infinities with NaN (ratio features, log(0), etc.)
        features = features.replace([np.inf, -np.inf], np.nan)

        # Interaction features
        if self.include_interactions:
            if self.verbose:
                print(f"  Computing interaction features...")
            interaction_feats = compute_interaction_features(features)
            features = pd.concat([features, interaction_feats], axis=1)

        # Replace any infinities introduced by interactions
        features = features.replace([np.inf, -np.inf], np.nan)

        # Deduplicate columns (e.g. RSI_5 from both indicator set and
        # multiscale features) — keep the first occurrence.
        if features.columns.duplicated().any():
            features = features.loc[:, ~features.columns.duplicated()]

        # Winsorize: clip each feature to [1st, 99th] percentile
        # Uses expanding window to avoid look-ahead bias
        features = _winsorize_expanding(features)

        # Targets
        targets = None
        if compute_targets_flag:
            targets = compute_targets(df, benchmark_close=benchmark_close)

        # Production mode: filter out RESEARCH_ONLY features
        if self.production_mode:
            pre_count = features.shape[1]
            features = _filter_causal_features(features)
            if self.verbose:
                dropped = pre_count - features.shape[1]
                print(f"  Production mode: dropped {dropped} research-only features")

        if self.verbose:
            print(f"  Total features: {features.shape[1]}")
            valid_rows = features.dropna(how="all").shape[0]
            print(f"  Valid rows: {valid_rows}/{len(features)}")

        return features, targets

    def compute_universe(
        self,
        data: Dict[str, pd.DataFrame],
        verbose: bool = True,
        compute_targets_flag: bool = True,
        benchmark_close: Optional[pd.Series] = None,
    ) -> tuple:
        """
        Compute features for all symbols, stacking into a single DataFrame.

        Args:
            data: {permno: OHLCV DataFrame} mapping.
            verbose: Print progress.
            compute_targets_flag: Whether to compute targets.
            benchmark_close: Benchmark (e.g. SPY) close prices for excess
                return targets.  When None, the pipeline loads SPY
                automatically via ``BENCHMARK`` config.

        Returns:
            (features_df, targets_df) — MultiIndex (permno, date)
        """
        # Load benchmark for excess-return targets when not supplied.
        if benchmark_close is None and compute_targets_flag:
            benchmark_close = self._load_benchmark_close(verbose=verbose)

        all_features = []
        all_targets = []

        for i, (permno, df) in enumerate(data.items()):
            if verbose:
                print(f"  [{i+1}/{len(data)}] {permno}...", end="", flush=True)

            feats, tgts = self.compute(
                df,
                compute_targets_flag=compute_targets_flag,
                benchmark_close=benchmark_close,
            )

            # Add PERMNO level to index
            feat_index_name = feats.index.name or "date"
            feats = feats.copy()
            feats.index = pd.MultiIndex.from_arrays(
                [
                    np.full(len(feats), str(permno), dtype=object),
                    feats.index,
                ],
                names=["permno", feat_index_name],
            )
            all_features.append(feats)
            if tgts is not None:
                tgt_index_name = tgts.index.name or "date"
                tgts = tgts.copy()
                tgts.index = pd.MultiIndex.from_arrays(
                    [
                        np.full(len(tgts), str(permno), dtype=object),
                        tgts.index,
                    ],
                    names=["permno", tgt_index_name],
                )
                all_targets.append(tgts)

            if verbose:
                valid = feats.dropna(how="all").shape[0]
                print(f" {valid} valid rows")

        features = pd.concat(all_features).sort_index()
        targets = pd.concat(all_targets).sort_index() if all_targets else None

        # Cross-asset research factors (lead-lag network and vol spillovers).
        if self.include_research_factors and self.include_cross_asset_factors:
            if verbose:
                print("  Computing research factors (cross-asset network)...")
            cross_asset = compute_cross_asset_research_factors(
                price_data=data,
                config=self.research_config,
            )
            if not cross_asset.empty:
                features = features.join(cross_asset, how="left")

        # HARX volatility spillover features (cross-asset)
        if self.include_research_factors and self.include_cross_asset_factors:
            if verbose:
                print("  Computing HARX spillover features...")
            try:
                from .harx_spillovers import compute_harx_spillovers
                # Build returns dict from price data
                returns_by_asset = {}
                for permno, df in data.items():
                    if "Close" in df.columns:
                        returns_by_asset[str(permno)] = df["Close"].pct_change().dropna()
                if len(returns_by_asset) >= 2:
                    harx_feats = compute_harx_spillovers(returns_by_asset)
                    if not harx_feats.empty:
                        features = features.join(harx_feats, how="left")
            except (ImportError, ValueError, RuntimeError) as e:
                if verbose:
                    print(f"  WARNING: HARX spillovers failed: {e}")

        # Correlation regime features (cross-asset)
        if self.include_research_factors and self.include_cross_asset_factors:
            if verbose:
                print("  Computing correlation regime features...")
            try:
                from ..regime.correlation import CorrelationRegimeDetector
                # Build returns dict from price data
                corr_returns = {}
                for permno, df in data.items():
                    if "Close" in df.columns:
                        corr_returns[str(permno)] = df["Close"].pct_change().dropna()
                if len(corr_returns) >= 2:
                    detector = CorrelationRegimeDetector()
                    corr_feats = detector.get_correlation_features(corr_returns)
                    if not corr_feats.empty:
                        # Correlation features are date-level — broadcast to
                        # all stocks via a join on the date level.
                        features = features.join(
                            corr_feats,
                            on=features.index.names[-1],
                            how="left",
                        )
            except (ImportError, ValueError, RuntimeError) as e:
                if verbose:
                    print(f"  WARNING: Correlation regime features failed: {e}")

        # Macro features (per-date, broadcast to all stocks)
        if self.include_research_factors:
            if verbose:
                print("  Computing macro features...")
            try:
                from .macro import MacroFeatureProvider
                macro_provider = MacroFeatureProvider()
                # Get date range from features index
                dates = features.index.get_level_values(-1)
                start_date = dates.min()
                end_date = dates.max()
                macro_df = macro_provider.get_macro_features(
                    start_date=str(start_date.date()) if hasattr(start_date, 'date') else str(start_date),
                    end_date=str(end_date.date()) if hasattr(end_date, 'date') else str(end_date),
                )
                if macro_df is not None and not macro_df.empty:
                    # Macro is date-level only — join on the date level of the MultiIndex
                    features = features.join(macro_df, on=features.index.names[-1], how="left")
            except (ImportError, ValueError, RuntimeError) as e:
                if verbose:
                    print(f"  WARNING: Macro features failed: {e}")

        # Intraday + LOB features (optional)
        # Strategy: try WRDS TAQmsec first, fall back to local IBKR cache
        if self.include_research_factors:
            intraday_rows = []
            lob_rows = []
            wrds_available = False
            try:
                from ..data.wrds_provider import WRDSProvider
                wrds = WRDSProvider()
                wrds_available = wrds.available()
            except ImportError:
                wrds = None

            if wrds_available:
                # Path 1: WRDS TAQmsec (original behavior, 1-min bars)
                try:
                    if verbose:
                        print("  Computing intraday + LOB features (TAQmsec)...")
                    from .intraday import compute_intraday_features
                    from .lob_features import compute_lob_features
                    for permno, df in data.items():
                        ticker = df.attrs.get("ticker", str(permno))
                        for date in df.index[-60:]:
                            date_str = str(date.date()) if hasattr(date, 'date') else str(date)
                            try:
                                intra = compute_intraday_features(ticker, date_str, wrds)
                                intra["permno"] = str(permno)
                                intra["date"] = date
                                intraday_rows.append(intra)
                            except (ValueError, KeyError, TypeError):
                                pass
                            try:
                                bars = wrds.get_taqmsec_ohlcv(
                                    ticker=ticker,
                                    timeframe="5m",
                                    start_date=date_str,
                                    end_date=date_str,
                                )
                                if bars is not None and not bars.empty:
                                    lob = compute_lob_features(bars)
                                    lob["permno"] = str(permno)
                                    lob["date"] = date
                                    lob_rows.append(lob)
                            except (ValueError, KeyError, TypeError):
                                pass
                except (ImportError, ValueError, RuntimeError) as e:
                    if verbose:
                        print(f"  WARNING: WRDS intraday/LOB features failed: {e}")

            if not wrds_available or (not intraday_rows and not lob_rows):
                # Path 2: Local IBKR cache fallback (LOB features only)
                try:
                    from ..data.local_cache import load_intraday_ohlcv
                    from .lob_features import compute_lob_features
                    from ..config import INTRADAY_MIN_BARS
                    if verbose:
                        print("  Computing LOB features from local IBKR cache...")
                    for permno, df in data.items():
                        ticker = df.attrs.get("ticker", str(permno))
                        # Try 30m bars first, then 1h
                        intra_bars = load_intraday_ohlcv(ticker, "30m")
                        intra_freq = "30min"
                        if intra_bars is None or len(intra_bars) < INTRADAY_MIN_BARS:
                            intra_bars = load_intraday_ohlcv(ticker, "1h")
                            intra_freq = "1h"
                        if intra_bars is None or len(intra_bars) < INTRADAY_MIN_BARS:
                            continue
                        # For each of the last 60 trading days, slice that day's bars
                        for date in df.index[-60:]:
                            date_str = str(date.date()) if hasattr(date, 'date') else str(date)
                            try:
                                day_bars = intra_bars.loc[date_str]
                                if isinstance(day_bars, pd.Series):
                                    continue  # only one bar — not enough
                                if len(day_bars) >= 5:
                                    lob = compute_lob_features(day_bars, freq=intra_freq)
                                    lob["permno"] = str(permno)
                                    lob["date"] = date
                                    lob_rows.append(lob)
                            except (KeyError, TypeError):
                                pass
                except (ImportError, ValueError, RuntimeError) as e:
                    if verbose:
                        print(f"  WARNING: Local intraday/LOB features failed: {e}")

            # Join intraday and LOB features into main features DataFrame
            try:
                if intraday_rows:
                    intra_df = pd.DataFrame(intraday_rows).set_index(["permno", "date"])
                    features = features.join(intra_df, how="left")
                if lob_rows:
                    lob_df = pd.DataFrame(lob_rows).set_index(["permno", "date"])
                    features = features.join(lob_df, how="left")
            except (ValueError, KeyError, TypeError) as e:
                if verbose:
                    print(f"  WARNING: Intraday/LOB join failed: {e}")

        # Cross-sectional z-scored momentum: for each date, z-score the
        # raw momentum across all stocks in the universe.  This produces a
        # truly relative signal that adjusts for market-wide momentum.
        if self.include_research_factors and len(data) > 1:
            if verbose:
                print("  Computing cross-sectional z-scored momentum...")
            date_level = features.index.names[-1]
            for h in [10, 20, 60]:
                raw_col = f"relative_mom_{h}"
                cs_col = f"cs_zmom_{h}"
                if raw_col in features.columns:
                    grouped = features[raw_col].groupby(level=date_level)
                    cs_mean = grouped.transform("mean")
                    cs_std = grouped.transform("std")
                    features[cs_col] = (features[raw_col] - cs_mean) / (cs_std + 1e-12)

        if verbose:
            print(f"\n  Universe total: {features.shape[0]} rows × {features.shape[1]} features")

        return features, targets

    @staticmethod
    def _load_benchmark_close(verbose: bool = False) -> Optional[pd.Series]:
        """Load benchmark close prices for excess-return target computation."""
        from ..config import BENCHMARK, LOOKBACK_YEARS
        from ..data.loader import load_ohlcv

        if verbose:
            print(f"  Loading benchmark ({BENCHMARK}) for excess-return targets...")
        try:
            bench_df = load_ohlcv(BENCHMARK, years=LOOKBACK_YEARS, use_cache=True)
            if bench_df is not None and "Close" in bench_df.columns:
                if verbose:
                    print(f"  Benchmark loaded: {len(bench_df)} bars")
                return bench_df["Close"]
        except (OSError, ValueError, RuntimeError) as exc:
            if verbose:
                print(f"  WARNING: Could not load benchmark {BENCHMARK}: {exc}")
        return None
