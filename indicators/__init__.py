"""
Quant Engine Indicators — self-contained copy of the technical indicator library.

All indicator classes from the original strategy_scanner are available here
so quant_engine has no external sys.path dependencies.
"""
from .indicators import (
    # Base
    Indicator,
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
    VolumeRatio, OBV, OBVSlope, MFI,
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
    AnchoredVWAP, PriceVsAnchoredVWAP, MultiVWAPPosition,
    # Value Area
    ValueAreaHigh, ValueAreaLow, POC,
    PriceVsPOC, ValueAreaPosition, AboveValueArea, BelowValueArea,
    # Beast666
    Beast666Proximity, Beast666Distance,
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
    # Registry
    get_all_indicators,
)

# The following indicators are available for external consumers and
# research notebooks but are NOT included in the production feature
# pipeline. To add them, instantiate in pipeline.py's indicator list.
__RESEARCH_INDICATORS__ = [
    "AnchoredVWAP", "PriceVsAnchoredVWAP", "MultiVWAPPosition",
    "Beast666Proximity", "Beast666Distance",
]

__all__ = [
    "Indicator",
    "ATR", "NATR", "BollingerBandWidth", "HistoricalVolatility",
    "BBWidthPercentile", "NATRPercentile", "VolatilitySqueeze",
    "RSI", "MACD", "MACDSignal", "MACDHistogram", "ROC",
    "Stochastic", "StochasticD", "WilliamsR", "CCI",
    "SMA", "EMA", "PriceVsSMA", "SMASlope", "ADX", "Aroon",
    "EMAAlignment", "TrendStrength", "PriceVsEMAStack",
    "MarketRegime", "VolatilityRegime",
    "VolumeRatio", "OBV", "OBVSlope", "MFI",
    "NetVolumeTrend", "VolumeForce", "AccumulationDistribution",
    "HigherHighs", "LowerLows", "CandleBody", "CandleDirection", "GapPercent",
    "DistanceFromHigh", "DistanceFromLow", "PricePercentile",
    "PivotHigh", "PivotLow", "NBarHighBreak", "NBarLowBreak", "RangeBreakout",
    "ATRTrailingStop", "ATRChannel", "RiskPerATR",
    "VWAP", "PriceVsVWAP", "VWAPBands",
    "AnchoredVWAP", "PriceVsAnchoredVWAP", "MultiVWAPPosition",
    "ValueAreaHigh", "ValueAreaLow", "POC",
    "PriceVsPOC", "ValueAreaPosition", "AboveValueArea", "BelowValueArea",
    "Beast666Proximity", "Beast666Distance",
    "ParkinsonVolatility", "GarmanKlassVolatility", "YangZhangVolatility",
    "VolatilityCone", "VolOfVol", "GARCHVolatility", "VolTermStructure",
    "HurstExponent", "MeanReversionHalfLife", "ZScore", "VarianceRatio",
    "Autocorrelation", "KalmanTrend",
    "ShannonEntropy", "ApproximateEntropy",
    "AmihudIlliquidity", "KyleLambda", "RollSpread",
    "FractalDimension", "DFA", "DominantCycle",
    "ReturnSkewness", "ReturnKurtosis",
    "CUSUMDetector", "RegimePersistence",
    "get_all_indicators",
]
