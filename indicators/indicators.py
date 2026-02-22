"""
Technical Indicator Library

A comprehensive library of technical indicators that can be combined
into trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class Indicator(ABC):
    """Base class for all indicators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the indicator's output column name."""
        pass

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Calculate indicator values. Returns a Series."""
        pass


# =============================================================================
# VOLATILITY INDICATORS
# =============================================================================

class ATR(Indicator):
    """Average True Range - measures volatility."""

    def __init__(self, period: int = 14):
        """Initialize ATR."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"ATR_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        high = df['High']
        low = df['Low']
        close = df['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()
        return atr


class NATR(Indicator):
    """Normalized ATR - ATR as percentage of close price."""

    def __init__(self, period: int = 14):
        """Initialize NATR."""
        self.period = period
        self._atr = ATR(period)

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"NATR_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        atr = self._atr.calculate(df)
        natr = (atr / df['Close']) * 100
        return natr


class BollingerBandWidth(Indicator):
    """Bollinger Band Width - measures volatility squeeze."""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """Initialize BollingerBandWidth."""
        self.period = period
        self.std_dev = std_dev

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"BBWidth_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        close = df['Close']
        sma = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()

        upper = sma + (self.std_dev * std)
        lower = sma - (self.std_dev * std)

        width = ((upper - lower) / sma) * 100
        return width


class HistoricalVolatility(Indicator):
    """Historical volatility (standard deviation of returns)."""

    def __init__(self, period: int = 20):
        """Initialize HistoricalVolatility."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"HV_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        returns = df['Close'].pct_change()
        hv = returns.rolling(window=self.period).std() * np.sqrt(252) * 100
        return hv


# =============================================================================
# MOMENTUM INDICATORS
# =============================================================================

class RSI(Indicator):
    """Relative Strength Index."""

    def __init__(self, period: int = 14):
        """Initialize RSI."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"RSI_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        delta = df['Close'].diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class MACD(Indicator):
    """MACD Line (difference between fast and slow EMA)."""

    def __init__(self, fast: int = 12, slow: int = 26):
        """Initialize MACD."""
        self.fast = fast
        self.slow = slow

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"MACD_{self.fast}_{self.slow}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        close = df['Close']
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        return ema_fast - ema_slow


class MACDSignal(Indicator):
    """MACD Signal Line."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """Initialize MACDSignal."""
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self._macd = MACD(fast, slow)

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"MACDSignal_{self.fast}_{self.slow}_{self.signal}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        macd = self._macd.calculate(df)
        signal = macd.ewm(span=self.signal, adjust=False).mean()
        return signal


class MACDHistogram(Indicator):
    """MACD Histogram (MACD - Signal)."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """Initialize MACDHistogram."""
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self._macd = MACD(fast, slow)
        self._signal = MACDSignal(fast, slow, signal)

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"MACDHist_{self.fast}_{self.slow}_{self.signal_period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        macd = self._macd.calculate(df)
        signal = self._signal.calculate(df)
        return macd - signal


class ROC(Indicator):
    """Rate of Change."""

    def __init__(self, period: int = 10):
        """Initialize ROC."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"ROC_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        close = df['Close']
        roc = ((close - close.shift(self.period)) / close.shift(self.period)) * 100
        return roc


class Stochastic(Indicator):
    """Stochastic %K."""

    def __init__(self, period: int = 14):
        """Initialize Stochastic."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"Stoch_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        low_min = df['Low'].rolling(window=self.period).min()
        high_max = df['High'].rolling(window=self.period).max()

        stoch = ((df['Close'] - low_min) / (high_max - low_min)) * 100
        return stoch


class StochasticD(Indicator):
    """Stochastic %D (smoothed %K)."""

    def __init__(self, k_period: int = 14, d_period: int = 3):
        """Initialize StochasticD."""
        self.k_period = k_period
        self.d_period = d_period
        self._stoch = Stochastic(k_period)

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"StochD_{self.k_period}_{self.d_period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        stoch_k = self._stoch.calculate(df)
        stoch_d = stoch_k.rolling(window=self.d_period).mean()
        return stoch_d


class WilliamsR(Indicator):
    """Williams %R."""

    def __init__(self, period: int = 14):
        """Initialize WilliamsR."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"WillR_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        high_max = df['High'].rolling(window=self.period).max()
        low_min = df['Low'].rolling(window=self.period).min()

        willr = ((high_max - df['Close']) / (high_max - low_min)) * -100
        return willr


class CCI(Indicator):
    """Commodity Channel Index."""

    def __init__(self, period: int = 20):
        """Initialize CCI."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"CCI_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma = tp.rolling(window=self.period).mean()
        mad = tp.rolling(window=self.period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True)

        cci = (tp - sma) / (0.015 * mad)
        return cci


# =============================================================================
# TREND INDICATORS
# =============================================================================

class SMA(Indicator):
    """Simple Moving Average."""

    def __init__(self, period: int = 20):
        """Initialize SMA."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"SMA_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        return df['Close'].rolling(window=self.period).mean()


class EMA(Indicator):
    """Exponential Moving Average."""

    def __init__(self, period: int = 20):
        """Initialize EMA."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"EMA_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        return df['Close'].ewm(span=self.period, adjust=False).mean()


class PriceVsSMA(Indicator):
    """Price distance from SMA (as percentage)."""

    def __init__(self, period: int = 50):
        """Initialize PriceVsSMA."""
        self.period = period
        self._sma = SMA(period)

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"PriceVsSMA_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        sma = self._sma.calculate(df)
        return ((df['Close'] - sma) / sma) * 100


class SMASlope(Indicator):
    """Slope of SMA (rate of change)."""

    def __init__(self, sma_period: int = 20, slope_period: int = 5):
        """Initialize SMASlope."""
        self.sma_period = sma_period
        self.slope_period = slope_period
        self._sma = SMA(sma_period)

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"SMASlope_{self.sma_period}_{self.slope_period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        sma = self._sma.calculate(df)
        slope = ((sma - sma.shift(self.slope_period)) / sma.shift(self.slope_period)) * 100
        return slope


class ADX(Indicator):
    """Average Directional Index - trend strength."""

    def __init__(self, period: int = 14):
        """Initialize ADX."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"ADX_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        high = df['High']
        low = df['Low']
        close = df['Close']

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=self.period).mean()
        plus_di = 100 * (plus_dm.rolling(window=self.period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=self.period).mean()
        return adx


class Aroon(Indicator):
    """Aroon Oscillator."""

    def __init__(self, period: int = 25):
        """Initialize Aroon."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"Aroon_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        period = self.period

        def aroon_up(x):
            return ((period - np.argmax(x)) / period) * 100

        def aroon_down(x):
            return ((period - np.argmin(x)) / period) * 100

        aroon_u = df['High'].rolling(window=period + 1).apply(aroon_up, raw=True)
        aroon_d = df['Low'].rolling(window=period + 1).apply(aroon_down, raw=True)

        return aroon_u - aroon_d


# =============================================================================
# VOLUME INDICATORS
# =============================================================================

class VolumeRatio(Indicator):
    """Current volume vs average volume."""

    def __init__(self, period: int = 20):
        """Initialize VolumeRatio."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"VolRatio_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        avg_vol = df['Volume'].rolling(window=self.period).mean()
        return df['Volume'] / avg_vol


class OBV(Indicator):
    """On-Balance Volume."""

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return "OBV"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        close = df['Close']
        volume = df['Volume']

        direction = np.where(close > close.shift(1), 1,
                            np.where(close < close.shift(1), -1, 0))

        obv = (volume * direction).cumsum()
        return pd.Series(obv, index=df.index)


class OBVSlope(Indicator):
    """OBV rate of change."""

    def __init__(self, period: int = 10):
        """Initialize OBVSlope."""
        self.period = period
        self._obv = OBV()

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"OBVSlope_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        obv = self._obv.calculate(df)
        obv_sma = obv.rolling(window=self.period).mean()
        slope = obv_sma.pct_change(self.period) * 100
        return slope


class MFI(Indicator):
    """Money Flow Index."""

    def __init__(self, period: int = 14):
        """Initialize MFI."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"MFI_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        mf = tp * df['Volume']

        positive_mf = mf.where(tp > tp.shift(1), 0)
        negative_mf = mf.where(tp < tp.shift(1), 0)

        mf_ratio = (positive_mf.rolling(window=self.period).sum() /
                   negative_mf.rolling(window=self.period).sum())

        mfi = 100 - (100 / (1 + mf_ratio))
        return mfi


# =============================================================================
# PRICE ACTION INDICATORS
# =============================================================================

class HigherHighs(Indicator):
    """Count of higher highs in lookback period."""

    def __init__(self, period: int = 10):
        """Initialize HigherHighs."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"HH_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        # Vectorized: count positive diffs within rolling window
        """Compute indicator values from the provided OHLCV dataframe."""
        higher = (df['High'].diff() > 0).astype(float)
        return higher.rolling(window=self.period - 1, min_periods=self.period - 1).sum()


class LowerLows(Indicator):
    """Count of lower lows in lookback period."""

    def __init__(self, period: int = 10):
        """Initialize LowerLows."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"LL_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        # Vectorized: count negative diffs within rolling window
        """Compute indicator values from the provided OHLCV dataframe."""
        lower = (df['Low'].diff() < 0).astype(float)
        return lower.rolling(window=self.period - 1, min_periods=self.period - 1).sum()


class CandleBody(Indicator):
    """Candle body size as percentage of range."""

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return "CandleBody"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        body = abs(df['Close'] - df['Open'])
        range_ = df['High'] - df['Low']
        return (body / range_) * 100


class CandleDirection(Indicator):
    """Candle direction streak (positive = up candles, negative = down)."""

    def __init__(self, period: int = 5):
        """Initialize CandleDirection."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"CandleDir_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        direction = np.where(df['Close'] > df['Open'], 1, -1)
        return pd.Series(direction, index=df.index).rolling(window=self.period).sum()


class GapPercent(Indicator):
    """Gap from previous close as percentage."""

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return "Gap"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        return ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)) * 100


# =============================================================================
# SUPPORT/RESISTANCE INDICATORS
# =============================================================================

class DistanceFromHigh(Indicator):
    """Distance from N-period high as percentage."""

    def __init__(self, period: int = 52):
        """Initialize DistanceFromHigh."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"DistHigh_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        high_max = df['High'].rolling(window=self.period).max()
        return ((df['Close'] - high_max) / high_max) * 100


class DistanceFromLow(Indicator):
    """Distance from N-period low as percentage."""

    def __init__(self, period: int = 52):
        """Initialize DistanceFromLow."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"DistLow_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        low_min = df['Low'].rolling(window=self.period).min()
        return ((df['Close'] - low_min) / low_min) * 100


class PricePercentile(Indicator):
    """Current price percentile within N-period range."""

    def __init__(self, period: int = 100):
        """Initialize PricePercentile."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"PricePct_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        def percentile(x):
            return (x[:-1] < x[-1]).sum() / len(x) * 100

        return df['Close'].rolling(window=self.period).apply(percentile, raw=True)


# =============================================================================
# VOLATILITY COMPRESSION INDICATORS
# =============================================================================

class BBWidthPercentile(Indicator):
    """
    Bollinger Band Width Percentile - identifies squeeze conditions.
    Low percentile = volatility compression = potential breakout setup.
    """

    def __init__(self, bb_period: int = 20, lookback: int = 100):
        """Initialize BBWidthPercentile."""
        self.bb_period = bb_period
        self.lookback = lookback
        self._bbwidth = BollingerBandWidth(bb_period)

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"BBWPct_{self.bb_period}_{self.lookback}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        bbwidth = self._bbwidth.calculate(df)

        def percentile_rank(x):
            return (x[:-1] < x[-1]).sum() / len(x) * 100

        return bbwidth.rolling(window=self.lookback).apply(percentile_rank, raw=True)


class NATRPercentile(Indicator):
    """
    NATR Percentile - where current volatility sits vs history.
    Low percentile = compression, potential breakout.
    """

    def __init__(self, natr_period: int = 14, lookback: int = 100):
        """Initialize NATRPercentile."""
        self.natr_period = natr_period
        self.lookback = lookback
        self._natr = NATR(natr_period)

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"NATRPct_{self.natr_period}_{self.lookback}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        natr = self._natr.calculate(df)

        def percentile_rank(x):
            return (x[:-1] < x[-1]).sum() / len(x) * 100

        return natr.rolling(window=self.lookback).apply(percentile_rank, raw=True)


class VolatilitySqueeze(Indicator):
    """
    Volatility Squeeze indicator - BB inside Keltner Channel.
    Returns 1 when in squeeze (low vol), 0 when not.
    """

    def __init__(self, period: int = 20, bb_mult: float = 2.0, kc_mult: float = 1.5):
        """Initialize VolatilitySqueeze."""
        self.period = period
        self.bb_mult = bb_mult
        self.kc_mult = kc_mult

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"Squeeze_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        close = df['Close']

        # Bollinger Bands
        sma = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        bb_upper = sma + (self.bb_mult * std)
        bb_lower = sma - (self.bb_mult * std)

        # Keltner Channel
        atr = ATR(self.period).calculate(df)
        kc_upper = sma + (self.kc_mult * atr)
        kc_lower = sma - (self.kc_mult * atr)

        # Squeeze: BB inside KC
        squeeze = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(int)
        return pd.Series(squeeze, index=df.index)


# =============================================================================
# ADVANCED VOLUME INDICATORS
# =============================================================================

class RVOL(Indicator):
    """
    Relative Volume - current volume vs same time period average.
    High RVOL = unusual activity = potential move.
    """

    def __init__(self, period: int = 20):
        """Initialize RVOL."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"RVOL_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        avg_vol = df['Volume'].rolling(window=self.period).mean()
        rvol = df['Volume'] / avg_vol
        return rvol


class NetVolumeTrend(Indicator):
    """
    Net Volume Trend - accumulation/distribution pressure.
    Positive = buying pressure, Negative = selling pressure.
    """

    def __init__(self, period: int = 14):
        """Initialize NetVolumeTrend."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"NetVol_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        # Volume direction based on close vs open
        """Compute indicator values from the provided OHLCV dataframe."""
        direction = np.sign(df['Close'] - df['Open'])
        net_volume = df['Volume'] * direction

        # Smooth with SMA
        net_vol_ma = net_volume.rolling(window=self.period).mean()

        # Normalize by average volume
        avg_vol = df['Volume'].rolling(window=self.period).mean()
        normalized = net_vol_ma / avg_vol

        return normalized


class VolumeForce(Indicator):
    """
    Volume Force Index - measures buying/selling pressure.
    Positive = bulls in control, Negative = bears in control.
    """

    def __init__(self, period: int = 13):
        """Initialize VolumeForce."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"VForce_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        force = df['Close'].diff() * df['Volume']
        force_ma = force.ewm(span=self.period, adjust=False).mean()
        return force_ma


class AccumulationDistribution(Indicator):
    """
    Accumulation/Distribution Line slope.
    Positive slope = accumulation, Negative slope = distribution.
    """

    def __init__(self, period: int = 10):
        """Initialize AccumulationDistribution."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"ADSlope_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        clv = clv.fillna(0)
        ad = (clv * df['Volume']).cumsum()
        ad_slope = ad.diff(self.period) / self.period
        return ad_slope


# =============================================================================
# EMA ALIGNMENT / TREND STRUCTURE
# =============================================================================

class EMAAlignment(Indicator):
    """
    EMA Alignment - checks if EMAs are properly stacked.
    +3 = perfect bull (8>21>50), -3 = perfect bear, 0 = mixed.
    """

    def __init__(self, fast: int = 8, medium: int = 21, slow: int = 50):
        """Initialize EMAAlignment."""
        self.fast = fast
        self.medium = medium
        self.slow = slow

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"EMAAlign_{self.fast}_{self.medium}_{self.slow}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        close = df['Close']
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_medium = close.ewm(span=self.medium, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()

        # Score alignment
        score = pd.Series(0, index=df.index)
        score += (ema_fast > ema_medium).astype(int) - (ema_fast < ema_medium).astype(int)
        score += (ema_medium > ema_slow).astype(int) - (ema_medium < ema_slow).astype(int)
        score += (close > ema_fast).astype(int) - (close < ema_fast).astype(int)

        return score


class TrendStrength(Indicator):
    """
    Combined trend strength using multiple factors.
    Higher = stronger trend, can be positive or negative.
    """

    def __init__(self, period: int = 20):
        """Initialize TrendStrength."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"TrendStr_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        close = df['Close']

        # Price above/below MA
        ma = close.rolling(window=self.period).mean()
        price_vs_ma = (close - ma) / ma

        # MA slope
        ma_slope = ma.pct_change(5)

        # ADX for trend strength (magnitude)
        adx = ADX(self.period).calculate(df)

        # Combine: direction from price_vs_ma, magnitude from ADX
        direction = np.sign(price_vs_ma)
        strength = direction * (adx / 100) * (1 + abs(ma_slope) * 10)

        return strength


class PriceVsEMAStack(Indicator):
    """
    Price position relative to EMA stack.
    +1 for each EMA price is above, -1 for each below.
    """

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return "PriceVsEMAs"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        close = df['Close']
        emas = [8, 13, 21, 34, 55]

        score = pd.Series(0, index=df.index)
        for period in emas:
            ema = close.ewm(span=period, adjust=False).mean()
            score += (close > ema).astype(int) - (close < ema).astype(int)

        return score


# =============================================================================
# PIVOT / BREAKOUT INDICATORS
# =============================================================================

class PivotHigh(Indicator):
    """
    Pivot High breakout - price breaks above N-bar high.
    Returns days since last pivot high break (0 = breaking now).
    """

    def __init__(self, left_bars: int = 5, right_bars: int = 5):
        """Initialize PivotHigh."""
        self.left_bars = left_bars
        self.right_bars = right_bars

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"PivotHi_{self.left_bars}_{self.right_bars}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        high = df['High']
        pivot_window = self.left_bars + self.right_bars + 1
        left = self.left_bars

        # Find pivot highs using raw numpy arrays
        pivots = high.rolling(window=pivot_window, center=True).apply(
            lambda x: x[left] if len(x) == pivot_window and
                      x[left] == x.max() else np.nan,
            raw=True
        )

        # Current high vs most recent pivot high
        pivot_high_level = pivots.ffill()
        breakout = (high > pivot_high_level).astype(int)

        return breakout


class PivotLow(Indicator):
    """
    Pivot Low breakdown - price breaks below N-bar low.
    Returns 1 when breaking down through pivot low.
    """

    def __init__(self, left_bars: int = 5, right_bars: int = 5):
        """Initialize PivotLow."""
        self.left_bars = left_bars
        self.right_bars = right_bars

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"PivotLo_{self.left_bars}_{self.right_bars}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        low = df['Low']
        pivot_window = self.left_bars + self.right_bars + 1
        left = self.left_bars

        # Find pivot lows using raw numpy arrays
        pivots = low.rolling(window=pivot_window, center=True).apply(
            lambda x: x[left] if len(x) == pivot_window and
                      x[left] == x.min() else np.nan,
            raw=True
        )

        # Current low vs most recent pivot low
        pivot_low_level = pivots.ffill()
        breakdown = (low < pivot_low_level).astype(int)

        return breakdown


class NBarHighBreak(Indicator):
    """
    Simple N-bar high breakout.
    Returns 1 when today's high breaks the high of last N bars.
    """

    def __init__(self, period: int = 5):
        """Initialize NBarHighBreak."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"HiBreak_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        prev_high = df['High'].shift(1).rolling(window=self.period).max()
        breakout = (df['High'] > prev_high).astype(int)
        return breakout


class NBarLowBreak(Indicator):
    """
    Simple N-bar low breakdown.
    Returns 1 when today's low breaks the low of last N bars.
    """

    def __init__(self, period: int = 5):
        """Initialize NBarLowBreak."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"LoBreak_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        prev_low = df['Low'].shift(1).rolling(window=self.period).min()
        breakdown = (df['Low'] < prev_low).astype(int)
        return breakdown


class RangeBreakout(Indicator):
    """
    Range Breakout - price breaks out of N-day range.
    +1 = upside breakout, -1 = downside breakout, 0 = inside range.
    """

    def __init__(self, period: int = 20):
        """Initialize RangeBreakout."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"RangeBO_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        prev_high = df['High'].shift(1).rolling(window=self.period).max()
        prev_low = df['Low'].shift(1).rolling(window=self.period).min()

        upside = (df['Close'] > prev_high).astype(int)
        downside = (df['Close'] < prev_low).astype(int) * -1

        return upside + downside


# =============================================================================
# ATR-BASED RISK INDICATORS
# =============================================================================

class ATRTrailingStop(Indicator):
    """
    Distance from ATR trailing stop.
    Positive = price above stop (in trade), Negative = below stop.
    Used for trend-following position management.
    """

    def __init__(self, period: int = 14, multiplier: float = 2.0):
        """Initialize ATRTrailingStop."""
        self.period = period
        self.multiplier = multiplier
        self._atr = ATR(period)

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"ATRStop_{self.period}_{self.multiplier}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        close = df['Close']
        atr = self._atr.calculate(df)

        # Chandelier Exit style trailing stop
        highest = close.rolling(window=self.period).max()
        stop_level = highest - (self.multiplier * atr)

        # Distance from stop as percentage
        distance = ((close - stop_level) / close) * 100
        return distance


class ATRChannel(Indicator):
    """
    Position within ATR channel.
    0 = at lower band, 100 = at upper band.
    """

    def __init__(self, period: int = 14, multiplier: float = 2.0):
        """Initialize ATRChannel."""
        self.period = period
        self.multiplier = multiplier
        self._atr = ATR(period)

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"ATRChan_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        close = df['Close']
        atr = self._atr.calculate(df)
        sma = close.rolling(window=self.period).mean()

        upper = sma + (self.multiplier * atr)
        lower = sma - (self.multiplier * atr)

        position = ((close - lower) / (upper - lower)) * 100
        return position


class RiskPerATR(Indicator):
    """
    Recent price range in ATR units.
    High value = extended move, low value = consolidation.
    """

    def __init__(self, atr_period: int = 14, range_period: int = 5):
        """Initialize RiskPerATR."""
        self.atr_period = atr_period
        self.range_period = range_period
        self._atr = ATR(atr_period)

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"RiskATR_{self.atr_period}_{self.range_period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        atr = self._atr.calculate(df)

        range_high = df['High'].rolling(window=self.range_period).max()
        range_low = df['Low'].rolling(window=self.range_period).min()
        price_range = range_high - range_low

        return price_range / atr


# =============================================================================
# REGIME INDICATORS
# =============================================================================

class MarketRegime(Indicator):
    """
    Market regime based on price action.
    +1 = uptrend, -1 = downtrend, 0 = ranging.
    Uses price vs MA and MA slope.
    """

    def __init__(self, period: int = 50):
        """Initialize MarketRegime."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"Regime_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        close = df['Close']
        ma = close.rolling(window=self.period).mean()
        ma_slope = ma.pct_change(10)

        # Uptrend: price above MA and MA rising
        uptrend = ((close > ma) & (ma_slope > 0.01)).astype(int)

        # Downtrend: price below MA and MA falling
        downtrend = ((close < ma) & (ma_slope < -0.01)).astype(int) * -1

        return uptrend + downtrend


class VolatilityRegime(Indicator):
    """
    Volatility regime classification.
    +1 = high vol, -1 = low vol, 0 = normal.
    """

    def __init__(self, period: int = 20, lookback: int = 100):
        """Initialize VolatilityRegime."""
        self.period = period
        self.lookback = lookback

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"VolRegime_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        returns = df['Close'].pct_change()
        current_vol = returns.rolling(window=self.period).std()
        vol_mean = current_vol.rolling(window=self.lookback).mean()
        vol_std = current_vol.rolling(window=self.lookback).std()

        z_score = (current_vol - vol_mean) / vol_std

        high_vol = (z_score > 1).astype(int)
        low_vol = (z_score < -1).astype(int) * -1

        return high_vol + low_vol


# =============================================================================
# VWAP INDICATORS
# =============================================================================

class VWAP(Indicator):
    """
    Volume Weighted Average Price - rolling calculation.
    For daily data, this approximates institutional fair value.
    """

    def __init__(self, period: int = 20):
        """Initialize VWAP."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"VWAP_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).rolling(window=self.period).sum() / \
               df['Volume'].rolling(window=self.period).sum()
        return vwap


class PriceVsVWAP(Indicator):
    """
    Price distance from VWAP as percentage.
    Positive = above VWAP (bullish), Negative = below (bearish).
    """

    def __init__(self, period: int = 20):
        """Initialize PriceVsVWAP."""
        self.period = period
        self._vwap = VWAP(period)

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"PriceVsVWAP_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        vwap = self._vwap.calculate(df)
        return ((df['Close'] - vwap) / vwap) * 100


class VWAPBands(Indicator):
    """
    VWAP Standard Deviation Bands.
    Returns position: 0=below -2SD, 50=at VWAP, 100=above +2SD.
    """

    def __init__(self, period: int = 20, num_std: float = 2.0):
        """Initialize VWAPBands."""
        self.period = period
        self.num_std = num_std
        self._vwap = VWAP(period)

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"VWAPBand_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        vwap = self._vwap.calculate(df)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3

        # Standard deviation of price from VWAP
        squared_diff = (typical_price - vwap) ** 2
        variance = squared_diff.rolling(window=self.period).mean()
        std = np.sqrt(variance)

        upper = vwap + (self.num_std * std)
        lower = vwap - (self.num_std * std)

        # Position within bands (0-100)
        position = ((df['Close'] - lower) / (upper - lower)) * 100
        return position.clip(0, 100)


class AnchoredVWAP(Indicator):
    """
    Anchored VWAP - VWAP calculated from N days ago.
    Different anchors give different institutional reference levels.
    Common anchors: 5 (week), 20 (month), 63 (quarter), 252 (year).
    """

    def __init__(self, anchor_days: int = 20):
        """Initialize AnchoredVWAP."""
        self.anchor_days = anchor_days

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"AVWAP_{self.anchor_days}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3

        # Rolling sum from anchor point
        tp_vol = typical_price * df['Volume']
        cum_tp_vol = tp_vol.rolling(window=self.anchor_days, min_periods=1).sum()
        cum_vol = df['Volume'].rolling(window=self.anchor_days, min_periods=1).sum()

        avwap = cum_tp_vol / cum_vol
        return avwap


class PriceVsAnchoredVWAP(Indicator):
    """
    Price distance from Anchored VWAP.
    Tests if price respects different VWAP anchors.
    """

    def __init__(self, anchor_days: int = 20):
        """Initialize PriceVsAnchoredVWAP."""
        self.anchor_days = anchor_days
        self._avwap = AnchoredVWAP(anchor_days)

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"PriceVsAVWAP_{self.anchor_days}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        avwap = self._avwap.calculate(df)
        return ((df['Close'] - avwap) / avwap) * 100


class MultiVWAPPosition(Indicator):
    """
    Position relative to multiple VWAP anchors.
    +3 = above all VWAPs (strong), -3 = below all (weak).
    Uses weekly, monthly, quarterly anchors.
    """

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return "MultiVWAP"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        close = df['Close']
        anchors = [5, 20, 63]  # Week, month, quarter

        score = pd.Series(0.0, index=df.index)
        for days in anchors:
            avwap = AnchoredVWAP(days).calculate(df)
            score += (close > avwap).astype(int) - (close < avwap).astype(int)

        return score


# =============================================================================
# VALUE AREA INDICATORS (Volume Profile Approximation)
# =============================================================================

class ValueAreaHigh(Indicator):
    """
    Value Area High approximation.
    Uses volume-weighted price distribution to find VAH.
    VAH = upper bound where 70% of volume traded.
    """

    def __init__(self, period: int = 20):
        """Initialize ValueAreaHigh."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"VAH_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        closes = df['Close'].values.astype(float)
        volumes = df['Volume'].values.astype(float)
        n = len(df)
        result = np.full(n, np.nan)

        for i in range(4, n):  # need at least 5 bars
            start = max(0, i - self.period + 1)
            window_prices = closes[start:i+1]
            window_volumes = volumes[start:i+1]

            total_vol = window_volumes.sum()
            if total_vol == 0:
                continue

            sorted_idx = np.argsort(window_prices)
            sorted_prices = window_prices[sorted_idx]
            sorted_volumes = window_volumes[sorted_idx]

            cum_vol = np.cumsum(sorted_volumes) / total_vol
            vah_idx = np.searchsorted(cum_vol, 0.85)
            vah_idx = min(vah_idx, len(sorted_prices) - 1)
            result[i] = sorted_prices[vah_idx]

        return pd.Series(result, index=df.index)


class ValueAreaLow(Indicator):
    """
    Value Area Low approximation.
    VAL = lower bound where 70% of volume traded.
    """

    def __init__(self, period: int = 20):
        """Initialize ValueAreaLow."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"VAL_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        closes = df['Close'].values.astype(float)
        volumes = df['Volume'].values.astype(float)
        n = len(df)
        result = np.full(n, np.nan)

        for i in range(4, n):  # need at least 5 bars
            start = max(0, i - self.period + 1)
            window_prices = closes[start:i+1]
            window_volumes = volumes[start:i+1]

            total_vol = window_volumes.sum()
            if total_vol == 0:
                continue

            sorted_idx = np.argsort(window_prices)
            sorted_prices = window_prices[sorted_idx]
            sorted_volumes = window_volumes[sorted_idx]

            cum_vol = np.cumsum(sorted_volumes) / total_vol
            val_idx = np.searchsorted(cum_vol, 0.15)
            val_idx = min(val_idx, len(sorted_prices) - 1)
            result[i] = sorted_prices[val_idx]

        return pd.Series(result, index=df.index)


class POC(Indicator):
    """
    Point of Control approximation.
    POC = price level with most volume (mode of volume distribution).
    """

    def __init__(self, period: int = 20, num_bins: int = 20):
        """Initialize POC."""
        self.period = period
        self.num_bins = num_bins

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"POC_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        closes = df['Close'].values.astype(float)
        volumes = df['Volume'].values.astype(float)
        n = len(df)
        num_bins = self.num_bins
        result = np.full(n, np.nan)

        for i in range(4, n):  # need at least 5 bars
            start = max(0, i - self.period + 1)
            window_prices = closes[start:i+1]
            window_volumes = volumes[start:i+1]

            price_min = window_prices.min()
            price_max = window_prices.max()
            if price_max == price_min:
                result[i] = price_min
                continue

            # Vectorized bin assignment
            bin_width = (price_max - price_min) / num_bins
            bin_indices = np.minimum(
                ((window_prices - price_min) / bin_width).astype(int),
                num_bins - 1)

            # Accumulate volume per bin
            bin_volumes = np.bincount(bin_indices, weights=window_volumes,
                                      minlength=num_bins)

            # POC is center of bin with max volume
            max_bin = np.argmax(bin_volumes)
            result[i] = price_min + (max_bin + 0.5) * bin_width

        return pd.Series(result, index=df.index)


class PriceVsPOC(Indicator):
    """
    Price distance from Point of Control.
    Positive = above POC, Negative = below.
    """

    def __init__(self, period: int = 20):
        """Initialize PriceVsPOC."""
        self.period = period
        self._poc = POC(period)

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"PriceVsPOC_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        poc = self._poc.calculate(df)
        return ((df['Close'] - poc) / poc) * 100


class ValueAreaPosition(Indicator):
    """
    Position within Value Area.
    0 = at VAL, 50 = at POC, 100 = at VAH.
    >100 or <0 = outside value area.
    """

    def __init__(self, period: int = 20):
        """Initialize ValueAreaPosition."""
        self.period = period
        self._vah = ValueAreaHigh(period)
        self._val = ValueAreaLow(period)

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"VAPos_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        vah = self._vah.calculate(df)
        val = self._val.calculate(df)
        close = df['Close']

        # Position as percentage of value area range
        va_range = vah - val
        position = ((close - val) / va_range) * 100

        return position


class AboveValueArea(Indicator):
    """
    Binary: 1 if price above VAH, 0 otherwise.
    Above VAH often indicates strength/breakout.
    """

    def __init__(self, period: int = 20):
        """Initialize AboveValueArea."""
        self.period = period
        self._vah = ValueAreaHigh(period)

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"AboveVA_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        vah = self._vah.calculate(df)
        return (df['Close'] > vah).astype(int)


class BelowValueArea(Indicator):
    """
    Binary: 1 if price below VAL, 0 otherwise.
    Below VAL often indicates weakness/breakdown.
    """

    def __init__(self, period: int = 20):
        """Initialize BelowValueArea."""
        self.period = period
        self._val = ValueAreaLow(period)

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"BelowVA_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        val = self._val.calculate(df)
        return (df['Close'] < val).astype(int)


# =============================================================================
# BEAST 666 — Proximity to "Triple-6" Price Levels
# =============================================================================

class Beast666Proximity(Indicator):
    """
    Beast 666 Proximity Score (0-100).

    Measures how close the current price is to the nearest "666 level" —
    any price containing three consecutive 6s in its significant digits.

    Examples of 666 levels:
        6.66, 16.66, 66.6, 166.6, 266.6, 366.6, 466.6, 566.6, 666.0,
        876.66, 1666, 2666, 6660, etc.

    The strategy hypothesis: prices tend to reverse at or near these
    psychologically significant "mark of the beast" levels.

    Score = 100 × exp(-pct_distance / tolerance)
        100 = exactly on a 666 level
        ~60 = within 0.25% of a 666 level
        ~37 = within 0.5% of a 666 level
        ~14 = within 1% of a 666 level
        ~0  = far from any 666 level

    Args:
        tolerance: Proximity band width as fraction of price.
                   0.005 (default) = 0.5% band. Smaller = tighter.
    """

    # Pre-computed 666-containing multipliers for level generation.
    # Applied at every power-of-10 from 10^-4 to 10^6 to cover prices
    # from penny stocks to AMZN/BRK.
    _BEAST_MULTIPLIERS = [
        6.66, 16.66, 26.66, 36.66, 46.66, 56.66, 66.6, 66.66,
        76.66, 86.66, 96.66, 106.66, 116.66, 126.66, 136.66,
        146.66, 156.66, 166.6, 176.66, 186.66, 196.66, 206.66,
        216.66, 226.66, 236.66, 246.66, 256.66, 266.6, 276.66,
        286.66, 296.66, 306.66, 316.66, 326.66, 336.66, 346.66,
        356.66, 366.6, 376.66, 386.66, 396.66, 406.66, 416.66,
        426.66, 436.66, 446.66, 456.66, 466.6, 476.66, 486.66,
        496.66, 506.66, 516.66, 526.66, 536.66, 546.66, 556.66,
        566.6, 576.66, 586.66, 596.66, 606.66, 616.66, 626.66,
        636.66, 646.66, 656.66, 666.0, 666.6, 676.66, 686.66,
        696.66, 706.66, 716.66, 726.66, 736.66, 746.66, 756.66,
        766.6, 776.66, 786.66, 796.66, 806.66, 816.66, 826.66,
        836.66, 846.66, 856.66, 866.6, 876.66, 886.66, 896.66,
        906.66, 916.66, 926.66, 936.66, 946.66, 956.66, 966.6,
        976.66, 986.66, 996.66,
        # Thousands with 666
        1666.0, 2666.0, 3666.0, 4666.0, 5666.0,
        6660.0, 6661.0, 6662.0, 6663.0, 6664.0,
        6665.0, 6666.0, 6667.0, 6668.0, 6669.0,
        7666.0, 8666.0, 9666.0,
    ]

    def __init__(self, tolerance: float = 0.005):
        """Initialize Beast666Proximity."""
        self.tolerance = tolerance
        # Pre-generate all 666 levels across exponent range
        self._all_levels = np.array(sorted(set(
            m * (10.0 ** exp)
            for exp in range(-4, 7)
            for m in self._BEAST_MULTIPLIERS
            if 0.01 <= m * (10.0 ** exp) <= 1_000_000
        )))

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        tol_str = str(self.tolerance).replace('.', 'p')
        return f"Beast666_{tol_str}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        close = df['Close'].values.astype(float)
        levels = self._all_levels
        n_levels = len(levels)

        # Vectorized: find nearest 666 level for all prices at once
        idx = np.searchsorted(levels, close)
        idx_lo = np.clip(idx - 1, 0, n_levels - 1)
        idx_hi = np.clip(idx, 0, n_levels - 1)

        dist_lo = np.abs(close - levels[idx_lo])
        dist_hi = np.abs(close - levels[idx_hi])

        # Handle edge cases where one side doesn't exist
        use_hi = (idx == 0)
        use_lo = (idx >= n_levels)
        nearest_dist = np.where(use_hi, dist_hi,
                       np.where(use_lo, dist_lo,
                                np.minimum(dist_lo, dist_hi)))

        pct_dist = nearest_dist / (np.abs(close) + 1e-12)
        scores = 100.0 * np.exp(-pct_dist / (self.tolerance + 1e-12))
        scores = np.where(close <= 0, 0.0, scores)

        return pd.Series(scores, index=df.index)


class Beast666Distance(Indicator):
    """
    Signed percent distance from the nearest 666 level.

    Positive = price is ABOVE the nearest 666 level.
    Negative = price is BELOW the nearest 666 level.

    Useful for detecting:
    - Approaching a 666 level from above (distance shrinking toward 0)
    - Bouncing off a 666 level (distance crosses from negative to positive)
    """

    def __init__(self):
        """Initialize Beast666Distance."""
        self._prox = Beast666Proximity(tolerance=0.005)

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return "Beast666Dist"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        close = df['Close'].values.astype(float)
        levels = self._prox._all_levels
        n_levels = len(levels)

        # Vectorized: find nearest 666 level for all prices at once
        idx = np.searchsorted(levels, close)
        idx_lo = np.clip(idx - 1, 0, n_levels - 1)
        idx_hi = np.clip(idx, 0, n_levels - 1)

        dist_lo = np.abs(close - levels[idx_lo])
        dist_hi = np.abs(close - levels[idx_hi])

        # Choose nearest level
        use_hi = (idx == 0)
        use_lo = (idx >= n_levels)
        nearest_level = np.where(use_hi, levels[idx_hi],
                        np.where(use_lo, levels[idx_lo],
                                 np.where(dist_lo <= dist_hi,
                                          levels[idx_lo], levels[idx_hi])))

        distances = (close - nearest_level) / (np.abs(close) + 1e-12) * 100
        distances = np.where(close <= 0, 0.0, distances)

        return pd.Series(distances, index=df.index)


# =============================================================================
# ADVANCED VOLATILITY MODELS
# =============================================================================

class ParkinsonVolatility(Indicator):
    """Parkinson range-based volatility estimator. More efficient than close-to-close."""

    def __init__(self, period: int = 20):
        """Initialize ParkinsonVolatility."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"ParkVol_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        log_hl_sq = (np.log(df['High'] / df['Low'])) ** 2
        factor = 1.0 / (4.0 * np.log(2))
        parkinson_var = log_hl_sq.rolling(window=self.period).mean() * factor
        return np.sqrt(parkinson_var * 252) * 100


class GarmanKlassVolatility(Indicator):
    """Garman-Klass OHLC volatility estimator. ~8x more efficient than close-to-close."""

    def __init__(self, period: int = 20):
        """Initialize GarmanKlassVolatility."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"GKVol_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        log_hl = np.log(df['High'] / df['Low'])
        log_co = np.log(df['Close'] / df['Open'])
        gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        avg_var = gk_var.rolling(window=self.period).mean()
        return np.sqrt(avg_var.clip(lower=0) * 252) * 100


class YangZhangVolatility(Indicator):
    """Yang-Zhang volatility combining overnight and Rogers-Satchell intraday components."""

    def __init__(self, period: int = 20):
        """Initialize YangZhangVolatility."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"YZVol_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        log_oc = np.log(df['Open'] / df['Close'].shift(1))
        log_co = np.log(df['Close'] / df['Open'])
        log_ho = np.log(df['High'] / df['Open'])
        log_lo = np.log(df['Low'] / df['Open'])

        sigma_o_sq = log_oc.rolling(window=self.period).var()

        log_cc = np.log(df['Close'] / df['Close'].shift(1))
        sigma_c_sq = log_cc.rolling(window=self.period).var()

        rs_var = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        sigma_rs_sq = rs_var.rolling(window=self.period).mean()

        n = self.period
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        yz_var = sigma_o_sq + k * sigma_c_sq + (1 - k) * sigma_rs_sq

        return np.sqrt(yz_var.clip(lower=0) * 252) * 100


class VolatilityCone(Indicator):
    """Percentile rank of current realized vol vs its historical distribution."""

    def __init__(self, vol_period: int = 20, lookback: int = 252):
        """Initialize VolatilityCone."""
        self.vol_period = vol_period
        self.lookback = lookback

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"VolCone_{self.vol_period}_{self.lookback}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        returns = df['Close'].pct_change()
        realized_vol = returns.rolling(window=self.vol_period).std() * np.sqrt(252)

        def percentile_rank(x):
            return (x[:-1] < x[-1]).sum() / (len(x) - 1) * 100

        return realized_vol.rolling(window=self.lookback).apply(percentile_rank, raw=True)


class VolOfVol(Indicator):
    """Volatility of volatility - rolling std of rolling volatility."""

    def __init__(self, vol_period: int = 20, vov_period: int = 60):
        """Initialize VolOfVol."""
        self.vol_period = vol_period
        self.vov_period = vov_period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"VoV_{self.vol_period}_{self.vov_period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        returns = df['Close'].pct_change()
        realized_vol = returns.rolling(window=self.vol_period).std() * np.sqrt(252)
        return realized_vol.rolling(window=self.vov_period).std() * 100


class GARCHVolatility(Indicator):
    """Simplified GARCH(1,1) volatility with fixed parameters."""

    def __init__(self, period: int = 252, alpha: float = 0.1, beta: float = 0.85):
        """Initialize GARCHVolatility."""
        self.period = period
        self.alpha = alpha
        self.beta = beta
        self.omega_scale = 1 - alpha - beta

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"GARCH_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        returns = df['Close'].pct_change().values
        n = len(returns)
        result = np.full(n, np.nan)

        if n < self.period + 1:
            return pd.Series(result, index=df.index)

        init_var = np.nanvar(returns[1:self.period + 1])
        omega = self.omega_scale * init_var
        sigma_sq = init_var

        for i in range(self.period + 1, n):
            r_prev = returns[i - 1]
            if np.isnan(r_prev):
                r_prev = 0.0
            sigma_sq = omega + self.alpha * r_prev**2 + self.beta * sigma_sq
            result[i] = np.sqrt(sigma_sq * 252) * 100

        return pd.Series(result, index=df.index)


class VolTermStructure(Indicator):
    """Ratio of short-term to long-term realized vol. >1 = backwardation (fear)."""

    def __init__(self, short_period: int = 10, long_period: int = 60):
        """Initialize VolTermStructure."""
        self.short_period = short_period
        self.long_period = long_period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"VolTS_{self.short_period}_{self.long_period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        returns = df['Close'].pct_change()
        short_vol = returns.rolling(window=self.short_period).std()
        long_vol = returns.rolling(window=self.long_period).std()
        return short_vol / long_vol.replace(0, np.nan)


# =============================================================================
# STATISTICAL / ECONOMETRIC
# =============================================================================

class HurstExponent(Indicator):
    """Hurst exponent via R/S analysis. H>0.5 trending, H<0.5 mean-reverting."""

    def __init__(self, period: int = 100):
        """Initialize HurstExponent."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"Hurst_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        prices = np.log(df['Close'].values.astype(float))
        n = len(prices)
        result = np.full(n, np.nan)

        for i in range(self.period, n):
            window = prices[i - self.period + 1:i + 1]
            returns = np.diff(window)
            m = len(returns)
            if m < 20:
                continue

            rs_values = []
            sizes = []
            size = m
            while size >= 8:
                n_blocks = m // size
                if n_blocks < 1:
                    size //= 2
                    continue
                rs_block = []
                for j in range(n_blocks):
                    block = returns[j * size:(j + 1) * size]
                    mean_block = block.mean()
                    deviations = np.cumsum(block - mean_block)
                    R = deviations.max() - deviations.min()
                    S = block.std(ddof=1)
                    if S > 1e-12:
                        rs_block.append(R / S)
                if rs_block:
                    rs_values.append(np.mean(rs_block))
                    sizes.append(size)
                size //= 2

            if len(rs_values) >= 2:
                log_sizes = np.log(np.array(sizes))
                log_rs = np.log(np.array(rs_values))
                A = np.column_stack([log_sizes, np.ones(len(log_sizes))])
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A, log_rs, rcond=None)
                    result[i] = coeffs[0]
                except (ValueError, KeyError, TypeError):
                    continue

        return pd.Series(result, index=df.index)


class MeanReversionHalfLife(Indicator):
    """Ornstein-Uhlenbeck half-life via OLS. Lower = faster mean reversion."""

    def __init__(self, period: int = 60):
        """Initialize MeanReversionHalfLife."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"HalfLife_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        prices = df['Close'].values.astype(float)
        n = len(prices)
        result = np.full(n, np.nan)

        for i in range(self.period, n):
            window = prices[i - self.period + 1:i + 1]
            y = np.diff(window)
            x = window[:-1]
            if len(y) < 10:
                continue

            A = np.column_stack([x, np.ones(len(x))])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                beta = coeffs[0]
                if beta < 0:
                    half_life = -np.log(2) / beta
                    result[i] = min(half_life, 1000)
            except (ValueError, KeyError, TypeError):
                continue

        return pd.Series(result, index=df.index)


class ZScore(Indicator):
    """Z-Score: standardized deviation from rolling mean."""

    def __init__(self, period: int = 20):
        """Initialize ZScore."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"ZScore_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        close = df['Close']
        mean = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        return (close - mean) / std.replace(0, np.nan)


class VarianceRatio(Indicator):
    """Lo-MacKinlay variance ratio. VR>1 = trending, VR<1 = mean-reverting."""

    def __init__(self, period: int = 100, k: int = 5):
        """Initialize VarianceRatio."""
        self.period = period
        self.k = k

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"VarRatio_{self.period}_{self.k}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        returns = df['Close'].pct_change()
        var_1 = returns.rolling(window=self.period).var()
        k_returns = df['Close'].pct_change(self.k)
        var_k = k_returns.rolling(window=self.period).var()
        return var_k / (self.k * var_1).replace(0, np.nan)


class Autocorrelation(Indicator):
    """Serial correlation of returns at lag k. Positive = momentum, negative = mean-reversion."""

    def __init__(self, period: int = 20, lag: int = 1):
        """Initialize Autocorrelation."""
        self.period = period
        self.lag = lag

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"AutoCorr_{self.period}_{self.lag}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        returns = df['Close'].pct_change()
        lag = self.lag

        def autocorr_k(x):
            if len(x) < lag + 2:
                return np.nan
            x1 = x[lag:]
            x0 = x[:-lag]
            m1 = x1.mean()
            m0 = x0.mean()
            num = ((x1 - m1) * (x0 - m0)).sum()
            den = np.sqrt(((x1 - m1)**2).sum() * ((x0 - m0)**2).sum())
            return num / den if den > 1e-12 else 0.0

        return returns.rolling(window=self.period).apply(autocorr_k, raw=True)


class KalmanTrend(Indicator):
    """1D Kalman filter for price trend extraction."""

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 1.0):
        """Initialize KalmanTrend."""
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        pn = str(self.process_noise).replace('.', 'p')
        mn = str(self.measurement_noise).replace('.', 'p')
        return f"Kalman_{pn}_{mn}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        prices = df['Close'].values.astype(float)
        n = len(prices)
        result = np.full(n, np.nan)

        if n < 2:
            return pd.Series(result, index=df.index)

        x = prices[0]
        P = 1.0
        Q = self.process_noise
        R = self.measurement_noise
        result[0] = x

        for i in range(1, n):
            if np.isnan(prices[i]):
                result[i] = x
                continue
            x_pred = x
            P_pred = P + Q
            K = P_pred / (P_pred + R)
            x = x_pred + K * (prices[i] - x_pred)
            P = (1 - K) * P_pred
            result[i] = x

        return pd.Series(result, index=df.index)


# =============================================================================
# INFORMATION THEORY
# =============================================================================

class ShannonEntropy(Indicator):
    """Shannon entropy of return distribution. High = uncertain, low = predictable."""

    def __init__(self, period: int = 20, n_bins: int = 10):
        """Initialize ShannonEntropy."""
        self.period = period
        self.n_bins = n_bins

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"Entropy_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        returns = df['Close'].pct_change()
        n_bins = self.n_bins

        def shannon_entropy(x):
            valid = x[~np.isnan(x)]
            if len(valid) < 5:
                return np.nan
            counts, _ = np.histogram(valid, bins=n_bins)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))

        return returns.rolling(window=self.period).apply(shannon_entropy, raw=True)


class ApproximateEntropy(Indicator):
    """Approximate Entropy (ApEn). Low = regular/predictable, high = complex/random."""

    def __init__(self, period: int = 50, m: int = 2, r_mult: float = 0.2):
        """Initialize ApproximateEntropy."""
        self.period = period
        self.m = m
        self.r_mult = r_mult

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"ApEn_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        prices = df['Close'].values.astype(float)
        n = len(prices)
        result = np.full(n, np.nan)
        m = self.m

        for i in range(self.period, n):
            window = prices[i - self.period + 1:i + 1]
            N = len(window)
            r = self.r_mult * np.std(window, ddof=1)
            if r < 1e-10:
                continue

            def _phi(m_dim):
                templates = np.array([window[j:j + m_dim] for j in range(N - m_dim + 1)])
                n_templates = len(templates)
                counts = np.zeros(n_templates)
                for j in range(n_templates):
                    dists = np.max(np.abs(templates - templates[j]), axis=1)
                    counts[j] = np.sum(dists <= r) / n_templates
                return np.log(counts[counts > 0]).mean()

            try:
                phi_m = _phi(m)
                phi_m1 = _phi(m + 1)
                result[i] = phi_m - phi_m1
            except (ValueError, KeyError, TypeError):
                continue

        return pd.Series(result, index=df.index)


# =============================================================================
# MICROSTRUCTURE
# =============================================================================

class AmihudIlliquidity(Indicator):
    """Amihud illiquidity ratio: |return| / dollar_volume. Higher = less liquid."""

    def __init__(self, period: int = 20):
        """Initialize AmihudIlliquidity."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"Amihud_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        returns = df['Close'].pct_change().abs()
        dollar_volume = df['Close'] * df['Volume']
        illiquidity = returns / dollar_volume.replace(0, np.nan)
        return illiquidity.rolling(window=self.period).mean() * 1e6


class KyleLambda(Indicator):
    """Kyle's lambda price impact coefficient via rolling regression."""

    def __init__(self, period: int = 20):
        """Initialize KyleLambda."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"Kyle_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        abs_price_change = df['Close'].diff().abs().values
        volume = df['Volume'].values.astype(float)
        n = len(df)
        result = np.full(n, np.nan)

        for i in range(self.period, n):
            y = abs_price_change[i - self.period + 1:i + 1]
            x = volume[i - self.period + 1:i + 1]
            valid = ~(np.isnan(y) | np.isnan(x) | (x == 0))
            if valid.sum() < 5:
                continue
            y_v = y[valid]
            x_v = x[valid]
            A = np.column_stack([x_v, np.ones(len(x_v))])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, y_v, rcond=None)
                result[i] = coeffs[0] * 1e6
            except (ValueError, KeyError, TypeError):
                continue

        return pd.Series(result, index=df.index)


class RollSpread(Indicator):
    """Roll's implied bid-ask spread in basis points."""

    def __init__(self, period: int = 20):
        """Initialize RollSpread."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"RollSpd_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        dp = df['Close'].diff()
        dp_lag = dp.shift(1)
        rolling_cov = dp.rolling(window=self.period).cov(dp_lag)
        neg_cov = (-rolling_cov).clip(lower=0)
        spread = 2 * np.sqrt(neg_cov)
        return (spread / df['Close'].replace(0, np.nan)) * 10000


# =============================================================================
# FRACTAL / COMPLEXITY
# =============================================================================

class FractalDimension(Indicator):
    """Higuchi fractal dimension. D~1 = smooth/trending, D~2 = rough/noisy."""

    def __init__(self, period: int = 100, k_max: int = 10):
        """Initialize FractalDimension."""
        self.period = period
        self.k_max = k_max

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"FracDim_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        prices = df['Close'].values.astype(float)
        n = len(prices)
        result = np.full(n, np.nan)

        for i in range(self.period, n):
            window = prices[i - self.period + 1:i + 1]
            N = len(window)

            lengths = []
            ks = []

            for k in range(1, min(self.k_max + 1, N // 4)):
                Lk_sum = 0
                count = 0
                for m_start in range(k):
                    indices = np.arange(m_start, N, k)
                    if len(indices) < 2:
                        continue
                    vals = window[indices]
                    L = np.sum(np.abs(np.diff(vals))) * (N - 1) / (k * len(indices) * k)
                    Lk_sum += L
                    count += 1
                if count > 0:
                    lengths.append(Lk_sum / count)
                    ks.append(k)

            if len(lengths) >= 2:
                log_k = np.log(1.0 / np.array(ks))
                log_L = np.log(np.array(lengths))
                valid = np.isfinite(log_k) & np.isfinite(log_L)
                if valid.sum() >= 2:
                    A = np.column_stack([log_k[valid], np.ones(valid.sum())])
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(A, log_L[valid], rcond=None)
                        result[i] = coeffs[0]
                    except (ValueError, KeyError, TypeError):
                        continue

        return pd.Series(result, index=df.index)


class DFA(Indicator):
    """Detrended Fluctuation Analysis. alpha>0.5 = persistent, alpha<0.5 = anti-persistent."""

    def __init__(self, period: int = 100):
        """Initialize DFA."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"DFA_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        prices = df['Close'].values.astype(float)
        n = len(prices)
        result = np.full(n, np.nan)

        for i in range(self.period, n):
            window = prices[i - self.period + 1:i + 1]
            log_window = np.log(window)
            log_window = log_window[np.isfinite(log_window)]
            if len(log_window) < 16:
                continue
            returns = np.diff(log_window)
            N = len(returns)

            y = np.cumsum(returns - returns.mean())

            box_sizes = []
            s = 4
            while s <= N // 4:
                box_sizes.append(s)
                s *= 2

            if len(box_sizes) < 2:
                continue

            fluctuations = []
            for s in box_sizes:
                n_boxes = N // s
                if n_boxes < 1:
                    continue
                F_sq = 0
                for j in range(n_boxes):
                    segment = y[j * s:(j + 1) * s]
                    t = np.arange(s, dtype=float)
                    A = np.column_stack([t, np.ones(s)])
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(A, segment, rcond=None)
                        trend = A @ coeffs
                        F_sq += np.mean((segment - trend) ** 2)
                    except (ValueError, KeyError, TypeError):
                        continue
                if n_boxes > 0:
                    fluctuations.append(np.sqrt(F_sq / n_boxes))

            if len(fluctuations) >= 2:
                log_s = np.log(np.array(box_sizes[:len(fluctuations)]))
                log_F = np.log(np.array(fluctuations))
                valid = np.isfinite(log_s) & np.isfinite(log_F)
                if valid.sum() >= 2:
                    A = np.column_stack([log_s[valid], np.ones(valid.sum())])
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(A, log_F[valid], rcond=None)
                        result[i] = coeffs[0]
                    except (ValueError, KeyError, TypeError):
                        continue

        return pd.Series(result, index=df.index)


# =============================================================================
# SPECTRAL ANALYSIS
# =============================================================================

class DominantCycle(Indicator):
    """FFT-based dominant cycle period in bars."""

    def __init__(self, period: int = 100):
        """Initialize DominantCycle."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"DomCycle_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        prices = df['Close'].values.astype(float)
        n = len(prices)
        result = np.full(n, np.nan)

        for i in range(self.period, n):
            window = prices[i - self.period + 1:i + 1]
            wlen = len(window)

            t = np.arange(wlen, dtype=float)
            A = np.column_stack([t, np.ones(wlen)])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, window, rcond=None)
                detrended = window - A @ coeffs
            except (ValueError, KeyError, TypeError):
                continue

            fft_vals = np.fft.rfft(detrended)
            magnitudes = np.abs(fft_vals)

            min_freq_idx = max(1, wlen // (wlen // 2)) if wlen > 2 else 1
            max_freq_idx = wlen // 3

            if max_freq_idx <= min_freq_idx:
                continue

            search_range = magnitudes[min_freq_idx:max_freq_idx + 1]
            if len(search_range) == 0:
                continue

            dominant_idx = np.argmax(search_range) + min_freq_idx
            if dominant_idx > 0:
                result[i] = wlen / dominant_idx

        return pd.Series(result, index=df.index)


# =============================================================================
# TAIL RISK / DISTRIBUTION
# =============================================================================

class ReturnSkewness(Indicator):
    """Rolling skewness of returns. Negative = left tail risk."""

    def __init__(self, period: int = 60):
        """Initialize ReturnSkewness."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"Skew_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        return df['Close'].pct_change().rolling(window=self.period).skew()


class ReturnKurtosis(Indicator):
    """Rolling excess kurtosis. High = fat tails (tail risk)."""

    def __init__(self, period: int = 60):
        """Initialize ReturnKurtosis."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"Kurt_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        return df['Close'].pct_change().rolling(window=self.period).kurt()


# =============================================================================
# REGIME DETECTION
# =============================================================================

class CUSUMDetector(Indicator):
    """CUSUM change-point detection. Output = bars since last regime change / period."""

    def __init__(self, period: int = 50, threshold: float = 4.0):
        """Initialize CUSUMDetector."""
        self.period = period
        self.threshold = threshold

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"CUSUM_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        returns = df['Close'].pct_change().values
        n = len(returns)
        result = np.full(n, np.nan)

        s_pos = 0.0
        s_neg = 0.0
        last_change_bar = 0

        for i in range(self.period, n):
            window_returns = returns[i - self.period + 1:i + 1]
            valid = window_returns[~np.isnan(window_returns)]
            if len(valid) < 10:
                continue

            mu = valid.mean()
            sigma = valid.std(ddof=1)
            if sigma < 1e-12:
                continue

            z = (returns[i] - mu) / sigma if not np.isnan(returns[i]) else 0.0

            s_pos = max(0, s_pos + z - 0.5)
            s_neg = max(0, s_neg - z - 0.5)

            if s_pos > self.threshold or s_neg > self.threshold:
                last_change_bar = i
                s_pos = 0.0
                s_neg = 0.0

            result[i] = (i - last_change_bar) / self.period

        return pd.Series(result, index=df.index)


class RegimePersistence(Indicator):
    """Consecutive bars in the same trend regime (price vs SMA)."""

    def __init__(self, period: int = 20):
        """Initialize RegimePersistence."""
        self.period = period

    @property
    def name(self) -> str:
        """Return the indicator's output column name."""
        return f"RegPersist_{self.period}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Compute indicator values from the provided OHLCV dataframe."""
        close = df['Close']
        sma = close.rolling(window=self.period).mean()
        above = (close > sma).astype(int)

        n = len(df)
        result = np.full(n, np.nan)
        count = 0
        prev_state = -1

        for i in range(self.period - 1, n):
            current_state = above.iloc[i]
            if np.isnan(current_state):
                continue
            if current_state == prev_state:
                count += 1
            else:
                count = 1
                prev_state = current_state
            result[i] = count

        return pd.Series(result, index=df.index)


# =============================================================================
# INDICATOR REGISTRY
# =============================================================================

def get_all_indicators() -> dict:
    """Return dictionary of all indicator classes."""
    return {
        # Volatility - Basic
        'ATR': ATR,
        'NATR': NATR,
        'BBWidth': BollingerBandWidth,
        'HV': HistoricalVolatility,

        # Volatility - Compression/Squeeze
        'BBWPct': BBWidthPercentile,
        'NATRPct': NATRPercentile,
        'Squeeze': VolatilitySqueeze,

        # Momentum
        'RSI': RSI,
        'MACD': MACD,
        'MACDSignal': MACDSignal,
        'MACDHist': MACDHistogram,
        'ROC': ROC,
        'Stoch': Stochastic,
        'StochD': StochasticD,
        'WillR': WilliamsR,
        'CCI': CCI,

        # Trend - Basic
        'SMA': SMA,
        'EMA': EMA,
        'PriceVsSMA': PriceVsSMA,
        'SMASlope': SMASlope,
        'ADX': ADX,
        'Aroon': Aroon,

        # Trend - Advanced/Alignment
        'EMAAlign': EMAAlignment,
        'TrendStr': TrendStrength,
        'PriceVsEMAs': PriceVsEMAStack,
        'Regime': MarketRegime,
        'VolRegime': VolatilityRegime,

        # Volume - Basic
        'VolRatio': VolumeRatio,
        'OBV': OBV,
        'OBVSlope': OBVSlope,
        'MFI': MFI,

        # Volume - Advanced
        'RVOL': RVOL,
        'NetVol': NetVolumeTrend,
        'VForce': VolumeForce,
        'ADSlope': AccumulationDistribution,

        # Price Action
        'HH': HigherHighs,
        'LL': LowerLows,
        'CandleBody': CandleBody,
        'CandleDir': CandleDirection,
        'Gap': GapPercent,

        # Support/Resistance
        'DistHigh': DistanceFromHigh,
        'DistLow': DistanceFromLow,
        'PricePct': PricePercentile,

        # Pivot/Breakout
        'PivotHi': PivotHigh,
        'PivotLo': PivotLow,
        'HiBreak': NBarHighBreak,
        'LoBreak': NBarLowBreak,
        'RangeBO': RangeBreakout,

        # ATR Risk Management
        'ATRStop': ATRTrailingStop,
        'ATRChan': ATRChannel,
        'RiskATR': RiskPerATR,

        # VWAP
        'VWAP': VWAP,
        'PriceVsVWAP': PriceVsVWAP,
        'VWAPBand': VWAPBands,
        'AVWAP': AnchoredVWAP,
        'PriceVsAVWAP': PriceVsAnchoredVWAP,
        'MultiVWAP': MultiVWAPPosition,

        # Value Area
        'VAH': ValueAreaHigh,
        'VAL': ValueAreaLow,
        'POC': POC,
        'PriceVsPOC': PriceVsPOC,
        'VAPos': ValueAreaPosition,
        'AboveVA': AboveValueArea,
        'BelowVA': BelowValueArea,

        # Beast 666
        'Beast666': Beast666Proximity,
        'Beast666Dist': Beast666Distance,

        # Advanced Volatility
        'ParkVol': ParkinsonVolatility,
        'GKVol': GarmanKlassVolatility,
        'YZVol': YangZhangVolatility,
        'VolCone': VolatilityCone,
        'VoV': VolOfVol,
        'GARCH': GARCHVolatility,
        'VolTS': VolTermStructure,

        # Statistical / Econometric
        'Hurst': HurstExponent,
        'HalfLife': MeanReversionHalfLife,
        'ZScore': ZScore,
        'VarRatio': VarianceRatio,
        'AutoCorr': Autocorrelation,
        'Kalman': KalmanTrend,

        # Information Theory
        'Entropy': ShannonEntropy,
        'ApEn': ApproximateEntropy,

        # Microstructure
        'Amihud': AmihudIlliquidity,
        'Kyle': KyleLambda,
        'RollSpd': RollSpread,

        # Fractal / Complexity
        'FracDim': FractalDimension,
        'DFA': DFA,

        # Spectral
        'DomCycle': DominantCycle,

        # Tail Risk / Distribution
        'Skew': ReturnSkewness,
        'Kurt': ReturnKurtosis,

        # Regime Detection
        'CUSUM': CUSUMDetector,
        'RegPersist': RegimePersistence,
    }


# Aliases for compound indicator names (for easier parsing)
INDICATOR_ALIASES = {
    'PriceVsAVWAP': PriceVsAnchoredVWAP,
    'PriceVsVWAP': PriceVsVWAP,
    'PriceVsPOC': PriceVsPOC,
    'EMAAlign': EMAAlignment,
    'PivotHi': PivotHigh,
    'PivotLo': PivotLow,
    'ATRStop': ATRTrailingStop,
    'RiskATR': RiskPerATR,
    'BBWPct': BBWidthPercentile,
    'NATRPct': NATRPercentile,
}


def create_indicator(name: str, **kwargs) -> Indicator:
    """Create an indicator by name with given parameters."""
    indicators = get_all_indicators()
    if name not in indicators:
        raise ValueError(f"Unknown indicator: {name}")
    return indicators[name](**kwargs)
