"""
Intraday microstructure features from WRDS TAQmsec tick data.

Provides ``compute_intraday_features()`` which uses the WRDSProvider's
TAQmsec method to pull intraday bars and derive market-microstructure
signals such as Amihud illiquidity, Kyle's lambda, VWAP deviation,
intraday volume ratio, realised 5-min volatility, and microstructure
noise.

All computation is self-contained: the function accepts a ticker, a date,
and a wrds_provider instance, and returns a dict of float features.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..config import MARKET_OPEN, MARKET_CLOSE


def compute_intraday_features(
    ticker: str,
    date: str,
    wrds_provider: Any,
) -> Dict[str, float]:
    """Compute intraday microstructure features for a single ticker on a date.

    Uses the WRDS TAQmsec provider to fetch 1-minute bars, then derives
    the following features:

    - ``intraday_vol_ratio`` : first-hour volume / last-hour volume
    - ``vwap_deviation``     : (close - VWAP) / VWAP
    - ``amihud_illiquidity`` : mean(|ret| / dollar_volume) * 1e6
    - ``kyle_lambda``        : OLS slope of signed_sqrt(vol) on price change
    - ``realized_vol_5m``    : annualised realised volatility from 5-min returns
    - ``microstructure_noise``: var(1m ret) / var(5m ret) - 1/5
                                (excess variance ratio, Hansen & Lunde style)

    Parameters
    ----------
    ticker : str
        Stock symbol (e.g. ``'AAPL'``).
    date : str
        Trading date in ``'YYYY-MM-DD'`` format.
    wrds_provider : WRDSProvider
        An initialised :class:`WRDSProvider` instance with TAQmsec access.

    Returns
    -------
    dict[str, float]
        Feature name to value mapping. Values are ``float('nan')`` when
        computation is not possible (e.g. insufficient data).
    """
    default: Dict[str, float] = {
        "intraday_vol_ratio": float("nan"),
        "vwap_deviation": float("nan"),
        "amihud_illiquidity": float("nan"),
        "kyle_lambda": float("nan"),
        "realized_vol_5m": float("nan"),
        "microstructure_noise": float("nan"),
    }

    if wrds_provider is None or not wrds_provider.available():
        return default

    try:
        # Fetch 1-minute bars for the single date
        bars_1m = wrds_provider.get_taqmsec_ohlcv(
            ticker=ticker,
            timeframe="1m",
            start_date=date,
            end_date=date,
        )
        if bars_1m is None or bars_1m.empty or len(bars_1m) < 10:
            return default

        close = pd.to_numeric(bars_1m["Close"], errors="coerce")
        volume = pd.to_numeric(bars_1m["Volume"], errors="coerce").fillna(0)
        high = pd.to_numeric(bars_1m["High"], errors="coerce")
        low = pd.to_numeric(bars_1m["Low"], errors="coerce")
        open_ = pd.to_numeric(bars_1m["Open"], errors="coerce")

        features: Dict[str, float] = {}

        # ── 1. Intraday volume ratio (first hour / last hour) ────────────
        try:
            first_hour = bars_1m.between_time(MARKET_OPEN, "10:30")
            last_hour = bars_1m.between_time("15:00", MARKET_CLOSE)
            vol_first = pd.to_numeric(first_hour["Volume"], errors="coerce").sum()
            vol_last = pd.to_numeric(last_hour["Volume"], errors="coerce").sum()
            features["intraday_vol_ratio"] = (
                float(vol_first / vol_last) if vol_last > 0 else float("nan")
            )
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            features["intraday_vol_ratio"] = float("nan")

        # ── 2. VWAP deviation ────────────────────────────────────────────
        try:
            typical_price = (high + low + close) / 3.0
            dollar_vol = typical_price * volume
            total_dollar = dollar_vol.sum()
            total_vol = volume.sum()
            if total_vol > 0:
                vwap = total_dollar / total_vol
                last_close = close.iloc[-1]
                features["vwap_deviation"] = float(
                    (last_close - vwap) / vwap
                ) if vwap > 0 else float("nan")
            else:
                features["vwap_deviation"] = float("nan")
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            features["vwap_deviation"] = float("nan")

        # ── 3. Amihud illiquidity ────────────────────────────────────────
        try:
            ret_1m = close.pct_change().replace([np.inf, -np.inf], np.nan)
            dollar_volume = close * volume
            valid = (dollar_volume > 0) & ret_1m.notna()
            if valid.sum() > 5:
                amihud = (ret_1m[valid].abs() / dollar_volume[valid]).mean() * 1e6
                features["amihud_illiquidity"] = float(amihud)
            else:
                features["amihud_illiquidity"] = float("nan")
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            features["amihud_illiquidity"] = float("nan")

        # ── 4. Kyle's lambda ────────────────────────────────────────────
        try:
            ret_1m = close.pct_change().replace([np.inf, -np.inf], np.nan)
            signed_vol = np.sign(ret_1m) * np.sqrt(volume)
            mask = ret_1m.notna() & signed_vol.notna() & np.isfinite(signed_vol)
            if mask.sum() > 10:
                y = ret_1m[mask].values
                x = signed_vol[mask].values
                # Simple OLS: lambda = cov(ret, signed_sqrt_vol) / var(signed_sqrt_vol)
                cov_xy = np.cov(y, x, ddof=1)
                if cov_xy.shape == (2, 2) and cov_xy[1, 1] > 1e-15:
                    kyle_lam = cov_xy[0, 1] / cov_xy[1, 1]
                    features["kyle_lambda"] = float(kyle_lam)
                else:
                    features["kyle_lambda"] = float("nan")
            else:
                features["kyle_lambda"] = float("nan")
        except (KeyError, ValueError, TypeError):
            features["kyle_lambda"] = float("nan")

        # ── 5. Realised volatility from 5-min returns ────────────────────
        try:
            bars_5m = close.resample("5min").last().dropna()
            if len(bars_5m) > 5:
                ret_5m = np.log(bars_5m / bars_5m.shift(1)).dropna()
                # Annualise: 78 five-minute bars per day, 252 trading days
                rv_5m = ret_5m.std() * np.sqrt(78.0 * 252.0)
                features["realized_vol_5m"] = float(rv_5m)
            else:
                features["realized_vol_5m"] = float("nan")
        except (KeyError, ValueError, TypeError):
            features["realized_vol_5m"] = float("nan")

        # ── 6. Microstructure noise ──────────────────────────────────────
        try:
            ret_1m = np.log(close / close.shift(1)).replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            bars_5m = close.resample("5min").last().dropna()
            ret_5m = np.log(bars_5m / bars_5m.shift(1)).dropna()

            if len(ret_1m) > 10 and len(ret_5m) > 5:
                var_1m = ret_1m.var()
                var_5m = ret_5m.var()
                if var_5m > 1e-15:
                    # Excess variance ratio: if returns were i.i.d.,
                    # var(1m) / var(5m) should be 1/5. Deviation indicates
                    # microstructure noise (positive) or autocorrelation.
                    noise = (var_1m / var_5m) - (1.0 / 5.0)
                    features["microstructure_noise"] = float(noise)
                else:
                    features["microstructure_noise"] = float("nan")
            else:
                features["microstructure_noise"] = float("nan")
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            features["microstructure_noise"] = float("nan")

        # Fill any missing keys with NaN
        for k in default:
            if k not in features:
                features[k] = float("nan")

        return features

    except (KeyError, ValueError, TypeError, RuntimeError):
        return default


# ---------------------------------------------------------------------------
# Causal VWAP alternative (rolling window, no future data)
# ---------------------------------------------------------------------------


def compute_rolling_vwap(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Compute causal rolling VWAP and deviation features.

    Unlike the full-day VWAP used in ``compute_intraday_features``, this
    uses a rolling window over *past* bars only — safe for intraday and
    daily prediction.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with 'High', 'Low', 'Close', and 'Volume' columns.
    window : int
        Number of past bars for the rolling calculation (default 20).

    Returns
    -------
    pd.DataFrame
        Columns: ``rolling_vwap_{window}``, ``rolling_vwap_deviation_{window}``.
    """
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")

    typical_price = (high + low + close) / 3.0
    dollar_volume = typical_price * volume

    min_periods = max(5, window // 3)
    rolling_dollar = dollar_volume.rolling(window, min_periods=min_periods).sum()
    rolling_vol = volume.rolling(window, min_periods=min_periods).sum()

    rolling_vwap = rolling_dollar / rolling_vol.replace(0.0, np.nan)
    deviation = (close - rolling_vwap) / rolling_vwap.replace(0.0, np.nan)

    out = pd.DataFrame(index=df.index)
    out[f"rolling_vwap_{window}"] = rolling_vwap
    out[f"rolling_vwap_deviation_{window}"] = deviation
    return out.replace([np.inf, -np.inf], np.nan)
