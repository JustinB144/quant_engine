"""
Markov LOB (Limit Order Book) features from intraday bar data (Tier 6.2).

Based on: "Markov models for limit order books" (Paper 1104.4596v1).

Computes microstructure features from aggregated intraday OHLCV bars
(e.g. TAQmsec 1-min or 5-min data).  Since raw order-book snapshots are
not available, all features are *proxies* derived from bar aggregates.

Single stock-day features:
  - ``trade_arrival_rate``: estimated Poisson lambda from exponential
    distribution fit of inter-bar durations.
  - ``quote_update_intensity``: number of price changes per unit time.
  - ``duration_between_trades_mean``: mean inter-trade duration (seconds).
  - ``duration_between_trades_std``: std of inter-trade durations.
  - ``price_impact_asymmetry``: buy vs. sell impact from signed volume.
  - ``queue_imbalance``: ratio of up-moves to down-moves.
  - ``fill_probability_proxy``: fraction of bars where volume > median.

Batch function ``compute_lob_features_batch`` processes multiple
stock-days and returns a DataFrame.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

_EPS = 1e-12


def _inter_bar_durations(index: pd.DatetimeIndex) -> np.ndarray:
    """Compute inter-bar durations in seconds from a DatetimeIndex."""
    if not isinstance(index, pd.DatetimeIndex):
        try:
            index = pd.to_datetime(index)
        except (ValueError, KeyError, TypeError):
            return np.array([], dtype=float)

    if len(index) < 2:
        return np.array([], dtype=float)

    diffs = np.diff(index.values).astype("timedelta64[s]").astype(float)
    # Filter out non-positive durations (duplicates, out-of-order)
    diffs = diffs[diffs > 0]
    return diffs


def _estimate_poisson_lambda(durations: np.ndarray) -> float:
    """
    Estimate trade arrival rate (lambda) from inter-arrival durations.

    Under a Poisson process, inter-arrival times are exponentially
    distributed with parameter lambda = 1 / mean(duration).
    """
    if len(durations) == 0:
        return float("nan")

    valid = durations[np.isfinite(durations) & (durations > 0)]
    if len(valid) == 0:
        return float("nan")

    mean_duration = np.mean(valid)
    if mean_duration < _EPS:
        return float("nan")

    return 1.0 / mean_duration


def _signed_volume(bars: pd.DataFrame) -> pd.Series:
    """
    Approximate trade direction using candle body (close - open).

    sign = +1 if close > open (buy pressure)
           -1 if close < open (sell pressure)
            0 if close == open
    """
    close = pd.to_numeric(bars["Close"], errors="coerce")
    open_ = pd.to_numeric(bars["Open"], errors="coerce")
    volume = pd.to_numeric(bars["Volume"], errors="coerce").fillna(0)

    direction = np.sign(close - open_).fillna(0)
    return direction * volume


def compute_lob_features(
    intraday_bars: pd.DataFrame,
    freq: str = "5min",
) -> Dict[str, float]:
    """
    Compute Markov LOB proxy features for a single stock-day.

    Parameters
    ----------
    intraday_bars : pd.DataFrame
        Intraday OHLCV bars for a single stock on a single trading day.
        Expected columns: Open, High, Low, Close, Volume.
        Index should be a DatetimeIndex (or convertible to one).
    freq : str
        Bar frequency hint (e.g. "1min", "5min").  Used for context but
        the actual durations are computed from the index timestamps.

    Returns
    -------
    dict[str, float]
        Feature name to value mapping.  Values are ``float('nan')``
        when computation is not possible.
    """
    default: Dict[str, float] = {
        "trade_arrival_rate": float("nan"),
        "quote_update_intensity": float("nan"),
        "duration_between_trades_mean": float("nan"),
        "duration_between_trades_std": float("nan"),
        "price_impact_asymmetry": float("nan"),
        "queue_imbalance": float("nan"),
        "fill_probability_proxy": float("nan"),
    }

    if intraday_bars is None or len(intraday_bars) < 3:
        return default

    # Ensure required columns exist
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(intraday_bars.columns)):
        return default

    features: Dict[str, float] = {}

    close = pd.to_numeric(intraday_bars["Close"], errors="coerce")
    open_ = pd.to_numeric(intraday_bars["Open"], errors="coerce")
    volume = pd.to_numeric(intraday_bars["Volume"], errors="coerce").fillna(0)

    # ------------------------------------------------------------------
    # 1. Trade arrival rate (Poisson lambda)
    # ------------------------------------------------------------------
    durations = _inter_bar_durations(intraday_bars.index)
    lam = _estimate_poisson_lambda(durations)
    features["trade_arrival_rate"] = lam

    # ------------------------------------------------------------------
    # 2. Duration statistics
    # ------------------------------------------------------------------
    if len(durations) > 0:
        valid_dur = durations[np.isfinite(durations) & (durations > 0)]
        if len(valid_dur) > 0:
            features["duration_between_trades_mean"] = float(np.mean(valid_dur))
            features["duration_between_trades_std"] = float(
                np.std(valid_dur, ddof=1) if len(valid_dur) > 1 else 0.0
            )
        else:
            features["duration_between_trades_mean"] = float("nan")
            features["duration_between_trades_std"] = float("nan")
    else:
        features["duration_between_trades_mean"] = float("nan")
        features["duration_between_trades_std"] = float("nan")

    # ------------------------------------------------------------------
    # 3. Quote update intensity
    # ------------------------------------------------------------------
    # Number of price changes per unit time
    price_changes = (close.diff().abs() > _EPS).sum()
    total_time_seconds = durations.sum() if len(durations) > 0 else 0.0
    if total_time_seconds > 0:
        features["quote_update_intensity"] = float(price_changes) / total_time_seconds
    else:
        features["quote_update_intensity"] = float("nan")

    # ------------------------------------------------------------------
    # 4. Price impact asymmetry (buy vs. sell)
    # ------------------------------------------------------------------
    try:
        signed_vol = _signed_volume(intraday_bars)
        price_change = close.diff()

        buy_mask = signed_vol > 0
        sell_mask = signed_vol < 0

        if buy_mask.sum() > 1 and sell_mask.sum() > 1:
            # Average price change per unit of buy volume vs sell volume
            buy_vol_total = signed_vol[buy_mask].sum()
            sell_vol_total = signed_vol[sell_mask].abs().sum()

            buy_impact = (
                price_change[buy_mask].sum() / (buy_vol_total + _EPS)
            )
            sell_impact = (
                price_change[sell_mask].sum() / (sell_vol_total + _EPS)
            )

            # Asymmetry: positive = buys move price more than sells
            features["price_impact_asymmetry"] = float(
                abs(buy_impact) - abs(sell_impact)
            )
        else:
            features["price_impact_asymmetry"] = float("nan")
    except (ValueError, KeyError, TypeError):
        features["price_impact_asymmetry"] = float("nan")

    # ------------------------------------------------------------------
    # 5. Queue imbalance (up-moves vs down-moves)
    # ------------------------------------------------------------------
    try:
        delta = close.diff()
        up_moves = (delta > 0).sum()
        down_moves = (delta < 0).sum()
        total_moves = up_moves + down_moves

        if total_moves > 0:
            features["queue_imbalance"] = float(up_moves - down_moves) / float(total_moves)
        else:
            features["queue_imbalance"] = float("nan")
    except (ValueError, KeyError, TypeError):
        features["queue_imbalance"] = float("nan")

    # ------------------------------------------------------------------
    # 6. Fill probability proxy
    # ------------------------------------------------------------------
    try:
        median_vol = volume.median()
        if median_vol > 0 and len(volume) > 0:
            features["fill_probability_proxy"] = float(
                (volume > median_vol).sum() / len(volume)
            )
        else:
            features["fill_probability_proxy"] = float("nan")
    except (ValueError, KeyError, TypeError):
        features["fill_probability_proxy"] = float("nan")

    # Fill any missing keys with NaN
    for k in default:
        if k not in features:
            features[k] = float("nan")

    return features


def compute_lob_features_batch(
    intraday_data: Dict[str, pd.DataFrame],
    freq: str = "5min",
) -> pd.DataFrame:
    """
    Compute LOB features for multiple stock-days in batch.

    Parameters
    ----------
    intraday_data : dict[str, pd.DataFrame]
        Mapping of ``"permno__YYYY-MM-DD"`` (or any unique key) to
        intraday OHLCV DataFrames.  The key format should allow
        extraction of permno and date.  Acceptable key formats:

        - ``"AAPL__2024-01-15"`` (double-underscore separator)
        - ``"12345__2024-01-15"`` (permno + date)
        - any string key (used as-is for the permno, date inferred
          from the DataFrame index)

    freq : str
        Bar frequency hint passed to ``compute_lob_features``.

    Returns
    -------
    pd.DataFrame
        MultiIndex (permno, date) DataFrame with LOB feature columns.
    """
    if not intraday_data:
        return pd.DataFrame()

    rows = []
    for key, bars in intraday_data.items():
        feats = compute_lob_features(bars, freq=freq)

        # Parse key into (permno, date)
        if "__" in str(key):
            parts = str(key).split("__", 1)
            permno = parts[0]
            date_str = parts[1] if len(parts) > 1 else None
        else:
            permno = str(key)
            date_str = None

        # Attempt to infer date from the DataFrame index if not in key
        if date_str is None:
            try:
                idx = pd.to_datetime(bars.index)
                date_str = str(idx[0].date())
            except (ValueError, KeyError, TypeError):
                date_str = str(key)

        feats["permno"] = permno
        feats["date"] = date_str
        rows.append(feats)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result.loc[:, "date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.set_index(["permno", "date"]).sort_index()

    # Drop the key columns if accidentally included
    feature_cols = [
        "trade_arrival_rate",
        "quote_update_intensity",
        "duration_between_trades_mean",
        "duration_between_trades_std",
        "price_impact_asymmetry",
        "queue_imbalance",
        "fill_probability_proxy",
    ]
    return result[[c for c in feature_cols if c in result.columns]]
