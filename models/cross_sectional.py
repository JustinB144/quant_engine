"""
Cross-Sectional Ranking Model — rank stocks relative to peers at each date.

Converts time-series predictions into cross-sectional signals by ranking
stocks within each date.  This addresses the limitation of purely
time-series models that predict absolute returns without considering
relative positioning across the universe.

Components:
    - cross_sectional_rank: percentile ranking + z-score + long/short signal
"""
from typing import Optional

import numpy as np
import pandas as pd


def cross_sectional_rank(
    predictions: pd.DataFrame,
    date_col: str = "date",
    prediction_col: str = "predicted_return",
    asset_col: Optional[str] = None,
    long_quantile: float = 0.80,
    short_quantile: float = 0.20,
) -> pd.DataFrame:
    """Rank stocks cross-sectionally by predicted return at each date.

    For each unique date in *predictions*, stocks are ranked by their
    predicted return.  Three derived columns are added:

    - ``cs_rank``: percentile rank in [0, 1] (1 = highest predicted return)
    - ``cs_zscore``: cross-sectional z-score (mean 0, std 1 within each date)
    - ``long_short_signal``: +1 for top quintile, -1 for bottom quintile,
      0 otherwise

    Parameters
    ----------
    predictions : pd.DataFrame
        Must contain a date column and a prediction column.  Can optionally
        contain an asset identifier column.  If the DataFrame has a
        MultiIndex with date as one level, the *date_col* parameter is used
        to identify that level.
    date_col : str, default "date"
        Column (or index level) identifying the date.
    prediction_col : str, default "predicted_return"
        Column containing the predicted return values to rank.
    asset_col : str, optional
        Column identifying the asset/stock.  Not required for ranking but
        preserved in output if present.
    long_quantile : float, default 0.80
        Percentile threshold for long signal (+1).  Stocks at or above this
        rank get +1.
    short_quantile : float, default 0.20
        Percentile threshold for short signal (-1).  Stocks at or below this
        rank get -1.

    Returns
    -------
    pd.DataFrame
        Copy of *predictions* with additional columns: ``cs_rank``,
        ``cs_zscore``, and ``long_short_signal``.

    Raises
    ------
    ValueError
        If *prediction_col* is not found in the DataFrame.
    """
    if not 0.0 < short_quantile < long_quantile < 1.0:
        raise ValueError(
            f"Quantile thresholds must satisfy 0 < short_quantile < long_quantile < 1, "
            f"got short_quantile={short_quantile}, long_quantile={long_quantile}"
        )

    # Work on a copy to avoid mutating the input
    df = predictions.copy()

    # Handle case where date is in the index
    date_in_index = False
    if date_col not in df.columns:
        if date_col in getattr(df.index, "names", []):
            df = df.reset_index()
            date_in_index = True
        elif isinstance(df.index, pd.DatetimeIndex) and date_col == "date":
            df.index.name = date_col
            df = df.reset_index()
            date_in_index = True
        else:
            raise ValueError(
                f"date_col '{date_col}' not found in columns or index"
            )

    if prediction_col not in df.columns:
        raise ValueError(
            f"prediction_col '{prediction_col}' not found in DataFrame columns. "
            f"Available: {list(df.columns)}"
        )

    # Compute cross-sectional statistics per date
    df["cs_rank"] = np.nan
    df["cs_zscore"] = np.nan
    df["long_short_signal"] = 0

    for date_val, group in df.groupby(date_col):
        idx = group.index
        preds = group[prediction_col]
        valid = preds.dropna()

        if len(valid) < 2:
            # Cannot rank fewer than 2 stocks
            df.loc[idx, "cs_rank"] = 0.5
            df.loc[idx, "cs_zscore"] = 0.0
            df.loc[idx, "long_short_signal"] = 0
            continue

        # Percentile rank: fraction of values <= current value
        ranks = preds.rank(method="average", pct=True)
        df.loc[idx, "cs_rank"] = ranks

        # Z-score: (x - mean) / std
        mean_val = preds.mean()
        std_val = preds.std()
        if std_val > 0:
            df.loc[idx, "cs_zscore"] = (preds - mean_val) / std_val
        else:
            df.loc[idx, "cs_zscore"] = 0.0

        # Long/short signal based on quantile thresholds
        signal = pd.Series(0, index=idx, dtype=int)
        signal[ranks >= long_quantile] = 1
        signal[ranks <= short_quantile] = -1
        df.loc[idx, "long_short_signal"] = signal

    # Cast signal column to int (use .loc for pandas 3.0 CoW compatibility)
    df.loc[:, "long_short_signal"] = df["long_short_signal"].fillna(0).astype(int)

    return df
