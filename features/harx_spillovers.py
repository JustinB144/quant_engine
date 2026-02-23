"""
HARX Volatility Spillover features (Tier 6.1).

Based on: "Heterogeneous Autoregressive models with Exogenous regressors"
(Paper 2601.03146v4).

Extends the standard HAR model (Corsi 2009) by including cross-market
realized volatility as exogenous regressors.  For each asset *i* in a
universe, the HARX regression is:

    RV_i(t+1) = alpha
              + beta_d * RV_i_daily(t)
              + beta_w * RV_i_weekly(t)
              + beta_m * RV_i_monthly(t)
              + SUM_j  gamma_j * RV_j_daily(t)      [j != i]

where RV_j_daily are the *exogenous* realized volatilities of other
assets.  The gamma_j coefficients capture volatility *spillover* from
asset j to asset i.

From these spillover coefficients, we derive a Diebold-Yilmaz-style
spillover index and per-asset directional spillover measures:
  - ``harx_spillover_from``:  how much volatility this asset *receives*
  - ``harx_spillover_to``:    how much it *transmits*
  - ``harx_net_spillover``:   to - from (net transmitter if positive)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

_EPS = 1e-12


def _realized_volatility(
    returns: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Rolling realized volatility (annualised std of returns)."""
    minp = min_periods if min_periods is not None else max(3, window // 2)
    return returns.rolling(window, min_periods=minp).std() * np.sqrt(252.0)


def _ols_lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS via numpy lstsq.  Returns coefficient vector."""
    result, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return result


def compute_harx_spillovers(
    returns_by_asset: Dict[str, pd.Series],
    rv_daily_window: int = 5,
    rv_weekly_window: int = 21,
    rv_monthly_window: int = 22,
    regression_window: int = 126,
    min_regression_obs: int = 60,
) -> pd.DataFrame:
    """
    Compute HARX cross-market volatility spillover features.

    Parameters
    ----------
    returns_by_asset : dict[str, pd.Series]
        Mapping of asset identifier (e.g. permno) to daily return series.
        All series should share a common DatetimeIndex.
    rv_daily_window : int
        Window for daily-horizon realized volatility (default 5).
    rv_weekly_window : int
        Window for weekly-horizon RV (default 5, kept as separate
        parameter for HAR nomenclature clarity).
    rv_monthly_window : int
        Window for monthly-horizon RV (default 22).
    regression_window : int
        Rolling window for the HARX OLS regression (default 126 ~ 6 months).
    min_regression_obs : int
        Minimum valid observations required in a regression window (default 60).

    Returns
    -------
    pd.DataFrame
        MultiIndex (permno, date) with columns:
        ``harx_spillover_from``, ``harx_spillover_to``,
        ``harx_net_spillover``.
    """
    # ------------------------------------------------------------------
    # 1. Build aligned return and RV matrices
    # ------------------------------------------------------------------
    asset_names = sorted(returns_by_asset.keys())
    if len(asset_names) < 2:
        return pd.DataFrame()

    # Align all return series on a common DatetimeIndex
    ret_df = pd.DataFrame(
        {name: returns_by_asset[name] for name in asset_names}
    ).sort_index().dropna(how="all")

    if ret_df.shape[0] < regression_window + rv_monthly_window + 2:
        return pd.DataFrame()

    n_dates, n_assets = ret_df.shape
    dates = ret_df.index

    # Compute RV at three horizons for each asset
    rv_daily = pd.DataFrame(index=dates)
    rv_weekly = pd.DataFrame(index=dates)
    rv_monthly = pd.DataFrame(index=dates)

    for name in asset_names:
        rv_daily[name] = _realized_volatility(ret_df[name], rv_daily_window)
        rv_weekly[name] = _realized_volatility(ret_df[name], rv_weekly_window)
        rv_monthly[name] = _realized_volatility(ret_df[name], rv_monthly_window)

    # ------------------------------------------------------------------
    # 2. Run rolling HARX regressions for each asset
    # ------------------------------------------------------------------
    # For each asset i, the design matrix X at time t has columns:
    #   [1, RV_i_daily(t), RV_i_weekly(t), RV_i_monthly(t),
    #    RV_j1_daily(t), RV_j2_daily(t), ...]
    # and y = RV_i_daily(t+1).

    # Spillover coefficient arrays
    # gamma_ij[t, i, j] = spillover from j to i at time t
    gamma_matrix = np.full((n_dates, n_assets, n_assets), np.nan, dtype=float)

    # Pre-extract numpy arrays for speed
    rv_d_vals = rv_daily.values.astype(float)   # (n_dates, n_assets)
    rv_w_vals = rv_weekly.values.astype(float)
    rv_m_vals = rv_monthly.values.astype(float)

    for i in range(n_assets):
        # Build the regression for asset i
        # y(t) = RV_i_daily(t+1), so we shift the target by -1
        y_full = rv_d_vals[1:, i]  # target: RV at t+1

        # Design matrix columns: intercept, own HAR, exogenous
        # intercept + 3 own + (n_assets-1) exogenous
        n_exog = n_assets - 1
        n_regressors = 1 + 3 + n_exog

        X_full = np.full((n_dates - 1, n_regressors), np.nan, dtype=float)
        X_full[:, 0] = 1.0  # intercept
        X_full[:, 1] = rv_d_vals[:-1, i]   # own daily RV at t
        X_full[:, 2] = rv_w_vals[:-1, i]   # own weekly RV at t
        X_full[:, 3] = rv_m_vals[:-1, i]   # own monthly RV at t

        # Exogenous: daily RV of other assets
        exog_col = 0
        exog_asset_indices = []
        for j in range(n_assets):
            if j == i:
                continue
            X_full[:, 4 + exog_col] = rv_d_vals[:-1, j]
            exog_asset_indices.append(j)
            exog_col += 1

        # Rolling regression
        for t in range(regression_window, n_dates - 1):
            start = t - regression_window
            end = t
            X_block = X_full[start:end]
            y_block = y_full[start:end]

            # Check for valid observations
            valid = np.all(np.isfinite(X_block), axis=1) & np.isfinite(y_block)
            if valid.sum() < min_regression_obs:
                continue

            X_valid = X_block[valid]
            y_valid = y_block[valid]

            try:
                beta = _ols_lstsq(X_valid, y_valid)
            except (ValueError, KeyError, TypeError):
                continue

            # Extract exogenous gamma coefficients (columns 4 onward)
            for k, j in enumerate(exog_asset_indices):
                # gamma_matrix[t+1] because y was RV at t+1
                gamma_matrix[t + 1, i, j] = beta[4 + k]

    # ------------------------------------------------------------------
    # 3. Compute Diebold-Yilmaz spillover measures
    # ------------------------------------------------------------------
    # spillover_from[i] = sum_j |gamma_ij| (how much vol i receives)
    # spillover_to[j]   = sum_i |gamma_ij| (how much vol j transmits)
    # net_spillover[i]  = spillover_to[i] - spillover_from[i]

    abs_gamma = np.abs(gamma_matrix)

    # spillover_from[t, i] = sum over j of |gamma[t, i, j]|
    spillover_from = np.nansum(abs_gamma, axis=2)  # (n_dates, n_assets)

    # spillover_to[t, j] = sum over i of |gamma[t, i, j]|
    spillover_to = np.nansum(abs_gamma, axis=1)    # (n_dates, n_assets)

    # Normalize by total system spillover for interpretability
    total_spillover = np.nansum(abs_gamma, axis=(1, 2), keepdims=False)  # (n_dates,)
    total_spillover_safe = np.where(total_spillover > _EPS, total_spillover, np.nan)

    spillover_from_norm = spillover_from / total_spillover_safe[:, None]
    spillover_to_norm = spillover_to / total_spillover_safe[:, None]
    net_spillover = spillover_to_norm - spillover_from_norm

    # Mark dates where we had no valid regressions as NaN
    no_data = np.all(np.isnan(gamma_matrix.reshape(n_dates, -1)), axis=1)
    spillover_from_norm[no_data] = np.nan
    spillover_to_norm[no_data] = np.nan
    net_spillover[no_data] = np.nan

    # Also mask dates where the asset itself has no data
    valid_mask = rv_daily.notna().values
    spillover_from_norm[~valid_mask] = np.nan
    spillover_to_norm[~valid_mask] = np.nan
    net_spillover[~valid_mask] = np.nan

    # ------------------------------------------------------------------
    # 4. Build output MultiIndex DataFrame
    # ------------------------------------------------------------------
    panel = []
    for i, name in enumerate(asset_names):
        f = pd.DataFrame(
            {
                "harx_spillover_from": spillover_from_norm[:, i],
                "harx_spillover_to": spillover_to_norm[:, i],
                "harx_net_spillover": net_spillover[:, i],
            },
            index=dates,
        )
        f.index.name = "date"
        f["permno"] = name
        f = f.set_index("permno", append=True).reorder_levels([1, 0])
        panel.append(f)

    if not panel:
        return pd.DataFrame()

    out = pd.concat(panel).sort_index()
    return out.replace([np.inf, -np.inf], np.nan)
