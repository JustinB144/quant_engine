"""
Factor-Based Portfolio Construction — factor decomposition and exposure analysis.

Decomposes stock returns into systematic factor exposures using OLS regression.
Detects when multiple positions share the same factor risk, enabling proper
diversification and risk budgeting.

Components:
    - compute_factor_exposures: OLS regression of returns against factors
    - compute_residual_returns: strips out systematic factor risk
"""
from typing import Optional

import numpy as np
import pandas as pd


def compute_factor_exposures(
    returns: pd.DataFrame,
    factor_returns: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Estimate factor betas for each asset via OLS regression.

    For each stock (column) in *returns*, run a time-series OLS regression
    against the provided *factor_returns*.  The resulting betas describe how
    much each stock's return moves per unit of factor return.

    If *factor_returns* is ``None``, a simple market factor is synthesised as
    the equal-weighted mean of all columns in *returns*.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns with a DatetimeIndex (or similar) and one column per
        asset.  Rows are time periods.
    factor_returns : pd.DataFrame, optional
        Factor returns with the same index frequency as *returns*.  Each column
        is one factor (e.g. "market", "size", "value").  If ``None``, an
        equal-weighted market factor is constructed automatically.

    Returns
    -------
    pd.DataFrame
        Factor betas with one row per asset (``returns.columns``) and one
        column per factor (``factor_returns.columns``).

    Raises
    ------
    ValueError
        If *returns* has fewer than 10 observations or no columns.

    Notes
    -----
    The regression includes an intercept (alpha) but only betas are returned.
    Missing values are forward-filled then back-filled before regression;
    remaining NaNs are dropped.
    """
    if returns.shape[0] < 10:
        raise ValueError(
            f"returns must have >= 10 observations, got {returns.shape[0]}"
        )
    if returns.shape[1] == 0:
        raise ValueError("returns DataFrame has no columns")

    # Clean returns
    clean_returns = (
        returns.replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
    )

    # Build default market factor if none provided
    if factor_returns is None:
        market = clean_returns.mean(axis=1)
        factor_returns = pd.DataFrame({"market": market}, index=clean_returns.index)
    else:
        factor_returns = (
            factor_returns.replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
        )

    # Align indices
    common_idx = clean_returns.index.intersection(factor_returns.index)
    if len(common_idx) < 10:
        raise ValueError(
            f"Fewer than 10 overlapping observations between returns and "
            f"factor_returns ({len(common_idx)} found)"
        )
    clean_returns = clean_returns.loc[common_idx]
    factor_returns = factor_returns.loc[common_idx]

    # Drop rows with any NaN across both DataFrames
    combined = pd.concat([clean_returns, factor_returns], axis=1)
    valid_mask = combined.notna().all(axis=1)
    clean_returns = clean_returns.loc[valid_mask]
    factor_returns = factor_returns.loc[valid_mask]

    if len(clean_returns) < 10:
        raise ValueError(
            f"Fewer than 10 valid observations after cleaning ({len(clean_returns)} found)"
        )

    # OLS regression: Y = X @ beta + epsilon
    # X includes intercept column
    n_factors = factor_returns.shape[1]
    X = np.column_stack([np.ones(len(factor_returns)), factor_returns.values])
    # X is (T, 1 + n_factors)

    # Use numerically stable least-squares solver instead of explicit
    # matrix inversion (handles ill-conditioned / rank-deficient matrices).
    betas = {}
    for asset in clean_returns.columns:
        y = clean_returns[asset].values
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        # coeffs[0] is intercept (alpha), coeffs[1:] are factor betas
        betas[asset] = coeffs[1:]

    beta_df = pd.DataFrame(
        betas,
        index=factor_returns.columns,
    ).T
    beta_df.index.name = "asset"

    return beta_df


def compute_residual_returns(
    returns: pd.DataFrame,
    factor_returns: Optional[pd.DataFrame] = None,
    factor_betas: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Strip out systematic factor exposure, returning idiosyncratic returns.

    Residual returns are computed as:

        ``residual_i(t) = return_i(t) - sum_k[ beta_ik * factor_k(t) ]``

    If *factor_betas* is not provided, they are estimated via
    :func:`compute_factor_exposures`.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (T x N).
    factor_returns : pd.DataFrame, optional
        Factor returns (T x K).  If ``None``, an equal-weighted market factor
        is constructed from *returns*.
    factor_betas : pd.DataFrame, optional
        Pre-computed factor betas (N x K) from :func:`compute_factor_exposures`.
        If ``None``, betas are estimated from *returns* and *factor_returns*.

    Returns
    -------
    pd.DataFrame
        Residual (idiosyncratic) returns with the same shape and index as
        *returns*.
    """
    # Clean returns
    clean_returns = (
        returns.replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
    )

    # Build default market factor if none provided
    if factor_returns is None:
        market = clean_returns.mean(axis=1)
        factor_returns = pd.DataFrame({"market": market}, index=clean_returns.index)
    else:
        factor_returns = (
            factor_returns.replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
        )

    # Estimate betas if not provided
    if factor_betas is None:
        factor_betas = compute_factor_exposures(returns, factor_returns)

    # Align indices
    common_idx = clean_returns.index.intersection(factor_returns.index)
    clean_returns = clean_returns.loc[common_idx]
    factor_returns = factor_returns.loc[common_idx]

    # Compute systematic component: factor_returns @ betas.T
    # factor_returns is (T, K), betas is (N, K) -> systematic is (T, N)
    common_assets = [a for a in clean_returns.columns if a in factor_betas.index]
    common_factors = [
        f for f in factor_returns.columns if f in factor_betas.columns
    ]

    if not common_assets or not common_factors:
        return clean_returns.copy()

    betas_aligned = factor_betas.loc[common_assets, common_factors]
    factors_aligned = factor_returns[common_factors]

    systematic = factors_aligned.values @ betas_aligned.values.T
    systematic_df = pd.DataFrame(
        systematic,
        index=clean_returns.index,
        columns=common_assets,
    )

    # Residual = actual - systematic
    residual = clean_returns[common_assets] - systematic_df

    # For assets without factor data, keep original returns
    result = clean_returns.copy()
    result[common_assets] = residual

    return result
