"""
Performance Attribution --- decompose portfolio returns into market, factor, and alpha.

Implements Brinson-style return attribution:

    total_return = benchmark_return + active_return
    active_return = market_contribution + factor_contribution + residual_alpha

Where:
    - market_contribution = beta * benchmark_return  (systematic exposure)
    - factor_contribution = sum of factor loadings * factor returns (if provided)
    - residual_alpha      = active_return - market_contribution - factor_contribution

This module provides:
    - ``decompose_returns``: Core decomposition into a dictionary of components.
    - ``compute_attribution_report``: Extended summary with risk-adjusted metrics.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _estimate_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """OLS beta of portfolio vs benchmark.

    beta = cov(portfolio, benchmark) / var(benchmark)
    """
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common) < 5:
        return 1.0  # Assume market-neutral fallback

    p = portfolio_returns.loc[common].values.astype(float)
    b = benchmark_returns.loc[common].values.astype(float)

    b_var = np.var(b, ddof=1)
    if b_var < 1e-14:
        return 1.0

    cov = np.cov(p, b, ddof=1)[0, 1]
    return float(cov / b_var)


def _estimate_factor_loadings(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> Dict[str, float]:
    """Multivariate OLS regression of excess returns on factor returns.

    Returns a dict mapping factor name -> loading (coefficient).
    """
    common = (
        portfolio_returns.index
        .intersection(benchmark_returns.index)
        .intersection(factor_returns.index)
    )
    if len(common) < max(10, factor_returns.shape[1] + 2):
        return {col: 0.0 for col in factor_returns.columns}

    excess = portfolio_returns.loc[common].values - benchmark_returns.loc[common].values
    X = factor_returns.loc[common].values.astype(float)
    y = excess.astype(float)

    # Add intercept
    X_with_const = np.column_stack([np.ones(len(y)), X])

    try:
        # OLS via normal equations: beta = (X'X)^-1 X'y
        coeffs = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {col: 0.0 for col in factor_returns.columns}

    # coeffs[0] is intercept (alpha), coeffs[1:] are factor loadings
    loadings: Dict[str, float] = {}
    for i, col in enumerate(factor_returns.columns):
        loadings[col] = float(coeffs[i + 1])

    return loadings


def decompose_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    factor_returns: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Decompose portfolio returns into market, factor, and alpha components.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily (or periodic) portfolio returns.
    benchmark_returns : pd.Series
        Daily benchmark returns (e.g., SPY).
    factor_returns : pd.DataFrame, optional
        Daily returns for each factor (columns = factor names).
        If provided, excess returns are regressed on these factors
        to separate factor contribution from residual alpha.

    Returns
    -------
    Dict[str, float]
        Keys:
        - ``total_return``: Cumulative portfolio return over the period.
        - ``benchmark_return``: Cumulative benchmark return.
        - ``active_return``: total_return - benchmark_return.
        - ``beta``: Portfolio beta vs benchmark.
        - ``market_contribution``: beta * benchmark_return.
        - ``factor_contribution``: Sum of factor-loading-weighted factor returns
          (0.0 if no factors provided).
        - ``residual_alpha``: active_return - market_contribution - factor_contribution.
        - ``factor_loadings``: Dict of factor name -> loading (empty if no factors).
    """
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common) < 2:
        return {
            "total_return": 0.0,
            "benchmark_return": 0.0,
            "active_return": 0.0,
            "beta": 1.0,
            "market_contribution": 0.0,
            "factor_contribution": 0.0,
            "residual_alpha": 0.0,
            "factor_loadings": {},
        }

    port = portfolio_returns.loc[common]
    bench = benchmark_returns.loc[common]

    # Cumulative returns (compounded)
    total_return = float((1 + port).prod() - 1)
    benchmark_return = float((1 + bench).prod() - 1)
    active_return = total_return - benchmark_return

    # Beta estimation
    beta = _estimate_beta(port, bench)

    # Market contribution: what the portfolio earned from systematic exposure
    market_contribution = beta * benchmark_return

    # Factor contribution
    factor_contribution = 0.0
    factor_loadings: Dict[str, float] = {}

    if factor_returns is not None and not factor_returns.empty:
        factor_common = common.intersection(factor_returns.index)
        if len(factor_common) >= max(10, factor_returns.shape[1] + 2):
            factor_loadings = _estimate_factor_loadings(port, bench, factor_returns)

            # Factor contribution = sum(loading_i * cumulative_factor_return_i)
            for factor_name, loading in factor_loadings.items():
                if factor_name in factor_returns.columns:
                    cum_factor = float(
                        (1 + factor_returns.loc[factor_common, factor_name]).prod() - 1
                    )
                    factor_contribution += loading * cum_factor

    # Residual alpha: what cannot be explained by market or factors
    residual_alpha = active_return - market_contribution - factor_contribution

    return {
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "active_return": active_return,
        "beta": beta,
        "market_contribution": market_contribution,
        "factor_contribution": factor_contribution,
        "residual_alpha": residual_alpha,
        "factor_loadings": factor_loadings,
    }


def compute_attribution_report(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    factor_returns: Optional[pd.DataFrame] = None,
    annual_trading_days: int = 252,
) -> Dict[str, object]:
    """Produce an extended attribution summary with risk-adjusted metrics.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.
    benchmark_returns : pd.Series
        Daily benchmark returns.
    factor_returns : pd.DataFrame, optional
        Daily factor returns for multi-factor decomposition.
    annual_trading_days : int
        Number of trading days per year for annualization.

    Returns
    -------
    Dict[str, object]
        All fields from ``decompose_returns`` plus:
        - ``annualized_active_return``
        - ``annualized_portfolio_return``
        - ``annualized_benchmark_return``
        - ``tracking_error``: Annualized std of active returns.
        - ``information_ratio``: Annualized active return / tracking error.
        - ``portfolio_sharpe``: Annualized portfolio return / vol (assumes Rf ~ 0).
        - ``benchmark_sharpe``: Annualized benchmark return / vol.
        - ``n_periods``: Number of overlapping return observations.
    """
    decomposition = decompose_returns(
        portfolio_returns, benchmark_returns, factor_returns
    )

    common = portfolio_returns.index.intersection(benchmark_returns.index)
    n_periods = len(common)

    if n_periods < 2:
        decomposition.update({
            "annualized_active_return": 0.0,
            "annualized_portfolio_return": 0.0,
            "annualized_benchmark_return": 0.0,
            "tracking_error": 0.0,
            "information_ratio": 0.0,
            "portfolio_sharpe": 0.0,
            "benchmark_sharpe": 0.0,
            "n_periods": n_periods,
        })
        return decomposition

    port = portfolio_returns.loc[common]
    bench = benchmark_returns.loc[common]
    active = port - bench

    # Annualized returns (geometric)
    years = n_periods / annual_trading_days
    ann_port = float((1 + decomposition["total_return"]) ** (1 / max(years, 1e-6)) - 1)
    ann_bench = float((1 + decomposition["benchmark_return"]) ** (1 / max(years, 1e-6)) - 1)
    ann_active = ann_port - ann_bench

    # Tracking error (annualized std of daily active returns)
    daily_te = float(active.std())
    tracking_error = daily_te * np.sqrt(annual_trading_days)

    # Information ratio
    information_ratio = (
        ann_active / tracking_error if tracking_error > 1e-10 else 0.0
    )

    # Sharpe ratios (excess over zero risk-free rate approximation)
    port_vol = float(port.std()) * np.sqrt(annual_trading_days)
    bench_vol = float(bench.std()) * np.sqrt(annual_trading_days)
    portfolio_sharpe = ann_port / port_vol if port_vol > 1e-10 else 0.0
    benchmark_sharpe = ann_bench / bench_vol if bench_vol > 1e-10 else 0.0

    decomposition.update({
        "annualized_active_return": ann_active,
        "annualized_portfolio_return": ann_port,
        "annualized_benchmark_return": ann_bench,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "portfolio_sharpe": portfolio_sharpe,
        "benchmark_sharpe": benchmark_sharpe,
        "n_periods": n_periods,
    })

    return decomposition
