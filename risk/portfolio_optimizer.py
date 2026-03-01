"""
Mean-Variance Portfolio Optimization — turnover-penalised portfolio construction.

Implements classical Markowitz optimisation with practical enhancements:
    - Turnover penalty to control transaction costs
    - Per-position size limits (long and short)
    - Portfolio-level volatility constraint
    - Sector neutrality constraints (NEW 12)
    - SLSQP-based quadratic optimisation via scipy

Components:
    - optimize_portfolio: full mean-variance optimizer with constraints
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# Mutable flag to log GICS_SECTORS warning only once per process.
_WARNED_GICS_EMPTY = [False]


def optimize_portfolio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    current_weights: Optional[pd.Series] = None,
    max_position: float = 0.10,
    max_portfolio_vol: float = 0.30,
    turnover_penalty: float = 0.001,
    risk_aversion: float = 1.0,
    sector_map: Optional[Dict[str, str]] = None,
    max_sector_exposure: Optional[float] = None,
) -> pd.Series:
    """Find optimal portfolio weights via mean-variance optimization.

    Solves the following optimisation problem:

        maximise  w' * mu  -  0.5 * gamma * w' * Sigma * w  -  lambda * |w - w_old|

    subject to:

        - sum(w) = 1
        - -max_position <= w_i <= max_position  for all i
        - sqrt(w' * Sigma * w * 252) <= max_portfolio_vol
        - abs(sum of weights in sector s) <= max_sector_exposure  for all s

    where ``mu`` is expected returns, ``Sigma`` is the covariance matrix,
    ``gamma`` is risk aversion, and ``lambda`` is the turnover penalty.

    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns for each asset.  Index values are asset identifiers.
    covariance : pd.DataFrame
        Covariance matrix of asset returns (daily scale).  Index and columns
        must align with *expected_returns*.
    current_weights : pd.Series, optional
        Current portfolio weights.  If ``None``, assumed to be zero
        (i.e. building a portfolio from scratch).
    max_position : float, default 0.10
        Maximum absolute weight per position.  Applies symmetrically to
        long and short positions: ``-max_position <= w_i <= max_position``.
    max_portfolio_vol : float, default 0.30
        Maximum annualised portfolio volatility.  The optimiser constrains
        ``sqrt(252 * w' Sigma w) <= max_portfolio_vol``.
    turnover_penalty : float, default 0.001
        Penalty per unit of absolute turnover.  Discourages excessive
        trading by penalising deviations from *current_weights*.
    risk_aversion : float, default 1.0
        Risk aversion parameter (gamma).  Higher values produce more
        conservative (lower variance) portfolios.
    sector_map : dict[str, str], optional
        Mapping of asset identifier to GICS sector name.  When provided
        together with *max_sector_exposure*, the optimiser constrains the
        absolute net weight in each sector.  Assets not present in the map
        are treated as belonging to an "Unknown" sector.
    max_sector_exposure : float, optional
        Maximum absolute net weight allowed in any single sector.  For
        example, 0.10 means no sector can have more than +/-10% net
        exposure.  Defaults to ``config.MAX_SECTOR_EXPOSURE`` when
        *sector_map* is provided and this parameter is ``None``.

    Returns
    -------
    pd.Series
        Optimal portfolio weights indexed by asset identifier.  Weights
        sum to 1 and respect position limits.

    Notes
    -----
    Because the turnover penalty introduces an L1 term (non-smooth), we
    reformulate it using auxiliary slack variables to keep the problem
    smooth for SLSQP.  Specifically, for each asset *i* we introduce
    ``t_i >= 0`` such that ``t_i >= w_i - w_old_i`` and
    ``t_i >= -(w_i - w_old_i)``.  The turnover penalty becomes
    ``lambda * sum(t_i)``, which is linear and smooth.

    Sector constraints are formulated as two inequality constraints per
    sector: ``sum_i_in_s(w_i) <= max_sector_exposure`` and
    ``-sum_i_in_s(w_i) <= max_sector_exposure``, which together enforce
    ``abs(sum_i_in_s(w_i)) <= max_sector_exposure``.
    """
    # Align assets across expected_returns and covariance
    assets = sorted(
        set(expected_returns.index) & set(covariance.index) & set(covariance.columns)
    )
    if len(assets) == 0:
        return pd.Series(dtype=float)

    n = len(assets)
    mu = expected_returns.loc[assets].values.astype(float)
    cov = covariance.loc[assets, assets].values.astype(float)

    # Ensure covariance is symmetric and PSD
    cov = 0.5 * (cov + cov.T)
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 0:
        cov += (abs(eigvals.min()) + 1e-8) * np.eye(n)

    # Current weights
    if current_weights is not None:
        w_old = np.array(
            [float(current_weights.get(a, 0.0)) for a in assets], dtype=float
        )
    else:
        w_old = np.zeros(n, dtype=float)

    # Decision variables: x = [w_1, ..., w_n, t_1, ..., t_n]
    # where t_i are slack variables for the L1 turnover penalty
    n_vars = 2 * n

    def objective(x: np.ndarray) -> float:
        """Negative of (expected return - risk - turnover cost)."""
        w = x[:n]
        t = x[n:]
        port_return = mu @ w
        port_variance = w @ cov @ w
        turnover_cost = turnover_penalty * np.sum(t)
        # We minimise, so negate the objective
        return -(port_return - 0.5 * risk_aversion * port_variance - turnover_cost)

    def grad_objective(x: np.ndarray) -> np.ndarray:
        """Gradient of the objective."""
        w = x[:n]
        g = np.zeros(n_vars)
        # d/dw of -(mu'w - 0.5*gamma*w'Cov*w - lambda*sum(t))
        g[:n] = -(mu - risk_aversion * cov @ w)
        # d/dt of -(-lambda*sum(t))
        g[n:] = turnover_penalty
        return g

    # Constraints
    constraints = []

    # 1. Weights sum to 1
    def weight_sum(x: np.ndarray) -> float:
        return float(np.sum(x[:n]) - 1.0)

    constraints.append({"type": "eq", "fun": weight_sum})

    # 2. Volatility constraint: max_vol^2 / 252 - w' Sigma w >= 0
    def vol_constraint(x: np.ndarray) -> float:
        w = x[:n]
        port_var = w @ cov @ w
        max_var = (max_portfolio_vol ** 2) / 252.0
        return float(max_var - port_var)

    constraints.append({"type": "ineq", "fun": vol_constraint})

    # 3. Slack variable constraints: t_i >= w_i - w_old_i and t_i >= -(w_i - w_old_i)
    for i in range(n):
        # t_i - (w_i - w_old_i) >= 0
        def slack_upper(x: np.ndarray, _i: int = i) -> float:
            return float(x[n + _i] - (x[_i] - w_old[_i]))

        # t_i + (w_i - w_old_i) >= 0  =>  t_i - (w_old_i - w_i) >= 0
        def slack_lower(x: np.ndarray, _i: int = i) -> float:
            return float(x[n + _i] + (x[_i] - w_old[_i]))

        constraints.append({"type": "ineq", "fun": slack_upper})
        constraints.append({"type": "ineq", "fun": slack_lower})

    # 4. Sector neutrality constraints: |sum of weights in sector| <= max_sector_exposure
    #    When sector_map is not explicitly provided, fall back to GICS_SECTORS from config.
    if sector_map is None or len(sector_map) == 0:
        from ..config import GICS_SECTORS
        if GICS_SECTORS:
            sector_map = GICS_SECTORS
        elif not _WARNED_GICS_EMPTY[0]:
            logger.warning(
                "GICS_SECTORS is empty — sector exposure constraint (%.0f%%) is NOT enforced",
                max_sector_exposure * 100 if max_sector_exposure is not None else 10,
            )
            _WARNED_GICS_EMPTY[0] = True
    if sector_map is not None and len(sector_map) > 0:
        # Resolve the effective max sector exposure.
        if max_sector_exposure is None:
            from ..config import MAX_SECTOR_EXPOSURE
            effective_sector_limit = MAX_SECTOR_EXPOSURE
        else:
            effective_sector_limit = max_sector_exposure

        # Group asset indices by sector.
        sector_to_indices: Dict[str, list] = {}
        for idx, asset in enumerate(assets):
            sector = sector_map.get(asset, "Unknown")
            sector_to_indices.setdefault(sector, []).append(idx)

        for sector, indices in sector_to_indices.items():
            # Upper bound: sum_i_in_s(w_i) <= max_sector_exposure
            # Reformulated as: max_sector_exposure - sum_i_in_s(w_i) >= 0
            def sector_upper(x: np.ndarray, _idx: list = indices) -> float:
                return float(effective_sector_limit - np.sum(x[_idx]))

            # Lower bound: -sum_i_in_s(w_i) <= max_sector_exposure
            # Reformulated as: max_sector_exposure + sum_i_in_s(w_i) >= 0
            def sector_lower(x: np.ndarray, _idx: list = indices) -> float:
                return float(effective_sector_limit + np.sum(x[_idx]))

            constraints.append({"type": "ineq", "fun": sector_upper})
            constraints.append({"type": "ineq", "fun": sector_lower})

    # Bounds: w_i in [-max_position, max_position], t_i in [0, inf)
    bounds = []
    for _ in range(n):
        bounds.append((-max_position, max_position))
    for _ in range(n):
        bounds.append((0.0, None))

    # Initial guess: equal weight for w, zero for t
    w0_equal = np.full(n, 1.0 / n)
    # Clip to bounds
    w0_equal = np.clip(w0_equal, -max_position, max_position)
    # Re-normalize so weights sum to 1
    w0_sum = w0_equal.sum()
    if abs(w0_sum) > 1e-10:
        w0_equal = w0_equal / w0_sum
    w0_equal = np.clip(w0_equal, -max_position, max_position)

    t0 = np.abs(w0_equal - w_old)
    x0 = np.concatenate([w0_equal, t0])

    result = minimize(
        fun=objective,
        x0=x0,
        jac=grad_objective,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
    )

    if not result.success:
        logger.warning(
            "Portfolio optimizer did not converge: %s. Falling back to constrained equal weight.",
            result.message,
        )
        equal_w = np.full(n, 1.0 / n)
        # Clip to max_position constraint
        if max_position is not None and max_position > 0:
            equal_w = np.minimum(equal_w, max_position)
        # Re-normalize to sum to 1, then iteratively project to respect max_position
        w_sum = equal_w.sum()
        if w_sum > 1e-10:
            equal_w = equal_w / w_sum
        if max_position is not None and max_position > 0:
            for _ in range(5):  # Iterative projection
                equal_w = np.minimum(equal_w, max_position)
                w_sum = equal_w.sum()
                if w_sum > 1e-10:
                    equal_w = equal_w / w_sum
                if np.all(equal_w <= max_position + 1e-8):
                    break
        return pd.Series(equal_w, index=assets, name="weight")

    optimal_weights = result.x[:n]

    # Normalise weights to ensure they sum exactly to 1
    w_sum = optimal_weights.sum()
    if abs(w_sum) > 1e-10:
        optimal_weights = optimal_weights / w_sum

    # Clip tiny weights to zero for cleanliness
    optimal_weights[np.abs(optimal_weights) < 1e-6] = 0.0

    # Re-normalise after clipping
    w_sum = optimal_weights.sum()
    if abs(w_sum) > 1e-10:
        optimal_weights = optimal_weights / w_sum

    # Re-clip any weights that now exceed max_position after renormalization
    for _ in range(5):  # Iterate to convergence (usually 1-2 passes)
        violations = np.abs(optimal_weights) > max_position
        if not violations.any():
            break
        optimal_weights[violations] = np.sign(optimal_weights[violations]) * max_position
        # Redistribute excess to non-capped weights
        excess = 1.0 - optimal_weights.sum()
        non_capped = ~violations & (np.abs(optimal_weights) > 0)
        if non_capped.any():
            non_capped_sum = np.abs(optimal_weights[non_capped]).sum()
            if non_capped_sum > 1e-10:
                optimal_weights[non_capped] += excess * (optimal_weights[non_capped] / non_capped_sum)

    assert np.all(np.abs(optimal_weights) <= max_position + 1e-8), \
        f"Weight exceeds max_position after normalization: {optimal_weights}"

    return pd.Series(optimal_weights, index=assets, name="weight")
