"""
Implied Volatility Surface Models.

Implements three core volatility models for options analysis:
    1. Black-Scholes: Analytical pricing, Greeks, IV solving
    2. Heston: Stochastic volatility with characteristic function pricing
    3. SVI: Stochastic Volatility Inspired smile parameterization

All models support surface generation for 3D visualization.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize, integrate, interpolate, stats


class OptionType(Enum):
    """Supported option contract types for pricing and volatility surface models."""
    CALL = "call"
    PUT = "put"


@dataclass
class Greeks:
    """Option Greeks container."""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass
class HestonParams:
    """Heston model parameters."""
    v0: float      # Initial variance
    theta: float   # Long-run variance
    kappa: float   # Mean reversion speed
    sigma: float   # Vol of vol
    rho: float     # Correlation between spot and vol

    def validate(self) -> bool:
        """Check Feller condition: 2*kappa*theta > sigma^2."""
        return 2 * self.kappa * self.theta > self.sigma ** 2


@dataclass
class SVIParams:
    """Raw SVI parameterization: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    a: float       # Vertical translation (overall variance level)
    b: float       # Angle between left and right asymptotes
    rho: float     # Rotation / skew (-1 < rho < 1)
    m: float       # Horizontal translation (smile center)
    sigma: float   # Smoothing at the money (ATM curvature)


class BlackScholes:
    """Black-Scholes option pricing and analytics."""

    @staticmethod
    def price(S, K, T, r, sigma, q=0.0, option_type=OptionType.CALL):
        """European option price via Black-Scholes."""
        # Handle edge cases
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            if T <= 0:
                payoff = max(S - K, 0) if option_type == OptionType.CALL else max(K - S, 0)
                return payoff
            if S <= 0 or K <= 0:
                return 0.0
            # sigma <= 0 with T > 0: deterministic case, return discounted intrinsic
            if option_type == OptionType.CALL:
                return max(0.0, S * np.exp(-q * T) - K * np.exp(-r * T))
            else:
                return max(0.0, K * np.exp(-r * T) - S * np.exp(-q * T))

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == OptionType.CALL:
            return S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)

    @staticmethod
    def greeks(S, K, T, r, sigma, q=0.0, option_type=OptionType.CALL):
        """Compute all Greeks."""
        if T <= 1e-10 or sigma <= 1e-10:
            return Greeks(delta=0, gamma=0, vega=0, theta=0, rho=0)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        pdf_d1 = stats.norm.pdf(d1)
        sqrt_T = np.sqrt(T)

        gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * sqrt_T)
        vega = S * np.exp(-q * T) * pdf_d1 * sqrt_T / 100  # per 1% move

        if option_type == OptionType.CALL:
            delta = np.exp(-q * T) * stats.norm.cdf(d1)
            theta = (-(S * sigma * np.exp(-q * T) * pdf_d1) / (2 * sqrt_T)
                     - r * K * np.exp(-r * T) * stats.norm.cdf(d2)
                     + q * S * np.exp(-q * T) * stats.norm.cdf(d1)) / 365
            rho_val = K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100
        else:
            delta = np.exp(-q * T) * (stats.norm.cdf(d1) - 1)
            theta = (-(S * sigma * np.exp(-q * T) * pdf_d1) / (2 * sqrt_T)
                     + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
                     - q * S * np.exp(-q * T) * stats.norm.cdf(-d1)) / 365
            rho_val = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho_val)

    @staticmethod
    def implied_vol(price, S, K, T, r, q=0.0, option_type=OptionType.CALL,
                    tol=1e-8, max_iter=100):
        """Solve for implied volatility using Brent's method with Newton warm-start."""
        if T <= 0 or price <= 0:
            return np.nan

        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if option_type == OptionType.CALL \
                    else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
        if price < intrinsic - 1e-10:
            return np.nan

        def objective(sigma):
            return BlackScholes.price(S, K, T, r, sigma, q, option_type) - price

        try:
            result = optimize.brentq(objective, 1e-6, 10.0, xtol=tol, maxiter=max_iter)
            return result
        except (ValueError, RuntimeError):
            return np.nan

    @staticmethod
    def iv_surface(S, strikes, expiries, r=0.05, q=0.0, base_vol=0.20):
        """Generate a BS IV surface (flat, used as baseline reference)."""
        K_grid, T_grid = np.meshgrid(strikes, expiries)
        iv_grid = np.full_like(K_grid, base_vol, dtype=float)
        return K_grid, T_grid, iv_grid


class HestonModel:
    """
    Heston (1993) stochastic volatility model.

    dS = (r-q)*S*dt + sqrt(v)*S*dW1
    dv = kappa*(theta - v)*dt + sigma*sqrt(v)*dW2
    dW1*dW2 = rho*dt

    Pricing via characteristic function and numerical integration
    (Albrecher et al. formulation for numerical stability).
    """

    def __init__(self, params: Optional[HestonParams] = None):
        """Initialize HestonModel."""
        self.params = params or HestonParams(v0=0.04, theta=0.04, kappa=1.5, sigma=0.3, rho=-0.7)

    def characteristic_function(self, u, T, r, q=0.0):
        """Heston characteristic function phi(u) for log-spot."""
        p = self.params
        # Use the Albrecher et al. rotation for numerical stability
        xi = p.kappa - p.sigma * p.rho * 1j * u
        d = np.sqrt(xi**2 + p.sigma**2 * (u * 1j + u**2))

        # Ensure proper branch selection
        g = (xi - d) / (xi + d)

        exp_dT = np.exp(-d * T)

        C = (r - q) * 1j * u * T + (p.kappa * p.theta / p.sigma**2) * (
            (xi - d) * T - 2 * np.log((1 - g * exp_dT) / (1 - g))
        )
        D = ((xi - d) / p.sigma**2) * ((1 - exp_dT) / (1 - g * exp_dT))

        return np.exp(C + D * p.v0)

    def price(self, S, K, T, r, q=0.0, option_type=OptionType.CALL):
        """Price European option via numerical integration of characteristic function."""
        if T <= 0 or S <= 0 or K <= 0:
            if T <= 0:
                payoff = max(S - K, 0) if option_type == OptionType.CALL else max(K - S, 0)
                return payoff
            return 0.0

        log_ratio = np.log(S / K)

        def integrand_P1(u):
            """First probability integral."""
            cf = self.characteristic_function(u - 1j, T, r, q)
            cf_0 = self.characteristic_function(-1j, T, r, q)
            if abs(cf_0) < 1e-20:
                return 0.0
            return np.real(np.exp(-1j * u * log_ratio) * cf / (1j * u * cf_0))

        def integrand_P2(u):
            """Second probability integral."""
            cf = self.characteristic_function(u, T, r, q)
            return np.real(np.exp(-1j * u * log_ratio) * cf / (1j * u))

        # Numerical integration
        limit = 100
        P1 = 0.5 + (1 / np.pi) * integrate.quad(integrand_P1, 1e-8, limit, limit=200)[0]
        P2 = 0.5 + (1 / np.pi) * integrate.quad(integrand_P2, 1e-8, limit, limit=200)[0]

        P1 = np.clip(P1, 0, 1)
        P2 = np.clip(P2, 0, 1)

        call_price = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
        call_price = max(call_price, 0)

        if option_type == OptionType.CALL:
            return call_price
        else:
            # Put-call parity
            return call_price - S * np.exp(-q * T) + K * np.exp(-r * T)

    def implied_vol(self, S, K, T, r, q=0.0, option_type=OptionType.CALL):
        """Compute BS-equivalent implied vol from Heston price."""
        heston_price = self.price(S, K, T, r, q, option_type)
        return BlackScholes.implied_vol(heston_price, S, K, T, r, q, option_type)

    def iv_surface(self, S, strikes, expiries, r=0.05, q=0.0):
        """Generate the Heston implied volatility surface."""
        n_K = len(strikes)
        n_T = len(expiries)
        iv_grid = np.full((n_T, n_K), np.nan)

        for i, T in enumerate(expiries):
            for j, K in enumerate(strikes):
                try:
                    iv = self.implied_vol(S, K, T, r, q)
                    iv_grid[i, j] = iv if np.isfinite(iv) else np.nan
                except (ValueError, RuntimeError):
                    iv_grid[i, j] = np.nan

        K_grid, T_grid = np.meshgrid(strikes, expiries)

        # Interpolate NaN values for smooth surface
        from scipy.interpolate import griddata
        valid = ~np.isnan(iv_grid)
        if valid.sum() > 3:
            points = np.column_stack((K_grid[valid], T_grid[valid]))
            values = iv_grid[valid]
            nan_mask = np.isnan(iv_grid)
            if nan_mask.any():
                nan_points = np.column_stack((K_grid[nan_mask], T_grid[nan_mask]))
                iv_grid[nan_mask] = griddata(points, values, nan_points, method='nearest')

        return K_grid, T_grid, iv_grid

    def calibrate(self, market_ivs, strikes, expiries, S, r=0.05, q=0.0):
        """
        Calibrate Heston parameters to market implied volatilities.

        Uses differential evolution for global search followed by
        Nelder-Mead for local refinement.
        """
        def objective(x):
            params = HestonParams(v0=x[0], theta=x[1], kappa=x[2], sigma=x[3], rho=x[4])
            self.params = params

            total_err = 0.0
            count = 0
            for i, T in enumerate(expiries):
                for j, K in enumerate(strikes):
                    if np.isnan(market_ivs[i, j]):
                        continue
                    try:
                        model_iv = self.implied_vol(S, K, T, r, q)
                        if np.isfinite(model_iv):
                            total_err += (model_iv - market_ivs[i, j])**2
                            count += 1
                    except (ValueError, RuntimeError):
                        total_err += 1.0
                        count += 1

            return total_err / max(count, 1)

        bounds = [
            (0.001, 1.0),   # v0
            (0.001, 1.0),   # theta
            (0.1, 10.0),    # kappa
            (0.05, 2.0),    # sigma
            (-0.99, 0.99),  # rho
        ]

        result = optimize.differential_evolution(
            objective, bounds, seed=42, maxiter=100, tol=1e-6, polish=True,
        )

        self.params = HestonParams(
            v0=result.x[0], theta=result.x[1], kappa=result.x[2],
            sigma=result.x[3], rho=result.x[4],
        )
        return self.params, result.fun


class SVIModel:
    """
    SVI (Stochastic Volatility Inspired) implied variance parameterization.

    Gatheral (2004): w(k) = a + b * (rho*(k - m) + sqrt((k - m)^2 + sigma^2))

    where:
        w = sigma_BS^2 * T  (total implied variance)
        k = log(K/F)        (log forward moneyness)
    """

    def __init__(self, params: Optional[SVIParams] = None):
        """Initialize SVIModel."""
        self.params = params or SVIParams(a=0.04, b=0.1, rho=-0.4, m=0.0, sigma=0.1)

    def total_variance(self, k):
        """Compute total implied variance w(k) for log-moneyness k."""
        p = self.params
        return p.a + p.b * (p.rho * (k - p.m) + np.sqrt((k - p.m)**2 + p.sigma**2))

    def implied_vol(self, k, T):
        """Compute implied volatility from SVI total variance."""
        w = self.total_variance(k)
        w = np.maximum(w, 1e-10)  # Ensure positive variance
        return np.sqrt(w / T)

    def iv_surface(self, S, strikes, expiries, r=0.05, q=0.0):
        """Generate the SVI implied volatility surface."""
        F_values = S * np.exp((r - q) * np.array(expiries))

        n_K = len(strikes)
        n_T = len(expiries)
        iv_grid = np.full((n_T, n_K), np.nan)

        for i, T in enumerate(expiries):
            F = F_values[i]
            for j, K in enumerate(strikes):
                k = np.log(K / F)
                try:
                    iv = self.implied_vol(k, T)
                    iv_grid[i, j] = iv if np.isfinite(iv) else np.nan
                except (ValueError, RuntimeError):
                    iv_grid[i, j] = np.nan

        K_grid, T_grid = np.meshgrid(strikes, expiries)
        return K_grid, T_grid, iv_grid

    def smile(self, T, S, r=0.05, q=0.0, n_strikes=50):
        """Generate a single smile curve for expiry T."""
        F = S * np.exp((r - q) * T)
        k_range = np.linspace(-0.5, 0.5, n_strikes)
        strikes = F * np.exp(k_range)
        ivs = np.array([self.implied_vol(k, T) for k in k_range])
        return strikes, ivs

    def calibrate(self, market_ivs, strikes, T, S, r=0.05, q=0.0):
        """Calibrate SVI parameters to a single expiry smile."""
        F = S * np.exp((r - q) * T)
        k_values = np.log(np.array(strikes) / F)
        target_w = np.array(market_ivs)**2 * T

        def objective(x):
            params = SVIParams(a=x[0], b=x[1], rho=x[2], m=x[3], sigma=x[4])
            self.params = params
            w_model = np.array([self.total_variance(k) for k in k_values])
            valid = np.isfinite(target_w) & np.isfinite(w_model)
            if valid.sum() == 0:
                return 1e10
            return np.sum((w_model[valid] - target_w[valid])**2)

        bounds = [
            (-0.5, 0.5),    # a
            (0.01, 1.0),    # b
            (-0.99, 0.99),  # rho
            (-0.5, 0.5),    # m
            (0.01, 1.0),    # sigma
        ]

        result = optimize.differential_evolution(
            objective, bounds, seed=42, maxiter=200, tol=1e-8,
        )

        self.params = SVIParams(
            a=result.x[0], b=result.x[1], rho=result.x[2],
            m=result.x[3], sigma=result.x[4],
        )
        return self.params, result.fun

    def check_no_butterfly_arbitrage(self):
        """Verify no-butterfly-arbitrage conditions on SVI parameters."""
        p = self.params
        conditions = {
            "b >= 0": p.b >= 0,
            "|rho| < 1": abs(p.rho) < 1,
            "sigma > 0": p.sigma > 0,
            "a + b*sigma*sqrt(1-rho^2) >= 0": (
                p.a + p.b * p.sigma * np.sqrt(1 - p.rho**2) >= 0
            ),
        }
        return conditions


class ArbitrageFreeSVIBuilder:
    """Arbitrage-aware SVI surface builder.

    Methodology follows the practical guidance in arXiv:1107.1834:
    - fit each maturity in total variance space ``w = sigma^2 T``;
    - use weighted least squares (vega/spread-inspired weights);
    - enforce no-butterfly admissibility on each SVI slice;
    - enforce no-calendar arbitrage by monotone total variance in maturity;
    - interpolate in total variance (not implied volatility).
    """

    def __init__(
        self,
        penalty_weight: float = 500.0,
        calendar_epsilon: float = 1e-7,
        max_iter: int = 300,
    ):
        """Initialize ArbitrageFreeSVIBuilder."""
        self.penalty_weight = float(max(1.0, penalty_weight))
        self.calendar_epsilon = float(max(0.0, calendar_epsilon))
        self.max_iter = int(max(50, max_iter))

    @staticmethod
    def _svi_total_variance(k: np.ndarray, params: SVIParams) -> np.ndarray:
        """Internal helper for svi total variance."""
        return params.a + params.b * (
            params.rho * (k - params.m) + np.sqrt((k - params.m) ** 2 + params.sigma ** 2)
        )

    @staticmethod
    def _initial_guess(k: np.ndarray, target_w: np.ndarray) -> np.ndarray:
        # Robust deterministic start from slice moments.
        """Internal helper for initial guess."""
        k = np.asarray(k, dtype=float)
        target_w = np.asarray(target_w, dtype=float)
        atm_idx = int(np.argmin(np.abs(k)))
        a0 = float(np.clip(target_w[atm_idx] * 0.75, 1e-6, 2.0))
        width = max(float(np.nanmax(k) - np.nanmin(k)), 1e-3)
        b0 = float(np.clip(np.nanstd(target_w) / width, 0.01, 2.0))
        rho0 = -0.3
        m0 = float(np.clip(np.nanmean(k), -1.0, 1.0))
        sigma0 = float(np.clip(width * 0.2, 0.02, 0.8))
        return np.array([a0, b0, rho0, m0, sigma0], dtype=float)

    @staticmethod
    def _vega_spread_weights(
        strikes: np.ndarray,
        ivs: np.ndarray,
        expiry: float,
        spot: float,
        r: float,
        q: float,
    ) -> np.ndarray:
        """Approximate bid-ask/vega weighting when quotes are unavailable."""
        T = max(float(expiry), 1e-6)
        K = np.asarray(strikes, dtype=float)
        sigma = np.clip(np.asarray(ivs, dtype=float), 0.01, 3.0)
        F = spot * np.exp((r - q) * T)
        d1 = (np.log(np.maximum(F, 1e-12) / np.maximum(K, 1e-12)) + 0.5 * sigma**2 * T) / (
            sigma * np.sqrt(T)
        )
        vega = spot * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T)
        # Synthetic spread proxy in volatility points.
        spread = np.maximum(0.0025, 0.10 * sigma)
        w = vega / spread
        w = np.clip(w, 1e-4, 1e6)
        return w / max(np.mean(w), 1e-8)

    def _slice_objective(
        self,
        x: np.ndarray,
        k: np.ndarray,
        target_w: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Internal helper for slice objective."""
        p = SVIParams(a=float(x[0]), b=float(x[1]), rho=float(x[2]), m=float(x[3]), sigma=float(x[4]))
        w_model = self._svi_total_variance(k, p)

        err = float(np.average((w_model - target_w) ** 2, weights=weights))

        # No-butterfly constraints and Lee wing bounds (practical penalties).
        penalties = 0.0
        butterfly_lb = p.a + p.b * p.sigma * np.sqrt(max(1.0 - p.rho**2, 1e-10))
        if butterfly_lb < 0:
            penalties += (abs(butterfly_lb) + 1e-8) ** 2

        wing = p.b * (1.0 + abs(p.rho))
        if wing > 4.0:
            penalties += (wing - 4.0) ** 2

        min_w = float(np.min(w_model))
        if min_w < 1e-10:
            penalties += (1e-10 - min_w) ** 2 * 25.0

        return err + self.penalty_weight * penalties

    def fit_slice(
        self,
        strikes: np.ndarray,
        ivs: np.ndarray,
        expiry: float,
        spot: float,
        r: float = 0.05,
        q: float = 0.0,
    ) -> Tuple[SVIParams, float]:
        """Fit one SVI smile using weighted total-variance loss."""
        K = np.asarray(strikes, dtype=float)
        sigma = np.asarray(ivs, dtype=float)
        T = float(expiry)
        F = spot * np.exp((r - q) * T)
        k = np.log(np.maximum(K, 1e-12) / max(F, 1e-12))
        target_w = np.clip(sigma**2 * T, 1e-10, 5.0)

        guess = self._initial_guess(k, target_w)
        bounds = [
            (-1.0, 2.0),    # a
            (1e-5, 5.0),    # b
            (-0.999, 0.999),  # rho
            (-2.0, 2.0),    # m
            (1e-5, 2.0),    # sigma
        ]

        weights = self._vega_spread_weights(K, sigma, T, spot, r, q)

        starts = [
            guess,
            np.array([guess[0] * 1.2, guess[1] * 0.8, -0.55, guess[3], max(0.03, guess[4] * 1.2)]),
            np.array([guess[0] * 0.9, guess[1] * 1.2, -0.15, guess[3] * 0.7, max(0.02, guess[4] * 0.8)]),
        ]

        best_res = None
        for x0 in starts:
            res = optimize.minimize(
                self._slice_objective,
                x0=x0,
                args=(k, target_w, weights),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.max_iter},
            )
            if best_res is None or res.fun < best_res.fun:
                best_res = res

        params = SVIParams(
            a=float(best_res.x[0]),
            b=float(best_res.x[1]),
            rho=float(best_res.x[2]),
            m=float(best_res.x[3]),
            sigma=float(best_res.x[4]),
        )
        return params, float(best_res.fun)

    @staticmethod
    def enforce_calendar_monotonicity(total_variance: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Force non-decreasing total variance in maturity for each strike."""
        out = np.asarray(total_variance, dtype=float).copy()
        if out.ndim != 2 or out.shape[0] <= 1:
            return out
        for i in range(1, out.shape[0]):
            out[i, :] = np.maximum(out[i, :], out[i - 1, :] + eps)
        return out

    @staticmethod
    def interpolate_total_variance(
        expiries: np.ndarray,
        total_variance: np.ndarray,
        query_expiries: np.ndarray,
    ) -> np.ndarray:
        """Linear maturity interpolation in total variance space."""
        expiries = np.asarray(expiries, dtype=float)
        query_expiries = np.asarray(query_expiries, dtype=float)
        tv = np.asarray(total_variance, dtype=float)
        interp = interpolate.interp1d(
            expiries,
            tv,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        return np.asarray(interp(query_expiries), dtype=float)

    def build_surface(
        self,
        spot: float,
        strikes: np.ndarray,
        expiries: np.ndarray,
        market_iv_grid: np.ndarray,
        r: float = 0.05,
        q: float = 0.0,
    ) -> Dict[str, object]:
        """Calibrate an arbitrage-aware SVI surface from market IV quotes."""
        strikes = np.asarray(strikes, dtype=float)
        expiries = np.asarray(expiries, dtype=float)
        market_iv_grid = np.asarray(market_iv_grid, dtype=float)

        if market_iv_grid.shape != (len(expiries), len(strikes)):
            raise ValueError("market_iv_grid shape must be (n_expiries, n_strikes)")
        if np.any(expiries <= 0):
            raise ValueError("All expiries must be positive")

        order = np.argsort(expiries)
        expiries_sorted = expiries[order]
        iv_sorted = market_iv_grid[order]

        fitted_params: List[SVIParams] = []
        objectives: List[float] = []
        raw_tv_rows: List[np.ndarray] = []

        prev_params = SVIParams(a=0.03, b=0.20, rho=-0.3, m=0.0, sigma=0.20)

        for i, T in enumerate(expiries_sorted):
            iv_slice = iv_sorted[i]
            valid = np.isfinite(iv_slice) & (iv_slice > 0.005)
            if valid.sum() < 5:
                params = prev_params
                obj = np.nan
            else:
                params, obj = self.fit_slice(
                    strikes=strikes[valid],
                    ivs=iv_slice[valid],
                    expiry=float(T),
                    spot=float(spot),
                    r=float(r),
                    q=float(q),
                )
                prev_params = params

            F = spot * np.exp((r - q) * T)
            k_full = np.log(np.maximum(strikes, 1e-12) / max(F, 1e-12))
            w_full = np.clip(self._svi_total_variance(k_full, params), 1e-10, 10.0)

            fitted_params.append(params)
            objectives.append(float(obj))
            raw_tv_rows.append(w_full)

        raw_tv = np.asarray(raw_tv_rows, dtype=float)
        adj_tv = self.enforce_calendar_monotonicity(raw_tv, eps=self.calendar_epsilon)

        raw_iv = np.sqrt(np.maximum(raw_tv, 1e-10) / expiries_sorted[:, None])
        adj_iv = np.sqrt(np.maximum(adj_tv, 1e-10) / expiries_sorted[:, None])

        K_grid, T_grid = np.meshgrid(strikes, expiries_sorted)
        return {
            "strikes": strikes,
            "expiries": expiries_sorted,
            "K_grid": K_grid,
            "T_grid": T_grid,
            "raw_total_variance": raw_tv,
            "adj_total_variance": adj_tv,
            "raw_iv_grid": raw_iv,
            "adj_iv_grid": adj_iv,
            "params": fitted_params,
            "objectives": objectives,
        }


def generate_synthetic_market_surface(S=100, r=0.05, q=0.01):
    """
    Generate a realistic synthetic market IV surface for demonstration.
    Uses Heston model with typical equity parameters.
    """
    heston = HestonModel(HestonParams(
        v0=0.04, theta=0.06, kappa=2.0, sigma=0.4, rho=-0.7,
    ))

    moneyness = np.linspace(0.8, 1.2, 25)
    strikes = S * moneyness
    expiries = np.array([0.083, 0.167, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

    K_grid, T_grid, iv_grid = heston.iv_surface(S, strikes, expiries, r, q)

    # Add small realistic noise
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 0.003, iv_grid.shape)
    iv_grid = np.clip(iv_grid + noise, 0.05, 1.0)

    return {
        "S": S, "r": r, "q": q,
        "strikes": strikes, "expiries": expiries,
        "moneyness": moneyness,
        "K_grid": K_grid, "T_grid": T_grid, "iv_grid": iv_grid,
    }


# ======================================================================
# IV Surface storage + interpolation
# ======================================================================

@dataclass
class IVPoint:
    """Single implied-volatility observation."""
    strike: float
    expiry: float       # time-to-expiry in years
    iv: float           # implied volatility (annualised)
    bid_iv: Optional[float] = None
    ask_iv: Optional[float] = None
    timestamp: Optional[str] = None


class IVSurface:
    """Store and interpolate an implied-volatility surface.

    The surface is represented as scattered (strike, expiry, iv) points
    that can be queried at arbitrary (K, T) via 2-D interpolation.

    Parameters
    ----------
    spot : float
        Current underlying spot price (used to convert strikes to
        log-moneyness internally).
    r : float
        Risk-free rate (used for forward-moneyness calculations).
    q : float
        Continuous dividend yield.
    """

    def __init__(self, spot: float, r: float = 0.05, q: float = 0.0):
        """Initialize IVSurface."""
        self.spot = spot
        self.r = r
        self.q = q
        self._points: List[IVPoint] = []
        self._interpolator: Optional[interpolate.CloughTocher2DInterpolator] = None
        self._dirty = True  # rebuild interpolator when data changes

    # ------------------------------------------------------------------
    # Data management
    # ------------------------------------------------------------------

    def add_point(self, strike: float, expiry: float, iv: float, **kwargs) -> None:
        """Add a single IV observation."""
        self._points.append(IVPoint(strike=strike, expiry=expiry, iv=iv, **kwargs))
        self._dirty = True

    def add_slice(self, strikes: np.ndarray, expiry: float, ivs: np.ndarray) -> None:
        """Add an entire smile (one expiry, many strikes)."""
        for k, iv in zip(strikes, ivs):
            if np.isfinite(iv) and iv > 0:
                self._points.append(IVPoint(strike=float(k), expiry=expiry, iv=float(iv)))
        self._dirty = True

    def add_surface(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,
        iv_grid: np.ndarray,
    ) -> None:
        """Add a full grid (n_expiry x n_strike) of IVs at once."""
        for i, T in enumerate(expiries):
            for j, K in enumerate(strikes):
                iv = iv_grid[i, j]
                if np.isfinite(iv) and iv > 0:
                    self._points.append(IVPoint(strike=float(K), expiry=float(T), iv=float(iv)))
        self._dirty = True

    @property
    def n_points(self) -> int:
        """n points."""
        return len(self._points)

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def _log_moneyness(self, K: float, T: float) -> float:
        """Convert strike to log forward-moneyness."""
        F = self.spot * np.exp((self.r - self.q) * T)
        return np.log(K / F)

    def _build_interpolator(self) -> None:
        """Rebuild the 2-D interpolator from stored points."""
        if not self._points:
            self._interpolator = None
            self._dirty = False
            return

        coords = np.array([
            [self._log_moneyness(p.strike, p.expiry), p.expiry]
            for p in self._points
        ])
        values = np.array([p.iv for p in self._points])

        try:
            self._interpolator = interpolate.CloughTocher2DInterpolator(
                coords, values, tol=1e-6,
            )
        except (ValueError, RuntimeError):
            # Fall back to linear interpolation if Clough-Tocher fails
            self._interpolator = interpolate.LinearNDInterpolator(
                coords, values,
            )
        self._dirty = False

    def get_iv(self, strike: float, expiry: float) -> float:
        """Interpolate IV at an arbitrary (strike, expiry) point.

        Returns ``np.nan`` if the point lies outside the convex hull
        of observed data.
        """
        if self._dirty:
            self._build_interpolator()
        if self._interpolator is None:
            return np.nan
        k = self._log_moneyness(strike, expiry)
        result = float(self._interpolator(k, expiry))
        return result if np.isfinite(result) else np.nan

    def get_smile(self, expiry: float, strikes: Optional[np.ndarray] = None,
                  n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Return an interpolated smile for a given expiry.

        Parameters
        ----------
        expiry : float
            Time-to-expiry (years).
        strikes : array, optional
            Strike prices at which to evaluate.  If ``None``, an
            evenly spaced grid between 80 % and 120 % of spot is used.
        n_points : int
            Number of points when generating a default strike grid.

        Returns
        -------
        (strikes, ivs) : tuple of arrays
        """
        if strikes is None:
            strikes = np.linspace(0.8 * self.spot, 1.2 * self.spot, n_points)
        ivs = np.array([self.get_iv(K, expiry) for K in strikes])
        return strikes, ivs

    # ------------------------------------------------------------------
    # KL-inspired decomposition
    # ------------------------------------------------------------------

    def decompose(
        self,
        expiry: float,
        strikes: Optional[np.ndarray] = None,
        n_points: int = 50,
    ) -> Dict[str, float]:
        """Extract Karhunen-Loeve-inspired principal modes from a smile.

        The first three orthogonal modes of equity IV smiles are
        well-known to be:

        1. **Level** -- the ATM implied volatility (parallel shift).
        2. **Slope** (skew) -- the gradient of IV with respect to
           log-moneyness at the money.
        3. **Curvature** (butterfly / convexity) -- the second
           derivative at the money.

        These correspond to the dominant eigenvectors of the
        covariance matrix of IV smile changes across time, as shown
        by Cont & da Fonseca (2002) and Fengler (2005).

        Parameters
        ----------
        expiry : float
            Time-to-expiry (years) of the smile to decompose.
        strikes : array, optional
            Strikes at which to evaluate.  Defaults to a grid
            around the forward.
        n_points : int
            Grid density when ``strikes`` is not provided.

        Returns
        -------
        dict
            ``level`` (ATM IV), ``slope`` (skew), ``curvature``
            (butterfly), plus ``atm_strike`` and ``forward``.
        """
        F = self.spot * np.exp((self.r - self.q) * expiry)

        if strikes is None:
            strikes = np.linspace(0.8 * self.spot, 1.2 * self.spot, n_points)

        ivs = np.array([self.get_iv(K, expiry) for K in strikes])
        k_vals = np.log(strikes / F)  # log-moneyness

        # Mask NaN values for robust fitting
        valid = np.isfinite(ivs)
        if valid.sum() < 3:
            return {
                "level": np.nan,
                "slope": np.nan,
                "curvature": np.nan,
                "atm_strike": F,
                "forward": F,
            }

        k_v = k_vals[valid]
        iv_v = ivs[valid]

        # Fit a second-order polynomial in log-moneyness:
        #   IV(k) ~ a0 + a1*k + a2*k^2
        # Then: level = a0 (ATM), slope = a1, curvature = 2*a2
        coeffs = np.polyfit(k_v, iv_v, deg=2)
        a2, a1, a0 = coeffs

        # ATM IV: evaluate the polynomial at k=0
        level = float(a0)
        slope = float(a1)
        curvature = float(2.0 * a2)

        return {
            "level": level,
            "slope": slope,
            "curvature": curvature,
            "atm_strike": float(F),
            "forward": float(F),
        }

    def decompose_surface(
        self,
        expiries: Optional[np.ndarray] = None,
    ) -> List[Dict[str, float]]:
        """Run ``decompose`` across multiple expiries.

        Parameters
        ----------
        expiries : array, optional
            Expiries to decompose.  If ``None``, the unique expiries
            present in the stored data are used.

        Returns
        -------
        list of dict
            One decomposition dict per expiry, augmented with an
            ``expiry`` key.
        """
        if expiries is None:
            expiries = np.unique([p.expiry for p in self._points])

        results = []
        for T in sorted(expiries):
            d = self.decompose(T)
            d["expiry"] = float(T)
            results.append(d)
        return results
