"""
Advanced Validation — Deflated Sharpe, PBO, Monte Carlo, capacity analysis.

Renaissance-grade validation to guard against strategy overfitting
and assess real-world viability.

References:
    - Bailey & Lopez de Prado (2014): "The Deflated Sharpe Ratio"
    - Bailey et al. (2017): "Probability of Backtest Overfitting"
    - White (2000): "Reality Check" bootstrap
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from ..config import RISK_FREE_RATE
from .sharpe_utils import compute_sharpe
from ._scipy_compat import sp_stats as stats


@dataclass
class DeflatedSharpeResult:
    """Result of Deflated Sharpe Ratio test."""
    observed_sharpe: float
    deflated_sharpe: float      # Sharpe after adjusting for multiple testing
    expected_max_sharpe: float  # Expected max Sharpe under null (from N trials)
    n_trials: int               # Number of strategy variants tested
    p_value: float              # Probability of observing this Sharpe by chance
    is_significant: bool        # p < 0.05
    deflated_sharpe_valid: bool = True  # False when deflated_sharpe is NaN


@dataclass
class PBOResult:
    """Probability of Backtest Overfitting result."""
    pbo: float                  # Probability of overfitting (0-1)
    n_combinations: int         # Total CSCV combinations tested
    degradation_rate: float     # Fraction of OOS results worse than median
    logits: List[float]         # Per-combination logit values
    is_overfit: bool            # PBO > 0.5
    pbo_valid: bool = True      # False when pbo is NaN (insufficient data)


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result."""
    n_simulations: int
    original_sharpe: float
    simulated_sharpes: np.ndarray   # Distribution under null
    p_value: float                   # Fraction of simulations >= original
    ci_95_lower: float              # 2.5th percentile Sharpe
    ci_95_upper: float              # 97.5th percentile Sharpe
    expected_drawdowns: np.ndarray   # Distribution of max drawdowns
    drawdown_95: float              # 95th percentile drawdown (worst case)
    is_significant: bool            # p < 0.05


@dataclass
class CapacityResult:
    """Strategy capacity analysis."""
    estimated_capacity_usd: float   # Dollar capacity before market impact
    avg_daily_volume: float         # Average daily dollar volume of universe
    participation_rate: float       # Our volume as fraction of market
    market_impact_bps: float        # Estimated market impact at full capacity
    trades_per_day: float
    capacity_constrained: bool
    # Stress-regime capacity (SPEC-V03)
    stress_capacity_usd: Optional[float] = None       # Capacity during stress regimes
    stress_market_impact_bps: Optional[float] = None   # Market impact during stress
    stress_participation_rate: Optional[float] = None   # Participation rate during stress
    stress_trades_per_day: Optional[float] = None       # Trade frequency during stress
    stress_avg_daily_volume: Optional[float] = None     # Volume during stress periods


@dataclass
class AdvancedValidationReport:
    """Complete advanced validation report."""
    deflated_sharpe: DeflatedSharpeResult
    pbo: Optional[PBOResult]
    monte_carlo: MonteCarloResult
    capacity: Optional[CapacityResult]
    overall_passes: bool
    summary: str


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_returns: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    annualization_factor: float = 1.0,
) -> DeflatedSharpeResult:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Adjusts the observed Sharpe ratio for the number of strategy
    variants that were tested (multiple testing correction).

    The expected maximum Sharpe under the null hypothesis (all strategies
    are random) grows as sqrt(2 * log(N)) where N = number of trials.

    Args:
        observed_sharpe: annualized Sharpe ratio of the strategy
        n_trials: number of strategy configurations tested
        n_returns: number of return observations
        skewness: return distribution skewness
        kurtosis: return distribution excess kurtosis
        annualization_factor: for de-annualizing back to per-period
    """
    if n_trials < 1:
        n_trials = 1
    if n_returns < 10:
        return DeflatedSharpeResult(
            observed_sharpe=observed_sharpe,
            deflated_sharpe=0,
            expected_max_sharpe=0,
            n_trials=n_trials,
            p_value=1.0,
            is_significant=False,
        )

    # Reject negative Sharpe outright — no statistical test needed.
    # A strategy with negative expected return cannot be significant
    # regardless of multiple-testing adjustment.
    if observed_sharpe <= 0:
        return DeflatedSharpeResult(
            observed_sharpe=observed_sharpe,
            deflated_sharpe=float('nan'),
            expected_max_sharpe=0.0,
            n_trials=n_trials,
            p_value=1.0,
            is_significant=False,
            deflated_sharpe_valid=False,
        )

    # Expected maximum Sharpe under null (Euler-Mascheroni approximation)
    euler_mascheroni = 0.5772156649
    expected_max = np.sqrt(2 * np.log(n_trials)) * (
        1 - euler_mascheroni / (2 * np.log(n_trials))
    ) if n_trials > 1 else 0

    # Standard error of Sharpe (accounting for non-normality)
    # SE(SR) = sqrt((1 + 0.5*SR^2 - skew*SR + (kurt/4)*SR^2) / (n-1))
    sr = observed_sharpe / annualization_factor if annualization_factor > 0 else observed_sharpe
    se = np.sqrt(
        (1 + 0.5 * sr**2 - skewness * sr + ((kurtosis - 3) / 4) * sr**2) /
        max(1, n_returns - 1)
    )

    # Deflated Sharpe = (observed - expected_max) / SE
    if se > 0:
        deflated = (sr - expected_max * se) / se  # per-period
        p_value = 1 - stats.norm.cdf(deflated)
    else:
        deflated = 0
        p_value = 1.0

    return DeflatedSharpeResult(
        observed_sharpe=observed_sharpe,
        deflated_sharpe=float(deflated * annualization_factor),
        expected_max_sharpe=float(expected_max * se * annualization_factor),
        n_trials=n_trials,
        p_value=float(p_value),
        is_significant=p_value < 0.05,
    )


def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame,
    n_partitions: int = 8,
) -> PBOResult:
    """
    Probability of Backtest Overfitting (Bailey et al., 2017).

    Uses Combinatorially Symmetric Cross-Validation (CSCV):
    1. Partition the return series into S sub-periods
    2. For each combination of S/2 sub-periods:
       - Train (optimize) on the selected half
       - Test on the remaining half
    3. PBO = fraction of combinations where the IS-best strategy
       underperforms the median OOS

    Args:
        returns_matrix: DataFrame where each column is a strategy variant
                       and each row is a return observation (time-ordered)
        n_partitions: number of time partitions (must be even)

    Returns:
        PBOResult with overfitting probability
    """
    from itertools import combinations

    n_strategies = returns_matrix.shape[1]
    n_obs = returns_matrix.shape[0]

    if n_strategies < 2:
        return PBOResult(pbo=float('nan'), n_combinations=0, degradation_rate=0,
                        logits=[], is_overfit=False, pbo_valid=False)

    # Ensure even number of partitions
    n_partitions = n_partitions if n_partitions % 2 == 0 else n_partitions + 1
    partition_size = n_obs // n_partitions

    if partition_size < 10:
        return PBOResult(pbo=float('nan'), n_combinations=0, degradation_rate=0.5,
                        logits=[], is_overfit=True, pbo_valid=False)

    # Partition data
    partitions = []
    for i in range(n_partitions):
        start = i * partition_size
        end = start + partition_size if i < n_partitions - 1 else n_obs
        partitions.append(returns_matrix.iloc[start:end])

    # Generate all combinations of S/2 partitions for training
    half = n_partitions // 2
    combos = list(combinations(range(n_partitions), half))

    # Limit combinations for computational feasibility.
    # 200 combinations provides better statistical power for reliable PBO
    # estimation while keeping runtime reasonable.
    max_combos = 200
    if len(combos) > max_combos:
        rng = np.random.RandomState(42)
        combo_indices = rng.choice(len(combos), max_combos, replace=False)
        combos = [combos[i] for i in combo_indices]

    logits = []
    n_degraded = 0

    for train_indices in combos:
        test_indices = [i for i in range(n_partitions) if i not in train_indices]

        # Concatenate train and test partitions
        train_data = pd.concat([partitions[i] for i in train_indices])
        test_data = pd.concat([partitions[i] for i in test_indices])

        # Find best strategy in-sample (by Sharpe)
        train_sharpes = pd.Series(
            {col: compute_sharpe(train_data[col].values, annualize=False) for col in train_data.columns}
        )
        best_is_strategy = train_sharpes.idxmax()

        # Evaluate best IS strategy out-of-sample
        oos_sharpe = compute_sharpe(test_data[best_is_strategy].values, annualize=False)

        # Compare to median OOS Sharpe across all strategies
        all_oos_sharpes = pd.Series(
            {col: compute_sharpe(test_data[col].values, annualize=False) for col in test_data.columns}
        )
        median_oos = all_oos_sharpes.median()

        # Logit: positive if IS-best underperforms OOS median
        if oos_sharpe < median_oos:
            n_degraded += 1
            logit = np.log(max(1e-10, (median_oos - oos_sharpe))) if median_oos > oos_sharpe else 0
        else:
            logit = -np.log(max(1e-10, (oos_sharpe - median_oos))) if oos_sharpe > median_oos else 0
        logits.append(float(logit))

    pbo = n_degraded / len(combos) if combos else 0.5
    degradation_rate = n_degraded / len(combos) if combos else 0.5

    # PBO logit: logit(pbo) gives a confidence measure.
    # logit > 0 means more likely than not that strategy is overfit.
    # Tightened threshold: PBO > 0.45 flags overfitting (academic standard
    # recommends concern at 0.4, rejection at 0.5; we use 0.45 as a
    # conservative middle ground).
    if 0 < pbo < 1:
        pbo_logit = float(np.log(pbo / (1 - pbo)))
    elif pbo >= 1:
        pbo_logit = float('inf')
    else:
        pbo_logit = float('-inf')

    is_overfit = pbo > 0.45 or pbo_logit > 0

    return PBOResult(
        pbo=float(pbo),
        n_combinations=len(combos),
        degradation_rate=float(degradation_rate),
        logits=logits,
        is_overfit=is_overfit,
    )


def monte_carlo_validation(
    trade_returns: np.ndarray,
    n_simulations: int = 1000,
    holding_days: int = 10,
    method: str = "shuffle",
) -> MonteCarloResult:
    """
    Monte Carlo validation of strategy performance.

    Tests whether observed performance could arise by chance by:
    1. Shuffling trade returns to destroy temporal structure
    2. Computing Sharpe ratio for each shuffled sequence
    3. Building a null distribution of Sharpe ratios
    4. Computing p-value: P(shuffled_sharpe >= observed_sharpe)

    Also estimates worst-case drawdown distribution.

    Args:
        trade_returns: array of per-trade net returns
        n_simulations: number of Monte Carlo trials
        holding_days: for annualization
        method: "shuffle" (permutation) or "bootstrap" (with replacement)
    """
    if len(trade_returns) < 10:
        return MonteCarloResult(
            n_simulations=0, original_sharpe=0,
            simulated_sharpes=np.array([]),
            p_value=1.0, ci_95_lower=0, ci_95_upper=0,
            expected_drawdowns=np.array([]),
            drawdown_95=0, is_significant=False,
        )

    returns = np.array(trade_returns, dtype=float)
    ann_factor = np.sqrt(252.0 / holding_days)

    # Original Sharpe
    rf_per_trade = RISK_FREE_RATE * holding_days / 252.0
    excess = returns - rf_per_trade
    original_sharpe = (excess.mean() / returns.std()) * ann_factor if returns.std() > 0 else 0

    rng = np.random.RandomState(42)
    sim_sharpes = np.zeros(n_simulations)
    sim_drawdowns = np.zeros(n_simulations)

    for i in range(n_simulations):
        if method == "shuffle":
            sim_returns = rng.permutation(returns)
        else:  # bootstrap
            sim_returns = rng.choice(returns, size=len(returns), replace=True)

        sim_excess = sim_returns - rf_per_trade
        sim_std = sim_returns.std()
        sim_sharpes[i] = (sim_excess.mean() / sim_std) * ann_factor if sim_std > 0 else 0

        # Max drawdown of this simulation
        cum_eq = np.cumprod(1 + sim_returns)
        running_max = np.maximum.accumulate(cum_eq)
        dd = ((cum_eq - running_max) / running_max).min()
        sim_drawdowns[i] = dd

    p_value = (sim_sharpes >= original_sharpe).mean()

    return MonteCarloResult(
        n_simulations=n_simulations,
        original_sharpe=float(original_sharpe),
        simulated_sharpes=sim_sharpes,
        p_value=float(p_value),
        ci_95_lower=float(np.percentile(sim_sharpes, 2.5)),
        ci_95_upper=float(np.percentile(sim_sharpes, 97.5)),
        expected_drawdowns=sim_drawdowns,
        # 5th percentile of negative drawdowns = 95th worst-case severity
        drawdown_95=float(np.percentile(sim_drawdowns, 5)),
        is_significant=p_value < 0.05,
    )


def _compute_capacity_metrics(
    trades: list,
    price_data: Dict[str, pd.DataFrame],
    capital_usd: float,
    max_participation_rate: float,
    impact_coefficient_bps: float,
) -> Optional[Dict[str, float]]:
    """Compute raw capacity metrics for a set of trades.

    Returns None if there is insufficient data (no trades, no price data,
    or no matching volume information).  Otherwise returns a dict with keys:
    estimated_capacity_usd, avg_daily_volume, participation_rate,
    market_impact_bps, trades_per_day, capacity_constrained.
    """
    if not trades or not price_data:
        return None

    # Compute average daily dollar volume for traded tickers
    traded_tickers = set(
        (
            t.ticker
            if hasattr(t, 'ticker')
            else t.get('permno', t.get('ticker', ''))
        )
        for t in trades
    )
    daily_volumes = []

    for ticker in traded_tickers:
        if ticker not in price_data:
            continue
        ohlcv = price_data[ticker]
        if len(ohlcv) < 20:
            continue
        # Average dollar volume (last 60 days)
        dollar_vol = (ohlcv["Close"] * ohlcv["Volume"]).iloc[-60:].mean()
        daily_volumes.append(float(dollar_vol))

    if not daily_volumes:
        return None

    avg_daily_vol = np.mean(daily_volumes)

    # Estimate trades per day
    trade_dates = set()
    for t in trades:
        entry = t.entry_date if hasattr(t, 'entry_date') else t.get('entry_date', '')
        trade_dates.add(entry)
    n_trading_days = len(trade_dates) if trade_dates else 1
    n_trades = len(trades)
    trades_per_day = n_trades / max(1, n_trading_days)

    # Median position size in dollars (robust to outliers and varying sizes)
    sizes = [
        t.position_size for t in trades
        if hasattr(t, 'position_size') and t.position_size > 0
    ]
    avg_size = float(np.median(sizes)) if sizes else 0.05
    position_usd = capital_usd * avg_size

    # Participation rate: our daily order flow / market volume
    daily_order_flow = position_usd * trades_per_day
    participation_rate = daily_order_flow / avg_daily_vol if avg_daily_vol > 0 else 0

    # Market impact: square-root model
    # impact_bps = impact_coefficient_bps * sqrt(participation_rate)
    market_impact = impact_coefficient_bps * np.sqrt(max(0.0, participation_rate))

    # Capacity estimate: capital where participation hits max_rate
    if trades_per_day > 0 and avg_size > 0:
        max_order = avg_daily_vol * max_participation_rate
        max_capital = max_order / (avg_size * trades_per_day) if avg_size * trades_per_day > 0 else 0
    else:
        max_capital = 0

    return {
        "estimated_capacity_usd": float(max_capital),
        "avg_daily_volume": float(avg_daily_vol),
        "participation_rate": float(participation_rate),
        "market_impact_bps": float(market_impact),
        "trades_per_day": float(trades_per_day),
        "capacity_constrained": participation_rate > max_participation_rate,
        "median_position_size": float(avg_size),
        "p75_position_size": float(np.percentile(sizes, 75)) if sizes else 0.05,
        "max_position_size": float(np.max(sizes)) if sizes else 0.05,
    }


def capacity_analysis(
    trades: list,
    price_data: Dict[str, pd.DataFrame],
    capital_usd: float = 1_000_000,
    max_participation_rate: float = 0.01,  # 1% of daily volume
    impact_coefficient_bps: float = 30.0,  # impact at 100% participation
    stress_regimes: Optional[List[int]] = None,
    min_stress_trades: int = 10,
) -> CapacityResult:
    """
    Estimate strategy capacity and market impact.

    Calculates how much capital the strategy can deploy before
    market impact erodes returns.  When ``stress_regimes`` is provided,
    also computes capacity using only trades that occurred during those
    regimes — liquidity typically collapses in stress, so stress capacity
    is the binding constraint for position sizing.

    Market impact model (square-root): impact = coeff * sqrt(Q / V)

    Args:
        trades: List of Trade objects (or dicts with compatible keys).
        price_data: Dict mapping ticker -> OHLCV DataFrame.
        capital_usd: Assumed capital base for participation calculation.
        max_participation_rate: Maximum acceptable fraction of daily volume.
        impact_coefficient_bps: Impact at 100% participation (basis points).
        stress_regimes: Regime codes (e.g. [2, 3]) to isolate for stress
            capacity.  ``None`` skips stress analysis.
        min_stress_trades: Minimum number of stress trades required for
            a meaningful stress capacity estimate.
    """
    empty = CapacityResult(
        estimated_capacity_usd=0, avg_daily_volume=0,
        participation_rate=0, market_impact_bps=0,
        trades_per_day=0, capacity_constrained=False,
    )

    overall = _compute_capacity_metrics(
        trades, price_data, capital_usd,
        max_participation_rate, impact_coefficient_bps,
    )
    if overall is None:
        return empty

    result = CapacityResult(
        estimated_capacity_usd=overall["estimated_capacity_usd"],
        avg_daily_volume=overall["avg_daily_volume"],
        participation_rate=overall["participation_rate"],
        market_impact_bps=overall["market_impact_bps"],
        trades_per_day=overall["trades_per_day"],
        capacity_constrained=overall["capacity_constrained"],
    )

    # ── Stress-regime capacity (SPEC-V03) ──
    if stress_regimes is not None:
        stress_regime_set = set(stress_regimes)
        stress_trades = [
            t for t in trades
            if (t.regime if hasattr(t, 'regime') else t.get('regime', -1))
            in stress_regime_set
        ]
        if len(stress_trades) >= min_stress_trades:
            stress = _compute_capacity_metrics(
                stress_trades, price_data, capital_usd,
                max_participation_rate, impact_coefficient_bps,
            )
            if stress is not None:
                result.stress_capacity_usd = stress["estimated_capacity_usd"]
                result.stress_market_impact_bps = stress["market_impact_bps"]
                result.stress_participation_rate = stress["participation_rate"]
                result.stress_trades_per_day = stress["trades_per_day"]
                result.stress_avg_daily_volume = stress["avg_daily_volume"]
                # The binding constraint is the worst case
                if result.stress_capacity_usd < result.estimated_capacity_usd:
                    result.capacity_constrained = (
                        result.capacity_constrained
                        or stress["participation_rate"] > max_participation_rate
                    )

    return result


def run_advanced_validation(
    trade_returns: np.ndarray,
    trades: list,
    price_data: Dict[str, pd.DataFrame],
    n_strategy_variants: int = 1,
    holding_days: int = 10,
    returns_matrix: Optional[pd.DataFrame] = None,
    verbose: bool = True,
    stress_regimes: Optional[List[int]] = None,
) -> AdvancedValidationReport:
    """
    Run all advanced validation tests.

    Args:
        trade_returns: per-trade net returns
        trades: list of Trade objects
        price_data: OHLCV data for all tickers
        n_strategy_variants: how many strategy configs were tested
        holding_days: average holding period
        returns_matrix: strategy variants for PBO (optional)
        verbose: print results
        stress_regimes: regime codes for stress capacity analysis (SPEC-V03)
    """
    returns = np.array(trade_returns, dtype=float)

    # ── Deflated Sharpe ──
    skew = float(pd.Series(returns).skew()) if len(returns) > 3 else 0
    kurt = float(pd.Series(returns).kurt()) + 3 if len(returns) > 3 else 3  # Convert excess to raw
    ann_factor = np.sqrt(252.0 / holding_days)
    observed_sharpe = (returns.mean() / returns.std()) * ann_factor if returns.std() > 0 else 0

    dsr = deflated_sharpe_ratio(
        observed_sharpe=observed_sharpe,
        n_trials=max(1, n_strategy_variants),
        n_returns=len(returns),
        skewness=skew,
        kurtosis=kurt,
        annualization_factor=ann_factor,
    )

    # ── PBO ──
    pbo_result = None
    if returns_matrix is not None and returns_matrix.shape[1] >= 2:
        pbo_result = probability_of_backtest_overfitting(returns_matrix)

    # ── Monte Carlo ──
    mc = monte_carlo_validation(returns, holding_days=holding_days)

    # ── Capacity ──
    cap = capacity_analysis(trades, price_data, stress_regimes=stress_regimes)

    # ── Overall assessment ──
    passes = (
        dsr.is_significant
        and mc.is_significant
        and (pbo_result is None or not pbo_result.is_overfit)
    )

    # Build summary string
    parts = []
    parts.append(f"DSR p={dsr.p_value:.3f} ({'PASS' if dsr.is_significant else 'FAIL'})")
    parts.append(f"MC p={mc.p_value:.3f} ({'PASS' if mc.is_significant else 'FAIL'})")
    if pbo_result:
        if pbo_result.pbo_valid:
            parts.append(f"PBO={pbo_result.pbo:.1%} ({'PASS' if not pbo_result.is_overfit else 'FAIL'})")
        else:
            parts.append("PBO=N/A (insufficient data)")
    if cap:
        parts.append(f"Capacity=${cap.estimated_capacity_usd:,.0f}")
    summary = " | ".join(parts)

    report = AdvancedValidationReport(
        deflated_sharpe=dsr,
        pbo=pbo_result,
        monte_carlo=mc,
        capacity=cap,
        overall_passes=passes,
        summary=summary,
    )

    if verbose:
        _print_report(report)

    return report


def _print_report(report: AdvancedValidationReport):
    """Pretty-print advanced validation report."""
    print(f"\n{'='*60}")
    print(f"ADVANCED VALIDATION")
    print(f"{'='*60}")

    dsr = report.deflated_sharpe
    print(f"\n  Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014):")
    print(f"    Observed Sharpe: {dsr.observed_sharpe:.2f}")
    if dsr.deflated_sharpe_valid:
        print(f"    Deflated Sharpe: {dsr.deflated_sharpe:.2f}")
    else:
        print(f"    Deflated Sharpe: N/A (negative observed Sharpe)")
    print(f"    Expected max (null): {dsr.expected_max_sharpe:.2f} (from {dsr.n_trials} trials)")
    print(f"    p-value: {dsr.p_value:.4f}")
    print(f"    Significant: {dsr.is_significant}")

    mc = report.monte_carlo
    print(f"\n  Monte Carlo Validation ({mc.n_simulations} simulations):")
    print(f"    Original Sharpe: {mc.original_sharpe:.2f}")
    print(f"    Simulated 95% CI: [{mc.ci_95_lower:.2f}, {mc.ci_95_upper:.2f}]")
    print(f"    p-value: {mc.p_value:.4f}")
    print(f"    95th pctile drawdown: {mc.drawdown_95:.1%}")
    print(f"    Significant: {mc.is_significant}")

    if report.pbo is not None:
        pbo = report.pbo
        print(f"\n  Probability of Backtest Overfitting:")
        if pbo.pbo_valid:
            print(f"    PBO: {pbo.pbo:.1%}")
        else:
            print(f"    PBO: N/A (insufficient data)")
        print(f"    Combinations tested: {pbo.n_combinations}")
        print(f"    Degradation rate: {pbo.degradation_rate:.1%}")
        print(f"    Overfit: {pbo.is_overfit}")

    if report.capacity is not None:
        cap = report.capacity
        print(f"\n  Capacity Analysis:")
        print(f"    Estimated capacity: ${cap.estimated_capacity_usd:,.0f}")
        print(f"    Avg daily volume: ${cap.avg_daily_volume:,.0f}")
        print(f"    Participation rate: {cap.participation_rate:.4%}")
        print(f"    Market impact: {cap.market_impact_bps:.1f} bps")
        print(f"    Capacity constrained: {cap.capacity_constrained}")
        if cap.stress_capacity_usd is not None:
            print(f"    Stress capacity: ${cap.stress_capacity_usd:,.0f}")
            print(f"    Stress market impact: {cap.stress_market_impact_bps:.1f} bps")
            print(f"    Stress participation: {cap.stress_participation_rate:.4%}")
            ratio = cap.stress_capacity_usd / cap.estimated_capacity_usd if cap.estimated_capacity_usd > 0 else 0
            print(f"    Stress/overall ratio: {ratio:.1%}")

    print(f"\n  OVERALL: {'PASS' if report.overall_passes else 'FAIL'}")
    print(f"  Summary: {report.summary}")
