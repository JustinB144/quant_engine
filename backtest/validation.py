"""
Walk-forward validation and statistical tests.

Ensures the model's predictions have genuine out-of-sample predictive power
and aren't artifacts of overfitting.

Anti-leakage measures:
    - Purge gap between train and test in walk-forward folds
    - Embargo period after purge gap
    - Spearman rank correlation throughout (robust to outliers)
    - Correct one-sided t-test (handles negative t-stat)
    - Benjamini-Hochberg FDR for multiple testing correction
    - Information coefficient significance test
"""
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import logging

import numpy as np
import pandas as pd

from ..config import IC_ROLLING_WINDOW, RISK_FREE_RATE, VALIDATION_FDR_FLOOR_ENABLED
from .sharpe_utils import compute_sharpe
from ._scipy_compat import sp_stats as stats

_logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """Per-fold walk-forward validation metrics for one temporal split."""
    fold: int
    train_size: int
    test_size: int
    train_corr: float  # Spearman rank correlation
    test_corr: float   # Spearman rank correlation
    test_mean_return: float  # mean actual return of predicted-positive trades
    # Fold-level metrics (Spec 04)
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_estimate: float = 0.0
    sample_count: int = 0


@dataclass
class WalkForwardResult:
    """Aggregate walk-forward validation summary and overfitting diagnostics."""
    n_folds: int
    folds: List[WalkForwardFold]
    avg_oos_corr: float
    avg_is_corr: float
    is_oos_gap: float
    oos_corr_std: float
    is_overfit: bool
    profitable_folds: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class StatisticalTests:
    # Prediction-actual Spearman correlation
    """Bundle of statistical significance tests for prediction quality and signal returns."""
    pred_actual_corr: float
    pred_actual_pval: float
    # Mean return of long signals
    long_mean_return: float
    long_tstat: float
    long_pval: float
    # Sharpe significance
    sharpe: float
    sharpe_se: float
    sharpe_pval: float
    # Information coefficient (rolling Spearman rank correlation)
    ic_mean: float
    ic_std: float
    ic_ir: float  # IC / std(IC) — information ratio
    ic_tstat: float  # t-test on IC values
    ic_pval: float   # p-value for IC significance
    # Multiple testing
    n_tests: int
    fdr_threshold: float  # Benjamini-Hochberg adjusted threshold
    passes: bool
    details: dict = field(default_factory=dict)
    null_comparison: Optional[object] = None  # NullModelResults when wired


@dataclass
class CPCVResult:
    """Combinatorial purged cross-validation summary metrics and pass/fail status."""
    n_partitions: int
    n_test_partitions: int
    n_combinations: int
    mean_is_corr: float
    mean_oos_corr: float
    median_oos_corr: float
    oos_corr_std: float
    mean_oos_return: float
    positive_oos_fraction: float
    passes: bool
    warnings: List[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class SPAResult:
    """Superior Predictive Ability (SPA) test result bundle."""
    observed_stat: float
    observed_mean: float
    benchmark_mean: float
    p_value: float
    n_bootstraps: int
    block_size: int
    passes: bool
    details: dict = field(default_factory=dict)


def walk_forward_validate(
    predictions: pd.Series,
    actuals: pd.Series,
    n_folds: int = 5,
    entry_threshold: float = 0.005,
    max_overfit_ratio: float = 2.5,
    purge_gap: int = 10,
    embargo: int = 5,
    max_train_samples: Optional[int] = None,
) -> WalkForwardResult:
    """
    Walk-forward validation of prediction quality with purge gap.

    Splits data temporally into n_folds windows.  By default the training
    window expands (all data up to the fold boundary).  When
    *max_train_samples* is set, the training window rolls forward so that
    only the most recent *max_train_samples* observations are used,
    preventing stale data from inflating measured skill.

    Anti-leakage:
        - Purge gap: removes `purge_gap` samples between train end and test start
        - Embargo: skips first `embargo` samples of test set

    Args:
        predictions: model's predicted returns
        actuals: actual forward returns
        n_folds: number of temporal folds
        entry_threshold: threshold for counting a prediction as "long"
        max_overfit_ratio: IS/OOS correlation ratio above which -> overfit
        purge_gap: number of samples to remove between train and test
        embargo: number of test samples to skip at start
        max_train_samples: if set, caps the training window size so that
            old data rolls off (rolling walk-forward instead of expanding)

    Returns:
        WalkForwardResult with per-fold and aggregate metrics
    """
    # Align and drop NaN
    valid = predictions.notna() & actuals.notna()
    preds = predictions[valid].values
    actual = actuals[valid].values
    n = len(preds)

    if n < 100:
        return WalkForwardResult(
            n_folds=0, folds=[], avg_oos_corr=0, avg_is_corr=0,
            is_oos_gap=0, oos_corr_std=0,
            is_overfit=True, profitable_folds=0,
            warnings=["Insufficient data for walk-forward validation"],
        )

    folds = []
    fold_size = n // (n_folds + 1)  # +1 so first fold has training data
    warnings = []

    for i in range(n_folds):
        # Train on data up to fold boundary minus purge gap
        test_start = (i + 1) * fold_size
        test_end = min(test_start + fold_size, n)

        # Apply purge gap: training data ends purge_gap samples before test
        train_end = max(0, test_start - purge_gap)
        # Apply embargo: test starts embargo samples after the fold boundary
        embargoed_start = min(test_start + embargo, test_end)

        if train_end < 50 or embargoed_start >= test_end or test_end - embargoed_start < 20:
            continue

        # Rolling window: cap training data to most recent max_train_samples
        train_start = 0
        if max_train_samples is not None and train_end > max_train_samples:
            train_start = train_end - max_train_samples

        train_preds = preds[train_start:train_end]
        train_actual = actual[train_start:train_end]
        test_preds = preds[embargoed_start:test_end]
        test_actual = actual[embargoed_start:test_end]

        # Spearman rank correlation (robust to outliers)
        if len(train_preds) > 2:
            train_corr, _ = stats.spearmanr(train_preds, train_actual)
            train_corr = float(train_corr) if not np.isnan(train_corr) else 0
        else:
            train_corr = 0

        if len(test_preds) > 2:
            test_corr, _ = stats.spearmanr(test_preds, test_actual)
            test_corr = float(test_corr) if not np.isnan(test_corr) else 0
        else:
            test_corr = 0

        # Mean return of predicted-long trades
        long_mask = test_preds > entry_threshold
        test_mean_ret = float(test_actual[long_mask].mean()) if long_mask.any() else 0

        # Fold-level signal quality metrics (Spec 04)
        fold_win_rate = 0.0
        fold_profit_factor = 0.0
        fold_sharpe = 0.0
        if long_mask.any():
            long_actual = test_actual[long_mask]
            n_long = len(long_actual)
            n_winners = int((long_actual > 0).sum())
            fold_win_rate = float(n_winners / n_long) if n_long > 0 else 0.0

            pos_sum = float(long_actual[long_actual > 0].sum())
            neg_sum = float(abs(long_actual[long_actual < 0].sum()))
            if neg_sum > 1e-12:
                fold_profit_factor = pos_sum / neg_sum
            elif pos_sum > 0:
                fold_profit_factor = 10.0  # capped stand-in for inf
            # else: 0.0

            if n_long > 1:
                fold_sharpe = compute_sharpe(long_actual.values if hasattr(long_actual, 'values') else long_actual, annualize=False)

        folds.append(WalkForwardFold(
            fold=i + 1,
            train_size=len(train_preds),
            test_size=len(test_preds),
            train_corr=train_corr,
            test_corr=test_corr,
            test_mean_return=test_mean_ret,
            win_rate=fold_win_rate,
            profit_factor=fold_profit_factor,
            sharpe_estimate=fold_sharpe,
            sample_count=len(test_preds),
        ))

    if not folds:
        return WalkForwardResult(
            n_folds=0, folds=[], avg_oos_corr=0, avg_is_corr=0,
            is_oos_gap=0, oos_corr_std=0,
            is_overfit=True, profitable_folds=0,
            warnings=["No valid folds"],
        )

    oos_corrs = [f.test_corr for f in folds]
    is_corrs = [f.train_corr for f in folds]

    avg_oos = np.mean(oos_corrs)
    avg_is = np.mean(is_corrs)
    gap = avg_is - avg_oos
    oos_std = np.std(oos_corrs)
    profitable = sum(1 for f in folds if f.test_mean_return > 0)

    # Overfit detection
    is_overfit = False
    if avg_oos > 0 and avg_is / (avg_oos + 1e-10) > max_overfit_ratio:
        is_overfit = True
        warnings.append(f"IS/OOS ratio = {avg_is/(avg_oos+1e-10):.2f} > {max_overfit_ratio}")
    if avg_oos <= 0:
        is_overfit = True
        warnings.append(f"Average OOS Spearman correlation <= 0 ({avg_oos:.4f})")
    if oos_std > 0.3:
        warnings.append(f"High OOS correlation variance ({oos_std:.4f})")

    return WalkForwardResult(
        n_folds=len(folds),
        folds=folds,
        avg_oos_corr=float(avg_oos),
        avg_is_corr=float(avg_is),
        is_oos_gap=float(gap),
        oos_corr_std=float(oos_std),
        is_overfit=is_overfit,
        profitable_folds=profitable,
        warnings=warnings,
    )


def _benjamini_hochberg(pvals: List[float], alpha: float = 0.05) -> float:
    """
    Benjamini-Hochberg procedure for multiple testing correction.

    Returns the adjusted significance threshold. p-values below this
    threshold are considered significant after controlling FDR.
    """
    if not pvals:
        return alpha

    m = len(pvals)
    sorted_pvals = sorted(pvals)
    threshold = alpha

    for k in range(m, 0, -1):
        if sorted_pvals[k - 1] <= (k / m) * alpha:
            threshold = (k / m) * alpha
            break
    else:
        threshold = 0.0  # none pass

    return threshold


def run_statistical_tests(
    predictions: pd.Series,
    actuals: pd.Series,
    trade_returns: np.ndarray,
    entry_threshold: float = 0.005,
    holding_days: int = 10,
) -> StatisticalTests:
    """
    Statistical tests for prediction quality.

    Tests:
        1. Prediction-actual Spearman correlation significance
        2. Mean return of long signals > 0 (one-sided t-test, correct direction)
        3. Sharpe ratio significance (Lo 2002 standard error)
        4. Information coefficient: rolling Spearman with t-test on IC values
        5. Benjamini-Hochberg FDR correction across all tests
    """
    valid = predictions.notna() & actuals.notna()
    preds = predictions[valid]
    actual = actuals[valid]

    all_pvals = []

    # 1. Prediction-actual Spearman correlation
    if len(preds) > 2:
        corr, corr_pval = stats.spearmanr(preds.values, actual.values)
        corr = float(corr) if not np.isnan(corr) else 0
        corr_pval = float(corr_pval) if not np.isnan(corr_pval) else 1
    else:
        corr, corr_pval = 0, 1
    all_pvals.append(corr_pval)

    # 2. Mean return of long signals (correct one-sided t-test)
    long_mask = preds > entry_threshold
    long_returns = actual[long_mask].values
    if len(long_returns) > 2:
        long_tstat, two_sided_pval = stats.ttest_1samp(long_returns, 0)
        long_mean = float(long_returns.mean())
        # One-sided: H1 is mean > 0
        # If t-stat is positive, p = two_sided / 2
        # If t-stat is negative, p = 1 - two_sided / 2
        if long_tstat > 0:
            long_pval = two_sided_pval / 2
        else:
            long_pval = 1 - two_sided_pval / 2
    else:
        long_tstat, long_pval, long_mean = 0, 1, 0
    all_pvals.append(long_pval)

    # 3. Sharpe ratio significance (Lo 2002)
    if len(trade_returns) > 2 and trade_returns.std() > 0:
        # Risk-free subtraction
        rf_per_trade = RISK_FREE_RATE * holding_days / 252.0
        excess = trade_returns - rf_per_trade
        sharpe = excess.mean() / trade_returns.std()
        n_trades = len(trade_returns)
        # Lo (2002) standard error accounting for skewness and kurtosis
        skew = float(stats.skew(trade_returns))
        kurt = float(stats.kurtosis(trade_returns))
        sharpe_se = np.sqrt(
            (1 + 0.5 * sharpe**2 - skew * sharpe + (kurt / 4) * sharpe**2) / n_trades
        )
        sharpe_z = sharpe / (sharpe_se + 1e-10)
        sharpe_pval = 1 - stats.norm.cdf(sharpe_z)  # one-sided
    else:
        sharpe, sharpe_se, sharpe_pval = 0, 1, 1
    all_pvals.append(sharpe_pval)

    # 4. Information coefficient (rolling Spearman rank correlation)
    ic_values = []
    window = IC_ROLLING_WINDOW
    for i in range(window, len(preds)):
        p = preds.iloc[i - window:i]
        a = actual.iloc[i - window:i]
        if p.std() > 0 and a.std() > 0:
            rc, _ = stats.spearmanr(p.values, a.values)
            if not np.isnan(rc):
                ic_values.append(rc)

    if len(ic_values) > 2:
        ic_mean = float(np.mean(ic_values))
        ic_std = float(np.std(ic_values))
        ic_ir = ic_mean / (ic_std + 1e-10)
        # t-test: is mean IC significantly > 0?
        ic_tstat, ic_two_pval = stats.ttest_1samp(ic_values, 0)
        # One-sided
        if ic_tstat > 0:
            ic_pval = ic_two_pval / 2
        else:
            ic_pval = 1 - ic_two_pval / 2
    else:
        ic_mean, ic_std, ic_ir = 0, 1, 0
        ic_tstat, ic_pval = 0, 1
    all_pvals.append(ic_pval)

    # 5. Benjamini-Hochberg FDR correction
    fdr_threshold = _benjamini_hochberg(all_pvals, alpha=0.05)

    # Pass criteria (all must hold after FDR correction):
    # - Spearman correlation p-value < FDR threshold
    # - Long signal return positive AND p-value < FDR threshold
    # - IC mean > 0 AND IC p-value < FDR threshold
    # - Sharpe p-value < 0.10 (relaxed, not FDR-corrected, as it's confirmatory)
    if fdr_threshold <= 0 and not VALIDATION_FDR_FLOOR_ENABLED:
        # BH says nothing is significant — respect that decision
        passes = False
    else:
        # When VALIDATION_FDR_FLOOR_ENABLED (deprecated), apply old 0.001 floor
        effective_fdr = (
            max(fdr_threshold, 0.001) if VALIDATION_FDR_FLOOR_ENABLED
            else fdr_threshold
        )
        passes = (
            corr_pval < effective_fdr
            and long_pval < effective_fdr
            and long_mean > 0
            and sharpe_pval < 0.10
            and ic_mean > 0
            and ic_pval < effective_fdr
        )

    return StatisticalTests(
        pred_actual_corr=float(corr),
        pred_actual_pval=float(corr_pval),
        long_mean_return=float(long_mean),
        long_tstat=float(long_tstat),
        long_pval=float(long_pval),
        sharpe=float(sharpe),
        sharpe_se=float(sharpe_se),
        sharpe_pval=float(sharpe_pval),
        ic_mean=float(ic_mean),
        ic_std=float(ic_std),
        ic_ir=float(ic_ir),
        ic_tstat=float(ic_tstat),
        ic_pval=float(ic_pval),
        n_tests=len(all_pvals),
        fdr_threshold=float(fdr_threshold),
        passes=passes,
        details={
            "n_predictions": len(preds),
            "n_long_signals": int(long_mask.sum()),
            "n_ic_windows": len(ic_values),
            "all_pvals": [float(p) for p in all_pvals],
            "fdr_adjusted": True,
        },
    )


def _partition_bounds(n_obs: int, n_partitions: int) -> List[Tuple[int, int]]:
    """Return contiguous [start, end) bounds for temporal partitions."""
    if n_obs <= 0:
        return []
    n_partitions = max(2, min(n_partitions, n_obs))
    bounds = []
    step = n_obs // n_partitions
    for i in range(n_partitions):
        start = i * step
        end = (i + 1) * step if i < n_partitions - 1 else n_obs
        if end > start:
            bounds.append((start, end))
    return bounds


def combinatorial_purged_cv(
    predictions: pd.Series,
    actuals: pd.Series,
    entry_threshold: float = 0.005,
    n_partitions: int = 8,
    n_test_partitions: int = 4,
    purge_gap: int = 10,
    embargo: int = 5,
    max_combinations: int = 200,
) -> CPCVResult:
    """
    Combinatorial Purged Cross-Validation for signal robustness.

    Evaluates IS/OOS consistency across many time-partition combinations.
    """
    valid = predictions.notna() & actuals.notna()
    preds = predictions[valid].values
    actual = actuals[valid].values
    n = len(preds)
    warnings: List[str] = []

    if n < 200:
        return CPCVResult(
            n_partitions=n_partitions,
            n_test_partitions=n_test_partitions,
            n_combinations=0,
            mean_is_corr=0.0,
            mean_oos_corr=0.0,
            median_oos_corr=0.0,
            oos_corr_std=0.0,
            mean_oos_return=0.0,
            positive_oos_fraction=0.0,
            passes=False,
            warnings=["Insufficient observations for CPCV"],
        )

    bounds = _partition_bounds(n, n_partitions)
    n_partitions_eff = len(bounds)
    if n_partitions_eff < 2:
        return CPCVResult(
            n_partitions=n_partitions_eff,
            n_test_partitions=0,
            n_combinations=0,
            mean_is_corr=0.0,
            mean_oos_corr=0.0,
            median_oos_corr=0.0,
            oos_corr_std=0.0,
            mean_oos_return=0.0,
            positive_oos_fraction=0.0,
            passes=False,
            warnings=["Unable to partition data for CPCV"],
        )

    n_test = max(1, min(n_test_partitions, n_partitions_eff - 1))
    combos = list(combinations(range(n_partitions_eff), n_test))
    if len(combos) > max_combinations:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(combos), size=max_combinations, replace=False)
        combos = [combos[int(i)] for i in idx]
        warnings.append(f"Combination set truncated to {max_combinations}")

    is_corrs = []
    oos_corrs = []
    oos_rets = []

    for test_parts in combos:
        test_mask = np.zeros(n, dtype=bool)
        for p in test_parts:
            start, end = bounds[p]
            test_mask[start:end] = True

        train_mask = ~test_mask
        for p in test_parts:
            start, end = bounds[p]
            purge_start = max(0, start - purge_gap)
            purge_end = min(n, end + embargo)
            train_mask[purge_start:purge_end] = False

        if train_mask.sum() < 60 or test_mask.sum() < 30:
            continue

        is_corr, _ = stats.spearmanr(preds[train_mask], actual[train_mask])
        oos_corr, _ = stats.spearmanr(preds[test_mask], actual[test_mask])
        is_corr = float(is_corr) if not np.isnan(is_corr) else 0.0
        oos_corr = float(oos_corr) if not np.isnan(oos_corr) else 0.0
        is_corrs.append(is_corr)
        oos_corrs.append(oos_corr)

        long_mask = test_mask & (preds > entry_threshold)
        if long_mask.any():
            oos_rets.append(float(actual[long_mask].mean()))
        else:
            oos_rets.append(0.0)

    if not oos_corrs:
        return CPCVResult(
            n_partitions=n_partitions_eff,
            n_test_partitions=n_test,
            n_combinations=0,
            mean_is_corr=0.0,
            mean_oos_corr=0.0,
            median_oos_corr=0.0,
            oos_corr_std=0.0,
            mean_oos_return=0.0,
            positive_oos_fraction=0.0,
            passes=False,
            warnings=["No valid CPCV combinations after purge/embargo"],
        )

    is_arr = np.array(is_corrs, dtype=float)
    oos_arr = np.array(oos_corrs, dtype=float)
    oos_ret_arr = np.array(oos_rets, dtype=float)
    positive_frac = float((oos_arr > 0).mean())

    passes = (
        float(np.median(oos_arr)) > 0.0
        and positive_frac >= 0.55
        and float(oos_ret_arr.mean()) > 0.0
    )
    if positive_frac < 0.55:
        warnings.append("Less than 55% of CPCV folds have positive OOS correlation")

    return CPCVResult(
        n_partitions=n_partitions_eff,
        n_test_partitions=n_test,
        n_combinations=len(oos_arr),
        mean_is_corr=float(is_arr.mean()),
        mean_oos_corr=float(oos_arr.mean()),
        median_oos_corr=float(np.median(oos_arr)),
        oos_corr_std=float(oos_arr.std()),
        mean_oos_return=float(oos_ret_arr.mean()),
        positive_oos_fraction=positive_frac,
        passes=passes,
        warnings=warnings,
        details={
            "best_oos_corr": float(oos_arr.max()),
            "worst_oos_corr": float(oos_arr.min()),
            "best_oos_return": float(oos_ret_arr.max()),
            "worst_oos_return": float(oos_ret_arr.min()),
        },
    )


def strategy_signal_returns(
    predictions: pd.Series,
    actuals: pd.Series,
    entry_threshold: float = 0.005,
    confidence: Optional[pd.Series] = None,
    min_confidence: float = 0.0,
) -> pd.Series:
    """
    Build per-sample strategy return series from prediction signals.
    """
    valid = predictions.notna() & actuals.notna()
    if confidence is not None:
        valid &= confidence.notna()
    preds = predictions[valid]
    actual = actuals[valid]

    signal_mask = preds > entry_threshold
    if confidence is not None:
        conf = confidence[valid]
        signal_mask &= conf >= min_confidence

    returns = pd.Series(0.0, index=preds.index)
    returns.loc[signal_mask] = actual.loc[signal_mask]
    return returns


def superior_predictive_ability(
    strategy_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    n_bootstraps: int = 1000,
    block_size: int = 10,
    random_state: int = 42,
) -> SPAResult:
    """
    Single-strategy SPA-style block-bootstrap test on differential returns.
    """
    sr = strategy_returns.dropna()
    if benchmark_returns is None:
        benchmark = pd.Series(0.0, index=sr.index)
    else:
        benchmark = benchmark_returns.reindex(sr.index).fillna(0.0)

    diff = (sr - benchmark).dropna()
    n = len(diff)
    if n < 40:
        return SPAResult(
            observed_stat=0.0,
            observed_mean=0.0,
            benchmark_mean=float(benchmark.mean()) if len(benchmark) > 0 else 0.0,
            p_value=1.0,
            n_bootstraps=0,
            block_size=block_size,
            passes=False,
            details={"warning": "Insufficient observations for SPA"},
        )

    arr = diff.values.astype(float)
    obs_mean = float(arr.mean())
    obs_stat = float(np.sqrt(n) * obs_mean)

    # Null-centered differential returns.
    centered = arr - obs_mean
    rng = np.random.RandomState(random_state)
    bsz = max(1, int(block_size))
    boot_stats = np.zeros(n_bootstraps, dtype=float)

    for i in range(n_bootstraps):
        sample_idx = []
        while len(sample_idx) < n:
            start = int(rng.randint(0, n))
            sample_idx.extend(((start + np.arange(bsz)) % n).tolist())
        idx = np.array(sample_idx[:n], dtype=int)
        boot = centered[idx]
        boot_stats[i] = np.sqrt(n) * float(boot.mean())

    p_value = float((boot_stats >= obs_stat).mean())
    passes = bool(obs_mean > 0 and p_value < 0.05)

    return SPAResult(
        observed_stat=obs_stat,
        observed_mean=obs_mean,
        benchmark_mean=float(benchmark.mean()),
        p_value=p_value,
        n_bootstraps=n_bootstraps,
        block_size=bsz,
        passes=passes,
        details={
            "n_observations": n,
            "boot_ci_95_low": float(np.percentile(boot_stats, 2.5)),
            "boot_ci_95_high": float(np.percentile(boot_stats, 97.5)),
        },
    )


# ── Walk-Forward with Embargo (Spec 08 T2) ──────────────────────────


@dataclass
class WalkForwardEmbargoFold:
    """Per-fold result from walk-forward evaluation with embargo."""
    fold: int
    train_start: str
    train_end: str
    embargo_start: str
    embargo_end: str
    test_start: str
    test_end: str
    train_sharpe: float
    test_sharpe: float
    train_ic: float
    test_ic: float
    overfit_gap_sharpe: float
    overfit_gap_ic: float
    test_n_samples: int


@dataclass
class WalkForwardEmbargoResult:
    """Aggregate walk-forward result across all folds."""
    n_folds: int
    folds: List[WalkForwardEmbargoFold]
    mean_train_sharpe: float
    mean_test_sharpe: float
    mean_overfit_gap: float
    is_overfit: bool
    warnings: List[str] = field(default_factory=list)


def walk_forward_with_embargo(
    returns: pd.Series,
    predictions: np.ndarray,
    train_window: int = 250,
    embargo: int = 5,
    test_window: int = 60,
    slide_freq: str = "weekly",
    risk_free_rate: float = RISK_FREE_RATE,
) -> WalkForwardEmbargoResult:
    """Walk-forward evaluation with embargo gap.

    Slides a fixed-width training window forward in time, leaving an
    embargo gap between training and test to prevent data leakage.

    Parameters
    ----------
    returns : pd.Series
        Actual returns indexed by datetime.
    predictions : np.ndarray
        Model predictions aligned to returns.
    train_window : int
        Training window size in bars.
    embargo : int
        Gap between train end and test start (bars).
    test_window : int
        Test window size in bars.
    slide_freq : str
        ``"weekly"`` (slide 5 bars) or ``"daily"`` (slide 1 bar).
    risk_free_rate : float
        Annual risk-free rate for Sharpe calculation.

    Returns
    -------
    WalkForwardEmbargoResult
    """
    pred_arr = np.asarray(predictions, dtype=float).ravel()
    ret_vals = returns.values.astype(float)
    n = min(len(ret_vals), len(pred_arr))
    ret_vals = ret_vals[:n]
    pred_arr = pred_arr[:n]
    index = returns.index[:n]

    slide = 5 if slide_freq == "weekly" else 1
    min_total = train_window + embargo + test_window
    warnings: List[str] = []

    if n < min_total:
        return WalkForwardEmbargoResult(
            n_folds=0, folds=[], mean_train_sharpe=0.0,
            mean_test_sharpe=0.0, mean_overfit_gap=0.0,
            is_overfit=True,
            warnings=["Insufficient data for walk-forward with embargo"],
        )

    folds: List[WalkForwardEmbargoFold] = []
    t = 0

    while t + min_total <= n:
        train_start = t
        train_end = t + train_window
        emb_start = train_end
        emb_end = emb_start + embargo
        test_start = emb_end
        test_end = min(test_start + test_window, n)

        if test_end - test_start < 10:
            break

        # Compute per-fold metrics
        train_ret = ret_vals[train_start:train_end]
        train_pred = pred_arr[train_start:train_end]
        test_ret = ret_vals[test_start:test_end]
        test_pred = pred_arr[test_start:test_end]

        train_sharpe = _sharpe(train_ret, risk_free_rate)
        test_sharpe = _sharpe(test_ret, risk_free_rate)
        train_ic = _spearman_ic(train_pred, train_ret)
        test_ic = _spearman_ic(test_pred, test_ret)

        folds.append(WalkForwardEmbargoFold(
            fold=len(folds) + 1,
            train_start=str(index[train_start]),
            train_end=str(index[train_end - 1]),
            embargo_start=str(index[emb_start]),
            embargo_end=str(index[min(emb_end - 1, n - 1)]),
            test_start=str(index[test_start]),
            test_end=str(index[test_end - 1]),
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            train_ic=train_ic,
            test_ic=test_ic,
            overfit_gap_sharpe=train_sharpe - test_sharpe,
            overfit_gap_ic=train_ic - test_ic,
            test_n_samples=test_end - test_start,
        ))

        t += slide

    if not folds:
        return WalkForwardEmbargoResult(
            n_folds=0, folds=[], mean_train_sharpe=0.0,
            mean_test_sharpe=0.0, mean_overfit_gap=0.0,
            is_overfit=True,
            warnings=["No valid folds produced"],
        )

    mean_train = float(np.mean([f.train_sharpe for f in folds]))
    mean_test = float(np.mean([f.test_sharpe for f in folds]))
    mean_gap = float(np.mean([f.overfit_gap_sharpe for f in folds]))

    is_overfit = mean_gap > 0.10 or mean_test <= 0
    if mean_gap > 0.10:
        warnings.append(f"Mean overfit gap = {mean_gap:.3f} > 0.10")
    if mean_test <= 0:
        warnings.append(f"Mean OOS Sharpe = {mean_test:.3f} <= 0")

    return WalkForwardEmbargoResult(
        n_folds=len(folds),
        folds=folds,
        mean_train_sharpe=mean_train,
        mean_test_sharpe=mean_test,
        mean_overfit_gap=mean_gap,
        is_overfit=is_overfit,
        warnings=warnings,
    )


# ── Rolling IC and Decay Detection (Spec 08 T3) ─────────────────────


def rolling_ic(
    predictions: np.ndarray,
    returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Compute rolling information coefficient (Spearman rank correlation).

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions.
    returns : pd.Series
        Actual returns (same length, same index).
    window : int
        Rolling window size in bars.

    Returns
    -------
    pd.Series
        Rolling IC values indexed by date. NaN for the initial ``window-1``
        bars.
    """
    pred_arr = np.asarray(predictions, dtype=float).ravel()
    ret_arr = returns.values.astype(float)
    n = min(len(pred_arr), len(ret_arr))
    pred_arr = pred_arr[:n]
    ret_arr = ret_arr[:n]
    index = returns.index[:n]

    ic_vals = np.full(n, np.nan, dtype=float)

    for i in range(window, n):
        p = pred_arr[i - window:i]
        r = ret_arr[i - window:i]
        valid = np.isfinite(p) & np.isfinite(r)
        if valid.sum() < 10:
            continue
        ic_vals[i] = _spearman_ic(p[valid], r[valid])

    return pd.Series(ic_vals, index=index, name="rolling_ic")


def detect_ic_decay(
    ic_series: pd.Series,
    decay_threshold: float = 0.02,
    window: int = 20,
) -> tuple:
    """Detect whether the information coefficient is decaying.

    Parameters
    ----------
    ic_series : pd.Series
        Rolling IC values (from :func:`rolling_ic`).
    decay_threshold : float
        IC values below this are considered weak.
    window : int
        Number of recent observations to check.

    Returns
    -------
    tuple[bool, dict]
        ``(decaying, info_dict)`` where ``info_dict`` has keys:
        ``current_ic``, ``mean_ic``, ``slope``, ``days_below_threshold``,
        ``pct_below_threshold``.
    """
    ic_clean = ic_series.dropna()

    if len(ic_clean) < window:
        return False, {
            "current_ic": float(ic_clean.iloc[-1]) if len(ic_clean) > 0 else 0.0,
            "mean_ic": float(ic_clean.mean()) if len(ic_clean) > 0 else 0.0,
            "slope": 0.0,
            "days_below_threshold": 0,
            "pct_below_threshold": 0.0,
            "warning": "Insufficient data for decay detection",
        }

    recent = ic_clean.iloc[-window:]
    current_ic = float(recent.iloc[-1])
    mean_ic = float(recent.mean())
    days_below = int((recent < decay_threshold).sum())
    pct_below = days_below / len(recent)

    # Linear trend of IC (slope via OLS)
    x = np.arange(len(ic_clean), dtype=float)
    y = ic_clean.values.astype(float)
    if len(x) > 2:
        # Fit linear regression
        x_mean = x.mean()
        y_mean = y.mean()
        slope = float(np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-12))
    else:
        slope = 0.0

    # Decay = IC below threshold for 80%+ of recent window OR declining slope
    decaying = (pct_below >= 0.80) or (slope < -0.001 and mean_ic < decay_threshold * 2)

    return decaying, {
        "current_ic": current_ic,
        "mean_ic": mean_ic,
        "slope": slope,
        "days_below_threshold": days_below,
        "pct_below_threshold": pct_below,
    }


# ── Private helpers ──────────────────────────────────────────────────


def _sharpe(returns: np.ndarray, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Annualized Sharpe ratio from a return array."""
    return compute_sharpe(returns, rf_annual=risk_free_rate, frequency="daily", annualize=True)


def _spearman_ic(predictions: np.ndarray, returns: np.ndarray) -> float:
    """Spearman rank correlation between predictions and returns."""
    if len(predictions) < 3:
        return 0.0
    corr, _ = stats.spearmanr(predictions, returns)
    return float(corr) if not np.isnan(corr) else 0.0


def attach_null_baselines(
    stat_result: StatisticalTests,
    ohlcv_dict: Dict[str, pd.DataFrame],
    null_baseline_enabled: bool = True,
) -> StatisticalTests:
    """Optionally compute and attach null model baselines to a StatisticalTests result.

    Parameters
    ----------
    stat_result : StatisticalTests
        Previously computed statistical test results.
    ohlcv_dict : dict
        Universe OHLCV data for null model computation.
    null_baseline_enabled : bool
        Whether to compute null baselines.

    Returns
    -------
    StatisticalTests
        The same result with ``null_comparison`` populated if enabled.
    """
    if not null_baseline_enabled:
        return stat_result

    from .null_models import compute_null_baselines

    try:
        null_results = compute_null_baselines(ohlcv_dict)
        stat_result.null_comparison = null_results
        _logger.info("Null baselines attached to validation results")
    except Exception:
        _logger.warning("Failed to compute null baselines", exc_info=True)

    return stat_result
