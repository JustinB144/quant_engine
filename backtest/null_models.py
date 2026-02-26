"""
Null model baselines — Truth Layer T4.

Provides three reference baselines to anchor backtest performance
evaluation.  Every strategy should meaningfully outperform these
null models; failure to do so is a strong overfitting signal.

Baselines:
    RandomBaseline    — randomly long/short each stock (coin flip)
    ZeroBaseline      — always flat (no positions)
    MomentumBaseline  — long if ROC(lookback) > 0, short otherwise

Usage:
    from quant_engine.backtest.null_models import (
        RandomBaseline, ZeroBaseline, MomentumBaseline,
    )

    random_preds = RandomBaseline(seed=42).generate_predictions(ohlcv_dict)
    zero_preds = ZeroBaseline().generate_predictions(ohlcv_dict)
    momentum_preds = MomentumBaseline(lookback=20).generate_predictions(ohlcv_dict)
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class NullBaselineMetrics:
    """Lightweight performance metrics for a null model baseline."""
    name: str
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0


@dataclass
class NullModelResults:
    """Aggregate results for all three null model baselines."""
    random: NullBaselineMetrics = field(default_factory=lambda: NullBaselineMetrics(name="random"))
    zero: NullBaselineMetrics = field(default_factory=lambda: NullBaselineMetrics(name="zero"))
    momentum: NullBaselineMetrics = field(default_factory=lambda: NullBaselineMetrics(name="momentum"))

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return dict of {baseline_name: {metric: value}}."""
        result = {}
        for baseline in [self.random, self.zero, self.momentum]:
            result[baseline.name] = {
                "total_return": baseline.total_return,
                "sharpe_ratio": baseline.sharpe_ratio,
                "max_drawdown": baseline.max_drawdown,
                "win_rate": baseline.win_rate,
                "n_trades": baseline.n_trades,
            }
        return result


class RandomBaseline:
    """Randomly long/short each stock with fixed probability.

    Parameters
    ----------
    long_prob : float
        Probability of going long on each bar.  Default 0.5 (fair coin).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, long_prob: float = 0.50, seed: int = 42):
        self.long_prob = long_prob
        self.seed = seed

    def generate_signals(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.Series]:
        """Generate random +1/-1 signals for each ticker.

        Parameters
        ----------
        ohlcv_dict : dict[str, pd.DataFrame]
            Mapping of ticker/permno -> OHLCV DataFrame.

        Returns
        -------
        dict[str, pd.Series]
            Mapping of ticker -> signal Series (+1 long, -1 short).
        """
        rng = np.random.RandomState(self.seed)
        signals: Dict[str, pd.Series] = {}

        for ticker, df in ohlcv_dict.items():
            n = len(df)
            raw = rng.choice(
                [-1, 1], size=n, p=[1 - self.long_prob, self.long_prob],
            )
            signals[ticker] = pd.Series(raw, index=df.index, name="signal", dtype=float)

        return signals

    def compute_returns(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
    ) -> NullBaselineMetrics:
        """Compute baseline returns from random signals.

        Uses next-bar-open entry convention to avoid look-ahead bias.
        """
        signals = self.generate_signals(ohlcv_dict)
        all_returns: List[float] = []

        for ticker, df in ohlcv_dict.items():
            if ticker not in signals:
                continue
            sig = signals[ticker]
            close = df["Close"].astype(float)
            daily_ret = close.pct_change().shift(-1)  # next-day return

            # Signal on bar t, return on bar t+1
            strategy_ret = sig.shift(1) * daily_ret
            strategy_ret = strategy_ret.dropna()
            all_returns.extend(strategy_ret.tolist())

        return _compute_metrics("random", all_returns)


class ZeroBaseline:
    """Always flat — zero positions, zero return.

    The zero baseline establishes the lower bound: any strategy should
    at least beat doing nothing (accounting for transaction costs).
    """

    def generate_signals(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.Series]:
        """Generate zero signals (always flat)."""
        signals: Dict[str, pd.Series] = {}
        for ticker, df in ohlcv_dict.items():
            signals[ticker] = pd.Series(
                0.0, index=df.index, name="signal", dtype=float,
            )
        return signals

    def compute_returns(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
    ) -> NullBaselineMetrics:
        """Zero baseline always returns zero."""
        return NullBaselineMetrics(
            name="zero",
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            n_trades=0,
        )


class MomentumBaseline:
    """Long if ROC(lookback) > 0, short otherwise.

    A simple momentum strategy serves as a benchmark for whether the
    ML model captures information beyond naive trend following.

    Parameters
    ----------
    lookback : int
        Number of bars for rate-of-change computation.  Default 20.
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def generate_signals(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.Series]:
        """Generate momentum-based +1/-1 signals."""
        signals: Dict[str, pd.Series] = {}

        for ticker, df in ohlcv_dict.items():
            close = df["Close"].astype(float)
            roc = close.pct_change(self.lookback)
            sig = (roc > 0).astype(float) * 2 - 1  # +1 if positive momentum, -1 otherwise
            signals[ticker] = pd.Series(sig, index=df.index, name="signal")

        return signals

    def compute_returns(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
    ) -> NullBaselineMetrics:
        """Compute baseline returns from momentum signals."""
        signals = self.generate_signals(ohlcv_dict)
        all_returns: List[float] = []

        for ticker, df in ohlcv_dict.items():
            if ticker not in signals:
                continue
            sig = signals[ticker]
            close = df["Close"].astype(float)
            daily_ret = close.pct_change().shift(-1)

            strategy_ret = sig.shift(1) * daily_ret
            strategy_ret = strategy_ret.dropna()
            all_returns.extend(strategy_ret.tolist())

        return _compute_metrics("momentum", all_returns)


def _compute_metrics(name: str, returns: List[float]) -> NullBaselineMetrics:
    """Compute basic performance metrics from a flat list of daily returns."""
    if not returns:
        return NullBaselineMetrics(name=name)

    ret_arr = np.array(returns, dtype=float)
    ret_arr = ret_arr[np.isfinite(ret_arr)]

    if len(ret_arr) == 0:
        return NullBaselineMetrics(name=name)

    total_return = float(np.prod(1 + ret_arr) - 1)

    # Annualized Sharpe (assume 252 trading days)
    mean_ret = float(np.mean(ret_arr))
    std_ret = float(np.std(ret_arr, ddof=1)) if len(ret_arr) > 1 else 1e-10
    sharpe = (mean_ret / max(std_ret, 1e-10)) * np.sqrt(252)

    # Max drawdown
    cum = np.cumprod(1 + ret_arr)
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum / running_max - 1
    max_dd = float(np.min(drawdowns))

    # Win rate
    n_positive = int(np.sum(ret_arr > 0))
    win_rate = n_positive / max(len(ret_arr), 1)

    return NullBaselineMetrics(
        name=name,
        total_return=round(total_return, 6),
        sharpe_ratio=round(float(sharpe), 4),
        max_drawdown=round(max_dd, 6),
        win_rate=round(win_rate, 4),
        n_trades=len(ret_arr),
    )


def compute_null_baselines(
    ohlcv_dict: Dict[str, pd.DataFrame],
    random_seed: int = 42,
    momentum_lookback: int = 20,
) -> NullModelResults:
    """Compute all three null model baselines for a universe.

    Parameters
    ----------
    ohlcv_dict : dict[str, pd.DataFrame]
        Universe OHLCV data.
    random_seed : int
        Seed for the random baseline.
    momentum_lookback : int
        Lookback period for the momentum baseline.

    Returns
    -------
    NullModelResults
        Aggregate metrics for random, zero, and momentum baselines.
    """
    logger.info("Computing null model baselines...")

    random_metrics = RandomBaseline(seed=random_seed).compute_returns(ohlcv_dict)
    zero_metrics = ZeroBaseline().compute_returns(ohlcv_dict)
    momentum_metrics = MomentumBaseline(lookback=momentum_lookback).compute_returns(ohlcv_dict)

    results = NullModelResults(
        random=random_metrics,
        zero=zero_metrics,
        momentum=momentum_metrics,
    )

    logger.info(
        "Null baselines computed — Random Sharpe=%.3f, Zero Sharpe=%.3f, Momentum Sharpe=%.3f",
        random_metrics.sharpe_ratio,
        zero_metrics.sharpe_ratio,
        momentum_metrics.sharpe_ratio,
    )

    return results
