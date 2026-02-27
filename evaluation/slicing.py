"""
Performance Slicing — regime-aware and condition-based return decomposition.

Partitions backtest returns into meaningful segments (regime-based, stress-based,
or custom) so that per-slice metrics reveal hidden regime dependence.

Primary slices (5):
    1. Normal — regime 0 (trending bull) or 1 (trending bear), no high-vol overlay
    2. High Vol — regime 3 (high_volatility)
    3. Crash — regime 3 AND cumulative return < -10%
    4. Recovery — regime 3 AND trailing 20-day return > 5%
    5. Trendless — regime 2 (mean_reverting) AND volatility > 1.5x rolling median
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import EVAL_MIN_SLICE_SAMPLES

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSlice:
    """A named filter that selects a subset of returns for separate evaluation.

    Parameters
    ----------
    name : str
        Human-readable slice label (e.g., "High Vol").
    condition : callable
        ``(metadata: pd.DataFrame) -> np.ndarray[bool]``
        Returns a boolean mask selecting rows that belong to this slice.
    min_samples : int
        Minimum observations required for reliable statistics.
    """

    name: str
    condition: Callable[[pd.DataFrame], np.ndarray]
    min_samples: int = EVAL_MIN_SLICE_SAMPLES

    def apply(
        self,
        returns: pd.Series,
        metadata: pd.DataFrame,
    ) -> Tuple[pd.Series, Dict]:
        """Apply the slice condition and return filtered returns with metadata.

        Parameters
        ----------
        returns : pd.Series
            Full return series (same index as metadata).
        metadata : pd.DataFrame
            Must include columns used by ``self.condition`` (e.g., ``regime``,
            ``volatility``, ``cumulative_return``).

        Returns
        -------
        tuple[pd.Series, dict]
            Filtered returns and info dict with keys:
            ``name``, ``n_samples``, ``n_days``, ``date_range``, ``low_confidence``.
        """
        mask = np.asarray(self.condition(metadata), dtype=bool)

        # Align mask length to returns
        if len(mask) != len(returns):
            logger.warning(
                "Slice '%s': mask length (%d) != returns length (%d); truncating",
                self.name, len(mask), len(returns),
            )
            min_len = min(len(mask), len(returns))
            mask = mask[:min_len]
            filtered = returns.iloc[:min_len][mask]
        else:
            filtered = returns[mask]

        n_samples = len(filtered)
        low_confidence = n_samples < self.min_samples

        if n_samples == 0:
            date_range = ("N/A", "N/A")
        elif hasattr(filtered.index, 'min'):
            date_range = (str(filtered.index.min()), str(filtered.index.max()))
        else:
            date_range = (str(filtered.index[0]), str(filtered.index[-1]))

        # Count unique dates (for multi-asset panels, n_days != n_samples)
        if hasattr(filtered.index, 'get_level_values'):
            try:
                dates = filtered.index.get_level_values(-1)
                n_days = dates.nunique()
            except Exception:
                n_days = n_samples
        else:
            n_days = n_samples

        info = {
            "name": self.name,
            "n_samples": n_samples,
            "n_days": n_days,
            "date_range": date_range,
            "low_confidence": low_confidence,
        }

        return filtered, info


class SliceRegistry:
    """Factory for standard performance slices."""

    @staticmethod
    def create_regime_slices(regime_states: np.ndarray) -> List[PerformanceSlice]:
        """Create the 5 primary regime-based slices.

        Parameters
        ----------
        regime_states : np.ndarray
            Integer array of regime labels (0-3) aligned with the return series.
            Used inside closures to build slice masks.

        Returns
        -------
        list[PerformanceSlice]
            Five slices: Normal, High Vol, Crash, Recovery, Trendless.
        """
        regimes = np.asarray(regime_states, dtype=int)

        # 1. Normal: trending bull (0) or trending bear (1)
        def _normal(meta: pd.DataFrame) -> np.ndarray:
            r = meta["regime"].values if "regime" in meta.columns else regimes
            return np.isin(r, [0, 1])

        # 2. High Vol: regime 3
        def _high_vol(meta: pd.DataFrame) -> np.ndarray:
            r = meta["regime"].values if "regime" in meta.columns else regimes
            return r == 3

        # 3. Crash: regime 3 AND cumulative return < -10%
        def _crash(meta: pd.DataFrame) -> np.ndarray:
            r = meta["regime"].values if "regime" in meta.columns else regimes
            is_hv = r == 3
            if "cumulative_return" in meta.columns:
                cum_ret = meta["cumulative_return"].values
                return is_hv & (cum_ret < -0.10)
            # Fallback: use drawdown if available
            if "drawdown" in meta.columns:
                dd = meta["drawdown"].values
                return is_hv & (dd < -0.10)
            return is_hv  # Degrade gracefully

        # 4. Recovery: regime 3 AND trailing return > 5%
        def _recovery(meta: pd.DataFrame) -> np.ndarray:
            r = meta["regime"].values if "regime" in meta.columns else regimes
            is_hv = r == 3
            if "trailing_return_20d" in meta.columns:
                trail = meta["trailing_return_20d"].values
                return is_hv & (trail > 0.05)
            return np.zeros(len(meta), dtype=bool)

        # 5. Trendless: regime 2 AND volatility > 1.5x rolling median
        def _trendless(meta: pd.DataFrame) -> np.ndarray:
            r = meta["regime"].values if "regime" in meta.columns else regimes
            is_mr = r == 2
            if "volatility" in meta.columns and "volatility_median" in meta.columns:
                vol = meta["volatility"].values
                vol_med = meta["volatility_median"].values
                return is_mr & (vol > 1.5 * vol_med)
            # Fallback: just mean-reverting regime
            return is_mr

        return [
            PerformanceSlice(name="Normal", condition=_normal),
            PerformanceSlice(name="High Vol", condition=_high_vol),
            PerformanceSlice(name="Crash", condition=_crash),
            PerformanceSlice(name="Recovery", condition=_recovery),
            PerformanceSlice(name="Trendless", condition=_trendless),
        ]

    @staticmethod
    def create_individual_regime_slices() -> List[PerformanceSlice]:
        """Create one slice per canonical regime (0-3) using metadata columns.

        Returns
        -------
        list[PerformanceSlice]
            Four slices, one per regime code.
        """
        from ..config import REGIME_NAMES

        slices = []
        for code in range(4):
            label = REGIME_NAMES.get(code, f"regime_{code}")

            def _make_cond(regime_code: int):
                def _cond(meta: pd.DataFrame) -> np.ndarray:
                    if "regime" in meta.columns:
                        return meta["regime"].values == regime_code
                    return np.zeros(len(meta), dtype=bool)
                return _cond

            slices.append(
                PerformanceSlice(name=label, condition=_make_cond(code))
            )
        return slices

    @staticmethod
    def create_uncertainty_slices(
        uncertainty: np.ndarray,
        n_quantiles: int = 3,
    ) -> List[PerformanceSlice]:
        """Create slices based on regime uncertainty quantiles.

        Partitions bars by their regime-uncertainty level (entropy of posterior
        probabilities), so that strategy performance can be evaluated separately
        during low-, mid-, and high-uncertainty periods.

        Parameters
        ----------
        uncertainty : np.ndarray
            Per-bar uncertainty values (typically entropy in [0, 1]).
        n_quantiles : int
            Number of quantile buckets (default 3: low / mid / high).

        Returns
        -------
        list[PerformanceSlice]
            ``n_quantiles`` slices named ``uncertainty_q1`` … ``uncertainty_qN``.
        """
        unc = np.asarray(uncertainty, dtype=float)
        quantile_edges = np.linspace(0, 1, n_quantiles + 1)
        thresholds = np.nanquantile(unc, quantile_edges)

        slices: List[PerformanceSlice] = []
        for i in range(n_quantiles):
            low = thresholds[i]
            high = thresholds[i + 1]
            label = f"uncertainty_q{i + 1}"
            is_first = i == 0
            is_last = i == n_quantiles - 1

            def _make_cond(lo: float, hi: float, first: bool, last: bool):
                def _cond(meta: pd.DataFrame) -> np.ndarray:
                    if "uncertainty" in meta.columns:
                        u = meta["uncertainty"].values
                    else:
                        u = unc[:len(meta)]
                    if first and last:
                        # Single quantile — select everything
                        return np.ones(len(u), dtype=bool)
                    elif first:
                        # First quantile: capture all values below upper edge
                        return u < hi
                    elif last:
                        # Last quantile: capture all values at or above lower edge
                        return u >= lo
                    else:
                        return (u >= lo) & (u < hi)
                return _cond

            slices.append(
                PerformanceSlice(
                    name=label,
                    condition=_make_cond(low, high, is_first, is_last),
                )
            )
        return slices

    @staticmethod
    def create_transition_slices(
        regimes: np.ndarray,
        window: int = 5,
    ) -> List[PerformanceSlice]:
        """Create slices based on proximity to regime transitions.

        A bar is ``near_transition`` if a regime change occurs within
        ``window`` bars (centred) of that bar.  All other bars are
        ``stable_regime``.

        Parameters
        ----------
        regimes : np.ndarray
            Integer regime labels per bar.
        window : int
            Lookback/lookahead window for transition proximity (default 5).

        Returns
        -------
        list[PerformanceSlice]
            Two slices: ``near_transition`` and ``stable_regime``.
        """
        regime_series = pd.Series(np.asarray(regimes, dtype=int))
        changes = regime_series.diff().abs() > 0
        near_transition = (
            changes.rolling(window, center=True, min_periods=1)
            .max()
            .fillna(0)
            .astype(bool)
        )
        near_mask = near_transition.values
        stable_mask = ~near_mask

        def _make_near_cond(mask: np.ndarray):
            def _cond(meta: pd.DataFrame) -> np.ndarray:
                if len(mask) == len(meta):
                    return mask
                return mask[:len(meta)]
            return _cond

        def _make_stable_cond(mask: np.ndarray):
            def _cond(meta: pd.DataFrame) -> np.ndarray:
                if len(mask) == len(meta):
                    return mask
                return mask[:len(meta)]
            return _cond

        return [
            PerformanceSlice(name="near_transition", condition=_make_near_cond(near_mask)),
            PerformanceSlice(name="stable_regime", condition=_make_stable_cond(stable_mask)),
        ]

    @staticmethod
    def build_metadata(
        returns: pd.Series,
        regime_states: np.ndarray,
        volatility: Optional[pd.Series] = None,
        uncertainty: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Build the metadata DataFrame expected by slice conditions.

        Parameters
        ----------
        returns : pd.Series
            Per-bar return series.
        regime_states : np.ndarray
            Regime labels per bar.
        volatility : pd.Series, optional
            Per-bar volatility (e.g., NATR or realized vol).
        uncertainty : np.ndarray, optional
            Per-bar regime uncertainty (entropy of posterior probabilities, [0, 1]).

        Returns
        -------
        pd.DataFrame
            Columns: regime, cumulative_return, drawdown, trailing_return_20d,
            volatility, volatility_median, and optionally uncertainty.
        """
        n = len(returns)
        meta = pd.DataFrame(index=returns.index)

        # Regime
        if len(regime_states) == n:
            meta["regime"] = regime_states
        else:
            meta["regime"] = np.full(n, 2, dtype=int)

        # Cumulative return (from start of series)
        cum_ret = (1 + returns).cumprod() - 1
        meta["cumulative_return"] = cum_ret.values

        # Drawdown
        cum_eq = (1 + returns).cumprod()
        running_max = cum_eq.cummax()
        meta["drawdown"] = ((cum_eq - running_max) / running_max).values

        # Trailing 20-day return
        trail_20 = returns.rolling(20, min_periods=1).sum()
        meta["trailing_return_20d"] = trail_20.values

        # Volatility
        if volatility is not None and len(volatility) == n:
            meta["volatility"] = volatility.values
            meta["volatility_median"] = volatility.rolling(
                252, min_periods=20
            ).median().values
        else:
            # Compute from returns as fallback
            vol = returns.rolling(20, min_periods=5).std() * np.sqrt(252)
            meta["volatility"] = vol.values
            meta["volatility_median"] = vol.rolling(
                252, min_periods=20
            ).median().values

        # Uncertainty
        if uncertainty is not None:
            unc_arr = np.asarray(uncertainty, dtype=float)
            if len(unc_arr) == n:
                meta["uncertainty"] = unc_arr
            else:
                logger.warning(
                    "Uncertainty length (%d) != returns length (%d); skipping",
                    len(unc_arr), n,
                )

        meta = meta.fillna(0.0)
        return meta
