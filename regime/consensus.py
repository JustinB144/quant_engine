"""Cross-Sectional Regime Consensus (SPEC_10 T6).

Computes the fraction of securities in each regime to determine market-wide
regime consensus.  High consensus (>80% in same regime) indicates a clear
market-wide regime.  Low consensus (<60%) or rapid drops in consensus serve
as early warnings for regime transitions.

Usage::

    consensus = RegimeConsensus()
    result = consensus.compute_consensus([0, 0, 0, 1, 0, 3])
    # result = {'consensus': 0.667, 'regime_pcts': {...}, 'consensus_regime': 0, ...}

    diverging, details = consensus.detect_divergence(consensus_history)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeConsensus:
    """Cross-sectional regime consensus analysis.

    Measures agreement across securities about the current market regime
    and detects when consensus is breaking down (potential regime transition).

    Parameters
    ----------
    consensus_threshold : float
        Fraction above which consensus is considered "high" (default 0.80).
    early_warning_threshold : float
        Fraction below which an early warning is triggered (default 0.60).
    divergence_window : int
        Lookback window (days) for trend-based divergence detection.
    divergence_slope_threshold : float
        Slope below which consensus is flagged as diverging (default -0.01).
    n_regimes : int
        Number of distinct regimes.
    """

    def __init__(
        self,
        consensus_threshold: Optional[float] = None,
        early_warning_threshold: Optional[float] = None,
        divergence_window: Optional[int] = None,
        divergence_slope_threshold: Optional[float] = None,
        n_regimes: int = 4,
    ):
        from ..config import (
            REGIME_CONSENSUS_THRESHOLD,
            REGIME_CONSENSUS_EARLY_WARNING,
            REGIME_CONSENSUS_DIVERGENCE_WINDOW,
            REGIME_CONSENSUS_DIVERGENCE_SLOPE,
        )

        self.consensus_threshold = (
            consensus_threshold if consensus_threshold is not None
            else REGIME_CONSENSUS_THRESHOLD
        )
        self.early_warning_threshold = (
            early_warning_threshold if early_warning_threshold is not None
            else REGIME_CONSENSUS_EARLY_WARNING
        )
        self.divergence_window = (
            divergence_window if divergence_window is not None
            else REGIME_CONSENSUS_DIVERGENCE_WINDOW
        )
        self.divergence_slope_threshold = (
            divergence_slope_threshold if divergence_slope_threshold is not None
            else REGIME_CONSENSUS_DIVERGENCE_SLOPE
        )
        self.n_regimes = n_regimes

        # History for divergence detection
        self._consensus_history: List[float] = []

    def compute_consensus(
        self,
        regime_per_security: Sequence[int],
    ) -> Dict:
        """Compute cross-sectional regime consensus.

        Parameters
        ----------
        regime_per_security : sequence of int
            Regime label for each security in the universe.

        Returns
        -------
        dict with keys:
            - ``consensus`` : float — max fraction in any single regime
            - ``regime_pcts`` : dict — {regime: fraction}
            - ``consensus_regime`` : int — regime with highest count
            - ``n_securities`` : int — total number of securities
            - ``is_high_consensus`` : bool — consensus >= threshold
            - ``is_early_warning`` : bool — consensus < early_warning_threshold
        """
        regimes = np.asarray(regime_per_security, dtype=int)
        n_total = len(regimes)

        # Validate and filter out-of-range regime labels
        for label in regimes:
            if label < 0 or label >= self.n_regimes:
                logger.warning("Out-of-range regime label %d in consensus input", int(label))
        valid_mask = (regimes >= 0) & (regimes < self.n_regimes)
        regimes = regimes[valid_mask]
        n_total = len(regimes)

        if n_total == 0:
            return {
                "consensus": 0.0,
                "regime_pcts": {},
                "consensus_regime": -1,
                "n_securities": 0,
                "is_high_consensus": False,
                "is_early_warning": True,
            }

        # Count per regime
        regime_pcts: Dict[int, float] = {}
        for r in range(self.n_regimes):
            count = (regimes == r).sum()
            regime_pcts[r] = count / n_total

        consensus_regime = int(max(regime_pcts, key=regime_pcts.get))
        consensus_value = regime_pcts[consensus_regime]

        # Update history
        self._consensus_history.append(consensus_value)

        return {
            "consensus": float(consensus_value),
            "regime_pcts": regime_pcts,
            "consensus_regime": consensus_regime,
            "n_securities": n_total,
            "is_high_consensus": bool(consensus_value >= self.consensus_threshold),
            "is_early_warning": bool(consensus_value < self.early_warning_threshold),
        }

    def detect_divergence(
        self,
        consensus_history: Optional[Sequence[float]] = None,
        window: Optional[int] = None,
    ) -> Tuple[bool, Dict]:
        """Detect if consensus is diverging (falling over time).

        Fits a linear trend to the consensus history over the lookback
        window.  If the slope is below the threshold, flags divergence.

        Parameters
        ----------
        consensus_history : sequence of float or None
            If None, uses internal history from ``compute_consensus`` calls.
        window : int or None
            Override the lookback window.

        Returns
        -------
        (diverging, details) : tuple
            ``diverging`` is True if consensus is falling significantly.
            ``details`` is a dict with slope, current consensus, etc.
        """
        history = (
            list(consensus_history) if consensus_history is not None
            else list(self._consensus_history)
        )
        win = window if window is not None else self.divergence_window

        if len(history) < max(3, win // 2):
            return False, {
                "slope": 0.0,
                "current_consensus": history[-1] if history else 0.0,
                "window": win,
                "n_observations": len(history),
                "reason": "insufficient_history",
            }

        # Use last `window` values
        recent = np.array(history[-win:])
        x = np.arange(len(recent), dtype=float)

        # Fit linear trend: y = slope * x + intercept
        if len(x) < 2:
            slope = 0.0
        else:
            slope = float(np.polyfit(x, recent, 1)[0])

        current = float(recent[-1])
        avg = float(recent.mean())
        diverging = slope < self.divergence_slope_threshold

        return diverging, {
            "slope": slope,
            "current_consensus": current,
            "historical_avg_consensus": avg,
            "window": win,
            "n_observations": len(recent),
        }

    def early_warning(
        self,
        consensus: float,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Check if consensus is below the early warning level.

        Parameters
        ----------
        consensus : float
            Current consensus value.
        threshold : float or None
            Override the early warning threshold.

        Returns
        -------
        (warning, reason) : tuple
            ``warning`` is True if consensus < threshold.
        """
        thr = threshold if threshold is not None else self.early_warning_threshold

        if consensus < thr:
            return True, (
                f"Consensus {consensus:.2%} below early warning threshold "
                f"{thr:.2%} — regime transition may be imminent"
            )
        return False, ""

    def compute_consensus_series(
        self,
        regime_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute consensus metrics for a full time series.

        Parameters
        ----------
        regime_matrix : pd.DataFrame
            Rows = dates, columns = securities, values = regime labels (int).

        Returns
        -------
        pd.DataFrame
            Columns: ``consensus``, ``consensus_regime``, ``is_high_consensus``,
            ``is_early_warning``, plus ``regime_pct_0`` through ``regime_pct_3``.
        """
        results = []
        self._consensus_history.clear()

        for date_idx in regime_matrix.index:
            row = regime_matrix.loc[date_idx].dropna().astype(int).values
            if len(row) == 0:
                results.append({
                    "consensus": np.nan,
                    "consensus_regime": -1,
                    "is_high_consensus": False,
                    "is_early_warning": True,
                })
                continue

            result = self.compute_consensus(row)
            row_data = {
                "consensus": result["consensus"],
                "consensus_regime": result["consensus_regime"],
                "is_high_consensus": result["is_high_consensus"],
                "is_early_warning": result["is_early_warning"],
            }
            for r in range(self.n_regimes):
                row_data[f"regime_pct_{r}"] = result["regime_pcts"].get(r, 0.0)

            results.append(row_data)

        return pd.DataFrame(results, index=regime_matrix.index)

    def reset_history(self) -> None:
        """Clear internal consensus history."""
        self._consensus_history.clear()
