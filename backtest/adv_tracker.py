"""
Average Daily Volume (ADV) tracker with EMA smoothing and volume trend analysis.

Spec 06 T2: Provides explicit ADV computation, volume trend tracking, and
participation limit adjustment based on volume conditions.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ADVTracker:
    """Track per-symbol ADV with exponential moving average smoothing.

    Maintains a rolling history of daily volumes per symbol and computes
    an EMA-smoothed ADV estimate.  The volume trend (current volume vs EMA)
    is used to adjust participation limits and cost multipliers.

    Parameters
    ----------
    lookback_days : int
        Maximum number of daily observations to retain per symbol.
    ema_span : int
        EMA span parameter (higher = slower response to volume changes).
    low_volume_cost_mult : float
        Multiplicative cost penalty on below-average volume days.
        Applied as: cost_mult = 1.0 + (low_volume_cost_mult - 1.0) * shortfall
        where shortfall = max(0, 1 - volume_trend).
    """

    def __init__(
        self,
        lookback_days: int = 20,
        ema_span: int = 20,
        low_volume_cost_mult: float = 1.5,
    ):
        self.lookback_days = max(5, int(lookback_days))
        self.ema_span = max(2, int(ema_span))
        self.low_volume_cost_mult = float(max(1.0, low_volume_cost_mult))
        self._alpha = 2.0 / (self.ema_span + 1)

        # Per-symbol state
        self._volume_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.lookback_days)
        )
        self._ema: Dict[str, float] = {}
        self._volume_trend: Dict[str, float] = {}
        self._zero_vol_count: Dict[str, int] = defaultdict(int)

    def update(self, symbol: str, daily_volume: float) -> None:
        """Update ADV estimates with the latest daily volume observation.

        Parameters
        ----------
        symbol : str
            Ticker or PERMNO identifier.
        daily_volume : float
            Today's share volume (must be > 0 to be useful).
        """
        daily_volume = float(max(0.0, daily_volume))
        if daily_volume <= 0:
            self._zero_vol_count[symbol] += 1
            if self._zero_vol_count[symbol] % 10 == 0:
                logger.warning(
                    "ADV tracker: %d zero-volume days dropped for %s",
                    self._zero_vol_count[symbol], symbol,
                )
            return

        self._volume_history[symbol].append(daily_volume)

        # Initialize or update EMA
        if symbol not in self._ema:
            self._ema[symbol] = daily_volume
        else:
            self._ema[symbol] = (
                self._alpha * daily_volume
                + (1.0 - self._alpha) * self._ema[symbol]
            )

        # Volume trend: ratio of current volume to EMA, capped at [0.25, 3.0]
        ema_val = self._ema[symbol]
        if ema_val > 0:
            trend = float(np.clip(daily_volume / ema_val, 0.25, 3.0))
        else:
            trend = 1.0
        self._volume_trend[symbol] = trend

    def update_from_series(
        self, symbol: str, volumes: list[float],
    ) -> None:
        """Bulk-initialize ADV from a historical volume series.

        Processes volumes in chronological order to warm up the EMA.
        """
        for v in volumes:
            self.update(symbol, v)

    def get_adv(self, symbol: str) -> float:
        """Get current ADV estimate (EMA-smoothed).

        Returns 0.0 if symbol has no history.
        """
        return self._ema.get(symbol, 0.0)

    def get_simple_adv(self, symbol: str) -> float:
        """Get simple average daily volume (arithmetic mean of history).

        Returns 0.0 if symbol has no history.
        """
        history = self._volume_history.get(symbol)
        if not history:
            return 0.0
        return float(np.mean(list(history)))

    def get_volume_trend(self, symbol: str) -> float:
        """Get volume trend ratio (current volume / EMA).

        Values > 1.0 indicate above-average volume; < 1.0 below-average.
        Clipped to [0.25, 3.0].  Returns 1.0 if symbol has no history.
        """
        return self._volume_trend.get(symbol, 1.0)

    def adjust_participation_limit(
        self, symbol: str, base_limit: float,
    ) -> float:
        """Adjust participation limit based on volume trend.

        On high-volume days (trend > 1.0), the participation limit is
        widened proportionally.  On low-volume days, it is tightened.
        Result is clipped to [base_limit * 0.5, base_limit * 2.0].

        Parameters
        ----------
        symbol : str
            Ticker or PERMNO.
        base_limit : float
            Base participation limit (e.g. 0.02 for 2% of ADV).

        Returns
        -------
        float
            Adjusted participation limit.
        """
        trend = self.get_volume_trend(symbol)
        adjusted = base_limit * trend
        return float(np.clip(adjusted, base_limit * 0.5, base_limit * 2.0))

    def get_volume_cost_adjustment(self, symbol: str) -> float:
        """Compute cost multiplier adjustment based on volume trend.

        Below-average volume increases costs linearly:
          trend >= 1.0  → 1.0 (no penalty)
          trend  = 0.5  → 1.0 + 0.5 * (low_volume_cost_mult - 1.0)
          trend  = 0.0  → low_volume_cost_mult

        Above-average volume provides a modest discount:
          trend  = 1.5  → 0.95
          trend  = 2.0  → 0.90

        Returns
        -------
        float
            Cost multiplier in [0.85, low_volume_cost_mult].
        """
        trend = self.get_volume_trend(symbol)
        if trend >= 1.0:
            # Modest discount for high volume (capped at 15% discount)
            discount = min(0.15, (trend - 1.0) * 0.10)
            return max(0.85, 1.0 - discount)
        else:
            # Penalty for low volume
            shortfall = 1.0 - trend  # 0..1
            penalty = shortfall * (self.low_volume_cost_mult - 1.0)
            return 1.0 + penalty

    def get_stats(self, symbol: str) -> Dict[str, float]:
        """Return a summary dict for logging/diagnostics."""
        return {
            "adv_ema": self.get_adv(symbol),
            "adv_simple": self.get_simple_adv(symbol),
            "volume_trend": self.get_volume_trend(symbol),
            "history_length": len(self._volume_history.get(symbol, [])),
            "participation_adj": self.adjust_participation_limit(symbol, 0.02),
            "cost_adj": self.get_volume_cost_adjustment(symbol),
        }

    def __repr__(self) -> str:
        return (
            f"ADVTracker(lookback={self.lookback_days}, "
            f"ema_span={self.ema_span}, symbols={len(self._ema)})"
        )
