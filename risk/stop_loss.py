"""
Stop Loss Manager — regime-aware ATR stops, trailing, time, and regime-change stops.

Implements multiple stop-loss strategies that can be composed:
    - ATR-based initial stop (regime-adjusted)
    - Trailing stop (regime-adjusted, locks in gains)
    - Time-based stop (exit after max holding period)
    - Regime-change stop (exit on regime transition)
    - Hard percentage stop (absolute limit)

Regime conditioning:
    ATR-based stop distances are scaled by REGIME_STOP_MULTIPLIER so that
    stops widen in high-volatility / mean-reverting regimes (avoiding noise
    stops) and tighten in trending-bear regimes (cutting losses faster).
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from ..config import REGIME_STOP_MULTIPLIER
from ..config import (HARD_STOP_PCT, ATR_STOP_MULTIPLIER, TRAILING_ATR_MULTIPLIER,
    TRAILING_ACTIVATION_PCT, MAX_HOLDING_DAYS)


class StopReason(Enum):
    """Enumerated reasons a stop-loss evaluation can trigger an exit."""
    NONE = "none"
    ATR_STOP = "atr_stop"
    TRAILING_STOP = "trailing_stop"
    TIME_STOP = "time_stop"
    REGIME_CHANGE = "regime_change"
    HARD_STOP = "hard_stop"
    TARGET_HIT = "target_hit"


@dataclass
class StopResult:
    """Result of stop-loss evaluation."""
    should_exit: bool
    reason: StopReason
    stop_price: float       # Current stop level
    trailing_high: float    # Highest price since entry
    unrealized_pnl: float   # Current unrealized P&L
    bars_held: int
    details: dict


class StopLossManager:
    """
    Multi-strategy stop-loss manager.

    Evaluates all stop conditions and triggers exit on the first
    one that fires. Designed to work with the backtester for
    realistic exit simulation.

    Stop types (all optional, checked in order):
        1. Hard stop: -X% from entry, no exceptions
        2. ATR stop: entry_price - N * ATR
        3. Trailing stop: highest_price - N * ATR (ratchets up)
        4. Time stop: exit after max_days
        5. Regime change: exit if regime shifts
        6. Profit target: exit at +X% gain
    """

    def __init__(
        self,
        hard_stop_pct: float = HARD_STOP_PCT,          # -8% hard stop
        atr_stop_multiplier: float = ATR_STOP_MULTIPLIER,       # 2x ATR initial stop
        trailing_atr_multiplier: float = TRAILING_ATR_MULTIPLIER,   # 1.5x ATR trailing
        trailing_activation_pct: float = TRAILING_ACTIVATION_PCT,  # Activate trailing after +2%
        max_holding_days: int = MAX_HOLDING_DAYS,             # Time stop
        regime_change_exit: bool = True,        # Exit on regime change
        profit_target_pct: Optional[float] = None,  # Optional profit target
    ):
        """Initialize StopLossManager."""
        self.hard_stop = hard_stop_pct
        self.atr_mult = atr_stop_multiplier
        self.trail_mult = trailing_atr_multiplier
        self.trail_activation = trailing_activation_pct
        self.max_days = max_holding_days
        self.regime_exit = regime_change_exit
        self.profit_target = profit_target_pct

    def evaluate(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
        atr: float,
        bars_held: int,
        entry_regime: int,
        current_regime: int,
    ) -> StopResult:
        """
        Evaluate all stop conditions for an open position.

        Args:
            entry_price: original entry price
            current_price: current market price
            highest_price: highest price since entry (for trailing)
            atr: current ATR value
            bars_held: number of bars held
            entry_regime: regime at time of entry
            current_regime: current detected regime

        Returns:
            StopResult with exit decision and details
        """
        unrealized = (current_price - entry_price) / entry_price

        # ── Regime-conditional ATR scaling ──
        # Look up the stop multiplier for the current regime.  A multiplier
        # >1 widens ATR-based stops (high-vol / mean-revert), <1 tightens
        # them (trending-bear).
        regime_mult = REGIME_STOP_MULTIPLIER.get(current_regime, 1.0)
        effective_atr_mult = self.atr_mult * regime_mult
        effective_trail_mult = self.trail_mult * regime_mult

        # ── 1. Hard stop ──
        if unrealized <= self.hard_stop:
            return StopResult(
                should_exit=True,
                reason=StopReason.HARD_STOP,
                stop_price=entry_price * (1 + self.hard_stop),
                trailing_high=highest_price,
                unrealized_pnl=unrealized,
                bars_held=bars_held,
                details={"hard_stop_level": self.hard_stop},
            )

        # ── 2. ATR stop (regime-adjusted) ──
        atr_stop_price = entry_price - effective_atr_mult * atr
        if current_price <= atr_stop_price:
            return StopResult(
                should_exit=True,
                reason=StopReason.ATR_STOP,
                stop_price=atr_stop_price,
                trailing_high=highest_price,
                unrealized_pnl=unrealized,
                bars_held=bars_held,
                details={
                    "atr": atr,
                    "atr_multiplier": effective_atr_mult,
                    "regime_stop_multiplier": regime_mult,
                },
            )

        # ── 3. Trailing stop (regime-adjusted, only active after minimum gain) ──
        trailing_stop_price = 0.0
        if unrealized >= self.trail_activation:
            trailing_stop_price = highest_price - effective_trail_mult * atr
            if current_price <= trailing_stop_price:
                return StopResult(
                    should_exit=True,
                    reason=StopReason.TRAILING_STOP,
                    stop_price=trailing_stop_price,
                    trailing_high=highest_price,
                    unrealized_pnl=unrealized,
                    bars_held=bars_held,
                    details={
                        "trailing_high": highest_price,
                        "atr": atr,
                        "trail_multiplier": effective_trail_mult,
                        "regime_stop_multiplier": regime_mult,
                    },
                )

        # ── 4. Time stop ──
        if bars_held >= self.max_days:
            return StopResult(
                should_exit=True,
                reason=StopReason.TIME_STOP,
                stop_price=current_price,
                trailing_high=highest_price,
                unrealized_pnl=unrealized,
                bars_held=bars_held,
                details={"max_days": self.max_days},
            )

        # ── 5. Regime change stop ──
        if self.regime_exit and current_regime != entry_regime:
            return StopResult(
                should_exit=True,
                reason=StopReason.REGIME_CHANGE,
                stop_price=current_price,
                trailing_high=highest_price,
                unrealized_pnl=unrealized,
                bars_held=bars_held,
                details={
                    "entry_regime": entry_regime,
                    "current_regime": current_regime,
                },
            )

        # ── 6. Profit target ──
        if self.profit_target is not None and unrealized >= self.profit_target:
            return StopResult(
                should_exit=True,
                reason=StopReason.TARGET_HIT,
                stop_price=entry_price * (1 + self.profit_target),
                trailing_high=highest_price,
                unrealized_pnl=unrealized,
                bars_held=bars_held,
                details={"target_pct": self.profit_target},
            )

        # ── No stop triggered ──
        # Report the tightest active stop
        active_stop = max(atr_stop_price, trailing_stop_price)
        return StopResult(
            should_exit=False,
            reason=StopReason.NONE,
            stop_price=active_stop,
            trailing_high=highest_price,
            unrealized_pnl=unrealized,
            bars_held=bars_held,
            details={},
        )

    def compute_initial_stop(
        self, entry_price: float, atr: float, regime: int = 0,
    ) -> float:
        """
        Compute the initial stop-loss price for a new position.

        Returns the tighter of regime-adjusted ATR stop and hard stop.

        Args:
            entry_price: position entry price
            atr: current Average True Range
            regime: current market regime (used to look up REGIME_STOP_MULTIPLIER)
        """
        regime_mult = REGIME_STOP_MULTIPLIER.get(regime, 1.0)
        atr_stop = entry_price - self.atr_mult * regime_mult * atr
        hard_stop = entry_price * (1 + self.hard_stop)
        return max(atr_stop, hard_stop)  # tighter = higher price = less risk

    def compute_risk_per_share(
        self, entry_price: float, atr: float, regime: int = 0,
    ) -> float:
        """
        Compute risk per share based on regime-adjusted stop distance.

        Useful for position sizing: shares = risk_budget / risk_per_share

        Args:
            entry_price: position entry price
            atr: current Average True Range
            regime: current market regime (used to look up REGIME_STOP_MULTIPLIER)
        """
        stop = self.compute_initial_stop(entry_price, atr, regime=regime)
        return max(0.01, entry_price - stop)
