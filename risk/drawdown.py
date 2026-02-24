"""
Drawdown Controller — circuit breakers and recovery protocols.

Implements multi-tier drawdown protection:
    - Tier 1 (Warning): Reduce position sizes
    - Tier 2 (Caution): Halt new entries
    - Tier 3 (Critical): Liquidate positions
    - Daily and weekly loss limits
    - Recovery mode with gradual re-entry
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import (DRAWDOWN_WARNING_THRESHOLD, DRAWDOWN_CAUTION_THRESHOLD,
    DRAWDOWN_CRITICAL_THRESHOLD, DRAWDOWN_DAILY_LOSS_LIMIT,
    DRAWDOWN_WEEKLY_LOSS_LIMIT, DRAWDOWN_RECOVERY_DAYS,
    DRAWDOWN_SIZE_MULT_WARNING, DRAWDOWN_SIZE_MULT_CAUTION)


class DrawdownState(Enum):
    """Discrete drawdown-control states used by the drawdown controller."""
    NORMAL = "normal"
    WARNING = "warning"         # Tier 1: reduce sizes
    CAUTION = "caution"         # Tier 2: halt new entries
    CRITICAL = "critical"       # Tier 3: liquidate
    RECOVERY = "recovery"       # Gradual re-entry after crisis


@dataclass
class DrawdownStatus:
    """Current drawdown state and action directives."""
    state: DrawdownState
    current_drawdown: float     # Current drawdown from peak (negative)
    peak_equity: float          # High water mark
    current_equity: float
    size_multiplier: float      # Apply to all position sizes (0 = no trading)
    allow_new_entries: bool
    force_liquidate: bool
    days_in_state: int
    daily_pnl: float
    weekly_pnl: float
    messages: List[str] = field(default_factory=list)


class DrawdownController:
    """
    Multi-tier drawdown protection with circuit breakers.

    Tracks equity in real-time and enforces increasingly strict
    risk controls as drawdown deepens.

    Tiers:
        NORMAL   (DD > -3%):  Full position sizes, all entries allowed
        WARNING  (DD > -5%):  Reduce sizes by 50%, warn
        CAUTION  (DD > -10%): No new entries, reduce existing by 75%
        CRITICAL (DD > -15%): Liquidate all positions
        RECOVERY (improving): Gradual re-entry over N days
    """

    def __init__(
        self,
        warning_threshold: float = DRAWDOWN_WARNING_THRESHOLD,      # -5% drawdown
        caution_threshold: float = DRAWDOWN_CAUTION_THRESHOLD,       # -10% drawdown
        critical_threshold: float = DRAWDOWN_CRITICAL_THRESHOLD,      # -15% drawdown
        daily_loss_limit: float = DRAWDOWN_DAILY_LOSS_LIMIT,        # -3% daily loss limit
        weekly_loss_limit: float = DRAWDOWN_WEEKLY_LOSS_LIMIT,       # -5% weekly loss limit
        recovery_days: int = DRAWDOWN_RECOVERY_DAYS,                # Days to return to full sizing
        initial_equity: float = 1.0,
    ):
        """Initialize DrawdownController."""
        self.warning_thresh = warning_threshold
        self.caution_thresh = caution_threshold
        self.critical_thresh = critical_threshold
        self.daily_limit = daily_loss_limit
        self.weekly_limit = weekly_loss_limit
        self.recovery_days = recovery_days

        # State tracking
        self.peak_equity = initial_equity
        self.current_equity = initial_equity
        self.state = DrawdownState.NORMAL
        self.days_in_state = 0
        self.daily_pnl_history: List[float] = []
        self.state_history: List[DrawdownState] = []
        self._recovery_start_day = 0
        self._total_days = 0

    def update(self, daily_pnl: float) -> DrawdownStatus:
        """
        Update equity and return current drawdown status.

        Args:
            daily_pnl: today's portfolio PnL as fraction of equity

        Returns:
            DrawdownStatus with action directives
        """
        self._total_days += 1
        self.current_equity *= (1 + daily_pnl)
        self.daily_pnl_history.append(daily_pnl)

        # Update high water mark
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        # Current drawdown (guard against near-zero peak)
        dd = (self.current_equity - self.peak_equity) / max(self.peak_equity, 1e-12)

        # Daily and weekly PnL
        daily = daily_pnl
        weekly = sum(self.daily_pnl_history[-5:]) if len(self.daily_pnl_history) >= 5 else daily

        # Determine state
        prev_state = self.state
        messages = []

        # Check daily/weekly limits first (override tier logic)
        if daily < self.daily_limit:
            new_state = DrawdownState.CAUTION
            messages.append(f"Daily loss {daily:.1%} breached limit {self.daily_limit:.1%}")
        elif weekly < self.weekly_limit:
            new_state = DrawdownState.CAUTION
            messages.append(f"Weekly loss {weekly:.1%} breached limit {self.weekly_limit:.1%}")
        elif dd <= self.critical_thresh:
            new_state = DrawdownState.CRITICAL
            messages.append(f"CRITICAL: Drawdown {dd:.1%} breached {self.critical_thresh:.1%}")
        elif dd <= self.caution_thresh:
            new_state = DrawdownState.CAUTION
            messages.append(f"CAUTION: Drawdown {dd:.1%} breached {self.caution_thresh:.1%}")
        elif dd <= self.warning_thresh:
            new_state = DrawdownState.WARNING
            messages.append(f"WARNING: Drawdown {dd:.1%} breached {self.warning_thresh:.1%}")
        elif prev_state in (DrawdownState.CAUTION, DrawdownState.CRITICAL):
            # Recovering from crisis — enter recovery mode
            new_state = DrawdownState.RECOVERY
            self._recovery_start_day = self._total_days
            messages.append("Entering RECOVERY mode — gradual re-entry")
        elif prev_state == DrawdownState.RECOVERY:
            # Check if recovery period is over
            days_recovering = self._total_days - self._recovery_start_day
            if days_recovering >= self.recovery_days:
                new_state = DrawdownState.NORMAL
                messages.append("Recovery complete — returning to NORMAL")
            else:
                new_state = DrawdownState.RECOVERY
                messages.append(f"Recovery day {days_recovering}/{self.recovery_days}")
        else:
            new_state = DrawdownState.NORMAL

        # Track state transitions
        if new_state != prev_state:
            self.days_in_state = 0
        else:
            self.days_in_state += 1

        self.state = new_state
        self.state_history.append(new_state)

        # Compute action directives
        size_mult, allow_new, force_liq = self._compute_actions(new_state, dd)

        return DrawdownStatus(
            state=new_state,
            current_drawdown=float(dd),
            peak_equity=float(self.peak_equity),
            current_equity=float(self.current_equity),
            size_multiplier=float(size_mult),
            allow_new_entries=allow_new,
            force_liquidate=force_liq,
            days_in_state=self.days_in_state,
            daily_pnl=float(daily),
            weekly_pnl=float(weekly),
            messages=messages,
        )

    def _compute_actions(
        self, state: DrawdownState, drawdown: float,
    ) -> tuple:
        """Determine size multiplier, entry permission, and liquidation flag.

        Recovery ramp uses a **quadratic** (concave) curve instead of linear,
        so the system is more cautious at the start of recovery — preventing
        immediate re-drawdown — and gradually returns to full sizing:

            scale = 0.25 + 0.75 * progress²

        At 25% of recovery: scale = 0.297  (linear would give 0.4375)
        At 50% of recovery: scale = 0.4375 (linear would give 0.625)
        At 75% of recovery: scale = 0.672  (linear would give 0.8125)
        At 100% recovery:   scale = 1.0

        New entries are only allowed after 30% of the recovery period has
        elapsed, adding further caution.
        """
        if state == DrawdownState.NORMAL:
            return 1.0, True, False
        elif state == DrawdownState.WARNING:
            return DRAWDOWN_SIZE_MULT_WARNING, True, False
        elif state == DrawdownState.CAUTION:
            return DRAWDOWN_SIZE_MULT_CAUTION, False, False
        elif state == DrawdownState.CRITICAL:
            return 0.0, False, True
        elif state == DrawdownState.RECOVERY:
            # Gradual re-entry: quadratic (concave) ramp from 0.25 to 1.0
            days_recovering = self._total_days - self._recovery_start_day
            progress = min(1.0, days_recovering / max(1, self.recovery_days))

            # Quadratic: cautious early recovery, faster later
            size_mult = 0.25 + 0.75 * (progress ** 2)

            # Only allow new entries after 30% of recovery period
            allow_new = progress >= 0.3
            return size_mult, allow_new, False
        return 1.0, True, False

    def reset(self, equity: float = 1.0):
        """Reset controller state."""
        self.peak_equity = equity
        self.current_equity = equity
        self.state = DrawdownState.NORMAL
        self.days_in_state = 0
        self.daily_pnl_history = []
        self.state_history = []
        self._recovery_start_day = 0
        self._total_days = 0

    def get_summary(self) -> Dict:
        """Get drawdown controller summary statistics."""
        if not self.daily_pnl_history:
            return {"state": self.state.value, "total_days": 0}

        equity_curve = np.cumprod([1 + r for r in self.daily_pnl_history])
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max

        # Count state occurrences
        state_counts = {}
        for s in self.state_history:
            state_counts[s.value] = state_counts.get(s.value, 0) + 1

        return {
            "state": self.state.value,
            "total_days": self._total_days,
            "max_drawdown": float(drawdowns.min()),
            "current_drawdown": float(drawdowns[-1]) if len(drawdowns) > 0 else 0,
            "days_in_normal": state_counts.get("normal", 0),
            "days_in_warning": state_counts.get("warning", 0),
            "days_in_caution": state_counts.get("caution", 0),
            "days_in_critical": state_counts.get("critical", 0),
            "days_in_recovery": state_counts.get("recovery", 0),
            "n_circuit_breakers": sum(
                1 for i in range(1, len(self.state_history))
                if self.state_history[i] == DrawdownState.CRITICAL
                and self.state_history[i-1] != DrawdownState.CRITICAL
            ),
        }
