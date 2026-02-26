"""
Position Sizing — Kelly criterion, volatility-scaled, and ATR-based methods.

Implements multiple sizing approaches that can be blended:
    - Kelly criterion (half-Kelly for conservatism)
    - Volatility targeting (inverse vol scaling)
    - ATR-based (risk per unit of volatility)
    - Composite blend of all methods (regime-conditional weights)
    - Regime-conditional Kelly (adjusts parameters by market regime)
    - Drawdown governor (convex exponential curve reduces sizing as drawdown increases)
    - Bayesian updating (Beta-Binomial conjugate prior for win rate)
    - Uncertainty-aware scaling (signal, regime, drift confidence)
    - Budget constraints (shock budget, turnover budget, concentration limit)

Spec 05: Risk Governor + Kelly Unification + Uncertainty-Aware Sizing
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Result of position sizing calculation."""
    ticker: str  # PERMNO in PERMNO-first mode (legacy field name kept for compatibility)
    raw_kelly: float        # Full Kelly fraction
    half_kelly: float       # Half-Kelly (recommended)
    vol_scaled: float       # Volatility-targeted size
    atr_based: float        # ATR-based size
    composite: float        # Blended final size
    max_allowed: float      # Position limit
    final_size: float       # min(composite, max_allowed)
    sizing_details: dict


class PositionSizer:
    """
    Multi-method position sizer with conservative blending.

    Kelly Criterion:
        f* = (p * b - q) / b
        where p = win probability, b = win/loss ratio, q = 1 - p
        Use half-Kelly for practical conservatism.

    Volatility Targeting:
        size = target_portfolio_vol / (realized_vol * sqrt(252/holding_days))
        Ensures each position contributes roughly equally to portfolio risk.

    ATR-Based:
        size = max_risk_per_trade / (N_atr * ATR / price)
        Scales position size to risk a fixed fraction of capital.

    Budget constraints (Spec 05):
        - Shock budget: reserve SHOCK_BUDGET_PCT of capital for tail events
        - Turnover budget: limit annualized portfolio turnover
        - Concentration limit: cap single-position notional as % of portfolio
    """

    def __init__(
        self,
        target_portfolio_vol: float = 0.15,     # 15% annual portfolio vol target
        max_position_pct: float = 0.10,          # 10% max per position
        min_position_pct: float = 0.01,          # 1% min per position
        max_risk_per_trade: float = 0.02,        # 2% max risk per trade
        kelly_fraction: float = 0.5,             # Half-Kelly
        atr_multiplier: float = 2.0,             # ATR stop distance
        blend_weights: Optional[dict] = None,    # Method blend weights
        max_portfolio_dd: float = 0.20,          # Max allowed portfolio drawdown
        bayesian_alpha: float = 2.0,             # Beta prior alpha (wins + 1)
        bayesian_beta: float = 2.0,              # Beta prior beta (losses + 1)
    ):
        """Initialize PositionSizer."""
        self.target_vol = target_portfolio_vol
        self.max_position = max_position_pct
        self.min_position = min_position_pct
        self.max_risk = max_risk_per_trade
        self.kelly_frac = kelly_fraction
        self.atr_mult = atr_multiplier
        self.blend_weights = blend_weights or {
            "kelly": 0.3,
            "vol_scaled": 0.4,
            "atr_based": 0.3,
        }
        self.max_portfolio_dd = max_portfolio_dd

        # Bayesian win-rate prior (Beta-Binomial conjugate)
        self.bayesian_alpha = bayesian_alpha
        self.bayesian_beta = bayesian_beta
        # Global Bayesian counters (fallback for cold-start)
        self._bayesian_wins = 0
        self._bayesian_losses = 0
        # Per-regime Bayesian counters: {regime_id (int): {wins, losses}}
        self._bayesian_regime: Dict[int, Dict[str, int]] = {
            0: {"wins": 0, "losses": 0},  # trending_bull
            1: {"wins": 0, "losses": 0},  # trending_bear
            2: {"wins": 0, "losses": 0},  # mean_reverting
            3: {"wins": 0, "losses": 0},  # high_volatility
        }

        # Regime-conditional historical statistics
        # Keys: regime label (str), Values: {"win_rate", "avg_win", "avg_loss", "n_trades"}
        self.regime_stats: Dict[str, Dict[str, float]] = {
            "trending_bull": {"win_rate": 0.55, "avg_win": 0.03, "avg_loss": -0.02, "n_trades": 0},
            "trending_bear": {"win_rate": 0.45, "avg_win": 0.02, "avg_loss": -0.025, "n_trades": 0},
            "mean_reverting": {"win_rate": 0.50, "avg_win": 0.02, "avg_loss": -0.02, "n_trades": 0},
            "high_volatility": {"win_rate": 0.42, "avg_win": 0.04, "avg_loss": -0.035, "n_trades": 0},
        }

        # Turnover tracking for budget enforcement (cumulative per-cycle)
        self._turnover_history: List[float] = []

        # Load risk governor configuration with safe defaults
        self._load_risk_governor_config()

    def _load_risk_governor_config(self) -> None:
        """Load risk governor parameters from config with safe defaults."""
        try:
            from ..config import (
                SHOCK_BUDGET_PCT,
                CONCENTRATION_LIMIT_PCT,
                TURNOVER_BUDGET_ENFORCEMENT,
                TURNOVER_BUDGET_LOOKBACK_DAYS,
                MAX_ANNUALIZED_TURNOVER,
                BLEND_WEIGHTS_STATIC,
                BLEND_WEIGHTS_BY_REGIME,
                UNCERTAINTY_SCALING_ENABLED,
                UNCERTAINTY_SIGNAL_WEIGHT,
                UNCERTAINTY_REGIME_WEIGHT,
                UNCERTAINTY_DRIFT_WEIGHT,
                UNCERTAINTY_REDUCTION_FACTOR,
                KELLY_BAYESIAN_ALPHA,
                KELLY_BAYESIAN_BETA,
                KELLY_MIN_SAMPLES_FOR_UPDATE,
            )
            self._shock_budget_pct = SHOCK_BUDGET_PCT
            self._concentration_limit_pct = CONCENTRATION_LIMIT_PCT
            self._turnover_budget_enforcement = TURNOVER_BUDGET_ENFORCEMENT
            self._turnover_budget_lookback_days = TURNOVER_BUDGET_LOOKBACK_DAYS
            self._max_annualized_turnover = MAX_ANNUALIZED_TURNOVER
            self._blend_weights_static = BLEND_WEIGHTS_STATIC
            self._blend_weights_by_regime = BLEND_WEIGHTS_BY_REGIME
            self._uncertainty_scaling_enabled = UNCERTAINTY_SCALING_ENABLED
            self._uncertainty_signal_weight = UNCERTAINTY_SIGNAL_WEIGHT
            self._uncertainty_regime_weight = UNCERTAINTY_REGIME_WEIGHT
            self._uncertainty_drift_weight = UNCERTAINTY_DRIFT_WEIGHT
            self._uncertainty_reduction_factor = UNCERTAINTY_REDUCTION_FACTOR
            self._kelly_min_samples = KELLY_MIN_SAMPLES_FOR_UPDATE
            # Update Bayesian priors from config
            self.bayesian_alpha = KELLY_BAYESIAN_ALPHA
            self.bayesian_beta = KELLY_BAYESIAN_BETA
        except ImportError:
            # Safe defaults if config params don't exist yet
            self._shock_budget_pct = 0.05
            self._concentration_limit_pct = 0.20
            self._turnover_budget_enforcement = True
            self._turnover_budget_lookback_days = 252
            self._max_annualized_turnover = 500.0
            self._blend_weights_static = {"kelly": 0.30, "vol_scaled": 0.40, "atr_based": 0.30}
            self._blend_weights_by_regime = None
            self._uncertainty_scaling_enabled = True
            self._uncertainty_signal_weight = 0.40
            self._uncertainty_regime_weight = 0.30
            self._uncertainty_drift_weight = 0.30
            self._uncertainty_reduction_factor = 0.30
            self._kelly_min_samples = 10

    # ── Core sizing methods ─────────────────────────────────────────────

    def size_position(
        self,
        ticker: str,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        realized_vol: float,
        atr: float,
        price: float,
        holding_days: int = 10,
        confidence: float = 0.5,
        n_current_positions: int = 0,
        max_positions: int = 20,
        regime: Optional[str] = None,
        current_drawdown: float = 0.0,
        n_trades: int = 100,
        signal_uncertainty: Optional[float] = None,
        regime_entropy: Optional[float] = None,
        drift_score: Optional[float] = None,
        portfolio_equity: Optional[float] = None,
        current_positions: Optional[Dict[str, float]] = None,
    ) -> PositionSize:
        """
        Calculate position size using multiple methods and blend.

        Args:
            ticker: security id (PERMNO in PERMNO-first mode)
            win_rate: historical win rate (0-1)
            avg_win: average winning trade return
            avg_loss: average losing trade return (negative)
            realized_vol: annualized realized volatility
            atr: current ATR value
            price: current price
            holding_days: expected holding period
            confidence: model confidence (0-1)
            n_current_positions: number of existing positions
            max_positions: maximum total positions
            regime: current market regime label (e.g. "trending_bull")
            current_drawdown: current portfolio drawdown as negative fraction (e.g. -0.05)
            n_trades: number of historical trades (used for small-sample penalty in Kelly)
            signal_uncertainty: 0-1, uncertainty of primary signal (0=certain, 1=uncertain)
            regime_entropy: 0-1, normalized entropy of regime state (0=certain, 1=uncertain)
            drift_score: 0-1, trend strength (0=no trend, 1=strong trend = high confidence)
            portfolio_equity: total portfolio equity for budget constraint calculations
            current_positions: {symbol: notional} dict for turnover budget calculations
        """
        # Apply regime-conditional adjustments if regime is known
        if regime is not None and regime in self.regime_stats:
            rs = self.regime_stats[regime]
            if rs["n_trades"] >= 20:
                # Use regime-specific stats when we have enough data
                win_rate = rs["win_rate"]
                avg_win = rs["avg_win"]
                avg_loss = rs["avg_loss"]

        # ── Kelly Criterion ──
        self._current_n_trades = n_trades
        raw_kelly = self._kelly(win_rate, avg_win, avg_loss)
        half_kelly = raw_kelly * self.kelly_frac

        # Apply drawdown governor to Kelly
        half_kelly = self._apply_drawdown_governor(half_kelly, current_drawdown, self.max_portfolio_dd)

        # ── Volatility-scaled ──
        vol_scaled = self._vol_scaled(realized_vol, holding_days, max_positions)

        # ── ATR-based ──
        atr_based = self._atr_based(atr, price)

        # ── Composite blend (regime-conditional weights) ──
        # Map drawdown-state regime names to uppercase for blend weight lookup
        blend_regime = self._map_regime_for_blend(regime)
        composite = self._blend_sizes(half_kelly, vol_scaled, atr_based, blend_regime)

        # Confidence scales position DOWN from blend, never UP.
        # Range: [0.5, 1.0] — full confidence = 100% of blend, low confidence = 50%.
        confidence_scalar = 0.5 + 0.5 * min(1.0, max(0.0, confidence))
        composite *= confidence_scalar

        # ── Uncertainty-aware scaling (Spec 05 T2) ──
        uncertainty_scale = self._compute_uncertainty_scale(
            signal_uncertainty, regime_entropy, drift_score,
        )
        composite *= uncertainty_scale

        # Apply position limits
        max_allowed = self.max_position
        # Reduce size if near max positions
        if n_current_positions > 0 and max_positions > 0:
            capacity_ratio = 1 - (n_current_positions / max_positions)
            max_allowed = min(max_allowed, max(self.min_position, capacity_ratio * self.max_position))

        final_size = float(np.clip(composite, self.min_position, max_allowed))

        # ── Budget constraints (Spec 05 T3/T4/T5) ──
        if portfolio_equity is not None and portfolio_equity > 0:
            final_size = self._apply_shock_budget(final_size, portfolio_equity)
            final_size = self._apply_concentration_limit(final_size, portfolio_equity, ticker)
            if current_positions is not None:
                final_size = self._apply_turnover_budget(
                    final_size, ticker, portfolio_equity,
                    current_positions,
                    self._turnover_budget_lookback_days,
                )

        return PositionSize(
            ticker=ticker,
            raw_kelly=float(raw_kelly),
            half_kelly=float(half_kelly),
            vol_scaled=float(vol_scaled),
            atr_based=float(atr_based),
            composite=float(composite),
            max_allowed=float(max_allowed),
            final_size=float(final_size),
            sizing_details={
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "realized_vol": realized_vol,
                "atr": atr,
                "confidence": confidence,
                "confidence_scalar": confidence_scalar,
                "uncertainty_scale": uncertainty_scale,
                "blend_regime": blend_regime,
            },
        )

    def size_position_paper_trader(
        self,
        ticker: str,
        kelly_history: List[float],
        atr: float,
        realized_vol: float,
        price: float,
        regime_state: str = "NORMAL",
        dd_ratio: float = 0.0,
        confidence: float = 0.5,
        holding_days: int = 10,
        n_current_positions: int = 0,
        max_positions: int = 20,
        position_notional_before: float = 0.0,
        portfolio_equity: float = 1_000_000.0,
        current_positions: Optional[Dict[str, float]] = None,
        signal_uncertainty: Optional[float] = None,
        regime_entropy: Optional[float] = None,
        drift_score: Optional[float] = None,
    ) -> float:
        """Compute position size as % of equity for paper trader usage.

        Unified interface that wraps size_position() with paper-trader-specific
        input preparation (computing win_rate from kelly_history, mapping regime
        states, etc.).

        Args:
            ticker: security identifier.
            kelly_history: recent P&L returns for win rate estimation.
            atr: current Average True Range.
            realized_vol: annualized realized volatility.
            price: current asset price.
            regime_state: drawdown controller state (NORMAL, WARNING, etc.).
            dd_ratio: current drawdown as negative fraction (e.g. -0.05).
            confidence: model confidence (0-1).
            holding_days: expected holding period.
            n_current_positions: number of existing positions.
            max_positions: max total positions.
            position_notional_before: current position notional for turnover calc.
            portfolio_equity: total portfolio equity.
            current_positions: {symbol: notional} for turnover budget.
            signal_uncertainty: 0-1, uncertainty of primary signal.
            regime_entropy: 0-1, normalized entropy of regime state.
            drift_score: 0-1, trend strength.

        Returns:
            Position size as fraction of equity (0-1).
        """
        # Compute win rate and trade stats from history
        arr = np.array(kelly_history, dtype=float)
        valid = arr[np.isfinite(arr)]

        if len(valid) >= 20:
            win_mask = valid > 0
            lose_mask = valid < 0
            win_rate = float(win_mask.mean())
            avg_win = float(valid[win_mask].mean()) if win_mask.any() else 0.02
            avg_loss = float(valid[lose_mask].mean()) if lose_mask.any() else -0.02
            if avg_loss >= 0:
                avg_loss = -0.02
            if avg_win <= 0:
                avg_win = 0.02
            n_trades = len(valid)
        else:
            win_rate = 0.50
            avg_win = 0.02
            avg_loss = -0.02
            n_trades = len(valid)

        # Build current_positions dict including this position's prior notional
        positions = dict(current_positions or {})
        if position_notional_before > 0:
            positions[ticker] = position_notional_before

        result = self.size_position(
            ticker=ticker,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            realized_vol=realized_vol,
            atr=atr,
            price=price,
            holding_days=holding_days,
            confidence=confidence,
            n_current_positions=n_current_positions,
            max_positions=max_positions,
            regime=None,  # Paper trader uses drawdown state, not regime label
            current_drawdown=dd_ratio,
            n_trades=n_trades,
            signal_uncertainty=signal_uncertainty,
            regime_entropy=regime_entropy,
            drift_score=drift_score,
            portfolio_equity=portfolio_equity,
            current_positions=positions,
        )

        logger.debug(
            "Paper trader sizing for %s: win_rate=%.3f, n_trades=%d, "
            "kelly=%.4f, vol=%.4f, atr=%.4f, final=%.4f, regime=%s",
            ticker, win_rate, n_trades,
            result.half_kelly, result.vol_scaled, result.atr_based,
            result.final_size, regime_state,
        )

        return result.final_size

    # ── Blend weights (regime-conditional) ──────────────────────────────

    def _map_regime_for_blend(self, regime: Optional[str]) -> str:
        """Map a regime label to the blend weight key.

        The blend weights use drawdown-state names (NORMAL, WARNING, etc.).
        Regime labels like 'trending_bull' map to NORMAL; 'high_volatility'
        maps to WARNING as a conservative default.
        """
        if regime is None:
            return "NORMAL"

        # Already in blend-weight key format
        upper = regime.upper()
        if upper in ("NORMAL", "WARNING", "CAUTION", "CRITICAL", "RECOVERY"):
            return upper

        # Map regime labels to blend weight keys
        regime_to_blend = {
            "trending_bull": "NORMAL",
            "trending_bear": "WARNING",
            "mean_reverting": "NORMAL",
            "high_volatility": "CAUTION",
        }
        return regime_to_blend.get(regime, "NORMAL")

    def _blend_sizes(
        self,
        kelly_size: float,
        vol_size: float,
        atr_size: float,
        regime_state: str = "NORMAL",
    ) -> float:
        """Blend three sizing methods using regime-conditional weights.

        Regime-conditional weights (BLEND_WEIGHTS_BY_REGIME) take precedence
        over static weights (BLEND_WEIGHTS_STATIC). Falls back to the
        constructor-provided blend_weights if neither config is available.
        """
        # Get regime-specific weights (try regime-conditional first)
        if self._blend_weights_by_regime:
            weights = self._blend_weights_by_regime.get(
                regime_state,
                self._blend_weights_by_regime.get("NORMAL", self._blend_weights_static),
            )
        elif self._blend_weights_static:
            weights = self._blend_weights_static
        else:
            weights = self.blend_weights

        # Validate weights sum to 1.0
        weight_sum = weights.get("kelly", 0.3) + weights.get("vol_scaled", 0.4) + weights.get("atr_based", 0.3)
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(
                "Blend weights don't sum to 1.0 for regime %s (sum=%.3f). Normalizing.",
                regime_state, weight_sum,
            )
            weights = {k: v / weight_sum for k, v in weights.items()}

        blended = (
            weights.get("kelly", 0.3) * kelly_size
            + weights.get("vol_scaled", 0.4) * vol_size
            + weights.get("atr_based", 0.3) * atr_size
        )

        logger.debug(
            "Blended size for %s: kelly=%.4f (w=%.2f), vol=%.4f (w=%.2f), "
            "atr=%.4f (w=%.2f) = %.4f",
            regime_state, kelly_size, weights.get("kelly", 0.3),
            vol_size, weights.get("vol_scaled", 0.4),
            atr_size, weights.get("atr_based", 0.3), blended,
        )

        return blended

    # ── Uncertainty-aware sizing (Spec 05 T2) ───────────────────────────

    def _compute_uncertainty_scale(
        self,
        signal_uncertainty: Optional[float],
        regime_entropy: Optional[float],
        drift_score: Optional[float],
    ) -> float:
        """Compute scaling factor [0, 1] based on uncertainty components.

        Returns 1.0 when uncertainty scaling is disabled or any input is None.
        Returns lower values when composite uncertainty is high.

        Args:
            signal_uncertainty: 0-1, higher = more uncertain signal.
            regime_entropy: 0-1, higher = more uncertain regime state.
            drift_score: 0-1, higher = stronger trend (higher confidence).

        Returns:
            Scale factor in [1 - max_reduction, 1.0].
        """
        if not self._uncertainty_scaling_enabled:
            return 1.0

        if signal_uncertainty is None or regime_entropy is None or drift_score is None:
            return 1.0

        # Validate inputs — treat out-of-range or NaN as missing
        def _validate(val: float) -> Optional[float]:
            if not np.isfinite(val) or val < 0.0 or val > 1.0:
                logger.warning(
                    "Uncertainty input out of range [0,1]: %.4f. Treating as missing.",
                    val,
                )
                return None
            return val

        sig_u = _validate(signal_uncertainty)
        reg_e = _validate(regime_entropy)
        drft = _validate(drift_score)

        if sig_u is None or reg_e is None or drft is None:
            return 1.0

        # Convert to confidence metrics (high = high confidence)
        signal_confidence = 1.0 - sig_u
        regime_confidence = 1.0 - reg_e
        drift_confidence = drft  # High drift = high confidence in direction

        # Weighted composite confidence
        composite_confidence = (
            self._uncertainty_signal_weight * signal_confidence
            + self._uncertainty_regime_weight * regime_confidence
            + self._uncertainty_drift_weight * drift_confidence
        )

        # Map confidence [0, 1] to size scale [1 - max_reduction, 1.0]
        max_reduction = self._uncertainty_reduction_factor
        scale = 1.0 - (max_reduction * (1.0 - composite_confidence))

        result = float(np.clip(scale, 1.0 - max_reduction, 1.0))

        logger.debug(
            "Uncertainty scale: signal_conf=%.3f, regime_conf=%.3f, "
            "drift_conf=%.3f, composite=%.3f, scale=%.3f",
            signal_confidence, regime_confidence, drift_confidence,
            composite_confidence, result,
        )

        return result

    # ── Budget constraints (Spec 05 T3/T4/T5) ──────────────────────────

    def _apply_shock_budget(
        self, position_size: float, portfolio_equity: float,
    ) -> float:
        """Ensure position doesn't exceed (1 - shock_budget_pct) of portfolio.

        The shock budget reserves a fraction of capital for tail events.
        No single position's notional can exceed the non-reserved portion.
        """
        max_deployable = 1.0 - self._shock_budget_pct
        if position_size > max_deployable:
            logger.warning(
                "Position size %.4f exceeds shock budget (max %.4f). Reduced.",
                position_size, max_deployable,
            )
            return max_deployable
        return position_size

    def _apply_concentration_limit(
        self, position_size: float, portfolio_equity: float,
        symbol: Optional[str] = None,
    ) -> float:
        """Ensure position doesn't exceed concentration limit as % of portfolio."""
        if position_size > self._concentration_limit_pct:
            logger.warning(
                "Position %s size %.4f exceeds concentration limit %.2f%%. Capped.",
                symbol or "?", position_size, self._concentration_limit_pct * 100,
            )
            return self._concentration_limit_pct
        return position_size

    def _apply_turnover_budget(
        self,
        proposed_size: float,
        symbol: str,
        portfolio_equity: float,
        current_positions: Dict[str, float],
        dates_in_period: int,
    ) -> float:
        """Scale down position size if forward turnover would exceed budget.

        Tracks cumulative turnover and enforces an annualized cap.
        """
        if not self._turnover_budget_enforcement:
            return proposed_size

        # Compute cumulative turnover from history
        total_turnover = sum(self._turnover_history) if self._turnover_history else 0.0
        days_elapsed = max(1, len(self._turnover_history))
        annualized_turnover = (total_turnover / days_elapsed) * 252

        # Compute forward turnover from this position change
        current_notional = current_positions.get(symbol, 0.0)
        proposed_notional = proposed_size * portfolio_equity
        if portfolio_equity > 0:
            position_turnover = abs(proposed_notional - current_notional) / portfolio_equity
        else:
            position_turnover = 0.0

        remaining_budget = self._max_annualized_turnover - annualized_turnover

        if position_turnover > remaining_budget and remaining_budget > 0:
            scale_factor = remaining_budget / position_turnover
            reduced_size = proposed_size * scale_factor
            logger.warning(
                "Turnover budget: %s scaled from %.4f to %.4f "
                "(annualized=%.1f, remaining=%.1f).",
                symbol, proposed_size, reduced_size,
                annualized_turnover, remaining_budget,
            )
            return reduced_size
        elif remaining_budget <= 0 and position_turnover > 0:
            logger.warning(
                "Turnover budget exhausted (annualized=%.1f). Not increasing %s.",
                annualized_turnover, symbol,
            )
            # Return current position size (no change) to avoid turnover
            if portfolio_equity > 0:
                return current_notional / portfolio_equity
            return 0.0

        # Record this turnover event
        self._turnover_history.append(position_turnover)

        return proposed_size

    def record_turnover(self, turnover_amount: float) -> None:
        """Record a turnover event for budget tracking.

        Called externally (e.g., by paper trader) after position changes.
        """
        self._turnover_history.append(turnover_amount)

    def reset_turnover_tracking(self) -> None:
        """Reset turnover tracking history."""
        self._turnover_history.clear()

    # ── Kelly methods ───────────────────────────────────────────────────

    def _kelly(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Kelly criterion: f* = (p*b - q) / b

        where p = win probability, b = win/loss ratio, q = 1-p

        Returns 0.0 for negative-edge signals (not min_position) — refuse to
        allocate capital when the expected edge is non-positive.  A small-sample
        penalty shrinks the Kelly fraction toward zero when ``_current_n_trades``
        is low, reaching full Kelly only after 50 trades.
        """
        if avg_loss >= -1e-9 or win_rate <= 0.0 or win_rate >= 1.0:
            return 0.0  # Invalid inputs — refuse to size

        p = win_rate
        q = 1.0 - p
        b = abs(avg_win / avg_loss)  # win/loss ratio

        kelly = (p * b - q) / b

        if kelly <= 0:
            return 0.0  # Negative edge — do not trade

        # Small-sample penalty: reduce Kelly when n_trades is low.
        # Shrink toward 0 as n_trades → 0; full Kelly only after 50 trades.
        n_trades = getattr(self, '_current_n_trades', 100)
        sample_penalty = min(1.0, n_trades / 50.0)
        kelly *= sample_penalty

        return min(kelly, self.max_position)

    def _vol_scaled(
        self, realized_vol: float, holding_days: int, max_positions: int,
    ) -> float:
        """
        Volatility-targeted sizing.

        size = target_vol / (asset_vol * sqrt(holding_period_fraction) * sqrt(n_positions))
        """
        if realized_vol <= 0:
            return self.min_position

        # Holding period volatility
        holding_vol = realized_vol * np.sqrt(holding_days / 252)

        # Diversification factor: spread risk across positions
        n_pos_factor = np.sqrt(max(1, max_positions))

        size = self.target_vol / (holding_vol * n_pos_factor)

        return np.clip(size, self.min_position, self.max_position)

    def _atr_based(self, atr: float, price: float) -> float:
        """
        ATR-based sizing: risk a fixed fraction of capital.

        size = max_risk / (atr_multiplier * atr / price)
        """
        if atr <= 0 or price <= 0:
            return self.min_position

        # Risk per share as % of price
        risk_per_share_pct = (self.atr_mult * atr) / price

        if risk_per_share_pct <= 0:
            return self.min_position

        size = self.max_risk / risk_per_share_pct

        return np.clip(size, self.min_position, self.max_position)

    def _apply_drawdown_governor(
        self, kelly_fraction: float, current_drawdown: float, max_allowed_dd: float = -0.20,
    ) -> float:
        """Reduce Kelly as drawdown approaches max using a convex (exponential) curve.

        Uses ``scale = exp(-k * dd_ratio^2)`` instead of linear ``1 - dd_ratio``.
        The squared term makes the curve more lenient early in the drawdown and
        more aggressive near the limit:

            At 50% of max_dd -> sizing ~ 61% (vs 50% linear)
            At 90% of max_dd -> sizing ~ 20% (vs 10% linear)

        Returns 0.0 if drawdown exceeds the maximum allowed.
        """
        if current_drawdown >= 0.0:
            return kelly_fraction  # No drawdown — full sizing

        dd_ratio = abs(current_drawdown) / abs(max_allowed_dd)

        if dd_ratio >= 1.0:
            return 0.0  # Breached max drawdown — no new positions

        # Convex scaling: slow reduction early, aggressive near limit.
        # exp(-k * dd_ratio^2) with k=2 gives:
        #   dd_ratio=0.5 -> scale~0.61 (lenient early)
        #   dd_ratio=0.9 -> scale~0.20 (aggressive late)
        k = 2.0
        scale = math.exp(-k * dd_ratio * dd_ratio)

        return kelly_fraction * scale

    # ── Bayesian Kelly updating ─────────────────────────────────────────

    def update_regime_stats(self, trades: pd.DataFrame, regime_col: str = "regime"):
        """Update regime-specific historical stats from completed trades.

        Accepts a full trade DataFrame with a ``regime`` column (integer IDs
        0-3) and updates stats for every regime that has enough data. This
        replaces the old interface that required pre-filtering by regime.

        Args:
            trades: DataFrame with ``net_return`` and ``regime`` columns.
            regime_col: column name containing integer regime IDs.
        """
        from ..config import REGIME_NAMES

        if trades.empty or "net_return" not in trades.columns:
            return
        if regime_col not in trades.columns:
            return

        for regime_id, regime_name in REGIME_NAMES.items():
            regime_trades = trades[trades[regime_col] == regime_id]
            returns = regime_trades["net_return"].dropna().values
            if len(returns) < 5:
                continue
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            self.regime_stats[regime_name] = {
                "win_rate": float((returns > 0).mean()),
                "avg_win": float(wins.mean()) if len(wins) > 0 else 0.01,
                "avg_loss": float(losses.mean()) if len(losses) > 0 else -0.01,
                "n_trades": len(returns),
            }

    def update_kelly_bayesian(self, trades: pd.DataFrame, regime_col: str = "regime"):
        """Update per-regime and global Bayesian posteriors from trade history.

        Maintains separate Beta-Binomial priors for each regime so that
        bull-market wins don't inflate bear-market posteriors.

        Args:
            trades: DataFrame with ``net_return`` column and optionally a
                ``regime`` column (integer regime IDs 0-3).
            regime_col: column name containing regime labels.
        """
        if trades.empty or "net_return" not in trades.columns:
            return

        returns = trades["net_return"].dropna()

        # Update global counters
        self._bayesian_wins = int((returns > 0).sum())
        self._bayesian_losses = int((returns <= 0).sum())

        # Update per-regime counters
        if regime_col in trades.columns:
            for regime_id in self._bayesian_regime:
                regime_mask = trades[regime_col] == regime_id
                regime_returns = trades.loc[regime_mask, "net_return"].dropna()
                if len(regime_returns) > 0:
                    wins = int((regime_returns > 0).sum())
                    losses = int((regime_returns <= 0).sum())
                    self._bayesian_regime[regime_id]["wins"] += wins
                    self._bayesian_regime[regime_id]["losses"] += losses

    def get_bayesian_kelly(
        self, avg_win: float = 0.02, avg_loss: float = -0.02, regime: Optional[int] = None,
    ) -> float:
        """Compute Kelly fraction using regime-specific Bayesian posterior.

        When a ``regime`` is specified and sufficient regime-specific data is
        available (>= min_samples trades), the posterior for that regime is used.
        Otherwise falls back to the global posterior.

        Uses configurable prior parameters (KELLY_BAYESIAN_ALPHA, KELLY_BAYESIAN_BETA)
        and minimum sample threshold (KELLY_MIN_SAMPLES_FOR_UPDATE).

        Args:
            avg_win: average winning trade return.
            avg_loss: average losing trade return (negative).
            regime: integer regime ID (0-3) or ``None`` for global.

        Returns:
            Half-Kelly fraction from Bayesian posterior win rate.
        """
        min_samples = self._kelly_min_samples

        # Try regime-specific posterior first
        if regime is not None and regime in self._bayesian_regime:
            stats = self._bayesian_regime[regime]
            wins, losses = stats["wins"], stats["losses"]
            if wins + losses >= min_samples:
                posterior_wr = (self.bayesian_alpha + wins) / (
                    self.bayesian_alpha + self.bayesian_beta + wins + losses
                )
                return self._kelly(posterior_wr, avg_win, avg_loss) * self.kelly_frac

        # Fallback to global posterior
        total_trades = self._bayesian_wins + self._bayesian_losses
        if total_trades < min_samples:
            # Insufficient data — use prior mode as estimate
            alpha = self.bayesian_alpha
            beta = self.bayesian_beta
            if (alpha + beta) > 2:
                prior_mode = (alpha - 1) / (alpha + beta - 2)
            else:
                prior_mode = 0.5
            logger.debug(
                "Insufficient trades (%d < %d). Using prior estimate %.3f.",
                total_trades, min_samples, prior_mode,
            )
            return self._kelly(prior_mode, avg_win, avg_loss) * self.kelly_frac

        alpha = self.bayesian_alpha + self._bayesian_wins
        beta = self.bayesian_beta + self._bayesian_losses
        posterior_win_rate = alpha / (alpha + beta)
        raw_kelly = self._kelly(posterior_win_rate, avg_win, avg_loss)

        logger.debug(
            "Kelly Bayesian: %d trades, posterior Beta(%.1f, %.1f), "
            "win_rate=%.3f, kelly=%.4f",
            total_trades, alpha, beta, posterior_win_rate, raw_kelly * self.kelly_frac,
        )

        return raw_kelly * self.kelly_frac

    # ── Portfolio-aware sizing ──────────────────────────────────────────

    def size_portfolio_aware(
        self,
        ticker: str,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        realized_vol: float,
        atr: float,
        price: float,
        existing_positions: Dict[str, float],
        returns_data: Optional[Dict[str, pd.Series]] = None,
        **kwargs,
    ) -> float:
        """Size position considering correlation with existing portfolio.

        Wraps :meth:`size_position` and applies a correlation penalty:
        positions that are highly correlated with the existing portfolio
        receive reduced allocation.

        Args:
            ticker: security identifier.
            win_rate: historical win rate (0-1).
            avg_win: average winning trade return.
            avg_loss: average losing trade return (negative).
            realized_vol: annualized realized volatility.
            atr: current ATR value.
            price: current price.
            existing_positions: ``{ticker: weight}`` of current portfolio.
            returns_data: ``{ticker: pd.Series}`` of return series for
                correlation computation.
            **kwargs: forwarded to :meth:`size_position`.

        Returns:
            Adjusted position size as a fraction of portfolio.
        """
        base_result = self.size_position(
            ticker=ticker,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            realized_vol=realized_vol,
            atr=atr,
            price=price,
            **kwargs,
        )
        base_size = base_result.final_size

        if not existing_positions or returns_data is None:
            return base_size

        ticker_returns = returns_data.get(ticker)
        if ticker_returns is None or len(ticker_returns) < 20:
            return base_size

        # Compute weight-adjusted average correlation with existing positions
        correlations = []
        total_abs_weight = 0.0
        for pos_ticker, pos_weight in existing_positions.items():
            pos_returns = returns_data.get(pos_ticker)
            if pos_returns is None:
                continue
            # Align series on shared index
            aligned = pd.concat([ticker_returns, pos_returns], axis=1).dropna()
            if len(aligned) < 20:
                continue
            corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            if not np.isfinite(corr):
                continue
            correlations.append(corr * abs(pos_weight))
            total_abs_weight += abs(pos_weight)

        if not correlations or total_abs_weight < 1e-9:
            return base_size

        # Weighted average correlation with portfolio
        avg_corr = sum(correlations) / total_abs_weight

        # Reduce size when highly correlated with existing positions.
        # corr=0 -> full size, corr=1 -> 50% size, corr<0 -> bonus (capped at 100%)
        correlation_penalty = 1.0 - 0.5 * max(0.0, avg_corr)

        return base_size * correlation_penalty

    def size_portfolio(
        self,
        signals: pd.DataFrame,
        price_data: dict,
        trade_history: Optional[pd.DataFrame] = None,
        max_positions: int = 20,
    ) -> pd.DataFrame:
        """
        Size all candidate positions for the portfolio.

        Args:
            signals: DataFrame with columns [permno|ticker, predicted_return, confidence, regime]
            price_data: dict of security-id -> OHLCV DataFrames
            trade_history: historical trades for win rate estimation
            max_positions: max simultaneous positions

        Returns:
            DataFrame with position sizes added
        """
        # Compute historical win rate and avg win/loss from trade history
        if trade_history is not None and len(trade_history) > 10:
            hist_returns = trade_history["net_return"].values
            win_rate = (hist_returns > 0).mean()
            avg_win = hist_returns[hist_returns > 0].mean() if (hist_returns > 0).any() else 0.01
            avg_loss = hist_returns[hist_returns < 0].mean() if (hist_returns < 0).any() else -0.01
        else:
            # Conservative defaults
            win_rate = 0.50
            avg_win = 0.02
            avg_loss = -0.02

        sizes = []
        id_col = "permno" if "permno" in signals.columns else "ticker"
        for _, row in signals.iterrows():
            ticker = str(row[id_col])
            if ticker not in price_data:
                continue

            ohlcv = price_data[ticker]
            if len(ohlcv) < 20:
                continue

            # Current ATR (14-period)
            high = ohlcv["High"].iloc[-14:]
            low = ohlcv["Low"].iloc[-14:]
            close = ohlcv["Close"].iloc[-15:]  # need one extra for prev close
            tr = pd.concat([
                high - low,
                (high - close.shift(1).iloc[-14:]).abs(),
                (low - close.shift(1).iloc[-14:]).abs(),
            ], axis=1).max(axis=1)
            atr = float(tr.mean())

            # Realized vol (annualized, 20-day)
            returns = ohlcv["Close"].pct_change().iloc[-20:]
            realized_vol = float(returns.std() * np.sqrt(252))

            price = float(ohlcv["Close"].iloc[-1])

            ps = self.size_position(
                ticker=ticker,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                realized_vol=realized_vol,
                atr=atr,
                price=price,
                confidence=float(row.get("confidence", 0.5)),
                n_current_positions=len(sizes),
                max_positions=max_positions,
            )
            sizes.append({
                "permno": ticker,
                "position_size": ps.final_size,
                "kelly_size": ps.half_kelly,
                "vol_size": ps.vol_scaled,
                "atr_size": ps.atr_based,
            })

        return pd.DataFrame(sizes)
