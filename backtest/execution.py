"""
Execution simulator with spread, market impact, and participation limits.

Spec 06 enhancements:
  - Structural state-aware cost multipliers (break probability, structure
    uncertainty, drift score, systemic stress)
  - ADV-aware participation limit adjustment via volume trend
  - Entry/exit urgency differentiation with cost acceptance limits
  - No-trade gate during extreme stress for low-urgency orders
  - Per-market-cap-segment impact coefficient support

SPEC-E03 enhancement:
  - Unified ShockModePolicy that consolidates shock/elevated/normal
    execution parameters from a ShockVector into a single policy object.

Includes a cost-model calibration routine that fits spread, impact, and fill
parameters from historical fill data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..regime.shock_vector import ShockVector

logger = logging.getLogger(__name__)


@dataclass
class ExecutionFill:
    """Simulated execution fill outcome returned by the execution model."""
    fill_price: float
    fill_ratio: float
    participation_rate: float
    impact_bps: float
    spread_bps: float
    event_spread_multiplier_applied: float = 1.0
    # Spec 06: structural state cost breakdown
    structural_mult: float = 1.0
    urgency_type: str = ""
    no_trade_blocked: bool = False
    urgency_reduced: bool = False
    cost_details: Dict = field(default_factory=dict)


@dataclass
class ShockModePolicy:
    """Unified shock-mode execution policy (SPEC-E03).

    Consolidates the system's response to market shocks, elevated
    uncertainty, and normal conditions into a single policy object that
    downstream execution and entry logic can consume without ad-hoc checks.

    Three tiers:
      1. **Shock** — active shock event (jump, changepoint, large move).
         Severely restricts participation, widens spreads, raises the
         confidence bar for new entries.
      2. **Elevated** — high HMM uncertainty (regime transition zone).
         Moderately restricts participation and raises confidence bar.
      3. **Normal** — calm market, default execution parameters.

    Attributes
    ----------
    is_active : bool
        True when operating in shock or elevated mode.
    tier : str
        ``"shock"``, ``"elevated"``, or ``"normal"``.
    max_participation_override : float
        Max participation rate to use for this bar's execution.
    spread_multiplier : float
        Multiplicative factor applied to effective spread.
    min_confidence_override : float
        Minimum signal confidence required to enter a new position.
    """

    is_active: bool
    tier: str
    max_participation_override: float
    spread_multiplier: float
    min_confidence_override: float

    @classmethod
    def from_shock_vector(
        cls,
        shock: "ShockVector",
        *,
        shock_max_participation: float = 0.005,
        shock_spread_mult: float = 2.0,
        shock_min_confidence: float = 0.80,
        elevated_max_participation: float = 0.01,
        elevated_spread_mult: float = 1.5,
        elevated_min_confidence: float = 0.65,
        elevated_uncertainty_threshold: float = 0.7,
        normal_max_participation: float = 0.02,
        normal_spread_mult: float = 1.0,
        normal_min_confidence: float = 0.50,
    ) -> "ShockModePolicy":
        """Derive execution policy from a ShockVector.

        Parameters
        ----------
        shock : ShockVector
            Current bar's structural state.
        shock_max_participation : float
            Max participation during full shock events.
        shock_spread_mult : float
            Spread multiplier during full shock events.
        shock_min_confidence : float
            Minimum confidence to enter during full shock events.
        elevated_max_participation : float
            Max participation during elevated uncertainty.
        elevated_spread_mult : float
            Spread multiplier during elevated uncertainty.
        elevated_min_confidence : float
            Minimum confidence to enter during elevated uncertainty.
        elevated_uncertainty_threshold : float
            HMM uncertainty above which elevated mode activates.
        normal_max_participation : float
            Max participation under normal conditions.
        normal_spread_mult : float
            Spread multiplier under normal conditions.
        normal_min_confidence : float
            Minimum confidence under normal conditions.

        Returns
        -------
        ShockModePolicy
        """
        if shock.is_shock_event():
            return cls(
                is_active=True,
                tier="shock",
                max_participation_override=shock_max_participation,
                spread_multiplier=shock_spread_mult,
                min_confidence_override=shock_min_confidence,
            )
        elif shock.hmm_uncertainty > elevated_uncertainty_threshold:
            return cls(
                is_active=True,
                tier="elevated",
                max_participation_override=elevated_max_participation,
                spread_multiplier=elevated_spread_mult,
                min_confidence_override=elevated_min_confidence,
            )
        return cls(
            is_active=False,
            tier="normal",
            max_participation_override=normal_max_participation,
            spread_multiplier=normal_spread_mult,
            min_confidence_override=normal_min_confidence,
        )

    @classmethod
    def normal_default(cls) -> "ShockModePolicy":
        """Return the default normal-mode policy (no overrides)."""
        return cls(
            is_active=False,
            tier="normal",
            max_participation_override=0.02,
            spread_multiplier=1.0,
            min_confidence_override=0.50,
        )


class ExecutionModel:
    """
    Market-impact model for backtests with structural state-aware costs.

    Extends the base spread + sqrt(participation) impact model with:
      - Structural state multipliers (break probability, uncertainty, drift, systemic stress)
      - ADV-based volume trend adjustment
      - Entry/exit urgency differentiation
      - No-trade gate during extreme stress
      - Per-market-cap-segment impact coefficients
    """

    def __init__(
        self,
        spread_bps: float = 3.0,
        max_participation_rate: float = 0.02,
        impact_coefficient_bps: float = 25.0,
        min_fill_ratio: float = 0.20,
        dynamic_costs: bool = True,
        dollar_volume_ref_usd: float = 25_000_000.0,
        vol_ref: float = 0.20,
        vol_spread_beta: float = 1.0,
        gap_spread_beta: float = 4.0,
        range_spread_beta: float = 2.0,
        vol_impact_beta: float = 1.0,
        # Spec 06: structural state config
        structural_stress_enabled: bool = True,
        break_prob_cost_mult: Optional[Dict[str, float]] = None,
        structure_uncertainty_cost_mult: float = 0.50,
        drift_score_cost_reduction: float = 0.20,
        systemic_stress_cost_mult: float = 0.30,
        # Spec 06: urgency config
        exit_urgency_cost_limit_mult: float = 1.5,
        entry_urgency_cost_limit_mult: float = 1.0,
        stress_pullback_min_size: float = 0.10,
        no_trade_stress_threshold: float = 0.95,
        # Spec 06: volume trend
        volume_trend_enabled: bool = True,
    ):
        """Initialize ExecutionModel."""
        self.spread_bps = float(max(0.0, spread_bps))
        self.max_participation = float(max(1e-6, max_participation_rate))
        self.impact_coeff = float(max(0.0, impact_coefficient_bps))
        self.min_fill_ratio = float(np.clip(min_fill_ratio, 0.0, 1.0))
        self.dynamic_costs = bool(dynamic_costs)
        self.dollar_volume_ref = float(max(1e-6, dollar_volume_ref_usd))
        self.vol_ref = float(max(1e-6, vol_ref))
        self.vol_spread_beta = float(max(0.0, vol_spread_beta))
        self.gap_spread_beta = float(max(0.0, gap_spread_beta))
        self.range_spread_beta = float(max(0.0, range_spread_beta))
        self.vol_impact_beta = float(max(0.0, vol_impact_beta))

        # Spec 06: structural state configuration
        self.structural_stress_enabled = bool(structural_stress_enabled)
        self.break_prob_cost_mult = break_prob_cost_mult or {
            "low": 1.0, "medium": 1.3, "high": 2.0,
        }
        self.structure_uncertainty_cost_mult = float(
            max(0.0, structure_uncertainty_cost_mult)
        )
        self.drift_score_cost_reduction = float(
            np.clip(drift_score_cost_reduction, 0.0, 0.50)
        )
        self.systemic_stress_cost_mult = float(
            max(0.0, systemic_stress_cost_mult)
        )

        # Spec 06: urgency configuration
        self.exit_urgency_cost_limit_mult = float(
            max(1.0, exit_urgency_cost_limit_mult)
        )
        self.entry_urgency_cost_limit_mult = float(
            max(0.5, entry_urgency_cost_limit_mult)
        )
        self.stress_pullback_min_size = float(
            np.clip(stress_pullback_min_size, 0.0, 0.50)
        )
        self.no_trade_stress_threshold = float(
            np.clip(no_trade_stress_threshold, 0.0, 1.0)
        )

        # Spec 06: volume trend
        self.volume_trend_enabled = bool(volume_trend_enabled)

        # Base transaction cost for urgency limit comparison (defaults to 20 bps)
        self._base_transaction_cost_bps = 20.0

    def set_base_transaction_cost_bps(self, bps: float) -> None:
        """Set the base transaction cost used for urgency limit comparison."""
        self._base_transaction_cost_bps = float(max(1.0, bps))

    # ── Structural State Multipliers (Spec 06 T1) ─────────────────────

    def _compute_break_probability_mult(
        self, break_prob: Optional[float],
    ) -> float:
        """Scale cost multiplier based on flash crash risk.

        Interpolates linearly between tier boundaries for smooth behavior:
          - break_prob < 0.05  → 'low' multiplier (1.0)
          - 0.05..0.15         → linear interpolation low→medium
          - 0.15..0.50         → linear interpolation medium→high
          - > 0.50             → 'high' multiplier (capped)
        """
        if break_prob is None or not np.isfinite(break_prob):
            return 1.0
        break_prob = float(np.clip(break_prob, 0.0, 1.0))

        mult_low = float(self.break_prob_cost_mult.get("low", 1.0))
        mult_med = float(self.break_prob_cost_mult.get("medium", 1.3))
        mult_high = float(self.break_prob_cost_mult.get("high", 2.0))

        if break_prob < 0.05:
            return mult_low
        elif break_prob < 0.15:
            interp = (break_prob - 0.05) / 0.10
            return mult_low + (mult_med - mult_low) * interp
        elif break_prob < 0.50:
            interp = (break_prob - 0.15) / 0.35
            return mult_med + (mult_high - mult_med) * interp
        else:
            return mult_high

    def _compute_structure_uncertainty_mult(
        self, uncertainty: Optional[float],
    ) -> float:
        """Scale cost multiplier based on regime state uncertainty (entropy).

        Higher uncertainty → higher costs (less confident about market state).
        Returns 1.0 + (uncertainty * coefficient).
        """
        if uncertainty is None or not np.isfinite(uncertainty):
            return 1.0
        uncertainty = float(np.clip(uncertainty, 0.0, 1.0))
        return 1.0 + (uncertainty * self.structure_uncertainty_cost_mult)

    def _compute_drift_score_mult(
        self, drift_score: Optional[float],
    ) -> float:
        """Reduce cost multiplier based on trend strength.

        High drift = high conviction trend → lower execution urgency → lower costs.
        Returns 1.0 - (drift_score * coefficient), floored at 0.70.
        """
        if drift_score is None or not np.isfinite(drift_score):
            return 1.0
        drift_score = float(np.clip(drift_score, 0.0, 1.0))
        return max(0.70, 1.0 - (drift_score * self.drift_score_cost_reduction))

    def _compute_systemic_stress_mult(
        self, systemic_stress: Optional[float],
    ) -> float:
        """Scale cost multiplier based on systemic stress (e.g. VIX percentile).

        Higher systemic stress → wider spreads, more impact → higher costs.
        Returns 1.0 + (systemic_stress * coefficient).
        """
        if systemic_stress is None or not np.isfinite(systemic_stress):
            return 1.0
        systemic_stress = float(np.clip(systemic_stress, 0.0, 1.0))
        return 1.0 + (systemic_stress * self.systemic_stress_cost_mult)

    def _compute_structural_multiplier(
        self,
        break_probability: Optional[float] = None,
        structure_uncertainty: Optional[float] = None,
        drift_score: Optional[float] = None,
        systemic_stress: Optional[float] = None,
    ) -> tuple[float, Dict[str, float]]:
        """Compute composite structural state multiplier.

        Individual multipliers are combined multiplicatively.
        Result is clipped to [0.70, 3.0] to prevent unrealistic extremes.

        Returns (composite_mult, details_dict).
        """
        if not self.structural_stress_enabled:
            return 1.0, {}

        break_mult = self._compute_break_probability_mult(break_probability)
        uncertainty_mult = self._compute_structure_uncertainty_mult(
            structure_uncertainty
        )
        drift_mult = self._compute_drift_score_mult(drift_score)
        stress_mult = self._compute_systemic_stress_mult(systemic_stress)

        composite = break_mult * uncertainty_mult * drift_mult * stress_mult
        composite = float(np.clip(composite, 0.70, 3.0))

        details = {
            "break_prob_mult": break_mult,
            "uncertainty_mult": uncertainty_mult,
            "drift_mult": drift_mult,
            "stress_mult": stress_mult,
            "composite_structural_mult": composite,
        }
        return composite, details

    # ── Cost Estimation (SPEC-E01) ──────────────────────────────────

    def estimate_cost(
        self,
        daily_volume: float,
        desired_notional: float,
        realized_vol: float | None = None,
        structure_uncertainty: float = 0.0,
        overnight_gap: float | None = None,
        intraday_range: float | None = None,
        reference_price: float = 100.0,
        event_spread_multiplier: float = 1.0,
        break_probability: float | None = None,
        drift_score: float | None = None,
        systemic_stress: float | None = None,
        volume_trend: float | None = None,
        impact_coeff_override: float | None = None,
        # SPEC-E03: shock mode overrides
        max_participation_override: float | None = None,
    ) -> float:
        """Estimate expected round-trip cost in basis points without executing.

        Uses the same spread + market impact model as :meth:`simulate` but
        returns only the total expected round-trip cost (entry + exit).  This
        supports the edge-after-costs trade gate in the backtest engine.

        Parameters
        ----------
        daily_volume : float
            Daily share volume.
        desired_notional : float
            Target execution notional in USD.
        realized_vol : float, optional
            Annualized realized volatility for dynamic cost scaling.
        structure_uncertainty : float
            Regime state entropy / uncertainty (0-1). Defaults to 0.
        overnight_gap : float, optional
            Overnight gap return for spread scaling.
        intraday_range : float, optional
            Intraday range for spread scaling.
        reference_price : float
            Reference price for dollar volume computation. Defaults to 100.
        event_spread_multiplier : float
            Multiplier for event-window spread widening (>= 1.0).
        break_probability : float, optional
            Probability of flash crash / structural break (0-1).
        drift_score : float, optional
            Trend conviction strength; high = lower costs (0-1).
        systemic_stress : float, optional
            Normalized systemic stress (0-1).
        volume_trend : float, optional
            Current volume relative to ADV EMA.
        impact_coeff_override : float, optional
            Calibrated impact coefficient for this security's market cap segment.

        Returns
        -------
        float
            Estimated round-trip cost in basis points.  This is 2x the
            one-way cost (half-spread + impact for entry, same for exit).
        """
        px = float(max(1e-9, reference_price))
        vol = float(max(0.0, daily_volume))
        desired = float(max(0.0, desired_notional))

        if desired <= 0 or vol <= 0:
            return 0.0

        # Participation limit with optional shock mode and volume trend adjustment
        effective_max_participation = self.max_participation

        # SPEC-E03: shock mode participation override
        if (
            max_participation_override is not None
            and np.isfinite(max_participation_override)
            and max_participation_override > 0
        ):
            effective_max_participation = float(max_participation_override)

        if (
            self.volume_trend_enabled
            and volume_trend is not None
            and np.isfinite(volume_trend)
        ):
            trend = float(np.clip(volume_trend, 0.5, 2.0))
            effective_max_participation = float(np.clip(
                effective_max_participation * trend,
                effective_max_participation * 0.5,
                effective_max_participation * 2.0,
            ))

        daily_dollar_volume = max(px * vol, 1e-9)
        max_notional = daily_dollar_volume * effective_max_participation
        fill_notional = min(desired, max_notional)
        participation = float(fill_notional / daily_dollar_volume)

        # Use per-segment impact coefficient if provided
        impact_coeff = (
            float(max(0.0, impact_coeff_override))
            if impact_coeff_override is not None
            and np.isfinite(impact_coeff_override)
            else self.impact_coeff
        )

        spread_bps = self.spread_bps

        if self.dynamic_costs:
            vol_component = 0.0
            if realized_vol is not None and np.isfinite(realized_vol):
                vol_component = max(
                    0.0, (float(realized_vol) - self.vol_ref) / self.vol_ref
                )

            gap_component = 0.0
            if overnight_gap is not None and np.isfinite(overnight_gap):
                gap_component = min(0.10, abs(float(overnight_gap)))

            range_component = 0.0
            if intraday_range is not None and np.isfinite(intraday_range):
                range_component = min(0.20, abs(float(intraday_range)))

            stress = 1.0
            stress += self.vol_spread_beta * vol_component
            stress += self.gap_spread_beta * gap_component
            stress += self.range_spread_beta * range_component

            liquidity_scalar = float(np.clip(
                np.sqrt(self.dollar_volume_ref / daily_dollar_volume),
                0.70,
                3.00,
            ))

            structural_mult, _ = self._compute_structural_multiplier(
                break_probability=break_probability,
                structure_uncertainty=structure_uncertainty,
                drift_score=drift_score,
                systemic_stress=systemic_stress,
            )

            spread_bps = (
                spread_bps * stress * liquidity_scalar * structural_mult
            )
            impact_coeff = (
                impact_coeff
                * (1.0 + self.vol_impact_beta * max(0.0, stress - 1.0))
                * liquidity_scalar
                * structural_mult
            )

        # Event-window spread blowout
        evt_mult = float(max(1.0, event_spread_multiplier))
        spread_bps = spread_bps * evt_mult

        impact_bps = float(impact_coeff * np.sqrt(max(0.0, participation)))
        half_spread_bps = 0.5 * spread_bps

        # One-way cost = half_spread + impact; round-trip = 2x
        one_way_cost_bps = half_spread_bps + impact_bps
        return float(2.0 * one_way_cost_bps)

    # ── Core Simulation ───────────────────────────────────────────────

    def simulate(
        self,
        side: str,
        reference_price: float,
        daily_volume: float,
        desired_notional_usd: float,
        force_full: bool = False,
        realized_vol: float | None = None,
        overnight_gap: float | None = None,
        intraday_range: float | None = None,
        event_spread_multiplier: float = 1.0,
        # Spec 06: structural state inputs (all optional, backward compatible)
        break_probability: float | None = None,
        structure_uncertainty: float | None = None,
        drift_score: float | None = None,
        systemic_stress: float | None = None,
        # Spec 06: urgency and volume trend
        urgency_type: str = "",
        volume_trend: float | None = None,
        # Spec 06: per-segment impact coefficient override
        impact_coeff_override: float | None = None,
        # SPEC-E03: shock mode participation override
        max_participation_override: float | None = None,
    ) -> ExecutionFill:
        """
        Simulate execution against daily volume capacity.

        Parameters
        ----------
        side : str
            "buy" or "sell".
        reference_price : float
            Mid-price at time of execution.
        daily_volume : float
            Daily share volume.
        desired_notional_usd : float
            Target execution notional in USD.
        force_full : bool
            If True, bypass participation limits (for forced exits).
        realized_vol, overnight_gap, intraday_range : float, optional
            Microstructure context for dynamic cost adjustments.
        event_spread_multiplier : float
            Multiplier for event-window spread widening (>= 1.0).
        break_probability : float, optional
            Probability of flash crash / structural break (0-1).
        structure_uncertainty : float, optional
            Regime state entropy / uncertainty (0-1).
        drift_score : float, optional
            Trend conviction strength; high = lower costs (0-1).
        systemic_stress : float, optional
            Normalized systemic stress, e.g. VIX percentile (0-1).
        urgency_type : str
            "entry" or "exit"; affects cost acceptance and no-trade gate.
        volume_trend : float, optional
            Current volume relative to ADV EMA (e.g. 1.2 = 20% above average).
        impact_coeff_override : float, optional
            Calibrated impact coefficient for this security's market cap segment.
        max_participation_override : float, optional
            SPEC-E03: Override effective max participation rate for this
            execution (e.g. during shock mode).
        """
        px = float(max(1e-9, reference_price))
        vol = float(max(0.0, daily_volume))
        desired = float(max(0.0, desired_notional_usd))

        if desired <= 0:
            return ExecutionFill(
                fill_price=px, fill_ratio=0.0, participation_rate=0.0,
                impact_bps=0.0, spread_bps=self.spread_bps,
                urgency_type=urgency_type,
            )

        # No-trade gate: block low-urgency orders during extreme stress
        if (
            urgency_type == "entry"
            and systemic_stress is not None
            and np.isfinite(systemic_stress)
            and float(systemic_stress) > self.no_trade_stress_threshold
        ):
            logger.warning(
                "No-trade gate: blocking entry due to extreme stress "
                "(systemic_stress=%.3f > threshold=%.3f)",
                systemic_stress,
                self.no_trade_stress_threshold,
            )
            return ExecutionFill(
                fill_price=px, fill_ratio=0.0, participation_rate=0.0,
                impact_bps=0.0, spread_bps=self.spread_bps,
                urgency_type=urgency_type, no_trade_blocked=True,
            )

        # Participation limit with optional shock mode and volume trend adjustment
        effective_max_participation = self.max_participation

        # SPEC-E03: shock mode participation override
        if (
            max_participation_override is not None
            and np.isfinite(max_participation_override)
            and max_participation_override > 0
        ):
            effective_max_participation = float(max_participation_override)

        if (
            self.volume_trend_enabled
            and volume_trend is not None
            and np.isfinite(volume_trend)
        ):
            trend = float(np.clip(volume_trend, 0.5, 2.0))
            effective_max_participation = float(np.clip(
                effective_max_participation * trend,
                effective_max_participation * 0.5,
                effective_max_participation * 2.0,
            ))

        daily_dollar_volume = max(px * vol, 1e-9)
        if force_full:
            fill_notional = desired
        else:
            max_notional = daily_dollar_volume * effective_max_participation
            fill_notional = min(desired, max_notional)

        fill_ratio = float(fill_notional / desired)
        if not force_full and fill_ratio < self.min_fill_ratio:
            return ExecutionFill(
                fill_price=px, fill_ratio=0.0, participation_rate=0.0,
                impact_bps=0.0, spread_bps=self.spread_bps,
                urgency_type=urgency_type,
            )

        participation = float(fill_notional / daily_dollar_volume)

        # Use per-segment impact coefficient if provided
        impact_coeff = (
            float(max(0.0, impact_coeff_override))
            if impact_coeff_override is not None
            and np.isfinite(impact_coeff_override)
            else self.impact_coeff
        )

        spread_bps = self.spread_bps
        structural_mult = 1.0
        cost_details: Dict = {}

        if self.dynamic_costs:
            vol_component = 0.0
            if realized_vol is not None and np.isfinite(realized_vol):
                vol_component = max(
                    0.0, (float(realized_vol) - self.vol_ref) / self.vol_ref
                )

            gap_component = 0.0
            if overnight_gap is not None and np.isfinite(overnight_gap):
                gap_component = min(0.10, abs(float(overnight_gap)))

            range_component = 0.0
            if intraday_range is not None and np.isfinite(intraday_range):
                range_component = min(0.20, abs(float(intraday_range)))

            stress = 1.0
            stress += self.vol_spread_beta * vol_component
            stress += self.gap_spread_beta * gap_component
            stress += self.range_spread_beta * range_component

            liquidity_scalar = float(np.clip(
                np.sqrt(self.dollar_volume_ref / daily_dollar_volume),
                0.70,
                3.00,
            ))

            # Spec 06: compute structural state multiplier
            structural_mult, struct_details = (
                self._compute_structural_multiplier(
                    break_probability=break_probability,
                    structure_uncertainty=structure_uncertainty,
                    drift_score=drift_score,
                    systemic_stress=systemic_stress,
                )
            )

            # Combine: base stress * liquidity * structural
            spread_bps = (
                spread_bps * stress * liquidity_scalar * structural_mult
            )
            impact_coeff = (
                impact_coeff
                * (1.0 + self.vol_impact_beta * max(0.0, stress - 1.0))
                * liquidity_scalar
                * structural_mult
            )

            cost_details = {
                "vol_component": vol_component,
                "gap_component": gap_component,
                "range_component": range_component,
                "base_stress": stress,
                "liquidity_scalar": liquidity_scalar,
                **struct_details,
            }

        # Event-window spread blowout: near-event periods widen spreads.
        evt_mult = float(max(1.0, event_spread_multiplier))
        spread_bps = spread_bps * evt_mult

        impact_bps = float(impact_coeff * np.sqrt(max(0.0, participation)))
        half_spread_bps = 0.5 * spread_bps
        total_bps = half_spread_bps + impact_bps

        cost_details["spread_bps_final"] = float(spread_bps)
        cost_details["impact_bps"] = float(impact_bps)
        cost_details["total_bps"] = float(total_bps)

        if side.lower() == "buy":
            fill_price = px * (1.0 + total_bps / 10000.0)
        elif side.lower() == "sell":
            fill_price = px * (1.0 - total_bps / 10000.0)
        else:
            raise ValueError("side must be 'buy' or 'sell'")

        # Spec 06: urgency-based cost acceptance check
        _urgency_reduced = False
        if urgency_type in ("entry", "exit") and self.dynamic_costs:
            base_cost = self._base_transaction_cost_bps
            if urgency_type == "exit":
                cost_limit = (
                    base_cost * self.exit_urgency_cost_limit_mult
                )
            else:
                cost_limit = (
                    base_cost * self.entry_urgency_cost_limit_mult
                )

            if total_bps > cost_limit and not force_full:
                reduction_factor = cost_limit / max(total_bps, 1e-9)
                min_keep = 1.0 - self.stress_pullback_min_size
                reduction_factor = max(reduction_factor, min_keep)
                fill_ratio = float(fill_ratio * reduction_factor)
                fill_notional = desired * fill_ratio
                _urgency_reduced = True
                if fill_ratio < self.min_fill_ratio:
                    return ExecutionFill(
                        fill_price=px, fill_ratio=0.0,
                        participation_rate=0.0,
                        impact_bps=0.0, spread_bps=float(spread_bps),
                        urgency_type=urgency_type,
                        urgency_reduced=True,
                        cost_details=cost_details,
                    )
                # Recompute participation and impact with reduced size
                participation = float(fill_notional / daily_dollar_volume)
                impact_bps = float(
                    impact_coeff * np.sqrt(max(0.0, participation))
                )
                cost_details["urgency_cost_limit"] = float(cost_limit)
                cost_details["urgency_reduction_factor"] = float(
                    reduction_factor
                )
                cost_details["impact_bps"] = float(impact_bps)
                logger.debug(
                    "Urgency cost check: total_bps=%.2f > limit=%.2f; "
                    "reduced fill_ratio to %.3f, recomputed "
                    "participation=%.6f, impact=%.2f bps",
                    total_bps,
                    cost_limit,
                    fill_ratio,
                    participation,
                    impact_bps,
                )

        return ExecutionFill(
            fill_price=float(fill_price),
            fill_ratio=float(np.clip(fill_ratio, 0.0, 1.0)),
            participation_rate=participation,
            impact_bps=float(impact_bps),
            spread_bps=float(spread_bps),
            event_spread_multiplier_applied=evt_mult,
            structural_mult=structural_mult,
            urgency_type=urgency_type,
            urgency_reduced=_urgency_reduced,
            cost_details=cost_details,
        )


# ── Default parameters (from config) ────────────────────────────────────
_DEFAULT_COST_PARAMS: Dict[str, float] = {
    "spread_bps": 3.0,
    "impact_coeff": 25.0,
    "fill_ratio": 1.0,
}


def calibrate_cost_model(
    fills: List[Dict],
    actual_prices: pd.DataFrame,
) -> Dict[str, float]:
    """Calibrate execution-cost model parameters from historical fills.

    Compares each historical fill record against the actual market price at
    the time of execution to compute realised spread, market impact, and
    fill ratio.  Returns parameter estimates suitable for initialising an
    :class:`ExecutionModel`.

    Parameters
    ----------
    fills : list of dict
        Historical fill records.  Each dict should contain at least:

        - ``ticker`` (str): symbol
        - ``timestamp`` (str or pd.Timestamp): execution time
        - ``fill_price`` (float): actual execution price
        - ``side`` (str): ``"buy"`` or ``"sell"``
        - ``desired_notional`` (float): intended trade notional ($)
        - ``filled_notional`` (float): actually filled notional ($)
        - ``participation_rate`` (float, optional): fraction of daily volume

    actual_prices : pd.DataFrame
        Market mid-prices with a DatetimeIndex and one column per ticker.
        Used as the reference (fair) price at each fill timestamp.

    Returns
    -------
    dict
        ``spread_bps`` : float — calibrated half-spread in basis points.
        ``impact_coeff`` : float — calibrated impact coefficient (bps per
        sqrt-participation).
        ``fill_ratio`` : float — average fraction of desired notional that
        was actually filled.

    Notes
    -----
    If *fills* is empty or no fills can be matched to *actual_prices*, the
    function returns the built-in default parameters from ``config.py``.
    """
    if not fills:
        return dict(_DEFAULT_COST_PARAMS)

    spreads_bps: list[float] = []
    impacts_bps: list[float] = []
    fill_ratios: list[float] = []

    for fill in fills:
        ticker = fill.get("ticker")
        ts = fill.get("timestamp")
        fill_px = fill.get("fill_price")
        side = fill.get("side", "").lower()

        if ticker is None or ts is None or fill_px is None:
            continue
        if side not in ("buy", "sell"):
            continue

        # Resolve reference price from actual_prices
        ts = pd.Timestamp(ts)
        if ticker not in actual_prices.columns:
            continue

        # Find the closest available date (on or before the fill timestamp)
        price_series = actual_prices[ticker].dropna()
        if price_series.empty:
            continue

        # Use the nearest available price on or before the fill time
        valid_idx = price_series.index[price_series.index <= ts]
        if len(valid_idx) == 0:
            # Fall back to the nearest available date
            nearest_idx = price_series.index[
                (price_series.index - ts).abs().argmin()
            ]
            ref_px = float(price_series.loc[nearest_idx])
        else:
            ref_px = float(price_series.loc[valid_idx[-1]])

        if ref_px <= 0:
            continue

        # Compute realised slippage in bps
        if side == "buy":
            slippage_bps = (float(fill_px) - ref_px) / ref_px * 10_000
        else:
            slippage_bps = (ref_px - float(fill_px)) / ref_px * 10_000

        # Decompose: half-spread + impact = total slippage
        participation = fill.get("participation_rate")
        if participation is not None and float(participation) > 0:
            sqrt_part = np.sqrt(float(participation))
            # Attribute minimum of slippage or a reasonable cap to spread
            est_spread_raw = min(max(slippage_bps * 0.3, 0.0), slippage_bps)
            est_spread = max(0.0, est_spread_raw)
            if est_spread_raw < 0.0:
                logger.warning(
                    "Negative spread estimate clipped to 0: original=%.2f bps",
                    est_spread_raw,
                )
            est_impact = max(slippage_bps - est_spread, 0.0)
            # impact_coeff = impact_bps / sqrt(participation)
            impact_coeff_obs = (
                est_impact / sqrt_part if sqrt_part > 1e-9 else 0.0
            )
            impacts_bps.append(impact_coeff_obs)
        else:
            est_spread = max(slippage_bps, 0.0)

        spreads_bps.append(est_spread)

        # Fill ratio
        desired = fill.get("desired_notional", 0.0)
        filled = fill.get("filled_notional", 0.0)
        if desired and float(desired) > 0:
            fill_ratios.append(float(filled) / float(desired))

    # If we could not compute any statistics, return defaults
    if not spreads_bps:
        return dict(_DEFAULT_COST_PARAMS)

    calibrated_spread = max(0.0, float(np.median(spreads_bps)))
    if float(np.median(spreads_bps)) < 0.0:
        logger.warning(
            "Negative spread estimate clipped to 0: original=%.2f bps",
            float(np.median(spreads_bps)),
        )
    calibrated: Dict[str, float] = {
        "spread_bps": calibrated_spread,
        "impact_coeff": (
            float(np.median(impacts_bps))
            if impacts_bps
            else _DEFAULT_COST_PARAMS["impact_coeff"]
        ),
        "fill_ratio": (
            float(np.mean(fill_ratios))
            if fill_ratios
            else _DEFAULT_COST_PARAMS["fill_ratio"]
        ),
    }
    return calibrated
