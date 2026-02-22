"""
Execution simulator with spread, market impact, and participation limits.

Includes a cost-model calibration routine that fits spread, impact, and fill
parameters from historical fill data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class ExecutionFill:
    """Simulated execution fill outcome returned by the execution model."""
    fill_price: float
    fill_ratio: float
    participation_rate: float
    impact_bps: float
    spread_bps: float
    event_spread_multiplier_applied: float = 1.0


class ExecutionModel:
    """
    Simple market-impact model for backtests.
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
    ) -> ExecutionFill:
        """
        Simulate execution against daily volume capacity.
        """
        px = float(max(1e-9, reference_price))
        vol = float(max(0.0, daily_volume))
        desired = float(max(0.0, desired_notional_usd))
        if desired <= 0:
            return ExecutionFill(px, 0.0, 0.0, 0.0, self.spread_bps)

        daily_dollar_volume = max(px * vol, 1e-9)
        if force_full:
            fill_notional = desired
        else:
            max_notional = daily_dollar_volume * self.max_participation
            fill_notional = min(desired, max_notional)

        fill_ratio = float(fill_notional / desired)
        if not force_full and fill_ratio < self.min_fill_ratio:
            return ExecutionFill(px, 0.0, 0.0, 0.0, self.spread_bps)

        participation = float(fill_notional / daily_dollar_volume)
        spread_bps = self.spread_bps
        impact_coeff = self.impact_coeff
        if self.dynamic_costs:
            vol_component = 0.0
            if realized_vol is not None and np.isfinite(realized_vol):
                vol_component = max(0.0, (float(realized_vol) - self.vol_ref) / self.vol_ref)

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

            liquidity_scalar = np.clip(
                np.sqrt(self.dollar_volume_ref / daily_dollar_volume),
                0.70,
                3.00,
            )
            spread_bps = spread_bps * stress * liquidity_scalar
            impact_coeff = impact_coeff * (1.0 + self.vol_impact_beta * max(0.0, stress - 1.0)) * liquidity_scalar

        # Event-window spread blowout: near-event periods widen spreads.
        evt_mult = float(max(1.0, event_spread_multiplier))
        spread_bps = spread_bps * evt_mult

        impact_bps = float(impact_coeff * np.sqrt(max(0.0, participation)))
        half_spread_bps = 0.5 * spread_bps
        total_bps = half_spread_bps + impact_bps

        if side.lower() == "buy":
            fill_price = px * (1.0 + total_bps / 10000.0)
        elif side.lower() == "sell":
            fill_price = px * (1.0 - total_bps / 10000.0)
        else:
            raise ValueError("side must be 'buy' or 'sell'")

        return ExecutionFill(
            fill_price=float(fill_price),
            fill_ratio=float(np.clip(fill_ratio, 0.0, 1.0)),
            participation_rate=participation,
            impact_bps=impact_bps,
            spread_bps=float(spread_bps),
            event_spread_multiplier_applied=evt_mult,
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
            est_spread = min(max(slippage_bps * 0.3, 0.0), slippage_bps)
            est_impact = max(slippage_bps - est_spread, 0.0)
            # impact_coeff = impact_bps / sqrt(participation)
            impact_coeff_obs = est_impact / sqrt_part if sqrt_part > 1e-9 else 0.0
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

    calibrated: Dict[str, float] = {
        "spread_bps": float(np.median(spreads_bps)),
        "impact_coeff": float(np.median(impacts_bps)) if impacts_bps else _DEFAULT_COST_PARAMS["impact_coeff"],
        "fill_ratio": float(np.mean(fill_ratios)) if fill_ratios else _DEFAULT_COST_PARAMS["fill_ratio"],
    }
    return calibrated
