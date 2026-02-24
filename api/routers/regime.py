"""Regime detection endpoints — metadata, current state, history."""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends

from ..cache.manager import CacheManager
from ..deps.providers import get_cache
from ..schemas.envelope import ApiResponse

router = APIRouter(prefix="/api/regime", tags=["regime"])


def _build_regime_metadata() -> dict:
    """Build regime metadata payload from config values."""
    from quant_engine.config import (
        REGIME_NAMES,
        REGIME_RISK_MULTIPLIER,
        REGIME_STOP_MULTIPLIER,
        REGIME_MODEL_TYPE,
        REGIME_ENSEMBLE_ENABLED,
    )

    regime_definitions = {
        0: {
            "definition": (
                "Market showing sustained upward momentum with low volatility. "
                "Hurst exponent > 0.55 and positive SMA slope."
            ),
            "detection": (
                "Identified when HMM posterior probability exceeds 50% for this state, "
                "characterized by positive returns, low NATR, and trending Hurst exponent."
            ),
            "color": "#22c55e",
        },
        1: {
            "definition": (
                "Market showing sustained downward momentum. "
                "Hurst exponent > 0.55 and negative SMA slope."
            ),
            "detection": (
                "Identified when HMM posterior shows dominant negative-return state "
                "with elevated volatility and downward SMA slope."
            ),
            "color": "#ef4444",
        },
        2: {
            "definition": (
                "Market oscillating without clear trend. Hurst exponent < 0.45."
            ),
            "detection": (
                "Identified by low Hurst exponent indicating anti-persistence, "
                "low ADX suggesting no directional trend, and oscillating returns."
            ),
            "color": "#eab308",
        },
        3: {
            "definition": (
                "Elevated market volatility regardless of direction. "
                "NATR exceeds 80th percentile of rolling 252-day window."
            ),
            "detection": (
                "Identified by NATR exceeding the 80th percentile threshold, "
                "high realized volatility, and often accompanied by volume spikes."
            ),
            "color": "#a855f7",
        },
    }

    regimes = {}
    for code in range(4):
        name_raw = REGIME_NAMES.get(code, f"regime_{code}")
        display_name = name_raw.replace("_", " ").title()
        defn = regime_definitions.get(code, {})
        size_mult = REGIME_RISK_MULTIPLIER.get(code, 1.0)
        stop_mult = REGIME_STOP_MULTIPLIER.get(code, 1.0)

        # Build portfolio impact description
        if size_mult == 1.0 and stop_mult == 1.0:
            impact_desc = "Full position sizes, standard stop losses. Best regime for directional strategies."
        elif size_mult < 0.7:
            impact_desc = (
                f"Reduced to {size_mult:.0%} position sizes, widest stops. "
                "Capital preservation priority."
            )
        elif stop_mult < 1.0:
            impact_desc = (
                f"Reduced to {size_mult:.0%} position sizes, tighter stops "
                f"({stop_mult:.1f}x). Defensive posture."
            )
        else:
            impact_desc = (
                f"{size_mult:.0%} position sizes, {stop_mult:.1f}x stop multiplier. "
                "Model has historically variable Sharpe in this regime."
            )

        regimes[str(code)] = {
            "name": display_name,
            "definition": defn.get("definition", ""),
            "detection": defn.get("detection", ""),
            "portfolio_impact": {
                "position_size_multiplier": size_mult,
                "stop_loss_multiplier": stop_mult,
                "description": impact_desc,
            },
            "color": defn.get("color", "#888888"),
        }

    return {
        "regimes": regimes,
        "detection_method": REGIME_MODEL_TYPE,
        "ensemble_enabled": REGIME_ENSEMBLE_ENABLED,
        "transition_matrix_explanation": (
            "Each cell (i,j) shows the probability of transitioning from regime i "
            "to regime j. Diagonal values show regime persistence (higher = more "
            "stable). Off-diagonal shows transition likelihood."
        ),
    }


@router.get("/metadata")
async def regime_metadata(cache: CacheManager = Depends(get_cache)) -> ApiResponse:
    """Return regime definitions, detection methodology, and portfolio impact."""
    cached = cache.get("regime:metadata")
    if cached is not None:
        return ApiResponse.from_cached(cached)
    t0 = time.monotonic()
    data = _build_regime_metadata()
    elapsed = (time.monotonic() - t0) * 1000
    cache.set("regime:metadata", data, ttl=3600)  # Changes rarely
    return ApiResponse.success(data, elapsed_ms=elapsed)
