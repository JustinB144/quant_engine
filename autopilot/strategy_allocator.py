"""
Regime-Aware Strategy Allocation — automatically adjust strategy parameters
based on detected market regime.

Instead of using fixed parameters across all regimes, this module provides
regime-specific strategy profiles that adjust entry thresholds, position sizes,
holding periods, and Kelly fractions based on the current market state.

Each profile is blended with a default profile based on regime confidence:
high confidence -> full regime profile, low confidence -> default parameters.

Integration points:
    - AutopilotEngine: use get_regime_profile() to adjust candidate parameters
    - Backtester: regime-aware parameter selection during simulation
    - PaperTrader: dynamic parameter adjustment during live paper-trading
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

from ..config import REGIME_NAMES


# ── Regime-specific strategy parameter profiles ──
# Each profile is tuned for the characteristic market behavior of its regime.
# Values are informed by empirical analysis of regime-conditional returns.

REGIME_STRATEGY_PROFILES: Dict[int, Dict] = {
    0: {  # trending_bull
        "name": "Momentum-Heavy",
        "entry_threshold": 0.008,     # Lower bar (ride trends)
        "position_size_pct": 0.08,    # Larger positions
        "max_positions": 25,          # More diversified
        "holding_days": 15,           # Hold longer (trend persists)
        "kelly_fraction": 0.60,       # More aggressive
        "confidence_threshold": 0.55, # Slightly lower (trends are reliable)
    },
    1: {  # trending_bear
        "name": "Defensive",
        "entry_threshold": 0.015,     # Higher bar (selective entries)
        "position_size_pct": 0.04,    # Smaller positions
        "max_positions": 10,          # Concentrated best ideas
        "holding_days": 5,            # Quick exits
        "kelly_fraction": 0.30,       # Conservative
        "confidence_threshold": 0.70, # High confidence required
    },
    2: {  # mean_reverting
        "name": "Mean-Reversion Focus",
        "entry_threshold": 0.010,     # Standard bar
        "position_size_pct": 0.06,    # Moderate positions
        "max_positions": 20,          # Standard diversification
        "holding_days": 10,           # Standard hold
        "kelly_fraction": 0.50,       # Moderate
        "confidence_threshold": 0.60, # Standard
    },
    3: {  # high_volatility
        "name": "Risk-Off",
        "entry_threshold": 0.025,     # Very high bar
        "position_size_pct": 0.03,    # Tiny positions
        "max_positions": 5,           # Minimal exposure
        "holding_days": 3,            # Very short holds
        "kelly_fraction": 0.20,       # Very conservative
        "confidence_threshold": 0.80, # Very high confidence required
    },
}

# Default profile (used when regime is unknown or confidence is low)
DEFAULT_PROFILE: Dict = {
    "name": "Default",
    "entry_threshold": 0.010,
    "position_size_pct": 0.05,
    "max_positions": 20,
    "holding_days": 10,
    "kelly_fraction": 0.50,
    "confidence_threshold": 0.60,
}


@dataclass
class StrategyProfile:
    """A complete set of strategy parameters for a given regime context."""
    name: str
    entry_threshold: float
    position_size_pct: float
    max_positions: int
    holding_days: int
    kelly_fraction: float
    confidence_threshold: float
    regime: int
    regime_confidence: float
    is_blended: bool = False


class StrategyAllocator:
    """
    Select and blend regime-specific strategy parameters.

    When regime confidence is high, parameters come almost entirely from
    the regime-specific profile.  When confidence is low, parameters
    converge to the default profile.

    Usage:
        allocator = StrategyAllocator()

        # Get regime-appropriate parameters
        profile = allocator.get_regime_profile(regime=0, regime_confidence=0.85)
        print(f"Entry threshold: {profile.entry_threshold}")
        print(f"Max positions: {profile.max_positions}")
    """

    def __init__(
        self,
        profiles: Optional[Dict[int, Dict]] = None,
        default_profile: Optional[Dict] = None,
    ):
        """
        Args:
            profiles: Override regime profiles.
            default_profile: Override default profile.
        """
        self.profiles = profiles or REGIME_STRATEGY_PROFILES
        self.default = default_profile or DEFAULT_PROFILE

    def get_regime_profile(
        self,
        regime: int,
        regime_confidence: float = 0.5,
    ) -> StrategyProfile:
        """Get blended strategy parameters for a given regime and confidence.

        Blending formula:
            param = confidence * regime_param + (1 - confidence) * default_param

        For integer parameters (max_positions, holding_days), the blended
        value is rounded to the nearest integer.

        Args:
            regime: Regime code (0-3).
            regime_confidence: Confidence in regime classification (0-1).

        Returns:
            StrategyProfile with blended parameters.
        """
        regime_confidence = max(0.0, min(1.0, regime_confidence))
        profile = self.profiles.get(regime, self.default)

        blended = {}
        for key in self.default:
            if key == "name":
                blended[key] = profile.get("name", self.default["name"])
                continue

            regime_val = profile.get(key, self.default[key])
            default_val = self.default[key]

            # Blend based on confidence
            if isinstance(default_val, int):
                raw = regime_confidence * regime_val + (1 - regime_confidence) * default_val
                blended[key] = max(1, round(raw))
            else:
                blended[key] = regime_confidence * regime_val + (1 - regime_confidence) * default_val

        return StrategyProfile(
            name=blended["name"],
            entry_threshold=blended["entry_threshold"],
            position_size_pct=blended["position_size_pct"],
            max_positions=blended["max_positions"],
            holding_days=blended["holding_days"],
            kelly_fraction=blended["kelly_fraction"],
            confidence_threshold=blended["confidence_threshold"],
            regime=regime,
            regime_confidence=regime_confidence,
            is_blended=regime_confidence < 0.95,
        )

    def get_all_profiles(self) -> Dict[int, StrategyProfile]:
        """Return all regime profiles at full confidence (for display/comparison)."""
        return {
            regime: self.get_regime_profile(regime, regime_confidence=1.0)
            for regime in self.profiles
        }

    def summarize(self) -> str:
        """Human-readable summary of all regime profiles."""
        lines = ["Regime Strategy Profiles:"]
        for regime, profile_dict in sorted(self.profiles.items()):
            name = REGIME_NAMES.get(regime, f"regime_{regime}")
            p = profile_dict
            lines.append(
                f"  {name} ({p.get('name', 'N/A')}): "
                f"entry={p.get('entry_threshold', 0):.3f}, "
                f"size={p.get('position_size_pct', 0):.2f}, "
                f"max_pos={p.get('max_positions', 0)}, "
                f"hold={p.get('holding_days', 0)}d, "
                f"kelly={p.get('kelly_fraction', 0):.2f}"
            )
        return "\n".join(lines)
