"""
Strategy discovery for execution-layer parameter variants.

This intentionally searches variations around a fixed predictive model:
entry/confirmation thresholds, risk mode, and portfolio concentration.
"""
from dataclasses import dataclass, asdict
from typing import Dict, List

from ..config import (
    ENTRY_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    POSITION_SIZE_PCT,
    DISCOVERY_ENTRY_MULTIPLIERS,
    DISCOVERY_CONFIDENCE_OFFSETS,
    DISCOVERY_RISK_VARIANTS,
    DISCOVERY_MAX_POSITIONS_VARIANTS,
)


@dataclass(frozen=True)
class StrategyCandidate:
    """Execution-parameter variant generated for backtest and promotion evaluation."""
    strategy_id: str
    horizon: int
    entry_threshold: float
    confidence_threshold: float
    use_risk_management: bool
    max_positions: int
    position_size_pct: float

    def to_dict(self) -> Dict:
        """Serialize StrategyCandidate to a dictionary."""
        return asdict(self)


class StrategyDiscovery:
    """Generates a deterministic candidate grid for backtest validation."""

    def __init__(
        self,
        base_entry_threshold: float = ENTRY_THRESHOLD,
        base_confidence_threshold: float = CONFIDENCE_THRESHOLD,
        base_position_size_pct: float = POSITION_SIZE_PCT,
    ):
        """Initialize StrategyDiscovery."""
        self.base_entry = base_entry_threshold
        self.base_conf = base_confidence_threshold
        self.base_size = base_position_size_pct

    def generate(self, horizon: int) -> List[StrategyCandidate]:
        """generate."""
        candidates: List[StrategyCandidate] = []
        for mult in DISCOVERY_ENTRY_MULTIPLIERS:
            entry = max(0.0005, self.base_entry * mult)
            for conf_offset in DISCOVERY_CONFIDENCE_OFFSETS:
                conf = min(0.99, max(0.05, self.base_conf + conf_offset))
                for risk_mode in DISCOVERY_RISK_VARIANTS:
                    for max_pos in DISCOVERY_MAX_POSITIONS_VARIANTS:
                        sid = (
                            f"h{horizon}"
                            f"_e{int(round(entry * 10000))}"
                            f"_c{int(round(conf * 100))}"
                            f"_r{int(risk_mode)}"
                            f"_m{max_pos}"
                        )
                        candidates.append(
                            StrategyCandidate(
                                strategy_id=sid,
                                horizon=horizon,
                                entry_threshold=float(entry),
                                confidence_threshold=float(conf),
                                use_risk_management=bool(risk_mode),
                                max_positions=int(max_pos),
                                position_size_pct=float(self.base_size),
                            ),
                        )
        return candidates

