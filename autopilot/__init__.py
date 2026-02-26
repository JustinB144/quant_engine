"""
Autopilot layer: discovery, promotion, and paper-trading orchestration.
"""

from .strategy_discovery import StrategyCandidate, StrategyDiscovery
from .promotion_gate import PromotionDecision, PromotionGate
from .registry import StrategyRegistry
from .paper_trader import PaperTrader
from .engine import AutopilotEngine
from .meta_labeler import MetaLabelingModel

__all__ = [
    "StrategyCandidate",
    "StrategyDiscovery",
    "PromotionDecision",
    "PromotionGate",
    "StrategyRegistry",
    "PaperTrader",
    "AutopilotEngine",
    "MetaLabelingModel",
]

