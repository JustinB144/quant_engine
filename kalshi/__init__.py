"""
Kalshi vertical for intraday event-market research.

Keeps event-market complexity isolated while staying inside quant_engine.
"""

from .client import KalshiClient
from .storage import EventTimeStore
from .provider import KalshiProvider
from .pipeline import KalshiPipeline
from .router import KalshiDataRouter
from .quality import QualityDimensions, StalePolicy
from .mapping_store import EventMarketMappingStore, EventMarketMappingRecord
from .distribution import DistributionConfig, build_distribution_panel
from .options import add_options_disagreement_features, build_options_reference_panel
from .promotion import EventPromotionConfig, evaluate_event_promotion
from .events import (
    EventFeatureConfig,
    add_reference_disagreement_features,
    asof_join,
    build_asset_response_labels,
    build_event_feature_panel,
    build_event_labels,
)
from .walkforward import (
    EventWalkForwardConfig,
    EventWalkForwardResult,
    evaluate_event_contract_metrics,
    run_event_walkforward,
)

__all__ = [
    "KalshiClient",
    "KalshiDataRouter",
    "EventTimeStore",
    "KalshiProvider",
    "KalshiPipeline",
    "QualityDimensions",
    "StalePolicy",
    "EventMarketMappingStore",
    "EventMarketMappingRecord",
    "DistributionConfig",
    "EventFeatureConfig",
    "EventWalkForwardConfig",
    "EventWalkForwardResult",
    "EventPromotionConfig",
    "evaluate_event_promotion",
    "evaluate_event_contract_metrics",
    "build_distribution_panel",
    "build_options_reference_panel",
    "add_options_disagreement_features",
    "build_event_feature_panel",
    "build_event_labels",
    "build_asset_response_labels",
    "run_event_walkforward",
    "add_reference_disagreement_features",
    "asof_join",
]
