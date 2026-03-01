"""
Data subpackage — self-contained data loading, caching, WRDS, and survivorship.
"""
from .loader import load_ohlcv, load_universe, load_survivorship_universe, load_with_delistings, warn_if_survivorship_biased
from .local_cache import save_ohlcv, load_ibkr_data, list_cached_tickers, cache_universe, normalize_ohlcv_columns, write_cache_meta
from .provider_registry import get_provider, list_providers, register_provider
from .quality import (
    DataQualityReport,
    assess_ohlcv_quality,
    check_ohlc_relationships,
    generate_quality_report,
    flag_degraded_stocks,
)
from .feature_store import FeatureStore

__all__ = [
    "load_ohlcv",
    "load_universe",
    "load_survivorship_universe",
    "load_with_delistings",
    "warn_if_survivorship_biased",
    "save_ohlcv",
    "load_ibkr_data",
    "list_cached_tickers",
    "cache_universe",
    "normalize_ohlcv_columns",
    "write_cache_meta",
    "get_provider",
    "list_providers",
    "register_provider",
    "DataQualityReport",
    "assess_ohlcv_quality",
    "check_ohlc_relationships",
    "generate_quality_report",
    "flag_degraded_stocks",
    "FeatureStore",
]
