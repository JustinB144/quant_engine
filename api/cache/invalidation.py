"""Event-driven cache invalidation helpers."""
from __future__ import annotations

from .manager import CacheManager


def invalidate_on_train(cache: CacheManager) -> None:
    """Clear caches affected by a new model training run."""
    cache.invalidate_pattern("model_health*")
    cache.invalidate_pattern("feature_importance*")
    cache.invalidate_pattern("signals*")
    cache.invalidate_pattern("dashboard*")


def invalidate_on_backtest(cache: CacheManager) -> None:
    """Clear caches affected by a new backtest run."""
    cache.invalidate_pattern("dashboard*")
    cache.invalidate_pattern("benchmark*")


def invalidate_on_data_refresh(cache: CacheManager) -> None:
    """Clear caches affected by fresh market data."""
    cache.invalidate_pattern("dashboard*")
    cache.invalidate_pattern("signals*")
    cache.invalidate_pattern("regime*")
    cache.invalidate_pattern("health*")


def invalidate_on_config_change(cache: CacheManager) -> None:
    """Clear all caches when runtime config is patched."""
    cache.invalidate_all()
