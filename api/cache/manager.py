"""In-memory TTL cache manager."""
from __future__ import annotations

import fnmatch
import time
from typing import Any, Dict, Optional


# Default TTLs in seconds per cache domain.
DEFAULT_TTLS: Dict[str, int] = {
    "dashboard": 30,
    "regime": 60,
    "model_health": 120,
    "feature_importance": 300,
    "signals": 30,
    "health": 120,
    "benchmark": 300,
}


class CacheManager:
    """Simple in-memory dict cache with per-key TTL."""

    def __init__(self) -> None:
        self._store: Dict[str, tuple] = {}  # key -> (value, expires_at)

    def get(self, key: str) -> Optional[Any]:
        """Return cached value or ``None`` if expired / missing."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store *value* under *key* with a TTL in seconds.

        If *ttl* is ``None``, infer from the key prefix via ``DEFAULT_TTLS``
        (fallback 60 s).
        """
        if ttl is None:
            prefix = key.split(":")[0] if ":" in key else key
            ttl = DEFAULT_TTLS.get(prefix, 60)
        self._store[key] = (value, time.monotonic() + ttl)

    def invalidate(self, key: str) -> None:
        """Remove a single key."""
        self._store.pop(key, None)

    def invalidate_pattern(self, pattern: str) -> int:
        """Remove all keys matching a glob *pattern*. Returns count removed."""
        to_remove = [k for k in self._store if fnmatch.fnmatch(k, pattern)]
        for k in to_remove:
            del self._store[k]
        return len(to_remove)

    def invalidate_all(self) -> None:
        """Flush the entire cache."""
        self._store.clear()
