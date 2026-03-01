"""In-memory TTL cache manager with thread safety and bounded size."""
from __future__ import annotations

import copy
import fnmatch
import threading
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
    """Thread-safe in-memory dict cache with per-key TTL and bounded size.

    All mutations are protected by a lock to prevent races from concurrent
    async handlers.  Values are deep-copied on retrieval so callers cannot
    accidentally mutate cached state.
    """

    def __init__(self, max_size: int = 100) -> None:
        self._store: Dict[str, tuple] = {}  # key -> (value, expires_at)
        self._lock = threading.Lock()
        self._max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        """Return a deep copy of the cached value or ``None`` if expired / missing."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return copy.deepcopy(value)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store *value* under *key* with a TTL in seconds.

        If *ttl* is ``None``, infer from the key prefix via ``DEFAULT_TTLS``
        (fallback 60 s).  When the cache exceeds ``max_size``, the oldest
        entry is evicted.
        """
        if ttl is None:
            prefix = key.split(":")[0] if ":" in key else key
            ttl = DEFAULT_TTLS.get(prefix, 60)
        with self._lock:
            self._store[key] = (value, time.monotonic() + ttl)
            # Evict oldest entries when over capacity
            if len(self._store) > self._max_size:
                self._evict_oldest()

    def invalidate(self, key: str) -> None:
        """Remove a single key."""
        with self._lock:
            self._store.pop(key, None)

    def invalidate_pattern(self, pattern: str) -> int:
        """Remove all keys matching a glob *pattern*. Returns count removed."""
        with self._lock:
            to_remove = [k for k in self._store if fnmatch.fnmatch(k, pattern)]
            for k in to_remove:
                del self._store[k]
            return len(to_remove)

    def invalidate_all(self) -> None:
        """Flush the entire cache."""
        with self._lock:
            self._store.clear()

    # ── Internal ──────────────────────────────────────────────────────

    def _evict_oldest(self) -> None:
        """Remove the entry with the earliest expiry. Must be called under lock."""
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k][1])
        del self._store[oldest_key]
