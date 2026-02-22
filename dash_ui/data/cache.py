"""
Caching layer for Dash UI data loading operations.

Provides a thread-safe cache instance and decorator for memoizing expensive
computations (data loading, metrics calculation, etc.) across requests.

The cache is initialized in server.py with Flask-Caching FileSystemCache
backend for persistent caching across server restarts.

Usage:
    from quant_engine.dash_ui.data.cache import cached

    @cached(timeout=300)  # Cache for 5 minutes
    def load_expensive_data():
        return compute_result()
"""
import tempfile
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

from flask_caching import Cache

# Global cache instance - initialized by server.py via init_cache()
_cache_dir = str(Path(tempfile.gettempdir()) / "qe_dash_cache")
cache = Cache(config={
    "CACHE_TYPE": "filesystem",
    "CACHE_DIR": _cache_dir,
    "CACHE_DEFAULT_TIMEOUT": 60,
})


def init_cache(app) -> None:
    """
    Initialize the Flask cache instance with a Dash application.

    Must be called once after creating the Dash app and configuring
    the Flask server with cache settings.

    Args:
        app: A Dash application instance (has .server attribute).

    Example:
        from dash_ui.app import create_app
        from dash_ui.data.cache import init_cache

        app = create_app()
        init_cache(app)  # Initialize cache
    """
    cache.init_app(app.server)


def cached(timeout: int = 60) -> Callable:
    """
    Decorator to cache function results for a specified duration.

    Wraps Flask-Caching's memoize decorator to cache expensive operations.
    Cache key is automatically derived from function name and arguments.

    Args:
        timeout: Cache duration in seconds (default: 60).

    Returns:
        Decorator function that wraps the target function.

    Example:
        @cached(timeout=300)
        def load_trades_data():
            return pd.read_csv(...)

        # First call computes result and caches it
        trades_1 = load_trades_data()  # ~1s to compute

        # Second call within 300s returns cached result
        trades_2 = load_trades_data()  # ~0.01s from cache
    """
    def decorator(func: Callable) -> Callable:
        # Create the memoized version once at decoration time
        memoized = cache.cached(timeout=timeout, key_prefix=func.__qualname__)(func)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return memoized(*args, **kwargs)
        return wrapper
    return decorator


__all__ = [
    "cache",
    "init_cache",
    "cached",
]
