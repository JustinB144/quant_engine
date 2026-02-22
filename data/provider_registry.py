"""
Provider registry for unified data-provider access (WRDS, Kalshi, ...).
"""
from __future__ import annotations

from typing import Callable, Dict, List

from .provider_base import DataProvider


ProviderFactory = Callable[..., DataProvider]


def _wrds_factory(**kwargs):
    """Lazily import and construct the WRDS provider."""
    from .wrds_provider import WRDSProvider

    return WRDSProvider(**kwargs)


def _kalshi_factory(**kwargs):
    """Lazily import and construct the Kalshi provider."""
    from ..kalshi.provider import KalshiProvider

    return KalshiProvider(**kwargs)


_REGISTRY: Dict[str, ProviderFactory] = {
    "wrds": _wrds_factory,
    "kalshi": _kalshi_factory,
}


def get_provider(name: str, **kwargs) -> DataProvider:
    """Construct a registered provider instance by name."""
    key = str(name).lower().strip()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")
    return _REGISTRY[key](**kwargs)


def list_providers() -> List[str]:
    """Return the names of supported data providers available through the registry."""
    return sorted(_REGISTRY.keys())


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register or override a provider factory under a normalized key."""
    key = str(name).lower().strip()
    if not key:
        raise ValueError("Provider name cannot be empty.")
    _REGISTRY[key] = factory
