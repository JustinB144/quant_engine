"""
Shared provider protocol for pluggable data connectors.
"""
from __future__ import annotations

from typing import Protocol


class DataProvider(Protocol):
    """Protocol defining the minimal interface expected from pluggable data providers."""
    def available(self) -> bool:
        """Return whether the resource is available in the current runtime."""
        ...

