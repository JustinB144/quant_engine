"""
Intraday data providers for multi-source download and validation.

Providers:
    AlpacaProvider      — Free, 200 req/min, 10+ year depth (primary)
    AlphaVantageProvider — Paid tier, 75 req/min, 20+ year depth (secondary)

IBKR remains the truth source for validation, accessed directly via ib_insync
in cross_source_validator.py rather than through this provider abstraction.
"""

from .alpaca_provider import AlpacaProvider
from .alpha_vantage_provider import AlphaVantageProvider

__all__ = [
    "AlpacaProvider",
    "AlphaVantageProvider",
]
