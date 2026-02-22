"""
Test module for provider registry behavior and regressions.
"""

import unittest

from quant_engine.data.provider_registry import get_provider, list_providers
from quant_engine.kalshi.client import KalshiClient


class ProviderRegistryTests(unittest.TestCase):
    """Test cases covering provider registry behavior and system invariants."""
    def test_registry_lists_core_providers(self):
        providers = list_providers()
        self.assertIn("wrds", providers)
        self.assertIn("kalshi", providers)

    def test_registry_rejects_unknown_provider(self):
        with self.assertRaises(ValueError):
            get_provider("does_not_exist")

    def test_registry_can_construct_kalshi_provider(self):
        provider = get_provider("kalshi", client=KalshiClient(api_key="", api_secret=""))
        self.assertTrue(hasattr(provider, "available"))


if __name__ == "__main__":
    unittest.main()

