"""
Tests for SPEC-B03: Fix broken import and error handling in alternative.py.

Verifies that:
  1. The import chain for the WRDS provider does not raise unexpected errors.
  2. AlternativeDataProvider in strict mode raises WRDSUnavailableError when
     WRDS is not reachable.
  3. AlternativeDataProvider in graceful mode (default) returns None and logs
     a warning instead of silently returning empty data.
  4. Module-level convenience functions (get_earnings_surprise, etc.) are
     importable and delegate correctly.
  5. Genuine code errors (e.g. AttributeError) inside wrds_provider are NOT
     swallowed — they propagate immediately.
"""

from __future__ import annotations

import importlib
import logging
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd


class TestImportChain(unittest.TestCase):
    """Verify that importing from data.alternative never raises ImportError."""

    def test_import_alternative_module(self):
        """The alternative module itself should be importable."""
        import quant_engine.data.alternative as mod
        self.assertTrue(hasattr(mod, "AlternativeDataProvider"))
        self.assertTrue(hasattr(mod, "compute_alternative_features"))
        self.assertTrue(hasattr(mod, "WRDSUnavailableError"))

    def test_import_convenience_functions(self):
        """Module-level convenience functions should be importable."""
        from quant_engine.data.alternative import (
            get_earnings_surprise,
            get_fundamentals,
            get_short_interest,
            get_options_flow,
            get_insider_transactions,
            get_institutional_ownership,
        )
        # Should be callables
        for fn in (
            get_earnings_surprise,
            get_fundamentals,
            get_short_interest,
            get_options_flow,
            get_insider_transactions,
            get_institutional_ownership,
        ):
            self.assertTrue(callable(fn))

    def test_import_wrds_unavailable_error(self):
        """WRDSUnavailableError should be importable."""
        from quant_engine.data.alternative import WRDSUnavailableError
        self.assertTrue(issubclass(WRDSUnavailableError, RuntimeError))

    def test_import_get_wrds_unavailable_reason(self):
        """get_wrds_unavailable_reason should be importable."""
        from quant_engine.data.alternative import get_wrds_unavailable_reason
        self.assertTrue(callable(get_wrds_unavailable_reason))


class TestStrictMode(unittest.TestCase):
    """Verify that strict=True raises WRDSUnavailableError when WRDS is down."""

    def setUp(self):
        # Reset the module-level singleton state before each test
        import quant_engine.data.alternative as mod
        mod._wrds_provider = None
        mod._wrds_import_error = None
        mod._default_provider = None

    def tearDown(self):
        import quant_engine.data.alternative as mod
        mod._wrds_provider = None
        mod._wrds_import_error = None
        mod._default_provider = None

    def test_strict_constructor_raises_when_wrds_unavailable(self):
        """AlternativeDataProvider(strict=True) raises WRDSUnavailableError."""
        from quant_engine.data.alternative import (
            AlternativeDataProvider,
            WRDSUnavailableError,
        )

        # Patch _get_wrds to simulate unavailability
        with patch("quant_engine.data.alternative._get_wrds", return_value=None):
            with self.assertRaises(WRDSUnavailableError) as ctx:
                AlternativeDataProvider(strict=True)
            self.assertIn("unavailable", str(ctx.exception).lower())

    def test_strict_method_raises_when_wrds_becomes_unavailable(self):
        """Even if construction succeeds, methods raise in strict mode if
        the WRDS connection drops."""
        from quant_engine.data.alternative import (
            AlternativeDataProvider,
            WRDSUnavailableError,
        )

        mock_wrds = MagicMock()
        mock_wrds.available.return_value = True

        with patch("quant_engine.data.alternative._get_wrds", return_value=mock_wrds):
            provider = AlternativeDataProvider(strict=True)

        # Now simulate connection dropping
        mock_wrds.available.return_value = False
        provider._wrds = mock_wrds

        with self.assertRaises(WRDSUnavailableError):
            provider.get_earnings_surprise("AAPL")

    def test_strict_mode_all_methods_raise(self):
        """All data methods should raise in strict mode when WRDS is down."""
        from quant_engine.data.alternative import (
            AlternativeDataProvider,
            WRDSUnavailableError,
        )

        # Create provider with mock, then break it
        mock_wrds = MagicMock()
        mock_wrds.available.return_value = True

        with patch("quant_engine.data.alternative._get_wrds", return_value=mock_wrds):
            provider = AlternativeDataProvider(strict=True)

        mock_wrds.available.return_value = False

        methods = [
            ("get_earnings_surprise", ("AAPL",)),
            ("get_options_flow", ("AAPL",)),
            ("get_short_interest", ("AAPL",)),
            ("get_insider_transactions", ("AAPL",)),
            ("get_institutional_ownership", ("AAPL",)),
        ]

        for method_name, args in methods:
            with self.subTest(method=method_name):
                method = getattr(provider, method_name)
                with self.assertRaises(WRDSUnavailableError):
                    method(*args)


class TestGracefulMode(unittest.TestCase):
    """Verify that default (graceful) mode returns None and logs warnings."""

    def setUp(self):
        import quant_engine.data.alternative as mod
        mod._wrds_provider = None
        mod._wrds_import_error = None
        mod._default_provider = None

    def tearDown(self):
        import quant_engine.data.alternative as mod
        mod._wrds_provider = None
        mod._wrds_import_error = None
        mod._default_provider = None

    def test_graceful_constructor_does_not_raise(self):
        """AlternativeDataProvider() should NOT raise when WRDS is unavailable."""
        from quant_engine.data.alternative import AlternativeDataProvider

        with patch("quant_engine.data.alternative._get_wrds", return_value=None):
            provider = AlternativeDataProvider()  # should not raise
            self.assertFalse(provider.is_available)

    def test_graceful_methods_return_none(self):
        """All data methods should return None in graceful mode."""
        from quant_engine.data.alternative import AlternativeDataProvider

        with patch("quant_engine.data.alternative._get_wrds", return_value=None):
            provider = AlternativeDataProvider()

        methods = [
            ("get_earnings_surprise", ("AAPL",)),
            ("get_options_flow", ("AAPL",)),
            ("get_short_interest", ("AAPL",)),
            ("get_insider_transactions", ("AAPL",)),
            ("get_institutional_ownership", ("AAPL",)),
        ]

        for method_name, args in methods:
            with self.subTest(method=method_name):
                result = getattr(provider, method_name)(*args)
                self.assertIsNone(result)

    def test_graceful_methods_log_warning(self):
        """Each method should log a WARNING (not DEBUG) when WRDS is unavailable."""
        from quant_engine.data.alternative import AlternativeDataProvider

        with patch("quant_engine.data.alternative._get_wrds", return_value=None):
            provider = AlternativeDataProvider()

            with self.assertLogs("quant_engine.data.alternative", level="WARNING") as cm:
                provider.get_earnings_surprise("AAPL")

        log_output = " ".join(cm.output)
        self.assertIn("WRDS unavailable", log_output)

    def test_is_available_property(self):
        """is_available should reflect the WRDS connection state."""
        from quant_engine.data.alternative import AlternativeDataProvider

        mock_wrds = MagicMock()
        mock_wrds.available.return_value = True

        with patch("quant_engine.data.alternative._get_wrds", return_value=mock_wrds):
            provider = AlternativeDataProvider()
            self.assertTrue(provider.is_available)

        with patch("quant_engine.data.alternative._get_wrds", return_value=None):
            provider2 = AlternativeDataProvider()
            self.assertFalse(provider2.is_available)

    def test_compute_alternative_features_returns_empty_when_unavailable(self):
        """compute_alternative_features should return empty DataFrame, not raise."""
        from quant_engine.data.alternative import (
            AlternativeDataProvider,
            compute_alternative_features,
        )

        with patch("quant_engine.data.alternative._get_wrds", return_value=None):
            provider = AlternativeDataProvider()
            result = compute_alternative_features("AAPL", provider=provider)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(result.empty)


class TestGetWrdsErrorPropagation(unittest.TestCase):
    """Verify that _get_wrds() propagates non-ImportError exceptions."""

    def setUp(self):
        import quant_engine.data.alternative as mod
        mod._wrds_provider = None
        mod._wrds_import_error = None

    def tearDown(self):
        import quant_engine.data.alternative as mod
        mod._wrds_provider = None
        mod._wrds_import_error = None

    def test_import_error_is_handled_gracefully(self):
        """ImportError (expected, e.g. wrds not installed) should return None."""
        from quant_engine.data.alternative import _get_wrds

        with patch("importlib.import_module", side_effect=ImportError("No module")):
            result = _get_wrds()
            self.assertIsNone(result)

    def test_non_import_error_propagates(self):
        """Non-ImportError (unexpected, e.g. AttributeError) should propagate."""
        from quant_engine.data.alternative import _get_wrds

        with patch(
            "importlib.import_module",
            side_effect=AttributeError("broken attribute"),
        ):
            with self.assertRaises(AttributeError):
                _get_wrds()

    def test_syntax_error_propagates(self):
        """SyntaxError inside wrds_provider should propagate, not be swallowed."""
        from quant_engine.data.alternative import _get_wrds

        with patch(
            "importlib.import_module",
            side_effect=SyntaxError("bad syntax"),
        ):
            with self.assertRaises(SyntaxError):
                _get_wrds()

    def test_runtime_error_propagates(self):
        """RuntimeError should propagate immediately."""
        from quant_engine.data.alternative import _get_wrds

        with patch(
            "importlib.import_module",
            side_effect=RuntimeError("something broke"),
        ):
            with self.assertRaises(RuntimeError):
                _get_wrds()


class TestGetWrdsUnavailableReason(unittest.TestCase):
    """Verify the get_wrds_unavailable_reason() diagnostic function."""

    def setUp(self):
        import quant_engine.data.alternative as mod
        mod._wrds_provider = None
        mod._wrds_import_error = None

    def tearDown(self):
        import quant_engine.data.alternative as mod
        mod._wrds_provider = None
        mod._wrds_import_error = None

    def test_returns_none_when_wrds_available(self):
        """Should return None when WRDS is successfully connected."""
        from quant_engine.data.alternative import get_wrds_unavailable_reason

        mock_wrds = MagicMock()
        mock_wrds.available.return_value = True
        mock_mod = MagicMock()
        mock_mod.get_wrds_provider.return_value = mock_wrds

        with patch("importlib.import_module", return_value=mock_mod):
            reason = get_wrds_unavailable_reason()
            self.assertIsNone(reason)

    def test_returns_reason_when_wrds_unavailable(self):
        """Should return a descriptive string when WRDS cannot be reached."""
        from quant_engine.data.alternative import get_wrds_unavailable_reason

        with patch("importlib.import_module", side_effect=ImportError("No module 'wrds'")):
            reason = get_wrds_unavailable_reason()
            self.assertIsNotNone(reason)
            self.assertIn("Could not import", reason)

    def test_returns_reason_when_wrds_not_available(self):
        """Should return credential hint when provider loads but is not available."""
        from quant_engine.data.alternative import get_wrds_unavailable_reason

        mock_wrds = MagicMock()
        mock_wrds.available.return_value = False
        mock_mod = MagicMock()
        mock_mod.get_wrds_provider.return_value = mock_wrds

        with patch("importlib.import_module", return_value=mock_mod):
            reason = get_wrds_unavailable_reason()
            self.assertIsNotNone(reason)
            self.assertIn("credentials", reason.lower())


class TestModuleLevelConvenienceFunctions(unittest.TestCase):
    """Verify module-level convenience functions work correctly."""

    def setUp(self):
        import quant_engine.data.alternative as mod
        mod._wrds_provider = None
        mod._wrds_import_error = None
        mod._default_provider = None

    def tearDown(self):
        import quant_engine.data.alternative as mod
        mod._wrds_provider = None
        mod._wrds_import_error = None
        mod._default_provider = None

    def test_get_earnings_surprise_returns_none_without_wrds(self):
        """Convenience function should return None without raising."""
        from quant_engine.data.alternative import get_earnings_surprise

        with patch("quant_engine.data.alternative._get_wrds", return_value=None):
            result = get_earnings_surprise("AAPL")
            self.assertIsNone(result)

    def test_get_earnings_surprise_delegates_to_provider(self):
        """Convenience function should call through to the provider method."""
        from quant_engine.data.alternative import get_earnings_surprise

        mock_wrds = MagicMock()
        mock_wrds.available.return_value = True
        fake_df = pd.DataFrame({"surprise_pct": [1.0]})
        mock_wrds.get_earnings_surprises.return_value = fake_df

        with patch("quant_engine.data.alternative._get_wrds", return_value=mock_wrds):
            cutoff = datetime(2024, 1, 1)
            get_earnings_surprise("AAPL", as_of_date=cutoff)
            # The WRDS provider method should have been called
            mock_wrds.get_earnings_surprises.assert_called_once()
            call_kwargs = mock_wrds.get_earnings_surprises.call_args.kwargs
            self.assertEqual(call_kwargs["end_date"], "2024-01-01")


class TestWRDSProviderFunctionValidation(unittest.TestCase):
    """Verify that _get_wrds() validates the provider function exists."""

    def setUp(self):
        import quant_engine.data.alternative as mod
        mod._wrds_provider = None
        mod._wrds_import_error = None

    def tearDown(self):
        import quant_engine.data.alternative as mod
        mod._wrds_provider = None
        mod._wrds_import_error = None

    def test_missing_factory_function_is_caught(self):
        """If the module has no get_wrds_provider, should fail gracefully."""
        from quant_engine.data.alternative import _get_wrds

        # Module exists but has no get_wrds_provider attribute
        mock_mod = MagicMock(spec=[])  # Empty spec → no attributes
        del mock_mod.get_wrds_provider  # Ensure it doesn't exist

        with patch("importlib.import_module", return_value=mock_mod):
            result = _get_wrds()
            self.assertIsNone(result)


class TestExistingTestsStillPass(unittest.TestCase):
    """Smoke test: verify that the existing B02 test patterns still work
    after the B03 error handling changes."""

    def test_provider_with_injected_wrds_still_works(self):
        """AlternativeDataProvider with an injected mock _wrds should work
        exactly as before for existing tests."""
        from quant_engine.data.alternative import AlternativeDataProvider

        mock_wrds = MagicMock()
        mock_wrds.available.return_value = True

        # Simulate the test_alternative_lookahead.py pattern:
        # create via __new__ and inject _wrds directly
        provider = AlternativeDataProvider.__new__(AlternativeDataProvider)
        provider.cache_dir = None
        provider.strict = False
        provider._wrds = mock_wrds

        # Should not raise, should delegate to mock
        mock_wrds.get_earnings_surprises.return_value = pd.DataFrame()
        result = provider.get_earnings_surprise("AAPL", lookback_days=90)
        self.assertIsNone(result)  # Empty DataFrame → returns None internally


if __name__ == "__main__":
    unittest.main()
