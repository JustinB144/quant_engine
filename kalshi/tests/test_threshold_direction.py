"""
Threshold direction correctness test (Instructions I.2).

Verifies _resolve_threshold_direction() and _resolve_threshold_direction_with_confidence()
correctly identify GE vs LE semantics from explicit metadata, rules text, and guesses.
"""
import unittest

from quant_engine.kalshi.distribution import (
    _resolve_threshold_direction,
    _resolve_threshold_direction_with_confidence,
)


class ThresholdDirectionTests(unittest.TestCase):
    """Tests for threshold direction resolution."""

    # ── Explicit metadata (high confidence) ──

    def test_explicit_ge_direction(self):
        result = _resolve_threshold_direction_with_confidence({"direction": "ge"})
        self.assertEqual(result.direction, "ge")
        self.assertEqual(result.source, "explicit_metadata")
        self.assertEqual(result.confidence, "high")

    def test_explicit_le_direction(self):
        result = _resolve_threshold_direction_with_confidence({"direction": "le"})
        self.assertEqual(result.direction, "le")
        self.assertEqual(result.source, "explicit_metadata")
        self.assertEqual(result.confidence, "high")

    def test_explicit_gte_alias(self):
        result = _resolve_threshold_direction_with_confidence({"direction": "gte"})
        self.assertEqual(result.direction, "ge")
        self.assertEqual(result.confidence, "high")

    def test_explicit_lte_alias(self):
        result = _resolve_threshold_direction_with_confidence({"direction": "lte"})
        self.assertEqual(result.direction, "le")
        self.assertEqual(result.confidence, "high")

    def test_explicit_ge_symbol(self):
        result = _resolve_threshold_direction_with_confidence({"direction": ">="})
        self.assertEqual(result.direction, "ge")

    def test_explicit_le_symbol(self):
        result = _resolve_threshold_direction_with_confidence({"direction": "<="})
        self.assertEqual(result.direction, "le")

    def test_payout_structure_above(self):
        result = _resolve_threshold_direction_with_confidence({"payout_structure": "above"})
        self.assertEqual(result.direction, "ge")
        self.assertEqual(result.source, "explicit_metadata")
        self.assertEqual(result.confidence, "high")

    def test_payout_structure_below(self):
        result = _resolve_threshold_direction_with_confidence({"payout_structure": "below"})
        self.assertEqual(result.direction, "le")
        self.assertEqual(result.source, "explicit_metadata")
        self.assertEqual(result.confidence, "high")

    # ── Rules text (medium confidence) ──

    def test_rules_text_greater_than(self):
        row = {"rules_text": "Pays if CPI is greater than or equal to 3.5%"}
        result = _resolve_threshold_direction_with_confidence(row)
        self.assertEqual(result.direction, "ge")
        self.assertEqual(result.source, "rules_text")
        self.assertEqual(result.confidence, "medium")

    def test_rules_text_less_than(self):
        row = {"rules_text": "Pays if unemployment is at most 4.0%"}
        result = _resolve_threshold_direction_with_confidence(row)
        self.assertEqual(result.direction, "le")
        self.assertEqual(result.source, "rules_text")
        self.assertEqual(result.confidence, "medium")

    def test_rules_text_above(self):
        row = {"rules_text": "Resolves Yes if value is above 250"}
        result = _resolve_threshold_direction_with_confidence(row)
        self.assertEqual(result.direction, "ge")
        self.assertEqual(result.source, "rules_text")

    def test_rules_text_below(self):
        row = {"rules_text": "Settles based on whether the rate is below 5%"}
        result = _resolve_threshold_direction_with_confidence(row)
        self.assertEqual(result.direction, "le")
        self.assertEqual(result.source, "rules_text")

    # ── Title/subtitle guess (low confidence) ──

    def test_title_guess_or_higher(self):
        row = {"title": "CPI 3.5% or higher"}
        result = _resolve_threshold_direction_with_confidence(row)
        self.assertEqual(result.direction, "ge")
        self.assertEqual(result.source, "guess")
        self.assertEqual(result.confidence, "low")

    def test_title_guess_or_lower(self):
        row = {"title": "Unemployment 4.0% or lower"}
        result = _resolve_threshold_direction_with_confidence(row)
        self.assertEqual(result.direction, "le")
        self.assertEqual(result.source, "guess")
        self.assertEqual(result.confidence, "low")

    # ── Unknown / no signal ──

    def test_no_direction_signal(self):
        result = _resolve_threshold_direction_with_confidence({"title": "some market"})
        self.assertIsNone(result.direction)
        self.assertEqual(result.confidence, "low")

    def test_empty_row(self):
        result = _resolve_threshold_direction_with_confidence({})
        self.assertIsNone(result.direction)

    # ── Legacy wrapper ──

    def test_legacy_resolve_returns_string(self):
        self.assertEqual(_resolve_threshold_direction({"direction": "ge"}), "ge")
        self.assertEqual(_resolve_threshold_direction({"direction": "le"}), "le")
        self.assertIsNone(_resolve_threshold_direction({}))


if __name__ == "__main__":
    unittest.main()
