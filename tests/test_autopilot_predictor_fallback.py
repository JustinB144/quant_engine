"""
Test module for autopilot predictor fallback behavior and regressions.
"""

import unittest
from unittest.mock import patch

import pandas as pd

from quant_engine.autopilot.engine import AutopilotEngine, HeuristicPredictor


class AutopilotPredictorFallbackTests(unittest.TestCase):
    """Test cases covering autopilot predictor fallback behavior and system invariants."""
    def test_ensure_predictor_falls_back_when_model_import_fails(self):
        idx = pd.date_range("2024-01-01", periods=12, freq="B")
        data = {
            "AAPL": pd.DataFrame(
                {
                    "Open": [100 + i for i in range(len(idx))],
                    "High": [101 + i for i in range(len(idx))],
                    "Low": [99 + i for i in range(len(idx))],
                    "Close": [100 + i for i in range(len(idx))],
                    "Volume": [1_000_000.0] * len(idx),
                },
                index=idx,
            ),
        }
        engine = AutopilotEngine(tickers=["AAPL"], verbose=False)

        with patch(
            "quant_engine.autopilot.engine.EnsemblePredictor",
            side_effect=ModuleNotFoundError("No module named 'sklearn'"),
        ):
            predictor = engine._ensure_predictor(data)

        self.assertIsInstance(predictor, HeuristicPredictor)

        feats = pd.DataFrame(
            {
                "return_20d": [0.0] * len(idx),
                "return_5d": [0.0] * len(idx),
                "PriceVsSMA_50": [0.0] * len(idx),
                "MACD_12_26": [0.0] * len(idx),
                "ZScore_20": [0.0] * len(idx),
            },
            index=idx,
        )
        regimes = pd.Series([0] * len(idx), index=idx)
        regime_conf = pd.Series([0.7] * len(idx), index=idx)
        preds = predictor.predict(feats, regimes, regime_conf)
        self.assertEqual(len(preds), len(idx))
        self.assertIn("predicted_return", preds.columns)
        self.assertIn("confidence", preds.columns)


if __name__ == "__main__":
    unittest.main()
