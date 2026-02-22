"""
Test module for covariance estimator behavior and regressions.
"""

import unittest

import numpy as np
import pandas as pd

from quant_engine.risk.covariance import CovarianceEstimator


class CovarianceEstimatorTests(unittest.TestCase):
    """Test cases covering covariance estimator behavior and system invariants."""
    def test_single_asset_covariance_is_2d_and_positive(self):
        returns = pd.DataFrame({"AAPL": np.array([0.01, -0.005, 0.002, 0.007, -0.001], dtype=float)})
        estimator = CovarianceEstimator(method="sample", shrinkage=0.2)
        estimate = estimator.estimate(returns)

        self.assertEqual(estimate.covariance.shape, (1, 1))
        self.assertGreater(float(estimate.covariance.iloc[0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
