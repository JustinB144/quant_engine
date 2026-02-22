"""
Test module for iv arbitrage builder behavior and regressions.
"""

import unittest

import numpy as np

from quant_engine.models.iv.models import ArbitrageFreeSVIBuilder, generate_synthetic_market_surface


class ArbitrageFreeSVIBuilderTests(unittest.TestCase):
    """Test cases covering iv arbitrage builder behavior and system invariants."""
    def test_build_surface_has_valid_shape_and_monotone_total_variance(self):
        market = generate_synthetic_market_surface(S=100.0, r=0.05, q=0.01)
        builder = ArbitrageFreeSVIBuilder(max_iter=120)

        out = builder.build_surface(
            spot=market["S"],
            strikes=market["strikes"],
            expiries=market["expiries"],
            market_iv_grid=market["iv_grid"],
            r=market["r"],
            q=market["q"],
        )

        self.assertEqual(out["adj_iv_grid"].shape, market["iv_grid"].shape)
        self.assertEqual(out["raw_total_variance"].shape, market["iv_grid"].shape)
        self.assertTrue(np.all(np.isfinite(out["adj_iv_grid"])))
        self.assertGreater(float(np.min(out["adj_iv_grid"])), 0.0)

        # Calendar no-arbitrage: total variance should be non-decreasing in T.
        cal_diff = np.diff(out["adj_total_variance"], axis=0)
        self.assertFalse(np.any(cal_diff < -1e-10))


if __name__ == "__main__":
    unittest.main()
