"""
Test module for delisting total return behavior and regressions.
"""

import unittest

import numpy as np
import pandas as pd

from quant_engine.features.pipeline import compute_indicator_features, compute_targets


class DelistingTotalReturnTests(unittest.TestCase):
    """Test cases covering delisting total return behavior and system invariants."""
    def test_target_uses_total_return_when_available(self):
        idx = pd.date_range("2024-01-01", periods=6, freq="B")
        close = pd.Series([100.0, 101.0, 102.0, 96.9, 96.9, 96.9], index=idx)
        ret = close.pct_change().fillna(0.0)
        total_ret = ret.copy()
        # Delisting-aware total return differs from plain return on this bar.
        total_ret.iloc[3] = -0.43

        df = pd.DataFrame(
            {
                "Open": close,
                "High": close,
                "Low": close,
                "Close": close,
                "Volume": 1_000_000.0,
                "Return": ret,
                "total_ret": total_ret,
                "delist_event": [0, 0, 0, 1, 0, 0],
            },
            index=idx,
        )

        targets = compute_targets(df, horizons=[1])
        # target_1d at t uses return stream at t+1.
        self.assertAlmostEqual(float(targets["target_1d"].iloc[2]), -0.43, places=6)
        self.assertNotAlmostEqual(float(targets["target_1d"].iloc[2]), float(ret.iloc[3]), places=6)

    def test_indicator_values_unaffected_by_delist_return_columns(self):
        rng = np.random.default_rng(7)
        idx = pd.date_range("2023-01-01", periods=320, freq="B")
        close = pd.Series(100 + np.cumsum(rng.normal(0, 1, len(idx))), index=idx)
        open_ = close.shift(1).fillna(close.iloc[0])
        high = np.maximum(open_, close) * 1.01
        low = np.minimum(open_, close) * 0.99
        volume = pd.Series(1_000_000 + rng.integers(-20_000, 20_000, len(idx)), index=idx)

        base = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=idx,
        )
        with_event = base.copy()
        with_event.loc[:, "Return"] = with_event["Close"].pct_change()
        with_event.loc[:, "total_ret"] = with_event["Return"]
        with_event.iloc[250, with_event.columns.get_loc("total_ret")] = -0.95
        with_event["dlret"] = np.nan
        with_event.iloc[250, with_event.columns.get_loc("dlret")] = -0.95
        with_event["delist_event"] = 0
        with_event.iloc[250, with_event.columns.get_loc("delist_event")] = 1

        f_base = compute_indicator_features(base, verbose=False)
        f_event = compute_indicator_features(with_event, verbose=False)

        self.assertEqual(list(f_base.index), list(f_event.index))
        common_cols = [c for c in f_base.columns if c in f_event.columns]
        for col in common_cols:
            a = pd.to_numeric(f_base[col], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(f_event[col], errors="coerce").to_numpy(dtype=float)
            self.assertTrue(np.allclose(a, b, equal_nan=True), msg=f"indicator mismatch: {col}")


if __name__ == "__main__":
    unittest.main()
