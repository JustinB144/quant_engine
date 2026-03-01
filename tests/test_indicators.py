"""Foundational unit tests for all indicator classes.

SPEC_AUDIT_FIX_24 T5: Validates the basic contract for every indicator:
  - Returns a pd.Series with matching index length
  - Produces no inf values on valid OHLCV data
  - Produces no inf values on flat-bar input (H==L==O==C)
"""

import numpy as np
import pandas as pd
import pytest

from indicators.indicators import get_all_indicators

_SAMPLE_DATA = pd.DataFrame({
    "Open": np.random.default_rng(42).uniform(100, 110, 300),
    "High": np.random.default_rng(42).uniform(105, 115, 300),
    "Low": np.random.default_rng(42).uniform(95, 105, 300),
    "Close": np.random.default_rng(42).uniform(100, 110, 300),
    "Volume": np.random.default_rng(42).uniform(1e6, 5e6, 300),
}, index=pd.bdate_range("2024-01-01", periods=300))
# Fix OHLC consistency
_SAMPLE_DATA["High"] = _SAMPLE_DATA[["Open", "High", "Close"]].max(axis=1)
_SAMPLE_DATA["Low"] = _SAMPLE_DATA[["Open", "Low", "Close"]].min(axis=1)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    return _SAMPLE_DATA.copy()


@pytest.mark.parametrize("name,cls", list(get_all_indicators().items()))
def test_indicator_returns_series(name: str, cls: type, sample_data: pd.DataFrame) -> None:
    """Every indicator must return a pd.Series with matching index."""
    indicator = cls()
    result = indicator.calculate(sample_data)
    assert isinstance(result, pd.Series), f"{name} returned {type(result)}"
    assert len(result) == len(sample_data), f"{name} length mismatch"


@pytest.mark.parametrize("name,cls", list(get_all_indicators().items()))
def test_indicator_no_inf(name: str, cls: type, sample_data: pd.DataFrame) -> None:
    """No indicator should produce inf values on valid OHLCV data."""
    indicator = cls()
    result = indicator.calculate(sample_data)
    finite_mask = np.isfinite(result.values) | np.isnan(result.values)
    assert np.all(finite_mask), (
        f"{name} produced inf values"
    )


def test_indicator_flat_bar_no_inf() -> None:
    """Flat-bar input (H==L==O==C) must not produce inf."""
    flat = _SAMPLE_DATA.copy()
    flat["High"] = flat["Low"] = flat["Open"] = flat["Close"] = 100.0
    for name, cls in get_all_indicators().items():
        indicator = cls()
        result = indicator.calculate(flat)
        assert not np.any(np.isinf(result.dropna().values)), (
            f"{name} produced inf on flat bars"
        )
