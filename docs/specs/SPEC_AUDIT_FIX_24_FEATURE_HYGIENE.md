# SPEC_AUDIT_FIX_24: Feature Engineering Code Hygiene & Test Coverage

**Priority:** LOW/MEDIUM — Causality tests too sparse to catch pivot leaks; duplicate indicator; dead code; zero dedicated indicator unit tests.
**Scope:** `tests/test_lookahead_detection.py`, `indicators/indicators.py`, `features/version.py`, `indicators/__init__.py`, `indicators/spectral.py`, `indicators/ssa.py`, `indicators/tail_risk.py`, `indicators/ot_divergence.py`, `indicators/eigenvalue.py`
**Estimated effort:** 3 hours
**Depends on:** SPEC_21 T1 (pivot fix must land first)
**Blocks:** Nothing

---

## Context

The feature engineering subsystem has six hygiene/test issues. F-08 (P2): lookahead CI tests check only boundary rows and miss `center=True` rolling window leaks. F-10 (P3): `features/version.py` (168 lines) is dead code with zero importers. F-11 (P3): `VolumeRatio` and `RVOL` are functionally identical (both compute `Volume / rolling_mean(Volume, 20)`). F-12 (P3): 5 structural analyzer files import pandas but never use it. F-13 (P3): 5 indicator classes exported by `__init__.py` but never consumed by pipeline. F-14 (P3): `INDICATOR_ALIASES` dict at indicators.py:2885-2896 is never consulted by `create_indicator()`. F-15 (P3): zero dedicated unit tests for 91 indicator classes.

---

## Tasks

### T1: Expand Lookahead CI Tests With Prefix-Replay and center=True Detection

**Problem:** `tests/test_lookahead_detection.py` has 2 test methods: `test_no_feature_uses_future_data()` (checks single boundary row) and `test_no_feature_uses_future_data_deep()` (checks rows at depths 1, 5, 10, 20). Neither detects the `center=True` rolling window leak in PivotHigh/PivotLow because the difference only manifests in specific bar alignments.

**File:** `tests/test_lookahead_detection.py`

**Implementation:**
1. Add a prefix-replay test that verifies every feature value at a truncation boundary matches the value computed from a longer series:
   ```python
   def test_prefix_replay_all_features():
       """Every feature at time t must be identical whether computed on
       data[:t+1] (prefix) or data[:t+N] (extended).

       This catches center=True rolling windows, forward-fill from future
       data, and any other look-ahead leak.
       """
       # Use synthetic data to avoid flaky external dependencies
       full_data = _generate_synthetic_ohlcv(bars=300)
       full_features = pipeline.compute(full_data)

       # Check multiple truncation points
       for cut_point in [100, 150, 200, 250]:
           prefix_data = full_data.iloc[:cut_point].copy()
           prefix_features = pipeline.compute(prefix_data)

           # Features at the last bar of the prefix must match
           check_idx = prefix_features.index[-1]
           for col in prefix_features.columns:
               prefix_val = prefix_features.loc[check_idx, col]
               full_val = full_features.loc[check_idx, col]
               if pd.notna(prefix_val) and pd.notna(full_val):
                   assert prefix_val == pytest.approx(full_val, nan_ok=True, rel=1e-10), (
                       f"Look-ahead detected in '{col}' at bar {cut_point}: "
                       f"prefix={prefix_val}, full={full_val}"
                   )
   ```
2. Add a specific test for `center=True` usage:
   ```python
   def test_no_center_true_in_indicators():
       """Verify no indicator uses center=True rolling windows."""
       import ast, inspect
       from indicators.indicators import Indicator
       for cls in Indicator.__subclasses__():
           source = inspect.getsource(cls)
           tree = ast.parse(source)
           for node in ast.walk(tree):
               if (isinstance(node, ast.keyword) and
                   node.arg == "center" and
                   isinstance(node.value, ast.Constant) and
                   node.value.value is True):
                   pytest.fail(
                       f"{cls.__name__} uses center=True in rolling() — "
                       f"this creates look-ahead bias"
                   )
   ```

**Acceptance:** The prefix-replay test passes for all features after SPEC_21 T1 fixes PivotHigh/PivotLow. The `center=True` AST check catches any future regressions.

---

### T2: Remove Duplicate RVOL Indicator

**Problem:** `VolumeRatio` (indicators.py:475-490) and `RVOL` (indicators.py:808-827) compute identically: `Volume / rolling_mean(Volume, period)` with `period=20`. Both are instantiated by the pipeline, producing redundant `VolRatio_20` and `RVOL_20` columns.

**File:** `indicators/indicators.py`, `features/pipeline.py`

**Implementation:**
1. Remove the `RVOL` class definition from indicators.py.
2. Update any references to `RVOL` to use `VolumeRatio` instead.
3. If `RVOL_20` is consumed downstream (check predictor, backtest, UI), add a feature alias:
   ```python
   # In pipeline after computing indicators:
   if "VolRatio_20" in features.columns and "RVOL_20" not in features.columns:
       features["RVOL_20"] = features["VolRatio_20"]
   ```
4. Remove `RVOL` from the pipeline's indicator instantiation list.
5. Remove `RVOL_20` from `FEATURE_METADATA` if present, or mark as alias.

**Acceptance:** Only one volume ratio indicator exists. No redundant column computation.

---

### T3: Remove Dead Code

**Problem:** Multiple dead code artifacts:
- `features/version.py` (168 lines): `FeatureVersion` and `FeatureRegistry` classes with zero importers.
- `indicators/indicators.py:2885-2896`: `INDICATOR_ALIASES` dict never consulted by `create_indicator()`.
- 5 analyzer files: `import pandas as pd` unused (spectral.py:12, ssa.py:12, tail_risk.py:12, ot_divergence.py:13, eigenvalue.py:16).

**Files:** `features/version.py`, `indicators/indicators.py`, `indicators/spectral.py`, `indicators/ssa.py`, `indicators/tail_risk.py`, `indicators/ot_divergence.py`, `indicators/eigenvalue.py`

**Implementation:**
1. Delete `features/version.py` entirely (confirmed zero importers via grep).
2. Remove `INDICATOR_ALIASES` dict from indicators.py:2885-2896 and any reference to it.
3. Remove `import pandas as pd` from the 5 analyzer files listed above.
4. Verify no other file references these deleted items (grep for `version.FeatureVersion`, `version.FeatureRegistry`, `INDICATOR_ALIASES`).

**Acceptance:** `ruff --select F` (unused imports/variables) produces zero findings for these files. `features/version.py` no longer exists. `INDICATOR_ALIASES` no longer exists.

---

### T4: Document Unexported Indicator Classes

**Problem:** `indicators/__init__.py` exports 5 classes not consumed by pipeline: `AnchoredVWAP`, `PriceVsAnchoredVWAP`, `MultiVWAPPosition`, `Beast666Proximity`, `Beast666Distance`. Their intended use is unclear.

**File:** `indicators/__init__.py`

**Implementation:**
1. Add a comment block documenting these as available for external/research use:
   ```python
   # The following indicators are available for external consumers and
   # research notebooks but are NOT included in the production feature
   # pipeline. To add them, instantiate in pipeline.py's indicator list.
   __RESEARCH_INDICATORS__ = [
       "AnchoredVWAP", "PriceVsAnchoredVWAP", "MultiVWAPPosition",
       "Beast666Proximity", "Beast666Distance",
   ]
   ```
2. Keep them exported (they may have external consumers), but make their status explicit.

**Acceptance:** The 5 research-only indicators are clearly documented as non-production.

---

### T5: Add Foundational Indicator Unit Tests

**Problem:** 91 concrete indicator classes have zero dedicated unit tests (`tests/test_indicator*` returns no matches). Indicators are only tested indirectly through feature pipeline integration tests.

**File:** `tests/test_indicators.py` (new file)

**Implementation:**
1. Create a parameterized test that validates basic properties for all indicator classes:
   ```python
   import pytest
   import pandas as pd
   import numpy as np
   from indicators.indicators import get_all_indicators

   _SAMPLE_DATA = pd.DataFrame({
       "Open": np.random.uniform(100, 110, 300),
       "High": np.random.uniform(105, 115, 300),
       "Low": np.random.uniform(95, 105, 300),
       "Close": np.random.uniform(100, 110, 300),
       "Volume": np.random.uniform(1e6, 5e6, 300),
   }, index=pd.bdate_range("2024-01-01", periods=300))
   # Fix OHLC consistency
   _SAMPLE_DATA["High"] = _SAMPLE_DATA[["Open", "High", "Close"]].max(axis=1)
   _SAMPLE_DATA["Low"] = _SAMPLE_DATA[["Open", "Low", "Close"]].min(axis=1)

   @pytest.fixture
   def sample_data():
       return _SAMPLE_DATA.copy()

   @pytest.mark.parametrize("name,cls", list(get_all_indicators().items()))
   def test_indicator_returns_series(name, cls, sample_data):
       """Every indicator must return a pd.Series with matching index."""
       indicator = cls()
       result = indicator.calculate(sample_data)
       assert isinstance(result, pd.Series), f"{name} returned {type(result)}"
       assert len(result) == len(sample_data), f"{name} length mismatch"

   @pytest.mark.parametrize("name,cls", list(get_all_indicators().items()))
   def test_indicator_no_inf(name, cls, sample_data):
       """No indicator should produce inf values on valid OHLCV data."""
       indicator = cls()
       result = indicator.calculate(sample_data)
       assert not np.any(np.isinf(result.values[np.isfinite(result.values)])), (
           f"{name} produced inf values"
       )

   def test_indicator_flat_bar_no_inf():
       """Flat-bar input (H==L==O==C) must not produce inf."""
       flat = _SAMPLE_DATA.copy()
       flat["High"] = flat["Low"] = flat["Open"] = flat["Close"] = 100.0
       for name, cls in get_all_indicators().items():
           indicator = cls()
           result = indicator.calculate(flat)
           assert not np.any(np.isinf(result.dropna().values)), (
               f"{name} produced inf on flat bars"
           )
   ```
2. These tests validate the foundational contract: every indicator returns a `pd.Series`, matches input length, and produces no infinities on valid data.

**Acceptance:** All 91 indicator classes pass the parameterized test suite. Flat-bar input produces no inf values (may produce NaN, which is acceptable).

---

## Verification

- [ ] Run `pytest tests/test_lookahead_detection.py` — prefix-replay and center=True tests pass
- [ ] Run `pytest tests/test_indicators.py` — all 91 indicators pass basic property tests
- [ ] Verify `ruff --select F` produces zero findings for cleaned files
- [ ] Verify `features/version.py` is deleted
- [ ] Verify `RVOL` class is removed and no downstream breakage
