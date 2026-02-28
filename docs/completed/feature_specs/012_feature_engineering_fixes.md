# Feature Spec: Feature Engineering — Fix Lookahead Bias, Leakage, and Missing Features

> **Status:** Approved
> **Author:** justin
> **Date:** 2026-02-23
> **Estimated effort:** ~10 hours across 7 tasks

---

## Why

The feature pipeline has several critical lookahead bias issues: (1) `research_factors.py` TSMom features use `shift(-h)` which looks FORWARD in time — `close.shift(-5)` is the price 5 bars in the future. (2) `options_factors.py` IV shock features (`delta_atm_iv_pre_post`, `skew_steepening`, `term_structure_kink`) all use `shift(-window)` to compare pre-event vs post-event, but post-event data is future data at prediction time. (3) `intraday.py` VWAP deviation uses full-day VWAP which isn't available intraday. (4) Feature selection in `trainer.py` uses permutation importance on validation data, but if the same validation set is reused across multiple selection rounds, this creates selection bias. (5) The pipeline doesn't separate causal (tradeable) features from non-causal (research-only) features, so it's easy to accidentally include forward-looking features in production.

## What

Fix all lookahead bias in feature computation, add a causal/non-causal feature tagging system, and add automated lookahead detection. Done means: no feature uses future data, all features are tagged as causal or research-only, and an automated test catches any new lookahead bias.

## Constraints

### Must-haves
- TSMom features corrected to use `shift(h)` (backward-looking lag, not forward)
- IV shock features redesigned to be causal (pre-event only)
- Feature tagging system: CAUSAL vs RESEARCH_ONLY
- Automated lookahead detection test in CI
- VWAP features marked as end-of-day only (not for intraday prediction)

### Must-nots
- Do NOT remove research features entirely (they're useful for analysis)
- Do NOT change the feature pipeline interface
- Do NOT break existing model artifacts (they'll be retrained)

## Tasks

### T1: Fix TSMom lookahead bias in research_factors.py

**What:** Change `shift(-h)` to proper backward-looking momentum calculation.

**Files:**
- `features/research_factors.py` — fix `compute_time_series_momentum_factors()`

**Implementation notes:**
- Current bug (uses future data):
  ```python
  mom_h = close.shift(-h) / close - 1  # WRONG: looks h bars into future
  ```
- Fix (backward-looking):
  ```python
  mom_h = close / close.shift(h) - 1  # RIGHT: compares current to h bars ago
  ```
- Also fix `TSMom_12m1m`:
  ```python
  # Current (WRONG): close.shift(21) / close.shift(252) - 1
  # This is actually backward-looking (correct) if using positive shift values
  # Verify: shift(21) = 21 bars ago, shift(252) = 252 bars ago
  # Result: price 21 days ago / price 252 days ago - 1 (12m-1m momentum)
  # This is CORRECT as written — it's the TSMom_h features that are wrong
  ```
- Rename for clarity: `TSMom_{h}` → `TSMom_lag{h}` to indicate these are lagged features
- Add `_feature_type` metadata: `{'TSMom_lag5': 'CAUSAL', ...}`

**Verify:**
```bash
python -c "
import pandas as pd, numpy as np
# Demonstrate the difference
close = pd.Series([100, 101, 102, 103, 104, 105], name='Close')
wrong = close.shift(-2) / close - 1  # Future: uses prices 2 bars ahead
right = close / close.shift(2) - 1  # Correct: compares to 2 bars ago
print('Wrong (lookahead):', wrong.values)
print('Right (causal):', right.values)
print('At index 0, wrong uses future price 102/100, right has NaN (no history)')
"
```

---

### T2: Fix IV shock features lookahead bias in options_factors.py

**What:** Redesign IV shock features to be causal (only look backward from current time).

**Files:**
- `features/options_factors.py` — fix `compute_iv_shock_features()`

**Implementation notes:**
- Current (uses future data):
  ```python
  delta_atm_iv_pre_post = iv30.shift(-window) - iv30.shift(window)  # Future - Past
  ```
- Fix (causal only):
  ```python
  # Measure recent IV change: compare current to W bars ago
  delta_atm_iv_change = iv30 - iv30.shift(window)         # Current - Past
  delta_atm_iv_velocity = delta_atm_iv_change - delta_atm_iv_change.shift(window)  # Acceleration
  
  # Skew change (not pre/post event, just recent change)
  skew_change = skew - skew.shift(window)
  
  # Term structure change
  term_ratio = iv30 / iv90 if iv90 is available
  term_change = term_ratio - term_ratio.shift(window)
  ```
- Rename features to reflect causality:
  - `delta_atm_iv_pre_post` → `delta_atm_iv_change_{w}d`
  - `skew_steepening` → `skew_change_{w}d`
  - `term_structure_kink` → `term_structure_change_{w}d`

**Verify:**
```bash
python -c "
print('IV shock features redesigned:')
print('  OLD: iv30.shift(-W) - iv30.shift(W) = uses future data')
print('  NEW: iv30 - iv30.shift(W) = compares current to past')
print('  No future data used in any IV feature')
"
```

---

### T3: Tag all features as CAUSAL or RESEARCH_ONLY

**What:** Add metadata to every feature indicating whether it's safe for production use.

**Files:**
- `features/pipeline.py` — add feature metadata system
- `features/research_factors.py` — tag all features
- `features/options_factors.py` — tag all features

**Implementation notes:**
- Add a feature registry:
  ```python
  FEATURE_METADATA = {
      # Core indicators (all causal)
      'rsi_14': {'type': 'CAUSAL', 'category': 'momentum', 'lookback': 14},
      'macd': {'type': 'CAUSAL', 'category': 'momentum', 'lookback': 26},
      
      # Research factors
      'TSMom_lag5': {'type': 'CAUSAL', 'category': 'momentum', 'lookback': 5},
      'vwap_deviation': {'type': 'END_OF_DAY', 'category': 'microstructure', 'lookback': 1},
      
      # Marked as research-only until fixed
      'relative_mom_5': {'type': 'RESEARCH_ONLY', 'category': 'momentum', 'reason': 'Cross-sectional adjustment needed at prediction time'},
  }
  ```
- In pipeline.py `compute()`:
  ```python
  if self.production_mode:
      # Filter out non-causal features
      causal_cols = [c for c in features.columns if FEATURE_METADATA.get(c, {}).get('type') == 'CAUSAL']
      features = features[causal_cols]
  ```
- Add `production_mode` parameter to FeaturePipeline constructor (default=False for backward compat)

**Verify:**
```bash
python -c "
from quant_engine.features.pipeline import FEATURE_METADATA
causal = sum(1 for v in FEATURE_METADATA.values() if v.get('type') == 'CAUSAL')
research = sum(1 for v in FEATURE_METADATA.values() if v.get('type') == 'RESEARCH_ONLY')
print(f'Features: {causal} causal, {research} research-only, {len(FEATURE_METADATA)} total')
"
```

---

### T4: Add automated lookahead bias detection test

**What:** Create a test that automatically detects if any feature uses future data.

**Files:**
- `tests/test_lookahead_detection.py` — new test file

**Implementation notes:**
- Test methodology:
  ```python
  def test_no_feature_uses_future_data():
      """
      Strategy: compute features on full data, then compute on truncated data.
      If truncating the LAST bar changes ANY feature value at bar N-2, that feature
      uses future data (the last bar shouldn't affect historical values).
      """
      import pandas as pd, numpy as np
      from quant_engine.features.pipeline import FeaturePipeline
      
      # Create synthetic data
      np.random.seed(42)
      n = 300
      df = pd.DataFrame({
          'Open': 100 + np.cumsum(np.random.randn(n) * 0.5),
          'High': 0, 'Low': 0, 'Close': 0, 'Volume': 1e6,
      }, index=pd.date_range('2020-01-01', periods=n, freq='B'))
      df['Close'] = df['Open'] + np.random.randn(n) * 0.5
      df['High'] = df[['Open', 'Close']].max(axis=1) + abs(np.random.randn(n)) * 0.3
      df['Low'] = df[['Open', 'Close']].min(axis=1) - abs(np.random.randn(n)) * 0.3
      
      pipe = FeaturePipeline(feature_mode='full')
      
      # Compute features on full data
      features_full, _ = pipe.compute(df, compute_targets_flag=False)
      
      # Compute features on data minus last bar
      features_truncated, _ = pipe.compute(df.iloc[:-1], compute_targets_flag=False)
      
      # Compare at the common last row (second-to-last bar of full data)
      check_idx = features_truncated.index[-1]
      if check_idx in features_full.index:
          full_row = features_full.loc[check_idx]
          trunc_row = features_truncated.loc[check_idx]
          
          # Find features that changed (lookahead detected)
          diff = (full_row - trunc_row).abs()
          lookahead_features = diff[diff > 1e-10].index.tolist()
          
          assert len(lookahead_features) == 0, (
              f"Lookahead bias detected in {len(lookahead_features)} features: "
              f"{lookahead_features[:10]}"
          )
  ```

**Verify:**
```bash
python -m pytest tests/test_lookahead_detection.py -v
```

---

### T5: Fix VWAP and intraday features for production use

**What:** Mark VWAP-based features as end-of-day only and add rolling VWAP alternative.

**Files:**
- `features/intraday.py` — add rolling VWAP, tag existing VWAP

**Implementation notes:**
- Current: `vwap_deviation = (close - vwap) / vwap` uses full-day VWAP
- Problem: full-day VWAP includes future intraday data if used for intraday prediction
- Fix for intraday use: rolling VWAP using only past data
  ```python
  def compute_rolling_vwap(df, window=20):
      """Causal VWAP: rolling volume-weighted average price over past W bars."""
      typical_price = (df['High'] + df['Low'] + df['Close']) / 3
      vwap = (typical_price * df['Volume']).rolling(window).sum() / df['Volume'].rolling(window).sum()
      return vwap
  ```
- Tag `vwap_deviation` as `END_OF_DAY` in feature metadata
- Add `rolling_vwap_deviation` as `CAUSAL` alternative

**Verify:**
```bash
python -c "
print('VWAP features:')
print('  vwap_deviation: END_OF_DAY (uses full-day volume)')
print('  rolling_vwap_deviation: CAUSAL (uses past W bars only)')
"
```

---

### T6: Add missing feature categories

**What:** Add commonly used quantitative features that are missing from the pipeline.

**Files:**
- `features/pipeline.py` — add new feature computation functions
- `features/research_factors.py` — add new research factors

**Implementation notes:**
- Missing features identified in audit:
  1. **Earnings-related**: days_to_earnings, post_earnings_drift, earnings_surprise_z
     - Requires earnings calendar data (can use placeholder until data available)
  2. **Sector rotation**: relative sector momentum, sector breadth
     - Compute from cross-asset data (requires GICS_SECTORS)
  3. **Liquidity provision signals**: bid-ask bounce, order flow imbalance persistence
     - Partial implementation exists in lob_features.py, extend for daily data
  4. **Realized-implied vol spread**: already partially implemented as vrp_30, enhance
  5. **Cross-sectional features**: z-scored momentum relative to universe, percentile rank
     - Computed in pipeline.py `compute_universe()`, not single-stock
- Implementation priority: features 4 and 5 are easiest (data already available)

**Verify:**
```bash
python -c "
print('New features to add:')
print('  1. Cross-sectional z-scored momentum (within compute_universe)')
print('  2. Enhanced VRP: vrp at multiple horizons (10d, 30d, 60d)')
print('  3. Sector rotation signals (when GICS_SECTORS populated)')
print('  4. Earnings dummies (when earnings calendar available)')
"
```

---

### T7: Test feature engineering fixes

**What:** Tests verifying lookahead bias is fixed and new features are causal.

**Files:**
- `tests/test_feature_fixes.py` — new test file

**Implementation notes:**
- Test cases:
  1. `test_tsmom_uses_backward_shift` — TSMom features use shift(h), not shift(-h)
  2. `test_iv_shock_no_future` — IV shock features only use shift(w), not shift(-w)
  3. `test_feature_metadata_exists` — All features in pipeline have metadata entry
  4. `test_production_mode_filters_research` — production_mode=True excludes RESEARCH_ONLY
  5. `test_rolling_vwap_is_causal` — Rolling VWAP at bar N doesn't change when bar N+1 is added
  6. `test_automated_lookahead_detection` — Full pipeline passes lookahead test

**Verify:**
```bash
python -m pytest tests/test_feature_fixes.py -v
```

---

## Validation

### Acceptance criteria
1. No feature in pipeline uses `shift(-h)` for negative h (forward shift)
2. All features have CAUSAL, END_OF_DAY, or RESEARCH_ONLY metadata tag
3. Production mode filters out non-causal features automatically
4. Automated lookahead detection test passes on full pipeline
5. TSMom features renamed to `TSMom_lag{h}` for clarity

### Rollback plan
- Feature renames: add column aliases for backward compatibility
- Metadata system: additive, doesn't affect computation
- Production mode filter: defaults to False (backward compatible)
