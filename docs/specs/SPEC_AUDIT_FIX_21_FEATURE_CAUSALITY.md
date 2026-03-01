# SPEC_AUDIT_FIX_21: Feature Causality & Metadata Correctness

**Priority:** CRITICAL — Two pivot indicators leak 5 future bars into training and live inference; 16 features bypass explicit causality typing; END_OF_DAY features pass through predictor gate unchecked.
**Scope:** `indicators/indicators.py`, `features/pipeline.py`, `models/predictor.py`
**Estimated effort:** 3 hours
**Depends on:** Nothing
**Blocks:** Nothing

---

## Context

The feature engineering subsystem has three causality enforcement gaps. First (F-01/P0), `PivotHigh` and `PivotLow` use `rolling(window=11, center=True)` which requires 5 future bars to classify a pivot at the current timestamp — these are classified as `CAUSAL` in `FEATURE_METADATA` (pipeline.py:149-150) and pass through the predictor causality gate because the gate only blocks `RESEARCH_ONLY`. Second (F-02/P1), 16 features emitted by macro, HARX, and correlation modules are absent from `FEATURE_METADATA`, defaulting to `CAUSAL` via `get_feature_type()` (pipeline.py:414-427) — all 16 happen to be genuinely causal today, but the default-to-CAUSAL policy is a latent risk. Third (F-05/P2), 13 `END_OF_DAY` features (requiring same-day close data) pass through the predictor gate because it only checks for `RESEARCH_ONLY`, creating a latent causality violation if the system is extended to intraday prediction.

---

## Tasks

### T1: Replace PivotHigh/PivotLow center=True With Strictly Causal Formulation

**Problem:** `indicators/indicators.py:1031` (`PivotHigh`) and `:1067` (`PivotLow`) use:
```python
pivots = high.rolling(window=pivot_window, center=True).apply(
    lambda x: x[left] if len(x) == pivot_window and
              x[left] == x.max() else np.nan,
    raw=True
)
```
With `left_bars=5, right_bars=5`, `pivot_window=11`, and `center=True`, the rolling window is centered on the current bar, meaning bars at positions `[t-5, ..., t, ..., t+5]` are examined. The `x[left]` accesses `x[5]` (the center/current bar) and checks if it is the maximum. But the window includes 5 **future** bars (`t+1` through `t+5`), creating a look-ahead leak.

**File:** `indicators/indicators.py`

**Implementation:**
1. Replace `center=True` with a strictly causal (left-only) formulation:
   ```python
   class PivotHigh(Indicator):
       def calculate(self, df: pd.DataFrame) -> pd.Series:
           high = df['High']
           # Strictly causal: only look back `left_bars + right_bars` bars.
           # A pivot high is confirmed when the bar at position `left_bars`
           # from the end of the window is the maximum of the entire window.
           # This means we wait `right_bars` bars after the candidate pivot
           # before confirming it — using only past data.
           pivot_window = self.left_bars + self.right_bars + 1

           pivots = high.rolling(window=pivot_window, min_periods=pivot_window).apply(
               lambda x: x[self.left_bars] if x[self.left_bars] == x.max() else np.nan,
               raw=True,
           )

           # Shift result forward by right_bars to align the pivot signal
           # with the bar where confirmation occurs (i.e., right_bars after
           # the actual pivot), NOT the pivot bar itself.
           # No shift needed — the rolling window already places the result
           # at the last bar of the window (the confirmation bar).

           pivot_high_level = pivots.ffill()
           breakout = (high > pivot_high_level).astype(int)
           return breakout
   ```
2. Apply the same fix to `PivotLow`:
   ```python
   pivots = low.rolling(window=pivot_window, min_periods=pivot_window).apply(
       lambda x: x[self.left_bars] if x[self.left_bars] == x.min() else np.nan,
       raw=True,
   )
   ```
3. The key insight: with `center=False` (default), a rolling window of 11 bars covers `[t-10, ..., t]`. The candidate pivot is at `x[left_bars]` = `x[5]`, which is `t-5`. By the time the window reaches `t`, we have seen 5 bars after the candidate — confirming the pivot using only past data.
4. Add a regression test that computes PivotHigh on truncated vs full data and verifies identical values at the truncation boundary.

**Acceptance:** `PivotHigh` and `PivotLow` produce identical output whether computed on `data[:t]` or `data[:t+100]` for all `t`. No `center=True` remains in any indicator.

---

### T2: Register All 16 Missing Features in FEATURE_METADATA

**Problem:** `get_feature_type()` at pipeline.py:414-427 defaults unknown feature names to `"CAUSAL"`. 16 features emitted by macro, HARX, and correlation modules have no entries in `FEATURE_METADATA`.

**File:** `features/pipeline.py`

**Implementation:**
1. Add all 16 features to `FEATURE_METADATA` with their verified causality types:
   ```python
   # Correlation regime features (from regime/correlation.py)
   "avg_pairwise_corr": {"type": "CAUSAL", "category": "correlation"},
   "corr_regime": {"type": "CAUSAL", "category": "correlation"},
   "corr_z_score": {"type": "CAUSAL", "category": "correlation"},

   # HARX spillover features (from features/harx_spillovers.py)
   "harx_spillover_from": {"type": "CAUSAL", "category": "spillover"},
   "harx_spillover_to": {"type": "CAUSAL", "category": "spillover"},
   "harx_net_spillover": {"type": "CAUSAL", "category": "spillover"},

   # Macro features (from features/macro.py)
   "macro_vix": {"type": "CAUSAL", "category": "macro"},
   "macro_vix_mom": {"type": "CAUSAL", "category": "macro"},
   "macro_term_spread": {"type": "CAUSAL", "category": "macro"},
   "macro_term_spread_mom": {"type": "CAUSAL", "category": "macro"},
   "macro_credit_spread": {"type": "CAUSAL", "category": "macro"},
   "macro_credit_spread_mom": {"type": "CAUSAL", "category": "macro"},
   "macro_initial_claims": {"type": "CAUSAL", "category": "macro"},
   "macro_initial_claims_mom": {"type": "CAUSAL", "category": "macro"},
   "macro_consumer_sentiment": {"type": "CAUSAL", "category": "macro"},
   "macro_consumer_sentiment_mom": {"type": "CAUSAL", "category": "macro"},
   ```
2. Change the default in `get_feature_type()` from `"CAUSAL"` to a safe fail-closed behavior:
   ```python
   def get_feature_type(feature_name: str) -> str:
       entry = FEATURE_METADATA.get(feature_name)
       if entry is not None:
           return entry["type"]
       if feature_name.startswith("X_"):
           return "CAUSAL"
       # CHANGED: unknown features default to RESEARCH_ONLY (fail-closed)
       # to prevent unregistered features from reaching production models.
       import logging
       logging.getLogger(__name__).warning(
           "Feature '%s' not in FEATURE_METADATA — defaulting to RESEARCH_ONLY",
           feature_name,
       )
       return "RESEARCH_ONLY"
   ```
3. This ensures any future unregistered feature is blocked from production rather than silently passing.

**Acceptance:** All 16 features have explicit `FEATURE_METADATA` entries. An unregistered feature name returns `"RESEARCH_ONLY"` with a warning log, not `"CAUSAL"`.

---

### T3: Add END_OF_DAY to Predictor Causality Gate

**Problem:** `models/predictor.py:213-227` only blocks `RESEARCH_ONLY` features. 13 `END_OF_DAY` features (intraday vol ratio, VWAP deviation, microstructure metrics) require same-day close data. If the system is extended to intraday prediction, these become forward-looking.

**File:** `models/predictor.py`

**Implementation:**
1. Add `END_OF_DAY` awareness to the causality gate:
   ```python
   if TRUTH_LAYER_ENFORCE_CAUSALITY:
       research_only = {
           col for col in features.columns
           if get_feature_type(col) == "RESEARCH_ONLY"
       }
       if research_only:
           raise ValueError(
               f"RESEARCH_ONLY features in prediction: {sorted(research_only)}. "
               f"Set TRUTH_LAYER_ENFORCE_CAUSALITY=False to override."
           )

       # NEW: warn about END_OF_DAY features (safe for daily, unsafe for intraday)
       end_of_day = {
           col for col in features.columns
           if get_feature_type(col) == "END_OF_DAY"
       }
       if end_of_day:
           import logging
           logging.getLogger(__name__).info(
               "Prediction includes %d END_OF_DAY features: %s. "
               "These require same-day close data and are only valid for "
               "end-of-day predictions.",
               len(end_of_day), sorted(end_of_day)[:5],
           )
   ```
2. Add a config constant `PREDICTION_MODE = "daily"` (default). When set to `"intraday"`, block END_OF_DAY features:
   ```python
   from ..config import PREDICTION_MODE
   if PREDICTION_MODE == "intraday" and end_of_day:
       raise ValueError(
           f"END_OF_DAY features in intraday prediction: {sorted(end_of_day)}. "
           f"Remove these features or switch to daily mode."
       )
   ```

**Acceptance:** In daily mode, END_OF_DAY features pass through with an INFO log. In intraday mode, they are blocked with a ValueError. RESEARCH_ONLY features remain blocked in all modes.

---

## Verification

- [ ] Run `pytest tests/ -k "lookahead or causality or predictor"` — all pass
- [ ] Verify PivotHigh/PivotLow produce identical results on truncated vs full data
- [ ] Verify all 16 features have FEATURE_METADATA entries
- [ ] Verify an unregistered feature name defaults to RESEARCH_ONLY with warning
- [ ] Verify END_OF_DAY features are logged in daily mode and blocked in intraday mode
