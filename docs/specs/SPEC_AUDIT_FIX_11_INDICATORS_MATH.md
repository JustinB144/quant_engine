# SPEC_AUDIT_FIX_11: Indicators Mathematical Correctness & Causality Fixes

**Priority:** CRITICAL — Aroon inversion flips signal interpretation; PivotHigh/Low look-ahead leaks into backtests.
**Scope:** `indicators/` — `indicators.py`, `tail_risk.py`
**Estimated effort:** 3 hours
**Depends on:** Nothing
**Blocks:** Nothing

---

## Context

The indicators module has two P0 math errors (Aroon oscillator is mathematically inverted, PivotHigh/PivotLow use centered rolling windows that look into the future), plus P1 issues with expected shortfall off-by-one, RegimePersistence NaN treatment, and dead alias code.

---

## Tasks

### T1: Fix Aroon Oscillator Inversion

**Problem:** `indicators.py:459,463` computes Aroon Up as `((period - np.argmax(x)) / period) * 100`. When the most recent bar IS the high, `np.argmax` returns the last index and `period - last_index` gives a low value. Standard Aroon Up should return 100 when the most recent bar is the high.

**File:** `indicators/indicators.py`

**Implementation:**
1. Fix the formula to standard Aroon:
   ```python
   # Standard Aroon: 100 * (period - periods_since_highest) / period
   # np.argmax on a window returns index of max from start of window
   # periods_since_highest = period - 1 - np.argmax(x)  (distance from end)
   aroon_up = lambda x: ((np.argmax(x)) / (len(x) - 1)) * 100
   aroon_down = lambda x: ((np.argmin(x)) / (len(x) - 1)) * 100
   ```
   Or more precisely:
   ```python
   aroon_up = lambda x: ((period - (period - 1 - np.argmax(x))) / period) * 100
   ```
   Simplified: `aroon_up = (np.argmax(x) / (period - 1)) * 100`
2. Verify against a reference implementation (TA-Lib or textbook).
3. Add unit test: for a series where the most recent bar is the highest, Aroon Up = 100.

**Acceptance:** Aroon Up = 100 when the most recent bar is the period high. Aroon Down = 100 when the most recent bar is the period low.

---

### T2: Fix PivotHigh/PivotLow Look-Ahead Bias

**Problem:** `indicators.py:1031,1067` use `rolling(window=pivot_window, center=True)` which makes the center bar look at future bars to the right. This is non-causal and introduces look-ahead bias.

**File:** `indicators/indicators.py`

**Implementation:**
1. Remove `center=True` and use a trailing window:
   ```python
   # OLD: .rolling(window=pivot_window, center=True)
   # NEW: Use trailing window (causal)
   highs = df['High'].rolling(window=pivot_window, center=False)
   pivot_high = (df['High'] == highs.max())  # Current bar is highest in trailing window
   ```
2. Document the semantic change: "PivotHigh now uses a trailing window. A pivot is identified when the current bar is the highest in the last N bars (no future data used)."
3. Update `FEATURE_METADATA` for PivotHigh/PivotLow from whatever type they are to `"CAUSAL"`.
4. Note: this changes signal timing. Pivots will be detected later than with centered windows. This is the correct behavior for trading.

**Acceptance:** PivotHigh/PivotLow compute values using only past and current data. No `center=True` in any rolling call.

---

### T3: Fix Expected Shortfall Off-By-One

**Problem:** `tail_risk.py:105,107` uses `np.partition(seg, k)[:k]` where `k = ceil(alpha * n)`. This can include a non-tail element and misses the kth element.

**File:** `indicators/tail_risk.py`

**Implementation:**
1. Fix the partition:
   ```python
   k = max(1, int(np.floor(alpha * len(seg))))  # floor, not ceil
   if k >= len(seg):
       k = len(seg) - 1
   worst_k = np.partition(seg, k)[:k]  # k smallest values (worst returns)
   es = float(np.mean(worst_k))
   ```
2. Alternatively, use `np.sort` which is clearer:
   ```python
   sorted_returns = np.sort(seg)  # ascending
   k = max(1, int(np.floor(alpha * len(seg))))
   es = float(np.mean(sorted_returns[:k]))
   ```
3. Add unit test: ES at 5% for `[-10, -5, -3, -1, 0, 1, 2, 3, 5, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]` (20 values) should be the mean of the worst 1 value = -10.

**Acceptance:** Expected shortfall at alpha correctly averages the worst floor(alpha × n) returns.

---

### T4: Fix RegimePersistence NaN-to-Zero Treatment

**Problem:** `indicators.py:2719,2728` converts state to int (NaN→0), then checks `np.isnan()` on the int (which never fires). NaN regime states are silently treated as regime 0.

**File:** `indicators/indicators.py`

**Implementation:**
1. Check for NaN BEFORE int conversion:
   ```python
   current_state = values[i]
   if np.isnan(current_state):
       persistence_counts.append(0)  # or np.nan
       continue
   current_state = int(current_state)
   ```
2. Alternatively, use `pd.isna()` which handles float NaN:
   ```python
   if pd.isna(current_state):
       continue
   ```

**Acceptance:** NaN regime states produce NaN/0 persistence, not false matches with regime 0.

---

### T5: Wire INDICATOR_ALIASES into create_indicator()

**Problem:** `indicators.py:2885,2899` defines `INDICATOR_ALIASES` but `create_indicator()` only checks `get_all_indicators()` and ignores aliases entirely.

**File:** `indicators/indicators.py`

**Implementation:**
1. In `create_indicator()`, check aliases before raising error:
   ```python
   def create_indicator(name: str, **kwargs):
       registry = get_all_indicators()
       if name in registry:
           return registry[name](**kwargs)
       # Check aliases
       canonical = INDICATOR_ALIASES.get(name)
       if canonical and canonical in registry:
           return registry[canonical](**kwargs)
       raise ValueError(f"Unknown indicator: {name}")
   ```

**Acceptance:** `create_indicator("RSI")` works if `INDICATOR_ALIASES["RSI"] = "RSI_14"` exists.

---

### T6: Sanitize Inf/NaN Output From Indicators

**Problem:** Multiple indicators can emit `inf`/`-inf` from division by zero (lines 102, 258, 300, 900, 1212, 1661). This propagates through feature engineering and model training.

**File:** `indicators/indicators.py`

**Implementation:**
1. Add a decorator or post-processing step to all indicator `compute()` methods:
   ```python
   def _sanitize_output(result: pd.DataFrame) -> pd.DataFrame:
       return result.replace([np.inf, -np.inf], np.nan)
   ```
2. Apply in the base class or in the pipeline's indicator computation loop.
3. Log when inf values are replaced for observability.

**Acceptance:** No indicator output contains inf or -inf values. All inf values are replaced with NaN.

---

## Verification

- [ ] Run `pytest tests/ -k "indicator"` — all pass
- [ ] Verify Aroon Up=100 when latest bar is period high
- [ ] Verify PivotHigh uses trailing window (no center=True)
- [ ] Verify ES at 5% for known data matches expected value
- [ ] Verify NaN regime states don't count as regime 0
- [ ] Verify no inf values in any indicator output
