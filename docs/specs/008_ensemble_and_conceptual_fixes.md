# Feature Spec: Fix Conceptual Issues — Ensemble Bug, Hardcoded Defaults, Missing Endpoints

> **Status:** Approved
> **Author:** justin
> **Date:** 2026-02-23
> **Estimated effort:** ~6 hours across 5 tasks

---

## Why

Several conceptual issues undermine system integrity: (1) The regime ensemble uses a duplicate of rule-based detection as the "jump model" when REGIME_JUMP_MODEL_ENABLED=False, making the 3-method voting actually 2-method with a phantom third vote. (2) Frontend pages hardcode config defaults (BacktestPage, ModelLabPage) instead of fetching from `/api/config`. (3) The Heston IV surface tab is completely broken (button disabled, endpoint never called). (4) No `/api/regime` dedicated endpoint exists despite regime detection being a core system. (5) The rule-based detector ignores REGIME_MIN_DURATION, producing 1-bar regime flickers.

## What

Fix all conceptual bugs that cause the system to behave differently from what it advertises. Done means: ensemble voting genuinely uses 3 independent methods, UI defaults match backend config, broken features are either fixed or clearly marked as "Coming Soon", and rule-based detection enforces minimum duration.

## Constraints

### Must-haves
- Ensemble voting: if jump model unavailable, use 2-method voting with threshold=2 (require unanimity), NOT fake 3rd vote
- Frontend defaults fetched from /api/config on page load
- Heston tab either connected to working endpoint or shows "Coming Soon" with explanation
- Rule-based detection enforces REGIME_MIN_DURATION

### Must-nots
- Do NOT change ensemble weights or thresholds without A/B backtest validation
- Do NOT remove the Heston tab entirely (it's partially implemented)

## Tasks

### T1: Fix ensemble duplicate placeholder bug

**What:** When REGIME_JUMP_MODEL_ENABLED=False, ensemble should use 2-method voting instead of faking a 3rd method.

**Files:**
- `regime/detector.py` — fix `detect_ensemble()` (lines 253-314)

**Implementation notes:**
- Current bug:
  ```python
  if REGIME_JUMP_MODEL_ENABLED:
      jump_out = self._jump_detect(features)
  else:
      jump_out = rule_out  # BUG: duplicates rule vote
  ```
- Fix:
  ```python
  methods = [("rule", rule_out), ("hmm", hmm_out)]
  if REGIME_JUMP_MODEL_ENABLED:
      methods.append(("jump", self._jump_detect(features)))

  n_methods = len(methods)
  threshold = min(REGIME_ENSEMBLE_CONSENSUS_THRESHOLD, n_methods)

  # Majority vote across available methods only
  for t in range(T):
      votes = [m[1].regime.iloc[t] for _, m in methods]
      regime_counts = Counter(votes)
      winner = regime_counts.most_common(1)[0]
      if winner[1] >= threshold:
          regime_vals[t] = winner[0]
      else:
          regime_vals[t] = hmm_out.regime.iloc[t]  # HMM tiebreaker

  # Probability blending: equal weights across available methods
  weight = 1.0 / n_methods
  blended_probs = sum(weight * m[1].probabilities for _, m in methods)
  ```

**Verify:**
```bash
python -c "
from quant_engine.regime.detector import RegimeDetector
from quant_engine.config import REGIME_JUMP_MODEL_ENABLED
print(f'Jump model enabled: {REGIME_JUMP_MODEL_ENABLED}')
# Ensemble should work correctly regardless of jump model status
"
```

---

### T2: Fetch frontend defaults from /api/config

**What:** BacktestPage and ModelLabPage should load their default config values from the backend instead of hardcoding them.

**Files:**
- `frontend/src/pages/BacktestPage.tsx` — fetch defaults from /api/config
- `frontend/src/pages/ModelLabPage.tsx` — fetch defaults from /api/config (TrainingTab)
- `frontend/src/api/endpoints.ts` — verify /config endpoint exists

**Implementation notes:**
- On page mount, fetch `/api/config` and populate form defaults:
  ```tsx
  const { data: config } = useQuery({ queryKey: ["config"], queryFn: () => api.get("/config") });
  const defaults = {
    horizon: config?.backtest?.default_horizon ?? 10,
    holding_period: config?.backtest?.holding_period ?? 10,
    entry_threshold: config?.entry_threshold ?? 0.01,
    max_positions: config?.max_positions ?? 10,
    position_size: config?.position_size ?? 0.1,
  };
  ```
- Show "Using server defaults" indicator when defaults are loaded
- If /config fetch fails, fall back to current hardcoded values (don't break the page)

**Verify:**
- Manual: Open Backtest page, verify config values match backend config.py values

---

### T3: Fix or properly disable Heston IV Surface tab

**What:** Either connect the Heston surface to a working backend endpoint or show a clear "Coming Soon" state with explanation.

**Files:**
- `frontend/src/pages/IVSurfacePage.tsx` — HestonSurfaceTab section

**Implementation notes:**
- Option A (if Heston computation exists in backend):
  - Connect to `/iv-surface/heston` endpoint
  - Enable the "Compute Surface" button
  - Show loading state during computation
- Option B (if Heston endpoint doesn't exist yet):
  - Replace empty state with:
    ```
    🚧 Heston Surface — Coming Soon

    The Heston stochastic volatility model requires server-side computation
    that is not yet connected to the API. The model code exists in
    models/iv/ but needs an API endpoint.

    What's implemented:
    ✅ SVI parametric surface (fully functional)
    ✅ Arbitrage-free SVI (fully functional)
    🔲 Heston model computation endpoint
    ```
  - Remove the disabled "Compute Surface" button (it's confusing)
  - Keep the Feller condition display (educational value)

**Verify:**
- Manual: Navigate to IV Surface → Heston tab. Should show clear status instead of empty/broken state.

---

### T4: Add duration smoothing to rule-based regime detection

**What:** Apply REGIME_MIN_DURATION enforcement to `_rule_detect()` output, matching HMM behavior.

**Files:**
- `regime/detector.py` — modify `_rule_detect()` method

**Implementation notes:**
- After computing regime labels (line ~107), apply duration smoothing:
  ```python
  # Apply minimum duration enforcement (same as HMM's _smooth_duration)
  if self.min_duration > 1:
      regime = self._apply_min_duration(regime, confidence)
  ```
- Create `_apply_min_duration(regime_series, confidence_series)` helper that:
  1. Finds runs shorter than `min_duration`
  2. Merges short runs with the adjacent regime that has higher confidence
  3. Same logic as HMM's `_smooth_duration()` but operating on pd.Series instead of np.ndarray
- This ensures rule-based detection doesn't produce 1-bar regime flickers that could trigger false regime change signals in retrain_trigger

**Verify:**
```bash
python -c "
from quant_engine.regime.detector import RegimeDetector
import pandas as pd, numpy as np
idx = pd.date_range('2020-01-01', periods=600, freq='B')
np.random.seed(42)
features = pd.DataFrame({
    'Close': np.cumsum(np.random.randn(600)*0.02)+100,
    'High': np.cumsum(np.random.randn(600)*0.02)+101,
    'Low': np.cumsum(np.random.randn(600)*0.02)+99,
    'Open': np.cumsum(np.random.randn(600)*0.02)+100,
    'Volume': np.random.randint(1e6, 1e7, 600).astype(float),
    'ret_1d': np.random.randn(600)*0.02,
    'vol_20d': 0.02 + np.random.rand(600)*0.01,
    'natr': 0.02 + np.random.rand(600)*0.01,
    'sma_slope': np.random.randn(600)*0.001,
    'hurst': 0.5 + np.random.randn(600)*0.1,
    'adx': 20 + np.random.randn(600)*5,
}, index=idx)
detector = RegimeDetector(method='rule', min_duration=3)
out = detector.detect_full(features)
# Check no regime run is shorter than 3 bars
runs = out.regime.ne(out.regime.shift()).cumsum()
run_lengths = out.regime.groupby(runs).count()
short_runs = (run_lengths < 3).sum()
print(f'Short runs (< 3 bars): {short_runs} (should be 0 or very few)')
"
```

---

### T5: Test all conceptual fixes

**What:** Tests verifying ensemble voting, config defaults, and duration smoothing work correctly.

**Files:**
- `tests/test_conceptual_fixes.py` — new test file

**Implementation notes:**
- Test cases:
  1. `test_ensemble_no_phantom_vote` — With JUMP_MODEL_ENABLED=False, ensemble uses 2 methods, not 3
  2. `test_ensemble_with_jump_model` — With JUMP_MODEL_ENABLED=True, ensemble uses 3 independent methods
  3. `test_rule_detect_min_duration` — No runs shorter than REGIME_MIN_DURATION in rule-based output
  4. `test_config_endpoint_has_defaults` — /api/config returns backtest defaults
  5. `test_heston_tab_state` — Heston tab shows appropriate status (not blank/broken)

**Verify:**
```bash
python -m pytest tests/test_conceptual_fixes.py -v
```

---

## Validation

### Acceptance criteria
1. Ensemble with 2 methods requires 2/2 agreement (not 2/3 with phantom vote)
2. Ensemble with 3 methods requires 2/3 agreement (genuine voting)
3. BacktestPage defaults match config.py values
4. Heston tab shows clear status (not broken/empty)
5. Rule-based detection produces no runs shorter than min_duration

### Rollback plan
- Ensemble fix: revert detect_ensemble() to previous code (voting logic change only)
- Frontend defaults: fall back to hardcoded values if /config fetch fails
- Duration smoothing: remove _apply_min_duration call (additive change)
