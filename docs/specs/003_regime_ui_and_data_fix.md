# Feature Spec: Fix Regime State Tab — Show Real Data & Explain Regimes

> **Status:** Approved
> **Author:** justin
> **Date:** 2026-02-23
> **Estimated effort:** ~6 hours across 5 tasks

---

## Why

The Regime State tab in the dashboard doesn't reliably show real data. The `compute_regime_payload()` function crashes with "boolean value of NA is ambiguous" when probabilities contain NaN values (partially fixed but root cause persists in HMM observation matrix edge cases). Additionally, the UI shows regime names ("Trending Bull", "Mean Reverting") without explaining what these regimes mean, how they're detected, or why the user should care. The transition matrix is displayed but users don't know how to interpret it.

## What

A fully functional Regime State tab that: always shows real regime data (never crashes), explains what each regime means and how it's detected, shows regime history timeline, explains the transition matrix, and provides actionable context ("In high-volatility regime: position sizes are reduced to 60%, stops are widened to 1.5x").

## Constraints

### Must-haves
- Regime payload never returns fallback/empty data when real data exists in cache
- NaN values in probabilities are handled at the detector level (not just in the API layer)
- UI explains each regime: definition, detection method, portfolio impact
- Transition matrix has hover tooltips explaining each cell
- Regime history shows a timeline of regime changes with durations

### Must-nots
- Do NOT change the RegimeOutput dataclass interface
- Do NOT modify downstream consumers (they already work)
- Do NOT add new detection methods (that's the jump model spec)

### Out of scope
- Replacing HMM with Jump Model (spec 001)
- Regime-based position sizing changes (already implemented)
- Real-time regime monitoring/alerting

## Current State

### Key files
| File | Role | Notes |
|------|------|-------|
| `api/services/data_helpers.py` | `compute_regime_payload()` — builds regime data for API | Partially fixed NA handling but detector can still produce NaN |
| `regime/detector.py` | `detect_full()` dispatches to detection methods | `_hmm_detect()` normalizes probs but can produce NaN if sum is 0 |
| `regime/hmm.py` | `build_hmm_observation_matrix()` — 11-feature standardization | `.fillna(0.0)` at end but division by zero in standardization possible |
| `frontend/src/pages/DashboardPage.tsx` | Dashboard with Regime tab | Tab fetches /dashboard/regime |
| `config.py` | `REGIME_NAMES`, `REGIME_RISK_MULTIPLIER`, `REGIME_STOP_MULTIPLIER` | Regime metadata exists but not exposed to UI |

## Tasks

### T1: Fix NaN propagation in regime probability pipeline

**What:** Ensure regime probabilities NEVER contain NaN at any stage, from observation matrix through to API output.

**Files:**
- `regime/detector.py` — add NaN guard after probability aggregation
- `regime/hmm.py` — add NaN guard after standardization in `build_hmm_observation_matrix()`

**Implementation notes:**
- In `_hmm_detect()` after line 181 (`probs = probs.div(...)`), add:
  ```python
  probs = probs.fillna(0.0)
  # If any row sums to 0 (all NaN before fillna), assign uniform distribution
  zero_rows = probs.sum(axis=1) == 0
  if zero_rows.any():
      probs.loc[zero_rows] = 0.25
  ```
- In `build_hmm_observation_matrix()`, after standardization, replace any inf/nan:
  ```python
  obs = obs.replace([np.inf, -np.inf], np.nan).fillna(0.0)
  ```
- In `_rule_detect()`, ensure confidence values are never NaN (clip and fillna)
- Same guards in `_jump_detect()`

**Verify:**
```bash
python -c "
import pandas as pd, numpy as np
from quant_engine.regime.detector import RegimeDetector
# Create features with intentional NaN to trigger edge cases
idx = pd.date_range('2020-01-01', periods=600, freq='B')
features = pd.DataFrame({
    'Close': np.cumsum(np.random.randn(600)*0.02)+100,
    'High': np.cumsum(np.random.randn(600)*0.02)+101,
    'Low': np.cumsum(np.random.randn(600)*0.02)+99,
    'Open': np.cumsum(np.random.randn(600)*0.02)+100,
    'Volume': np.random.randint(1e6, 1e7, 600).astype(float),
}, index=idx)
features.loc[features.index[:50], 'Volume'] = np.nan  # inject NaN
features['ret_1d'] = features['Close'].pct_change()
features['vol_20d'] = features['ret_1d'].rolling(20).std()
features['natr'] = (features['High']-features['Low'])/features['Close']
features['sma_slope'] = features['Close'].rolling(20).mean().pct_change(5)
features['hurst'] = 0.5
features['adx'] = 25.0
features = features.iloc[50:]  # keep NaN-free part but with derived NaN
detector = RegimeDetector(method='hmm')
out = detector.detect_full(features)
assert not out.probabilities.isna().any().any(), 'Probabilities contain NaN!'
assert not out.regime.isna().any(), 'Regime contains NaN!'
print('No NaN in regime output')
"
```

---

### T2: Add /api/regime/metadata endpoint

**What:** Create an API endpoint that returns regime definitions, detection methodology, and portfolio impact for the UI to display.

**Files:**
- `api/routers/system_health.py` (or create `api/routers/regime.py`) — add endpoint
- `config.py` — already has REGIME_NAMES, REGIME_RISK_MULTIPLIER, REGIME_STOP_MULTIPLIER

**Implementation notes:**
- Endpoint: `GET /api/regime/metadata`
- Response:
```json
{
  "regimes": {
    "0": {
      "name": "Trending Bull",
      "definition": "Market showing sustained upward momentum with low volatility. Hurst exponent > 0.55 and positive SMA slope.",
      "detection": "Identified when HMM/Jump Model posterior probability exceeds 50% for this state, characterized by positive returns, low NATR, and trending Hurst exponent.",
      "portfolio_impact": {
        "position_size_multiplier": 1.0,
        "stop_loss_multiplier": 1.0,
        "description": "Full position sizes, standard stop losses. Best regime for directional strategies."
      },
      "color": "#22c55e"
    },
    "1": {
      "name": "Trending Bear",
      "definition": "Market showing sustained downward momentum. Hurst exponent > 0.55 and negative SMA slope.",
      "portfolio_impact": {
        "position_size_multiplier": 0.85,
        "stop_loss_multiplier": 0.8,
        "description": "Reduced to 85% position sizes, tighter stops. Defensive posture."
      },
      "color": "#ef4444"
    },
    "2": {
      "name": "Mean Reverting",
      "definition": "Market oscillating without clear trend. Hurst exponent < 0.45.",
      "portfolio_impact": {
        "position_size_multiplier": 0.95,
        "stop_loss_multiplier": 1.2,
        "description": "95% position sizes, wider stops to allow oscillation. Model has historically low Sharpe in this regime."
      },
      "color": "#eab308"
    },
    "3": {
      "name": "High Volatility",
      "definition": "Elevated market volatility regardless of direction. NATR exceeds 80th percentile of rolling 252-day window.",
      "portfolio_impact": {
        "position_size_multiplier": 0.60,
        "stop_loss_multiplier": 1.5,
        "description": "Reduced to 60% position sizes, widest stops. Capital preservation priority."
      },
      "color": "#a855f7"
    }
  },
  "detection_method": "hmm",
  "ensemble_enabled": true,
  "transition_matrix_explanation": "Each cell (i,j) shows the probability of transitioning from regime i to regime j. Diagonal values show regime persistence (higher = more stable). Off-diagonal shows transition likelihood."
}
```

**Verify:**
```bash
# After starting server:
curl -s http://localhost:8000/api/regime/metadata | python -m json.tool
```

---

### T3: Add regime history timeline to detector output

**What:** Add a `regime_changes` list to the regime payload that shows when each regime transition occurred and how long each regime lasted.

**Files:**
- `api/services/data_helpers.py` — enhance `compute_regime_payload()` to include regime change timeline

**Implementation notes:**
- After computing regime series, detect transitions:
  ```python
  changes = []
  prev_regime = None
  start_date = None
  for date, regime_val in out.regime.items():
      if regime_val != prev_regime:
          if prev_regime is not None:
              changes.append({
                  "from_regime": REGIME_NAMES.get(prev_regime, f"Regime {prev_regime}"),
                  "to_regime": REGIME_NAMES.get(regime_val, f"Regime {regime_val}"),
                  "date": date.strftime("%Y-%m-%d"),
                  "duration_days": (date - start_date).days,
              })
          start_date = date
          prev_regime = regime_val
  ```
- Add to return dict: `"regime_changes": changes[-20:]` (last 20 transitions)
- Add: `"current_regime_duration_days": (history.index[-1] - start_date).days`

**Verify:**
```bash
python -c "
from quant_engine.api.services.data_helpers import compute_regime_payload
from quant_engine.config import DATA_CACHE_DIR
result = compute_regime_payload(DATA_CACHE_DIR)
print('Changes:', len(result.get('regime_changes', [])))
print('Current duration:', result.get('current_regime_duration_days'))
"
```

---

### T4: Update frontend Regime tab with explanations and timeline

**What:** Redesign the Regime tab to show regime explanations, portfolio impact, transition timeline, and interactive transition matrix.

**Files:**
- `frontend/src/pages/DashboardPage.tsx` — update Regime tab content
- `frontend/src/components/charts/HeatmapChart.tsx` — add tooltip support for transition matrix cells

**Implementation notes:**
- Fetch `/api/regime/metadata` on page load (cache it — changes rarely)
- Regime State section:
  - Current regime badge with definition text below it
  - Portfolio impact card: "Position sizes at 60%, stops at 1.5x" (from metadata)
  - Duration: "In this regime for 12 days"
- Probability chart: existing stacked area, keep as-is
- Transition matrix: add tooltips on each cell: "23% chance of transitioning from Trending Bull to High Volatility"
- Regime history timeline: horizontal timeline showing colored blocks for each regime period (last 240 days)
  - Each block width proportional to duration
  - Hover shows date range and duration

**Verify:**
- Manual: Navigate to Dashboard → Regime tab. Verify explanations shown, timeline visible, matrix has tooltips.

---

### T5: Test regime data pipeline end-to-end

**What:** Integration tests verifying regime data flows from detector through API to UI-consumable format without NaN or errors.

**Files:**
- `tests/test_regime_payload.py` — new test file

**Implementation notes:**
- Test cases:
  1. `test_regime_payload_no_nan` — compute_regime_payload returns valid data with no NaN in probs
  2. `test_regime_payload_with_sparse_data` — Works with minimal cache (single parquet)
  3. `test_regime_changes_timeline` — Timeline has correct transitions
  4. `test_regime_metadata_endpoint` — /api/regime/metadata returns all 4 regimes with required fields
  5. `test_regime_fallback_on_missing_cache` — Returns graceful fallback when no cache exists

**Verify:**
```bash
python -m pytest tests/test_regime_payload.py -v
```

---

## Validation

### Acceptance criteria
1. Regime tab shows real data (not fallback) when cache parquets exist
2. No "boolean value of NA is ambiguous" warning in logs
3. Each regime has visible definition, detection method, and portfolio impact
4. Transition matrix cells have hover tooltips
5. Regime history timeline shows last 240 days of regime blocks
6. Probabilities sum to 1.0 (within floating point tolerance) on every row

### Rollback plan
- NaN fixes are purely additive guards — safe to keep even if UI changes revert
- Regime metadata endpoint is new — remove route if needed
- Frontend changes are additive — old Regime tab structure still works without metadata
