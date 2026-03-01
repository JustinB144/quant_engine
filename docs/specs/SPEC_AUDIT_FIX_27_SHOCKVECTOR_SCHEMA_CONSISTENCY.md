# SPEC_AUDIT_FIX_27: ShockVector Schema & Construction Consistency

**Priority:** HIGH — `from_dict` breaks documented forward compatibility; two construction paths produce structurally incompatible ShockVectors; validator has coverage gaps.
**Scope:** `regime/shock_vector.py`, `regime/detector.py`
**Estimated effort:** 3 hours
**Depends on:** Nothing
**Blocks:** Nothing

---

## Context

Four schema and construction issues were identified in `ShockVector`. First (F-01/P1), `ShockVector.from_dict()` at shock_vector.py:128-133 calls `cls(**d)` without filtering unknown keys — if a V2-serialized dict contains fields not in the V1 dataclass, `TypeError` is raised, directly contradicting the forward-compatibility promise at shock_vector.py:7-8 ("V1 consumers can safely ignore fields added in V2+"). Second (F-02/P1), the two ShockVector construction paths produce entirely disjoint structural features: `compute_shock_vectors()` at shock_vector.py:469-472 produces `{drift_score, systemic_stress}`, while `detect_with_shock_context()` at detector.py:778-790 produces `{spectral_entropy, ssa_trend_strength, jump_intensity, eigenvalue_concentration}`. The backtest engine (engine.py:446-447, 579-580) accesses only `drift_score` and `systemic_stress` via `.get()`. Third (F-04/P2), `compute_shock_vectors()` at shock_vector.py:487 hardcodes `ensemble_model_type="hmm"` for all bars, while `detect_with_shock_context()` at detector.py:805 correctly uses `regime_out.model_type`. Fourth (F-10/P3), `ShockVectorValidator.validate()` at shock_vector.py:176-281 validates 10 of 13 fields but omits type checks for `timestamp`, `jump_detected`, and `jump_magnitude`.

---

## Tasks

### T1: Add Unknown-Key Filtering to ShockVector.from_dict for Forward Compatibility

**Problem:** `shock_vector.py:128-133` — `from_dict()` does `d = dict(data)`, pops only `transition_matrix`, then calls `cls(**d)`. If the dict contains any key not in the dataclass definition (e.g., a V2 field), `cls(**d)` raises `TypeError: __init__() got an unexpected keyword argument`. This breaks the documented promise at lines 7-8.

**File:** `regime/shock_vector.py`

**Implementation:**
1. Filter to known fields before calling `cls(**d)`:
   ```python
   from dataclasses import fields as dataclass_fields

   @classmethod
   def from_dict(cls, data: Dict) -> "ShockVector":
       d = dict(data)
       ts = d.get("timestamp")
       if isinstance(ts, str):
           d["timestamp"] = datetime.fromisoformat(ts)
       d.pop("transition_matrix", None)
       # Filter to known fields for forward compatibility (V1 ignores V2+ fields)
       known = {f.name for f in dataclass_fields(cls)}
       unknown = set(d.keys()) - known
       if unknown:
           import logging
           logging.getLogger(__name__).debug(
               "ShockVector.from_dict ignoring unknown keys (possible newer schema): %s",
               unknown,
           )
       d = {k: v for k, v in d.items() if k in known}
       return cls(**d)
   ```
2. Add a unit test: construct a dict with extra keys (`{"new_v2_field": 42, ...}`), verify `from_dict` succeeds and ignores the unknown field.

**Acceptance:** `ShockVector.from_dict({"schema_version": "1.0", ..., "future_field": 99})` returns a valid ShockVector without raising `TypeError`. Unknown keys are logged at DEBUG level.

---

### T2: Unify Structural Feature Computation Between Construction Paths

**Problem:** `compute_shock_vectors()` at shock_vector.py:469-472 produces `{drift_score, systemic_stress}` (computed from price, SMA, ATR, rolling vol). `detect_with_shock_context()` at detector.py:778-790 produces `{spectral_entropy, ssa_trend_strength, jump_intensity, eigenvalue_concentration}` (pulled from feature columns). The two sets are entirely disjoint. The backtest engine accesses `drift_score` and `systemic_stress` — if any consumer used `detect_with_shock_context`, those `.get()` calls would silently return `None`.

**Files:** `regime/shock_vector.py`, `regime/detector.py`

**Implementation:**

**Option A (Recommended): Create a shared structural feature builder:**
```python
# In shock_vector.py, add a helper:
def _build_structural_features(
    ohlcv: pd.DataFrame,
    features: Optional[pd.DataFrame],
    bar_idx: int,
    vol_lookback: int = 20,
) -> Dict[str, float]:
    """Build a unified structural feature dict for a single bar.

    Computes drift_score and systemic_stress from OHLCV data.
    If a feature DataFrame is provided, also extracts spectral_entropy,
    ssa_trend_strength, jump_intensity, and eigenvalue_concentration.
    """
    result: Dict[str, float] = {}

    # Price-derived features (always available from OHLCV)
    # ... drift_score computation from existing lines 427-447 ...
    # ... systemic_stress computation from existing lines 449-461 ...

    # Feature-derived features (available when feature DataFrame is provided)
    if features is not None:
        _feature_cols = {
            "spectral_entropy": "SpectralEntropy_252",
            "ssa_trend_strength": "SSATrendStr_60",
            "jump_intensity": "JumpIntensity_20",
            "eigenvalue_concentration": "EigenConcentration_60",
        }
        for key, col in _feature_cols.items():
            if col in features.columns:
                val = features[col].iloc[bar_idx]
                if np.isfinite(val):
                    result[key] = float(val)

    return result
```

Both `compute_shock_vectors()` and `detect_with_shock_context()` call this helper, ensuring every ShockVector has the same structural feature schema (with missing features as absent keys, not different keys).

**Option B: Add drift_score and systemic_stress to detect_with_shock_context:**

Compute them from the OHLCV data that `detect_with_shock_context` already receives, matching the batch path.

**Acceptance:** Both `compute_shock_vectors()` and `detect_with_shock_context()` produce ShockVectors with the same structural feature key set. A test constructs ShockVectors via both paths and asserts `set(sv1.structural_features.keys()) == set(sv2.structural_features.keys())`.

---

### T3: Pass Actual Model Type to compute_shock_vectors

**Problem:** `shock_vector.py:487` hardcodes `ensemble_model_type="hmm"` in every ShockVector. `detect_with_shock_context()` at detector.py:805 correctly uses `regime_out.model_type`. When the system uses ensemble or jump detection, the batch ShockVectors incorrectly claim "hmm".

**File:** `regime/shock_vector.py`

**Implementation:**
1. Add an `ensemble_model_type` parameter to `compute_shock_vectors()`:
   ```python
   def compute_shock_vectors(
       ohlcv: pd.DataFrame,
       regime_series: Optional[pd.Series] = None,
       regime_confidence_series: Optional[pd.Series] = None,
       ticker: str = "",
       vol_lookback: int = 20,
       ensemble_model_type: str = "hmm",  # NEW — caller provides actual type
   ) -> Dict[pd.Timestamp, ShockVector]:
   ```
2. Use the parameter at line 487 instead of the hardcoded string:
   ```python
   sv = ShockVector(
       ...
       ensemble_model_type=ensemble_model_type,  # Was hardcoded "hmm"
       ...
   )
   ```
3. Update callers to pass the correct model type:
   - `backtest/engine.py:875-882`: Extract model type from the regime detection output and pass it through.
   - `run_backtest.py`: If the regime detection output is available, pass `regime_out.model_type`.

**Acceptance:** When using ensemble detection, `compute_shock_vectors` produces ShockVectors with `ensemble_model_type="ensemble"`, not `"hmm"`. A test verifies the field reflects the actual model type.

---

### T4: Complete ShockVectorValidator Field Coverage

**Problem:** `ShockVectorValidator.validate()` at shock_vector.py:176-281 validates 10 of 13 fields but omits:
- `timestamp` — no type check (could be `"not_a_datetime"`)
- `jump_detected` — no type check (could be `42` instead of `bool`)
- `jump_magnitude` — no type/range check

**File:** `regime/shock_vector.py`

**Implementation:**
Add validation blocks after the existing checks (before the return statement):
```python
# Timestamp type check
if not isinstance(sv.timestamp, datetime):
    errors.append(
        f"timestamp must be datetime, got {type(sv.timestamp).__name__}"
    )

# Jump detected type check
if not isinstance(sv.jump_detected, bool):
    errors.append(
        f"jump_detected must be bool, got {type(sv.jump_detected).__name__}: {sv.jump_detected}"
    )

# Jump magnitude type and range check
if not isinstance(sv.jump_magnitude, (int, float)):
    errors.append(
        f"jump_magnitude must be numeric, got {type(sv.jump_magnitude).__name__}"
    )
elif not np.isfinite(sv.jump_magnitude):
    errors.append(f"jump_magnitude must be finite, got {sv.jump_magnitude}")
```

**Acceptance:** A ShockVector with `timestamp="not_a_datetime"` fails validation. A ShockVector with `jump_detected=42` fails validation. A ShockVector with `jump_magnitude=np.inf` fails validation. All currently-valid ShockVectors continue to pass.

---

## Verification

- [ ] Run `pytest tests/ -k "shock_vector or regime"` — all pass
- [ ] `ShockVector.from_dict` with unknown keys succeeds (forward compat)
- [ ] Both construction paths produce identical structural feature key sets
- [ ] `ensemble_model_type` reflects actual model type in batch ShockVectors
- [ ] Validator catches invalid timestamp, jump_detected, and jump_magnitude types
