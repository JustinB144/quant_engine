# SPEC_AUDIT_FIX_10: Features Causality Enforcement & Data Integrity Fixes

**Priority:** CRITICAL — Causality violations directly cause look-ahead bias in backtests and live trading.
**Scope:** `features/` — `pipeline.py`, `research_factors.py`, `macro.py`, `version.py`
**Estimated effort:** 4–5 hours
**Depends on:** Nothing
**Blocks:** SPEC_11 (indicators causality fix)

---

## Context

The feature engineering pipeline has critical causality violations: "RESEARCH_ONLY" filter mode is not actually enforced, unknown/unregistered features default to "CAUSAL" (fail-open), the VolSpillover_Net indicator uses a wrong operand mixing different semantics, DTW backtracking produces negative indices, monthly macro forward-fill is too short for monthly data, and feature collisions are silently dropped.

---

## Tasks

### T1: Enforce RESEARCH_ONLY Causality Filter

**Problem:** `pipeline.py` `_filter_causal_features()` only handles `"CAUSAL"` and `"END_OF_DAY"` modes. When `causality_filter="RESEARCH_ONLY"` is set, it falls through and returns everything — defeating the purpose of research-mode restrictions.

**File:** `features/pipeline.py`

**Implementation:**
1. In `_filter_causal_features()`, add explicit handling:
   ```python
   if causality_filter == "RESEARCH_ONLY":
       # Include CAUSAL + RESEARCH_ONLY features, exclude END_OF_DAY
       allowed_types = {"CAUSAL", "RESEARCH_ONLY"}
       return [f for f in features if get_feature_type(f) in allowed_types]
   elif causality_filter == "CAUSAL":
       # Strictest: only causally safe features
       return [f for f in features if get_feature_type(f) == "CAUSAL"]
   elif causality_filter == "END_OF_DAY":
       # All features available after market close
       return features  # All types allowed
   else:
       raise ValueError(f"Unknown causality_filter: {causality_filter}")
   ```
2. Add test for each filter mode.

**Acceptance:** `causality_filter="RESEARCH_ONLY"` includes CAUSAL and RESEARCH_ONLY features but excludes END_OF_DAY. Unknown filter values raise an error.

---

### T2: Make Unknown Features Fail-Closed

**Problem:** `pipeline.py` `get_feature_type()` returns `"CAUSAL"` for any feature not in `FEATURE_METADATA`, silently permitting unknown features to pass causality checks.

**File:** `features/pipeline.py`

**Implementation:**
1. Change default return to `"UNKNOWN"`:
   ```python
   def get_feature_type(feature_name: str) -> str:
       return FEATURE_METADATA.get(feature_name, {}).get("type", "UNKNOWN")
   ```
2. In `_filter_causal_features()`, treat `"UNKNOWN"` as non-causal (fail-closed):
   - In `"CAUSAL"` mode: exclude unknown features.
   - In `"RESEARCH_ONLY"` mode: exclude unknown features.
   - In `"END_OF_DAY"` mode: include unknown features (permissive).
3. Log a warning for each unknown feature encountered:
   ```python
   logger.warning("Feature '%s' has no causality metadata, treating as UNKNOWN", feature_name)
   ```
4. Add a `FEATURE_METADATA_COVERAGE_CHECK` that runs at pipeline startup and warns if >10% of computed features lack metadata.

**Acceptance:** A new feature added without metadata is excluded from CAUSAL mode. Warning is logged. Coverage check runs at startup.

---

### T3: Fix VolSpillover_Net Wrong Operand

**Problem:** `research_factors.py` line ~632 computes `VolSpillover_Net` as `vol_out - recv_cent` instead of `vol_out - vol_in`. This mixes volatility spillover with momentum-centrality semantics.

**File:** `features/research_factors.py`

**Implementation:**
1. Fix the computation:
   ```python
   # OLD: "VolSpillover_Net": vol_out[:, i] - recv_cent[:, i],
   # NEW:
   "VolSpillover_Net": vol_out[:, i] - vol_in[:, i],
   ```
2. Verify `vol_in` is computed and available in scope (it should be the incoming volatility spillover from other assets).
3. Add a unit test with synthetic data verifying that VolSpillover_Net = vol_out - vol_in.

**Acceptance:** VolSpillover_Net is the difference between outgoing and incoming volatility spillover, both in volatility-space.

---

### T4: Fix DTW Backtracking Negative Indices

**Problem:** `research_factors.py` DTW backtrack loop appends `(i-1, j-1)` BEFORE checking if `i==0` or `j==0`, producing negative indices in the path.

**File:** `features/research_factors.py`

**Implementation:**
1. Move boundary check BEFORE the append:
   ```python
   while i > 0 or j > 0:
       if i == 0:
           j -= 1
           path.append((i, j))
       elif j == 0:
           i -= 1
           path.append((i, j))
       else:
           # Standard backtrack: choose minimum cost neighbor
           costs = [cost_matrix[i-1, j-1], cost_matrix[i-1, j], cost_matrix[i, j-1]]
           min_idx = np.argmin(costs)
           if min_idx == 0:
               i, j = i-1, j-1
           elif min_idx == 1:
               i -= 1
           else:
               j -= 1
           path.append((i, j))
   ```
2. Add assertion: `assert all(i >= 0 and j >= 0 for i, j in path), "Negative indices in DTW path"`.

**Acceptance:** DTW path contains no negative indices. Assertion holds for all valid inputs.

---

### T5: Fix Monthly Macro Forward-Fill Limit

**Problem:** `macro.py:236` uses `ffill(limit=5)` for monthly data, but monthly releases are ~21 business days apart. Only 5 days are filled, leaving ~16 days as NaN.

**File:** `features/macro.py`

**Implementation:**
1. Determine the data frequency and set limit accordingly:
   ```python
   # For monthly data, ffill up to 25 business days (one month + buffer)
   if freq == "monthly":
       aligned = raw.reindex(date_range).ffill(limit=25)
   elif freq == "weekly":
       aligned = raw.reindex(date_range).ffill(limit=7)
   else:
       aligned = raw.reindex(date_range).ffill(limit=5)
   ```
2. If the `freq` parameter isn't available, infer from the data gap between observations.
3. Add a `staleness_warning` flag when data hasn't updated in >1.5× expected frequency.

**Acceptance:** Monthly macro features (UMCSENT, etc.) have values for all business days between releases, not just the first 5 days.

---

### T6: Fix Feature Collision Silent Drop

**Problem:** `pipeline.py:1147-1148` silently drops duplicate feature columns using pandas `duplicated(keep='first')` with no logging.

**File:** `features/pipeline.py`

**Implementation:**
1. Before deduplication, log which features are duplicated:
   ```python
   dupes = features.columns[features.columns.duplicated()].tolist()
   if dupes:
       logger.warning("Duplicate features detected and dropped (kept first): %s", dupes)
   ```
2. Optionally raise an error in strict mode (`STRICT_FEATURE_DEDUP = True` from config) to force resolution.

**Acceptance:** Duplicate features produce a warning log listing the dropped columns.

---

### T7: Fix FeatureVersion Frozen Dataclass Mutation

**Problem:** `version.py:23-31` declares `@dataclass(frozen=True)` but mutates `feature_names` in `__post_init__`. This violates the frozen contract.

**File:** `features/version.py`

**Implementation:**
1. Use `object.__setattr__` in `__post_init__` (standard pattern for frozen dataclasses):
   ```python
   def __post_init__(self):
       object.__setattr__(self, 'feature_names', tuple(sorted(self.feature_names)))
   ```
2. Change `feature_names` type from `List[str]` to `Tuple[str, ...]` for true immutability.

**Acceptance:** `FeatureVersion` is truly immutable. `v.feature_names = [...]` raises `FrozenInstanceError`.

---

### T8: Fix Compatibility Check Ignoring Extra Features

**Problem:** `version.py:72-74,145` marks compatibility as True when current features are a superset of model features, ignoring extra features that could indicate feature-space drift.

**File:** `features/version.py`

**Implementation:**
1. Report extra features in the compatibility result:
   ```python
   def check_compatibility(self, other: FeatureVersion) -> dict:
       missing = set(other.feature_names) - set(self.feature_names)
       extra = set(self.feature_names) - set(other.feature_names)
       return {
           "compatible": len(missing) == 0,
           "missing_features": list(missing),
           "extra_features": list(extra),
           "drift_warning": len(extra) > 0,
       }
   ```
2. Log a warning when extra features are present but compatibility is still True.

**Acceptance:** Compatibility check reports extra features. `drift_warning=True` when current has features the model doesn't expect.

---

## Verification

- [ ] Run `pytest tests/ -k "feature or pipeline or version"` — all pass
- [ ] Verify RESEARCH_ONLY filter mode works correctly
- [ ] Verify unknown features are excluded from CAUSAL mode
- [ ] Verify VolSpillover_Net uses vol_out - vol_in
- [ ] Verify DTW path has no negative indices
- [ ] Verify monthly macro features have values between releases
