# SPEC_AUDIT_FIX_13: Kalshi Data Integrity, Storage & Distribution Fixes

**Priority:** CRITICAL — Data integrity errors in distribution construction affect all Kalshi research output.
**Scope:** `kalshi/` — `distribution.py`, `events.py`, `storage.py`, `walkforward.py`, `regimes.py`
**Estimated effort:** 4–5 hours
**Depends on:** Nothing
**Blocks:** Nothing

---

## Context

The Kalshi subsystem has critical data integrity issues: threshold-direction inference uses substring matching that can misclassify contracts, event-to-market merge can mis-assign market IDs, INSERT OR REPLACE can null out previously stored fields, walk-forward capacity is overstated by counting duplicates, regime tagging drops valid zero values, unresolved-direction distributions still compute moments, and version ordering is lexicographic (unsafe for v10 vs v2).

---

## Tasks

### T1: Fix Threshold-Direction Substring Misclassification

**Problem:** `distribution.py:164,167` matches raw substrings like `"ge"`, `"le"`, `"over"` in free text, which can trigger on unrelated words (e.g., "recovery" matches "over", "danger" matches "ge").

**File:** `kalshi/distribution.py`

**Implementation:**
1. Use word-boundary regex instead of substring search:
   ```python
   import re
   _ABOVE_PATTERN = re.compile(r'\b(>=|above|over|greater\s+than|at\s+least|or\s+higher)\b', re.IGNORECASE)
   _BELOW_PATTERN = re.compile(r'\b(<=|below|under|less\s+than|at\s+most|or\s+lower)\b', re.IGNORECASE)

   def infer_threshold_direction(rules_text: str) -> Optional[str]:
       if _ABOVE_PATTERN.search(rules_text):
           return "above"
       if _BELOW_PATTERN.search(rules_text):
           return "below"
       return None
   ```
2. Add test cases: `"recovery above 50%"` → "above"; `"large gathering"` → None (not misclassified by "ge").

**Acceptance:** Words like "recovery", "danger", "altogether" do not trigger false threshold-direction matches.

---

### T2: Fix Event-to-Market Fallback Merge

**Problem:** `events.py:134,139` uses index-aligned `fillna()` after a merge, which can mis-assign market_id when the fallback frame has different index alignment.

**File:** `kalshi/events.py`

**Implementation:**
1. Use keyed merge instead of index alignment:
   ```python
   # Merge on event_id (or appropriate key), not index position
   result = out.merge(fallback[['event_id', 'market_id']], on='event_id', how='left', suffixes=('', '_fallback'))
   result['market_id'] = result['market_id'].fillna(result['market_id_fallback'])
   result = result.drop(columns=['market_id_fallback'])
   ```
2. Add assertion: no duplicate market_id assignments per event.

**Acceptance:** Each event maps to exactly one market_id. Fallback merge uses key alignment, not positional.

---

### T3: Fix Storage INSERT OR REPLACE Partial Row Nullification

**Problem:** `storage.py:153,160` constructs INSERT OR REPLACE with only the columns present in the incoming row. SQLite replaces the entire row, setting missing columns to NULL.

**File:** `kalshi/storage.py`

**Implementation:**
1. Use INSERT ... ON CONFLICT DO UPDATE instead of INSERT OR REPLACE:
   ```python
   # Build upsert statement that only updates provided columns
   key_cols = self._get_primary_key_cols(table)
   update_cols = [c for c in cols if c not in key_cols]
   conflict_clause = ", ".join(key_cols)
   update_clause = ", ".join(f"{c} = excluded.{c}" for c in update_cols)

   sql = f"""INSERT INTO {table} ({col_sql}) VALUES ({placeholders})
             ON CONFLICT({conflict_clause}) DO UPDATE SET {update_clause}"""
   ```
2. This preserves existing column values not present in the update.

**Acceptance:** Upserting a row with columns A and B does not null out column C's existing value.

---

### T4: Fix Walk-Forward Capacity Overstatement

**Problem:** `walkforward.py:451,208` counts per-row traces including multi-horizon duplicates, so event frequency and capacity constraints are overstated.

**File:** `kalshi/walkforward.py`

**Implementation:**
1. Deduplicate by event_id before computing capacity metrics:
   ```python
   unique_events = traces.drop_duplicates(subset=['event_id'])
   event_frequency = len(unique_events) / n_periods
   ```
2. Compute capacity per-event (not per-trace-row).

**Acceptance:** Capacity metrics reflect unique events, not duplicate horizon rows.

---

### T5: Fix Regime Tagging Zero-Value Drop

**Problem:** `regimes.py:80` uses `value or np.nan` which converts `0.0` (falsy) to NaN. Zero is a valid regime/inflation value.

**File:** `kalshi/regimes.py`

**Implementation:**
1. Replace `or` with explicit None check:
   ```python
   # OLD: classify_inflation_regime(cpi_yoy or np.nan)
   # NEW:
   classify_inflation_regime(cpi_yoy if cpi_yoy is not None else np.nan)
   ```
2. Apply the same fix to all `value or np.nan` patterns in the file.

**Acceptance:** `classify_inflation_regime(0.0)` returns a valid regime classification, not NaN.

---

### T6: Prevent Moment Computation for Unresolved Direction

**Problem:** `distribution.py:683,721` computes distribution moments even when threshold direction is None (unresolved), producing non-NaN statistics that are semantically invalid.

**File:** `kalshi/distribution.py`

**Implementation:**
1. Early return when direction is None:
   ```python
   if direction is None:
       return DistributionResult(
           support=support, mass=None, mean=np.nan, std=np.nan, skew=np.nan,
           direction_resolved=False, quality="unresolved_direction"
       )
   ```
2. Add `direction_resolved: bool` field to the result if not present.

**Acceptance:** Unresolved-direction distributions return NaN moments and `direction_resolved=False`.

---

### T7: Fix Lexicographic Version Ordering

**Problem:** `storage.py:633` sorts `mapping_version` as text, so "v10" sorts before "v2".

**File:** `kalshi/storage.py`

**Implementation:**
1. Use numeric sorting:
   ```python
   # Option A: Store version as integer in DB
   # Option B: Sort in Python after query
   df = df.sort_values('mapping_version', key=lambda s: s.str.extract(r'(\d+)')[0].astype(int))
   ```
2. If using SQL ORDER BY, cast or extract the numeric part:
   ```sql
   ORDER BY CAST(REPLACE(mapping_version, 'v', '') AS INTEGER) DESC
   ```

**Acceptance:** Version "v10" sorts after "v2", not before.

---

### T8: Enforce Bin Validation in Quality Gating

**Problem:** `distribution.py:542,755` computes bin validation but does not use it to gate quality. Structurally invalid bins can pass through.

**File:** `kalshi/distribution.py`

**Implementation:**
1. After bin validation, check result and set quality flag:
   ```python
   if not bin_validation_passed:
       result.quality = "quality_low"
       result.quality_flags.append("invalid_bin_structure")
   ```
2. In the downstream feature generation, skip distributions marked `quality_low` or treat them as missing.

**Acceptance:** Distributions with invalid bin structure are flagged `quality_low` and excluded from feature generation.

---

## Verification

- [ ] Run `pytest kalshi/tests/ -v` — all pass
- [ ] Verify threshold direction inference with edge-case strings
- [ ] Verify upsert preserves existing columns
- [ ] Verify capacity counts unique events
- [ ] Verify zero regime values are preserved
- [ ] Verify version ordering is numeric
