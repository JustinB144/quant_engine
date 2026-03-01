# SPEC_AUDIT_FIX_03: API Response Contracts, Config & Data Quality Fixes

**Priority:** HIGH
**Scope:** `api/` — `routers/`, `schemas/`, `ab_testing.py`
**Estimated effort:** 3 hours
**Depends on:** SPEC_01 (auth) should land first
**Blocks:** Nothing

---

## Context

Multiple API endpoints violate the `ApiResponse` envelope contract, return placeholder data as success, silently discard metadata fields, invalidate caches at the wrong time, and use incorrect REST semantics. The A/B statistical test implementation has a formula error that invalidates inference.

---

## Tasks

### T1: Fix Placeholder Endpoints Returning Success Status

**Problem:** `routers/diagnostics.py:34` and `routers/risk.py:34` return hardcoded demo/schema data with `"status": "ok"`, misleading operators into trusting non-live analytics.

**Files:** `api/routers/diagnostics.py`, `api/routers/risk.py`

**Implementation:**
1. Both endpoints should return `ApiResponse` envelopes (they currently return raw dicts — also fixes finding #18).
2. When no real data is available, set `ok=False` and include `error="No live data connected. Showing schema only."` with a `data_mode="placeholder"` field in `meta`.
3. When real data is available, return normally with `ok=True`.

**Acceptance:** GET `/diagnostics` without live data returns `{"ok": false, "error": "No live data connected...", "data": {...schema...}, "meta": {"data_mode": "placeholder"}}`.

---

### T2: Add regime_trade_policy to ResponseMeta

**Problem:** `routers/signals.py:26` sets `meta_fields["regime_trade_policy"]` but `schemas/envelope.py:12` `ResponseMeta` does not declare this field. Pydantic silently drops it.

**File:** `api/schemas/envelope.py`

**Implementation:**
1. Add `regime_trade_policy: Optional[str] = None` to the `ResponseMeta` model.
2. Verify signals endpoint now includes the field in responses.

**Acceptance:** GET `/signals` response `meta` object includes `regime_trade_policy` when set.

---

### T3: Move Backtest Cache Invalidation to Job Completion

**Problem:** `routers/backtests.py:88` invalidates cache immediately on job submission (queue time), not on successful completion. This causes stale/empty cache windows during the run.

**File:** `api/routers/backtests.py`

**Implementation:**
1. Remove `invalidate_on_backtest(cache)` from the submission path.
2. In the job's completion callback (or in the job wrapper function), call `invalidate_on_backtest(cache)` only after the backtest succeeds.
3. If using the `JobRunner`, add an `on_success` callback parameter that runs after the job function returns successfully.

**Acceptance:** Cache is only invalidated after a backtest job transitions to `succeeded`. A failed backtest does not invalidate cache.

---

### T4: Fix data_explorer POST Using Query Params

**Problem:** `routers/data_explorer.py:461,465` defines a POST endpoint (`/ticker/{ticker}/indicators/batch`) but takes all inputs as `Query()` parameters instead of a request body.

**File:** `api/routers/data_explorer.py`

**Implementation:**
1. Create a Pydantic model:
   ```python
   class BatchIndicatorRequest(BaseModel):
       timeframe: str = "1d"
       indicators: List[str] = ["rsi_14", "macd", "bollinger_20"]
   ```
2. Change endpoint signature to accept `body: BatchIndicatorRequest` as a `Body()` parameter.
3. Update frontend `useData.ts` calls to send JSON body instead of query params.

**Acceptance:** POST `/ticker/AAPL/indicators/batch` with JSON body `{"timeframe": "1d", "indicators": ["rsi_14"]}` works correctly.

---

### T5: Fix Parquet File Selection to Use Newest Match

**Problem:** `routers/data_explorer.py:55-56` returns the first file from `sorted(glob(...))`, which is alphabetical, not the newest.

**File:** `api/routers/data_explorer.py`

**Implementation:**
1. Replace alphabetical sort with modification time sort:
   ```python
   matches = list(cache_dir.glob(f"{ticker_upper}_{pat}_*.parquet"))
   if matches:
       return max(matches, key=lambda p: p.stat().st_mtime)
   ```

**Acceptance:** When multiple parquet files exist for the same ticker/pattern, the most recently modified file is returned.

---

### T6: Fix Mutable Defaults in Pydantic Schemas

**Problem:** `schemas/autopilot.py:21`, `schemas/dashboard.py:29`, `schemas/system_health.py:13`, `schemas/signals.py:25` use `{}` and `[]` as default values.

**Files:** All files in `api/schemas/`

**Implementation:**
1. Replace all `= {}` with `= Field(default_factory=dict)`.
2. Replace all `= []` with `= Field(default_factory=list)`.
3. Add `from pydantic import Field` import where missing.

**Acceptance:** No mutable default arguments in any schema model. `grep -rn "= {}" api/schemas/` and `grep -rn "= \[\]" api/schemas/` return no matches.

---

### T7: Fix A/B Statistical Test Newey-West Implementation

**Problem:** `ab_testing.py:292,303` computes Newey-West HAC variance on concatenated centered arms rather than computing per-arm variance and combining properly. This invalidates the t-test inference.

**File:** `api/ab_testing.py`

**Implementation:**
1. Compute Newey-West variance **separately** for each arm:
   ```python
   nw_var_a = _newey_west_variance(returns_a - returns_a.mean(), max_lag)
   nw_var_b = _newey_west_variance(returns_b - returns_b.mean(), max_lag)
   se = np.sqrt(nw_var_a / len(returns_a) + nw_var_b / len(returns_b))
   ```
2. Remove the concatenation approach at line 293-296.
3. The t-statistic becomes: `t = (mean_a - mean_b) / se`.
4. Add a unit test with known synthetic data where the correct answer is known.

**Acceptance:** A/B test with two arms of known different means produces statistically significant result. Unit test passes with synthetic data.

---

## Verification

- [ ] Run `pytest tests/api/ -v`
- [ ] Verify diagnostics endpoint returns `ok=false` with placeholder mode
- [ ] Verify signals endpoint includes `regime_trade_policy` in meta
- [ ] Verify backtest cache only invalidated on completion
- [ ] Verify batch indicators accepts JSON body
- [ ] Verify parquet selection returns newest file
