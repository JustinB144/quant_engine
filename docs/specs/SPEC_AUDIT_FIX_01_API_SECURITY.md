# SPEC_AUDIT_FIX_01: API Security & Authentication Hardening

**Priority:** CRITICAL — Execute first. Security vulnerabilities must be sealed before any other work.
**Scope:** `api/` — `main.py`, `routers/`, `services/kalshi_service.py`, `ab_testing.py`, `config.py`
**Estimated effort:** 3–4 hours
**Depends on:** Nothing
**Blocks:** All other API specs

---

## Context

The API layer exposes all mutation/compute endpoints without authentication, has a SQL injection vector, allows path traversal through A/B test variant names, and misconfigures CORS to allow credentialed cross-origin requests from any origin. These are all verified against source code.

---

## Tasks

### T1: Add Authentication Middleware for Mutation Endpoints

**Problem:** All POST/PATCH/DELETE endpoints (train, backtest, autopilot run-cycle, config patch) are callable by any network-reachable client with zero authentication.

**Files:** `api/main.py`, all files in `api/routers/` that define POST/PATCH/DELETE endpoints

**Implementation:**
1. Create `api/deps/auth.py` with a dependency that validates a bearer token or API key from the `Authorization` header (or a configurable `X-API-Key` header).
2. Add a config constant `API_AUTH_ENABLED` (default `True`) and `API_AUTH_TOKEN` (read from environment variable `QUANT_ENGINE_API_TOKEN`).
3. Create a FastAPI dependency `require_auth` that:
   - Returns immediately if `API_AUTH_ENABLED` is `False` (local dev mode).
   - Reads the token from the request header.
   - Raises `HTTPException(401)` if missing or invalid.
4. Apply `Depends(require_auth)` to all POST/PATCH/DELETE route handlers in: `routers/model_lab.py`, `routers/backtests.py`, `routers/autopilot.py`, `routers/config_mgmt.py`, and any other mutation endpoints.
5. GET/read-only endpoints remain unauthenticated (observability should be accessible).

**Acceptance:** Sending a POST to `/api/v1/model-lab/train` without a valid token returns 401. With a valid token, it proceeds normally. When `API_AUTH_ENABLED=False`, all endpoints work without tokens.

---

### T2: Fix SQL Injection in Kalshi Query Construction

**Problem:** `services/kalshi_service.py:45` interpolates user input directly into a SQL `where=` clause using f-strings.

**File:** `api/services/kalshi_service.py`

**Implementation:**
1. Replace all f-string SQL interpolation with parameterized queries.
2. Where the storage layer's `query_df()` method accepts a `where=` string, refactor to accept parameters separately: `store.query_df("kalshi_contracts", where="event_id = ?", params=(market_id,))`.
3. If the storage layer (`kalshi/storage.py`) does not support parameterized `query_df`, add that support (see SPEC_AUDIT_FIX_15 for Kalshi storage fixes — coordinate).
4. Audit all other `query_df` calls in `api/services/kalshi_service.py` for the same pattern.

**Acceptance:** No f-string SQL interpolation remains. A market_id value of `' OR 1=1 --` does not execute injected SQL.

---

### T3: Fix CORS Configuration

**Problem:** `main.py:118-124` sets `allow_origins=["*"]` (from `config.py:32`) with `allow_credentials=True`. Per CORS spec, `*` with credentials is invalid and browsers handle it inconsistently.

**File:** `api/main.py`, `api/config.py`

**Implementation:**
1. In `api/config.py`, change `CORS_ORIGINS` default from `["*"]` to `["http://localhost:5173", "http://localhost:8000"]` (dev defaults).
2. Add `CORS_ORIGINS` to the runtime-configurable whitelist, reading from environment variable `QUANT_ENGINE_CORS_ORIGINS` (comma-separated).
3. In `main.py`, only set `allow_credentials=True` when origins are explicitly listed (not `*`).
4. Add a startup warning if `CORS_ORIGINS` contains `"*"`.

**Acceptance:** Default startup does not use `*` for CORS origins. `allow_credentials=True` is only active with explicit origin list.

---

### T4: Sanitize A/B Test Variant Names for Path Traversal

**Problem:** `ab_testing.py:592,712,784` use `variant.name` directly in filesystem paths without sanitization. A name like `../../etc/passwd` could escape the intended directory.

**File:** `api/ab_testing.py`

**Implementation:**
1. Add a `_sanitize_name(name: str) -> str` helper that:
   - Strips or replaces `/`, `\`, `..`, null bytes.
   - Allows only alphanumeric, hyphens, underscores.
   - Raises `ValueError` if the result is empty.
2. Apply `_sanitize_name()` to `variant.name` and `test.test_id` before any `Path` construction at lines 592, 712, 784.
3. Also validate at ABTest/Variant creation time (constructor or Pydantic validator).

**Acceptance:** A variant name containing `../` raises an error rather than constructing a path outside the trades directory.

---

### T5: Fix Runtime Config Boolean Coercion

**Problem:** `api/config.py:68-70` uses `target_type(value)` for type coercion. For booleans, `bool("false")` evaluates to `True` because any non-empty string is truthy.

**File:** `api/config.py`

**Implementation:**
1. In the config patching logic, add special handling for `bool` targets:
   ```python
   if target_type is bool:
       if isinstance(value, str):
           coerced = value.lower() in ("true", "1", "yes")
       else:
           coerced = bool(value)
   else:
       coerced = target_type(value)
   ```
2. Add unit test: patching a bool config key with string `"false"` sets it to `False`.

**Acceptance:** `PATCH /config` with `{"SOME_BOOL_KEY": "false"}` correctly sets the value to `False`.

---

## Improvement Beyond Audit

### T6: Add Config Semantic Validation

**Problem:** Runtime config patching accepts any type-coercible value without range or invariant checks. A Kelly fraction of 500.0 or a negative transaction cost would be accepted.

**File:** `api/config.py`, `api/routers/config_mgmt.py`

**Implementation:**
1. Create a `CONFIG_VALIDATORS` dict mapping config keys to validation functions:
   ```python
   CONFIG_VALIDATORS = {
       "KELLY_FRACTION": lambda v: 0.0 <= v <= 1.0,
       "TRANSACTION_COST_BPS": lambda v: 0.0 <= v <= 500.0,
       "MAX_POSITIONS": lambda v: 1 <= v <= 100,
       # ... add for all ACTIVE config keys with natural bounds
   }
   ```
2. In the patch endpoint, after coercion, run the validator if one exists. Return 422 with a descriptive error on failure.

**Acceptance:** Patching `KELLY_FRACTION` to `5.0` returns a 422 error explaining the valid range.

---

## Verification

- [ ] Run existing API tests: `pytest tests/api/ -v`
- [ ] Manually test auth flow: unauthenticated POST → 401, authenticated POST → 200/202
- [ ] Manually test config patch: `"false"` → bool False, out-of-range → 422
- [ ] Verify CORS headers in browser dev tools
- [ ] Attempt SQL injection via kalshi service endpoint → no injection
- [ ] Attempt path traversal via A/B test variant name → rejected
