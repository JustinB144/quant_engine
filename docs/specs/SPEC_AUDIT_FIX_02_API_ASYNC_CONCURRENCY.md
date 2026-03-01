# SPEC_AUDIT_FIX_02: API Async, Concurrency & Job System Fixes

**Priority:** HIGH — Concurrency bugs cause silent data corruption and event loop blocking.
**Scope:** `api/` — `jobs/runner.py`, `routers/system_health.py`, `cache/manager.py`, `routers/logs.py`
**Estimated effort:** 3–4 hours
**Depends on:** Nothing
**Blocks:** Nothing (can run in parallel with SPEC_01)

---

## Context

The API layer has several concurrency and async correctness issues: jobs cannot be truly cancelled, SQLite calls block the event loop, the in-memory cache has race conditions, job submission is unbounded, and the log handler attaches at import time.

---

## Tasks

### T1: Add Cooperative Cancellation to Job Runner

**Problem:** `jobs/runner.py:53,83` — `task.cancel()` only cancels the asyncio wrapper, but the actual work runs in `asyncio.to_thread()` and continues executing with side effects.

**File:** `api/jobs/runner.py`

**Implementation:**
1. Add a `threading.Event` cancellation token to each job:
   ```python
   cancel_event = threading.Event()
   ```
2. Pass `cancel_event` to the job function alongside `progress_callback`.
3. Update all long-running job functions (train, backtest, autopilot) to periodically check `cancel_event.is_set()` and raise a `JobCancelled` exception.
4. In `cancel()`, set the event AND cancel the asyncio task.
5. Update `JobRecord` to include a `cancel_event` field.
6. Add `cancelled` as a terminal job status with proper cleanup.

**Acceptance:** Cancelling a running training job stops the thread within 5 seconds. Job status transitions to `cancelled`.

---

### T2: Fix Sync SQLite Blocking Event Loop

**Problem:** `routers/system_health.py:51` calls `svc.get_health_history_with_trends()` synchronously inside an async route handler, blocking the event loop.

**File:** `api/routers/system_health.py`

**Implementation:**
1. Wrap the synchronous call in `asyncio.to_thread()`:
   ```python
   data = await asyncio.to_thread(svc.get_health_history_with_trends, limit=90)
   ```
2. Audit all other routes in `routers/system_health.py` for synchronous calls to `HealthService` methods that touch SQLite. Wrap any found.
3. Check `routers/logs.py` and `routers/dashboard.py` for the same pattern.

**Acceptance:** Health history endpoint does not block concurrent requests. No synchronous SQLite operations in async handlers.

---

### T3: Fix model_age Timezone Mismatch

**Problem:** `routers/system_health.py:76-77` subtracts a potentially timezone-aware `datetime.fromisoformat()` result from timezone-naive `datetime.now()`, which raises `TypeError`.

**File:** `api/routers/system_health.py`

**Implementation:**
1. Replace `datetime.now()` with `datetime.now(timezone.utc)` (import from `datetime`).
2. Ensure `train_date` is also UTC-aware: if `fromisoformat()` returns naive, treat as UTC:
   ```python
   train_date = datetime.fromisoformat(train_date_str)
   if train_date.tzinfo is None:
       train_date = train_date.replace(tzinfo=timezone.utc)
   age_days = (datetime.now(timezone.utc) - train_date).days
   ```

**Acceptance:** `model_age` endpoint returns correct age regardless of whether the stored timestamp includes timezone info.

---

### T4: Add Thread-Safe Cache Manager

**Problem:** `cache/manager.py:25,36,55` — The in-memory TTL cache uses a plain dict with no synchronization. Concurrent async tasks can race on reads/writes, and cached objects are returned by reference (caller mutation risk).

**File:** `api/cache/manager.py`

**Implementation:**
1. Add an `asyncio.Lock` to the cache manager:
   ```python
   self._lock = asyncio.Lock()
   ```
2. Wrap all `_store` mutations (`__setitem__`, `__delitem__`, `get`, eviction) with `async with self._lock`.
3. Return deep copies of cached values (or document that callers must not mutate):
   ```python
   import copy
   return copy.deepcopy(value)
   ```
   If deep copy is too expensive for large DataFrames, return the object but add a `_frozen` flag or use `types.MappingProxyType` for dicts.
4. Add a `max_size` parameter to limit cache entries (default 100). Evict oldest on overflow.

**Acceptance:** Concurrent cache access from multiple async handlers does not corrupt state. Cache has a bounded size.

---

### T5: Bound Job Submission Queue

**Problem:** `jobs/runner.py:21,35` — The semaphore only limits concurrent execution (default 2), but any number of jobs can be submitted and queued, consuming unbounded memory.

**File:** `api/jobs/runner.py`

**Implementation:**
1. Add a `max_queued` parameter (default 20).
2. Track pending (queued but not running) job count.
3. In `submit()`, reject new jobs with HTTP 429 if pending count exceeds `max_queued`.
4. Return a clear error message: `"Job queue full. {max_queued} jobs pending. Try again later."`

**Acceptance:** Submitting 25 jobs when `max_queued=20` causes the 21st+ to return 429. Completing a job frees a queue slot.

---

### T6: Fix Log Buffer Handler Attachment

**Problem:** `routers/logs.py:30-32` attaches a `_BufferHandler` to the root `quant_engine` logger at module import time. This can duplicate handlers across reloads and makes the handler impossible to test or disable.

**File:** `api/routers/logs.py`

**Implementation:**
1. Move handler attachment into a `setup_log_buffer()` function.
2. Call it once during app startup (`main.py` lifespan), not at import time.
3. Add a guard to prevent duplicate attachment:
   ```python
   if not any(isinstance(h, _BufferHandler) for h in logger.handlers):
       logger.addHandler(_handler)
   ```
4. Add a `teardown_log_buffer()` for test cleanup.

**Acceptance:** Reloading the module does not add duplicate handlers. Buffer can be cleanly attached/detached.

---

### T7: Fix Orchestrator Silent Failure Dropping

**Problem:** `orchestrator.py:248,315` catches `(ValueError, KeyError, TypeError)` with `pass`, silently dropping prediction and backtest failures with no logging or error propagation.

**File:** `api/orchestrator.py`

**Implementation:**
1. Replace bare `pass` with structured logging:
   ```python
   except (ValueError, KeyError, TypeError) as exc:
       logger.warning("Prediction failed for %s: %s", ticker, exc)
       failed_tickers.append({"ticker": ticker, "error": str(exc)})
   ```
2. Include `failed_tickers` in the orchestrator's return value / job result so the caller knows which tickers failed.
3. If >50% of tickers fail, escalate to `logger.error` and set job status to `failed` (not `succeeded`).

**Acceptance:** A failed prediction for ticker AAPL appears in logs and in the job result. Majority failure causes job failure status.

---

## Verification

- [ ] Run `pytest tests/api/ -v`
- [ ] Test job cancellation: submit long job → cancel → verify thread stops
- [ ] Test concurrent cache access: fire 50 simultaneous requests → no race errors
- [ ] Test job queue limit: submit `max_queued + 5` jobs → verify 429 responses
- [ ] Test health endpoint under concurrent load: no event loop blocking
