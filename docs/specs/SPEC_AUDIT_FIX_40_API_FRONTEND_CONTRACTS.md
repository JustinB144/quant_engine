# SPEC_AUDIT_FIX_40: API–Frontend Contract Alignment & Job Lifecycle Fixes

**Priority:** HIGH — Job lifecycle is broken end-to-end (status mismatch + SSE wiring + compute schema drift); system_health error paths mislead consumers.
**Scope:** `api/jobs/`, `api/routers/system_health.py`, `api/schemas/compute.py`, `frontend/src/types/`, `frontend/src/hooks/`, `frontend/src/pages/`
**Estimated effort:** 5–6 hours
**Depends on:** SPEC_02 (job runner fixes), SPEC_03 (envelope fixes), SPEC_12 (frontend type safety)
**Blocks:** Nothing

---

## Context

Two independent audits of Subsystem 10 (API & Frontend) identified critical contract mismatches between the backend and frontend that are NOT covered by existing specs. SPEC_01–04 cover API security, async/concurrency, envelope contracts, and health service decomposition. SPEC_12 covers frontend type safety and client-side fixes. SPEC_38 covers kalshi_service.py. However, none of those specs address the **job lifecycle contract drift**, **SSE event wiring mismatch**, or **compute request schema drift** — all of which are confirmed broken end-to-end.

### Cross-Audit Reconciliation

| Finding | Auditor 1 (Claude) | Auditor 2 (Claude) | Existing Spec | Disposition |
|---------|--------------------|--------------------|---------------|-------------|
| diagnostics.py/risk.py envelope bypass | A-01/A-02 (P0), A-03/A-04 (P1) | P1 | **SPEC_03 T1** | Already covered |
| kalshi_service.py broken API | A-05/A-06 (P1), A-12/A-13 (P2) | — | **SPEC_01 T2 + SPEC_38 T1** | Already covered |
| HealthService God Object | A-07 (P2) | — | **SPEC_04 T5** | Already covered |
| Sync SQLite in async | A-08 (P2) | — | **SPEC_02 T2** | Already covered |
| Job status contract drift | — | P1 | Not in any spec | **NEW → T1** |
| SSE event wiring mismatch | — | P1 | Not in any spec | **NEW → T2** |
| Compute request schema drift | — | P1 | Not in any spec | **NEW → T3** |
| system_health error returns ok=True | A-10 (P2) | — | Not in any spec | **NEW → T4** |
| Circular api<->autopilot tracking | A-09 (P2) | — | Not in any spec | **NEW → T5** |
| Cancelled queued job SSE hang | A-19 (P3), A-20 (P3) | P2 | **SPEC_02 T1** (partial) | **NEW → T6** (SSE-specific) |

---

## Tasks

### T1: Fix Job Status Contract Drift Between Backend and Frontend

**Problem:** The backend `JobStatus` enum (`api/jobs/models.py:11`) uses `queued`/`succeeded` and field name `progress_message`. The frontend `JobStatus` type (`frontend/src/types/jobs.ts:1`) uses `pending`/`completed` and field name `message`. The frontend's `useJobProgress.ts:20-23` checks for `status === 'pending'` and `status === 'running'` — but the backend never sends `"pending"`, it sends `"queued"`. The frontend's `useJobProgress.ts:49` defaults to `'pending'`, but the backend defaults to `'queued'`. Job completion detection is broken because the frontend checks for `"completed"` but the backend sends `"succeeded"`.

**Files:** `frontend/src/types/jobs.ts`, `frontend/src/hooks/useJobProgress.ts`, `frontend/src/hooks/useJobs.ts` (or `frontend/src/api/queries/useJobs.ts`)

**Implementation:**

Option A (preferred — frontend aligns to backend): The backend enum is the canonical source. Update the frontend to match:

1. Update `types/jobs.ts`:
   ```typescript
   export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'
   ```
2. Update `types/jobs.ts` field name from `message` to `progress_message`:
   ```typescript
   export interface JobRecord {
     job_id: string
     job_type: string
     status: JobStatus
     progress_message?: string  // was: message
     // ...
   }
   ```
3. Update all frontend status checks:
   - `'pending'` → `'queued'`
   - `'completed'` → `'succeeded'`
   - `message` → `progress_message`
4. Search for every occurrence in `useJobProgress.ts`, `useJobs.ts`, `JobMonitor.tsx`, and any other files referencing these status values.

Option B (backend adds aliases): Add an API serialization layer that maps backend statuses to frontend-friendly names. This is NOT recommended because it adds indirection — better to have one canonical set.

**Acceptance:** Frontend correctly detects when a job is queued (not `undefined`), correctly detects completion (`succeeded`), and displays progress messages using the correct field name. Job polling and SSE work end-to-end.

---

### T2: Fix SSE Event Wiring Mismatch

**Problem:** The backend (`api/routers/jobs.py:57`) emits SSE events with named event types:
```python
yield {"event": event.get("event", "message"), "data": json.dumps(event)}
```
This produces events like `event: started`, `event: progress`, `event: completed`, `event: failed`, `event: done`.

The frontend (`frontend/src/hooks/useSSE.ts:34`) only registers:
```typescript
es.onmessage = (e) => onMessageRef.current(e)
```
Per the SSE specification, `onmessage` only handles events with NO event type or event type `"message"`. Named events (like `started`, `progress`, `completed`) require `addEventListener('eventname', handler)`. Since the backend NEVER emits default/unnamed events, the frontend receives NOTHING from the SSE stream.

**Files:** `frontend/src/hooks/useSSE.ts`, `api/routers/jobs.py`

**Implementation:**

Option A (preferred — fix frontend to listen for named events):
1. In `useSSE.ts`, add listeners for all expected event types:
   ```typescript
   const EVENT_TYPES = ['status', 'started', 'progress', 'completed', 'failed', 'cancelled', 'done', 'message'] as const;

   useEffect(() => {
     const es = new EventSource(url);
     eventSourceRef.current = es;

     const handler = (e: MessageEvent) => {
       onMessageRef.current(e);
     };

     // Listen for all named events AND the default message event
     EVENT_TYPES.forEach(type => {
       es.addEventListener(type, handler);
     });
     es.onmessage = handler;  // Catch any unnamed events as fallback

     es.onerror = (e) => { ... };

     return () => {
       EVENT_TYPES.forEach(type => {
         es.removeEventListener(type, handler);
       });
       es.close();
     };
   }, [url]);
   ```

Option B (fix backend to emit unnamed events): Change `jobs.py:57` to always use `"message"` as the event type:
```python
yield {"event": "message", "data": json.dumps(event)}
```
This is simpler but loses the ability to differentiate event types on the client side.

**Recommendation:** Use Option A so the frontend can distinguish between event types (progress updates vs. completion vs. failure). This enables richer UX — e.g., showing a progress bar for `progress` events and a completion toast for `completed`.

**Acceptance:** Frontend receives and processes all SSE events from the backend. Progress messages appear in real-time during job execution. Job completion triggers the correct UI state update.

---

### T3: Fix Compute Request Schema Drift

**Problem:** The frontend sends fields in BacktestRequest and TrainRequest that the backend schema does not define, so they are silently ignored by Pydantic:

Frontend `BacktestRequest` (compute.ts:8-14):
- `holding_period`, `max_positions`, `entry_threshold`, `position_size` — NOT in backend schema

Backend `BacktestRequest` (schemas/compute.py:21-30):
- Has `tickers`, `years`, `feature_mode`, `risk_management`, `version` — NOT in frontend type

Frontend `TrainRequest` (compute.ts:1-6):
- Uses `survivorship_filter` — backend expects `survivorship` (different field name)

Frontend `BacktestPage.tsx:11-17` sends defaults like `holding_period: 10`, `max_positions: 10`, `entry_threshold: 0.01`, `position_size: 0.1` — the backend never sees these.

**Files:** `frontend/src/types/compute.ts`, `api/schemas/compute.py`, `frontend/src/pages/BacktestPage.tsx`, `frontend/src/pages/ModelLabPage/TrainingTab.tsx`, `frontend/src/components/SignalControls.tsx`

**Implementation:**

1. Audit the orchestrator to determine which fields are actually used by the compute pipeline:
   - Check `api/orchestrator.py` `backtest()` and `train()` method signatures
   - Check `backtest/engine.py` for `holding_period`, `max_positions`, `entry_threshold`, `position_size`
   - Determine if these fields are consumed anywhere in the pipeline or are truly dead

2. For fields that ARE consumed by the pipeline but missing from the backend schema, add them:
   ```python
   class BacktestRequest(BaseModel):
       horizon: int = 10
       tickers: Optional[List[str]] = None
       years: int = 15
       feature_mode: str = "core"
       risk_management: bool = False
       version: str = "latest"
       full_universe: bool = False
       # Add fields consumed by pipeline:
       holding_period: int = 10
       max_positions: int = 10
       entry_threshold: float = 0.01
       position_size: float = 0.1
   ```

3. For fields that are NOT consumed anywhere, remove them from the frontend types and UI components.

4. Fix the `survivorship_filter` → `survivorship` field name mismatch:
   ```typescript
   // frontend/src/types/compute.ts
   export interface TrainRequest {
     horizons?: number[]
     feature_mode?: string
     survivorship?: boolean  // was: survivorship_filter
     full_universe?: boolean
   }
   ```

5. Update the frontend types to include all backend fields:
   ```typescript
   export interface BacktestRequest {
     horizon?: number
     tickers?: string[]
     years?: number
     feature_mode?: string
     risk_management?: boolean
     version?: string
     full_universe?: boolean
     // ... plus any pipeline fields confirmed in step 2
   }
   ```

6. Update `BacktestPage.tsx` and `TrainingTab.tsx` to send the correct field names.

**Acceptance:** Every field sent by the frontend is defined in the backend schema. Every field expected by the backend is present in the frontend type. No silently ignored fields. `survivorship_filter` → `survivorship` name is aligned.

---

### T4: Fix system_health.py Error Paths Returning ok=True

**Problem:** `api/routers/system_health.py:85-88` — the `/model-age` endpoint catches `(OSError, json.JSONDecodeError, ValueError, ImportError)` and returns `ApiResponse.success({"status": "error", ...})`. Similarly at lines 118-120, `/data-mode` catches exceptions and returns `ApiResponse.success({"mode": "unknown", "status": "error", ...})`. Both should use `ApiResponse.fail()` so the frontend (which checks `json.ok`) correctly identifies errors.

**File:** `api/routers/system_health.py`

**Implementation:**
1. Change the error paths in `/model-age`:
   ```python
   except (OSError, json.JSONDecodeError, ValueError, ImportError) as e:
       elapsed = (time.monotonic() - t0) * 1000
       return ApiResponse.fail(
           f"Could not determine model age: {e}",
           elapsed_ms=elapsed,
       )
   ```
2. Change the error paths in `/data-mode`:
   ```python
   except (OSError, ValueError, ImportError) as e:
       elapsed = (time.monotonic() - t0) * 1000
       return ApiResponse.fail(
           f"Could not determine data mode: {e}",
           elapsed_ms=elapsed,
       )
   ```
3. The frontend will now correctly see `ok=false` and can display an appropriate error state.
4. If the endpoint should return partial data on error (e.g., `age_days: null` alongside the error), use `ApiResponse.fail()` with the partial data in the `data` field (if `ApiResponse.fail()` supports a data parameter), or return `ok=false` with the data in a separate field.

**Acceptance:** When model metadata is unreadable, `/model-age` returns `ok=false`. Frontend displays error state instead of showing `null` age as if everything is fine.

---

### T5: Extract Shared Tracking Module to Break Circular api<->autopilot Dependency

**Problem:** `api/services/health_service.py` provides IC tracking, disagreement tracking, and execution quality tracking functions that are consumed by `autopilot/engine.py:1868,1911` and `autopilot/paper_trader.py:189,532`. This creates a circular dependency: `api/jobs/autopilot_job.py` → `autopilot/engine.py` → `api/services/health_service.py`. All imports are lazy with try/except guards (runtime-safe), but structurally fragile — any top-level import change breaks the cycle.

**Files:** `api/services/health_service.py`, new file `quant_engine/tracking/` module

**Implementation:**
1. Create a new `quant_engine/tracking/` package:
   ```
   quant_engine/tracking/__init__.py
   quant_engine/tracking/ic_tracker.py
   quant_engine/tracking/disagreement_tracker.py
   quant_engine/tracking/execution_tracker.py
   ```
2. Move the following functions from `health_service.py` into the tracking package:
   - `save_ic_snapshot()` / `get_ic_history()` → `ic_tracker.py`
   - `save_disagreement_snapshot()` / `get_disagreement_history()` → `disagreement_tracker.py`
   - `save_execution_quality_fill()` / `get_execution_quality_summary()` → `execution_tracker.py`
3. Each tracker module owns its own SQLite database initialization and operations.
4. Update `autopilot/engine.py` and `autopilot/paper_trader.py` to import from `quant_engine.tracking` instead of `api.services.health_service`:
   ```python
   # Before (lazy import of api module):
   from quant_engine.api.services.health_service import save_ic_snapshot

   # After (clean import of shared module):
   from quant_engine.tracking.ic_tracker import save_ic_snapshot
   ```
5. Update `health_service.py` to delegate to the tracking modules instead of having inline implementations.
6. The imports from `autopilot` → `tracking` are now non-circular (tracking has no dependency on api or autopilot).
7. The lazy import guards in autopilot can remain as defensive patterns but are no longer structurally required.

**Acceptance:** `autopilot/engine.py` imports tracking functions without touching `api/`. No circular dependency between `api/` and `autopilot/`. All existing IC, disagreement, and execution quality tracking functionality works unchanged.

---

### T6: Fix Cancelled Queued Job SSE Subscriber Hang

**Problem:** When a queued (non-running) job is cancelled via `runner.cancel()` → `store.cancel_job()`, the store updates the database status to `cancelled` but does NOT emit an SSE event. If a client has already subscribed to events via `subscribe_events()`, the subscriber's queue is waiting in `queue.get()` and will never receive a terminal event (`done` or `cancelled`), causing the SSE connection to hang indefinitely.

SPEC_02 T1 covers cooperative cancellation for running jobs (threading.Event), but does not address the SSE subscriber leak for queued jobs.

**File:** `api/jobs/runner.py`

**Implementation:**
1. After `store.cancel_job()` succeeds, emit a `cancelled` event to any subscribers:
   ```python
   async def cancel(self, job_id: str) -> bool:
       task = self._active_tasks.get(job_id)
       if task and not task.done():
           task.cancel()
           return True
       cancelled = await self._store.cancel_job(job_id)
       if cancelled:
           # Notify SSE subscribers that the queued job was cancelled
           await self._emit(job_id, {
               "event": "cancelled",
               "job_id": job_id,
               "status": "cancelled",
           })
           # Also emit terminal "done" event so subscriber loop breaks
           await self._emit(job_id, {
               "event": "done",
               "job_id": job_id,
           })
       return cancelled
   ```
2. Add a timeout to the subscriber loop as a safety net:
   ```python
   async def subscribe_events(self, job_id: str) -> AsyncGenerator[Dict[str, Any], None]:
       queue: asyncio.Queue = asyncio.Queue()
       self._event_subscribers.setdefault(job_id, []).append(queue)
       try:
           rec = await self._store.get_job(job_id)
           if rec:
               yield {"event": "status", "data": rec.model_dump()}
               # If already terminal, break immediately
               if rec.status in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
                   return

           while True:
               try:
                   event = await asyncio.wait_for(queue.get(), timeout=300)  # 5min safety timeout
               except asyncio.TimeoutError:
                   # Check if job is in terminal state
                   rec = await self._store.get_job(job_id)
                   if rec and rec.status in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
                       yield {"event": "done", "job_id": job_id}
                       break
                   continue
               yield event
               if event.get("event") in ("done", "cancelled"):
                   break
       finally:
           subs = self._event_subscribers.get(job_id, [])
           if queue in subs:
               subs.remove(queue)
   ```

**Acceptance:** Cancelling a queued job while an SSE subscriber is connected sends a `cancelled` event and terminates the stream. No SSE connections hang indefinitely.

---

## Verification

- [ ] Run `pytest tests/api/ -v` — all pass
- [ ] Run `npm run typecheck` — zero errors
- [ ] End-to-end test: submit job → frontend shows "queued" → job runs → frontend shows progress → job completes → frontend shows "succeeded"
- [ ] Test SSE: submit job → subscribe to SSE → verify progress events arrive in frontend
- [ ] Test compute schemas: submit backtest with all frontend fields → backend correctly receives and uses them
- [ ] Test system_health error: trigger model-age error → verify `ok=false` in response
- [ ] Test cancel queued job: queue a job → subscribe SSE → cancel → verify SSE stream terminates
- [ ] Verify no circular imports: `python -c "from quant_engine.autopilot.engine import AutopilotEngine"` works without api imports
