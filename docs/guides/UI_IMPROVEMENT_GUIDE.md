# Quant Engine UI Improvement Guide (Current React/FastAPI Stack)

Source-verified improvement guide for the current web UI. This document replaces older Dash/tkinter-era guidance and is based on the active `frontend/` + `api/` implementation.

## Current State (Source Snapshot)

- Frontend source files (`frontend/src`): 131 (7,935 LOC)
- Top-level routed pages: 10
- API query hooks: 12
- API mutation hooks: 6

## What Is Already Good (Keep These Patterns)

- Clear separation of concerns: routes/pages, shared UI components, query/mutation hooks, and Zustand stores.
- Centralized endpoint constants and a shared API envelope client (`frontend/src/api/client.ts`).
- Lazy route loading with error boundaries (`frontend/src/App.tsx`).
- Background job UX infrastructure exists (polling + SSE + job monitor).

## Highest-Value Improvements (Source-Verified Gaps)

### 1. Fix frontend/backend job contract drift

Why it matters: `api/jobs/models.py` emits `queued/succeeded/progress_message`, while `frontend/src/types/jobs.ts` expects `pending/completed/message`. This can break status display and completion handling.

Source to check:
- `api/jobs/models.py`
- `api/jobs/store.py`
- `frontend/src/types/jobs.ts`
- `frontend/src/hooks/useJobProgress.ts`
- `frontend/src/components/job/JobMonitor.tsx`

### 2. Add a dedicated Kalshi API contract (optional but high leverage)

Why it matters: `api/services/kalshi_service.py` exists, but no Kalshi router is mounted. Kalshi UI surfaces rely on indirect artifact/state access, which limits observability and consistency.

Source to check:
- `api/services/kalshi_service.py`
- `api/routers/__init__.py` (router registration list)
- `frontend/src/pages/AutopilotPage/KalshiEventsTab.tsx`

### 3. Strengthen API type alignment and generated types

Why it matters: frontend types under `frontend/src/types/*` are handwritten and can drift from backend schemas (`api/schemas/*`, `api/jobs/models.py`, service payloads).

Source to check:
- `frontend/src/types/*`
- `api/schemas/*`
- `api/routers/*` response payloads
- `scripts/generate_types.py` (if used in your workflow)

### 4. Improve cache transparency in the UI

Why it matters: backend `ResponseMeta` includes `cache_hit`, `elapsed_ms`, `data_mode`, warnings, and provenance fields, but not every page surfaces them consistently.

Source to check:
- `api/schemas/envelope.py`
- `frontend/src/types/api.ts`
- page components that display `meta` (Dashboard, Benchmark, Signals, Autopilot)

### 5. Consolidate page-level loading/error UX patterns

Why it matters: pages currently mix inline error text, `ErrorPanel`, and bespoke loading placeholders. A more consistent pattern reduces maintenance and surprises.

Source to check:
- `frontend/src/pages/*Page.tsx`
- `frontend/src/components/ui/ErrorPanel.tsx`
- `frontend/src/components/ui/Spinner.tsx`
- `frontend/src/components/ui/EmptyState.tsx`

## Medium-Term Improvements

- Add frontend test coverage for API hooks and key page flows (especially jobs and config patching).
- Add route-level access patterns for feature flags / unavailable backend subsystems (for example Kalshi disabled mode via `KALSHI_ENABLED`).
- Normalize chart data adapters so page components stay thinner (more transform logic in hooks/util layers).
- Add explicit contract tests for `ApiResponse.meta` fields consumed by UI badges and banners.

## Improvement Workflow (Recommended)

1. Update backend contract first (`api/schemas`, routers/services, job models) or document the intended change.
2. Align frontend endpoint hook/types.
3. Validate affected pages manually using the dev stack (`run_server.py` + `npm run dev`).
4. Update `docs/reference/FRONTEND_UI_REFERENCE.md` and `docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md` if the contract changed.
