# SPEC_AUDIT_FIX_12: Frontend Type Safety, API Contract & Security Fixes

**Priority:** HIGH — Type mismatches cause runtime crashes; URL encoding and error handling affect reliability.
**Scope:** `frontend/src/` — types, hooks, components, client
**Estimated effort:** 3–4 hours
**Depends on:** Nothing (frontend-only changes)
**Blocks:** Nothing

---

## Context

The frontend has four P0 type contract mismatches where components reference fields not present in declared types, plus P1 issues with URL encoding, cache invalidation, ECharts API misuse, completion callback deduplication, hardcoded URLs, error handling, and unsafe assumptions about API response shape.

---

## Tasks

### T1: Fix Type Mismatches (4 P0 Issues)

**Problem:** Multiple components access fields not declared in their TypeScript interfaces, relying on `any` casts or runtime `undefined` behavior.

**Files:**
- `FeatureDiffViewer.tsx` / `types/models.ts` — accesses `previous_importance` not in `FeatureImportance`
- `PaperPnLTracker.tsx` / `types/autopilot.ts` — accesses `trade_history` and `cumulative_pnl` not in `PaperState`/`PaperTrade`
- `BacktestResults.tsx` / `types/backtests.ts` — accesses `sharpe_pvalue`, `spa_pvalue`, `dsr_pvalue`, `ic_pvalue` not in `BacktestResult`

**Implementation:**
1. For each mismatch, determine if the backend actually returns these fields (check FastAPI schemas and actual API responses).
2. **If backend returns them:** Add the fields to the TypeScript types:
   ```typescript
   // models.ts
   interface FeatureImportance {
     global_importance: Record<string, number>;
     regime_heatmap: Record<string, Record<string, number>>;
     previous_importance?: Record<string, number>;  // ADD
   }

   // autopilot.ts
   interface PaperState {
     // ... existing fields ...
     trade_history?: PaperTrade[];  // ADD
     initial_capital?: number;      // ADD
   }
   interface PaperTrade {
     // ... existing fields ...
     cumulative_pnl?: number;       // ADD
   }

   // backtests.ts
   interface BacktestResult {
     // ... existing fields ...
     sharpe_pvalue?: number;  // ADD
     spa_pvalue?: number;     // ADD
     dsr_pvalue?: number;     // ADD
     ic_pvalue?: number;      // ADD
   }
   ```
3. **If backend does NOT return them:** Fix the component to handle missing fields gracefully with optional chaining (`?.`) and fallback values.
4. Run `npm run typecheck` and fix all remaining type errors.

**Acceptance:** `npm run typecheck` passes with zero errors. No `any` casts hiding type mismatches.

---

### T2: Fix URL Encoding for Dynamic Values

**Problem:** `endpoints.ts:35,53`, `useData.ts:51,75`, `useBacktests.ts:18`, `useSignals.ts:9` concatenate ticker symbols and parameters directly into URLs without `encodeURIComponent()`.

**Files:** `api/endpoints.ts`, `hooks/useData.ts`, `hooks/useBacktests.ts`, `hooks/useSignals.ts`

**Implementation:**
1. In `endpoints.ts`, encode all dynamic path segments:
   ```typescript
   export const DATA_TICKER = (ticker: string) =>
     `/data/ticker/${encodeURIComponent(ticker)}`;
   ```
2. In hook files, encode query parameter values:
   ```typescript
   const url = `${DATA_TICKER(ticker)}?years=${encodeURIComponent(String(years))}`;
   ```
3. Apply consistently across all endpoint functions and hook URL constructions.

**Acceptance:** A ticker containing special characters (e.g., `BRK.B`) produces a valid URL.

---

### T3: Fix usePatchConfig Cache Invalidation

**Problem:** `usePatchConfig.ts:11` invalidates `['config']` but not `['config-status']`, leaving stale UI state after config changes.

**File:** `hooks/usePatchConfig.ts`

**Implementation:**
1. Invalidate both query keys on success:
   ```typescript
   onSuccess: () => {
     queryClient.invalidateQueries({ queryKey: ['config'] });
     queryClient.invalidateQueries({ queryKey: ['config-status'] });
   }
   ```

**Acceptance:** After patching config, both config and config-status UI elements refresh.

---

### T4: Fix ScatterChart symbolSize Array

**Problem:** `ScatterChart.tsx:50` sets `symbolSize` to an array, which is not valid for ECharts scatter series (expects number or function).

**File:** `components/ScatterChart.tsx`

**Implementation:**
1. Change to a function that maps data index to size:
   ```typescript
   symbolSize: (value: number[], params: any) => {
     const point = data[params.dataIndex];
     return point?.size ?? 8;
   }
   ```

**Acceptance:** Scatter chart renders with correctly sized markers.

---

### T5: Fix JobMonitor onComplete Re-invocation

**Problem:** `JobMonitor.tsx:14` — The useEffect fires onComplete on every render where status is complete, without checking if it already fired.

**File:** `components/JobMonitor.tsx`

**Implementation:**
1. Add a ref to track whether onComplete has been called:
   ```typescript
   const completedRef = useRef(false);
   useEffect(() => {
     if (status === 'succeeded' && !completedRef.current) {
       completedRef.current = true;
       onComplete?.(result);
     }
   }, [status, result, onComplete]);
   ```
2. Reset `completedRef.current = false` when a new job starts (if the component is reused).

**Acceptance:** onComplete fires exactly once per job completion, not on every render.

---

### T6: Fix StatusBar Hardcoded URLs

**Problem:** `StatusBar.tsx:30,97` uses `fetch('/api/v1/...')` directly and hardcodes `localhost:8000`.

**File:** `components/StatusBar.tsx`

**Implementation:**
1. Import and use the shared API client and endpoint constants:
   ```typescript
   import { get } from '../api/client';
   import { SYSTEM_HEALTH } from '../api/endpoints';
   ```
2. Remove hardcoded `localhost:8000` reference. Use relative URLs or the configured base URL.

**Acceptance:** StatusBar uses the shared API client. No hardcoded host/port.

---

### T7: Fix API Client Error Handling

**Problem:** `client.ts:29,31` discards structured error envelopes on non-2xx responses, throwing only raw text. Also forces `Content-Type: application/json` on all requests including GET.

**File:** `api/client.ts`

**Implementation:**
1. Parse error envelope:
   ```typescript
   if (!res.ok) {
     try {
       const errorJson = await res.json();
       throw new ApiError(errorJson.error || errorJson.message || res.statusText, res.status, errorJson);
     } catch (e) {
       if (e instanceof ApiError) throw e;
       throw new ApiError(await res.text(), res.status);
     }
   }
   ```
2. Update `ApiError` class to include structured data: `constructor(message, status, data?)`.
3. Only set `Content-Type: application/json` for POST/PATCH/PUT, not GET:
   ```typescript
   const headers: Record<string, string> = {};
   if (options?.body) {
     headers['Content-Type'] = 'application/json';
   }
   ```

**Acceptance:** API errors include structured error data. GET requests don't send Content-Type header.

---

### T8: Fix DataProvenanceBadge Null Safety

**Problem:** `DataProvenanceBadge.tsx:49` accesses `meta.warnings.length` without null check.

**File:** `components/DataProvenanceBadge.tsx`

**Implementation:**
1. Add optional chaining:
   ```typescript
   const warningCount = meta?.warnings?.length ?? 0;
   ```

**Acceptance:** DataProvenanceBadge renders without crash when `meta.warnings` is undefined.

---

## Verification

- [ ] Run `npm run typecheck` — zero errors
- [ ] Run `npm run build` — builds successfully
- [ ] Test StatusBar renders with shared client
- [ ] Test JobMonitor fires onComplete exactly once
- [ ] Test API error responses include structured data
- [ ] Test ScatterChart renders with variable marker sizes
