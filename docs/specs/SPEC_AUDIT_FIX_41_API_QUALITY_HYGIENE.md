# SPEC_AUDIT_FIX_41: API Quality Gating, Cache Wiring & Service Hardening

**Priority:** MEDIUM — Dead invalidation hook, unused schemas, unprotected artifact reads, and assorted service hygiene.
**Scope:** `api/cache/invalidation.py`, `api/schemas/`, `api/routers/`, `api/services/`, `api/jobs/runner.py`
**Estimated effort:** 3–4 hours
**Depends on:** SPEC_03 (envelope fixes), SPEC_04 (health service)
**Blocks:** Nothing

---

## Context

Two independent audits of Subsystem 10 (API & Frontend) identified medium and low priority findings beyond what existing specs (SPEC_01–04, SPEC_12, SPEC_38) and the new SPEC_40 cover. These relate to dead cache invalidation code, unused schema models, unprotected artifact file reads, and assorted service hygiene issues that degrade reliability and maintainability.

### Cross-Audit Traceability

| Finding | Auditor 1 (Claude) | Auditor 2 (Claude) | Existing Spec | Disposition |
|---------|--------------------|--------------------|---------------|-------------|
| `invalidate_on_data_refresh` dead code | A-11 (P2) | P3 | Not in any spec | **NEW → T1** |
| Schema models defined but unused | A-14 (P2) | — | Not in any spec | **NEW → T2** |
| `json.load()` unprotected in services | A-22/A-23 (P3) | P2 | Not in any spec | **NEW → T3** |
| ModelService no error handling | A-24 (P3) | — | Not in any spec | **NEW → T4** |
| `_not_found()` bypasses ApiResponse | A-15 (P3) | — | Not in any spec | **NEW → T5** |
| logs.py missing elapsed_ms | A-16 (P3) | — | Not in any spec | **NEW → T5** |

### Deferred Items (P3-LOW / INFO — Document Only)

The following items from both audits are acknowledged but deferred:

| ID | File | Description | Rationale for Deferral |
|----|------|-------------|----------------------|
| A-17 | health_service.py:1913+ | Class-level mutable path singletons | Functional; address during SPEC_04 T5 decomposition |
| A-18 | health_service.py:123 | Naive `datetime.now()` inconsistent with UTC | Address with SPEC_02 T3 (timezone mismatch) |
| A-19 | runner.py:83-89 | TOCTOU race in job cancellation | Theoretical; mitigated by SPEC_40 T6 safety timeout |
| A-21 | runner.py:22 | `_active_tasks` includes queued-but-not-running | Cosmetic naming; no functional impact |
| A-25 | health_risk_feedback.py:247 | YAML config path uses `__import__` | Fragile but functional; low risk |
| A-26 | paper_trader.py:228 | Calls private `ABTestRegistry._save()` | Fragile coupling; address when A/B testing is refactored |
| A-I1 | api/config.py | Default CORS `*` | **SPEC_01 T3** already covers this |
| A-I3 | orchestrator.py | predict()/backtest() file write side effects | Documented behavior; no action needed |
| A-I4 | data_helpers.py | Hardcoded feature inventory counts | May drift; document as maintenance note |
| A-I7 | system_health.py | datetime.now() without timezone | Covered by SPEC_02 T3 |
| A-I9 | All routers | ApiResponse generic `T` never specified | Cosmetic; OpenAPI docs improvement |
| A-I10 | health_confidence.py | Hard `from scipy import stats` at module level | Breaks lazy import pattern but scipy is always available |
| A-I11 | ab_testing.py | MD5 for ticker assignment | Functional; only fails on FIPS-restricted systems |

---

## Tasks

### T1: Wire `invalidate_on_data_refresh` Into Data Refresh Pathway

**Problem:** `api/cache/invalidation.py:21` defines `invalidate_on_data_refresh()` which clears dashboard, signals, benchmark, and feature_importance cache domains. However, it is never called anywhere in the codebase. External data refreshes (new WRDS data, manual cache updates) leave stale cache for up to 300 seconds (the longest TTL) until natural expiry.

**Files:** `api/cache/invalidation.py`, `api/services/data_service.py`, `api/routers/data_explorer.py`

**Implementation:**
1. Identify all data refresh pathways:
   - `data_service.py` — any method that fetches or refreshes external data
   - `data_explorer.py` — any endpoint that triggers data reload
   - Startup/lifespan events that populate initial data
2. Call `invalidate_on_data_refresh(cache)` after successful data refresh operations:
   ```python
   # In data_service.py after successful data fetch:
   from ..cache.invalidation import invalidate_on_data_refresh
   invalidate_on_data_refresh(cache_manager)
   ```
3. If no explicit data refresh endpoint exists, wire it into the orchestrator's `load_and_prepare()` method (which is called before train/backtest):
   ```python
   # In orchestrator.py load_and_prepare():
   # After data loading completes
   try:
       from .cache.invalidation import invalidate_on_data_refresh
       from .deps.providers import get_cache
       invalidate_on_data_refresh(get_cache())
   except Exception:
       pass  # Cache invalidation failure is non-fatal
   ```
4. Also consider calling it during app startup lifespan to clear stale cache from previous runs.

**Acceptance:** After a data refresh, dashboard/signals/benchmark/feature_importance caches are invalidated immediately. The function is called from at least one live code path.

---

### T2: Wire Typed Schema Models Into Router Signatures

**Problem:** `api/schemas/` contains well-defined Pydantic models (`DashboardKPIs`, `BacktestSummary`, `QuickStatus`, `ModelLabResult`, etc.) but NO router uses them. All routers pass untyped `dict` objects to `ApiResponse.success(data)`. This means: (a) OpenAPI docs show `data: Any` instead of structured schemas, (b) no Pydantic validation on response data, (c) the schema definitions are dead code.

**Files:** All files in `api/routers/`, `api/schemas/`

**Implementation:**
1. For each router, identify the matching schema model:
   - `dashboard.py` → `schemas/dashboard.py::DashboardKPIs`
   - `backtests.py` → `schemas/backtests.py::BacktestSummary`
   - `system_health.py` → `schemas/system_health.py::QuickStatus`
   - `model_lab.py` → `schemas/model_lab.py::ModelLabResult`
   - `signals.py` → `schemas/signals.py::SignalData`
   - `autopilot.py` → `schemas/autopilot.py::AutopilotState`
2. Update router return types to use `ApiResponse[SchemaModel]`:
   ```python
   from ..schemas.dashboard import DashboardKPIs

   @router.get("/kpis", response_model=ApiResponse[DashboardKPIs])
   async def get_kpis():
       data = _build_kpis()
       return ApiResponse.success(DashboardKPIs(**data))
   ```
3. If the service layer returns dicts with more/fewer fields than the schema, update the schema model to match actual output — or use `.model_construct()` for lenient construction.
4. This improves OpenAPI documentation quality, adds runtime response validation, and eliminates dead schema code.

**Note:** This is a gradual improvement. Start with the 3 most-used endpoints (dashboard, backtests, system_health) and expand.

**Acceptance:** At least 3 router endpoints use typed schema models. OpenAPI `/docs` shows structured response schemas instead of `Any`. Schema files are no longer dead code.

---

### T3: Add Error Handling to Artifact File Reads

**Problem:** Multiple service modules read JSON artifact files without protection against corrupt or malformed content:
- `backtest_service.py:24-25` — `json.load(f)` on summary file, no try/except
- `results_service.py:24-25` — `json.load(f)` on results file, no try/except
- `backtest_service.py:51` — `pd.read_csv()` on backtest trades, no try/except for corrupt CSV

A corrupt artifact file causes an unhandled exception that propagates as HTTP 500 with no useful error message.

**Files:** `api/services/backtest_service.py`, `api/services/results_service.py`

**Implementation:**
1. Wrap all artifact reads in defensive try/except:
   ```python
   # backtest_service.py
   def get_backtest_summary(self, result_id: str) -> Optional[Dict]:
       summary_path = self._results_dir / result_id / "summary.json"
       if not summary_path.exists():
           return None
       try:
           with open(summary_path) as f:
               return json.load(f)
       except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
           logger.warning("Corrupt backtest summary %s: %s", result_id, e)
           return None  # or raise a custom ArtifactCorruptError
   ```
2. Apply the same pattern to `results_service.py`:
   ```python
   def get_result(self, result_id: str) -> Optional[Dict]:
       path = self._results_dir / result_id / "result.json"
       if not path.exists():
           return None
       try:
           with open(path) as f:
               return json.load(f)
       except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
           logger.warning("Corrupt result file %s: %s", result_id, e)
           return None
   ```
3. For CSV reads, add similar protection:
   ```python
   try:
       trades_df = pd.read_csv(trades_path)
   except (pd.errors.ParserError, UnicodeDecodeError, OSError) as e:
       logger.warning("Corrupt trades CSV %s: %s", result_id, e)
       trades_df = pd.DataFrame()
   ```
4. The router layer should check for `None` returns and use `ApiResponse.fail("Artifact unavailable or corrupt")`.

**Acceptance:** A corrupt `summary.json` returns a graceful error message instead of HTTP 500. A warning is logged for investigation.

---

### T4: Add Error Handling to ModelService Initialization

**Problem:** `api/services/model_service.py:17-21,89-98` instantiates `ModelRegistry()` and `ModelGovernance()` without error handling. If the model directory doesn't exist or the registry is corrupt, the service crashes on import/instantiation rather than gracefully degrading.

**File:** `api/services/model_service.py`

**Implementation:**
1. Wrap registry/governance initialization:
   ```python
   class ModelService:
       def __init__(self):
           try:
               from quant_engine.models.registry import ModelRegistry
               self._registry = ModelRegistry()
           except Exception as e:
               logger.warning("ModelRegistry unavailable: %s", e)
               self._registry = None

           try:
               from quant_engine.models.governance import ModelGovernance
               self._governance = ModelGovernance()
           except Exception as e:
               logger.warning("ModelGovernance unavailable: %s", e)
               self._governance = None
   ```
2. In each service method, check if the dependency is available:
   ```python
   def get_model_info(self, version: str = "latest"):
       if self._registry is None:
           return {"error": "Model registry unavailable", "status": "degraded"}
       # ... existing logic
   ```
3. This ensures the API layer starts even when model subsystem is partially broken.

**Acceptance:** API starts successfully even when model registry directory is missing. Model endpoints return graceful degradation messages instead of 500 errors.

---

### T5: Fix Minor Router Envelope Inconsistencies

**Problem:** Two minor envelope violations:
1. `routers/jobs.py:20-22` — `_not_found()` returns `JSONResponse(status_code=404, ...)` bypassing the declared `-> ApiResponse` return type and the global error handler pattern.
2. `routers/logs.py:38` — does not include `elapsed_ms` in the response meta, unlike all other endpoints that measure response time.

**File:** `api/routers/jobs.py`, `api/routers/logs.py`

**Implementation:**
1. Fix `_not_found()` in jobs.py:
   ```python
   def _not_found(job_id: str) -> ApiResponse:
       return ApiResponse.fail(f"Job '{job_id}' not found", status_code=404)
   ```
   If `ApiResponse.fail()` doesn't support `status_code`, raise the custom `NotFoundError` exception instead (which the global error handler maps to 404).

2. Add timing to logs.py:
   ```python
   @router.get("/logs")
   async def get_logs(limit: int = 100):
       t0 = time.monotonic()
       logs = _handler.get_records(limit)
       elapsed = (time.monotonic() - t0) * 1000
       return ApiResponse.success(logs, elapsed_ms=elapsed)
   ```

**Acceptance:** `_not_found()` returns a proper `ApiResponse.fail()` envelope. Logs endpoint includes `elapsed_ms` in meta.

---

## Verification

- [ ] Run `pytest tests/api/ -v` — all pass
- [ ] Verify `invalidate_on_data_refresh` is called during at least one live code path
- [ ] Verify at least 3 router endpoints use typed schema response models
- [ ] Verify corrupt JSON artifact file returns graceful error, not 500
- [ ] Verify API starts when model registry directory is missing
- [ ] Verify `_not_found()` returns ApiResponse envelope
- [ ] Verify logs endpoint includes `elapsed_ms`
