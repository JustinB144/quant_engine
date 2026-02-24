# Feature Spec: Data Loading Diagnostics & "No Data Loaded" Fix

> **Status:** Approved
> **Author:** justin
> **Date:** 2026-02-23
> **Estimated effort:** ~5 hours across 4 tasks

---

## Why

The predict job fails with "No data loaded — check data sources" but gives no information about WHY tickers were rejected. The orchestrator calls `load_universe(..., verbose=False)` which silently swallows all skip reasons. Additionally, `orchestrator.py` line 69 imports `DATA_DIR` from config which doesn't exist (only `DATA_CACHE_DIR` exists), causing the diagnostic code itself to potentially crash. Users have no way to know if their data is stale, missing PERMNOs, failing quality gates, or simply not cached.

## What

A diagnostic system where: every rejected ticker has a recorded reason, the error message tells the user exactly what failed and how to fix it, and the UI shows data loading status with per-ticker health. Done means: "No data loaded" errors include actionable diagnostics, and the Data Explorer shows a data status panel with cache freshness and quality info.

## Constraints

### Must-haves
- Fix `DATA_DIR` → `DATA_CACHE_DIR` import in orchestrator.py
- Per-ticker skip reasons logged at WARNING level even with verbose=False
- Error messages include: which tickers were attempted, which failed and why, what config settings might be wrong
- Data status visible in UI (cache freshness, data source, quality gate results)

### Must-nots
- Do NOT change load_universe return type (still returns Dict[str, DataFrame])
- Do NOT make data loading slower with extra validation
- Do NOT auto-retry failed tickers (user should fix root cause)

### Out of scope
- Automatic WRDS data re-download
- Data quality improvement (separate concern)
- Cache management UI (delete/refresh individual tickers)

## Tasks

### T1: Fix DATA_DIR import and improve orchestrator diagnostics

**What:** Fix the broken import and make the error message actionable.

**Files:**
- `api/orchestrator.py` — fix line 69: `DATA_DIR` → `DATA_CACHE_DIR`

**Implementation notes:**
- Change: `from quant_engine.config import DATA_DIR` → `from quant_engine.config import DATA_CACHE_DIR`
- Change: `cache_dir = DATA_DIR / "cache"` → `cache_dir = DATA_CACHE_DIR`
- Add: check `load_universe` return diagnostics (the skip logging added earlier will populate the log)

**Verify:**
```bash
python -c "from quant_engine.api.orchestrator import PipelineOrchestrator; print('Import OK')"
```

---

### T2: Add /api/data/status endpoint for cache health

**What:** New endpoint showing per-ticker cache status, freshness, data source, and quality.

**Files:**
- `api/routers/data_explorer.py` — add endpoint

**Implementation notes:**
- Endpoint: `GET /api/data/status`
- Scans `DATA_CACHE_DIR` for all parquets
- For each ticker found:
  ```json
  {
    "ticker": "AAPL",
    "permno": "14593",
    "source": "wrds",
    "last_bar_date": "2026-02-21",
    "total_bars": 3750,
    "timeframes_available": ["1d", "5min", "15min"],
    "quality": "GOOD",
    "freshness": "FRESH",
    "days_stale": 2
  }
  ```
- Freshness categories: FRESH (<3 days), STALE (3-14 days), VERY_STALE (>14 days)
- Cache response for 60 seconds (scanning filesystem every request is expensive)

**Verify:**
```bash
curl -s http://localhost:8000/api/data/status | python -m json.tool | head -30
```

---

### T3: Add data status panel to DataExplorerPage

**What:** Show cache health info in the Data Explorer UI.

**Files:**
- `frontend/src/pages/DataExplorerPage.tsx` — add status panel

**Implementation notes:**
- Below the ticker selector, show summary bar:
  - "142 tickers cached | 138 fresh | 4 stale | Source: WRDS"
- When a ticker is selected, show its specific status:
  - Last bar date, total bars, source, quality gate result
  - If stale (>7 days): yellow warning "Data may be stale — last updated Feb 14"
  - If quality POOR: red warning with specific issues

**Verify:**
- Manual: Data Explorer shows cache status summary and per-ticker info

---

### T4: Test data loading diagnostics

**What:** Tests verifying diagnostic messages are useful.

**Files:**
- `tests/test_data_diagnostics.py` — new test file

**Implementation notes:**
- Test cases:
  1. `test_orchestrator_imports_data_cache_dir` — No ImportError on DATA_DIR
  2. `test_load_universe_logs_skip_reasons` — With verbose=False, still logs WARNING with skip reasons
  3. `test_error_message_includes_diagnostics` — RuntimeError message includes ticker list, WRDS status, cache count
  4. `test_data_status_endpoint` — Returns valid status for cached tickers

**Verify:**
```bash
python -m pytest tests/test_data_diagnostics.py -v
```

---

## Validation

### Acceptance criteria
1. `from quant_engine.api.orchestrator import PipelineOrchestrator` succeeds (no DATA_DIR crash)
2. "No data loaded" error includes WRDS_ENABLED, REQUIRE_PERMNO, cache file count, and attempted tickers
3. Data Explorer shows per-ticker cache status (freshness, source, quality)
4. Skip reasons logged at WARNING level for every rejected ticker

### Rollback plan
- DATA_DIR fix is a one-line change — safe, no rollback needed
- Status endpoint is additive — remove route if issues
- Frontend status panel is additive — hide with CSS if needed
