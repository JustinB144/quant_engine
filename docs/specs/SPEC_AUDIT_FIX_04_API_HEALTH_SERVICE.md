# SPEC_AUDIT_FIX_04: Health Service Performance & Decomposition

**Priority:** HIGH
**Scope:** `api/services/health_service.py` (~2000+ LOC monolith)
**Estimated effort:** 4–5 hours
**Depends on:** Nothing
**Blocks:** Nothing

---

## Context

The HealthService is a ~2000+ line monolith mixing scoring logic, SQLite persistence, alerting, trend analytics, and heavy I/O cache scans. It persists a snapshot on every health request (mutation on read), executes synchronous full-cache-scans in the request path, uses hardcoded thresholds and relative file paths.

---

## Tasks

### T1: Throttle Health Snapshot Persistence

**Problem:** `services/health_service.py:201` calls `self.save_health_snapshot(runtime_health)` on every `get_detailed_health()` call, so every observability request mutates the health history database and can distort trends/alerts.

**File:** `api/services/health_service.py`

**Implementation:**
1. Add instance variable `self._last_snapshot_time: float = 0.0` and constant `HEALTH_SNAPSHOT_MIN_INTERVAL_SECONDS = 300` (5 minutes, configurable).
2. In `get_detailed_health()`, only call `save_health_snapshot()` if `time.time() - self._last_snapshot_time >= HEALTH_SNAPSHOT_MIN_INTERVAL_SECONDS`.
3. Update `self._last_snapshot_time` after successful save.

**Acceptance:** Calling `get_detailed_health()` 10 times in 1 minute produces at most 1 database write.

---

### T2: Move Heavy Cache Scans to Background Task

**Problem:** `services/health_service.py:1099` (`_check_data_anomalies`) and `:1191` (`_check_market_microstructure`) iterate through all parquet files synchronously in the request path, causing latency spikes.

**File:** `api/services/health_service.py`

**Implementation:**
1. Create a background task that runs `_check_data_anomalies()` and `_check_market_microstructure()` periodically (every 15 minutes).
2. Store results in instance variables: `self._cached_anomaly_result` and `self._cached_microstructure_result`, with a `self._cache_scan_timestamp`.
3. In `get_detailed_health()`, read cached results instead of running scans inline.
4. Use `asyncio.create_task()` with a periodic loop in the app lifespan, or use a `BackgroundTasks` mechanism.
5. On first request before any background scan has run, return `{"status": "pending", "message": "Initial scan in progress"}` for these checks.

**Acceptance:** Health endpoint responds in <200ms even with 1000+ parquet files in cache. Background scan runs every 15 minutes.

---

### T3: Make Trend Detection Thresholds Configurable

**Problem:** `services/health_service.py:2103,2105` uses hardcoded `±0.5` slope thresholds for trend classification, which are too coarse for subtle drift detection.

**File:** `api/services/health_service.py`, `api/config.py`

**Implementation:**
1. Add to `api/config.py`:
   ```python
   HEALTH_TREND_IMPROVING_THRESHOLD = 0.5
   HEALTH_TREND_DEGRADING_THRESHOLD = -0.5
   ```
2. Replace hardcoded values with config references.
3. Add to the runtime-configurable whitelist so operators can tune live.

**Acceptance:** Config-driven thresholds. Changing `HEALTH_TREND_DEGRADING_THRESHOLD` to `-0.2` causes more sensitive degradation detection.

---

### T4: Fix Feature-Stability Path to Use Config

**Problem:** `services/health_service.py:1817` uses hardcoded relative path `"results/feature_stability_history.json"` which depends on current working directory.

**File:** `api/services/health_service.py`

**Implementation:**
1. Import `RESULTS_DIR` from config.
2. Replace `Path("results/feature_stability_history.json")` with `Path(RESULTS_DIR) / "feature_stability_history.json"`.

**Acceptance:** Health service finds feature stability file regardless of CWD.

---

### T5: Decompose HealthService into Focused Modules

**Problem:** HealthService mixes scoring, storage, alerting, DB schema, and trend analytics in a single ~2000+ LOC class, harming testability and reliability.

**Files:** Create new files: `api/services/health_scoring.py`, `api/services/health_storage.py`, `api/services/health_checks.py`, `api/services/health_alerts.py`

**Implementation:**
1. **`health_checks.py`**: Move all `_check_*` methods (data anomalies, microstructure, feature stability, model staleness, etc.) into standalone functions. Each function takes required data as parameters (no self reference).
2. **`health_scoring.py`**: Move domain scoring logic (score aggregation, severity classification, red flag detection).
3. **`health_storage.py`**: Move SQLite operations (snapshot persistence, history retrieval, trend computation).
4. **`health_alerts.py`**: Move alerting logic (threshold evaluation, notification dispatch).
5. **`health_service.py`**: Reduce to a thin orchestrator that imports from the above modules and coordinates the health check pipeline.
6. This is a pure refactor — no behavior changes. All existing tests must pass unchanged.

**Acceptance:** `health_service.py` reduced to <300 LOC. All existing health-related tests pass. Each new module is independently testable.

---

## Verification

- [ ] Run `pytest tests/api/ -v` — all pass
- [ ] Time health endpoint with 500+ parquet files — <200ms response
- [ ] Verify snapshot throttling — repeated requests don't flood DB
- [ ] Verify feature stability file found from any CWD
- [ ] Each new module importable and testable independently
