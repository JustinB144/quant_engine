# Feature Spec: Health System Transparency & Trustworthiness

> **Status:** Approved
> **Author:** justin
> **Date:** 2026-02-23
> **Estimated effort:** ~10 hours across 7 tasks

---

## Why

The health system currently shows scores (0-100) with no explanation of what they mean, how they're calculated, or whether the underlying checks are meaningful. Users see a score like "75" but have no way to evaluate whether that number is trustworthy. Several checks return hardcoded fallback scores (50.0) when data is unavailable, which inflates the overall score and creates a false sense of health. The domain scoring uses simple averages without weighting by severity — a catastrophic data integrity failure counts the same as a minor feature drift.

## What

A transparent, auditable health system where: every score explains its methodology to the user, unavailable checks are clearly marked as "UNAVAILABLE" (not scored), domain scores weight checks by severity, and the UI shows exactly what each check measures with thresholds visible. Done means: the health page shows a methodology panel, each check has a human-readable explanation, and no hardcoded fallback scores inflate the result.

## Constraints

### Must-haves
- Every `_check_*` method must return a structured result with: score, status (PASS/WARN/FAIL/UNAVAILABLE), explanation string, methodology string, and thresholds used
- UNAVAILABLE checks must be excluded from domain score calculation (not counted as 50)
- Domain scores must use severity-weighted averages (critical checks weight 3x, standard 1x)
- UI must show expandable methodology panel for each check
- Health history: store last 30 scores with timestamps for trend visualization
- Overall score calculation must be documented and visible in the UI

### Must-nots
- Do NOT remove any existing checks — only improve their output format and scoring logic
- Do NOT change the API endpoint paths (/health, /health/detailed)
- Do NOT make health checks slower (keep under 5s total)

### Out of scope
- Adding new health checks (separate spec)
- Real-time health monitoring/alerting
- Health score-based autopilot gating

## Current State

### Key files
| File | Role | Notes |
|------|------|-------|
| `api/services/health_service.py` | 15 comprehensive health checks across 5 domains | All checks return hardcoded score=50.0 on failure; no methodology strings |
| `api/services/data_helpers.py` | 6 basic health checks (_check_data_integrity, _check_walkforward, etc.) | Scores start at 0 and earn points, but thresholds not documented |
| `api/routers/system_health.py` | /health and /health/detailed endpoints | Returns JSON with scores but no methodology |
| `frontend/src/pages/SystemHealthPage.tsx` | Health dashboard UI | Shows scores + PASS/WARN/FAIL but no explanation of what each score means |

### Existing patterns to follow
- Health checks return dicts with "score", "status", "details" keys
- Domain weights: Data Integrity 25%, Signal Quality 25%, Risk Management 20%, Execution Quality 15%, Model Governance 15%
- Status thresholds: Green > 80, Amber > 50, Red < 50

## Tasks

### T1: Create HealthCheckResult dataclass with methodology fields

**What:** Define a structured return type for all health checks that includes methodology and threshold documentation.

**Files:**
- `api/services/health_service.py` — add dataclass at top of file

**Implementation notes:**
```python
@dataclass
class HealthCheckResult:
    name: str                    # e.g. "Signal Decay"
    score: float                 # 0-100
    status: str                  # "PASS", "WARN", "FAIL", "UNAVAILABLE"
    explanation: str             # Human-readable: "Signal autocorrelation is 0.08, below the 0.15 threshold"
    methodology: str             # "Measures lag-1 autocorrelation of trading signals. Low autocorrelation indicates signals aren't stale."
    thresholds: Dict[str, float] # {"pass": 0.15, "warn": 0.25, "fail": 0.40}
    severity: str                # "critical", "standard", "informational"
    data_available: bool         # False if check couldn't run due to missing data
    raw_value: Optional[float]   # The actual measured value (e.g., 0.08 for autocorrelation)
```

**Verify:**
```bash
python -c "from quant_engine.api.services.health_service import HealthCheckResult; print('Dataclass OK')"
```

---

### T2: Refactor all _check_* methods to return HealthCheckResult

**What:** Update all 15 check methods in health_service.py and 6 in data_helpers.py to return HealthCheckResult instead of plain dicts. Replace all hardcoded score=50.0 fallbacks with status="UNAVAILABLE", data_available=False.

**Files:**
- `api/services/health_service.py` — all 15 _check_* methods
- `api/services/data_helpers.py` — all 6 _check_* functions

**Implementation notes:**
- For each check, add:
  - `methodology` string explaining WHAT it measures and WHY it matters
  - `thresholds` dict showing the exact cutoff values
  - `severity` classification: critical (data integrity, signal quality), standard (risk, execution), informational (governance)
- When data is unavailable, return `HealthCheckResult(score=0, status="UNAVAILABLE", data_available=False, explanation="Trade history not available — run a backtest first")`
- Remove ALL hardcoded score=50.0 fallbacks

**Verify:**
```bash
python -c "
from quant_engine.api.services.health_service import HealthService
svc = HealthService()
# Should not crash and should return HealthCheckResult objects
print('HealthService instantiated OK')
"
```

---

### T3: Implement severity-weighted domain scoring

**What:** Replace simple average domain scoring with severity-weighted scoring that excludes UNAVAILABLE checks.

**Files:**
- `api/services/health_service.py` — modify `_domain_score()` and `compute_comprehensive_health()`

**Implementation notes:**
- Severity weights: critical=3.0, standard=1.0, informational=0.5
- Domain score = weighted_sum(available_scores) / sum(available_weights)
- If ALL checks in a domain are UNAVAILABLE, domain score = None (not 0, not 50)
- Overall score = weighted mean of available domain scores only
- Add `available_check_count` and `total_check_count` to domain output

**Verify:**
```bash
python -c "
# Verify UNAVAILABLE checks don't inflate scores
# If all checks are unavailable, overall should be None, not 50
print('Scoring logic updated')
"
```

---

### T4: Add health history storage

**What:** Store health scores with timestamps in a SQLite table so the UI can show trends.

**Files:**
- `api/services/health_service.py` — add `save_health_snapshot()` and `get_health_history()`
- `api/routers/system_health.py` — add `/health/history` endpoint

**Implementation notes:**
- Table: `health_history (timestamp TEXT, overall_score REAL, domain_scores TEXT, check_results TEXT)`
- Store JSON blob of full results
- Keep last 30 snapshots (configurable)
- Save snapshot every time `/health/detailed` is called
- History endpoint returns list of {timestamp, overall_score, domain_scores}

**Verify:**
```bash
python -c "
import sqlite3, json
# Verify table exists after first health check
print('Health history storage ready')
"
```

---

### T5: Update /health/detailed API response format

**What:** Enrich the API response with methodology, thresholds, and scoring transparency.

**Files:**
- `api/routers/system_health.py` — modify response format
- `api/schemas/health.py` — add Pydantic response models (create if needed)

**Implementation notes:**
- Response structure:
```json
{
  "overall_score": 72.5,
  "overall_methodology": "Weighted average of 5 domains: Data Integrity (25%), Signal Quality (25%), Risk Management (20%), Execution Quality (15%), Model Governance (15%). Only checks with available data are scored.",
  "domains": {
    "data_integrity": {
      "score": 85.0,
      "weight": 0.25,
      "checks_available": 3,
      "checks_total": 3,
      "checks": [
        {
          "name": "Signal Decay",
          "score": 90.0,
          "status": "PASS",
          "explanation": "Signal autocorrelation is 0.08, well below the 0.15 decay threshold",
          "methodology": "Measures lag-1 autocorrelation of trade signals...",
          "thresholds": {"pass": 0.15, "warn": 0.25},
          "severity": "critical",
          "raw_value": 0.08
        }
      ]
    }
  },
  "history": [...]
}
```

**Verify:**
```bash
# Start server and hit endpoint
curl -s http://localhost:8000/api/health/detailed | python -m json.tool | head -30
```

---

### T6: Update frontend SystemHealthPage with methodology panels

**What:** Show scoring methodology, thresholds, raw values, and trend chart in the health dashboard.

**Files:**
- `frontend/src/pages/SystemHealthPage.tsx` — major update
- `frontend/src/components/charts/LineChart.tsx` — reuse for health history trend

**Implementation notes:**
- Add "How Scores Are Calculated" expandable panel at top of page
- For each check, show:
  - Score with color coding
  - Status badge (PASS/WARN/FAIL/UNAVAILABLE)
  - Raw measured value (e.g., "Autocorrelation: 0.08")
  - Threshold bar: visual indicator showing where the raw value falls relative to pass/warn/fail thresholds
  - Expandable methodology text
- UNAVAILABLE checks shown with gray badge and "Data not yet available" message (NOT hidden)
- Add health trend mini-chart (sparkline of last 30 overall scores)
- Show domain weights next to domain names ("Data Integrity (25%)")

**Verify:**
- Manual: Navigate to /health page, verify methodology panels expand, UNAVAILABLE checks show correctly, trend chart appears

---

### T7: Verify end-to-end health transparency

**What:** Integration test confirming the full pipeline works and scores are honest.

**Files:**
- `tests/test_health_transparency.py` — new test file

**Implementation notes:**
- Test cases:
  1. `test_unavailable_checks_not_scored` — When no backtest data exists, health score should be None or reflect only available checks, NOT inflated by 50.0 defaults
  2. `test_methodology_strings_present` — Every HealthCheckResult has non-empty methodology
  3. `test_severity_weighting` — Critical check failure drops score more than informational check failure
  4. `test_health_history_stored` — After calling detailed health, history table has a new row
  5. `test_api_response_has_methodology` — /health/detailed response includes methodology fields

**Verify:**
```bash
python -m pytest tests/test_health_transparency.py -v
```

---

## Validation

### Acceptance criteria
1. No health check returns hardcoded score=50.0 — all unavailable checks return status="UNAVAILABLE"
2. Every check has a non-empty methodology string explaining what it measures
3. Every check has a thresholds dict showing pass/warn/fail cutoffs
4. Domain scores exclude unavailable checks from calculation
5. UI shows methodology panels that expand to explain each score
6. Health history is stored and a trend chart is visible
7. A system with no backtest data shows honest scores (mostly UNAVAILABLE), not inflated 50s

### Rollback plan
- Revert health_service.py and data_helpers.py to previous versions
- Frontend changes are additive (methodology panels) — can be hidden via CSS
