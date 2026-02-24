# Feature Spec: Fix All UI Errors, Warnings, and Log-Visible Issues

> **Status:** Approved
> **Author:** justin
> **Date:** 2026-02-23
> **Estimated effort:** ~8 hours across 6 tasks

---

## Why

The system logs show persistent errors and warnings that undermine trust: "boolean value of NA is ambiguous" (regime detection), "No data loaded — check data sources" (predict jobs), and other runtime warnings. Even after partial fixes, the root causes persist because they're symptoms of deeper issues — NaN propagation in the regime pipeline, missing data source fallbacks, and silent exception swallowing. The Logs page shows these errors in real-time, making the system look unreliable.

## What

Eliminate all recurring errors and warnings from the logs. Every error that can't be prevented should have a clear, actionable message. Every warning should explain what's wrong and how to fix it. Done means: a fresh system startup shows zero ERROR-level log entries, and WARNING-level entries all include remediation instructions.

## Constraints

### Must-haves
- Zero recurring ERROR-level log entries during normal operation
- All WARNING entries include: what happened, why, and how to fix
- Exception handlers log the full context (ticker, timeframe, data shape) not just the exception message
- Log messages use structured format: `logger.warning("Action failed: %s | ticker=%s | reason=%s", ...)`

### Must-nots
- Do NOT suppress errors by catching and ignoring them
- Do NOT lower log levels to hide problems (no ERROR → WARNING downgrades without fixing root cause)
- Do NOT remove the Logs page or filter out errors client-side

### Out of scope
- External log aggregation (ELK, Datadog, etc.)
- Log rotation and retention policies
- Email/Slack alerting on errors

## Tasks

### T1: Fix all NaN-related warnings in regime detection pipeline

**What:** Eliminate NaN propagation at every stage of the regime pipeline.

**Files:**
- `regime/detector.py` — NaN guards in _hmm_detect, _jump_detect, _rule_detect, detect_ensemble
- `regime/hmm.py` — NaN guard after standardization in build_hmm_observation_matrix
- `api/services/data_helpers.py` — Already partially fixed, verify completeness

**Implementation notes:**
- Root causes of NaN in regime probabilities:
  1. `build_hmm_observation_matrix()` standardization: if a feature column has zero variance, z-score = 0/0 = NaN
     - Fix: after standardization, check for zero-variance columns and fill with 0.0
  2. `_hmm_detect()` probability normalization: `probs.div(probs.sum(axis=1).replace(0, 1), axis=0)` can produce NaN if probs contain NaN before normalization
     - Fix: `probs = probs.fillna(0.0)` BEFORE normalization, then normalize, then fillna again
  3. `_rule_detect()` confidence calculation uses Hurst exponent which can be NaN
     - Fix: `hurst = features.get("hurst", pd.Series(0.5, index=features.index)).fillna(0.5)`
  4. `detect_ensemble()` probability blending can propagate NaN from any sub-method
     - Fix: fillna(0.0) on each method's probabilities before blending

**Verify:**
```bash
python -c "
import warnings
warnings.filterwarnings('error')  # Turn warnings into errors
import pandas as pd, numpy as np
from quant_engine.regime.detector import RegimeDetector
# Create features with edge cases
idx = pd.date_range('2020-01-01', periods=600, freq='B')
np.random.seed(42)
features = pd.DataFrame({
    'Close': np.cumsum(np.random.randn(600)*0.02)+100,
    'High': np.cumsum(np.random.randn(600)*0.02)+101,
    'Low': np.cumsum(np.random.randn(600)*0.02)+99,
    'Open': np.cumsum(np.random.randn(600)*0.02)+100,
    'Volume': np.random.randint(1e6, 1e7, 600).astype(float),
}, index=idx)
features['ret_1d'] = features['Close'].pct_change()
features['vol_20d'] = features['ret_1d'].rolling(20).std()
features['natr'] = (features['High']-features['Low'])/features['Close']
features['sma_slope'] = features['Close'].rolling(20).mean().pct_change(5)
features['hurst'] = np.nan  # Intentional NaN
features['adx'] = 25.0
features = features.dropna(subset=['ret_1d'])
detector = RegimeDetector(method='hmm')
out = detector.detect_full(features)
assert not out.probabilities.isna().any().any(), 'NaN in probabilities'
assert not out.regime.isna().any(), 'NaN in regime'
print('All NaN guards working')
"
```

---

### T2: Improve "No data loaded" error with actionable remediation

**What:** Make the RuntimeError message tell the user exactly how to fix the problem.

**Files:**
- `api/orchestrator.py` — enhance error message
- `data/loader.py` — ensure skip reasons are captured

**Implementation notes:**
- Error message format:
  ```
  No data loaded — all 25 tickers were rejected.

  Diagnostics:
    WRDS_ENABLED=True, REQUIRE_PERMNO=True, years=2
    Cache: 47 daily files, 0 _1d.parquet files

  Top skip reasons:
    - 15 tickers: "permno unresolved" → Set REQUIRE_PERMNO=False in config.py or run run_wrds_daily_refresh.py
    - 8 tickers: "insufficient data (< 500 bars)" → Increase years parameter or check data sources
    - 2 tickers: "quality gate failed" → Check MAX_ZERO_VOLUME_FRACTION in config.py

  To debug: Set verbose=True in load_universe() or check logs for per-ticker details.
  ```
- `load_universe()` should return skip reasons alongside data (add to module-level tracker)
- `orchestrator.py` reads the skip tracker and formats the error message

**Verify:**
```bash
python -c "
from quant_engine.api.orchestrator import PipelineOrchestrator
orch = PipelineOrchestrator()
try:
    orch.load_and_prepare(tickers=['NONEXISTENT_TICKER_XYZ'], years=1)
except RuntimeError as e:
    assert 'Diagnostics' in str(e) or 'rejected' in str(e), f'Bad error: {e}'
    print('Error message is actionable')
"
```

---

### T3: Add structured logging with context to all exception handlers

**What:** Every except block that logs should include the full context (what was being done, with what data).

**Files:**
- `api/services/health_service.py` — all catch blocks in _check_* methods
- `api/services/data_helpers.py` — all catch blocks
- `regime/detector.py` — HMM/Jump model fallback catches
- `data/loader.py` — WRDS/cache fallback catches

**Implementation notes:**
- Pattern to follow:
  ```python
  # BAD (current):
  except Exception as e:
      logger.warning("Check failed: %s", e)
      return {"score": 50.0}

  # GOOD (target):
  except Exception as e:
      logger.warning(
          "Health check '%s' failed: %s | model_dir=%s | trade_count=%d",
          check_name, e, model_dir, len(trades),
          exc_info=True,
      )
      return HealthCheckResult(
          name=check_name,
          score=0,
          status="UNAVAILABLE",
          explanation=f"Check could not run: {e}",
          data_available=False,
      )
  ```
- Use `exc_info=True` for ERROR level, omit for WARNING level
- Include relevant variables in log message (ticker, permno, data shape, timeframe)

**Verify:**
```bash
# Check that health service logs include context
python -c "
import logging
logging.basicConfig(level=logging.WARNING)
from quant_engine.api.services.health_service import HealthService
svc = HealthService()
# Run checks — any warnings should now include context
print('Structured logging in place')
"
```

---

### T4: Add startup health check that surfaces config issues

**What:** On server startup, run a quick validation of config settings and log any issues.

**Files:**
- `api/main.py` (or wherever FastAPI app is created) — add startup validation
- `config.py` — add `validate_config()` function

**Implementation notes:**
- `validate_config()` checks:
  1. `GICS_SECTORS` is empty → WARNING: "Sector exposure constraints disabled"
  2. `OPTIONMETRICS_ENABLED` is True but no OptionMetrics code → WARNING
  3. `WRDS_ENABLED` is True but WRDS credentials missing → WARNING
  4. `DATA_CACHE_DIR` doesn't exist or is empty → WARNING: "No cached data"
  5. `MODEL_DIR` doesn't exist or has no models → WARNING: "No trained models"
  6. `REGIME_MODEL_TYPE` not in ("hmm", "rule", "jump") → ERROR
- Run on startup, log all findings
- Add `/api/config/validate` endpoint that returns the same findings as JSON

**Verify:**
```bash
python -c "from quant_engine.config import validate_config; issues = validate_config(); print(f'{len(issues)} config issues found')"
```

---

### T5: Add log level filtering and structured log format

**What:** Improve log output format and add log level configuration.

**Files:**
- `api/main.py` or `run_server.py` — configure logging format

**Implementation notes:**
- Structured log format:
  ```
  2026-02-23 08:21:01 | ERROR | quant_engine.api.jobs.runner | Job 0d0c72eb failed | ticker=AAPL | reason=No data loaded
  ```
- Add `LOG_LEVEL` config constant (default: "INFO")
- Add `LOG_FORMAT` config: "structured" (default) or "json" (for machine parsing)
- Ensure all loggers use `getLogger(__name__)` (most already do)

**Verify:**
```bash
python -c "
import logging
from quant_engine.config import LOG_LEVEL
logging.basicConfig(level=getattr(logging, LOG_LEVEL, 'INFO'))
print(f'Log level: {LOG_LEVEL}')
"
```

---

### T6: Test that common operations produce zero errors

**What:** Integration test that runs common operations and asserts no ERROR-level log entries.

**Files:**
- `tests/test_zero_errors.py` — new test file

**Implementation notes:**
- Capture log output during test execution
- Run: health check, regime detection (if cache exists), config validation
- Assert: zero ERROR entries (WARNING is OK)
- If cache exists, run: data loading for a single ticker, feature computation
- If no cache, verify graceful degradation (no crashes, clear messages)

**Verify:**
```bash
python -m pytest tests/test_zero_errors.py -v
```

---

## Validation

### Acceptance criteria
1. Server startup shows zero ERROR entries in logs
2. Regime detection on cached data produces zero warnings about NA/NaN
3. "No data loaded" error includes per-ticker skip reasons and remediation steps
4. All exception handlers log full context (ticker, data shape, what was being attempted)
5. Config validation surfaces all known issues at startup
6. Logs page shows clean output during normal operation

### Rollback plan
- Log format changes are non-functional — revert by changing format string
- Structured logging is additive — old log messages still work
- validate_config() is new — remove call from startup if issues
