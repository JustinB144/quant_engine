# Feature Spec: Config Cleanup — Fix Dead Flags, Empty Maps, and Contradictions

> **Status:** Approved
> **Author:** justin
> **Date:** 2026-02-23
> **Estimated effort:** ~4 hours across 4 tasks

---

## Why

`config.py` has several problems that cause silent failures: `GICS_SECTORS` is an empty dict which means `MAX_SECTOR_EXPOSURE = 0.10` is never enforced (portfolio optimizer silently skips sector constraints), `OPTIONMETRICS_ENABLED = True` but no code uses it, `ALMGREN_CHRISS_ENABLED = True` but it's not wired into execution, and `REGIME_2_TRADE_ENABLED = False` is a hardcoded override rather than an adaptive threshold. These create a gap between what the config advertises and what actually runs.

## What

Clean config.py so every flag either works or is explicitly marked as not-yet-implemented. Populate or remove `GICS_SECTORS`. Document which flags are active vs planned. Done means: every enabled flag has corresponding functional code, and disabled/empty flags have clear comments explaining their status.

## Constraints

### Must-haves
- GICS_SECTORS either populated from a data source or MAX_SECTOR_EXPOSURE disabled with clear comment
- OPTIONMETRICS_ENABLED set to False with TODO comment (no code uses it)
- ALMGREN_CHRISS_ENABLED set to False with TODO comment (not wired into execution)
- Every config constant has a comment explaining: what it does, whether it's active, and what depends on it

### Must-nots
- Do NOT remove config constants (downstream code may reference them)
- Do NOT change values that affect active functionality
- Do NOT populate GICS_SECTORS with fake data

## Tasks

### T1: Audit every config flag for active usage

**What:** Grep every config constant to confirm it's actually used by running code.

**Files:**
- `config.py` — add comments with usage status

**Implementation notes:**
- For each constant, grep codebase for imports/usage
- Mark each as: ACTIVE (used by running code), PLACEHOLDER (defined but not functionally wired), or DEPRECATED
- Add header comment block at top of config.py explaining the status system

**Verify:**
```bash
python -c "from quant_engine.config import *; print('All config imports OK')"
```

---

### T2: Fix GICS_SECTORS and sector exposure enforcement

**What:** Either populate GICS_SECTORS or disable sector constraints with clear documentation.

**Files:**
- `config.py` — update GICS_SECTORS and MAX_SECTOR_EXPOSURE
- `risk/portfolio_optimizer.py` — add warning when sector constraints skipped

**Implementation notes:**
- Since WRDS Compustat access exists, add a helper function to populate GICS_SECTORS from the WRDS comp.company table
- If WRDS not available, set `GICS_SECTORS = {}` with comment:
  ```python
  # PLACEHOLDER: Populate via `run_wrds_daily_refresh.py --gics` or manually.
  # When empty, MAX_SECTOR_EXPOSURE constraint is NOT enforced in portfolio_optimizer.
  ```
- In portfolio_optimizer.py, when GICS_SECTORS is empty, log WARNING:
  ```python
  logger.warning("GICS_SECTORS is empty — sector exposure constraint (%.0f%%) is NOT enforced", MAX_SECTOR_EXPOSURE * 100)
  ```

**Verify:**
```bash
python -c "from quant_engine.config import GICS_SECTORS, MAX_SECTOR_EXPOSURE; print(f'GICS populated: {len(GICS_SECTORS) > 0}, Max sector: {MAX_SECTOR_EXPOSURE}')"
```

---

### T3: Disable unused feature flags with documentation

**What:** Set OPTIONMETRICS_ENABLED and ALMGREN_CHRISS_ENABLED to False with clear TODO comments.

**Files:**
- `config.py` — update flags

**Implementation notes:**
```python
# PLACEHOLDER — OptionMetrics IV surface data integration.
# Set to True once api/routers/iv_surface.py has a working /iv-surface/heston endpoint
# and data loader merges OptionMetrics surface into OHLCV panels.
OPTIONMETRICS_ENABLED = False

# PLACEHOLDER — Almgren-Chriss optimal execution model.
# Set to True once backtest/execution.py wires AC model into trade simulation.
# Currently, the AC model exists in code but is not called during backtests.
ALMGREN_CHRISS_ENABLED = False
```

**Verify:**
```bash
python -c "from quant_engine.config import OPTIONMETRICS_ENABLED, ALMGREN_CHRISS_ENABLED; assert not OPTIONMETRICS_ENABLED; assert not ALMGREN_CHRISS_ENABLED; print('Placeholder flags disabled')"
```

---

### T4: Expose active config to /api/config endpoint with status annotations

**What:** Make the /api/config endpoint return config values with active/placeholder status so the UI can show which features are actually running.

**Files:**
- `api/routers/config_mgmt.py` — enhance response

**Implementation notes:**
- Response should include:
```json
{
  "regime": {
    "model_type": {"value": "hmm", "status": "active"},
    "ensemble_enabled": {"value": true, "status": "active"},
    "jump_model_enabled": {"value": true, "status": "active"}
  },
  "risk": {
    "max_sector_exposure": {"value": 0.10, "status": "inactive", "reason": "GICS_SECTORS is empty"},
    "almgren_chriss": {"value": false, "status": "placeholder"}
  },
  "data": {
    "wrds_enabled": {"value": true, "status": "active"},
    "optionmetrics": {"value": false, "status": "placeholder"}
  }
}
```

**Verify:**
```bash
curl -s http://localhost:8000/api/config | python -m json.tool | head -20
```

---

## Validation

### Acceptance criteria
1. Every config flag has a comment explaining its status (active/placeholder/deprecated)
2. OPTIONMETRICS_ENABLED and ALMGREN_CHRISS_ENABLED are False
3. Portfolio optimizer logs WARNING when sector constraints skipped
4. /api/config returns status annotations for feature flags
5. No config imports cause ImportError

### Rollback plan
- Config flag changes (True→False) are trivially reversible
- Comments are documentation-only changes
