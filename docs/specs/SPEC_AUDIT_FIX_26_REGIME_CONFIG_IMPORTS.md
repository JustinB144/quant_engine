# SPEC_AUDIT_FIX_26: Regime Config Wiring & Import Brittleness Fixes

**Priority:** MEDIUM/LOW — ShockVector hardcodes schema version and changepoint threshold instead of using config; regime package import requires scipy unconditionally.
**Scope:** `regime/shock_vector.py`, `regime/__init__.py`, `regime/bocpd.py`
**Estimated effort:** 1.5 hours
**Depends on:** Nothing
**Blocks:** Nothing

---

## Context

Two config/import issues were identified. First (F-05/P2), `compute_shock_vectors()` at shock_vector.py:475 hardcodes `schema_version="1.0"` instead of using `SHOCK_VECTOR_SCHEMA_VERSION` from config.py:296. Meanwhile, `detect_with_shock_context()` at detector.py:793 correctly uses the config constant. Additionally, `is_shock_event()` at shock_vector.py:135 uses a hardcoded default of `0.50` for the changepoint threshold, while `BOCPD_CHANGEPOINT_THRESHOLD` is imported in detector.py:42 but never passed through to this method. Second (F-08/P3), `regime/__init__.py:3` unconditionally imports `BOCPDDetector` from `bocpd.py`, which unconditionally imports `scipy.special.gammaln` at line 33. This means importing the regime package (`from regime import RegimeDetector`) transitively requires scipy, even if the consumer never uses BOCPD.

---

## Tasks

### T1: Wire Config Constants Into compute_shock_vectors and is_shock_event

**Problem:** `shock_vector.py:475` hardcodes `schema_version="1.0"` while `config.py:296` defines `SHOCK_VECTOR_SCHEMA_VERSION`. `shock_vector.py:135` defaults `changepoint_threshold=0.50` while `BOCPD_CHANGEPOINT_THRESHOLD` exists in config but is never threaded to this call site.

**File:** `regime/shock_vector.py`

**Implementation:**
1. Import and use `SHOCK_VECTOR_SCHEMA_VERSION` in `compute_shock_vectors()`:
   ```python
   from ..config import SHOCK_VECTOR_SCHEMA_VERSION

   # In compute_shock_vectors(), line 474-475:
   sv = ShockVector(
       schema_version=SHOCK_VECTOR_SCHEMA_VERSION,  # Was hardcoded "1.0"
       timestamp=ts,
       ...
   )
   ```
2. Add `changepoint_threshold` as a parameter to `compute_shock_vectors()` with a config-driven default:
   ```python
   def compute_shock_vectors(
       ohlcv: pd.DataFrame,
       regime_series: Optional[pd.Series] = None,
       regime_confidence_series: Optional[pd.Series] = None,
       ticker: str = "",
       vol_lookback: int = 20,
       changepoint_threshold: Optional[float] = None,  # NEW
   ) -> Dict[pd.Timestamp, ShockVector]:
       if changepoint_threshold is None:
           try:
               from ..config import BOCPD_CHANGEPOINT_THRESHOLD
               changepoint_threshold = BOCPD_CHANGEPOINT_THRESHOLD
           except ImportError:
               changepoint_threshold = 0.50
   ```
3. Pass the threshold to `is_shock_event()` calls downstream:
   ```python
   sv.is_shock_event(changepoint_threshold=changepoint_threshold)
   ```

**Acceptance:** Changing `SHOCK_VECTOR_SCHEMA_VERSION` in config changes the schema version in ShockVectors produced by both `detect_with_shock_context()` and `compute_shock_vectors()`. Changing `BOCPD_CHANGEPOINT_THRESHOLD` changes the shock event detection threshold.

---

### T2: Make BOCPD Import Conditional in regime/__init__.py

**Problem:** `regime/__init__.py:3` unconditionally imports `BOCPDDetector` from `bocpd.py`, which imports `scipy.special.gammaln` at module level. Any consumer importing the regime package (e.g., `from regime import RegimeDetector`) transitively requires scipy, even if they don't use BOCPD.

**Files:** `regime/__init__.py`, `regime/bocpd.py`

**Implementation:**

**Option A (Recommended): Lazy import in __init__.py:**
```python
# regime/__init__.py

# Core imports (no heavy dependencies)
from .confidence_calibrator import ConfidenceCalibrator
from .consensus import RegimeConsensus
from .correlation import CorrelationRegimeDetector
from .detector import (
    RegimeDetector,
    RegimeOutput,
    detect_regimes_batch,
    validate_hmm_observation_features,
)
from .hmm import GaussianHMM, HMMFitResult
from .jump_model import StatisticalJumpModel, JumpModelResult
from .jump_model_pypi import PyPIJumpModel
from .online_update import OnlineRegimeUpdater
from .shock_vector import ShockVector, ShockVectorValidator, compute_shock_vectors
from .uncertainty_gate import UncertaintyGate

# BOCPD requires scipy — import lazily to avoid hard scipy dependency
# for consumers that don't use change-point detection.
def __getattr__(name):
    """Lazy import for optional BOCPD components."""
    if name in ("BOCPDDetector", "BOCPDResult", "BOCPDBatchResult"):
        from .bocpd import BOCPDDetector, BOCPDResult, BOCPDBatchResult
        globals()["BOCPDDetector"] = BOCPDDetector
        globals()["BOCPDResult"] = BOCPDResult
        globals()["BOCPDBatchResult"] = BOCPDBatchResult
        return globals()[name]
    raise AttributeError(f"module 'regime' has no attribute {name!r}")

__all__ = [
    "BOCPDDetector",
    "BOCPDResult",
    "BOCPDBatchResult",
    "ConfidenceCalibrator",
    "CorrelationRegimeDetector",
    "RegimeConsensus",
    "RegimeDetector",
    "RegimeOutput",
    "GaussianHMM",
    "HMMFitResult",
    "OnlineRegimeUpdater",
    "StatisticalJumpModel",
    "JumpModelResult",
    "PyPIJumpModel",
    "ShockVector",
    "ShockVectorValidator",
    "UncertaintyGate",
    "compute_shock_vectors",
    "detect_regimes_batch",
    "validate_hmm_observation_features",
]
```

**Option B (Alternative): Conditional import with try/except in bocpd.py:**
```python
# bocpd.py line 33:
try:
    from scipy.special import gammaln
except ImportError:
    gammaln = None
    import logging
    logging.getLogger(__name__).warning(
        "scipy not available — BOCPD features disabled"
    )
```
Then guard all methods that use `gammaln` with an availability check.

Option A is preferred because it keeps BOCPD's internal code clean and moves the optionality to the package boundary.

**Acceptance:** `from regime import RegimeDetector` works without scipy installed. `from regime import BOCPDDetector` raises `ImportError` with a clear message if scipy is missing.

---

## Verification

- [ ] Run `pytest tests/ -k "regime or shock_vector"` — all pass
- [ ] Verify ShockVector schema_version matches config constant
- [ ] Verify changing BOCPD_CHANGEPOINT_THRESHOLD affects is_shock_event()
- [ ] Verify `from regime import RegimeDetector` works without scipy (mock scipy as unavailable)
- [ ] Verify `from regime import BOCPDDetector` triggers lazy import and requires scipy
