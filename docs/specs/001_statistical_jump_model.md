# Feature Spec: Replace HMM with Production Jump Model Package

> **Status:** Approved
> **Author:** justin
> **Date:** 2026-02-23
> **Estimated effort:** ~12 hours across 8 tasks

---

## Why

The current HMM regime detector suffers from known weaknesses: sensitivity to model mis-specification (real financial returns aren't Gaussian), implicit persistence control via sticky transition priors (indirect, hard to tune), and frequent false regime switches that require post-hoc duration smoothing. The codebase already has a custom `StatisticalJumpModel` in `regime/jump_model.py` (243 lines), but it's a basic implementation that lacks the refinements of the peer-reviewed `jumpmodels` PyPI package (arXiv 2402.05272) — specifically: continuous (probabilistic) jump models, sparse feature selection, cross-validated lambda tuning, and online prediction support.

## What

Replace the custom `StatisticalJumpModel` with the `jumpmodels` PyPI package as the primary regime detection backend, while keeping the HMM and rule-based methods as ensemble fallbacks. Done means: the system defaults to `REGIME_MODEL_TYPE = "jump"`, all 12 consumer modules work without changes (same `RegimeOutput` interface), backtests show equal or better Sharpe and max-drawdown vs the HMM baseline, and the ensemble mode uses the new Jump Model instead of the old custom one.

## Constraints

### Must-haves
- `RegimeOutput` dataclass interface is unchanged — no downstream consumer modifications required
- `regime_features()` output columns are identical (regime, regime_confidence, regime_prob_*, etc.)
- `build_hmm_observation_matrix()` is reused for feature preparation (already standardizes 11 features)
- `map_raw_states_to_regimes()` and `map_raw_states_to_regimes_stable()` still map raw K states → 4 canonical regimes
- Config-driven: all new hyperparameters live in `config.py` with `REGIME_JUMP_*` prefix
- Feature flag: `REGIME_JUMP_USE_PYPI_PACKAGE = True` to toggle between old custom and new package
- Cross-validated lambda selection using time-series splits (not in-sample BIC)
- Online prediction support via `.predict_online()` for real-time API/dashboard use
- Ensemble voting still works with 3 methods (rule, HMM, jump)

### Must-nots
- Do NOT remove the existing `GaussianHMM` class or `hmm.py` — it stays as ensemble fallback
- Do NOT change the `RegimeOutput` dataclass fields or their types
- Do NOT modify any of the 12 consumer modules (run_train.py, run_backtest.py, autopilot/engine.py, etc.)
- Do NOT hard-delete the old `jump_model.py` — rename it to `jump_model_legacy.py` for reference

### Out of scope
- Sparse Jump Model feature selection (future spec — requires feature importance analysis first)
- Online lambda adaptation (future — requires regime change frequency monitoring)
- GPU acceleration for jump model fitting
- Replacing the rule-based detector

## Current State

### Key files
| File | Role | Notes |
|------|------|-------|
| `regime/detector.py` (522 lines) | Main entry point, dispatches to `_hmm_detect()`, `_jump_detect()`, `_rule_detect()`, `detect_ensemble()` | `_jump_detect()` at lines 202-251 uses custom `StatisticalJumpModel` |
| `regime/hmm.py` (563 lines) | `GaussianHMM` class, `build_hmm_observation_matrix()`, `map_raw_states_to_regimes()`, `select_hmm_states_bic()` | Keep entirely, reuse observation matrix builder and state mapper |
| `regime/jump_model.py` (243 lines) | Custom `StatisticalJumpModel` with basic DP segmentation | Will be renamed to `jump_model_legacy.py`; replaced by PyPI wrapper |
| `regime/__init__.py` | Exports `RegimeDetector`, `RegimeOutput` | May need to export new module |
| `config.py` (lines 140-162) | All `REGIME_*` constants | Add new `REGIME_JUMP_*` constants for PyPI package |
| `api/services/data_helpers.py` | `compute_regime_payload()` — consumes `RegimeOutput` | No changes needed (interface preserved) |
| `autopilot/engine.py` | Consumes `RegimeDetector` | No changes needed |
| `risk/covariance.py` | `compute_regime_covariance()` — uses regime series | No changes needed |

### Existing patterns to follow
- All regime config constants use `REGIME_` prefix in `config.py`
- Detector methods return `RegimeOutput` dataclass — never raw arrays
- Observation matrix is standardized via `build_hmm_observation_matrix()` before model fitting
- Raw states (0..K-1) are mapped to canonical regimes (0..3) via `map_raw_states_to_regimes()`
- Probabilities are aggregated from raw K states into 4 canonical columns: `regime_prob_0` through `regime_prob_3`
- Confidence = `probs.max(axis=1).clip(0, 1)`
- Uncertainty = normalized Shannon entropy of posterior probabilities
- Logging uses `logging.getLogger(__name__)` — no print statements
- Error handling: catch `(ValueError, RuntimeError, np.linalg.LinAlgError)` and fall back to rule-based

### Configuration (current relevant values)
```python
REGIME_MODEL_TYPE = "hmm"                    # Will change to "jump"
REGIME_JUMP_MODEL_ENABLED = True             # Already enabled for ensemble
REGIME_JUMP_PENALTY = 0.02                   # Will be replaced by CV-selected lambda
REGIME_EXPECTED_CHANGES_PER_YEAR = 4         # Keep as prior for lambda initialization
REGIME_ENSEMBLE_ENABLED = True               # Keep, uses new jump model
REGIME_ENSEMBLE_CONSENSUS_THRESHOLD = 2      # Keep
```

---

## Tasks

### T1: Install jumpmodels package and add to dependencies

**What:** Add `jumpmodels` to project dependencies and verify import works.

**Files:**
- `pyproject.toml` — add `jumpmodels>=0.1.1` to `[project.optional-dependencies] ml`

**Implementation notes:**
- The package requires numpy, pandas, scipy, scikit-learn (all already installed)
- Verify `from jumpmodels import JumpModel` succeeds
- Check compatibility with Python 3.10+ (the package requires >=3.8)

**Verify:**
```bash
pip install jumpmodels --break-system-packages
python -c "from jumpmodels import JumpModel; print('jumpmodels imported OK, version:', JumpModel.__module__)"
```

---

### T2: Add new config constants for PyPI Jump Model

**What:** Add all new `REGIME_JUMP_*` configuration constants to `config.py`.

**Files:**
- `config.py` — add new constants after existing REGIME_JUMP_* section (around line 155)

**Implementation notes:**
- New constants needed:
  ```python
  # PyPI jumpmodels package configuration
  REGIME_JUMP_USE_PYPI_PACKAGE = True          # Toggle: True=PyPI, False=legacy custom
  REGIME_JUMP_CV_FOLDS = 5                     # Time-series CV folds for lambda selection
  REGIME_JUMP_LAMBDA_RANGE = (0.005, 0.15)     # Search range for jump penalty
  REGIME_JUMP_LAMBDA_STEPS = 20                # Grid points for lambda search
  REGIME_JUMP_MAX_ITER = 50                    # Coordinate descent iterations
  REGIME_JUMP_TOL = 1e-6                       # Convergence tolerance
  REGIME_JUMP_USE_CONTINUOUS = True             # Continuous JM for soft probabilities
  REGIME_JUMP_MODE_LOSS_WEIGHT = 0.1           # Mode loss penalty (continuous JM)
  ```
- Keep existing `REGIME_JUMP_PENALTY` and `REGIME_EXPECTED_CHANGES_PER_YEAR` as fallbacks

**Verify:**
```bash
python -c "from quant_engine.config import REGIME_JUMP_USE_PYPI_PACKAGE, REGIME_JUMP_CV_FOLDS; print('Config OK:', REGIME_JUMP_USE_PYPI_PACKAGE, REGIME_JUMP_CV_FOLDS)"
```

---

### T3: Create new jump_model_pypi.py wrapper module

**What:** Create a wrapper around the `jumpmodels` PyPI package that conforms to the existing interface pattern (returns arrays/dataframes compatible with `map_raw_states_to_regimes()` and `RegimeOutput` construction).

**Files:**
- `regime/jump_model.py` → rename to `regime/jump_model_legacy.py`
- `regime/jump_model_pypi.py` — new file (the main wrapper)

**Implementation notes:**
- The wrapper class `PyPIJumpModel` should:
  1. Accept the same feature matrix X as `GaussianHMM.fit()` (output of `build_hmm_observation_matrix()`)
  2. Use `jumpmodels.JumpModel` (continuous mode if `REGIME_JUMP_USE_CONTINUOUS` is True)
  3. Implement time-series cross-validation for lambda selection:
     - Split data into `REGIME_JUMP_CV_FOLDS` temporal folds (no shuffling — walk-forward)
     - For each lambda in `np.linspace(*REGIME_JUMP_LAMBDA_RANGE, REGIME_JUMP_LAMBDA_STEPS)`:
       - Fit on train fold, predict on validation fold
       - Score by Sharpe ratio of a simple regime-following strategy (long in bull, flat in bear/high-vol)
     - Select lambda with highest average validation Sharpe
  4. Return a result object with:
     - `regime_sequence: np.ndarray` — shape (T,), integer state labels
     - `regime_probs: np.ndarray` — shape (T, K), soft probabilities
     - `centroids: np.ndarray` — shape (K, D)
     - `jump_penalty: float` — selected lambda
     - `converged: bool`
  5. Implement `.predict_online(x_new)` for single-step prediction (delegates to package's `.predict_online()`)
  6. Handle edge cases: insufficient data (<200 rows → fall back to legacy), all-NaN features, convergence failure
- Reuse the existing `JumpModelResult` dataclass from `jump_model_legacy.py` for the return type

**Verify:**
```bash
python -c "
from quant_engine.regime.jump_model_pypi import PyPIJumpModel
import numpy as np
X = np.random.randn(500, 4)
model = PyPIJumpModel(n_regimes=4)
result = model.fit(X)
print('States shape:', result.regime_sequence.shape)
print('Probs shape:', result.regime_probs.shape)
print('Lambda:', result.jump_penalty)
print('Converged:', result.converged)
"
```

---

### T4: Wire PyPI Jump Model into RegimeDetector._jump_detect()

**What:** Update `_jump_detect()` in `detector.py` to use the new `PyPIJumpModel` wrapper when `REGIME_JUMP_USE_PYPI_PACKAGE` is True, falling back to legacy otherwise.

**Files:**
- `regime/detector.py` — modify `_jump_detect()` method (lines 202-251)

**Implementation notes:**
- At the top of `_jump_detect()`:
  ```python
  from quant_engine.config import REGIME_JUMP_USE_PYPI_PACKAGE
  if REGIME_JUMP_USE_PYPI_PACKAGE:
      from .jump_model_pypi import PyPIJumpModel
      # ... use PyPIJumpModel
  else:
      from .jump_model_legacy import StatisticalJumpModel
      # ... existing code path
  ```
- The observation matrix is already built via `build_hmm_observation_matrix(features)` — reuse it
- Map raw states to canonical regimes via `map_raw_states_to_regimes(result.regime_sequence, features)`
- Aggregate probabilities into 4 canonical columns (same pattern as `_hmm_detect()` lines 176-181)
- Set `transition_matrix=None` since Jump Models don't produce one (this is already handled — line 510 in `data_helpers.py` falls back to `np.eye(4)`)
- Compute uncertainty via `self.get_regime_uncertainty(probs)`
- Wrap in `try/except (ValueError, RuntimeError, ImportError)` → fall back to `self._rule_detect(features)`

**Verify:**
```bash
python -c "
from quant_engine.regime.detector import RegimeDetector
import pandas as pd, numpy as np
# Create minimal test features
np.random.seed(42)
idx = pd.date_range('2020-01-01', periods=600, freq='B')
features = pd.DataFrame({
    'Close': np.cumsum(np.random.randn(600) * 0.02) + 100,
    'High': np.cumsum(np.random.randn(600) * 0.02) + 101,
    'Low': np.cumsum(np.random.randn(600) * 0.02) + 99,
    'Open': np.cumsum(np.random.randn(600) * 0.02) + 100,
    'Volume': np.random.randint(1e6, 1e7, 600),
}, index=idx)
features['ret_1d'] = features['Close'].pct_change()
features['vol_20d'] = features['ret_1d'].rolling(20).std()
features['natr'] = (features['High'] - features['Low']) / features['Close']
features['sma_slope'] = features['Close'].rolling(20).mean().pct_change(5)
features['hurst'] = 0.5
features['adx'] = 25.0
features = features.dropna()

detector = RegimeDetector(method='jump')
out = detector.detect_full(features)
print('Model type:', out.model_type)
print('Regime shape:', out.regime.shape)
print('Probs shape:', out.probabilities.shape)
print('Unique regimes:', sorted(out.regime.unique()))
print('Has uncertainty:', out.uncertainty is not None)
"
```

---

### T5: Update default config to use Jump Model as primary

**What:** Change `REGIME_MODEL_TYPE` default from `"hmm"` to `"jump"` so the system uses the Jump Model by default. Keep ensemble enabled.

**Files:**
- `config.py` — change `REGIME_MODEL_TYPE = "hmm"` to `REGIME_MODEL_TYPE = "jump"`

**Implementation notes:**
- This is a one-line change but has system-wide impact
- The ensemble mode (`REGIME_ENSEMBLE_ENABLED = True`) will still use all three methods for voting
- When ensemble is disabled, the system will default to jump model alone
- All consumers go through `RegimeDetector.detect_full()` which dispatches based on this config value

**Verify:**
```bash
python -c "from quant_engine.config import REGIME_MODEL_TYPE; assert REGIME_MODEL_TYPE == 'jump', f'Expected jump, got {REGIME_MODEL_TYPE}'; print('Default is now jump model')"
```

---

### T6: Write unit tests for PyPI Jump Model wrapper

**What:** Comprehensive unit tests for the new `jump_model_pypi.py` module.

**Files:**
- `tests/test_jump_model_pypi.py` — new test file

**Implementation notes:**
- Test cases:
  1. `test_fit_basic` — fit on random data, verify output shapes and types
  2. `test_fit_returns_valid_regimes` — all regime labels in {0, ..., K-1}
  3. `test_fit_probs_sum_to_one` — each row of regime_probs sums to ~1.0
  4. `test_cv_lambda_selection` — verify CV selects a lambda within configured range
  5. `test_fallback_on_short_data` — <200 rows → falls back to legacy or raises cleanly
  6. `test_nan_handling` — features with NaN → handled without crash
  7. `test_predict_online_shape` — single-step prediction returns correct shape
  8. `test_regime_persistence` — with higher lambda, regimes are more persistent (fewer transitions)
  9. `test_continuous_vs_discrete` — continuous mode produces soft probabilities, discrete produces hard
  10. `test_canonical_mapping` — raw states correctly map to 4 canonical regimes via `map_raw_states_to_regimes()`
- Use `@pytest.mark.unit` marker
- Use `conftest.py` fixtures if a shared feature DataFrame fixture exists

**Verify:**
```bash
python -m pytest tests/test_jump_model_pypi.py -v --tb=short
```

---

### T7: Write integration test for full regime detection pipeline

**What:** End-to-end integration test that loads real cached data, runs the full pipeline with jump model, and verifies RegimeOutput compatibility with downstream consumers.

**Files:**
- `tests/test_regime_integration.py` — new or extend existing

**Implementation notes:**
- Test cases:
  1. `test_detect_full_jump_model` — full `detect_full()` with method="jump" on real-ish features
  2. `test_detect_ensemble_includes_jump` — ensemble mode includes jump model vote
  3. `test_regime_features_unchanged` — `regime_features()` output has same columns regardless of method
  4. `test_compute_regime_payload_jump` — `compute_regime_payload()` works with jump model output (tests the NA fix from earlier)
  5. `test_regime_covariance_with_jump` — `compute_regime_covariance()` works with jump-produced regime series
- Use `@pytest.mark.integration` marker
- If no cached data available, generate synthetic OHLCV with known regime structure (e.g., low-vol uptrend → high-vol crash → sideways)

**Verify:**
```bash
python -m pytest tests/test_regime_integration.py -v --tb=short -m integration
```

---

### T8: A/B comparison backtest — Jump Model vs HMM baseline

**What:** Run identical backtests using HMM and Jump Model, compare key metrics, and save results.

**Files:**
- `scripts/compare_regime_models.py` — new comparison script

**Implementation notes:**
- Script should:
  1. Load a representative universe (UNIVERSE_QUICK or 10-20 tickers)
  2. Run full pipeline twice: once with `REGIME_MODEL_TYPE="hmm"`, once with `"jump"`
  3. For each, run backtest with identical parameters (same horizon, same risk management settings)
  4. Compare and print table:
     - Sharpe ratio
     - Sortino ratio
     - Max drawdown
     - Win rate
     - Trades per year
     - Regime transition frequency (count of regime changes per year)
     - Average regime duration (days)
  5. Save comparison to `results/regime_model_comparison.json`
- This is a diagnostic script, not a test — it produces data for human decision-making
- If Jump Model is worse on any key metric, log a WARNING (don't fail)

**Verify:**
```bash
python scripts/compare_regime_models.py
# Check that results/regime_model_comparison.json exists and has both model entries
python -c "import json; d = json.load(open('results/regime_model_comparison.json')); print(json.dumps(d, indent=2))"
```

---

## Validation

### Acceptance criteria
1. `python -c "from jumpmodels import JumpModel"` succeeds
2. All existing tests pass: `python -m pytest tests/ -x --tb=short`
3. `RegimeDetector(method="jump").detect_full(features)` returns a valid `RegimeOutput` with all fields populated
4. `RegimeDetector(method="hmm").detect_full(features)` still works (HMM not broken)
5. `detect_ensemble()` uses the new PyPI jump model, not the legacy custom one
6. `compute_regime_payload()` in `data_helpers.py` handles jump model output without the "boolean value of NA is ambiguous" error
7. A/B backtest shows Jump Model Sharpe >= HMM Sharpe (or within 0.05)
8. A/B backtest shows Jump Model max drawdown <= HMM max drawdown (or within 1%)
9. Regime transition frequency is lower with Jump Model than HMM (the whole point of explicit persistence)

### Verification steps
```bash
# Full test suite
python -m pytest tests/ -x --tb=short -q

# Specific regime tests
python -m pytest tests/test_jump_model_pypi.py tests/test_regime_integration.py -v

# A/B comparison
python scripts/compare_regime_models.py

# Smoke test the API endpoint
python -c "
from quant_engine.api.services.data_helpers import compute_regime_payload
from pathlib import Path
from quant_engine.config import DATA_DIR
result = compute_regime_payload(DATA_DIR / 'cache')
print('Regime payload:', result['current_label'], result['current_probs'])
"
```

### Rollback plan
- Set `REGIME_JUMP_USE_PYPI_PACKAGE = False` in config.py → reverts to legacy custom jump model
- Set `REGIME_MODEL_TYPE = "hmm"` in config.py → reverts to HMM as primary
- Both are single-line config changes with zero code impact

---

## Notes

**Design alternative considered:** Using `jumpmodels.SparseJumpModel` for automatic feature selection. Decided to defer to a follow-up spec because: (a) we first need feature importance analysis to validate which of the 11 observation features matter, and (b) the Sparse JM adds another hyperparameter (L1 penalty) that needs tuning infrastructure.

**Why continuous mode:** The `jumpmodels` package supports both discrete (hard labels) and continuous (soft probabilities) modes. We use continuous because: the rest of the system relies heavily on `regime_prob_*` columns for weighted ensemble predictions, confidence scoring, and uncertainty quantification. Hard labels would lose this information.

**Lambda selection strategy:** The paper recommends optimizing lambda for downstream trading performance (Sharpe ratio), not statistical fit (BIC/AIC). Our CV implementation follows this by scoring each lambda candidate on a simple regime-following strategy's Sharpe. This directly optimizes what we care about.

**References:**
- arXiv 2402.05272: "Downside Risk Reduction Using Regime-Switching Signals"
- PyPI package: https://pypi.org/project/jumpmodels/
- GitHub: https://github.com/Yizhan-Oliver-Shu/jump-models
