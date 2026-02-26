# Feature Spec: Foundational Hardening — Truth Layer

> **Status:** Draft
> **Author:** Claude Opus 4
> **Date:** 2026-02-26
> **Estimated effort:** 85 hours across 6 tasks

---

## Why

The current quant engine executes backtests and produces predictions without explicit, locked-down preconditions. Critical execution assumptions (entry price type, return definition, label horizon) are implicit in code. Data quality checks exist but do NOT block backtest execution on corruption. Feature causality tags document intent but are never enforced at runtime. Null models have no baselines. Transaction costs are fixed but never stress-tested. Cache staleness is computed using `date.today()` instead of trading calendar, causing silent time shifts.

These gaps create systematic risk:
- **Data integrity**: Corrupted OHLCV data triggers only warnings, allowing downstream models to train on garbage
- **Leakage**: END_OF_DAY and RESEARCH_ONLY features are not runtime-checked; they can leak into live predictions
- **Overfitting**: No null model baselines (random, zero, momentum) to anchor performance evaluation
- **Cost blindness**: Execution cost assumptions are never swept; slippage models never stress-tested against regime changes
- **Cache rot**: `_cache_is_usable()` uses `date.today()` not trading calendar; on holidays/weekends, cache appears fresh when it's stale

**Impact**: Models trained on this foundation are fundamentally unvalidated. Performance metrics are systematically inflated. Production deployment carries catastrophic risk.

---

## What

Implement a "Truth Layer" that locks down foundational execution contracts before any modeling or backtesting begins:

1. **Global preconditions**: Add `RET_TYPE`, `PX_TYPE`, `LABEL_H`, `ENTRY_PRICE` to config; validate at engine init
2. **Data integrity preflight**: Build fail-fast system that blocks corrupt OHLCV from entering pipeline
3. **Leakage tripwires**: Enforce FEATURE_METADATA causality tags at runtime; detect time-shift leakage
4. **Null model baselines**: Compute random, zero, momentum baselines for every backtest
5. **Cost stress defaults**: Parameterize transaction cost multiplier; sweep [0.5x, 1.0x, 2.0x, 5.0x]
6. **Cache staleness fix**: Replace `date.today()` with last trading day via trading calendar

These changes are **non-invasive**—backward compatible with existing backtest/prediction pipelines. They create guardrails that catch corruption before it propagates.

---

## Constraints

### Must-haves

- Global preconditions stored in `config.py`, validated at engine initialization
- `assess_ohlcv_quality()` return blocks `load_universe()` on ANY failed check (not warnings-only)
- Runtime enforcement of FEATURE_METADATA["type"] at feature pipeline compute time
- Time-shift tripwire test: verify label data NOT available at prediction time
- Null baselines (random, zero, momentum) computed and stored alongside every backtest result
- Cache staleness computed from trading calendar, not `date.today()`
- All changes backward-compatible; existing configs continue to work with sensible defaults

### Must-nots

- Do NOT modify backtest engine logic; only add preflight checks
- Do NOT break existing pipelines; all changes must be additive or config-gated
- Do NOT require model retraining; the Truth Layer applies at execution time only
- Do NOT introduce circular dependencies (config → features → config)

### Out of scope

- Wallets / portfolio-level position limits (separate feature)
- Multi-asset correlation collapse scenarios (regime module enhancement)
- Microstructure realism (execution module enhancement)
- Sensitivity analysis for feature engineering choices (modeling improvement)

---

## Current State

### Key files

| Module | File | Role | Status |
|--------|------|------|--------|
| Config | `/sessions/fervent-optimistic-fermat/mnt/quant_engine/config.py` | 150+ constants; config_structured.py has typed dataclasses | ACTIVE |
| Data | `/sessions/fervent-optimistic-fermat/mnt/quant_engine/data/loader.py` | `load_ohlcv()`, `load_universe()`, `_cache_is_usable()`; uses `date.today()` bug | ACTIVE |
| Quality | `/sessions/fervent-optimistic-fermat/mnt/quant_engine/data/quality.py` | `assess_ohlcv_quality()` with MAX_MISSING_BAR_FRACTION=0.05, MAX_ZERO_VOLUME_FRACTION=0.25, MAX_ABS_DAILY_RETURN=0.40 | ACTIVE |
| Backtest | `/sessions/fervent-optimistic-fermat/mnt/quant_engine/backtest/engine.py` | 1869 lines; next-bar-open entry, Almgren-Chriss, regime-2 suppression | ACTIVE |
| Features | `/sessions/fervent-optimistic-fermat/mnt/quant_engine/features/pipeline.py` | 1273 lines; 100+ features; FEATURE_METADATA with causality tags | ACTIVE |
| Validation | `/sessions/fervent-optimistic-fermat/mnt/quant_engine/validation.py` | `walk_forward_validate()`, `run_statistical_tests()`, `combinatorial_purged_cv()` | ACTIVE |

### Existing patterns to follow

1. **Config layering**: Use typed dataclasses in `config_structured.py` for structured data; keep constants in `config.py`
2. **Status annotations**: All config constants tagged with `# STATUS: ACTIVE | PLACEHOLDER | DEPRECATED`
3. **Causality tags**: FEATURE_METADATA dict structure with `{"type": "CAUSAL|END_OF_DAY|RESEARCH_ONLY", "category": "..."}`
4. **Quality reporting**: DataQualityReport dataclass with `passed: bool`, `metrics: Dict`, `warnings: List[str]`
5. **Backtest dataclasses**: BacktestResult, Trade use field() with defaults and repr=False for large objects

### Configuration

Current relevant constants:
```python
# data/quality.py reads these from config
MAX_MISSING_BAR_FRACTION = 0.05      # STATUS: ACTIVE
MAX_ZERO_VOLUME_FRACTION = 0.25      # STATUS: ACTIVE
MAX_ABS_DAILY_RETURN = 0.40          # STATUS: ACTIVE

# backtest/engine.py
TRANSACTION_COST_BPS = 20            # STATUS: ACTIVE
ENTRY_THRESHOLD = 0.02               # STATUS: ACTIVE
CONFIDENCE_THRESHOLD = 0.55          # STATUS: ACTIVE
POSITION_SIZE_PCT = 0.05             # STATUS: ACTIVE
BACKTEST_ASSUMED_CAPITAL_USD = 1e6   # STATUS: ACTIVE

# regime
REGIME_MODEL_TYPE = "hmm"            # STATUS: ACTIVE
REGIME_HMM_STATES = 4                # STATUS: ACTIVE
```

Missing constants (to be added):
```python
# Return definition (MUST be locked down)
RET_TYPE = "log"                     # "log" or "simple"; determines how return is computed
LABEL_H = 5                          # Hold this many days; labels are RET_TYPE-returns over next LABEL_H bars
PX_TYPE = "close"                    # "close" or "open"; price used for entry/exit
ENTRY_PRICE_TYPE = "next_bar_open"   # "next_bar_open" (current), "market_on_open", "limit_10bp"
```

---

## Tasks

### T1: Global Preconditions Contract

**What:** Add RET_TYPE, PX_TYPE, LABEL_H, ENTRY_PRICE to config and validate at engine init. Build a `PreconditionsValidator` that runs before any modeling or backtesting.

**Files:**
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/config.py` — add 4 new constants
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/config_structured.py` — add PreconditionsConfig dataclass with validation
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/backtest/engine.py` — add preflight check in BacktestEngine.__init__()
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/models/trainer.py` — add preflight check in ModelTrainer.__init__()

**Implementation notes:**

1. Add to config.py (after line 40, Kalshi section):
   ```python
   # ── Execution Contract ──────────────────────────────────────────────────
   RET_TYPE = "log"                      # STATUS: ACTIVE — "log" (log returns) or "simple" (pct returns)
   LABEL_H = 5                           # STATUS: ACTIVE — label horizon in trading days
   PX_TYPE = "close"                     # STATUS: ACTIVE — "close" or "open" for price baseline
   ENTRY_PRICE_TYPE = "next_bar_open"    # STATUS: ACTIVE — "next_bar_open" (no look-ahead)
   ```

2. Create PreconditionsConfig in config_structured.py:
   ```python
   from enum import Enum
   from dataclasses import dataclass

   class ReturnType(Enum):
       LOG = "log"
       SIMPLE = "simple"

   class PriceType(Enum):
       CLOSE = "close"
       OPEN = "open"

   class EntryType(Enum):
       NEXT_BAR_OPEN = "next_bar_open"
       MARKET_ON_OPEN = "market_on_open"
       LIMIT_10BP = "limit_10bp"

   @dataclass
   class PreconditionsConfig:
       ret_type: ReturnType = ReturnType.LOG
       label_h: int = 5                # trading days
       px_type: PriceType = PriceType.CLOSE
       entry_price_type: EntryType = EntryType.NEXT_BAR_OPEN

       def __post_init__(self):
           # Validation
           if self.label_h < 1:
               raise ValueError("LABEL_H must be >= 1")
           if self.label_h > 60:
               raise ValueError("LABEL_H > 60 days is unrealistic; check config")
   ```

3. Add validator function in new file `/sessions/fervent-optimistic-fermat/mnt/quant_engine/validation/preconditions.py`:
   ```python
   from typing import Tuple
   from ..config import RET_TYPE, LABEL_H, PX_TYPE, ENTRY_PRICE_TYPE
   from ..config_structured import PreconditionsConfig

   def validate_execution_contract() -> Tuple[bool, str]:
       """Check that execution preconditions are locked and sensible."""
       try:
           cfg = PreconditionsConfig(
               ret_type=RET_TYPE,
               label_h=LABEL_H,
               px_type=PX_TYPE,
               entry_price_type=ENTRY_PRICE_TYPE,
           )
           return True, f"Execution contract OK: {cfg}"
       except ValueError as e:
           return False, str(e)
   ```

4. Call validator in BacktestEngine.__init__() and ModelTrainer.__init__():
   ```python
   from ..validation.preconditions import validate_execution_contract

   def __init__(self, ...):
       ok, msg = validate_execution_contract()
       if not ok:
           raise RuntimeError(f"Preconditions failed: {msg}")
   ```

**Verify:**
- `pytest tests/validation/test_preconditions.py::test_validate_execution_contract`
- Manual: `python -c "from quant_engine.validation.preconditions import validate_execution_contract; print(validate_execution_contract())"`

---

### T2: Data Integrity Preflight System

**What:** Build a fail-fast system that blocks corrupt OHLCV from entering the pipeline. Extend `assess_ohlcv_quality()` to emit exceptions, not warnings. Create `DataIntegrityValidator` that runs before load_universe() returns data.

**Files:**
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/data/quality.py` — add fail_on_error parameter to assess_ohlcv_quality()
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/data/loader.py` — add preflight check in load_universe()
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/validation/data_integrity.py` (new) — DataIntegrityValidator class

**Implementation notes:**

1. Modify assess_ohlcv_quality() in quality.py (line 56):
   ```python
   def assess_ohlcv_quality(
       df: pd.DataFrame,
       max_missing_bar_fraction: float = MAX_MISSING_BAR_FRACTION,
       max_zero_volume_fraction: float = MAX_ZERO_VOLUME_FRACTION,
       max_abs_daily_return: float = MAX_ABS_DAILY_RETURN,
       fail_on_error: bool = False,  # NEW
   ) -> DataQualityReport:
       """Assess OHLCV quality; optionally raise on failure."""
       # ... existing logic ...
       report = DataQualityReport(passed=len(warnings) == 0, metrics=metrics, warnings=warnings)

       if fail_on_error and not report.passed:
           raise ValueError(f"Data quality check failed: {report.warnings}")

       return report
   ```

2. Create `/sessions/fervent-optimistic-fermat/mnt/quant_engine/validation/data_integrity.py`:
   ```python
   from dataclasses import dataclass
   from typing import Dict, List, Tuple
   import pandas as pd
   from ..data.quality import assess_ohlcv_quality, DataQualityReport

   @dataclass
   class DataIntegrityCheckResult:
       passed: bool
       n_stocks_passed: int
       n_stocks_failed: int
       failed_tickers: List[str]
       failed_reasons: Dict[str, List[str]]  # {ticker: [warning1, warning2, ...]}

   class DataIntegrityValidator:
       def __init__(self, fail_fast: bool = True):
           self.fail_fast = fail_fast

       def validate_universe(
           self,
           ohlcv_dict: Dict[str, pd.DataFrame]
       ) -> DataIntegrityCheckResult:
           """Check all stocks in universe; fail if any corrupted (if fail_fast=True)."""
           failed_tickers = []
           failed_reasons = {}

           for ticker, df in ohlcv_dict.items():
               try:
                   report = assess_ohlcv_quality(df, fail_on_error=True)
               except ValueError as e:
                   failed_tickers.append(ticker)
                   failed_reasons[ticker] = [str(e)]
                   if self.fail_fast:
                       raise RuntimeError(
                           f"Data integrity check failed at {ticker}: {e}"
                       ) from e

           result = DataIntegrityCheckResult(
               passed=len(failed_tickers) == 0,
               n_stocks_passed=len(ohlcv_dict) - len(failed_tickers),
               n_stocks_failed=len(failed_tickers),
               failed_tickers=failed_tickers,
               failed_reasons=failed_reasons,
           )
           return result
   ```

3. Add to data/loader.py in load_universe() (before return):
   ```python
   from ..validation.data_integrity import DataIntegrityValidator

   def load_universe(tickers, ...) -> Dict[str, pd.DataFrame]:
       # ... existing load logic ...
       ohlcv_dict = { ... }

       # Preflight check
       validator = DataIntegrityValidator(fail_fast=True)
       result = validator.validate_universe(ohlcv_dict)
       if not result.passed:
           logger.error(
               f"Data integrity check failed; {result.n_stocks_failed} tickers corrupted: {result.failed_tickers}"
           )
           raise RuntimeError(f"Universe contains {result.n_stocks_failed} corrupted tickers")

       return ohlcv_dict
   ```

**Verify:**
- Add test case with synthetic corrupted OHLCV:
  ```python
  def test_data_integrity_validator_blocks_missing_bars():
      df = pd.DataFrame({...})  # 50% missing bars
      validator = DataIntegrityValidator(fail_fast=True)
      with pytest.raises(RuntimeError):
          validator.validate_universe({"CORRUPT_STOCK": df})
  ```
- Manual: `python -c "from quant_engine.validation.data_integrity import DataIntegrityValidator; print(DataIntegrityValidator().validate_universe({'SPY': df}))"`

---

### T3: Leakage Tripwires

**What:** Enforce FEATURE_METADATA["type"] at runtime. Detect time-shift leakage with explicit tripwire test. Prevent END_OF_DAY and RESEARCH_ONLY features from being used in live predictions.

**Files:**
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/features/pipeline.py` — add runtime causality check in compute()
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/validation/leakage_detection.py` (new) — LeakageDetector class with time-shift test

**Implementation notes:**

1. Add runtime check to FeaturePipeline.compute() in features/pipeline.py. Find the compute() method and add:
   ```python
   def compute(
       self,
       ohlcv: pd.DataFrame,
       *,
       causality_filter: str = "CAUSAL",  # NEW: "CAUSAL", "END_OF_DAY", "RESEARCH_ONLY", "ALL"
       enforce_causality: bool = True,    # NEW: if True, raise on non-matching causality
   ) -> pd.DataFrame:
       """
       Compute features from OHLCV.

       Parameters
       ----------
       causality_filter : str
           Only compute features matching this causality type.
           "CAUSAL": safe for live prediction
           "END_OF_DAY": requires full-day close; safe for daily-close prediction only
           "RESEARCH_ONLY": cross-sectional or non-causal; offline only
           "ALL": compute all features (backward-compat, for offline analysis)

       enforce_causality : bool
           If True, raise ValueError if any computed feature violates causality_filter.
           If False, log warning and skip.
       """
       # ... existing feature compute code ...

       # NEW: Causality enforcement
       computed_features = {...}  # After all features computed

       if enforce_causality and causality_filter != "ALL":
           violated = []
           for fname in computed_features.columns:
               feature_type = FEATURE_METADATA.get(fname, {}).get("type", "CAUSAL")
               if causality_filter == "CAUSAL" and feature_type != "CAUSAL":
                   violated.append((fname, feature_type))
               elif causality_filter == "END_OF_DAY" and feature_type not in ("CAUSAL", "END_OF_DAY"):
                   violated.append((fname, feature_type))

           if violated:
               raise ValueError(
                   f"Causality violation: {len(violated)} features have non-{causality_filter} type:\n"
                   + "\n".join(f"  {fname}: {ftype}" for fname, ftype in violated)
               )

       return computed_features
   ```

2. Create `/sessions/fervent-optimistic-fermat/mnt/quant_engine/validation/leakage_detection.py`:
   ```python
   from dataclasses import dataclass
   from typing import Dict, List, Tuple
   import pandas as pd
   import numpy as np

   @dataclass
   class LeakageTestResult:
       passed: bool
       n_violations: int
       violations: List[Dict]  # [{feature: str, shift_lag: int, correlation: float}, ...]

   class LeakageDetector:
       """Detects forward-looking leakage via time-shift correlation test."""

       def __init__(self, shift_range: List[int] = None):
           """Initialize detector. Default shift_range = [1, 2, 3, 5, 10]."""
           self.shift_range = shift_range or [1, 2, 3, 5, 10]

       def test_time_shift_leakage(
           self,
           features: pd.DataFrame,
           labels: pd.Series,
           threshold_corr: float = 0.20,  # Suspicious if abs(corr) > 20%
       ) -> LeakageTestResult:
           """
           Check if features have suspicious correlation with FUTURE labels.

           For each feature, compute correlation with labels shifted forward by [1, 2, 3, ...] bars.
           If correlation with forward-shifted label > threshold, flag as leakage.
           """
           violations = []

           # Align features and labels
           common_index = features.index.intersection(labels.index)
           feat_aligned = features.loc[common_index]
           label_aligned = labels.loc[common_index]

           for shift in self.shift_range:
               label_shifted = label_aligned.shift(-shift)  # Negative = future
               mask = label_shifted.notna()

               for col in feat_aligned.columns:
                   if not mask.any():
                       continue

                   corr = feat_aligned.loc[mask, col].corr(label_shifted.loc[mask])
                   if abs(corr) > threshold_corr:
                       violations.append({
                           "feature": col,
                           "shift_lag": shift,
                           "correlation": float(corr),
                       })

           result = LeakageTestResult(
               passed=len(violations) == 0,
               n_violations=len(violations),
               violations=violations,
           )
           return result
   ```

3. Add tripwire call in validation.py before training:
   ```python
   from .leakage_detection import LeakageDetector

   def run_leakage_checks(
       features: pd.DataFrame,
       labels: pd.Series,
   ) -> bool:
       """Run leakage detection; raise on failure."""
       detector = LeakageDetector(shift_range=[1, 2, 3, 5])
       result = detector.test_time_shift_leakage(features, labels, threshold_corr=0.20)

       if not result.passed:
           raise RuntimeError(
               f"Leakage detected: {result.n_violations} features have suspicious forward correlation:\n"
               + "\n".join(
                   f"  {v['feature']}: corr={v['correlation']:.3f} at lag={v['shift_lag']}"
                   for v in result.violations[:10]  # Show top 10
               )
           )
       return True
   ```

**Verify:**
- Create synthetic test with forward leakage: feature = labels.shift(-1)
  ```python
  def test_leakage_detector_catches_forward_shift():
      features = pd.DataFrame({"X": np.random.randn(100)})
      labels = pd.Series(np.random.randn(100))
      features["Y"] = labels.shift(-1)  # Obvious leakage

      detector = LeakageDetector()
      result = detector.test_time_shift_leakage(features, labels)
      assert not result.passed
      assert len(result.violations) > 0
  ```

---

### T4: Null Model Baselines

**What:** Compute and store baseline models (random, zero, momentum) for every backtest. Compare strategy performance against baselines; flag if strategy underperforms null.

**Files:**
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/backtest/null_models.py` (new) — RandomBaseline, ZeroBaseline, MomentumBaseline classes
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/backtest/engine.py` — call baseline generators before strategy backtest

**Implementation notes:**

1. Create `/sessions/fervent-optimistic-fermat/mnt/quant_engine/backtest/null_models.py`:
   ```python
   from dataclasses import dataclass
   from typing import Dict, Optional
   import numpy as np
   import pandas as pd
   from .engine import BacktestResult, Trade

   @dataclass
   class NullModelResults:
       random_baseline: BacktestResult
       zero_baseline: BacktestResult
       momentum_baseline: BacktestResult

   class RandomBaseline:
       """Randomly long/short each stock."""

       def __init__(self, long_prob: float = 0.50, random_seed: int = 42):
           self.long_prob = long_prob
           self.random_seed = random_seed

       def generate_predictions(
           self,
           ohlcv_dict: Dict[str, pd.DataFrame],
       ) -> Dict[str, pd.Series]:
           """Generate random predictions for each ticker."""
           rng = np.random.RandomState(self.random_seed)
           predictions = {}

           for ticker, df in ohlcv_dict.items():
               n = len(df)
               pred = rng.choice([-1, 1], size=n, p=[1-self.long_prob, self.long_prob])
               predictions[ticker] = pd.Series(pred, index=df.index, name="pred")

           return predictions

   class ZeroBaseline:
       """Always flat (zero positions)."""

       def generate_predictions(
           self,
           ohlcv_dict: Dict[str, pd.DataFrame],
       ) -> Dict[str, pd.Series]:
           """Generate zero predictions."""
           predictions = {}
           for ticker, df in ohlcv_dict.items():
               predictions[ticker] = pd.Series(0.0, index=df.index, name="pred")
           return predictions

   class MomentumBaseline:
       """Long if ROC(20) > 0, else short."""

       def __init__(self, lookback: int = 20):
           self.lookback = lookback

       def generate_predictions(
           self,
           ohlcv_dict: Dict[str, pd.DataFrame],
       ) -> Dict[str, pd.Series]:
           """Generate momentum-based predictions."""
           predictions = {}

           for ticker, df in ohlcv_dict.items():
               close = df["Close"].astype(float)
               roc = close.pct_change(self.lookback)
               pred = (roc > 0).astype(int) * 2 - 1  # +1 if roc > 0, -1 else
               predictions[ticker] = pd.Series(pred, index=df.index, name="pred")

           return predictions

   def generate_null_baselines(
       ohlcv_dict: Dict[str, pd.DataFrame],
       backtest_engine,  # BacktestEngine instance
   ) -> NullModelResults:
       """Generate all three null baselines using the same backtest engine."""

       baselines = NullModelResults(
           random_baseline=None,
           zero_baseline=None,
           momentum_baseline=None,
       )

       # Random baseline
       random_gen = RandomBaseline(random_seed=42)
       random_preds = random_gen.generate_predictions(ohlcv_dict)
       baselines.random_baseline = backtest_engine.run(
           predictions=random_preds,
           description="null_random",
       )

       # Zero baseline (should have near-zero return and Sharpe)
       zero_gen = ZeroBaseline()
       zero_preds = zero_gen.generate_predictions(ohlcv_dict)
       baselines.zero_baseline = backtest_engine.run(
           predictions=zero_preds,
           description="null_zero",
       )

       # Momentum baseline
       momentum_gen = MomentumBaseline(lookback=20)
       momentum_preds = momentum_gen.generate_predictions(ohlcv_dict)
       baselines.momentum_baseline = backtest_engine.run(
           predictions=momentum_preds,
           description="null_momentum",
       )

       return baselines
   ```

2. Modify BacktestResult in engine.py to include baseline comparison:
   ```python
   @dataclass
   class BacktestResult:
       # ... existing fields ...
       null_baselines: Optional[NullModelResults] = field(default=None, repr=False)

       def summarize_vs_null(self) -> Dict[str, float]:
           """Compare strategy Sharpe/return vs null baselines."""
           if self.null_baselines is None:
               return {}

           return {
               "sharpe_vs_random": self.sharpe_ratio - self.null_baselines.random_baseline.sharpe_ratio,
               "sharpe_vs_zero": self.sharpe_ratio - self.null_baselines.zero_baseline.sharpe_ratio,
               "sharpe_vs_momentum": self.sharpe_ratio - self.null_baselines.momentum_baseline.sharpe_ratio,
               "return_vs_random": self.total_return - self.null_baselines.random_baseline.total_return,
               "return_vs_zero": self.total_return - self.null_baselines.zero_baseline.total_return,
               "return_vs_momentum": self.total_return - self.null_baselines.momentum_baseline.total_return,
           }
   ```

3. Call baseline generator in BacktestEngine.run():
   ```python
   def run(self, predictions: Dict[str, pd.Series], ...) -> BacktestResult:
       # ... existing backtest logic ...
       result = BacktestResult(...)

       # Compute null baselines (optional, can be gated by config flag)
       if getattr(self, 'compute_null_baselines', True):
           from .null_models import generate_null_baselines
           result.null_baselines = generate_null_baselines(self.ohlcv_dict, self)

       return result
   ```

**Verify:**
- Test that random baseline has Sharpe near 0:
  ```python
  def test_random_baseline_sharpe_near_zero():
      result = baselines.random_baseline
      assert abs(result.sharpe_ratio) < 0.5  # Random walk should have near-zero Sharpe
  ```

---

### T5: Cost Stress Sweep Defaults

**What:** Add configuration for transaction cost multiplier sweep. Run backtest at [0.5x, 1.0x, 2.0x, 5.0x] cost levels; report performance curve. Identify breakeven transaction cost.

**Files:**
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/config.py` — add COST_STRESS_MULTIPLIERS
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/backtest/cost_stress.py` (new) — CostStressTester class
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/backtest/engine.py` — hook in cost stress reporter

**Implementation notes:**

1. Add to config.py:
   ```python
   # ── Cost Stress Testing ────────────────────────────────────────────────
   COST_STRESS_MULTIPLIERS = [0.5, 1.0, 2.0, 5.0]  # STATUS: ACTIVE — cost sweep factors
   COST_STRESS_ENABLED = True                      # STATUS: ACTIVE — run cost stress by default
   ```

2. Create `/sessions/fervent-optimistic-fermat/mnt/quant_engine/backtest/cost_stress.py`:
   ```python
   from dataclasses import dataclass
   from typing import Dict, List
   import pandas as pd
   from .engine import BacktestResult, BacktestEngine

   @dataclass
   class CostStressResult:
       base_cost_bps: float
       multipliers: List[float]
       results: Dict[float, BacktestResult]  # {multiplier: result}
       breakeven_cost_bps: float  # Cost at which Sharpe ratio = 0

   class CostStressTester:
       """Test strategy robustness to cost assumptions."""

       def __init__(self, base_cost_bps: float, multipliers: List[float] = None):
           self.base_cost_bps = base_cost_bps
           self.multipliers = multipliers or [0.5, 1.0, 2.0, 5.0]

       def stress_test(
           self,
           backtest_engine: BacktestEngine,
           predictions: Dict,  # from run()
       ) -> CostStressResult:
           """Run backtest at multiple cost levels."""

           results = {}
           original_cost = backtest_engine.cost_bps

           try:
               for mult in self.multipliers:
                   new_cost = self.base_cost_bps * mult
                   backtest_engine.cost_bps = new_cost
                   result = backtest_engine.run(predictions)
                   results[mult] = result
           finally:
               backtest_engine.cost_bps = original_cost

           # Estimate breakeven cost
           sharpes = [r.sharpe_ratio for r in results.values()]
           if sharpes[0] > 0 and sharpes[-1] < 0:
               # Linear interpolation to find breakeven
               from scipy.interpolate import interp1d
               mults_sorted = sorted(results.keys())
               sharpes_sorted = [results[m].sharpe_ratio for m in mults_sorted]
               interp = interp1d(mults_sorted, sharpes_sorted, kind="linear")
               zero_crossing = float(interp(1.0))  # Rough estimate
               breakeven_cost = self.base_cost_bps * zero_crossing
           else:
               breakeven_cost = float('inf') if sharpes[-1] > 0 else 0.0

           return CostStressResult(
               base_cost_bps=self.base_cost_bps,
               multipliers=list(self.multipliers),
               results=results,
               breakeven_cost_bps=breakeven_cost,
           )

       def report(self, result: CostStressResult) -> str:
           """Generate human-readable cost stress report."""
           lines = [
               f"Cost Stress Test Report",
               f"{'Multiplier':<12} {'Cost (bps)':<12} {'Sharpe':<10} {'Return':<10}",
               "-" * 44,
           ]
           for mult in sorted(result.results.keys()):
               r = result.results[mult]
               cost = result.base_cost_bps * mult
               lines.append(
                   f"{mult:<12.1f} {cost:<12.1f} {r.sharpe_ratio:<10.3f} {r.total_return:<10.3f}"
               )

           lines.append(f"\nBreakeven cost: {result.breakeven_cost_bps:.1f} bps")
           return "\n".join(lines)
   ```

3. Add to BacktestResult:
   ```python
   @dataclass
   class BacktestResult:
       # ... existing fields ...
       cost_stress_result: Optional[CostStressResult] = field(default=None, repr=False)
   ```

4. Hook into engine.py run() method:
   ```python
   from .cost_stress import CostStressTester
   from ..config import COST_STRESS_ENABLED, COST_STRESS_MULTIPLIERS, TRANSACTION_COST_BPS

   def run(self, predictions: Dict[str, pd.Series], ...) -> BacktestResult:
       # ... existing backtest logic ...
       result = BacktestResult(...)

       # Cost stress testing (optional)
       if COST_STRESS_ENABLED:
           tester = CostStressTester(
               base_cost_bps=TRANSACTION_COST_BPS,
               multipliers=COST_STRESS_MULTIPLIERS,
           )
           result.cost_stress_result = tester.stress_test(self, predictions)

       return result
   ```

**Verify:**
- Test cost stress at 1.0x should match original result:
  ```python
  def test_cost_stress_1x_matches_original():
      tester = CostStressTester(20.0, [1.0])
      result = tester.stress_test(engine, predictions)
      assert abs(result.results[1.0].sharpe_ratio - original_sharpe) < 0.01
  ```

---

### T6: Fix Cache Staleness to Use Trading Calendar

**What:** Replace `date.today()` in `_cache_is_usable()` with last trading day from trading calendar. Ensure cache validity is checked relative to trading calendar, not wall-clock time.

**Files:**
- `/sessions/fervent-optimistic-fermat/mnt/quant_engine/data/loader.py` — fix _cache_is_usable() logic

**Implementation notes:**

1. Locate _cache_is_usable() in data/loader.py (currently uses date.today() on line ~XX):
   ```python
   # CURRENT (BUGGY):
   def _cache_is_usable(cache_time: datetime, max_age_days: int) -> bool:
       return (date.today() - cache_time.date()).days < max_age_days

   # FIXED:
   from datetime import datetime
   import pandas as pd
   try:
       import pandas_market_calendars as mcal
       NYSE_CAL = mcal.get_calendar("NYSE")
   except ImportError:
       NYSE_CAL = None

   def _get_last_trading_day(ref_date: datetime = None) -> pd.Timestamp:
       """Get last trading day before ref_date (or today if ref_date is None)."""
       if ref_date is None:
           ref_date = pd.Timestamp.now()
       else:
           ref_date = pd.Timestamp(ref_date)

       if NYSE_CAL is None:
           # Fallback: use pandas bdate_range
           business_days = pd.bdate_range(end=ref_date, periods=1)
           return business_days[-1]

       # Use NYSE calendar
       schedule = NYSE_CAL.schedule(start_date=ref_date - pd.Timedelta(days=10), end_date=ref_date)
       if len(schedule) == 0:
           return ref_date  # No trading days found; return ref_date as fallback
       return pd.Timestamp(schedule.index[-1])

   def _cache_is_usable(cache_time: pd.Timestamp, max_age_days: int) -> bool:
       """Check if cache is fresh relative to last trading day."""
       last_trading_day = _get_last_trading_day()
       cache_date = pd.Timestamp(cache_time)

       # Count trading days between cache_time and last_trading_day
       if NYSE_CAL is not None:
           schedule = NYSE_CAL.schedule(start_date=cache_date, end_date=last_trading_day)
           trading_days_elapsed = len(schedule)
       else:
           # Fallback: count business days
           trading_days_elapsed = len(pd.bdate_range(start=cache_date, end=last_trading_day))

       return trading_days_elapsed < max_age_days
   ```

2. Update docstring for _cache_is_usable():
   ```python
   def _cache_is_usable(cache_time: pd.Timestamp, max_age_days: int) -> bool:
       """
       Check if cache is fresh relative to the last trading day.

       Uses NYSE trading calendar to count elapsed trading days.
       Falls back to business days if pandas_market_calendars is unavailable.

       Parameters
       ----------
       cache_time : pd.Timestamp
           Timestamp when cache was created
       max_age_days : int
           Maximum age in trading days before cache is considered stale

       Returns
       -------
       bool
           True if cache is younger than max_age_days (in trading days)
       """
   ```

3. Update callers of _cache_is_usable() to pass pd.Timestamp:
   - Locate all calls to _cache_is_usable() in data/loader.py
   - Ensure cache_time is converted to pd.Timestamp before calling

**Verify:**
- Test on weekend: cache from Friday should still be valid if max_age_days > 1
  ```python
  def test_cache_staleness_weekend_friendly():
      friday = pd.Timestamp("2026-02-20")  # Friday
      monday = pd.Timestamp("2026-02-23")  # Monday (weekend in between)
      result = _cache_is_usable(friday, max_age_days=5)
      assert result  # Friday cache should still be valid on Monday
  ```
- Test on holiday: cache before holiday should not count holiday as trading day
  ```python
  def test_cache_staleness_holiday_aware():
      day_before_holiday = pd.Timestamp("2026-01-16")  # Fri before MLK day (1/20)
      day_after_holiday = pd.Timestamp("2026-01-21")   # Wed after MLK day
      result = _cache_is_usable(day_before_holiday, max_age_days=1)
      assert result  # Only 1 trading day elapsed (Tue), not 3 calendar days
  ```

---

## Validation

### Acceptance criteria

1. **Global preconditions**: Config validation runs at engine init; model training raises ValueError if RET_TYPE, LABEL_H, PX_TYPE, ENTRY_PRICE_TYPE are unset or invalid
2. **Data integrity**: load_universe() raises RuntimeError if > 1 stock fails quality checks; zero stocks pass when all data is corrupted
3. **Leakage tripwires**: Feature pipeline raises ValueError if RESEARCH_ONLY features are computed with causality_filter="CAUSAL"; LeakageDetector catches 100% of time-shifted labels in synthetic test
4. **Null baselines**: Every backtest includes RandomBaseline, ZeroBaseline, MomentumBaseline results; ZeroBaseline has Sharpe < 0.1
5. **Cost stress**: CostStressTester runs at [0.5x, 1.0x, 2.0x, 5.0x] multipliers; 1.0x result matches original backtest within 1 basis point
6. **Cache staleness**: _cache_is_usable() respects trading calendar; Friday cache is valid on Monday even if max_age_days=1

### Verification steps

1. **Unit tests** (pytest):
   ```bash
   pytest tests/validation/test_preconditions.py -v
   pytest tests/validation/test_data_integrity.py -v
   pytest tests/validation/test_leakage_detection.py -v
   pytest tests/backtest/test_null_models.py -v
   pytest tests/backtest/test_cost_stress.py -v
   pytest tests/data/test_cache_staleness.py -v
   ```

2. **Integration test** (end-to-end backtest):
   ```bash
   python scripts/run_backtest.py --enable-truth-layer --universe SPY,QQQ,IWM
   ```
   Expected output includes:
   - "Preconditions validation: PASSED"
   - "Data integrity check: PASSED (N stocks)"
   - "Leakage detection: PASSED"
   - "Null baselines computed: Random Sharpe=..., Zero Sharpe=..., Momentum Sharpe=..."
   - "Cost stress sweep: Base 20 bps → Breakeven XXX bps"

3. **Manual validation** (on live data):
   - Run on a weekend after market close: verify cache is not marked stale
   - Corrupt one stock's OHLCV (set 60% of volume to 0): verify load_universe() raises error
   - Add a forward-leaking feature to FEATURE_METADATA: verify pipeline raises ValueError

### Rollback plan

If Truth Layer breaks existing workflows:

1. **Preconditions validation**: Add `TRUTH_LAYER_STRICT_PRECONDITIONS = False` to config.py; make validation non-blocking
2. **Data integrity**: Add `TRUTH_LAYER_FAIL_ON_CORRUPT = False`; convert to warnings-only
3. **Leakage tripwires**: Add `TRUTH_LAYER_ENFORCE_CAUSALITY = False`; skip runtime checks
4. **Null baselines**: Add `TRUTH_LAYER_COMPUTE_NULL_BASELINES = False`; skip baseline generation
5. **Cost stress**: Add `TRUTH_LAYER_COST_STRESS_ENABLED = False`; skip cost sweep
6. **Cache staleness**: Revert _cache_is_usable() to `date.today()` logic (add `USE_TRADING_CALENDAR = False`)

Each can be independently disabled without breaking downstream code.

---

## Notes

- **Backward compatibility**: All Truth Layer checks are implemented as preflight validations before core backtest/prediction logic. Existing models/configs continue to work with sensible defaults.
- **Performance**: Data integrity and leakage checks add ~2–5% overhead to run time. Cost stress sweep adds 4–5x time (multiple backtests). Can be disabled via config flags.
- **Operational visibility**: All validation results logged at INFO level; failures logged at ERROR level with actionable messages for ops teams.
- **Future enhancements**:
  - Add config versioning (lock specific versions of RET_TYPE, LABEL_H for production models)
  - Add statistical significance testing for null baseline outperformance
  - Add portfolio-level position limit stress (separate spec)
  - Add microstructure realism (execution slippage, market impact curves)
