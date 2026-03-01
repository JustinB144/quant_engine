# JOB 2: DEPENDENCY EXTRACTION
## LLM Audit Workflow — Step 2 of 7

**Purpose**: Extract every cross-module import relationship in the codebase, producing a complete edge list that maps which files depend on which other files, what symbols they import, and whether the import is top-level or lazy/conditional.

**Estimated effort**: 1 LLM session, ~45 minutes
**Input required**: Full repo access + MODULE_INVENTORY.yaml from Job 1
**Output**: `DEPENDENCY_EDGES.json` + `DEPENDENCY_MATRIX.md` saved to `docs/audit/data/`
**Depends on**: Job 1 (Module Inventory)
**Feeds into**: Job 3 (Hotspot Scoring), Job 4 (Interface Boundaries), Job 5 (Subsystem Clustering)

---

## WHAT YOU ARE BUILDING

Two artifacts:
1. A JSON file containing every import edge between files in different modules
2. A markdown file containing a module-level adjacency matrix summarizing the edges

---

## CRITICAL CONTEXT: IMPORT PATTERNS IN THIS CODEBASE

This codebase uses TWO import patterns. You MUST scan for both:

### Pattern 1: Relative imports (used INSIDE packages)
```python
from ..config import ROOT_DIR                    # parent package
from ..regime.shock_vector import ShockVector     # sibling package
from .execution import ExecutionModel             # same package
```

### Pattern 2: Absolute imports (used in run_*.py entry points and scripts)
```python
from quant_engine.config import ROOT_DIR
from quant_engine.data.loader import load_universe
```

### Pattern 3: Lazy/conditional imports (INSIDE function bodies)
```python
def some_function():
    from ..risk.position_sizer import PositionSizer  # Only loaded when called
```

**WHY THIS MATTERS**: Lazy imports represent optional coupling. They won't cause import-time failures but will cause runtime failures if the target file changes. They are HARDER to find because they're indented inside functions. You MUST scan for them.

---

## EXACT INSTRUCTIONS

### Step 1: Extract ALL cross-module imports

For each of the 14 module directories, run:

```bash
# Top-level imports (at file start, no indentation)
grep -rn "^from \.\." /path/to/quant_engine/MODULE/ --include="*.py" | grep -v __pycache__

# Lazy imports (indented, inside function bodies)
grep -rn "^    .*from \.\." /path/to/quant_engine/MODULE/ --include="*.py" | grep -v __pycache__
grep -rn "^        .*from \.\." /path/to/quant_engine/MODULE/ --include="*.py" | grep -v __pycache__
```

For root-level files (run_*.py, scripts/):
```bash
grep -rn "from quant_engine\." /path/to/quant_engine/run_*.py
grep -rn "from quant_engine\." /path/to/quant_engine/scripts/*.py
```

### Step 2: Parse each import into a structured edge

For EVERY import line found, produce a JSON edge:

```json
{
  "source_file": "autopilot/engine.py",
  "source_module": "autopilot",
  "source_line": 20,
  "target_file": "backtest/engine.py",
  "target_module": "backtest",
  "import_statement": "from ..backtest.engine import Backtester",
  "symbols_imported": ["Backtester"],
  "import_type": "top_level",
  "notes": ""
}
```

The `import_type` field MUST be one of:
- `"top_level"` — Import at module level (lines with no indentation, typically lines 1-100)
- `"lazy"` — Import inside a function body (indented lines, typically after line 100)
- `"conditional"` — Import inside a try/except or if-guard

### Step 3: Classify same-module vs cross-module

**Cross-module** imports are between DIFFERENT top-level directories. These are what we care about.
- `from ..backtest.engine import Backtester` in autopilot/engine.py → CROSS-MODULE (autopilot→backtest)
- `from .execution import ExecutionModel` in backtest/engine.py → SAME-MODULE (internal to backtest)

**Include BOTH in the output** but tag them:
- `"cross_module": true` for cross-module imports
- `"cross_module": false` for same-module imports

### Step 4: Build the adjacency matrix

Aggregate cross-module edges into a module-level matrix. For each cell, count the number of distinct edges.

Format as a markdown table:

```
| Source ↓ / Target → | config | data | features | indicators | regime | models | backtest | risk | evaluation | validation | autopilot | kalshi | api | utils |
```

Each cell contains the number of import edges (0 if none).

### Step 5: Verify against known edges

The following edges were VERIFIED by reading actual source code on 2026-02-27. Your extraction MUST find all of these. If any are missing, your scan is incomplete.

**VERIFIED CROSS-MODULE IMPORTS (with exact file and line numbers):**

**data/ imports from:**
- config: data/feature_store.py:31, data/loader.py:34, data/local_cache.py:22, data/quality.py:25, data/intraday_quality.py:25(lazy), data/survivorship.py:371(lazy), data/loader.py:601(lazy), data/loader.py:701(lazy), data/local_cache.py:748(lazy)
- validation: data/loader.py:567(lazy) → validation.data_integrity.DataIntegrityValidator
- kalshi: data/provider_registry.py:23(lazy) → kalshi.provider.KalshiProvider

**features/ imports from:**
- indicators: features/pipeline.py:21 (top-level, 21+ symbols), features/pipeline.py:769(lazy)→spectral, :789(lazy)→ssa, :809(lazy)→tail_risk, :836(lazy)→ot_divergence, :1337(lazy)→eigenvalue
- config: features/intraday.py:21, features/pipeline.py:750(lazy), :871(lazy), :924(lazy), :1120(lazy), :1326(lazy), :1460(lazy), :1527(lazy)
- regime: features/pipeline.py:1303(lazy) → regime.correlation.CorrelationRegimeDetector
- data: features/pipeline.py:1413(lazy) → data.wrds_provider.WRDSProvider, :1458(lazy) → data.local_cache.load_intraday_ohlcv, :1528(lazy) → data.loader.load_ohlcv

**regime/ imports from:**
- config: regime/detector.py:24, regime/consensus.py:55(lazy), regime/hmm.py:560(lazy), regime/jump_model_pypi.py:84(lazy), regime/jump_model_pypi.py:314(lazy), regime/online_update.py:60(lazy), regime/uncertainty_gate.py:52(lazy), regime/detector.py:240(lazy), :324(lazy), :402(lazy), :933(lazy)

**models/ imports from:**
- config: models/trainer.py:77, models/predictor.py:21, models/versioning.py:22, models/governance.py:10-11, models/feature_stability.py:26, models/retrain_trigger.py:31
- features: models/predictor.py:22 → features.pipeline.get_feature_type
- validation: models/trainer.py:219(lazy) → validation.preconditions.enforce_preconditions

**backtest/ imports from:**
- config: backtest/engine.py:26, backtest/validation.py:23, backtest/optimal_execution.py:17, backtest/engine.py:1713(lazy), :2375(lazy)
- regime: backtest/engine.py:77 → regime.shock_vector (compute_shock_vectors, ShockVector), :78 → regime.uncertainty_gate (UncertaintyGate), backtest/execution.py:29(lazy) → regime.shock_vector.ShockVector
- validation: backtest/engine.py:198(lazy) → validation.preconditions.enforce_preconditions
- risk: backtest/engine.py:316-320(lazy) → risk.position_sizer.PositionSizer, risk.drawdown.DrawdownController, risk.stop_loss.StopLossManager, risk.portfolio_risk.PortfolioRiskManager, risk.metrics.RiskMetrics

**risk/ imports from:**
- config: risk/drawdown.py:18, risk/portfolio_risk.py:23, risk/stop_loss.py:23-24, risk/portfolio_optimizer.py:190(lazy), :202(lazy), risk/position_sizer.py:136(lazy), :819(lazy)
- regime: risk/position_sizer.py:27 → regime.uncertainty_gate.UncertaintyGate

**evaluation/ imports from:**
- config: evaluation/engine.py:27, evaluation/calibration_analysis.py:19, evaluation/fragility.py:20, evaluation/metrics.py:20, evaluation/ml_diagnostics.py:26, evaluation/slicing.py:23, :193(lazy)
- models: evaluation/calibration_analysis.py:92(lazy) → models.calibration (compute_ece, compute_reliability_curve)
- backtest: evaluation/engine.py:272(lazy) → backtest.validation.walk_forward_with_embargo, :315(lazy) → backtest.validation (rolling_ic, detect_ic_decay)

**validation/ imports from:**
- config: validation/preconditions.py:16
- config_structured: validation/preconditions.py:23 → config_structured.PreconditionsConfig
- data: validation/data_integrity.py:20 → data.quality (assess_ohlcv_quality, DataQualityReport)

**autopilot/ imports from:**
- backtest: autopilot/engine.py:20-26 (Backtester, advanced_validation funcs, validation funcs), autopilot/promotion_gate.py:12 → backtest.engine.BacktestResult, autopilot/paper_trader.py:55-57 (ExecutionModel, ADVTracker, CostCalibrator), autopilot/engine.py:1405(lazy) → backtest.execution.ExecutionModel
- models: autopilot/engine.py:33,57-59 (walk_forward, cross_sectional_rank, EnsemblePredictor, ModelTrainer), autopilot/engine.py:565(lazy) → models.cross_sectional
- config: autopilot/engine.py:34, autopilot/meta_labeler.py:23, autopilot/paper_trader.py:13, autopilot/promotion_gate.py:13, autopilot/registry.py:10, autopilot/strategy_allocator.py:20, autopilot/strategy_discovery.py:10, autopilot/engine.py:1906(lazy)
- data: autopilot/engine.py:54-55 (load_survivorship_universe, load_universe, filter_panel_by_point_in_time_universe)
- features: autopilot/engine.py:56 → features.pipeline.FeaturePipeline
- regime: autopilot/engine.py:60-61 (RegimeDetector, UncertaintyGate)
- risk: autopilot/engine.py:62 → risk.portfolio_optimizer.optimize_portfolio, :971(lazy) → risk.position_sizer.PositionSizer, :1528(lazy) → risk.covariance, autopilot/paper_trader.py:58-60 (PositionSizer, StopLossManager, PortfolioRiskManager), :99(lazy) → risk.drawdown.DrawdownController
- api: autopilot/engine.py:1868(lazy) → api.services.health_service.HealthService, :1911(lazy) → api.services.health_service.HealthService, autopilot/paper_trader.py:173(lazy) → api.services.health_risk_feedback, :189(lazy) → api.services.health_service.HealthService, :211(lazy) → api.ab_testing.ABTestRegistry, :532(lazy) → api.services.health_service.HealthService

**kalshi/ imports from:**
- config: kalshi/provider.py:14
- features: kalshi/options.py:11 → features.options_factors.compute_option_surface_factors
- autopilot: kalshi/pipeline.py:25 → autopilot.promotion_gate.PromotionDecision, kalshi/promotion.py:12-13 → autopilot.promotion_gate (PromotionDecision, PromotionGate), autopilot.strategy_discovery.StrategyCandidate
- backtest: kalshi/promotion.py:14 → backtest.engine.BacktestResult, kalshi/walkforward.py:12 → backtest.advanced_validation (deflated_sharpe_ratio, monte_carlo_validation)

**utils/ imports from:**
- config: utils/logging.py:114(lazy) → config (ALERT_HISTORY_FILE, ALERT_WEBHOOK_URL)

**api/ imports from (internal only — all ..submodule patterns):**
- api/deps/providers.py:6 → api.config (ApiSettings, RuntimeConfig)
- api/routers/* → api.cache, api.deps, api.schemas, api.services, api.jobs, api.errors
- api/jobs/* → api.orchestrator.PipelineOrchestrator
- api/routers/config_mgmt.py:11 → api.config.RuntimeConfig (NOTE: this is api's OWN config.py, not root config.py)

**run_*.py entry points import from (absolute paths):**
- run_train.py: config, data.loader, features.pipeline, regime.detector, models.governance, models.trainer, models.versioning, reproducibility
- run_backtest.py: config, data.loader, data.survivorship, features.pipeline, regime.detector, models.predictor, backtest.engine, backtest.validation, reproducibility, backtest.advanced_validation(lazy:382)
- run_predict.py: config, data.loader, features.pipeline, regime.detector, models.predictor, reproducibility
- run_autopilot.py: autopilot.engine, config, reproducibility
- run_retrain.py: config, data.loader, features.pipeline, regime.detector, models.governance, models.trainer, models.retrain_trigger, models.versioning, reproducibility
- run_kalshi_event_pipeline.py: config, kalshi.distribution, kalshi.events, kalshi.pipeline, kalshi.promotion, kalshi.walkforward
- run_wrds_daily_refresh.py: config, data.local_cache, data.survivorship, data.wrds_provider(lazy:406,702,746), data.local_cache(lazy:676)
- run_rehydrate_cache_metadata.py: config, data.local_cache
- run_server.py: (launches FastAPI, imports api.main)

**scripts/ import from:**
- scripts/compare_regime_models.py: config, regime.detector, regime.hmm, features.pipeline(lazy:48)
- scripts/ibkr_daily_gapfill.py: config, data.local_cache

---

## OUTPUT FORMAT

### DEPENDENCY_EDGES.json

```json
{
  "metadata": {
    "generated": "YYYY-MM-DD",
    "total_edges": <count>,
    "cross_module_edges": <count>,
    "same_module_edges": <count>
  },
  "edges": [
    {
      "source_file": "autopilot/engine.py",
      "source_module": "autopilot",
      "source_line": 20,
      "target_file": "backtest/engine.py",
      "target_module": "backtest",
      "import_statement": "from ..backtest.engine import Backtester",
      "symbols_imported": ["Backtester"],
      "import_type": "top_level",
      "cross_module": true
    }
  ]
}
```

### DEPENDENCY_MATRIX.md

A markdown table showing edge counts between modules, plus a summary of:
- Total cross-module edges per module (fan-in and fan-out)
- Which modules are isolated (zero cross-module imports)
- Which modules are hubs (5+ cross-module connections)

---

## VERIFICATION CHECKLIST

Before declaring this job complete:

- [ ] Every cross-module import from the VERIFIED list above was found
- [ ] Lazy imports (indented `from ..`) were captured with correct line numbers
- [ ] The JSON edge list contains the exact `import_statement` string (copy-paste from grep)
- [ ] The adjacency matrix row sums match the fan-out count for each module
- [ ] The adjacency matrix column sums match the fan-in count for each module
- [ ] No edges were fabricated — every edge has a real source file and line number
- [ ] Same-module imports (from .submodule) are included but tagged `cross_module: false`

---

## KNOWN EDGE CASES

1. **api/routers/config_mgmt.py imports from `..config`** — This is api's OWN config.py (api/config.py), NOT root config.py. Tag it as same-module.

2. **autopilot/paper_trader.py imports from api.services** — This is a CROSS-MODULE import from autopilot→api. It creates a circular architectural dependency (api serves autopilot, but autopilot imports from api). Flag this in notes.

3. **data/provider_registry.py imports from kalshi.provider** — Conditional import inside a try block. Tag as `"conditional"`.

4. **features/pipeline.py has ~18 lazy imports** — Many are deep inside compute functions. You must scan past line 750 to find them all.

5. **backtest/engine.py has lazy risk imports at lines 316-320** — These are inside the `_init_risk_managed_mode()` method, NOT at file top.
