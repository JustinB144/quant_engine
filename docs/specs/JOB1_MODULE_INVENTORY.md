# JOB 1: MODULE INVENTORY
## LLM Audit Workflow — Step 1 of 7

**Purpose**: Produce a complete inventory card for every Python package and root-level file in the quant_engine codebase. This is the foundation that all subsequent jobs depend on.

**Estimated effort**: 1 LLM session, ~30 minutes
**Input required**: Full repo access
**Output**: `MODULE_INVENTORY.yaml` saved to `docs/audit/`
**Depends on**: Nothing (run first)
**Feeds into**: Job 2 (Dependency Extraction), Job 3 (Hotspot Scoring)

---

## WHAT YOU ARE BUILDING

A structured YAML file containing one "module card" for every top-level package directory and every root-level Python file. Each card captures: what the module does, how many files it has, which files are largest/most important, and what classes/functions it exports to other modules.

---

## EXACT INSTRUCTIONS

### Step 1: Generate the file tree

Run this command and save the output — you will reference it throughout:

```bash
find /path/to/quant_engine -name "*.py" -not -path "*__pycache__*" -not -path "*.egg-info*" -not -path "*/node_modules/*" -not -path "*/.venv/*" | sort
```

### Step 2: For each module directory listed below, produce a card

You MUST inventory these 14 package directories plus the root-level files:

**Package directories:**
1. `data/` (including `data/providers/` subdirectory)
2. `features/`
3. `indicators/`
4. `regime/`
5. `models/` (including `models/iv/` subdirectory)
6. `backtest/`
7. `risk/`
8. `evaluation/`
9. `validation/`
10. `autopilot/`
11. `kalshi/` (including `kalshi/tests/` subdirectory)
12. `api/` (including subdirectories: `routers/`, `services/`, `schemas/`, `jobs/`, `cache/`, `deps/`)
13. `utils/`
14. `scripts/`

**Root-level files** (treat as a 15th group called "root"):
- `config.py`
- `config_structured.py`
- `reproducibility.py`
- `run_train.py`
- `run_backtest.py`
- `run_predict.py`
- `run_autopilot.py`
- `run_retrain.py`
- `run_server.py`
- `run_kalshi_event_pipeline.py`
- `run_wrds_daily_refresh.py`
- `run_rehydrate_cache_metadata.py`
- `__init__.py`

### Step 3: Card format

For EACH module, produce a card in this exact YAML format:

```yaml
- module: <directory_name>
  path: <relative path from repo root>
  purpose: >
    <1-3 sentence description. Get this from reading the module's __init__.py
    and the docstrings at the top of its main files. Do NOT guess — if there's
    no docstring, describe based on what the code actually does.>
  file_count: <exact count of .py files in this directory and subdirectories>
  total_lines: <sum of line counts for all .py files>
  files:
    - path: <relative path to each .py file>
      lines: <exact line count from wc -l>
      role: >
        <1 sentence describing what this file does. Read the file's module
        docstring or first class/function to determine this. Do NOT guess.>
  public_interface:
    classes:
      - name: <ClassName>
        defined_in: <relative file path>
        exported: <true if listed in __init__.py or __all__, false otherwise>
        description: <1 sentence from docstring>
    functions:
      - name: <function_name>
        defined_in: <relative file path>
        exported: <true if listed in __init__.py or __all__, false otherwise>
        description: <1 sentence from docstring>
    constants:
      - name: <CONSTANT_NAME>
        defined_in: <relative file path>
        description: <what it configures>
```

### Step 4: How to determine public_interface

A class or function is "public" if ANY of these are true:
1. It appears in the module's `__init__.py` imports
2. It appears in an `__all__` list
3. Another module imports it (you can check this by grepping for the symbol name in other directories)

For determining what OTHER modules import, run:
```bash
grep -rn "from.*<module_name>.*import" /path/to/quant_engine/ --include="*.py" | grep -v __pycache__ | grep -v "<module_name>/"
```
This finds all imports OF this module FROM other modules.

### Step 5: Verification checklist

Before declaring this job complete, verify:

- [ ] Every .py file in the repo appears in exactly one module card
- [ ] Line counts match `wc -l` output (not estimated)
- [ ] Every `__init__.py` was read and its exports are captured
- [ ] The `purpose` field was derived from actual docstrings/code, not guessed
- [ ] No module directory was skipped

---

## VERIFIED REFERENCE DATA

The following data was extracted from the actual codebase on 2026-02-27. Use it as a CHECK against your own findings — do not copy it blindly.

### File counts per module (verified):
| Module | File Count |
|--------|-----------|
| api | 59 |
| tests | 99 |
| kalshi | 25 |
| models | 16 |
| risk | 16 |
| data | 15 |
| regime | 13 |
| features | 10 |
| backtest | 11 |
| evaluation | 8 |
| autopilot | 8 |
| indicators | 7 |
| validation | 5 |
| scripts | 5 |
| utils | 2 |

### Largest files (verified, lines):
| File | Lines |
|------|-------|
| api/services/health_service.py | 2,929 |
| indicators/indicators.py | 2,904 |
| backtest/engine.py | 2,488 |
| autopilot/engine.py | 1,927 |
| models/trainer.py | 1,818 |
| data/wrds_provider.py | 1,620 |
| features/pipeline.py | 1,541 |
| risk/position_sizer.py | 1,254 |
| autopilot/paper_trader.py | 1,254 |

### __init__.py export summary (verified):

**data/__init__.py exports**: DataQualityReport, FeatureStore, load_ohlcv, load_universe, load_survivorship_universe, load_with_delistings, save_ohlcv, load_ibkr_data, list_cached_tickers, cache_universe, get_provider, list_providers, register_provider, assess_ohlcv_quality, check_ohlc_relationships, generate_quality_report, flag_degraded_stocks

**features/__init__.py exports**: FEATURE_METADATA, get_feature_type

**indicators/__init__.py exports**: Indicator (base), 90+ indicator classes (ATR, RSI, MACD, SMA, EMA, ADX, Aroon, OBV, MFI, RVOL, etc.), get_all_indicators

**regime/__init__.py exports**: BOCPDDetector, BOCPDResult, ConfidenceCalibrator, RegimeConsensus, CorrelationRegimeDetector, RegimeDetector, RegimeOutput, GaussianHMM, HMMFitResult, OnlineRegimeUpdater, StatisticalJumpModel, JumpModelResult, PyPIJumpModel, ShockVector, ShockVectorValidator, UncertaintyGate, compute_shock_vectors, detect_regimes_batch, validate_hmm_observation_features

**models/__init__.py exports**: ModelGovernance, ChampionRecord, ConfidenceCalibrator, TabularNet, FeatureStabilityTracker, DistributionShiftDetector, ConformalPredictor, cross_sectional_rank, walk_forward_select

**backtest/__init__.py**: Empty (no explicit exports)

**risk/__init__.py exports**: PositionSizer, PositionSize, PortfolioRiskManager, ConstraintMultiplier, RiskCheck, UniverseConfig, ConfigError, FactorExposureManager, DrawdownController, DrawdownState, DrawdownStatus, RiskMetrics, StopLossManager, StopReason, StopResult, CovarianceEstimator, FactorExposureMonitor, compute_regime_covariance, get_regime_covariance, compute_factor_exposures, compute_residual_returns, optimize_portfolio, decompose_returns, compute_attribution_report, compute_rolling_attribution, run_stress_scenarios, run_historical_drawdown_test, correlation_stress_test, factor_stress_test, replay_with_stress_constraints, compute_robustness_score, optimize_rebalance_cost, CRISIS_SCENARIOS

**evaluation/__init__.py exports**: PerformanceSlice, SliceRegistry, EvaluationEngine, compute_slice_metrics, decile_spread, pnl_concentration, drawdown_distribution, recovery_time_distribution, detect_critical_slowing_down, feature_importance_drift, ensemble_disagreement, analyze_calibration

**validation/__init__.py**: Empty (no explicit exports)

**autopilot/__init__.py exports**: StrategyCandidate, StrategyDiscovery, PromotionDecision, PromotionGate, StrategyRegistry, PaperTrader, AutopilotEngine, MetaLabelingModel

**kalshi/__init__.py exports**: KalshiClient, EventTimeStore, KalshiProvider, KalshiPipeline, KalshiDataRouter, QualityDimensions, StalePolicy, EventMarketMappingStore, EventMarketMappingRecord, DistributionConfig, EventFeatureConfig, EventWalkForwardConfig, EventWalkForwardResult, EventPromotionConfig, build_distribution_panel, build_options_reference_panel, add_options_disagreement_features, build_event_feature_panel, build_event_labels, build_asset_response_labels, run_event_walkforward, evaluate_event_promotion, evaluate_event_contract_metrics, add_reference_disagreement_features, asof_join

**api/__init__.py**: Empty (FastAPI backend)

**utils/__init__.py**: Empty

---

## OUTPUT LOCATION

Save the completed inventory to: `docs/audit/MODULE_INVENTORY.yaml`

Create the `docs/audit/` directory if it doesn't exist.
