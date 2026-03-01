# Audit Report: Subsystem 08 — Autopilot (Strategy Discovery)

> **Status:** Complete
> **Auditor:** Claude Opus 4.6
> **Date:** 2026-02-28
> **Spec:** `docs/audit/subsystem_specs/SPEC_AUDIT_08_AUTOPILOT.md`

---

## Findings (Severity-Ranked)

### F-01 — INTERFACE_CONTRACTS.yaml documents stale `walk_forward_validate` signature [P1 HIGH]

- **File:** `docs/audit/data/INTERFACE_CONTRACTS.yaml:360-367`
- **Proof:** The contract documents:
  ```
  signature: 'def walk_forward_validate(features: pd.DataFrame, targets: pd.Series,
    regimes: pd.Series, n_folds: int = 5, horizon: int = 10, verbose: bool = False)
    -> WalkForwardResult'
  ```
  The actual function signature at `backtest/validation.py:197-206` is:
  ```python
  def walk_forward_validate(
      predictions: pd.Series, actuals: pd.Series, n_folds: int = 5,
      entry_threshold: float = 0.005, max_overfit_ratio: float = 2.5,
      purge_gap: int = 10, embargo: int = 5,
      max_train_samples: Optional[int] = None,
  ) -> WalkForwardResult
  ```
  The engine calls this correctly at `autopilot/engine.py:807-814` using `predictions=`, `actuals=`, `entry_threshold=`, `purge_gap=`, `embargo=`, `max_train_samples=`.
- **Impact:** The INTERFACE_CONTRACTS.yaml is the authoritative boundary contract reference for audits. Any downstream audit relying on this contract will have an incorrect understanding of the `walk_forward_validate` API surface. The code itself is correct — this is a documentation drift issue, not a runtime bug.
- **Recommendation:** Update the INTERFACE_CONTRACTS.yaml entry (boundary `autopilot_to_backtest_31`) to reflect the actual signature.

### F-02 — `strategy_allocator.py` is unused — not integrated into engine or paper trader [P2 MEDIUM]

- **File:** `autopilot/strategy_allocator.py` (196 lines)
- **Proof:**
  - `grep -r "StrategyAllocator\|strategy_allocator" autopilot/engine.py autopilot/paper_trader.py` returns zero matches.
  - `StrategyAllocator` is not exported from `autopilot/__init__.py` (lines 5-21).
  - The module's own docstring says it should be consumed by `AutopilotEngine`, `Backtester`, and `PaperTrader` — none of which import it.
  - The only non-test consumer is `config.py:616` which declares `REGIME_STRATEGY_ALLOCATION_ENABLED = True`, but no code checks this flag.
  - Test coverage exists in `tests/test_system_innovations.py:38-412`.
- **Impact:** 196 lines of regime-aware parameter blending logic exist but have zero production integration. The `REGIME_STRATEGY_ALLOCATION_ENABLED` config flag is active but unread. This is dead code that misleads about system capabilities. The module's docstring overstates integration.
- **Recommendation:** Either integrate `StrategyAllocator` into `engine.py`/`paper_trader.py` as designed, or mark the module and config flag as `PLACEHOLDER` with a tracking issue.

### F-03 — `ENSEMBLE_DISAGREEMENT_WARN_THRESHOLD` import uses try/except with hardcoded fallback [P2 MEDIUM]

- **File:** `autopilot/engine.py:1905-1908`
- **Proof:**
  ```python
  try:
      from ..config import ENSEMBLE_DISAGREEMENT_WARN_THRESHOLD
  except ImportError:
      ENSEMBLE_DISAGREEMENT_WARN_THRESHOLD = 0.015
  ```
  The constant exists at `config.py:683` with value `0.015`. The `try/except ImportError` guard is unnecessary — this is not a circular import or optional dependency. Every other config import in `engine.py` (lines 34-53) is a normal top-level import without a guard.
- **Impact:** If the config constant is renamed or removed, the fallback silently takes over with a hardcoded value, masking the configuration change. This is inconsistent with the established import pattern in the same file.
- **Recommendation:** Move to top-level import alongside other config imports (line 34-53 block) without try/except.

### F-04 — Composite promotion score uses mixed-scale raw metrics without normalization [P2 MEDIUM]

- **File:** `autopilot/promotion_gate.py:304-325`
- **Proof:**
  ```python
  score = (
      1.30 * result.sharpe_ratio          # typical range: 0.5-3.0
      + 0.80 * result.annualized_return    # typical range: 0.05-0.50
      + 0.35 * result.win_rate             # range: 0.0-1.0
      + 0.20 * min(result.profit_factor, 5.0)  # range: 0.0-5.0
      + 0.01 * min(result.total_trades, 5000)  # range: 0-5000 -> 0-50
      + 0.80 * max(result.max_drawdown, -0.50) # range: -0.50-0.0
  )
  ```
  The `0.01 * total_trades` term can contribute up to +50 to the composite score, dominating all other terms combined (which typically sum to ~3-5). A strategy with 5000 trades and mediocre metrics would outscore a strategy with 50 trades and excellent metrics.
- **Impact:** The ranking score used to prioritize promotions is heavily skewed by trade count. This could promote high-frequency-but-mediocre strategies over lower-frequency-but-superior ones.
- **Recommendation:** Either normalize all terms to comparable scales (e.g., z-scores or [0,1] ranges) or reduce the trade-count coefficient. The current `min(total_trades, 5000)` cap helps but `0.01 * 5000 = 50` still dominates.

### F-05 — `_save_ic_to_health_tracking` best-IC tracking has O(n^2) scan and fragile equality check [P2 MEDIUM]

- **File:** `autopilot/engine.py:1846-1858`
- **Proof:**
  ```python
  for d in decisions:
      ...
      if ic is not None:
          ic_values.append(float(ic))
          ...
          if float(ic) == max(ic_values):  # re-scans list each iteration
              cand = d.candidate ...
              best_strategy_id = cand.strategy_id
  ```
  1. `max(ic_values)` is called inside the loop, re-scanning the growing list each iteration — O(n^2).
  2. Floating-point equality (`==`) for max comparison is fragile; identical-valued ICs could miss updates due to float representation.
- **Impact:** With typical candidate counts (< 100), performance is negligible. However, the float equality check could misattribute the "best" strategy when multiple candidates have identical IC values to floating-point precision.
- **Recommendation:** Track `best_ic` and `best_strategy_id` as running variables updated by `>` comparison, eliminating both the O(n^2) scan and float equality issue.

### F-06 — Paper trader health gate reads only latest health score without staleness check [P3 LOW]

- **File:** `autopilot/paper_trader.py:189-198`
- **Proof:**
  ```python
  history = svc.get_health_history(limit=1)
  if history:
      latest_score = history[-1].get("overall_score")
      if latest_score is not None:
          self._health_risk_gate.update_health(latest_score)
          return self._health_risk_gate.apply_health_gate(position_size_pct, latest_score)
  ```
  No timestamp check on the health record. If the health service hasn't updated in days, the paper trader still uses the stale score for position sizing.
- **Impact:** In degraded-health-service scenarios, paper trading could operate with an outdated health score indefinitely, neither failing safe (reduced sizing) nor explicitly marking the score as stale.
- **Recommendation:** Check the health record timestamp against a configurable staleness threshold (e.g., 24h). If stale, either skip the gate (pass-through) or apply a conservative default.

### F-07 — `INTERFACE_CONTRACTS.yaml` documents `size_position` (21 params) but paper trader uses `size_position_paper_trader` (18 params) [P3 LOW]

- **File:** `docs/audit/data/INTERFACE_CONTRACTS.yaml` (boundary `autopilot_to_risk_30`)
- **Proof:** The contract documents `PositionSizer.size_position()` with 21 parameters as the interface consumed by autopilot. The actual call in `autopilot/paper_trader.py` uses `size_position_paper_trader()` (defined at `risk/position_sizer.py:338`) with 18 parameters. These are two different methods with different signatures.
- **Impact:** Boundary documentation overstates the coupling surface. The paper trader's actual dependency is narrower than documented.
- **Recommendation:** Add `size_position_paper_trader` as a separate contract entry under `autopilot_to_risk_30` or update the existing entry.

### F-08 — `__init__.py` exports do not include `StrategyAllocator` [P3 LOW]

- **File:** `autopilot/__init__.py:5-21`
- **Proof:** `__all__` exports 8 symbols: `StrategyCandidate`, `StrategyDiscovery`, `PromotionDecision`, `PromotionGate`, `StrategyRegistry`, `PaperTrader`, `AutopilotEngine`, `MetaLabelingModel`. `StrategyAllocator` is omitted despite being a public class in the package.
- **Impact:** External consumers using `from autopilot import StrategyAllocator` would get an import error unless they import directly from the submodule. This is consistent with F-02 (module is unused), but if/when integration happens, the export should be added.
- **Recommendation:** Add to `__init__.py` exports if/when the module is integrated, or add a comment noting the intentional omission.

---

## Executive Summary

All **8 autopilot files** and **4,480 lines** were reviewed line-by-line, with read-only contract verification against `backtest/validation.py`, `backtest/execution.py`, `risk/position_sizer.py`, `api/services/health_service.py`, `api/ab_testing.py`, `kalshi/promotion.py`, and relevant `config.py` constants. The orchestration pipeline (discovery → evaluation → promotion → paper-trading) is structurally sound. All 6 circular API dependencies are properly guarded with lazy imports and try/except. The promotion gate implements all required statistical gates (DSR, PBO, CPCV, Monte Carlo, SPA, walk-forward, capacity, stress resilience). However, contract documentation drift exists in INTERFACE_CONTRACTS.yaml, 196 lines of strategy allocation code are dead, and the composite promotion score has a trade-count scaling issue.

- **P0:** 0
- **P1:** 1
- **P2:** 4
- **P3:** 3

No production code changes were made.

---

## Scope & Ledger (T1)

### File Coverage

| # | File | Lines | Reviewed |
|---|---|---:|---|
| 1 | `autopilot/engine.py` | 1927 | Yes |
| 2 | `autopilot/paper_trader.py` | 1254 | Yes |
| 3 | `autopilot/meta_labeler.py` | 468 | Yes |
| 4 | `autopilot/promotion_gate.py` | 424 | Yes |
| 5 | `autopilot/strategy_allocator.py` | 196 | Yes |
| 6 | `autopilot/registry.py` | 110 | Yes |
| 7 | `autopilot/strategy_discovery.py` | 79 | Yes |
| 8 | `autopilot/__init__.py` | 22 | Yes |

**Total: 4,480 lines reviewed (100% of Subsystem 8).**

### Read-Only Contract Files Reviewed

| File | Why |
|---|---|
| `backtest/validation.py` | `walk_forward_validate` signature parity (boundary `autopilot_to_backtest_31`) |
| `backtest/execution.py` | `ExecutionModel` constructor parity with `paper_trader.py` usage |
| `risk/position_sizer.py` | `size_position_paper_trader()` method signature and parameter contract |
| `risk/stop_loss.py` | `StopLossManager` import contract |
| `risk/portfolio_risk.py` | `PortfolioRiskManager` import contract |
| `api/services/health_service.py` | Circular import targets: `HealthService`, `save_ic_snapshot`, `save_disagreement_snapshot`, `save_execution_quality_fill` |
| `api/services/health_risk_feedback.py` | Circular import target: `create_health_risk_gate` |
| `api/ab_testing.py` | Circular import target: `ABTestRegistry` |
| `kalshi/promotion.py` | Consumer of `PromotionDecision`, `PromotionGate`, `StrategyCandidate` |
| `kalshi/pipeline.py` | Consumer of `PromotionDecision` |
| `api/jobs/autopilot_job.py` | Consumer of `AutopilotEngine` |
| `config.py` | All 60+ config constants imported by autopilot files |

### Shared Artifact Schema Snapshot

| Artifact | Producer | Key Fields | Consumers |
|---|---|---|---|
| `latest_cycle.json` (`AUTOPILOT_CYCLE_REPORT`) | `engine.py:run_cycle()` | `timestamp`, `n_candidates`, `n_passed`, `promotions[]`, `paper_report{}`, `predictor_info{}`, `ic_mean`, `disagreement_mean` | `api/jobs/autopilot_job.py`, dashboard display |
| `strategy_registry.json` (`STRATEGY_REGISTRY_PATH`) | `registry.py:apply_promotions()` | `active[]` (strategy_id, score, promoted_at, metrics), `history[]` | `paper_trader.py:_load_state()`, API consumers |
| `paper_state.json` (`PAPER_STATE_PATH`) | `paper_trader.py:_save_state()` | `equity`, `positions{}`, `pending_orders[]`, `trade_history[]`, `last_cycle_ts`, `kelly_history{}` | `paper_trader.py:_load_state()`, API dashboard |

---

## Orchestration Correctness Pass (T2)

### Discovery → Evaluation → Promotion → Paper-Trading Flow

The control flow in `engine.py:run_cycle()` (line 1691) executes:

1. **Ensure predictor** (`_ensure_predictor()`, line 299): Loads or trains an `EnsemblePredictor`. Falls back to `HeuristicPredictor` if model artifacts are unavailable. Fallback is explicit (logged at WARNING level) and returns a valid predictor interface.

2. **Generate candidates** (`StrategyDiscovery.generate()`, line 1706): Deterministic Cartesian product of entry multipliers × confidence offsets × risk variants × max position variants. No randomness, reproducible.

3. **Walk-forward predictions** (`_walk_forward_predictions()`, line 447): Expanding-window folds via `_expanding_walk_forward_folds()` from `models/walk_forward.py`. Predictions are merged across folds with NaN-safe alignment.

4. **Evaluate candidates** (`_evaluate_candidates()`, line 607): Per-candidate loop running:
   - Signal generation with entry/confidence thresholds
   - Walk-forward validation (`walk_forward_validate`, line 807)
   - Full backtest via `Backtester` (line 873)
   - Statistical tests: DSR, PBO, Monte Carlo, CPCV, SPA (lines 911-1010)
   - Regime-conditional performance analysis (lines 1025-1098)
   - Stress-regime capacity analysis (lines 1112-1175)
   - Meta-labeling confidence filtering when enabled (line 1220)

5. **Promotion gate** (`PromotionGate.evaluate()`, line 128): Multi-gate evaluation (see T3 below). Composite score ranking for passers.

6. **Registry update** (`StrategyRegistry.apply_promotions()`, line 1770): Capped at `PROMOTION_MAX_ACTIVE_STRATEGIES`, ranked by score, persisted to JSON.

7. **Paper-trading cycle** (`PaperTrader.run_cycle()`, line 1785): Stateful execution using promoted strategies (see T3 below).

8. **Health tracking** (`_save_ic_to_health_tracking()`, line 1829; `_save_disagreement_to_health_tracking()`, line 1881): Persists IC and disagreement metrics to health database for monitoring.

### Fallback Behavior Verification

- **Missing predictor:** `_ensure_predictor()` (line 299) catches `ModelTrainer` failures and returns `HeuristicPredictor` (defined at line 71). HeuristicPredictor implements `predict()` with signal-based heuristics, providing a valid but degraded prediction interface. **PASS.**
- **Missing features:** Feature pipeline failures are caught at line 1698 with explicit error logging. **PASS.**
- **Missing regime:** `RegimeDetector` failure returns `None` regime; downstream code handles `None` regime throughout. **PASS.**

---

## Paper-Trading Execution/Risk Parity Pass (T3)

### Execution Model Parity

`paper_trader.py:55` imports `ExecutionModel` from `backtest/execution.py`. The `ExecutionModel` constructor at `backtest/execution.py:169` accepts the same `EXEC_*` config constants that `paper_trader.py` imports at lines 24-53. Both use identical cost model parameters:

| Parameter | Paper Trader Import (line) | Execution Model Usage |
|---|---|---|
| `EXEC_SPREAD_BPS` | 24 | Base spread cost |
| `EXEC_MAX_PARTICIPATION` | 25 | Participation rate cap |
| `EXEC_IMPACT_COEFF_BPS` | 26 | Market impact coefficient |
| `EXEC_MIN_FILL_RATIO` | 27 | Minimum fill threshold |
| `EXEC_DYNAMIC_COSTS` | 28 | Dynamic cost toggle |
| `EXEC_STRUCTURAL_STRESS_ENABLED` | 35 | Structural stress overlay |
| All 11 remaining EXEC_* params | 29-53 | Matching cost model inputs |

**Verdict: Full parity confirmed.** Paper trader uses the same `ExecutionModel` class and identical config constants as backtest execution.

### Position Sizing Parity

`paper_trader.py:58` imports `PositionSizer` from `risk/position_sizer.py`. The `_position_size_pct()` method (line 651) calls `PositionSizer.size_position_paper_trader()` (defined at `risk/position_sizer.py:338`) with 18 parameters including Kelly sizing, regime state, drawdown ratio, regime entropy, and drift score. This is a paper-trader-specific method that wraps the core sizing logic with paper-trading context. **PASS.**

### Stop Loss and Drawdown Controls

- `StopLossManager` imported from `risk/stop_loss.py` (line 59). Applied in position management logic.
- `PortfolioRiskManager` imported from `risk/portfolio_risk.py` (line 60). Applied for portfolio-level risk limits.
- Maximum drawdown tracking in paper trader state with circuit-breaker behavior. **PASS.**

### State Recovery and Persistence

- State saved to `PAPER_STATE_PATH` (paper_state.json) via `_save_state()`.
- State loaded via `_load_state()` on initialization with graceful handling of missing/corrupt files.
- Schema includes: equity, positions, pending orders, trade history, Kelly history, timestamps.
- **PASS** — recovery is reliable; missing state starts fresh.

---

## API Coupling and Degradation-Safety Pass (T4)

### Circular Dependency Inventory

All 6 circular edges between autopilot and api are confirmed with lazy imports and try/except guards:

| # | File:Line | Import Target | Guard Type | Failure Behavior |
|---|---|---|---|---|
| 1 | `engine.py:1868` | `api.services.health_service.HealthService` | Lazy (inside method), outer try/except | Logs warning, continues cycle |
| 2 | `engine.py:1911` | `api.services.health_service.HealthService` | Lazy (inside method), outer try/except | Logs warning, continues cycle |
| 3 | `paper_trader.py:173` | `api.services.health_risk_feedback.create_health_risk_gate` | Static method, try/except | Returns `None`, health gate disabled |
| 4 | `paper_trader.py:189` | `api.services.health_service.HealthService` | Inside method, try/except | Logs debug, returns original size |
| 5 | `paper_trader.py:211` | `api.ab_testing.ABTestRegistry` | Inside method, try/except | Logs debug, returns `None` |
| 6 | `paper_trader.py:532` | `api.services.health_service.HealthService` | Inside method, try/except | Logs debug, drops record |

### Degradation Analysis

**All 6 circular imports degrade safely:**

1. **IC tracking (engine.py:1868):** If HealthService is unavailable, the exception is caught and logged at WARNING level. The cycle continues normally — IC tracking is purely observational.
2. **Disagreement tracking (engine.py:1911):** Same pattern as IC tracking. Non-blocking, observational only.
3. **Health risk gate (paper_trader.py:173):** `_init_health_gate()` returns `None` on failure. All callers check `if self._health_risk_gate is None: return position_size_pct` — full pass-through.
4. **Health gate apply (paper_trader.py:189):** Wrapped in try/except, returns original `position_size_pct` on any failure.
5. **A/B testing (paper_trader.py:211):** Returns `None`, and all A/B test call sites check for `None` before applying variants.
6. **Execution quality (paper_trader.py:532):** Pure telemetry — failure is silently dropped with debug log.

**No hidden hard dependency on API boot order.** The autopilot subsystem can run its full cycle with zero API services available. **PASS.**

---

## Meta-Labeling and Allocation Pass (T5)

### Meta-Labeling Model (`meta_labeler.py`)

- **Optional dependency guards:** XGBoost and joblib are conditionally imported with `_HAS_XGB`/`_HAS_JOBLIB` flags (lines 33-45). All model operations check these flags before proceeding. **PASS.**
- **Feature schema:** `META_FEATURE_COLUMNS` (10 features) defines canonical ordering. `build_meta_features()` constructs features from predictions, volatility, regime, and market context.
- **Model persistence:** Saves to `MODEL_DIR/meta_labeler/` with timestamped filenames. `save()`/`load()` use joblib serialization with version tracking.
- **Confidence filtering:** `predict_confidence()` returns P(signal_correct). Engine applies `META_LABELING_CONFIDENCE_THRESHOLD` filter at line 1220.
- **Retrain frequency:** Controlled by `META_LABELING_RETRAIN_FREQ_DAYS` with stale-model detection.
- **Min samples guard:** `META_LABELING_MIN_SAMPLES` prevents training on insufficient data.

### Strategy Allocator (`strategy_allocator.py`)

As documented in **F-02**, this module is complete and tested but not integrated:
- Defines `REGIME_STRATEGY_PROFILES` for regimes 0-3 with 7 parameters each.
- `StrategyAllocator.get_regime_profile()` blends regime-specific and default profiles weighted by confidence.
- Integer parameters (max_positions, holding_days) are properly rounded.
- `REGIME_NAMES` import from config is used only in `summarize()` display method.
- **Integration status: NOT CONNECTED.** Neither `engine.py` nor `paper_trader.py` imports or uses this module.

### Registry (`registry.py`)

- `StrategyRegistry` manages active strategies persisted to `STRATEGY_REGISTRY_PATH`.
- `apply_promotions()` ranks decisions by score, caps to `PROMOTION_MAX_ACTIVE_STRATEGIES`.
- History pruning: keeps last 2000 entries to prevent unbounded growth.
- `ActiveStrategy` dataclass: `strategy_id`, `score`, `promoted_at`, `entry_threshold`, `confidence_threshold`, `metrics`.
- Thread-safety: No explicit locking. Single-threaded autopilot cycle assumption is correct for current architecture.

---

## Boundary Contract Pass (T6)

### `autopilot_to_api_circular_5` Disposition

**PASS with caveat.** All 6 circular edges are properly guarded (see T4 above). No runtime coupling exists — all imports are lazy and failure-safe. The architectural coupling is telemetry-only (health tracking, A/B testing, execution quality). The autopilot subsystem can operate completely independently of the API layer.

### `autopilot_to_multi_4` Disposition

**PASS.** Autopilot imports 23+ symbols across 8 upstream modules:
- `backtest`: `Backtester`, `ExecutionModel`, `ADVTracker`, `CostCalibrator`, `walk_forward_validate`, `run_statistical_tests`, `combinatorial_purged_cv`, `superior_predictive_ability`, `strategy_signal_returns`, `capacity_analysis`, `deflated_sharpe_ratio`, `probability_of_backtest_overfitting`, `_expanding_walk_forward_folds`
- `models`: `EnsemblePredictor`, `ModelTrainer`, `cross_sectional_rank`
- `risk`: `PositionSizer`, `StopLossManager`, `PortfolioRiskManager`, `optimize_portfolio`
- `regime`: `RegimeDetector`, `UncertaintyGate`
- `data`: `load_survivorship_universe`, `load_universe`, `filter_panel_by_point_in_time_universe`
- `features`: `FeaturePipeline`

All imports are top-level (lines 20-67) and stable. No version-gated or conditional imports for upstream production dependencies.

### `autopilot_to_backtest_31` Disposition

**PARTIAL — contract documentation drift (F-01).** Code-level coupling is correct: `engine.py` calls `walk_forward_validate` with the actual signature. But `INTERFACE_CONTRACTS.yaml` documents an outdated signature.

### `kalshi_to_autopilot_9` Disposition

**PASS.** Verified consumers:
- `kalshi/promotion.py:12`: imports `PromotionDecision`, `PromotionGate`
- `kalshi/promotion.py:13`: imports `StrategyCandidate`
- `kalshi/pipeline.py:25`: imports `PromotionDecision`

All imported symbols are stable public API defined in `autopilot/promotion_gate.py` and `autopilot/strategy_discovery.py`.

### `autopilot_to_config_25` Disposition

**PASS.** All config constants are verified to exist:
- `engine.py` imports 18 constants (lines 34-53) — all exist in `config.py`
- `paper_trader.py` imports 41 constants (lines 13-54) — all exist in `config.py`
- `promotion_gate.py` imports 29+ constants (lines 13-42) — all exist in `config.py`
- `meta_labeler.py` imports 5 constants (lines 23-29) — all exist in `config.py`
- `strategy_discovery.py` imports 7 constants — all exist in `config.py`

Exception: `ENSEMBLE_DISAGREEMENT_WARN_THRESHOLD` at `engine.py:1906` uses try/except ImportError instead of top-level import (see F-03). The constant exists at `config.py:683`.

### `autopilot_to_models_28` Disposition

**PASS.** `EnsemblePredictor`, `ModelTrainer`, and `cross_sectional_rank` are imported and used correctly. `_expanding_walk_forward_folds` from `models/walk_forward.py` is used for fold generation.

### `autopilot_to_risk_30` Disposition

**PASS with documentation caveat (F-07).** `PositionSizer.size_position_paper_trader()` is the actual method used (18 params at `risk/position_sizer.py:338`), not `size_position()` (21 params) as documented in INTERFACE_CONTRACTS.yaml.

### `autopilot_to_data_26` Disposition

**PASS.** `load_survivorship_universe`, `load_universe`, and `filter_panel_by_point_in_time_universe` are imported at top-level and used in data loading paths.

### `autopilot_to_regime_29` Disposition

**PASS.** `RegimeDetector` and `UncertaintyGate` are imported at top-level and used for regime detection and entropy-based position gating.

### `autopilot_to_features_27` Disposition

**PASS.** `FeaturePipeline` is imported at top-level and used for feature generation.

---

## Architectural Concern: autopilot ↔ api Coupling

The autopilot subsystem has 6 circular dependency edges to the `api` layer, all arising from health tracking and A/B testing telemetry. While all are properly guarded with lazy imports and try/except, this represents the largest circular coupling surface in the codebase.

**Current state:** Safe. All circular paths are non-blocking telemetry. No API service failure can prevent autopilot from completing a cycle.

**Risk vector:** If future changes add decision-critical logic that depends on API health responses (e.g., gating promotions on health scores), the current lazy-import pattern would silently degrade to "no health data" rather than failing explicitly. The health gate in `paper_trader.py:179-201` is the closest to decision-critical — it scales position sizes based on health scores. Its degradation (pass-through on failure) is safe but means health-degraded markets could see full-sized positions.

**Recommendation:** Consider extracting a thin `health_client` interface that autopilot depends on, with the API layer providing the implementation. This would formalize the dependency direction and make the circular imports unnecessary.

---

## Promotion Gate Correctness (T2/T3 Detail)

### Gate Checks Verified

| Gate | File:Line | Config Constants | Verified |
|---|---|---|---|
| Minimum trades | `promotion_gate.py:149` | `PROMOTION_MIN_TRADES` | Yes |
| Win rate | `promotion_gate.py:151` | `PROMOTION_MIN_WIN_RATE` | Yes |
| Sharpe ratio | `promotion_gate.py:153` | `PROMOTION_MIN_SHARPE` | Yes |
| Profit factor | `promotion_gate.py:155` | `PROMOTION_MIN_PROFIT_FACTOR` | Yes |
| Max drawdown | `promotion_gate.py:157` | `PROMOTION_MAX_DRAWDOWN` | Yes (negative comparison is correct) |
| Annualized return | `promotion_gate.py:159` | `PROMOTION_MIN_ANNUAL_RETURN` | Yes |
| Negative Sharpe rejection | `promotion_gate.py:165` | None (hardcoded <= 0) | Yes |
| DSR significance | `promotion_gate.py:181` | `PROMOTION_MAX_DSR_PVALUE` | Yes |
| Monte Carlo significance | `promotion_gate.py:185` | None (boolean check) | Yes |
| PBO threshold | `promotion_gate.py:188-192` | `PROMOTION_MAX_PBO` | Yes |
| Capacity constraint | `promotion_gate.py:194-196` | `PROMOTION_REQUIRE_CAPACITY_UNCONSTRAINED` | Yes |
| Capacity utilization | `promotion_gate.py:196` | `PROMOTION_MAX_CAPACITY_UTILIZATION` | Yes |
| Stress capacity (SPEC-V02) | `promotion_gate.py:201-210` | `PROMOTION_MIN_STRESS_CAPACITY_USD` | Yes |
| Stress capacity ratio | `promotion_gate.py:211-214` | `PROMOTION_MIN_STRESS_CAPACITY_RATIO` | Yes |
| WF OOS correlation | `promotion_gate.py:228-232` | `PROMOTION_MIN_WF_OOS_CORR`, `PROMOTION_MIN_WF_POSITIVE_FOLD_FRACTION` | Yes |
| WF IS/OOS gap | `promotion_gate.py:234` | `PROMOTION_MAX_WF_IS_OOS_GAP` | Yes |
| Regime positive fraction | `promotion_gate.py:240` | `PROMOTION_REGIME_POSITIVE_FRACTION` | Yes |
| Fold consistency (Spec 04) | `promotion_gate.py:344-357` | `PROMOTION_WEIGHT_FOLD_CONSISTENCY` | Yes |

### Max Drawdown Comparison Note

At `promotion_gate.py:157`: `if result.max_drawdown < self.max_drawdown` — this is correct because `max_drawdown` values are negative (e.g., -0.15 means 15% drawdown). A result with -0.25 drawdown would be `< -0.20` threshold, correctly triggering rejection.

### Event Strategy Gate

`evaluate_event_strategy()` (line 402) merges event-specific metrics with standard contract metrics and delegates to `evaluate()` with `event_mode=True`. Event mode relaxes PBO requirement (line 189: skips `pbo_unavailable` reason). Additional event gates: `surprise_hit_rate` (line 291) and `event_regime_stability` (line 297).

---

## Verification Commands Executed

- `wc -l autopilot/*.py`
  - Output: 8 files, `4480 total`
- `grep -r "StrategyAllocator\|strategy_allocator" autopilot/engine.py autopilot/paper_trader.py`
  - Output: zero matches (confirms F-02)
- `grep -r "StrategyAllocator\|strategy_allocator" **/*.py`
  - Output: `strategy_allocator.py` self-references, test file, config comment, script reference only
- Verified all 6 circular edges line-by-line with try/except guard patterns
- Verified `walk_forward_validate` actual signature at `backtest/validation.py:197` vs. INTERFACE_CONTRACTS.yaml entry
- Verified `size_position_paper_trader` actual signature at `risk/position_sizer.py:338`
- Verified `ENSEMBLE_DISAGREEMENT_WARN_THRESHOLD` exists at `config.py:683`
- Verified `kalshi/promotion.py` and `kalshi/pipeline.py` import targets

---

## Acceptance Criteria Status

1. **Full line coverage across all 8 autopilot files:** **PASS** (4,480/4,480 lines)
2. **Promotion, paper-trading, and artifact contracts validated:** **PASS** (all gates verified, artifact schemas captured)
3. **API coupling degradation paths verified:** **PASS** (all 6 circular edges degrade safely)
4. **No unresolved P0/P1 findings in hotspot files:** **PARTIAL** (0 P0; 1 P1 — contract documentation drift, not a code defect)

---

## Final Assessment

Subsystem 08 is architecturally sound. The orchestration pipeline correctly chains discovery, evaluation, promotion, and paper-trading. All statistical gates (DSR, PBO, CPCV, Monte Carlo, SPA, walk-forward, capacity, stress resilience) are correctly wired. The 6 circular API dependencies are all properly guarded with lazy imports and graceful degradation. The highest-priority issues are contract documentation drift (F-01) and dead code (F-02, 196 lines). The composite promotion score scaling issue (F-04) warrants review to ensure ranking fidelity.
