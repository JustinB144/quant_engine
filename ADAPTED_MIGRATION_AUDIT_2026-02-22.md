# Quant Engine: Adapted System Audit & Migration Instructions

**Date:** 2026-02-22
**Base audit:** Codex System Audit 2026-02-22 (GPT-5)
**Review & adaptation by:** Claude Opus 4 (source-verified against live codebase + system reference DOCX)
**Scope:** Validate original audit findings, add missing gaps, strengthen migration instructions, add conceptual workflow improvements

---

## How to read this document

This document has three parts:

1. **Audit review** — What the original Codex audit got right, what it missed, and what needs correction
2. **Conceptual workflow improvements** — Architectural and workflow changes discovered by investigating the actual runtime code paths (not just documentation)
3. **Adapted migration instructions** — The complete FastAPI + React build brief with all corrections and additions applied

---

# PART 1: AUDIT REVIEW

## Original findings confirmed (no changes needed)

The following Codex findings are accurate and verified against live source:

- **P0-1 (Model Lab training callback):** Confirmed. `ModelTrainer.__init__` does not accept `tickers`, `horizon`, `feature_mode`, `ensemble` kwargs. The actual entry point is `train_ensemble(features, targets, regimes, regime_probabilities, horizon, ...)`. The demo fallback masks this completely.
- **P0-2 (Signal Desk import):** Confirmed. `ModelPredictor` does not exist; the class is `EnsemblePredictor`. Its `predict()` method requires `features`, `regimes`, `regime_confidence`, and optionally `regime_probabilities` — not a single `latest` argument.
- **P0-3 (Autopilot missing module):** Confirmed. `autopilot/discovery.py` does not exist (it is `strategy_discovery.py`), and `run_discovery_cycle()` is not a real function. Registry schema is `{"active": [], "history": []}`, not `{"strategies": [...]}`.
- **P0-4 (Kalshi DB schema drift):** Confirmed. UI queries `probability_snapshots` which does not exist; the actual table is `kalshi_distributions`. Column names are also mismatched.
- **P0-5 (Backtest page not running backtests):** Confirmed. The callback loads a pre-computed CSV and ignores the user-selected parameters.
- **P1-6 through P1-8 (test failures):** Confirmed. Cache metadata sidecar path, n_jobs=-1 in permutation importance, and Kalshi direction="above"/"below" handling are all real defects with failing tests.
- **P1-9 (silent demo fallbacks):** Confirmed across at least 6 pages. This is the single highest-risk integrity problem in the current UI.
- **P2-13 through P2-21:** All confirmed as stated.

## Findings that need correction or clarification

### Correction 1: Codex audit understates the backtest page problem (P0-5)

The audit says the page "does not run a backtest." This is accurate but incomplete. The deeper issue is that the page accepts user parameters (`holding_period`, `max_positions`, `entry_threshold`, `risk_mgmt`) through interactive controls, implying these affect the output. They do not. The parameters are accepted by the callback function signature but never passed to any engine. This is not just a missing feature — it is actively misleading because the UI gives the appearance of parameterized execution.

**Adapted recommendation:** The FastAPI replacement must either genuinely execute parameterized backtests (via async job) or remove the parameter controls entirely. Do not ship controls that do not affect output.

### Correction 2: The audit misses a critical gap in `loaders.py` reusability

The audit recommends "Optionally reuse selective pure helpers from `dash_ui/data/loaders.py` only after review." This is too generous. After investigation, at least the following loaders.py functions contain hardcoded simulated data or incomplete implementations:

- `collect_health_data()` → promotion funnel counts are hardcoded constants, not read from autopilot state
- `_check_walkforward()` → never populates `wf_windows`; UI always renders synthetic fold data
- `compute_regime_payload()` → works but has no error metadata in its return type
- `compute_model_health()` → partially works but silently returns defaults when model directory is missing

**Adapted recommendation:** Do not reuse `loaders.py` functions as-is. Instead, use them as a reference for what the FastAPI service layer needs to compute, then rewrite each function with proper error handling, provenance tracking, and actual data source connections.

### Correction 3: The audit does not address the `run_*.py` script → API gap

The five `run_*.py` scripts (`run_train.py`, `run_predict.py`, `run_backtest.py`, `run_autopilot.py`, `run_retrain.py`) are the actual working orchestration code. They correctly chain: data loading → feature pipeline → regime detection → model training/prediction → backtesting → validation. The Dash UI does NOT use these scripts — instead, it reimplements (incorrectly) the same orchestration inside callbacks.

**Adapted recommendation:** The FastAPI service layer should mirror the orchestration patterns in the `run_*.py` scripts, not the Dash callbacks. These scripts are the source of truth for correct method signatures and data flow.

## Findings the audit missed entirely

### NEW-1: EnsemblePredictor has a hidden regime-2 suppression

When the detected regime is mean-reverting (regime 2), `EnsemblePredictor.predict()` internally suppresses confidence to near-zero. The current Dash UI does not surface this, so a user looking at the Signal Desk would see low-confidence predictions without understanding why. The predicted_return value is still computed but the confidence is artificially depressed.

**Recommendation:** The FastAPI prediction response must include a `regime_suppressed: bool` flag and an explanation field when suppression is active. The React frontend should display a visible indicator (e.g., "Regime 2 suppression active — momentum signals unreliable in mean-reverting conditions").

### NEW-2: Cross-sectional ranking is applied silently in autopilot but not in predict

`run_predict.py` outputs raw predicted returns. `AutopilotEngine._predict_universe()` applies cross-sectional z-scoring (`cs_zscore = (pred - mean) / std` per date) before passing predictions to the backtester. This means autopilot backtest results and signal desk predictions are on different scales and not directly comparable.

**Recommendation:** The FastAPI prediction endpoint should include both `raw_predicted_return` and `cs_zscore` in its response. The backtest endpoint should document which input it uses. The frontend should display both values and clarify which one drives trade entry.

### NEW-3: Walk-forward fallback to single split is silent

`AutopilotEngine.run_cycle()` attempts expanding-window walk-forward validation. If insufficient data exists for multiple folds, it silently falls back to a single 80/20 train/test split with no OOS guarantee flagging. The `strict_oos` parameter exists but the fallback is not reported to the caller.

**Recommendation:** The FastAPI autopilot response must include `walk_forward_mode: "full" | "single_split" | "heuristic_fallback"` and `n_oos_folds_achieved: int`. The frontend should display an amber warning when the system fell back to single-split mode.

### NEW-4: Autopilot uses HeuristicPredictor as emergency fallback

If no trained model exists and training fails, `AutopilotEngine` falls back to `HeuristicPredictor` — a simple momentum+mean-reversion heuristic with no ML. This is a reasonable safety valve, but the current system does not report when this fallback is active. If the heuristic predictor drives paper trading, the user should know.

**Recommendation:** Include `predictor_type: "ensemble" | "heuristic"` in all prediction and autopilot responses. Display a red warning badge when heuristic fallback is active.

### NEW-5: Position sizing method varies by code path

Three different position sizing approaches exist: simple fixed percentage (backtest simple mode), Kelly/Vol/ATR blend (backtest risk-managed mode), and confidence-weighted optimizer weights (autopilot). The current UI does not indicate which method was used.

**Recommendation:** Include `sizing_method: "fixed" | "kelly_vol_atr" | "optimizer_weighted"` in backtest and paper trading results.

### NEW-6: The Data Explorer timeframe collision bug has downstream implications

The Codex audit correctly identifies that Data Explorer keys by ticker only (P2-11). What it misses is that this same pattern affects the benchmark comparison and backtest pages, which also load data into stores without timeframe context. If a user loads 1h data for AAPL in the Data Explorer, then navigates to the backtest page which expects daily data, the cached data could be incorrect.

**Recommendation:** All data loading in the FastAPI layer must include `(ticker, timeframe)` as the cache key. The API response schema must always include `timeframe` as a required field. The React frontend should validate that the data timeframe matches the page's expectation.

### NEW-7: No health check for stale trained models

The system has retrain triggers (schedule, performance, IC drift, etc.) but the Dash UI does not display model staleness as a metric. A model trained 45 days ago with a 30-day retrain schedule should show as overdue.

**Recommendation:** Add `model_staleness_days: int` and `retrain_overdue: bool` to the dashboard summary endpoint. Display as an amber/red indicator on the dashboard.

### NEW-8: Pandas 3.0 forward-compatibility risk (from original P3-22, not addressed in original migration plan)

The original Codex audit identified 524 FutureWarnings from pandas related to Copy-on-Write and chained assignment behavior. These appear in core modules: `features/pipeline.py`, `backtest/engine.py`, `models/cross_sectional.py`, `data/survivorship.py`, and others. The original audit correctly flagged this but the migration instructions did not include it as a remediation step.

**Recommendation:** Add a pandas-compatibility cleanup pass to Phase 0 (stabilize backend). Prefer `.loc[...] = ...` with explicit `.copy()` when transforming slices/views. Set `pd.options.mode.copy_on_write = True` in test configuration to surface breakage early. This is not a blocker for migration but should be addressed before pandas 3.0 adoption to prevent silent behavioral changes in feature computation and backtesting.

### NEW-9: Config is not hot-reloadable

All 300+ config values are imported at module level. Changing config.py requires restarting the entire application. For a trading system where you might want to adjust entry thresholds, drawdown limits, or universe composition during market hours, this is a significant operational limitation.

**Recommendation:** The FastAPI backend should load config values at request time (or with a short cache), not at import time. Consider a `GET /api/config` and `PATCH /api/config` endpoint for runtime-adjustable parameters (with validation). Non-adjustable parameters (paths, DB connections) can remain static.

---

# PART 2: CONCEPTUAL WORKFLOW IMPROVEMENTS

These are architectural and workflow improvements discovered by tracing the actual runtime code paths.

## Improvement 1: Unified orchestration service

**Current state:** Five separate `run_*.py` scripts duplicate the same data-loading → feature-computation → regime-detection chain. The Dash UI duplicates it again (incorrectly). This means the same 15-line orchestration pattern exists in at least 6 places.

**Recommended change:** Create a single `QuantEngineOrchestrator` service class in the FastAPI layer that owns the canonical pipeline:

```
load_universe(tickers, years, source_policy)
  → compute_features(data, mode, include_targets)
    → detect_regimes(features)
      → predict(features, regimes, version)
        → backtest(predictions, data, params)
          → validate(backtest_result)
```

Each FastAPI endpoint calls into this orchestrator at the appropriate depth. Training calls the full chain. Prediction skips backtesting. Dashboard summary calls a cached snapshot. This eliminates duplication and ensures every code path uses the same correct method signatures.

## Improvement 2: Provenance-first response architecture

**Current state:** No response includes metadata about where data came from, how stale it is, or whether fallbacks were used.

**Recommended change:** Every API response wraps its payload in a standard envelope:

```json
{
  "data": { ... },
  "meta": {
    "data_mode": "live",
    "generated_at": "2026-02-22T14:30:00Z",
    "staleness_seconds": 45,
    "source_summary": "Model v20260220_143022, cache from 2026-02-21",
    "warnings": [],
    "predictor_type": "ensemble",
    "walk_forward_mode": "full",
    "regime_suppressed": false
  }
}
```

This is not optional. Every endpoint, every response. The React frontend renders a persistent `DataProvenanceBadge` component on every panel. The badge is green for live, amber for stale/fallback, red for demo/error.

## Improvement 3: Event-driven model lifecycle notifications

**Current state:** Retrain triggers are checked only when `run_retrain.py` runs manually. The UI shows retrain status passively via cached metrics.

**Recommended change:** The FastAPI backend should run a lightweight background monitor (every 5 minutes) that checks retrain triggers and model staleness. When a trigger fires, it should:
1. Log the trigger to a persistent audit trail
2. Emit a notification via SSE to connected frontends
3. Optionally auto-queue a retrain job (configurable)

This transforms the system from "check when asked" to "alert when needed."

## Improvement 4: Granular caching with invalidation

**Current state:** Flask-Caching with 60s timeout on everything. No way to invalidate specific caches. No distinction between data that changes every 30 seconds (regime state) and data that changes monthly (model metadata).

**Recommended cache tiers:**

| Data type | Cache duration | Invalidation trigger |
|-----------|---------------|---------------------|
| Model metadata / feature importance | 1 hour | New model trained |
| Regime state | 30 seconds | None (time-based) |
| Dashboard KPIs | 60 seconds | None (time-based) |
| System health | 5 minutes | Config change |
| Backtest results | Infinite (immutable) | None |
| Predictions | 30 seconds | None (time-based) |
| Data quality | 10 minutes | New data ingested |

The FastAPI layer should use a simple in-memory cache (e.g., `cachetools.TTLCache`) per endpoint, with explicit invalidation when training or data ingestion completes.

## Improvement 5: Separate read-only and write (job) endpoints

**Current state:** The Dash UI mixes data display and computation in the same callbacks. A user viewing the dashboard triggers expensive regime detection and risk computation.

**Recommended change:** Split all endpoints into two categories:

**Read-only (fast, cached):** `GET /api/dashboard/summary`, `GET /api/system-health/summary`, `GET /api/signals/latest`, etc. These serve cached snapshots and return in <100ms.

**Compute (async job):** `POST /api/jobs/train`, `POST /api/jobs/backtest`, `POST /api/jobs/predict`, `POST /api/jobs/autopilot`. These return immediately with a job ID. The frontend polls or subscribes via SSE for progress and completion.

This prevents the UI from becoming unresponsive during heavy computation and gives the user explicit control over when expensive operations run.

## Improvement 6: Explicit demo mode as a first-class feature

**Current state:** Demo mode is an unintentional side effect of exception handling. It activates silently when backend calls fail.

**Recommended change:** Demo mode should be a deliberate, opt-in feature controlled by a config flag or query parameter (`?demo=true`). When demo mode is active:
- Every response includes `"data_mode": "demo"` in metadata
- The frontend displays a persistent banner: "DEMO MODE — Data is simulated"
- Demo data is deterministic (seeded RNG, not `hash()`)
- Demo mode is useful for UI development and presentations, but never activated by accident

When demo mode is NOT active, errors produce explicit error responses with diagnostic information, never synthetic data.

## Improvement 7: Portfolio optimizer weights need transparency

**Current state:** `AutopilotEngine._compute_optimizer_weights()` produces confidence-weighted portfolio weights, but these weights are not surfaced anywhere. The user sees trade entries/exits but not why a particular position was sized the way it was.

**Recommended change:** The autopilot and paper trading endpoints should include a `position_sizing_breakdown` per trade showing: base weight from optimizer, confidence scaling factor, regime risk multiplier, drawdown state multiplier, and final weight. This makes the system auditable.

## Improvement 8: Feature pipeline versioning

**Current state:** Features are computed fresh each run. The feature store (`data/feature_store.py`) provides point-in-time caching but there is no versioning of the feature pipeline itself. If you add a new indicator, old cached features become stale with no warning.

**Recommended change:** Compute a pipeline fingerprint (hash of indicator list + computation parameters) and include it in feature store metadata. When loading cached features, verify the fingerprint matches the current pipeline. If not, force recomputation and log a warning. Include `feature_pipeline_version` in training and prediction responses.

---

# PART 3: ADAPTED MIGRATION INSTRUCTIONS

This is the complete, corrected build brief incorporating all findings above.

## Migration goals (unchanged from original, confirmed correct)

Build a new UI stack that:
- Preserves the concepts and information architecture of the Dash UI
- Removes all Dash UI code and starts from a clean implementation
- Is professional, fast, explicit about data provenance, and reliable
- Uses FastAPI for backend APIs and React for frontend
- Does not directly expose backend internals/files/schemas to the frontend

## Non-goals (unchanged, confirmed correct)

- Do not port Dash callback code line-for-line
- Do not reuse Dash page files or component code
- Do not preserve current demo-fallback behavior by default
- Do not make the frontend depend on raw filesystem paths or DB table names

## Concept parity map (corrected)

Original is correct. No changes needed to the route mapping:

| Current Dash Page | New React Route | Notes |
|-------------------|----------------|-------|
| Dashboard | `/dashboard` | Add model staleness indicator, retrain-overdue badge |
| System Health | `/system-health` | Mark each subsection as `live` or `illustrative` |
| Data Explorer | `/data-explorer` | Key by `(ticker, timeframe)`, include provenance |
| Model Lab | `/model-lab` | Training via async job, not synchronous callback |
| Signal Desk | `/signal-desk` | Include raw_predicted_return AND cs_zscore |
| Backtest & Risk | `/backtests` | Real parameterized execution via async job |
| IV Surface | `/iv-surface` | Backend-computed surfaces, frontend renders only |
| S&P Comparison | `/benchmark-comparison` | Fix animation interval leak |
| System Logs | `/logs` | SSE stream, no global logger side effects |
| Autopilot & Events | `/autopilot` | Include walk_forward_mode, predictor_type |

## Target architecture (adapted)

### Backend (`FastAPI`)

Location: `quant_engine/api/`

```
api/
├── main.py                    # FastAPI app factory with lifespan
├── config.py                  # Runtime-adjustable config loader
├── orchestrator.py            # Unified pipeline orchestration (NEW)
├── routers/
│   ├── dashboard.py
│   ├── system_health.py
│   ├── data_explorer.py
│   ├── model_lab.py
│   ├── signals.py
│   ├── backtests.py
│   ├── benchmark.py
│   ├── logs.py
│   ├── autopilot.py
│   ├── kalshi.py
│   ├── config_mgmt.py         # Runtime config read/update (NEW)
│   └── jobs.py
├── schemas/
│   ├── envelope.py            # Standard response wrapper with meta (NEW)
│   ├── dashboard.py
│   ├── system_health.py
│   ├── data_explorer.py
│   ├── model_lab.py
│   ├── signals.py
│   ├── backtests.py
│   ├── benchmark.py
│   ├── logs.py
│   ├── autopilot.py
│   ├── kalshi.py
│   └── jobs.py
├── services/
│   ├── data_service.py        # Wraps data/loader.py + cache + quality
│   ├── feature_service.py     # Wraps features/pipeline.py
│   ├── regime_service.py      # Wraps regime/detector.py
│   ├── model_service.py       # Wraps models/predictor.py + trainer.py
│   ├── backtest_service.py    # Wraps backtest/engine.py + validation
│   ├── risk_service.py        # Wraps risk/* modules
│   ├── autopilot_service.py   # Wraps autopilot/engine.py
│   ├── kalshi_service.py      # Wraps kalshi/* modules
│   ├── health_service.py      # System health computation (replaces loaders.py)
│   └── monitor_service.py     # Background retrain-trigger monitor (NEW)
├── jobs/
│   ├── runner.py              # SQLite-backed job queue (lightweight)
│   ├── train_job.py
│   ├── backtest_job.py
│   ├── predict_job.py
│   └── autopilot_job.py
├── cache/
│   ├── manager.py             # TTL cache with per-endpoint configuration
│   └── invalidation.py        # Event-driven cache invalidation
├── deps/
│   ├── providers.py           # Data provider lifecycle (lifespan-managed)
│   └── auth.py                # Optional API key auth for non-localhost
└── errors.py                  # Typed exception → HTTP response mapping
```

### Frontend (`React + TypeScript`)

Location: `quant_engine/webapp/`

**Recommended stack (adapted based on research):**

- **Build:** Vite
- **Framework:** React 19 + TypeScript
- **Routing:** React Router v7
- **Server state:** TanStack Query v5 (`gcTime` replaces `cacheTime`)
- **Local UI state:** Zustand (minimal, only for sidebar collapse, theme toggle, etc.)
- **OHLCV / candlestick charts:** TradingView Lightweight Charts (45KB, canvas-based, handles 10K+ points)
- **Analytical charts (equity curves, rolling metrics, heatmaps):** Apache ECharts (faster than Plotly for large datasets, better memory footprint)
- **3D IV surfaces only:** Plotly React (justified for specialized 3D financial visualization)
- **Tables:** TanStack Table with virtualization (lighter than AG Grid for this use case, no license needed)
- **Styling:** Tailwind CSS + design tokens (CSS custom properties)
- **Real-time:** SSE for job progress, polling for periodic snapshots

**Why these specific choices:**
- TradingView Lightweight Charts outperforms Plotly and ECharts for candlestick rendering — it was built specifically for financial data by the TradingView team
- ECharts handles large time series (10+ years daily data) without the Chrome memory issues that Plotly can cause
- TanStack Table is sufficient for trade logs and signal tables; AG Grid is enterprise overhead you don't need for a single-user system
- SSE is simpler than WebSocket for the unidirectional job-progress use case; WebSocket adds connection management complexity without benefit here

## Critical design requirements (adapted with additions)

### 1) Standard response envelope (REQUIRED — expanded from original)

Every API response must use this structure:

```python
class ResponseMeta(BaseModel):
    data_mode: Literal["live", "fallback", "demo"]
    generated_at: datetime
    staleness_seconds: Optional[int] = None
    warnings: list[str] = []
    source_summary: str
    # NEW fields from investigation:
    predictor_type: Optional[Literal["ensemble", "heuristic"]] = None
    walk_forward_mode: Optional[Literal["full", "single_split", "heuristic_fallback"]] = None
    regime_suppressed: Optional[bool] = None
    feature_pipeline_version: Optional[str] = None
    model_version: Optional[str] = None
    sizing_method: Optional[Literal["fixed", "kelly_vol_atr", "optimizer_weighted"]] = None

class ApiResponse(BaseModel, Generic[T]):
    data: T
    meta: ResponseMeta
```

### 2) Async jobs for long-running operations (expanded)

**Job system choice:** SQLite-backed job queue (not Celery, not ARQ). Reasoning: single-user system, no need for Redis dependency, SQLite handles 15K jobs/s which far exceeds requirements, file-based persistence survives restarts.

```
POST /api/jobs/train       → returns {job_id, status: "queued"}
POST /api/jobs/backtest    → returns {job_id, status: "queued"}
POST /api/jobs/predict     → returns {job_id, status: "queued"}
POST /api/jobs/autopilot   → returns {job_id, status: "queued"}
GET  /api/jobs/{id}        → returns {job_id, status, progress_pct, result?, error?}
GET  /api/jobs/{id}/events → SSE stream: progress updates + completion
POST /api/jobs/{id}/cancel → cancel if running
GET  /api/jobs/recent      → last 20 jobs with status
```

Job statuses: `queued` → `running` → `succeeded` | `failed` | `cancelled`

### 3) Typed contracts (confirmed, add detail)

**Critical: use the ACTUAL method signatures from `run_*.py` scripts, not from Dash callbacks.**

Correct signatures discovered from source investigation:

```python
# ModelTrainer.train_ensemble() — the REAL interface
train_ensemble(
    features: pd.DataFrame,            # MultiIndex (permno, date)
    targets: pd.Series,                # Same index
    regimes: pd.Series,                # Regime labels 0-3
    regime_probabilities: pd.DataFrame, # regime_prob_0..3 columns
    horizon: int,
    verbose: bool = True,             # NOTE: defaults to True in source
    versioned: bool = True,
    survivorship_mode: bool = False,
    recency_weight: bool = False,
) → EnsembleResult

# EnsemblePredictor.predict() — the REAL interface
predict(
    features: pd.DataFrame,
    regimes: pd.Series,
    regime_confidence: pd.Series,
    regime_probabilities: Optional[pd.DataFrame] = None,
) → pd.DataFrame  # columns: predicted_return, confidence, regime, blend_alpha, etc.

# Backtester.run() — the REAL interface
run(
    predictions: pd.DataFrame,         # MultiIndex (permno, date)
    price_data: Dict[str, pd.DataFrame],
    verbose: bool = False,
) → BacktestResult

# FeaturePipeline.compute_universe() — the REAL interface
compute_universe(
    data: Dict[str, pd.DataFrame],     # OHLCV per permno
    verbose: bool = True,              # NOTE: defaults to True in source
    compute_targets_flag: bool = True,
) → Tuple[pd.DataFrame, pd.DataFrame]  # (features, targets)

# RegimeDetector.regime_features() — the REAL interface
regime_features(
    features: pd.DataFrame,            # Single permno, date index
) → pd.DataFrame  # columns: regime, regime_confidence, regime_prob_0..3, etc.
```

### 4) Server-side caching (adapted with tiered durations)

| Endpoint | Cache TTL | Invalidation |
|----------|-----------|-------------|
| Dashboard KPIs | 60s | Time-based |
| Regime state | 30s | Time-based |
| System health | 300s | Config change or model retrain |
| Model metadata / feature importance | 3600s | New model trained |
| Backtest results (by job ID) | Infinite | Immutable |
| Predictions (latest) | 30s | Time-based |
| Data quality | 600s | New data ingested |
| Benchmark comparison | 60s | Time-based |
| Autopilot registry | 300s | Promotion event |

### 5) Error transparency (expanded with demo mode separation)

**Demo mode is a deliberate opt-in feature, never an accident:**

```python
# In API config
DEMO_MODE: bool = False  # Set via env var or config endpoint

# In every service function
if not data_available and not settings.DEMO_MODE:
    raise DataUnavailableError("No trained model found", diagnostic_info={...})
elif not data_available and settings.DEMO_MODE:
    return DemoResponse(data=generate_deterministic_demo(), meta={"data_mode": "demo"})
```

Error responses use structured JSON:

```json
{
  "error": {
    "type": "model_not_found",
    "message": "No trained model exists for horizon=10. Run training first.",
    "diagnostic": {
      "model_dir": "trained_models/",
      "versions_found": [],
      "suggestion": "POST /api/jobs/train with horizon=10"
    }
  }
}
```

## Backend API design (corrected endpoint blueprint)

### Dashboard

```
GET /api/dashboard/summary
  Response: {
    kpis: {
      portfolio_value, thirty_day_return, sharpe_ratio,
      current_regime, current_regime_name,
      retrain_status: {triggered: bool, reasons: [], overdue: bool, days_since_train: int},
      cv_gap, data_quality_score, system_health_score,
      model_staleness_days  # NEW
    },
    equity_series: [{date, strategy_return, benchmark_return}],
    regime: {
      current: {label: int, name: str, confidence: float, probabilities: [4 floats]},
      history: [{date, regime, confidence}],
      transition_matrix: [[4x4 floats]]
    },
    model_health: {
      holdout_r2, holdout_ic, ic_drift, cv_gap,
      active_retrain_triggers: [str]
    },
    feature_importance: {
      global_top20: [{name, importance}],
      regime_heatmap: {regime_0: [{name, importance}], ...}
    },
    recent_trades: [{ticker, entry_date, exit_date, predicted_return, actual_return, net_return, regime, confidence, exit_reason}],
    risk: {sharpe, sortino, max_drawdown, var_95, cvar_95, calmar, tail_ratio}
  }
```

**Split endpoints available if payload too large:**
```
GET /api/dashboard/kpis
GET /api/dashboard/equity
GET /api/dashboard/regime
GET /api/dashboard/model-health
GET /api/dashboard/risk
GET /api/dashboard/trades?limit=200
GET /api/dashboard/features
```

### System Health

```
GET /api/system-health/summary
  Response: {
    scores: {overall, data_integrity, promotion_gate, walk_forward, execution_realism, complexity},
    statuses: {overall: "healthy"|"caution"|"critical", ...per_score},
    strengths: [str],
    vulnerabilities: [str]
  }

GET /api/system-health/details
  Response: {
    data_integrity: {
      checks: [{name, passed: bool, value, threshold, fidelity: "live"|"illustrative"}],
      universe_freshness: {ticker: last_update_date}
    },
    promotion_gate: {
      requirements: [{gate, threshold, current_value}],
      funnel: {total_candidates, passed_sharpe, passed_dsr, passed_pbo, promoted},
      funnel_fidelity: "live"|"illustrative"  # CRITICAL: honest about whether this is real data
    },
    walk_forward: {
      windows: [{fold, train_start, train_end, test_start, test_end, is_corr, oos_corr}],
      windows_fidelity: "live"|"illustrative",
      avg_oos_corr, is_oos_gap, overfit_detected: bool
    },
    execution_realism: {checks: [{name, passed, details}]},
    complexity: {n_features, n_config_params, features_to_obs_ratio, checks: [...]}
  }
```

### Data Explorer

```
POST /api/data-explorer/load
  Request: {tickers: [str], timeframe: "daily"|"4h"|"1h"|"30m", source_policy: "trusted_first"|"any"}
  Response: {
    series: {
      "AAPL": {
        timeframe: "daily",  # ALWAYS included
        source: "wrds"|"cache"|"yfinance",
        quality_score: 92,
        data: [{date, open, high, low, close, volume}],  # downsampled if >5000 points
        total_bars: 3752,
        returned_bars: 2500,
        date_range: {start, end}
      }
    }
  }

GET /api/data-explorer/{ticker}/quality?timeframe=daily
  Response: {
    missing_bar_fraction, zero_volume_fraction, extreme_returns_count,
    duplicate_timestamps, composite_score, degraded: bool
  }
```

### Model Lab

```
GET /api/model-lab/features/importance?horizon=10&version=latest
  Response: {
    global_importance: [{feature_name, importance, rank}],
    regime_importance: {0: [{feature_name, importance}], ...},
    model_version: str,
    feature_pipeline_version: str  # NEW: pipeline fingerprint
  }

GET /api/model-lab/regime/payload
  Response: {
    current_regime: {label, name, confidence, probabilities, duration_bars},
    history: [{date, regime, confidence}],
    model_type: "hmm"|"rule",
    transition_matrix: [[4x4]]
  }

POST /api/jobs/train
  Request: {
    horizon: 10,
    universe: "full"|"quick"|["AAPL","NVDA"],
    years: 10,
    feature_mode: "core"|"full",
    survivorship: true,
    recency_weight: false
  }
  → {job_id: str, status: "queued"}
```

### Signal Desk

```
POST /api/signals/generate
  Request: {horizon: 10, universe: "full"|"quick", version: "latest"|"champion"|str, top_n: 20}
  Response: {
    as_of_date: str,
    predictions: [{
      permno: str, ticker: str,
      predicted_return: float,     # raw model output
      cs_zscore: float,            # NEW: cross-sectional z-score
      confidence: float,
      regime: int, regime_name: str,
      blend_alpha: float,
      regime_suppressed: bool,     # NEW: whether regime-2 suppression was active
    }],
    signals_above_threshold: int,
    total_predictions: int,
    predictor_type: "ensemble"|"heuristic"  # NEW
  }

GET /api/signals/latest?horizon=10
  → cached version of most recent generation
```

### Backtests

```
POST /api/jobs/backtest
  Request: {
    horizon: 10,
    universe: "full"|"quick",
    years: 15,
    entry_threshold: 0.005,
    confidence_threshold: 0.4,
    max_positions: 20,
    use_risk_management: true,
    version: "latest"
  }
  → {job_id: str, status: "queued"}

GET /api/backtests/{job_id}/result
  Response: {
    summary: {
      total_trades, win_rate, sharpe, sortino, max_drawdown, annual_return, profit_factor,
      sizing_method: "fixed"|"kelly_vol_atr"  # NEW: which sizing was used
    },
    equity_series: [{date, equity, drawdown}],
    trades: [{permno, ticker, entry_date, exit_date, predicted_return, actual_return,
              net_return, regime, confidence, exit_reason,
              position_size_pct, sizing_breakdown: {...}}],  # NEW: sizing transparency
    regime_breakdown: {0: {count, sharpe, win_rate}, ...},
    validation: {
      walk_forward: {n_folds, avg_oos_corr, is_oos_gap, passes: bool,
                     walk_forward_mode: str},  # NEW: full vs single_split
      statistical_tests: {spearman: {stat, p}, long_mean: {stat, p}, ...},
      dsr: {observed_sharpe, dsr_adjusted, p_value, n_variants_tested},
      pbo: {value, passes: bool}
    }
  }

GET /api/backtests/latest?horizon=10
GET /api/backtests/{job_id}/trades.csv
```

### IV Surface

```
POST /api/iv/surface
  Request: {ticker: str, model: "svi"|"heston"|"arb_aware", expiries: [int], strikes: [float]}
  Response: {
    surface_points: [{strike, expiry_days, iv}],
    model_params: {...},
    decomposition: {level: [float], slope: [float], curvature: [float]},
    smiles: [{expiry_days, points: [{strike, iv}]}]
  }
```

### Benchmark Comparison

```
GET /api/benchmark-comparison?horizon=10&period=ALL
  Response: {
    strategy_cumulative: [{date, return}],
    benchmark_cumulative: [{date, return}],
    rolling_alpha: [{date, alpha}],
    rolling_beta: [{date, beta}],
    rolling_correlation: [{date, corr}],
    drawdown_comparison: {strategy_max_dd, benchmark_max_dd, dates: [{date, strategy_dd, benchmark_dd}]},
    risk_adjusted: {strategy_sharpe, benchmark_sharpe, information_ratio, tracking_error}
  }
```

### Logs

```
GET /api/logs?level=INFO&limit=500&after=<timestamp>
  Response: {entries: [{timestamp, level, module, message}], total_available: int}

GET /api/logs/stream
  → SSE: {event: "log", data: {timestamp, level, module, message}}

POST /api/logs/clear
  → {cleared: int}
```

**Critical:** Log handler must be configured in `api/main.py` lifespan, NOT in a router module. No import-time side effects.

### Autopilot & Kalshi

```
GET /api/autopilot/registry
  Response: {
    active_strategies: [{strategy_id, entry_threshold, confidence_threshold,
                        max_positions, sharpe, win_rate, promoted_at, model_version}],
    history: [{strategy_id, status, score, promoted_at, deactivated_at}]
  }

POST /api/jobs/autopilot
  Request: {horizon: 10, years: 15, feature_mode: "core", max_candidates: 24, walk_forward: true}
  → {job_id, status: "queued"}

GET /api/autopilot/paper-summary
  Response: {
    equity: float, starting_capital: float,
    daily_pnl: [{date, equity, daily_return}],
    open_positions: [{ticker, side, size, entry_price, current_price, unrealized_pnl}],
    sharpe: float, max_drawdown: float,
    predictor_type: "ensemble"|"heuristic",  # NEW
    sizing_method: "kelly_weighted"           # NEW
  }

GET /api/kalshi/events?event_type=CPI
  Response: {
    events: [{event_id, event_type, release_date, status}],
    distributions: {current: [{bucket_label, probability}], history: [{date, distribution: [...]}]},
    walkforward: {accuracy: float, calibration: float, n_events: int},
    disagreement: [{date, direction, magnitude}]
  }
```

### Config Management (NEW)

```
GET /api/config
  Response: {
    adjustable: {
      entry_threshold: 0.005,
      confidence_threshold: 0.6,
      max_positions: 20,
      regime_2_trade_enabled: false,
      ...
    },
    static: {
      data_cache_dir: "(not shown)",
      wrds_enabled: true,
      ...
    }
  }

PATCH /api/config
  Request: {entry_threshold: 0.003, max_positions: 30}
  Response: {updated: {entry_threshold: {old: 0.005, new: 0.003}, ...}}
```

## Frontend implementation instructions (adapted)

### 1) App shell

Same concepts as original audit, with these additions:
- Global `DataProvenanceBadge` component rendered on every panel (green/amber/red)
- Global `DemoModeBanner` when demo mode is active
- Model staleness indicator in the top status bar
- Retrain-overdue badge next to model version display

### 2) Design system

Original audit's token list is correct. Add:
- `color.chart.regime[0-3]` — one color per regime for consistent identification across pages
- `color.provenance.live / fallback / demo` — badge colors
- `animation.pulse` — for the live status indicator

### 3) Data fetching (corrected TanStack Query config)

```typescript
// TanStack Query v5: gcTime replaces cacheTime

// Static model metadata
useQuery({
  queryKey: ['model', 'features', horizon, version],
  queryFn: () => api.getFeatureImportance(horizon, version),
  staleTime: 60 * 60 * 1000,     // 1 hour
  gcTime: 24 * 60 * 60 * 1000,   // 24 hours
})

// Regime state (periodic refresh)
useQuery({
  queryKey: ['regime', 'current'],
  queryFn: () => api.getRegimePayload(),
  staleTime: 20_000,              // 20 seconds
  gcTime: 5 * 60 * 1000,          // 5 minutes
  refetchInterval: 30_000,        // 30-second polling
})

// Backtest results (immutable)
useQuery({
  queryKey: ['backtest', jobId],
  queryFn: () => api.getBacktestResult(jobId),
  staleTime: Infinity,
  gcTime: 7 * 24 * 60 * 60 * 1000, // 1 week
  enabled: !!jobId,
})

// Dashboard KPIs (moderate refresh)
useQuery({
  queryKey: ['dashboard', 'kpis'],
  queryFn: () => api.getDashboardKPIs(),
  staleTime: 30_000,
  gcTime: 5 * 60 * 1000,
  refetchInterval: 60_000,
})
```

### 4) Chart library usage

```
TradingView Lightweight Charts:
  → Data Explorer OHLCV candlesticks
  → Any price chart (equity curves can use area series)

Apache ECharts:
  → Dashboard equity curve + benchmark overlay
  → Rolling metrics (Sharpe, volatility, drawdown)
  → Regime probability stacked area
  → Feature importance bar charts
  → Heatmaps (regime × feature importance)
  → Promotion funnel bar chart
  → Walk-forward fold performance

Plotly React:
  → IV Surface 3D mesh (only use case that justifies Plotly)
  → Transition matrix heatmap (optional, ECharts can do this too)
```

### 5) Long-running jobs UX

```
1. User clicks "Run Backtest" → POST /api/jobs/backtest → receives job_id
2. Frontend shows job card: [Backtest • Queued • ...]
3. Frontend subscribes to GET /api/jobs/{id}/events (SSE)
4. SSE emits: {progress: 0.15, message: "Computing features for 60 tickers..."}
5. Job card updates: [Backtest • Running • 15% • Computing features...]
6. SSE emits: {progress: 1.0, status: "succeeded", result_url: "/api/backtests/{id}/result"}
7. Job card shows: [Backtest • Complete • View Results]
8. User clicks → navigates to results page with cached data
```

**Recent jobs persist in Zustand store (survives page navigation, cleared on tab close).**

### 6) Performance rules (confirmed + additions)

- Downsample time series >2500 points on backend (largest-triangle-three-buckets)
- Virtualize tables with >100 rows (TanStack Virtual)
- Lazy-load page components (React.lazy + Suspense per route)
- Do not fetch all 10 pages' data on initial load — fetch on navigation
- Chart data should use typed arrays where possible for ECharts performance
- For Data Explorer: paginate OHLCV response (return first 2500 bars, offer "load more")

## Implementation plan (corrected phase ordering)

### Phase 0: Stabilize backend (bugs first — DO THIS BEFORE MIGRATION) — INVESTIGATED 2026-02-23

**Investigation results (verified against live source code):**

| Original Bug | Status | Detail |
|-------------|--------|--------|
| `data/local_cache.py` sidecar path | Exists (LOW) | 6 `read_parquet()` calls without `filters=` — optimization miss, not a crash |
| `kalshi/distribution.py` scipy guard | **Does not exist** | File doesn't import or use scipy.stats at all |
| `models/trainer.py` LightGBM callback | **Does not exist** | Code calls `.fit(X, y, sample_weight=w)` with no callbacks |
| Pandas `swaplevel()` deprecation | **Confirmed** | 15+ locations across `autopilot/engine.py`, `run_backtest.py`, `regime/detector.py`, etc. |
| Pandas `df.append()` removal | **Does not exist** | Only list `.append()` found, no DataFrame `.append()` |

**Remaining Phase 0 work:**
1. ~~Fix `kalshi/distribution.py` direction handling~~ — Not a real bug
2. ~~Fix `models/trainer.py` callback API~~ — Not a real bug
3. Fix `data/local_cache.py` `read_parquet()` to add `filters=` param for performance (LOW priority)
4. Fix `swaplevel()` → `reorder_levels()` in 15+ files for pandas 3.0 compatibility (MEDIUM priority)
5. Begin pandas 3.0 compatibility cleanup: fix chained assignment patterns in `features/pipeline.py`, `backtest/engine.py`, `models/cross_sectional.py`, `data/survivorship.py`

### Phase 1: Build FastAPI backend — COMPLETE (commit `e25437d`)

1. Created `api/` package with `main.py`, lifespan-managed providers, and response envelope
2. Implemented `orchestrator.py` — unified pipeline chain mirroring `run_*.py` scripts
3. Built 27 read-only + compute endpoints across 11 routers
4. Added SQLite job queue with train/backtest/predict/autopilot jobs
5. Added SSE endpoints for job progress and log streaming
6. Added tiered caching with per-endpoint TTLs

### Phase 2: Build React frontend — COMPLETE (commit `3185fa2`)

136 files, 12,542 insertions. All 10 pages implemented:
1. App shell + sidebar + design system + DataProvenanceBadge
2. Dashboard (8 KPI cards, 6 tabs, live API data)
3. Signal Desk (signal generation, rankings table, distribution + scatter charts)
4. Backtest & Risk (config panel, job submission, equity curve, risk metrics, trade analysis)
5. Data Explorer (universe selector, candlestick chart with volume, quality metrics)
6. System Health (radar chart, score cards, collapsible check panels)
7. Benchmark Comparison (metric cards, detailed comparison)
8. System Logs (level filters, search, auto-refresh 3s)
9. Model Lab (features, regime, training tabs with job submission)
10. Autopilot (strategy candidates, paper trading, Kalshi, lifecycle tabs)
11. IV Surface (SVI + Heston tabs with 3D plotting)

### Phase 3: Parity validation — COMPLETE (2026-02-23)

**Validation results:**

All 10 sidebar routes match (same labels, same paths). All core concepts covered.

**Verified working:**
- Provenance badges on all chart panels
- Error states handled with ErrorPanel + ErrorBoundary (route-aware with `key={location.pathname}`)
- Job lifecycle works (train, backtest, predict, autopilot)
- CSV export on TradeTable
- CandlestickChart includes volume pane
- Auto-refresh: logs (3s), dashboard (60s)
- All API calls use correct field names matching actual responses
- TypeScript compiles clean, production build succeeds

**Known parity gaps (by category):**

*Category A — Backend API limitations (API returns aggregate metrics, no time series):*
- Benchmark page: missing equity comparison, rolling correlation, rolling alpha/beta, relative strength, drawdown comparison charts (API only returns aggregate stats)
- Dashboard Risk tab: missing return distribution histogram and rolling risk charts
- Dashboard Portfolio tab: missing benchmark overlay and attribution chart/text
- Backtest Risk tab: missing VaR waterfall, rolling risk, return distribution charts

*Category B — Design decisions (acceptable trade-offs):*
- System Health: React uses collapsible check panels instead of 6 separate tabs with charts (same information, different presentation)
- Dashboard KPI cards: show different metrics than Dash (Win Rate/Avg Return vs Portfolio Value/30D Return) based on what the API provides
- IV Surface: 2 tabs (SVI, Heston) instead of 3 (Arb-Aware SVI requires server-side computation)

*Category C — Enhancement opportunities:*
- Data Explorer: missing bottom stats bar (52W Hi/Lo, Avg Vol, Day Change) — computable from bar data
- Model Lab: missing feature correlation heatmap
- Benchmark: missing animation mode

### Phase 4: Cutover — COMPLETE (2026-02-23)

**Actions taken:**
1. Extracted data loading functions from `dash_ui/data/loaders.py` into `api/services/data_helpers.py`
2. Updated all API service imports: `health_service.py`, `backtest_service.py`, `model_service.py`, `regime_service.py`, `routers/benchmark.py`
3. Verified all 14 API endpoints still work after import changes
4. Deleted `dash_ui/` directory (both root and nested `quant_engine/dash_ui/`)
5. Deleted `run_dash.py` (both root and nested)
6. Verified zero remaining `dash_ui` imports in active code

**Cutover checklist status:**
1. FastAPI endpoints live and tested — DONE
2. React pages cover all routes — DONE (10/10)
3. Provenance badges on all pages — DONE
4. Jobs run through backend — DONE
5. No `dash_ui` imports remain — DONE
6. Dash UI deleted — DONE
7. Dependency cleanup — PENDING (no `pyproject.toml` or `requirements.txt` exists yet)
8. New UI runbook — PENDING

---

### Phase 5: Backend API Enhancements (from NEW findings + Strategic Directions)

These items add time series data to the API and implement audit recommendations.

**Priority 1 — Add time series endpoints to close parity gaps:**
1. `GET /api/benchmark/equity-curves` — Returns strategy + SPY cumulative return time series for equity comparison chart
2. `GET /api/benchmark/rolling-metrics` — Returns rolling 60D correlation, alpha, beta, relative strength time series
3. `GET /api/dashboard/returns-distribution` — Returns daily return histogram data with VaR/CVaR lines
4. `GET /api/dashboard/rolling-risk` — Returns rolling volatility, Sharpe, drawdown time series
5. `GET /api/dashboard/equity` — Add benchmark overlay series to existing equity curve response
6. `GET /api/dashboard/attribution` — Returns tech-minus-def, momentum-spread factor returns

**Priority 2 — NEW findings from audit (transparency features):**
7. Add `regime_suppressed: bool` flag to prediction responses (NEW-1)
8. Add `cs_zscore` field to signal table (NEW-2)
9. Add `walk_forward_mode` to autopilot/backtest responses (NEW-3)
10. Add `predictor_type` badge to prediction responses (NEW-4) — field exists in `ResponseMeta`, needs frontend display
11. Add `sizing_method` to backtest responses (NEW-5)
12. Add `model_staleness_days` and `retrain_overdue` to dashboard KPIs (NEW-7)

**Priority 3 — Operational improvements:**
13. `GET /api/config` and `PATCH /api/config` endpoints for runtime-adjustable parameters (NEW-9)
14. Background monitor service for retrain triggers (Improvement 3)
15. Event-driven cache invalidation on model retrain (Improvement 4)

### Phase 6: Frontend Enhancements (matching Phase 5 API additions)

Once Phase 5 API endpoints exist, add the corresponding React components:

**Priority 1 — Chart parity:**
1. BenchmarkPage: Add equity comparison LineChart, rolling correlation, alpha/beta, relative strength, drawdown comparison charts
2. Dashboard RiskTab: Add return distribution HistogramChart with VaR lines, rolling risk DualAxisChart
3. Dashboard EquityCurveTab: Add benchmark overlay line to equity curve
4. Dashboard: Add AttributionTab or merge into EquityCurveTab (BarChart + text)

**Priority 2 — NEW finding displays:**
5. SignalDeskPage: Add regime suppression indicator (amber warning when active)
6. SignalTable: Add `cs_zscore` column
7. BacktestResults: Display `walk_forward_mode` and `sizing_method`
8. Dashboard KPIGrid: Add model staleness indicator card, retrain-overdue badge
9. AutopilotPage: Show `predictor_type` badge (red for heuristic)

**Priority 3 — Missing features:**
10. DataExplorerPage: Add bottom stats bar (Last Price, Day Change, 52W Hi/Lo, Avg Vol)
11. ModelLabPage: Add feature correlation heatmap
12. IVSurfacePage: Add Arb-Aware SVI tab (requires backend Heston endpoint)
13. BenchmarkPage: Add animation mode for progressive chart reveal

### Phase 7: Project Infrastructure (from Strategic Directions)

1. Create `pyproject.toml` with dependency groups: core, api, dash (legacy, removable now), dev (Direction 1)
2. Create `tests/conftest.py` with shared fixtures and pytest markers (Direction 3)
3. Add `run_server.py` — single-command launch for API + frontend static serving (Direction 10)
4. Enable OpenAPI docs at `/docs` and export `docs/openapi.json` (Direction 14)
5. TypeScript type generation from Pydantic schemas via `pydantic2ts` (Direction 12)
6. Fix `swaplevel()` → `reorder_levels()` in 15+ files (Phase 0 remaining)
7. Pandas 3.0 compatibility pass with `pd.options.mode.copy_on_write = True` in tests

## Detailed LLM implementation instructions (corrected copy/paste-ready)

This is the corrected version of the original audit's 20-step instruction list. Changes are marked with [CHANGED] or [NEW].

1. Create a new FastAPI backend under `quant_engine/api/` with routers for dashboard, system-health, data-explorer, model-lab, signals, backtests, benchmark-comparison, logs, autopilot, kalshi, and config-management. [CHANGED: added config-management router]

2. Define Pydantic schemas for every response using a standard `ApiResponse[T]` envelope containing `data: T` and `meta: ResponseMeta`. The `ResponseMeta` must include `data_mode`, `generated_at`, `warnings`, `source_summary`, plus domain-specific fields: `predictor_type`, `walk_forward_mode`, `regime_suppressed`, `feature_pipeline_version`, `model_version`, `sizing_method`. [CHANGED: expanded meta fields]

3. Do not call Dash code. Do not import `dash_ui/pages/*` in the API. Do not reuse `dash_ui/data/loaders.py` functions as-is — use them only as a reference for what needs to be computed, then rewrite with proper error handling and provenance. [CHANGED: clarified loaders.py reuse policy]

4. Create a unified `QuantEngineOrchestrator` service class that mirrors the orchestration pattern in `run_train.py`, `run_predict.py`, `run_backtest.py`, and `run_autopilot.py`. This is the single source of truth for correct method signatures and data flow. Do not reimplent orchestration in each router. [NEW]

5. Use existing engine modules as the source of truth (`data`, `features`, `models`, `backtest`, `autopilot`, `kalshi`). Call their actual interfaces — specifically `ModelTrainer.train_ensemble()` (not `.train()`), `EnsemblePredictor.predict(features, regimes, regime_confidence, regime_probabilities)` (not `.predict(latest)`), and `FeaturePipeline.compute_universe(data)`. [CHANGED: specified correct method names]

6. Add backend adapter services that convert raw backend objects/DataFrames into stable JSON DTOs. Handle MultiIndex (permno, date) → flat records conversion. Handle ticker ↔ permno translation via WRDS metadata or cache sidecars. [CHANGED: added permno handling]

7. Add a SQLite-backed job system for training/backtests/prediction/autopilot. Expose `POST /api/jobs/...`, `GET /api/jobs/{id}`, `GET /api/jobs/{id}/events` (SSE), and `POST /api/jobs/{id}/cancel`. Do not use Celery or ARQ — SQLite is sufficient for a single-user system and avoids the Redis dependency. [CHANGED: specified SQLite, not generic]

8. Create a React + TypeScript frontend (`webapp/`) with React Router v7 and TanStack Query v5. Note: in TanStack Query v5, `cacheTime` is renamed to `gcTime`. [CHANGED: version specifics]

9. Build a new design system and app shell from scratch. Keep the same Bloomberg-dark concept but implement it as CSS custom property tokens. Include a `DataProvenanceBadge` component (green/amber/red) that renders on every panel. Include a `DemoModeBanner` for when demo mode is active. [CHANGED: added specific components]

10. Replace silent fallbacks with explicit error states. Demo mode is a deliberate opt-in feature controlled by a config flag, never activated by exception handling. When demo mode is off, errors produce structured error JSON with diagnostic info and actionable suggestions. [CHANGED: clarified demo mode policy]

11. Use TradingView Lightweight Charts for OHLCV candlesticks and price charts. Use Apache ECharts for all other 2D analytical charts (equity curves, rolling metrics, heatmaps, bar charts). Use Plotly React only for 3D IV surface visualization. [CHANGED: specified TradingView LC instead of generic ECharts-for-everything]

12. Use TanStack Table with virtualization for trade logs and signal tables. No need for AG Grid licensing overhead for a single-user system. [CHANGED: TanStack Table instead of AG Grid]

13. Add frontend route pages matching the concept map. Configure TanStack Query with tiered `staleTime`/`gcTime` per data type: Infinity for immutable backtest results, 20-30s for regime/predictions, 1 hour for model metadata. [CHANGED: specified tiering]

14. For Data Explorer, key loaded data by `(ticker, timeframe)` in the query key. Validate that the response `timeframe` field matches the page expectation. Include provenance metadata (source, quality score) in the display. [UNCHANGED]

15. For Signal Desk, display BOTH `predicted_return` (raw model output) AND `cs_zscore` (cross-sectional ranking). Show a visible indicator when regime-2 suppression is active. Show the predictor type (ensemble vs heuristic). [NEW: expanded requirements]

16. For Backtest page, implement real parameterized execution via job submission. Display the `sizing_method` and `walk_forward_mode` in results. Include position sizing breakdown per trade if available. Remove any parameter controls that do not actually affect output. [CHANGED: added sizing/walkforward transparency]

17. For Autopilot and Kalshi pages, build API adapters that read current registry/state/DB schemas. Map `autopilot/registry.py`'s `{"active": [], "history": []}` schema correctly. Map `kalshi/storage.py`'s `kalshi_distributions` table correctly. Do not use the old `probability_snapshots` or `strategies` key names from the Dash UI. [CHANGED: specified correct schema names]

18. Add contract tests that verify Pydantic response schemas match what the frontend expects. Add integration tests that call through to actual backend modules with synthetic but realistic data. Specifically test: correct `EnsemblePredictor.predict()` signature, correct `ModelTrainer.train_ensemble()` signature, correct `Backtester.run()` signature. [CHANGED: specified what to test]

19. Add an end-to-end smoke test suite: dashboard loads with real or demo data, data explorer loads a ticker, signal generation completes, log SSE stream connects, backtest job lifecycle (submit → poll → result) completes. [UNCHANGED]

20. Only after feature parity and acceptance, delete the entire `dash_ui` package and `run_dash.py`. Verify no active entrypoints import from `quant_engine.dash_ui`. Update dependency manifests. [UNCHANGED]

21. [NEW] Add a background monitor service that checks retrain triggers every 5 minutes and emits notifications via SSE when a trigger fires. Include model staleness in the dashboard KPI response.

22. [NEW] Implement tiered server-side caching with explicit invalidation: model-retrain events invalidate model metadata caches, data-ingestion events invalidate quality caches, config changes invalidate health caches.

23. [NEW] Add `GET /api/config` and `PATCH /api/config` endpoints for runtime-adjustable parameters (entry_threshold, confidence_threshold, max_positions, regime_2_trade_enabled, drawdown thresholds). Validate all changes against min/max bounds. Non-adjustable parameters (paths, DB connections) remain static.

---

## Suggested remediation order (complete)

**Before migration (stabilize current system):**
1. Fix `data/local_cache.py` metadata sidecar path (tests already exist and fail)
2. Fix `kalshi/distribution.py` direction alias handling (tests already exist and fail)
3. Add sequential fallback for `models/trainer.py` permutation importance (tests already exist and fail)

**During migration:**
4. Build FastAPI orchestrator + read-only endpoints
5. Build SQLite job system + compute endpoints
6. Build React shell + Dashboard page
7. Build remaining pages in priority order
8. Parallel run validation
9. Cutover and cleanup

---

## Files reviewed for this adaptation

**System reference:** `Quant_Engine_System_Reference.docx` (full 1223-line conversion)

**Live source files investigated:**
- `run_train.py`, `run_predict.py`, `run_backtest.py`, `run_autopilot.py`, `run_retrain.py`
- `models/trainer.py`, `models/predictor.py`, `models/governance.py`
- `features/pipeline.py`
- `regime/detector.py`, `regime/hmm.py`
- `backtest/engine.py`, `backtest/validation.py`
- `autopilot/engine.py`, `autopilot/strategy_discovery.py`, `autopilot/registry.py`, `autopilot/paper_trader.py`
- `data/loader.py`, `data/local_cache.py`, `data/provider_registry.py`
- `kalshi/storage.py`, `kalshi/distribution.py`
- `dash_ui/app.py`, `dash_ui/server.py`, `dash_ui/data/loaders.py`
- `dash_ui/pages/dashboard.py`, `dash_ui/pages/system_health.py`, `dash_ui/pages/model_lab.py`, `dash_ui/pages/signal_desk.py`, `dash_ui/pages/backtest_risk.py`, `dash_ui/pages/autopilot_kalshi.py`, `dash_ui/pages/data_explorer.py`, `dash_ui/pages/sp_comparison.py`
- `config.py`

**Original audit reviewed:** `CODEX_SYSTEM_AUDIT_2026-02-22.md` (1078 lines, all 22 findings verified)

---

# PART 4: STRATEGIC DIRECTIONS FOR THE MIGRATION

These directions go beyond the audit findings and migration instructions. They address project-level decisions that will shape the quality and maintainability of the new stack.

## Direction 1: No dependency manifest exists — create one immediately

The current repo has no `requirements.txt`, `pyproject.toml`, or `setup.cfg`. Dependencies are implicit (discovered by running imports and seeing what breaks). This is a blocking risk for the migration because:
- You cannot reliably set up a clean environment
- You cannot distinguish Dash-era dependencies from core dependencies
- New FastAPI/React dependencies have no canonical place to live

**Action:** Before writing any migration code, create a `pyproject.toml` with dependency groups:

```toml
[project]
name = "quant_engine"
version = "1.0.0"
requires-python = ">=3.10"

dependencies = [
    # Core (always needed)
    "pandas>=2.0",
    "numpy",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "joblib",
    "scipy",
    # Data
    "pyarrow",        # parquet support
    "requests",       # Kalshi API
    "duckdb",         # Kalshi storage
    "fredapi",        # FRED macro data
    # Regime
    "hmmlearn",
    # IV Surface
    "plotly",         # keep for backend surface computation
]

[project.optional-dependencies]
api = [
    "fastapi>=0.110",
    "uvicorn[standard]",
    "pydantic>=2.0",
    "sse-starlette",  # SSE support
]
dash = [
    # Legacy — remove after cutover
    "dash>=2.14",
    "dash-bootstrap-components",
    "flask-caching",
]
dev = [
    "pytest",
    "pytest-asyncio",  # for FastAPI test client
    "httpx",           # for FastAPI TestClient
]

[tool.pytest.ini_options]
filterwarnings = ["error::FutureWarning"]  # catch pandas 3.0 issues early
```

This lets you `pip install -e ".[api]"` for the new stack and `pip install -e ".[dash]"` for the old stack during parallel run, and `pip install -e ".[dev]"` for testing.

## Direction 2: The system is batch-oriented — do not over-engineer real-time

After investigating the actual codebase, the system has zero async patterns, zero WebSocket usage, and zero streaming infrastructure. Everything runs as batch cycles:
- `run_autopilot.py` runs one cycle, writes JSON/CSV, exits
- `run_predict.py` runs once, saves predictions CSV, exits
- `run_backtest.py` runs once, saves results, exits
- Paper trading updates on each autopilot cycle (not continuous)
- Kalshi client uses synchronous `requests` with a thread-safe rate limiter

**What this means for the migration:** Do not build a WebSocket infrastructure you don't need. The right real-time strategy is:

| Data type | Update mechanism | Why |
|-----------|-----------------|-----|
| Dashboard KPIs | Polling (60s) | Backed by cached file reads, not live computation |
| Regime state | Polling (30s) | Regime detection runs in batch, not continuously |
| Predictions | On-demand (user clicks "Generate") | Expensive computation, not streaming |
| Job progress | SSE | Unidirectional, server pushes progress to one client |
| Log stream | SSE | Unidirectional, server pushes log lines |
| Paper trading P&L | Polling (60s) | Updated only on autopilot cycle completion |
| Kalshi events | Polling (300s) | Events are infrequent (monthly economic releases) |

**SSE is the only real-time technology you need.** Use it for job progress and log streaming. Everything else is polling with TanStack Query's `refetchInterval`. Do not introduce WebSocket complexity.

## Direction 3: Conftest.py and test infrastructure need creation

The test suite has 20 files but no `conftest.py`, no shared fixtures, and no global configuration. Each test file creates its own synthetic data from scratch. This means:
- Expensive fixture setup is duplicated across test files
- No consistent synthetic data factory
- No pytest markers for slow/integration/unit separation

**Action:** Create `tests/conftest.py` with:

```python
import pytest
import pandas as pd
import numpy as np

@pytest.fixture(scope="session")
def synthetic_universe():
    """10 stocks, 500 trading days of OHLCV — reused across all tests."""
    np.random.seed(42)
    tickers = [f"TEST{i:03d}" for i in range(10)]
    dates = pd.bdate_range("2020-01-01", periods=500)
    data = {}
    for t in tickers:
        prices = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.02, 500))
        data[t] = pd.DataFrame({
            "open": prices * (1 + np.random.uniform(-0.005, 0.005, 500)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.01, 500))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.01, 500))),
            "close": prices,
            "volume": np.random.randint(100000, 10000000, 500).astype(float),
        }, index=dates)
    return data

# pytest markers
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m not slow')")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "api: marks FastAPI endpoint tests")
```

Also create `tests/api/` for FastAPI-specific tests using `httpx.AsyncClient`:

```python
# tests/api/conftest.py
import pytest
from httpx import AsyncClient, ASGITransport
from quant_engine.api.main import create_app

@pytest.fixture
async def api_client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
```

## Direction 4: DataFrame serialization strategy matters for performance

The current Dash UI serializes DataFrames to JSON via `dcc.Store`, which is extremely inefficient for large datasets (10 years × 60 tickers × 300 features = millions of rows). The FastAPI replacement must handle this better.

**Recommended serialization strategy:**

| Data type | Serialization | Why |
|-----------|--------------|-----|
| Small payloads (<1000 rows) | JSON (Pydantic models) | Human-readable, debuggable, standard |
| Trade logs (1000-5000 rows) | JSON with pagination | `?page=1&per_page=200` or cursor-based |
| OHLCV time series (2500+ points) | JSON with downsampling | Largest-triangle-three-buckets on backend, return max 2500 points |
| Full feature matrix (export) | CSV download endpoint | `GET /api/exports/features.csv` — never pass through JSON |
| Backtest equity curves | JSON (already small) | One point per trading day, ~2500 points max |

**Critical: never serialize a full 60-ticker × 300-feature × 2500-day panel as JSON.** That's 45 million cells. Instead:
- Dashboard shows pre-aggregated summaries
- Data Explorer shows one ticker at a time
- Signal Desk shows latest date's predictions only (60 rows)
- Feature importance shows top-20 features only
- Full exports use CSV download, not API response

**Backend downsampling function to implement:**

```python
def downsample_timeseries(series: list[dict], max_points: int = 2500) -> list[dict]:
    """Largest-triangle-three-buckets downsampling for chart display."""
    if len(series) <= max_points:
        return series
    # LTTB algorithm preserves visual shape while reducing points
    ...
```

## Direction 5: The autopilot engine is 926 lines — wrap it, don't rewrite it

`autopilot/engine.py` is the most complex module in the system. It has:
- A 4-level predictor fallback chain (champion → latest → retrain → heuristic)
- Walk-forward prediction with cross-sectional ranking
- Portfolio optimization with confidence weighting
- Strategy discovery, evaluation, and promotion
- Paper trading execution

The FastAPI service layer should **wrap** this module, not rewrite any of its logic. The correct pattern:

```python
# api/services/autopilot_service.py
class AutopilotService:
    def __init__(self, config: dict):
        self.engine = AutopilotEngine(config)

    async def run_cycle(self, params: AutopilotRequest) -> AutopilotResponse:
        """Run in thread pool to avoid blocking the event loop."""
        report = await asyncio.to_thread(self.engine.run_cycle)
        return AutopilotResponse(
            data=self._transform_report(report),
            meta=ResponseMeta(
                data_mode="live",
                predictor_type=report.get("predictor_type", "ensemble"),
                walk_forward_mode=report.get("wf_mode", "unknown"),
                ...
            )
        )

    def _transform_report(self, report: dict) -> dict:
        """Convert engine report dict to stable API schema."""
        # Map internal keys to API-stable keys
        # Handle missing fields gracefully
        # Never expose raw file paths
        ...
```

**The same pattern applies to all complex modules:** backtest engine, feature pipeline, regime detector. Wrap them in `asyncio.to_thread()` calls (they're all synchronous CPU-bound code), translate their outputs to Pydantic DTOs, and add provenance metadata.

## Direction 6: PERMNO is the internal identity — ticker is the display identity

Throughout the codebase, stocks are identified by CRSP PERMNO (a permanent numeric identifier that doesn't change when a company changes its ticker symbol). The Dash UI inconsistently uses ticker in some places and permno in others. The new stack must be consistent:

**Rule:** Backend uses PERMNO everywhere internally. API responses include BOTH `permno` and `ticker`. Frontend displays ticker to the user but uses permno as the stable key for caching and cross-referencing.

```python
# Every prediction/trade/signal response includes both
class StockIdentity(BaseModel):
    permno: str          # "14593" — stable, never changes
    ticker: str          # "AAPL" — display name, can change
    company_name: str | None = None  # optional, from WRDS metadata
```

This matters because if a company changes its ticker (happens regularly — e.g., Facebook → Meta changed FB → META), PERMNO stays the same. The cache, model artifacts, and backtest results all use PERMNO. If the frontend only knows tickers, cross-referencing breaks.

## Direction 7: The results/ directory is the implicit API — formalize it

Currently, `run_backtest.py` writes to `results/backtest_{horizon}d_trades.csv` and `results/backtest_{horizon}d_summary.json`. `run_predict.py` writes to `results/predictions_{horizon}d.csv`. The Dash UI reads these files directly. This file-based interface is actually the system's implicit API.

**Action:** Formalize this as a results registry:

```python
# api/services/results_service.py
class ResultsService:
    """Manages the results/ directory as a structured artifact store."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir

    def get_latest_backtest(self, horizon: int) -> BacktestSummary | None:
        summary_path = self.results_dir / f"backtest_{horizon}d_summary.json"
        trades_path = self.results_dir / f"backtest_{horizon}d_trades.csv"
        if not summary_path.exists():
            return None
        return BacktestSummary(
            summary=json.loads(summary_path.read_text()),
            trades_path=trades_path,
            generated_at=datetime.fromtimestamp(summary_path.stat().st_mtime),
            staleness_seconds=(datetime.now() - datetime.fromtimestamp(summary_path.stat().st_mtime)).seconds,
        )

    def get_latest_predictions(self, horizon: int) -> PredictionSet | None:
        ...

    def list_all_results(self) -> list[ResultManifest]:
        """Inventory of everything in results/."""
        ...
```

This service becomes the single point of access for read-only dashboard endpoints. It replaces the scattered file reads across Dash pages. New backtest/prediction jobs write through this service, which also handles cache invalidation.

## Direction 8: The HeuristicPredictor needs a dedicated API indicator

The `HeuristicPredictor` (autopilot/engine.py lines 50-123) is a simple momentum+mean-reversion heuristic that activates when no ML model is available. It produces predictions that look structurally similar to ensemble predictions (same columns, same confidence range) but are fundamentally different in quality. The current system logs which predictor is active but does not surface this in any structured way.

**Action:** The heuristic predictor already tags its predictions internally. Propagate this through the entire chain:

```
HeuristicPredictor.predict() → sets predictor_type="heuristic" in output
  → AutopilotEngine reads predictor_type from output
    → API response includes predictor_type in ResponseMeta
      → Frontend displays RED badge: "HEURISTIC MODE — No trained model available"
```

This prevents a scenario where a user trusts paper trading results that are actually driven by a simple heuristic, not the sophisticated 5-model ensemble they think is running.

## Direction 9: Build the FastAPI app to be runnable alongside `run_*.py` scripts

The `run_*.py` scripts are the working CLI tools that researchers use today. The FastAPI backend should NOT replace them — it should coexist. Both the API and the CLI scripts should call the same underlying engine modules.

**Architecture:**

```
run_train.py ──────┐
run_predict.py ────┤
run_backtest.py ───┤──→ Core Engine Modules (data/, features/, models/, backtest/, autopilot/)
run_autopilot.py ──┤
                   │
api/services/ ─────┘──→ Same modules, wrapped in Pydantic DTOs + async threading
```

**Practical implication:** If a researcher runs `python run_backtest.py --horizon 10` from the terminal, the results should appear in the React dashboard on next refresh (because both write to `results/`). If a user submits a backtest job via the UI, it should produce identical results to the CLI script with the same parameters.

**Test this explicitly:** Add an integration test that runs `run_backtest.py` via subprocess, then calls `GET /api/backtests/latest?horizon=10` and verifies the results match.

## Direction 10: Plan for the React build/serve architecture

The audit document specifies Vite + React but doesn't address how the frontend is served in production. There are two common patterns:

**Option A: Separate servers (recommended for development)**
```
uvicorn quant_engine.api.main:app --port 8000    # FastAPI backend
cd webapp && npm run dev                           # Vite dev server on port 5173
# Vite proxies /api/* to localhost:8000
```

**Option B: FastAPI serves the built React app (recommended for production)**
```python
# api/main.py
from fastapi.staticfiles import StaticFiles

app = FastAPI()
# Mount API routers first
app.include_router(dashboard_router, prefix="/api")
...
# Then serve React build as static files (catch-all for SPA routing)
app.mount("/", StaticFiles(directory="webapp/dist", html=True), name="webapp")
```

**Action:** Configure both modes. Use Option A during development (hot reload on both sides). Use Option B for the single-process production deployment. Add a `run_server.py` script:

```python
# run_server.py — replaces run_dash.py
import uvicorn
from quant_engine.api.main import create_app

if __name__ == "__main__":
    app = create_app(serve_frontend=True)  # mounts webapp/dist/
    uvicorn.run(app, host="127.0.0.1", port=8050)
```

This gives you one command (`python run_server.py`) that serves both the API and the React frontend on the same port, same as the current `python run_dash.py` experience.

## Direction 11: Vite proxy configuration for development

During development, the React dev server needs to proxy API calls to the FastAPI backend. Add this to `webapp/vite.config.ts`:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/api/logs/stream': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // SSE needs special handling — don't buffer
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            proxyRes.headers['cache-control'] = 'no-cache';
            proxyRes.headers['connection'] = 'keep-alive';
          });
        },
      },
    },
  },
  build: {
    outDir: 'dist',
  },
})
```

## Direction 12: TypeScript type generation from Pydantic schemas

Rather than manually maintaining TypeScript interfaces that mirror your Pydantic models, generate them automatically:

```bash
pip install pydantic-to-typescript
# Generates webapp/src/api/types.ts from all Pydantic models in api/schemas/
pydantic2ts --module quant_engine.api.schemas --output webapp/src/api/types.ts
```

This guarantees that the frontend TypeScript types always match the backend Pydantic schemas. If you change a schema, regenerate the types and the TypeScript compiler will catch any mismatches.

Add this to your development workflow:
```json
// webapp/package.json
{
  "scripts": {
    "generate-types": "cd .. && pydantic2ts --module quant_engine.api.schemas --output webapp/src/api/types.ts",
    "dev": "npm run generate-types && vite"
  }
}
```

## Direction 13: The Kalshi module is the most isolated — migrate it last but design its API first

The Kalshi integration is the most self-contained subsystem: it has its own client, storage (DuckDB), distribution inference, event features, walk-forward, and promotion gate. It barely interacts with the equity pipeline (only through shared regime detection). But the current Dash UI's Kalshi tab is also the most broken (wrong DB schema, wrong table names, wrong column names).

**Strategy:** Design the Kalshi API contract now (it's in the adapted audit), but implement it last in the React frontend. The Kalshi endpoints are valuable but lower priority than the core equity trading pipeline. The backend service should be built early (Phase 1) because it exercises the DuckDB adapter pattern, but the React page can wait until Phase 2 step 10.

## Direction 14: Add an OpenAPI schema export for documentation

FastAPI generates OpenAPI (Swagger) documentation automatically. This is extremely valuable for your use case because:
- You can hand the OpenAPI spec to an LLM to generate frontend API client code
- You get interactive API documentation at `/docs` (Swagger UI)
- You can validate frontend requests against the schema

**Action:** Enable the docs endpoint in production (not just development):

```python
app = FastAPI(
    title="Quant Engine API",
    version="1.0.0",
    description="Backend API for the Quant Engine trading research platform",
    docs_url="/docs",        # Swagger UI
    redoc_url="/redoc",      # ReDoc alternative
    openapi_url="/openapi.json",
)
```

After building the API, export the OpenAPI spec and include it in the repo:

```bash
python -c "from quant_engine.api.main import create_app; import json; print(json.dumps(create_app().openapi(), indent=2))" > docs/openapi.json
```

This becomes a contract document that both the frontend developer (or LLM) and the backend tests reference.

---

## Summary of all new directions

| # | Direction | Action |
|---|-----------|--------|
| 1 | No dependency manifest | Create `pyproject.toml` with dependency groups before anything else |
| 2 | Batch-oriented system | Use polling + SSE only, no WebSocket |
| 3 | No test infrastructure | Create `conftest.py` with shared fixtures and pytest markers |
| 4 | DataFrame serialization | Downsample time series, paginate tables, CSV for exports |
| 5 | Autopilot is 926 lines | Wrap in async service, do not rewrite |
| 6 | PERMNO vs ticker | Backend uses PERMNO, API returns both, frontend displays ticker |
| 7 | File-based results | Formalize as ResultsService with staleness tracking |
| 8 | HeuristicPredictor | Propagate predictor_type through entire chain to UI |
| 9 | CLI coexistence | API and run_*.py scripts share same engine modules |
| 10 | React serve architecture | Separate dev servers, single production server |
| 11 | Vite proxy | Configure /api proxy with SSE-safe settings |
| 12 | TypeScript from Pydantic | Auto-generate types.ts from schemas |
| 13 | Kalshi isolation | Design API first, implement UI last |
| 14 | OpenAPI export | Enable /docs, export spec to repo |
