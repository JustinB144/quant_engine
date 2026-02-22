# Quant Engine System Audit (Source-Backed)

Date: 2026-02-22
Auditor: Codex (GPT-5)
Repo: `quant_engine`

## Scope and method

This audit used:

- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/Quant_Engine_System_Reference.docx` (extracted with `python-docx`) for intended system behavior and UI intent.
- Direct source inspection across core modules and the full Dash UI implementation.
- Automated test execution (`pytest -q`) to capture reproducible failures.
- Contract checks against live imports in the local environment.

Important limitation:

- This is a thorough, source-backed audit of the current repository state, but no audit can guarantee discovery of every bug. Findings below are concrete and evidenced.

## Executive summary

The backend core is substantial and test coverage is meaningful, but the current system has several confirmed defects and a high-risk UI integrity problem:

- 6 reproducible test failures (cache metadata rehydration, model trainer sandbox compatibility, Kalshi threshold-direction handling).
- Multiple Dash pages silently fall back to demo/synthetic data when backend calls fail, which can present healthy-looking UI output while real functionality is broken.
- Several Dash callbacks are wired to non-existent backend APIs/classes/modules, meaning "live" features cannot work as implemented.
- The Dash UI contains significant schema drift versus current autopilot/Kalshi storage backends.

The strongest recommendation is a staged replacement of the Dash UI with a FastAPI backend + React frontend, preserving feature concepts only (not Dash code), with explicit provenance labels (`live`, `fallback`, `demo`) and contract-tested API endpoints.

## Test audit results (reproducible)

Command run:

- `pytest -q`

Observed result:

- `6 failed, 107 passed, 3 skipped` (plus 524 warnings)

Confirmed failing areas:

1. Cache metadata rehydration sidecar path behavior
2. Model trainer feature-selection parallelism in restricted environments
3. Kalshi threshold-direction semantics for `direction="above"/"below"`

## Findings (prioritized)

## P0 - UI live features are broken or misleading

### 1) Model Lab training callback calls a non-existent backend API

Evidence:

- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/model_lab.py:771` constructs `ModelTrainer(tickers=..., horizon=..., feature_mode=..., ensemble=...)`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/model_lab.py:777` calls `trainer.train()`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/models/trainer.py:206` shows `ModelTrainer.__init__` does not accept those kwargs
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/models/trainer.py` has no `def train(...)` method (only `train_ensemble(...)` at `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/models/trainer.py:453`)

Reproduced locally:

- `TypeError: ModelTrainer.__init__() got an unexpected keyword argument 'tickers'`

Impact:

- The "Training" tab cannot run real training.
- The callback falls into demo mode and displays synthetic CV metrics, masking real integration failure.

Recommended fix:

- Replace direct model-training UI callback logic with a backend job endpoint (FastAPI) that invokes the actual training pipeline contract (`FeaturePipeline` + `RegimeDetector` + `ModelTrainer.train_ensemble(...)`).
- Remove demo fallback for backend contract errors; surface explicit error + diagnostics.

### 2) Signal Desk "live model" path is impossible with current backend

Evidence:

- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/signal_desk.py:231` imports `ModelPredictor`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/models/predictor.py` defines `EnsemblePredictor`, not `ModelPredictor`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/signal_desk.py:251` calls `predictor.predict(latest)` with one argument
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/models/predictor.py:160` shows `EnsemblePredictor.predict(...)` requires `features`, `regimes`, and `regime_confidence` (plus optional `regime_probabilities`)

Reproduced locally:

- `ImportError cannot import name 'ModelPredictor'`

Impact:

- Signal Desk live predictions cannot work as implemented.
- UI silently falls back to demo signals (`/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/signal_desk.py:318-327`), creating a false sense of operational readiness.

Recommended fix:

- Introduce a backend signal-generation service that owns the full pipeline (data load -> feature compute -> regime detect -> predictor call) and returns a typed response.
- UI should never directly instantiate backend ML objects.

### 3) Autopilot page imports a missing module/function and parses the wrong registry schema

Evidence (missing import/function):

- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/autopilot_kalshi.py:474` imports `quant_engine.autopilot.discovery`
- No such module exists in `autopilot/` (repo file list shows `strategy_discovery.py`, not `discovery.py`)
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/autopilot_kalshi.py:475` calls `run_discovery_cycle()`
- No `run_discovery_cycle` symbol exists in `autopilot/*.py`

Evidence (registry schema mismatch):

- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/autopilot_kalshi.py:444` expects `registry["strategies"]`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/autopilot/registry.py:44` actual registry payload shape is `{"active": [], "history": []}`

Impact:

- Autopilot UI "live" discovery/registry integration is broken and falls back to demo.

Recommended fix:

- Add a dedicated API adapter for autopilot status that maps actual backend state (`StrategyRegistry`, `AutopilotEngine`) into a stable DTO for the frontend.
- Do not let UI read raw registry JSON files directly.

### 4) Kalshi UI queries an outdated/nonexistent DB schema and silently falls back to demo

Evidence:

- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/autopilot_kalshi.py:615` queries `probability_snapshots` (table not present in current storage schema)
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/kalshi/storage.py:319` defines `kalshi_distributions` (current distribution table)
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/autopilot_kalshi.py:625-633` expects `event_outcomes` columns like `event_date`, `actual`, `forecast`, `wf_accuracy`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/kalshi/storage.py:275-286` `event_outcomes` schema is `event_id`, `realized_value`, `release_ts`, `asof_ts`, `source`, `learned_at_ts` (no `event_date/actual/forecast/wf_accuracy`)

Impact:

- Kalshi page will fail on real DB access and silently show demo data.

Recommended fix:

- Add backend endpoints that translate current Kalshi storage schema into page-ready view models.
- Version API contracts separately from DB schema.

### 5) Backtest & Risk page is labeled as an interactive backtest runner but does not run a backtest

Evidence:

- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/backtest_risk.py:370` callback name `run_backtest`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/backtest_risk.py:373-375` only loads `results/backtest_10d_trades.csv`
- UI parameters `holding_period`, `max_positions`, `risk_mgmt` are accepted but not used in engine execution (`/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/backtest_risk.py:370-405`)

Impact:

- UI implies parameterized backtest execution but actually filters/preprocesses an existing CSV.
- User actions may appear to have computational effect when they do not.

Recommended fix:

- Rename page actions to "Load Results" until a real backend job exists, or implement real async backtest execution via API.

## P1 - Confirmed backend defects (tests failing)

### 6) Cache metadata rehydration writes/reads the wrong sidecar path for `_daily_` files

Evidence:

- `_cache_meta_path(...)` returns file-adjacent metadata for filenames containing `_daily_`:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/data/local_cache.py:123-135`
- `_read_cache_meta(...)` only checks `_cache_meta_path(...)` and `with_suffix(".meta.json")`, which are the same for `_daily_` files:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/data/local_cache.py:137-153`
- `rehydrate_cache_metadata(...)` uses `_cache_meta_path(...)` for existing-sidecar detection:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/data/local_cache.py:546-549`
- Tests expect canonical daily sidecar path `TICKER_1d.meta.json`:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/tests/test_cache_metadata_rehydrate.py:47`
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/tests/test_cache_metadata_rehydrate.py:60`
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/tests/test_cache_metadata_rehydrate.py:79`

Reproduced behavior:

- `_cache_meta_path("AAPL_daily_...csv") -> AAPL_daily_....meta.json` (not `AAPL_1d.meta.json`)

Impact:

- Rehydration misses existing canonical daily metadata.
- Source overwrite logic fails against canonical sidecars.
- Loader/rehydration metadata compatibility for legacy daily files is inconsistent.

Recommended fix:

- For daily files, standardize on canonical sidecar `TICKER_1d.meta.json`.
- In `_read_cache_meta`, check both canonical and file-adjacent paths to preserve backward compatibility.
- In `rehydrate_cache_metadata`, consider sidecar "exists" if either canonical or legacy-adjacent path exists.
- If both exist, define precedence and merge policy (recommended: canonical wins, legacy migrated once).

### 7) Model trainer feature selection hardcodes multiprocessing (`n_jobs=-1`) and fails in restricted environments

Evidence:

- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/models/trainer.py:426-433` calls `permutation_importance(..., n_jobs=-1)`
- Pytest failures:
  - `tests/test_integration.py::TestFullPipelineSynthetic::test_training_produces_result`
  - `tests/test_integration.py::TestCvGapHardBlock::test_cv_gap_hard_block`
- Exception observed: `PermissionError` from joblib/loky semaphore limits (`SC_SEM_NSEMS_MAX`)

Impact:

- Training/integration tests fail in sandboxed CI/dev environments.
- UI/automation flows depending on training may fail nondeterministically by environment.

Recommended fix:

- Wrap permutation importance in a retry path:
  - First try configured `n_jobs`
  - On `PermissionError`, `OSError`, or `NotImplementedError`, retry with `n_jobs=1`
- Expose `permutation_importance_n_jobs` as a constructor/config option (default `-1`, overrideable to `1` in CI/sandbox).

### 8) Kalshi threshold direction resolver ignores `direction="above"/"below"` in explicit metadata field

Evidence:

- `_resolve_threshold_direction_with_confidence(...)` checks `row["direction"]` only for `ge/gte/>=` and `le/lte/<=`:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/kalshi/distribution.py:149-155`
- It checks `above/below` only in `payout_structure`, not `direction`:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/kalshi/distribution.py:156-160`
- Regression test explicitly passes `direction="above"` and `direction="below"`:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/tests/test_kalshi_hardening.py:89-90`

Observed behavior:

- Tail probabilities become `NaN` because direction is treated as unknown and `direction_known=False`, causing fallback path that returns `tail_p_* = NaN`.

Impact:

- Threshold-curve Kalshi distribution features lose tail probabilities for valid contracts.
- Downstream Kalshi signals/quality metrics degrade.

Recommended fix:

- Accept `above/over` and `below/under` aliases in the explicit `direction` field (same normalization used for `payout_structure`).
- Keep test coverage as-is; it already captures the regression.

## P1 - UI/backend data integrity and provenance issues

### 9) Dash UI hides real errors behind demo/synthetic fallbacks across multiple pages

Evidence (examples):

- Feature tab demo fallback on any exception:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/model_lab.py:457-460`
- Regime detection demo fallback on any exception:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/model_lab.py:596-605`
- Training demo fallback on any exception:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/model_lab.py:809-845`
- Signal Desk demo fallback if live path fails:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/signal_desk.py:318-327`
- Data Explorer generates demo OHLCV if cache+yfinance fail:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/data_explorer.py:274-287`
- Autopilot/Kalshi page demo fallbacks:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/autopilot_kalshi.py:483-489`
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/autopilot_kalshi.py:552-555`
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/autopilot_kalshi.py:652-655`

Impact:

- Operational failures can be mistaken for valid live output.
- Users cannot reliably tell what is real vs simulated.

Recommended fix:

- Every response/view model must include `data_mode` (`live`, `fallback`, `demo`) plus `warnings[]`.
- Show persistent page-level provenance badge and last-success timestamp.
- Only use demo mode when explicitly enabled in settings or development mode.

### 10) Dashboard error state is captured but not surfaced to the user

Evidence:

- Dashboard loader stores exception details in payload:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/dashboard.py:381-384`
- Metric cards return blank placeholders when `loaded=False`:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/dashboard.py:404-406`
- Tab content renders "Loading data..." when `loaded=False`, even if an error is present:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/dashboard.py:476-480`

Impact:

- Real failures can look like a loading issue.
- Troubleshooting is harder than necessary.

Recommended fix:

- Add explicit error rendering path showing message + trace ID (not full traceback in production UI).
- Put traceback in server logs/API diagnostics endpoint instead.

### 11) Data Explorer stores data keyed only by ticker, causing timeframe collisions

Evidence:

- Store shape is ticker -> OHLCV payload (`dcc.Store(id="de-loaded-data", data={})`)
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/data_explorer.py:71`
- Load callback writes `store[ticker] = _df_to_store(df)` regardless of selected timeframe:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/data_explorer.py:369-394`

Impact:

- Loading `AAPL` daily then `AAPL` 1h overwrites one with the other.
- UI loses provenance/timeframe context for loaded series.

Recommended fix:

- Key by `(ticker, timeframe)` or nested structure `store[timeframe][ticker]`.
- Include metadata fields (`timeframe`, `source_mode`, `source_path`, `asof`) in stored payloads.

### 12) Data Explorer can silently label demo-generated data as "loaded" without provenance

Evidence:

- `_download_ticker(...)` returns synthetic demo data after cache and yfinance fallbacks fail:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/data_explorer.py:274-287`
- `load_data(...)` treats returned DataFrame as successful load and reports "Loaded X ticker(s)":
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/data_explorer.py:390-409`

Impact:

- Users can unknowingly analyze fake OHLCV series.

Recommended fix:

- Return structured result (`data`, `source_mode`, `source_name`, `warnings`) instead of raw `DataFrame`.
- Display explicit visual badge on charts/tables when data is synthetic.

## P2 - Schema drift / path mismatches / UX correctness issues

### 13) Inconsistent backtest trade file paths across pages

Evidence:

- Dashboard uses `results/backtest_trades.csv`:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/dashboard.py:189`
- Backtest & Risk uses `results/backtest_10d_trades.csv`:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/backtest_risk.py:373`
- S&P Comparison also uses `results/backtest_10d_trades.csv`:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/sp_comparison.py:179`

Impact:

- Different pages may show inconsistent/empty datasets for the same session.

Recommended fix:

- Centralize results artifact resolution in one backend service function (by horizon and canonical manifest).

### 14) S&P Comparison animation loop does not stop its interval when animation completes

Evidence:

- Interval is enabled/disabled only by toggle callback:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/sp_comparison.py:432-443`
- Animation callback reaches completion and returns `no_update` for figure but does not disable interval:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/sp_comparison.py:466-468`

Impact:

- `dcc.Interval` continues firing at 40ms after animation completion, wasting CPU and server cycles.

Recommended fix:

- Add `animation-interval.disabled` as an output of the animation callback (or a separate completion callback) and disable when `frame >= total`.

### 15) Dash page ordering has duplicates (`order=2`)

Evidence:

- Data Explorer registered with `order=2`:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/data_explorer.py:23`
- System Logs registered with `order=2`:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/system_logs.py:25`

Impact:

- Unstable page ordering in registry-driven navigation or menus (even if sidebar is manually ordered today).

Recommended fix:

- Enforce unique route sort order values.

### 16) System Logs page changes global logger configuration at import time

Evidence:

- Handler installation runs on module import:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/system_logs.py:76-77`
- It modifies the `quant_engine` root logger level and handlers:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/system_logs.py:67-74`

Impact:

- Importing the page can change logging behavior globally (verbosity, handler composition), which is a hidden side effect.

Recommended fix:

- Move handler installation to app startup with explicit opt-in and idempotent configuration.
- Avoid changing root logger levels in a page module.

### 17) Demo data generator claims reproducibility but uses salted `hash()`

Evidence:

- Comment says demo generators use reproducible seeds:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/autopilot_kalshi.py:52`
- `_demo_kalshi_events(...)` seeds RNG with `42 + hash(event_type) % 100`:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/autopilot_kalshi.py:140`

Impact:

- Python string hashing is process-salted by default, so demo outputs may vary between runs.

Recommended fix:

- Use a stable hash (e.g., `hashlib.sha256(event_type.encode()).hexdigest()`) or a fixed lookup map per event type.

## P2 - System Health page fidelity issues (mixes real checks with simulated outputs)

### 18) System Health payload omits real walk-forward windows; UI always simulates fold performance visuals

Evidence:

- `SystemHealthPayload` defines `wf_windows`:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/data/loaders.py:403`
- `collect_health_data()` never sets `payload.wf_windows` (only checks/score):
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/data/loaders.py:438-441`
- UI renders simulated windows when `wf_windows` is empty:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/system_health.py:711-729`

Impact:

- Walk-forward tab charts are synthetic even when the page appears to be a live audit.

Recommended fix:

- Populate `wf_windows` from actual validation/backtest outputs or label the panel "illustrative."

### 19) Promotion funnel in health data is hard-coded, not read from actual autopilot state

Evidence:

- `_check_promotion_contract()` returns fixed funnel counts:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/data/loaders.py:560-567`

Impact:

- System Health page can misrepresent current autopilot/promotion pipeline status.

Recommended fix:

- Source promotion funnel from `StrategyRegistry` / promotion audit artifacts, not constants.

### 20) Walk-forward tab uses unseeded random IS/OOS values in UI rendering

Evidence:

- Synthetic fold returns generated with `np.random.uniform(...)` during rendering:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/system_health.py:727-728`

Impact:

- Same page can show different performance visuals across reloads with no backend change.

Recommended fix:

- Remove random generation from render path.
- Either show explicit placeholder state or deterministic examples behind a `demo` label.

## P2 - Paper/Kalshi/autopilot UI schema mismatches causing silent demo mode

### 21) Paper Trading UI expects `equity_curve` in state, but paper trader state schema does not provide it

Evidence:

- UI expects `state["equity_curve"]` and `state["positions"]`:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/autopilot_kalshi.py:538-547`
- `PaperTrader` persisted state schema is `cash`, `realized_pnl`, `positions`, `trades`, `last_update` (no `equity_curve`):
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/autopilot/paper_trader.py:62-68`

Impact:

- Real paper-trading state loads partially or falls back to demo equity curve.

Recommended fix:

- Create a backend "paper summary" endpoint that computes equity curve from state + marks, instead of expecting raw persisted chart data.

## P3 - Forward-compatibility and maintainability risks

### 22) Pandas 3.0 compatibility risk: 500+ warnings in test suite

Evidence:

- `pytest -q` emitted 524 warnings, including multiple `FutureWarning` messages related to upcoming pandas Copy-on-Write / chained assignment behavior.
- Representative impacted code paths from warning summary include:
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/features/pipeline.py:255`
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/features/pipeline.py:262`
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/features/pipeline.py:263`
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/features/pipeline.py:424`
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/backtest/engine.py:619`
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/models/cross_sectional.py:134`
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/data/survivorship.py:383`
  - `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/data/survivorship.py:389`

Impact:

- Future pandas upgrades may change behavior or break code paths subtly.

Recommended fix:

- Add a CI warning budget and fix warnings incrementally.
- Prefer `.loc[...] = ...` with explicit `.copy()` when transforming slices/views.
- Add a pandas-next compatibility test job.

## Architectural audit observations (current Dash UI)

The current Dash UI is conceptually strong but implementation quality is uneven:

Strengths:

- Clear multi-page decomposition (`dashboard`, `system_health`, `data_explorer`, `model_lab`, `signal_desk`, `backtest_risk`, `iv_surface`, `sp_comparison`, `system_logs`, `autopilot_kalshi`)
- Shared visual theme and reusable components (`metric_card`, `page_header`, `sidebar`)
- Attempts to isolate data loading logic in `dash_ui/data/loaders.py`
- Good user-facing feature coverage (risk, regime, diagnostics, IV, Kalshi)

Weaknesses:

- UI directly imports and orchestrates backend engine code inside callbacks
- Silent exception swallowing + demo fallbacks hide real failures
- File/schema coupling is brittle and outdated in multiple pages
- Heavy payload serialization into `dcc.Store` harms scalability/performance
- No clear data provenance contract or typed API surface

## FastAPI + React migration instructions (blank-slate, same concepts)

This section is intentionally detailed so you can hand it to another LLM to implement.

## Migration goals

Build a new UI stack that:

- Preserves the concepts and information architecture of the Dash UI
- Removes all Dash UI code and starts from a clean implementation
- Is professional, fast, explicit about data provenance, and reliable
- Uses FastAPI for backend APIs and React for frontend
- Does not directly expose backend internals/files/schemas to the frontend

## Non-goals (important)

- Do not port Dash callback code line-for-line.
- Do not reuse Dash page files or component code.
- Do not preserve current demo-fallback behavior by default.
- Do not make the frontend depend on raw filesystem paths or DB table names.

## Concept parity map (Dash -> React routes, same ideas)

Use the same concepts, but rebuild the UI from scratch:

1. `Dashboard` -> `/dashboard`
   - KPI cards
   - Portfolio vs benchmark equity
   - Regime state/probabilities/transition matrix
   - Model health
   - Feature importance
   - Trade log
   - Risk analytics

2. `System Health` -> `/system-health`
   - Scorecards
   - Config + data integrity audit
   - Promotion contract status
   - Walk-forward integrity
   - Execution realism
   - Complexity monitor

3. `Data Explorer` -> `/data-explorer`
   - Ticker/timeframe data loading
   - Price + volume charts
   - Data quality report

4. `Model Lab` -> `/model-lab`
   - Feature importance and regime diagnostics
   - Training job launch + progress + results

5. `Signal Desk` -> `/signal-desk`
   - Current predictions and ranking
   - Confidence vs return diagnostics
   - Distribution diagnostics

6. `Backtest & Risk` -> `/backtests`
   - Backtest job submission/history/results
   - Risk analytics
   - Trade analysis + export

7. `IV Surface` -> `/iv-surface`
   - Interactive SVI/Heston/arb-aware views
   - 3D surface and smiles

8. `S&P Comparison` -> `/benchmark-comparison`
   - Strategy vs SPY analytics
   - Rolling correlation/alpha/beta/drawdown

9. `System Logs` -> `/logs`
   - Real log stream
   - Filters and retention

10. `Autopilot & Events` -> `/autopilot`
   - Discovery/promotions registry
   - Paper trading summary
   - Kalshi event analytics

## Target architecture (recommended)

### Backend (`FastAPI`)

Create a new backend package, e.g.:

- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/api/`

Recommended structure:

- `api/main.py` - FastAPI app factory
- `api/routers/` - route modules per feature domain
- `api/schemas/` - Pydantic request/response models
- `api/services/` - orchestration over core quant_engine modules
- `api/jobs/` - async job runner abstractions (training/backtests)
- `api/cache/` - caching abstraction (Redis or local fallback)
- `api/deps/` - dependency injection (config, auth, stores)
- `api/errors.py` - typed exception mapping

### Frontend (`React + TypeScript`)

Create a new frontend app, e.g.:

- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/webapp/`

Recommended stack:

- Vite
- React + TypeScript
- React Router
- TanStack Query (server state)
- Zustand (small local UI state)
- AG Grid or TanStack Table (tables)
- ECharts for most 2D charts (performance)
- Plotly React only for IV 3D surfaces (specialized 3D)
- CSS modules or Tailwind + design tokens (prefer typed theme tokens)

## Critical design requirements (professional + fast + reliable)

### 1) Explicit data provenance (required)

Every API response must include:

- `data_mode`: `live` | `fallback` | `demo`
- `generated_at`
- `warnings[]`
- `source_summary` (cache/DB/model versions used)
- `staleness_seconds` where relevant

Frontend must display a visible badge on every page/panel.

### 2) No heavy computations in request thread for long-running actions

Training/backtests/autopilot discovery should be job-based:

- `POST /jobs/train`
- `POST /jobs/backtest`
- `POST /jobs/autopilot/discovery`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/events` (SSE preferred)

Use a persistent job registry (SQLite or Redis) with statuses:

- `queued`, `running`, `succeeded`, `failed`, `cancelled`

### 3) Typed contracts between frontend and backend

Never let React read:

- raw CSV files
- raw JSON registry files
- raw DuckDB table names
- raw `quant_engine` class internals

All frontend data should come from Pydantic DTOs with versioned schema.

### 4) Server-side caching for expensive read endpoints

Examples:

- dashboard summary snapshots (30-60s)
- system health summary (60-300s)
- regime payload (30-60s)
- feature importance metadata (300s)
- benchmark comparison snapshots (60s)

Do not cache job endpoints.

### 5) Error transparency (no silent fallback)

Rules:

- Fallbacks must be explicit in response metadata
- Demo mode only if user/dev flag enabled
- Backend exceptions return structured error JSON
- Frontend shows actionable error panels, not endless spinners

## Backend API design (endpoint blueprint)

Implement these first (read-only), then jobs.

### Dashboard

- `GET /api/dashboard/summary`
  - KPI values
  - equity series
  - benchmark series
  - model health snapshot
  - regime snapshot
  - top feature importance
  - recent trades
  - risk metrics

Split if payload becomes too large:

- `GET /api/dashboard/kpis`
- `GET /api/dashboard/equity`
- `GET /api/dashboard/regime`
- `GET /api/dashboard/model-health`
- `GET /api/dashboard/risk`
- `GET /api/dashboard/trades`

### System Health

- `GET /api/system-health/summary`
- `GET /api/system-health/details`

Requirements:

- Mark each subsection as `live` or `illustrative`.
- Do not synthesize random fold returns in API responses.

### Data Explorer

- `POST /api/data-explorer/load`
  - request: tickers[], timeframe, source_policy
  - response: series metadata + paged/downsampled data
- `GET /api/data-explorer/{ticker}/quality?timeframe=daily`

Important:

- Include `timeframe` in response identity.
- Avoid returning huge raw arrays for large universes unless requested.

### Model Lab

- `GET /api/model-lab/features/importance`
- `GET /api/model-lab/regime/payload`
- `POST /api/model-lab/train` -> create job

### Signal Desk

- `POST /api/signals/generate`
  - request: `horizon`, `universe`, `top_n`
  - response: predictions + diagnostics + provenance

Backend must orchestrate:

- data load
- feature pipeline
- regime detection
- `EnsemblePredictor`

### Backtests

- `POST /api/backtests/run` -> create job
- `GET /api/backtests/{job_id}/result`
- `GET /api/backtests/latest`
- `GET /api/backtests/{job_id}/trades.csv`

### IV Surface

- `POST /api/iv/svi/surface`
- `POST /api/iv/heston/surface`
- `POST /api/iv/svi/arb-aware`

Prefer backend-calculated surfaces; frontend only renders.

### Benchmark Comparison

- `GET /api/benchmark-comparison?period=ALL&horizon=10`

### Logs

- `GET /api/logs?level=INFO&limit=500`
- `GET /api/logs/stream` (SSE)
- `POST /api/logs/clear` (if you want the UI clear button)

Do not install handlers in route modules. Configure logging in app startup.

### Autopilot & Kalshi

- `GET /api/autopilot/registry`
- `POST /api/autopilot/discovery` -> job
- `GET /api/autopilot/paper-summary`
- `GET /api/kalshi/events?event_type=CPI`
- `GET /api/kalshi/distributions/current?event_type=CPI`
- `GET /api/kalshi/walkforward?event_type=CPI`

Backend must translate current `kalshi/storage.py` schema to UI DTOs.

## Frontend implementation instructions (React)

### 1) App shell and layout

Rebuild the current concepts, but not the Dash layout code:

- Fixed left sidebar navigation
- Main content region
- Global status/header strip (clock optional)
- Route-based pages

Professional UI requirements:

- Responsive desktop-first layout with mobile collapse
- Keyboard-accessible sidebar and tabs
- Consistent spacing scale and typography
- Clear empty/error/loading states
- Visible data provenance badges on every panel

### 2) Design system (new, not copied from Dash CSS)

Keep the general Bloomberg-dark concept if desired, but build a real tokenized theme:

- `color.background.canvas`
- `color.background.panel`
- `color.border.default`
- `color.text.primary`
- `color.text.secondary`
- `color.status.success/warn/error/info`
- `color.chart.series[n]`
- `radius.sm/md/lg`
- `shadow.panel`
- `space.1..10`
- `font.family.ui`, `font.family.mono`

Component primitives to implement first:

- `AppShell`
- `SidebarNav`
- `TopStatusBar`
- `PageHeader`
- `MetricCard`
- `Panel`
- `StatusBadge`
- `DataProvenanceBadge`
- `ErrorState`
- `EmptyState`
- `LoadingBlock`
- `TableToolbar`

### 3) Data fetching and caching

Use TanStack Query:

- Query keys per page/panel
- `staleTime` aligned to backend cache durations
- Automatic retry only for transient errors
- No retry for contract errors (`4xx`)

Use optimistic updates only for simple UI actions (e.g., toggles), not for jobs.

### 4) Long-running jobs UX

For training/backtests/autopilot:

- Submit job -> immediately show job card with state
- Stream logs/progress via SSE (or poll every 1-2s)
- Persist recent jobs in local UI state
- Allow cancellation when supported
- Link to result artifact(s)

### 5) Charts and tables

Use:

- ECharts for high-frequency/rolling 2D analytics (fast)
- Plotly React for IV 3D surfaces and specialized financial plots if needed
- AG Grid (or TanStack Table) for large trade/signal tables with sorting/filtering/export

Performance rules:

- Downsample long time series on backend or frontend (largest-triangle or fixed bucket)
- Virtualize large tables
- Avoid shipping full raw arrays for every page at once
- Do not store multi-page payloads in global browser state by default

## "Same ideas, blank slate" implementation plan

This is the exact plan to preserve concepts while deleting old UI later.

### Phase 0 - Freeze concepts (no code reuse)

Create a spec doc (new) that lists:

- routes/pages
- widgets per page
- controls per widget
- data needed per widget
- interactions (refresh, filters, export, drilldown)
- provenance behavior

Use the current Dash pages and the system reference DOCX only as a concept inventory.

### Phase 1 - Build backend adapters around existing engine modules

Implement FastAPI services that call existing backend code safely:

- Reuse core modules (`data/*`, `features/*`, `models/*`, `backtest/*`, `autopilot/*`, `kalshi/*`)
- Do not reuse `dash_ui/pages/*` callback logic
- Optionally reuse selective pure helpers from `dash_ui/data/loaders.py` only after review (many functions are placeholders/simulated)

Add contract tests for each endpoint.

### Phase 2 - Build frontend shell + one page at a time

Recommended order:

1. `Dashboard` (highest visibility)
2. `Data Explorer` (core data troubleshooting)
3. `System Health` (but only with truthful live/illustrative labeling)
4. `Signal Desk`
5. `Backtests`
6. `Benchmark Comparison`
7. `Logs`
8. `Model Lab`
9. `Autopilot`
10. `IV Surface`

### Phase 3 - Parallel run and parity validation

Run Dash and React/FastAPI side-by-side temporarily.

Validation checklist per page:

- Same concept coverage
- Correct backend wiring (no demo fallback unless explicitly enabled)
- Provenance badge present
- Error states visible
- Performance acceptable on realistic payloads
- Exports/logs/jobs functional

### Phase 4 - Remove old Dash UI files (only after parity signoff)

Delete only after the new stack is accepted:

- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/` (entire package)
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/run_dash.py`
- Dash-related docs that are obsolete (after replacing with new docs)
- Dash dependencies from environment/package management config

Before deletion:

- Archive a screenshot set of old UI for reference
- Freeze route/feature parity checklist
- Confirm no production entrypoints still import `dash_ui`

## Explicit old UI removal instructions (required)

This must be treated as part of the migration, not optional cleanup.

### Rule

- Do not keep the old Dash UI running in parallel long-term.
- After FastAPI + React parity is signed off, remove the Dash UI code and Dash entrypoints entirely.
- The new frontend should reuse concepts only, not Dash files/components/callbacks.

### What to remove

Delete these old UI assets after cutover:

- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/` (all pages, components, theme, server, app)
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/run_dash.py`
- Any legacy Dash launch scripts/docs that reference Dash as the primary UI
- Dash-specific dependencies from the project environment (for example: `dash`, `dash-bootstrap-components`, `plotly` if no longer needed in backend, `flask-caching` if only used for Dash)

### What not to copy into the new UI

- Dash callback logic (`@callback`)
- `dcc.Store` payload designs
- Demo fallback behavior that hides backend failures
- Direct file reads from the frontend layer
- Dash CSS classes/layout structure

### Safe cutover checklist

1. New FastAPI endpoints are live and contract-tested.
2. New React pages cover all required concepts/routes.
3. Provenance badges and explicit error states exist on all pages.
4. Backtest/training/autopilot actions run through backend jobs (no fake UI actions).
5. No imports of `quant_engine.dash_ui` remain in active entrypoints.
6. Dash UI directory and `run_dash.py` are deleted.
7. Dependency manifests/env docs are updated to remove Dash-era setup steps.
8. New UI runbook replaces old Dash run instructions.

## Detailed implementation instructions for your LLM (copy/paste ready)

Use this as a direct build brief.

1. Create a new FastAPI backend under `quant_engine/api` with routers for dashboard, system-health, data-explorer, model-lab, signals, backtests, benchmark-comparison, logs, autopilot, and kalshi.
2. Define Pydantic schemas for every response. Include `data_mode`, `generated_at`, `warnings`, and `source_summary` in all top-level responses.
3. Do not call Dash code. Do not import `dash_ui/pages/*` in the API.
4. Use existing engine modules as the source of truth (`data`, `features`, `models`, `backtest`, `autopilot`, `kalshi`).
5. Add backend adapter services that convert raw backend objects/dataframes into stable JSON DTOs.
6. Add a job system for training/backtests/autopilot discovery. Expose `POST /jobs/...`, `GET /jobs/{id}`, and SSE progress streams.
7. Create a React + TypeScript frontend (`webapp/`) with React Router and TanStack Query.
8. Build a new design system and app shell from scratch. Keep the same page concepts but do not port Dash markup/CSS.
9. Implement a visible provenance badge on every panel. If data is demo/fallback, it must be obvious.
10. Replace silent fallbacks with explicit error states. Demo mode should be opt-in.
11. Use ECharts for most 2D charts and Plotly React only where 3D or complex financial charts justify it.
12. Use AG Grid or TanStack Table for large tables with virtualization and CSV export.
13. Add frontend route pages matching the concept map (`/dashboard`, `/system-health`, `/data-explorer`, etc.).
14. For Data Explorer, key loaded data by `(ticker, timeframe)` and keep provenance metadata in state.
15. For Signal Desk, implement backend orchestration that uses `EnsemblePredictor` correctly (features + regimes + regime confidence).
16. For Model Lab training, implement backend job submission instead of synchronous UI-callback training.
17. For Autopilot and Kalshi pages, build API adapters that read current registry/state/DB schemas (not legacy table names or old JSON shapes).
18. Add integration tests that verify UI-backend contracts and prevent the current class/module mismatch regressions.
19. Add an end-to-end smoke test suite for core pages (dashboard load, data explorer load, signal generation, logs stream, backtest job lifecycle).
20. Only after feature parity and acceptance, delete the entire `dash_ui` package and `run_dash.py`.

## Suggested remediation order (bugs first, then migration)

If you want to stabilize the current repo before migration:

1. Fix `data/local_cache.py` metadata sidecar path compatibility (tests already failing)
2. Fix `kalshi/distribution.py` direction alias handling (`above`/`below`)
3. Add sequential fallback for `models/trainer.py` permutation importance
4. Remove/flag silent demo fallbacks in Dash pages
5. Stop wiring Dash callbacks to nonexistent backend APIs
6. Begin FastAPI + React replacement

## Files reviewed (high-value subset)

Core/backend:

- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/data/local_cache.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/models/trainer.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/models/predictor.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/kalshi/distribution.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/kalshi/storage.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/autopilot/engine.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/autopilot/registry.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/autopilot/paper_trader.py`

Dash UI (full page set and shell):

- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/app.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/server.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/theme.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/data/loaders.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/dashboard.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/system_health.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/data_explorer.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/model_lab.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/signal_desk.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/backtest_risk.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/iv_surface.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/sp_comparison.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/system_logs.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/dash_ui/pages/autopilot_kalshi.py`

Tests/docs used as evidence/context:

- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/tests/test_cache_metadata_rehydrate.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/tests/test_integration.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/tests/test_kalshi_hardening.py`
- `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/Quant_Engine_System_Reference.docx`
