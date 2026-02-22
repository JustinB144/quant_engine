# Quant Engine LLM Context Spec

## Purpose

This is the LLM-oriented version of the system audit.
It is a working context contract for any model that will modify or analyze `quant_engine`.

Use this before touching code.

Primary deep source-based reference:
- `docs/reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md`

Human-oriented companion:
- `docs/guides/QUANT_ENGINE_HUMAN_SYSTEM_GUIDE.md`

## LLM Mission Context (What This Repository Is)

`quant_engine` is a multi-subsystem quantitative research and operations framework that combines:
- equity data ingestion and point-in-time normalization,
- feature engineering,
- regime detection (rule-based + HMM-based),
- model training/versioning/prediction,
- realistic backtesting + risk management,
- strategy autopilot lifecycle management,
- Kalshi event-market research pipeline,
- and a Dash UI for operational visibility and workflow tooling.

It is not a single-script strategy. It is a pipeline + governance + UI system.

## Non-Negotiable Architecture Contracts (Preserve These)

### 1. PERMNO-first identity contract
- Treat `permno` as the canonical identity when available.
- `ticker` is display/compatibility metadata.
- Do not introduce new ticker-first joins in core data/backtest/autopilot paths.

Why:
- Prevents historical misjoins due to ticker instability.

### 2. No-leakage time alignment contract
- Feature joins must be strict backward/as-of joins.
- Event feature rows must remain strictly pre-release (`asof_ts < release_ts`).
- Walk-forward splits must preserve purge/embargo behavior.

Why:
- Leakage invalidates research results.

### 3. Promotion-gate contract
- Do not bypass `autopilot.promotion_gate` logic to promote strategies.
- Promotion requires robustness and capacity-aware checks, not just good returns.

Why:
- This is the system’s deployment-quality filter.

### 4. Dash UI is the active UI contract
- Active UI stack is `dash_ui` and `run_dash.py`.
- Legacy `ui/` and `run_dashboard.py` have been removed.
- New UI work should target `dash_ui/*` only.

### 5. Data provenance and quality contract
- Preserve metadata/provenance sidecars and quality scoring behavior.
- Prefer trusted cache/WRDS workflows over ad-hoc direct fetches in core paths.

## Current Top-Level System Map (Intent-Oriented)

- `data/`
  - Provider adapters, caching, quality checks, survivorship-safe loading
- `features/`
  - Feature engineering pipeline and factor families
- `regime/`
  - Regime classification (rule/HMM) + correlation regime features
- `models/`
  - Trainer, predictor, versioning, governance, retrain triggers, IV models
- `backtest/`
  - Signal-to-trade simulation + validation and execution realism
- `risk/`
  - Position sizing, portfolio risk, drawdown, stress, attribution, optimization
- `autopilot/`
  - Candidate discovery, promotion gate, registry, paper trader, orchestration engine
- `kalshi/`
  - Event-market client/provider/store/distribution/features/walk-forward/promotion
- `dash_ui/`
  - Active human-facing UI (Dash pages/components/theme/assets)
- root `run_*.py`
  - Operational entry points for core workflows

## Active Workflow Entry Points (CLI)

- `run_train.py`
- `run_predict.py`
- `run_backtest.py`
- `run_retrain.py`
- `run_autopilot.py`
- `run_kalshi_event_pipeline.py`
- `run_dash.py`
- `run_wrds_daily_refresh.py`

## Regime Detection Contract (Detailed LLM Notes)

### Canonical regime labels
- `0` = bull trend
- `1` = bear trend
- `2` = mean-reverting/default
- `3` = high volatility

### `regime.detector`
Responsibilities:
- Expose a stable system-facing API for regime labels + confidence + posterior probabilities.
- Support both rule and HMM engines.
- Output additional derived features (`regime_duration`, one-hot columns, transition probability).

Modification rules:
- Preserve output columns and meanings unless explicitly migrating all downstream consumers.
- Preserve fallback behavior from HMM -> rule engine when fit/data is insufficient.

### `regime.hmm`
Responsibilities:
- In-repo Gaussian HMM implementation with sticky transitions and duration smoothing.
- BIC-based state count selection support.
- Mapping raw hidden states to canonical semantic regimes.

Modification rules:
- Preserve numerical-stability safeguards (regularization, fallback behavior).
- Preserve hidden-state-to-semantic mapping interface (`map_raw_states_to_regimes`).
- Changes to observation matrix or mapping can alter downstream model/backtest behavior; treat as high impact.

### `regime.correlation`
Responsibilities:
- Detect correlation-spike conditions and expose correlation regime features.

Modification rules:
- Preserve column names and semantics if used in feature pipeline/backtests.

## Data and Feature Contracts (LLM Notes)

### `data.loader`
- Central loading orchestrator used by multiple workflows.
- Prefer adding behavior here rather than duplicating fetch logic in run scripts.
- Preserve quality-gating and fallback order semantics.

### `data.survivorship`
- Point-in-time membership handling is a core anti-bias mechanism.
- Changes here can silently invalidate historical research if mishandled.

### `features.pipeline`
- This is a contract module for train/predict feature consistency.
- Feature additions/removals should consider artifact compatibility and predictor expectations.

## Modeling Contracts (LLM Notes)

### `models.trainer`
- Implements anti-leakage training/validation discipline.
- Changes to splitting, CV, or gating logic affect all reported performance numbers.

### `models.predictor`
- Runtime artifact resolution and inference path.
- Preserves version/champion loading, regime-aware blending, and confidence composition behavior.

Modification rules:
- Do not silently fall back to a different version when explicit version load fails.
- Preserve output schema used by backtest/autopilot/UI pages.

## Backtest/Risk Contracts (LLM Notes)

### `backtest.engine.Backtester`
- Core execution simulator with simple and risk-managed paths.
- Includes realistic costs/impact and PERMNO assertions.

Modification rules:
- Treat changes to execution assumptions, cost models, and signal gating as behaviorally high-risk.
- Preserve result schema fields consumed by autopilot/UI/tests.

### `backtest.advanced_validation`
- Provides DSR/PBO/Monte Carlo/capacity metrics.
- Promotion gate depends on outputs from this area.

### `risk/*`
- Shared risk components used beyond the backtester (autopilot paper trading, analytics).
- Preserve API stability when possible.

## Autopilot Contracts (LLM Notes)

### `autopilot.engine`
- Orchestrates the full strategy lifecycle.
- Has fallback predictor path for environments without sklearn artifacts/deps.
- Enforces PERMNO-first assumptions in multiple checkpoints.

Modification rules:
- Preserve promotion flow ordering (discover -> evaluate -> gate -> registry -> paper).
- Preserve fallback behavior unless explicitly removing it across tests/docs/UI.

### `autopilot.strategy_discovery`
- Generates deterministic candidate grids.
- If changing search space semantics, document impact on promotion results and runtime cost.

### `autopilot.promotion_gate`
- The deployability contract.
- Thresholds and metrics encode system quality policy.

Modification rules:
- Do not weaken checks without explicit user approval.
- If adding metrics, keep backwards-compatible fields when possible.

### `autopilot.paper_trader`
- Stateful paper execution for promoted strategies.
- Uses cash accounting, entry/exit rules, and bounded history persistence.

Modification rules:
- Preserve state schema compatibility or include migration logic.

## Kalshi Event Pipeline Contracts (LLM Notes)

### `kalshi.client`
- Signed API client with rate limiting, retries, historical/live routing.
- Preserves canonical signing payload format and auth header behavior.

### `kalshi.storage`
- Stable event-time schema for Kalshi + macro event research.
- Schema changes are high impact; prefer additive changes.

### `kalshi.distribution`
- Converts contract quotes into distribution snapshots with quality/repair metadata.
- Includes threshold-direction resolution, monotonic repairs, stale quote logic, tail features, distance metrics.

Modification rules:
- Preserve output metadata fields if downstream UI/models/tests depend on them.
- Preserve quality-low signaling and confidence flags for uncertain semantics.

### `kalshi.events`
- Builds event-centric, pre-release feature panels and labels.

Modification rules:
- Preserve strict as-of join semantics and leakage guard checks.
- Preserve `asof_ts < release_ts` invariant.

### `kalshi.walkforward`
- Event-level walk-forward with purge/embargo and advanced contract metrics.

Modification rules:
- Preserve purge/embargo and trial-counting concepts.
- Any split logic changes require test updates and explicit rationale.

### `kalshi.promotion`
- Bridges event walk-forward results into standard promotion gating.
- Preserve compatibility with `autopilot.promotion_gate` and `BacktestResult`-like expectations.

## Dash UI Contract (LLM Notes)

### Active UI stack
- `dash_ui/*` only
- `run_dash.py` is the launcher

### Design intent
The UI is page-based because each page maps to a distinct operational question (health, data, model/regime, signals, backtest/risk, IV, benchmark comparison, autopilot/events).

Modification rules:
- Prefer page-local changes for page-local behavior.
- Use `dash_ui/components/*` for reusable UI primitives and chart patterns.
- Preserve page routes and major IDs unless you also update all callbacks and references.

### Active pages (conceptual)
- Dashboard
- System Health
- Data Explorer
- Model Lab
- Signal Desk
- Backtest & Risk
- IV Surface
- S&P Comparison
- Autopilot & Events

## Tests as Behavioral Spec (LLM Use)

Treat tests as architecture constraints, not just syntax checks.

High-signal test themes:
- no-leakage and panel split discipline
- delisting total return correctness
- dynamic execution cost behavior
- autopilot fallback behavior
- Kalshi hardening (signature, stale quotes, direction semantics, bin validity, event walk-forward)

When making changes:
- identify which test theme your change touches before editing,
- preserve intent even if exact implementation changes.

## Recommended LLM Workflow Before Editing

1. Read `docs/reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md`
2. Read this file (`docs/reports/QUANT_ENGINE_LLM_CONTEXT_SPEC.md`)
3. Read only the modules directly relevant to the requested change
4. Check for invariants touched (PERMNO, leakage, promotion gate, Dash UI, schema compatibility)
5. Make minimal changes
6. Run targeted validation/tests if possible
7. Summarize behavior impact, not just diff lines

## Safe Change Heuristics (LLM)

Good low-risk edits:
- UI copy/layout tweaks inside one Dash page
- additive visualization logic in `dash_ui/components/chart_utils.py`
- additive metrics fields (with backward-compatible defaults)
- docs/report updates

Higher-risk edits (require explicit impact analysis):
- data loader fallback logic
- survivorship filtering / point-in-time joins
- regime label semantics or mapping
- model split/CV logic
- backtest execution/cost assumptions
- promotion-gate thresholds/criteria
- Kalshi as-of joins / distribution reconstruction semantics
- storage schema changes

## Current Cleanup State (LLM Awareness)

- Legacy UI removed: `ui/`, `run_dashboard.py`
- Organized docs live under:
  - `docs/guides`
  - `docs/reports`
  - `docs/plans`
  - `docs/notes`
- Only remaining `ui` mention in active code is a historical comment in `dash_ui/theme.py`

## Prompt Snippet For Future LLMs (Reuse)

Read `docs/reports/QUANT_ENGINE_LLM_CONTEXT_SPEC.md` and `docs/reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md` first. Treat PERMNO-first identity, strict no-leakage joins, promotion-gate robustness checks, and Dash (`dash_ui`) as hard architectural constraints. Then inspect only the modules directly related to the task before changing code.
